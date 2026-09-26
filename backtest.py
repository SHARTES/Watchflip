"""Backtest — would the shortlist have worked over the last few days?

    python backtest.py

It replays recent auctions as if they were live. For each vintage-Omega lot
that closed in the test window, it rewinds to a few hours before the close,
uses only what you could have seen then, and asks what the shortlist would
have said. Then it checks what actually happened.

Rules that keep it honest:

* The model is trained only on lots that closed BEFORE the test window.
* The "current bid" at decision time is the bid one to five hours before
  close — the same number the live shortlist reads.
* A flagged lot counts as won if it actually closed at or under your max bid,
  and the price paid is the actual final price.

What it cannot test is the resale price. Profit here assumes you resell at the
same realism factor the shortlist uses.

Lots valued by the feature tier are reported separately and WITHOUT a profit
figure. Their value comes from the model itself, so the backtest would only be
the model agreeing with itself. What it does report is how the feature estimate
compares with real evidence on lots that have both — that is the honest test of
whether the feature tier can be trusted where it is the only evidence. The backtest checks the buying side —
how often you win and at what price — not whether the watches then sell.

One small compromise: eBay prices are today's, not the prices on the day each
lot closed. Asks move slowly, and this is eBay data rather than Catawiki
results, so it does not leak the answer — but it is not perfectly historical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import db
import model
import shortlist as sl

TEST_DAYS = 5     # replay the last five days
CAL_DAYS = 3      # of the training period, calibrate on its last three days


def seller_info(ids) -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute(
            "select lot_id, seller_name, url from lots where lot_id = any(%s)",
            (list(ids),),
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["seller_name", "url"])
    return pd.DataFrame(rows).set_index("lot_id")


def replay() -> tuple[pd.DataFrame, dict]:
    closed = model.load()
    end = closed["close_time"].max()
    start = end - pd.Timedelta(days=TEST_DAYS)
    train = closed[closed["close_time"] < start].reset_index(drop=True)
    test = closed[closed["close_time"] >= start].reset_index(drop=True)

    sl.CAL_DAYS = CAL_DAYS
    live, X_test, test_b, _, _, fair = sl.build_live_model(train, test)
    pred = live.predict(X_test)

    ref_to_line = {}
    for ref, wm in zip(closed["reference_number"], closed["watch_model"]):
        k = model.ref_key(ref)
        if k:
            ref_to_line.setdefault(k, sl.line_of(wm))
    by_ref, by_line = sl.load_asks(ref_to_line)
    # Only sales from before the window — the test lots must not price themselves.
    cw_train = sl.catawiki_by_ref(train)
    sl.calibrate_ratio(train, by_ref)
    sl.calibrate_feature_bias(live, by_ref)

    tm = model.target_mask(test_b)
    info = {"start": start, "end": end, "train": len(train),
            "target": int(tm.sum()), "filtered": 0, "no_decision_point": 0,
            "no_value": 0}

    rows = []
    for i, (idx, lot) in enumerate(test_b.iterrows()):
        if not tm[i]:
            continue
        if not sl.passes_filters(lot):
            info["filtered"] += 1
            continue
        # The decision moment is a few hours before close. Without a snapshot
        # from then, there is nothing the live shortlist could have read.
        if pd.isna(lot["late_bid"]):
            info["no_decision_point"] += 1
            continue

        line = sl.line_of(lot["watch_model"])
        f_i = fair.get(idx)
        v = sl.value_for(lot, by_ref, by_line, cw_train, f_i)
        if v is None:
            info["no_value"] += 1
            continue
        value, realism, _, basis = v

        non_eu = sl.eu_status(lot["seller_country"]) == "non_eu"
        ceiling, net_sale = sl.max_bid(value, realism, non_eu, sl.margin_for(basis))
        current = float(lot["late_bid"])
        final = float(lot["final_price"])

        p20, p50, p80 = (float(pred.loc[idx, c]) for c in ("p20", "p50", "p80"))
        p_win = sl.p_under(ceiling, p20, p50, p80) if ceiling > current else 0.0

        rows.append({
            "lot_id": lot["lot_id"],
            "closed": lot["close_time"],
            "line": line,
            "reference": lot["reference_number"] if isinstance(lot["reference_number"], str) else "—",
            "decision_bid": current,
            "p50": p50,
            "max_bid": ceiling,
            "p_win": p_win,
            "flagged": ceiling > current and p_win >= sl.MIN_WIN_PROB,
            "final": final,
            "won": final <= ceiling,
            "landed": sl.landed(final, non_eu),
            "profit": net_sale - sl.landed(final, non_eu),
            "non_eu": non_eu,
            "basis": basis,
            "value": value,
            "fair": float(f_i) if f_i is not None else float("nan"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.join(seller_info(df["lot_id"]), on="lot_id")
    return df, info


def report(df: pd.DataFrame, info: dict) -> None:
    days = (info["end"] - info["start"]).total_seconds() / 86400
    print(f"BACKTEST — {days:.0f} days of real auctions replayed, "
          f"{info['start']:%d %b} → {info['end']:%d %b}")
    print(f"model trained on {info['train']:,} earlier lots · resale assumed at "
          f"{sl.REALISM_EBAY:.0%} of the lower-quarter cleaned eBay ask")
    print(sl.ratio_text())
    print(f"{sl.bias_text()}\n")

    if df.empty:
        print("No lots to replay.")
        return

    feat = df[df["basis"] == "feature"]
    df = df[df["basis"] != "feature"]
    flagged = df[df["flagged"]]
    won = flagged[flagged["won"]]

    print("1. What the shortlist would have seen")
    print(f"   vintage-Omega lots in the window            {info['target']:>6}")
    print(f"   condition Good or better                    {info['target'] - info['filtered']:>6}")
    print(f"   with a bid a few hours before close         "
          f"{info['target'] - info['filtered'] - info['no_decision_point']:>6}")
    print(f"   with reference evidence (eBay or Catawiki)  {len(df):>6}")
    print(f"   valued only from features (see section 6)   {len(feat):>6}")
    print(f"   flagged as worth watching                   {len(flagged):>6}\n")

    print("2. What would have happened if you bid your max on the flagged lots")
    print(f"   won                                         {len(won):>6} of {len(flagged)}")
    print(f"   the model expected to win                   {flagged['p_win'].sum():>6.1f}"
          "   ← close to the line above = honest probabilities")
    if len(won):
        print(f"   total spent, landed                       €{won['landed'].sum():>7,.0f}")
        print(f"   profit if resold as assumed               €{won['profit'].sum():>7,.0f}"
              f"   (€{won['profit'].mean():.0f} per watch)")
        print(f"   losing buys among the wins                  {int((won['profit'] < 0).sum()):>6}")
        print(f"   wins per week                               {len(won) / days * 7:>6.1f}")
    print()

    every = df[df["won"]]
    print("3. Compared with bidding your max on every lot, no model")
    print(f"   lots you would have had to watch            {len(df):>6}   vs {len(flagged)} flagged")
    print(f"   won                                         {len(every):>6}   vs {len(won)}")
    print(f"   profit if resold as assumed               €{every['profit'].sum():>7,.0f}"
          f"   vs €{won['profit'].sum():,.0f}")
    print("   The max bid comes from the eBay value, so the model cannot win you")
    print("   more lots. Its job is focus: fewer lots to watch, for most of the wins.\n")

    if len(won):
        gap = (won["final"] / won["p50"] - 1).mean()
        side = "above" if gap >= 0 else "below"
        print("4. Warning signs to check by hand")
        print(f"   wins closed on average {abs(gap):.0%} {side} what the model expected")
        if gap >= 0:
            print("   → not bargains by auction standards; any profit comes from the resale estimate")
        print(f"   wins from non-EU sellers                    {int(won['non_eu'].sum()):>6}")
        print(f"   wins valued from eBay asks                  {int((won['basis'] == 'ebay').sum()):>6}")
        print(f"   wins valued from Catawiki sales             {int((won['basis'] == 'catawiki').sum()):>6}")
        print(f"   wins valued from the line                   {int((won['basis'] == 'line').sum()):>6}")
        print("   A big discount is either a real bargain or a flaw the listing hides.")
        print("   Only looking at the lots can tell which.\n")

        shown = min(len(won), 15)
        print(f"5. The lots you would have won — top {shown} of {len(won)} by profit. "
              "Open a few and judge them")
        print(f"   {'closed':<8}{'line':<15}{'ref':<14}{'final':>7}{'max':>7}{'profit':>8}  seller")
        for _, r in won.sort_values("profit", ascending=False).head(15).iterrows():
            print(f"   {r['closed']:%d %b}  {r['line']:<15}{str(r['reference'])[:13]:<14}"
                  f"{r['final']:>7.0f}{r['max_bid']:>7.0f}{r['profit']:>8.0f}  "
                  f"{str(r.get('seller_name') or '')[:22]}")
            print(f"   {'':<8}{r.get('url') or ''}")


def report_features(feat: pd.DataFrame, evidence: pd.DataFrame, days: float) -> None:
    print("\n6. Feature tier — lots with no reference evidence")
    print("   No profit shown: the value comes from the model, so it cannot check itself.")

    both = evidence.dropna(subset=["fair"])
    if len(both):
        est = both["fair"] * sl.ASK_TO_HAMMER / sl.FEAT_BIAS
        err = (est / both["value"] - 1).abs()
        bias = (est / both["value"] - 1).median()
        print(f"   check on {len(both)} lots that also had real evidence (after the ÷{sl.FEAT_BIAS:.2f} correction):")
        print(f"     feature estimate vs evidence, median gap   {err.median():>6.0%}")
        print(f"     direction                                  "
              f"{'too high' if bias > 0 else 'too low'} by {abs(bias):.0%}")
        print("   Under ~25% gap: usable where it is the only evidence.")
        print("   Over ~40%: treat feature-tier lots as leads to look at, not buys.")

    if feat.empty:
        print("   no feature-valued lots in the window")
        return
    f_flag = feat[feat["flagged"]]
    f_won = f_flag[f_flag["won"]]
    print(f"   feature-valued lots with a late bid         {len(feat):>6}")
    print(f"   flagged                                     {len(f_flag):>6}")
    print(f"   would have won                              {len(f_won):>6}"
          f"   ({len(f_won) / days * 7:.1f} per week)")
    if len(f_won):
        disc = (1 - f_won["final"] / f_won["fair"]).median()
        print(f"   they closed a median {disc:.0%} below the typical price of similar listings")
        print("   Open every one of these. Is it a plain listing of a decent watch,")
        print("   or is the reason it went cheap visible in the photos?")
        print(f"   {'closed':<8}{'line':<15}{'ref':<14}{'final':>7}{'typical':>9}  seller")
        for _, r in f_won.sort_values("final").head(12).iterrows():
            print(f"   {r['closed']:%d %b}  {r['line']:<15}{str(r['reference'])[:13]:<14}"
                  f"{r['final']:>7.0f}{r['fair']:>9.0f}  {str(r.get('seller_name') or '')[:22]}")
            print(f"   {'':<8}{r.get('url') or ''}")


if __name__ == "__main__":
    df, info = replay()
    report(df, info)
    if not df.empty:
        days = (info["end"] - info["start"]).total_seconds() / 86400
        report_features(df[df["basis"] == "feature"], df[df["basis"] != "feature"], days)

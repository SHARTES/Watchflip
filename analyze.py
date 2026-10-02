"""One-off analysis: what does the system look like at a higher budget?

    python analyze.py

Three parts, all read-only:

  A. Funnel     how many target watches exist, and how many survive each step
                (sold, affordable, good condition, seen before close, priced)
  B. Models     the hammer and listing models trained on ALL watches versus
                trained on the target slice only, compared on the same future
                window of target lots
  C. Replay     the last TEST_DAYS days replayed with the better model under
                the settings below, split by valuation tier, with a resale
                sensitivity and what €1,500 of capital could actually carry

Settings are deliberately moderate rather than conservative. Change them here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import backtest as bt
import db
import model
import shortlist as sl
from config import cfg

# ------------------------------------------------------------------ settings
BUDGET_ALL_IN = 700.0      # max total cost per watch, fees and shipping included
TEST_DAYS = 7              # replay window; everything before it is training data
CAPITAL = 1500.0

SETTINGS = {               # (resale share of the value estimate, required margin)
    "ebay":     (1.00, 0.20),   # value is already the lower-quarter ask
    "catawiki": (1.00, 0.20),   # same scale as eBay: the ratio is measured
    "feature":  (0.90, 0.30),   # 10% below the evidence tiers
}
SENSITIVITY = (0.85, 0.95, 1.05)   # resale shares to test for evidence-valued lots
SELL_WEEKS = 4                     # assumed time to sell one watch

object.__setattr__(cfg, "max_all_in_cost", BUDGET_ALL_IN)
MAX_HAMMER = (BUDGET_ALL_IN - sl.fixed_in()) / sl.fee_mult()   # EU seller


def line(label, value, note=""):
    print(f"   {label:<46}{value:>8}{('   ' + note) if note else ''}")


# ------------------------------------------------------------------ A. funnel
def funnel(sold: pd.DataFrame) -> None:
    with db.connect() as conn:
        rows = conn.execute("""
            -- analyze funnel
            select l.lot_id, l.brand, l.gender, l.watch_year, l.watch_period,
                   l.close_time, r.sold, r.lot_id as has_result
            from lots l left join lot_results r using (lot_id)
        """).fetchall()
    allx = pd.DataFrame(rows)
    allx["watch_year"] = pd.to_numeric(allx["watch_year"], errors="coerce")
    allx["close_time"] = pd.to_datetime(allx["close_time"], utc=True)
    t_all = allx[model.target_mask(allx)]
    now = pd.Timestamp.now(tz="UTC")

    t = sold[model.target_mask(sold)].copy()
    t["cond_ok"] = t.apply(sl.passes_filters, axis=1)
    afford = t["final_price"] <= MAX_HAMMER

    ref_to_line = {}
    for ref, wm in zip(sold["reference_number"], sold["watch_model"]):
        k = model.ref_key(ref)
        if k:
            ref_to_line.setdefault(k, sl.line_of(wm))
    by_ref, _ = sl.load_asks(ref_to_line)
    cw = sl.catawiki_by_ref(sold)
    k = t["reference_number"].map(model.ref_key)
    has_ebay = t.apply(lambda lot: sl.ebay_value(lot, by_ref) is not None, axis=1)
    has_cw = t.apply(lambda lot: sl.catawiki_value(lot, cw) is not None, axis=1)
    evidence = has_ebay | has_cw

    base = t[afford & t["cond_ok"]]
    late = base["late_bid"].notna()
    ev = evidence[base.index]

    print("A. FUNNEL — vintage men's/unisex Omega, whole history")
    print(f"   budget €{BUDGET_ALL_IN:.0f} all-in → max hammer ≈ €{MAX_HAMMER:.0f} (EU seller)\n")
    line("target lots collected, all states", f"{len(t_all):,}")
    line("  still open right now", f"{int((t_all['has_result'].isna() & (t_all['close_time'] > now)).sum()):,}")
    line("  closed and sold", f"{len(t):,}")
    line("sold at or under the max hammer", f"{int(afford.sum()):,}",
         f"{afford.mean():.0%} of sold")
    line("  … and condition Good or better", f"{len(base):,}")
    line("  … with a bid seen 1–5 h before close", f"{int(late.sum()):,}",
         f"{late.mean():.0%} — the rest closed unseen")
    line("  … priced from eBay asks", f"{int(has_ebay[base.index].sum()):,}")
    line("  … priced from earlier Catawiki sales", f"{int((has_cw & ~has_ebay)[base.index].sum()):,}")
    line("  … any reference evidence", f"{int(ev.sum()):,}", f"{ev.mean():.0%}")
    line("  … seen before close AND evidence", f"{int((late & ev).sum()):,}",
         "what the shortlist can fully price")
    print()


# ------------------------------------------------------------------ B. models
def fit_variant(train, test, only_target: bool, seller: bool = False):
    tr = train[model.target_mask(train)].reset_index(drop=True) if only_target else train
    keep = sl.USE_SELLER
    sl.USE_SELLER = seller
    try:
        live, X_te, te_b, adj, counts, fair = sl.build_live_model(tr, test)
    finally:
        sl.USE_SELLER = keep
    return live.predict(X_te), fair, te_b, len(tr), live


def coverage(y, p20, p80):
    below = (y < p20).mean() * 100
    above = (y > p80).mean() * 100
    return below, 100 - below - above, above


def compare_models(train, test):
    sl.CAL_DAYS = 3
    tm = model.target_mask(test)
    y = test["final_price"].to_numpy(float)
    late = test["late_bid"].notna().to_numpy()

    gm = train["final_price"].median()
    t_tr = train[model.target_mask(train)]
    base = test["watch_model"].map(t_tr.groupby("watch_model")["final_price"].median())
    base = base.fillna(t_tr["final_price"].median()).to_numpy(float)

    print(f"B. MODELS — trained before {test['close_time'].min():%d %b}, "
          f"tested on {int(tm.sum())} target lots after it\n")
    print(f"   {'':<30}{'train n':>8}{'all':>8}{'late bid':>10}{'no late':>9}"
          f"{'listing':>9}{'band in/60':>12}")
    print(f"   {'baseline: target line median':<30}{len(t_tr):>8}"
          f"{model.mape(base[tm], y[tm]):>7.1f}%")

    results = {}
    variants = (("trained on all watches", False, False),
                ("trained on target only", True, False),
                ("all watches + seller", False, True),
                ("target only + seller", True, True))
    for name, only, seller in variants:
        pred, fair, te_b, n, live = fit_variant(train, test, only, seller)
        p50 = pred["p50"].to_numpy(); p20 = pred["p20"].to_numpy(); p80 = pred["p80"].to_numpy()
        fv = fair.to_numpy()
        _, inside, _ = coverage(y[tm], p20[tm], p80[tm])
        print(f"   {name:<30}{n:>8}"
              f"{model.mape(p50[tm], y[tm]):>7.1f}%"
              f"{model.mape(p50[tm & late], y[tm & late]):>9.1f}%"
              f"{model.mape(p50[tm & ~late], y[tm & ~late]):>8.1f}%"
              f"{model.mape(fv[tm], y[tm]):>8.1f}%"
              f"{inside:>11.0f}%")
        results[name] = (pred, fair, te_b, model.mape(p50[tm], y[tm]), live, seller)
    print("   'listing' = the bid-free model behind the feature tier. Lower is better;")
    print("   'band in' is the share of prices inside p20–p80, which should be about 60%.\n")

    gains = [results[a][3] - results[b][3] for a, b in
             (("trained on all watches", "all watches + seller"),
              ("trained on target only", "target only + seller"))]
    if min(gains) >= 0.3:
        print(f"   seller helps on both training sets ({gains[0]:+.1f} / {gains[1]:+.1f} points)"
              f" → worth turning on: USE_SELLER = True in shortlist.py")
    else:
        print(f"   seller does not clearly help ({gains[0]:+.1f} / {gains[1]:+.1f} points)"
              f" → leave USE_SELLER off")
    # Part C replays what the live shortlist would do, so it only picks among
    # the variants matching the live seller setting.
    pool = {k: v for k, v in results.items() if v[5] == sl.USE_SELLER}
    best = min(pool, key=lambda k: pool[k][3])
    print(f"   → using the model {best} for part C\n")
    pred, fair, te_b, _, live, _ = pool[best]
    return pred, fair, te_b, live


# ------------------------------------------------------------------ C. replay
def replay(train, test, pred, fair, te_b, live):
    ref_to_line = {}
    for ref, wm in zip(pd.concat([train, test])["reference_number"],
                       pd.concat([train, test])["watch_model"]):
        k = model.ref_key(ref)
        if k:
            ref_to_line.setdefault(k, sl.line_of(wm))
    by_ref, by_line = sl.load_asks(ref_to_line)
    cw_train = sl.catawiki_by_ref(train)
    sl.calibrate_ratio(train, by_ref)
    sl.calibrate_feature_bias(live, by_ref)
    sl.calibrate_margin(train, by_ref, cw_train)
    tm = model.target_mask(te_b)

    rows = []
    for i, (idx, lot) in enumerate(te_b.iterrows()):
        if not tm[i] or not sl.passes_filters(lot) or pd.isna(lot["late_bid"]):
            continue
        v = sl.value_for(lot, by_ref, by_line, cw_train, fair.get(idx))
        if v is None:
            continue
        value, _, _, kind = v
        non_eu = sl.eu_status(lot["seller_country"]) == "non_eu"
        p20, p50, p80 = (float(pred.loc[idx, c]) for c in ("p20", "p50", "p80"))
        rows.append({"lot_id": lot["lot_id"], "idx": idx, "kind": kind, "value": value,
                     "sigma": getattr(v, "sigma", np.nan),
                     "non_eu": non_eu, "current": float(lot["late_bid"]),
                     "final": float(lot["final_price"]), "p20": p20, "p50": p50, "p80": p80,
                     "fair": float(fair.get(idx, np.nan)), "line": sl.line_of(lot["watch_model"]),
                     "reference": lot["reference_number"] if isinstance(lot["reference_number"], str) else "—",
                     "closed": lot["close_time"]})
    df = pd.DataFrame(rows)
    return df


def evaluate(df, realism_by_kind, rule: str = "fixed"):
    """rule 'fixed': the margins in SETTINGS. rule 'uncertainty': margin from
    how sure each value is (shortlist.margin_for), the live default."""
    out = []
    for _, r in df.iterrows():
        realism, margin = realism_by_kind[r["kind"]]
        if rule == "uncertainty":
            keep = sl.USE_UNCERTAINTY_MARGIN
            sl.USE_UNCERTAINTY_MARGIN = True
            margin = sl.margin_for(r["kind"], r.get("sigma"), center=margin)
            sl.USE_UNCERTAINTY_MARGIN = keep
        ceiling, net = sl.max_bid(r["value"], realism, r["non_eu"], margin)
        p_win = sl.p_under(ceiling, r["p20"], r["p50"], r["p80"]) if ceiling > r["current"] else 0.0
        flagged = ceiling > r["current"] and p_win >= sl.MIN_WIN_PROB
        won = flagged and r["final"] <= ceiling
        land = sl.landed(r["final"], r["non_eu"])
        out.append({**r.to_dict(), "ceiling": ceiling, "p_win": p_win, "flagged": flagged,
                    "won": won, "landed": land, "profit": net - land})
    return pd.DataFrame(out)


def report_replay(df, days):
    print(f"C. REPLAY — last {days:.0f} days, budget €{BUDGET_ALL_IN:.0f}, moderate settings")
    print(f"   {sl.ratio_text()}")
    print(f"   {sl.bias_text()}")
    c = sl.CHANNELS[sl.PLAN_CHANNEL]
    print(f"   profit planned on {sl.PLAN_CHANNEL} (fee {c['fee']:.1%} + €{c['fixed'] + c['ship']:.0f}) "
          f"after a {sl.WARRANTY_RESERVE:.0%} reserve; inbound shipping €{cfg.expected_inbound_shipping:.0f}")
    for k, (r, m) in SETTINGS.items():
        print(f"   {k:<9} resale {r:.0%} of value, margin {m:.0%}")
    print()
    if df.empty:
        print("   nothing to replay")
        return None

    ev = evaluate(df, SETTINGS)
    print(f"   {'tier':<22}{'priced':>7}{'flagged':>9}{'won':>6}{'expected':>10}"
          f"{'profit':>9}{'per watch':>11}{'losers':>8}")
    for kind, label in (("ebay", "eBay asks"), ("catawiki", "Catawiki sales"),
                        ("feature", "features (unverified)")):
        s = ev[ev["kind"] == kind]
        f = s[s["flagged"]]
        w = s[s["won"]]
        prof = f"{w['profit'].sum():>9.0f}" if len(w) else f"{'—':>9}"
        per = f"{w['profit'].mean():>11.0f}" if len(w) else f"{'—':>11}"
        print(f"   {label:<22}{len(s):>7}{len(f):>9}{len(w):>6}{f['p_win'].sum():>10.1f}"
              f"{prof}{per}{int((w['profit'] < 0).sum()):>8}")
    print("   'expected' = wins the model predicted; close to 'won' means honest odds.")
    print("   Feature-tier profit is the model agreeing with itself — treat it as a lead count.\n")

    # Which margin rule? Same lots, same model — only the required margin differs.
    # Stress = keep exactly the bids each rule made, then assume every resale
    # actually came in 15% below the estimate. A good rule loses less there.
    print("   margin rule, evidence-valued lots:")
    print(f"   {'rule':<14}{'avg margin':>11}{'won':>6}{'profit':>9}{'losers':>8}"
          f"{'if resale −15%':>16}{'losers':>8}")
    evid = df[df["kind"] != "feature"]
    for rule in ("fixed", "uncertainty"):
        e = evaluate(evid, SETTINGS, rule)
        w = e[e["won"]]
        if len(w):
            realism = w["kind"].map(lambda k: SETTINGS[k][0])
            sale = w["value"] * realism * 0.85
            stress = sale.map(sl.net_from) - w["landed"]
        else:
            stress = pd.Series(dtype=float)
        avg_m = evid.apply(lambda r: sl.margin_for(r["kind"], r.get("sigma"), center=SETTINGS[r["kind"]][1])
                           if rule == "uncertainty" else SETTINGS[r["kind"]][1], axis=1).mean()
        print(f"   {rule:<14}{avg_m:>11.0%}{len(w):>6}{w['profit'].sum():>9.0f}"
              f"{int((w['profit'] < 0).sum()):>8}{stress.sum():>16.0f}{int((stress < 0).sum()):>8}")
    print("   Prefer the rule with more profit in both columns; if they split, the")
    print("   −15% column matters more until real sales confirm the resale level.\n")

    both = ev[ev["kind"] != "feature"].dropna(subset=["fair"])
    if len(both):
        for label, div in (("before correction", 1.0), ("after correction ", sl.FEAT_BIAS)):
            gap = (both["fair"] * sl.ASK_TO_HAMMER / div / both["value"] - 1)
            print(f"   feature estimate vs real evidence, {label} ({len(both)} lots): "
                  f"median gap {gap.abs().median():.0%}, "
                  f"{'too high' if gap.median() > 0 else 'too low'} by {abs(gap.median()):.0%}")
        print("   'after' should sit near 0%. The correction was measured on earlier")
        print("   lots, so this is a fair out-of-sample check.\n")

    print("   resale sensitivity, evidence-valued lots only:")
    print(f"   {'resale':>9}{'won':>6}{'profit':>9}{'per watch':>11}{'per week':>10}")
    for r in SENSITIVITY:
        s = evaluate(df[df["kind"] != "feature"],
                     {**SETTINGS, "ebay": (r, SETTINGS["ebay"][1]),
                      "catawiki": (r, SETTINGS["catawiki"][1])})
        w = s[s["won"]]
        per = w["profit"].mean() if len(w) else 0
        print(f"   {r:>9.0%}{len(w):>6}{w['profit'].sum():>9.0f}{per:>11.0f}{len(w) / days * 7:>10.1f}")
    print()

    w = ev[ev["won"] & (ev["kind"] != "feature")]
    if len(w):
        per_week = len(w) / days * 7
        avg_land = w["landed"].mean()
        carry = CAPITAL / avg_land
        can_buy = carry / SELL_WEEKS
        limit = min(per_week, can_buy)
        print(f"   capital check with €{CAPITAL:,.0f}:")
        print(f"     average landed cost of a win                €{avg_land:,.0f}")
        print(f"     watches you can hold at once                {carry:.1f}")
        print(f"     buys per week if each sells in {SELL_WEEKS} weeks       {can_buy:.1f}")
        print(f"     winnable per week (evidence tiers)          {per_week:.1f}")
        print(f"     → realistic pace {limit:.1f}/week, about "
              f"€{limit * w['profit'].mean() * 4.3:,.0f}/month if resale holds\n")

    wins = ev[ev["won"]].sort_values("profit", ascending=False)
    if len(wins):
        info = bt.seller_info(wins["lot_id"])
        wins = wins.join(info, on="lot_id")
        print(f"   won lots ({len(wins)}) — open them and judge:")
        print(f"   {'closed':<8}{'tier':<10}{'line':<15}{'ref':<14}{'final':>7}{'max':>7}{'profit':>8}  seller")
        for _, r in wins.head(20).iterrows():
            print(f"   {r['closed']:%d %b}  {r['kind']:<10}{r['line']:<15}{str(r['reference'])[:13]:<14}"
                  f"{r['final']:>7.0f}{r['ceiling']:>7.0f}{r['profit']:>8.0f}  "
                  f"{str(r.get('seller_name') or '')[:20]}")
            print(f"   {'':<8}{r.get('url') or ''}")
    return ev


def main():
    sold = model.load()
    funnel(sold)

    end = sold["close_time"].max()
    start = end - pd.Timedelta(days=TEST_DAYS)
    train = sold[sold["close_time"] < start].reset_index(drop=True)
    test = sold[sold["close_time"] >= start].reset_index(drop=True)

    pred, fair, te_b, live = compare_models(train, test)
    df = replay(train, test, pred, fair, te_b, live)
    report_replay(df, (end - start).total_seconds() / 86400)


if __name__ == "__main__":
    main()

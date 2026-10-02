"""Explain one lot: where its value and profit come from.

    python explain.py 106866757
    python explain.py https://www.catawiki.com/en/l/106866757-omega-de-ville-...

Prints, for a single lot:

  1. the eBay listings behind its value — every one, with price, link and a
     warning when it does not count (a spare part, another currency, a gold
     or two-tone variant against a steel watch, a far outlier)
  2. earlier Catawiki sales of the same reference
  3. the value the system used: the lower quarter of the listings that count
     (✗ marks the ones that do not), next to the raw median for comparison
  4. the full cost chain from that value down to the maximum bid, and the
     cash profit on each selling channel (in person, Chrono24, eBay)

Nothing is changed. It uses the same settings as the live shortlist.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd

import asks
import db
import model
import shortlist as sl

def lot_id_from(arg: str) -> str:
    m = re.search(r"/l/(\d+)", arg)
    return m.group(1) if m else arg.strip()


def load_lot(lot_id: str) -> pd.DataFrame:
    select = sl.OPEN_QUERY.split("from lots l")[0]
    q = select + """,
    r.final_price, r.sold
from lots l
left join lot_results r using (lot_id)
where l.lot_id = %s
"""
    with db.connect() as conn:
        rows = conn.execute(model.with_photos(conn, q), (lot_id,)).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    for col in ("estimate_low", "estimate_high", "case_diameter_mm", "bid_24h", "late_bid",
                "bid_count", "photo_count", "watch_year", "current_bid", "final_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ebay_link(comp_id: str) -> str:
    m = re.search(r"\|(\d+)\|", str(comp_id))
    return f"https://www.ebay.de/itm/{m.group(1)}" if m else str(comp_id)


def flags_for(row, lot_material: str, median: float, line=None) -> list[str]:
    f = [x for x in asks.listing_flags(row.get("title"), row.get("currency"))
         if x != "gold/two-tone"]
    why = asks.mismatch(row.get("title"), lot_material, line)
    if why:
        f.append(why)
    if median and (row["price"] > asks.OUTLIER_HIGH * median
                   or row["price"] < asks.OUTLIER_LOW * median):
        f.append("outlier")
    return f


def cost_chain(value: float, realism: float, margin: float, non_eu: bool, bid: float | None,
               list_price: float | None = None):
    sale = value * realism
    ceiling, net = sl.max_bid(value, realism, non_eu, margin)
    lp = list_price if list_price and list_price > sale * 1.03 else None

    def row(label, amount):
        print(f"   {label:<58}{amount:>10}")

    row("value (lower quarter of comparable asks)", f"€{value:,.0f}")
    row(f"× {realism:.0%} = quick-sale price (priced to move, after offers)", f"€{sale:,.0f}")
    if lp:
        row("first list price (middle of comparable asks)", f"€{lp:,.0f}")
    print()
    eur = sl._eur

    cols = [sale] + ([lp] if lp else [])
    print(f"   {'what reaches you, by channel':<34}{'fee':>12}{'shipping':>10}"
          + "".join(f"{'at ' + eur(p):>11}" for p in cols))
    for c, s in sl.CHANNELS.items():
        fee = f"{s['fee']:.1%}" + (f"+€{s['fixed']:.2f}" if s["fixed"] else "")
        name = c + (" ← max bid planned here" if c == sl.PLAN_CHANNEL else "")
        print(f"   {name:<34}{fee:>12}{eur(s['ship']):>10}"
              + "".join(f"{eur(sl.keep(p, c)):>11}" for p in cols))
    print()
    row(f"planning net: {sl.PLAN_CHANNEL} at €{sale:,.0f} − {sl.WARRANTY_RESERVE:.0%} reserve "
        f"(€{sl.WARRANTY_RESERVE * sale:,.0f})", f"€{net:,.0f}")
    print("   (the reserve is the average cost of repairs and returns — not paid on")
    print("    every watch, so it is left out of the cash profits below)")
    vat = (f" + {sl.IMPORT_VAT:.0%} VAT on hammer + shipping + €{sl.IMPORT_CLEARANCE:.0f} customs handling"
           if non_eu else "")
    print(f"   landed cost = hammer × {sl.fee_mult():.2f} + €{sl.fixed_in():.0f} "
          f"(fee €{sl.cfg.catawiki_buyer_fixed_fee:.0f} + shipping "
          f"€{sl.cfg.expected_inbound_shipping:.0f}){vat}")
    row(f"max bid: {margin:.0%} margin on the planning net, ≤ €{sl.cfg.max_all_in_cost:.0f} all-in",
        f"€{ceiling:,.0f}")
    print()
    points = [("at your max bid", ceiling)]
    if bid:
        points.insert(0, ("at the price shown", bid))
    prices = [sale] + ([lp] if lp else [])
    for label, h in points:
        land = sl.landed(h, non_eu)
        print(f"   {label}: hammer €{h:,.0f} → landed €{land:,.0f}")
        for p in prices:
            cash = " · ".join(f"{c} {sl._eur(sl.keep(p, c) - land)}" for c in sl.CHANNELS)
            print(f"      cash profit if it sells for €{p:,.0f}:  {cash}")


def main(arg: str) -> None:
    lot_id = lot_id_from(arg)
    lot = load_lot(lot_id)
    if lot.empty:
        sys.exit(f"lot {lot_id} is not in the database")
    r = lot.iloc[0]
    key = model.ref_key(r["reference_number"])
    text = f"{r['title'] or ''} {r['description'] or ''}"
    material = model.lot_material(r)
    line = asks.lot_line(r)
    serviced = asks.lot_serviced(r)
    non_eu = sl.eu_status(r["seller_country"]) == "non_eu"

    print(f"{r['title']}")
    print(f"{r['url']}")
    state = (f"sold for €{r['final_price']:.0f}" if pd.notna(r["final_price"])
             else f"open, current bid €{r['current_bid']:.0f}" if pd.notna(r["current_bid"])
             else "no bid seen")
    print(f"reference {r['reference_number'] or '—'} (key {key or '—'}) · {line or 'line ?'} · material {material} · "
          f"{'serviced' if serviced else 'as-is (no service stated)'} · "
          f"condition {r['watch_condition']} · seller {r['seller_country'] or '?'}"
          f"{' (non-EU)' if non_eu else ''} · {state}\n")

    # 1. eBay
    asks_all = asks.load().get(key) if key else None
    asks_df = asks_all if asks_all is not None else pd.DataFrame()
    if len(asks_df):
        dealer = asks.service_mismatch(asks_df, material, line, serviced)
        keep = asks.usable(asks_df, material, line) & ~dealer
        no_out = asks.drop_outliers(asks_df.loc[keep, "price"])
        kept = asks.drop_repeats(no_out)
        median = float(asks_df["price"].median())
        print(f"1. eBay listings for this reference — {len(asks_df)} live, median €{median:.0f}")
        clean_med = float(kept.median()) if len(kept) else float("nan")
        for i, a in asks_df.iterrows():
            f = flags_for(a, material, float(no_out.median()) if len(no_out) else float("nan"), line)
            if dealer[i]:
                f = [x for x in f if x != "serviced/warranty"] + ["serviced/warranty, yours is as-is"]
            if i in no_out.index and i not in kept.index:
                f.append("same price as another ask")
            used = "  " if i in kept.index else "✗ "
            mark = f"  ⚠ {', '.join(f)}" if f else ""
            print(f"   {used}€{a['price']:>6.0f} {str(a.get('currency') or ''):<4}"
                  f"{str(a['title'])[:60]:<61}{mark}")
            print(f"            {ebay_link(a['comp_id'])}")
        if len(kept):
            print(f"   ✗ = not counted. Counted {len(kept)} of {len(asks_df)}: "
                  f"cheapest €{kept.min():.0f}, lower quarter €{np.quantile(kept, asks.VALUE_QUANTILE):.0f}, "
                  f"median €{clean_med:.0f}")
        else:
            print("   none of them count for this watch")
        print("   These are ASKING prices. A new seller has to list near the cheaper end;")
        print("   dealer listings marked serviced/warranty sit at the top"
              + (" and do not count for an as-is watch while 3+ as-is asks remain.\n"
                 if not serviced else ".\n"))
    else:
        print("1. eBay: no live listings stored for this reference\n")

    # 2. Catawiki
    closed = model.load()
    same = closed[closed["reference_number"].map(model.ref_key) == key] if key else closed.iloc[0:0]
    same = same[same["lot_id"] != lot_id]
    if len(same):
        def why_not(s_):
            m = model.lot_material(s_)
            if material in ("steel", "plated", "gold", "bicolor") and m not in (material, "unknown", "other"):
                return f"{m}, yours is {material}"
            return asks.line_mismatch(f"{s_['watch_model'] or ''} {s_['title'] or ''}", line)
        reasons = same.apply(why_not, axis=1)
        counted_cw = same[reasons.isna()]
        print(f"2. Catawiki sales of the same reference — {len(same)}, "
              f"{len(counted_cw)} comparable"
              + (f", median of those €{counted_cw['final_price'].median():.0f}" if len(counted_cw) else ""))
        recent = same.sort_values("close_time").tail(12)
        if len(same) > len(recent):
            print(f"   (latest {len(recent)} shown)")
        for i, s_ in recent.iterrows():
            mark = "✗ " if reasons[i] else "  "
            note = f"  ⚠ {reasons[i]}" if reasons[i] else ""
            print(f"   {mark}{s_['close_time']:%d %b}  €{s_['final_price']:>6.0f}  "
                  f"{str(s_['title'])[:56]:<57}{note}")
        if len(counted_cw) < sl.MIN_REF_SALES:
            print(f"   fewer than {sl.MIN_REF_SALES} comparable sales — not used for the value")
        print()
    else:
        print("2. Catawiki: no other sales of this reference yet\n")

    # 3. value the system uses
    by_ref, by_line = sl.load_asks({})
    cw = sl.catawiki_by_ref(closed[closed["lot_id"] != lot_id])
    sl.calibrate_ratio(closed[closed["lot_id"] != lot_id], by_ref)
    sl.calibrate_margin(closed[closed["lot_id"] != lot_id], by_ref, cw)
    fair = None
    v = sl.value_for(r, by_ref, by_line, cw)
    if v is None and sl.USE_FEATURE_TIER:
        live, X, ob, _, _, fair_s = sl.build_live_model(closed[closed["lot_id"] != lot_id],
                                                        lot.drop(columns=["final_price", "sold"]))
        sl.calibrate_feature_bias(live, by_ref)
        fair = float(fair_s.iloc[0])
        v = sl.value_for(r, by_ref, by_line, cw, fair)
    if v is None:
        print("3. No value — the shortlist skips this lot.")
        return
    value, realism, basis, kind = v
    sigma = getattr(v, "sigma", None)
    margin = sl.margin_for(kind, sigma)
    print(f"3. Value used: €{value:.0f} from {basis}")
    if kind == "ebay" and "combined" in basis:
        print("   = geometric mean of the two: the eBay value (lower quarter of comparable")
        print(f"     asks) and comparable Catawiki sales × {sl.ASK_TO_HAMMER:.2f} "
              f"(the measured eBay-to-auction ratio)")
    elif kind == "ebay":
        print(f"   = lower quarter of the counted eBay asks (the raw median would be "
              f"€{asks_df['price'].median():.0f})")
    if kind == "catawiki":
        print(f"   = Catawiki median × {sl.ASK_TO_HAMMER:.2f}")
        print(f"   ({sl.ratio_text()})")
    if kind == "feature":
        print(f"   = typical Catawiki price for similar listings €{fair:.0f} × {sl.ASK_TO_HAMMER:.2f}"
              f" ÷ {sl.FEAT_BIAS:.2f}")
        print(f"   ({sl.ratio_text()};")
        print(f"    {sl.bias_text()})")
    if sigma is not None:
        ref = sl.SIGMA_REF.get("evidence")
        typ = f" (typical ±{ref:.0%})" if ref is not None and kind != "feature" else ""
        print(f"   uncertainty of this value: ±{sigma:.0%}{typ} → required margin {margin:.0%}")
    print()

    # 4. cost chain
    print(f"4. From value to profit  (settings: {kind} tier, resale {realism:.0%}, margin {margin:.0%})")
    bid = r["final_price"] if pd.notna(r["final_price"]) else r["current_bid"]
    cost_chain(value, realism, margin, non_eu, float(bid) if pd.notna(bid) else None,
               getattr(v, "list_price", None))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

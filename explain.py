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
  4. the full cost chain from that value down to the maximum bid and profit

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


def flags_for(row, lot_material: str, median: float) -> list[str]:
    f = [x for x in asks.listing_flags(row.get("title"), row.get("currency"))
         if x != "gold/two-tone"]
    why = asks.mismatch(row.get("title"), lot_material)
    if why:
        f.append(why)
    if median and (row["price"] > asks.OUTLIER_HIGH * median
                   or row["price"] < asks.OUTLIER_LOW * median):
        f.append("outlier")
    return f


def cost_chain(value: float, realism: float, margin: float, non_eu: bool, bid: float | None):
    sale = value * realism
    reserve = sale * sl.WARRANTY_RESERVE
    fee = sale * sl.SALE_FEE
    net = sale - reserve - fee - sl.OUTBOUND_SHIPPING
    ceiling, _ = sl.max_bid(value, realism, non_eu, margin)

    def row(label, amount):
        print(f"   {label:<44}{amount:>10}")

    row("value (what it fetches, as listed)", f"€{value:,.0f}")
    row(f"× {realism:.0%} you realistically get", f"€{sale:,.0f}")
    row(f"− {sl.SALE_FEE:.0%} platform fee", f"−€{fee:,.0f}")
    row(f"− {sl.WARRANTY_RESERVE:.0%} warranty/return reserve", f"−€{reserve:,.0f}")
    row("− outbound insured shipping", f"−€{sl.OUTBOUND_SHIPPING:,.0f}")
    row("= net from the sale", f"€{net:,.0f}")
    print()
    vat = " + 20% import VAT" if non_eu else ""
    print(f"   landed cost = hammer × {sl.fee_mult():.2f} + €{sl.fixed_in():.0f} "
          f"(fee €{sl.cfg.catawiki_buyer_fixed_fee:.0f} + shipping "
          f"€{sl.cfg.expected_inbound_shipping:.0f}){vat}")
    row(f"max bid: {margin:.0%} margin, ≤ €{sl.cfg.max_all_in_cost:.0f} all-in", f"€{ceiling:,.0f}")
    points = [("at your max bid", ceiling)]
    if bid:
        points.insert(0, ("at the bid shown", bid))
    for label, h in points:
        land = sl.landed(h, non_eu)
        print(f"   profit {label:<17} hammer €{h:>5,.0f} → landed €{land:>5,.0f} → "
              f"profit €{net - land:>5,.0f} ({(net - land) / land:>4.0%})")


def main(arg: str) -> None:
    lot_id = lot_id_from(arg)
    lot = load_lot(lot_id)
    if lot.empty:
        sys.exit(f"lot {lot_id} is not in the database")
    r = lot.iloc[0]
    key = model.ref_key(r["reference_number"])
    text = f"{r['title'] or ''} {r['description'] or ''}"
    material = model.lot_material(r)
    non_eu = sl.eu_status(r["seller_country"]) == "non_eu"

    print(f"{r['title']}")
    print(f"{r['url']}")
    state = (f"sold for €{r['final_price']:.0f}" if pd.notna(r["final_price"])
             else f"open, current bid €{r['current_bid']:.0f}" if pd.notna(r["current_bid"])
             else "no bid seen")
    print(f"reference {r['reference_number'] or '—'} (key {key or '—'}) · material {material} · "
          f"condition {r['watch_condition']} · seller {r['seller_country'] or '?'}"
          f"{' (non-EU)' if non_eu else ''} · {state}\n")

    # 1. eBay
    asks_all = asks.load().get(key) if key else None
    asks_df = asks_all if asks_all is not None else pd.DataFrame()
    if len(asks_df):
        keep = asks.usable(asks_df, material)
        no_out = asks.drop_outliers(asks_df.loc[keep, "price"])
        kept = asks.drop_repeats(no_out)
        median = float(asks_df["price"].median())
        print(f"1. eBay listings for this reference — {len(asks_df)} live, median €{median:.0f}")
        clean_med = float(kept.median()) if len(kept) else float("nan")
        for i, a in asks_df.iterrows():
            f = flags_for(a, material, float(no_out.median()) if len(no_out) else float("nan"))
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
        print("   dealer listings marked serviced/warranty sit at the top.\n")
    else:
        print("1. eBay: no live listings stored for this reference\n")

    # 2. Catawiki
    closed = model.load()
    same = closed[closed["reference_number"].map(model.ref_key) == key] if key else closed.iloc[0:0]
    same = same[same["lot_id"] != lot_id]
    if len(same):
        print(f"2. Catawiki sales of the same reference — {len(same)}, "
              f"median €{same['final_price'].median():.0f}")
        recent = same.sort_values("close_time").tail(12)
        if len(same) > len(recent):
            print(f"   (latest {len(recent)} shown)")
        for _, s in recent.iterrows():
            print(f"   {s['close_time']:%d %b}  €{s['final_price']:>6.0f}  {str(s['title'])[:70]}")
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
    cost_chain(value, realism, margin, non_eu, float(bid) if pd.notna(bid) else None)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

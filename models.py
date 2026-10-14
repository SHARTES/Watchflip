"""Which models to specialise in. Read-only.

    python models.py              # vintage Omega references, best first
    python models.py --min 3      # include references with 3+ sales (default 4)
    python models.py --weeks 6    # look back 6 weeks (default: everything)
    python models.py --top 40

For every reference that sold often enough on Catawiki, using the same
valuation as the alerts (eBay + logged prices, Catawiki sales, competition
cap, confidence grade):

  sold/wk     how often it comes up (sold lots per week)
  hammer      median hammer price
  resale      quick-sale price the alerts would plan with
  max bid     the alert's max bid for it: 25% margin on Chrono24 after fees
              and the 5% reserve, inside your all-in budget
  ≤max/wk     how many sales a week closed at or below that max bid — the
              deals you could actually have won
  profit      cash profit per such deal, sold in person at the quick-sale price
              (Chrono24 takes about 6.5% + shipping off that)
  €/wk        ≤max/wk × profit: what the model offered you per week, had you
              won every lot under your max. The ranking.

A good specialty: 1+ deals a week, €100+ profit each, grade A or B, steel,
and a watch you like. Five to ten of those, known inside out, beat a hundred
references you have to research from scratch for every alert.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import model
import shortlist as sl


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min", type=int, default=4, help="minimum sales of a reference (default 4)")
    p.add_argument("--weeks", type=float, default=None, help="look back this many weeks")
    p.add_argument("--top", type=int, default=30)
    a = p.parse_args()

    closed = model.load()
    by_ref, by_line = sl.load_asks({})
    cw = sl.catawiki_by_ref(closed)
    sl.calibrate_ratio(closed, by_ref)
    sl.calibrate_margin(closed, by_ref, cw)

    t = closed[model.target_mask(closed)].copy()
    if a.weeks:
        t = t[t["close_time"] > t["close_time"].max() - pd.Timedelta(weeks=a.weeks)]
    t = t[t.apply(sl.passes_filters, axis=1)]
    t["k"] = t["reference_number"].map(model.ref_key)
    t = t.dropna(subset=["k"])
    weeks = max((t["close_time"].max() - t["close_time"].min()).days / 7, 1)

    rows, no_value = [], 0
    for k, g in t.groupby("k"):
        if len(g) < a.min:
            continue
        g = g.sort_values("close_time")
        lot = g.iloc[-1]                        # the latest sale stands for the reference
        v = sl.value_for(lot, by_ref, by_line, cw)
        if v is None:
            no_value += 1
            continue
        value, realism = v[0], v[1]
        quick = value * realism
        ceiling, _ = sl.max_bid(value, realism, False, sl.margin_for(v[3], v.sigma))
        prices = g["final_price"].astype(float)
        under = prices[prices <= ceiling]
        profit = (sl.keep(quick, "in person") - sl.landed(float(under.median()), False)
                  if len(under) else np.nan)
        mats = g.apply(model.lot_material, axis=1)
        rows.append({
            "ref": str(lot["reference_number"])[:14], "line": sl.line_of(lot["watch_model"]),
            "n": len(g), "per_wk": len(g) / weeks, "hammer": prices.median(),
            "resale": quick, "grade": v.grade, "max": ceiling,
            "under_wk": len(under) / weeks, "profit": profit,
            "eur_wk": (len(under) / weeks) * profit if len(under) else 0.0,
            "steel": (mats == "steel").mean(), "title": str(lot["title"])[:48],
        })
    if not rows:
        print("No reference has enough sales yet.")
        return
    df = pd.DataFrame(rows).sort_values("eur_wk", ascending=False)

    print(f"Vintage-Omega references with {a.min}+ sales over {weeks:.1f} weeks "
          f"({len(df)} valued, {no_value} without enough evidence to value)\n")
    print(f"{'#':>3} {'reference':<15}{'line':<14}{'sold/wk':>8}{'hammer':>8}{'resale':>8}{'grade':>6}"
          f"{'max bid':>8}{'≤max/wk':>8}{'profit':>8}{'€/wk':>7}{'steel':>7}")
    for i, (_, r) in enumerate(df.head(a.top).iterrows(), 1):
        prof = f"{r['profit']:.0f}" if np.isfinite(r["profit"]) else "—"
        print(f"{i:>3} {r['ref']:<15}{r['line'][:13]:<14}{r['per_wk']:>8.1f}{r['hammer']:>8.0f}"
              f"{r['resale']:>8.0f}{r['grade']:>6}{r['max']:>8.0f}{r['under_wk']:>8.1f}{prof:>8}"
              f"{r['eur_wk']:>7.0f}{r['steel']:>7.0%}")
    print("\nThe example title for each, to recognise it:")
    for i, (_, r) in enumerate(df.head(a.top).iterrows(), 1):
        print(f"{i:>3} {r['ref']:<15}{r['title']}")
    print("\nRead a line in full: python explain.py <lot id of any recent sale of it>")


if __name__ == "__main__":
    main()

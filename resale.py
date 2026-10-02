"""What sells on eBay, how fast, and at what price — read from vanished listings.

    python resale.py

Read-only. Real sale prices are not available (Marketplace Insights was
refused), so this reads the lifecycle of the asks comps.py keeps collecting: a
listing that vanishes probably sold near its last price; one that sits for
months is a wish. Not proof — sellers withdraw and eBay relists — so relists
(a new listing of the same reference at the same price within a week) are
taken out, and every number says how many listings it rests on.

  A. clearing level   where vanished asks sat among the comparable asks live
                      at the same moment. Tests the shortlist's quick-sale
                      price: lower quarter × 0.95.
  B. speed            how long vanished listings were up, and how fast each
                      line and material turns over — what to buy for the exit
  C. price cuts       how often and how far sellers lower their ask
  D. service premium  serviced/warranty asks against as-is asks of the same
                      reference: does a €150–250 service pay for itself?

Your own sales (python trade.py show) beat all of this once there are a few.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import asks
import db
import model

MIN_POOL = 4            # comparable asks needed to place a vanished listing
RELIST_DAYS = 8         # a same-price listing appearing this soon after = relist
RELIST_GAP = 0.02       # "same price": within 2%
STALE_DAYS = 30         # a listing up this long without selling is a wish
MIN_GROUP = 10          # listings needed before a line/material row is shown
SERVICE_COST = (150, 250)
# eBay returned at most 100 results per query until 2 Oct (comps.PAGE_LIMIT is
# 200 since). A query that filled the page saw only the top of eBay's
# best-match order, which reshuffles: there a "vanished" listing often just
# slipped below the cut. Disappearances count only for references whose last
# query came back below this — complete result sets, under either limit.
COMPLETE_BELOW = 100


def load() -> pd.DataFrame:
    with db.connect() as conn:
        rows = conn.execute("""
            select c.comp_id, c.title, c.price, c.first_price, c.currency, c.query_reference,
                   c.first_seen_at, c.last_seen_at, c.disappeared_at, c.seen_count,
                   q.results as query_results
            from comps c
            left join comp_queries q
              on q.query_brand = c.query_brand and q.query_reference = c.query_reference
            where c.price > 0""").fetchall()
    c = pd.DataFrame(rows)
    if c.empty:
        return c
    for col in ("first_seen_at", "last_seen_at", "disappeared_at"):
        c[col] = pd.to_datetime(c[col], utc=True)
    for col in ("price", "first_price", "query_results"):
        c[col] = pd.to_numeric(c[col], errors="coerce")
    c["k"] = c["query_reference"].map(model.ref_key)
    c = c.dropna(subset=["k", "price"])
    t = c["title"].fillna("")
    c = c[(c["currency"].fillna("EUR").str.upper() == "EUR") & ~t.map(lambda s: bool(asks.PARTS_RE.search(s)))]
    t = c["title"].fillna("")
    # The listing's own material, in the lot's terms: gold-coloured without a
    # karat stamp counts as plated (asks.py treats it the same way).
    c = c.assign(
        mat=t.map(asks.listing_material).replace({"goldtone": "plated"}),
        line=t.map(lambda s: next((n for n in asks.LINE_RES if n in asks.lines_in(s)), None)),
        serviced=t.map(lambda s: bool(asks.SERVICED_RE.search(s))),
        gone=c["disappeared_at"].notna(),
        complete=c["query_results"] < COMPLETE_BELOW,
    )
    return c.reset_index(drop=True)


def mark_relists(c: pd.DataFrame) -> pd.Series:
    """True for vanished listings that came back as a new listing — same
    reference and title, price within RELIST_GAP, first seen within RELIST_DAYS."""
    out = pd.Series(False, index=c.index)
    norm = c["title"].fillna("").str.lower().str.split().str.join(" ")
    for _, g in c.groupby("k"):
        for i, v in g[g["gone"]].iterrows():
            lo = v["last_seen_at"] - pd.Timedelta(days=1)
            hi = v["disappeared_at"] + pd.Timedelta(days=RELIST_DAYS)
            other = g[(g.index != i) & g["first_seen_at"].between(lo, hi)
                      & (norm[g.index] == norm[i])]
            if ((other["price"] / v["price"] - 1).abs() <= RELIST_GAP).any():
                out[i] = True
    return out


def place(c: pd.DataFrame, row, at) -> dict | None:
    """Where `row`'s price sat among comparable asks live at time `at`."""
    g = c[(c["k"] == row["k"]) & (c["first_seen_at"] <= at)
          & (c["disappeared_at"].isna() | (c["disappeared_at"] > at))]
    if row.name not in g.index:
        g = pd.concat([g, c.loc[[row.name]]])
    prices = asks.counted(g, row["mat"], row["line"], bool(row["serviced"]))
    if len(prices) < MIN_POOL:
        return None
    p = row["price"]
    rank = ((prices < p).sum() + 0.5 * (prices == p).sum()) / len(prices)
    return {"rank": float(rank), "to_p25": p / float(np.quantile(prices, asks.VALUE_QUANTILE)),
            "to_median": p / float(prices.median()), "pool": len(prices)}


def clearing(c: pd.DataFrame, sold: pd.DataFrame, now) -> None:
    print("A. CLEARING LEVEL — where vanished asks sat among comparable asks live at the time")
    placed = [x for x in (place(c, r, r["last_seen_at"]) for _, r in sold.iterrows()) if x]
    stale = c[~c["gone"] & c["complete"] & (c["first_seen_at"] < now - pd.Timedelta(days=STALE_DAYS))]
    stale_placed = [x for x in (place(c, r, now) for _, r in stale.iterrows()) if x]
    print(f"   vanished listings, relists removed, with {MIN_POOL}+ comparable asks: {len(placed)}")
    if len(placed) < 10:
        print("   too few to read yet — run again in a week or two.\n")
        return
    p = pd.DataFrame(placed)
    print(f"   median position among comparable asks     {p['rank'].median():>6.0%}"
          f"   (cheapest = 0%, dearest = 100%)")
    if stale_placed:
        print(f"   … listings still up after {STALE_DAYS}+ days         "
              f"{pd.DataFrame(stale_placed)['rank'].median():>6.0%}   ({len(stale_placed)} listings)")
    r25, rmed = p["to_p25"].median(), p["to_median"].median()
    print(f"   median vanished ask ÷ lower quarter       {r25:>6.2f}")
    print(f"   median vanished ask ÷ median ask          {rmed:>6.2f}")
    if 0.40 <= p["rank"].median() <= 0.60:
        print("   Vanished asks sit mid-range, not at the cheap end: either watches sell")
        print("   across the range (condition and photos decide), or many disappearances")
        print("   are withdrawals, not sales. Read the ratio below as an upper bound.")
    import shortlist
    used = shortlist.REALISM_EBAY
    implied = r25 * 0.95
    verdict = ("about right" if abs(implied - used) <= 0.06 else
               "conservative — watches leave above it" if implied > used else
               "optimistic — watches leave below it")
    print(f"   → if buyers then knock ~5% off, a sale lands near {implied:.2f}× the lower quarter;")
    print(f"     the shortlist assumes {used:.2f}× (REALISM_EBAY): {verdict}.\n")


def speed(c: pd.DataFrame, sold: pd.DataFrame, now) -> None:
    print("B. SPEED — what turns over, by line and material")
    start = c["first_seen_at"].min()
    window = min(30.0, (now - start).total_seconds() / 86400)
    if window < 7:
        print("   less than a week of tracking — come back later.\n")
        return
    up = (sold["last_seen_at"] - sold["first_seen_at"]).dt.total_seconds() / 86400
    if len(up):
        print(f"   vanished listings were up at least {up.median():.0f} days (median; we only see them"
              f" from the first poll), middle half {up.quantile(.25):.0f}–{up.quantile(.75):.0f}")
    recent = sold[sold["disappeared_at"] > now - pd.Timedelta(days=window)]
    live = c[~c["gone"] & c["complete"]]
    rows = []
    for (line, mat), g in live.groupby([live["line"].fillna("no line"), "mat"]):
        if len(g) < MIN_GROUP:
            continue
        n_out = int(((recent["line"].fillna("no line") == line) & (recent["mat"] == mat)).sum())
        rate = n_out / (window / 7) / len(g)
        rows.append((line, mat, len(g), float(g["price"].median()), n_out, rate))
    if not rows:
        print(f"   no line/material with {MIN_GROUP}+ live listings yet.\n")
        return
    print(f"   over the last {window:.0f} days:")
    print(f"   {'line':<15}{'material':<10}{'live':>6}{'median ask':>12}{'vanished':>10}{'per week':>10}")
    for line, mat, n, med, out, rate in sorted(rows, key=lambda x: -x[5]):
        print(f"   {line:<15}{mat:<10}{n:>6}{med:>12,.0f}{out:>10}{rate:>10.1%}")
    print("   'per week' = share of the live listings that vanish in a week. Higher =")
    print("   more buyers for that kind of watch: easier exits, less capital stuck.\n")


def cuts(c: pd.DataFrame) -> None:
    print("C. PRICE CUTS — sellers lowering their ask")
    seen = c[c["seen_count"] >= 2].dropna(subset=["first_price"])
    seen = seen[seen["first_price"] > 0]
    if len(seen) < 20:
        print("   too few listings seen twice yet.\n")
        return
    cut = seen["price"] < seen["first_price"] * 0.99
    size = 1 - seen.loc[cut, "price"] / seen.loc[cut, "first_price"]
    print(f"   listings seen 2+ times: {len(seen):,} · lowered their price: {cut.mean():.0%}"
          + (f" · median cut {size.median():.0%}" if cut.any() else ""))
    gone = seen[seen["gone"]]
    if len(gone) >= 10:
        print(f"   of those that vanished: {(gone['price'] < gone['first_price'] * 0.99).mean():.0%} "
              f"had been cut first")
    print("   A high share means asks are opening positions — plan to come down too.\n")


def service_premium(c: pd.DataFrame) -> None:
    print("D. SERVICE PREMIUM — serviced/warranty asks vs as-is asks, same reference and material")
    live = c[~c["gone"]]
    rows = []
    for (k, mat), g in live.groupby(["k", "mat"]):
        s, a = g.loc[g["serviced"], "price"], g.loc[~g["serviced"], "price"]
        if len(s) >= 2 and len(a) >= 2:
            rows.append((float(s.median()), float(a.median())))
    if len(rows) < 5:
        print(f"   only {len(rows)} references have 2+ of each — not enough yet.\n")
        return
    r = pd.DataFrame(rows, columns=["s", "a"])
    ratio = float(np.exp(np.median(np.log(r["s"] / r["a"]))))
    typical = float(r["a"].median())
    gain = typical * (ratio - 1)
    side = "above" if ratio >= 1 else "below"
    print(f"   {len(r)} references · serviced asks sit {abs(ratio - 1):.0%} {side} as-is asks "
          f"(≈ €{abs(gain):,.0f} on a typical €{typical:,.0f} watch)")
    lo, hi = SERVICE_COST
    verdict = ("covers a service with room to spare" if gain > hi * 1.3 else
               "about covers a service — only worth it if it also sells faster" if gain > lo else
               "does not cover a service — sell as-is, describe it honestly")
    print(f"   a service costs about €{lo}–{hi} and takes weeks → the premium {verdict}.")
    print("   These are dealer asks with a warranty, so a private seller gets less of it.\n")


def main() -> None:
    c = load()
    if c.empty:
        print("No eBay listings stored yet.")
        return
    now = pd.Timestamp.now(tz="UTC")
    relist = mark_relists(c)
    sold = c[c["gone"] & ~relist & c["complete"]]
    print(f"eBay listings tracked since {c['first_seen_at'].min():%d %b}: {len(c):,} "
          f"(EUR, parts removed) · live {int((~c['gone']).sum()):,} · vanished {int(c['gone'].sum()):,}, "
          f"of which {int(relist.sum()):,} came back as relists")
    refs = c.groupby("k")["complete"].first()
    print(f"references with a complete eBay result set (under {COMPLETE_BELOW} results): "
          f"{int(refs.sum()):,} of {len(refs):,} · vanished there, relists removed: {len(sold):,}")
    print("A and B use only those: in a full result set a listing can drop out without selling.\n")
    clearing(c, sold, now)
    speed(c, sold, now)
    cuts(c)
    service_premium(c)


if __name__ == "__main__":
    main()

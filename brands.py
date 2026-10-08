"""Which brands besides Omega could Retour buy and sell? Read-only.

    python brands.py              # vintage men's/unisex, 1950–1989, every brand
    python brands.py tank         # Seiko dress/tank-style pieces in detail
    python brands.py seiko        # one brand in detail: models and references
                                  # ("seiko, better lines" is King/Grand Seiko,
                                  # Lord Marvel/Matic, classic chronos and divers)

Every closed lot in the database, not just the Omega target, grouped by
brand. For each brand: how many vintage watches close per week, how many sell,
at what prices, how many of those fit the €700 all-in budget, and how
repeatable the references are (a reference that sells again and again is one
the price model can learn and the shop can restock).

    sold/wk      sold vintage lots per week (the deal flow)
    €80–cap/wk   sold per week between €80 (below that, fees and shipping eat
                 the margin) and the highest hammer the budget allows
    sell-thru    sold ÷ closed (the rest missed their reserve)
    p25/med/p75  hammer prices of sold lots
    steel        share of sold lots with a steel case (easier to resell)
    refs 3+      references sold 3+ times; "spread" is how far apart their
                 prices fall (p75 ÷ p25, median across those references):
                 wide = more noise, and more room for a mispriced lot
"""

from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np
import pandas as pd

import db
import model
from config import cfg

QUERY = """
select l.lot_id, l.brand, l.watch_model, l.reference_number, l.gender,
       l.watch_year, l.watch_period, l.watch_condition, l.close_time, l.title,
       l.description,
       jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Case material").value') #>> '{}' as case_material,
       r.final_price, r.sold
from lots l
join lot_results r using (lot_id)
"""
MIN_SOLD = 15          # brands with fewer vintage sales are not listed


def load() -> pd.DataFrame:
    with db.connect() as conn:
        df = pd.DataFrame(conn.execute(QUERY).fetchall())
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["watch_year"] = pd.to_numeric(df["watch_year"], errors="coerce")
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["brand_k"] = df["brand"].fillna("?").astype(str).str.strip().map(brand_key)
    # Seiko's better lines are a different business from Seiko 5: split them.
    better = re.compile(cfg.premium_seiko_pattern, re.I)
    text = (df["watch_model"].fillna("") + " " + df["title"].fillna("") + " "
            + df["reference_number"].fillna(""))
    df.loc[(df["brand_k"] == "seiko") & text.map(lambda t: bool(better.search(t))),
           "brand_k"] = "seiko, better lines"
    era = (df["watch_year"].between(1950, 1989)
           | df["watch_period"].astype(str).str.match(r"^(1950|1960|1970|1980)"))
    df["vintage"] = era & df["gender"].isin(["men", "unisex"])
    # Seiko dress/tank style counts for any gender and up to 1999 (config.py).
    style = re.compile(cfg.style_seiko_pattern, re.I)
    names = re.compile(cfg.style_seiko_names, re.I)
    late = (df["watch_year"].between(1960, 1999)
            | df["watch_period"].astype(str).str.match(r"^(196|197|198|199)"))
    df["tank"] = (df["brand_k"].str.startswith("seiko") & late
                  & ((df["watch_model"].fillna("") + " " + df["title"].fillna("")).map(
                      lambda t: bool(style.search(t)))
                     | df["description"].fillna("").map(lambda t: bool(names.search(t)))))
    df["mat"] = df.apply(model.lot_material, axis=1)
    return df


def brand_key(name: str) -> str:
    """Lower-case, accents removed: 'Universal Genève' and 'Universal Geneve' are one brand."""
    return "".join(c for c in unicodedata.normalize("NFKD", name)
                   if not unicodedata.combining(c)).lower()


def max_hammer() -> float:
    return (cfg.max_all_in_cost - cfg.catawiki_buyer_fixed_fee
            - cfg.expected_inbound_shipping) / (1 + cfg.catawiki_buyer_fee_rate)


def ref_spread(sold: pd.DataFrame) -> tuple[int, float]:
    """(references sold 3+ times, median p75 ÷ p25 of their prices)."""
    k = sold["reference_number"].map(model.ref_key)
    g = sold.assign(k=k).dropna(subset=["k"]).groupby("k")["final_price"]
    st = g.agg(n="size", q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
    st = st[st["n"] >= 3]
    if st.empty:
        return 0, float("nan")
    return int(len(st)), float((st["q75"] / st["q25"].clip(lower=1)).median())


def overview(df: pd.DataFrame) -> None:
    v = df[df["vintage"]]
    weeks = max((v["close_time"].max() - v["close_time"].min()).days / 7, 1)
    cap = max_hammer()
    print(f"Vintage men's/unisex lots with a result: {len(v):,} over {weeks:.1f} weeks "
          f"({v['close_time'].min():%d %b} → {v['close_time'].max():%d %b})")
    print(f"Budget: €{cfg.max_all_in_cost:.0f} all-in → hammer up to about €{cap:.0f} (EU seller)\n")
    rows = []
    for b, g in v.groupby("brand_k"):
        s = g[g["sold"] == True]
        if len(s) < MIN_SOLD:
            continue
        p = s["final_price"]
        n_refs, spread = ref_spread(s)
        rows.append({
            "brand": b, "closed": len(g), "sold": len(s), "sold_wk": len(s) / weeks,
            "budget_wk": (p.between(80, cap)).sum() / weeks,
            "thru": len(s) / len(g), "p25": p.quantile(.25), "med": p.median(), "p75": p.quantile(.75),
            "steel": (s["mat"] == "steel").mean(), "refs3": n_refs, "spread": spread,
        })
    if not rows:
        print("No brand has enough vintage sales yet.")
        return
    t = pd.DataFrame(rows).sort_values("budget_wk", ascending=False)
    band = f"€80–{cap:.0f}/wk"
    print(f"{'brand':<20}{'sold/wk':>8}{band:>13}{'sell-thru':>10}"
          f"{'p25':>7}{'med':>7}{'p75':>7}{'steel':>7}{'refs 3+':>8}{'spread':>8}")
    for _, r in t.iterrows():
        sp = f"{r['spread']:.2f}" if np.isfinite(r["spread"]) else "—"
        print(f"{r['brand'][:19]:<20}{r['sold_wk']:>8.1f}{r['budget_wk']:>13.1f}{r['thru']:>10.0%}"
              f"{r['p25']:>7.0f}{r['med']:>7.0f}{r['p75']:>7.0f}{r['steel']:>7.0%}"
              f"{int(r['refs3']):>8}{sp:>8}")
    others = v[~v["brand_k"].isin(t["brand"])]
    print(f"\n({others['brand_k'].nunique()} smaller brands with fewer than {MIN_SOLD} vintage sales "
          f"left out — {int((others['sold'] == True).sum())} sales between them)")

    t = df[df["tank"]]
    s = t[t["sold"] == True]
    if len(s):
        p = s["final_price"]
        tw = max((t["close_time"].max() - t["close_time"].min()).days / 7, 1)
        print(f"\nSeiko dress and tank style (Dolce, Chariot, Lassale, Credor, rectangular…; any gender, 1960–1999):")
        print(f"   {len(s)} sold of {len(t)} closed, {len(s) / tw:.1f} a week · hammer p25 €{p.quantile(.25):.0f}, "
              f"median €{p.median():.0f}, p75 €{p.quantile(.75):.0f} · details: python brands.py tank")


def detail(df: pd.DataFrame, brand: str) -> None:
    b = brand_key(brand.strip())
    if b == "tank":
        g = df[df["tank"]]
        brand = "Seiko dress and tank style"
    else:
        g = df[df["vintage"] & df["brand_k"].str.contains(b, regex=False)]
    s = g[g["sold"] == True]
    if s.empty:
        print(f"No sold vintage lots for '{brand}'.")
        return
    weeks = max((df["close_time"].max() - df["close_time"].min()).days / 7, 1)
    print(f"{brand}: {len(s)} vintage sales of {len(g)} closed, {len(s) / weeks:.1f} a week · "
          f"median €{s['final_price'].median():.0f}\n")
    print("by model line (Catawiki's 'Model' field):")
    m = s.groupby(s["watch_model"].fillna("?").str.strip())["final_price"].agg(["count", "median", "min", "max"])
    for name, r in m.sort_values("count", ascending=False).head(15).iterrows():
        print(f"   {str(name)[:34]:<35}{int(r['count']):>5} sold   median €{r['median']:>6.0f}"
              f"   €{r['min']:.0f}–{r['max']:.0f}")
    print("\nmost-sold references:")
    k = s.assign(k=s["reference_number"].map(model.ref_key)).dropna(subset=["k"])
    refs = k.groupby("k").agg(n=("final_price", "size"), med=("final_price", "median"),
                              lo=("final_price", "min"), hi=("final_price", "max"),
                              example=("reference_number", "first"))
    for _, r in refs.sort_values("n", ascending=False).head(15).iterrows():
        print(f"   {str(r['example'])[:20]:<21}{int(r['n']):>4} sold   median €{r['med']:>6.0f}"
              f"   €{r['lo']:.0f}–{r['hi']:.0f}")
    print(f"\nmaterial: " + "  ".join(f"{m_} {c / len(s):.0%}" for m_, c in s["mat"].value_counts().items()))
    print(f"no reference number given: {s['reference_number'].isna().mean():.0%} of sales")


if __name__ == "__main__":
    data = load()
    if data.empty:
        sys.exit("No closed lots yet.")
    if len(sys.argv) > 1:
        detail(data, " ".join(sys.argv[1:]))
    else:
        overview(data)

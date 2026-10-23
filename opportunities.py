"""Where else is there money? Every brand and model line in the database. Read-only.

    python opportunities.py              # all lines with 8+ sales
    python opportunities.py --min 5
    python opportunities.py --top 40

A. Every brand + model line (Catawiki's "Model" field), any gender, any era:
     sold/wk     sold lots per week
     ≤cap/wk     of those, how many fit your all-in budget
     sell-thru   sold ÷ closed (the rest missed their reserve)
     p25/med/p75 hammer prices
     spread      p75 ÷ p25: how far apart prices for the "same" watch fall —
                 wide = lots of noise, and more chances to buy the cheap end
     cheap       share of sales below 60% of the line's median — the bargains
     steel       share with a steel case
   Two rankings: most budget lots per week, and widest spreads.

B. Anatomy of a bargain: across lines with enough sales, do lots that close
   far below their line's median have something in common — the weekday and
   hour they close, how many photos they have, where the seller is? If a
   pattern is strong, that is where to hunt, in any brand.

These are auction prices, not resale prices. A line that looks good here
still needs its resale side checked (market.py, Chrono24, eBay sold).
"""

from __future__ import annotations

import argparse
import re
import unicodedata

import numpy as np
import pandas as pd

import db
import model
from config import cfg

QUERY = """
select l.lot_id, l.brand, l.watch_model, l.title, l.gender, l.watch_year, l.watch_period,
       l.close_time, l.photo_count, l.seller_country, l.reference_number,
       jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Case material").value') #>> '{}' as case_material,
       r.final_price, r.sold
from lots l
join lot_results r using (lot_id)
"""
UNNAMED = {"", "?", "-", "unknown", "vintage", "other", "none", "n/a", "vintage watch", "watch"}
EU = {"AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE", "IT",
      "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"}


def norm(s) -> str:
    s = "".join(c for c in unicodedata.normalize("NFKD", str(s or ""))
                if not unicodedata.combining(c)).lower().strip()
    return re.sub(r"\s+", " ", s)


def load() -> pd.DataFrame:
    with db.connect() as conn:
        df = pd.DataFrame(conn.execute(model.with_photos(conn, QUERY)).fetchall())
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    df["final_price"] = pd.to_numeric(df["final_price"], errors="coerce")
    df["photo_count"] = pd.to_numeric(df["photo_count"], errors="coerce")
    df["brand_k"] = df["brand"].map(norm)
    line = df["watch_model"].map(norm)
    df["line"] = line.where(~line.isin(UNNAMED), "(no model given)")
    df["mat"] = df.apply(model.lot_material, axis=1)
    return df


def max_hammer() -> float:
    return (cfg.max_all_in_cost - cfg.catawiki_buyer_fixed_fee
            - cfg.expected_inbound_shipping) / (1 + cfg.catawiki_buyer_fee_rate)


def lines(df: pd.DataFrame, min_sold: int) -> pd.DataFrame:
    weeks = max((df["close_time"].max() - df["close_time"].min()).days / 7, 1)
    cap = max_hammer()
    rows = []
    for (b, ln), g in df.groupby(["brand_k", "line"]):
        s = g[g["sold"] == True]
        if len(s) < min_sold or b in ("", "?"):
            continue
        p = s["final_price"]
        med = p.median()
        rows.append({
            "brand": b, "line": ln, "sold_wk": len(s) / weeks,
            "cap_wk": (p <= cap).sum() / weeks, "thru": len(s) / len(g),
            "p25": p.quantile(.25), "med": med, "p75": p.quantile(.75),
            "spread": p.quantile(.75) / max(p.quantile(.25), 1),
            "cheap": (p < 0.6 * med).mean(), "steel": (s["mat"] == "steel").mean(),
        })
    return pd.DataFrame(rows), weeks


def show(t: pd.DataFrame, title: str, n: int) -> None:
    print(title)
    print(f"   {'brand':<18}{'model line':<26}{'sold/wk':>8}{'≤cap/wk':>8}{'sell-thru':>10}"
          f"{'p25':>7}{'med':>7}{'p75':>7}{'spread':>7}{'cheap':>7}{'steel':>7}")
    for _, r in t.head(n).iterrows():
        print(f"   {r['brand'][:17]:<18}{r['line'][:25]:<26}{r['sold_wk']:>8.1f}{r['cap_wk']:>8.1f}"
              f"{r['thru']:>10.0%}{r['p25']:>7.0f}{r['med']:>7.0f}{r['p75']:>7.0f}"
              f"{r['spread']:>7.2f}{r['cheap']:>7.0%}{r['steel']:>7.0%}")
    print()


def anatomy(df: pd.DataFrame, t: pd.DataFrame) -> None:
    """Do bargains share a closing time, photo count or seller location?"""
    keep = set(zip(t["brand"], t["line"]))
    s = df[(df["sold"] == True) & df.apply(lambda r: (r["brand_k"], r["line"]) in keep, axis=1)].copy()
    s = s[s["line"] != "(no model given)"]
    if len(s) < 200:
        print("B. Not enough sales in named lines yet for the bargain analysis.")
        return
    s["ratio"] = s["final_price"] / s.groupby(["brand_k", "line"])["final_price"].transform("median")
    s["bargain"] = s["ratio"] < 0.7
    local = s["close_time"].dt.tz_convert(cfg.timezone)
    s["day"] = np.where(local.dt.dayofweek >= 5, "weekend", "weekday")
    s["slot"] = pd.cut(local.dt.hour, [-1, 11, 16, 19, 23], labels=["before 12", "12–17", "17–20", "20–24"])
    s["photos"] = pd.cut(s["photo_count"], [-1, 5, 9, 99], labels=["≤5 photos", "6–9 photos", "10+ photos"])
    s["seller"] = np.where(s["seller_country"].astype(str).str.upper().str[:2].isin(EU), "EU seller",
                           np.where(s["seller_country"].isna(), "seller unknown", "non-EU seller"))
    base = s["bargain"].mean()
    print(f"B. Anatomy of a bargain — {len(s):,} sales in {len(keep)} lines; "
          f"{base:.0%} closed under 70% of their line's median\n")
    print(f"   {'':<22}{'lots':>7}{'bargains':>10}{'vs average':>12}   median ÷ line median")
    for col, label in (("day", "Closing day"), ("slot", "Closing time (Vienna)"),
                       ("photos", "Photos"), ("seller", "Seller")):
        print(f"   {label}")
        for k, g in s.groupby(col, observed=True):
            if len(g) < 30:
                continue
            lift = g["bargain"].mean() / base if base else np.nan
            print(f"     {str(k):<20}{len(g):>7}{g['bargain'].mean():>10.0%}{lift:>11.1f}×"
                  f"   {g['ratio'].median():.2f}")
    print("\n   'vs average' above 1.3× is worth acting on: hunt where bargains concentrate.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--min", type=int, default=8, help="minimum sales per line (default 8)")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    df = load()
    if df.empty:
        raise SystemExit("No closed lots yet.")
    t, weeks = lines(df, a.min)
    if t.empty:
        raise SystemExit("No line has enough sales yet.")
    print(f"All closed lots: {len(df):,} over {weeks:.1f} weeks · budget hammer up to €{max_hammer():.0f}\n")
    show(t.sort_values("cap_wk", ascending=False),
         f"A1. Most budget lots per week ({len(t)} lines with {a.min}+ sales)", a.top)
    named = t[t["line"] != "(no model given)"]
    show(named[named["cap_wk"] >= 0.5].sort_values("spread", ascending=False),
         "A2. Widest spreads among lines with 0.5+ budget lots a week — where cheap buys happen", a.top)
    anatomy(df, t)


if __name__ == "__main__":
    main()

"""Turning live eBay asks into a resale value you could actually get.

The median of every listing for a reference overstated what a new seller can
realise: the listings mix in spare parts, other currencies, solid-gold and
two-tone versions, and far outliers, and a buyer comparing listings goes to
the cheaper end first. So the value is the LOWER QUARTER of the cleaned asks
— close to where you would have to list to sell in reasonable time.

Shared by shortlist.py (live), backtest.py, analyze.py and explain.py, so they
all agree on the number.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import model

VALUE_QUANTILE = 0.25     # lower quarter of the cleaned asks

# A part listing says so explicitly: "dial only", "Zifferblatt für …", "for parts".
# Words like "dial" on their own describe the watch ("TV Dial", "Black Dial").
PARTS_RE = re.compile(
    r"\b(only|nur|for parts|ersatzteile?|spares?|defekt|defective|not working|"
    r"ohne uhr|without watch)\b"
    r"|\b(dial|zifferblatt|crown|krone|uhrwerk|movement|caseback|case back|bezel|"
    r"strap|armband|bracelet|box|zeiger|hands|glas|crystal|buckle)\s+(for|für|fits|passend)\b"
    r"|^(dial|zifferblatt|crown|krone|uhrwerk|movement|caseback|bezel|strap|armband|box)\b",
    re.I)
# Solid gold, and the pricier two-tone / diamond variants of a steel reference.
GOLD_VARIANT_RE = re.compile(
    r"\b(18|14)\s?(k|kt|ct|karat)\b|\b750\b|massiv\w*\s?gold|solid gold"
    r"|stahl\s?/\s?gold|steel\s?/\s?gold|gold\s?/\s?stahl|zweifarbig|bicolou?r|two[\s-]?tone"
    r"|diamant|diamond|brillant",
    re.I)
SERVICED_RE = re.compile(r"servic|revidiert|revision|überholt|warranty|garantie", re.I)

# Relative to the median of what is left. The low bound only catches
# unflagged accessories: a cheap real watch is the best evidence there is,
# and a high median is often just dealer stock.
OUTLIER_HIGH, OUTLIER_LOW = 2.0, 0.3

# One dealer often lists the same watch several times at the same price
# ("VROM12", "VROM13", …). Asks within this share of each other count once,
# so a single shop cannot set the value. No seller data is used or stored.
DUPLICATE_GAP = 0.01


def listing_flags(title, currency) -> list[str]:
    """Flags that do not depend on the lot being valued."""
    t = str(title or "")
    f = []
    if PARTS_RE.search(t):
        f.append("part?")
    ccy = str(currency or "EUR").upper()
    if ccy != "EUR":
        f.append(ccy)
    if GOLD_VARIANT_RE.search(t):
        f.append("gold/two-tone")
    if SERVICED_RE.search(t):
        f.append("serviced/warranty")
    return f


def lot_material(lot) -> str:
    return model.material(f"{lot.get('title') or ''} {lot.get('description') or ''}")


def usable(listings: pd.DataFrame, material: str) -> pd.Series:
    """Which listings count for a lot of this material (outliers handled later)."""
    parts = listings["title"].fillna("").map(lambda t: bool(PARTS_RE.search(t)))
    other_ccy = listings["currency"].fillna("EUR").str.upper() != "EUR"
    gold = listings["title"].fillna("").map(lambda t: bool(GOLD_VARIANT_RE.search(t)))
    keep = ~parts & ~other_ccy
    if material != "gold":
        keep &= ~gold
    return keep


def drop_outliers(prices: pd.Series) -> pd.Series:
    if prices.empty:
        return prices
    med = prices.median()
    return prices[(prices <= OUTLIER_HIGH * med) & (prices >= OUTLIER_LOW * med)]


def drop_repeats(prices: pd.Series) -> pd.Series:
    """Keep one ask per cluster of near-identical prices (cheapest first)."""
    prices = prices.sort_values()
    keep, last = [], None
    for i, p in prices.items():
        if last is None or p > last * (1 + DUPLICATE_GAP):
            keep.append(i)
            last = p
    return prices.loc[keep]


def counted(listings: pd.DataFrame, material: str) -> pd.Series:
    """The asks that count for a lot of this material, one per price cluster."""
    return drop_repeats(drop_outliers(listings.loc[usable(listings, material), "price"]))


def value(listings: pd.DataFrame, material: str) -> tuple[float, int] | None:
    """(lower-quarter ask, listings used) for one reference, or None if empty."""
    if listings is None or listings.empty:
        return None
    prices = counted(listings, material)
    if prices.empty:
        return None
    return float(np.quantile(prices, VALUE_QUANTILE)), int(len(prices))


def load() -> dict[str, pd.DataFrame]:
    """Live eBay listings per normalised reference: {ref_key: listings}."""
    import db
    with db.connect() as conn:
        rows = conn.execute(
            "select comp_id, title, price, currency, query_reference from comps "
            "where disappeared_at is null and price > 0"
        ).fetchall()
    a = pd.DataFrame(rows)
    if a.empty:
        return {}
    a["k"] = a["query_reference"].map(model.ref_key)
    a["price"] = pd.to_numeric(a["price"], errors="coerce")
    a = a.dropna(subset=["k", "price"])
    return {k: g.sort_values("price").reset_index(drop=True) for k, g in a.groupby("k")}

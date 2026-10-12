"""Prices you see yourself: Chrono24, Willhaben, a dealer, a flea market.

    python market.py add 136.011 chrono24 450 520 610         # three asking prices
    python market.py add 136.011 willhaben 480 --plated --cond "very good"
    python market.py add 166.0216 willhaben 590 --sold        # what one really sold for
    python market.py list                                     # everything still counting
    python market.py list 136.011
    python market.py drop 17                                  # remove entry 17
    python market.py audit                                    # what the pricing does now

Chrono24 and Willhaben do not allow automated collection, so the agent never
reads them. What you look at yourself, you log here in seconds, and from then
on it counts like an eBay listing of that reference:

  asks     join the comparable listings. They help reach the 3 listings a
           reference needs for an eBay-style value, and the competition (the
           cheapest comparable ask, the second cheapest from 5 asks on) caps
           the quick-sale price (shortlist.COMPETITION_CAP).
  --sold   a price a watch really fetched. The quick-sale price is never
           planned above the median of these.

Material and service matter as they do for eBay listings: a --plated ask does
not count against a steel lot, and a --serviced dealer ask does not count
against an as-is lot while 3+ as-is asks remain. Without --steel/--plated/
--gold the material is unknown and the ask counts for any material.

Asks stop counting after ASK_DAYS (asking prices go stale), sold prices after
SOLD_DAYS. Only the price and these few facts are stored: no links, no seller.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

import db
import model

ASK_DAYS = 45
SOLD_DAYS = 180
CHANNELS = ("chrono24", "willhaben", "ebay", "catawiki", "dorotheum", "dealer", "instagram", "other")

SCHEMA = """
create table if not exists market_comps (
    id         bigserial primary key,
    reference  text not null,
    channel    text not null,
    kind       text not null default 'ask',       -- ask | sold
    price      numeric(12, 2) not null,
    material   text,                              -- steel | plated | gold | null = unknown
    serviced   boolean not null default false,
    condition  text,
    note       text,
    seen_on    date not null default current_date
);
alter table market_comps enable row level security;
"""

LIVE = """
select * from market_comps
where (kind = 'ask'  and seen_on > current_date - %(ask)s::int)
   or (kind = 'sold' and seen_on > current_date - %(sold)s::int)
order by reference, price
"""


def _setup(conn) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def _exists(conn) -> bool:
    return conn.execute("select to_regclass('market_comps') as t").fetchone()["t"] is not None


def _rows() -> list[dict]:
    with db.connect() as conn:
        if not _exists(conn):
            return []
        return conn.execute(LIVE, {"ask": ASK_DAYS, "sold": SOLD_DAYS}).fetchall()


def _title(r) -> str:
    """A listing-style title, so the eBay filters read material and service
    from it exactly as they do from a real listing title."""
    parts = {"steel": ["stainless steel"], "plated": ["gold plated"], "gold": ["18k gold"]}.get(
        r.get("material"), [])
    if r.get("serviced"):
        parts.append("serviced")
    if r.get("condition"):
        parts.append(str(r["condition"]))
    parts.append(f"(logged from {r['channel']} on {r['seen_on']:%d %b})")
    return " ".join(parts)


def as_listings() -> pd.DataFrame:
    """Logged asks in the shape asks.load() uses for eBay listings."""
    rows = [r for r in _rows() if r["kind"] == "ask"]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame({
        "comp_id": [f"m{r['id']}" for r in rows],
        "title": [_title(r) for r in rows],
        "price": [float(r["price"]) for r in rows],
        "currency": "EUR",
        "query_reference": [r["reference"] for r in rows],
        "source": [r["channel"] for r in rows],
    })


def sold_by_ref() -> dict[str, list[tuple[float, str, bool]]]:
    """{ref_key: [(price, material, serviced), …]} for logged real sales."""
    out: dict[str, list] = {}
    for r in _rows():
        k = model.ref_key(r["reference"])
        if r["kind"] == "sold" and k:
            out.setdefault(k, []).append((float(r["price"]), r["material"] or "unknown",
                                          bool(r["serviced"])))
    return out


# --------------------------------------------------------------------- CLI

def log_prices(reference: str, channel: str, kind: str, prices, material: str | None = None,
               serviced: bool = False, condition: str | None = None, note: str | None = None,
               on: date | None = None) -> list[int]:
    """Store prices for a reference; returns the new entry ids."""
    ids = []
    with db.connect() as conn:
        _setup(conn)
        for p in prices:
            ids.append(conn.execute(
                "insert into market_comps (reference, channel, kind, price, material, serviced, "
                "condition, note, seen_on) values (%s, %s, %s, %s, %s, %s, %s, %s, %s) returning id",
                (reference.strip(), channel.lower(), kind, p, material, serviced,
                 condition, note, on or date.today())).fetchone()["id"])
        conn.commit()
    return ids


def drop_ids(ids) -> int:
    with db.connect() as conn:
        _setup(conn)
        n = conn.execute("delete from market_comps where id = any(%s)", (list(ids),)).rowcount
        conn.commit()
    return n


def add(a) -> None:
    if not model.ref_key(a.reference):
        sys.exit(f"'{a.reference}' does not look like a reference number")
    material = "steel" if a.steel else "plated" if a.plated else "gold" if a.gold else None
    kind = "sold" if a.sold else "ask"
    log_prices(a.reference, a.channel, kind, a.prices, material, a.serviced, a.cond, a.note, a.on)
    what = "sold price" if a.sold else "ask"
    print(f"logged {len(a.prices)} {what}{'s' if len(a.prices) > 1 else ''} for {a.reference} "
          f"on {a.channel}: " + ", ".join(f"€{p:.0f}" for p in a.prices)
          + (f" ({material})" if material else " (material not given: counts for any)"))


def list_(a) -> None:
    rows = _rows()
    if a.reference:
        k = model.ref_key(a.reference)
        rows = [r for r in rows if model.ref_key(r["reference"]) == k]
    if not rows:
        print("Nothing logged" + (f" for {a.reference}" if a.reference else "") + " that still counts.")
        return
    print(f"{'id':>5}  {'reference':<14}{'channel':<11}{'kind':<6}{'price':>7}  "
          f"{'material':<9}{'serv.':<6}{'seen':<8}condition / note")
    for r in rows:
        print(f"{r['id']:>5}  {r['reference'][:13]:<14}{r['channel'][:10]:<11}{r['kind']:<6}"
              f"{float(r['price']):>7.0f}  {(r['material'] or '—'):<9}{('yes' if r['serviced'] else ''):<6}"
              f"{r['seen_on']:%d %b}  {' · '.join(x for x in (r['condition'], r['note']) if x)}")
    print(f"\nAsks count for {ASK_DAYS} days, sold prices for {SOLD_DAYS}.")


def drop(a) -> None:
    n = drop_ids([a.id])
    print(f"removed entry {a.id}" if n else f"no entry {a.id}")


def audit(a) -> None:
    """How the competition cap and the confidence grades act on real lots:
    every vintage-Omega lot closed in the last DAYS days, plus the open ones,
    valued from evidence (eBay, Catawiki, logged prices). Lots with no
    evidence (the feature tier) need the full model and are only counted."""
    import numpy as np
    import shortlist as sl

    closed = model.load()
    by_ref, by_line = sl.load_asks({})
    cw = sl.catawiki_by_ref(closed)
    sl.calibrate_ratio(closed, by_ref)
    sl.calibrate_margin(closed, by_ref, cw)

    with db.connect() as conn:
        alerted = {r["lot_id"] for r in conn.execute("select lot_id from shortlist_alerts").fetchall()}
    recent = closed[closed["close_time"] > closed["close_time"].max() - pd.Timedelta(days=a.days)]
    open_ = sl.load_open()
    lots = pd.concat([recent, open_], ignore_index=True) if len(open_) else recent
    lots = lots[model.target_mask(lots)]

    rows, no_evidence, filtered = [], 0, 0
    for _, lot in lots.iterrows():
        if not sl.passes_filters(lot):
            filtered += 1
            continue
        v = sl.value_for(lot, by_ref, by_line, cw)          # no feature tier here
        if v is None:
            comp = sl.competition(lot, by_ref)
            no_evidence += 1
            rows.append({"lot": lot["lot_id"], "grade": "C" if len(comp) else "D", "before": np.nan,
                         "after": np.nan, "beat": float(comp["price"].iloc[0]) if len(comp) else np.nan,
                         "n": len(comp), "alerted": lot["lot_id"] in alerted,
                         "ref": lot["reference_number"]})
            continue
        rows.append({"lot": lot["lot_id"], "grade": v.grade, "before": v.v1 * v[1],
                     "after": v[0] * v[1], "beat": v.beat if v.beat else np.nan,
                     "n": v.n_asks, "alerted": lot["lot_id"] in alerted, "ref": lot["reference_number"]})
    df = pd.DataFrame(rows)
    if df.empty:
        print("No vintage-Omega lots in that window.")
        return

    ev = df.dropna(subset=["before"])
    capped = ev[ev["after"] < ev["before"] - 0.5]
    print(f"Vintage-Omega lots, closed in the last {a.days} days or open now: {len(df) + filtered} "
          f"({filtered} below your condition filter, left out)\n")
    print("Confidence grades")
    for g, text in sl.GRADE_TEXT.items():
        n = int((df["grade"] == g).sum())
        al = int(((df["grade"] == g) & df["alerted"]).sum())
        print(f"   {g}  {n:>5} lots  ({al} were alerted)   {text}")
    print(f"   (D and part of C: no reference evidence, valued by the feature model in the "
          f"live shortlist; {no_evidence} lots here)\n")
    if len(ev):
        cut = (capped["after"] / capped["before"] - 1)
        print("New checks: never above the competition (asks under half the value ignored;")
        print("the cheapest of a few, about the 12% point of many) or logged sale prices;")
        print("eBay and Catawiki more than 25% apart → the lower one, unless eBay sits below")
        print("even the auction price (then eBay is ignored)")
        print(f"   lots valued from evidence: {len(ev)}")
        print(f"   valued lower than before:  {len(capped)} ({len(capped) / len(ev):.0%})"
              + (f", median {cut.median():.0%}, largest {cut.min():.0%}" if len(capped) else ""))
        al = ev[ev["alerted"]]
        if len(al):
            alc = al[al["after"] < al["before"] - 0.5]
            print(f"   of the lots you were alerted on: {len(alc)} of {len(al)} would now be "
                  f"valued lower")
        print()
        show = capped.assign(cut=capped["after"] / capped["before"] - 1).sort_values("cut").head(a.top)
        if len(show):
            print(f"   {'lot':<11}{'reference':<14}{'grade':<7}{'quick sale before':>18}{'now':>8}"
                  f"{'competition':>14}{'asks':>6}")
            for _, r in show.iterrows():
                print(f"   {r['lot']:<11}{str(r['ref'])[:13]:<14}{r['grade']:<7}{r['before']:>18.0f}"
                      f"{r['after']:>8.0f}{r['beat']:>14.0f}{int(r['n']):>6}"
                      + ("   ← alerted" if r["alerted"] else ""))
    print("\nTo see one lot in full: python explain.py <lot id>")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="log one or more prices for a reference")
    s.add_argument("reference")
    s.add_argument("channel", help=", ".join(CHANNELS))
    s.add_argument("prices", type=float, nargs="+")
    s.add_argument("--sold", action="store_true", help="a real sale price, not an ask")
    m = s.add_mutually_exclusive_group()
    m.add_argument("--steel", action="store_true")
    m.add_argument("--plated", action="store_true")
    m.add_argument("--gold", action="store_true", help="solid gold or two-tone")
    s.add_argument("--serviced", action="store_true", help="sold serviced or with a warranty")
    s.add_argument("--cond", help='condition as listed, e.g. "very good"')
    s.add_argument("--note")
    s.add_argument("--on", type=date.fromisoformat, help="date seen, YYYY-MM-DD (default today)")
    s.set_defaults(fn=add)

    s = sub.add_parser("list"); s.add_argument("reference", nargs="?"); s.set_defaults(fn=list_)
    s = sub.add_parser("drop"); s.add_argument("id", type=int); s.set_defaults(fn=drop)
    s = sub.add_parser("audit"); s.add_argument("--days", type=int, default=14)
    s.add_argument("--top", type=int, default=15); s.set_defaults(fn=audit)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

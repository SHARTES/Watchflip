"""Trade log — what you actually paid, spent and sold for.

    python trade.py buy  106921180 --hammer 460 --paid 533.40 --value 745 --max 457
    python trade.py cost 106921180 --amount 180 --note "service, Uhrmacher X"
    python trade.py list 106921180 --price 790
    python trade.py sell 106921180 --price 720 --shipping 15 --fees 0
    python trade.py show

Every assumption in the shortlist — that you sell at 95% of the cheaper-quarter
eBay value, that a watch sells in about four weeks, that €15 covers shipping —
is a guess until real sales replace it. This records each trade from purchase
to sale, and `show` compares what happened with what the system expected:

  realised ÷ value   what you sold for, as a share of the value at purchase
                     (the shortlist assumes REALISM_EBAY, see shortlist.py)
  days to sell       from purchase to sale (the capital check assumes 28)
  profit             sale − shipping − fees − purchase − every extra cost

Nothing here touches the auction data; it is your own ledger.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import db

SCHEMA = """
create table if not exists trades (
    lot_id        text primary key,
    title         text,
    bought_on     date not null default current_date,
    hammer        numeric(12, 2) not null,
    paid_total    numeric(12, 2) not null,     -- everything paid to Catawiki, incl. fees and shipping
    value_at_buy  numeric(12, 2),              -- the system's resale value when you bid
    max_bid       numeric(12, 2),              -- the max bid it suggested
    listed_price  numeric(12, 2),
    listed_on     date,
    sold_price    numeric(12, 2),
    sold_on       date,
    ship_out      numeric(12, 2) default 0,
    sale_fees     numeric(12, 2) default 0,
    channel       text,
    notes         text
);
create table if not exists trade_costs (
    id        bigserial primary key,
    lot_id    text not null references trades (lot_id) on delete cascade,
    spent_on  date not null default current_date,
    amount    numeric(12, 2) not null,
    note      text
);
alter table trades enable row level security;
alter table trade_costs enable row level security;
"""


def _setup(conn) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def _lot_id(arg: str) -> str:
    import re
    m = re.search(r"/l/(\d+)", arg)
    return m.group(1) if m else arg.strip()


def buy(a) -> None:
    lot = _lot_id(a.lot)
    with db.connect() as conn:
        _setup(conn)
        row = conn.execute("select title from lots where lot_id = %s", (lot,)).fetchall()
        title = row[0]["title"] if row else None
        max_bid = a.max
        if max_bid is None:
            try:
                r = conn.execute("select max_bid from shortlist_alerts where lot_id = %s",
                                 (lot,)).fetchall()
                max_bid = float(r[0]["max_bid"]) if r else None
            except Exception:
                conn.rollback()
        conn.execute("""
            insert into trades (lot_id, title, bought_on, hammer, paid_total, value_at_buy, max_bid, notes)
            values (%s, %s, %s, %s, %s, %s, %s, %s)
            on conflict (lot_id) do update set
                hammer = excluded.hammer, paid_total = excluded.paid_total,
                value_at_buy = coalesce(excluded.value_at_buy, trades.value_at_buy),
                max_bid = coalesce(excluded.max_bid, trades.max_bid),
                notes = coalesce(excluded.notes, trades.notes)
        """, (lot, title, a.on or date.today(), a.hammer, a.paid, a.value, max_bid, a.note))
        conn.commit()
    print(f"recorded purchase of {lot}: hammer €{a.hammer:.0f}, paid €{a.paid:.2f}"
          + (f", value at buy €{a.value:.0f}" if a.value else ""))


def cost(a) -> None:
    lot = _lot_id(a.lot)
    with db.connect() as conn:
        _setup(conn)
        conn.execute("insert into trade_costs (lot_id, spent_on, amount, note) values (%s, %s, %s, %s)",
                     (lot, a.on or date.today(), a.amount, a.note))
        conn.commit()
    print(f"added €{a.amount:.2f} to {lot}" + (f" ({a.note})" if a.note else ""))


def list_(a) -> None:
    lot = _lot_id(a.lot)
    with db.connect() as conn:
        _setup(conn)
        conn.execute("update trades set listed_price = %s, listed_on = %s, channel = coalesce(%s, channel) "
                     "where lot_id = %s", (a.price, a.on or date.today(), a.channel, lot))
        conn.commit()
    print(f"{lot} listed at €{a.price:.0f}")


def sell(a) -> None:
    lot = _lot_id(a.lot)
    with db.connect() as conn:
        _setup(conn)
        conn.execute("""
            update trades set sold_price = %s, sold_on = %s, ship_out = %s, sale_fees = %s,
                              channel = coalesce(%s, channel)
            where lot_id = %s
        """, (a.price, a.on or date.today(), a.shipping, a.fees, a.channel, lot))
        conn.commit()
    print(f"{lot} sold for €{a.price:.0f}")


def show(a) -> None:
    with db.connect() as conn:
        _setup(conn)
        rows = conn.execute("""
            select t.*, coalesce((select sum(c.amount) from trade_costs c
                                  where c.lot_id = t.lot_id), 0) as extra
            from trades t order by t.bought_on, t.lot_id
        """).fetchall()
    if not rows:
        print("No trades yet.")
        return

    f = lambda x: float(x) if x is not None else None
    print(f"{'bought':<11}{'lot':<11}{'paid':>7}{'extra':>7}{'value':>7}{'sold':>7}"
          f"{'÷value':>8}{'days':>6}{'profit':>8}  title")
    held, done = [], []
    for r in rows:
        paid, extra, value = f(r["paid_total"]), f(r["extra"]), f(r["value_at_buy"])
        sold = f(r["sold_price"])
        cost_in = paid + extra
        if sold is not None:
            profit = sold - f(r["ship_out"] or 0) - f(r["sale_fees"] or 0) - cost_in
            days = (r["sold_on"] - r["bought_on"]).days
            ratio = sold / value if value else None
            done.append((profit, cost_in, days, ratio))
        else:
            profit = days = ratio = None
            held.append(cost_in)
        print(f"{r['bought_on']!s:<11}{r['lot_id']:<11}{paid:>7.0f}{extra:>7.0f}"
              f"{(f'{value:.0f}' if value else '—'):>7}{(f'{sold:.0f}' if sold else '—'):>7}"
              f"{(f'{ratio:.2f}' if ratio else '—'):>8}{(str(days) if days is not None else '—'):>6}"
              f"{(f'{profit:.0f}' if profit is not None else '—'):>8}  {str(r['title'] or '')[:40]}")

    print()
    if held:
        print(f"held: {len(held)} watches, €{sum(held):,.0f} of capital tied up")
    if done:
        n = len(done)
        prof = sum(p for p, *_ in done)
        ratios = [r for *_, r in done if r]
        print(f"sold: {n} · profit €{prof:,.0f} (€{prof / n:,.0f} per watch, "
              f"{prof / sum(c for _, c, *_ in done):.0%} on money in)")
        print(f"      median days to sell {sorted(d for _, _, d, _ in done)[n // 2]}"
              f"   (the capital check assumes 28)")
        if ratios:
            ratios.sort()
            try:
                from shortlist import REALISM_EBAY as assumed
                note = f"   (the shortlist assumes {assumed:.2f})"
            except Exception:
                note = ""
            print(f"      median realised ÷ value {ratios[len(ratios) // 2]:.2f}{note}")
        if n < 5:
            print("      Fewer than 5 sales — read these as anecdotes, not rates.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    on = dict(type=date.fromisoformat, help="date, YYYY-MM-DD (default today)")

    b = sub.add_parser("buy"); b.add_argument("lot")
    b.add_argument("--hammer", type=float, required=True)
    b.add_argument("--paid", type=float, required=True, help="total paid incl. fees and shipping")
    b.add_argument("--value", type=float, help="resale value shown in the alert")
    b.add_argument("--max", type=float, help="max bid shown in the alert")
    b.add_argument("--note"); b.add_argument("--on", **on); b.set_defaults(fn=buy)

    c = sub.add_parser("cost"); c.add_argument("lot")
    c.add_argument("--amount", type=float, required=True)
    c.add_argument("--note"); c.add_argument("--on", **on); c.set_defaults(fn=cost)

    l = sub.add_parser("list"); l.add_argument("lot")
    l.add_argument("--price", type=float, required=True)
    l.add_argument("--channel", help="instagram, ebay, …"); l.add_argument("--on", **on)
    l.set_defaults(fn=list_)

    s = sub.add_parser("sell"); s.add_argument("lot")
    s.add_argument("--price", type=float, required=True)
    s.add_argument("--shipping", type=float, default=0.0)
    s.add_argument("--fees", type=float, default=0.0)
    s.add_argument("--channel"); s.add_argument("--on", **on); s.set_defaults(fn=sell)

    sub.add_parser("show").set_defaults(fn=show)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

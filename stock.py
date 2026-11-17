"""Stock clock — a nudge when a watch has been for sale too long.

Capital is the limit: money sitting in a watch that does not sell buys nothing
else. Once a day this looks at every watch in the ledger (trade.py) that is not
sold yet and sends one Telegram message when it reaches a step:

    14 days for sale   5% under the first list price
    28 days for sale   10% under the first list price
    45 days for sale   down to the floor: what it cost you + 20%
                       (and at 60 days: sell at cost and free the money)

A step never goes below the floor before day 45, and nothing is sent when the
watch is already listed at or under the step's price. Days count from the first
listing date, which `trade.py list` keeps, so changing the price does not
restart the clock. A watch bought but still without a list price after
UNLISTED_DAYS gets one reminder to record it.

    python stock.py          # send what is due (run.py serve does this every morning)
    python stock.py --show   # print the clock for every watch held; send nothing

Only the ledger and Telegram are used.
"""

from __future__ import annotations

import logging
import math
import sys
from datetime import date
from html import escape

import db
import replies
import trade

log = logging.getLogger("stock")

STEPS = ((14, 0.95), (28, 0.90), (45, None))   # (days for sale, share of the first list price)
FLOOR_MARGIN = 0.20       # the floor is cost × (1 + this): your minimum margin
LAST_MARGIN = 0.10        # day 45, when the watch is already at or under the floor
COST_DAYS = 60            # from here: sell at cost
UNLISTED_DAYS = 10        # bought this long ago and no list price recorded → one reminder

SCHEMA = """
create table if not exists stock_nudges (
    lot_id   text not null,
    step     int  not null,           -- days of the step; 0 = "no list price recorded"
    sent_at  timestamptz not null default now(),
    primary key (lot_id, step)
);
alter table stock_nudges enable row level security;
"""

HELD = """
select t.lot_id, t.title, t.bought_on, t.paid_total, t.listed_price, t.listed_on,
       t.first_listed_on, t.first_listed_price, t.channel,
       coalesce((select sum(c.amount) from trade_costs c where c.lot_id = t.lot_id), 0) as extra
from trades t
where t.sold_price is null
order by t.bought_on, t.lot_id
"""


def nice(price: float) -> int:
    """A shop price ending in 9, never below `price`: 268.67 → 269, 270 → 279."""
    return int(math.ceil((price + 1) / 10.0) * 10 - 1)


def near(price: float) -> int:
    """The nearest shop price ending in 9: 303.05 → 299, 287.10 → 289."""
    return int(round((price + 1) / 10.0) * 10 - 1)


def plan(row, today: date | None = None) -> dict:
    """Where one held watch stands: {days, cost, floor, listed, step, target, note}.

    step is the furthest step reached (0 = no list price recorded), or None
    when nothing is due; target is the price that step asks for, or None when
    the watch is already at or under it."""
    today = today or date.today()
    cost = float(row["paid_total"]) + float(row["extra"])
    floor = cost * (1 + FLOOR_MARGIN)
    out = {"cost": cost, "floor": floor, "listed": None, "step": None, "target": None, "days": None}
    if row["listed_price"] is None:
        out["days"] = (today - row["bought_on"]).days
        out["step"] = 0 if out["days"] >= UNLISTED_DAYS else None
        return out
    listed = float(row["listed_price"])
    first = float(row["first_listed_price"] or listed)
    since = row["first_listed_on"] or row["listed_on"] or row["bought_on"]
    days = (today - since).days
    out.update(listed=listed, days=days)
    due = [(d, share) for d, share in STEPS if days >= d]
    if not due:
        return out
    d, share = due[-1]
    if share is not None:
        target = max(near(first * share), nice(floor))
    else:
        target = nice(floor)
        if listed <= target:                      # already at the floor: one last step
            target = nice(cost * (1 + LAST_MARGIN))
    out["step"] = d
    out["target"] = target if target < listed else None
    return out


def message(row, p: dict) -> str:
    name = escape(str(row["title"] or row["lot_id"])[:60])
    eur = lambda x: f"€{x:,.0f}"
    if p["step"] == 0:
        return (f"⏳ <b>{name}</b> was bought {p['days']} days ago and has no list price recorded.\n"
                f"It cost you {eur(p['cost'])}; the floor (cost + {FLOOR_MARGIN:.0%}) is {eur(p['floor'])}.\n"
                f"Once it is for sale, record it so the price clock can run:\n"
                f"<code>python trade.py list {row['lot_id']} --price …</code>")
    left = p["target"] - p["cost"]
    lines = [f"⏳ <b>{name}</b> has been for sale {p['days']} days "
             f"(listed at {eur(p['listed'])}, cost {eur(p['cost'])}).",
             f"Day {p['step']} step: drop to <b>{eur(p['target'])}</b>. That still leaves "
             f"{eur(left)} ({left / p['cost']:.0%}) before any platform fee."]
    if p["step"] >= STEPS[-1][0]:
        lines.append(f"If it is still there at {COST_DAYS} days, sell at cost ({eur(p['cost'])}) "
                     f"and put the money into the next watch.")
    lines.append(f"After you change the price: "
                 f"<code>python trade.py list {row['lot_id']} --price {p['target']}</code>")
    return "\n".join(lines)


def _held(conn) -> list[dict]:
    trade._setup(conn)
    conn.execute(SCHEMA)
    conn.commit()
    return conn.execute(HELD).fetchall()


def run() -> int:
    """Send the nudges that are due. Returns how many were sent."""
    sent = 0
    with db.connect() as conn:
        for row in _held(conn):
            p = plan(row)
            if p["step"] is None:
                continue
            done = {r["step"] for r in conn.execute(
                "select step from stock_nudges where lot_id = %s", (row["lot_id"],)).fetchall()}
            if p["step"] in done:
                continue
            if p["step"] == 0 or p["target"] is not None:
                if replies.send_html(message(row, p), row["lot_id"]) is None:
                    continue                      # not sent: try again tomorrow
                sent += 1
            # this step and every earlier one are settled, sent or not needed
            for d in [0] + [d for d, _ in STEPS if d <= p["step"]]:
                conn.execute("insert into stock_nudges (lot_id, step) values (%s, %s) "
                             "on conflict do nothing", (row["lot_id"], d))
            conn.commit()
    if sent:
        log.info("sent %d stock nudges", sent)
    return sent


def show() -> None:
    with db.connect() as conn:
        rows = _held(conn)
    if not rows:
        print("No unsold watches in the ledger.")
        return
    print(f"{'lot':<11}{'cost':>7}{'floor':>7}{'listed':>8}{'days':>6}  next")
    for row in rows:
        p = plan(row)
        if p["listed"] is None:
            nxt = f"no list price recorded ({p['days']} days since purchase)"
        elif p["target"] is not None:
            nxt = f"day {p['step']} step reached: drop to €{p['target']}"
        else:
            ahead = [d for d, _ in STEPS if d > p["days"]]
            nxt = (f"next step at day {ahead[0]}" if ahead
                   else f"at or under the floor — sell at cost from day {COST_DAYS}")
        listed = f"{p['listed']:.0f}" if p["listed"] is not None else "—"
        print(f"{row['lot_id']:<11}{p['cost']:>7.0f}{p['floor']:>7.0f}{listed:>8}"
              f"{p['days']:>6}  {nxt}  · {str(row['title'] or '')[:36]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if "--show" in sys.argv:
        show()
    else:
        run()

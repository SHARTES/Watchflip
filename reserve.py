"""Reserve play and lot memory — what happens around a lot that does not sell.

On Catawiki a lot whose reserve is not met ends unsold, and the seller can then
offer it to the highest bidder. Many of those lots are also listed again. Three
small things use that:

  memory(conn, lot_id)   earlier runs of the same listing — same seller, same
                         title, same description text — and how each one ended.
                         Shown in alerts and in answers to "ask the bot".
  followup()             after the close: a lot you were alerted on, or asked
                         about, ended UNSOLD with the last bid under your max.
                         Tells you what an offer is worth accepting.
  relists()              one of those unsold lots is on sale again.

    python reserve.py            # one pass of followup + relists (what run.py schedules)
    python reserve.py 108328727  # print the memory for one lot

"Same listing" is a text match, so a dealer who sells several watches under one
identical description shows up too; the line in the alert says "same listing",
not "same watch".

Only the database and Telegram are used. Nothing here opens a web page.
"""

from __future__ import annotations

import logging
import sys
from html import escape

import pandas as pd

import db
import replies
from config import cfg

log = logging.getLogger("reserve")

SCHEMA = """
create table if not exists asked_lots (
    lot_id          text primary key references lots (lot_id) on delete cascade,
    asked_at        timestamptz not null default now(),
    max_bid         numeric(12, 2),      -- the bid the answer told you to place
    fair_max        numeric(12, 2),      -- set when a higher fair max was shown next to it
    tier            text,
    followed_up_at  timestamptz
);
alter table asked_lots enable row level security;
create table if not exists relist_notices (
    lot_id          text primary key references lots (lot_id) on delete cascade,   -- the new lot
    earlier_lot_id  text,
    sent_at         timestamptz not null default now()
);
alter table relist_notices enable row level security;
create index if not exists lots_seller_idx on lots (seller_name);
"""
_READY = False

FOLLOWUP_DAYS = 3         # how long after a result a follow-up is still worth sending
RELIST_DAYS = 60          # how far back an unsold lot is matched against new listings
MEMORY_RUNS = 4           # earlier runs shown in an alert

# Same seller, same title, same description text, closed before this lot.
MEMORY = """
select p.lot_id, p.close_time, r.sold, r.final_price,
       (select s.current_bid from bid_snapshots s where s.lot_id = p.lot_id
        order by s.observed_at desc limit 1) as last_bid
from lots l
join lots p on p.seller_name = l.seller_name and p.title = l.title and p.lot_id <> l.lot_id
           and md5(coalesce(p.description, '')) = md5(coalesce(l.description, ''))
join lot_results r on r.lot_id = p.lot_id
where l.lot_id = %s and l.seller_name is not null
  and p.close_time < coalesce(l.close_time, now())
order by p.close_time desc
limit %s
"""

# Lots you had a max for (alerted or asked), with a result, not yet followed up.
FOLLOWUP = """
with watched as (
    select lot_id, max_bid, fair_max, tier, 0 as src from shortlist_alerts where followed_up_at is null
    union all
    select lot_id, max_bid, fair_max, tier, 1 as src from asked_lots where followed_up_at is null
)
select distinct on (w.lot_id) w.lot_id, w.max_bid, w.fair_max, w.tier,
       l.url, l.watch_model, l.reference_number, r.sold,
       (select s.current_bid from bid_snapshots s where s.lot_id = w.lot_id
        order by s.observed_at desc limit 1) as last_bid
from watched w
join lots l using (lot_id)
join lot_results r using (lot_id)
where r.recorded_at > now() - make_interval(days => %s)
order by w.lot_id, w.src
"""

# An unsold lot you had a max for is open again under a new lot number.
RELISTS = """
select distinct on (n.lot_id) n.lot_id, n.url, n.close_time, n.watch_model, n.reference_number,
       e.lot_id as earlier_lot_id, e.close_time as earlier_close, w.max_bid, w.fair_max,
       (select s.current_bid from bid_snapshots s where s.lot_id = e.lot_id
        order by s.observed_at desc limit 1) as last_bid
from (select lot_id, max_bid, fair_max from shortlist_alerts
      union all
      select lot_id, max_bid, fair_max from asked_lots) w
join lots e on e.lot_id = w.lot_id
join lot_results r on r.lot_id = e.lot_id and not r.sold
join lots n on n.seller_name = e.seller_name and n.title = e.title and n.lot_id <> e.lot_id
           and md5(coalesce(n.description, '')) = md5(coalesce(e.description, ''))
left join lot_results nr on nr.lot_id = n.lot_id
where e.seller_name is not null
  and e.close_time > now() - make_interval(days => %s)
  and nr.lot_id is null and n.close_time > now() and n.close_time > e.close_time
  and not exists (select 1 from relist_notices x where x.lot_id = n.lot_id)
  and not exists (select 1 from shortlist_alerts a where a.lot_id = n.lot_id)
order by n.lot_id, e.close_time desc
"""


def setup(conn) -> None:
    """Create the tables this module needs. Runs once per process."""
    global _READY
    if _READY:
        return
    import shortlist
    shortlist.alerts_schema(conn)
    conn.execute(SCHEMA)
    conn.commit()
    _READY = True


def memory(conn, lot_id: str, limit: int = MEMORY_RUNS) -> list[dict]:
    """Earlier runs of the same listing, newest first:
    [{lot_id, close_time, sold, final_price, last_bid}, …]."""
    return conn.execute(MEMORY, (str(lot_id), limit)).fetchall()


def remember_ask(conn, lot_id: str, max_bid: float | None, fair_max: float | None,
                 tier: str | None) -> None:
    """Note a lot you asked about, so it gets the same follow-up as an alert."""
    setup(conn)
    conn.execute("""
        insert into asked_lots (lot_id, max_bid, fair_max, tier) values (%s, %s, %s, %s)
        on conflict (lot_id) do update set asked_at = now(), max_bid = excluded.max_bid,
            fair_max = excluded.fair_max, tier = excluded.tier""",
                 (str(lot_id), max_bid, fair_max, tier))
    conn.commit()


def _name(row) -> str:
    import shortlist
    line = shortlist.line_of(row["watch_model"])
    ref = row["reference_number"] if isinstance(row["reference_number"], str) else ""
    return escape(" ".join(f"Omega {line if line != 'other' else ''} {ref}".split()))


def _day(ts) -> str:
    return pd.Timestamp(ts).tz_convert(cfg.timezone).strftime("%a %-d %b")


def followup() -> int:
    """Message for each watched lot that ended unsold under your max. Returns how many."""
    import shortlist
    eur = shortlist._eur
    sent = 0
    with db.connect() as conn:
        setup(conn)
        rows = conn.execute(FOLLOWUP, (FOLLOWUP_DAYS,)).fetchall()
        for r in rows:
            top = max(float(r["max_bid"] or 0), float(r["fair_max"] or 0))
            bid = float(r["last_bid"]) if r["last_bid"] is not None else None
            if not r["sold"] and bid is not None and top > 0 and bid < top:
                which = "fair max" if (r["tier"] == "fair" or r["fair_max"] is not None) else "max"
                text = (f"🔁 <b>{_name(r)}</b> ended unsold.\n"
                        f"Last bid seen {eur(bid)}, under your {which} of <b>{eur(top)}</b>.\n"
                        f"If you were the top bidder, the seller can send you an offer, usually "
                        f"within a day: accept up to {eur(top)}, not more.\n"
                        f"If you were not, it may be listed again — I will tell you if the same "
                        f"listing comes back.\n"
                        f'<a href="{escape(r["url"], quote=True)}">Open on Catawiki</a>')
                if replies.send_html(text, r["lot_id"]) is None:
                    continue                  # Telegram refused: try again on the next pass
                sent += 1
            for table in ("shortlist_alerts", "asked_lots"):
                conn.execute(f"update {table} set followed_up_at = now() where lot_id = %s",
                             (r["lot_id"],))
            conn.commit()
    if sent:
        log.info("sent %d unsold follow-ups", sent)
    return sent


def relists() -> int:
    """Message when an unsold lot you had a max for is on sale again. Returns how many."""
    import shortlist
    eur = shortlist._eur
    sent = 0
    with db.connect() as conn:
        setup(conn)
        rows = conn.execute(RELISTS, (RELIST_DAYS,)).fetchall()
        for r in rows:
            top = max(float(r["max_bid"] or 0), float(r["fair_max"] or 0))
            last = (f" with a last bid of {eur(float(r['last_bid']))}"
                    if r["last_bid"] is not None else "")
            text = (f"🔁 <b>{_name(r)}</b> is back on sale — closes {_day(r['close_time'])} "
                    f"at {pd.Timestamp(r['close_time']).tz_convert(cfg.timezone):%H:%M}.\n"
                    f"The same listing ended unsold on {_day(r['earlier_close'])}{last}; "
                    f"your max then was {eur(top)}.\n"
                    f"Reply <i>ask</i> to this message for today's max bid.\n"
                    f'<a href="{escape(r["url"], quote=True)}">Open on Catawiki</a>')
            if replies.send_html(text, r["lot_id"]) is None:
                continue
            conn.execute("insert into relist_notices (lot_id, earlier_lot_id) values (%s, %s) "
                         "on conflict (lot_id) do nothing", (r["lot_id"], r["earlier_lot_id"]))
            conn.commit()
            sent += 1
    if sent:
        log.info("sent %d relist notices", sent)
    return sent


def run() -> None:
    """One pass of both; each part fails on its own."""
    for part in (followup, relists):
        try:
            part()
        except Exception:
            log.exception("%s failed", part.__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if len(sys.argv) > 1:
        import shortlist
        with db.connect() as conn:
            mem = memory(conn, sys.argv[1], 10)
        print(shortlist.memory_text(mem) or "No earlier run of this listing is stored.")
    else:
        run()

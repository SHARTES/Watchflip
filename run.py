"""Entrypoint.

    python run.py poll      # one polling pass
    python run.py sweep     # one sweeping pass
    python run.py health    # how much data do I actually have
    python run.py serve     # long-running scheduler
    python run.py reserve   # unsold follow-ups and relist notices, one pass
    python run.py stock     # price reminders for unsold stock, one pass
"""

from __future__ import annotations

import logging
import sys
import comps
import db
import poller
import replies
import reserve
import stock
import sweeper
import shortlist

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("run")


def health() -> None:
    with db.connect() as conn:
        h = db.health(conn)
    print(f"lots              {h['lots']:>8,}")
    print(f"closed w/ result  {h['results']:>8,}")
    print(f"bid snapshots     {h['snapshots']:>8,}")
    print(f"last poll         {h['last_poll']}")
    print(f"last sweep        {h['last_sweep']}")
    remaining = max(0, 1500 - (h["results"] or 0))
    print(f"\n{remaining:,} more closed lots until the Phase 1 gate.")


def serve() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    sched = BlockingScheduler(timezone="Europe/Vienna")
    sched.add_job(poller.monitor, "cron", minute="*/15", id="monitor",
                  max_instances=1, misfire_grace_time=300)
    # Most lots close between 18:00 and 23:00. Discover (20+ min) and the
    # sweeper (several minutes) hold the only browser, and while they do the
    # monitor skips its pass and the reminder cannot read the live bid. So both
    # stay out of the evening: discover 00:10–16:10 every 2 h and 23:10; results
    # are swept outside 18:00–22:59. Lots are listed days ahead, so none is missed.
    sched.add_job(poller.discover, "cron", hour="0-16/2,23", minute="10", id="discover",
                  max_instances=1, misfire_grace_time=3600)
    sched.add_job(sweeper.run, "cron", hour="0-17,23", minute="20,50", id="sweeper",
                  max_instances=1, misfire_grace_time=900)
    sched.add_job(comps.run, "cron", hour="9, 21", minute="40", id="comps",
                  max_instances=1, misfire_grace_time=3600)
    sched.add_job(lambda: shortlist.run(send=True), "cron", minute="5,35",
                 id="shortlist", max_instances=1, misfire_grace_time=600)
    sched.add_job(shortlist.remind, "cron", minute="*/5", id="remind",
                   max_instances=1, misfire_grace_time=120)
    # What you send the bot in Telegram (replies.py): prices replied to an alert,
    # and "ask the bot" — a lot link or number, answered with its max bid (ask.py).
    # Every minute, so an answer comes while an auction is still running.
    sched.add_job(replies.poll, "cron", minute="*", id="replies",
                  max_instances=1, misfire_grace_time=50)
    # After the close: lots you had a max for that ended unsold under it, and
    # the same listings coming back (reserve.py). Database and Telegram only.
    sched.add_job(reserve.run, "cron", minute="25,55", id="reserve",
                  max_instances=1, misfire_grace_time=600)
    # Unsold stock that has been for sale 14, 28 or 45 days (stock.py).
    sched.add_job(stock.run, "cron", hour="10", minute="0", id="stock",
                  max_instances=1, misfire_grace_time=6 * 3600)
    log.info("scheduler: monitor every 15 min; discover every 2 h and sweep twice an "
             "hour, both paused 17:00–23:00 for the evening closes")
    sched.start()

COMMANDS = {
    "poll": poller.discover,
    "discover": poller.discover,
    "monitor": poller.monitor,
    "sweep": sweeper.run,
    "health": health,
    "serve": serve,
    "comps": comps.run,
    "replies": replies.poll,
    "reserve": reserve.run,
    "stock": stock.run,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "health"
    if cmd not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[cmd]()

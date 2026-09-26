"""Entrypoint.

    python run.py poll      # one polling pass
    python run.py sweep     # one sweeping pass
    python run.py health    # how much data do I actually have
    python run.py serve     # long-running scheduler
"""

from __future__ import annotations

import logging
import sys
import comps
import db
import poller
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
    sched.add_job(poller.discover, "cron", hour="*/2", minute="10", id="discover",
                  max_instances=1, misfire_grace_time=3600)
    sched.add_job(sweeper.run, "cron", minute="20,50", id="sweeper",
                  max_instances=1, misfire_grace_time=900)
    sched.add_job(comps.run, "cron", hour="9, 21", minute="40", id="comps",
                  max_instances=1, misfire_grace_time=3600)
    sched.add_job(lambda: shortlist.run(send=True), "cron", minute="5,35",
                 id="shortlist", max_instances=1, misfire_grace_time=600)
    sched.add_job(shortlist.remind, "cron", minute="*/5", id="remind",
                   max_instances=1, misfire_grace_time=120)
    log.info("scheduler: monitor every 15 min, discover every 2h, sweep twice an hour")
    sched.start()

COMMANDS = {
    "poll": poller.discover,
    "discover": poller.discover,
    "monitor": poller.monitor,
    "sweep": sweeper.run,
    "health": health,
    "serve": serve,
    "comps": comps.run,
}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "health"
    if cmd not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[cmd]()

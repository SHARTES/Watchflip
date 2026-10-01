"""Record how much interest a lot has, at the moment the monitor sees it.

Called from poller._snapshot(). Saves, per visit:
  favorite_count — people who saved the lot ("66 other people are watching")
  n_bids_listed  — bids in the visible bid history
  n_bidders      — distinct bidders among them (a count only; bidder
                   identifiers are never stored or tracked)

Timestamped, so the model later learns only from what was visible before the
close. Never breaks the bid snapshot: runs inside a savepoint and logs a
warning on any problem.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import db

log = logging.getLogger("demand")

SCHEMA = """
create table if not exists demand_snapshots (
    id               bigserial primary key,
    lot_id           text not null references lots (lot_id) on delete cascade,
    observed_at      timestamptz not null default now(),
    minutes_to_close numeric(10, 2),
    favorite_count   int,
    n_bids_listed    int,
    n_bidders        int
);
create index if not exists demand_snapshots_lot_idx on demand_snapshots (lot_id, observed_at);
alter table demand_snapshots enable row level security;
"""
_ready = False


def record(conn, lot_id, rec, close_time) -> None:
    global _ready
    try:
        raw = (rec or {}).get("raw") or {}
        details = raw.get("lotDetailsData") or {}
        fav = details.get("favoriteCount")
        bids = (((raw.get("biddingBlockResponse") or {}).get("biddingHistory") or {})
                .get("bids")) or []
        if fav is None and not bids:
            return
        tokens = {b.get("bidderToken") for b in bids
                  if isinstance(b, dict) and b.get("bidderToken")}
        mins = None
        if close_time is not None:
            mins = round((close_time - datetime.now(timezone.utc)).total_seconds() / 60, 2)

        if not _ready:
            # Own connection, committed at once: a later rollback of the
            # monitor's transaction must not undo the table.
            with db.connect() as setup:
                setup.execute(SCHEMA)
                setup.commit()
            _ready = True

        with conn.transaction():          # savepoint inside the monitor's transaction
            conn.execute(
                "insert into demand_snapshots (lot_id, minutes_to_close, favorite_count, "
                "n_bids_listed, n_bidders) values (%s, %s, %s, %s, %s)",
                (lot_id, mins, int(fav) if isinstance(fav, (int, float)) else None,
                 len(bids), len(tokens)),
            )
    except Exception:
        log.warning("demand snapshot skipped for %s", lot_id, exc_info=True)

"""Database access. Thin wrapper over psycopg3 — no ORM, nothing clever."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
import time
import logging
import psycopg
from psycopg.rows import dict_row

from config import cfg

log = logging.getLogger(__name__)


@contextmanager
def connect(retries: int = 3):
    """Open a connection, retrying briefly on DNS or network failure.

    The laptop sleeps and wakes; a job that fires before Wi-Fi is back sees
    an unresolvable host. Waiting a few seconds costs nothing and saves the
    whole pass — the next discover is two hours away.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with psycopg.connect(cfg.database_url, row_factory=dict_row,
                                 connect_timeout=10) as conn:
                yield conn
                return
        except psycopg.OperationalError as exc:
            last = exc
            if attempt < retries - 1:
                log.warning("db connect failed (%s), retrying in %ds",
                            exc.__class__.__name__, 5 * (attempt + 1))
                time.sleep(5 * (attempt + 1))
    raise last


def upsert_lot(conn, lot: dict[str, Any]) -> bool:
    """Insert a lot, or refresh last_seen_at if we already have it.

    Returns True when the lot is new to us.
    """
    row = conn.execute(
        """
        insert into lots (
            lot_id, url, auction_id, category_slug, title, description,
            photo_urls, seller_country, seller_name,
            brand, watch_model, reference_number, watch_period, watch_year,
            watch_condition, movement, case_diameter_mm,
            estimate_low, estimate_high, currency, close_time, raw
        )
        values (
            %(lot_id)s, %(url)s, %(auction_id)s, %(category_slug)s,
            %(title)s, %(description)s, %(photo_urls)s,
            %(seller_country)s, %(seller_name)s,
            %(brand)s, %(watch_model)s, %(reference_number)s, %(watch_period)s,
            %(watch_year)s, %(watch_condition)s, %(movement)s, %(case_diameter_mm)s,
            %(estimate_low)s, %(estimate_high)s, %(currency)s,
            %(close_time)s, %(raw)s
        )
        on conflict (lot_id) do update set
            last_seen_at = now(),
            close_time   = coalesce(excluded.close_time, lots.close_time),
            title        = coalesce(excluded.title, lots.title),
            brand        = coalesce(excluded.brand, lots.brand),
            watch_model  = coalesce(excluded.watch_model, lots.watch_model),
            reference_number = coalesce(excluded.reference_number, lots.reference_number)
        returning (xmax = 0) as inserted
        """,
        {
            "brand": None,
            "watch_model": None,
            "reference_number": None,
            "watch_period": None,
            "watch_year": None,
            "watch_condition": None,
            "movement": None,
            "case_diameter_mm": None,
            **lot,
            "photo_urls": json.dumps(lot.get("photo_urls") or []),
            "raw": json.dumps(lot.get("raw") or {}),
        },
    ).fetchone()
    return bool(row["inserted"])


def insert_snapshot(
    conn,
    lot_id: str,
    current_bid: float | None,
    bid_count: int | None,
    reserve_met: bool | None,
    close_time: datetime | None,
) -> None:
    minutes_to_close = None
    if close_time is not None:
        delta = close_time - datetime.now(timezone.utc)
        minutes_to_close = round(delta.total_seconds() / 60, 2)

    conn.execute(
        """
        insert into bid_snapshots
            (lot_id, current_bid, bid_count, reserve_met, minutes_to_close)
        values (%s, %s, %s, %s, %s)
        """,
        (lot_id, current_bid, bid_count, reserve_met, minutes_to_close),
    )


def record_result(
    conn,
    lot_id: str,
    final_price: float | None,
    sold: bool,
    bid_count: int | None,
    closed_at: datetime | None,
) -> bool:
    row = conn.execute(
        """
        insert into lot_results (lot_id, final_price, sold, bid_count, closed_at)
        values (%s, %s, %s, %s, %s)
        on conflict (lot_id) do nothing
        returning lot_id
        """,
        (lot_id, final_price, sold, bid_count, closed_at),
    ).fetchone()
    return row is not None


def lots_awaiting_result(conn) -> list[dict[str, Any]]:
    """Closed long enough ago to have a final price, not too old to bother."""
    return conn.execute(
        """
        select l.lot_id, l.url, l.close_time
        from lots l
        left join lot_results r using (lot_id)
        where r.lot_id is null
          and l.close_time is not null
          and l.close_time < now() - make_interval(mins => %s)
          and l.close_time > now() - make_interval(hours => %s)
        order by l.close_time
        limit 200
        """,
        (cfg.sweep_delay_minutes, cfg.sweep_giveup_hours),
    ).fetchall()


def known_lot_ids(conn, lot_ids: Iterable[str]) -> set[str]:
    ids = list(lot_ids)
    if not ids:
        return set()
    rows = conn.execute(
        "select lot_id from lots where lot_id = any(%s)", (ids,)
    ).fetchall()
    return {r["lot_id"] for r in rows}


def log_run(conn, job: str, **counts) -> None:
    conn.execute(
        """
        insert into fetch_log (job, lots_seen, lots_new, results_new, errors, note)
        values (%s, %s, %s, %s, %s, %s)
        """,
        (
            job,
            counts.get("lots_seen", 0),
            counts.get("lots_new", 0),
            counts.get("results_new", 0),
            counts.get("errors", 0),
            counts.get("note"),
        ),
    )


def health(conn) -> dict[str, Any]:
    return conn.execute(
        """
        select
            (select count(*) from lots)                            as lots,
            (select count(*) from lot_results)                     as results,
            (select count(*) from bid_snapshots)                   as snapshots,
            (select max(ran_at) from fetch_log where job='discover')  as last_poll,
            (select max(ran_at) from fetch_log where job='sweeper') as last_sweep
        """
    ).fetchone()


def update_close_time(conn, lot_id: str, close_time) -> None:
    conn.execute(
        "update lots set close_time = %s, last_seen_at = now() where lot_id = %s",
        (close_time, lot_id),
    )


def lots_due_for_monitoring(conn, limit: int = 40) -> list[dict[str, Any]]:
    """Open lots whose last snapshot is older than their cadence allows.

    Cadence tightens as the close approaches: nothing beyond 72h, then daily,
    six-hourly, hourly, and every 15 minutes in the final hour. Almost all the
    information in an auction arrives at the end, so that is where the page
    budget goes.
    """
    return conn.execute(
        """
        select l.lot_id, l.url, l.close_time,
               round(extract(epoch from (l.close_time - now())) / 3600.0, 2)
                   as hours_left
        from lots l
        left join lot_results r using (lot_id)
        where r.lot_id is null
          and l.close_time > now()
          and l.close_time < now() + interval '72 hours'
          and coalesce(
                (select max(s.observed_at) from bid_snapshots s
                 where s.lot_id = l.lot_id),
                to_timestamp(0)
              ) < now() - case
                when l.close_time - now() < interval '1 hour'   then interval '15 minutes'
                when l.close_time - now() < interval '6 hours'  then interval '1 hour'
                when l.close_time - now() < interval '24 hours' then interval '6 hours'
                else interval '24 hours'
              end
        order by l.close_time
        limit %s
        """,
        (limit,),
    ).fetchall()
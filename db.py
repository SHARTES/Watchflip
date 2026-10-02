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

    Only the connect is retried. An error inside the `with` body (a dropped
    connection mid-query) is raised as it is: retrying there made Python
    report "generator didn't stop after throw()" instead of the real error.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            conn = psycopg.connect(cfg.database_url, row_factory=dict_row,
                                   connect_timeout=10)
        except psycopg.OperationalError as exc:
            last = exc
            if attempt < retries - 1:
                log.warning("db connect failed (%s), retrying in %ds",
                            exc.__class__.__name__, 5 * (attempt + 1))
                time.sleep(5 * (attempt + 1))
            continue
        with conn:                      # commit on success, roll back on error, close
            yield conn
        return
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
            **{k: _no_nul(v) for k, v in lot.items()},
            "photo_urls": json.dumps(_no_nul(lot.get("photo_urls") or [])),
            "raw": json.dumps(_no_nul(lot.get("raw") or {})),
        },
    ).fetchone()
    return bool(row["inserted"])


def _no_nul(value):
    """Remove NUL characters, which PostgreSQL refuses in text and jsonb.

    A seller's description occasionally carries one (pasted from a PDF or a
    form); without this the whole lot failed to save and was lost.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {_no_nul(k): _no_nul(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_nul(v) for v in value]
    return value


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


# Which open lots matter most to the monitor. 0 = the vintage Omegas you are
# alerted on, 1 = vintage lots of the candidate brands (better Seikos only),
# 2 = everything else, which still feeds the price model with what is left.
MONITOR_TIER_SQL = """
    case
      when not (coalesce(l.gender in ('men', 'unisex'), false)
                and (coalesce(l.watch_year between 1950 and 1989, false)
                     or coalesce(l.watch_period ~ '^(1950|1960|1970|1980)', false)))
        then 2
      when lower(l.brand) = lower(%(target)s) then 0
      when lower(translate(l.brand, 'èéêÈÉÊ', 'eeeeee')) = any(%(candidates)s)
           and (lower(l.brand) <> 'seiko'
                or concat_ws(' ', l.watch_model, l.title, l.reference_number) ~* %(premium)s)
        then 1
      else 2
    end"""


def lots_due_for_monitoring(conn, limit: int = 40) -> list[dict[str, Any]]:
    """Open lots whose last snapshot is older than their cadence allows.

    Cadence tightens as the close approaches: nothing beyond 72h, then daily,
    six-hourly, hourly, and every 15 minutes in the final hour. Almost all the
    information in an auction arrives at the end, so that is where the page
    budget goes.

    When more lots are due than one pass can read, your own lots go first
    (MONITOR_TIER_SQL), each tier soonest close first, so the ones skipped
    are never the ones you might bid on.
    """
    return conn.execute(
        """
        select l.lot_id, l.url, l.close_time,
               round(extract(epoch from (l.close_time - now())) / 3600.0, 2)
                   as hours_left,
               """ + MONITOR_TIER_SQL + """ as tier
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
        order by tier, l.close_time
        limit %(limit)s
        """,
        {"limit": limit, "target": cfg.target_brand, "candidates": cfg.candidate_keys,
         "premium": cfg.premium_seiko_pattern},
    ).fetchall()
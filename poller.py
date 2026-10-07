"""Poller: discover new lots, and re-snapshot open ones as they near closing.

Two separate passes with different economics:

  discover() walks the catalogue and fetches only lots we have never seen.
  Each new lot costs one page and yields a row plus one snapshot; after that
  the sweeper can collect its final price even if we never look again.

  monitor() ignores the catalogue entirely and works from our own database,
  re-snapshotting open lots on a ladder that tightens as the close approaches.
  Almost all the information in an auction arrives at the end, so that is
  where the page budget goes.

Both passes abandon their work the moment the session looks blocked. A
blocked session that keeps requesting turns a temporary block into a
permanent one.
"""

from __future__ import annotations

import logging

import db
import demand
import re
from config import cfg
from fetcher import fetcher
from parse import LOT_URL_RE, looks_gone, parse_listing_page, parse_lot_page

log = logging.getLogger("poller")


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")


def is_target_lot(rec: dict) -> bool:
    """Whether this lot is in the vertical we intend to trade.

    This is telemetry, not a gate. Rows are stored regardless: the hammer
    model learns how the auction itself behaves — how it punishes bad photos,
    odd closing hours, thin descriptions — and that needs the whole catalogue,
    not one brand. The vertical is expressed as a WHERE clause at training
    time, where it costs nothing and can be changed.
    """
    brand = rec.get("brand")
    if not isinstance(brand, str) or brand.casefold() != cfg.target_brand.casefold():
        return False

    model = rec.get("watch_model")
    if not isinstance(model, str) or not any(
        t in model.casefold() for t in cfg.target_models
    ):
        return False

    year = rec.get("watch_year")
    if isinstance(year, int):
        return cfg.vintage_year_min <= year <= cfg.vintage_year_max

    # Three quarters of Catawiki sellers give a decade rather than a year —
    # "1970-1979". That is precise enough for the only thing this decides,
    # which is vintage versus modern. Treated as an overlap, not containment:
    # a 1940-1949 lot reaches into a range starting at 1950 and is worth
    # keeping, while 2010-2020 is not.
    period = rec.get("watch_period")
    if isinstance(period, str):
        years = [int(y) for y in YEAR_RE.findall(period)]
        if years:
            return (
                min(years) <= cfg.vintage_year_max
                and max(years) >= cfg.vintage_year_min
            )

    return False

def is_within_purchase_budget(rec: dict) -> bool:
    """Whether the current bid still fits the all-in cap.

    Not a buy signal — just a rough guardrail for the future alert layer.
    It ignores per-lot shipping, resale fees and any servicing the watch needs.
    """
    bid = rec.get("_current_bid")
    return bid is not None and bid <= cfg.max_entry_bid


def _snapshot(conn, lot_id, rec, close_time) -> None:
    demand.record(conn, lot_id, rec, close_time)
    db.insert_snapshot(
        conn,
        lot_id=lot_id,
        current_bid=rec.get("_current_bid"),
        bid_count=rec.get("_bid_count"),
        reserve_met=(
            bool(rec["_reserve_met"]) if rec.get("_reserve_met") is not None else None
        ),
        close_time=close_time,
    )


def discover_lot_urls(f) -> list[str]:
    """Collect lot URLs from every configured catalogue source.

    Sources are walked in order, so put the target vertical first: the caller
    truncates at max_lots_per_run and whatever comes last gets dropped.
    """
    urls: list[str] = []

    for base_url, pages in cfg.watch_urls:
        for page in range(1, pages + 1):
            if f.should_stop:
                log.error("session blocked — stopping catalogue walk")
                break

            sep = "&" if "?" in base_url else "?"
            url = base_url if page == 1 else f"{base_url}{sep}page={page}"
            html = f.get_html(url)
            if not html:
                continue

            found = parse_listing_page(html)
            log.info("listing page %d of %s -> %d lot links", page, base_url, len(found))
            if not found:
                break
            urls.extend(found)

        if f.should_stop:
            break

    seen: set[str] = set()
    return [u for u in urls if not (u in seen or seen.add(u))]


def discover() -> None:
    """Walk the catalogue and fetch every lot we have not seen before."""
    stored = new = targets = affordable = errors = 0

    with db.connect() as conn, fetcher(lock_timeout=3600) as f:
        urls = discover_lot_urls(f)
        log.info("%d distinct lot urls in catalogue", len(urls))

        by_id: dict[str, str] = {}
        for u in urls:
            m = LOT_URL_RE.search(u)
            if m:
                by_id.setdefault(m.group(1), u)

        already = db.known_lot_ids(conn, by_id.keys())
        fresh = [u for lid, u in by_id.items() if lid not in already]
        log.info("%d already known, %d new", len(already), len(fresh))

        if len(fresh) > cfg.max_lots_per_run:
            log.warning(
                "%d new lots but the cap is %d — the rest wait for the next pass",
                len(fresh), cfg.max_lots_per_run,
            )

        for url in fresh[: cfg.max_lots_per_run]:
            if f.should_stop:
                log.error("session blocked — aborting discover pass")
                break

            html = f.get_html(url)
            if not html:
                errors += 1
                continue

            rec = parse_lot_page(html, url)
            if rec is None:
                errors += 1
                log.warning("could not parse %s", url)
                continue

            bid = rec.get("_current_bid")
            if bid is not None and bid > cfg.training_max_price:
                continue

            if is_target_lot(rec):
                targets += 1
                affordable += int(is_within_purchase_budget(rec))

            try:
                new += int(db.upsert_lot(
                    conn, {k: v for k, v in rec.items() if not k.startswith("_")}
                ))
                _snapshot(conn, rec["lot_id"], rec, rec.get("close_time"))
                conn.commit()
                stored += 1
            except Exception:
                errors += 1
                conn.rollback()
                log.exception("db write failed for %s", rec["lot_id"])

        db.log_run(conn, "discover", lots_seen=stored, lots_new=new, errors=errors,
                   note=f"targets={targets} affordable={affordable}")
        conn.commit()

    log.info(
        "discover: %d stored (%d new), %d on-target, %d within budget, %d errors",
        stored, new, targets, affordable, errors,
    )


def monitor(limit: int | None = None) -> None:
    """Re-snapshot open lots that the closing-time ladder says are due.

    Yields quickly if discover is holding the browser. Monitor runs four times
    an hour and a deep discover pass takes the better part of one, so whoever
    waits patiently here starves the other. Monitor is the one that can afford
    to skip: its lots come back on the next pass, while a lost discover means
    lots that close without ever being recorded.
    """
    limit = limit or cfg.monitor_batch_size

    with db.connect() as conn:
        due = db.lots_due_for_monitoring(conn, limit)

    if not due:
        log.info("nothing due for monitoring")
        return

    log.info("%d lots due (%d yours, %d candidate brands), nearest closes in %.1f h",
             len(due), sum(r.get("tier") == 0 for r in due), sum(r.get("tier") == 1 for r in due),
             min(float(r["hours_left"]) for r in due))
    if len(due) >= limit:
        log.warning(
            "the ladder is saturated at %d lots — snapshots are being skipped; "
            "raise monitor_batch_size or loosen the cadence",
            limit,
        )

    snapped = extended = errors = gone = 0

    try:
        with fetcher(lock_timeout=60) as f, db.connect() as conn:
            for row in due:
                if f.should_stop:
                    log.error("session blocked — aborting monitor pass")
                    break

                html = f.get_html(row["url"])
                if not html:
                    errors += 1
                    continue

                rec = parse_lot_page(html, row["url"])
                if rec is None:
                    if looks_gone(html):
                        # Pulled before closing (by the seller or Catawiki).
                        # Record it the way the sweeper does, so it stops
                        # being scored and alerted as an open lot.
                        try:
                            db.record_result(conn, lot_id=row["lot_id"], final_price=None,
                                             sold=False, bid_count=None,
                                             closed_at=row["close_time"])
                            conn.commit()
                            gone += 1
                        except Exception:
                            errors += 1
                            conn.rollback()
                            log.exception("could not record removed lot %s", row["lot_id"])
                        continue
                    errors += 1
                    continue

                close_time = rec.get("close_time") or row["close_time"]
                try:
                    # Catawiki extends lots in the final minutes. Recording the
                    # new close time keeps the sweeper from arriving while the
                    # lot is still running.
                    if rec.get("close_time") and rec["close_time"] != row["close_time"]:
                        db.update_close_time(conn, row["lot_id"], rec["close_time"])
                        extended += 1

                    _snapshot(conn, row["lot_id"], rec, close_time)
                    conn.commit()
                    snapped += 1
                except Exception:
                    errors += 1
                    conn.rollback()
                    log.exception("snapshot failed for %s", row["lot_id"])

            db.log_run(conn, "monitor", lots_seen=snapped, errors=errors,
                       note=f"extended={extended} gone={gone}")
            conn.commit()
    except TimeoutError:
        log.info("browser busy (discover is probably running) — skipping this pass")
        return

    log.info("monitor: %d snapshots, %d extensions, %d removed, %d errors",
             snapped, extended, gone, errors)


run = discover


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    run()
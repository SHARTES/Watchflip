"""Sweeper: revisit closed lots and record the final price.

Never writes a result for a lot that is still running. Catawiki extends lots
in the final minutes, so the close_time we stored earlier is not authoritative
— only the one on the page right now is.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from bs4 import BeautifulSoup
import parse
import db
from fetcher import fetcher
from parse import MONEY_RE, as_float, parse_lot_page

log = logging.getLogger("sweeper")

STILL_OPEN = "still_open"
UNKNOWN = "unknown"

UNSOLD_MARKERS = (
    "not sold",
    "reserve not met",
    "no bids",
    "unsold",
    "closed without",
)


def outcome(html: str, rec: dict | None):
    """Return (final_price, sold), or STILL_OPEN / UNKNOWN."""
    now = datetime.now(timezone.utc)

    if rec is None:
        return UNKNOWN

    close_time = rec.get("close_time")
    if close_time is None:
        return UNKNOWN
    if close_time > now:
        return STILL_OPEN

    if rec.get("_sold") is True:
        price = rec.get("_current_bid")
        return (price, True) if price else UNKNOWN
    if rec.get("_sold") is False:
        return None, False

    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True).lower()
    if any(marker in text for marker in UNSOLD_MARKERS):
        return None, False

    m = re.search(
        r"(?:winning bid|sold for|final bid|hammer)\D{0,20}" + MONEY_RE.pattern,
        text, re.I,
    )
    if m:
        return as_float(m.group(0)), True

    return UNKNOWN


def run() -> None:
    recorded = still_open = unknown = errors = gone = 0

    with db.connect() as conn:
        pending = db.lots_awaiting_result(conn)

    if not pending:
        log.info("nothing to sweep")
        return

    log.info("%d lots awaiting a result", len(pending))

    with fetcher(lock_timeout=1800) as f, db.connect() as conn:
        for row in pending:
            if f.should_stop:
                log.error("session blocked — aborting sweep")
                break

            html = f.get_html(row["url"])
            if not html:
                errors += 1
                continue

            # Catawiki removes some lot pages entirely. Retrying those every
            # half hour for five days burns pages for nothing, so record them
            # once as unsold and let them leave the queue.
            if parse.looks_gone(html):
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

            rec = parse_lot_page(html, row["url"])
            res = outcome(html, rec)
            try:
                if res == STILL_OPEN:
                    still_open += 1
                    db.update_close_time(conn, row["lot_id"], rec["close_time"])
                    conn.commit()
                    log.info("%s still open, extended to %s",
                             row["lot_id"], rec["close_time"])
                    continue

                if res == UNKNOWN:
                    unknown += 1
                    log.warning("%s: could not determine outcome, will retry",
                                row["lot_id"])
                    continue

                price, sold = res
                wrote = db.record_result(
                    conn,
                    lot_id=row["lot_id"],
                    final_price=price,
                    sold=sold,
                    bid_count=(rec or {}).get("_bid_count"),
                    closed_at=rec.get("close_time") or row["close_time"],
                )
                recorded += int(wrote)
                conn.commit()
            except Exception:
                errors += 1
                conn.rollback()
                log.exception("could not record result for %s", row["lot_id"])

        db.log_run(conn, "sweeper", results_new=recorded, errors=errors,
                   note=f"still_open={still_open} unknown={unknown} gone={gone}")
        conn.commit()

    log.info("sweeper: %d recorded, %d still open, %d unknown, %d errors",
             recorded, still_open, unknown, errors)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    run()
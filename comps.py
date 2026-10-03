"""Price reference collection from eBay.

    python comps.py            # one collection pass
    python comps.py status     # what is in the table
    python comps.py calibrate  # how far asks sit above Catawiki realised prices
    python comps.py coverage   # which open target lots have a price reference

Why it works this way
---------------------

Marketplace Insights was refused, so there is no source of realised eBay
prices. Active listings are asks, and asks are a censored sample: the cheap
ones have already sold and left, the overpriced ones remain. Reading the median
ask as a market price will make everything look profitable.

What is obtainable is the *lifecycle*. Poll the same reference repeatedly and
record which item ids vanish. A listing that disappears in three weeks probably
sold near its asking price. One that sits six months is fiction. The gap
between what is still listed and what has vanished is a usable signal, and it
is the closest honest proxy available through the public API.

Caveats worth keeping in mind when reading the output:

* Disappearance is not proof of sale. Sellers withdraw listings, and eBay
  relists automatically — which appears as a vanish followed by a new id for
  the same watch.
* A reference with two or three vanished listings tells you nothing. Treat
  thin references as unknown rather than as cheap.

PRIVACY: no seller fields are read or stored anywhere in this module. That is
a commitment made to eBay, not an oversight. See schema_comps_v2.sql.
"""

from __future__ import annotations

import json
import logging
import sys
import time

import db
import ebay
from config import cfg

log = logging.getLogger("comps")

# How often to re-poll a reference. Short enough to catch disappearances while
# they are still informative, long enough not to burn the daily allowance.
REFRESH_DAYS = 4

# Omega's references come first; the candidate brands share what is left.
# Twice a day this stays far below the eBay Browse API's daily allowance.
MAX_QUERIES_PER_RUN = 90

# Below this many vanished listings, treat a reference's numbers as unknown.
MIN_VANISHED = 5

# Results asked for per query (the Browse API's maximum). A result set that
# fills the page is only the top of eBay's best-match order, which reshuffles
# between polls: a listing missing from it may simply have slipped below the
# cut, so disappearances are only recorded when the page was NOT full.
PAGE_LIMIT = 200


# ------------------------------------------------------------- what to query

def references_to_poll(conn, limit: int) -> list[dict]:
    """References attached to lots we could still bid on, soonest close first.

    Priority is deliberate: a reference on a lot closing tomorrow is worth a
    call, one on a lot that closed last week is not.
    """
    return conn.execute(
        """
        select l.brand, l.reference_number,
               count(*) as open_lots,
               min(l.close_time) as next_close
        from lots l
        left join lot_results r using (lot_id)
        left join comp_queries q
               on q.query_brand = l.brand
              and q.query_reference = l.reference_number
        where (
                -- Vintage men's/unisex only. Without this the budget goes on
                -- modern Seamasters and Speedmasters, which are not what we buy.
                ((l.brand ilike %(target)s
                  or (lower(translate(l.brand, 'èéêÈÉÊ', 'eeeeee')) = any(%(candidates)s)
                      and (lower(l.brand) <> 'seiko'
                           or concat_ws(' ', l.watch_model, l.title, l.reference_number) ~* %(premium)s)))
                 and l.gender in ('men', 'unisex')
                 and (l.watch_year between 1950 and 1989
                      or l.watch_period ~ '^(1950|1960|1970|1980)'))
                -- Seiko dress/tank style: any gender, up to 1999.
                or (lower(l.brand) = 'seiko'
                    and concat_ws(' ', l.watch_model, l.title) ~* %(style)s
                    and (l.watch_year between 1960 and 1999
                         or l.watch_period ~ '^(196|197|198|199)')))
          and l.reference_number is not null
          and l.reference_number ~ '^[0-9]'
          and length(l.reference_number) between 4 and 20
          and r.lot_id is null
          and l.close_time > now()
          and (q.last_run_at is null
               or q.last_run_at < now() - make_interval(days => %(refresh)s))
        group by 1, 2
        order by (l.brand ilike %(target)s) desc, min(l.close_time)
        limit %(limit)s
        """,
        {"target": cfg.target_brand, "candidates": cfg.candidate_keys,
         "premium": cfg.premium_seiko_pattern, "style": cfg.style_seiko_pattern,
         "refresh": REFRESH_DAYS, "limit": limit},
    ).fetchall()


# --------------------------------------------------------------- persistence

def upsert_listing(conn, row: dict) -> None:
    """Insert a listing, or mark an existing one as seen again."""
    conn.execute(
        """
        insert into comps (
            comp_id, source, marketplace, title, reference_number,
            item_condition, price, first_price, currency, is_sold, sold_at,
            query_brand, query_reference, raw
        )
        values (
            %(comp_id)s, %(source)s, %(marketplace)s, %(title)s,
            %(reference_number)s, %(item_condition)s, %(price)s, %(price)s,
            %(currency)s, %(is_sold)s, %(sold_at)s, %(query_brand)s,
            %(query_reference)s, %(raw)s
        )
        on conflict (comp_id) do update set
            price          = excluded.price,
            last_seen_at   = now(),
            seen_count     = comps.seen_count + 1,
            -- A listing that reappears was never gone; clear the flag rather
            -- than leaving a false disappearance in the history.
            disappeared_at = null
        """,
        {**row, "raw": json.dumps({})},
    )


def mark_vanished(conn, brand: str, reference: str, seen_ids: list[str]) -> int:
    """Flag listings for this reference that were absent from the latest result.

    Only listings seen at least twice are eligible: one observed once and then
    missing is more likely a paging artefact than a sale.
    """
    rows = conn.execute(
        """
        update comps
           set disappeared_at = now()
         where query_brand = %s
           and query_reference = %s
           and disappeared_at is null
           and seen_count >= 2
           and last_seen_at < now() - interval '1 hour'
           and not (comp_id = any(%s))
        returning comp_id
        """,
        (brand, reference, seen_ids or [""]),
    ).fetchall()
    return len(rows)


def record_query(conn, brand, reference, results, vanished, source) -> None:
    conn.execute(
        """
        insert into comp_queries
            (query_brand, query_reference, results, sold_results, source,
             vanished_this_run)
        values (%s, %s, %s, 0, %s, %s)
        on conflict (query_brand, query_reference) do update set
            last_run_at       = now(),
            results           = excluded.results,
            source            = excluded.source,
            vanished_this_run = excluded.vanished_this_run
        """,
        (brand, reference, results, source, vanished),
    )


# --------------------------------------------------------------------- steps

def run(limit: int = MAX_QUERIES_PER_RUN) -> None:
    client = ebay.EbayClient()

    with db.connect() as conn:
        targets = references_to_poll(conn, limit)

    if not targets:
        log.info("no open target lots need a price reference right now")
        return

    log.info("%d references to poll", len(targets))
    queried = stored = vanished_total = errors = 0

    with db.connect() as conn:
        for t in targets:
            brand, ref = t["brand"], t["reference_number"]
            query = f"{brand} {ref}"

            try:
                items, source = client.search_sold(query, limit=PAGE_LIMIT)
            except ebay.EbayError:
                errors += 1
                log.exception("query failed for %s", query)
                continue

            queried += 1
            seen_ids: list[str] = []

            for item in items:
                row = ebay.to_comp(item, source, brand, ref)
                if row is None or row["price"] is None:
                    continue
                try:
                    upsert_listing(conn, row)
                    seen_ids.append(row["comp_id"])
                    stored += 1
                except Exception:
                    errors += 1
                    conn.rollback()
                    log.exception("could not store listing for %s", query)

            gone = (mark_vanished(conn, brand, ref, seen_ids)
                    if len(items) < PAGE_LIMIT else 0)
            vanished_total += gone
            record_query(conn, brand, ref, len(items), gone, source)
            conn.commit()

            log.info("%-26s %3d live, %2d vanished  (%d open lots, next %s)",
                     query, len(seen_ids), gone, t["open_lots"],
                     t["next_close"].strftime("%d %b"))
            time.sleep(0.4)

    log.info("comps: %d references, %d listings seen, %d newly vanished, %d errors",
             queried, stored, vanished_total, errors)


def status() -> None:
    with db.connect() as conn:
        top = conn.execute(
            """
            select query_reference, listings, still_listed, vanished,
                   asking_median, vanished_median, avg_days_listed
            from ask_summary
            where vanished >= %s
            order by vanished desc
            limit 15
            """,
            (MIN_VANISHED,),
        ).fetchall()
        overall = conn.execute(
            """
            select count(*) as listings,
                   count(*) filter (where disappeared_at is not null) as vanished,
                   count(distinct query_reference) as refs,
                   min(first_seen_at)::date as tracking_since
            from comps
            """
        ).fetchone()

    print(f"{overall['listings']:,} listings across {overall['refs']} references, "
          f"tracked since {overall['tracking_since']}")
    print(f"{overall['vanished']:,} have vanished\n")

    if not top:
        print(f"No reference has {MIN_VANISHED}+ vanished listings yet.\n"
              "Disappearance needs repeated polls over days — run this daily\n"
              "and come back. Until then there is nothing here to read.")
        return

    print(f"{'reference':<22}{'live':>6}{'gone':>6}{'asking':>9}{'vanished':>10}{'days':>7}")
    for r in top:
        print(f"{r['query_reference']:<22}{r['still_listed']:>6}{r['vanished']:>6}"
              f"{r['asking_median'] or 0:>9.0f}{r['vanished_median'] or 0:>10.0f}"
              f"{r['avg_days_listed'] or 0:>7.1f}")

    print("\nIf 'vanished' sits well below 'asking', the cheap listings are the\n"
          "ones leaving — which is what you would expect if they are selling.\n"
          "If the two are equal, disappearance is not tracking sales here.")


def calibrate() -> None:
    """How far do eBay asks sit above what Catawiki actually realises?

    This is the number behind a rule like "it asks 500, so 450 is safe". It
    does not give a retail resale price — it bounds how optimistic asks are
    relative to auction reality, which is enough to stop the margin being a
    guess.
    """
    with db.connect() as conn:
        rows = conn.execute(
            """
            select c.query_reference,
                   percentile_cont(0.5) within group (order by c.price) as ask,
                   percentile_cont(0.5) within group (order by r.final_price) as realised,
                   count(distinct c.comp_id) as n_asks,
                   count(distinct l.lot_id)  as n_sales
            from comps c
            join lots l on l.reference_number = c.query_reference
                       and l.brand ilike c.query_brand
            join lot_results r on r.lot_id = l.lot_id and r.sold
            where c.price > 0 and r.final_price > 0
              and c.query_brand ilike %s
              and l.gender in ('men', 'unisex')
              and (l.watch_year between 1950 and 1989
                   or l.watch_period ~ '^(1950|1960|1970|1980)')
            group by 1
            having count(distinct l.lot_id) >= 1 and count(distinct c.comp_id) >= 3
            order by 5 desc
            """,
            (cfg.target_brand,),
        ).fetchall()

    if not rows:
        print("No reference has both eBay asks and a Catawiki sale yet.\n"
              "Run a few collection passes first.")
        return

    pairs = [(r, float(r["ask"]) / float(r["realised"]))
             for r in rows if r["realised"] and float(r["realised"]) > 0]
    ratios = sorted(x for _, x in pairs)
    n = len(ratios)

    print(f"{n} references with both an eBay ask and a Catawiki sale\n")
    print(f"{'reference':<22}{'ask':>9}{'catawiki':>11}{'ratio':>8}{'n':>5}")
    for r, ratio in pairs[:15]:
        print(f"{r['query_reference']:<22}{float(r['ask']):>9.0f}"
              f"{float(r['realised']):>11.0f}{ratio:>7.2f}x{r['n_sales']:>5}")

    def pct(p):
        return ratios[min(int(n * p), n - 1)]

    print(f"\nask / realised:  p25 {pct(.25):.2f}x   median {pct(.5):.2f}x   "
          f"p75 {pct(.75):.2f}x")
    print("\nA wide spread here means the ratio is not stable enough to use as a\n"
          "single discount factor, and a fixed safety margin would be a guess\n"
          "dressed up as a rule.")


def coverage() -> None:
    """Which open target lots have a usable price reference, and which do not."""
    with db.connect() as conn:
        row = conn.execute(
            """
            with open_target as (
              select l.lot_id, l.reference_number, l.brand
              from lots l left join lot_results r using (lot_id)
              where r.lot_id is null and l.close_time > now()
                and l.brand ilike %s
                and (l.watch_year between 1950 and 1989
                     or l.watch_period ~ '^(1950|1960|1970|1980)')
            )
            select count(*) as open_lots,
                   count(reference_number) as with_ref,
                   count(*) filter (where exists (
                     select 1 from ask_summary a
                     where a.query_reference = open_target.reference_number
                       and a.listings >= 3)) as with_asks,
                   count(*) filter (where exists (
                     select 1 from ask_summary a
                     where a.query_reference = open_target.reference_number
                       and a.vanished >= %s)) as with_vanished
            from open_target
            """,
            (cfg.target_brand, MIN_VANISHED),
        ).fetchone()

    print(f"open target lots        {row['open_lots']:>6,}")
    print(f"  with a reference      {row['with_ref']:>6,}")
    print(f"  with eBay asks        {row['with_asks']:>6,}")
    print(f"  with vanished data    {row['with_vanished']:>6,}")
    print("\nThe last line is what a value estimate can actually rest on.\n"
          "Everything above it is a lot you would have to price by judgement.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "status": status, "calibrate": calibrate,
     "coverage": coverage}.get(cmd, lambda: sys.exit(__doc__))()

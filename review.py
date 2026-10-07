"""One-page review of the database and the last two weeks. Read-only.

    python review.py
"""
import db
from config import cfg

CAP = (cfg.max_all_in_cost - cfg.expected_inbound_shipping
       - cfg.catawiki_buyer_fixed_fee) / (1 + cfg.catawiki_buyer_fee_rate)
TARGET = """lower(l.brand) = 'omega' and l.gender in ('men', 'unisex')
    and (l.watch_year between 1950 and 1989 or l.watch_period ~ '^(1950|1960|1970|1980)')"""


def show(conn, title, sql, params=None):
    rows = conn.execute(sql, params or {}).fetchall()
    print(f"\n== {title}")
    if not rows:
        print("   (nothing)")
        return
    cols = list(rows[0].keys())
    w = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    print("   " + "  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    for r in rows:
        print("   " + "  ".join(str(r[c]).ljust(w[i]) for i, c in enumerate(cols)))


with db.connect() as conn:
    show(conn, "A. What is stored", """
        select (select count(*) from lots) as lots,
               (select count(*) from lot_results) as closed,
               (select count(*) from lot_results where sold) as sold,
               (select count(*) from bid_snapshots) as bid_snapshots,
               (select count(*) from comps) as ebay_listings,
               (select count(*) from shortlist_alerts) as alerts_sent,
               (select to_char(min(first_seen_at), 'DD Mon') from lots) as since""")

    show(conn, "B. Vintage Omega per week (the target)", f"""
        select to_char(date_trunc('week', l.close_time), 'DD Mon') as week,
               count(*) as closed,
               count(*) filter (where r.sold) as sold,
               count(*) filter (where r.sold and r.final_price <= %(cap)s) as sold_in_budget,
               round(percentile_cont(.5) within group (order by r.final_price)
                     filter (where r.sold)) as median_hammer
        from lots l join lot_results r using (lot_id)
        where {TARGET} and l.close_time > now() - interval '6 weeks'
        group by 1, date_trunc('week', l.close_time) order by date_trunc('week', l.close_time)""",
         {"cap": CAP})

    show(conn, "C. Alerts, last 10 days", """
        select to_char(a.sent_at at time zone 'Europe/Vienna', 'DD Mon HH24:MI') as sent,
               left(l.title, 42) as title, round(a.max_bid) as max_bid,
               round(a.exp_profit) as exp_profit,
               case when r.lot_id is null then 'open'
                    when not r.sold and r.final_price is null and r.bid_count is null then 'PAGE GONE'
                    when not r.sold then 'unsold'
                    when r.final_price <= a.max_bid then 'winnable'
                    else 'went higher' end as outcome,
               round(r.final_price) as hammer, l.lot_id
        from shortlist_alerts a join lots l using (lot_id)
        left join lot_results r using (lot_id)
        where a.sent_at > now() - interval '10 days'
        order by a.sent_at""")

    show(conn, "D. All alerts: how they ended", """
        select count(*) as alerts,
               count(*) filter (where r.sold and r.final_price <= a.max_bid) as winnable,
               count(*) filter (where r.sold and r.final_price > a.max_bid) as went_higher,
               count(*) filter (where not r.sold and r.final_price is null and r.bid_count is null) as page_gone,
               round((percentile_cont(.5) within group (order by r.final_price / nullif(a.max_bid, 0))
                     filter (where r.sold and r.final_price > a.max_bid))::numeric, 2) as higher_by_x
        from shortlist_alerts a left join lot_results r using (lot_id)""")

    show(conn, "E. What vintage Omega actually sold in budget, last 3 weeks (by model line)", f"""
        select coalesce(l.watch_model, '?') as model, count(*) as sold,
               round(percentile_cont(.5) within group (order by r.final_price)) as median_hammer
        from lots l join lot_results r using (lot_id)
        where {TARGET} and r.sold and r.final_price <= %(cap)s
          and l.close_time > now() - interval '21 days'
        group by 1 order by 2 desc limit 12""", {"cap": CAP})

    show(conn, "F. Other brands sold in budget, last 3 weeks", """
        select l.brand, count(*) as sold,
               round(percentile_cont(.5) within group (order by r.final_price)) as median_hammer
        from lots l join lot_results r using (lot_id)
        where lower(l.brand) in ('seiko', 'longines', 'universal genève', 'universal geneve', 'tissot')
          and l.gender in ('men', 'unisex') and r.sold
          and r.final_price between 80 and %(cap)s
          and l.close_time > now() - interval '21 days'
        group by 1 order by 2 desc""", {"cap": CAP})

    show(conn, "G. Your trades", """
        select lot_id, left(title, 35) as title, hammer, paid_total, listed_price,
               sold_price, channel from trades order by bought_on""")

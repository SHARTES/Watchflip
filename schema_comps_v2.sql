-- comps v2 — track listings over time, not just snapshots.
--
-- Without Marketplace Insights there is no source of realised eBay prices.
-- What is obtainable is the lifecycle of a listing: when it appeared, how long
-- it stayed, and when it vanished. A listing that disappears quickly probably
-- sold near its asking price; one that sits for months is a wish. Neither is
-- as good as a transaction record, but the difference between the two is
-- measurable and it is the closest honest proxy available.
--
-- PRIVACY: no seller or buyer identifiers here, ever. We told eBay this
-- application persists no user personal data in order to claim exemption from
-- Marketplace Account Deletion notifications. Keep that true.

alter table comps add column if not exists first_seen_at  timestamptz not null default now();
alter table comps add column if not exists last_seen_at   timestamptz not null default now();
alter table comps add column if not exists disappeared_at timestamptz;
alter table comps add column if not exists seen_count     int not null default 1;
alter table comps add column if not exists first_price    numeric(12, 2);

create index if not exists comps_alive_idx on comps (query_reference, disappeared_at);

alter table comp_queries add column if not exists vanished_this_run int not null default 0;

-- Per reference: what is being asked, and what vanished.
--
-- Read `vanished_median` as a soft upper bound on the clearing price, not as a
-- sale price. Listings also disappear because the seller withdrew them, and
-- eBay relists automatically, which shows up as a disappearance followed by a
-- new id. Treat a reference with few vanished items as unknown.
create or replace view ask_summary as
select
    query_brand,
    query_reference,
    count(*)                                              as listings,
    count(*) filter (where disappeared_at is null)        as still_listed,
    count(*) filter (where disappeared_at is not null)    as vanished,
    round(percentile_cont(0.5) within group (
        order by price) filter (where disappeared_at is null)::numeric, 2) as asking_median,
    round(percentile_cont(0.5) within group (
        order by price) filter (where disappeared_at is not null)::numeric, 2) as vanished_median,
    round(percentile_cont(0.25) within group (
        order by price) filter (where disappeared_at is not null)::numeric, 2) as vanished_p25,
    round(avg(extract(epoch from (disappeared_at - first_seen_at)) / 86400.0)
          filter (where disappeared_at is not null)::numeric, 1) as avg_days_listed,
    max(last_seen_at)                                     as last_polled
from comps
where price is not null and price > 0
group by 1, 2;

-- Comparable sales from eBay. Run once against the project database.
--
-- PRIVACY NOTE, load-bearing: no seller or buyer identifiers are stored here,
-- by design. We told eBay this application persists no user personal data in
-- order to claim exemption from Marketplace Account Deletion notifications.
-- Do not add sellerUsername, feedback score, location or any account field to
-- this table — that statement has to stay true.

create table if not exists comps (
    comp_id          text primary key,        -- eBay item id
    source           text not null,           -- ebay_insights | ebay_browse
    marketplace      text,
    title            text,
    reference_number text,                    -- normalised, from the query
    item_condition   text,
    price            numeric(12, 2),
    currency         text,
    is_sold          boolean not null,
    sold_at          timestamptz,             -- null for active listings
    query_brand      text,
    query_reference  text,
    fetched_at       timestamptz not null default now(),
    raw              jsonb
);

create index if not exists comps_ref_idx      on comps (query_brand, query_reference);
create index if not exists comps_sold_idx     on comps (is_sold, sold_at);

-- What we have already asked eBay, so we do not burn the daily call budget
-- re-querying the same reference every hour.
create table if not exists comp_queries (
    query_brand      text not null,
    query_reference  text not null,
    last_run_at      timestamptz not null default now(),
    results          int not null default 0,
    sold_results     int not null default 0,
    source           text,
    primary key (query_brand, query_reference)
);

-- Median realised price per reference, with the sample size alongside it.
-- n matters more than the median: below roughly eight sales the number is
-- noise dressed up as a fact.
create or replace view comp_summary as
select
    query_brand,
    query_reference,
    count(*)                                                             as n,
    round(percentile_cont(0.2) within group (order by price)::numeric, 2) as p20,
    round(percentile_cont(0.5) within group (order by price)::numeric, 2) as median,
    round(percentile_cont(0.8) within group (order by price)::numeric, 2) as p80,
    min(price)                                                           as min_price,
    max(price)                                                           as max_price,
    max(sold_at)                                                         as newest_sale
from comps
where is_sold and price is not null
group by 1, 2;

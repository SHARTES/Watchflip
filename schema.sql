-- Phase 0 schema. Run once against your Supabase / Postgres instance.
-- Everything here is append-mostly: the point of Phase 0 is to accumulate
-- closed lots with final prices, which is the only asset you cannot buy later.

create table if not exists lots (
    lot_id          text primary key,
    url             text not null,
    auction_id      text,
    category_slug   text,
    title           text,
    description     text,
    photo_urls      jsonb  not null default '[]'::jsonb,
    photo_count     int    generated always as (jsonb_array_length(photo_urls)) stored,
    seller_country  text,
    seller_name     text,
    brand           text,
    watch_model     text,
    reference_number text,
    watch_period    text,
    watch_year      smallint,
    watch_condition text,
    movement        text,
    case_diameter_mm numeric(5, 2),
    estimate_low    numeric(12, 2),
    estimate_high   numeric(12, 2),
    currency        text   not null default 'EUR',
    close_time      timestamptz,
    first_seen_at   timestamptz not null default now(),
    last_seen_at    timestamptz not null default now(),
    raw             jsonb
);

-- Safe migration for databases created before the Omega-focused collector.
alter table lots add column if not exists brand text;
alter table lots add column if not exists watch_model text;
alter table lots add column if not exists reference_number text;
alter table lots add column if not exists watch_period text;
alter table lots add column if not exists watch_year smallint;
alter table lots add column if not exists watch_condition text;
alter table lots add column if not exists movement text;
alter table lots add column if not exists case_diameter_mm numeric(5, 2);

create index if not exists lots_close_time_idx on lots (close_time);
create index if not exists lots_category_idx   on lots (category_slug);
create index if not exists lots_reference_idx  on lots (brand, reference_number);
create index if not exists lots_watch_slice_idx on lots (brand, watch_model, watch_year);

-- One row per observation. The bid trajectory is a strong feature for the
-- hammer model later, so sample it even though it feels redundant now.
create table if not exists bid_snapshots (
    id           bigserial primary key,
    lot_id       text not null references lots (lot_id) on delete cascade,
    observed_at  timestamptz not null default now(),
    current_bid  numeric(12, 2),
    bid_count    int,
    reserve_met  boolean,
    minutes_to_close numeric(10, 2)
);

create index if not exists bid_snapshots_lot_idx
    on bid_snapshots (lot_id, observed_at desc);

-- The training labels. One row per lot, written once by the sweeper.
create table if not exists lot_results (
    lot_id       text primary key references lots (lot_id) on delete cascade,
    final_price  numeric(12, 2),
    sold         boolean not null,
    bid_count    int,
    closed_at    timestamptz,
    recorded_at  timestamptz not null default now()
);

-- Operational visibility. If the poller silently breaks you want to know
-- from a query, not from noticing the table stopped growing three weeks on.
create table if not exists fetch_log (
    id           bigserial primary key,
    ran_at       timestamptz not null default now(),
    job          text not null,
    lots_seen    int not null default 0,
    lots_new     int not null default 0,
    results_new  int not null default 0,
    errors       int not null default 0,
    note         text
);

-- Convenience view: everything the models will eventually train on.
create or replace view training_lots as
select
    l.lot_id,
    l.title,
    l.description,
    l.category_slug,
    l.photo_count,
    l.seller_country,
    l.estimate_low,
    l.estimate_high,
    l.close_time,
    extract(hour from l.close_time at time zone 'Europe/Vienna') as close_hour_vienna,
    extract(dow  from l.close_time at time zone 'Europe/Vienna') as close_dow_vienna,
    r.final_price,
    r.sold,
    r.bid_count,
    (
        select s.current_bid
        from bid_snapshots s
        where s.lot_id = l.lot_id
          and s.minutes_to_close between 1380 and 1500   -- roughly T-24h
        order by s.minutes_to_close
        limit 1
    ) as bid_at_t_minus_24h,
    (
        select count(*) from bid_snapshots s where s.lot_id = l.lot_id
    ) as snapshot_count,
    -- New fields are appended so an existing view can be upgraded safely.
    l.brand,
    l.watch_model,
    l.reference_number,
    l.watch_period,
    l.watch_year,
    l.watch_condition,
    l.movement,
    l.case_diameter_mm
from lots l
join lot_results r using (lot_id);

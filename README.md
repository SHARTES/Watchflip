# watchflip — phase 0

Collects Catawiki lots, bid trajectories, and final prices into Postgres.
No models, no alerts, no bidding. The only goal is to accumulate closed lots
with known final prices, because that data arrives at the speed auctions close
and cannot be bought or backfilled later.

## Current research focus

The collector is configured for vintage **Omega** watches from 1950–1989 in
the Seamaster, Genève and De Ville families.  It keeps comparable results up
to €800, while the future alert layer uses a separate all-in buying limit of
€300. With the default €25 inbound-shipping reserve and Catawiki's 9% + €3
buyer fee, that means a current maximum bid of **€249.54**.

Change this focus in `config.py`; do not use the €300 purchase cap to shrink
the training set, since higher-priced examples with the same reference are
valuable comparables.

**Gate to move on: 1,500 rows in `lot_results`.**

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

cp .env.example .env      # fill in DATABASE_URL
psql "$DATABASE_URL" -f schema.sql
```

## Calibrate the parser first

This is the step people skip and then discover three weeks later that every
price in the database is 100x wrong.

```bash
HEADLESS=0 python probe.py https://www.catawiki.com/en/l/<some-real-lot>
```

Read the output carefully and do three things:

1. Map the real key names into `CANDIDATES` in `parse.py`.
2. Answer the cents question. If the page shows €210 and the parser says
   €21,000, set `MINOR_UNITS = True`.
3. Confirm `close_time` is right. Everything downstream keys off it.

Re-run `probe.py` until every field matches what you see on screen.

## Run

```bash
python run.py poll      # one pass
python run.py sweep     # one pass
python run.py health    # how far from the gate
python run.py serve     # scheduler: poll every 15 min, sweep twice an hour
```

For the first catalogue run, use an interactive browser so you can accept any
consent prompt or see an access restriction yourself:

```bash
HEADLESS=0 python run.py poll
```

If the log says `Catawiki denied access`, stop the collector. Do not retry in a
tight loop or try to bypass the restriction; resolve it in the normal browser
session and only then return to the scheduled mode.

Deploy `serve` on a small VPS under systemd or `screen`. It needs a browser,
so a container without Chromium will not work.

## Operating notes

- `PAGE_DELAY` is 6 seconds with jitter. Leave it. The value of this system is
  that it runs uninterrupted for months, and nothing threatens that more than
  hammering the site. Scraping and automated access are very likely against
  Catawiki's terms — you are accepting an account-ban risk, so keep the
  footprint small and do not put money in escrow on an account you are also
  scraping from.
- Start with one category and the narrow price band in `config.py`. Widening
  before the parser is proven just multiplies bad rows.
- Check `python run.py health` weekly. If `results` stops growing while `lots`
  keeps growing, the sweeper's outcome detection has broken against a markup
  change — fix `UNSOLD_MARKERS` and the regex in `sweeper.py`.

## What's deliberately missing

Feature extraction, the comps store, both models, the decision engine and the
Telegram alerter. None of them can be validated without the data this collects,
so building them now would just be building on a guess.

## Sanity queries

```sql
-- is the data plausible?
select count(*), min(final_price), percentile_cont(0.5) within group (order by final_price),
       max(final_price)
from lot_results where sold;

-- do we have trajectories, not just endpoints?
select avg(snapshot_count)::numeric(6,1) from training_lots;

-- does closing hour matter? (the first hypothesis worth testing)
select close_hour_vienna, count(*),
       avg(final_price)::numeric(10,2) as avg_price
from training_lots where sold
group by 1 order by 1;
```

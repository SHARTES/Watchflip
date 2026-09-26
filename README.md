# watchflip

**Can you predict what a vintage watch will sell for at auction — well enough to buy below value and resell at a profit?**

watchflip monitors Catawiki auctions of vintage Omega watches (men's and unisex, 1950–1989), predicts each lot's closing price with a calibrated uncertainty band, estimates what the watch would resell for, and sends a short list of lots worth bidding on to Telegram. A human looks at the photos and places the bid. There is no auto-bidding.

It is a research project first. The most useful results are the ones that said *no*.

---

## Key findings

1. **The auction is mostly efficient.** On a typical evening, 52 of 63 candidate lots were already bid past the price at which they could still be profitable. The model's job turned out to be *focus*: finding the few lots per week that close below value, not beating the market on average.
2. **Resale value is the hard part, not the hammer price.** The closing price can be predicted to ~12% (median absolute error). The resale value is where naive estimates were off by up to 3×.
3. **Asking prices lie in predictable ways.** The median of live eBay listings overstated realistic resale value for every reference checked by hand. Fixing it took four explicit rules (below). One case: a Seamaster 166.0213 was valued at €1,703 from the raw median, €1,524 after removing parts and outliers, and €1,032 after collapsing one dealer's repeated listings — the one comparable auction sale was €800.
4. **A plausible feature can be wrong.** Seller identity looked promising (some sellers' lots close far above expectation). Tested as a leave-one-out encoding on held-out data, it made the model *worse* (+0.6 to +0.9 points of error) and was left out.
5. **Most of the edge is on the sell side.** At realistic resale prices the margin per watch is €120–250, and capital, not deal flow, limits volume. Profit depends on selling above the cheapest comparable listing — presentation, service history, trust — which no model on the buy side can supply.

---

## How it works

```
 Catawiki ──► discover (every 2 h) ──► lots
          ──► monitor  (every 15 min, tightening toward close) ──► bid_snapshots
          ──► sweeper  (twice an hour, after close) ──► lot_results   ← training labels
 eBay API ──► comps    (twice a day) ──► live asking prices per reference

 lots + snapshots + results + comps
          ──► model.py      closing-price model (p20 / p50 / p80)
          ──► asks.py       resale value from cleaned eBay listings
          ──► shortlist.py  max bid, win probability, profit ──► Telegram
                            + last-call reminder with the live bid 30 min before close
```

Python, Playwright, PostgreSQL (Supabase), scikit-learn, APScheduler under macOS `launchd`.

### 1. Closing-price model

- **Target:** log hammer price of sold lots. Prices are right-skewed; squared error on raw euros would let a few expensive lots dominate.
- **Model:** gradient-boosted quantile regression (`HistGradientBoostingRegressor`) for the 20th, 50th and 80th percentiles.
- **Features:** line, year, case size, condition, movement, material and condition keywords parsed from the text (solid gold vs. plated vs. steel, serviced, box, papers, defects), photo count, Catawiki's estimate where present, closing time, a smoothed leave-one-out reference-price encoding, and the **bid 1–5 hours before close** — legitimate, because that is when the bidding decision is made.
- **Split by time, never at random.** Lots closing the same evening share bidders; a random split leaks and flatters the score.
- **Conformal calibration.** Raw quantile bands were too narrow (54% of prices fell outside a band meant to hold 40%). The band edges are shifted by an amount measured on a held-out week, separately for lots with and without a late bid, because their accuracy differs threefold.

| Held-out vintage-Omega lots (271) | Median abs. error | With late bid | Without | Inside p20–p80 (target 60%) |
|---|---|---|---|---|
| Baseline: median price of the model line | 29.9% | | | |
| Model | **12.3–12.7%** | 9.3–10.8% | 23.7–30.6% | 61–63% |

### 2. Resale value

Realised eBay prices are not available (Marketplace Insights is restricted), so resale value is built from **live asking prices**, in three tiers from most to least specific:

1. **eBay listings of the same reference**, cleaned:
   - drop spare parts ("dial only", "for parts"), other currencies, and far outliers;
   - drop solid-gold, two-tone and diamond variants when valuing a steel watch;
   - count near-identical prices once — a single dealer listing the same watch four times must not set the value;
   - take the **lower quarter**, not the median: a new seller has to price near the cheaper end to sell.
2. **Earlier Catawiki sales of the same reference**, scaled to eBay level by a ratio **measured each run** from references that have both (currently 1.39× on 24 references).
3. **A listing-only model** for lots with no comparables: what similar watches typically fetch. Its overestimate is **measured on a held-out week each run and divided out**. Before correction it read 14% too high against real evidence; after, 2% too low.

### 3. Decision

- **Landed cost** = hammer + 9% buyer's fee + €3 + inbound shipping, + 20% import VAT for sellers outside the EU.
- **Net sale** = value × a realism factor − outbound shipping − a 5% returns reserve.
- **Max bid** = the highest hammer that keeps the required margin (25% for evidence-based values, 35% for model-only values), capped by the budget.
- **Win probability** = the calibrated band's probability of closing at or below the max bid.
- Lots above 20% win probability go to Telegram ~3 hours before close; a reminder ~30 minutes before close re-reads the live bid and says either "still under your max" or "skip it".

### 4. Backtest

`backtest.py` and `analyze.py` replay recent auctions as if live: the model trains only on earlier lots, the decision uses only the bid visible hours before close, and a lot counts as won only if it actually closed at or below the max bid (the winner's curse is built in). The model's predicted number of wins has tracked actual wins closely (e.g. 4.5 expected vs. 2 won; 5.9 vs. 6), which is the check that the probabilities are honest.

---

## Things that went wrong, and what they taught

- **A 3× overvaluation.** The first resale estimate (median eBay ask) looked like large profits. Checking won lots against eBay by hand showed the gap; `explain.py` now prints every listing behind a value with a reason for each one excluded.
- **A spurious time-of-day effect.** Closing hour appeared to move price 3×. Controlling for model line removed it entirely — expensive lines simply cluster at 20:00.
- **Label corruption, caught before it happened.** Catawiki extends lots in the final minutes. The sweeper would have written a live bid as a final price; it now re-reads the end time and refuses to record a result for a lot still running.
- **A broken feature.** Photo count scraped every image on the page and capped at 12. It is now read from the lot data (5–47 photos per lot), which improved the listing-only model from 21.7% to 20.3% error.
- **A sleeping laptop.** Coverage of the decisive last-hours bid fell from 99% to 0% over a week, silently, because the host slept. The fix was operational, not modelling — and a reminder that monitoring the pipeline matters as much as the model.

---

## Constraints

- **Polite, single-session collection.** Several seconds between page loads, one browser session, and a hard stop after repeated denials. No IP rotation or fingerprint spoofing.
- **No personal data from eBay is stored.** Comparable listings keep price, title and condition only — never seller identifiers.
- **No data in this repository.** Code only.
- **No auto-bidding.** Every bid is placed by a person who has looked at the photos. Dial condition, redials and case wear live in the images, which the model does not see.

---

## Repository

| File | Purpose |
|---|---|
| `run.py`, `run-daemon.sh` | Scheduler entry point and `launchd` wrapper |
| `poller.py`, `sweeper.py`, `fetcher.py`, `parse.py` | Discovery, monitoring, results, polite browser session, page parsing |
| `comps.py`, `ebay.py` | eBay Browse API collection of comparable listings |
| `model.py` | Features, closing-price model, conformal calibration |
| `asks.py` | Cleaning eBay listings into a resale value |
| `shortlist.py` | Live scoring, Telegram alerts and reminders |
| `backtest.py`, `analyze.py` | Honest replays, model comparison, sensitivity and capital checks |
| `explain.py` | Full breakdown of one lot's value and profit |
| `photos.py`, `when.py`, `q.py`, `probe.py` | Photo-count backfill, when to keep the host awake, ad-hoc SQL, parser probe |
| `schema*.sql` | Database schema |

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # fill in database URL, eBay and Telegram credentials
psql "$DATABASE_URL" -f schema.sql -f schema_comps_v2.sql
python run.py serve               # or install run-daemon.sh as a launchd agent
python analyze.py                 # once there is data: funnel, models, replay
```

---

## Limitations and next steps

- **Resale is still an assumption.** The next step is a trade log — buy price, service cost, sale price, days to sell — so real sales replace the realism factor.
- **Condition lives in the photos.** A vision model flagging redials, damaged dials or replaced hands would address the largest remaining source of error.
- **The host must stay awake.** Catawiki blocks headless browsers, so collection runs on a laptop; an always-on machine would close the coverage gaps.

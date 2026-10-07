"""Shortlist — which open lots are worth bidding on right now.

    python shortlist.py              # print the ranked shortlist
    python shortlist.py --send       # print, and send new lots to Telegram
    python shortlist.py --remind     # last-call reminders for alerted lots closing soon
    python shortlist.py --countries  # list seller countries, to check the EU mapping

For every open vintage-Omega lot closing within WINDOW_HOURS it combines:

  predicted close   the hammer model, retrained on every closed lot, with its
                    band edges calibrated on the most recent week
  expected resale   the lower quarter of the live eBay asks for the same
                    reference, after dropping parts, other currencies, other
                    materials and model lines, serviced dealer stock (for an
                    as-is lot) and outliers; else earlier Catawiki sales or
                    the listing-only model
  total cost        hammer + Catawiki fee + inbound shipping, plus import VAT
                    and customs handling when the seller is outside the EU
  selling costs     per channel (CHANNELS): in person, Chrono24, eBay

and turns them into a maximum bid (planned on PLAN_CHANNEL, after a reserve
for repairs and returns), the chance the lot closes under it, and the cash
profit on each channel. Nothing here places bids. It tells you where to look.

Every constant below is a judgement call. They are collected here so they can
be changed in one place as your own sales replace the assumptions.
"""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from urllib.parse import quote_plus

import numpy as np
import pandas as pd

import db
import asks
import model
from config import cfg

log = logging.getLogger("shortlist")

WINDOW_HOURS = 6          # score lots closing within this many hours
CAL_DAYS = 7              # calibrate on this many most recent days

# eBay tier: the value is already the lower quarter of the cleaned asks (see
# asks.py), roughly where a new seller has to list.
# value × realism is the QUICK-SALE price: priced among the cheapest quarter of
# comparable watches, after a buyer's offer. Retail, but priced to move.
# Re-set it from the median "realised ÷ value" in trade.py show once there are
# 5+ real sales; resale.py's clearing level is a second, weaker check.
REALISM_EBAY = 0.95
REALISM_REF = REALISM_EBAY  # Catawiki tier: same scale as eBay since the ratio is measured
REALISM_LINE = 0.75       # the same, when only a line-level ask is available
MIN_REF_LISTINGS = 3      # cleaned eBay listings needed to trust a reference

TARGET_MARGIN = 0.25      # required profit over landed cost (fixed rule)

# Margin from uncertainty. Each value carries σ — how unsure it is, in log
# terms: few or scattered comparables, eBay and Catawiki disagreeing. The
# required margin is moved around the usual level by that uncertainty:
#     margin = usual margin + MARGIN_PER_SIGMA × (σ − typical σ)
# "Typical σ" is measured on past evidence-valued lots each run, so on
# average the agent is exactly as cautious as before — it just spends that
# caution where the evidence is thin and bids more freely where it is solid.
# analyze.py compares this with the fixed margins on past auctions.
USE_UNCERTAINTY_MARGIN = True
MARGIN_PER_SIGMA = 1.0
MARGIN_MIN, MARGIN_MAX = 0.15, 0.45
SIGMA_REF = {"evidence": None}   # set by calibrate_margin()
SIGMA_FLOOR = 0.05        # even many tight comparables are asks, not sales

# ------------------------------------------------------------ selling costs
# Where you sell decides how much of the sale price you keep. Per channel:
#   fee    share of the price the platform takes
#   fixed  € per sale
#   ship   what sending it costs YOU (insured, price incl. shipping)
# Checked Oct 2026 for a private seller in Austria:
#   in person  Willhaben or Instagram, paid on hand-over: nothing. Willhaben
#              PayLivery (Austria only) is free for the seller too; the buyer
#              pays postage, but Post AG's standard cover ends at €510.
#              Instagram by bank transfer: nothing; PayPal goods & services
#              3.4% + €0.35.
#   Chrono24   private seller 6.5%, paid out via escrow after delivery; you
#              may charge the buyer for insured shipping (then set ship 0).
#   eBay       eBay.at private seller: 11% + 0.42% operating fee + €0.35, on
#              the total including postage.
CHANNELS = {
    "in person": {"fee": 0.0,    "fixed": 0.0,  "ship": 0.0},
    "Chrono24":  {"fee": 0.065,  "fixed": 0.0,  "ship": 15.0},
    "eBay":      {"fee": 0.1142, "fixed": 0.35, "ship": 15.0},
}
# The max bid is planned on this channel: the big market you can always fall
# back to if nobody local buys. Selling in person is then upside. Once
# trade.py shows where your sales really happen, plan on that channel.
PLAN_CHANNEL = "Chrono24"

# Expected cost of what goes wrong after you buy — a return, a regulation, a
# crystal, a watch that stops a week after the sale — as a share of the sale
# price. It is not paid on every watch; it is the average. Applied to the max
# bid only, never shown as money you pay. trade.py costs replace it later.
WARRANTY_RESERVE = 0.05

IMPORT_VAT = 0.20         # Austrian import VAT on goods from outside the EU,
                          # charged on the price + shipping (not Catawiki's fee)
IMPORT_CLEARANCE = 20.0   # the carrier's customs handling charge; duty may add more

MIN_WIN_PROB = 0.20       # below this, a lot is not worth your attention
MIN_CONDITION = 3         # 3 = 'Good'; Fair and Poor are filtered out

# Value from the same reference's Catawiki sales, converted to a resale
# estimate with the ratio between the eBay value and the Catawiki hammer.
#
# The ratio is MEASURED on every run, by calibrate_ratio(), from references
# that have both: for each sold target lot with an eBay value, log(value /
# hammer); median per reference, so one busy reference cannot dominate; then
# the median across references. The old fixed 1.41 was measured against the
# MEDIAN eBay ask; the eBay value is now the cheaper quarter, so 1.41 made the
# Catawiki and feature tiers about 20% too optimistic — exactly the gap the
# backtests kept reporting.
MIN_REF_SALES = 2         # earlier Catawiki sales needed to trust a reference
ASK_TO_HAMMER = 1.20      # fallback only, used if too few references to measure
MIN_RATIO_REFS = 8        # references needed before the measured ratio is used
# When a reference has both eBay listings and Catawiki sales, use the geometric
# mean of the two. Asking prices drift above what watches really fetch, and a
# handful of auction results is noisy; together they are steadier than either.
BLEND_EVIDENCE = True
RATIO_INFO = {"ratio": ASK_TO_HAMMER, "refs": 0, "lots": 0, "measured": False}

# The line-level fallback values an ordinary watch as if it were a typical
# member of a wide range. The backtests showed it inflating profit on exactly
# the lots you then judged by eye. Off by default; the lot is skipped instead.
USE_LINE_FALLBACK = False

# Third tier: value a watch from its features — "what do listings like this one
# usually fetch on Catawiki?" — using a model that sees only the listing, never
# the bids. It covers the lots with no reference evidence at all, which is most
# of them. It is also the tier most exposed to the winner's curse: a lot heading
# well below "typical" may be a bargain or may be flawed in a way the listing
# does not say. So it gets stricter settings, and every hit is flagged for a look.
USE_FEATURE_TIER = True
REALISM_FEAT = 0.85       # 10% below the evidence tiers: no comparables

# The listing-only model overestimates on the lots that matter (measured +9%
# to +19% against real eBay evidence). FEAT_BIAS corrects that. It is measured
# on every run by calibrate_feature_bias(): the listing model is fitted
# WITHOUT the latest week, predicts that week, and is compared with the eBay
# value of the lots there that have one. Never below 1.0 — the correction may
# only make the feature tier more careful, not more optimistic.
FEAT_BIAS = 1.00          # fallback when the latest week has too few lots
                          # (measured 1.00 live on 1 Oct; 1.10 over-corrected)
MIN_BIAS_LOTS = 15
BIAS_INFO = {"bias": FEAT_BIAS, "lots": 0, "measured": False}
FEAT_MARGIN = 0.35        # stricter than TARGET_MARGIN

# References you never want to see again, whatever the numbers say.
EXCLUDE_REFS = {
    "31858801",     # Seamaster, not to your taste
}
TOP_N = 8

# Alerts: the first message when a lot is inside ALERT_HOURS of closing (late
# enough that the bid you see means something), then one short reminder
# REMIND_MINUTES before the end saying whether it is still under your max.
ALERT_HOURS = 3.0
REMIND_MINUTES = 30

# Seller encoding: how a Catawiki seller's lots price relative to what they
# are (model.add_seller_encoding). Off until analyze.py shows it lowers the
# error on your data — part B compares the model with and without it.
USE_SELLER = False

# Features a live lot cannot have yet. bid_count in the training data is the
# FINAL count from the result, and snapshot count grows with a lot's lifetime —
# both would leak the outcome if used to score an open lot.
LIVE_EXCLUDE = ("bid_count", "snapshots")

# The listing-only model also drops every bid. It must answer "what is a watch
# like this worth at auction", not "where is this particular auction heading".
LISTING_EXCLUDE = LIVE_EXCLUDE + ("late_bid", "bid_24h")

EU = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "EL",
    "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI",
    "ES", "SE",
    "AUSTRIA", "BELGIUM", "BULGARIA", "CROATIA", "CYPRUS", "CZECHIA",
    "CZECH REPUBLIC", "DENMARK", "ESTONIA", "FINLAND", "FRANCE", "GERMANY",
    "GREECE", "HUNGARY", "IRELAND", "ITALY", "LATVIA", "LITHUANIA", "LUXEMBOURG",
    "MALTA", "NETHERLANDS", "POLAND", "PORTUGAL", "ROMANIA", "SLOVAKIA",
    "SLOVENIA", "SPAIN", "SWEDEN",
}

Z80 = 0.8416              # standard-normal quantile of the 80th percentile


# ---------------------------------------------------------------- helpers

def line_of(watch_model) -> str:
    m = watch_model.lower() if isinstance(watch_model, str) else ""
    if "constellation" in m:
        return "Constellation"
    if "de ville" in m or "deville" in m:
        return "De Ville"
    if re.search(r"gen.ve", m):
        return "Genève"
    if "speedmaster" in m:
        return "Speedmaster"
    if "seamaster" in m:
        return "Seamaster"
    return "other"


def passes_filters(lot) -> bool:
    """Rules that encode your own judgement, applied before any scoring."""
    cond = model.CONDITION_ORDER.get(str(lot["watch_condition"]).lower())
    if cond is not None and cond < MIN_CONDITION:
        return False
    k = model.ref_key(lot["reference_number"])
    if k and k in {model.ref_key(r) for r in EXCLUDE_REFS}:
        return False
    if str(lot.get("repainted_dial") or "").strip().lower() == "yes":
        return False             # a redial is worth a fraction of an original
    return True


def catawiki_by_ref(closed: pd.DataFrame) -> pd.DataFrame:
    """Catawiki sales per reference, from lots that already sold.

    Keeps each sale's price and case material, so a plated lot is compared
    with plated (or unstated) sales of its reference, not with steel ones.
    """
    c = closed.assign(k=closed["reference_number"].map(model.ref_key))
    c = c.dropna(subset=["k"])
    if c.empty:
        return pd.DataFrame(columns=["median", "count", "sales"])
    c = c.assign(mat=c.apply(model.lot_material, axis=1),
                 lines=(c["watch_model"].fillna("") + " " + c["title"].fillna("")).map(asks.lines_in))
    out = c.groupby("k")["final_price"].agg(["median", "count"])
    out["sales"] = c.groupby("k").apply(
        lambda g: list(zip(g["final_price"].astype(float), g["mat"], g["lines"])))
    return out


def catawiki_value(lot, cw_by_ref, with_spread: bool = False):
    """(median Catawiki price of comparable sales, how many[, log spread]) or None."""
    k = model.ref_key(lot["reference_number"])
    if not k or k not in cw_by_ref.index:
        return None
    mat, line = model.lot_material(lot), asks.lot_line(lot)
    sales = cw_by_ref.loc[k, "sales"]
    if mat in ("steel", "plated", "gold", "bicolor"):
        sales = [x for x in sales if x[1] in (mat, "unknown", "other")]
    if line:
        sales = [x for x in sales if not x[2] or line in x[2]]
    if len(sales) < MIN_REF_SALES:
        return None
    prices = np.array([x[0] for x in sales], float)
    if not with_spread:
        return float(np.median(prices)), len(sales)
    sd = float(np.std(np.log(prices), ddof=1)) if len(prices) > 1 else 0.5
    return float(np.median(prices)), len(sales), sd


def ebay_value(lot, by_ref) -> tuple[float, int] | None:
    """(lower-quarter cleaned ask, listings used) if there are enough, else None."""
    k = model.ref_key(lot["reference_number"])
    if not k or k not in by_ref:
        return None
    v = asks.value(by_ref[k], asks.lot_material(lot), asks.lot_line(lot), asks.lot_serviced(lot))
    return v if v is not None and v[1] >= MIN_REF_LISTINGS else None


def calibrate_ratio(closed: pd.DataFrame, by_ref) -> dict:
    """Measure eBay value ÷ Catawiki hammer, and use it for the other tiers.

    Only sold vintage-Omega lots that pass your filters, so the ratio is the
    one for the watches you actually buy. Pass TRAINING lots only when
    replaying the past, so a test lot never helps set its own value.
    """
    global ASK_TO_HAMMER
    t = closed[model.target_mask(closed)]
    rows = []
    for _, lot in t.iterrows():
        if not passes_filters(lot) or not lot["final_price"] > 0:
            continue
        ev = ebay_value(lot, by_ref)
        if ev is not None:
            rows.append((model.ref_key(lot["reference_number"]),
                         math.log(ev[0] / float(lot["final_price"]))))
    info = {"ratio": ASK_TO_HAMMER, "refs": 0, "lots": len(rows), "measured": False}
    if rows:
        per_ref = pd.DataFrame(rows, columns=["k", "lr"]).groupby("k")["lr"].median()
        info["refs"] = len(per_ref)
        if len(per_ref) >= MIN_RATIO_REFS:
            ASK_TO_HAMMER = float(math.exp(per_ref.median()))
            # Spread across references, to show how much one number hides.
            info.update(ratio=ASK_TO_HAMMER, measured=True,
                        p25=float(math.exp(per_ref.quantile(0.25))),
                        p75=float(math.exp(per_ref.quantile(0.75))),
                        # robust spread of the per-reference ratio (IQR / 1.349)
                        sigma=float((per_ref.quantile(0.75) - per_ref.quantile(0.25)) / 1.349))
    RATIO_INFO.clear()
    RATIO_INFO.update(info)
    return info


def calibrate_feature_bias(live, by_ref) -> dict:
    """How far the listing-only model overshoots real eBay evidence, held out.

    Uses only lots from the calibration week, predicted by a listing model that
    never saw them, and only eBay evidence (Catawiki evidence would include the
    lot's own price). Call after calibrate_ratio().
    """
    global FEAT_BIAS
    rows, fair = getattr(live, "cal_rows", None), getattr(live, "fair_cal", None)
    lr = []
    if rows is not None and fair is not None and len(rows):
        tm = model.target_mask(rows)
        for i, (idx, lot) in enumerate(rows.iterrows()):
            if not tm[i] or not passes_filters(lot):
                continue
            ev = ebay_value(lot, by_ref)
            if ev is not None and fair.get(idx, 0) > 0:
                lr.append(math.log(float(fair[idx]) * ASK_TO_HAMMER / ev[0]))
    info = {"bias": FEAT_BIAS, "lots": len(lr), "measured": False}
    if len(lr) >= MIN_BIAS_LOTS:
        FEAT_BIAS = max(1.0, float(math.exp(np.median(lr))))
        q75, q25 = np.percentile(lr, [75, 25])
        info.update(bias=FEAT_BIAS, measured=True, raw=float(math.exp(np.median(lr))),
                    sigma=float((q75 - q25) / 1.349))
    BIAS_INFO.clear()
    BIAS_INFO.update(info)
    return info


def bias_text() -> str:
    i = BIAS_INFO
    if not i.get("measured"):
        return (f"feature correction ÷{i['bias']:.2f} (fallback — {i['lots']} held-out "
                f"lots with eBay evidence, need {MIN_BIAS_LOTS})")
    note = "" if i["raw"] >= 1.0 else f", measured {i['raw']:.2f} but never below 1.00"
    return (f"feature correction ÷{i['bias']:.2f}, measured on {i['lots']} held-out "
            f"lots with eBay evidence{note}")


def ratio_text() -> str:
    i = RATIO_INFO
    if not i.get("measured"):
        return (f"value ratio {i['ratio']:.2f} (fallback — only {i['refs']} references "
                f"with both eBay and Catawiki evidence, need {MIN_RATIO_REFS})")
    return (f"value ratio {i['ratio']:.2f} × Catawiki hammer, measured on {i['refs']} "
            f"references / {i['lots']} lots (middle half {i['p25']:.2f}–{i['p75']:.2f})")


class Value(tuple):
    """(value, realism, basis, kind) — unpacks like before — plus .sigma and
    .list_price (the median comparable ask, when eBay listings exist)."""
    sigma: float | None = None
    list_price: float | None = None


def _value(value, realism, basis, kind, sigma, list_price=None):
    v = Value((value, realism, basis, kind))
    v.sigma = float(sigma) if sigma is not None else None
    v.list_price = float(list_price) if list_price else None
    return v


def ebay_stats(lot, by_ref) -> dict | None:
    k = model.ref_key(lot["reference_number"])
    if not k or k not in by_ref:
        return None
    s = asks.summary(by_ref[k], asks.lot_material(lot), asks.lot_line(lot), asks.lot_serviced(lot))
    return s if s is not None and s["n"] >= MIN_REF_LISTINGS else None


def value_for(lot, by_ref, by_line, cw_by_ref, fair=None):
    """Expected resale for one lot, from the most specific evidence available.

    Returns (value, realism, description, kind) or None.
      1. live eBay asks for this exact reference, cleaned, lower quarter
      2. Catawiki sales of this exact reference, scaled up to ask level
      3. the listing-only model's typical price, scaled up to ask level
      4. the line-level fallback, only if USE_LINE_FALLBACK is on

    `fair` is the listing-only model's typical Catawiki price for this lot.
    """
    es = ebay_stats(lot, by_ref)
    cw = catawiki_value(lot, cw_by_ref, with_spread=True)
    s_ratio = RATIO_INFO.get("sigma", 0.20)
    if es is not None:
        ve, n, sd_e, med = es["value"], es["n"], es["sd"], es["median"]
        sig_e = math.sqrt(sd_e ** 2 / n + SIGMA_FLOOR ** 2)
    if cw is not None:
        vc, m, sd_c = cw[0] * ASK_TO_HAMMER, cw[1], cw[2]
        # Converting auction prices to eBay level adds the ratio's own spread.
        sig_c = math.sqrt(sd_c ** 2 / m + s_ratio ** 2)
    if es is not None and cw is not None and BLEND_EVIDENCE:
        d = abs(math.log(ve / vc))
        # Averaging two sources halves the noise, unless they disagree more
        # than their noise explains — then the disagreement is the uncertainty.
        sig = max(0.5 * math.sqrt(sig_e ** 2 + sig_c ** 2), d / 2)
        return _value(math.sqrt(ve * vc), REALISM_EBAY,
                      f"eBay €{ve:.0f} ({n} listings) and Catawiki €{vc:.0f} ({m} sales), combined",
                      "ebay", sig, med)
    if es is not None:
        return _value(ve, REALISM_EBAY, f"the cheaper quarter of {n} eBay listings", "ebay", sig_e, med)
    if cw is not None:
        return _value(vc, REALISM_REF, f"{m} Catawiki sales", "catawiki", sig_c)
    if USE_FEATURE_TIER and fair is not None and np.isfinite(fair) and fair > 0:
        return _value(float(fair) * ASK_TO_HAMMER / FEAT_BIAS, REALISM_FEAT,
                      f"similar lots, which typically fetch €{fair:.0f} at auction", "feature",
                      BIAS_INFO.get("sigma", 0.20))
    if USE_LINE_FALLBACK:
        line = line_of(lot["watch_model"])
        if line in by_line:
            return _value(float(by_line[line]), REALISM_LINE, f"{line} line value", "line", 0.35)
    return None


def watch_year_text(lot) -> str:
    y = lot.get("watch_year")
    if pd.notna(y):
        return str(int(y))
    p = lot.get("watch_period")
    return p.replace("-", "–") if isinstance(p, str) else ""


def ebay_search(ref) -> str | None:
    if not isinstance(ref, str) or not ref.strip():
        return None
    m = model.DOTTED_REF_RE.search(ref)
    term = m.group(0) if m else ref.strip()
    return "https://www.ebay.de/sch/i.html?_nkw=" + quote_plus(f"Omega {term}")


def eu_status(country) -> str:
    if not isinstance(country, str) or not country.strip():
        return "unknown"
    return "eu" if country.strip().upper() in EU else "non_eu"


def fee_mult() -> float:
    return 1 + cfg.catawiki_buyer_fee_rate


def fixed_in() -> float:
    return cfg.catawiki_buyer_fixed_fee + cfg.expected_inbound_shipping


def landed(bid: float, non_eu: bool) -> float:
    """Everything you pay to have the watch in your hands."""
    cost = bid * fee_mult() + fixed_in()
    if non_eu:
        cost += IMPORT_VAT * (bid + cfg.expected_inbound_shipping) + IMPORT_CLEARANCE
    return cost


def hammer_for(cost: float, non_eu: bool) -> float:
    """The hammer price whose landed cost is `cost` (the inverse of landed)."""
    per_euro = fee_mult() + (IMPORT_VAT if non_eu else 0)
    fixed = fixed_in() + ((IMPORT_VAT * cfg.expected_inbound_shipping + IMPORT_CLEARANCE)
                          if non_eu else 0)
    return (cost - fixed) / per_euro


def keep(price: float, channel: str) -> float:
    """What you are left with from a sale at `price` on `channel`, after the
    platform fee and shipping — the money that reaches you."""
    c = CHANNELS[channel]
    return price * (1 - c["fee"]) - c["fixed"] - c["ship"]


def net_from(sale: float, channel: str | None = None) -> float:
    """Planning net: what you keep on the plan channel, less the reserve."""
    return keep(sale, channel or PLAN_CHANNEL) - WARRANTY_RESERVE * sale


def margin_for(kind: str, sigma: float | None = None, center: float | None = None) -> float:
    usual = center if center is not None else (FEAT_MARGIN if kind == "feature" else TARGET_MARGIN)
    ref = SIGMA_REF.get("evidence")
    if (not USE_UNCERTAINTY_MARGIN or kind == "feature" or sigma is None
            or not np.isfinite(sigma) or ref is None):
        return usual
    return float(np.clip(usual + MARGIN_PER_SIGMA * (sigma - ref), MARGIN_MIN, MARGIN_MAX))


def calibrate_margin(closed: pd.DataFrame, by_ref, cw_by_ref) -> float | None:
    """Typical uncertainty of evidence-based values, measured on sold lots.
    Call after calibrate_ratio(). Pass training lots only when replaying."""
    t = closed[model.target_mask(closed)]
    sig = []
    for _, lot in t.iterrows():
        if not passes_filters(lot):
            continue
        v = value_for(lot, by_ref, {}, cw_by_ref, None)
        if v is not None and v[3] in ("ebay", "catawiki") and v.sigma is not None:
            sig.append(v.sigma)
    SIGMA_REF["evidence"] = float(np.median(sig)) if len(sig) >= 10 else None
    return SIGMA_REF["evidence"]


def max_bid(value: float, realism: float, non_eu: bool,
            margin: float = TARGET_MARGIN) -> tuple[float, float]:
    """Highest hammer that still clears the margin, capped by the budget.

    Planned on PLAN_CHANNEL at the quick-sale price, after the reserve.
    Returns (max bid, planning net from the sale)."""
    net_sale = net_from(value * realism)
    by_margin = hammer_for(net_sale / (1 + margin), non_eu)
    by_budget = hammer_for(cfg.max_all_in_cost, non_eu)
    return min(by_margin, by_budget), net_sale


def p_under(x: float, p20: float, p50: float, p80: float) -> float:
    """Chance the lot closes at or below x, from the calibrated band.

    Treats the price as log-normal around p50, with a separate spread on each
    side so a lopsided band stays lopsided.
    """
    if x <= 0 or p50 <= 0:
        return 0.0
    lx, l50 = math.log(x), math.log(p50)
    side = (l50 - math.log(p20)) if lx < l50 else (math.log(p80) - l50)
    s = max(side / Z80, 1e-6)
    return 0.5 * (1 + math.erf((lx - l50) / (s * math.sqrt(2))))


# ------------------------------------------------------------------- data

OPEN_QUERY = """
select
    l.lot_id, l.url, l.title, l.description, l.brand, l.watch_model, l.gender,
    l.reference_number, l.watch_period, l.watch_year, l.watch_condition,
    l.movement, l.case_diameter_mm, l.photo_count, l.seller_country, l.seller_name,
    l.estimate_low, l.estimate_high, l.close_time,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Case material").value') #>> '{}' as case_material,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Repainted dial").value') #>> '{}' as repainted_dial,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Original box included").value') #>> '{}' as box_spec,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Original papers included").value') #>> '{}' as papers_spec,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Original warranty included").value') #>> '{}' as warranty_spec,
    jsonb_path_query_first(l.raw, '$.lotDetailsData.specifications[*] ? (@.name == "Band material").value') #>> '{}' as band_material,
    l.raw #>> '{lotDetailsData,seo,ldSchema,aggregateRating,reviewCount}' as seller_reviews,
    l.raw #>> '{lotDetailsData,seo,ldSchema,aggregateRating,ratingValue}' as seller_rating,
    (select s.bid_count from bid_snapshots s where s.lot_id = l.lot_id
     order by s.observed_at desc limit 1) as bid_count,
    (select count(*) from bid_snapshots s where s.lot_id = l.lot_id) as snapshots,
    (select s.current_bid from bid_snapshots s
     where s.lot_id = l.lot_id and s.minutes_to_close between 1200 and 1680
     order by s.minutes_to_close limit 1) as bid_24h,
    (select s.current_bid from bid_snapshots s
     where s.lot_id = l.lot_id and s.minutes_to_close between 60 and 300
     order by s.minutes_to_close limit 1) as late_bid,
    (select s.current_bid from bid_snapshots s where s.lot_id = l.lot_id
     order by s.observed_at desc limit 1) as current_bid,
    (select max(s.observed_at) from bid_snapshots s where s.lot_id = l.lot_id) as last_seen
from lots l
left join lot_results r using (lot_id)
where r.lot_id is null
  and l.close_time > now()
  and l.close_time < now() + make_interval(hours => %s)
"""


def load_open() -> pd.DataFrame:
    try:
        import photos
        photos.fill_new()          # photo counts for lots found since the last run
    except Exception:
        log.exception("photo count fill failed — continuing with what is stored")
    with db.connect() as conn:
        rows = conn.execute(model.with_photos(conn, OPEN_QUERY), (WINDOW_HOURS,)).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close_time"] = pd.to_datetime(df["close_time"], utc=True)
    for col in ("estimate_low", "estimate_high", "case_diameter_mm", "bid_24h",
                "late_bid", "bid_count", "photo_count", "watch_year", "current_bid"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_asks(ref_to_line: dict) -> tuple[dict, dict]:
    """Live eBay listings per normalised reference, and a value per line.

    by_ref is {ref_key: listings}; the value for a particular lot is computed
    in ebay_value(), because which listings count depends on the lot (a steel
    watch is not priced from solid-gold listings).
    """
    by_ref = asks.load()
    rows = []
    for k, g in by_ref.items():
        v = asks.value(g, "steel")
        if v is not None and v[1] >= MIN_REF_LISTINGS and k in ref_to_line:
            rows.append((ref_to_line[k], v[0]))
    # Lower quarter of reference values. A line spans cheap steel and gold
    # chronometers alike, and the middle of that range overvalues an ordinary
    # watch — the backtest showed line-valued lots dominating the "profit".
    by_line = (pd.DataFrame(rows, columns=["line", "v"]).groupby("line")["v"]
               .quantile(0.25).to_dict()) if rows else {}
    return by_ref, by_line


# ------------------------------------------------------------------ model

def build_live_model(closed: pd.DataFrame, open_: pd.DataFrame):
    """Calibrate on the latest week, then refit on everything closed.

    Featurising closed and open lots together keeps the category codes for
    brand, line, movement and country consistent between the two.
    """
    both = pd.concat([closed.assign(_open=False), open_.assign(_open=True)],
                     ignore_index=True)
    f = model.featurise(both)
    cols = [c for c in f.columns if c not in LIVE_EXCLUDE]

    c_idx = both.index[~both["_open"]]
    o_idx = both.index[both["_open"]]
    closed_b, open_b = both.loc[c_idx], both.loc[o_idx]

    cutoff = closed_b["close_time"].max() - pd.Timedelta(days=CAL_DAYS)
    fit_df = closed_b[closed_b["close_time"] <= cutoff]
    cal_df = closed_b[closed_b["close_time"] > cutoff]

    X_fit, X_cal = model.add_ref_encoding(f.loc[fit_df.index, cols],
                                          f.loc[cal_df.index, cols], fit_df, cal_df)
    if USE_SELLER:
        X_fit, X_cal = model.add_seller_encoding(X_fit, X_cal, fit_df, cal_df)
    calibrated, counts = model.fit_calibrated(fit_df, cal_df, X_fit, X_cal)

    # The offsets were measured on held-out data; the point predictions are
    # better with the newest week included, so refit on all of it and keep them.
    X_all, X_open = model.add_ref_encoding(f.loc[c_idx, cols], f.loc[o_idx, cols],
                                           closed_b, open_b)
    if USE_SELLER:
        X_all, X_open = model.add_seller_encoding(X_all, X_open, closed_b, open_b)
    y_all = np.log(closed_b["final_price"].to_numpy(float))
    models = {q: model.fit(X_all, y_all, q) for q in (0.2, 0.5, 0.8)}
    live = model.CalibratedModel(models, calibrated.adj, list(X_all.columns))

    # Listing-only model: the typical price of a watch like this, ignoring how
    # this particular auction is going. Used by the feature valuation tier.
    lcols = [c for c in f.columns if c not in LISTING_EXCLUDE]
    # Held-out check for the feature tier: a listing model that has not seen
    # the calibration week predicts it. Used by calibrate_feature_bias().
    Lf, Lc = model.add_ref_encoding(f.loc[fit_df.index, lcols], f.loc[cal_df.index, lcols],
                                    fit_df, cal_df)
    y_fit = np.log(fit_df["final_price"].to_numpy(float))
    live.fair_cal = pd.Series(np.exp(model.fit(Lf, y_fit, 0.5).predict(Lc)),
                              index=cal_df.index)
    live.cal_rows = cal_df
    L_all, L_open = model.add_ref_encoding(f.loc[c_idx, lcols], f.loc[o_idx, lcols],
                                           closed_b, open_b)
    # No seller encoding here on purpose: this model sets the RESALE value, and
    # a watch is not worth more to your buyer because of who sold it to you.
    listing = model.fit(L_all, y_all, 0.5)
    fair = pd.Series(np.exp(listing.predict(L_open)), index=o_idx)

    return live, X_open, open_b, calibrated.adj, counts, fair


# ---------------------------------------------------------------- scoring

def score() -> tuple[pd.DataFrame, dict]:
    stats = {"window": 0, "target": 0, "filtered": 0, "no_value": 0,
             "past_ceiling": 0, "unlikely": 0, "kept": 0}

    open_ = load_open()
    if open_.empty:
        return pd.DataFrame(), stats
    stats["window"] = len(open_)

    closed = model.load()
    live, X_open, open_b, adj, counts, fair = build_live_model(closed, open_)
    stats["calibration"] = (adj, counts)

    pred = live.predict(X_open)

    ref_to_line = {}
    for frame in (closed, open_):
        for ref, wm in zip(frame["reference_number"], frame["watch_model"]):
            k = model.ref_key(ref)
            if k:
                ref_to_line.setdefault(k, line_of(wm))
    by_ref, by_line = load_asks(ref_to_line)
    cw_by_ref = catawiki_by_ref(closed)
    calibrate_ratio(closed, by_ref)
    calibrate_feature_bias(live, by_ref)
    calibrate_margin(closed, by_ref, cw_by_ref)

    tm = model.target_mask(open_b)
    stats["target"] = int(tm.sum())

    rows = []
    now = pd.Timestamp.now(tz="UTC")
    for i, (idx, lot) in enumerate(open_b.iterrows()):
        if not tm[i]:
            continue
        if not passes_filters(lot):
            stats["filtered"] += 1
            continue

        line = line_of(lot["watch_model"])
        v = value_for(lot, by_ref, by_line, cw_by_ref, fair.get(idx))
        if v is None:
            stats["no_value"] += 1
            continue
        value, realism, basis, kind = v

        country = eu_status(lot["seller_country"])
        non_eu = country == "non_eu"
        margin = margin_for(kind, getattr(v, "sigma", None))
        ceiling, net_sale = max_bid(value, realism, non_eu, margin)

        current = float(lot["current_bid"]) if pd.notna(lot["current_bid"]) else 0.0
        if ceiling <= current:
            stats["past_ceiling"] += 1
            continue

        p20, p50, p80 = (float(pred.loc[idx, c]) for c in ("p20", "p50", "p80"))
        p_win = p_under(ceiling, p20, p50, p80)
        if p_win < MIN_WIN_PROB:
            stats["unlikely"] += 1
            continue

        win_price = max(min(p50, ceiling), current)
        profit = net_sale - landed(win_price, non_eu)

        flags = []
        if non_eu:
            flags.append("non-EU +20% VAT")
        if country == "unknown":
            flags.append("country unknown")
        active_late = pd.notna(X_open.loc[idx, "late_bid"])
        if not active_late:
            flags.append("no late bid" if pd.isna(lot["late_bid"]) else "bidding not started")
        if kind == "line":
            flags.append("line-level value")
        if kind == "feature":
            flags.append("feature-valued: no reference evidence, check photos closely")

        rows.append({
            "lot_id": lot["lot_id"],
            "hours": (lot["close_time"] - now).total_seconds() / 3600,
            "line": line,
            "reference": lot["reference_number"] if isinstance(lot["reference_number"], str) else "—",
            "title": lot["title"],
            "current": current,
            "p20": p20, "p50": p50, "p80": p80,
            "value": value, "basis": basis,
            "max_bid": ceiling,
            "p_win": p_win,
            "profit": profit,
            "exp_profit": p_win * profit,
            "flags": ", ".join(flags),
            "url": lot["url"],
            "compare": ebay_search(lot["reference_number"]),
            "realism": realism, "net_sale": net_sale, "kind": kind,
            "margin": margin, "sigma": getattr(v, "sigma", None),
            "sale": value * realism,
            # first list price: the median comparable ask, never below the quick price
            "list_price": (max(v.list_price, value * realism)
                           if getattr(v, "list_price", None) else None),
            "profit_at_max": net_sale - landed(ceiling, non_eu),
            "landed_at_max": landed(ceiling, non_eu),
            "close_time": lot["close_time"],
            "year": watch_year_text(lot),
            "condition": str(lot["watch_condition"]).split(" - ")[0]
                         if isinstance(lot["watch_condition"], str) else "",
            "country": country,
            "has_late_bid": active_late,
            "seen_at": lot.get("last_seen"),
            "serviced": asks.lot_serviced(lot),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("exp_profit", ascending=False)
    stats["kept"] = len(out)
    return out, stats


# ----------------------------------------------------------------- output

def print_table(out: pd.DataFrame, stats: dict) -> None:
    adj, counts = stats.get("calibration", ({}, {}))
    print(f"lots closing within {WINDOW_HOURS} h: {stats['window']}   "
          f"vintage Omega target: {stats['target']}")
    print(f"skipped — condition below Good: {stats['filtered']}   "
          f"no price reference: {stats['no_value']}   "
          f"already past your ceiling: {stats['past_ceiling']}   "
          f"under {MIN_WIN_PROB:.0%} chance: {stats['unlikely']}")
    if adj:
        parts = [f"{r} n={counts.get(r, 0)}" for r in adj]
        print(f"calibrated on the last {CAL_DAYS} days ({', '.join(parts)})")
    print(ratio_text())
    print(bias_text())
    if SIGMA_REF.get("evidence") is not None and USE_UNCERTAINTY_MARGIN:
        print(f"margin from uncertainty: {TARGET_MARGIN:.0%} at the typical ±"
              f"{SIGMA_REF['evidence']:.0%}, from {MARGIN_MIN:.0%} to {MARGIN_MAX:.0%}")
    print()

    if out.empty:
        print("Nothing worth a look right now.")
        return
    print(f"'if won' = profit if you win at your max bid (the worst case), selling on "
          f"{PLAN_CHANNEL} at the quick-sale price, after the {WARRANTY_RESERVE:.0%} reserve;")
    print("'exp. €' = that profit × the chance of winning, used for ranking.\n")

    print(f"{'closes':>7}  {'line':<15}{'ref':<14}{'now':>6}{'likely':>8}"
          f"{'max bid':>9}{'chance':>8}{'if won':>8}{'exp. €':>8}")
    for _, r in out.head(TOP_N).iterrows():
        print(f"{r['hours']:>6.1f}h  {r['line']:<15}{str(r['reference'])[:13]:<14}"
              f"{r['current']:>6.0f}{r['p50']:>8.0f}{r['max_bid']:>9.0f}"
              f"{r['p_win']:>7.0%}{r['profit_at_max']:>8.0f}{r['exp_profit']:>8.0f}")
        extra = f"band €{r['p20']:.0f}–{r['p80']:.0f} · value from {r['basis']}"
        if r["flags"]:
            extra += f" · {r['flags']}"
        print(f"{'':>9}{extra}")
        print(f"{'':>9}{breakdown(r)}")
        print(f"{'':>9}cash profit at €{r['sale']:.0f} if won at the max: "
              + " · ".join(f"{c} €{p:.0f}" for c, p in channel_profits(r["sale"], r["landed_at_max"])))
        print(f"{'':>9}{r['url']}")
        if isinstance(r.get("compare"), str):
            print(f"{'':>9}compare: {r['compare']}")
        print(f"{'':>9}details: python explain.py {r['lot_id']}")


def channel_profits(price: float, landed_cost: float) -> list[tuple[str, float]]:
    """Cash profit per selling channel for a sale at `price` — fee and
    shipping taken off, the reserve not (it is an average, not a bill)."""
    return [(c, keep(price, c) - landed_cost) for c in CHANNELS]


def breakdown(r) -> str:
    """One line showing where the planning number comes from."""
    sale = r["value"] * r["realism"]
    return (f"value €{r['value']:.0f} × {r['realism']:.0%} = quick sale €{sale:.0f} → "
            f"net €{r['net_sale']:.0f} on {PLAN_CHANNEL} after fee, shipping + "
            f"{WARRANTY_RESERVE:.0%} reserve · max bid €{r['max_bid']:.0f} keeps {r['margin']:.0%} margin"
            + (f" (uncertainty ±{r['sigma']:.0%})" if r.get("sigma") is not None else ""))


EVIDENCE = {
    "ebay": "Solid — priced from comparable eBay listings",
    "catawiki": "Good — priced from earlier Catawiki sales of this reference",
    "feature": "Weak — no direct comparables, estimated from similar lots",
    "line": "Weak — estimated from the model line as a whole",
}


def _eur(x: float) -> str:
    x = round(x)
    return f"−€{-x:,.0f}" if x < 0 else f"€{x:,.0f}"


def _closes(r) -> str:
    local = r["close_time"].tz_convert(cfg.timezone)
    mins = max(0, int(round(r["hours"] * 60)))
    left = f"{mins // 60} h {mins % 60:02d} min" if mins >= 60 else f"{mins} min"
    today = pd.Timestamp.now(tz=cfg.timezone).date()
    day = "today" if local.date() == today else local.strftime("%a %d %b")
    return f"Closes {day} at {local:%H:%M} · in {left}"


def message(r) -> str:
    """The Telegram alert, in HTML (bold headings, tappable links)."""
    from html import escape

    name = f"Omega {r['line'] if r['line'] != 'other' else ''} {r['reference'] if r['reference'] != '—' else ''}"
    name = " ".join(name.split())
    details = " · ".join(x for x in (r["year"], r["condition"]) if x)
    sale = r["value"] * r["realism"]
    margin = r["profit_at_max"] / r["landed_at_max"] if r["landed_at_max"] else 0
    non_eu = r["country"] == "non_eu"

    out = [f"⌚ <b>{escape(name)}</b>"]
    if details:
        out.append(escape(details))
    out.append(_closes(r))
    out.append("")

    out.append(f"<b>Bid up to {_eur(r['max_bid'])}</b> — and not a euro more")
    now = "no bids yet" if r["current"] <= 0 else f"bid {_eur(r['current'])}"
    seen = r.get("seen_at")
    if seen is not None and pd.notna(seen):
        seen = pd.Timestamp(seen)
        seen = seen.tz_localize("UTC") if seen.tzinfo is None else seen
        age = (pd.Timestamp.now(tz="UTC") - seen).total_seconds() / 60
        now += f" at {seen.tz_convert(cfg.timezone):%H:%M}"
        if age > STALE_MINUTES:
            now += f" ({int(age)} min ago — check the page)"
    out.append(f"{now.capitalize()} · likely to close near {_eur(r['p50'])} "
               f"(usually {_eur(r['p20'])}–{_eur(r['p80'])})")
    out.append(f"Chance it closes at or under your max: <b>{r['p_win']:.0%}</b>")
    out.append("")

    out.append(f"<b>If you win at {_eur(r['max_bid'])}</b>")
    vat = " + import VAT and customs handling" if non_eu else ""
    out.append(f"You pay {_eur(r['landed_at_max'])} all-in (Catawiki fee + shipping{vat})")
    lp = r.get("list_price")
    if lp is not None and pd.notna(lp) and lp > sale * 1.03:
        out.append(f"List at about {_eur(lp)} (middle of comparable asks), "
                   f"quick-sale price {_eur(sale)}")
    else:
        out.append(f"Quick-sale price about {_eur(sale)}")
    profits = " · ".join(f"{c} {_eur(p)}" for c, p in channel_profits(sale, r["landed_at_max"]))
    out.append(f"Profit if it sells for {_eur(sale)}: <b>{profits}</b>")
    out.append(f"The max bid still keeps {margin:.0%} selling on {PLAN_CHANNEL}, "
               f"after a {WARRANTY_RESERVE:.0%} reserve for repairs and returns")
    if r["p50"] < r["max_bid"] * 0.95:
        saved = r["landed_at_max"] - landed(r["p50"], non_eu)
        out.append(f"If it closes near the likely {_eur(r['p50'])}: {_eur(saved)} more on each")
    out.append("")

    out.append(f"<b>Evidence:</b> {EVIDENCE.get(r['kind'], r['kind'])}")
    out.append(f"Resale value {_eur(r['value'])}, from {escape(str(r['basis']))}")

    warn = []
    if r.get("serviced"):
        warn.append("Listing mentions a service — ask the seller for the receipt or date")
    else:
        warn.append("No service mentioned — a service would cost €150–250 on top")
    if r["kind"] in ("feature", "line"):
        warn.append("No comparable sales — check the photos closely before bidding")
    if non_eu:
        warn.append("Seller outside the EU — import VAT is already included above")
    if r["country"] == "unknown":
        warn.append("Seller country unknown — VAT may apply on top")
    if not r["has_late_bid"]:
        warn.append("Bidding has barely started, so the likely price is estimated from "
                    "the listing and the chance is less reliable")
    warn.append("Look at the dial, hands and case in every photo — "
                "damage is often not in the description")
    out.append("")
    out.append("<b>Before you bid</b>")
    out.extend(f"• {escape(w)}" for w in warn)
    out.append("")

    links = [f'<a href="{escape(r["url"], quote=True)}">Open on Catawiki</a>']
    if isinstance(r.get("compare"), str):
        links.append(f'<a href="{escape(r["compare"], quote=True)}">Compare on eBay</a>')
    out.append(" · ".join(links))
    out.append(f"<code>python explain.py {escape(str(r['lot_id']))}</code>")
    return "\n".join(out)


def send_new(out: pd.DataFrame) -> None:
    import requests

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — nothing sent")
        return
    if out.empty:
        return

    with db.connect() as conn:
        conn.execute("""
            create table if not exists shortlist_alerts (
                lot_id     text primary key references lots (lot_id) on delete cascade,
                sent_at    timestamptz not null default now(),
                max_bid    numeric(12, 2),
                p_win      numeric(5, 3),
                exp_profit numeric(12, 2)
            )""")
        conn.execute("alter table shortlist_alerts enable row level security")
        sent = {r["lot_id"] for r in
                conn.execute("select lot_id from shortlist_alerts").fetchall()}
        conn.commit()

        fresh = out[~out["lot_id"].isin(sent) & (out["hours"] <= ALERT_HOURS)].head(TOP_N)
        for _, r in fresh.iterrows():
            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": message(r), "parse_mode": "HTML",
                      "disable_web_page_preview": "true"},
                timeout=20,
            )
            if not resp.ok:
                log.warning("telegram refused %s: %s", r["lot_id"], resp.text[:200])
                continue
            conn.execute(
                "insert into shortlist_alerts (lot_id, max_bid, p_win, exp_profit) "
                "values (%s, %s, %s, %s) on conflict (lot_id) do nothing",
                (r["lot_id"], round(r["max_bid"], 2), round(r["p_win"], 3),
                 round(r["exp_profit"], 2)),
            )
            conn.commit()
        log.info("sent %d new lots to Telegram", len(fresh))


STALE_MINUTES = 20        # a stored bid older than this is not shown as "now"
LAST_CHANCE_MINUTES = 12  # if the live check keeps failing, send anyway after this
REMIND_WAIT_SECONDS = 480 # how long the reminder waits for the browser


def _fetch_live(due: list[dict]) -> dict:
    """Open each lot page once, right now, and return {lot_id: (bid, seen, close)}.

    A lot whose page Catawiki has removed comes back as {lot_id: None}, and is
    recorded as closed unsold so nothing scores or reminds about it again.

    Uses the same browser, lock and politeness as the monitor, and stores what
    it reads as a normal snapshot. If the browser is busy or blocked, returns
    what it managed — the caller falls back or retries.
    """
    import poller
    from fetcher import fetcher
    from parse import looks_gone, parse_lot_page

    live = {}
    try:
        # A monitor pass in the evening holds the browser up to ~10 minutes;
        # waiting for it beats sending a bid that is minutes old.
        with fetcher(lock_timeout=REMIND_WAIT_SECONDS) as f, db.connect() as conn:
            for r in due:
                if f.should_stop:
                    break
                html = f.get_html(r["url"])
                rec = parse_lot_page(html, r["url"]) if html else None
                if rec is None and html and looks_gone(html):
                    db.record_result(conn, lot_id=r["lot_id"], final_price=None,
                                     sold=False, bid_count=None,
                                     closed_at=r["close_time"])
                    conn.commit()
                    live[r["lot_id"]] = None
                    continue
                if rec is None or rec.get("_current_bid") is None:
                    continue
                close = rec.get("close_time") or r["close_time"]
                if rec.get("close_time") and rec["close_time"] != r["close_time"]:
                    db.update_close_time(conn, r["lot_id"], rec["close_time"])
                poller._snapshot(conn, r["lot_id"], rec, close)
                conn.commit()
                live[r["lot_id"]] = (float(rec["_current_bid"]),
                                     pd.Timestamp.now(tz="UTC"), pd.Timestamp(close))
    except Exception:
        log.exception("live bid check failed — will retry or fall back")
    return live


def remind() -> None:
    """Last call for alerted lots closing soon: still under your max, or not.

    Checks the CURRENT bid on the lot page just before sending. If that check
    fails, it tries again on the next run (every 5 min); only in the last
    LAST_CHANCE_MINUTES does it send the stored bid instead, clearly marked
    as old.
    """
    import requests
    from html import escape

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        log.warning("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set — nothing sent")
        return

    with db.connect() as conn:
        conn.execute("alter table shortlist_alerts add column if not exists "
                     "reminded_at timestamptz")
        conn.commit()
        due = conn.execute("""
            select a.lot_id, a.max_bid, l.url, l.close_time, l.watch_model,
                   l.reference_number,
                   (select s.current_bid from bid_snapshots s where s.lot_id = a.lot_id
                    order by s.observed_at desc limit 1) as bid,
                   (select s.observed_at from bid_snapshots s where s.lot_id = a.lot_id
                    order by s.observed_at desc limit 1) as seen_at
            from shortlist_alerts a join lots l using (lot_id)
            left join lot_results r using (lot_id)
            where a.reminded_at is null and r.lot_id is null
              and l.close_time > now()
              and l.close_time <= now() + make_interval(mins => %s)
        """, (REMIND_MINUTES,)).fetchall()
    if not due:
        return

    live = _fetch_live(due)
    tz = cfg.timezone
    now = pd.Timestamp.now(tz="UTC")
    sent = 0

    with db.connect() as conn:
        for r in due:
            if r["lot_id"] in live and live[r["lot_id"]] is None:
                # Removed from Catawiki: a link to a dead page helps nobody.
                conn.execute("update shortlist_alerts set reminded_at = now() "
                             "where lot_id = %s", (r["lot_id"],))
                conn.commit()
                log.info("alerted lot %s was removed from Catawiki — no reminder", r["lot_id"])
                continue
            if r["lot_id"] in live:
                bid, seen, close_utc = live[r["lot_id"]]
                fresh = True
            else:
                bid = float(r["bid"]) if r["bid"] is not None else 0.0
                seen = pd.Timestamp(r["seen_at"]) if r["seen_at"] is not None else None
                close_utc = pd.Timestamp(r["close_time"])
                fresh = seen is not None and (now - seen).total_seconds() < STALE_MINUTES * 60
                mins_left = (close_utc - now).total_seconds() / 60
                if not fresh and mins_left > LAST_CHANCE_MINUTES:
                    continue            # try the live check again in 5 minutes

            close = close_utc.tz_convert(tz)
            mins = max(0, int((close_utc - now).total_seconds() // 60))
            ref = r["reference_number"] if isinstance(r["reference_number"], str) else ""
            line = line_of(r["watch_model"])
            name = escape(" ".join(f"Omega {line if line != 'other' else ''} {ref}".split()))
            mx = float(r["max_bid"])
            link = f'<a href="{escape(r["url"], quote=True)}">Open on Catawiki</a>'

            if fresh:
                when = f"right now ({seen.tz_convert(tz):%H:%M})" if r["lot_id"] in live \
                    else f"as of {seen.tz_convert(tz):%H:%M}"
                if bid < mx:
                    text = (f"⏰ <b>{name}</b> closes at {close:%H:%M} · in {mins} min\n"
                            f"Bid {_eur(bid)} {when} — still under your max of <b>{_eur(mx)}</b>.\n"
                            f"If you still like it, place your max bid now.\n{link}")
                else:
                    text = (f"✋ <b>{name}</b> closes at {close:%H:%M}\n"
                            f"Bid {_eur(bid)} {when} — already over your max of {_eur(mx)}. "
                            f"Skip this one.")
            else:
                age = (f"at {seen.tz_convert(tz):%H:%M}, {int((now - seen).total_seconds() // 60)} min ago"
                       if seen is not None else "never")
                text = (f"⏰ <b>{name}</b> closes at {close:%H:%M} · in {mins} min\n"
                        f"⚠️ Could not check the current bid. Last seen: {_eur(bid)} {age} — "
                        f"it has probably moved.\n"
                        f"Your max is <b>{_eur(mx)}</b>. Open the lot and check before bidding.\n{link}")

            resp = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": text, "parse_mode": "HTML",
                      "disable_web_page_preview": "true"},
                timeout=20,
            )
            if not resp.ok:
                log.warning("telegram refused reminder %s: %s", r["lot_id"], resp.text[:200])
                continue
            conn.execute("update shortlist_alerts set reminded_at = now() where lot_id = %s",
                         (r["lot_id"],))
            conn.commit()
            sent += 1
    if sent:
        log.info("sent %d reminders (%d with a live bid)", sent,
                 sum(v is not None for v in live.values()))


def countries() -> None:
    with db.connect() as conn:
        rows = conn.execute(
            "select seller_country, count(*) as lots from lots "
            "group by 1 order by 2 desc"
        ).fetchall()
    for r in rows:
        print(f"  {str(r['seller_country']):<24}{r['lots']:>6}   {eu_status(r['seller_country'])}")


def run(send: bool = False) -> None:
    out, stats = score()
    print_table(out, stats)
    if send:
        send_new(out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if "--countries" in sys.argv:
        countries()
    elif "--remind" in sys.argv:
        remind()
    else:
        run(send="--send" in sys.argv)

"""Ask the bot about one lot — the check you would otherwise do by hand.

Send the bot a Catawiki link or a lot number in Telegram (replies.py), or run

    python ask.py 108328727
    python ask.py https://www.catawiki.com/en/l/108328727-omega-constellation-…

and it answers with the same numbers an alert carries, for any lot, whether or
not it is one of your focus models or close to its end:

  the verdict        under your safe max, only under the fair max, or past both
  both max bids      and what each costs you all-in
  the value          with its confidence grade and what it is based on
  past sales         what this reference fetched on Catawiki
  lot memory         whether the same listing ran before, and how it ended
  the cash rule      whether the all-in cost fits half of your free cash
  what to check      what the listing says, or leaves out

The valuation is the shortlist's own (shortlist.value_for, max_bid, fair_bid),
so an answer and an alert for the same lot agree. The current bid is read from
the lot page when the browser is free, otherwise the last stored bid is shown
with its age. A lot that is not in the database yet is fetched once.

A lot you asked about gets the same "ended unsold under your max" follow-up as
an alerted one (reserve.py).
"""

from __future__ import annotations

import logging
import sys
from html import escape

import numpy as np
import pandas as pd

import asks
import db
import explain
import model
import replies
import reserve
import shortlist as sl
from config import cfg

log = logging.getLogger("ask")

LIVE_WAIT_SECONDS = 25    # how long to wait for the browser before using the stored bid
ASKS_MAX_AGE = 45 * 60    # reuse the eBay asks the shortlist read up to this long ago
LAST_SALES = 3            # most recent comparable Catawiki sales listed in the answer


def _lot_url(lot_id: str) -> str:
    return f"https://www.catawiki.com/en/l/{lot_id}"


def _past_sales(closed: pd.DataFrame, lot, key: str | None) -> pd.DataFrame:
    """Comparable Catawiki sales of the same reference (same material and line)."""
    if not key:
        return closed.iloc[0:0]
    same = closed[closed["reference_number"].map(model.ref_key) == key]
    if same.empty:
        return same
    material, line = model.lot_material(lot), asks.lot_line(lot)

    def comparable(s) -> bool:
        m = model.lot_material(s)
        if material in ("steel", "plated", "gold", "bicolor") and m not in (material, "unknown", "other"):
            return False
        return asks.line_mismatch(f"{s['watch_model'] or ''} {s['title'] or ''}", line) is None

    return same[same.apply(comparable, axis=1)].sort_values("close_time")


def evaluate(lot_id: str, url: str | None = None, live: bool = True) -> dict:
    """Everything the answer needs, as a dict. {"status": "unknown"} when the lot
    is not stored and could not be fetched."""
    lot_id = str(lot_id)
    lot = explain.load_lot(lot_id)
    if lot.empty:
        import poller
        got = poller.fetch_one(url or _lot_url(lot_id), LIVE_WAIT_SECONDS) if live else None
        lot = explain.load_lot(got or lot_id)
        if lot.empty:
            return {"status": "unknown", "lot_id": lot_id}
        live = False                      # the fetch just stored a fresh bid
    r = lot.iloc[0]
    now = pd.Timestamp.now(tz="UTC")
    closed_out = pd.notna(r.get("sold")) or (pd.notna(r["close_time"]) and r["close_time"] <= now)

    current = float(r["current_bid"]) if pd.notna(r["current_bid"]) else None
    seen = pd.Timestamp(r["last_seen"]) if pd.notna(r.get("last_seen")) else None
    if seen is not None and seen.tzinfo is None:
        seen = seen.tz_localize("UTC")
    reserve_met, close_time, fresh = r.get("reserve_met"), r["close_time"], False
    if live and not closed_out:
        res = sl._fetch_live([{"lot_id": lot_id, "url": r["url"], "close_time": r["close_time"]}],
                             wait=LIVE_WAIT_SECONDS)
        if lot_id in res and res[lot_id] is None:
            closed_out = True             # the page is gone: withdrawn, recorded as unsold
        elif res.get(lot_id):
            current, seen, close_time, rm = res[lot_id]
            reserve_met = rm if rm is not None else reserve_met
            fresh = True

    country = sl.eu_status(r["seller_country"])
    non_eu = country == "non_eu"
    key = model.ref_key(r["reference_number"])

    with sl.PIPELINE_LOCK:
        closed = model.load()
        closed = closed[closed["lot_id"] != lot_id]
        by_ref, by_line = sl.load_asks({}, max_age=ASKS_MAX_AGE)
        cw = sl.catawiki_by_ref(closed)
        sl.calibrate_ratio(closed, by_ref)
        sl.calibrate_margin(closed, by_ref, cw)
        v = sl.value_for(r, by_ref, by_line, cw)
        if v is None and sl.USE_FEATURE_TIER:
            live_model, _, _, _, _, fair_s = sl.build_live_model(
                closed, lot.drop(columns=["final_price", "sold"]))
            sl.calibrate_feature_bias(live_model, by_ref)
            v = sl.value_for(r, by_ref, by_line, cw, float(fair_s.iloc[0]))
        past = _past_sales(closed, r, key)

        out = {"status": "ok", "lot_id": lot_id, "lot": r, "closed": bool(closed_out),
               "current": current, "seen": seen, "fresh": fresh, "close_time": close_time,
               "reserve_met": reserve_met, "country": country, "non_eu": non_eu,
               "passes": bool(sl.passes_filters(r)), "risks": sl.risk_flags(r),
               "serviced": asks.lot_serviced(r), "past": past, "value": None}
        if v is not None:
            value, realism, basis, kind = v
            margin = sl.margin_for(kind, getattr(v, "sigma", None))
            safe_max, net_safe = sl.max_bid(value, realism, non_eu, margin)
            fair = sl.fair_bid(v, value, realism, country)
            out.update(value=value, realism=realism, basis=basis, kind=kind, grade=v.grade,
                       margin=margin, safe_max=safe_max, net_safe=net_safe,
                       fair_max=fair[0] if fair and fair[0] > safe_max else None,
                       net_fair=fair[1] if fair else None,
                       competitors=v.competitors or [])

    with db.connect() as conn:
        import trade
        out["memory"] = sl._extra(conn, lambda: reserve.memory(conn, lot_id))
        out["cash"] = sl._extra(conn, lambda: trade.cash(conn))
        if out["value"] is not None and not out["closed"]:
            top_is_fair = out["fair_max"] is not None and (current or 0) >= out["safe_max"]
            sl._extra(conn, lambda: reserve.remember_ask(
                conn, lot_id,
                round(out["fair_max"] if top_is_fair else out["safe_max"], 2),
                None if top_is_fair or out["fair_max"] is None else round(out["fair_max"], 2),
                "fair" if top_is_fair else "safe"))
    return out


def _name(r) -> str:
    brand = r["brand"] if isinstance(r["brand"], str) else "Lot"
    line = sl.line_of(r["watch_model"])
    model_name = line if line != "other" else (r["watch_model"] if isinstance(r["watch_model"], str) else "")
    ref = r["reference_number"] if isinstance(r["reference_number"], str) else ""
    return " ".join(f"{brand} {model_name} {ref}".split())[:80]


def _when(close_time, now) -> str:
    local = pd.Timestamp(close_time).tz_convert(cfg.timezone)
    mins = int((pd.Timestamp(close_time) - now).total_seconds() // 60)
    day = "today" if local.date() == now.tz_convert(cfg.timezone).date() else local.strftime("%a %-d %b")
    if mins < 0:
        return f"Closed {day} at {local:%H:%M}"
    left = f"{mins // 60} h {mins % 60:02d} min" if mins >= 60 else f"{mins} min"
    return f"Closes {day} at {local:%H:%M} · in {left}"


def text(e: dict) -> str:
    """The Telegram answer, in HTML."""
    if e["status"] != "ok":
        return (f"I do not have lot {escape(e['lot_id'])} and could not read its page just now "
                f"(the browser may be busy collecting). Send the full Catawiki link, or try "
                f"again in a few minutes.")
    r, eur, now = e["lot"], sl._eur, pd.Timestamp.now(tz="UTC")
    non_eu = e["non_eu"]
    cond = str(r["watch_condition"]).split(" - ")[0] if isinstance(r["watch_condition"], str) else ""
    where = {"eu": "EU seller", "non_eu": "seller outside the EU", "unknown": "seller country unknown"}[e["country"]]
    country = r["seller_country"].upper() if isinstance(r["seller_country"], str) else ""
    out = [f"🔎 <b>{escape(_name(r))}</b>",
           escape(" · ".join(x for x in (sl.watch_year_text(r), cond, f"{where} {country}".strip()) if x)),
           _when(e["close_time"], now)]

    # ---- where the bidding stands
    cur = e["current"]
    if e["closed"]:
        if pd.notna(r.get("final_price")):
            out.append(f"Sold for {eur(float(r['final_price']))}")
        elif pd.notna(r.get("sold")):
            out.append("Ended without a sale" + (f" · last bid seen {eur(cur)}" if cur else ""))
        else:
            out.append("Closed — result not recorded yet" + (f" · last bid seen {eur(cur)}" if cur else ""))
    else:
        if cur is None:
            bid = "No bid seen yet"
        elif e["fresh"]:
            bid = f"Bid {eur(cur)} right now ({e['seen'].tz_convert(cfg.timezone):%H:%M})"
        elif e["seen"] is not None:
            age = int((now - e["seen"]).total_seconds() // 60)
            bid = (f"Bid {eur(cur)} at {e['seen'].tz_convert(cfg.timezone):%H:%M}"
                   + (f" ({age} min ago — check the page)" if age > sl.STALE_MINUTES else ""))
        else:
            bid = f"Bid {eur(cur)}"
        if sl.reserve_unmet(e["reserve_met"]):
            bid += " · reserve not met"
        out.append(bid)
    mem = sl.memory_text(e.get("memory"))
    if mem:
        out.append(escape(mem))
    out.append("")

    if e["value"] is None:
        out.append("<b>No max bid:</b> there is no price evidence for this watch (no reference "
                   "match and no similar lots). Check the links yourself before bidding.")
        return "\n".join(out + _checks(e) + _links(r))

    safe, fair = e["safe_max"], e["fair_max"]
    top = fair if fair is not None else safe
    bid_now = cur or 0.0
    act = None                                   # the max to act on, if any
    if e["closed"]:
        out.append(f"<b>It would have been worth up to {eur(top)}</b>")
    elif bid_now < safe:
        act = safe
        out.append(f"✅ <b>Bid up to {eur(safe)}</b> — and not a euro more")
        if fair is not None:
            out.append(f"Fair max {eur(fair)}: up to there it still makes {sl.FAIR_MARGIN:.0%} "
                       f"if you sell it yourself with no platform fee")
    elif fair is not None and bid_now < fair:
        act = fair
        out.append(f"⚖️ <b>Fair deal only: bid up to {eur(fair)}</b>")
        out.append(f"The safe max of {eur(safe)} is already passed. This pays only if you sell "
                   f"it yourself with no platform fee")
    else:
        out.append(f"✋ <b>Skip it</b> — the bid is already at or over your max of {eur(top)}")
        if fair is None and e["country"] == "eu" and e["kind"] in ("ebay", "catawiki"):
            out.append(f"No fair max here: fewer than {sl.FAIR_MIN_SALES} comparable Catawiki sales")
    if act is not None and sl.reserve_unmet(e["reserve_met"]):
        out.append("Reserve not met yet. Bid your max anyway: if it ends unsold with you as top "
                   "bidder, the seller can offer it to you — accept only up to your max")
    if not e["passes"]:
        out.append("⚠️ Condition below Good, or a repainted dial: the shortlist would skip this lot")

    # ---- what each bid costs
    vat = ", import VAT and customs handling" if non_eu else ""
    steps = [("safe max", safe)] + ([("fair max", fair)] if fair is not None else [])
    if cur and not e["closed"]:
        steps = [("the bid now", cur)] + steps
    out.append("All-in (Catawiki fee, shipping" + vat + "): "
               + " · ".join(f"{eur(sl.landed(b, non_eu))} at {n} {eur(b)}" for n, b in steps))
    cash = sl.cash_text(e.get("cash"), sl.landed(act if act is not None else top, non_eu))
    if cash:
        out.append(cash)
    out.append("")

    # ---- what it is worth
    sale = e["value"] * e["realism"]
    out.append(f"<b>Worth:</b> quick-sale price about {eur(sale)} (resale value {eur(e['value'])})")
    out.append(f"Confidence {e['grade']}: {escape(sl.GRADE_TEXT.get(e['grade'], ''))}"
               if e.get("grade") else f"Evidence: {sl.EVIDENCE.get(e['kind'], e['kind'])}")
    out.append(f"From {escape(str(e['basis']))}")
    past = e["past"]
    if len(past):
        p = past["final_price"].astype(float)
        last = " · ".join(f"{eur(x)} ({t.tz_convert(cfg.timezone):%-d %b})"
                          for x, t in zip(p.tail(LAST_SALES), past["close_time"].tail(LAST_SALES)))
        spread = (f", middle half {eur(float(np.quantile(p, 0.25)))}–{eur(float(np.quantile(p, 0.75)))}"
                  if len(p) >= 4 else "")
        out.append(f"Catawiki sales of this reference: {len(p)}, typical {eur(float(p.median()))}"
                   f"{spread}. Latest: {last}")
    else:
        out.append("No comparable Catawiki sale of this reference is stored")
    comps = e.get("competitors") or []
    if comps:
        out.append("Cheapest comparable for sale: " + ", ".join(sl._competitor(c) for c in comps))
    at = act if act is not None else top
    cost = sl.landed(at, non_eu)
    out.append(f"Profit if won at {eur(at)} and sold for {eur(sale)}: "
               + " · ".join(f"{c} {eur(pr)}" for c, pr in sl.channel_profits(sale, cost)))
    return "\n".join(out + _checks(e) + _links(r))


def _checks(e: dict) -> list[str]:
    warn = list(e.get("risks") or [])
    warn.append("Listing mentions a service — ask the seller for the receipt or date" if e.get("serviced")
                else "No service mentioned — a service would cost €150–250 on top")
    if e.get("grade") in ("C", "D") or e.get("value") is None:
        warn.append("Thin evidence — look this reference up on Chrono24 and Willhaben before you bid")
    if e["non_eu"]:
        warn.append("Seller outside the EU — import VAT is already in the all-in figures; "
                    "customs can take 2–3 weeks")
    if e["country"] == "unknown":
        warn.append("Seller country unknown — VAT may apply on top")
    warn.append("Look at the dial, hands and case in every photo — damage is often not in the description")
    return ["", "<b>Before you bid</b>"] + [f"• {escape(w)}" for w in warn]


def _links(r) -> list[str]:
    links = [f'<a href="{escape(r["url"], quote=True)}">Open on Catawiki</a>']
    links += [f'<a href="{escape(u, quote=True)}">{n}</a>' for n, u in sl.search_links(r["reference_number"])]
    return ["", " · ".join(links),
            "Saw prices? Reply to this message: <i>sold 540 575</i> · <i>c24 450</i> · <i>wh 480</i>"]


def reply(lot_id: str, url: str | None = None, reply_to: int | None = None) -> None:
    """Answer in Telegram (called by replies.py)."""
    replies.send_html(f"⏳ Checking lot {escape(str(lot_id))} …", reply_to=reply_to)
    try:
        answer = text(evaluate(lot_id, url))
    except Exception:
        log.exception("could not evaluate lot %s", lot_id)
        answer = (f"Something went wrong while checking lot {escape(str(lot_id))} — the error is in "
                  f"logs/err.log. Try:  python ask.py {escape(str(lot_id))}")
    replies.send_html(answer, lot_id, reply_to)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    import re
    arg = args[0]
    lot = explain.lot_id_from(arg)
    html = text(evaluate(lot, arg if arg.startswith("http") else None, live="--stored" not in sys.argv))
    print(re.sub(r"<[^>]+>", "", html).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">"))

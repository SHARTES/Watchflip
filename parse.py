"""Parsing.

READ THIS BEFORE YOU TRUST ANYTHING BELOW.

This is the one module I cannot write blind and have it be correct. Catawiki
renders from an embedded JSON blob whose exact key names change over time, and
any CSS selector I guess today is a selector that breaks next month. So the
approach here is deliberately defensive:

  1. Find every embedded JSON blob on the page (__NEXT_DATA__, JSON-LD, any
     application/json script tag).
  2. Walk them recursively looking for objects that *smell* like a lot — an id
     plus a title plus something price-shaped.
  3. Map fields by trying a list of candidate key names.
  4. Fall back to visible-text regex only for the numbers.

Run `python probe.py <lot-url>` once. It dumps the real key names it found.
Add them to the CANDIDATES lists below and this module becomes exact instead of
merely resilient. Budget twenty minutes for that; it is the highest-value
twenty minutes in the whole project.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterator

from bs4 import BeautifulSoup
from dateutil import parser as dateparser

# Candidate key names, tried in order. Extend from probe.py output.
CANDIDATES: dict[str, tuple[str, ...]] = {
    "lot_id": ("lotId", "id", "objectId", "lot_id"),
    "title": ("title", "name", "lotTitle", "headline"),
    "description": ("description", "subtitle", "specifications", "summary"),
    "current_bid": ("currentBidAmount", "currentBid", "highestBid", "bidAmount",
                    "amount", "price", "currentPrice"),
    "bid_count": ("bidCount", "bidsCount", "numberOfBids", "totalBids"),
    "close_time": ("endTime", "closingTime", "expiresAt", "endsAt", "closeTime"),
    "estimate_low": ("estimateLow", "minEstimate", "lowEstimate", "estimateMin"),
    "estimate_high": ("estimateHigh", "maxEstimate", "highEstimate", "estimateMax"),
    "seller_country": ("sellerCountry", "country", "countryCode", "shipsFrom"),
    "seller_name": ("sellerName", "seller", "sellerUsername"),
    "reserve_met": ("reserveMet", "isReserveMet", "reservePriceMet"),
    "sold": ("isSold", "sold", "hasSold"),
    "auction_id": ("auctionId", "auction_id", "auctionNumber"),
}

LOT_URL_RE = re.compile(r"/l/(\d+)-[a-z0-9\-]+", re.I)
MONEY_RE = re.compile(r"(?:€|EUR)\s*([\d.,]+)", re.I)


# ---------------------------------------------------------------- JSON finding

def embedded_json(html: str) -> Iterator[Any]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("script"):
        t = (tag.get("type") or "").lower()
        if tag.get("id") == "__NEXT_DATA__" or "json" in t:
            text = tag.string or tag.get_text() or ""
            text = text.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except (ValueError, TypeError):
                continue


def catawiki_page_props(html: str) -> dict[str, Any] | None:
    """Return the server-rendered Catawiki page data, when it is present.

    Catawiki's lot detail data is split into a few sibling objects inside
    Next.js ``pageProps``.  Treating each object as an independent "lot" loses
    the link between the title, auction end time and current bid, which is why
    the generic JSON walker below is only a fallback for this site.
    """
    for blob in embedded_json(html):
        if not isinstance(blob, dict):
            continue
        props = blob.get("props")
        page_props = props.get("pageProps") if isinstance(props, dict) else None
        if isinstance(page_props, dict) and isinstance(
            page_props.get("lotDetailsData"), dict
        ):
            return page_props
    return None


def walk(node: Any) -> Iterator[dict]:
    """Yield every dict anywhere in a nested JSON structure."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def pick(d: dict, field: str) -> Any:
    for key in CANDIDATES[field]:
        if key in d and d[key] not in (None, "", []):
            return d[key]
    return None


def looks_like_lot(d: dict) -> bool:
    has_id = pick(d, "lot_id") is not None
    has_title = isinstance(pick(d, "title"), str)
    has_price = pick(d, "current_bid") is not None or pick(d, "close_time") is not None
    return has_id and has_title and has_price


# ------------------------------------------------------------------ coercion

# Does the embedded JSON store money in cents rather than euros? Marketplaces
# are split roughly 50/50 on this and guessing is how you end up with a
# database that is silently 100x wrong. probe.py tells you which it is; set
# this once and never think about it again.
MINOR_UNITS = False

# Keys whose value is unambiguously in minor units regardless of the flag.
_CENT_KEYS = ("cents", "minorunits", "amountincents", "amountminor")


def _parse_number(raw: str) -> float | None:
    """Parse a number written in either European or Anglo convention."""
    raw = raw.strip()
    if "," in raw and "." in raw:
        # Whichever separator comes last is the decimal point.
        if raw.rindex(",") > raw.rindex("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        tail = raw.split(",")[-1]
        raw = raw.replace(",", ".") if len(tail) <= 2 else raw.replace(",", "")
    elif "." in raw:
        tail = raw.split(".")[-1]
        # "1.200" in a European price is twelve hundred, not one point two.
        # A group of exactly three digits after a dot is a thousands separator.
        if len(tail) == 3 and raw.count(".") >= 1 and not raw.startswith("0."):
            raw = raw.replace(".", "")
    try:
        return float(raw)
    except ValueError:
        return None


def as_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) / 100 if MINOR_UNITS else float(v)
    if isinstance(v, dict):
        # Catawiki localises monetary amounts as {"EUR": 440, ...}.
        if isinstance(v.get("EUR"), (int, float)):
            return float(v["EUR"])
        for k in v:
            if k.lower() in _CENT_KEYS and isinstance(v[k], (int, float)):
                return float(v[k]) / 100
        for k in ("amount", "value", "price"):
            if k in v:
                return as_float(v[k])
        return None
    if isinstance(v, str):
        m = MONEY_RE.search(v) or re.search(r"([\d.,]+)", v)
        return _parse_number(m.group(1) if m.lastindex else m.group(0)) if m else None
    return None


def as_int(v: Any) -> int | None:
    f = as_float(v)
    return int(f) if f is not None else None


def as_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        ts = float(v)
        if ts > 1e11:          # milliseconds
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(v, str):
        try:
            dt = dateparser.parse(v)
        except (ValueError, OverflowError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    return None


# ------------------------------------------------------------------- parsers

def parse_listing_page(html: str, base: str = "https://www.catawiki.com") -> list[str]:
    """Return absolute lot URLs found on a category or auction page."""
    urls: list[str] = []
    seen: set[str] = set()
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        m = LOT_URL_RE.search(a["href"])
        if not m:
            continue
        lot_id = m.group(1)
        if lot_id in seen:
            continue
        seen.add(lot_id)
        href = a["href"]
        urls.append(href if href.startswith("http") else base + href)
    return urls


def parse_lot_page(html: str, url: str) -> dict[str, Any] | None:
    """Extract a lot record from a lot detail page."""
    page_props = catawiki_page_props(html)
    if page_props is not None:
        return parse_catawiki_lot_page(page_props, url)

    best: dict | None = None
    for blob in embedded_json(html):
        for node in walk(blob):
            if looks_like_lot(node):
                # Prefer the richest candidate — the page usually contains a
                # slim card object and one fat detail object.
                if best is None or len(node) > len(best):
                    best = node

    url_id = LOT_URL_RE.search(url)
    lot_id = str(pick(best, "lot_id")) if best else None
    if not lot_id and url_id:
        lot_id = url_id.group(1)
    if not lot_id:
        return None

    photos = extract_photos(html)
    soup = BeautifulSoup(html, "lxml")

    record: dict[str, Any] = {
        "lot_id": lot_id,
        "url": url.split("?")[0],
        "auction_id": str(pick(best, "auction_id")) if best and pick(best, "auction_id") else None,
        "category_slug": None,
        "title": (pick(best, "title") if best else None) or text_of(soup, "h1"),
        "description": (pick(best, "description") if best else None) or meta_description(soup),
        "photo_urls": photos,
        "seller_country": pick(best, "seller_country") if best else None,
        "seller_name": stringify(pick(best, "seller_name")) if best else None,
        "estimate_low": as_float(pick(best, "estimate_low")) if best else None,
        "estimate_high": as_float(pick(best, "estimate_high")) if best else None,
        "currency": "EUR",
        "close_time": as_dt(pick(best, "close_time")) if best else None,
        "raw": best or {},
        # live state, stored as a snapshot rather than on the lot row
        "_current_bid": as_float(pick(best, "current_bid")) if best else None,
        "_bid_count": as_int(pick(best, "bid_count")) if best else None,
        "_reserve_met": pick(best, "reserve_met") if best else None,
        "_sold": pick(best, "sold") if best else None,
    }

    if record["_current_bid"] is None:
        record["_current_bid"] = visible_price(soup)

    return record


def parse_catawiki_lot_page(page_props: dict[str, Any], url: str) -> dict[str, Any] | None:
    """Parse Catawiki's current Next.js lot-detail payload.

    This payload was captured by ``probe.py`` on 2026-09-11.  Keep this parser
    explicit: the price and close time are live-auction fields and must not be
    confused with the expert estimate or a buy-now offer elsewhere on the page.
    """
    details = page_props.get("lotDetailsData")
    bidding = page_props.get("biddingBlockResponse")
    tracking = page_props.get("dataLayerBase")
    if not isinstance(details, dict):
        return None
    if not isinstance(bidding, dict):
        bidding = {}
    if not isinstance(tracking, dict):
        tracking = {}

    lot_id = details.get("lotId") or page_props.get("lotId") or tracking.get("lot_id")
    if lot_id is None:
        match = LOT_URL_RE.search(url)
        lot_id = match.group(1) if match else None
    if lot_id is None:
        return None

    estimate = details.get("expertsEstimate")
    estimate = estimate if isinstance(estimate, dict) else {}
    seller = details.get("sellerInfo")
    seller = seller if isinstance(seller, dict) else {}
    address = seller.get("address")
    address = address if isinstance(address, dict) else {}
    country = address.get("country")
    country = country if isinstance(country, dict) else {}
    category = details.get("category")
    category = category if isinstance(category, dict) else {}
    category_url = category.get("url")
    watch = extract_watch_attributes(details.get("specifications"))

    history = bidding.get("biddingHistory")
    history = history if isinstance(history, dict) else {}
    bids = history.get("bids")
    bids = bids if isinstance(bids, list) else []
    bid_count = None
    for bid in bids:
        if isinstance(bid, dict) and bid.get("totalBids") is not None:
            bid_count = as_int(bid["totalBids"])
            break
    if bid_count is None and bids:
        bid_count = len(bids)

    photos: list[str] = []
    for image in details.get("images", []):
        if not isinstance(image, dict):
            continue
        photo = image.get("large") or image.get("medium") or image.get("thumbnail")
        if isinstance(photo, str) and photo not in photos:
            photos.append(photo)

    reserve_met = bidding.get("reservePriceMet")
    sold = bidding.get("sold")
    return {
        "lot_id": str(lot_id),
        "url": url.split("?")[0],
        "auction_id": str(details.get("auctionId") or tracking.get("auction_id"))
        if details.get("auctionId") or tracking.get("auction_id")
        else None,
        "category_slug": category_url.rstrip("/").rsplit("/", 1)[-1]
        if isinstance(category_url, str)
        else None,
        "title": details.get("lotTitle"),
        "description": details.get("description"),
        "photo_urls": photos[:12],
        "seller_country": country.get("shortCode") or country.get("name"),
        "seller_name": seller.get("sellerName") or seller.get("userName"),
        **watch,
        "estimate_low": as_float(estimate.get("min")),
        "estimate_high": as_float(estimate.get("max")),
        "currency": "EUR",
        "close_time": as_dt(bidding.get("biddingEndTime")),
        "raw": {
            "source": "catawiki_next_page_props",
            "lotDetailsData": details,
            "biddingBlockResponse": bidding,
        },
        "_current_bid": as_float(bidding.get("localizedCurrentBidAmount")),
        "_bid_count": bid_count,
        "_reserve_met": reserve_met if isinstance(reserve_met, bool) else None,
        "_sold": sold if isinstance(sold, bool) else None,
    }


def extract_watch_attributes(specifications: Any) -> dict[str, Any]:
    """Turn Catawiki's watch specification list into stable query fields."""
    values: dict[str, str] = {}
    if isinstance(specifications, list):
        for specification in specifications:
            if not isinstance(specification, dict):
                continue
            name = specification.get("name")
            value = specification.get("value")
            if isinstance(name, str) and isinstance(value, str) and value.strip():
                values[name.casefold()] = value.strip()

    year = None
    raw_year = values.get("year")
    if raw_year:
        match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", raw_year)
        if match:
            year = int(match.group(1))

    diameter = None
    raw_diameter = values.get("case diameter")
    if raw_diameter:
        match = re.search(r"(\d+(?:[.,]\d+)?)\s*mm\b", raw_diameter, re.I)
        if match:
            diameter = _parse_number(match.group(1))

    return {
        "brand": values.get("brand"),
        "watch_model": values.get("model"),
        "reference_number": normalise_reference(values.get("reference number")),
        "watch_period": values.get("period"),
        "watch_year": year,
        "watch_condition": values.get("condition"),
        "movement": values.get("movement"),
        "case_diameter_mm": diameter,
    }


def normalise_reference(value: str | None) -> str | None:
    """Normalise spacing and a leading 'Ref.' without guessing a reference."""
    if not value:
        return None
    value = re.sub(r"^\s*(?:ref(?:erence)?\.?\s*)", "", value, flags=re.I)
    value = re.sub(r"\s+", "", value).upper()
    return value or None


def extract_photos(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls: list[str] = []
    for img in soup.find_all("img", src=True):
        src = img["src"]
        if "catawiki" in src and re.search(r"\.(jpe?g|png|webp)", src, re.I):
            clean = src.split("?")[0]
            if clean not in urls:
                urls.append(clean)
    return urls[:12]


def visible_price(soup: BeautifulSoup) -> float | None:
    m = MONEY_RE.search(soup.get_text(" ", strip=True))
    return as_float(m.group(0)) if m else None


def text_of(soup: BeautifulSoup, sel: str) -> str | None:
    el = soup.select_one(sel)
    return el.get_text(" ", strip=True) if el else None


def meta_description(soup: BeautifulSoup) -> str | None:
    el = soup.find("meta", attrs={"name": "description"})
    return el.get("content") if el else None


def stringify(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        for k in ("name", "username", "displayName"):
            if isinstance(v.get(k), str):
                return v[k]
    return str(v)[:200]


NOT_FOUND_MARKERS = (
    "page not found",
    "this lot is no longer available",
    "sorry, we can't find",
)


def looks_gone(html: str) -> bool:
    """True when Catawiki has removed the lot page entirely.

    Searches the rendered text rather than raw HTML: the source contains
    escaped entities and markup between the words, so a plain substring match
    against html misses "Page Not Found" even when it is plainly on screen.
    """
    text = BeautifulSoup(html, "lxml").get_text(" ", strip=True).casefold()
    return any(m in text for m in NOT_FOUND_MARKERS)
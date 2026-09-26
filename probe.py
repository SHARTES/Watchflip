"""Run this ONCE against a real lot URL before trusting the parser.

    python probe.py https://www.catawiki.com/en/l/12345678-some-watch

It dumps the embedded JSON objects that look like a lot, so you can read the
actual key names and paste them into CANDIDATES in parse.py. It also writes the
full page HTML and the raw JSON to ./probe-output/ so you can grep around.
"""

from __future__ import annotations

import json
import pathlib
import sys

from fetcher import fetcher
from parse import (
    catawiki_page_props,
    embedded_json,
    looks_like_lot,
    parse_lot_page,
    walk,
)

OUT = pathlib.Path("probe-output")


def main(url: str) -> None:
    OUT.mkdir(exist_ok=True)

    with fetcher() as f:
        html = f.get_html(url)

    if not html:
        print("Could not fetch the page. If it hung, try HEADLESS=0 and watch "
              "what the browser is being shown — it may be a consent wall.")
        return

    (OUT / "page.html").write_text(html, encoding="utf-8")
    print(f"wrote {OUT / 'page.html'}  ({len(html):,} bytes)")

    blobs = list(embedded_json(html))
    print(f"found {len(blobs)} embedded JSON blob(s)")
    for i, blob in enumerate(blobs):
        path = OUT / f"blob-{i}.json"
        path.write_text(json.dumps(blob, indent=2, default=str), encoding="utf-8")
        print(f"  wrote {path}")

    candidates = []
    for blob in blobs:
        for node in walk(blob):
            if looks_like_lot(node):
                candidates.append(node)

    page_props = catawiki_page_props(html)
    if page_props:
        details = page_props.get("lotDetailsData", {})
        bidding = page_props.get("biddingBlockResponse", {})
        history = bidding.get("biddingHistory", {}) if isinstance(bidding, dict) else {}
        bids = history.get("bids", []) if isinstance(history, dict) else []
        total_bids = None
        if bids and isinstance(bids[0], dict):
            total_bids = bids[0].get("totalBids")
        print("\n--- Catawiki Next.js lot payload ---\n")
        print(f"  lot id                      {page_props.get('lotId')}")
        print(f"  title                       {details.get('lotTitle')!r}")
        print(f"  current bid (EUR)           {bidding.get('localizedCurrentBidAmount')}")
        print(f"  total bids                  {total_bids}")
        print(f"  bidding end (epoch ms)      {bidding.get('biddingEndTime')}")
        print(f"  reserve met                 {bidding.get('reservePriceMet')}")
        print(f"  sold                        {bidding.get('sold')}")

    print(f"\n{len(candidates)} object(s) look like a lot.")
    if candidates:
        richest = max(candidates, key=len)
        print("\nRichest candidate's keys:\n")
        for k, v in sorted(richest.items()):
            preview = json.dumps(v, default=str)
            if len(preview) > 90:
                preview = preview[:87] + "..."
            print(f"  {k:<28} {preview}")

    print("\n--- what parse_lot_page() currently extracts ---\n")
    rec = parse_lot_page(html, url)
    if rec is None:
        print("NOTHING. The parser found no lot. Open blob-*.json and map the "
              "keys manually into CANDIDATES.")
        return
    for k, v in rec.items():
        if k == "raw":
            continue
        print(f"  {k:<18} {v!r}"[:160])

    print("\n--- the cents check (do not skip this) ---\n")
    bid = rec.get("_current_bid")
    if bid is None:
        print("  No current bid extracted. Find it in blob-*.json and add its "
              "key to CANDIDATES['current_bid'].")
    else:
        print(f"  Parser says the current bid is EUR {bid:,.2f}")
        print(f"  If the page actually shows EUR {bid / 100:,.2f}, the JSON is "
              "in cents:\n  set MINOR_UNITS = True in parse.py and re-run this.")

    print("\nCheck every line above against what the page actually shows. "
          "The fields that matter most are close_time and current_bid — if "
          "either is None or wrong, fix it before you let this run for weeks.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])

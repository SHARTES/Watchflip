"""eBay API client.

Two search backends, in order of usefulness:

  Marketplace Insights  — actual sold prices. This is the one that matters.
                          Requires a separate access request to eBay.
  Browse                — active listings only. Asking prices, not sale
                          prices, and on a marketplace where vintage watches
                          sit unsold for months those are wishes rather than
                          evidence. Useful for shaking out the matching logic
                          while the Insights request is pending; not useful
                          for training a value model.

The client tries Insights first and falls back to Browse, reporting which one
it used so nothing downstream silently treats asking prices as sales.

No seller fields are read or returned anywhere in this module. See the note in
schema_comps.sql — that is a commitment, not an oversight.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("ebay")

OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
INSIGHTS_URL = "https://api.ebay.com/buy/marketplace_insights/v1_beta/item_sales/search"
BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"

SCOPE = "https://api.ebay.com/oauth/api_scope"

CLIENT_ID = os.environ.get("EBAY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("EBAY_CLIENT_SECRET", "")
MARKETPLACE = os.environ.get("EBAY_MARKETPLACE", "EBAY_DE")

# Watches category. Narrows results and keeps unrelated junk out of the comps.
CATEGORY_WRISTWATCHES = "31387"


class EbayError(RuntimeError):
    pass


class EbayClient:
    def __init__(self):
        if not CLIENT_ID or not CLIENT_SECRET:
            raise EbayError("EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set")
        self._token: str | None = None
        self._token_expires_at = 0.0
        self._session = requests.Session()
        self.insights_available: bool | None = None

    # ------------------------------------------------------------------ auth

    def _fetch_token(self) -> None:
        basic = base64.b64encode(
            f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
        ).decode()

        r = self._session.post(
            OAUTH_URL,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials", "scope": SCOPE},
            timeout=30,
        )
        if r.status_code != 200:
            raise EbayError(f"token request failed ({r.status_code}): {r.text[:300]}")

        payload = r.json()
        self._token = payload["access_token"]
        # Refresh a minute early rather than discovering expiry mid-pass.
        self._token_expires_at = time.time() + payload.get("expires_in", 7200) - 60
        log.info("got eBay token, valid for %ss", payload.get("expires_in"))

    def _headers(self) -> dict[str, str]:
        if not self._token or time.time() >= self._token_expires_at:
            self._fetch_token()
        return {
            "Authorization": f"Bearer {self._token}",
            "X-EBAY-C-MARKETPLACE-ID": MARKETPLACE,
            "Accept": "application/json",
        }

    # ---------------------------------------------------------------- search

    def search_sold(self, query: str, limit: int = 50, days: int = 90
                    ) -> tuple[list[dict], str]:
        """Return (items, source). Tries real sales, falls back to listings."""
        if self.insights_available is not False:
            try:
                items = self._insights(query, limit, days)
                self.insights_available = True
                return items, "ebay_insights"
            except EbayError as exc:
                if "403" in str(exc) or "Insufficient permissions" in str(exc):
                    if self.insights_available is None:
                        log.warning(
                            "Marketplace Insights is not granted on this keyset — "
                            "falling back to active listings. These are asking "
                            "prices, not sales; do not train a value model on them."
                        )
                    self.insights_available = False
                else:
                    raise

        return self._browse(query, limit), "ebay_browse"

    def _insights(self, query: str, limit: int, days: int) -> list[dict]:
        # Insights wants an explicit window; last N days of completed sales.
        start = time.strftime(
            "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - days * 86400)
        )
        params = {
            "q": query,
            "limit": str(min(limit, 200)),
            "category_ids": CATEGORY_WRISTWATCHES,
            "filter": f"lastSoldDate:[{start}..]",
        }
        r = self._session.get(INSIGHTS_URL, headers=self._headers(),
                              params=params, timeout=45)
        if r.status_code != 200:
            raise EbayError(f"insights {r.status_code}: {r.text[:300]}")
        return r.json().get("itemSales", [])

    def _browse(self, query: str, limit: int) -> list[dict]:
        params = {
            "q": query,
            "limit": str(min(limit, 200)),
            "category_ids": CATEGORY_WRISTWATCHES,
        }
        r = self._session.get(BROWSE_URL, headers=self._headers(),
                              params=params, timeout=45)
        if r.status_code != 200:
            raise EbayError(f"browse {r.status_code}: {r.text[:300]}")
        return r.json().get("itemSummaries", [])


# ------------------------------------------------------------- normalisation

def _money(node: Any) -> tuple[float | None, str | None]:
    if not isinstance(node, dict):
        return None, None
    try:
        return float(node.get("value")), node.get("currency")
    except (TypeError, ValueError):
        return None, node.get("currency")


def to_comp(item: dict, source: str, brand: str, reference: str) -> dict | None:
    """Flatten an eBay item into a comps row.

    Deliberately drops everything about the seller. Only the fields listed in
    the exemption statement are read.
    """
    comp_id = item.get("itemId")
    if not comp_id:
        return None

    is_sold = source == "ebay_insights"
    price_node = item.get("lastSoldPrice") if is_sold else item.get("price")
    price, currency = _money(price_node)

    return {
        "comp_id": str(comp_id),
        "source": source,
        "marketplace": MARKETPLACE,
        "title": item.get("title"),
        "reference_number": reference,
        "item_condition": item.get("condition"),
        "price": price,
        "currency": currency,
        "is_sold": is_sold,
        "sold_at": item.get("lastSoldDate") if is_sold else None,
        "query_brand": brand,
        "query_reference": reference,
    }

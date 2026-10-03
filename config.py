"""Configuration. Everything tunable lives here."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    database_url: str = os.environ.get("DATABASE_URL", "")

    # Omega first: discover walks sources in order and truncates at
    # max_lots_per_run, so the target vertical always gets budget. Then the
    # candidate brands (below), then the broad catalogue, which takes whatever
    # is left and feeds the hammer model.
    watch_urls: tuple = (
        ("https://www.catawiki.com/en/x/1331", 40),   # Omega
        ("https://www.catawiki.com/en/x/1383", 10),   # Seiko
        ("https://www.catawiki.com/en/x/1337", 6),    # Longines
        ("https://www.catawiki.com/en/x/1453", 4),    # Universal Genève
        ("https://www.catawiki.com/en/x/1341", 4),    # Tissot
        ("https://www.catawiki.com/en/c/333-wristwatches", 4),
    )

    # Used for tagging and, later, for alert filtering.  Not a collection filter.
    target_brand: str = "Omega"
    target_models: tuple = (
        "seamaster", "genève", "geneve", "de ville",
        "constellation", "speedmaster", "dynamic",
    )
    # Brands collected and watched next to Omega, to decide after about three
    # weeks of data (python brands.py) which ones to trade. No alerts yet.
    # Their vintage lots get monitor time right after Omega's, and their
    # references get eBay comparables. For Seiko only the better lines count:
    # King Seiko, Grand Seiko, Lord Marvel, Lord Matic and the classic
    # chronograph and diver references; a Seiko 5 does not pay for the work.
    candidate_brands: tuple = ("Seiko", "Longines", "Universal Genève", "Tissot")
    # Works in Python (re, case-insensitive) and in PostgreSQL (~*) alike.
    premium_seiko_pattern: str = (
        r"king ?seiko|grand ?seiko|lord ?marvel|lord ?matic"
        r"|(^|[^0-9])(6139|6138|6105|6306|6309|5626|5625|5645|5646|5606|4402|4420"
        r"|4502|4520|4522|5722|5740|5245|5246|6145|6146|6185|6186)([^0-9]|$)"
    )
    # Seiko dress watches in the Cartier Tank spirit: Dolce, Chariot, Lassale,
    # Credor, Exceline, and anything rectangular, square or tonneau. Often
    # quartz, often 24–30 mm and listed as women's or unisex, often from the
    # 1980s–90s, so these count for any gender and up to 1999. Cheap at
    # auction; what they resell for styled is what the data has to show.
    style_seiko_pattern: str = (
        r"dolce|chariot|lassale|credor|exceline|tank|rectang|rechteck|square"
        r"|carr[ée]|tonneau|curved"
    )
    vintage_year_min: int = 1950
    vintage_year_max: int = 1989
    training_max_price: float = 3000.0

    # Purchase guardrail, used by the future decision/alert layer.  This is
    # the all-in cap, not the bid shown on Catawiki.
    max_all_in_cost: float = 700.0
    expected_inbound_shipping: float = 30.0
    catawiki_buyer_fee_rate: float = 0.09
    catawiki_buyer_fixed_fee: float = 3.0

    @property
    def candidate_keys(self) -> list:
        """Candidate brands lower-cased without accents, as SQL compares them."""
        import unicodedata
        return ["".join(c for c in unicodedata.normalize("NFKD", b) if not unicodedata.combining(c)).lower()
                for b in self.candidate_brands]

    @property
    def max_entry_bid(self) -> float:
        """Highest bid that fits the all-in budget using the shipping reserve."""
        return round(
            (self.max_all_in_cost - self.expected_inbound_shipping
             - self.catawiki_buyer_fixed_fee) / (1 + self.catawiki_buyer_fee_rate),
            2,
        )

    # Politeness. These are deliberately slow. Do not lower them.
    page_delay_seconds: float = float(os.environ.get("PAGE_DELAY", "6.0"))
    delay_jitter_seconds: float = 3.0
    max_listing_pages: int = int(os.environ.get("MAX_LISTING_PAGES", "12"))
    max_lots_per_run: int = int(os.environ.get("MAX_LOTS_PER_RUN", "150"))
    monitor_batch_size: int = int(os.environ.get("MONITOR_BATCH", "150"))

    # Browser
    headless: bool = os.environ.get("HEADLESS", "1") == "1"
    user_data_dir: str = os.environ.get("USER_DATA_DIR", "./.browser-profile")
    nav_timeout_ms: int = 30_000

    # Sweeper: how long after close_time before we look for the final price,
    # and how long we keep retrying before giving up on a lot.
    sweep_delay_minutes: int = 20
    sweep_giveup_hours: int = 120

    timezone: str = "Europe/Vienna"


cfg = Config()

if not cfg.database_url:
    raise SystemExit(
        "DATABASE_URL is not set. Copy .env.example to .env and fill it in."
    )

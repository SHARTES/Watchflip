"""Configuration. Everything tunable lives here."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    database_url: str = os.environ.get("DATABASE_URL", "")

        # Omega first: discover walks sources in order and truncates at
    # max_lots_per_run, so the target vertical always gets budget. The broad
    # catalogue takes whatever is left and feeds the hammer model.
    watch_urls: tuple = (
        ("https://www.catawiki.com/en/x/1331", 40),
        ("https://www.catawiki.com/en/c/333-wristwatches", 4),
    )

    # Used for tagging and, later, for alert filtering.  Not a collection filter.
    target_brand: str = "Omega"
    target_models: tuple = (
        "seamaster", "genève", "geneve", "de ville",
        "constellation", "speedmaster", "dynamic",
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

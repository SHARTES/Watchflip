"""Page fetching.

One browser, one profile, one page at a time, with a sleep between every
navigation. Slow and boring on purpose: a single well-behaved session that
runs for months is worth far more than a fast one that gets blocked in a week.

Denials are tracked, not just logged. A blocked session that keeps requesting
is the fastest way to turn a temporary block into a permanent one, so callers
check `should_stop` and abandon the pass.
"""

from __future__ import annotations

import errno
import fcntl
import logging
import pathlib
import random
import time
from contextlib import contextmanager

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from config import cfg

log = logging.getLogger(__name__)

ACCESS_DENIED_MARKERS = (
    "access denied",
    "you don't have permission to access",
    "errors.edgesuite.net",
)

# Consecutive denials before a pass is abandoned. Three is enough to rule out
# a single bad page without hammering a session that has clearly been blocked.
DENIAL_LIMIT = 3

LOCK_PATH = pathlib.Path(cfg.user_data_dir + ".lock")


@contextmanager
def browser_lock(timeout_s: int = 180):
    """Only one job may drive the browser profile at a time."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fh = LOCK_PATH.open("w")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError as exc:
            if exc.errno not in (errno.EAGAIN, errno.EACCES):
                raise
            if time.monotonic() > deadline:
                fh.close()
                raise TimeoutError("browser profile is busy")
            time.sleep(5)
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


class Fetcher:
    def __init__(self, pw):
        self._ctx = pw.chromium.launch_persistent_context(
            user_data_dir=cfg.user_data_dir,
            headless=cfg.headless,
            locale="en-GB",
            timezone_id=cfg.timezone,
            viewport={"width": 1440, "height": 900},
            args=["--disk-cache-size=52428800"],
        )
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        self._page.set_default_navigation_timeout(cfg.nav_timeout_ms)
        self._last_nav = 0.0

        self.consecutive_denials = 0
        self.denials = 0
        self.pages_fetched = 0

    @property
    def should_stop(self) -> bool:
        """True once the session looks blocked. Callers must honour this."""
        return self.consecutive_denials >= DENIAL_LIMIT

    def _wait_turn(self) -> None:
        delay = cfg.page_delay_seconds + random.uniform(0, cfg.delay_jitter_seconds)
        # Back off hard after a denial rather than walking straight back in.
        if self.consecutive_denials:
            delay *= 2 ** self.consecutive_denials
        elapsed = time.monotonic() - self._last_nav
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def get_html(self, url: str) -> str | None:
        """Navigate and return the rendered HTML, or None on failure."""
        if self.should_stop:
            return None

        self._wait_turn()
        try:
            self._page.goto(url, wait_until="domcontentloaded")
            self._page.wait_for_timeout(random.randint(900, 2200))
            html = self._page.content()

            if any(m in html.casefold() for m in ACCESS_DENIED_MARKERS):
                self.consecutive_denials += 1
                self.denials += 1
                log.warning(
                    "Catawiki denied access to %s (%d in a row)",
                    url, self.consecutive_denials,
                )
                if self.should_stop:
                    log.error(
                        "%d consecutive denials — the session looks blocked. "
                        "Leave it alone for a few hours rather than retrying.",
                        self.consecutive_denials,
                    )
                return None

            self.consecutive_denials = 0
            self.pages_fetched += 1
            return html

        except PlaywrightError as exc:
            log.warning("fetch failed for %s: %s", url, exc)
            return None
        finally:
            self._last_nav = time.monotonic()

    def close(self) -> None:
        if self.denials:
            log.info(
                "session summary: %d pages fetched, %d denials",
                self.pages_fetched, self.denials,
            )
        try:
            self._ctx.close()
        except PlaywrightError:
            pass


@contextmanager
def fetcher(lock_timeout: int = 180):
    with browser_lock(lock_timeout), sync_playwright() as pw:
        f = Fetcher(pw)
        try:
            yield f
        finally:
            f.close()
"""Generic LLM-assisted store crawler.

Wraps ``LlmListingExtractor`` into an ``OfficialStoreCrawler``-compatible class
so any store can be monitored without writing a bespoke CSS-selector crawler.

Usage example::

    from market_monitor.llm_listing_extractor import LlmListingExtractor
    from market_monitor.llm_store_crawler import LlmStoreCrawler
    from market_monitor.http import HttpClient

    extractor = LlmListingExtractor()
    crawler = LlmStoreCrawler(
        store_name="animate",
        urls=["https://www.animate.co.jp/special/lottery/"],
        http_client=HttpClient(),
        extractor=extractor,
    )
    listings = crawler.fetch_listings()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .official_store_base import OfficialStoreListing

if TYPE_CHECKING:
    from .http import HttpClient
    from .llm_listing_extractor import LlmListingExtractor

logger = logging.getLogger(__name__)


@dataclass
class LlmStoreCrawler:
    """Crawls one or more store URLs and extracts listings via ``LlmListingExtractor``.

    Implements the ``OfficialStoreCrawler`` protocol (``store_name`` attribute +
    ``fetch_listings()`` method) so it can be used as a drop-in replacement for
    hand-written CSS crawlers.

    Deduplication is done by ``title`` (case-insensitive strip) — the first
    occurrence wins.  If the extractor returns nothing for a URL, a warning is
    logged and that URL is silently skipped so the overall list is not poisoned.
    """

    store_name: str
    urls: list[str]
    http_client: "HttpClient"
    extractor: "LlmListingExtractor"
    timeout_seconds: int = 30

    def fetch_listings(self, *, timeout_seconds: int | None = None) -> list[OfficialStoreListing]:
        """Fetch and extract listings from all configured URLs.

        Parameters
        ----------
        timeout_seconds:
            Override the instance-level timeout for this call.

        Returns
        -------
        list[OfficialStoreListing]
            Deduplicated listings; empty list on complete failure.
        """
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        all_listings: list[OfficialStoreListing] = []

        for url in self.urls:
            try:
                html = self.http_client.get_text(url, timeout_seconds=effective_timeout)
            except Exception:
                logger.warning(
                    "LlmStoreCrawler[%s]: failed to fetch url=%s", self.store_name, url
                )
                continue

            extracted = self.extractor.extract(
                html,
                store_name=self.store_name,
                base_url=url,
            )

            if not extracted:
                logger.warning(
                    "LlmStoreCrawler[%s]: extractor returned no listings for url=%s",
                    self.store_name,
                    url,
                )
                continue

            all_listings.extend(extracted)
            logger.debug(
                "LlmStoreCrawler[%s]: extracted %d listings from url=%s",
                self.store_name,
                len(extracted),
                url,
            )

        deduped = _dedup_by_title(all_listings)
        logger.info(
            "LlmStoreCrawler[%s]: fetch_listings complete listings=%d (raw=%d urls=%d)",
            self.store_name,
            len(deduped),
            len(all_listings),
            len(self.urls),
        )
        return deduped


def _dedup_by_title(listings: list[OfficialStoreListing]) -> list[OfficialStoreListing]:
    """Return *listings* with duplicates removed; first occurrence wins."""
    seen: set[str] = set()
    result: list[OfficialStoreListing] = []
    for listing in listings:
        key = listing.title.strip().lower()
        if key not in seen:
            seen.add(key)
            result.append(listing)
    return result

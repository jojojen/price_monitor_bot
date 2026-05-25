"""Trend-driven crawler that pre-populates the perceptual-hash database.

Runs on a background thread (default: every 6 hours, configurable). For each
TCG game we care about:
  1. Pull top-N ranked products from Snkrdunk's search-result HTML
     (`snkrdunk_ranking.iter_ranked_products`)
  2. For each product: download its thumbnail image, compute dHash,
     upsert into `card_image_fingerprints` with confidence_source="crawl"

After the first crawl, the most-popular boxes / cards already have hash
entries, so a user uploading a photo of them hits the fingerprint fast path
in `TcgImagePriceService.parse_image` and gets a result in milliseconds.

Design notes:
- Idempotent: `MonitorDatabase.upsert_card_image_fingerprint` is
  INSERT-or-touch-last-seen, so re-runs are cheap.
- Per-product errors don't abort the whole crawl (download flakes, hash
  failures, etc. just log + skip).
- Threading: sibling of `MarketplaceWatchMonitor`, same daemon-thread +
  Event-based stop pattern.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from market_monitor.http import HttpClient
from market_monitor.models import CardImageFingerprint, utc_now
from market_monitor.storage import MonitorDatabase

from .image_fingerprint import DEFAULT_ALGO, compute_dhash
from .snkrdunk_ranking import RankedProduct, iter_ranked_products

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CrawlSummary:
    game: str
    fetched: int
    persisted: int
    skipped: int

    def __str__(self) -> str:
        return f"{self.game}: fetched={self.fetched} persisted={self.persisted} skipped={self.skipped}"


class CardImageCrawler:
    def __init__(
        self,
        *,
        database: MonitorDatabase,
        http_client: HttpClient | None = None,
        games: Sequence[str] = ("pokemon", "ws", "union_arena"),
        per_game_limit: int = 30,
    ) -> None:
        self.database = database
        self.http = http_client or HttpClient()
        self.games = tuple(games)
        self.per_game_limit = per_game_limit

    def crawl_once(self) -> list[CrawlSummary]:
        summaries: list[CrawlSummary] = []
        for game in self.games:
            summaries.append(self._crawl_game(game))
        logger.info("Image crawl pass complete: %s", "; ".join(str(s) for s in summaries))
        return summaries

    def _crawl_game(self, game: str) -> CrawlSummary:
        try:
            products = iter_ranked_products(
                game=game, http_client=self.http, limit=self.per_game_limit,
            )
        except Exception as exc:
            logger.warning("ranking fetch crashed game=%s: %s", game, exc)
            return CrawlSummary(game=game, fetched=0, persisted=0, skipped=0)
        persisted = 0
        skipped = 0
        for product in products:
            if self._ingest_product(game, product):
                persisted += 1
            else:
                skipped += 1
        return CrawlSummary(
            game=game, fetched=len(products), persisted=persisted, skipped=skipped,
        )

    def _ingest_product(self, game: str, product: RankedProduct) -> bool:
        try:
            image_bytes = self.http.get_bytes(product.image_url, timeout_seconds=10)
        except Exception as exc:
            logger.debug("image fetch skipped url=%s error=%s", product.image_url, exc)
            return False
        target_hash = compute_dhash(image_bytes=image_bytes)
        if not target_hash:
            return False
        now = utc_now()
        fingerprint_id = MonitorDatabase._fingerprint_id(target_hash, DEFAULT_ALGO)
        fp = CardImageFingerprint(
            fingerprint_id=fingerprint_id,
            game=game,
            item_kind=product.item_kind,
            title=product.title,
            card_number=None,
            rarity=None,
            set_code=None,
            source_url=product.product_url,
            image_url=product.image_url,
            perceptual_hash=target_hash,
            fingerprint_algo=DEFAULT_ALGO,
            confidence_source="crawl",
            captured_at=now,
            last_seen_at=now,
        )
        try:
            self.database.upsert_card_image_fingerprint(fp)
        except Exception as exc:
            logger.warning(
                "upsert fingerprint failed game=%s title=%r error=%s",
                game, product.title, exc,
            )
            return False
        return True


class CardImageCrawlMonitor:
    """Background daemon thread that runs `CardImageCrawler.crawl_once()`
    on a fixed interval. Mirrors `MarketplaceWatchMonitor`'s pattern so the
    rest of the codebase doesn't have to learn a new scheduling primitive."""

    def __init__(
        self,
        *,
        crawler: CardImageCrawler,
        interval_seconds: int = 6 * 3600,  # 6 hours
        initial_delay_seconds: int = 60,
    ) -> None:
        self._crawler = crawler
        self._interval = max(60, int(interval_seconds))
        self._initial_delay = max(0, int(initial_delay_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="CardImageCrawlMonitor", daemon=True,
        )
        self._thread.start()
        logger.info(
            "CardImageCrawlMonitor started interval_seconds=%d games=%s",
            self._interval, self._crawler.games,
        )

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        # Initial delay so we don't slam external services right at boot
        if self._stop.wait(timeout=self._initial_delay):
            return
        while not self._stop.is_set():
            start = time.monotonic()
            try:
                self._crawler.crawl_once()
            except Exception:
                logger.exception("CardImageCrawler.crawl_once crashed")
            elapsed = time.monotonic() - start
            wait_secs = max(60, int(self._interval - elapsed))
            if self._stop.wait(timeout=wait_secs):
                return

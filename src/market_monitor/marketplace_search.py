"""Multi-source marketplace search abstraction.

Each marketplace (Mercari, Rakuma, future Yuyutei/PayPay/etc.) implements the
``MarketplaceSearchClient`` Protocol so the monitor can dispatch by source
without knowing the underlying scraper details.

Adding a new source = (a) write a search function that returns ``list[dict]``
with the canonical keys, (b) wrap it in a class with a ``source_name``
attribute and a ``search(...)`` method, (c) register the instance in the
monitor's clients map.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .mercari_search import DEFAULT_CONDITION_IDS, search_mercari

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketplaceListing:
    """Normalised listing returned by every ``MarketplaceSearchClient``.

    Optional fields default to None / sensible defaults so adding more fields
    later is non-breaking for existing callers and clients."""

    source: str
    item_id: str        # source-local id, NOT globally unique
    title: str
    price_jpy: int
    url: str
    thumbnail_url: str | None = None
    # Optional: populated by sources that have the data (Yuyutei et al.).
    stock_count: int | None = None
    listing_kind: str = "fixed_price"   # "fixed_price" / future "auction"


class MarketplaceSearchClient(Protocol):
    """Protocol every marketplace search client implements.

    The ``source_name`` attribute identifies the marketplace. ``search(...)``
    runs a free-text + price-max query and returns canonical listings.

    ``source_options`` is the per-source escape hatch — clients pull the keys
    they recognise (Mercari uses ``condition_ids``; future Yuyutei may use
    ``game`` / ``set_code``) and silently ignore the rest. This keeps the
    monitor dispatch loop source-agnostic.
    """

    source_name: str

    def search(
        self,
        query: str,
        *,
        price_max: int,
        max_results: int = 30,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] = ...,
    ) -> list[MarketplaceListing]: ...


class MercariSearchClient:
    """``MarketplaceSearchClient`` backed by the existing ``search_mercari``
    Playwright scraper. Source-specific option recognised: ``condition_ids``
    (tuple of Mercari condition enum ints 1-6)."""

    source_name: str = "mercari"

    def search(
        self,
        query: str,
        *,
        price_max: int,
        max_results: int = 30,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] | None = None,
    ) -> list[MarketplaceListing]:
        opts = dict(source_options or {})
        raw_conditions = opts.get("condition_ids")
        if raw_conditions is None:
            condition_ids: tuple[int, ...] | None = DEFAULT_CONDITION_IDS
        elif isinstance(raw_conditions, (list, tuple)):
            condition_ids = tuple(int(c) for c in raw_conditions if isinstance(c, (int, str)))
            condition_ids = condition_ids or DEFAULT_CONDITION_IDS
        else:
            condition_ids = DEFAULT_CONDITION_IDS
        try:
            raw = search_mercari(
                query,
                price_max=price_max,
                max_results=max_results,
                timeout_ms=timeout_ms,
                condition_ids=condition_ids,
            )
        except Exception:
            logger.exception(
                "MercariSearchClient: search failed query=%s price_max=%d",
                query, price_max,
            )
            return []
        return [
            MarketplaceListing(
                source="mercari",
                item_id=str(item.get("item_id", "")),
                title=str(item.get("title", "")),
                price_jpy=int(item.get("price_jpy", 0)),
                url=str(item.get("url", "")),
                thumbnail_url=item.get("thumbnail_url") or None,
            )
            for item in raw
            if item.get("item_id") and item.get("url")
        ]


def listing_to_record(listing: MarketplaceListing) -> dict[str, Any]:
    """Convert a ``MarketplaceListing`` into the dict shape ``record_marketplace_hits``
    consumes. Centralising the conversion here keeps the monitor loop tidy and
    documents the expected DB-write shape in one place."""
    return {
        "item_id": listing.item_id,
        "title": listing.title,
        "price_jpy": listing.price_jpy,
        "url": listing.url,
        "thumbnail_url": listing.thumbnail_url,
        "stock_count": listing.stock_count,
        "listing_kind": listing.listing_kind,
    }

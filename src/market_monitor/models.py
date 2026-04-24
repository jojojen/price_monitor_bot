from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Mapping

PriceKind = Literal["ask", "bid", "market", "last_sale"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class TrackedItem:
    item_id: str
    item_type: str
    category: str
    title: str
    aliases: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketOffer:
    source: str
    listing_id: str
    url: str
    title: str
    price_jpy: int
    price_kind: PriceKind
    captured_at: datetime
    source_category: str
    availability: str | None = None
    condition: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)
    score: float | None = None


@dataclass(frozen=True, slots=True)
class FairValueEstimate:
    item_id: str
    amount_jpy: int
    confidence: float
    sample_count: int
    reasoning: tuple[str, ...]
    computed_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class WatchRule:
    rule_id: str
    item_id: str
    source_scope: str = "all"
    discount_threshold_pct: float = 15.0
    enabled: bool = True
    schedule_minutes: int = 30

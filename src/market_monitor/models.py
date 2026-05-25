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


FeedbackConfidence = Literal["high", "medium", "low"]
FeedbackStatus = Literal[
    "analyzed",
    "fetch_failed",
    "extraction_failed",
    "low_consistency",
    "low_consensus",
]


@dataclass(frozen=True, slots=True)
class PriceFeedbackEvent:
    feedback_id: str
    chat_id: str | None
    item_id: str
    game: str
    item_kind: str
    original_fair_value_jpy: int | None
    claimed_url: str
    claimed_domain: str
    url_hash: str
    extracted_price_jpy_pass1: int | None
    extracted_price_jpy_pass2: int | None
    consistency_pct: float | None
    consensus_pct: float | None
    extraction_confidence: FeedbackConfidence
    raw_html_gzipped: bytes | None
    llm_notes_json: str
    status: FeedbackStatus
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class DomainTrust:
    domain_id: str
    game: str
    item_kind: str
    domain: str
    vote_count: int
    consensus_success_count: int
    consensus_fail_count: int
    bayes_accuracy_score: float
    suspended: bool = False
    first_seen_at: datetime = field(default_factory=utc_now)
    last_extraction_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ExtractionExample:
    example_id: str
    game: str
    item_kind: str
    domain: str
    title: str
    price_jpy: int
    captured_from_feedback_id: str
    captured_at: datetime = field(default_factory=utc_now)


# Layer 1 of the proactive image-recognition cache: perceptual-hash fingerprints
# of known product images. Populated by:
#   (a) crawler (Layer 2 — snkrdunk ranking, /trend hot cards) — confidence_source="crawl"
#   (b) successful image lookups — confidence_source="user_resolved"
ConfidenceSource = Literal["crawl", "user_resolved", "verified"]


@dataclass(frozen=True, slots=True)
class CardImageFingerprint:
    fingerprint_id: str
    game: str
    item_kind: str
    title: str
    card_number: str | None
    rarity: str | None
    set_code: str | None
    source_url: str         # where we found the product (e.g. snkrdunk product page)
    image_url: str          # direct URL of the image hashed
    perceptual_hash: str    # hex string; dhash 64-bit = 16 hex chars
    fingerprint_algo: str = "dhash"
    confidence_source: ConfidenceSource = "crawl"
    captured_at: datetime = field(default_factory=utc_now)
    last_seen_at: datetime = field(default_factory=utc_now)

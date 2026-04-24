from __future__ import annotations

import json
import re
from collections import Counter

from market_monitor import ReferenceSource
from market_monitor.models import MarketOffer
from market_monitor.pricing import FairValueCalculator
from tcg_tracker.service import TcgLookupResult

_GRADE_TITLE_RE = re.compile(r"\b(?P<company>PSA|BGS|ARS|CGC)\s*(?P<grade>\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)
_SECTION_RAW = "raw"
_SECTION_PSA10 = "psa10"
_SECTION_OTHER_GRADED = "other_graded"


def format_jpy(amount: int) -> str:
    return f"￥{amount:,}"


def format_lookup_result(result: TcgLookupResult) -> str:
    spec = result.spec
    header_parts = [f"[{spec.game}] {spec.title}"]
    if spec.card_number:
        header_parts.append(spec.card_number)
    if spec.rarity:
        header_parts.append(spec.rarity)

    lines = [" | ".join(header_parts)]
    for note in result.notes:
        lines.append(f"Note: {note}")
    if not result.offers:
        lines.append("No matching offers were found on the active reference sources.")
        return "\n".join(lines)

    if result.fair_value is not None:
        lines.append(
            f"Fair Value: {format_jpy(result.fair_value.amount_jpy)} | confidence={result.fair_value.confidence:.2f}"
        )

    best_ask = min(
        (offer for offer in result.offers if offer.price_kind == "ask"),
        default=None,
        key=lambda offer: offer.price_jpy,
    )
    best_market = min(
        (offer for offer in result.offers if offer.price_kind in {"market", "last_sale"}),
        default=None,
        key=lambda offer: offer.price_jpy,
    )
    best_bid = max(
        (offer for offer in result.offers if offer.price_kind == "bid"),
        default=None,
        key=lambda offer: offer.price_jpy,
    )

    if best_ask is not None:
        lines.append(f"Best Ask: {format_jpy(best_ask.price_jpy)} ({best_ask.source})")
    if best_market is not None:
        lines.append(f"Best Market: {format_jpy(best_market.price_jpy)} ({best_market.source})")
    if best_bid is not None:
        lines.append(f"Best Bid: {format_jpy(best_bid.price_jpy)} ({best_bid.source})")

    source_summary = ", ".join(
        f"{source} x{count}" for source, count in sorted(Counter(offer.source for offer in result.offers).items())
    )
    lines.append(f"Sources: {source_summary}")
    lines.append("Offers:")
    for offer in result.offers[:6]:
        metadata = [
            value
            for value in (
                offer.attributes.get("card_number", ""),
                offer.attributes.get("rarity", ""),
                offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""),
            )
            if value
        ]
        metadata_text = " / ".join(metadata) if metadata else "n/a"
        score = f"{offer.score:.1f}" if offer.score is not None else "n/a"
        lines.append(
            f"- [{offer.source} | {offer.price_kind}] {format_jpy(offer.price_jpy)} | "
            f"{offer.title} | {metadata_text} | score={score}"
        )
        lines.append(f"  {offer.url}")

    return "\n".join(lines)


def _normalize_grade_label(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"[^A-Z0-9.]", "", value.upper())
    return normalized or None


def _offer_grade_label(offer: MarketOffer) -> str | None:
    grade_attr = _normalize_grade_label(offer.attributes.get("grade_label"))
    if grade_attr is not None:
        return grade_attr

    match = _GRADE_TITLE_RE.search(offer.title)
    if match:
        return _normalize_grade_label(f"{match.group('company')}{match.group('grade')}")

    if offer.condition == "graded" or offer.attributes.get("is_graded") == "1":
        return "GRADED"
    return None


def _offer_section(offer: MarketOffer) -> str:
    grade_label = _offer_grade_label(offer)
    if grade_label is None:
        return _SECTION_RAW
    if grade_label == "PSA10":
        return _SECTION_PSA10
    return _SECTION_OTHER_GRADED


def _best_ask(offers: list[MarketOffer]) -> MarketOffer | None:
    return min((o for o in offers if o.price_kind == "ask"), default=None, key=lambda o: o.price_jpy)


def _best_market(offers: list[MarketOffer]) -> MarketOffer | None:
    return min((o for o in offers if o.price_kind in {"market", "last_sale"}), default=None, key=lambda o: o.price_jpy)


def _best_bid(offers: list[MarketOffer]) -> MarketOffer | None:
    return max((o for o in offers if o.price_kind == "bid"), default=None, key=lambda o: o.price_jpy)


def _avg_price(offers: list[MarketOffer]) -> int | None:
    prices = [o.price_jpy for o in offers if o.price_kind in {"market", "last_sale", "ask"}]
    if not prices:
        return None
    return sum(prices) // len(prices)


def _section_fair_value(result: TcgLookupResult, offers: list[MarketOffer]) -> str | None:
    fair_value = FairValueCalculator().calculate(result.item.item_id, offers)
    if fair_value is None:
        return None
    return f"Fair Value: {format_jpy(fair_value.amount_jpy)} | confidence={fair_value.confidence:.2f}"


def _section_reference_offer(offers: list[MarketOffer]) -> MarketOffer | None:
    for selector in (_best_market, _best_ask, _best_bid):
        offer = selector(offers)
        if offer is not None:
            return offer
    return None


def format_lookup_result_telegram(result: TcgLookupResult) -> str:
    spec = result.spec
    header_parts = [f"[{spec.game}] {spec.title}"]
    if spec.card_number:
        header_parts.append(spec.card_number)
    if spec.rarity:
        header_parts.append(spec.rarity)

    lines = [" | ".join(header_parts)]

    if not result.offers:
        lines.append("No matching offers were found on the active reference sources.")
        return "\n".join(lines)

    raw_offers = [offer for offer in result.offers if _offer_section(offer) == _SECTION_RAW]
    psa10_offers = [offer for offer in result.offers if _offer_section(offer) == _SECTION_PSA10]
    other_graded_offers = [offer for offer in result.offers if _offer_section(offer) == _SECTION_OTHER_GRADED]

    def append_section(label: str, section_offers: list[MarketOffer]) -> None:
        if not section_offers:
            return

        lines.append("")
        lines.append(label)

        fair_value_text = _section_fair_value(result, section_offers)
        if fair_value_text is not None:
            lines.append(fair_value_text)

        avg = _avg_price(section_offers)
        bid = _best_bid(section_offers)
        ask = _best_ask(section_offers)
        market = _best_market(section_offers)
        reference_offer = _section_reference_offer(section_offers)

        if avg is not None:
            lines.append(f"Avg Price: {format_jpy(avg)}")
        if bid is not None:
            lines.append(f"Best Bid: {format_jpy(bid.price_jpy)} ({bid.source})")
        if ask is not None:
            lines.append(f"Best Ask: {format_jpy(ask.price_jpy)} ({ask.source})")
        if market is not None:
            lines.append(f"Best Market: {format_jpy(market.price_jpy)} ({market.source})")
        if reference_offer is not None:
            lines.append(f"Source URL: {reference_offer.url}")
        if avg is None and bid is None and ask is None and market is None:
            lines.append("n/a")

    append_section("Raw", raw_offers)
    append_section("PSA 10", psa10_offers)
    append_section("其他鑑定卡", other_graded_offers)

    source_summary = ", ".join(
        f"{source} x{count}" for source, count in sorted(Counter(offer.source for offer in result.offers).items())
    )
    lines.append("")
    lines.append(f"Sources: {source_summary}")

    return "\n".join(lines)


def lookup_result_payload(result: TcgLookupResult) -> dict[str, object]:
    return {
        "item": {
            "item_id": result.item.item_id,
            "title": result.item.title,
            "attributes": dict(result.item.attributes),
        },
        "notes": list(result.notes),
        "fair_value": None
        if result.fair_value is None
        else {
            "amount_jpy": result.fair_value.amount_jpy,
            "confidence": result.fair_value.confidence,
            "sample_count": result.fair_value.sample_count,
            "reasoning": list(result.fair_value.reasoning),
        },
        "offers": [
            {
                "source": offer.source,
                "price_kind": offer.price_kind,
                "title": offer.title,
                "url": offer.url,
                "price_jpy": offer.price_jpy,
                "attributes": dict(offer.attributes),
                "score": offer.score,
            }
            for offer in result.offers
        ],
    }


def lookup_result_to_json(result: TcgLookupResult) -> str:
    payload = lookup_result_payload(result)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def format_reference_sources(sources: tuple[ReferenceSource, ...]) -> str:
    if not sources:
        return "No reference sources matched the requested filters."

    lines = ["Reference sources:"]
    for source in sources:
        games = ",".join(source.games)
        roles = ",".join(source.reference_roles)
        lines.append(
            f"- {source.name} [{source.id}] | games={games} | kind={source.source_kind} | "
            f"trust={source.trust_score:.2f} | price_weight={source.price_weight:.2f}"
        )
        lines.append(f"  roles={roles}")
        lines.append(f"  url={source.url}")
        lines.append(f"  notes={source.notes}")
    return "\n".join(lines)


def reference_sources_payload(sources: tuple[ReferenceSource, ...]) -> list[dict[str, object]]:
    return [
        {
            "id": source.id,
            "name": source.name,
            "games": list(source.games),
            "source_kind": source.source_kind,
            "reference_roles": list(source.reference_roles),
            "price_weight": source.price_weight,
            "trust_score": source.trust_score,
            "url": source.url,
            "notes": source.notes,
        }
        for source in sources
    ]


def reference_sources_to_json(sources: tuple[ReferenceSource, ...]) -> str:
    payload = reference_sources_payload(sources)
    return json.dumps(payload, ensure_ascii=False, indent=2)

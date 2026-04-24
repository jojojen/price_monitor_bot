from __future__ import annotations

from typing import Iterable

from .models import FairValueEstimate, MarketOffer

KIND_WEIGHTS = {
    "ask": 1.0,
    "market": 0.95,
    "last_sale": 1.05,
    "bid": 0.65,
}

KIND_ADJUSTMENTS = {
    "ask": 1.0,
    "market": 1.0,
    "last_sale": 1.0,
    "bid": 1.25,
}


def weighted_median(points: Iterable[tuple[int, float]]) -> int:
    ordered = sorted(points, key=lambda point: point[0])
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0:
        raise ValueError("weighted_median requires at least one positive weight")

    midpoint = total_weight / 2
    running_weight = 0.0
    for value, weight in ordered:
        running_weight += weight
        if running_weight >= midpoint:
            return value
    return ordered[-1][0]


class FairValueCalculator:
    def calculate(self, item_id: str, offers: Iterable[MarketOffer]) -> FairValueEstimate | None:
        offer_list = list(offers)
        if not offer_list:
            return None

        calibrated_points: list[tuple[int, float]] = []
        sources: set[str] = set()
        kinds: set[str] = set()
        reasoning: list[str] = []

        for offer in offer_list:
            weight = KIND_WEIGHTS.get(offer.price_kind, 0.5)
            adjusted_price = int(round(offer.price_jpy * KIND_ADJUSTMENTS.get(offer.price_kind, 1.0)))

            if offer.availability and "×" in offer.availability:
                weight *= 0.9
            if offer.score is not None and offer.score < 55:
                weight *= 0.7

            calibrated_points.append((adjusted_price, weight))
            sources.add(offer.source)
            kinds.add(offer.price_kind)

        calibrated_values = [value for value, _ in calibrated_points]
        if len(calibrated_points) == 1:
            fair_value = calibrated_values[0]
        elif len(calibrated_points) == 2:
            fair_value = round(sum(calibrated_values) / 2)
            reasoning.append("small samples use a midpoint across adjusted values")
        else:
            fair_value = weighted_median(calibrated_points)

        if "ask" in kinds:
            reasoning.append("ask prices anchor the upper side of the fair value")
        if "bid" in kinds:
            reasoning.append("bid prices are uplifted to approximate resale value")
        if len(sources) == 1:
            reasoning.append("confidence is limited because only one source is active in phase one")

        confidence = min(0.95, 0.35 + 0.1 * len(offer_list) + 0.12 * len(kinds) + 0.08 * len(sources))

        return FairValueEstimate(
            item_id=item_id,
            amount_jpy=fair_value,
            confidence=round(confidence, 2),
            sample_count=len(offer_list),
            reasoning=tuple(reasoning),
        )

from __future__ import annotations

from datetime import datetime, timezone

from market_monitor.models import MarketOffer
from market_monitor.pricing import FairValueCalculator, weighted_median


def test_weighted_median_prefers_central_weighted_point() -> None:
    assert weighted_median([(100, 1.0), (200, 2.0), (500, 0.5)]) == 200


def test_fair_value_blends_ask_and_bid() -> None:
    calculator = FairValueCalculator()
    offers = [
        MarketOffer(
            source="yuyutei",
            listing_id="a",
            url="https://example.com/a",
            title="Example Card",
            price_jpy=10000,
            price_kind="ask",
            captured_at=datetime.now(timezone.utc),
            source_category="poc",
            attributes={},
        ),
        MarketOffer(
            source="yuyutei",
            listing_id="b",
            url="https://example.com/b",
            title="Example Card",
            price_jpy=7000,
            price_kind="bid",
            captured_at=datetime.now(timezone.utc),
            source_category="poc",
            attributes={},
        ),
    ]

    fair_value = calculator.calculate("item-1", offers)

    assert fair_value is not None
    assert fair_value.amount_jpy == 9375
    assert fair_value.sample_count == 2
    assert fair_value.confidence > 0.5

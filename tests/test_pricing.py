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


def _offer(
    *,
    price: int,
    product_kind: str | None = None,
    price_kind: str = "ask",
    listing_id: str = "x",
) -> MarketOffer:
    attributes: dict[str, str] = {}
    if product_kind is not None:
        attributes["product_kind"] = product_kind
    return MarketOffer(
        source="cardrush_pokemon",
        listing_id=listing_id,
        url=f"https://example.com/{listing_id}",
        title="Example",
        price_jpy=price,
        price_kind=price_kind,
        captured_at=datetime.now(timezone.utc),
        source_category="specialty_store",
        attributes=attributes,
    )


def test_calculate_filters_by_expected_product_kind() -> None:
    calculator = FairValueCalculator()
    offers = [
        _offer(price=1800, listing_id="single-a"),
        _offer(price=1900, listing_id="single-b"),
        _offer(price=12000, product_kind="sealed_box", listing_id="box-a"),
    ]

    fair_value = calculator.calculate("item-1", offers, expected_product_kind="sealed_box")

    assert fair_value is not None
    assert fair_value.amount_jpy == 12000
    assert fair_value.sample_count == 1


def test_calculate_returns_none_when_no_offer_matches_expected_kind() -> None:
    calculator = FairValueCalculator()
    offers = [
        _offer(price=1800, listing_id="single-a"),
        _offer(price=1900, listing_id="single-b"),
    ]

    fair_value = calculator.calculate("item-1", offers, expected_product_kind="sealed_box")

    assert fair_value is None


def test_calculate_treats_missing_product_kind_as_card() -> None:
    calculator = FairValueCalculator()
    offers = [
        _offer(price=1800, listing_id="a"),
        _offer(price=2000, product_kind="card", listing_id="b"),
    ]

    fair_value = calculator.calculate("item-1", offers, expected_product_kind="card")

    assert fair_value is not None
    assert fair_value.sample_count == 2

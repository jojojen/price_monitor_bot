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
    is_graded: bool = False,
) -> MarketOffer:
    attributes: dict[str, str] = {}
    if product_kind is not None:
        attributes["product_kind"] = product_kind
    if is_graded:
        attributes["is_graded"] = "1"
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


def test_calculate_excludes_graded_offers_by_default() -> None:
    """Default behaviour excludes graded (PSA / BGS / CGC) listings from
    the fair-value sample — they distort the median upward. Mirrors the
    UNION ARENA 綾波レイ bug: 3 raw at ~¥4-5k + 1 PSA10 at ¥14k → without
    exclusion FV becomes ¥7k+; with exclusion FV ≈ raw median."""
    calculator = FairValueCalculator()
    offers = [
        _offer(price=4500, listing_id="raw-a"),
        _offer(price=5000, listing_id="raw-b"),
        _offer(price=5200, listing_id="raw-c"),
        _offer(price=14000, listing_id="psa10", is_graded=True),  # outlier
    ]

    fair_value = calculator.calculate("item-1", offers)

    assert fair_value is not None
    # Sample count should be 3 raw (graded excluded)
    assert fair_value.sample_count == 3
    # Fair value should be around the raw median (~5000), not pulled toward
    # the 14000 PSA10 outlier
    assert 4500 <= fair_value.amount_jpy <= 5500


def test_calculate_includes_graded_when_exclude_graded_false() -> None:
    """Opt-in path for callers that explicitly want graded prices (e.g. a
    hypothetical /lookup graded command)."""
    calculator = FairValueCalculator()
    offers = [
        _offer(price=5000, listing_id="raw-a"),
        _offer(price=14000, listing_id="psa10", is_graded=True),
    ]

    fair_value = calculator.calculate("item-1", offers, exclude_graded=False)

    assert fair_value is not None
    assert fair_value.sample_count == 2  # both offers counted


def test_calculate_returns_none_when_only_graded_offers() -> None:
    """When every offer is graded and exclude_graded is True (default),
    return None gracefully rather than synthesising a fair value from 0
    samples or accidentally including the graded items."""
    calculator = FairValueCalculator()
    offers = [
        _offer(price=14000, listing_id="psa10", is_graded=True),
        _offer(price=15000, listing_id="psa9", is_graded=True),
    ]

    fair_value = calculator.calculate("item-1", offers)

    assert fair_value is None

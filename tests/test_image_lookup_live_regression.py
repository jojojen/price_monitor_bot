from __future__ import annotations

import os

import pytest

from tcg_tracker.image_lookup import TcgImageLookupOutcome, TcgImagePriceService

from tests.image_lookup_case_fixtures import iter_image_lookup_live_cases

pytestmark = [
    pytest.mark.live_image_lookup,
    pytest.mark.skipif(
        os.getenv("OPENCLAW_RUN_LIVE_IMAGE_FIXTURES") != "1",
        reason="Set OPENCLAW_RUN_LIVE_IMAGE_FIXTURES=1 to run live image lookup regression fixtures.",
    ),
]


@pytest.fixture(scope="module")
def image_lookup_service() -> TcgImagePriceService:
    return TcgImagePriceService()


@pytest.mark.parametrize(
    "case",
    iter_image_lookup_live_cases(),
    ids=lambda case: case.case_id,
)
def test_live_image_lookup_regression_case(
    image_lookup_service: TcgImagePriceService,
    case,
) -> None:
    outcome = image_lookup_service.lookup_image(case.image_path, persist=False)
    _assert_outcome_matches_expected(outcome, case.payload["expected"])


def _assert_outcome_matches_expected(
    outcome: TcgImageLookupOutcome,
    expected: object,
) -> None:
    assert isinstance(expected, dict)
    assert outcome.status == expected["status"]

    expected_parsed = expected.get("parsed")
    if isinstance(expected_parsed, dict):
        for key, value in expected_parsed.items():
            assert getattr(outcome.parsed, key) == value

    expected_lookup_spec = expected.get("lookup_spec")
    if isinstance(expected_lookup_spec, dict):
        assert outcome.lookup_result is not None
        for key, value in expected_lookup_spec.items():
            assert getattr(outcome.lookup_result.spec, key) == value

    expected_offer_count = expected.get("offer_count")
    actual_offer_count = 0 if outcome.lookup_result is None else len(outcome.lookup_result.offers)
    if isinstance(expected_offer_count, int):
        assert actual_offer_count == expected_offer_count

    minimum_offer_count = expected.get("min_offer_count")
    if isinstance(minimum_offer_count, int):
        assert actual_offer_count >= minimum_offer_count

    maximum_offer_count = expected.get("max_offer_count")
    if isinstance(maximum_offer_count, int):
        assert actual_offer_count <= maximum_offer_count

    required_sources = expected.get("required_sources")
    if isinstance(required_sources, list):
        actual_sources = set() if outcome.lookup_result is None else {offer.source for offer in outcome.lookup_result.offers}
        for source in required_sources:
            assert source in actual_sources

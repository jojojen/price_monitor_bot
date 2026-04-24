from __future__ import annotations

from tests.image_lookup_case_fixtures import iter_image_lookup_live_cases


def test_image_lookup_live_cases_have_expected_metadata() -> None:
    cases = iter_image_lookup_live_cases()

    assert cases
    for case in cases:
        assert case.payload["case_id"] == case.case_id
        assert case.image_path.exists()
        assert case.expected_path.exists()

        expected = case.payload["expected"]
        assert isinstance(expected, dict)
        assert isinstance(expected.get("status"), str)

        parsed = expected.get("parsed")
        if parsed is not None:
            assert isinstance(parsed, dict)

        lookup_spec = expected.get("lookup_spec")
        if lookup_spec is not None:
            assert isinstance(lookup_spec, dict)

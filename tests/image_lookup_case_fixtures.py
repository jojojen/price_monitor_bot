from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_LOOKUP_LIVE_CASES_ROOT = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "image_lookup"
    / "live_regression_cases"
)
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class ImageLookupLiveCase:
    case_id: str
    case_dir: Path
    image_path: Path
    expected_path: Path
    payload: dict[str, object]


def iter_image_lookup_live_cases() -> tuple[ImageLookupLiveCase, ...]:
    cases: list[ImageLookupLiveCase] = []
    for case_dir in sorted(IMAGE_LOOKUP_LIVE_CASES_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        expected_path = case_dir / "expected.json"
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        image_candidates = sorted(
            path
            for path in case_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if len(image_candidates) != 1:
            raise AssertionError(
                f"Expected exactly one image fixture in {case_dir}, found {len(image_candidates)}."
            )
        cases.append(
            ImageLookupLiveCase(
                case_id=case_dir.name,
                case_dir=case_dir,
                image_path=image_candidates[0],
                expected_path=expected_path,
                payload=payload,
            )
        )
    return tuple(cases)


def get_image_lookup_live_case(case_id: str) -> ImageLookupLiveCase:
    for case in iter_image_lookup_live_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(f"Unknown image lookup live case: {case_id}")

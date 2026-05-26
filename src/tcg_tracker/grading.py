"""Cross-source grading-detection helper.

PSA / BGS / CGC graded cards trade at multiples of the raw-card price.
Before this module, only the Magi client had its own specialised grading
detection — Mercari / Cardrush / Yuyutei / Snkrdunk / Surugaya all
returned graded listings without any flag, so a PSA10 outlier flowed
straight into `FairValueCalculator` and inflated the median.

Usage at scrape time:

    from .grading import looks_like_graded
    attributes = {...}
    if looks_like_graded(listing_title) or looks_like_graded(listing_detail):
        attributes["is_graded"] = "1"

Downstream (`FairValueCalculator.calculate(exclude_graded=True)`,
`OpportunityPriceChecker.check`) reads `attributes["is_graded"] == "1"`
to filter samples.
"""

from __future__ import annotations

import re


# Patterns:
#   "PSA10", "PSA 10", "PSA9.5" (some grades have decimals)
#   "BGS 9.5", "BGS9.5"
#   "CGC 10", "CGC10"
#   Japanese 鑑定 markers: 鑑定済 / 鑑定品 / 鑑定書 / 鑑定カード
_GRADING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bPSA\s?\d{1,2}(?:\.\d)?\b", re.IGNORECASE),
    re.compile(r"\bBGS\s?\d{1,2}(?:\.\d)?\b", re.IGNORECASE),
    re.compile(r"\bCGC\s?\d{1,2}(?:\.\d)?\b", re.IGNORECASE),
    re.compile(r"鑑定済|鑑定品|鑑定書|鑑定カード"),
)


def looks_like_graded(text: str | None) -> bool:
    """Return True if the given text contains a recognised grading marker.
    Safe to call on empty / None — returns False without raising."""
    if not text:
        return False
    return any(pattern.search(text) for pattern in _GRADING_PATTERNS)

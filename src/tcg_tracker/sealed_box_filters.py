"""Shared helpers for classifying TCG marketplace listings as sealed boxes."""

from __future__ import annotations

import re

_BOX_POSITIVE_MARKERS: tuple[str, ...] = (
    # Explicit "sealed BOX" suffix
    "未開封box",
    "未開封 box",
    "未開封　box",
    "1box",
    "１box",
    "30パック入",
    # Booster/expansion box product line suffixes
    "ブースターボックス",
    "ブースターbox",
    "拡張パックbox",
    "ハイクラスパックbox",
    # Pokemon-specific real sealed product lines that legitimately end in ボックス
    "プレミアムトレーナーボックス",
    "プレミアムトレーナー box",
    "コレクションボックス",
    "シールドボックス",
    "プロモボックス",
)

# Strong negative signals: rarity codes, explicit single-card phrasing, and
# product types that are NOT booster-style sealed boxes (starter sets and
# accessories — these are sealed but priced differently and confuse the fair
# value of a booster box query).
_SINGLE_CARD_MARKERS: tuple[str, ...] = (
    # Pokemon single-card rarity codes (matched with surrounding whitespace so
    # they don't trigger on substrings inside longer words).
    " ur ",
    " sr ",
    " sar ",
    " ar ",
    " hr ",
    " rr ",
    " chr ",
    "シングル",
    "1枚",
    "single card",
    # Starter / theme decks — sealed but not booster boxes
    "スターターセット",
    "スターターデッキ",
    "starter set",
    "starter deck",
    # Accessories that happen to ship in a "box"
    "デッキケース",
    "デッキボックス",
    "カードファイルセット",
    "カードボックス",
)

# Card-number patterns like "349/190", "001/078" — a hard tell for single cards.
_CARD_NUMBER_RE = re.compile(r"\b\d{2,3}\s*/\s*\d{2,3}\b")


def looks_like_sealed_box_listing(text: str) -> bool:
    """Return True only if ``text`` reads like a sealed-box product listing.

    Uses an allow-list of explicit box markers (pack count, BOX suffix on a known
    product line, etc.) and rejects anything that also carries strong
    single-card signals (rarity codes, card-number pattern). The old "any
    occurrence of 'box'" rule misclassified single-card listings whose
    surrounding text mentioned a set's BOX edition.
    """
    if not text:
        return False
    padded = " " + text.lower() + " "
    if _CARD_NUMBER_RE.search(padded):
        return False
    if any(marker in padded for marker in _SINGLE_CARD_MARKERS):
        return False
    return any(marker in padded for marker in _BOX_POSITIVE_MARKERS)

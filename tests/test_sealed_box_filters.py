"""Tests for `looks_like_sealed_box_listing` — the shared classifier used by
cardrush.py and magi.py to decide whether a marketplace listing represents a
sealed booster box or a single card."""
from __future__ import annotations

import pytest

from tcg_tracker.sealed_box_filters import looks_like_sealed_box_listing


@pytest.mark.parametrize(
    "text",
    [
        "強化拡張パック『ポケモンカード151』(SV2a)【未開封BOX】70,800円 在庫数 7点",
        "ポケモンカードゲーム MEGAアビスアイ 拡張パックBOX 30パック入 14,800円",
        "ハイクラスパック MEGAドリームex 未開封BOX ¥ 17,800 出品数 36",
        "ブースターボックス Scarlet & Violet 4500円",
        "ポケカ 1BOX 新品未開封 12000円",
    ],
)
def test_accepts_explicit_sealed_box_listings(text: str) -> None:
    assert looks_like_sealed_box_listing(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Single card with explicit card number — strongest negative signal
        "ポケモンカード MEGAアビスアイ SR 001/078 1,880円",
        "リザードンex SAR 349/190 29,000円",
        "ゲンガーEX TD 010/049 ¥ 42,800",
        # Single card whose listing happens to mention BOX in cross-sell text
        "ポケモンカード MEGAアビスアイ 単品 SR 1,880円 関連商品BOX",
        # Rarity-coded single cards with no card number but with " SR " etc.
        "ピカチュウ SR シングル 800円",
        # No box markers, no rarity — should default to False (not enough info)
        "ポケモンカード 通常パック 1パック",
        # Empty / whitespace
        "",
        "   ",
    ],
)
def test_rejects_non_sealed_box_listings(text: str) -> None:
    assert looks_like_sealed_box_listing(text) is False


def test_rejects_card_file_set_with_box_substring_but_card_number() -> None:
    text = "ポケモンカード151 リザードン SR 006/165 ファイルセット 9,980円"
    assert looks_like_sealed_box_listing(text) is False


def test_accepts_real_box_even_with_box_in_unrelated_phrase() -> None:
    text = "ポケモンカードゲーム 強化拡張パック ポケモンカード151 未開封BOX 30パック入 ¥7,800"
    assert looks_like_sealed_box_listing(text) is True

from __future__ import annotations

from dataclasses import dataclass

from .catalog import TcgCardSpec


@dataclass(frozen=True, slots=True)
class LiveExample:
    spec: TcgCardSpec
    expected_card_number: str
    expected_rarity: str


EXAMPLE_CARDS: tuple[LiveExample, ...] = (
    LiveExample(
        spec=TcgCardSpec(
            game="pokemon",
            title="ピカチュウex",
            card_number="132/106",
            rarity="SAR",
            set_code="sv08",
            set_name="超電ブレイカー",
        ),
        expected_card_number="132/106",
        expected_rarity="SAR",
    ),
    LiveExample(
        spec=TcgCardSpec(
            game="pokemon",
            title="ピカチュウex",
            card_number="234/193",
            rarity="SAR",
            set_code="m02a",
            set_name="MEGAドリームex",
        ),
        expected_card_number="234/193",
        expected_rarity="SAR",
    ),
    LiveExample(
        spec=TcgCardSpec(
            game="ws",
            title="15th Anniversary カレン(サイン入り)",
            card_number="KMS/W133-002SSP",
            rarity="SSP",
            set_code="kms",
            set_name="きんいろモザイク 15th Anniversary",
        ),
        expected_card_number="KMS/W133-002SSP",
        expected_rarity="SSP",
    ),
)

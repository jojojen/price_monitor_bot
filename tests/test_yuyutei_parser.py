from __future__ import annotations

from pathlib import Path

from datetime import datetime, timezone

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.matching import minimum_match_score, score_tcg_offer
from tcg_tracker.yuyutei import YuyuteiClient

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FixtureHttpClient(HttpClient):
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        super().__init__(user_agent="fixture")

    def get_text(self, url: str, *, params=None, encoding="utf-8") -> str:  # type: ignore[override]
        target = url
        if "/sell/" in target:
            return self.responses["sell"]
        if "/buy/" in target:
            return self.responses["buy"]
        raise AssertionError(f"unexpected url: {url}")


def test_pokemon_lookup_matches_exact_card_number_and_rarity() -> None:
    client = YuyuteiClient(
        FixtureHttpClient(
            {
                "sell": (FIXTURE_DIR / "yuyutei_pokemon_sell_search.html").read_text(encoding="utf-8"),
                "buy": (FIXTURE_DIR / "yuyutei_pokemon_buy_search.html").read_text(encoding="utf-8"),
            }
        )
    )
    spec = TcgCardSpec(game="pokemon", title="ピカチュウex", card_number="132/106", rarity="SAR", set_code="sv08")

    offers = client.lookup(spec)

    assert offers
    assert offers[0].attributes["card_number"] == "132/106"
    assert offers[0].attributes["rarity"] == "SAR"
    assert any(offer.price_kind == "ask" and offer.price_jpy == 99800 for offer in offers)
    assert any(offer.price_kind == "bid" and offer.price_jpy == 80000 for offer in offers)
    bid_offer = next(offer for offer in offers if offer.price_kind == "bid" and offer.price_jpy == 80000)
    assert bid_offer.attributes["price_change_direction"] == "up"
    assert bid_offer.attributes["compare_price_jpy"] == "35000"


def test_ws_lookup_matches_signed_card() -> None:
    client = YuyuteiClient(
        FixtureHttpClient(
            {
                "sell": (FIXTURE_DIR / "yuyutei_ws_sell_karen.html").read_text(encoding="utf-8"),
                "buy": (FIXTURE_DIR / "yuyutei_ws_buy_karen.html").read_text(encoding="utf-8"),
            }
        )
    )
    spec = TcgCardSpec(
        game="ws",
        title="15th Anniversary カレン(サイン入り)",
        card_number="KMS/W133-002SSP",
        rarity="SSP",
        set_code="kms",
    )

    offers = client.lookup(spec)

    assert offers
    assert offers[0].attributes["card_number"] == "KMS/W133-002SSP"
    assert offers[0].attributes["rarity"] == "SSP"
    assert any(offer.price_kind == "ask" and offer.price_jpy == 24800 for offer in offers)
    assert any(offer.price_kind == "bid" and offer.price_jpy == 15000 for offer in offers)
    bid_offer = next(offer for offer in offers if offer.price_kind == "bid" and offer.price_jpy == 15000)
    assert bid_offer.attributes["price_change_direction"] == "up"


def test_name_only_lookup_returns_candidate_variants() -> None:
    client = YuyuteiClient(
        FixtureHttpClient(
            {
                "sell": (FIXTURE_DIR / "yuyutei_pokemon_sell_search.html").read_text(encoding="utf-8"),
                "buy": (FIXTURE_DIR / "yuyutei_pokemon_buy_search.html").read_text(encoding="utf-8"),
            }
        )
    )
    spec = TcgCardSpec(game="pokemon", title="ピカチュウex")

    offers = client.lookup(spec)
    candidate_numbers = {offer.attributes["card_number"] for offer in offers}

    assert offers
    assert "132/106" in candidate_numbers
    assert "234/193" in candidate_numbers


def test_no_results_return_empty_list() -> None:
    client = YuyuteiClient(
        FixtureHttpClient(
            {
                "sell": (FIXTURE_DIR / "yuyutei_ws_sell_search.html").read_text(encoding="utf-8"),
                "buy": (FIXTURE_DIR / "yuyutei_ws_sell_search.html").read_text(encoding="utf-8"),
            }
        )
    )
    spec = TcgCardSpec(game="ws", title="喜多川 海夢")

    assert client.lookup(spec) == []


def test_search_terms_include_card_number_and_pokemon_ex_variant() -> None:
    spec = TcgCardSpec(game="pokemon", title="メガシビルドン", card_number="235/193", rarity="SAR")

    terms = YuyuteiClient._search_terms(spec)  # type: ignore[attr-defined]

    assert terms[0] == "235/193"
    assert "メガシビルドン" in terms
    assert "メガシビルドンex" in terms


def test_union_arena_search_terms_include_padded_card_number_variant() -> None:
    spec = TcgCardSpec(game="union area", title="綾波レイ", card_number="UAPR/EVA-1-71")

    terms = YuyuteiClient._search_terms(spec)  # type: ignore[attr-defined]

    assert terms[0] == "UAPR/EVA-1-71"
    assert "UAPR/EVA-1-071" in terms
    assert "綾波レイ" in terms


def test_union_arena_matching_tolerates_different_set_code_prefix() -> None:
    """Yuyutei lists this card as UA44BT/... while users / other sites use UAPR/...
    The same EVA-1-071 suffix should still produce a match."""
    spec = TcgCardSpec(game="union_arena", title="綾波レイ", card_number="UAPR/EVA-1-071")
    offer = MarketOffer(
        source="yuyutei",
        listing_id="ua44bt:eva1",
        url="https://yuyu-tei.jp/sell/ua/card/eva1/10093",
        title="綾波 レイ",
        price_jpy=2200,
        price_kind="ask",
        captured_at=datetime.now(timezone.utc),
        source_category="ua",
        attributes={
            "card_number": "UA44BT/EVA-1-071",
            "rarity": "R",
            "version_code": "eva1",
            "image_alt": "UA44BT/EVA-1-071 R 綾波 レイ",
        },
    )

    score = score_tcg_offer(spec, offer)

    assert score >= minimum_match_score(spec)


def test_union_arena_matching_still_rejects_different_card_suffix() -> None:
    spec = TcgCardSpec(game="union_arena", title="綾波レイ", card_number="UAPR/EVA-1-071")
    offer = MarketOffer(
        source="yuyutei",
        listing_id="other",
        url="https://example.com/other",
        title="シンジ",
        price_jpy=2200,
        price_kind="ask",
        captured_at=datetime.now(timezone.utc),
        source_category="ua",
        attributes={
            "card_number": "UA44BT/EVA-1-072",
            "rarity": "R",
            "version_code": "eva1",
        },
    )

    assert score_tcg_offer(spec, offer) < minimum_match_score(spec)

from __future__ import annotations

from pathlib import Path

from market_monitor.http import HttpClient
from tcg_tracker.catalog import TcgCardSpec
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

from __future__ import annotations

from urllib.parse import urlencode

from market_monitor.http import HttpClient
from tcg_tracker.cardrush import CARDRUSH_PRODUCT_LIST_URL, CardrushPokemonClient
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.magi import MAGI_PRODUCT_SEARCH_URL, MagiProductClient


class FixtureHttpClient(HttpClient):
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        super().__init__(user_agent="fixture")

    def get_text(self, url: str, *, params=None, encoding="utf-8", headers=None) -> str:  # type: ignore[override]
        target = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            target = f"{url}{separator}{query}"
        if target not in self.responses:
            raise AssertionError(f"unexpected url: {target}")
        return self.responses[target]


class FailingHttpClient(HttpClient):
    def __init__(self) -> None:
        self.calls = 0
        super().__init__(user_agent="fixture")

    def get_text(self, url: str, *, params=None, encoding="utf-8", headers=None) -> str:  # type: ignore[override]
        self.calls += 1
        raise RuntimeError("blocked")


def test_cardrush_lookup_matches_precise_pokemon_variant() -> None:
    CardrushPokemonClient.reset_temporary_disable()
    html = """
    <ul class="item_list">
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/198765">メガシビルドンex【SAR】{ 235/193 } [ M2a ] 1,180円 (税込) 在庫数 140枚</a>
        </div>
      </li>
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/198766">メガシビルドンex【MA】{ 225/193 } [ M2a ] 780円 (税込) 在庫数 120枚</a>
        </div>
      </li>
    </ul>
    """
    client = CardrushPokemonClient(
        FixtureHttpClient(
            {
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=235%2F193": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%93%E3%83%AB%E3%83%89%E3%83%B3": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=SAR": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%93%E3%83%AB%E3%83%89%E3%83%B3ex": html,
            }
        )
    )

    offers = client.lookup(
        TcgCardSpec(game="pokemon", title="メガシビルドン", card_number="235/193", rarity="SAR")
    )

    assert len(offers) == 1
    assert offers[0].source == "cardrush_pokemon"
    assert offers[0].price_jpy == 1180
    assert offers[0].attributes["card_number"] == "235/193"
    assert offers[0].attributes["version_code"] == "m2a"


def test_cardrush_lookup_still_matches_when_input_card_number_drops_zero_padding() -> None:
    CardrushPokemonClient.reset_temporary_disable()
    html = """
    <ul class="item_list">
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/71713">メガリザードンXex【SAR】{ 110/080 } [ M2 ] 79,800円 (税込) 在庫数 18枚</a>
        </div>
      </li>
    </ul>
    """
    client = CardrushPokemonClient(
        FixtureHttpClient(
            {
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=110%2F80": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=110%2F080": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=%E3%83%A1%E3%82%AC%E3%83%AA%E3%82%B6%E3%83%BC%E3%83%89%E3%83%B3Xex": html,
                f"{CARDRUSH_PRODUCT_LIST_URL}?keyword=%E3%83%A1%E3%82%AC%E3%83%AA%E3%82%B6%E3%83%BC%E3%83%89%E3%83%B3X": html,
            }
        )
    )

    offers = client.lookup(
        TcgCardSpec(game="pokemon", title="メガリザードンXex", card_number="110/80", rarity="SAR", set_code="m2")
    )

    assert len(offers) == 1
    assert offers[0].price_jpy == 79800
    assert offers[0].attributes["card_number"] == "110/080"


def test_cardrush_temporarily_disables_after_repeated_transport_failures() -> None:
    CardrushPokemonClient.reset_temporary_disable()
    failing_http_client = FailingHttpClient()
    first_client = CardrushPokemonClient(failing_http_client)
    spec = TcgCardSpec(game="pokemon", title="リザードンex", card_number="349/190", rarity="SAR")

    assert first_client.lookup(spec) == []
    assert failing_http_client.calls == 1

    second_http_client = FailingHttpClient()
    second_client = CardrushPokemonClient(second_http_client)
    assert second_client.lookup(spec) == []
    assert second_http_client.calls == 0

    CardrushPokemonClient.reset_temporary_disable()


def test_magi_lookup_matches_ws_card() -> None:
    html = """
    <div class="product-list__box">
      <a href="/products/7001">ワンダーランドのセカイ 初音ミク TD PJS/S91-T51 ¥ 80 ~ 出品数 3</a>
    </div>
    """
    client = MagiProductClient(
        FixtureHttpClient(
            {
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=PJS%2FS91-T51": html,
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=%E3%83%AF%E3%83%B3%E3%83%80%E3%83%BC%E3%83%A9%E3%83%B3%E3%83%89%E3%81%AE%E3%82%BB%E3%82%AB%E3%82%A4+%E5%88%9D%E9%9F%B3%E3%83%9F%E3%82%AF": html,
            }
        )
    )

    offers = client.lookup(
        TcgCardSpec(game="ws", title="ワンダーランドのセカイ 初音ミク", card_number="PJS/S91-T51")
    )

    assert len(offers) == 1
    assert offers[0].source == "magi"
    assert offers[0].price_kind == "market"
    assert offers[0].price_jpy == 80
    assert offers[0].attributes["card_number"] == "PJS/S91-T51"
    assert offers[0].attributes["rarity"] == "TD"


def test_magi_lookup_matches_pokemon_card_number() -> None:
    html = """
    <div class="product-list__box">
      <a href="/products/8201">リザードンex SAR 349/190 ¥ 29,000 ~ 出品数 4</a>
    </div>
    """
    client = MagiProductClient(
        FixtureHttpClient(
            {
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=349%2F190": html,
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=%E3%83%AA%E3%82%B6%E3%83%BC%E3%83%89%E3%83%B3ex": html,
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=%E3%83%AA%E3%82%B6%E3%83%BC%E3%83%89%E3%83%B3": html,
                f"{MAGI_PRODUCT_SEARCH_URL}?forms_search_items%5Bkeyword%5D=SAR": html,
            }
        )
    )

    offers = client.lookup(
        TcgCardSpec(game="pokemon", title="リザードンex", card_number="349/190", rarity="SAR")
    )

    assert len(offers) == 1
    assert offers[0].price_jpy == 29000
    assert offers[0].attributes["card_number"] == "349/190"
    assert offers[0].attributes["rarity"] == "SAR"

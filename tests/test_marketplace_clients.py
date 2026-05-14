from __future__ import annotations

from urllib.parse import urlencode

from market_monitor.http import HttpClient
from tcg_tracker.cardrush import (
    CARDRUSH_PRODUCT_LIST_URL,
    CARDRUSH_YUGIOH_PRODUCT_LIST_URL,
    CardrushPokemonClient,
    CardrushYugiohClient,
)
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.magi import MAGI_PRODUCT_SEARCH_URL, MagiProductClient
from tcg_tracker.mercari_reference import MercariReferenceClient
from tcg_tracker.surugaya import SURUGAYA_SEARCH_URL, SurugayaClient
from tcg_tracker.search_terms import build_lookup_terms


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


def test_cardrush_yugioh_lookup_matches_exact_variant() -> None:
    CardrushYugiohClient.reset_temporary_disable()
    spec = TcgCardSpec(game="ygo", title="青眼の白龍", card_number="QCCP-JP001", rarity="ウルトラ")
    html = """
    <ul class="item_list">
      <li class="list_item_cell list_item_84978">
        <div class="item_data clearfix" data-product-id="84978">
          <a href="https://www.cardrush.jp/product/84978">
            <span class="goods_name">青眼の白龍【ウルトラ】{QCCP-JP001}《モンスター》</span>
          </a>
          <p><span class="figure">1,580円</span></p>
          <p class="stock">在庫数 34枚</p>
        </div>
      </li>
      <li class="list_item_cell list_item_45650">
        <div class="item_data clearfix" data-product-id="45650">
          <a href="https://www.cardrush.jp/product/45650">
            <span class="goods_name">青眼の白龍(初期)【シークレット】{-}《モンスター》</span>
          </a>
          <p><span class="figure">10,800,000円</span></p>
          <p class="stock">在庫数 1枚</p>
        </div>
      </li>
    </ul>
    """
    client = CardrushYugiohClient(
        FixtureHttpClient(
            {
                f"{CARDRUSH_YUGIOH_PRODUCT_LIST_URL}?{urlencode({'keyword': term})}": html
                for term in build_lookup_terms(spec)
            }
        )
    )

    offers = client.lookup(spec)

    assert len(offers) == 1
    assert offers[0].source == "cardrush_yugioh"
    assert offers[0].price_jpy == 1580
    assert offers[0].attributes["card_number"] == "QCCP-JP001"
    assert offers[0].attributes["set_code"] == "qccp"


def test_mercari_reference_matches_union_arena_padded_card_number() -> None:
    calls: list[str] = []

    def fake_search(query: str, **kwargs) -> list[dict[str, object]]:
        calls.append(query)
        return [
            {
                "item_id": "mua1",
                "title": "UAPR/EVA-1-071 綾波レイ SR ユニオンアリーナ",
                "price_jpy": 2200,
                "url": "https://jp.mercari.com/item/mua1",
                "thumbnail_url": "https://example.com/card.jpg",
            }
        ]

    client = MercariReferenceClient(search_fn=fake_search, max_results=3)

    offers = client.lookup(TcgCardSpec(game="union area", title="綾波レイ", card_number="UAPR/EVA-1-71"))

    assert offers
    assert calls[0] == "UAPR/EVA-1-71 綾波レイ"
    assert "UAPR/EVA-1-071 綾波レイ" in calls
    assert offers[0].source == "mercari"
    assert offers[0].attributes["card_number"] == "UAPR/EVA-1-071"
    assert offers[0].attributes["set_code"] == "uapr"


def test_surugaya_lookup_matches_union_arena_card_with_buylist_price() -> None:
    search_html = """
    <div class="item_box first_item">
      <p class="title">
        <a href="/product/detail/GU659612?branch_number=0080">UAPR/EVA-1-071[R]：(キラ)綾波 レイ（未開封）</a>
      </p>
      <p class="price_teika">中古： ￥8,317 税込</p>
    </div>
    """
    detail_html = """
    <h1>ユニオンアリーナ/R/キャラクター UAPR/EVA-1-071[R]：(キラ)綾波 レイ（未開封）</h1>
    <label data-label="中古&nbsp;未開封">
      <span class="text-price-detail price-buy">8,317円 (税込)</span>
    </label>
    <input type="hidden" name="amount_max" value="1" class="amount_max">
    <button type="button" class="btn_buy btn cart1">カートに入れる</button>
    <a href="https://www.suruga-ya.jp/kaitori/kaitori_detail/GU659612">
      <span>買取価格：</span><span class="text-red purchase-price">4,200円</span>
    </a>
    """
    client = SurugayaClient(
        FixtureHttpClient(
            {
                f"{SURUGAYA_SEARCH_URL}?search_word=UAPR%2FEVA-1-071": search_html,
                f"{SURUGAYA_SEARCH_URL}?search_word=UAPR%2FEVA-1-71": "",
                f"{SURUGAYA_SEARCH_URL}?search_word=UAPR+EVA+1+071": "",
                "https://www.suruga-ya.jp/product/detail/GU659612": detail_html,
            }
        )
    )

    offers = client.lookup(TcgCardSpec(game="union area", title="綾波レイ", card_number="UAPR/EVA-1-71"))

    assert len(offers) == 2
    assert {offer.price_kind for offer in offers} == {"ask", "bid"}
    assert any(offer.price_kind == "ask" and offer.price_jpy == 8317 for offer in offers)
    assert any(offer.price_kind == "bid" and offer.price_jpy == 4200 for offer in offers)
    assert {offer.attributes["card_number"] for offer in offers} == {"UAPR/EVA-1-071"}
    assert {offer.attributes["rarity"] for offer in offers} == {"R"}
    assert all((offer.score or 0) >= 40 for offer in offers)


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


def test_cardrush_lookup_matches_pokemon_sealed_box() -> None:
    CardrushPokemonClient.reset_temporary_disable()
    spec = TcgCardSpec(
        game="pokemon",
        title="強化拡張パック ポケモンカード151",
        item_kind="sealed_box",
        set_code="sv2a",
    )
    html = """
    <ul class="item_list">
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/46191">強化拡張パック『ポケモンカード151』(SV2a)【未開封BOX】{-} 70,800円 (税込) 在庫数 7点</a>
        </div>
      </li>
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/43870">ポケモンカード151 カードファイルセット フシギバナ・リザードン・カメックス【未開封BOX】{-} 9,980円 (税込) ×</a>
        </div>
      </li>
    </ul>
    """
    client = CardrushPokemonClient(
        FixtureHttpClient(
            {
                f"{CARDRUSH_PRODUCT_LIST_URL}?{urlencode({'keyword': term})}": html
                for term in build_lookup_terms(spec)
            }
        )
    )

    offers = client.lookup(spec)

    assert len(offers) == 1
    assert offers[0].price_jpy == 70800
    assert offers[0].attributes["product_kind"] == "sealed_box"
    assert offers[0].attributes["set_code"] == "sv2a"
    assert "70,800円" not in offers[0].title


def test_magi_lookup_matches_pokemon_sealed_box() -> None:
    spec = TcgCardSpec(
        game="pokemon",
        title="ハイクラスパック MEGAドリームex",
        item_kind="sealed_box",
    )
    html = """
    <div class="product-list__box">
      <a href="/products/2915333">ハイクラスパック MEGAドリームex 未開封BOX ¥ 17,800 ~ 出品数 36</a>
    </div>
    <div class="product-list__box">
      <a href="/products/447483">ゲンガーEX TD 010/049 ¥ 42,800 ~ 出品数 1</a>
    </div>
    """
    client = MagiProductClient(
        FixtureHttpClient(
            {
                f"{MAGI_PRODUCT_SEARCH_URL}?{urlencode({'forms_search_items[keyword]': term})}": html
                for term in build_lookup_terms(spec)
            }
        )
    )

    offers = client.lookup(spec)

    assert len(offers) == 1
    assert offers[0].price_jpy == 17800
    assert offers[0].attributes["product_kind"] == "sealed_box"
    assert offers[0].attributes["card_number"] == ""

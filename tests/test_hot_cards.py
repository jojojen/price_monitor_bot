from __future__ import annotations

from pathlib import Path

from market_monitor.http import HttpClient
from tcg_tracker.hot_cards import (
    HotCardBuySignal,
    HotCardReference,
    HotCardSocialSignal,
    YUYUTEI_WS_TOP_URL,
    TcgHotCardService,
    _ParsedHotItem,
    _parse_cardrush_text,
    _parse_magi_text,
    _parse_snkrdunk_text,
    _parse_yahoo_realtime_signal,
)
from tcg_tracker.catalog import TcgCardSpec

FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FixtureHttpClient(HttpClient):
    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        super().__init__(user_agent="fixture")

    def get_text(self, url: str, *, params=None, encoding="utf-8", headers=None) -> str:  # type: ignore[override]
        if url in self.responses:
            return self.responses[url]
        raise AssertionError(f"unexpected url: {url}")


def _buy_signal(
    *,
    bid: int | None,
    ask: int | None,
    previous_bid: int | None = None,
    signal_label: str | None = None,
) -> HotCardBuySignal:
    ratio = None if bid is None or ask is None or ask <= 0 else round(min(1.0, bid / ask), 4)
    support_ratio = 0.0
    if bid is not None:
        support_ratio += 0.35
    if ratio is not None:
        support_ratio += ratio * 0.50
    if bid is not None and ask is not None:
        support_ratio += 0.15
    momentum_boost = 0.0
    if signal_label == "priceup":
        if previous_bid is None or previous_bid <= 0 or bid is None or bid <= previous_bid:
            momentum_boost = 0.03
        else:
            momentum_boost = min(0.06, 0.03 + (((bid - previous_bid) / previous_bid) * 0.03))
    support_ratio += momentum_boost
    return HotCardBuySignal(
        best_ask_jpy=ask,
        best_bid_jpy=bid,
        previous_bid_jpy=previous_bid,
        ask_count=0 if ask is None else 1,
        bid_count=0 if bid is None else 1,
        bid_ask_ratio=ratio,
        buy_support_ratio=round(min(1.0, support_ratio), 4),
        momentum_boost_ratio=round(momentum_boost, 4),
        buy_signal_label=signal_label,
        references=(HotCardReference(label="stub", url="https://example.com/buy-signal"),),
    )


class StubHotCardService(TcgHotCardService):
    def __init__(
        self,
        *,
        buy_signals: dict[str, HotCardBuySignal] | None = None,
        social_signals: dict[str, HotCardSocialSignal] | None = None,
    ) -> None:
        super().__init__()
        self._buy_signals = buy_signals or {}
        self._social_signals = social_signals or {}

    def _lookup_buy_signal(  # type: ignore[override]
        self,
        spec: TcgCardSpec,
        *,
        reference_ask_jpy: int | None = None,
    ) -> HotCardBuySignal | None:
        return self._buy_signals.get(spec.card_number or spec.title)

    def _lookup_social_signal(self, *, game: str, item: _ParsedHotItem) -> HotCardSocialSignal | None:  # type: ignore[override]
        return self._social_signals.get(item.title)


def test_parse_cardrush_text_extracts_core_fields() -> None:
    parsed = _parse_cardrush_text(
        "〔状態A-〕ピカチュウex【SAR】{234/193} [ [状態A-]M2a ] 59,800円 (税込) 在庫数 5枚",
        detail_url="https://www.cardrush-pokemon.jp/product/123",
        board_url="https://www.cardrush-pokemon.jp/product-group/22?sort=rank&num=100",
        thumbnail_url="https://www.cardrush-pokemon.jp/image.jpg",
    )

    assert parsed is not None
    assert parsed.title == "ピカチュウex"
    assert parsed.card_number == "234/193"
    assert parsed.rarity == "SAR"
    assert parsed.set_code == "m2a"
    assert parsed.price_jpy == 59800
    assert parsed.listing_count == 5
    assert parsed.condition == "状態A-"
    assert parsed.thumbnail_url == "https://www.cardrush-pokemon.jp/image.jpg"


def test_parse_magi_text_extracts_ws_card_fields() -> None:
    parsed = _parse_magi_text(
        "“夏の思い出”蒼(サイン入り) SP SMP/W60-051SP ¥ 22,800 ~ 出品数 1",
        detail_url="https://magi.camp/products/123",
        board_url="https://magi.camp/series/7/products",
        thumbnail_url="https://magi.camp/image.jpg",
    )

    assert parsed is not None
    assert parsed.title == "“夏の思い出”蒼(サイン入り)"
    assert parsed.rarity == "SP"
    assert parsed.card_number == "SMP/W60-051SP"
    assert parsed.set_code == "smp"
    assert parsed.price_jpy == 22800
    assert parsed.listing_count == 1
    assert parsed.is_graded is False
    assert parsed.thumbnail_url == "https://magi.camp/image.jpg"


def test_parse_magi_text_handles_codes_with_letter_prefix_after_hyphen() -> None:
    parsed = _parse_magi_text(
        "〖PSA10〗舞台の上で 天音かなた(サイン入り) SP HOL/W91-T108SP - 出品数 0",
        detail_url="https://magi.camp/products/456",
        board_url="https://magi.camp/series/7/products",
    )

    assert parsed is not None
    assert parsed.title == "舞台の上で 天音かなた(サイン入り)"
    assert parsed.rarity == "SP"
    assert parsed.card_number == "HOL/W91-T108SP"
    assert parsed.listing_count == 0
    assert parsed.is_graded is True


def test_parse_snkrdunk_text_extracts_core_fields() -> None:
    parsed = _parse_snkrdunk_text(
        "???圯x SAR [M2a 234/193](?舀?喳?摨舫??MEGA?脩?ex)",
        detail_url="https://snkrdunk.com/apparels/730956?slide=right",
        board_url="https://snkrdunk.com/articles/31649/",
        thumbnail_url="https://cdn.snkrdunk.com/example.webp",
        note="trend",
        source_label="SNKRDUNK monthly trades",
        source_rank=5,
        demand_ratio=0.92,
    )

    assert parsed is not None
    assert parsed.title == "???圯x"
    assert parsed.rarity == "SAR"
    assert parsed.card_number == "234/193"
    assert parsed.set_code == "m2a"
    assert parsed.source_label == "SNKRDUNK monthly trades"
    assert parsed.source_rank == 5
    assert parsed.demand_ratio == 0.92


def test_parse_snkrdunk_text_handles_ws_code_with_redundant_rarity_token() -> None:
    parsed = _parse_snkrdunk_text(
        'Steamboat Mickey OR[Dds/S104-100OR OR](Booster Pack Disney100)',
        detail_url="https://snkrdunk.com/apparels/120130?slide=right",
        board_url="https://snkrdunk.com/articles/26509/",
        note="annual sales",
        source_label="SNKRDUNK annual sales ranking",
        source_rank=1,
        demand_ratio=0.8,
    )

    assert parsed is not None
    assert parsed.title == "Steamboat Mickey"
    assert parsed.rarity == "OR"
    assert parsed.card_number == "Dds/S104-100OR"
    assert parsed.set_code == "dds"


def test_parse_snkrdunk_article_apparel_items_extracts_ws_singles() -> None:
    html = """
    <article-content
      :apparels='[
        {
          "id": 778223,
          "localizedName": "Weiss Schwarz Booster Pack Summer Pockets Box",
          "displayPrice": "¥13,000",
          "totalListingCount": 1,
          "primaryMedia": {"imageUrl": "https://cdn.example.com/box.webp"}
        },
        {
          "id": 798318,
          "localizedName": "\\"A Girl Gazing at the Sea\\" Shiroha SSP [SMP/W137-038SSP](Booster Pack Summer Pockets)",
          "displayPrice": "¥80,000",
          "totalListingCount": 3,
          "primaryMedia": {"imageUrl": "https://cdn.example.com/shiroha.webp"}
        }
      ]'
    ></article-content>
    """

    service = TcgHotCardService()
    items = service._parse_snkrdunk_article_apparel_items(  # type: ignore[attr-defined]
        html,
        board_url="https://snkrdunk.com/articles/31956/",
        source_label="SNKRDUNK Summer Pockets initial market",
        source_weight=0.48,
        note="initial market",
    )

    assert len(items) == 1
    assert items[0].title == '"A Girl Gazing at the Sea" Shiroha'
    assert items[0].card_number == "SMP/W137-038SSP"
    assert items[0].rarity == "SSP"
    assert items[0].price_jpy == 80000
    assert items[0].listing_count == 3
    assert items[0].thumbnail_url == "https://cdn.example.com/shiroha.webp"
    assert items[0].source_rank == 1
    assert items[0].demand_ratio > 0


def test_parse_snkrdunk_heading_ranking_items_skips_box_like_entries() -> None:
    html = """
    <div class="article-content">
      <h3>■第1位 Steamboat Mickey OR[Dds/S104-100OR OR](Booster Pack Disney100)</h3>
      <a href="/apparels/120130?ref=articles_hobby&amp;slide=right">
        <img src="https://cdn.example.com/mickey.webp" />
        <div>¥2,000,000〜</div>
      </a>
      <h3>■第2位 Weiss Schwarz Booster Pack Disney100 Box</h3>
      <a href="/apparels/122977?ref=articles_hobby&amp;slide=right">
        <img src="https://cdn.example.com/box.webp" />
        <div>¥45,000〜</div>
      </a>
    </div>
    """

    service = TcgHotCardService()
    items = service._parse_snkrdunk_heading_ranking_items(  # type: ignore[attr-defined]
        html,
        board_url="https://snkrdunk.com/articles/26509/",
        source_label="SNKRDUNK annual sales ranking",
        source_weight=0.54,
        note="annual sales",
    )

    assert len(items) == 1
    assert items[0].title == "Steamboat Mickey"
    assert items[0].card_number == "Dds/S104-100OR"
    assert items[0].price_jpy == 2000000
    assert items[0].source_rank == 1


def test_hot_card_service_merges_duplicate_variants() -> None:
    service = StubHotCardService(
        buy_signals={
            "234/193": _buy_signal(bid=48000, ask=59800),
            "237/193": _buy_signal(bid=20000, ask=34800),
        }
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="pokemon",
        parsed_items=[
            _parse_cardrush_text(
                "〔状態A-〕ピカチュウex【SAR】{234/193} [ [状態A-]M2a ] 59,800円 (税込) 在庫数 5枚",
                detail_url="https://www.cardrush-pokemon.jp/product/123",
                board_url="https://www.cardrush-pokemon.jp/product-group/22?sort=rank&num=100",
            ),
            _parse_cardrush_text(
                "ピカチュウex【SAR】{234/193} [ M2a ] 61,800円 (税込) 在庫数 2枚",
                detail_url="https://www.cardrush-pokemon.jp/product/124",
                board_url="https://www.cardrush-pokemon.jp/product-group/22?sort=rank&num=100",
            ),
            _parse_cardrush_text(
                "ロケット団のミュウツーex【SAR】{237/193} [ M2a ] 34,800円 (税込) 在庫数 1枚",
                detail_url="https://www.cardrush-pokemon.jp/product/125",
                board_url="https://www.cardrush-pokemon.jp/product-group/22?sort=rank&num=100",
            ),
        ],
        limit=10,
    )

    assert len(entries) == 2
    assert entries[0].title == "ピカチュウex"
    assert entries[0].listing_count == 7
    assert entries[0].best_bid_jpy == 48000
    assert entries[0].references[1].url == "https://www.cardrush-pokemon.jp/product/124"


def test_hot_card_service_prioritizes_buy_support_over_source_rank() -> None:
    service = StubHotCardService(
        buy_signals={
            "AAA/W11-001SP": _buy_signal(bid=5000, ask=30000),
            "AAA/W11-002SP": _buy_signal(bid=25000, ask=28000),
        }
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="ws",
        parsed_items=[
            _ParsedHotItem(
                title="rank_only_card",
                price_jpy=30000,
                thumbnail_url=None,
                card_number="AAA/W11-001SP",
                rarity="SP",
                set_code="aaa",
                listing_count=0,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/rank-only",
                board_url="https://example.com/board",
                note="rank only",
            ),
            _ParsedHotItem(
                title="active_card",
                price_jpy=28000,
                thumbnail_url=None,
                card_number="AAA/W11-002SP",
                rarity="SP",
                set_code="aaa",
                listing_count=3,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/active",
                board_url="https://example.com/board",
                note="active",
            ),
        ],
        limit=10,
    )

    assert len(entries) == 2
    assert entries[0].title == "active_card"
    assert entries[0].hot_score > 0


def test_hot_card_service_prefers_raw_copies_when_depth_is_equal() -> None:
    service = StubHotCardService(
        buy_signals={
            "BBB/W22-001SSP": _buy_signal(bid=32000, ask=40000),
            "BBB/W22-002SSP": _buy_signal(bid=32000, ask=40000),
        }
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="ws",
        parsed_items=[
            _ParsedHotItem(
                title="graded_card",
                price_jpy=50000,
                thumbnail_url=None,
                card_number="BBB/W22-001SSP",
                rarity="SSP",
                set_code="bbb",
                listing_count=2,
                is_graded=True,
                condition=None,
                detail_url="https://example.com/graded",
                board_url="https://example.com/board",
                note="graded",
            ),
            _ParsedHotItem(
                title="raw_card",
                price_jpy=45000,
                thumbnail_url=None,
                card_number="BBB/W22-002SSP",
                rarity="SSP",
                set_code="bbb",
                listing_count=2,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/raw",
                board_url="https://example.com/board",
                note="raw",
            ),
        ],
        limit=10,
    )

    assert entries[0].title == "raw_card"
    assert entries[1].title == "graded_card"
    assert entries[0].hot_score > entries[1].hot_score


def test_hot_card_service_uses_attention_only_as_secondary_signal() -> None:
    service = StubHotCardService(
        buy_signals={
            "001/001": _buy_signal(bid=12000, ask=15000),
            "002/001": _buy_signal(bid=7000, ask=12000),
        },
        social_signals={
            "attention_card": HotCardSocialSignal(
                query="attention card",
                search_url="https://example.com/social",
                matched_post_count=8,
                engagement_count=2200,
                score_ratio=0.95,
            ),
        },
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="pokemon",
        parsed_items=[
            _ParsedHotItem(
                title="deep_card",
                price_jpy=15000,
                thumbnail_url=None,
                card_number="001/001",
                rarity="SAR",
                set_code="aaa",
                listing_count=6,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/deep",
                board_url="https://example.com/board",
                note="deep",
            ),
            _ParsedHotItem(
                title="attention_card",
                price_jpy=12000,
                thumbnail_url=None,
                card_number="002/001",
                rarity="SAR",
                set_code="aaa",
                listing_count=2,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/attention",
                board_url="https://example.com/board",
                note="attention",
            ),
        ],
        limit=10,
    )

    assert entries[0].title == "deep_card"
    assert entries[1].title == "attention_card"
    assert entries[1].attention_score > entries[0].attention_score


def test_hot_card_service_boosts_explicit_store_side_buy_up_signal() -> None:
    service = StubHotCardService(
        buy_signals={
            "AAA/001": _buy_signal(bid=25000, ask=30000, previous_bid=12000, signal_label="priceup"),
            "BBB/001": _buy_signal(bid=24000, ask=29000),
        }
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="pokemon",
        parsed_items=[
            _ParsedHotItem(
                title="buy_up_card",
                price_jpy=30000,
                thumbnail_url=None,
                card_number="AAA/001",
                rarity="SAR",
                set_code="aaa",
                listing_count=1,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/buy-up",
                board_url="https://example.com/board",
                note="buy-up",
            ),
            _ParsedHotItem(
                title="plain_card",
                price_jpy=29000,
                thumbnail_url=None,
                card_number="BBB/001",
                rarity="SAR",
                set_code="aaa",
                listing_count=9,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/plain",
                board_url="https://example.com/board",
                note="plain",
            ),
        ],
        limit=10,
    )

    assert entries[0].title == "buy_up_card"
    assert entries[0].buy_signal_label == "priceup"
    assert entries[0].momentum_boost_score > 0
    assert any("Store-side buy pressure signal" in note for note in entries[0].notes)


def test_hot_card_service_prioritizes_recent_trade_activity_signal() -> None:
    service = StubHotCardService(
        buy_signals={
            "AAA/001": _buy_signal(bid=18000, ask=28000),
            "BBB/001": _buy_signal(bid=34000, ask=35000),
        }
    )
    entries = service._build_ranked_entries(  # type: ignore[attr-defined]
        game="pokemon",
        parsed_items=[
            _ParsedHotItem(
                title="iconic_trade_card",
                price_jpy=28000,
                thumbnail_url=None,
                card_number="AAA/001",
                rarity="SAR",
                set_code="aaa",
                listing_count=1,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/iconic",
                board_url="https://example.com/monthly-trades",
                note="monthly trades",
                source_label="SNKRDUNK monthly trades",
                source_rank=4,
                demand_ratio=0.95,
            ),
            _ParsedHotItem(
                title="obscure_bid_card",
                price_jpy=35000,
                thumbnail_url=None,
                card_number="BBB/001",
                rarity="SAR",
                set_code="bbb",
                listing_count=1,
                is_graded=False,
                condition=None,
                detail_url="https://example.com/obscure",
                board_url="https://example.com/cardrush",
                note="cardrush",
                source_label="Cardrush category rank",
                source_rank=1,
                demand_ratio=0.08,
            ),
        ],
        limit=10,
    )

    assert entries[0].title == "iconic_trade_card"
    assert entries[0].hot_score > entries[1].hot_score
    assert any("Recent market activity signal" in note for note in entries[0].notes)


def test_resolve_lookup_spec_uses_hot_card_metadata_for_precise_variant() -> None:
    class StubHotCardService(TcgHotCardService):
        def _load_source_items(self, game: str) -> list[_ParsedHotItem]:  # type: ignore[override]
            return [
                _ParsedHotItem(
                    title="メガシビルドンex",
                    price_jpy=780,
                    thumbnail_url=None,
                    card_number="225/193",
                    rarity="MA",
                    set_code="m2a",
                    listing_count=535,
                    is_graded=False,
                    condition=None,
                    detail_url="https://example.com/225",
                    board_url="https://example.com/board",
                    note="stub",
                ),
                _ParsedHotItem(
                    title="メガシビルドンex",
                    price_jpy=1280,
                    thumbnail_url=None,
                    card_number="235/193",
                    rarity="SAR",
                    set_code="m2a",
                    listing_count=140,
                    is_graded=False,
                    condition=None,
                    detail_url="https://example.com/235",
                    board_url="https://example.com/board",
                    note="stub",
                ),
            ]

    service = StubHotCardService()
    resolved = service.resolve_lookup_spec(
        TcgCardSpec(game="pokemon", title="メガシビルドン", rarity="SAR"),
    )

    assert resolved is not None
    assert resolved.title == "メガシビルドンex"
    assert resolved.card_number == "235/193"
    assert resolved.rarity == "SAR"
    assert resolved.set_code == "m2a"


def test_parse_cardrush_pokemon_items_extracts_thumbnail_url() -> None:
    html = """
    <ul class="item_list">
      <li class="list_item_cell">
        <div class="item_data">
          <a href="/product/777">
            <div class="global_photo">
              <img
                src="https://cdn.example.com/thumb-160.jpg"
                data-x2="https://cdn.example.com/thumb-320.jpg"
                alt="メガシビルドンex【SAR】{235/193}"
              />
            </div>
            <p class="item_name">
              <span class="goods_name">メガシビルドンex【SAR】{235/193}</span>
            </p>
            <div class="item_info">
              <p class="selling_price"><span class="figure">12,800円</span></p>
              <p class="stock">在庫数 14枚</p>
            </div>
          </a>
        </div>
      </li>
    </ul>
    """

    service = TcgHotCardService()
    items = service._parse_cardrush_pokemon_items(html)  # type: ignore[attr-defined]

    assert len(items) == 1
    assert items[0].thumbnail_url == "https://cdn.example.com/thumb-320.jpg"
    assert items[0].detail_url == "https://www.cardrush-pokemon.jp/product/777"


def test_parse_magi_ws_items_extracts_thumbnail_url() -> None:
    html = """
    <div class="product-list__box">
      <a href="/products/999">
        <figure class="product-list__thumbnail">
          <img
            src="/assets/fallback.jpg"
            data-src="https://magi.camp/cdn/ws-thumb.jpg"
            alt="初音ミク SSP"
          />
        </figure>
        <div class="product-list__product-name">ワンダーランドのセカイ 初音ミク SSP PJS/S91-T51</div>
        <ul class="product-list__price-box">
          <li class="product-list__price-box-price">¥ 1,280 ~</li>
        </ul>
        <div class="product-list__item-count">出品数 4</div>
      </a>
    </div>
    """

    service = TcgHotCardService()
    items = service._parse_magi_ws_items(html)  # type: ignore[attr-defined]

    assert len(items) == 1
    assert items[0].thumbnail_url == "https://magi.camp/cdn/ws-thumb.jpg"
    assert items[0].detail_url == "https://magi.camp/products/999"


def test_parse_yuyutei_ws_carousel_items_extracts_featured_cards() -> None:
    html = (FIXTURE_DIR / "yuyutei_ws_sell_search.html").read_text(encoding="utf-8")

    service = TcgHotCardService()
    items = service._parse_yuyutei_ws_carousel_items(  # type: ignore[attr-defined]
        html,
        board_url=YUYUTEI_WS_TOP_URL,
        carousel_id="recommendedItemList",
        source_label="Yuyutei featured singles",
        source_weight=0.34,
        note="featured",
        minimum_price_jpy=5000,
    )

    assert items
    assert items[0].card_number == "PJS/S125-083EX"
    assert items[0].rarity == "SEC"
    assert items[0].price_jpy == 19800
    assert items[0].detail_url == "https://yuyu-tei.jp/sell/ws/card/pjs3.0/10263"
    assert items[0].board_url == YUYUTEI_WS_TOP_URL
    assert items[0].source_label == "Yuyutei featured singles"


def test_load_ws_board_items_uses_live_sources_instead_of_legacy_articles() -> None:
    service = TcgHotCardService(
        http_client=FixtureHttpClient(
            {
                YUYUTEI_WS_TOP_URL: (FIXTURE_DIR / "yuyutei_ws_sell_search.html").read_text(encoding="utf-8"),
            }
        )
    )

    items, methodology = service._load_ws_board_items()  # type: ignore[attr-defined]

    assert items
    assert all(item.board_url == YUYUTEI_WS_TOP_URL for item in items)
    assert any(item.source_label == "Yuyutei featured singles" for item in items)
    assert any(item.source_label == "Yuyutei latest-release spotlight" for item in items)
    assert "Static SNKRDUNK article rankings are no longer used as primary WS market-activity inputs" in methodology


def test_parse_yahoo_realtime_signal_extracts_matched_posts_and_engagement() -> None:
    html = """
    <div id="sr">
      <div>
        <div class="Tweet_TweetContainer__abc">
          <a
            class="Tweet_TweetMenu__abc"
            data-cl-params="_cl_link:menu;reply:5;retweet:12;like:140;quote:3"
          >menu</a>
          <p class="Tweet_body__abc">メガシビルドンex ポケカ SAR を買った。価格の動きが気になる。</p>
        </div>
        <div class="Tweet_TweetContainer__abc">
          <a
            class="Tweet_TweetMenu__abc"
            data-cl-params="_cl_link:menu;reply:1;retweet:8;like:52;quote:0"
          >menu</a>
          <p class="Tweet_body__abc">メガシビルドンex SAR が高流動性で追いやすい。</p>
        </div>
      </div>
    </div>
    """

    signal = _parse_yahoo_realtime_signal(
        html=html,
        query="メガシビルドンex ポケカ SAR",
        search_url="https://search.yahoo.co.jp/realtime/search?p=%E3%83%A1%E3%82%AC%E3%82%B7%E3%83%93%E3%83%AB%E3%83%89%E3%83%B3ex+%E3%83%9D%E3%82%B1%E3%82%AB+SAR",
    )

    assert signal is not None
    assert signal.matched_post_count == 2
    assert signal.engagement_count == 221
    assert signal.score_ratio > 0

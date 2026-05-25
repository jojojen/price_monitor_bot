from __future__ import annotations

from datetime import datetime, timezone

from market_monitor.models import MarketOffer
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.service import TcgPriceService


class StubYuyuteiClient:
    def __init__(self, offers: list[MarketOffer]) -> None:
        self.offers = offers

    def lookup(self, spec: TcgCardSpec) -> list[MarketOffer]:
        return list(self.offers)


class StubLookupClient:
    def __init__(self, offers: list[MarketOffer]) -> None:
        self.offers = offers

    def lookup(self, spec: TcgCardSpec) -> list[MarketOffer]:
        return list(self.offers)


class TimeoutLookupClient:
    def lookup(self, spec: TcgCardSpec) -> list[MarketOffer]:
        raise TimeoutError("read timed out")


def test_name_only_lookup_is_marked_ambiguous_and_skips_fair_value(tmp_path) -> None:
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        yuyutei_client=StubYuyuteiClient(
            [
                MarketOffer(
                    source="yuyutei",
                    listing_id="sv08-ask",
                    url="https://example.com/sv08",
                    title="ピカチュウex",
                    price_jpy=99800,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "132/106", "rarity": "SAR", "version_code": "sv08"},
                ),
                MarketOffer(
                    source="yuyutei",
                    listing_id="m02a-ask",
                    url="https://example.com/m02a",
                    title="ピカチュウex",
                    price_jpy=59900,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "234/193", "rarity": "SAR", "version_code": "m02a"},
                ),
            ]
        ),
    )

    result = service.lookup(TcgCardSpec(game="pokemon", title="ピカチュウex"), persist=False)

    assert len(result.offers) == 2
    assert result.fair_value is None
    assert result.notes
    assert "Multiple variants matched" in result.notes[0]


def test_precise_lookup_keeps_fair_value(tmp_path) -> None:
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        yuyutei_client=StubYuyuteiClient(
            [
                MarketOffer(
                    source="yuyutei",
                    listing_id="sv08-ask",
                    url="https://example.com/sv08-ask",
                    title="ピカチュウex",
                    price_jpy=99800,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "132/106", "rarity": "SAR", "version_code": "sv08"},
                ),
                MarketOffer(
                    source="yuyutei",
                    listing_id="sv08-bid",
                    url="https://example.com/sv08-bid",
                    title="ピカチュウex",
                    price_jpy=80000,
                    price_kind="bid",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "132/106", "rarity": "SAR", "version_code": "sv08"},
                ),
            ]
        ),
    )

    result = service.lookup(
        TcgCardSpec(game="pokemon", title="ピカチュウex", card_number="132/106", rarity="SAR", set_code="sv08"),
        persist=False,
    )

    assert len(result.offers) == 2
    assert result.fair_value is not None
    assert result.fair_value.amount_jpy == 99900
    assert result.notes == ()


def test_lookup_aggregates_secondary_reference_sources(tmp_path) -> None:
    shared_timestamp = datetime.now(timezone.utc)
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        reference_clients=(
            StubLookupClient(
                [
                    MarketOffer(
                        source="yuyutei",
                        listing_id="sv4a-ask",
                        url="https://example.com/yuyutei",
                        title="リザードンex",
                        price_jpy=59800,
                        price_kind="ask",
                        captured_at=shared_timestamp,
                        source_category="specialty_store",
                        attributes={"card_number": "349/190", "rarity": "SAR", "version_code": "sv4a"},
                    )
                ]
            ),
            StubLookupClient(
                [
                    MarketOffer(
                        source="cardrush_pokemon",
                        listing_id="cardrush-ask",
                        url="https://example.com/cardrush",
                        title="リザードンex",
                        price_jpy=57200,
                        price_kind="ask",
                        captured_at=shared_timestamp,
                        source_category="specialty_store",
                        attributes={"card_number": "349/190", "rarity": "SAR", "version_code": "sv4a"},
                    )
                ]
            ),
            StubLookupClient(
                [
                    MarketOffer(
                        source="magi",
                        listing_id="magi-market",
                        url="https://example.com/magi",
                        title="リザードンex",
                        price_jpy=54000,
                        price_kind="market",
                        captured_at=shared_timestamp,
                        source_category="marketplace",
                        attributes={"card_number": "349/190", "rarity": "SAR"},
                    )
                ]
            ),
        ),
    )

    result = service.lookup(
        TcgCardSpec(game="pokemon", title="リザードンex", card_number="349/190", rarity="SAR", set_code="sv4a"),
        persist=False,
    )

    assert len(result.offers) == 3
    assert {offer.source for offer in result.offers} == {"yuyutei", "cardrush_pokemon", "magi"}
    assert result.fair_value is not None
    assert result.fair_value.sample_count == 3


def test_lookup_continues_when_one_source_times_out(tmp_path, caplog) -> None:
    shared_timestamp = datetime.now(timezone.utc)
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        reference_clients=(
            TimeoutLookupClient(),
            StubLookupClient(
                [
                    MarketOffer(
                        source="cardrush_pokemon",
                        listing_id="cardrush-ask",
                        url="https://example.com/cardrush",
                        title="メガルカリオex",
                        price_jpy=65800,
                        price_kind="ask",
                        captured_at=shared_timestamp,
                        source_category="specialty_store",
                        attributes={"card_number": "092/063", "rarity": "MUR", "version_code": "m1l"},
                    )
                ]
            ),
        ),
    )

    with caplog.at_level("WARNING"):
        result = service.lookup(
            TcgCardSpec(game="pokemon", title="メガルカリオex", card_number="092/063", rarity="MUR", set_code="m1l"),
            persist=False,
        )

    assert len(result.offers) == 1
    assert result.offers[0].source == "cardrush_pokemon"
    assert "Source client timed out" in caplog.text
    assert "Traceback" not in caplog.text


def test_sealed_box_lookup_drops_starter_sets_and_low_priced_listings(tmp_path) -> None:
    """Regression for MEGA アビスアイ: the bot used to return ¥1,930 by
    treating cardrush single cards / starter sets as sealed boxes. The
    new pipeline must drop those and surface only the real booster /
    premium-trainer / hi-class boxes."""
    now = datetime.now(timezone.utc)
    sealed_attrs = {"product_kind": "sealed_box"}
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        reference_clients=(
            StubLookupClient(
                [
                    # Cardrush returns 2 real premium-trainer-boxes (¥7,980 / ¥8,480)
                    MarketOffer(
                        source="cardrush_pokemon", listing_id="69744",
                        url="https://www.cardrush-pokemon.jp/product/69744",
                        title="プレミアムトレーナーボックス MEGA", price_jpy=8480,
                        price_kind="ask", captured_at=now,
                        source_category="specialty_store",
                        attributes=sealed_attrs, score=42.0,
                    ),
                    MarketOffer(
                        source="cardrush_pokemon", listing_id="71062",
                        url="https://www.cardrush-pokemon.jp/product/71062",
                        title="プレミアムトレーナーボックス MEGA", price_jpy=7980,
                        price_kind="ask", captured_at=now,
                        source_category="specialty_store",
                        attributes=sealed_attrs, score=42.0,
                    ),
                    # And 2 starter sets that were the source of the bug (not tagged
                    # as sealed_box anymore by the parser)
                    MarketOffer(
                        source="cardrush_pokemon", listing_id="72035",
                        url="https://www.cardrush-pokemon.jp/product/72035",
                        title="スターターセット MEGA メガディアンシーex",
                        price_jpy=1880, price_kind="ask", captured_at=now,
                        source_category="specialty_store",
                        attributes={}, score=12.0,
                    ),
                ]
            ),
            StubLookupClient(
                [
                    # Magi has the real expansion boxes (mirrors actual data from
                    # the bot's 2026-05-25 MEGA アビスアイ lookup).
                    MarketOffer(
                        source="magi", listing_id="2915333",
                        url="https://magi.camp/products/2915333",
                        title="ハイクラスパック MEGAドリームex 未開封BOX",
                        price_jpy=16800, price_kind="market", captured_at=now,
                        source_category="marketplace",
                        attributes=sealed_attrs, score=52.0,
                    ),
                    MarketOffer(
                        source="magi", listing_id="2781611",
                        url="https://magi.camp/products/2781611",
                        title="MEGA 拡張パック メガシンフォニア 未開封BOX",
                        price_jpy=22593, price_kind="market", captured_at=now,
                        source_category="marketplace",
                        attributes=sealed_attrs, score=52.0,
                    ),
                    MarketOffer(
                        source="magi", listing_id="2781610",
                        url="https://magi.camp/products/2781610",
                        title="MEGA 拡張パック メガブレイブ 未開封BOX",
                        price_jpy=26500, price_kind="market", captured_at=now,
                        source_category="marketplace",
                        attributes=sealed_attrs, score=52.0,
                    ),
                ]
            ),
        ),
    )

    result = service.lookup(
        TcgCardSpec(
            game="pokemon", title="ポケモンカードゲーム MEGA アビスアイ",
            item_kind="sealed_box",
        ),
        persist=False,
    )

    # Starter set is gone; only real sealed boxes remain.
    sources = {(o.source, o.listing_id) for o in result.offers}
    assert ("cardrush_pokemon", "72035") not in sources
    assert all(offer.price_jpy >= 4_000 for offer in result.offers)
    # Fair value should reflect the real boxes, comfortably above ¥10,000.
    assert result.fair_value is not None
    assert result.fair_value.amount_jpy >= 10_000

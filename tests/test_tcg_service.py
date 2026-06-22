from __future__ import annotations

import time
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


class SlowStubClient:
    def __init__(self, source: str, delay_s: float = 0.1) -> None:
        self.source = source
        self.delay_s = delay_s

    def lookup(self, spec: TcgCardSpec) -> list[MarketOffer]:
        time.sleep(self.delay_s)
        return [
            MarketOffer(
                source=self.source,
                listing_id=f"{self.source}-1",
                url=f"https://example.com/{self.source}",
                title="テスト",
                price_jpy=10000,
                price_kind="market",
                captured_at=datetime.now(timezone.utc),
                source_category="marketplace",
                attributes={"card_number": "001/100", "rarity": "SR", "version_code": "sv01"},
            )
        ]


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
    assert "Source client transient failure" in caplog.text
    assert "Traceback" not in caplog.text


def test_lookup_fans_out_reference_clients_in_parallel(tmp_path) -> None:
    """Three 0.1s stubs must complete in well under 0.3s once the fan-out
    runs concurrently. If clients ran serially the total would be ~0.3s."""
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        reference_clients=(
            SlowStubClient("a", 0.1),
            SlowStubClient("b", 0.1),
            SlowStubClient("c", 0.1),
        ),
    )

    start = time.perf_counter()
    result = service.lookup(
        TcgCardSpec(
            game="pokemon",
            title="テスト",
            card_number="001/100",
            rarity="SR",
            set_code="sv01",
        ),
        persist=False,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.25, f"clients appear to run serially, elapsed={elapsed:.3f}s"
    assert {offer.source for offer in result.offers} == {"a", "b", "c"}


def test_lookup_notes_include_learned_reference_sites(tmp_path) -> None:
    """Pre-populated domain_trust rows (with real votes) must surface as a
    '📚 社群指名可參考來源' line in the lookup notes."""
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        reference_clients=(
            StubLookupClient(
                [
                    MarketOffer(
                        source="yuyutei", listing_id="abc",
                        url="https://example.com/abc", title="リザードンex",
                        price_jpy=59800, price_kind="ask",
                        captured_at=datetime.now(timezone.utc),
                        source_category="specialty_store",
                        attributes={"card_number": "349/190", "rarity": "SAR", "version_code": "sv4a"},
                    )
                ]
            ),
        ),
    )

    # Inject two real votes for distinct domains, both in the same game/kind
    service.database.bump_domain_trust(
        game="pokemon", item_kind="card", domain="yuyu-tei.jp", success=True,
    )
    service.database.bump_domain_trust(
        game="pokemon", item_kind="card", domain="cardrush.jp", success=True,
    )

    result = service.lookup(
        TcgCardSpec(game="pokemon", title="リザードンex", card_number="349/190", rarity="SAR", set_code="sv4a"),
        persist=False,
    )
    learned_note = next((n for n in result.notes if "社群指名可參考來源" in n), None)
    assert learned_note is not None, f"missing learned-sites hint in notes: {result.notes}"
    assert "yuyu-tei.jp" in learned_note
    assert "cardrush.jp" in learned_note


def test_tier1_completes_fast_tier2_dropped_on_grace_timeout(tmp_path) -> None:
    """Tier 1 stubs return immediately; Tier 2 stub sleeps 5s. With a 0.3s
    grace, total wall-clock should be ~0.3-0.5s and we get Tier 1 offers
    only — the slow Tier 2 stub gets dropped."""
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tier1_clients=(
            SlowStubClient("snkrdunk", 0.05),
            SlowStubClient("yuyutei", 0.05),
        ),
        tier2_clients=(
            SlowStubClient("slow-cardrush", 5.0),
        ),
        tier2_grace_seconds=0.3,
    )

    start = time.perf_counter()
    result = service.lookup(
        TcgCardSpec(game="pokemon", title="テスト", card_number="001/100", rarity="SR", set_code="sv01"),
        persist=False,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"tiered fan-out blocked on tier 2; elapsed={elapsed:.3f}s"
    sources = {offer.source for offer in result.offers}
    assert "snkrdunk" in sources
    assert "yuyutei" in sources
    assert "slow-cardrush" not in sources  # dropped by grace timeout


def test_tier2_fast_enough_is_included(tmp_path) -> None:
    """If Tier 2 finishes within grace window, its offers ARE included."""
    service = TcgPriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tier1_clients=(SlowStubClient("snkrdunk", 0.05),),
        tier2_clients=(SlowStubClient("fast-cardrush", 0.05),),
        tier2_grace_seconds=1.0,
    )
    result = service.lookup(
        TcgCardSpec(game="pokemon", title="テスト", card_number="001/100", rarity="SR", set_code="sv01"),
        persist=False,
    )
    sources = {offer.source for offer in result.offers}
    assert sources == {"snkrdunk", "fast-cardrush"}


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

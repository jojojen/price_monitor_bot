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

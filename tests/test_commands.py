from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.service import TcgLookupResult

from price_monitor_bot import commands


class StubService:
    def __init__(self, *args, **kwargs) -> None:
        self.calls: list[TcgCardSpec] = []

    def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
        self.calls.append(spec)
        if spec.card_number == "235/193":
            offer = MarketOffer(
                source="yuyutei",
                listing_id="m02a-235",
                url="https://example.com/235",
                title="メガシビルドンex",
                price_jpy=1280,
                price_kind="ask",
                captured_at=datetime.now(timezone.utc),
                source_category="poc",
                attributes={"card_number": "235/193", "rarity": "SAR", "version_code": "m02a"},
            )
            fair_value = FairValueEstimate(
                item_id="tcg-test",
                amount_jpy=1280,
                confidence=0.8,
                sample_count=1,
                reasoning=("stub",),
            )
            item = TrackedItem(
                item_id="tcg-test",
                item_type="tcg_card",
                category="tcg",
                title="メガシビルドンex",
                attributes={"game": "pokemon", "card_number": "235/193", "rarity": "SAR", "set_code": "m2a"},
            )
            return TcgLookupResult(spec=spec, item=item, offers=(offer,), fair_value=fair_value, notes=())

        item = TrackedItem(
            item_id="tcg-test-empty",
            item_type="tcg_card",
            category="tcg",
            title=spec.title,
            attributes={"game": spec.game},
        )
        return TcgLookupResult(
            spec=spec,
            item=item,
            offers=(),
            fair_value=None,
            notes=("No matching offers were found on the current source.",),
        )


class StubHintService:
    def resolve_lookup_spec(self, spec: TcgCardSpec) -> TcgCardSpec | None:
        if spec.title == "メガシビルドン" and spec.rarity == "SAR":
            return replace(spec, title="メガシビルドンex", card_number="235/193", set_code="m2a")
        return None


def test_lookup_card_uses_hot_card_fallback(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(commands, "TcgPriceService", StubService)

    result = commands.lookup_card(
        db_path=tmp_path / "monitor.sqlite3",
        game="pokemon",
        name="メガシビルドン",
        rarity="SAR",
        persist=False,
        hot_card_service=StubHintService(),  # type: ignore[arg-type]
    )

    assert result.offers
    assert result.spec.title == "メガシビルドンex"
    assert result.spec.card_number == "235/193"
    assert "Resolved the query via liquidity-source metadata fallback" in result.notes[0]

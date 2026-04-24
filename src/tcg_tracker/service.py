from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import socket
from pathlib import Path
from typing import Protocol

from market_monitor.http import HttpClient
from market_monitor.normalize import normalize_card_number, normalize_text
from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem, WatchRule
from market_monitor.pricing import FairValueCalculator
from market_monitor.storage import MonitorDatabase

from .catalog import TcgCardSpec
from .cardrush import CardrushPokemonClient
from .magi import MagiProductClient
from .yuyutei import YuyuteiClient

logger = logging.getLogger(__name__)
_TRANSIENT_SOURCE_EXCEPTIONS = (TimeoutError, socket.timeout)
_DEFAULT_USER_AGENT = "OpenClawPriceMonitor/0.1 (+https://local-dev)"
_DEFAULT_LOG_RAW_RESULT_LIMIT = 20


@dataclass(frozen=True, slots=True)
class TcgLookupResult:
    spec: TcgCardSpec
    item: TrackedItem
    offers: tuple[MarketOffer, ...]
    fair_value: FairValueEstimate | None
    notes: tuple[str, ...] = ()


class OfferLookupClient(Protocol):
    def lookup(self, spec: TcgCardSpec) -> list[MarketOffer]:
        ...


class TcgPriceService:
    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        yuyutei_client: YuyuteiClient | None = None,
        reference_clients: tuple[OfferLookupClient, ...] | list[OfferLookupClient] | None = None,
        pricing: FairValueCalculator | None = None,
        yuyutei_user_agent: str = _DEFAULT_USER_AGENT,
        log_raw_result_limit: int = _DEFAULT_LOG_RAW_RESULT_LIMIT,
    ) -> None:
        resolved_db_path = Path(db_path or "data/monitor.sqlite3")
        self._log_raw_result_limit = log_raw_result_limit

        self.database = MonitorDatabase(resolved_db_path)
        self.database.bootstrap()
        primary_client = yuyutei_client or YuyuteiClient(
            HttpClient(user_agent=yuyutei_user_agent)
        )
        if reference_clients is not None:
            self.reference_clients = tuple(reference_clients)
        elif yuyutei_client is not None:
            self.reference_clients = (primary_client,)
        else:
            shared_http_client = HttpClient(user_agent=yuyutei_user_agent)
            self.reference_clients = (
                primary_client,
                CardrushPokemonClient(shared_http_client),
                MagiProductClient(shared_http_client),
            )
        self.pricing = pricing or FairValueCalculator()

    def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
        item = spec.to_tracked_item()
        logger.info(
            "TCG service lookup started item_id=%s game=%s title=%s card_number=%s rarity=%s set_code=%s persist=%s",
            item.item_id,
            spec.game,
            spec.title,
            spec.card_number,
            spec.rarity,
            spec.set_code,
            persist,
        )
        offers = tuple(self._lookup_offers(spec))
        fair_value = self.pricing.calculate(item.item_id, offers) if self._can_calculate_fair_value(spec, offers) else None
        notes = self._build_lookup_notes(spec, offers, fair_value)
        logger.info(
            "TCG service lookup finished item_id=%s offers=%s sources=%s fair_value=%s notes=%s",
            item.item_id,
            len(offers),
            sorted({offer.source for offer in offers}),
            None if fair_value is None else fair_value.amount_jpy,
            list(notes),
        )
        logger.debug(
            "Processed offers item_id=%s offers=%s",
            item.item_id,
            [_offer_summary(offer) for offer in offers[: self._log_raw_result_limit]],
        )

        if persist:
            self.database.upsert_item(item)
            self.database.save_offers(item.item_id, offers)
            if fair_value is not None:
                self.database.save_snapshot(fair_value)

        return TcgLookupResult(spec=spec, item=item, offers=offers, fair_value=fair_value, notes=notes)

    def _lookup_offers(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[tuple[str, str, str, int]] = set()
        for client in self.reference_clients:
            client_name = type(client).__name__
            logger.debug("Querying source client=%s title=%s", client_name, spec.title)
            try:
                client_offers = client.lookup(spec)
            except _TRANSIENT_SOURCE_EXCEPTIONS as exc:
                logger.warning(
                    "Source client timed out client=%s title=%s error=%s",
                    client_name,
                    spec.title,
                    exc,
                )
                continue
            except Exception:
                logger.exception("Source client failed client=%s title=%s", client_name, spec.title)
                continue
            logger.debug(
                "Source client returned client=%s count=%s offers=%s",
                client_name,
                len(client_offers),
                [_offer_summary(offer) for offer in client_offers[: self._log_raw_result_limit]],
            )

            for offer in client_offers:
                dedupe_key = (offer.source, offer.url, offer.price_kind, offer.price_jpy)
                if dedupe_key in seen:
                    logger.debug("Deduped duplicate offer source=%s url=%s price_kind=%s", offer.source, offer.url, offer.price_kind)
                    continue
                seen.add(dedupe_key)
                offers.append(offer)

        offers.sort(key=self._offer_sort_key)
        if spec.item_kind == "sealed_box":
            offers = self._filter_sealed_box_offer_cluster(offers)
        return offers

    def seed_watchlist(
        self,
        specs: list[TcgCardSpec] | tuple[TcgCardSpec, ...],
        *,
        discount_threshold_pct: float = 15.0,
        schedule_minutes: int = 30,
    ) -> None:
        for spec in specs:
            item = spec.to_tracked_item()
            self.database.upsert_item(item)
            self.database.save_watch_rule(
                WatchRule(
                    rule_id=f"watch-{item.item_id}",
                    item_id=item.item_id,
                    source_scope="tcg",
                    discount_threshold_pct=discount_threshold_pct,
                    enabled=True,
                    schedule_minutes=schedule_minutes,
                )
            )

    @staticmethod
    def _can_calculate_fair_value(spec: TcgCardSpec, offers: tuple[MarketOffer, ...]) -> bool:
        if not offers:
            return False
        if spec.item_kind == "sealed_box":
            return True
        if spec.card_number:
            return True
        return len(TcgPriceService._variant_keys(offers)) == 1

    @staticmethod
    def _build_lookup_notes(
        spec: TcgCardSpec,
        offers: tuple[MarketOffer, ...],
        fair_value: FairValueEstimate | None,
    ) -> tuple[str, ...]:
        notes: list[str] = []
        variant_count = len(TcgPriceService._variant_keys(offers))

        if not offers:
            notes.append("No matching offers were found on the current reference sources.")
            if spec.item_kind == "sealed_box":
                notes.append("Try adding the product line, such as booster pack or high-class pack, to narrow the box search.")
                return tuple(notes)
            if not any((spec.card_number, spec.rarity, spec.set_code, spec.set_name)):
                notes.append("Try adding card number, rarity, or set code to narrow the search.")
            return tuple(notes)

        if fair_value is None and variant_count > 1:
            notes.append(
                f"Multiple variants matched this name-only query ({variant_count} variants), so no single fair value was computed."
            )
            notes.append("Add card number, rarity, or set code to narrow the result.")

        return tuple(notes)

    @staticmethod
    def _variant_keys(offers: tuple[MarketOffer, ...]) -> set[tuple[str, str, str]]:
        variants: set[tuple[str, str, str]] = set()
        for offer in offers:
            card_number = normalize_card_number(offer.attributes.get("card_number", ""))
            rarity = normalize_text(offer.attributes.get("rarity", ""))
            version_code = normalize_text(offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""))
            variants.add((card_number, rarity, version_code))
        return variants

    @staticmethod
    def _offer_sort_key(offer: MarketOffer) -> tuple[float, int, int, str]:
        kind_priority = {
            "ask": 3,
            "market": 2,
            "last_sale": 2,
            "bid": 1,
        }.get(offer.price_kind, 0)
        price_sort = -offer.price_jpy if offer.price_kind == "bid" else offer.price_jpy
        return (
            -(offer.score or 0.0),
            -kind_priority,
            price_sort,
            normalize_text(offer.title),
        )

    @staticmethod
    def _filter_sealed_box_offer_cluster(offers: list[MarketOffer]) -> list[MarketOffer]:
        if len(offers) <= 1:
            return offers

        clusters: dict[str, list[MarketOffer]] = {}
        for offer in offers:
            key = _sealed_box_cluster_key(offer.title)
            clusters.setdefault(key, []).append(offer)

        best_cluster = max(
            clusters.values(),
            key=lambda group: (
                max(offer.score or 0.0 for offer in group),
                len(group),
                sum(offer.score or 0.0 for offer in group) / len(group),
            ),
        )
        best_cluster.sort(key=TcgPriceService._offer_sort_key)
        return best_cluster


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "card_number": offer.attributes.get("card_number", ""),
        "rarity": offer.attributes.get("rarity", ""),
        "set_code": offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""),
        "score": offer.score,
    }


def _sealed_box_cluster_key(title: str) -> str:
    normalized = normalize_text(title)
    normalized = re.sub(r"(未開封\s*box|box|ボックス)", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[『』「」【】\[\]\(\)\-]", "", normalized)
    return normalized.strip()

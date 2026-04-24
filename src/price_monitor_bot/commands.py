from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

from market_monitor import ReferenceSource, filter_reference_sources, load_reference_sources
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.hot_cards import TcgHotCardService
from tcg_tracker.service import TcgLookupResult, TcgPriceService

logger = logging.getLogger(__name__)


def build_card_spec(
    *,
    game: str,
    name: str,
    card_number: str | None = None,
    rarity: str | None = None,
    set_code: str | None = None,
    set_name: str | None = None,
    aliases: tuple[str, ...] = (),
    extra_keywords: tuple[str, ...] = (),
) -> TcgCardSpec:
    return TcgCardSpec(
        game=game,
        title=name,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        set_name=set_name,
        aliases=aliases,
        extra_keywords=extra_keywords,
    )


def lookup_card(
    *,
    db_path: str | Path,
    game: str,
    name: str,
    card_number: str | None = None,
    rarity: str | None = None,
    set_code: str | None = None,
    set_name: str | None = None,
    aliases: tuple[str, ...] = (),
    extra_keywords: tuple[str, ...] = (),
    persist: bool = True,
    hot_card_service: TcgHotCardService | None = None,
) -> TcgLookupResult:
    logger.info(
        "Lookup requested game=%s name=%s card_number=%s rarity=%s set_code=%s persist=%s",
        game,
        name,
        card_number,
        rarity,
        set_code,
        persist,
    )
    service = TcgPriceService(db_path=db_path)
    spec = build_card_spec(
        game=game,
        name=name,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        set_name=set_name,
        aliases=aliases,
        extra_keywords=extra_keywords,
    )
    return _lookup_with_hot_card_fallback(
        service=service,
        spec=spec,
        persist=persist,
        hot_card_service=hot_card_service or TcgHotCardService(),
    )


def seed_example_watchlist(*, db_path: str | Path, config_path: str | Path) -> int:
    service = TcgPriceService(db_path=db_path)
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    specs = [
        TcgCardSpec(
            game=item["game"],
            title=item["title"],
            card_number=item.get("card_number"),
            rarity=item.get("rarity"),
            set_code=item.get("set_code"),
            set_name=item.get("set_name"),
        )
        for item in payload
    ]
    service.seed_watchlist(specs)
    return len(specs)


def list_reference_sources(
    *,
    config_path: str | Path = "config/reference_sources.json",
    game: str | None = None,
    source_kind: str | None = None,
    reference_role: str | None = None,
) -> tuple[ReferenceSource, ...]:
    sources = load_reference_sources(config_path)
    return filter_reference_sources(
        sources,
        game=game,
        source_kind=source_kind,
        reference_role=reference_role,
    )


def _lookup_with_hot_card_fallback(
    *,
    service: TcgPriceService,
    spec: TcgCardSpec,
    persist: bool,
    hot_card_service: TcgHotCardService,
) -> TcgLookupResult:
    initial = service.lookup(spec, persist=False)
    logger.debug(
        "Primary lookup completed title=%s offers=%s fair_value=%s notes=%s",
        spec.title,
        len(initial.offers),
        None if initial.fair_value is None else initial.fair_value.amount_jpy,
        list(initial.notes),
    )
    if initial.offers:
        return service.lookup(spec, persist=True) if persist else initial

    try:
        resolved_spec = hot_card_service.resolve_lookup_spec(spec)
    except Exception:
        logger.exception("Hot-card fallback resolution failed title=%s", spec.title)
        resolved_spec = None
    if resolved_spec is None:
        logger.info("No hot-card fallback spec available title=%s", spec.title)
        return service.lookup(spec, persist=True) if persist else initial

    logger.info(
        "Hot-card fallback resolved title=%s -> resolved_title=%s resolved_card_number=%s resolved_rarity=%s resolved_set_code=%s",
        spec.title,
        resolved_spec.title,
        resolved_spec.card_number,
        resolved_spec.rarity,
        resolved_spec.set_code,
    )
    resolved_result = service.lookup(resolved_spec, persist=persist)
    if not resolved_result.offers:
        logger.info("Resolved fallback spec still produced no offers resolved_title=%s", resolved_spec.title)
        return resolved_result

    resolution_note = (
        "Resolved the query via liquidity-source metadata fallback: "
        f"{resolved_spec.title} / {resolved_spec.card_number or 'n/a'} / {resolved_spec.rarity or 'n/a'}"
    )
    logger.info(
        "Fallback lookup succeeded resolved_title=%s offers=%s fair_value=%s",
        resolved_spec.title,
        len(resolved_result.offers),
        None if resolved_result.fair_value is None else resolved_result.fair_value.amount_jpy,
    )
    return replace(
        resolved_result,
        notes=(resolution_note, *resolved_result.notes),
    )

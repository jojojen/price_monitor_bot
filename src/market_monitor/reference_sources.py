from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReferenceSource:
    id: str
    name: str
    games: tuple[str, ...]
    source_kind: str
    reference_roles: tuple[str, ...]
    price_weight: float
    trust_score: float
    url: str
    notes: str


def load_reference_sources(path: str | Path = "config/reference_sources.json") -> tuple[ReferenceSource, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        ReferenceSource(
            id=item["id"],
            name=item["name"],
            games=tuple(item["games"]),
            source_kind=item["source_kind"],
            reference_roles=tuple(item["reference_roles"]),
            price_weight=float(item["price_weight"]),
            trust_score=float(item["trust_score"]),
            url=item["url"],
            notes=item["notes"],
        )
        for item in payload
    )


def filter_reference_sources(
    sources: Iterable[ReferenceSource],
    *,
    game: str | None = None,
    source_kind: str | None = None,
    reference_role: str | None = None,
) -> tuple[ReferenceSource, ...]:
    filtered = tuple(
        source
        for source in sources
        if (game is None or game in source.games)
        and (source_kind is None or source.source_kind == source_kind)
        and (reference_role is None or reference_role in source.reference_roles)
    )
    return tuple(sorted(filtered, key=lambda source: (-source.trust_score, -source.price_weight, source.id)))

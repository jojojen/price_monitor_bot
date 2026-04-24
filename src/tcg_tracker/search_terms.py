from __future__ import annotations

import re

from .catalog import TcgCardSpec


def build_lookup_terms(spec: TcgCardSpec) -> tuple[str, ...]:
    terms: list[str] = []
    if spec.card_number:
        terms.append(spec.card_number)
        if spec.game == "pokemon":
            terms.extend(_pokemon_card_number_variants(spec.card_number))

    for keyword in spec.keywords():
        if keyword:
            terms.append(keyword)

    if spec.game == "pokemon":
        terms.extend(_pokemon_title_variants(spec.title))
        for alias in spec.aliases:
            terms.extend(_pokemon_title_variants(alias))

    deduped: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def pokemon_title_variants(title: str) -> tuple[str, ...]:
    return _pokemon_title_variants(title)


def _pokemon_title_variants(title: str) -> tuple[str, ...]:
    cleaned = title.strip()
    if not cleaned:
        return ()

    lower = cleaned.lower()
    if lower.endswith("ex"):
        base = cleaned[:-2].strip()
        return tuple(value for value in (cleaned, base) if value)
    return cleaned, f"{cleaned}ex"


def _pokemon_card_number_variants(card_number: str) -> tuple[str, ...]:
    match = re.fullmatch(r"(?P<numerator>\d{1,3})/(?P<denominator>\d{1,3})", card_number.strip())
    if match is None:
        return ()

    numerator = int(match.group("numerator"))
    denominator = int(match.group("denominator"))
    if numerator <= 0 or denominator <= 0:
        return ()

    original = card_number.strip()
    compact = f"{numerator}/{denominator}"
    padded = f"{numerator:03d}/{denominator:03d}"
    return tuple(
        value
        for value in (compact, padded)
        if value and value != original
    )

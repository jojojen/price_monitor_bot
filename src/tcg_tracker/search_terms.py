from __future__ import annotations

import re

from .catalog import TcgCardSpec


def build_lookup_terms(spec: TcgCardSpec) -> tuple[str, ...]:
    if spec.item_kind == "sealed_box":
        return _build_sealed_box_lookup_terms(spec)

    terms: list[str] = []
    if spec.card_number:
        terms.append(spec.card_number)
        if spec.game == "pokemon":
            terms.extend(_pokemon_card_number_variants(spec.card_number))
        if spec.game in {"yugioh", "union_arena"}:
            terms.extend(generic_card_number_variants(spec.card_number))

    for keyword in spec.keywords():
        if keyword:
            terms.append(keyword)

    if spec.game == "pokemon":
        terms.extend(_pokemon_title_variants(spec.title))
        for alias in spec.aliases:
            terms.extend(_pokemon_title_variants(alias))
    if spec.game in {"yugioh", "union_arena"}:
        terms.extend(_japanese_spacing_variants(spec.title))
        for alias in spec.aliases:
            terms.extend(_japanese_spacing_variants(alias))

    deduped: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _build_sealed_box_lookup_terms(spec: TcgCardSpec) -> tuple[str, ...]:
    terms: list[str] = []
    for keyword in spec.keywords():
        if keyword:
            terms.append(keyword)
            if spec.game == "pokemon":
                terms.extend(_pokemon_title_variants(keyword))
            terms.extend(_sealed_box_ascii_token_variants(keyword))

    if spec.set_code:
        terms.append(spec.set_code)

    box_variants: list[str] = []
    for base_term in list(terms):
        stripped = base_term.strip()
        if not stripped:
            continue
        box_variants.extend(
            (
                f"{stripped} box",
                f"{stripped} BOX",
                f"{stripped} 未開封BOX",
                f"{stripped} 未開封 BOX",
            )
        )
    terms.extend(box_variants)

    deduped: list[str] = []
    for term in terms:
        cleaned = term.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _sealed_box_ascii_token_variants(value: str) -> tuple[str, ...]:
    tokens = [token for token in re.findall(r"[A-Za-z0-9]+", value or "") if token.lower() not in {"box"}]
    if not tokens:
        return ()

    variants: list[str] = []
    normalized_tokens = [token.upper() if len(token) >= 4 else token.lower() for token in tokens]
    variants.append(" ".join(normalized_tokens))
    if len(tokens) >= 2:
        variants.append(f"{tokens[0].upper()} {tokens[-1].lower()}")
    return tuple(variant for variant in variants if variant and variant != value.strip())


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


def _japanese_spacing_variants(title: str) -> tuple[str, ...]:
    cleaned = title.strip()
    if not cleaned:
        return ()
    variants = (
        cleaned.replace("・", ""),
        cleaned.replace("・", " "),
        cleaned.replace(" ", ""),
    )
    return tuple(value for value in variants if value and value != cleaned)


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


def generic_card_number_variants(card_number: str) -> tuple[str, ...]:
    original = card_number.strip().upper()
    if not original:
        return ()

    variants: list[str] = []
    padded = _pad_trailing_card_number(original)
    if padded and padded != original:
        variants.append(padded)

    depadded = _depad_trailing_card_number(original)
    if depadded and depadded != original:
        variants.append(depadded)

    tokenized = re.sub(r"[-/]", " ", padded or original)
    tokenized = " ".join(tokenized.split())
    if tokenized and tokenized not in {original, *variants}:
        variants.append(tokenized)

    deduped: list[str] = []
    for variant in variants:
        if variant and variant not in deduped:
            deduped.append(variant)
    return tuple(deduped)


def _pad_trailing_card_number(value: str) -> str | None:
    match = re.fullmatch(r"(?P<prefix>.+[-/])(?P<number>\d{1,3})", value.strip().upper())
    if match is None:
        return None
    return f"{match.group('prefix')}{int(match.group('number')):03d}"


def _depad_trailing_card_number(value: str) -> str | None:
    match = re.fullmatch(r"(?P<prefix>.+[-/])(?P<number>0{1,2}\d{1,3})", value.strip().upper())
    if match is None:
        return None
    return f"{match.group('prefix')}{int(match.group('number'))}"

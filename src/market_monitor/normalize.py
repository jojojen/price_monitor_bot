from __future__ import annotations

import re
import unicodedata

SPACE_RE = re.compile(r"\s+")
BRACKET_RE = re.compile(r"[【】\[\]()（）]")


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = BRACKET_RE.sub(" ", value)
    value = value.strip().lower()
    return SPACE_RE.sub(" ", value)


def normalize_card_number(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = SPACE_RE.sub("", value.upper())
    return value.replace("／", "/")


def contains_all_keywords(text: str, keywords: tuple[str, ...] | list[str]) -> bool:
    haystack = normalize_text(text)
    return all(normalize_text(keyword) in haystack for keyword in keywords if keyword)

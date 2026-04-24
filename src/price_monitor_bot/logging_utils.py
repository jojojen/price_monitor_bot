from __future__ import annotations

_MASK_IDENTIFIERS_IN_LOGS = True


def mask_identifier(value: str | int | None) -> str:
    if value is None:
        return "n/a"
    text = str(value)
    if not _MASK_IDENTIFIERS_IN_LOGS:
        return text
    if len(text) <= 4:
        return "*" * len(text)
    return f"{text[:2]}***{text[-2:]}"


def trim_for_log(value: str, *, limit: int = 240) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[: limit - 3]}..."

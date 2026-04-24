from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

from market_monitor.models import TrackedItem
from market_monitor.normalize import normalize_card_number

GAME_CODES = {
    "pokemon": "poc",
    "ws": "ws",
}


def build_tcg_item_id(game: str, title: str, card_number: str | None, rarity: str | None, set_code: str | None) -> str:
    payload = "|".join([game, title, card_number or "", rarity or "", set_code or ""])
    digest = sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"tcg-{digest}"


@dataclass(frozen=True, slots=True)
class TcgCardSpec:
    game: str
    title: str
    card_number: str | None = None
    rarity: str | None = None
    set_code: str | None = None
    set_name: str | None = None
    aliases: tuple[str, ...] = ()
    extra_keywords: tuple[str, ...] = ()
    item_id: str | None = None

    def __post_init__(self) -> None:
        if self.game not in GAME_CODES:
            raise ValueError(f"unsupported game: {self.game}")

    @property
    def source_code(self) -> str:
        return GAME_CODES[self.game]

    @property
    def normalized_card_number(self) -> str:
        return normalize_card_number(self.card_number or "")

    def resolved_item_id(self) -> str:
        return self.item_id or build_tcg_item_id(
            self.game,
            self.title,
            self.card_number,
            self.rarity,
            self.set_code,
        )

    def keywords(self) -> tuple[str, ...]:
        values = [self.title, *self.aliases, *self.extra_keywords]
        deduped: list[str] = []
        for value in values:
            if value and value not in deduped:
                deduped.append(value)
        return tuple(deduped)

    def to_tracked_item(self) -> TrackedItem:
        attributes = {
            "game": self.game,
            "source_code": self.source_code,
        }
        if self.card_number:
            attributes["card_number"] = self.card_number
        if self.rarity:
            attributes["rarity"] = self.rarity
        if self.set_code:
            attributes["set_code"] = self.set_code
        if self.set_name:
            attributes["set_name"] = self.set_name

        return TrackedItem(
            item_id=self.resolved_item_id(),
            item_type="tcg_card",
            category="tcg",
            title=self.title,
            aliases=self.aliases,
            attributes=attributes,
        )

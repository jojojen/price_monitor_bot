from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import re

from market_monitor.models import TrackedItem
from market_monitor.normalize import normalize_card_number

GAME_CODES = {
    "pokemon": "poc",
    "ws": "ws",
    "yugioh": "ygo",
    "union_arena": "ua",
}
SUPPORTED_GAMES = tuple(GAME_CODES)
GAME_ALIASES = {
    "pokemon": "pokemon",
    "ptcg": "pokemon",
    "ポケモン": "pokemon",
    "寶可夢": "pokemon",
    "宝可梦": "pokemon",
    "寶可卡": "pokemon",
    "宝可卡": "pokemon",
    "ws": "ws",
    "weiss": "ws",
    "weiss_schwarz": "ws",
    "weiß_schwarz": "ws",
    "ヴァイス": "ws",
    "yugioh": "yugioh",
    "ygo": "yugioh",
    "yu_gi_oh": "yugioh",
    "遊戯王": "yugioh",
    "遊戲王": "yugioh",
    "游戏王": "yugioh",
    "遊戯王ocg": "yugioh",
    "遊戯王ラッシュデュエル": "yugioh",
    "union_arena": "union_arena",
    "unionarena": "union_arena",
    "union_area": "union_arena",
    "unionarea": "union_arena",
    "ua": "union_arena",
    "ユニオンアリーナ": "union_arena",
}
ITEM_KINDS = {"card", "sealed_box"}


def normalize_game_key(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower()
    if not key:
        return None
    key = key.replace("-", "_").replace("/", "_")
    key = key.replace("☆", "").replace("・", "")
    key = re.sub(r"\s+", "_", key)
    return GAME_ALIASES.get(key)


def supported_game_hint() -> str:
    return "pokemon, ws, yugioh/ygo, union_arena/ua"


def build_tcg_item_id(
    game: str,
    title: str,
    card_number: str | None,
    rarity: str | None,
    set_code: str | None,
    *,
    item_kind: str = "card",
) -> str:
    payload = "|".join([item_kind, game, title, card_number or "", rarity or "", set_code or ""])
    digest = sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"tcg-{digest}"


@dataclass(frozen=True, slots=True)
class TcgCardSpec:
    game: str
    title: str
    item_kind: str = "card"
    card_number: str | None = None
    rarity: str | None = None
    set_code: str | None = None
    set_name: str | None = None
    aliases: tuple[str, ...] = ()
    extra_keywords: tuple[str, ...] = ()
    item_id: str | None = None

    def __post_init__(self) -> None:
        normalized_game = normalize_game_key(self.game)
        if normalized_game not in GAME_CODES:
            raise ValueError(f"unsupported game: {self.game}")
        object.__setattr__(self, "game", normalized_game)
        if self.item_kind not in ITEM_KINDS:
            raise ValueError(f"unsupported item kind: {self.item_kind}")

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
            item_kind=self.item_kind,
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
            "item_kind": self.item_kind,
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
            item_type="tcg_sealed_box" if self.item_kind == "sealed_box" else "tcg_card",
            category="tcg",
            title=self.title,
            aliases=self.aliases,
            attributes=attributes,
        )

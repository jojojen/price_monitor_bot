from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.hot_cards import HotCardBoard, HotCardEntry, HotCardReference
from tcg_tracker.image_lookup import ParsedCardImage, TcgImageLookupOutcome
from tcg_tracker.service import TcgLookupResult
from tests.image_lookup_case_fixtures import get_image_lookup_live_case

from price_monitor_bot.formatters import format_lookup_result_telegram
from price_monitor_bot.natural_language import (
    TelegramNaturalLanguageIntent,
    _normalize_intent,
    fallback_route_telegram_natural_language,
)
from price_monitor_bot.bot import (
    PendingTelegramPhotoClarification,
    PendingTelegramTextClarification,
    PhotoLookupReply,
    PollHeartbeat,
    RegisteredCommand,
    TelegramCommandProcessor,
    TelegramFileAttachment,
    TelegramPhotoIntentAnalysis,
    TelegramPhotoIntentOption,
    TelegramLookupQuery,
    TelegramReputationQuery,
    TelegramReputationDelivery,
    TelegramTextIntentOption,
    _build_text_intent_candidates,
    _drain_pending_updates,
    _is_conflict_error,
    build_processing_ack,
    format_liquidity_board,
    format_photo_lookup_result,
    handle_telegram_callback_query,
    handle_telegram_message,
    parse_lookup_command,
    parse_reputation_snapshot_command,
    start_poll_watchdog,
)

# Every call to `handle_telegram_message` now sends an immediate intake ack
# before kicking off the real processing pipeline; the assertions below
# include these to lock in that behaviour.
PHOTO_INTAKE_ACK = "已收到圖片，開始解讀使用者意圖"
TEXT_INTAKE_ACK = "已收到訊息，開始解讀使用者意圖"


def _stub_board() -> HotCardBoard:
    return HotCardBoard(
        game="pokemon",
        label="Pokemon Liquidity Board",
        methodology="stub methodology",
        generated_at=datetime.now(timezone.utc),
        items=(
            HotCardEntry(
                game="pokemon",
                rank=1,
                title="Pikachu ex",
                price_jpy=99800,
                thumbnail_url="https://example.com/pikachu.jpg",
                card_number="132/106",
                rarity="SAR",
                set_code="sv08",
                listing_count=5,
                best_ask_jpy=99800,
                best_bid_jpy=80000,
                previous_bid_jpy=50000,
                bid_ask_ratio=0.8016,
                buy_support_score=90.08,
                momentum_boost_score=6.0,
                buy_signal_label="priceup",
                hot_score=88.2,
                attention_score=41.7,
                social_post_count=3,
                social_engagement_count=120,
                notes=("stub note",),
                is_graded=False,
                references=(HotCardReference(label="Ranking Source", url="https://example.com/rank"),),
            ),
        ),
    )


class FakeTelegramClient:
    def __init__(self, sample_path: Path | None = None) -> None:
        self.sample_path = sample_path
        self.sent_messages: list[str] = []
        self.sent_documents: list[tuple[str, str | None]] = []
        self.sent_photos: list[tuple[str, str | None]] = []
        self.edited_messages: list[dict[str, object]] = []
        self.answered_callbacks: list[dict[str, object]] = []

    def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.sent_messages.append(text)
        return {"chat_id": str(chat_id), "text": text, "reply_markup": reply_markup}

    def edit_message_text(
        self,
        *,
        chat_id: str | int,
        message_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        record = {
            "chat_id": str(chat_id),
            "message_id": message_id,
            "text": text,
            "reply_markup": reply_markup,
        }
        self.edited_messages.append(record)
        return record

    def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> dict[str, object]:
        record = {
            "callback_query_id": callback_query_id,
            "text": text,
            "show_alert": show_alert,
        }
        self.answered_callbacks.append(record)
        return record

    def send_document(self, *, chat_id: str | int, document_path: Path, caption: str | None = None) -> dict[str, object]:
        self.sent_documents.append((document_path.name, caption))
        return {"chat_id": str(chat_id), "document": document_path.name, "caption": caption}

    def send_photo(self, *, chat_id: str | int, photo_path: Path, caption: str | None = None) -> dict[str, object]:
        self.sent_photos.append((photo_path.name, caption))
        return {"chat_id": str(chat_id), "photo": photo_path.name, "caption": caption}

    def get_file(self, *, file_id: str) -> dict[str, object]:
        assert self.sample_path is not None
        return {"file_path": self.sample_path.name, "file_id": file_id}

    def download_file(self, *, file_path: str) -> bytes:
        assert self.sample_path is not None
        assert file_path == self.sample_path.name
        return self.sample_path.read_bytes()


class StubNaturalLanguageRouter:
    def __init__(self, intent: TelegramNaturalLanguageIntent | None) -> None:
        self.intent = intent
        self.seen_texts: list[str] = []

    def route(self, text: str) -> TelegramNaturalLanguageIntent | None:
        self.seen_texts.append(text)
        return self.intent


def _ambiguous_photo_analysis(
    *,
    parsed_game: str | None = "pokemon",
    parsed_item_kind: str | None = "card",
    parsed_title: str | None = None,
) -> TelegramPhotoIntentAnalysis:
    return TelegramPhotoIntentAnalysis(
        options=(
            TelegramPhotoIntentOption(1, "pokemon_card_price", "要我查這張寶可夢卡市價嗎？", "/scan pokemon"),
            TelegramPhotoIntentOption(2, "yugioh_card_price", "要我查這張遊戲王卡市價嗎？", "/scan yugioh"),
            TelegramPhotoIntentOption(3, "pokemon_box_price", "要我查這個寶可夢卡盒市價嗎？", "/scan pokemon"),
        ),
        parsed_game=parsed_game,
        parsed_item_kind=parsed_item_kind,
        parsed_title=parsed_title,
    )


def test_parse_lookup_command_supports_pipe_format() -> None:
    query = parse_lookup_command("pokemon | Pikachu ex | 132/106 | SAR | sv08")

    assert query == TelegramLookupQuery(
        game="pokemon",
        name="Pikachu ex",
        card_number="132/106",
        rarity="SAR",
        set_code="sv08",
    )


def test_parse_lookup_command_supports_simple_format() -> None:
    query = parse_lookup_command("ws Hatsune Miku")

    assert query == TelegramLookupQuery(
        game="ws",
        name="Hatsune Miku",
    )


def test_parse_lookup_command_supports_yugioh_and_union_arena_aliases() -> None:
    ygo_query = parse_lookup_command("ygo | 青眼の白龍 | QCCP-JP001 | ウルトラ")
    ua_query = parse_lookup_command("ua 綾波レイ")

    assert ygo_query == TelegramLookupQuery(
        game="yugioh",
        name="青眼の白龍",
        card_number="QCCP-JP001",
        rarity="ウルトラ",
        set_code="qccp",
    )
    assert ua_query == TelegramLookupQuery(game="union_arena", name="綾波レイ")


def test_parse_lookup_command_recovers_card_metadata_from_simple_format() -> None:
    ua_query = parse_lookup_command("union_arena uapr/eva-1-071 綾波レイ")
    ua_card_query = parse_lookup_command("ua card uapr/eva-1-071 綾波レイ")
    pokemon_query = parse_lookup_command("pokemon リザードンex 201/165 SAR")

    assert ua_query == TelegramLookupQuery(
        game="union_arena",
        name="綾波レイ",
        card_number="UAPR/EVA-1-071",
        set_code="uapr",
    )
    assert ua_card_query == ua_query
    assert pokemon_query == TelegramLookupQuery(
        game="pokemon",
        name="リザードンex",
        card_number="201/165",
        rarity="SAR",
    )


def test_command_processor_restricts_unconfigured_chat() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"999"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    assert processor.build_reply(chat_id="123", text="/ping") is None


def test_command_processor_handles_price_and_trend_aliases() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}:{query.card_number}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    lookup_reply = processor.build_reply(chat_id="123", text="/price pokemon | Pikachu ex | 132/106")
    trend_reply = processor.build_reply(chat_id="123", text="/trend pokemon")
    hot_reply = processor.build_reply(chat_id="123", text="/hot pokemon 1")

    assert lookup_reply == "pokemon:Pikachu ex:132/106"
    assert "Pokemon Liquidity Board" in trend_reply
    assert "bid " in trend_reply and "80,000" in trend_reply
    assert "ask " in trend_reply and "99,800" in trend_reply
    assert "boost 6.00" in trend_reply
    assert "Pokemon Liquidity Board" in hot_reply
    assert "\n1. Pikachu ex\n" in hot_reply


def test_command_processor_help_lists_trend_and_scan_commands() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    help_reply = processor.build_reply(chat_id="123", text="/help")

    assert "/trend pokemon" in help_reply
    assert "/price pokemon | Pikachu ex | 132/106 | SAR | sv08" in help_reply
    assert "/snapshot https://jp.mercari.com/item/m123456789" in help_reply
    assert "/scan pokemon" in help_reply
    assert "/hunt status" in help_reply


def test_command_processor_handles_hunt_status() -> None:
    def _stub_hunt(remainder: str, chat_id: str) -> str:
        if remainder.strip() in {"status", ""}:
            return "targets: Umbreon"
        return "unknown"

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        command_handlers={"/hunt": RegisteredCommand(_stub_hunt)},
    )

    assert processor.build_reply(chat_id="123", text="/hunt status") == "targets: Umbreon"


def test_parse_reputation_snapshot_command_requires_url() -> None:
    query = parse_reputation_snapshot_command("https://jp.mercari.com/item/m123456789")

    assert query == TelegramReputationQuery(query_url="https://jp.mercari.com/item/m123456789")


def test_command_processor_handles_snapshot_command() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        reputation_renderer=lambda query: f"snapshot:{query.query_url}",
    )

    reply = processor.build_reply(chat_id="123", text="/snapshot https://jp.mercari.com/item/m123456789")

    assert reply == "snapshot:https://jp.mercari.com/item/m123456789"


def test_command_processor_handles_natural_language_lookup_via_router() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="lookup_card",
            game="pokemon",
            name="Pikachu ex",
            card_number="132/106",
            rarity="SAR",
            set_code="sv08",
            confidence=0.98,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}:{query.card_number}:{query.rarity}:{query.set_code}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    reply = processor.build_reply(chat_id="123", text="幫我查 pokemon Pikachu ex 132/106 SAR sv08")

    assert reply == "pokemon:Pikachu ex:132/106:SAR:sv08"
    assert router.seen_texts == ["幫我查 pokemon Pikachu ex 132/106 SAR sv08"]


def test_command_processor_handles_natural_language_trend_via_router() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="pokemon",
            limit=3,
            confidence=0.91,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    reply = processor.build_reply(chat_id="123", text="pokemon 熱門前 3")

    assert "Pokemon Liquidity Board" in reply
    assert router.seen_texts == ["pokemon 熱門前 3"]


def test_command_processor_builds_ack_for_natural_language_trend() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=5,
            confidence=0.94,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        ),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    plan = processor.build_reply_plan(chat_id="123", text="ws 熱門前 5")

    assert plan.ack == "已理解查詢內容，相當於 /trend ws 5，開始整理資料。"
    reply = plan.execute()
    assert reply is not None
    assert "WS Liquidity Board" in reply
    assert router.seen_texts == ["ws 熱門前 5"]


def test_build_processing_ack_for_heavy_actions() -> None:
    assert build_processing_ack(text="/price pokemon Pikachu ex") == "收到查價指令，開始處理。"
    assert build_processing_ack(text="/trend pokemon") == "收到趨勢榜查詢，開始整理資料。"
    assert build_processing_ack(text="/snapshot https://jp.mercari.com/item/m123456789") == (
        "收到信譽快照查詢，先檢查既有 proof，必要時建立新快照。"
    )
    assert build_processing_ack(has_photo=True) == "收到圖片，開始解析與查價。"
    assert build_processing_ack(text="/ping") is None


def test_handle_telegram_message_sends_ack_then_photo_result() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"photo:{query.game_hint}:{query.title_hint}:{query.image_path.suffix}",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "/scan pokemon Pikachu ex",
        },
    )

    assert replies == (
        PHOTO_INTAKE_ACK,
        "收到圖片，開始解析與查價。",
        "photo:pokemon:Pikachu ex:.jpg",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_ack_then_text_result() -> None:
    client = FakeTelegramClient()
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "/price pokemon Pikachu ex",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到查價指令，開始處理。",
        "pokemon:Pikachu ex",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_ignores_generic_price_caption_as_title_hint() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"photo:{query.game_hint}:{query.title_hint}:{query.image_path.suffix}",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "查這個box市價",
        },
    )

    assert replies == (
        PHOTO_INTAKE_ACK,
        build_processing_ack(has_photo=True),
        "photo:None:None:.jpg",
    )


def test_handle_telegram_message_clarifies_image_without_caption() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(parsed_title="Pikachu ex"),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )

    assert len(replies) == 2
    assert replies[0] == PHOTO_INTAKE_ACK
    assert "請點按鈕" in replies[1]
    assert "1. 要我查這張寶可夢卡市價嗎？" in replies[1]
    assert "4. 都不是，請回答：否，[您的意圖]" in replies[1]
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_uses_explicit_price_caption_without_clarifying() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)

    def analyzer(query):
        raise AssertionError("photo intent analyzer should not run for explicit price captions")

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=analyzer,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"photo:{query.game_hint}:{query.title_hint}",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "查這張寶可夢卡市價",
        },
    )

    assert replies == (
        PHOTO_INTAKE_ACK,
        build_processing_ack(has_photo=True),
        "photo:pokemon:None",
    )


def test_handle_telegram_message_runs_selected_photo_option_after_clarification() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}",
        message={
            "chat": {"id": "123"},
            "text": "1",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我就照第 1 個方式處理。",
        "resolved:/scan pokemon:pokemon",
    )


def test_handle_telegram_message_supports_photo_override_text() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}:{query.item_kind_hint}",
        message={
            "chat": {"id": "123"},
            "text": "否，查這張遊戲王卡市價",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我改照你補充的意思處理：查這張遊戲王卡市價",
        "resolved:/scan yugioh:yugioh:card",
    )


def test_handle_telegram_message_card_selection_overrides_box_guess() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(parsed_item_kind="sealed_box"),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "photo": [{"file_id": "photo-1", "file_size": 128}]},
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}:{query.item_kind_hint}",
        message={"chat": {"id": "123"}, "text": "2"},
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我就照第 2 個方式處理。",
        "resolved:/scan yugioh:yugioh:card",
    )


def test_handle_telegram_message_supports_photo_identify_override() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(
            parsed_game="pokemon",
            parsed_item_kind="sealed_box",
            parsed_title="ポケモンカード151",
        ),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "否，這是什麼",
        },
    )

    assert replies == (TEXT_INTAKE_ACK, "我目前看起來比較像 寶可夢卡盒：ポケモンカード151")
    assert processor.get_pending_photo_clarification("123") is None


def test_handle_telegram_message_reminds_when_photo_clarification_reply_is_unrecognized() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "蛤",
        },
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    assert "我現在在等你確認這張圖要怎麼處理" in replies[1]
    assert "1. 要我查這張寶可夢卡市價嗎？" in replies[1]


def test_handle_telegram_message_requests_override_when_other_option_is_selected() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "4",
        },
    )

    assert replies == (TEXT_INTAKE_ACK, "好，請直接回答：否，[您的意圖]")


def test_pending_photo_clarification_expires(tmp_path: Path) -> None:
    sample_path = tmp_path / "pending.jpg"
    sample_path.write_bytes(b"stub")
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )
    processor.set_pending_photo_clarification(
        PendingTelegramPhotoClarification(
            chat_id="123",
            image_path=sample_path,
            caption=None,
            file_id="photo-1",
            options=_ambiguous_photo_analysis().options,
            created_at=0.0,
        )
    )
    assert processor.get_pending_photo_clarification("123") is None


def test_handle_telegram_message_sends_snapshot_ack_then_result(tmp_path: Path) -> None:
    client = FakeTelegramClient()
    pdf_path = tmp_path / "proof_123.pdf"
    png_path = tmp_path / "proof_123.png"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nstub")
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        reputation_renderer=lambda query: TelegramReputationDelivery(
            summary_text=f"snapshot:{query.query_url}",
            attachments=(
                TelegramFileAttachment(kind="document", path=pdf_path, caption="Reputation snapshot PDF"),
                TelegramFileAttachment(kind="photo", path=png_path, caption="Reputation snapshot preview"),
            ),
            cleanup_paths=(pdf_path, png_path),
        ),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "/snapshot https://jp.mercari.com/item/m123456789",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到信譽快照查詢，先檢查既有 proof，必要時建立新快照。",
        "snapshot:https://jp.mercari.com/item/m123456789",
    )
    assert client.sent_messages == list(replies)
    assert client.sent_documents == [("proof_123.pdf", "Reputation snapshot PDF")]
    assert client.sent_photos == [("proof_123.png", "Reputation snapshot preview")]
    assert not pdf_path.exists()
    assert not png_path.exists()


def test_handle_telegram_message_sends_natural_language_ack_then_result() -> None:
    client = FakeTelegramClient()
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=5,
            confidence=0.94,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        ),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "ws 熱門前 5",
        },
    )

    assert replies[0] == TEXT_INTAKE_ACK
    assert replies[1] == "已理解查詢內容，相當於 /trend ws 5，開始整理資料。"
    assert "WS Liquidity Board" in replies[2]
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_natural_language_ack_before_running_heavy_work() -> None:
    client = FakeTelegramClient()
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=3,
            confidence=0.94,
        )
    )

    def board_loader() -> tuple[HotCardBoard, ...]:
        assert client.sent_messages == [TEXT_INTAKE_ACK, "已理解查詢內容，相當於 /trend ws 3，開始整理資料。"]
        return (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        )

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=board_loader,
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "查ws熱門前3",
        },
    )

    assert replies[0] == TEXT_INTAKE_ACK
    assert replies[1] == "已理解查詢內容，相當於 /trend ws 3，開始整理資料。"
    assert "WS Liquidity Board" in replies[2]


def _mixed_grade_lookup_result() -> TcgLookupResult:
    now = datetime.now(timezone.utc)
    offers = (
        MarketOffer(source="cardrush_pokemon", listing_id="a1", url="https://cardrush.example/a1",
                    title="ピカチュウex 132/106", price_jpy=32800, price_kind="ask",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="cardrush_pokemon", listing_id="a2", url="https://cardrush.example/a2",
                    title="ピカチュウex 132/106", price_jpy=28000, price_kind="bid",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="magi", listing_id="m1", url="https://magi.example/m1",
                    title="ピカチュウex 132/106", price_jpy=30500, price_kind="market",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="cardrush_pokemon", listing_id="p1", url="https://cardrush.example/p1",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=98000, price_kind="ask",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
        MarketOffer(source="cardrush_pokemon", listing_id="p2", url="https://cardrush.example/p2",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=88000, price_kind="bid",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
        MarketOffer(source="magi", listing_id="p3", url="https://magi.example/p3",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=92000, price_kind="market",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
    )
    spec = TcgCardSpec(game="pokemon", title="Pikachu ex", card_number="132/106", rarity="SAR")
    item = TrackedItem(item_id="x", item_type="card", category="tcg", title="Pikachu ex")
    fv = FairValueEstimate(item_id="x", amount_jpy=32500, confidence=0.72, sample_count=3, reasoning=())
    return TcgLookupResult(spec=spec, item=item, offers=offers, fair_value=fv, notes=("sample note",))


def test_format_lookup_result_telegram_shows_raw_and_psa10_sections_without_mixed_total_price() -> None:
    text = format_lookup_result_telegram(_mixed_grade_lookup_result())

    assert "Raw" in text
    assert "PSA 10" in text

    raw_section = text.split("Raw", 1)[1].split("PSA 10", 1)[0]
    psa_section = text.split("PSA 10", 1)[1].split("Sources:", 1)[0]
    header_section = text.split("Raw", 1)[0]

    for section_name, section in (("raw", raw_section), ("psa10", psa_section)):
        assert "Fair Value:" in section, f"{section_name} section missing Fair Value"
        assert "Avg Price:" in section, f"{section_name} section missing Avg Price"
        assert "Best Bid:" in section, f"{section_name} section missing Best Bid"
        assert "Best Ask:" in section, f"{section_name} section missing Best Ask"
        assert "Best Market:" in section, f"{section_name} section missing Best Market"

    assert "Fair Value: ￥32,500" not in header_section
    assert "Fair Value: ￥32,800" in raw_section
    assert "￥31,100" in raw_section or "￥31,200" in raw_section or "￥31,650" in raw_section
    assert "￥28,000" in raw_section
    assert "￥32,800" in raw_section
    assert "￥30,500" in raw_section
    assert "Source URL: https://magi.example/m1" in raw_section
    assert "Fair Value: ￥98,000" in psa_section
    assert "￥95,000" in psa_section
    assert "￥88,000" in psa_section
    assert "￥98,000" in psa_section
    assert "￥92,000" in psa_section
    assert "Source URL: https://magi.example/p3" in psa_section
    assert "Offers:" not in text


def test_format_lookup_result_telegram_has_no_scan_or_note_noise() -> None:
    text = format_lookup_result_telegram(_mixed_grade_lookup_result())

    for forbidden in ("Image scan result", "Detected game", "Detected card", "Detected fields", "Note:", "sample note"):
        assert forbidden not in text, f"Telegram output should not contain {forbidden!r}: {text!r}"


def test_format_photo_lookup_result_is_identical_to_telegram_lookup_result() -> None:
    lookup_result = _mixed_grade_lookup_result()
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="Pikachu ex",
        aliases=(),
        card_number="132/106",
        rarity="SAR",
        set_code=None,
        raw_text="",
        extracted_lines=(),
    )
    outcome = TcgImageLookupOutcome(
        status="success",
        parsed=parsed,
        lookup_result=lookup_result,
        warnings=("sample warning",),
    )

    text = format_photo_lookup_result(outcome)

    assert text == format_lookup_result_telegram(lookup_result)
    for forbidden in ("Image scan result", "Detected game", "Detected card", "Detected fields", "sample warning", "sample note"):
        assert forbidden not in text


def test_format_lookup_result_telegram_detects_psa10_from_title_with_spacing() -> None:
    now = datetime.now(timezone.utc)
    lookup_result = TcgLookupResult(
        spec=TcgCardSpec(game="pokemon", title="Pikachu ex", card_number="132/106", rarity="SAR"),
        item=TrackedItem(item_id="x", item_type="card", category="tcg", title="Pikachu ex"),
        offers=(
            MarketOffer(
                source="magi",
                listing_id="p1",
                url="https://magi.example/p1",
                title="PSA 10 ピカチュウex 132/106",
                price_jpy=92000,
                price_kind="market",
                captured_at=now,
                source_category="marketplace",
                condition="graded",
            ),
        ),
        fair_value=None,
    )

    text = format_lookup_result_telegram(lookup_result)

    assert "PSA 10" in text
    assert "其他鑑定卡" not in text


def test_format_lookup_result_telegram_supports_sealed_box_products() -> None:
    now = datetime.now(timezone.utc)
    lookup_result = TcgLookupResult(
        spec=TcgCardSpec(game="pokemon", title="強化拡張パック ポケモンカード151", item_kind="sealed_box", set_code="sv2a"),
        item=TrackedItem(item_id="box151", item_type="tcg_sealed_box", category="tcg", title="強化拡張パック ポケモンカード151"),
        offers=(
            MarketOffer(
                source="cardrush_pokemon",
                listing_id="c1",
                url="https://cardrush.example/box151",
                title="強化拡張パック ポケモンカード151 未開封BOX",
                price_jpy=70800,
                price_kind="ask",
                captured_at=now,
                source_category="specialty_store",
                attributes={"product_kind": "sealed_box", "set_code": "sv2a"},
            ),
            MarketOffer(
                source="magi",
                listing_id="m1",
                url="https://magi.example/box151",
                title="強化拡張パック ポケモンカード151 未開封BOX",
                price_jpy=70000,
                price_kind="market",
                captured_at=now,
                source_category="marketplace",
                attributes={"product_kind": "sealed_box", "set_code": "sv2a"},
            ),
        ),
        fair_value=FairValueEstimate(item_id="box151", amount_jpy=70400, confidence=0.78, sample_count=2, reasoning=()),
    )

    text = format_lookup_result_telegram(lookup_result)

    assert "[pokemon sealed box] 強化拡張パック ポケモンカード151" in text
    assert "Raw" not in text
    assert "Fair Value:" in text
    assert "Best Ask:" in text
    assert "Best Market:" in text
    assert "Source URL: https://magi.example/box151" in text


def test_format_lookup_result_telegram_sealed_box_skips_mis_tagged_single_cards() -> None:
    """Regression for MEGA アビスアイ: cardrush returned single-card pages
    (¥1,880 / ¥1,930) that were mis-tagged as sealed boxes. The Telegram
    output must NOT surface those prices as a Fair Value / Best Ask for a
    sealed-box query."""
    now = datetime.now(timezone.utc)
    lookup_result = TcgLookupResult(
        spec=TcgCardSpec(
            game="pokemon", title="ポケモンカードゲーム MEGA アビスアイ",
            item_kind="sealed_box",
        ),
        item=TrackedItem(
            item_id="megaabyss", item_type="tcg_sealed_box", category="tcg",
            title="ポケモンカードゲーム MEGA アビスアイ",
        ),
        offers=(
            MarketOffer(
                source="cardrush_pokemon",
                listing_id="72035",
                url="https://www.cardrush-pokemon.jp/product/72035",
                title="MEGA アビスアイ SR 001/078",
                price_jpy=1880,
                price_kind="ask",
                captured_at=now,
                source_category="specialty_store",
                # Mis-tagged: parser saw "box" in surrounding text. Should be filtered out.
                attributes={"product_kind": "sealed_box"},
            ),
            MarketOffer(
                source="cardrush_pokemon",
                listing_id="72036",
                url="https://www.cardrush-pokemon.jp/product/72036",
                title="MEGA アビスアイ SR 002/078",
                price_jpy=1930,
                price_kind="ask",
                captured_at=now,
                source_category="specialty_store",
                attributes={"product_kind": "sealed_box"},
            ),
        ),
        fair_value=None,
    )

    text = format_lookup_result_telegram(lookup_result)

    assert "Fair Value: ￥1,930" not in text
    assert "Best Ask: ￥1,880" not in text
    assert "No sealed-box listings were found" in text


def test_format_liquidity_board_includes_reference_url() -> None:
    text = format_liquidity_board(_stub_board(), limit=1)

    assert "https://example.com/rank" in text
    assert "liq 88.20" in text
    assert "attn 41.70" in text
    assert "support 90.08" in text
    assert "buy-up" in text
    assert "stub methodology" not in text


def test_fallback_router_infers_ws_game_surrounded_by_cjk() -> None:
    intent = fallback_route_telegram_natural_language("查熱門ws前三")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 3


def test_fallback_router_infers_ws_game_with_arabic_limit() -> None:
    intent = fallback_route_telegram_natural_language("ws熱門前5")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 5


def test_fallback_router_infers_chinese_numeral_limit() -> None:
    intent = fallback_route_telegram_natural_language("pokemon 熱門前十")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "pokemon"
    assert intent.limit == 10


def test_fallback_router_defaults_limit_when_no_number_given() -> None:
    intent = fallback_route_telegram_natural_language("ws 熱門")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 5


def _llm_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "intent": "lookup_card",
        "game": None,
        "name": None,
        "card_number": None,
        "rarity": None,
        "set_code": None,
        "limit": None,
        "confidence": 0.6,
        "watch_query": None,
        "watch_price_threshold": None,
        "watch_id": None,
        "query_url": None,
        "sns_handle": None,
        "sns_keyword": None,
        "sns_buzz_query": None,
    }
    payload.update(overrides)
    return payload


def test_normalize_intent_recovers_union_arena_card_number_from_name() -> None:
    intent = _normalize_intent(
        _llm_payload(game="union_arena", name="uapr/eva-1-071 綾波レイ")
    )

    assert intent.intent == "lookup_card"
    assert intent.name == "綾波レイ"
    assert intent.card_number == "UAPR/EVA-1-071"
    assert intent.set_code == "uapr"


def test_normalize_intent_recovers_pokemon_card_number_and_rarity_from_name() -> None:
    intent = _normalize_intent(
        _llm_payload(game="pokemon", name="リザードンex 201/165 SAR")
    )

    assert intent.name == "リザードンex"
    assert intent.card_number == "201/165"
    assert intent.rarity == "SAR"


def test_normalize_intent_leaves_already_structured_fields_alone() -> None:
    intent = _normalize_intent(
        _llm_payload(
            game="union_arena",
            name="綾波レイ",
            card_number="UAPR/EVA-1-71",
            rarity="SR",
            set_code="uapr",
        )
    )

    assert intent.name == "綾波レイ"
    assert intent.card_number == "UAPR/EVA-1-71"
    assert intent.rarity == "SR"
    assert intent.set_code == "uapr"


def test_fallback_router_preserves_union_arena_card_number_in_name() -> None:
    intent = fallback_route_telegram_natural_language("查 UA卡 uapr/eva-1-071 綾波レイ 價格")

    assert intent is not None
    assert intent.intent == "lookup_card"
    assert intent.game == "union_arena"
    assert intent.card_number == "UAPR/EVA-1-071"
    assert intent.name == "綾波レイ"
    # `ua` must no longer be stripped out of the card-number prefix
    assert "pr/eva" not in intent.name.lower()


# ─── Text intent clarification (mirrors the photo flow) ───────────────────────


def _make_clarifying_processor(
    router_intent: TelegramNaturalLanguageIntent | None,
    *,
    research_renderer=lambda q: f"research:{q.query}",
    lookup_renderer=lambda q: f"lookup:{q.game}:{q.name}",
) -> TelegramCommandProcessor:
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lookup_renderer,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        research_renderer=research_renderer,
        natural_language_router=StubNaturalLanguageRouter(router_intent),
    )


def test_handle_telegram_message_clarifies_when_intent_is_unknown() -> None:
    # Stub returns unknown — _route_natural_language filters that out and the
    # fallback rules also return None for this neutral text, so the processor
    # must fall back to the clarification flow rather than the generic
    # "Unknown command" reply.
    client = FakeTelegramClient()
    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "今天心情很差"},
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    assert "請點按鈕" in replies[1]
    # Falls back to /search + /help options when nothing else fits.
    assert "上網搜尋" in replies[1]
    assert "/help" in replies[1]
    assert "都不是，請回答：否，[您的意圖]" in replies[1]
    assert processor.get_pending_text_clarification("123") is not None


def test_route_natural_language_prefers_generic_fast_path_over_llm() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(intent="sns_delete", sns_handle="example_tcg", confidence=0.9)
    )
    processor = _make_clarifying_processor(router.intent)
    processor._natural_language_router = router

    intent = processor._route_natural_language("把 @example_tcg 的 filter 全部拿掉")

    assert intent is not None
    assert intent.intent == "sns_clear_filter"
    assert intent.sns_handle == "example_tcg"
    assert router.seen_texts == []


def test_handle_telegram_message_clarifies_when_intent_confidence_is_low() -> None:
    # Confidence below the 0.55 threshold should trigger clarification even
    # when the LLM has a guess — exactly the "don't guess silently" rule that
    # already applies on the photo side.
    client = FakeTelegramClient()
    low_conf_intent = TelegramNaturalLanguageIntent(
        intent="web_research",
        research_query="why are pokemon cards popular",
        confidence=0.4,
    )
    processor = _make_clarifying_processor(low_conf_intent)

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "why are pokemon cards popular"},
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    reply = replies[1]
    assert "請點按鈕" in reply
    # The LLM's top guess shows up first.
    assert "1. 上網搜尋" in reply
    # Game keyword "pokemon" added a /price alternative.
    assert "/price" in reply
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    assert pending.original_text == "why are pokemon cards popular"
    assert pending.options[0].intent.intent == "web_research"


def test_handle_telegram_message_runs_selected_text_option_after_clarification() -> None:
    client = FakeTelegramClient()
    research_calls: list[str] = []

    def research_renderer(query):
        research_calls.append(query.query)
        return f"web:{query.query}"

    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
        research_renderer=research_renderer,
    )

    # Step 1: trigger the clarification options.
    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "ありがとう"},
    )
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    # Find which option is web_research so we exercise the right number.
    web_option = next(
        opt for opt in pending.options if opt.intent.intent == "web_research"
    )

    # Step 2: reply with that option number.
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": str(web_option.option_number)},
    )

    assert any(
        reply.startswith(f"收到，我就照第 {web_option.option_number} 個方式處理。")
        for reply in replies
    )
    assert research_calls == ["ありがとう"]
    assert "web:ありがとう" in replies[-1]
    # Pending state must be consumed after a successful selection.
    assert processor.get_pending_text_clarification("123") is None


_MERCARI_ITEM_URL = "https://jp.mercari.com/item/m68332847288"
_MERCARI_SHOPS_URL = "https://jp.mercari.com/shops/product/2JPEa5BQcDaCwxJamae4fp"


def test_build_text_intent_candidates_offers_research_and_snapshot_for_mercari_product_url() -> None:
    # The router judged a bare product link ambiguous (low confidence). The
    # clarification builder must surface exactly the two product reads — deep
    # research vs seller reputation — not the generic candidate spread.
    top = TelegramNaturalLanguageIntent(
        intent="product_research", query_url=_MERCARI_ITEM_URL, confidence=0.4
    )
    options = _build_text_intent_candidates(_MERCARI_ITEM_URL, top)

    assert [opt.intent.intent for opt in options] == [
        "product_research",
        "reputation_snapshot",
    ]
    assert all(opt.intent.query_url == _MERCARI_ITEM_URL for opt in options)
    assert "深度商品研究" in options[0].prompt
    assert "信譽快照" in options[1].prompt


def test_build_text_intent_candidates_shops_url_also_gets_the_pair() -> None:
    top = TelegramNaturalLanguageIntent(
        intent="reputation_snapshot", query_url=_MERCARI_SHOPS_URL, confidence=0.45
    )
    options = _build_text_intent_candidates(_MERCARI_SHOPS_URL, top)

    assert [opt.intent.intent for opt in options] == [
        "product_research",
        "reputation_snapshot",
    ]


def test_build_text_intent_candidates_pair_independent_of_top_intent_guess() -> None:
    # Live: qwen3:14b labels a bare Mercari URL `add_watch` @0.5 (below the
    # ambiguity threshold). The URL still reads as research-vs-snapshot, so the
    # builder offers that pair regardless of the router's low-confidence guess.
    top = TelegramNaturalLanguageIntent(intent="add_watch", confidence=0.5)
    options = _build_text_intent_candidates(_MERCARI_ITEM_URL, top)

    assert [opt.intent.intent for opt in options] == [
        "product_research",
        "reputation_snapshot",
    ]
    assert all(opt.intent.query_url == _MERCARI_ITEM_URL for opt in options)


def test_build_text_intent_candidates_non_mercari_url_keeps_generic_spread() -> None:
    # A plain article URL is NOT a product page, so the research/snapshot pair
    # must not be forced — the generic reputation/web_research spread applies.
    text = "https://example.com/some-article"
    top = TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1)
    options = _build_text_intent_candidates(text, top)

    kinds = [opt.intent.intent for opt in options]
    assert "product_research" not in kinds
    assert "reputation_snapshot" in kinds


def test_bare_mercari_product_url_offers_two_button_choice() -> None:
    # End-to-end: a bare product URL the LLM marks ambiguous (confidence < 0.55)
    # must produce the 2-option clarification, NOT silently run one command.
    client = FakeTelegramClient()
    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(
            intent="product_research", query_url=_MERCARI_ITEM_URL, confidence=0.4
        ),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": _MERCARI_ITEM_URL},
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    menu = replies[1]
    assert "請點按鈕" in menu
    assert "深度商品研究" in menu
    assert "信譽快照" in menu
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    assert [opt.intent.intent for opt in pending.options] == [
        "product_research",
        "reputation_snapshot",
    ]


def test_selecting_research_option_dispatches_to_research_command() -> None:
    # Picking the research option must build a background plan that runs the
    # injected /research command with the product URL and the chat id.
    client = FakeTelegramClient()
    research_calls: list[tuple[str, str]] = []

    def research_handler(remainder: str, chat_id: str) -> str:
        research_calls.append((remainder, chat_id))
        return f"研究報告：{remainder}"

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda q: "unused",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=StubNaturalLanguageRouter(
            TelegramNaturalLanguageIntent(
                intent="product_research", query_url=_MERCARI_ITEM_URL, confidence=0.4
            )
        ),
        command_handlers={
            "/research": RegisteredCommand(
                research_handler, ack="收到，開始研究…", background=True
            )
        },
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": _MERCARI_ITEM_URL},
    )
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    research_option = next(
        opt for opt in pending.options if opt.intent.intent == "product_research"
    )

    # Research is slow → background plan: ack is sent, the factory deferred.
    plan = processor.build_pending_text_reply_plan(
        chat_id="123", text=str(research_option.option_number)
    )
    assert plan is not None
    assert plan.run_in_background is True
    assert "收到，開始研究…" in plan.ack
    assert research_calls == []  # not run until the factory executes
    assert plan.execute() == f"研究報告：{_MERCARI_ITEM_URL}"
    assert research_calls == [(_MERCARI_ITEM_URL, "123")]
    # Selecting an option consumes the pending clarification.
    assert processor.get_pending_text_clarification("123") is None


def test_handle_telegram_message_text_clarification_other_option_requests_override() -> None:
    client = FakeTelegramClient()
    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "在嗎"},
    )
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    sentinel = len(pending.options) + 1

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": str(sentinel)},
    )

    assert replies == (TEXT_INTAKE_ACK, "好，請直接回答：否，[您的意圖]")
    # Pending state remains so the follow-up "否，..." can be matched.
    assert processor.get_pending_text_clarification("123") is not None


def test_handle_telegram_message_text_clarification_override_reroutes_through_router() -> None:
    client = FakeTelegramClient()
    captured: list[TelegramLookupQuery] = []

    def lookup_renderer(query):
        captured.append(query)
        return f"lookup:{query.game}:{query.name}"

    # The first message is unknown; the override text "查 pokemon Pikachu ex"
    # then comes back through the router and gets a confident lookup intent.
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1)
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lookup_renderer,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "蛤"},
    )
    assert processor.get_pending_text_clarification("123") is not None

    # Re-target the stub so the override path gets a confident answer.
    router.intent = TelegramNaturalLanguageIntent(
        intent="lookup_card",
        game="pokemon",
        name="Pikachu ex",
        confidence=0.92,
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "否，查 pokemon Pikachu ex"},
    )

    assert any("收到，我改照你補充的意思處理：查 pokemon Pikachu ex" in r for r in replies)
    assert any(r.endswith("lookup:pokemon:Pikachu ex") for r in replies)
    assert captured and captured[0].game == "pokemon" and captured[0].name == "Pikachu ex"
    # Pending state was popped and not re-established because the override was confident.
    assert processor.get_pending_text_clarification("123") is None


# ─── Per-intent fallback router behaviour (locks in the slim from this change) ─


CANONICAL_FALLBACK_PHRASES: tuple[tuple[str, str], ...] = (
    # trend_board — must require an explicit ranking word, never plain "趨勢".
    ("pokemon 熱門前 5", "trend_board"),
    ("ws trending top 3", "trend_board"),
    ("幫我看 pokemon liquidity", "trend_board"),
    ("遊戲王最近熱門排行", "trend_board"),
    # web_research — only canonical question phrasings stay reliable in fallback.
    ("為什麼Pokemon卡這麼貴", "web_research"),
    ("why are pokemon cards popular", "web_research"),
    # watch / list / remove / update — touched as little as possible.
    ("追蹤 初音ミク SSP 5萬以下", "add_watch"),
    ("我的追蹤清單", "list_watches"),
    ("取消追蹤 abc12345", "remove_watch"),
    ("把 abc12345 改成 4萬", "update_watch_price"),
    # reputation, sns — untouched lists, still work.
    ("信用 https://jp.mercari.com/user/profile/123", "reputation_snapshot"),
    ("追蹤 @example_news", "sns_add_account"),
    ("snslist", "sns_list"),
    ("取消追蹤 @example_news", "sns_delete"),
    # sns_buzz — slimmed; canonical phrasings keep working, generic
    # "什麼熱門" no longer over-eats (asserted separately below).
    ("snsbuzz amd", "sns_buzz"),
    ("整理 amd 最近的熱門討論", "sns_buzz"),
    ("buzz on amd", "sns_buzz"),
    # status / tools — only canonical roots kept (狀態 / health / 工具 / catalog / 功能清單).
    ("目前狀態", "status"),
    ("服務 health", "status"),
    ("tools", "tools"),
    ("catalog", "tools"),
    ("功能清單", "tools"),
    # opportunity_remove — canonical phrasings.
    ("移除機會目標 2", "opportunity_remove"),
    ("remove target 3", "opportunity_remove"),
    ("I am not interested in Umbreon ex SAR anymore", "opportunity_remove"),
)


def test_fallback_router_canonical_phrases_keep_routing() -> None:
    for phrase, expected_intent in CANONICAL_FALLBACK_PHRASES:
        result = fallback_route_telegram_natural_language(phrase)
        assert result is not None, f"phrase={phrase!r} unexpectedly returned None"
        assert result.intent == expected_intent, (
            f"phrase={phrase!r} routed to {result.intent}, expected {expected_intent}"
        )


def test_fallback_router_does_not_route_bare_trend_keyword_to_trend_board() -> None:
    # "趨勢" was removed from _TREND_KEYWORDS because in Chinese it usually
    # means "direction", not "ranking". A bare "最近趨勢" should fall through
    # to the LLM (or to clarification options) rather than fire a confident
    # trend_board fallback.
    assert fallback_route_telegram_natural_language("最近趨勢") is None


def test_fallback_router_no_longer_overeats_with_sns_buzz() -> None:
    # Before the slim "遊戲王最近熱門排行" matched _SNS_BUZZ_KEYWORDS via
    # "什麼熱門"/"最近熱門" and routed to sns_buzz despite the clear ranking
    # intent. After dropping those entries from _SNS_BUZZ_KEYWORDS it goes
    # to trend_board as the user would expect.
    result = fallback_route_telegram_natural_language("遊戲王最近熱門排行")
    assert result is not None
    assert result.intent == "trend_board"
    assert result.game == "yugioh"


def test_handle_photo_message_installs_clarification_when_renderer_returns_unresolved() -> None:
    # End-to-end check that when the photo renderer returns a
    # PhotoLookupReply with a pending_clarification (the
    # unresolved/rejected_sanity path), the dispatcher both replies with the
    # clarification text AND installs the pending photo state so the user's
    # next text message can use the "否，[您的意圖]" override flow.
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    def stub_photo_renderer(query):
        pending = PendingTelegramPhotoClarification(
            chat_id=str(query.chat_id),
            image_path=query.image_path,
            caption=query.caption,
            file_id=query.file_id,
            options=(),
            parsed_game="pokemon",
            parsed_item_kind="card",
            parsed_title=None,
        )
        return PhotoLookupReply(
            text="我看不太清楚這張卡。可以直接告訴我卡片名稱嗎？",
            pending_clarification=pending,
        )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=stub_photo_renderer,
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "/scan pokemon",  # would normally trigger direct lookup
        },
    )

    assert any("我看不太清楚這張卡" in reply for reply in replies)
    # The pending photo state must be installed so a follow-up "否，..."
    # message reaches the existing override flow.
    pending = processor.get_pending_photo_clarification("123")
    assert pending is not None
    assert pending.parsed_game == "pokemon"


# ── Callback-query dispatch (inline button: ❌ 刪除追蹤) ───────────────────────

class _FakeAccountRule:
    """Stand-in for sns_monitor.models.AccountWatch used only by the
    callback-query dispatch tests below (the real model lives in a separate
    package that this bot's venv does not import)."""

    def __init__(self, rule_id: str, screen_name: str) -> None:
        self.rule_id = rule_id
        self.screen_name = screen_name
        self.include_keywords: tuple[str, ...] = ()
        self.query: str | None = None


class _FakeSnsDatabase:
    def __init__(self, rules: list[_FakeAccountRule]) -> None:
        self._rules: list[_FakeAccountRule] = list(rules)
        self.deleted: list[str] = []

    def list_watch_rules(self):
        return list(self._rules)

    def delete_watch_rule(self, rule_id: str) -> bool:
        for idx, rule in enumerate(self._rules):
            if rule.rule_id == rule_id:
                self._rules.pop(idx)
                self.deleted.append(rule_id)
                return True
        return False


def _processor_with_sns_db(sns_db) -> TelegramCommandProcessor:
    """Build a TelegramCommandProcessor wired with inline SNS registry handlers.

    The SNS command/callback handlers are now in aka_no_claw's openclaw_adapter
    (not importable from this venv), so we inline minimal equivalents here so
    the dispatch-mechanism tests remain valid.
    """
    from price_monitor_bot.list_view import LIST_VIEW_MODE_READ, ListRow, build_list_view

    # -- inline SNS list view (mirrors build_snslist_view_fn logic) -----------
    def _sl_view(*, page: int = 0, mode: str = LIST_VIEW_MODE_READ):
        if sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。", None, 0
        rules = list(sns_db.list_watch_rules())
        rules.sort(key=lambda r: (not getattr(r, "enabled", True), r.rule_id))
        items = []
        for rule in rules:
            source = getattr(rule, "source", "x")
            screen_name = getattr(rule, "screen_name", None)
            query_text = getattr(rule, "query", None)
            category = getattr(rule, "category", None)
            if screen_name:
                handle_display = f"r/{screen_name}" if source == "reddit" else f"@{screen_name}"
                include_kw = getattr(rule, "include_keywords", ()) or ()
                filters = f" filter[{', '.join(include_kw)}]" if include_kw else ""
                info, short = f"{handle_display}{filters}", handle_display
            elif query_text:
                info, short = f'"{query_text}"', f'"{query_text[:18]}"'
            elif category:
                info = short = f"Trend:{category}"
            else:
                info = "Unknown"; short = rule.rule_id[:8]
            domains = getattr(rule, "domains", ())
            domain_seg = f" domain[{', '.join(domains)}]" if domains else " domain[?]"
            sched_seg = (f" schedule:{rule.schedule_minutes}m"
                         if getattr(rule, "schedule_minutes", None) else "")
            status = "✓" if getattr(rule, "enabled", True) else "✗"
            text_block = (f"  {status} [{source}] {info}{domain_seg}{sched_seg}"
                          f" ({rule.rule_id[:8]}…)")
            items.append(ListRow(id=rule.rule_id, text=text_block, short_label=short))
        return build_list_view(
            list_kind="sl", items=items, page=page, mode=mode,
            list_title="📋 SNS 監控規則",
            empty_message="尚無 SNS 監控規則。\n用法：/snsadd @username",
        )

    # -- inline snsdel callback (mirrors build_snsdel_callback_handler logic) -
    def _snsdel_cb(payload: str, original_text: str, chat_id: str):
        handle = payload.lstrip("@")
        rules = list(sns_db.list_watch_rules())
        rule_id = None
        for rule in rules:
            if (rule.rule_id == handle or rule.rule_id.startswith(handle)
                    or getattr(rule, "screen_name", "").lower() == handle.lower()):
                rule_id = rule.rule_id
                break
        if rule_id is None:
            return (f"已經不在追蹤 @{handle}",
                    f"{original_text}\n\n✓ 已刪除 @{handle}（先前已移除）", None)
        found = sns_db.delete_watch_rule(rule_id)
        if found:
            return f"已刪除 @{handle}", f"{original_text}\n\n✓ 已刪除 @{handle}", None
        return (f"已經不在追蹤 @{handle}",
                f"{original_text}\n\n✓ 已刪除 @{handle}（先前已移除）", None)

    # -- inline sl deleter ----------------------------------------------------
    def _sl_deleter(rule_id: str) -> bool:
        return bool(sns_db.delete_watch_rule(rule_id)) if sns_db else False

    class _SnsProcessor(TelegramCommandProcessor):
        """Thin subclass adding render_snslist_view for backward-compat in tests."""
        def render_snslist_view(self, *, page: int = 0, mode: str = LIST_VIEW_MODE_READ):
            return _sl_view(page=page, mode=mode)

    return _SnsProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        sns_db=sns_db,
        callback_handlers={"snsdel": _snsdel_cb},
        view_handlers={"sl": _sl_view},
        item_deleter_handlers={"sl": (_sl_deleter, "SNS 規則")},
    )


def _callback_update(*, chat_id: str, data: str, text: str = "🔎 自動加入追蹤 @foo") -> dict:
    return {
        "id": "cbq-1",
        "data": data,
        "message": {
            "message_id": 42,
            "chat": {"id": chat_id},
            "text": text,
        },
    }


def test_callback_query_snsdel_deletes_rule_and_edits_message() -> None:
    sns_db = _FakeSnsDatabase([_FakeAccountRule("abc12345", "foo")])
    processor = _processor_with_sns_db(sns_db)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="snsdel:foo"),
    )

    assert sns_db.deleted == ["abc12345"]
    assert len(client.edited_messages) == 1
    edited = client.edited_messages[0]
    assert edited["message_id"] == 42
    assert "✓ 已刪除 @foo" in edited["text"]
    assert edited["reply_markup"] is None  # keyboard cleared
    assert len(client.answered_callbacks) == 1
    assert "已刪除" in client.answered_callbacks[0]["text"]


def test_callback_query_uses_async_handler_hook_and_returns_early() -> None:
    client = FakeTelegramClient()

    class _AsyncProcessor(TelegramCommandProcessor):
        def handle_callback_query_async(self, **kwargs):
            client.answer_callback_query(
                callback_query_id=kwargs["callback_id"],
                text="背景處理中",
            )
            return True

    processor = _AsyncProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="goal:__goal_confirm__"),
    )

    assert len(client.answered_callbacks) == 1
    assert client.answered_callbacks[0]["text"] == "背景處理中"
    assert client.edited_messages == []


def _setup_snsfb_processor(tmp_path):
    """Build a processor whose _sns_db is a real SnsDatabase pointing at a
    tmp file, seeded with one AccountWatch rule. Returns (processor, db, rule_id).

    The sns_monitor package is only available in aka_no_claw's combined venv
    (where the whole stack runs). When running price_monitor_bot's own venv
    standalone the import fails — skip the test in that case rather than
    fail."""
    sns_monitor = pytest.importorskip("sns_monitor")
    from sns_monitor.models import AccountWatch
    from sns_monitor.storage import SnsDatabase
    db = SnsDatabase(tmp_path / "snsfb_test.sqlite3")
    db.bootstrap()
    rule = AccountWatch(
        rule_id="r1234567890",
        screen_name="snsuser",
        user_id=None,
        label="snsuser",
        include_keywords=(),
        domains=(),
        enabled=True,
        schedule_minutes=60,
        chat_id="123",
    )
    db.save_watch_rule(rule)

    def _snsfb_cb(payload: str, original_text: str, chat_id: str) -> tuple:
        parts = payload.split(":", 2)
        if len(parts) != 3 or parts[0] not in {"up", "down", "bought"}:
            return "未知回饋", None, None
        kind, tweet_id, rule_id = parts
        db_path = getattr(db, "path", None)
        if db_path is None:
            return "SNS monitor 未啟用，無法寫入回饋", None, None
        from sns_monitor.feedback import record_sns_feedback
        result = record_sns_feedback(db=db, tweet_id=tweet_id, rule_id=rule_id,
                                     chat_id=str(chat_id), kind=kind)
        if result.get("status") != "ok":
            return f"記錄失敗：{result.get('reason', 'unknown')}", None, None
        side_effects = list(result.get("side_effects") or ())
        if kind == "up":
            toast = "✓ 已記錄 👍（已提高同類推文推播機率）"
        elif kind == "bought":
            toast = "✓ 已記錄 💰（已提高同類推文推播機率）"
        else:
            if "rule_disabled" in side_effects:
                toast = "✓ 已標記不感興趣（累計過閾值，rule 自動停用）"
            else:
                toast = "✓ 已標記不感興趣（24h cooldown）"
        return toast, f"{original_text}\n\n{toast}", None

    proc = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        sns_db=db,
        callback_handlers={"snsfb": _snsfb_cb},
    )
    return proc, db, rule.rule_id


def test_callback_query_snsfb_up_writes_db_and_clears_keyboard(tmp_path) -> None:
    import sqlite3
    processor, db, rule_id = _setup_snsfb_processor(tmp_path)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-snsfb-up",
            "data": f"snsfb:up:tw-111:{rule_id}",
            "message": {
                "message_id": 5,
                "chat": {"id": "123"},
                "text": "🐦 X 帳號通知 — @snsuser",
            },
        },
    )

    assert "✓ 已記錄 👍" in client.answered_callbacks[0]["text"]
    edited = client.edited_messages[0]
    assert "✓ 已記錄 👍" in edited["text"]
    assert edited["reply_markup"] is None  # keyboard cleared
    # Feedback row persisted
    with sqlite3.connect(db.path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM sns_post_feedback WHERE tweet_id = ?", ("tw-111",)
        ).fetchone()
    assert row is not None and row["feedback_kind"] == "up"


def test_callback_query_snsfb_down_shows_cooldown_toast(tmp_path) -> None:
    processor, db, rule_id = _setup_snsfb_processor(tmp_path)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-snsfb-down",
            "data": f"snsfb:down:tw-222:{rule_id}",
            "message": {
                "message_id": 5,
                "chat": {"id": "123"},
                "text": "🐦 X 帳號通知 — @snsuser",
            },
        },
    )

    assert "24h cooldown" in client.answered_callbacks[0]["text"]
    refreshed = db.get_watch_rule(rule_id)
    assert refreshed.cooldown_until is not None
    assert refreshed.enabled is True  # not yet disabled


def test_callback_query_snsfb_bought_shortens_schedule(tmp_path) -> None:
    processor, db, rule_id = _setup_snsfb_processor(tmp_path)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-snsfb-bought",
            "data": f"snsfb:bought:tw-333:{rule_id}",
            "message": {
                "message_id": 5,
                "chat": {"id": "123"},
                "text": "🐦 X 帳號通知 — @snsuser",
            },
        },
    )

    toast = client.answered_callbacks[0]["text"]
    assert "💰" in toast
    assert "✓ 已記錄 💰" in toast


def test_callback_query_snsfb_invalid_kind_rejected(tmp_path) -> None:
    processor, _db, rule_id = _setup_snsfb_processor(tmp_path)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-snsfb-bad",
            "data": f"snsfb:nope:tw-444:{rule_id}",
            "message": {
                "message_id": 5,
                "chat": {"id": "123"},
                "text": "🐦 X 帳號通知 — @snsuser",
            },
        },
    )
    assert client.answered_callbacks[0]["text"] == "未知回饋"


def test_callback_query_snsdel_idempotent_when_rule_already_gone() -> None:
    sns_db = _FakeSnsDatabase([])  # no rules
    processor = _processor_with_sns_db(sns_db)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="snsdel:foo"),
    )

    assert sns_db.deleted == []
    # Still edits the source message + answers the callback so the user
    # sees a confirmation either way.
    assert len(client.edited_messages) == 1
    assert "已刪除 @foo" in client.edited_messages[0]["text"]
    assert client.answered_callbacks
    assert "已經不在追蹤" in client.answered_callbacks[0]["text"]


def test_callback_query_from_unauthorized_chat_is_rejected() -> None:
    sns_db = _FakeSnsDatabase([_FakeAccountRule("abc12345", "foo")])
    processor = _processor_with_sns_db(sns_db)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="999", data="snsdel:foo"),
    )

    # Rule must not be touched, source message must not be edited.
    assert sns_db.deleted == []
    assert client.edited_messages == []
    # We still answer the callback (Telegram requires it within ~15s to
    # remove the loading spinner on the button), just without a toast.
    assert client.answered_callbacks
    assert client.answered_callbacks[0]["text"] is None


# ── Paginated /snslist view (read mode + edit mode + callback nav) ────────────

def _make_snslist_processor(handles: list[str], *, enabled: bool = True) -> tuple[TelegramCommandProcessor, _FakeSnsDatabase]:
    rules = [_FakeAccountRule(f"rid{i:04x}", handle) for i, handle in enumerate(handles)]
    if not enabled:
        for r in rules:
            r.enabled = False  # type: ignore[attr-defined]
    db = _FakeSnsDatabase(rules)
    proc = _processor_with_sns_db(db)
    return proc, db


# `enabled` is read by render_snslist_view sort key — patch FakeAccountRule.
_FakeAccountRule.enabled = True  # type: ignore[attr-defined]
_FakeAccountRule.domains = ()  # type: ignore[attr-defined]


def test_snslist_view_read_mode_no_delete_buttons() -> None:
    """Read mode shows text only; keyboard has no per-row delete rows."""
    processor, _ = _make_snslist_processor(["alice", "bob", "carol"])

    text, kb, page = processor.render_snslist_view()

    assert "📋 SNS 監控規則" in text
    assert "第 1/1 頁" in text
    assert "共 3 筆" in text
    assert "@alice" in text and "@bob" in text and "@carol" in text
    assert page == 0
    assert kb is not None
    # Only one row of buttons (navigation), no delete rows.
    assert len(kb["inline_keyboard"]) == 1
    nav_buttons = kb["inline_keyboard"][0]
    button_texts = {b["text"] for b in nav_buttons}
    assert "✏️ 編輯" in button_texts
    assert "✖️ 關閉" in button_texts
    # Single page: no prev/next.
    assert "⬅️ 上頁" not in button_texts
    assert "下頁 ➡️" not in button_texts


def test_snslist_view_edit_mode_has_one_delete_button_per_visible_row() -> None:
    processor, _ = _make_snslist_processor(["alice", "bob", "carol"])

    text, kb, _ = processor.render_snslist_view(mode="e")

    # 3 delete-button rows + 1 navigation row
    assert len(kb["inline_keyboard"]) == 4
    delete_rows = kb["inline_keyboard"][:3]
    delete_data = [row[0]["callback_data"] for row in delete_rows]
    assert delete_data == ["del:sl:rid0000", "del:sl:rid0001", "del:sl:rid0002"]
    assert all("❌ 刪除" in row[0]["text"] for row in delete_rows)
    nav_button_texts = {b["text"] for b in kb["inline_keyboard"][-1]}
    assert "✓ 完成" in nav_button_texts


def test_snslist_view_pagination_boundaries_first_and_last_page() -> None:
    handles = [f"user{i:02d}" for i in range(12)]  # 12 items → 3 pages of 5/5/2
    processor, _ = _make_snslist_processor(handles)

    # Page 0
    _, kb0, page0 = processor.render_snslist_view(page=0)
    nav0 = {b["text"] for b in kb0["inline_keyboard"][-1]}
    assert page0 == 0
    assert "⬅️ 上頁" not in nav0  # no prev on first page
    assert "下頁 ➡️" in nav0

    # Page 2 (last)
    _, kb2, page2 = processor.render_snslist_view(page=2)
    nav2 = {b["text"] for b in kb2["inline_keyboard"][-1]}
    assert page2 == 2
    assert "⬅️ 上頁" in nav2
    assert "下頁 ➡️" not in nav2  # no next on last page

    # Out-of-bounds page is clamped
    _, _, clamped = processor.render_snslist_view(page=99)
    assert clamped == 2


def test_callback_query_pg_renders_target_page_without_db_mutation() -> None:
    """A `pg:sl:1:e` tap should edit_message_text the page-2 view and NOT
    touch the database."""
    handles = [f"u{i:02d}" for i in range(8)]  # 8 → 2 pages
    processor, db = _make_snslist_processor(handles)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-1",
            "data": "pg:sl:1:e",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "stale text — will be replaced",
            },
        },
    )

    assert db.deleted == []  # navigation must not delete anything
    assert len(client.edited_messages) == 1
    edited = client.edited_messages[0]
    assert "第 2/2 頁" in edited["text"]
    assert edited["reply_markup"] is not None  # has keyboard
    assert client.answered_callbacks
    assert client.answered_callbacks[0]["text"] is None


def test_callback_query_del_sl_removes_row_and_rerenders_same_page() -> None:
    handles = [f"u{i:02d}" for i in range(8)]
    processor, db = _make_snslist_processor(handles)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-2",
            "data": "del:sl:rid0003",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "📋 SNS 監控規則  第 1/2 頁（共 8 筆）\n...",
            },
        },
    )

    assert db.deleted == ["rid0003"]
    edited = client.edited_messages[0]
    # After deletion total is 7 → 2 pages of 5/2; we were on page 0, still page 0.
    assert "第 1/2 頁" in edited["text"]
    assert "共 7 筆" in edited["text"]
    # Re-render is in edit mode so we can keep deleting.
    nav_buttons = {b["text"] for b in edited["reply_markup"]["inline_keyboard"][-1]}
    assert "✓ 完成" in nav_buttons
    assert client.answered_callbacks[0]["text"] == "✓ 已刪除"


def test_callback_query_del_sl_last_row_on_last_page_jumps_back() -> None:
    """If the deleted row was the only item on the last page, the re-render
    should drop back to the previous page rather than showing an empty page."""
    handles = [f"u{i:02d}" for i in range(6)]  # 6 → page 0 has 5, page 1 has 1
    processor, db = _make_snslist_processor(handles)
    client = FakeTelegramClient()

    # The only item on page 1 is "rid0005" (sorted by rule_id).
    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-3",
            "data": "del:sl:rid0005",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "📋 SNS 監控規則  第 2/2 頁（共 6 筆）\n...",
            },
        },
    )

    assert db.deleted == ["rid0005"]
    edited = client.edited_messages[0]
    # 5 items left → 1 page. After deleting the lone item on page 2, we land
    # on page 1.
    assert "第 1/1 頁" in edited["text"]
    assert "共 5 筆" in edited["text"]


def test_callback_query_close_sl_clears_keyboard() -> None:
    processor, db = _make_snslist_processor(["alice"])
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-4",
            "data": "close:sl",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "📋 SNS 監控規則  第 1/1 頁",
            },
        },
    )

    assert db.deleted == []
    edited = client.edited_messages[0]
    assert "（已關閉）" in edited["text"]
    assert edited["reply_markup"] is None


def test_snslist_view_empty_shows_friendly_text_and_no_keyboard() -> None:
    processor, _ = _make_snslist_processor([])
    text, kb, page = processor.render_snslist_view()
    assert "尚無 SNS 監控規則" in text
    assert kb is None
    assert page == 0


# ── Paginated /watchlist view ─────────────────────────────────────────────────

class _FakeMercariWatch:
    """Lightweight stand-in for MarketplaceWatch (v2 markets-array shape).
    Attribute name kept for legacy tests; new tests should construct a real
    MarketplaceWatch directly."""

    def __init__(
        self,
        watch_id: str,
        query: str,
        *,
        price: int = 5000,
        enabled: bool = True,
        condition_ids: tuple[int, ...] = (1, 2, 3),
        markets: tuple[str, ...] = ("mercari",),
    ) -> None:
        self.watch_id = watch_id
        self.markets = markets
        self.query = query
        self.price_threshold_jpy = price
        self.enabled = enabled
        self.chat_id = "0"
        self.last_checked_at = None
        self.created_at = "2026-01-01"
        self.updated_at = "2026-01-01"
        self.market_options: dict[str, dict] = {}
        for m in markets:
            self.market_options[m] = (
                {"condition_ids": list(condition_ids)} if m == "mercari" else {}
            )

    def options_for(self, market: str) -> dict:
        return dict(self.market_options.get(market) or {})


class _FakeWatchDatabase:
    def __init__(self, watches: list[_FakeMercariWatch]) -> None:
        self._watches = list(watches)
        self.deleted: list[str] = []

    def list_marketplace_watchlist(self, *, market: str | None = None):
        if market is None:
            return list(self._watches)
        return [w for w in self._watches if market in w.markets]

    def delete_marketplace_watch(self, watch_id: str) -> bool:
        for idx, w in enumerate(self._watches):
            if w.watch_id == watch_id:
                self._watches.pop(idx)
                self.deleted.append(watch_id)
                return True
        return False


def _make_watchlist_processor(queries: list[str]) -> tuple[TelegramCommandProcessor, _FakeWatchDatabase]:
    watches = [_FakeMercariWatch(f"wid{i:04x}", q) for i, q in enumerate(queries)]
    db = _FakeWatchDatabase(watches)
    proc = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        watch_db=db,
    )
    return proc, db


def test_watchlist_view_renders_with_per_item_delete_in_edit_mode() -> None:
    processor, _ = _make_watchlist_processor(["pikachu", "charizard", "umbreon"])

    text_read, kb_read, _ = processor.render_watchlist_view()
    assert "Marketplace 追蹤" in text_read
    assert "共 3 筆" in text_read
    # Read mode: 1 nav row only
    assert len(kb_read["inline_keyboard"]) == 1

    text_edit, kb_edit, _ = processor.render_watchlist_view(mode="e")
    # 3 items × (1 label row + 1 action row) + 1 nav row = 7 rows
    assert len(kb_edit["inline_keyboard"]) == 7
    # Label rows are at even indices (0, 2, 4); action rows at odd (1, 3, 5)
    label_data = [kb_edit["inline_keyboard"][i][0]["callback_data"] for i in (0, 2, 4)]
    assert label_data == ["noop", "noop", "noop"]
    delete_data = [kb_edit["inline_keyboard"][i][0]["callback_data"] for i in (1, 3, 5)]
    assert delete_data == ["del:wl:wid0000", "del:wl:wid0001", "del:wl:wid0002"]


def test_callback_query_del_wl_removes_watch_and_rerenders() -> None:
    processor, db = _make_watchlist_processor([f"q{i:02d}" for i in range(8)])
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-wl",
            "data": "del:wl:wid0003",
            "message": {
                "message_id": 11,
                "chat": {"id": "123"},
                "text": "📋 Marketplace 追蹤  第 1/2 頁（共 8 筆）\n…",
            },
        },
    )

    assert db.deleted == ["wid0003"]
    edited = client.edited_messages[0]
    assert "第 1/2 頁" in edited["text"]
    assert "共 7 筆" in edited["text"]
    assert client.answered_callbacks[0]["text"] == "✓ 已刪除"


# ── /watchlist 編輯模式的「🎛 設定狀態」按鈕 + cond: callback dispatch ─────────

def test_watchlist_edit_mode_each_row_has_delete_and_condition_buttons() -> None:
    processor, _ = _make_watchlist_processor(["alpha", "beta"])
    _, kb, _ = processor.render_watchlist_view(mode="e")
    # 2 watches × (1 label + 1 action) + 1 nav = 5 rows
    assert len(kb["inline_keyboard"]) == 5
    # Row 0 = label for item 0 (noop), Row 1 = action for item 0
    label_row = kb["inline_keyboard"][0]
    assert label_row[0]["callback_data"] == "noop"
    assert "alpha" in label_row[0]["text"]
    first_watch_row = kb["inline_keyboard"][1]
    assert len(first_watch_row) == 2
    assert first_watch_row[0]["callback_data"] == "del:wl:wid0000"
    assert first_watch_row[1]["callback_data"] == "wedit:wid0000"
    assert "✏️" in first_watch_row[1]["text"]


def _setup_watchlist_callback(client_cls=FakeTelegramClient):
    """Helper: build processor + client with one watch and a mutable DB
    that simulates condition_ids updates."""

    class _DB:
        def __init__(self):
            self._w = _FakeMercariWatch("wid0000", "alpha")
            self.update_calls: list[dict] = []

        def list_marketplace_watchlist(self, *, market: str | None = None):
            if market is None or market in self._w.markets:
                return [self._w]
            return []

        def get_marketplace_watch(self, watch_id):
            return self._w if watch_id == self._w.watch_id else None

        def delete_marketplace_watch(self, watch_id):
            if watch_id == self._w.watch_id:
                return True
            return False

        def update_marketplace_watch(self, watch_id, *, market_options=None, **_kwargs):
            self.update_calls.append({"watch_id": watch_id, "market_options": market_options})
            if market_options is not None:
                self._w.market_options = {
                    k: dict(v) for k, v in market_options.items()
                }
            return True

    db = _DB()
    proc = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        watch_db=db,
    )
    return proc, db, client_cls()


def test_callback_query_cond_open_renders_picker_with_six_checkboxes() -> None:
    processor, db, client = _setup_watchlist_callback()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-cond-open",
            "data": "cond:wid0000:open",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "📋 Marketplace 追蹤（多站）  第 1/1 頁（共 1 筆）",
            },
        },
    )

    assert db.update_calls == []  # opening shouldn't write
    edited = client.edited_messages[0]
    assert "🎛 設定狀態" in edited["text"]
    assert "alpha" in edited["text"]
    kb = edited["reply_markup"]["inline_keyboard"]
    # 6 condition rows + 1 done row
    assert len(kb) == 7
    callback_data = [row[0]["callback_data"] for row in kb[:6]]
    assert callback_data == [
        "cond:wid0000:t:1", "cond:wid0000:t:2", "cond:wid0000:t:3",
        "cond:wid0000:t:4", "cond:wid0000:t:5", "cond:wid0000:t:6",
    ]
    # Top 3 should already be checked.
    assert "☑" in kb[0][0]["text"]
    assert "☑" in kb[1][0]["text"]
    assert "☑" in kb[2][0]["text"]
    assert "☐" in kb[3][0]["text"]
    assert kb[6][0]["callback_data"] == "cond:wid0000:done"


def test_callback_query_cond_toggle_adds_new_condition_and_persists() -> None:
    processor, db, client = _setup_watchlist_callback()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-cond-t",
            "data": "cond:wid0000:t:4",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "🎛 設定狀態：alpha\n目前接受：…",
            },
        },
    )

    assert db.update_calls == [
        {"watch_id": "wid0000",
         "market_options": {"mercari": {"condition_ids": [1, 2, 3, 4]}}}
    ]
    # Re-rendered picker should now show 4 boxes ticked.
    edited = client.edited_messages[0]
    kb = edited["reply_markup"]["inline_keyboard"]
    assert "☑" in kb[3][0]["text"]  # ID 4 now ticked


def test_callback_query_cond_toggle_refuses_to_empty_all_conditions() -> None:
    processor, db, client = _setup_watchlist_callback()
    # Pre-shrink to only ID 3 so the next toggle would clear all.
    processor._watch_db._w.market_options = {"mercari": {"condition_ids": [3]}}

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-cond-empty",
            "data": "cond:wid0000:t:3",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "🎛 設定狀態：alpha\n…",
            },
        },
    )

    # DB must NOT have been touched.
    assert db.update_calls == []
    assert client.answered_callbacks[0]["text"] == "至少要保留一個狀態"


def test_callback_query_cond_done_returns_to_watchlist_edit_mode() -> None:
    processor, db, client = _setup_watchlist_callback()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-cond-done",
            "data": "cond:wid0000:done",
            "message": {
                "message_id": 7,
                "chat": {"id": "123"},
                "text": "🎛 設定狀態：alpha\n…",
            },
        },
    )

    edited = client.edited_messages[0]
    # cond:done now returns to the single-watch edit view, not the watchlist
    assert "✏️" in edited["text"]
    assert "💰 上限" in edited["text"]
    kb_rows = edited["reply_markup"]["inline_keyboard"]
    # First row is "修改上限", last row is "← 返回清單"
    assert any("wprc:" in r[0]["callback_data"] for r in kb_rows)
    assert any("wback:" in r[0]["callback_data"] for r in kb_rows)


# ── Paginated /hunt view ──────────────────────────────────────────────────────

def _make_huntlist_processor(candidates: list[dict[str, object]]) -> tuple[TelegramCommandProcessor, dict]:
    """Build a processor wired with stub opportunity view/deleter via registry.

    Returns (processor, scratch) where scratch is a dict the test can inspect
    to see which candidate_id the deleter was called with.
    """
    from price_monitor_bot.list_view import LIST_VIEW_MODE_READ, ListRow, build_list_view

    scratch: dict[str, object] = {"removed": [], "candidates": list(candidates)}

    def _hl_view(*, page: int = 0, mode: str = LIST_VIEW_MODE_READ):
        items = []
        for c in scratch["candidates"]:
            cid = str(c.get("candidate_id") or "")
            if not cid:
                continue
            game = str(c.get("game") or "?")
            title = str(c.get("title") or "(no title)")
            heat = c.get("heat_score")
            heat_text = f"{float(heat):.0f}" if heat is not None else "?"
            sq = str(c.get("search_query") or "")
            items.append(
                ListRow(
                    id=cid,
                    text=f"[{game}] {title}\n  heat={heat_text}  search: {sq}",
                    short_label=f"[{game}] {title[:18]}",
                )
            )
        return build_list_view(
            list_kind="hl",
            items=items,
            page=page,
            mode=mode,
            list_title="📋 Opportunity 候選",
            empty_message="目前沒有 Opportunity 候選。等下一輪 agent tick 收集到目標後再看。",
        )

    def _hl_deleter(candidate_id: str) -> bool:
        for c in scratch["candidates"]:
            if c["candidate_id"] == candidate_id:
                scratch["candidates"].remove(c)
                scratch["removed"].append(candidate_id)
                return True
        return False

    proc = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        view_handlers={"hl": _hl_view},
        item_deleter_handlers={"hl": (_hl_deleter, "Opportunity 候選")},
    )
    return proc, scratch


def _make_hunt_candidate(i: int) -> dict[str, object]:
    return {
        "candidate_id": f"opp_{i:016x}",
        "game": "pokemon",
        "product_type": "single_card",
        "title": f"Test card #{i}",
        "heat_score": 100 + i,
        "search_query": f"q{i}",
        "last_checked_at": None,
        "reason": None,
    }


def test_huntlist_view_renders_with_per_item_delete_in_edit_mode() -> None:
    processor, _ = _make_huntlist_processor([_make_hunt_candidate(i) for i in range(3)])

    view = processor._view_registry["hl"]
    text_r, kb_r, _ = view()
    assert "📋 Opportunity 候選" in text_r
    assert "共 3 筆" in text_r
    assert len(kb_r["inline_keyboard"]) == 1  # nav only

    _, kb_e, _ = view(mode="e")
    assert len(kb_e["inline_keyboard"]) == 4  # 3 deletes + nav
    delete_data = [row[0]["callback_data"] for row in kb_e["inline_keyboard"][:3]]
    assert delete_data == [
        "del:hl:opp_0000000000000000",
        "del:hl:opp_0000000000000001",
        "del:hl:opp_0000000000000002",
    ]


def test_callback_query_del_hl_dismisses_candidate_and_rerenders() -> None:
    processor, scratch = _make_huntlist_processor([_make_hunt_candidate(i) for i in range(8)])
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-hl",
            "data": "del:hl:opp_0000000000000003",
            "message": {
                "message_id": 21,
                "chat": {"id": "123"},
                "text": "📋 Opportunity 候選  第 1/2 頁（共 8 筆）\n…",
            },
        },
    )

    assert scratch["removed"] == ["opp_0000000000000003"]
    edited = client.edited_messages[0]
    assert "第 1/2 頁" in edited["text"]
    assert "共 7 筆" in edited["text"]
    assert client.answered_callbacks[0]["text"] == "✓ 已刪除"


def test_huntlist_view_empty_when_no_provider_results() -> None:
    processor, _ = _make_huntlist_processor([])
    view = processor._view_registry["hl"]
    text, kb, page = view()
    assert "目前沒有 Opportunity 候選" in text
    assert kb is None
    assert page == 0


# ── Clarification flows: inline buttons for photo / text option selection ─────

def _last_message_reply_markup(client: FakeTelegramClient) -> dict[str, object] | None:
    """Helper: the FakeTelegramClient.sent_messages list only keeps text. To
    inspect the reply_markup we need to drop down to the underlying calls —
    but the test fake's send_message returns the keyword dict it received,
    so we record by patching."""
    # No-op in current fake; tests below use a wrapper FakeClient with capture.
    return None


class CapturingTelegramClient(FakeTelegramClient):
    """FakeTelegramClient that also records each send_message's reply_markup."""

    def __init__(self, sample_path: Path | None = None) -> None:
        super().__init__(sample_path=sample_path)
        self.sent_markups: list[dict[str, object] | None] = []

    def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.sent_markups.append(reply_markup)
        return super().send_message(chat_id=chat_id, text=text, reply_markup=reply_markup)


def test_photo_clarification_message_carries_popt_inline_keyboard() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = CapturingTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )

    # Find the clarification message (the one containing the numbered options
    # — it's the final reply, not the intake/processing acks).
    clar_idx = next(
        i for i, t in enumerate(client.sent_messages) if "請點按鈕" in t
    )
    kb = client.sent_markups[clar_idx]
    assert kb is not None
    # `_ambiguous_photo_analysis` yields 3 numbered options.
    buttons = [row[0] for row in kb["inline_keyboard"]]
    assert all(b["callback_data"].startswith("popt:") for b in buttons)
    assert [b["callback_data"] for b in buttons] == ["popt:1", "popt:2", "popt:3"]


def test_text_clarification_message_carries_topt_inline_keyboard() -> None:
    client = CapturingTelegramClient()
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="unknown",
            game=None,
            name=None,
            confidence=0.1,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "why are pokemon cards popular"},
    )

    clar_idx = next(
        (i for i, t in enumerate(client.sent_messages) if "請點按鈕" in t),
        None,
    )
    assert clar_idx is not None, client.sent_messages
    kb = client.sent_markups[clar_idx]
    assert kb is not None
    buttons = [row[0] for row in kb["inline_keyboard"]]
    assert all(b["callback_data"].startswith("topt:") for b in buttons)
    assert {int(b["callback_data"].split(":")[1]) for b in buttons} == {
        opt for opt in range(1, len(buttons) + 1)
    }


def test_callback_query_popt_selects_option_and_runs_photo_action() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = CapturingTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    # 1. Upload an ambiguous photo → bot installs pending clarification.
    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )

    # 2. User taps `popt:1` button instead of typing "1".
    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-popt",
            "data": "popt:1",
            "message": {
                "message_id": 31,
                "chat": {"id": "123"},
                "text": "我先看了一下，這張圖看起來比較像「寶可夢卡片」…\n請點按鈕…",
            },
        },
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}",
    )

    # Original clarification message gets edited to show the selection and
    # have its keyboard cleared.
    assert len(client.edited_messages) == 1
    edited = client.edited_messages[0]
    assert "✓ 已選 1." in edited["text"]
    assert edited["reply_markup"] is None
    # Callback toast confirms the choice.
    assert any(a["text"] == "已選 1" for a in client.answered_callbacks)
    # The downstream action that "1" would have triggered must run — the
    # plan's ack and reply land as fresh messages.
    assert any("第 1 個方式處理" in m for m in client.sent_messages)
    assert any(m.startswith("resolved:") for m in client.sent_messages)


def test_callback_query_topt_selects_option_and_runs_text_action() -> None:
    client = CapturingTelegramClient()
    nl_intent = TelegramNaturalLanguageIntent(
        intent="unknown",
        game=None,
        name=None,
        confidence=0.1,
    )
    router = StubNaturalLanguageRouter(nl_intent)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: f"price:{query.name}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "why are pokemon cards popular"},
    )

    # Tap the first option button.
    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-topt",
            "data": "topt:1",
            "message": {
                "message_id": 42,
                "chat": {"id": "123"},
                "text": "我看了一下你的訊息…\n請點按鈕…",
            },
        },
    )

    assert len(client.edited_messages) == 1
    edited = client.edited_messages[0]
    assert "✓ 已選 1." in edited["text"]
    assert edited["reply_markup"] is None
    assert any(a["text"] == "已選 1" for a in client.answered_callbacks)


def test_callback_query_popt_with_no_pending_state_shows_expired_toast() -> None:
    client = CapturingTelegramClient()
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    # No pending state set; user taps a stale button.
    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-stale",
            "data": "popt:2",
            "message": {
                "message_id": 99,
                "chat": {"id": "123"},
                "text": "（已過期的釐清訊息）",
            },
        },
    )

    # The button is removed (keyboard cleared) and toast says expired.
    assert client.edited_messages
    assert client.edited_messages[0]["reply_markup"] is None
    assert "已過期" in client.edited_messages[0]["text"]
    assert client.answered_callbacks
    assert client.answered_callbacks[0]["text"] == "選項已處理或過期"
    # No fresh action messages should be sent.
    assert client.sent_messages == []


# ── SNS bulk filter — NL fallback + preview/confirm/cancel callbacks ──────────

def test_fallback_routes_bulk_filter_tcg_add() -> None:
    intent = fallback_route_telegram_natural_language(
        "把每個跟 tcg 相關的 sns 追蹤帳號 filter 都加上「抽選」"
    )
    assert intent is not None
    assert intent.intent == "sns_bulk_add_filter"
    assert intent.bulk_target_domain == "tcg"
    assert intent.bulk_filter_keywords == ("抽選",)


def test_fallback_routes_bulk_filter_pokemon_specific() -> None:
    intent = fallback_route_telegram_natural_language(
        "幫所有 pokemon 帳號加上 抽選 filter"
    )
    assert intent is not None
    assert intent.intent == "sns_bulk_add_filter"
    assert intent.bulk_target_domain == "pokemon"
    assert intent.bulk_filter_keywords == ("抽選",)


def test_fallback_routes_bulk_filter_yugioh_with_quote() -> None:
    intent = fallback_route_telegram_natural_language(
        "所有遊戲王帳號的篩選都加上 新弾"
    )
    assert intent is not None
    assert intent.intent == "sns_bulk_add_filter"
    assert intent.bulk_target_domain == "yugioh"
    assert intent.bulk_filter_keywords == ("新弾",)


def test_fallback_does_not_misroute_single_handle_add_filter() -> None:
    """`/snsadd @foo filter[…]`-style single-handle requests must continue
    routing to `sns_add_account`, not bulk."""
    intent = fallback_route_telegram_natural_language(
        "幫我把 @tenbai_hakase 加上 [抽選] 篩選"
    )
    assert intent is not None
    assert intent.intent == "sns_add_account"


# Live SnsDatabase round-trips for the bulk e2e path. The price_monitor_bot
# venv doesn't import sns_monitor; skip these tests there.
import pytest as _pytest
try:
    from sns_monitor.storage import SnsDatabase as _RealSnsDatabase  # noqa: F401
    _HAVE_SNS_MONITOR = True
except ImportError:
    _HAVE_SNS_MONITOR = False


@_pytest.mark.skipif(not _HAVE_SNS_MONITOR, reason="sns_monitor package not installed in this venv")
def _make_bulk_processor_with_tcg_rules(tmp_path) -> tuple[TelegramCommandProcessor, object]:
    from sns_monitor.models import AccountWatch
    from sns_monitor.storage import SnsDatabase

    db = SnsDatabase(tmp_path / "sns.sqlite3")
    db.bootstrap()
    for name, domains in [
        ("poke_news", ("pokemon", "tcg")),
        ("yugioh_jp", ("yugioh",)),
        ("politics_bot", ("politic",)),  # should NOT be picked up
    ]:
        db.save_watch_rule(
            AccountWatch(
                rule_id=SnsDatabase._watch_rule_id("account", name),
                screen_name=name,
                user_id=None,
                label=f"@{name}",
                include_keywords=(),
                domains=domains,
                enabled=True,
                schedule_minutes=15,
                chat_id="0",
                last_checked_at=None,
            )
        )
    proc = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        sns_db=db,
    )
    return proc, db


@_pytest.mark.skipif(not _HAVE_SNS_MONITOR, reason="sns_monitor package not installed in this venv")
def test_sns_bulk_add_filter_preview_lists_affected_and_sets_pending(tmp_path) -> None:
    processor, _ = _make_bulk_processor_with_tcg_rules(tmp_path)

    plan = processor._build_sns_bulk_add_filter_plan(
        chat_id="123", target_domain="tcg", keywords=("抽選",)
    )

    assert "找到 2 個" in plan.reply
    assert "@poke_news" in plan.reply
    assert "@yugioh_jp" in plan.reply
    assert "@politics_bot" not in plan.reply
    kb = plan.reply_markup
    assert kb is not None
    flat = [b for row in kb["inline_keyboard"] for b in row]
    assert {b["callback_data"] for b in flat} == {"bulk:c", "bulk:x"}
    # Pending state must be installed.
    pending = processor.get_pending_sns_bulk_update("123")
    assert pending is not None
    assert pending.bulk_target_domain == "tcg"
    assert pending.keywords == ("抽選",)
    assert len(pending.affected_rule_ids) == 2


@_pytest.mark.skipif(not _HAVE_SNS_MONITOR, reason="sns_monitor package not installed in this venv")
def test_sns_bulk_add_filter_confirm_callback_updates_db(tmp_path) -> None:
    processor, db = _make_bulk_processor_with_tcg_rules(tmp_path)
    # Set up pending via the preview builder.
    processor._build_sns_bulk_add_filter_plan(
        chat_id="123", target_domain="tcg", keywords=("抽選",)
    )
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-bulk-c",
            "data": "bulk:c",
            "message": {
                "message_id": 100,
                "chat": {"id": "123"},
                "text": "🎯 找到 2 個 tcg 相關帳號…",
            },
        },
    )

    # Both TCG rules now have the new keyword; politics_bot is untouched.
    for handle in ("poke_news", "yugioh_jp"):
        rule = db.get_watch_rule(_RealSnsDatabase._watch_rule_id("account", handle))
        assert rule.include_keywords == ("抽選",)
    politics = db.get_watch_rule(_RealSnsDatabase._watch_rule_id("account", "politics_bot"))
    assert politics.include_keywords == ()
    assert "✓ 已修改 2 個帳號" in client.edited_messages[0]["text"]
    assert processor.get_pending_sns_bulk_update("123") is None


@_pytest.mark.skipif(not _HAVE_SNS_MONITOR, reason="sns_monitor package not installed in this venv")
def test_sns_bulk_add_filter_cancel_callback_leaves_db_untouched(tmp_path) -> None:
    processor, db = _make_bulk_processor_with_tcg_rules(tmp_path)
    processor._build_sns_bulk_add_filter_plan(
        chat_id="123", target_domain="tcg", keywords=("抽選",)
    )
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-bulk-x",
            "data": "bulk:x",
            "message": {
                "message_id": 100,
                "chat": {"id": "123"},
                "text": "🎯 找到 2 個 tcg 相關帳號…",
            },
        },
    )

    for handle in ("poke_news", "yugioh_jp", "politics_bot"):
        rule = db.get_watch_rule(_RealSnsDatabase._watch_rule_id("account", handle))
        assert rule.include_keywords == ()
    assert "已取消" in client.edited_messages[0]["text"]
    assert processor.get_pending_sns_bulk_update("123") is None


@_pytest.mark.skipif(not _HAVE_SNS_MONITOR, reason="sns_monitor package not installed in this venv")
def test_sns_bulk_add_filter_callback_with_no_pending_shows_expired_toast(tmp_path) -> None:
    processor, _ = _make_bulk_processor_with_tcg_rules(tmp_path)
    # No pending — user is tapping a stale button.
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-bulk-stale",
            "data": "bulk:c",
            "message": {
                "message_id": 100,
                "chat": {"id": "123"},
                "text": "old preview text",
            },
        },
    )

    assert client.answered_callbacks[0]["text"] == "操作已過期，請重新輸入"
    # The message gets edited with the expired marker, but no DB write.
    assert "已過期" in client.edited_messages[0]["text"]


# ── Price-feedback loop UI (Phase 1) ────────────────────────────────────────


class _FakeFeedbackService:
    def __init__(self, summary: str = "fake summary") -> None:
        self.calls: list[dict] = []
        self.summary = summary

    def submit(self, *, item, spec, chat_id, original_fair_value_jpy, claimed_url):
        self.calls.append({
            "item_id": item.item_id, "game": spec.game, "kind": spec.item_kind,
            "chat_id": chat_id, "fair_value": original_fair_value_jpy,
            "url": claimed_url,
        })
        from collections import namedtuple
        Out = namedtuple("Out", ["summary_for_user"])
        return Out(summary_for_user=self.summary)


def test_fbprc_callback_stores_pending_and_sends_force_reply(tmp_path: Path) -> None:
    """Tap '不合理' button → bot stores pending feedback state and sends a
    ForceReply prompt with the [fbprc:item_id] tag embedded."""
    from market_monitor.storage import MonitorDatabase

    watch_db = MonitorDatabase(tmp_path / "fbui.sqlite3")
    watch_db.bootstrap()
    item = TrackedItem(
        item_id="tcg-abcdef0123456789",
        item_type="tcg_sealed_box",
        category="tcg",
        title="MEGA アビスアイ",
        attributes={"game": "pokemon", "item_kind": "sealed_box"},
    )
    watch_db.upsert_item(item)

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        watch_db=watch_db,
        feedback_service=_FakeFeedbackService(),
    )
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query={
            "id": "cbq-fbprc",
            "data": f"fbprc:{item.item_id}",
            "message": {
                "message_id": 99,
                "chat": {"id": "123"},
                "text": "lookup result here",
            },
        },
    )

    # ForceReply message was sent
    assert any(f"[fbprc:{item.item_id}]" in msg for msg in client.sent_messages)
    # Pending state stored
    pending = processor.get_pending_price_feedback("123")
    assert pending is not None
    assert pending.item_id == item.item_id


def test_fbprc_url_reply_calls_feedback_service(tmp_path: Path) -> None:
    """User pastes a URL as reply to the ForceReply prompt → handler matches
    the [fbprc:item_id] tag, consumes pending state, and routes to the
    feedback_service."""
    from market_monitor.storage import MonitorDatabase

    watch_db = MonitorDatabase(tmp_path / "fbui2.sqlite3")
    watch_db.bootstrap()
    item = TrackedItem(
        item_id="tcg-abcdef0123456789",
        item_type="tcg_sealed_box",
        category="tcg",
        title="MEGA アビスアイ",
        attributes={"game": "pokemon", "item_kind": "sealed_box"},
    )
    watch_db.upsert_item(item)

    fake_service = _FakeFeedbackService(summary="✅ 已記錄\n網站: yuyu-tei.jp")
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        watch_db=watch_db,
        feedback_service=fake_service,
    )
    # Simulate the pending state that the callback would have set
    from price_monitor_bot.bot import PendingTelegramPriceFeedback
    processor.set_pending_price_feedback(PendingTelegramPriceFeedback(
        chat_id="123", item_id=item.item_id, original_fair_value_jpy=16800,
    ))

    client = FakeTelegramClient()
    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda q: "noop",
        message={
            "chat": {"id": "123"},
            "text": "https://yuyu-tei.jp/sealed/abc",
            "reply_to_message": {
                "text": f"請貼上 URL [fbprc:{item.item_id}]",
            },
        },
    )

    assert len(fake_service.calls) == 1
    call = fake_service.calls[0]
    assert call["item_id"] == item.item_id
    assert call["url"] == "https://yuyu-tei.jp/sealed/abc"
    assert call["fair_value"] == 16800
    # Pending state consumed
    assert processor.get_pending_price_feedback("123") is None
    # Bot replied with the service's summary
    assert any("已記錄" in m for m in client.sent_messages)


def test_fbprc_url_reply_rejects_non_url(tmp_path: Path) -> None:
    from market_monitor.storage import MonitorDatabase
    from price_monitor_bot.bot import PendingTelegramPriceFeedback

    watch_db = MonitorDatabase(tmp_path / "fbui3.sqlite3")
    watch_db.bootstrap()
    item = TrackedItem(
        item_id="tcg-xy",
        item_type="tcg_sealed_box", category="tcg", title="X",
        attributes={"game": "pokemon", "item_kind": "sealed_box"},
    )
    watch_db.upsert_item(item)

    fake_service = _FakeFeedbackService()
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        watch_db=watch_db,
        feedback_service=fake_service,
    )
    processor.set_pending_price_feedback(
        PendingTelegramPriceFeedback(chat_id="123", item_id=item.item_id, original_fair_value_jpy=None)
    )

    client = FakeTelegramClient()
    handle_telegram_message(
        client=client, processor=processor,
        photo_renderer=lambda q: "noop",
        message={
            "chat": {"id": "123"},
            "text": "not a url at all",
            "reply_to_message": {"text": f"請貼上 URL [fbprc:{item.item_id}]"},
        },
    )

    assert fake_service.calls == []  # service NOT called
    assert any("不是合法 URL" in m for m in client.sent_messages)


# ── Poll-loop health helpers ───────────────────────────────────────────────


def test_poll_heartbeat_is_stale_after_threshold(tmp_path: Path) -> None:
    hb = PollHeartbeat(tmp_path / "hb")
    hb.touch()
    stale_now = 500.0
    beat = hb.last_beat()
    assert beat is not None
    assert not hb.is_stale(10.0, now=beat + 5.0)
    assert hb.is_stale(10.0, now=beat + 11.0)


def test_poll_heartbeat_never_stale_if_never_written(tmp_path: Path) -> None:
    hb = PollHeartbeat(tmp_path / "hb_fresh")
    assert hb.last_beat() is None
    assert not hb.is_stale(0.0)


def test_is_conflict_error_detects_409() -> None:
    assert _is_conflict_error(RuntimeError("Telegram API HTTP 409 for getUpdates."))


def test_is_conflict_error_detects_conflict_text() -> None:
    assert _is_conflict_error(RuntimeError("Conflict: terminated by other getUpdates"))


def test_is_conflict_error_ignores_other_errors() -> None:
    assert not _is_conflict_error(RuntimeError("Telegram API HTTP 500 for getUpdates."))
    assert not _is_conflict_error(RuntimeError("Connection refused"))


def test_drain_pending_updates_returns_offset_on_success() -> None:
    calls: list[int] = []

    class _Client:
        def get_updates(self, *, offset=None, timeout=0):
            calls.append(timeout)
            return [{"update_id": 42}]

    result = _drain_pending_updates(_Client(), sleep_fn=lambda _: None)
    assert result == 43
    assert calls == [0]


def test_drain_pending_updates_returns_none_when_empty() -> None:
    class _Client:
        def get_updates(self, *, offset=None, timeout=0):
            return []

    assert _drain_pending_updates(_Client(), sleep_fn=lambda _: None) is None


def test_drain_pending_updates_retries_on_409_then_succeeds() -> None:
    attempt = [0]

    class _Client:
        def get_updates(self, *, offset=None, timeout=0):
            attempt[0] += 1
            if attempt[0] < 3:
                raise RuntimeError("Telegram API HTTP 409 for getUpdates.")
            return [{"update_id": 7}]

    slept: list[float] = []
    result = _drain_pending_updates(_Client(), backoff_seconds=1.0, sleep_fn=slept.append)
    assert result == 8
    assert attempt[0] == 3
    assert len(slept) == 2  # slept twice before succeeding


def test_drain_pending_updates_gives_up_after_max_attempts() -> None:
    class _Client:
        def get_updates(self, *, offset=None, timeout=0):
            raise RuntimeError("Telegram API HTTP 409 for getUpdates.")

    result = _drain_pending_updates(
        _Client(), max_attempts=3, backoff_seconds=0.0, sleep_fn=lambda _: None
    )
    assert result is None


def test_drain_pending_updates_reraises_non_conflict_error() -> None:
    class _Client:
        def get_updates(self, *, offset=None, timeout=0):
            raise RuntimeError("Telegram API HTTP 500 for getUpdates.")

    with pytest.raises(RuntimeError, match="500"):
        _drain_pending_updates(_Client(), sleep_fn=lambda _: None)


def test_start_poll_watchdog_sends_alert_on_stale_heartbeat(tmp_path: Path) -> None:
    import time as _time

    hb = PollHeartbeat(tmp_path / "hb_wd")
    alerted: list[str] = []

    class _Client:
        def send_message(self, *, chat_id, text, **_kw):
            alerted.append(text)
            return {}

    # Write a heartbeat that is already 200s stale.
    (tmp_path / "hb_wd").write_text(str(_time.time() - 200))

    start_poll_watchdog(
        heartbeat=hb,
        client=_Client(),
        alert_chat_ids=frozenset(["999"]),
        max_age_seconds=10.0,
        check_interval_seconds=0.05,
        exit_on_stale=False,
    )
    _time.sleep(0.2)
    assert any("心跳停止" in m for m in alerted)


# ── /quiz command + quiz: callback ────────────────────────────────────────────


def _quiz_processor(*, quiz_handler=None, quiz_callback_handler=None):
    command_handlers = {}
    if quiz_handler is not None:
        command_handlers["/quiz"] = RegisteredCommand(
            quiz_handler,
            ack="收到，正在出題（地端模型，可能要一點時間）…",
            background=True,
        )
    callback_handlers = {}
    if quiz_callback_handler is not None:
        callback_handlers["quiz"] = quiz_callback_handler
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        command_handlers=command_handlers,
        callback_handlers=callback_handlers,
    )


def test_quiz_command_runs_in_background_and_returns_question_with_keyboard() -> None:
    seen = []
    markup = {"inline_keyboard": [[{"text": "A", "callback_data": "quiz:a:qid:0"}]]}

    def handler(raw, chat_id):
        seen.append((raw, chat_id))
        return ("🎴 JLPT N1 測驗\n\nA. foo\nB. bar", markup)

    processor = _quiz_processor(quiz_handler=handler)
    plan = processor.build_reply_plan(chat_id="123", text="/quiz JLPTN1 miku")

    assert plan is not None
    assert plan.run_in_background is True  # slow local-LLM op must not block poll loop
    assert plan.ack  # an ack is shown immediately
    text, factory_markup = plan._execute_unpacked()
    assert "JLPT N1 測驗" in text
    assert factory_markup == markup
    # remainder after the command word is forwarded to the handler
    assert seen and seen[0][0] == "JLPTN1 miku" and seen[0][1] == "123"


def _fetch_processor(*, fetch_renderer=None) -> TelegramCommandProcessor:
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda q: "unused",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        fetch_renderer=fetch_renderer,
    )


def test_fetch_command_forwards_url_and_prompt_to_renderer() -> None:
    seen: list[tuple[str, str]] = []

    def renderer(url, prompt):
        seen.append((url, prompt))
        return f"答案：{url}"

    processor = _fetch_processor(fetch_renderer=renderer)
    plan = processor.build_reply_plan(
        chat_id="123", text="/fetch https://example.com/a 這篇的重點是什麼",
    )

    assert plan is not None
    assert plan.ack  # heavy op acks immediately
    reply = plan.execute()
    assert reply == "答案：https://example.com/a"
    assert seen == [("https://example.com/a", "這篇的重點是什麼")]


def test_fetch_command_disabled_when_no_renderer() -> None:
    processor = _fetch_processor(fetch_renderer=None)
    plan = processor.build_reply_plan(chat_id="123", text="/fetch https://example.com/a x")
    reply = plan.execute()
    assert "尚未設定" in reply


def test_fetch_command_requires_url() -> None:
    processor = _fetch_processor(fetch_renderer=lambda u, p: "unused")
    plan = processor.build_reply_plan(chat_id="123", text="/fetch")
    reply = plan.execute()
    assert "請提供網址" in reply


def test_quiz_command_unregistered_falls_through_to_unknown() -> None:
    # No /quiz in the registry → generic dispatch is skipped and the command
    # falls through to the built-in "Unknown command" reply.
    processor = _quiz_processor(quiz_handler=None)
    plan = processor.build_reply_plan(chat_id="123", text="/quiz")
    text, _ = plan._execute_unpacked()
    assert "Unknown command" in text


def test_quiz_handler_string_result_is_wrapped() -> None:
    processor = _quiz_processor(quiz_handler=lambda raw, cid: "純文字回覆")
    plan = processor.build_reply_plan(chat_id="123", text="/quiz review")
    text, markup = plan._execute_unpacked()
    assert text == "純文字回覆"
    assert markup is None


def test_quiz_callback_grades_and_edits_message() -> None:
    captured = {}

    def cb(payload, original_text, chat_id):
        captured["payload"] = payload
        captured["original"] = original_text
        captured["chat_id"] = chat_id
        return ("✅ 答對了！", original_text + "\n\n✅ 正解！", None)

    processor = _quiz_processor(quiz_callback_handler=cb)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(
            chat_id="123", data="quiz:a:abc123:1", text="🎴 JLPT N1 測驗"
        ),
    )

    assert captured["payload"] == "a:abc123:1"
    assert captured["original"] == "🎴 JLPT N1 測驗"
    assert captured["chat_id"] == "123"
    assert len(client.edited_messages) == 1
    edited = client.edited_messages[0]
    assert "✅ 正解！" in edited["text"]
    assert edited["reply_markup"] is None  # keyboard cleared after answering
    assert client.answered_callbacks[0]["text"] == "✅ 答對了！"


def test_quiz_callback_unregistered_falls_through() -> None:
    processor = _quiz_processor(quiz_callback_handler=None)
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="quiz:a:abc:0"),
    )

    # No registered "quiz" prefix → no edit, default unknown-button toast.
    assert client.edited_messages == []
    assert client.answered_callbacks[0]["text"] == "未知按鈕"


# ── command/callback registry extension point ─────────────────────────────────


def _registry_processor(*, command_handlers=None, callback_handlers=None):
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        command_handlers=command_handlers or {},
        callback_handlers=callback_handlers or {},
    )


def test_registry_sync_command_returns_text_and_markup() -> None:
    markup = {"inline_keyboard": [[{"text": "x", "callback_data": "voice:speed:+"}]]}
    seen = []

    def handler(remainder, chat_id):
        seen.append((remainder, chat_id))
        return ("語音參數", markup)

    processor = _registry_processor(
        command_handlers={"/voice": RegisteredCommand(handler)}
    )
    plan = processor.build_reply_plan(chat_id="123", text="/voice speed 1.5")

    assert plan.run_in_background is False
    assert plan.ack is None
    assert plan.execute() == "語音參數"
    assert plan.reply_markup == markup
    assert seen == [("speed 1.5", "123")]


def test_registry_background_command_uses_factory_and_ack() -> None:
    calls = []

    def handler(remainder, chat_id):
        calls.append((remainder, chat_id))
        return f"done:{remainder}:{chat_id}"

    spec = RegisteredCommand(handler, ack="收到…", background=True)
    processor = _registry_processor(command_handlers={"/say": spec})
    plan = processor.build_reply_plan(chat_id="123", text="/say こんにちは")

    assert plan.run_in_background is True
    assert plan.ack == "收到…"
    # The handler must not have run yet — only the factory defers it.
    assert calls == []
    reply = plan.execute()
    assert reply == "done:こんにちは:123"
    assert calls == [("こんにちは", "123")]


def test_registry_command_error_returns_friendly_plan() -> None:
    def boom(remainder, chat_id):
        raise RuntimeError("爆炸")

    processor = _registry_processor(
        command_handlers={"/voice": RegisteredCommand(boom)}
    )
    plan = processor.build_reply_plan(chat_id="123", text="/voice x")
    reply = plan.execute()
    assert "指令失敗" in reply
    assert "爆炸" in reply


def test_registry_does_not_shadow_builtin_commands() -> None:
    # A registry must never intercept built-ins like /ping.
    called = []
    processor = _registry_processor(
        command_handlers={
            "/ping": RegisteredCommand(lambda r, c: called.append(1) or "hijacked")
        }
    )
    plan = processor.build_reply_plan(chat_id="123", text="/ping")
    assert plan.execute() == "pong"
    assert called == []


def test_registry_callback_routes_and_rerenders() -> None:
    def cb(payload, original_text, chat_id):
        return ("✅", original_text + f"\n{payload}/{chat_id}", None)

    processor = _registry_processor(callback_handlers={"voice": cb})
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="voice:speed:+", text="參數"),
    )

    assert len(client.edited_messages) == 1
    assert "speed:+/123" in client.edited_messages[0]["text"]
    assert client.answered_callbacks[0]["text"] == "✅"


def test_registry_callback_does_not_shadow_builtin_prefix() -> None:
    # Built-in prefixes (e.g. noop) must still win over an unrelated registry.
    processor = _registry_processor(callback_handlers={"voice": lambda *a: ("x", "y", None)})
    client = FakeTelegramClient()

    handle_telegram_callback_query(
        client=client,
        processor=processor,
        callback_query=_callback_update(chat_id="123", data="noop", text="label"),
    )

    # noop silently acknowledges with no edit.
    assert client.edited_messages == []


# --- #52 live catalog fallback: unknown_text_handler intercept ----------------

def _catalog_processor(handler):
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        unknown_text_handler=handler,
    )


def test_unknown_text_handler_reply_is_used() -> None:
    called: list = []

    def handler(text, chat_id):
        called.append((text, chat_id))
        return ("大阪 晴", {"inline_keyboard": [[{"text": "x", "callback_data": "y"}]]})

    processor = _catalog_processor(handler)
    plan = processor.build_reply_plan(chat_id="123", text="查大阪天氣")
    assert plan.reply == "大阪 晴"
    assert plan.reply_markup is not None
    assert called == [("查大阪天氣", "123")]


def test_unknown_text_handler_none_falls_through_to_default() -> None:
    called: list = []

    def handler(text, chat_id):
        called.append(text)
        return None

    processor = _catalog_processor(handler)
    plan = processor.build_reply_plan(chat_id="123", text="查大阪天氣")
    # Handler was consulted, returned nothing → fall through to the generic
    # clarification menu (or, absent candidates, the unknown-command reply).
    assert called == ["查大阪天氣"]
    assert plan.reply != "大阪 晴"
    assert ("/search" in plan.reply) or ("Unknown command" in plan.reply)


def test_unknown_text_handler_not_called_for_slash_commands() -> None:
    called: list = []
    processor = _catalog_processor(lambda text, chat_id: called.append(text) or ("x", None))
    plan = processor.build_reply_plan(chat_id="123", text="/bogus")
    assert "Unknown command" in plan.reply
    assert called == []


def test_unknown_text_handler_exception_falls_through() -> None:
    def boom(text, chat_id):
        raise RuntimeError("nope")

    processor = _catalog_processor(boom)
    plan = processor.build_reply_plan(chat_id="123", text="查大阪天氣")
    # A throwing handler is swallowed; routing falls through to the menu.
    assert plan.reply != "大阪 晴"
    assert ("/search" in plan.reply) or ("Unknown command" in plan.reply)

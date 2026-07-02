"""Characterization tests for TelegramCommandProcessor.build_reply_plan's
branch order, written BEFORE the Phase 2 telegram_core split
(docs/TELEGRAM_CORE_EXTRACTION_PLAN.md §4 Phase 2, aka_no_claw repo). These
must keep passing UNMODIFIED after the split into
telegram_core.CoreCommandProcessor + PriceCommandProcessor — that's the
stage-gate proof that the split didn't reorder dispatch priority.

Locked-in order: allowlist -> empty-content guard -> built-ins
(/start /help /ping /status /tools) -> registry -> domain command sets ->
natural-language routing -> unknown-text hook -> clarification -> fallback.
"""
from __future__ import annotations

from datetime import datetime, timezone

from tcg_tracker.hot_cards import HotCardBoard, HotCardEntry

from price_monitor_bot.bot import RegisteredCommand, TelegramCommandProcessor


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
            ),
        ),
    )


def _make_processor(**kwargs) -> TelegramCommandProcessor:
    kwargs.setdefault("lookup_renderer", lambda query: query.name)
    kwargs.setdefault("board_loader", lambda: (_stub_board(),))
    kwargs.setdefault("catalog_renderer", lambda: "catalog")
    kwargs.setdefault("allowed_chat_ids", frozenset({"123"}))
    return TelegramCommandProcessor(**kwargs)


def test_builtin_help_wins_over_a_registered_help_handler() -> None:
    """/start and /help are checked before registry dispatch — a registered
    handler for the same name is unreachable."""
    processor = _make_processor(
        command_handlers={"/help": RegisteredCommand(lambda remainder, chat_id: "registry help")},
    )
    reply = processor.build_reply(chat_id="123", text="/help")
    assert reply != "registry help"
    assert "/trend pokemon" in reply  # the real catalog, unchanged


def test_builtin_ping_and_status_are_not_overridable_by_registry() -> None:
    processor = _make_processor(
        command_handlers={
            "/ping": RegisteredCommand(lambda remainder, chat_id: "registry pong"),
            "/status": RegisteredCommand(lambda remainder, chat_id: "registry status"),
        },
    )
    assert processor.build_reply(chat_id="123", text="/ping") == "pong"
    assert processor.build_reply(chat_id="123", text="/status") == "OpenClaw Telegram bot is online."


def test_registry_dispatch_wins_over_hardcoded_price_command_set() -> None:
    """/price is normally routed via PRICE_LOOKUP_COMMANDS; a registered
    handler for the same name must be dispatched first."""
    processor = _make_processor(
        command_handlers={"/price": RegisteredCommand(lambda remainder, chat_id: f"registry:{remainder}")},
    )
    assert processor.build_reply(chat_id="123", text="/price pokemon Pikachu") == "registry:pokemon Pikachu"


def test_unrecognized_slash_command_returns_domain_unknown_message() -> None:
    processor = _make_processor()
    reply = processor.build_reply(chat_id="123", text="/nope")
    assert reply == (
        "Unknown command. Use /help, /price, /trend, /snapshot, /search, "
        "or send a photo with /scan. You can also ask in natural language."
    )


def test_unrecognized_bare_text_with_no_router_offers_clarification_menu() -> None:
    """No natural_language_router/intent_fast_path configured -> ambiguous ->
    falls through to the generic did-you-mean clarification menu (a
    web_research + help candidate pair is always offered for plain text),
    NOT straight to the unknown-command fallback."""
    processor = _make_processor()
    reply = processor.build_reply(chat_id="123", text="random chatter with no signal")
    assert "請點按鈕（或回覆數字）：" in reply
    assert "(相當於 /search)" in reply
    assert "(相當於 /help)" in reply


def test_unknown_text_handler_runs_before_the_clarification_menu() -> None:
    """For ambiguous non-slash text, the injected unknown_text_handler (aka's
    /new dynamic-tool hook) gets first crack, before the generic
    did-you-mean clarification menu."""
    calls: list[str] = []

    def handler(text: str, chat_id: str):
        calls.append(text)
        return "handled by dynamic tool", None

    processor = _make_processor(unknown_text_handler=handler)
    reply = processor.build_reply(chat_id="123", text="some ambiguous chatter")
    assert reply == "handled by dynamic tool"
    assert calls == ["some ambiguous chatter"]


def test_unknown_text_handler_returning_none_falls_back_to_clarification_menu() -> None:
    processor = _make_processor(unknown_text_handler=lambda text, chat_id: None)
    reply = processor.build_reply(chat_id="123", text="random chatter with no signal")
    assert "請點按鈕（或回覆數字）：" in reply
    assert "(相當於 /search)" in reply


def test_empty_content_guard_before_any_dispatch() -> None:
    processor = _make_processor()
    assert processor.build_reply(chat_id="123", text="   ") == (
        "Empty command. Use /help to see supported commands."
    )


def test_none_text_returns_none() -> None:
    processor = _make_processor()
    assert processor.build_reply(chat_id="123", text=None) is None


def test_unauthorized_chat_denied_before_empty_content_guard() -> None:
    processor = _make_processor(allowed_chat_ids=frozenset({"999"}))
    assert processor.build_reply(chat_id="123", text="") is None

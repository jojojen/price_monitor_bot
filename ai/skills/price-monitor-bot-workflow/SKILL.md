---
name: price-monitor-bot-workflow
description: Work on the standalone price_monitor_bot project. Use when modifying reusable market monitoring logic, TCG source parsers, image lookup flows, reference source configuration, Telegram command behavior, or project tests. Preserve the package boundaries between market_monitor, tcg_tracker, and price_monitor_bot, and verify source-role changes before shipping them.
---

# Price Monitor Bot Workflow

## Overview

Use this skill before changing pricing behavior, parser logic, or bot-facing flows in this repository.
This repo is intentionally split into reusable monitoring core, TCG-specific logic, and the standalone bot package.

## Layer Boundaries

Choose one primary layer before editing:

- `src/market_monitor`
  - Generic monitoring models, storage, HTTP helpers, normalization, pricing, and source catalog loading
- `src/tcg_tracker`
  - TCG-specific parsers, matching rules, live checks, hot-card logic, image lookup, and card catalog behavior
- `src/price_monitor_bot`
  - Standalone command, formatter, natural-language, and Telegram bot wiring

If a change crosses more than one layer, keep the shared logic in the lowest valid layer instead of duplicating it higher up.

## Source Rules

Treat source metadata as part of the product contract:

- `official_metadata` sources are for normalization and verification, not direct fair-value estimation
- `specialty_store` sources provide higher-trust ask and bid references
- `marketplace` and `market_content` sources provide live market depth, listing, and trend signals
- When updating `config/reference_sources.json`, keep `source_kind`, `reference_roles`, and `price_weight` aligned with real behavior
- Do not silently mix metadata-only sources into pricing logic

## Common Task Map

Use these entry points:

- Source parser changes
  - Read the parser module in `src/tcg_tracker/`
  - Update fixture-backed tests first when possible
  - Verify with `tests/test_yuyutei_parser.py` or the matching parser test file
- Pricing or source weighting changes
  - Read `src/market_monitor/pricing.py` and `config/reference_sources.json`
  - Verify with `tests/test_pricing.py` and `tests/test_reference_sources.py`
- Telegram command or formatter changes
  - Read `src/price_monitor_bot/commands.py`, `bot.py`, and `formatters.py`
  - Verify with `tests/test_commands.py` and `tests/test_telegram_bot.py`
- Image lookup or OCR-adjacent changes
  - Read `src/tcg_tracker/image_lookup.py` and `local_vision.py`
  - Verify with `tests/test_image_lookup.py`, `tests/test_local_vision.py`, and fixture metadata tests
- Hot-card ranking changes
  - Read `src/tcg_tracker/hot_cards.py`
  - Verify with `tests/test_hot_cards.py`

## Workflow

1. Identify the owning layer and state assumptions.
2. Read the nearby module and the most relevant tests before editing.
3. Add or adjust a regression test first for bug fixes and behavior changes.
4. Implement the smallest change that preserves the existing boundaries.
5. Run targeted tests before broader checks.
6. Explicitly call out skipped live-network verification.

## Verification

Prefer focused commands:

- `python -m pytest tests/test_pricing.py tests/test_reference_sources.py`
- `python -m pytest tests/test_yuyutei_parser.py`
- `python -m pytest tests/test_commands.py tests/test_telegram_bot.py`
- `python -m pytest tests/test_image_lookup.py tests/test_local_vision.py`
- `python -m pytest tests/test_hot_cards.py`

Use live checks only when the task truly requires them:

- `python -m pytest -m live_image_lookup`

## Guardrails

- Do not hardcode card catalogs or live price data.
- Do not weaken parser provenance or source attribution.
- Do not change public command behavior without updating tests.
- Prefer fixture-based verification over ad hoc manual assertions.

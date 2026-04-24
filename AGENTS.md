# AGENTS.md

This repository versions its Codex workflow files alongside the codebase.

## Repo-local skill

- Primary skill: `ai/skills/price-monitor-bot-workflow/SKILL.md`
- Install or refresh it into Codex with `scripts/install-codex-skills.ps1`
- Restart Codex after installation so the new skill is discovered

## Expectations

- Keep `market_monitor`, `tcg_tracker`, and `price_monitor_bot` responsibilities separate
- Treat source metadata as a contract, not a loose hint
- Prefer regression tests and fixture-backed parser checks
- Explicitly note when live-network verification was not run

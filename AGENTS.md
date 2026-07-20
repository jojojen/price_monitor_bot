# AGENTS.md

## First Principle — General Correctness

1. **Correctness first means general correctness, not correctness for one case.**
   Never make the current example pass with hardcoded keywords, values, output
   text, or exception branches. Prefer a structural solution that removes
   special cases; a sound fix should normally reduce total code and branch
   count. If a proposed fix adds case-specific code, stop and redesign it.

2. **Research uncertainty before coding.** When the correct general solution
   is unclear, consult current primary sources and proven implementations before
   changing code. Use that evidence to define a general contract or design;
   never replace uncertainty with a case-specific hardcode.

3. **Use ASD-STE100 as the language principle for comments and documentation.**
   During development, write code comments, technical notes, and user-facing
   project documentation in ASD Simplified Technical English style: short direct
   sentences, active voice, one instruction per sentence where practical,
   consistent terms for the same concept, and concrete nouns instead of vague
   references. Avoid idioms, rhetorical phrasing, hidden assumptions, and
   unnecessary adjectives. When exact ASD-STE100 compliance is impractical,
   prefer clarity, consistency, and unambiguous technical meaning.

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

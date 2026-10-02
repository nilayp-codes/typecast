# CLAUDE.md

This repo is the **Structured Output Agent** (package `soa` in `src/soa/`, CLI `soa`). Full specs live in `docs/`:
PRD, ARCHITECTURE, RULES, DESIGN, TASKS, MEMORY (each `docs/<NAME>.md`).

## Session start
1. Read `docs/MEMORY.md`.
2. Read `docs/RULES.md`.
3. Read the current phase of `docs/TASKS.md`.

Don't assume anything. If it isn't in the docs or the code, ask.

## Hard constraints (from RULES.md, compressed)
- Pydantic v2 only. No raw dicts across module boundaries; pass models.
- `mypy --strict` must pass on `src/`.
- Functions ≤ 40 lines.
- LLM calls go only through `soa/llm.py`.
- Tests never hit a real LLM; mock `llm.call()`.
- Prompts are module-level constants in `agent.py`.
- The schema shown to the model comes from `Model.model_json_schema()`, never hand-written.

## Workflow
1. Pick one `docs/TASKS.md` item. Only one at a time.
2. Implement it.
3. Run `uv run pytest` and `uv run mypy --strict src/`. Both must pass.
4. Tick its box in `docs/TASKS.md`.
5. Commit as `type: summary` (e.g. `feat: add schema validator`).
6. Move to the next item.

Stop and ask if a task is ambiguous or conflicts with a doc.

## Commands
```
uv run pytest               # tests
uv run mypy --strict src/   # type check
uv run soa ...              # run the CLI
```

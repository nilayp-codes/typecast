# ARCHITECTURE — Structured Output Agent

## Stack
- Python 3.11+, `uv` for env/deps
- Pydantic v2 — schemas + validation
- LiteLLM — provider-agnostic client (swap models in one line)
- Typer — CLI
- pytest — tests
- No DB. JSONL files for logs and eval results.

## Repo layout
```
structured-output-agent/
├── src/soa/
│   ├── schemas.py      # JobPosting, Skill, Location, enums
│   ├── tools.py        # tool functions + their arg/response models
│   ├── llm.py          # thin LiteLLM wrapper: call(), token/cost capture
│   ├── agent.py        # extract(): prompt → call → validate → retry
│   ├── logger.py       # AttemptLog model + JSONL writer
│   ├── eval.py         # run dataset, compute metrics, write table
│   └── cli.py          # `soa extract`, `soa eval`
├── data/
│   ├── jds/            # raw .txt job descriptions
│   └── labels/         # hand-labelled gold JSON (25-30 files)
├── logs/               # attempts.jsonl (gitignored)
├── results/            # eval outputs (committed)
├── tests/
├── scripts/            # standalone scripts, e.g. baseline.py (Phase 2 real-API baseline run)
├── .github/workflows/ci.yml   # pytest + mypy --strict src/ on push and PR (Phase 7)
└── docs/  (these .md files)
```

## Core flow
```
text ──► build prompt (schema + instructions)
          │
          ▼
     LLM call ──► raw string
          │
          ▼
  JobPosting.model_validate_json(raw)
     │ ok                     │ ValidationError
     ▼                        ▼
  log success           log failure (error + raw)
  return model          attempt < MAX? ──► append error to messages, retry
                              │ no
                              ▼
                       raise ExtractionFailed (logged)
```

## Tool flow
```
model requests tool ──► validate args (ToolArgs model)
                          │ invalid → return error to model as tool result
                          ▼
                     run tool ──► validate response (ToolResponse model)
                          │ invalid → log + raise (bug in our code, not the LLM)
                          ▼
                     return result to model
```

## Key decisions
| Decision | Choice | Why |
|---|---|---|
| Retry strategy | Feed exact Pydantic error back | Model self-corrects on specific errors far better than blind retry |
| Max attempts | 3 | Diminishing returns past 2; caps cost |
| API errors | Separate from validation: retryable (429/5xx/timeout) waits 2, 4, 8 s and re-sends the same call, max 3 times per call; non-retryable raises at once. Each is logged as `error_type="APIError"` and never consumes a validation attempt | A provider outage says nothing about the model's output, and must not burn the 3-attempt budget |
| Validation | `model_validate_json` (not `json.loads` + validate) | Single step, better error locations |
| JSON mode | Off by default, togglable. The eval runs it both on and off, as a required comparison (setups A-D) | Measure raw model behaviour first; answer "why not just native structured outputs?" with numbers |
| Logging | Pydantic `AttemptLog` → JSONL | Logs are themselves validated; trivially parseable for eval |
| Client | LiteLLM | Model swap for evals; aligns with LiteLLM ecosystem |

## Interfaces
```python
def extract(text: str, model: str, max_attempts: int = 3, json_mode: bool = False, *,
            doc_id: str = "adhoc", run_id: str | None = None, log_path: Path | None = None) -> ExtractionResult
# ExtractionResult: posting: JobPosting, attempts: int, api_retries: int, total_tokens: int, cost_usd: float,
#                   cost_known: bool, latency_ms: int   (totals over every model call; cost_known is False if any call's cost is unknown)
# doc_id / run_id / log_path only label and place the log lines; they do not change extraction.
# extract_once(...) is the single-attempt building block (used by the Phase 2 baseline): it keeps its signature and
# delegates to attempt_once(messages, ...), which extract() calls with the growing conversation.
```

## Eval setups: how retry and native mode are toggled
Both toggles are plain arguments, so the eval runs the same code path four ways:

| Setup | Retry loop | Native mode | Call |
|---|---|---|---|
| A | off | off | `extract(text, model, max_attempts=1, json_mode=False)` |
| B | on | off | `extract(text, model, max_attempts=3, json_mode=False)` |
| C | off | on | `extract(text, model, max_attempts=1, json_mode=True)` |
| D | on | on | `extract(text, model, max_attempts=3, json_mode=True)` |

- **Retry:** `max_attempts=1` is the no-retry path: one call, one validation, and failure raises `ExtractionFailed`
  after logging. `max_attempts` defaults to 3 in `extract()`; the CLI reads `SOA_MAX_ATTEMPTS` and passes it in.
- **Native mode:** native mode means the provider's schema mode, built from `JobPosting.model_json_schema()`
  (`response_format` of type `json_schema`, or the provider's equivalent via LiteLLM). It is not `json_object`:
  that only guarantees valid JSON, so it cannot test the "shape, not meaning" claim, and it is not one of the four
  setups. With `json_mode=True`, `agent.py` passes the model class to `llm.call()` as
  `response_model: type[BaseModel] | None` (`None` when native mode is off), not a schema dict, so rule 2 holds.
  `llm.py` calls `model_json_schema()` on it and builds the provider setting. Rule 9 still holds (the schema is
  generated from the model, never hand-written), and so does rule 5: no provider SDK or `response_format` outside
  `llm.py`. (As of Phase 2, `llm.call()` takes `json_mode: bool` and maps it to `json_object`; replacing that with
  `response_model` is a Phase 5 task.)
- **Provider schema subsets:** provider strict-schema modes support only a subset of JSON Schema. If a constraint
  must be dropped for the provider, record which ones in `results/` and still validate the output with our full
  Pydantic model. Dropped constraints are reported as a finding in the README.
- **Models** come from config/env (`SOA_MODEL_PRIMARY`, `SOA_MODEL_SECONDARY`) and are passed in as the `model`
  argument; nothing in the logic names a model (rule 6).
- **Model pair (2026-10-15):** both models are Qwen models served through OpenRouter. `SOA_MODEL_PRIMARY` is the weak
  one (`qwen-2.5-7b-instruct`), `SOA_MODEL_SECONDARY` the strong one (`qwen3-235b-a22b-2507`). One provider key
  (`OPENROUTER_API_KEY`) covers both. Which one is actually weaker is measured by the Phase 2 baseline, not assumed.
- **Unsupported setups:** if a model does not support native schema mode, `eval.py` records C and D for that model
  as "not supported" and never a number. How support is detected is decided in Phase 6.
- **Which documents:** the 4-setup comparison (A-D) runs on the gold set (25-30 JDs) for both models. Setup B
  (retry on, native off) also runs on all 50 JDs for the headline validity numbers.

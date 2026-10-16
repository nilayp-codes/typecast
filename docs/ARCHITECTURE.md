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
│   ├── tools.py        # normalize_skill, classify_location (deterministic post-processing) + arg/response models
│   ├── errors.py       # format_validation_errors (rule 11) and redact_keys, shared by agent.py, tools.py, cli.py
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
├── scripts/            # standalone scripts: baseline.py (Phase 2 real-API baseline run),
│                       #   make_label_templates.py (TODO templates in data/labels/), check_labels.py (validate labels)
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
The two tools (`normalize_skill`, `classify_location`) are **deterministic post-processing after validation. They
are not model-called.** The model never sees them and they cost no LLM calls, so the retry loop, attempt counting,
cost and the A-D setups are unchanged. Agentic tool calling (the model requests the tools mid-extraction) is a v2
experiment, to be measured against this.
```
validated JobPosting (raw, what the model produced)
        │
        ▼
 postprocess(raw, source text)
   skills:   normalize_skill(NormalizeSkillArgs) per skill ──► merge duplicates (required = any merged required)
   location: classify_location(JD header = first 10 non-empty lines) ──► FILL ONLY:
             city/country when null, mode when "unspecified"; model values are never overridden,
             disagreements (model value vs tool value) are logged and returned in ExtractionResult.disagreements
        │ invalid tool args → structured ToolError, logged; the original value is kept
        ▼
 re-validate the final JobPosting
        │ invalid → log + raise ToolResponseError (a bug in our code, not the LLM)
        ▼
 ExtractionResult(posting = normalized, raw_posting = raw)
```
Both are kept because setups A-D and the validity metrics are measured on `raw_posting`, and field accuracy is
reported raw and normalized (Phase 6). Post-processing can be turned off (`extract(..., normalize=False)`).

## Key decisions
| Decision | Choice | Why |
|---|---|---|
| Retry strategy | Feed exact Pydantic error back | Model self-corrects on specific errors far better than blind retry |
| Max attempts | 3 | Diminishing returns past 2; caps cost |
| API errors | Separate from validation: retryable (429/5xx/timeout) waits 2, 4, 8 s and re-sends the same call, max 3 times per call; non-retryable raises at once. Each is logged as `error_type="APIError"` and never consumes a validation attempt | A provider outage says nothing about the model's output, and must not burn the 3-attempt budget |
| Validation | `model_validate_json` (not `json.loads` + validate) | Single step, better error locations |
| Native mode (`json_mode`) | Off by default, togglable. There is ONE native mode: the provider's JSON-schema mode built from `JobPosting.model_json_schema()` (no plain `json_object` path). The eval runs it both on and off, as a required comparison (setups A-D) | Measure raw model behaviour first; answer "why not just native structured outputs?" with numbers |
| Logging | Pydantic `AttemptLog` → JSONL | Logs are themselves validated; trivially parseable for eval |
| Client | LiteLLM | Model swap for evals; aligns with LiteLLM ecosystem |

## Interfaces
```python
def extract(text: str, model: str, max_attempts: int = 3, json_mode: bool = False, *,
            doc_id: str = "adhoc", run_id: str | None = None, log_path: Path | None = None,
            normalize: bool = True) -> ExtractionResult
# ExtractionResult: posting: JobPosting (post-processed), raw_posting: JobPosting (what the model produced),
#                   disagreements: list[Disagreement] (field, model_value, tool_value; also logged),
#                   failed_attempts: list[AttemptFailure] (attempt, error_type, error; the rejected ones),
#                   attempts: int, api_retries: int, total_tokens: int, cost_usd: float, cost_known: bool,
#                   latency_ms: int   (totals over every model call; cost_known is False if any call's cost is unknown)
# doc_id / run_id / log_path only label and place the log lines; they do not change extraction.
# json_mode=True = native schema mode: every model call gets response_model=JobPosting (retries too).
# normalize=False skips tools.postprocess: posting == raw_posting. Validity metrics use raw_posting.
# Post-processing failing re-validation raises tools.ToolResponseError (our bug), not ExtractionFailed.
# ExtractionFailed carries doc_id, attempts (list[AttemptResult]), failures (list[AttemptFailure]), totals
# (tokens, cost, cost_known, latency) and last_error, so the CLI can report a failed run in the same format.
# cli.py reads SOA_MODEL_PRIMARY / SOA_MAX_ATTEMPTS and passes them in; extract() itself reads no environment.
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
  `llm.py`. The signature is `call(messages, model, response_model: type[BaseModel] | None = None)`; with a model
  it sends `response_format={"type": "json_schema", "json_schema": {"name": <class name>, "schema":
  <model_json_schema()>, "strict": True}}`. `json_mode` stays the boolean name on `extract()` and the CLI
  (`--json-mode`) and means `response_model=JobPosting`; there is no `json_object` path any more.
- **Provider rejects schema mode:** a 400 (LiteLLM `BadRequestError`, which includes `UnsupportedParamsError`) on
  a request that asked for schema mode becomes
  `LLMAPIError(retryable=False, "schema mode unsupported by <model>: <provider message>")` only when the
  provider's message (or the exception class name) mentions `response_format`, `json_schema` or `unsupported`.
  Any other 400 stays a generic non-retryable `LLMAPIError` with the provider's text. No backoff either way, and
  it is logged as an `APIError` line. `eval.py` reports C and D for a model that raises the "unsupported" form as
  "not supported".
- **No silent downgrade on OpenRouter:** OpenRouter drops request parameters the routed provider lacks. So for a
  `openrouter/...` model with `response_model` set, `llm.py` also sends
  `extra_body={"provider": {"require_parameters": True}}` (OpenRouter provider routing), which routes only to
  providers supporting every parameter. It is not sent without `response_model` or for other providers. If no
  provider qualifies OpenRouter returns an error instead of a degraded answer; its exact status for that case is
  not in the docs and is unverified (a 404 would not be labelled "unsupported", only a 400 is).
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

## Eval metrics (`eval.py`)
`run_eval(model, setup, data_dir, labels_dir, limit)` calls `extract()` once per JD (name order, first `limit`) with a
fresh `run_id` (`eval-<setup>-<id>-<doc>`) and the setup's `max_attempts` / `json_mode`. `ExtractionFailed` marks the
doc invalid and the run continues; a doc whose calls all fail at the API level is `api_error`. The result is an
`EvalReport`, built by the pure `build_report(docs, logs, labels)`: the validity numbers are read back from
`logs/attempts.jsonl` (only the run's own `run_id`s), the postings come from `ExtractionResult`.
- **n / n_scored / n_api_failed:** n = documents run; n_scored = those with at least one model response;
  n_api_failed = the rest. A doc with no response says nothing about the model, so it is left out of the validity
  numbers and counted separately. `APIError` log lines never count as attempts.
- **first-try %** = scored docs whose first model response (attempt 1) was valid; **final %** = scored docs with any
  valid response; **avg calls** = model responses per scored doc (so a failed doc contributes its attempts).
- **cost** = sum over model responses, `n/a` if any of them has an unknown cost (`APIError` lines are not counted);
  **avg latency** = summed model-call time per scored doc, averaged. **Tool disagreements** = sum over valid docs.
- **Field accuracy** (percent) is computed twice, on `raw_posting` and on the normalized `posting`, over labelled docs
  that produced a valid posting (validity is reported separately; `n_labelled` and `n_labelled_with_output` show the
  coverage). title / company: case-insensitive exact; seniority, mode, salary_mentioned, min_years_experience, city,
  country: exact; skills: precision / recall / F1 on casefolded names, micro-averaged over docs. `overall` = mean of
  the eight fields and skills F1. Labels that fail validation or still contain `TODO` are skipped and listed.
- **Output:** `write_results(report)` saves `results/<date>_<model-slug>_<setup>.json` (the whole `EvalReport`,
  including every doc's raw and normalized posting, so a run can be re-scored without API calls; a name already taken
  gets `_2`, `_3`..., nothing is overwritten) and appends one row to `results/summary.md`:
  `model | setup | n | first-try | final | avg calls | field acc (norm) | cost`. `n` reads `N (K scored)` when some
  docs got no model response; the percentages are over the scored ones. Stored values keep full precision; the table
  rounds to one decimal only for display.
- **Failure breakdown:** every rejected attempt (not `APIError`), counted by `error_type` and by the location of its
  first error line (numeric path parts become `*`: `skills.*.name`).

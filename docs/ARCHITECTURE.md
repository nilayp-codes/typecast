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
│   └── labels/         # hand-labelled gold JSON (10 files)
├── logs/               # attempts.jsonl (gitignored)
├── results/            # eval outputs (committed)
├── tests/
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
| Validation | `model_validate_json` (not `json.loads` + validate) | Single step, better error locations |
| JSON mode | Off by default, togglable | Measure raw model behaviour first, then compare |
| Logging | Pydantic `AttemptLog` → JSONL | Logs are themselves validated; trivially parseable for eval |
| Client | LiteLLM | Model swap for evals; aligns with LiteLLM ecosystem |

## Interfaces
```python
def extract(text: str, model: str, max_attempts: int = 3) -> ExtractionResult
# ExtractionResult: posting: JobPosting, attempts: int, total_tokens: int, cost_usd: float
```

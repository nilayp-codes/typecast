# Baseline: openrouter/qwen/qwen-2.5-7b-instruct, setup A: plain prompt, no retry

- date: 2026-11-03
- setup A: plain prompt, no retry (json_mode off; the model gets the prompt once, a validation failure is not retried)
- model: `openrouter/qwen/qwen-2.5-7b-instruct` (results are tied to this exact string)
- run_id: `baseline-A-2026-11-03-b56df7`
- API errors: the script waits 20/40/60 s and re-sends the same doc (503, per-minute 429); pause between docs 5s
- code at run time: commit `f4dcae1` (prompt frozen at `d90d70a`: the seniority text there added "Architect -> lead")
- JD set: 11 docs planned (run on jd_001-jd_011 only). The JD set will grow to 30 for Phase 6, so this is a baseline on a small sample, not the final set
- run status: complete
- docs attempted: 11 of 11 planned
- reached validation (got a model response): 11; API errors with no response: 0
- first-try valid: 9/11 (81.8%) of docs that reached validation
- total tokens (prompt + completion): 21,762
- total cost: $0.002580 (ESTIMATE: LiteLLM has no price for this model, so cost_known=False on every call; computed from OpenRouter's list prices for the tokens used. The billed amount may differ by provider)

## Per document

| doc_id | ok | error_type | first error loc | total tokens | latency (ms) | cost (USD) | cost_known | api retries |
|---|---|---|---|---|---|---|---|---|
| jd_001 | yes | - | - | 1,684 | 3,180 | 0.000183 | False (cost estimated from list price) | 0 |
| jd_002 | no | ValidationError | skills | 2,640 | 6,310 | 0.000345 | False (cost estimated from list price) | 0 |
| jd_003 | no | ValidationError | skills | 2,754 | 4,720 | 0.000342 | False (cost estimated from list price) | 0 |
| jd_004 | yes | - | - | 2,134 | 3,235 | 0.000247 | False (cost estimated from list price) | 0 |
| jd_005 | yes | - | - | 2,046 | 6,012 | 0.000255 | False (cost estimated from list price) | 0 |
| jd_006 | yes | - | - | 1,548 | 1,526 | 0.000169 | False (cost estimated from list price) | 0 |
| jd_007 | yes | - | - | 1,301 | 1,507 | 0.000144 | False (cost estimated from list price) | 0 |
| jd_008 | yes | - | - | 1,837 | 3,473 | 0.000224 | False (cost estimated from list price) | 0 |
| jd_009 | yes | - | - | 2,015 | 3,116 | 0.000247 | False (cost estimated from list price) | 0 |
| jd_010 | yes | - | - | 2,125 | 3,498 | 0.000241 | False (cost estimated from list price) | 0 |
| jd_011 | yes | - | - | 1,678 | 1,684 | 0.000182 | False (cost estimated from list price) | 0 |

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

| failure kind | docs | count |
|---|---|---|
| constraint: too_long on skills | jd_002, jd_003 | 2 |

## First error line per failed document (API errors in full)

- jd_002: skills: List should have at most 30 items after validation, not 39 (got: [{'name': 'LLM prompt engineering', 'required': True}, {'...)
- jd_003: skills: List should have at most 30 items after validation, not 31 (got: [{'name': 'Python', 'required': True}, {'name': 'FastAPI'...)

# Baseline: openrouter/qwen/qwen3-235b-a22b-2507, setup A: plain prompt, no retry

- date: 2026-11-03
- setup A: plain prompt, no retry (json_mode off; the model gets the prompt once, a validation failure is not retried)
- model: `openrouter/qwen/qwen3-235b-a22b-2507` (results are tied to this exact string)
- run_id: `baseline-A-2026-11-03-70d7cc`
- API errors: the script waits 20/40/60 s and re-sends the same doc (503, per-minute 429); pause between docs 5s
- code at run time: commit `f4dcae1` (prompt frozen at `d90d70a`: the seniority text there added "Architect -> lead")
- JD set: 11 docs planned (run on jd_001-jd_011 only). The JD set will grow to 30 for Phase 6, so this is a baseline on a small sample, not the final set
- run status: complete
- docs attempted: 11 of 11 planned
- reached validation (got a model response): 11; API errors with no response: 0
- first-try valid: 7/11 (63.6%) of docs that reached validation
- total tokens (prompt + completion): 20,810
- total cost: $0.003291 (ESTIMATE: LiteLLM has no price for this model, so cost_known=False on every call; computed from OpenRouter's list prices for the tokens used. The billed amount may differ by provider)

## Per document

| doc_id | ok | error_type | first error loc | total tokens | latency (ms) | cost (USD) | cost_known | api retries |
|---|---|---|---|---|---|---|---|---|
| jd_001 | yes | - | - | 1,623 | 6,668 | 0.000185 | False (cost estimated from list price) | 0 |
| jd_002 | no | ValidationError | skills | 2,289 | 23,159 | 0.000418 | False (cost estimated from list price) | 0 |
| jd_003 | no | ValidationError | skills | 2,595 | 18,218 | 0.000466 | False (cost estimated from list price) | 0 |
| jd_004 | no | ValidationError | skills | 2,255 | 18,466 | 0.000415 | False (cost estimated from list price) | 0 |
| jd_005 | no | ValidationError | skills | 1,882 | 21,861 | 0.000327 | False (cost estimated from list price) | 0 |
| jd_006 | yes | - | - | 1,553 | 10,306 | 0.000207 | False (cost estimated from list price) | 0 |
| jd_007 | yes | - | - | 1,356 | 5,714 | 0.000212 | False (cost estimated from list price) | 0 |
| jd_008 | yes | - | - | 1,642 | 13,180 | 0.000242 | False (cost estimated from list price) | 0 |
| jd_009 | yes | - | - | 1,876 | 15,298 | 0.000315 | False (cost estimated from list price) | 0 |
| jd_010 | yes | - | - | 1,971 | 8,007 | 0.000239 | False (cost estimated from list price) | 0 |
| jd_011 | yes | - | - | 1,768 | 12,865 | 0.000264 | False (cost estimated from list price) | 0 |

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

| failure kind | docs | count |
|---|---|---|
| constraint: too_long on skills | jd_002, jd_003, jd_004, jd_005 | 4 |

## First error line per failed document (API errors in full)

- jd_002: skills: List should have at most 30 items after validation, not 43 (got: [{'name': 'RAG', 'required': True}, {'name': 'Prompt Engi...)
- jd_003: skills: List should have at most 30 items after validation, not 47 (got: [{'name': 'Kafka', 'required': True}, {'name': 'NATS', 'r...)
- jd_004: skills: List should have at most 30 items after validation, not 42 (got: [{'name': 'Python', 'required': True}, {'name': 'Large La...)
- jd_005: skills: List should have at most 30 items after validation, not 31 (got: [{'name': 'Python', 'required': True}, {'name': 'LangChai...)

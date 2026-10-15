# Baseline: gemini/gemini-3-flash-preview, single shot, no retries

- date: 2026-10-15
- model: `gemini/gemini-3-flash-preview` (results are tied to this exact string)
- run_id: `baseline-2026-10-15-b3400c`
- json_mode: off; retries: none; pause between calls: 5s
- code at run time: commit `619be12` (prompt frozen, `619be12`)
- first-try valid: 4/11
- total tokens (prompt + completion): 15,907

## Per document

| doc_id | ok | error_type | first error loc | total tokens | latency (ms) |
|---|---|---|---|---|---|
| jd_001 | no | APIError | - | - | - |
| jd_002 | yes | - | - | 4,115 | 9,261 |
| jd_003 | no | APIError | - | - | - |
| jd_004 | yes | - | - | 3,301 | 6,324 |
| jd_005 | yes | - | - | 4,265 | 10,672 |
| jd_006 | no | APIError | - | - | - |
| jd_007 | no | APIError | - | - | - |
| jd_008 | yes | - | - | 4,226 | 12,085 |
| jd_009 | no | APIError | - | - | - |
| jd_010 | no | APIError | - | - | - |
| jd_011 | no | APIError | - | - | - |

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

| failure kind | docs | count |
|---|---|---|
| API error | jd_001, jd_003, jd_006, jd_007, jd_009, jd_010, jd_011 | 7 |

## First error line per failed document

- jd_001: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - { "error": { "code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.", "status": "UNAVAILABLE" } }
- jd_003: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - { "error": { "code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.", "status": "UNAVAILABLE" } }
- jd_006: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - { "error": { "code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.", "status": "UNAVAILABLE" } }
- jd_007: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - { "error": { "code": 503, "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.", "status": "UNAVAILABLE" } }
- jd_009: APIError: RateLimitError: litellm.RateLimitError: litellm.RateLimitError: geminiException - { "error": { "code": 429, "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limit
- jd_010: APIError: RateLimitError: litellm.RateLimitError: litellm.RateLimitError: geminiException - { "error": { "code": 429, "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limit
- jd_011: APIError: RateLimitError: litellm.RateLimitError: litellm.RateLimitError: geminiException - { "error": { "code": 429, "message": "You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limit

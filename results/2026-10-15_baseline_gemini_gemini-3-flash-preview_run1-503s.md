# Baseline: gemini/gemini-3-flash-preview, single shot, no retries

- date: 2026-10-15
- model: `gemini/gemini-3-flash-preview` (results are tied to this exact string)
- run_id: `baseline-2026-10-15-f9cead`
- json_mode: off; retries: none; pause between calls: 5s
- code at run time: commit `619be12` (prompt frozen, `619be12`)
- first-try valid: 1/11
- total tokens (prompt + completion): 4,922

## Per document

| doc_id | ok | error_type | first error loc | total tokens | latency (ms) |
|---|---|---|---|---|---|
| jd_001 | no | APIError | - | - | - |
| jd_002 | no | APIError | - | - | - |
| jd_003 | no | APIError | - | - | - |
| jd_004 | no | APIError | - | - | - |
| jd_005 | yes | - | - | 4,922 | 14,282 |
| jd_006 | no | APIError | - | - | - |
| jd_007 | no | APIError | - | - | - |
| jd_008 | no | APIError | - | - | - |
| jd_009 | no | APIError | - | - | - |
| jd_010 | no | APIError | - | - | - |
| jd_011 | no | APIError | - | - | - |

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

| failure kind | docs | count |
|---|---|---|
| API error | jd_001, jd_002, jd_003, jd_004, jd_006, jd_007, jd_008, jd_009, jd_010, jd_011 | 10 |

## First error line per failed document

- jd_001: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_002: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_003: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_004: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_006: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_007: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_008: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_009: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_010: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {
- jd_011: APIError: ServiceUnavailableError: litellm.ServiceUnavailableError: GeminiException - {

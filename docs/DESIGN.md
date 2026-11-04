# DESIGN — Structured Output Agent

No UI in v1. "Design" here = schema design, CLI output, log format, README presentation.

## 1. Schema design
```python
class Seniority(StrEnum):
    intern = "intern"; junior = "junior"; mid = "mid"; senior = "senior"; lead = "lead"

class WorkMode(StrEnum):
    remote = "remote"; hybrid = "hybrid"; onsite = "onsite"; unspecified = "unspecified"

# Every model: model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)
# Every field: Field(description=...) so model_json_schema() explains it to the LLM.

class Skill(BaseModel):
    name: str = Field(min_length=1, max_length=40)   # normalized, e.g. "PostgreSQL"
    required: bool

class Location(BaseModel):
    city: str | None       # required but nullable: the model must emit null explicitly
    country: str | None    # required but nullable
    mode: WorkMode

class JobPosting(BaseModel):
    title: str = Field(min_length=2, max_length=100)
    company: str | None    # required but nullable
    seniority: Seniority
    min_years_experience: int | None = Field(ge=0, le=30)   # required but nullable
    skills: list[Skill] = Field(min_length=1, max_length=30)
    location: Location
    salary_mentioned: bool
```
Validators:
- skills: no duplicate names. Compare with whitespace stripped, then casefolded.
- if seniority == intern → min_years_experience must be None or 0

### Gold-label guide (`data/labels/`)
The rules below are the labelling rules AND the text of the schema `Field(description=...)` strings, so the
model's instructions and the gold labels match. Labels are model-drafted and human-verified (corrections in
`docs/label_corrections.log`).
- **skills scope:** named technologies, tools, languages, frameworks, platforms, protocols, named technical methods
  (e.g. RAG, LoRA) and CS fundamentals (e.g. data structures and algorithms). Exclude soft skills, degrees,
  certifications, spoken languages, industry domains, architecture concepts and vague entries ("Coding", "AI").
- **skill.required:** true if in the requirements, the described work or the posting's Skills list. Alternatives
  ("A or B") take the status of their section. False if preferred, nice-to-have, bonus, "a plus", optional, only
  an example ("e.g."), or only in the company's tech-stack section (include those, as false).
- **skill names:** canonical name if known, else the JD's own wording.
- **more than 30 skills (labelling only, not in the schema text):** required first, then preferred, each in order
  of appearance, cut at 30.
- **seniority:** (1) title first: Intern -> intern; Junior/Associate -> junior; Senior/Sr -> senior;
  Lead/Staff/Principal/Architect -> lead. (2) Then level words in the description ("entry level", "fresher",
  "junior/mid-level"; use the lowest level stated). (3) Then minimum years: 0-1 junior, 2-4 mid, 5+ senior.
  (4) Else mid.
- **min_years_experience:** the lower bound; the requirements section beats the header; "no experience required"
  -> 0; not stated -> null.
- **location.city:** one city -> that city, normalised (e.g. "Mumbai Metropolitan Region" -> Mumbai); several cities
  in one metro region -> the region name (e.g. Delhi NCR); several distinct cities -> null.
- **location.country:** filled when all places are in one country; fully remote with no place -> null.
- **location.mode:** an explicit label, policy or statement ("Remote Work Policy", "work remotely", "working out of
  our office"). Onsite plus an explicit remote option -> hybrid. Directly contradicting labels -> unspecified.
  Office perks alone are not a mode statement -> unspecified.
- **salary_mentioned:** true only for a pay figure or range (monthly pay and stipends count); "competitive",
  budgets and company revenue do not.

**Principle:** constraints should be strict enough that a sloppy output *fails*, so the retry loop has real work to do.

## 2. CLI output
`soa extract jd.txt`
```
stderr:  ✓ Valid after 2 attempts  (1,240 tokens · $0.0009 · 2.1s)
stderr:    attempt 1 ✗ skills.3.name: String should have at most 40 characters
stdout:  { ...pretty JSON... }
```
**stdout carries only the JSON** (so `soa extract jd.txt > out.json` and pipes work); all status output goes to stderr.
`soa extract <file> [--model M] [--json-mode] [--no-normalize] [--max-attempts N]`
- `--json-mode` = the provider's JSON-schema mode, built from `JobPosting.model_json_schema()` (the only native mode).
  A provider that rejects it exits 2 with `schema mode unsupported by <model>`.
- Defaults: model from `SOA_MODEL_PRIMARY`, attempts from `SOA_MAX_ATTEMPTS` (else 3). `.env` is loaded from the
  working directory (or a parent); variables already set are not overridden. The file is read as UTF-8 (a BOM is dropped).
- Header: `✓ Valid after N attempt(s)  (tokens · cost · seconds)`. Cost is `n/a` when `cost_known` is false. Seconds is the
  summed model-call time (backoff waits excluded). `· N API retries` is appended when there were any. A failed run
  prints `✗ Failed after N attempts  (...)`.
- Then one block per rejected attempt (`attempt N ✗ <first error>`, any further errors aligned underneath), then
  any `tool disagreement` lines, then pretty JSON of the normalized posting (`--no-normalize`: the model's raw
  output, and no post-processing). Only the JSON goes to stdout; the header, attempt lines, tool disagreement
  lines and error messages go to stderr. A failed run prints nothing on stdout.
- Exit codes: 0 success; 1 `ExtractionFailed` (or a post-processing bug); 2 bad input file or config, or an API
  error (including retries exhausted).
- Windows console: stdout and stderr are switched to UTF-8; if that is impossible for stderr (where the status
  lines go) and the stream cannot carry ✓/✗/·, they are printed as `OK` / `FAIL` / `-`. API key values are never printed.

`soa eval --model <model>`
```
model         n   first-try  final   avg calls  cost
<model>       30  xx.x%      xx.x%   x.xx       $x.xx   (placeholder values, not results)
```
`soa eval --model <m> [--setup A|B|C|D|all] [--limit N] [--data-dir PATH] [--labels-dir PATH] [--results-dir PATH] [--pause SECONDS] [--resume EVAL_ID] [--yes]`.
`--pause` is the wait between documents (default 2; `SOA_EVAL_PAUSE` in the environment replaces the default, and the
flag beats both). Every invocation prints an `eval id`; `--resume <eval id>` continues that run after it stopped: documents
it already finished (valid, or all attempts used) are rebuilt from `logs/attempts.jsonl` and not sent again, documents
that were cut short or only had API errors run again, and the report is completed in place (same `results/` file, same
`summary.md` row) instead of a second one being added. It needs the same `--model`.
`--setup` picks the setup (default B; `all` runs A, B, C, D in turn): A = plain prompt, no retry; B = plain prompt +
retry; C = schema mode, no retry; D = schema mode + retry. (This replaces the earlier `--json-mode` / `--no-retry`
pair on `eval`; `soa extract` keeps `--json-mode`.) `--data-dir` is the folder of JD `.txt` files (default
`data/jds/`); the gold labels are read from `--labels-dir` (default: the sibling `labels/` folder of `--data-dir`), and
field accuracy is skipped when no label is usable. **These are real model calls.** Before sending anything the command
prints the plan (model, documents, labels, calls per setup, `estimated model calls: LOW to HIGH`: every doc valid at
once vs every doc using all its attempts; API-level retries come on top) and asks `Make these model calls now?`;
`--yes` skips the question, and declining (or no terminal input) exits 1 having sent nothing. Each setup then saves
`results/<date>_<model-slug>_<setup>.json` and appends a row to `results/summary.md` (`--results-dir`), and the
rows are printed as a markdown table followed by notes (API errors, docs without a model response, skipped labels,
tool disagreements). Metric definitions: ARCHITECTURE.md "Eval metrics". The runs are combined into the 4-setup
comparison table:
```
EXAMPLE ONLY - placeholder values, not results
model    setup  retry  native  first-try  final  avg calls  field acc  cost
<model>  A      off    off     xx.x%      xx.x%  1.00       xx.x%      $x.xx
<model>  B      on     off     xx.x%      xx.x%  x.xx       xx.x%      $x.xx
<model>  C      off    on      xx.x%      xx.x%  1.00       xx.x%      $x.xx
<model>  D      on     on      xx.x%      xx.x%  x.xx       xx.x%      $x.xx
<model2> C      off    on      not supported (no numbers)
```

## 3. Log format (`logs/attempts.jsonl`, one line per attempt)
```json
{"run_id":"…","doc_id":"jd_014","attempt":1,"model":"gpt-4o-mini",
 "setup":"B","json_mode":false,
 "ok":false,"error_type":"ValidationError","error":"skills.3.name: …",
 "raw_output":"…","latency_ms":1840,"prompt_tokens":910,
 "completion_tokens":330,"cost_usd":0.0006,"cost_known":true,"cost_source":"provider","ts":"…"}
```
`cost_source` says where `cost_usd` came from: `"provider"` = the cost the provider reported for that request
(OpenRouter's `usage.cost`, in USD), `"estimate"` = list price (LiteLLM's price map, else OpenRouter's public model
list) times the token counts, `"none"` = no cost available (`cost_usd` is 0.0 and `cost_known` is false), `null` =
the line was written before the field existed. Eval reports carry `cost_source` too: `provider` only if every
response had the provider's own figure, otherwise `estimate`, or n/a if any cost is unknown; `summary.md` marks an
estimate with `~`.
`temperature` is the temperature SENT with the request (`null` = none sent, so the provider's default applied; today
`llm.call` sends none unless a caller passes one). `provider` is the upstream provider OpenRouter routed the request to,
read from the response's `provider` field (`null` = not reported, or a line written before the field existed). Eval
reports carry `temperature` (the single value sent, else `null`) and `providers` (responses per upstream provider).
`setup` is `"A"`, `"B"`, `"C"` or `"D"` (see ARCHITECTURE.md "Eval setups"), or `null` for non-eval runs.
`json_mode` is a bool (true = schema mode was requested). Both fields are on `AttemptLog` and written on every line,
including `APIError` lines. They default to `null` / `false`, so lines written before Phase 6 still parse.

## 4. README layout
1. CI badge at the top, then the one-line pitch
2. **Eval table** (above the fold)
3. **Why not just native JSON mode?** The 4-setup comparison table (A-D) plus our 3-point answer:
   (a) native modes guarantee shape, not meaning (they miss duplicate skills, the intern/experience rule, wrong
   seniority); (b) valid JSON != correct data (field accuracy measures this); (c) open-weight OpenRouter models
   often lack strict schema mode.
4. 15-sec GIF of `soa extract`
5. How it works (the flow diagram from ARCHITECTURE)
6. Most common failure types (from logs) + what fixed them
7. **Design decisions**: strict mode, required-but-nullable fields, error-feedback format (one line per error),
   why feeding errors back beats blind retry
8. Run it yourself
9. What I'd do next (includes the validator-only MCP server and the PyPI package)

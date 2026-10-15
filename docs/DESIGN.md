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

**Seniority mapping (also the labelling rule for `data/labels/`; gold labels must follow it):**
1. Title first: Intern -> intern; Junior/Associate -> junior; Senior -> senior; Lead, Staff or Principal -> lead.
2. If the title gives no level, use minimum years required: 0-1 -> junior, 2-4 -> mid, 5+ -> senior.
3. If neither is stated -> mid.

**Principle:** constraints should be strict enough that a sloppy output *fails*, so the retry loop has real work to do.

## 2. CLI output
`soa extract jd.txt`
```
✓ Valid after 2 attempts  (1,240 tokens · $0.0009 · 2.1s)
  attempt 1 ✗ skills.3.name: String should have at most 40 characters
{ ...pretty JSON... }
```
`soa eval --model gpt-4o-mini`
```
model         n   first-try  final   avg calls  cost
gpt-4o-mini   50  82.0%      100%    1.20       $0.04
```
`soa eval --model <m> [--json-mode] [--no-retry]`: the two flags select the setup. A = `--no-retry`, B = (default),
C = `--no-retry --json-mode`, D = `--json-mode`. The runs are combined into the 4-setup comparison table:
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
 "ok":false,"error_type":"ValidationError","error":"skills.3.name: …",
 "raw_output":"…","latency_ms":1840,"prompt_tokens":910,
 "completion_tokens":330,"cost_usd":0.0006,"ts":"…"}
```

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

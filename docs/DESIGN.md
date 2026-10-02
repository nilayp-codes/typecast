# DESIGN — Structured Output Agent

No UI in v1. "Design" here = schema design, CLI output, log format, README presentation.

## 1. Schema design
```python
class Seniority(str, Enum):
    intern = "intern"; junior = "junior"; mid = "mid"; senior = "senior"; lead = "lead"

class WorkMode(str, Enum):
    remote = "remote"; hybrid = "hybrid"; onsite = "onsite"; unspecified = "unspecified"

class Skill(BaseModel):
    name: str = Field(min_length=1, max_length=40)   # normalized, e.g. "PostgreSQL"
    required: bool

class Location(BaseModel):
    city: str | None
    country: str | None
    mode: WorkMode

class JobPosting(BaseModel):
    title: str = Field(min_length=2, max_length=100)
    company: str | None
    seniority: Seniority
    min_years_experience: int | None = Field(default=None, ge=0, le=30)
    skills: list[Skill] = Field(min_length=1, max_length=30)
    location: Location
    salary_mentioned: bool
```
Validators:
- skills: no duplicate names (case-insensitive)
- if seniority == intern → min_years_experience must be None or 0

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

## 3. Log format (`logs/attempts.jsonl`, one line per attempt)
```json
{"run_id":"…","doc_id":"jd_014","attempt":1,"model":"gpt-4o-mini",
 "ok":false,"error_type":"ValidationError","error":"skills.3.name: …",
 "raw_output":"…","latency_ms":1840,"prompt_tokens":910,
 "completion_tokens":330,"cost_usd":0.0006,"ts":"…"}
```

## 4. README layout
1. One-line pitch
2. **Eval table** (above the fold)
3. 15-sec GIF of `soa extract`
4. How it works (the flow diagram from ARCHITECTURE)
5. Most common failure types (from logs) + what fixed them
6. Run it yourself
7. What I'd do next

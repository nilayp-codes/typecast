# Typecast

LLM agent that turns messy job descriptions into schema-validated, typed JSON — and measures how often that fails.

**Key findings** (two Qwen models via OpenRouter, 30 real job descriptions):

- **Schema mode makes output valid, not correct.** In setup C both models
  were valid 98.9% of the time, at lower cost than retry, yet end-to-end
  accuracy was 67.1% (7B) and 84.7% (235B).
- **Error-feedback retry fixed the small model, not the large one.**
  Setup B was 98.9% valid for the 7B but 88.9% for the 235B, which failed
  on jd_002 in all four setup-B runs. A count-aware retry message did not
  change that.
- **No run reached my 90% field-accuracy goal (best: 87.3%), and runs
  vary.** The same setup moved by about 8 points end to end between runs,
  so a gap between two single runs is not conclusive.

**60-second walkthrough:**



https://github.com/user-attachments/assets/c4a2572c-88a2-49a3-8e20-176010877154


CLI and package name: `soa` · MCP server on PyPI: [`typecast-mcp`](https://pypi.org/project/typecast-mcp/)

[![CI](https://github.com/nilayp-codes/typecast/actions/workflows/ci.yml/badge.svg)](https://github.com/nilayp-codes/typecast/actions/workflows/ci.yml)

## Results

Two open-weight models through OpenRouter, 30 real job descriptions, four setups. Field accuracy is scored against 20
hand-verified gold labels (see [Limitations](#limitations) for how they were made). With 3 runs, the figure is the
mean and the brackets are min–max across the runs; with 1 run it is that single run. A = plain prompt, no retry; B =
plain prompt + error-feedback retry; C = provider schema mode, no retry; D = schema mode + retry.

| model | setup | runs | final valid | end-to-end accuracy | cost per 30-doc run |
|---|---|---|---|---|---|
| qwen-2.5-7b-instruct | A · plain | 1 | 83.3% | 50.4% | $0.0081 |
| qwen-2.5-7b-instruct | B · plain + retry | 3 | 98.9% (96.7%-100.0%) | 67.1% (62.9%-70.4%) | $0.0103 ($0.0101-$0.0105) |
| qwen-2.5-7b-instruct | C · schema | 3 | 98.9% (96.7%-100.0%) | 67.1% (65.0%-69.8%) | $0.0076 ($0.0076-$0.0076) |
| qwen-2.5-7b-instruct | D · schema + retry | 1 | 100.0% | 70.0% | $0.0080 |
| qwen3-235b-a22b-2507 | A · plain | 1 | 70.0% | 60.4% | $0.0120 |
| qwen3-235b-a22b-2507 | B · plain + retry | 3 | 88.9% (86.7%-93.3%) | 75.6% (72.8%-80.9%) | $0.0211 ($0.0199-$0.0234) |
| qwen3-235b-a22b-2507 | C · schema | 3 | 98.9% (96.7%-100.0%) | 84.7% (82.8%-85.8%) | $0.0136 ($0.0134-$0.0140) |
| qwen3-235b-a22b-2507 | D · schema + retry | 1 | 100.0% | 85.4% | $0.0143 |

<details>
<summary>Full table: first-try valid, avg calls and field accuracy too</summary>

| model | setup | runs | first-try valid | final valid | avg calls | field accuracy (valid outputs) | end-to-end accuracy | cost per 30-doc run |
|---|---|---|---|---|---|---|---|---|
| qwen-2.5-7b-instruct | A: plain prompt, no retry | single run | 83.3% | 83.3% | 1.00 | 62.9% | 50.4% | $0.0081 |
| qwen-2.5-7b-instruct | B: plain prompt + error-feedback retry | mean of 3 runs (min–max) | 82.2% (80.0%-83.3%) | 98.9% (96.7%-100.0%) | 1.20 (1.17-1.23) | 68.2% (66.0%-70.4%) | 67.1% (62.9%-70.4%) | $0.0103 ($0.0101-$0.0105) |
| qwen-2.5-7b-instruct | C: schema mode, no retry | mean of 3 runs (min–max) | 98.9% (96.7%-100.0%) | 98.9% (96.7%-100.0%) | 1.00 (1.00-1.00) | 68.2% (66.5%-69.8%) | 67.1% (65.0%-69.8%) | $0.0076 ($0.0076-$0.0076) |
| qwen-2.5-7b-instruct | D: schema mode + retry | single run | 96.7% | 100.0% | 1.03 | 70.0% | 70.0% | $0.0080 |
| qwen3-235b-a22b-2507 | A: plain prompt, no retry | single run | 70.0% | 70.0% | 1.00 | 86.6% | 60.4% | $0.0120 |
| qwen3-235b-a22b-2507 | B: plain prompt + error-feedback retry | mean of 3 runs (min–max) | 76.7% (73.3%-80.0%) | 88.9% (86.7%-93.3%) | 1.34 (1.33-1.37) | 85.6% (85.1%-86.0%) | 75.6% (72.8%-80.9%) | $0.0211 ($0.0199-$0.0234) |
| qwen3-235b-a22b-2507 | C: schema mode, no retry | mean of 3 runs (min–max) | 98.9% (96.7%-100.0%) | 98.9% (96.7%-100.0%) | 1.00 (1.00-1.00) | 86.1% (85.6%-87.0%) | 84.7% (82.8%-85.8%) | $0.0136 ($0.0134-$0.0140) |
| qwen3-235b-a22b-2507 | D: schema mode + retry | single run | 100.0% | 100.0% | 1.00 | 85.4% | 85.4% | $0.0143 |

</details>

How to read it:

- **"Mean of 3 runs (min–max)"** (B and C) is the mean of three runs on the same 30 documents, with the lowest and
  highest run in brackets. It is a spread across runs, not an interval estimate. The B runs are the three runs made
  after the retry-feedback fix described below. The C runs are the first C run plus two repeats. **"Single run"**
  (A and D) is one draw each: the repeats show the same setup moving by about 8 points end to end between runs
  (235B setup B: 72.8% to 80.9%). Each run's own 95% interval (Wilson for the validity columns, bootstrap for the
  accuracy columns) is in the per-run table at the end of this section; the single runs of A and D have none, because
  they were made before the interval code existed.
- **First-try / final valid**: share of the 30 documents whose first reply, or any reply within 3 attempts, passed
  validation. **Avg calls**: model calls per document.
- **Field accuracy (valid outputs)** is measured only on documents that produced a valid posting, so it flatters a
  setup that fails often (235B setup A: 86.6% field accuracy, but only 70.0% of the labelled documents had a valid
  output to score). **End-to-end accuracy** counts a failed document as wrong on every field and is the number to
  trust when setups differ in validity.
- Cost is the provider-reported cost of one run over the 30 documents. Temperature was not set (the provider default
  applied, value unknown).

### My targets, and whether they were met

I set three targets for setup B before running anything: final validity of 30/30, at most 1.3 calls per document on
average, and at least 90% field accuracy. Result, per run (the three runs after the retry fix):

| target (setup B) | qwen-2.5-7b-instruct | qwen3-235b-a22b-2507 |
|---|---|---|
| final validity 30/30 | **met in 2 of 3 runs** (100.0%, 100.0%, 96.7%) | **missed in all 3** (86.7%, 93.3%, 86.7%) |
| avg calls <= 1.3 | met (1.20, 1.17, 1.23) | **missed** (1.37, 1.33, 1.33) |
| field accuracy >= 90% | **missed** (70.4%, 68.1%, 66.0%) | **missed** (85.7%, 85.1%, 86.0%) |

The 90% field-accuracy target was missed by every run of both models, in every setup. The best number anywhere is
87.3% (235B, setup B, first-round run).

<details>
<summary>Every run, with 95% intervals (results/summary.md, verbatim)</summary>

Rows 1-8: the first round, one run per model and setup (A, B, C, D for the 7B, then for the 235B). These were written
before the interval code existed, so they have none. Rows 9-18: the repeat round, in order 7B B x3, 7B C x2, 235B B x3,
235B C x2. Intervals are in brackets. Columns: `field acc` is over valid outputs, `coverage` is the share of the 20
labelled documents with a valid output, `e2e acc` counts failures as wrong.

| model | setup | n | first-try | final | avg calls | field acc (norm) | coverage | e2e acc (norm) | cost |
|---|---|---|---|---|---|---|---|---|---|
| openrouter/qwen/qwen-2.5-7b-instruct | A | 30 | 83.3% | 83.3% | 1.00 | 62.9% | 80.0% | 50.4% | $0.0081 |
| openrouter/qwen/qwen-2.5-7b-instruct | B | 30 | 86.7% | 96.7% | 1.17 | 68.3% | 95.0% | 64.8% | $0.0102 |
| openrouter/qwen/qwen-2.5-7b-instruct | C | 30 | 100.0% | 100.0% | 1.00 | 69.8% | 100.0% | 69.8% | $0.0076 |
| openrouter/qwen/qwen-2.5-7b-instruct | D | 30 | 96.7% | 100.0% | 1.03 | 70.0% | 100.0% | 70.0% | $0.0080 |
| openrouter/qwen/qwen3-235b-a22b-2507 | A | 30 | 70.0% | 70.0% | 1.00 | 86.6% | 70.0% | 60.4% | $0.0120 |
| openrouter/qwen/qwen3-235b-a22b-2507 | B | 30 | 83.3% | 86.7% | 1.33 | 87.3% | 85.0% | 74.3% | $0.0186 |
| openrouter/qwen/qwen3-235b-a22b-2507 | C | 30 | 100.0% | 100.0% | 1.00 | 85.8% | 100.0% | 85.8% | $0.0140 |
| openrouter/qwen/qwen3-235b-a22b-2507 | D | 30 | 100.0% | 100.0% | 1.00 | 85.4% | 100.0% | 85.4% | $0.0143 |
| openrouter/qwen/qwen-2.5-7b-instruct | B | 30 | 80.0% [62.7-90.5] | 100.0% [88.6-100.0] | 1.20 | 70.4% [64.4-76.9] | 100.0% | 70.4% [64.4-76.9] | $0.0104 |
| openrouter/qwen/qwen-2.5-7b-instruct | B | 30 | 83.3% [66.4-92.7] | 100.0% [88.6-100.0] | 1.17 | 68.1% [61.4-74.7] | 100.0% | 68.1% [61.4-74.7] | $0.0101 |
| openrouter/qwen/qwen-2.5-7b-instruct | B | 30 | 83.3% [66.4-92.7] | 96.7% [83.3-99.4] | 1.23 | 66.0% [59.6-72.2] | 95.0% | 62.9% [54.5-70.5] | $0.0105 |
| openrouter/qwen/qwen-2.5-7b-instruct | C | 30 | 100.0% [88.6-100.0] | 100.0% [88.6-100.0] | 1.00 | 66.5% [59.6-73.8] | 100.0% | 66.5% [59.6-73.8] | $0.0076 |
| openrouter/qwen/qwen-2.5-7b-instruct | C | 30 | 96.7% [83.3-99.4] | 96.7% [83.3-99.4] | 1.00 | 68.4% [61.3-75.0] | 95.0% | 65.0% [55.2-73.3] | $0.0076 |
| openrouter/qwen/qwen3-235b-a22b-2507 | B | 30 | 76.7% [59.1-88.2] | 86.7% [70.3-94.7] | 1.37 | 85.7% [81.3-89.6] | 85.0% | 72.8% [58.4-84.8] | $0.0234 |
| openrouter/qwen/qwen3-235b-a22b-2507 | B | 30 | 73.3% [55.6-85.8] | 93.3% [78.7-98.2] | 1.33 | 85.1% [81.1-88.9] | 95.0% | 80.9% [71.1-87.9] | $0.0199 |
| openrouter/qwen/qwen3-235b-a22b-2507 | B | 30 | 80.0% [62.7-90.5] | 86.7% [70.3-94.7] | 1.33 | 86.0% [82.0-89.6] | 85.0% | 73.1% [58.7-84.9] | $0.0202 |
| openrouter/qwen/qwen3-235b-a22b-2507 | C | 30 | 96.7% [83.3-99.4] | 96.7% [83.3-99.4] | 1.00 | 87.0% [83.7-89.9] | 95.0% | 82.8% [72.3-89.0] | $0.0136 |
| openrouter/qwen/qwen3-235b-a22b-2507 | C | 30 | 100.0% [88.6-100.0] | 100.0% [88.6-100.0] | 1.00 | 85.6% [82.0-88.8] | 100.0% | 85.6% [82.0-88.8] | $0.0134 |

The A and D figures in the main table are rows 1-8 (A and D) above. The first-round B rows (7B: 86.7% / 96.7%,
235B: 83.3% / 86.7% first-try / final) are the runs from before the retry fix; the first-round C rows are the first
of the three C runs in the main table. The aggregate over the repeats is in `results/aggregate.md`; the raw per-run
reports are `results/2026-11-04_*.json`.

</details>

<!-- GIF: soa extract demo -->

## How it works

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

The schema shown to the model is generated from the Pydantic model (`JobPosting.model_json_schema()`), never written
by hand, and the reply is checked with the same model. A failed reply goes back to the model with the validator's
errors, one line per error, for up to 3 attempts. After a valid reply, two deterministic tools tidy the result
(canonical skill names, a location check against the document header). They are plain code, not model calls, so they
add no LLM calls and the setups below stay comparable. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The four setups: **A** plain prompt, no retry. **B** plain prompt, errors fed back (up to 3 attempts). **C** the
provider's strict JSON-schema mode, no retry. **D** schema mode plus retry.

## Why not just use native structured outputs?

Schema mode helped a lot, and it is cheaper. It also does not do the whole job.

**What it fixed.** Setup C was valid on the first try 98.9% (96.7%-100.0%) of the time for both models, against a
single plain-prompt run (A) of 83.3% for the 7B and 70.0% for the 235B. The 30-item cap on `skills` was the constraint
that models actually broke in plain mode; under schema mode the report counts replies that broke a constraint the
schema carries, and it was **0 in every C and D run** (6 C runs, 2 D runs). C also cost less than B: $0.0076 against
$0.0103 for the 7B, $0.0136 against $0.0211 for the 235B (means per run).

**What it does not do:**

1. **It guarantees shape, not meaning.** The remaining failures in C were rules no JSON schema can express. Two of the
   six C runs had one document rejected for duplicate skill names ("LangChain" and "Langchain"; "Python" twice; both rejections are in
   [results/failure_examples.jsonl](results/failure_examples.jsonl)), which is why C's final validity is 98.9%
   (96.7%-100.0%) and not 100%. The intern-implies-no-experience rule has the same property (it cannot be stated in a
   schema); it did not fire in any of the 637 logged attempt lines of this project (the last line of that file
   records the count), so I have no evidence either way on it.
2. **Valid is not correct.** Schema mode made the 235B valid 98.9% (96.7%-100.0%) of the time, and its field accuracy
   is 86.1% (85.6%-87.0%), with end-to-end accuracy 84.7% (82.8%-85.8%): below my 90% goal. The 7B in C is just as
   valid and gets 68.2% (66.5%-69.8%) field accuracy; its seniority accuracy was 55.0%, 55.0% and 52.6% over the three
   C runs. Validity says nothing about whether the seniority is right.
3. **It may trade recall for validity.** The cap is enforced by the model emitting shorter skill lists, and skills
   are where accuracy is weakest. For the 235B, skills F1 was 66.9%, 72.1% and 65.7% in the three C runs, against
   71.4%, 71.6% and 68.3% in the three B runs (means 68.2% vs 70.4%, with overlapping ranges). In the first-round
   single runs it was 72.1% (A) and 73.8% (B) against 66.9% (C) and 63.4% (D). The repeats show this is a tendency,
   not a clean effect, and with n=20 labelled documents I would not claim more.
4. **It depends on the provider.** Both models accepted strict `json_schema` here, but only because requests were sent
   with OpenRouter's `require_parameters` so they could only route to providers that support it. The 7B was always
   served by one provider; the 235B was split across three.

## Most common failure types, and what fixed them

From the plain-prompt runs and the retry runs (counts are failing attempts whose first error was at that field,
`failures.by_first_loc` in the reports):

| failure | setup A (one run) | setup B, 3 runs | what fixed it |
|---|---|---|---|
| `skills` over the 30-item cap | 7B: 3 of 5 failing docs; 235B: 8 of 9 failing docs | 7B: 5, 5, 3 failing attempts; 235B: 14, 11, 14 | Retry feedback fixes most of it. Schema mode removes it. |
| skill name over 40 characters (`skills.*.name`) | 7B 2, 235B 1 | 7B: 0, 0, 3; 235B: 1, 1, 0 | Retry |
| `null` instead of true/false (`skills.*.required`, `salary_mentioned`) | none | 7B: 2 attempts at `skills.*.required` in one run, 1 at `salary_mentioned` in another | Retry |
| duplicate skill names (also counted under `skills`, since that is where the error is reported) | not counted separately | not counted separately | Retry fixes it: all 5 duplicate-name rejections in setup B were valid on the next attempt (each rejected attempt and its valid successor are in `results/failure_examples.jsonl`). **Schema mode cannot**: it was the only failure left in C (two documents, same file). |
| malformed JSON, markdown fences, prose, extra fields, invalid enum values | none: no run has a failure at the root, at `seniority`, `mode` or any other enum or extra field | none | Nothing needed |

Almost everything that goes wrong is a count cap, not a formatting problem. In the first 11-document baseline every
failure of both models was a too-long `skills` list (`results/2026-11-03_baseline_*.md`). The first thing I tried was
to make the feedback more specific. The default feedback line is the validator's own wording (for example `skills:
List should have at most 30 items after validation, not 42`). I replaced it for over-long lists with a count and a
fix, for example `skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones`. The result is
mixed:

- It did **not** lift the 235B's final validity in setup B: 86.7% before, then 86.7%, 93.3% and 86.7% across three
  runs. Each run's interval contains the old figure, so at n=30 no effect is shown.
- The four 235B documents that failed in the first B run (jd_002, jd_003, jd_005, jd_023) passed in 6 of 12
  document-runs afterwards (jd_002 0 of 3, jd_003 1 of 3, jd_005 2 of 3, jd_023 3 of 3), but other documents
  (jd_029, jd_030, jd_004) failed instead. I cannot separate the fix from run-to-run variation. The first attempt
  is identical before and after the fix, yet first-try validity moved from 83.3% to 76.7%, 73.3% and 80.0%; that
  spread is run-to-run variation (sampling plus a provider mix I did not log in the first round).
- **jd_002 failed for the 235B in all four setup-B runs.** Every attempt is in
  [results/failure_examples.jsonl](results/failure_examples.jsonl): 42, 31, 31 skills in the first run (before the
  fix), then 70, 31, 31 and 47, 31, 31 and 44, 31, 31 in the three runs after it. The model goes from 42-70 skills down
  to 31 on the first retry and then stays at 31. For a 31-item reply the new feedback line is `skills: has 31 items,
  maximum is 30 - remove 1, starting with required=false ones` (built by the code; the prompt text itself is not
  logged). This is the clearest open failure: the model does not make the last cut.

## Design decisions

- **`strict=True` on every model.** Pydantic's default lax mode quietly turns `"3"` into `3` and `"yes"` into
  `True`, so a sloppy reply passed and the retry loop had nothing to do. Strict mode rejects both.
- **Required-but-nullable fields** (`company`, `city`, `country`, `min_years_experience`). No default: the model has
  to write `null` explicitly, so "not stated" is a decision the model made, and a forgotten key is an error rather
  than a silent `null`.
- **One line per error in the retry prompt**, `loc: msg (got: <input, clipped to 60 chars>)`, rendered from
  `ValidationError.errors()` instead of `str(error)`. The raw string carries a version-specific prefix, a truncated
  `input_value`, and a documentation URL, which are noise for a model. The one exception is an over-long list, which
  gets the count-aware line shown above. I did not run a blind-retry control (retry with no error text), so I have no
  measurement of feedback versus plain resampling; setup A against B is retry-with-feedback against nothing.
- **Tools are deterministic post-processing, not model-called.** Canonical skill names and a location check run
  after validation as plain code. That keeps every setup at one prompt and the retry/attempt/cost counts clean, and
  the model's own output is kept (`raw_posting`) so validity is measured on what the model actually said. Letting the
  model call the tools is a separate experiment I have not run.
- **API errors are separate from validation errors.** A 429 or 5xx is retried with backoff (2, 4, 8 s) and never uses
  up one of the 3 validation attempts.
- **The 30-skill cap stays.** The first plain-prompt baseline showed every failure was this one constraint, so I kept
  it as the test of what retries and schema modes can do rather than loosening it to make numbers look better.

## Limitations

- **30 job descriptions, 20 gold labels.** One document moves a validity rate by 3.3 points and a field accuracy
  by 5 points; the intervals above are wide for that reason (for example 235B setup B final validity 93.3%
  [78.7-98.2] in one run). Five of the 30 are non-technical or semi-technical roles, kept on purpose as an
  out-of-domain slice; two of them (jd_012, jd_019) are among the 20 labelled.
- **The gold labels were drafted by a model (Claude) and then checked by me.** I verified every field and made 17
  corrections ([docs/label_corrections.log](docs/label_corrections.log)). One person, one pass, so any systematic
  mistake of mine is in the labels. The labelling rules were written into the schema's field descriptions after the
  first baseline, so the models and the labels follow the same rules; runs on the earlier prompt are not comparable.
- **Noise floor on accuracy, from a blind check.** On 5 other documents (jd_004, 005, 008, 014, 015, none of them in
  the gold set) I labelled blind and compared against Claude's sealed labels written beforehand, with the same
  per-field metrics as the eval (my labels as the reference): title 80.0%, company 100.0%, seniority 80.0%, mode
  80.0%, salary_mentioned 100.0%, min_years_experience 100.0%, city 80.0%, country 100.0%, skills F1 61.0% (precision
  52.9%, recall 72.0%), overall 86.8% ([results/label_agreement.md](results/label_agreement.md)). Five documents is
  a sanity check, not a rate: one disagreement moves a field by 20 points. It does suggest that on skills two careful
  labelers disagree about as much as the models disagree with the gold labels, and that the 235B's roughly 86%
  field accuracy is close to the level two labelers reach with each other, so the remaining gap to my 90% target may
  partly be label noise rather than model error. I have not shown that.
- **Runs are from one day, two models of one family, one provider layer.** Temperature was not set. The 7B was always
  routed to one provider; the 235B to three, and the provider was not logged in the first round, so provider mix and
  sampling noise are mixed into the run-to-run spread.
- **The post-processing tools were not tuned against the final labelling rules**, so the location check and the labels
  can disagree on some documents.
- Setup D is a single run per model; only B and C were repeated.
- **Data.** The 30 job descriptions in `data/jds/` were copied from public postings (LinkedIn 13, Work at a Startup 11,
  Wellfound 6, counted from `data/jds/manifest.csv`) for evaluation only; I will remove any of them on request. Five
  rows have no source URL recorded in `manifest.csv`: jd_001, jd_002, jd_003, jd_004 and jd_010 (all LinkedIn; their
  notes say "URL not recorded").

## Run it yourself

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/nilayp-codes/typecast.git
cd typecast
uv sync
cp .env.example .env        # then set OPENROUTER_API_KEY in .env (never commit it)

uv run soa extract data/jds/jd_001.txt                 # the JSON on stdout, a status line on stderr
uv run soa extract data/jds/jd_001.txt --json-mode     # provider schema mode
uv run soa extract data/jds/jd_001.txt > out.json

uv run soa eval --model openrouter/qwen/qwen-2.5-7b-instruct --setup B --limit 5
uv run soa eval --model openrouter/qwen/qwen3-235b-a22b-2507 --setup all
```

Known issue: on Windows, clone into a short path, because LiteLLM writes a cache file under `.venv` and a very deep path exceeds Windows' path-length limit (`soa extract` then fails with `APIConnectionError ... [Errno 2]`).

`soa eval` makes real model calls: it prints the plan and the estimated number of calls and asks before sending
anything (`--yes` skips the question). In my runs one setup over the 30 documents cost between $0.0076 and $0.0234.
It saves `results/<date>_<model>_<setup>.json` and appends a row to `results/summary.md`; `--resume <eval id>`
continues an interrupted run. To combine repeated runs: `uv run python scripts/aggregate_runs.py results/<files>.json`
(writes `results/aggregate.md`). The 30 job descriptions are in `data/jds/` and the gold labels in `data/labels/`.

Tests and type checks (no real model is ever called in tests):

```
uv run pytest
uv run mypy --strict src/ packages/typecast-mcp/src/
```

## Use it from an AI agent (MCP server)

`typecast-mcp` is a small [MCP](https://modelcontextprotocol.io) server that checks a job-posting JSON against the
`JobPosting` schema and returns one-line errors an agent can act on. It has no LLM, needs no API key and makes no
network calls. It is its own package ([`packages/typecast-mcp/`](packages/typecast-mcp/)) with two dependencies,
`pydantic` and `mcp`, so installing it does not pull in LiteLLM or the rest of the eval harness.

Two tools:

- `get_job_posting_schema`: the JSON Schema of a job posting. It does not carry every rule (duplicate skill names and
  the intern/experience rule are only checked by the validator).
- `validate_job_posting`: takes one argument, `posting`, the whole job-posting JSON as ONE string. Returns `valid`, a
  list of error lines (`field: problem (got: value)`), and the parsed posting when valid.

**Install** (needs [uv](https://docs.astral.sh/uv/)). Claude Desktop, Cursor and other clients:

```json
{
  "mcpServers": {
    "typecast": {
      "command": "uvx",
      "args": ["typecast-mcp"]
    }
  }
}
```

Claude Code: `claude mcp add typecast -- uvx typecast-mcp`. On Windows, if the client cannot find `uvx`, put the full
path to `uvx.exe` in `command`. Without uv: `pip install typecast-mcp` and use `typecast-mcp` as the command.

**From a clone.** The repo ships a `.mcp.json` that runs the server from the working tree (`uv run typecast-mcp`).
Open the folder in Claude Code, approve `typecast` when asked, and check it with `/mcp`. Approving runs `uv run` in
this folder, which installs the locked dependencies and starts the server. Only approve it in a clone you trust.

**A real example.** Qwen3-235B's reply on jd_002 had 70 skills. An agent checked it with `validate_job_posting` and got
one line back:

```
skills: has 70 items, maximum is 30 - remove 40, starting with required=false ones
```

The agent fixed the posting in one round. This was one run with Claude Sonnet 5.5, not a benchmark.

**What it does not check:** whether the values are correct. A posting can be valid and still have the wrong seniority
or the wrong skills; see [Why not just use native structured outputs?](#why-not-just-use-native-structured-outputs).

Details, limits and design decisions: [docs/MCP.md](docs/MCP.md).

## What I'd do next

1. ~~A validator-only MCP server~~: done, see [Use it from an AI agent](#use-it-from-an-ai-agent-mcp-server).
2. ~~Publish on PyPI~~: done for the MCP server, [`typecast-mcp`](https://pypi.org/project/typecast-mcp/).
3. Fix the remaining failure the data points at: the model that cannot make the last cut on jd_002, for example by
   telling it which items to drop. Re-measure with temperature fixed and the provider pinned, so repeats differ less.
4. A second labeler for the gold set, a bigger document set, and repeats of setup D, so the field-accuracy
   numbers stop being limited by n=20.

## License

MIT. See [LICENSE](LICENSE).

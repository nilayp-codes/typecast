"""Eval harness: run one setup over a folder of job descriptions and score it. Metric definitions: ARCHITECTURE.md.

`run_eval` makes the extract() calls and returns an `EvalReport`; `build_report` is the pure scoring step (log lines
and postings in, report out), so it is tested without any model. The plan helpers at the bottom belong to the
`soa eval` stub and go away when the CLI is wired to `run_eval`.
"""

import itertools
import logging
import re
import time
import uuid
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from soa.agent import DEFAULT_MAX_ATTEMPTS, ExtractionFailed, extract
from soa.errors import redact_keys
from soa.llm import LLMAPIError
from soa.logger import API_ERROR_TYPE, AttemptLog, Setup, default_log_path, read_log
from soa.schemas import JobPosting
from soa.tools import postprocess

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/jds")
UNSUPPORTED_STOP_AFTER = 2  # consecutive "schema mode unsupported" docs after which a C/D run stops
DEFAULT_PAUSE_SECONDS = 2.0  # between documents; the baseline hit 429s after about 25 unpaced calls
LABEL_PLACEHOLDER = "TODO"  # scripts/make_label_templates.py; a label still containing it is not gold yet


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


# --- setups -----------------------------------------------------------------


def setup_args(setup: Setup, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> tuple[int, bool]:
    """(max_attempts, json_mode) for a setup: A plain/no retry, B plain/retry, C schema/no retry, D schema/retry."""
    retry = setup in ("B", "D")
    return (max_attempts if retry else 1), setup in ("C", "D")


# --- report models ----------------------------------------------------------


class DocRun(StrictModel):
    """What extract() gave for one document. `raw_posting` is the model's output, `posting` the post-processed one."""

    doc_id: str
    run_id: str
    # invalid = ExtractionFailed; api_error = no model response; unsupported = the provider cannot do schema mode
    status: Literal["valid", "invalid", "api_error", "unsupported"]
    raw_posting: JobPosting | None = None
    posting: JobPosting | None = None
    disagreements: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def postings_match_status(self) -> Self:
        has_output = self.raw_posting is not None and self.posting is not None
        if (self.status == "valid") != has_output:
            raise ValueError("a valid DocRun carries both postings; any other status carries none")
        return self


class Labels(StrictModel):
    """Gold labels by doc id, plus the ids of label files that were skipped (invalid, unreadable or still TODO)."""

    by_doc: dict[str, JobPosting]
    skipped: list[str]


class FieldAccuracy(StrictModel):
    """Agreement with the gold labels, in percent (0-100), over labelled docs that produced a valid posting.

    title / company: case-insensitive exact match. seniority, mode, salary_mentioned, min_years_experience, city and
    country: exact match. skills: precision / recall / F1 on casefolded names, micro-averaged over docs.
    `overall` is the plain mean of the nine numbers: the eight fields above and skills_f1.
    """

    n: int = Field(ge=1)
    title: float
    company: float
    seniority: float
    mode: float
    salary_mentioned: float
    min_years_experience: float
    city: float
    country: float
    skills_precision: float
    skills_recall: float
    skills_f1: float
    overall: float


class FailureBreakdown(StrictModel):
    """Rejected attempts of the run (API errors excluded), most frequent first."""

    by_error_type: dict[str, int]
    by_first_loc: dict[str, int]  # location of the first error line, numeric path parts as `*` (skills.*.name)


class EvalReport(StrictModel):
    """One setup on one model over one set of documents. Rates are percentages (0-100), None when undefined.

    n = documents run; n_scored = those that got at least one model response; n_api_failed = n - n_scored (no
    response, so they say nothing about validity and are left out of the validity numbers). Validity numbers and
    avg_calls are over n_scored. total_cost_usd is None ("n/a") if any model response had an unknown cost.
    """

    model: str
    setup: Setup
    json_mode: bool
    max_attempts: int
    run_date: str
    eval_id: str  # the `soa eval` invocation (in every run_id); `--resume` and write_results use it to find the run
    n: int
    n_scored: int
    n_api_failed: int
    n_unsupported: int  # docs whose schema-mode request was rejected as unsupported (a subset of n_api_failed)
    # setup C/D and EVERY doc was rejected as unsupported, or the run stopped after UNSUPPORTED_STOP_AFTER in a row
    # (then n counts only the docs run): the report carries no numbers
    not_supported: bool
    first_try_valid_pct: float | None
    final_valid_pct: float | None
    avg_calls: float | None
    total_cost_usd: float | None
    # where total_cost_usd comes from: "provider" = every response carried the provider's own cost; "estimate" = at
    # least one was computed from list prices (or logged before sources were recorded); None = n/a
    cost_source: Literal["provider", "estimate"] | None
    avg_latency_ms: float | None
    api_error_count: int
    disagreement_count: int
    n_labelled: int  # usable labelled docs in this run
    n_labelled_with_output: int
    coverage_pct: float | None  # n_labelled_with_output / n_labelled
    labels_skipped: list[str]
    accuracy_raw: FieldAccuracy | None  # accuracy on valid outputs only
    accuracy_normalized: FieldAccuracy | None
    end_to_end_raw: FieldAccuracy | None  # over all usable labelled docs; no valid output scores 0 on every field
    end_to_end_normalized: FieldAccuracy | None
    failures: FailureBreakdown
    constraint_violations: ConstraintViolations | None  # setups C and D only (schema mode requested)
    docs: list[DocRun]


# --- validity, cost, failures (from the log) --------------------------------


class Validity(StrictModel):
    n_scored: int
    first_try: int
    final: int
    calls: int
    cost_usd: float | None
    cost_source: Literal["provider", "estimate"] | None
    latency_ms: int
    api_errors: int


def _lines_by_run(docs: list[DocRun], logs: list[AttemptLog]) -> dict[str, list[AttemptLog]]:
    grouped: dict[str, list[AttemptLog]] = {d.run_id: [] for d in docs}
    for line in logs:
        if line.run_id in grouped:
            grouped[line.run_id].append(line)
    return grouped


def _responses(lines: list[AttemptLog]) -> list[AttemptLog]:
    """The model responses among a run's log lines (APIError lines are not), in attempt order."""
    return sorted((x for x in lines if x.error_type != API_ERROR_TYPE), key=lambda x: x.attempt)


def validity_stats(docs: list[DocRun], logs: list[AttemptLog]) -> Validity:
    """Validity, calls, cost and latency from the log lines of these docs' runs. APIError lines only count as such."""
    scored = first_try = final = calls = latency = api_errors = 0
    cost, cost_known, all_provider = 0.0, True, True
    for lines in _lines_by_run(docs, logs).values():
        api_errors += sum(1 for x in lines if x.error_type == API_ERROR_TYPE)
        responses = _responses(lines)
        if not responses:
            continue
        all_provider = all_provider and all(x.cost_source == "provider" for x in responses)
        scored += 1
        calls += len(responses)
        first_try += responses[0].attempt == 1 and responses[0].ok
        final += any(x.ok for x in responses)
        cost += sum(x.cost_usd for x in responses)
        cost_known = cost_known and all(x.cost_known for x in responses)
        latency += sum(x.latency_ms for x in responses)
    source: Literal["provider", "estimate"] | None = None
    if cost_known and scored:
        source = "provider" if all_provider else "estimate"
    return Validity(
        n_scored=scored, first_try=first_try, final=final, calls=calls,
        cost_usd=cost if cost_known else None, cost_source=source, latency_ms=latency, api_errors=api_errors,
    )


def first_error_loc(error: str) -> str:
    """`skills.3.name: String should...` -> `skills.*.name`. The first line only; `(unknown)` if there is no loc."""
    first = error.splitlines()[0] if error else ""
    loc, sep, _ = first.partition(": ")
    if not sep:
        return "(unknown)"
    return ".".join("*" if part.isdigit() else part for part in loc.split("."))


def _most_frequent_first(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def failure_breakdown(docs: list[DocRun], logs: list[AttemptLog]) -> FailureBreakdown:
    by_type: Counter[str] = Counter()
    by_loc: Counter[str] = Counter()
    for lines in _lines_by_run(docs, logs).values():
        for line in _responses(lines):
            if not line.ok:
                by_type[line.error_type or "(none)"] += 1
                by_loc[first_error_loc(line.error or "")] += 1
    return FailureBreakdown(by_error_type=_most_frequent_first(by_type), by_first_loc=_most_frequent_first(by_loc))


# Constraints a JSON Schema can express, which the provider's schema mode was asked to enforce. Matched on the start
# of the validator's message (the same pydantic wording for every field); the names are the breakdown keys.
SCHEMA_CONSTRAINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("string_length", ("String should have at least", "String should have at most")),
    ("list_size", ("List should have at least", "List should have at most")),
    ("enum", ("Input should be '",)),
    ("type", ("Input should be a valid",)),
    ("number_bound", ("Input should be greater than", "Input should be less than")),
    ("required_field", ("Field required",)),
    ("extra_field", ("Extra inputs are not permitted",)),
)


def schema_constraint(error_line: str) -> str | None:
    """Which schema constraint a `loc: msg (got: ...)` line violates, or None.

    None for rules no schema carries (the duplicate-skills and intern rules: `Value error, ...`) and for
    `Invalid JSON`: those are not something the provider's schema mode could have enforced.
    """
    _, sep, rest = error_line.partition(": ")
    if not sep:
        return None
    for name, prefixes in SCHEMA_CONSTRAINTS:
        if rest.startswith(prefixes):
            return name
    return None


class ConstraintViolations(StrictModel):
    """Replies that came back in schema mode and still broke a constraint the schema carries (setups C and D).

    n_outputs = rejected model replies with at least one such violation (a retried doc can contribute several);
    by_constraint = replies containing at least one violation of that kind (so it can sum to more than n_outputs).
    A non-zero count means the provider dropped or does not enforce that constraint.
    """

    n_outputs: int
    by_constraint: dict[str, int]


def constraint_violations(docs: list[DocRun], logs: list[AttemptLog]) -> ConstraintViolations:
    n_outputs = 0
    by_constraint: Counter[str] = Counter()
    for lines in _lines_by_run(docs, logs).values():
        for line in (x for x in _responses(lines) if not x.ok):
            kinds = {kind for e in (line.error or "").splitlines() if (kind := schema_constraint(e))}
            n_outputs += bool(kinds)
            by_constraint.update(kinds)
    return ConstraintViolations(n_outputs=n_outputs, by_constraint=_most_frequent_first(by_constraint))


def _pct(part: int, whole: int) -> float | None:
    return 100 * part / whole if whole else None


# --- field accuracy ---------------------------------------------------------


def _same_text(gold: str | None, predicted: str | None) -> bool:
    if gold is None or predicted is None:
        return gold is predicted
    return gold.casefold() == predicted.casefold()


FIELD_MATCHERS: dict[str, Callable[[JobPosting, JobPosting], bool]] = {
    "title": lambda g, p: _same_text(g.title, p.title),
    "company": lambda g, p: _same_text(g.company, p.company),
    "seniority": lambda g, p: g.seniority == p.seniority,
    "mode": lambda g, p: g.location.mode == p.location.mode,
    "salary_mentioned": lambda g, p: g.salary_mentioned == p.salary_mentioned,
    "min_years_experience": lambda g, p: g.min_years_experience == p.min_years_experience,
    "city": lambda g, p: g.location.city == p.location.city,
    "country": lambda g, p: g.location.country == p.location.country,
}


def _skill_names(posting: JobPosting) -> set[str]:
    return {skill.name.strip().casefold() for skill in posting.skills}


Pair = tuple[JobPosting, JobPosting | None]  # (gold, predicted); predicted None = no valid output for that doc


def _skill_scores(pairs: Sequence[Pair]) -> tuple[float, float, float]:
    """(precision, recall, F1) in percent, micro-averaged. A gold posting has >= 1 skill, so recall and F1 are defined.

    A missing prediction adds its gold skills as misses and no predicted skills; precision is 0.0 if nothing at all
    was predicted.
    """
    tp = fp = fn = 0
    for gold, predicted in pairs:
        g = _skill_names(gold)
        p = set() if predicted is None else _skill_names(predicted)
        tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0
    return precision, 100 * tp / (tp + fn), 100 * 2 * tp / (2 * tp + fp + fn)


def score_fields(pairs: Sequence[Pair]) -> FieldAccuracy:
    """Score (gold, predicted) pairs; a pair with no prediction scores 0 on every field. Needs at least one pair."""
    rates = {
        name: 100 * sum(p is not None and match(g, p) for g, p in pairs) / len(pairs)
        for name, match in FIELD_MATCHERS.items()
    }
    precision, recall, f1 = _skill_scores(pairs)
    overall = (sum(rates.values()) + f1) / (len(rates) + 1)
    return FieldAccuracy(
        n=len(pairs), **rates, skills_precision=precision, skills_recall=recall, skills_f1=f1, overall=overall
    )


def load_labels(labels_dir: Path) -> Labels:
    """Read <labels_dir>/*.json as JobPostings. Invalid, unreadable or still-TODO files are skipped and listed."""
    by_doc: dict[str, JobPosting] = {}
    skipped: list[str] = []
    paths = sorted(labels_dir.glob("*.json")) if labels_dir.is_dir() else []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8-sig")
            if LABEL_PLACEHOLDER in text:
                raise ValueError(f"still contains {LABEL_PLACEHOLDER}")
            by_doc[path.stem] = JobPosting.model_validate_json(text)
        except (OSError, UnicodeDecodeError, ValueError) as exc:  # pydantic's ValidationError is a ValueError
            logger.warning("label %s skipped: %s", path.name, str(exc).splitlines()[0])
            skipped.append(path.stem)
    return Labels(by_doc=by_doc, skipped=skipped)


class Accuracy(StrictModel):
    """Everything the labels say about a run. None where there is nothing to score."""

    n_labelled: int
    n_with_output: int
    raw: FieldAccuracy | None
    normalized: FieldAccuracy | None
    end_to_end_raw: FieldAccuracy | None
    end_to_end_normalized: FieldAccuracy | None


def _score(pairs: Sequence[Pair]) -> FieldAccuracy | None:
    return score_fields(pairs) if pairs else None


def _accuracy(docs: list[DocRun], labels: Labels) -> Accuracy:
    """Accuracy over valid outputs only, and end to end over every labelled doc (no valid output scores 0)."""
    labelled = [d for d in docs if d.doc_id in labels.by_doc]
    with_output = [d for d in labelled if d.raw_posting is not None]
    return Accuracy(
        n_labelled=len(labelled),
        n_with_output=len(with_output),
        raw=_score([(labels.by_doc[d.doc_id], d.raw_posting) for d in with_output]),
        normalized=_score([(labels.by_doc[d.doc_id], d.posting) for d in with_output]),
        end_to_end_raw=_score([(labels.by_doc[d.doc_id], d.raw_posting) for d in labelled]),
        end_to_end_normalized=_score([(labels.by_doc[d.doc_id], d.posting) for d in labelled]),
    )


def _no_accuracy(docs: list[DocRun], labels: Labels) -> Accuracy:
    """A not-supported run gets no accuracy numbers, not even zeros (the labelled count is still shown)."""
    n_labelled = sum(d.doc_id in labels.by_doc for d in docs)
    return Accuracy(
        n_labelled=n_labelled, n_with_output=0, raw=None, normalized=None, end_to_end_raw=None,
        end_to_end_normalized=None,
    )


# --- the report -------------------------------------------------------------


def build_report(
    *, model: str, setup: Setup, max_attempts: int, docs: list[DocRun], logs: list[AttemptLog], labels: Labels,
    run_date: str, eval_id: str = "adhoc", stopped_unsupported: bool = False,
) -> EvalReport:
    """Score a run. `logs` may hold other runs' lines too; only the docs' run_ids are read."""
    v = validity_stats(docs, logs)
    n_unsupported = sum(d.status == "unsupported" for d in docs)
    json_mode = setup_args(setup)[1]
    not_supported = json_mode and (stopped_unsupported or n_unsupported == len(docs) > 0)
    acc = _no_accuracy(docs, labels) if not_supported else _accuracy(docs, labels)
    doc_ids = {d.doc_id for d in docs}
    return EvalReport(
        model=model, setup=setup, json_mode=json_mode, max_attempts=max_attempts, run_date=run_date, eval_id=eval_id,
        n=len(docs), n_scored=v.n_scored, n_api_failed=len(docs) - v.n_scored,
        n_unsupported=n_unsupported, not_supported=not_supported,
        first_try_valid_pct=_pct(v.first_try, v.n_scored), final_valid_pct=_pct(v.final, v.n_scored),
        avg_calls=v.calls / v.n_scored if v.n_scored else None,
        total_cost_usd=v.cost_usd, cost_source=v.cost_source,
        avg_latency_ms=v.latency_ms / v.n_scored if v.n_scored else None,
        api_error_count=v.api_errors, disagreement_count=sum(d.disagreements for d in docs),
        n_labelled=acc.n_labelled, n_labelled_with_output=acc.n_with_output,
        coverage_pct=None if not_supported else _pct(acc.n_with_output, acc.n_labelled),
        labels_skipped=[s for s in labels.skipped if s in doc_ids],
        accuracy_raw=acc.raw, accuracy_normalized=acc.normalized,
        end_to_end_raw=acc.end_to_end_raw, end_to_end_normalized=acc.end_to_end_normalized,
        failures=failure_breakdown(docs, logs),
        constraint_violations=constraint_violations(docs, logs) if json_mode and not not_supported else None,
        docs=docs,
    )


# --- running ----------------------------------------------------------------


def list_jds(data_dir: Path, limit: int | None = None) -> list[Path]:
    """The .txt job descriptions in name order, at most `limit`. FileNotFoundError if there are none."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {data_dir}")
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"no .txt job descriptions in {data_dir}")
    return paths[:limit] if limit else paths


def new_eval_id() -> str:
    """A short id for one `soa eval` invocation; it is in every run_id, so `--resume` can find the run in the log."""
    return uuid.uuid4().hex[:6]


def base_run_id(setup: Setup, eval_id: str, doc_id: str) -> str:
    return f"eval-{setup}-{eval_id}-{doc_id}"


def _revision(run_id: str, base: str) -> int | None:
    """1 for `base`, N for `base-rN` (a doc re-run on resume), None for any other run_id."""
    match = re.fullmatch(re.escape(base) + r"(?:-r(\d+))?", run_id)
    if match is None:
        return None
    return int(match.group(1)) if match.group(1) else 1


def _revisions(logs: list[AttemptLog], base: str) -> dict[int, list[AttemptLog]]:
    found: dict[int, list[AttemptLog]] = {}
    for line in logs:
        revision = _revision(line.run_id, base)
        if revision is not None:
            found.setdefault(revision, []).append(line)
    return found


def next_run_id(logs: list[AttemptLog], base: str) -> str:
    """`base` for a doc not run yet, else `base-r<N+1>`: a re-run never appends to an earlier run's log lines."""
    revisions = _revisions(logs, base)
    return base if not revisions else f"{base}-r{max(revisions) + 1}"


def finished_doc(path: Path, setup: Setup, max_attempts: int, eval_id: str, logs: list[AttemptLog]) -> DocRun | None:
    """The DocRun of a doc whose latest logged run already ended in a valid posting or in used-up attempts.

    Rebuilt from the log (the model's reply, then the deterministic post-processing again), so no model call is made.
    None for a doc with no run, or whose latest run was cut short or only had API errors: resume runs those again.
    """
    base = base_run_id(setup, eval_id, path.stem)
    revisions = _revisions(logs, base)
    if not revisions:
        return None
    latest = max(revisions)
    run_id = base if latest == 1 else f"{base}-r{latest}"
    responses = _responses(revisions[latest])
    valid = next((x for x in responses if x.ok), None)
    if valid is None:
        if len(responses) < max_attempts:
            return None
        return DocRun(doc_id=path.stem, run_id=run_id, status="invalid")
    raw = JobPosting.model_validate_json(valid.raw_output)
    processed = postprocess(raw, path.read_text(encoding="utf-8-sig"), doc_id=path.stem)
    return DocRun(
        doc_id=path.stem, run_id=run_id, status="valid", raw_posting=raw, posting=processed.posting,
        disagreements=len(processed.disagreements),
    )


def run_one(
    path: Path, model: str, setup: Setup, max_attempts: int, json_mode: bool, run_id: str, log_path: Path
) -> DocRun:
    """extract() on one document under `run_id`. ExtractionFailed and API errors become a DocRun, not a raise."""
    doc_id = path.stem
    text = path.read_text(encoding="utf-8-sig")
    try:
        result = extract(
            text, model, max_attempts, json_mode, doc_id=doc_id, run_id=run_id, log_path=log_path, setup=setup
        )
    except ExtractionFailed:
        return DocRun(doc_id=doc_id, run_id=run_id, status="invalid")
    except LLMAPIError as exc:
        logger.error("%s: no model response (%s)", doc_id, redact_keys(" ".join(str(exc).split())))
        return DocRun(doc_id=doc_id, run_id=run_id, status="unsupported" if exc.schema_unsupported else "api_error")
    return DocRun(
        doc_id=doc_id, run_id=run_id, status="valid", raw_posting=result.raw_posting, posting=result.posting,
        disagreements=len(result.disagreements),
    )


def _run_docs(
    paths: list[Path], model: str, setup: Setup, attempts: int, json_mode: bool, eval_id: str, log: Path,
    pause: float, resume: bool,
) -> tuple[list[DocRun], bool]:
    """(DocRuns, stopped): finished docs rebuilt from the log on resume, the rest run with `pause` between them.

    A schema-mode run stops (`stopped` True, the remaining docs are not run) after UNSUPPORTED_STOP_AFTER docs in a
    row were rejected as unsupported: the provider will not change its answer, so more calls only cost time.
    """
    earlier = read_log(log) if resume else []
    docs: list[DocRun] = []
    called = False  # the first model call of this invocation is not preceded by a pause
    for path in paths:
        if json_mode and _trailing_unsupported(docs) >= UNSUPPORTED_STOP_AFTER:
            return docs, True
        done = finished_doc(path, setup, attempts, eval_id, earlier) if resume else None
        if done is None:
            if called and pause > 0:
                time.sleep(pause)
            called = True
            run_id = next_run_id(earlier, base_run_id(setup, eval_id, path.stem))
            done = run_one(path, model, setup, attempts, json_mode, run_id, log)
        docs.append(done)
    return docs, False


def _trailing_unsupported(docs: list[DocRun]) -> int:
    count = 0
    for doc in reversed(docs):
        if doc.status != "unsupported":
            break
        count += 1
    return count


def run_eval(
    model: str,
    setup: Setup,
    data_dir: Path,
    labels_dir: Path,
    limit: int | None = None,
    *,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    log_path: Path | None = None,
    run_date: str | None = None,
    pause: float = 0.0,
    eval_id: str | None = None,
    resume: bool = False,
) -> EvalReport:
    """Run `setup` on every document (up to `limit`) and score it. REAL model calls unless llm.call is mocked.

    Each document gets its own run_id (`eval-<setup>-<eval_id>-<doc>`); ExtractionFailed is counted as an invalid
    document and the run continues. A document whose calls all fail at the API level is counted as `api_error`.
    `pause` seconds are slept between documents that make model calls (not before the first, not for skipped ones).
    With `resume`, documents of run `eval_id` that already finished in the log (valid, or all attempts used) are
    rebuilt from it without a model call; the rest, including docs that only had API errors, run again under a new
    run_id (`-r2`, ...), and the report covers all of them.
    """
    paths = list_jds(data_dir, limit)
    attempts, json_mode = setup_args(setup, max_attempts)
    log = log_path or default_log_path()
    eval_id = eval_id or new_eval_id()
    docs, stopped = _run_docs(paths, model, setup, attempts, json_mode, eval_id, log, pause, resume)
    return build_report(
        model=model, setup=setup, max_attempts=attempts, docs=docs, logs=read_log(log), labels=load_labels(labels_dir),
        run_date=run_date or date.today().isoformat(), eval_id=eval_id, stopped_unsupported=stopped,
    )


# --- results/ ---------------------------------------------------------------

RESULTS_DIR = Path("results")
SUMMARY_FILE = "summary.md"
SUMMARY_HEADER = (
    "| model | setup | n | first-try | final | avg calls | field acc (norm) | coverage | e2e acc (norm) | cost |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def model_slug(model: str) -> str:
    """A file-name-safe model string: `openrouter/qwen/qwen-2.5-7b-instruct` -> `openrouter_qwen_qwen-2.5-7b-instruct`."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model).strip("_")


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def summary_row(report: EvalReport) -> str:
    """One markdown table row for summary.md. `n` shows the scored count when some docs got no model response."""
    n = str(report.n) if report.n_scored == report.n else f"{report.n} ({report.n_scored} scored)"
    model = report.model.replace("|", "\\|")
    if report.not_supported:  # no numbers at all, not even zeros
        cells = [model, report.setup, n, f"not supported by {model}", "-", "-", "-", "-", "-", "-"]
        return "| " + " | ".join(cells) + " |\n"
    accuracy = report.accuracy_normalized.overall if report.accuracy_normalized else None
    e2e = report.end_to_end_normalized.overall if report.end_to_end_normalized else None
    calls = "n/a" if report.avg_calls is None else f"{report.avg_calls:.2f}"
    cost = "n/a" if report.total_cost_usd is None else f"${report.total_cost_usd:.4f}"
    if report.cost_source == "estimate" and report.total_cost_usd is not None:
        cost = "~" + cost  # a list-price estimate, not the provider's figure
    cells = [
        model, report.setup, n, _fmt_pct(report.first_try_valid_pct),
        _fmt_pct(report.final_valid_pct), calls, _fmt_pct(accuracy), _fmt_pct(report.coverage_pct), _fmt_pct(e2e),
        cost,
    ]
    return "| " + " | ".join(cells) + " |\n"


def _write_new(directory: Path, stem: str, suffix: str, text: str) -> Path:
    """Write `<stem><suffix>`, or `<stem>_2<suffix>`, `_3`... if taken: an earlier result is never overwritten."""
    for n in itertools.count(1):
        path = directory / f"{stem}{'' if n == 1 else f'_{n}'}{suffix}"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except FileExistsError:
            continue
        return path
    raise AssertionError("unreachable")  # itertools.count never ends


def _append_summary(path: Path, row: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not existing.strip():
        prefix = SUMMARY_HEADER
    else:
        prefix = "" if existing.endswith("\n") else "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(prefix + row)


def _earlier_report(results_dir: Path, report: EvalReport) -> tuple[Path, EvalReport] | None:
    """The saved report of the same invocation (eval_id), model and setup, if there is one: a resumed run's first part."""
    for path in sorted(results_dir.glob("*.json")):
        try:
            old = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):  # not a report, or one written before eval_id existed
            continue
        if (old.eval_id, old.model, old.setup) == (report.eval_id, report.model, report.setup):
            return path, old
    return None


def _replace_summary_row(path: Path, old_row: str, new_row: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True) if path.is_file() else []
    if old_row not in lines:  # the row was edited or removed by hand: add the new one instead of guessing
        _append_summary(path, new_row)
        return
    lines[lines.index(old_row)] = new_row
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def write_results(report: EvalReport, results_dir: Path = RESULTS_DIR, *, resumed: bool = False) -> Path:
    """Save `<date>_<model-slug>_<setup>.json` (the full report) and append a row to summary.md. Returns the JSON path.

    A name already taken gets `_2`, `_3`... and an earlier result is never overwritten, with one exception: when
    `resumed`, the saved report of the SAME invocation (same eval_id, model and setup), i.e. the first part of the
    run, is replaced by the complete report and its summary row is replaced too. Without one, it is saved as new.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    text = report.model_dump_json(indent=2) + "\n"
    earlier = _earlier_report(results_dir, report) if resumed else None
    if earlier is not None:
        path, old = earlier
        path.write_text(text, encoding="utf-8", newline="\n")
        _replace_summary_row(results_dir / SUMMARY_FILE, summary_row(old), summary_row(report))
        return path
    stem = f"{report.run_date}_{model_slug(report.model)}_{report.setup}"
    json_path = _write_new(results_dir, stem, ".json", text)
    _append_summary(results_dir / SUMMARY_FILE, summary_row(report))
    return json_path


# --- the plan shown before a run ---------------------------------------------

SETUPS: tuple[Setup, ...] = ("A", "B", "C", "D")


def parse_setups(value: str) -> list[Setup]:
    """`A`..`D` (any case) or `all`. Raises ValueError on anything else."""
    key = value.strip().upper()
    if key == "ALL":
        return list(SETUPS)
    for setup in SETUPS:
        if key == setup:
            return [setup]
    raise ValueError(f"unknown setup {value!r}: use A, B, C, D or all")


def setup_label(setup: Setup) -> str:
    attempts, native = setup_args(setup)
    return f"setup {setup} (retry {'on' if attempts > 1 else 'off'}, native {'on' if native else 'off'})"


def check_resume(eval_id: str, model: str, log_path: Path) -> None:
    """Raise ValueError unless the log holds lines of eval run `eval_id`, all made with `model`."""
    mine = [x for x in read_log(log_path) if re.match(rf"eval-[ABCD]-{re.escape(eval_id)}-", x.run_id)]
    if not mine:
        raise ValueError(f"no eval run with id {eval_id!r} in {log_path}")
    others = sorted({x.model for x in mine} - {model})
    if others:
        raise ValueError(f"eval run {eval_id} was made with {', '.join(others)}, not {model}")


class EvalPlan(StrictModel):
    """What a run would cover and how many model calls it can make, shown before anything is sent."""

    model: str
    setups: list[Setup]
    data_dir: Path
    labels_dir: Path
    limit: int | None
    n_docs: int
    n_labels: int
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    pause: float = 0.0
    resume: str | None = None

    def call_range(self) -> tuple[int, int]:
        """(fewest, most) model calls: every doc valid at once vs every doc using all its attempts.

        Provider-level retries after API errors (up to 3 per call) come on top and are not counted.
        """
        most = sum(self.n_docs * setup_args(s, self.max_attempts)[0] for s in self.setups)
        return self.n_docs * len(self.setups), most

    def describe(self) -> list[str]:
        accuracy = (
            f"{self.n_labels} label files in {self.labels_dir}" if self.n_labels
            else f"skipped (no labels in {self.labels_dir})"
        )
        low, high = self.call_range()
        calls = str(low) if low == high else f"{low} to {high}"
        lines = [
            f"model: {self.model}",
            f"documents: {self.n_docs} in {self.data_dir}" + (f" (limit {self.limit})" if self.limit else ""),
            f"field accuracy: {accuracy}",
        ]
        lines += [f"{setup_label(s)}: {self.n_docs} to {self.n_docs * setup_args(s, self.max_attempts)[0]} calls"
                  for s in self.setups]
        if self.pause > 0:
            lines.append(f"pause between documents: {self.pause:g} s")
        if self.resume:
            lines.append(f"resuming eval {self.resume}: documents that already finished in the log are not sent again")
        return lines + [f"estimated model calls: {calls} (real API calls{', upper bound when resuming' if self.resume else ''})"]


def labels_dir_for(data_dir: Path) -> Path:
    """Gold labels sit beside the JD folder: data/jds -> data/labels (ARCHITECTURE.md layout)."""
    return data_dir.parent / "labels"


def plan_eval(
    model: str,
    setups: list[Setup],
    data_dir: Path,
    labels_dir: Path | None = None,
    limit: int | None = None,
    *,
    pause: float = 0.0,
    resume: str | None = None,
) -> EvalPlan:
    """Count the documents and label files. Raises FileNotFoundError if the folder is missing or has no .txt files."""
    n_docs = len(list_jds(data_dir, limit))
    labels = labels_dir or labels_dir_for(data_dir)
    n_labels = len(list(labels.glob("*.json"))) if labels.is_dir() else 0
    return EvalPlan(
        model=model, setups=setups, data_dir=data_dir, labels_dir=labels, limit=limit, n_docs=n_docs,
        n_labels=n_labels, pause=pause, resume=resume,
    )


def report_notes(report: EvalReport) -> list[str]:
    """Short lines about things the summary row hides: API failures, skipped labels, tool disagreements."""
    notes: list[str] = []
    if report.not_supported:
        return [f"setup {report.setup}: schema mode is not supported by {report.model} (all {report.n} doc(s) rejected)"]
    if report.n_unsupported:
        notes.append(f"{report.n_unsupported} doc(s) were rejected because schema mode is unsupported")
    if report.constraint_violations is not None:
        violations = report.constraint_violations
        detail = ", ".join(f"{name} {count}" for name, count in violations.by_constraint.items())
        notes.append(f"constraint violations under schema mode: {violations.n_outputs}" + (f" ({detail})" if detail else ""))
    if report.api_error_count:
        notes.append(f"{report.api_error_count} API error line(s) in the log (not counted as validation attempts)")
    if report.n_api_failed:
        notes.append(f"{report.n_api_failed} doc(s) got no model response and are left out of the validity numbers")
    if report.labels_skipped:
        notes.append(f"labels skipped (invalid or still TODO): {', '.join(report.labels_skipped)}")
    if report.disagreement_count:
        notes.append(f"{report.disagreement_count} model-vs-header-tool disagreement(s)")
    return [f"setup {report.setup}: {note}" for note in notes]

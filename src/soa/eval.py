"""Eval harness: run one setup over a folder of job descriptions and score it. Metric definitions: ARCHITECTURE.md.

`run_eval` makes the extract() calls and returns an `EvalReport`; `build_report` is the pure scoring step (log lines
and postings in, report out), so it is tested without any model. The plan helpers at the bottom belong to the
`soa eval` stub and go away when the CLI is wired to `run_eval`.
"""

import itertools
import logging
import re
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

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/jds")
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
    status: Literal["valid", "invalid", "api_error"]  # invalid = ExtractionFailed; api_error = no model response
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
    n: int
    n_scored: int
    n_api_failed: int
    first_try_valid_pct: float | None
    final_valid_pct: float | None
    avg_calls: float | None
    total_cost_usd: float | None
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
    docs: list[DocRun]


# --- validity, cost, failures (from the log) --------------------------------


class Validity(StrictModel):
    n_scored: int
    first_try: int
    final: int
    calls: int
    cost_usd: float | None
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
    cost, cost_known = 0.0, True
    for lines in _lines_by_run(docs, logs).values():
        api_errors += sum(1 for x in lines if x.error_type == API_ERROR_TYPE)
        responses = _responses(lines)
        if not responses:
            continue
        scored += 1
        calls += len(responses)
        first_try += responses[0].attempt == 1 and responses[0].ok
        final += any(x.ok for x in responses)
        cost += sum(x.cost_usd for x in responses)
        cost_known = cost_known and all(x.cost_known for x in responses)
        latency += sum(x.latency_ms for x in responses)
    return Validity(
        n_scored=scored, first_try=first_try, final=final, calls=calls,
        cost_usd=cost if cost_known else None, latency_ms=latency, api_errors=api_errors,
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


# --- the report -------------------------------------------------------------


def build_report(
    *, model: str, setup: Setup, max_attempts: int, docs: list[DocRun], logs: list[AttemptLog], labels: Labels,
    run_date: str,
) -> EvalReport:
    """Score a run. `logs` may hold other runs' lines too; only the docs' run_ids are read."""
    v = validity_stats(docs, logs)
    acc = _accuracy(docs, labels)
    doc_ids = {d.doc_id for d in docs}
    return EvalReport(
        model=model, setup=setup, json_mode=setup_args(setup)[1], max_attempts=max_attempts, run_date=run_date,
        n=len(docs), n_scored=v.n_scored, n_api_failed=len(docs) - v.n_scored,
        first_try_valid_pct=_pct(v.first_try, v.n_scored), final_valid_pct=_pct(v.final, v.n_scored),
        avg_calls=v.calls / v.n_scored if v.n_scored else None,
        total_cost_usd=v.cost_usd,
        avg_latency_ms=v.latency_ms / v.n_scored if v.n_scored else None,
        api_error_count=v.api_errors, disagreement_count=sum(d.disagreements for d in docs),
        n_labelled=acc.n_labelled, n_labelled_with_output=acc.n_with_output,
        coverage_pct=_pct(acc.n_with_output, acc.n_labelled),
        labels_skipped=[s for s in labels.skipped if s in doc_ids],
        accuracy_raw=acc.raw, accuracy_normalized=acc.normalized,
        end_to_end_raw=acc.end_to_end_raw, end_to_end_normalized=acc.end_to_end_normalized,
        failures=failure_breakdown(docs, logs), docs=docs,
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


def run_one(
    path: Path, model: str, setup: Setup, max_attempts: int, json_mode: bool, eval_id: str, log_path: Path
) -> DocRun:
    """extract() on one document with a fresh run_id. ExtractionFailed and API errors become a DocRun, not a raise."""
    doc_id = path.stem
    run_id = f"eval-{setup}-{eval_id}-{doc_id}"
    text = path.read_text(encoding="utf-8-sig")
    try:
        result = extract(
            text, model, max_attempts, json_mode, doc_id=doc_id, run_id=run_id, log_path=log_path, setup=setup
        )
    except ExtractionFailed:
        return DocRun(doc_id=doc_id, run_id=run_id, status="invalid")
    except LLMAPIError as exc:
        logger.error("%s: no model response (%s)", doc_id, redact_keys(" ".join(str(exc).split())))
        return DocRun(doc_id=doc_id, run_id=run_id, status="api_error")
    return DocRun(
        doc_id=doc_id, run_id=run_id, status="valid", raw_posting=result.raw_posting, posting=result.posting,
        disagreements=len(result.disagreements),
    )


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
) -> EvalReport:
    """Run `setup` on every document (up to `limit`) and score it. REAL model calls unless llm.call is mocked.

    Each document gets a fresh run_id; ExtractionFailed is counted as an invalid document and the run continues.
    A document whose calls all fail at the API level is counted as `api_error`, not as invalid.
    """
    paths = list_jds(data_dir, limit)
    attempts, json_mode = setup_args(setup, max_attempts)
    log = log_path or default_log_path()
    eval_id = uuid.uuid4().hex[:6]
    docs = [run_one(p, model, setup, attempts, json_mode, eval_id, log) for p in paths]
    return build_report(
        model=model, setup=setup, max_attempts=attempts, docs=docs, logs=read_log(log), labels=load_labels(labels_dir),
        run_date=run_date or date.today().isoformat(),
    )


# --- results/ ---------------------------------------------------------------

RESULTS_DIR = Path("results")
SUMMARY_FILE = "summary.md"
SUMMARY_HEADER = (
    "| model | setup | n | first-try | final | avg calls | field acc (norm) | coverage | cost |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def model_slug(model: str) -> str:
    """A file-name-safe model string: `openrouter/qwen/qwen-2.5-7b-instruct` -> `openrouter_qwen_qwen-2.5-7b-instruct`."""
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model).strip("_")


def _fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def summary_row(report: EvalReport) -> str:
    """One markdown table row for summary.md. `n` shows the scored count when some docs got no model response."""
    n = str(report.n) if report.n_scored == report.n else f"{report.n} ({report.n_scored} scored)"
    accuracy = report.accuracy_normalized.overall if report.accuracy_normalized else None
    calls = "n/a" if report.avg_calls is None else f"{report.avg_calls:.2f}"
    cost = "n/a" if report.total_cost_usd is None else f"${report.total_cost_usd:.4f}"
    cells = [
        report.model.replace("|", "\\|"), report.setup, n, _fmt_pct(report.first_try_valid_pct),
        _fmt_pct(report.final_valid_pct), calls, _fmt_pct(accuracy), _fmt_pct(report.coverage_pct), cost,
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


def write_results(report: EvalReport, results_dir: Path = RESULTS_DIR) -> Path:
    """Save `<date>_<model-slug>_<setup>.json` (the full report) and append a row to summary.md. Returns the JSON path."""
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.run_date}_{model_slug(report.model)}_{report.setup}"
    json_path = _write_new(results_dir, stem, ".json", report.model_dump_json(indent=2) + "\n")
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
        return lines + [f"estimated model calls: {calls} (real API calls)"]


def labels_dir_for(data_dir: Path) -> Path:
    """Gold labels sit beside the JD folder: data/jds -> data/labels (ARCHITECTURE.md layout)."""
    return data_dir.parent / "labels"


def plan_eval(
    model: str, setups: list[Setup], data_dir: Path, labels_dir: Path | None = None, limit: int | None = None
) -> EvalPlan:
    """Count the documents and label files. Raises FileNotFoundError if the folder is missing or has no .txt files."""
    n_docs = len(list_jds(data_dir, limit))
    labels = labels_dir or labels_dir_for(data_dir)
    n_labels = len(list(labels.glob("*.json"))) if labels.is_dir() else 0
    return EvalPlan(
        model=model, setups=setups, data_dir=data_dir, labels_dir=labels, limit=limit, n_docs=n_docs, n_labels=n_labels
    )


def report_notes(report: EvalReport) -> list[str]:
    """Short lines about things the summary row hides: API failures, skipped labels, tool disagreements."""
    notes: list[str] = []
    if report.api_error_count:
        notes.append(f"{report.api_error_count} API error line(s) in the log (not counted as validation attempts)")
    if report.n_api_failed:
        notes.append(f"{report.n_api_failed} doc(s) got no model response and are left out of the validity numbers")
    if report.labels_skipped:
        notes.append(f"labels skipped (invalid or still TODO): {', '.join(report.labels_skipped)}")
    if report.disagreement_count:
        notes.append(f"{report.disagreement_count} model-vs-header-tool disagreement(s)")
    return [f"setup {report.setup}: {note}" for note in notes]

"""eval.py: metric math on fixture logs and labels, then run_eval with llm.call mocked (rule 13)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from soa import agent, llm
from soa.eval import (
    DocRun,
    Labels,
    build_report,
    check_resume,
    failure_breakdown,
    first_error_loc,
    load_labels,
    next_run_id,
    parse_setups,
    plan_eval,
    report_notes,
    run_eval,
    schema_constraint,
    score_fields,
    summary_row,
    setup_args,
    validity_stats,
)
from soa.errors import format_validation_errors
from soa.llm import CostSource, LLMAPIError, LLMResponse, Message
from soa.logger import AttemptLog, Setup, append_log, read_log
from soa.schemas import JobPosting

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": "Acme",
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}, {"name": "SQL", "required": False}],
    "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
    "salary_mentioned": False,
}


def posting(**overrides: Any) -> JobPosting:
    return JobPosting.model_validate_json(json.dumps({**VALID, **overrides}))  # strict models need JSON mode


def skills(*names: str) -> dict[str, Any]:
    return {"skills": [{"name": n, "required": True} for n in names]}


# --- fixture log lines ------------------------------------------------------


def line(
    run_id: str, attempt: int = 1, *, ok: bool = True, error_type: str | None = None, error: str | None = None,
    cost: float = 0.001, known: bool = True, latency: int = 1000, setup: Setup | None = "B",
    source: CostSource | None = "provider",
) -> AttemptLog:
    if not ok:
        error_type = error_type or "ValidationError"
        error = error or "seniority: Input should be 'intern', 'junior', 'mid', 'senior' or 'lead' (got: 'x')"
    return AttemptLog(
        run_id=run_id, doc_id=f"doc-{run_id}", attempt=attempt, model="m", setup=setup, json_mode=False, ok=ok,
        error_type=error_type, error=error, raw_output="{}", latency_ms=latency, prompt_tokens=100,
        completion_tokens=20, cost_usd=cost, cost_known=known, cost_source=source if known else "none",
    )


def api_line(run_id: str, attempt: int = 1) -> AttemptLog:
    """What agent._log_api_error writes: no output, no tokens, cost 0.0 and cost_known False."""
    return AttemptLog(
        run_id=run_id, doc_id=f"doc-{run_id}", attempt=attempt, model="m", setup="B", json_mode=False, ok=False,
        error_type="APIError", error="RateLimitError: slow down", raw_output="", latency_ms=50, prompt_tokens=0,
        completion_tokens=0, cost_usd=0.0, cost_known=False, cost_source="none",
    )


def doc(run_id: str, status: str = "valid", **kwargs: Any) -> DocRun:
    postings: dict[str, Any] = {"raw_posting": posting(), "posting": posting()} if status == "valid" else {}
    return DocRun(doc_id=f"doc-{run_id}", run_id=run_id, status=status, **postings, **kwargs)  # type: ignore[arg-type]


# r1 valid first try; r2 fails once then valid; r3 three failures (ExtractionFailed); r4 only API errors;
# r5 one API error, then valid on its first model response
DOCS = [doc("r1"), doc("r2"), doc("r3", "invalid"), doc("r4", "api_error"), doc("r5")]
LOGS = [
    line("r1"),
    line("r2", 1, ok=False), line("r2", 2),
    line("r3", 1, ok=False), line("r3", 2, ok=False), line("r3", 3, ok=False),
    api_line("r4"), api_line("r4"), api_line("r4"),
    api_line("r5"), line("r5", 1),
]


def report(docs: list[DocRun] = DOCS, logs: list[AttemptLog] = LOGS, labels: Labels | None = None) -> Any:
    return build_report(
        model="m", setup="B", max_attempts=3, docs=docs, logs=logs,
        labels=labels or Labels(by_doc={}, skipped=[]), run_date="2026-10-17",
    )


# --- validity, calls, cost, latency -----------------------------------------


def test_validity_counts_come_from_model_responses_only() -> None:
    r = report()
    assert (r.n, r.n_scored, r.n_api_failed) == (5, 4, 1)
    assert r.first_try_valid_pct == pytest.approx(50.0)  # r1, r5 of the 4 that got a response
    assert r.final_valid_pct == pytest.approx(75.0)  # r1, r2, r5; r3 stayed invalid
    assert r.avg_calls == pytest.approx(7 / 4)  # 1 + 2 + 3 + 1 model responses, API errors not counted
    assert r.avg_latency_ms == pytest.approx(7000 / 4)


def test_api_error_lines_are_counted_but_never_enter_validity_cost_or_latency() -> None:
    r = report()
    assert r.api_error_count == 4
    assert r.total_cost_usd == pytest.approx(0.007)  # 7 responses; APIError lines (cost_known False) do not make it n/a


def test_a_doc_with_only_api_errors_is_not_scored() -> None:
    r = report([doc("r4", "api_error")], [api_line("r4"), api_line("r4", 2)])
    assert (r.n, r.n_scored, r.n_api_failed, r.api_error_count) == (1, 0, 1, 2)
    assert (r.first_try_valid_pct, r.final_valid_pct, r.avg_calls, r.avg_latency_ms) == (None, None, None, None)


def test_api_error_before_a_first_valid_reply_still_counts_as_first_try() -> None:
    r = report([doc("r5")], [api_line("r5"), line("r5", 1)])
    assert r.first_try_valid_pct == 100.0 and r.api_error_count == 1


def test_an_extraction_failed_doc_counts_as_invalid() -> None:
    r = report([doc("r3", "invalid")], [line("r3", 1, ok=False), line("r3", 2, ok=False), line("r3", 3, ok=False)])
    assert (r.n_scored, r.first_try_valid_pct, r.final_valid_pct, r.avg_calls) == (1, 0.0, 0.0, 3.0)


def test_unknown_cost_makes_the_total_n_a() -> None:
    logs = [line("r1", cost=0.0, known=False), line("r2", cost=0.002)]
    r = report([doc("r1"), doc("r2")], logs)
    assert r.total_cost_usd is None


def test_lines_of_other_runs_are_ignored() -> None:
    other = [line("zzz", ok=False), api_line("zzz")]
    assert report(logs=LOGS + other) == report()


def test_validity_stats_order_of_lines_does_not_matter() -> None:
    shuffled = [LOGS[2], LOGS[1], *LOGS[3:], LOGS[0]]
    assert validity_stats(DOCS, shuffled) == validity_stats(DOCS, LOGS)


# --- failure breakdown --------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "loc"),
    [
        ("skills.3.name: String should have at most 40 characters (got: 'x')", "skills.*.name"),
        ("skills.12.required: Input should be a valid boolean (got: 'yes')", "skills.*.required"),
        ("location.mode: Input should be 'remote'... (got: 'x')\nseniority: other", "location.mode"),
        ("(root): Invalid JSON: expected value (got: '```json')", "(root)"),
        ("title: Field required (got: {})", "title"),
        ("something without a location", "(unknown)"),
        ("", "(unknown)"),
    ],
)
def test_first_error_loc(error: str, loc: str) -> None:
    assert first_error_loc(error) == loc


def test_failure_breakdown_counts_by_type_and_first_loc_excluding_api_errors() -> None:
    logs = [
        line("r1", 1, ok=False, error="skills.0.name: too long (got: 'x')\nseniority: bad (got: 'y')"),
        line("r1", 2, ok=False, error="skills.5.name: too long (got: 'x')"),
        line("r2", 1, ok=False, error_type="JSONDecodeError", error="(root): Invalid JSON (got: 'z')"),
        line("r2", 2, ok=False, error="title: Field required (got: {})"),
        line("r2", 3),
        api_line("r1"), api_line("r2"),
    ]
    b = failure_breakdown([doc("r1"), doc("r2")], logs)
    assert b.by_error_type == {"ValidationError": 3, "JSONDecodeError": 1}
    assert b.by_first_loc == {"skills.*.name": 2, "(root)": 1, "title": 1}  # most frequent first; only each first line
    assert list(b.by_first_loc)[0] == "skills.*.name"


def test_no_failures_gives_an_empty_breakdown() -> None:
    b = failure_breakdown([doc("r1")], [line("r1")])
    assert (b.by_error_type, b.by_first_loc) == ({}, {})


# --- field accuracy -----------------------------------------------------------


def test_perfect_prediction_scores_100_everywhere() -> None:
    acc = score_fields([(posting(), posting())])
    assert acc.overall == 100.0 and acc.skills_f1 == 100.0 and acc.n == 1


def test_skills_are_compared_casefolded_with_precision_recall_f1() -> None:
    gold, predicted = posting(**skills("Python", "SQL")), posting(**skills("python", "sql", "Go"))
    acc = score_fields([(gold, predicted)])
    assert acc.skills_precision == pytest.approx(200 / 3)  # tp 2, fp 1
    assert acc.skills_recall == 100.0
    assert acc.skills_f1 == pytest.approx(80.0)  # 2*2 / (2*2 + 1 + 0)


def test_skill_scores_are_micro_averaged_over_docs() -> None:
    pairs = [
        (posting(**skills("A", "B")), posting(**skills("a"))),  # tp 1, fn 1
        (posting(**skills("C")), posting(**skills("c", "d", "e"))),  # tp 1, fp 2
    ]
    acc = score_fields(pairs)
    assert acc.skills_precision == pytest.approx(100 * 2 / 4)
    assert acc.skills_recall == pytest.approx(100 * 2 / 3)
    assert acc.skills_f1 == pytest.approx(100 * 4 / 7)


def test_title_and_company_ignore_case_but_city_and_country_do_not() -> None:
    gold = posting()
    predicted = posting(title="BACKEND engineer", company="acme", location={**VALID["location"], "city": "pune"})
    acc = score_fields([(gold, predicted)])
    assert (acc.title, acc.company) == (100.0, 100.0)
    assert (acc.city, acc.country) == (0.0, 100.0)


def test_none_matches_none_but_not_a_value() -> None:
    gold = posting(company=None, min_years_experience=None)
    acc = score_fields([(gold, posting(company=None, min_years_experience=None)), (gold, posting())])
    assert (acc.company, acc.min_years_experience) == (50.0, 50.0)


def test_each_exact_field_is_scored_on_its_own() -> None:
    gold = posting()
    wrong = posting(
        seniority="senior", min_years_experience=5, salary_mentioned=True,
        location={"city": "Pune", "country": "India", "mode": "remote"},
    )
    acc = score_fields([(gold, wrong)])
    assert (acc.seniority, acc.min_years_experience, acc.salary_mentioned, acc.mode) == (0.0, 0.0, 0.0, 0.0)
    assert (acc.title, acc.company, acc.city, acc.country) == (100.0, 100.0, 100.0, 100.0)
    # five of the nine numbers are 100 (title, company, city, country, skills_f1)
    assert acc.overall == pytest.approx(100 * 5 / 9)


def test_raw_and_normalized_accuracy_differ_when_post_processing_changed_something() -> None:
    gold = posting(**skills("PostgreSQL"))
    raw = posting(**skills("postgres"), location={"city": None, "country": "India", "mode": "hybrid"})
    normalized = posting(**skills("PostgreSQL"))
    run = DocRun(doc_id="doc-r1", run_id="r1", status="valid", raw_posting=raw, posting=normalized)
    r = report([run], [line("r1")], Labels(by_doc={"doc-r1": gold}, skipped=[]))
    assert r.accuracy_raw.skills_f1 == 0.0 and r.accuracy_raw.city == 0.0
    assert r.accuracy_normalized.skills_f1 == 100.0 and r.accuracy_normalized.city == 100.0
    assert r.accuracy_normalized.overall > r.accuracy_raw.overall


def test_accuracy_covers_labelled_docs_with_valid_output_only() -> None:
    labels = Labels(by_doc={"doc-r1": posting(), "doc-r3": posting(), "doc-r4": posting()}, skipped=[])
    r = report(labels=labels)  # r3 is invalid, r4 had only API errors, r2/r5 are unlabelled
    assert (r.n_labelled, r.n_labelled_with_output) == (3, 1)
    assert r.accuracy_raw.n == 1 and r.accuracy_normalized.n == 1


def test_no_labels_skips_accuracy() -> None:
    r = report()
    assert (r.n_labelled, r.n_labelled_with_output) == (0, 0)
    assert (r.accuracy_raw, r.accuracy_normalized) == (None, None)
    assert (r.coverage_pct, r.end_to_end_raw, r.end_to_end_normalized) == (None, None, None)


def test_a_missing_prediction_scores_zero_on_every_field() -> None:
    acc = score_fields([(posting(), posting()), (posting(), None)])
    assert (acc.n, acc.title, acc.seniority, acc.city, acc.salary_mentioned) == (2, 50.0, 50.0, 50.0, 50.0)
    assert (acc.skills_precision, acc.skills_recall, acc.skills_f1) == (100.0, 50.0, pytest.approx(200 / 3))
    nothing = score_fields([(posting(), None)])
    assert (nothing.overall, nothing.skills_precision, nothing.skills_recall, nothing.skills_f1) == (0.0, 0.0, 0.0, 0.0)


def test_coverage_is_labelled_docs_with_output_over_usable_labelled_docs() -> None:
    labels = Labels(by_doc={"doc-r1": posting(), "doc-r3": posting(), "doc-r4": posting()}, skipped=["doc-r2"])
    r = report(labels=labels)  # r1 valid; r3 invalid; r4 api_error; r2 has a skipped label so is not usable
    assert r.coverage_pct == pytest.approx(100 / 3)
    assert r.accuracy_raw is not None and r.accuracy_raw.overall == 100.0  # on valid outputs only
    assert r.end_to_end_raw is not None and r.end_to_end_raw.n == 3
    # r3 and r4 count as wrong: eight fields at 1/3; skills tp 2, fn 4 -> F1 = 4/(4+4) = 50%
    assert r.end_to_end_raw.skills_f1 == pytest.approx(50.0)
    assert r.end_to_end_raw.overall == pytest.approx((8 * 100 / 3 + 50) / 9)
    assert r.end_to_end_normalized is not None and r.end_to_end_normalized.title == pytest.approx(100 / 3)


def test_labels_without_any_valid_output_skip_accuracy_but_say_why() -> None:
    r = report([doc("r3", "invalid")], [line("r3", ok=False)], Labels(by_doc={"doc-r3": posting()}, skipped=[]))
    assert (r.n_labelled, r.n_labelled_with_output, r.accuracy_raw) == (1, 0, None)


def test_skipped_labels_are_reported_only_for_docs_in_the_run() -> None:
    r = report(labels=Labels(by_doc={}, skipped=["doc-r2", "jd_099"]))
    assert r.labels_skipped == ["doc-r2"]


def test_disagreements_are_summed() -> None:
    r = report([doc("r1", disagreements=2), doc("r2", disagreements=1)], [line("r1"), line("r2")])
    assert r.disagreement_count == 3


# --- labels ---------------------------------------------------------------------


def test_load_labels_skips_invalid_unreadable_and_todo_files(tmp_path: Path) -> None:
    (tmp_path / "jd_001.json").write_text(json.dumps(VALID), encoding="utf-8")
    (tmp_path / "jd_002.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "jd_003.json").write_text(json.dumps({**VALID, "seniority": "wizard"}), encoding="utf-8")
    (tmp_path / "jd_004.json").write_text(json.dumps({**VALID, "title": "TODO: check"}), encoding="utf-8")
    (tmp_path / "jd_005.json").write_bytes(b"\xff\xfe\x00bad")
    labels = load_labels(tmp_path)
    assert list(labels.by_doc) == ["jd_001"]
    assert labels.skipped == ["jd_002", "jd_003", "jd_004", "jd_005"]


def test_load_labels_of_a_missing_folder_is_empty(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "nope") == Labels(by_doc={}, skipped=[])


# --- models and setups ------------------------------------------------------------


def test_a_valid_docrun_needs_postings_and_the_others_must_not_have_them() -> None:
    with pytest.raises(ValidationError):
        DocRun(doc_id="d", run_id="r", status="valid")
    with pytest.raises(ValidationError):
        DocRun(doc_id="d", run_id="r", status="invalid", raw_posting=posting(), posting=posting())


@pytest.mark.parametrize(
    ("setup", "expected"), [("A", (1, False)), ("B", (3, False)), ("C", (1, True)), ("D", (3, True))]
)
def test_setup_args(setup: Setup, expected: tuple[int, bool]) -> None:
    assert setup_args(setup) == expected
    assert setup_args(setup, 5)[0] == (5 if setup in ("B", "D") else 1)


# --- run_eval with a mocked llm.call ------------------------------------------------


def reply(obj: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(obj), prompt_tokens=100, completion_tokens=50, cost_usd=0.002, cost_known=True,
        cost_source="provider", latency_ms=500, model="m",
    )


BAD = reply({**VALID, "seniority": "wizard"})
GOOD = reply(VALID)


def make_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scripts: dict[str, list[LLMResponse | LLMAPIError]]
) -> tuple[Path, Path, list[tuple[str, type[BaseModel] | None]]]:
    """JD files named after `scripts`' keys; llm.call answers each doc from its script. Returns (jds, labels, calls)."""
    jds, labels = tmp_path / "jds", tmp_path / "labels"
    jds.mkdir()
    labels.mkdir()
    for name in scripts:
        (jds / f"{name}.txt").write_text(f"MARKER-{name}\nBackend Engineer\nAcme\nPune, India\nHybrid\n", encoding="utf-8")
    calls: list[tuple[str, type[BaseModel] | None]] = []

    def fake(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
        name = next(n for n in scripts if f"MARKER-{n}" in messages[1].content)
        calls.append((name, response_model))
        item = scripts[name].pop(0)
        if isinstance(item, BaseException):  # an LLMAPIError, or a KeyboardInterrupt to cut a run short
            raise item
        return item

    monkeypatch.setattr(llm, "call", fake)
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    return jds, labels, calls


def test_run_eval_continues_after_failures_and_scores_every_document(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[LLMResponse | LLMAPIError]] = {
        "jd_001": [GOOD],
        "jd_002": [BAD, GOOD],
        "jd_003": [BAD, BAD, BAD],
        "jd_004": [LLMAPIError("AuthenticationError: bad key", model="m", retryable=False, status_code=401)],
    }
    jds, labels, calls = make_world(tmp_path, monkeypatch, scripts)
    log = tmp_path / "attempts.jsonl"
    (labels / "jd_001.json").write_text(json.dumps(VALID), encoding="utf-8")

    r = run_eval("m", "B", jds, labels, log_path=log, run_date="2026-10-17")

    assert [d.status for d in r.docs] == ["valid", "valid", "invalid", "api_error"]
    assert (r.n, r.n_scored, r.n_api_failed) == (4, 3, 1)
    assert r.first_try_valid_pct == pytest.approx(100 / 3)
    assert r.final_valid_pct == pytest.approx(200 / 3)
    assert r.avg_calls == pytest.approx(6 / 3)
    assert r.api_error_count == 1
    assert r.failures.by_error_type == {"ValidationError": 4}
    assert r.failures.by_first_loc == {"seniority": 4}
    assert r.accuracy_normalized is not None and r.accuracy_normalized.n == 1
    assert len(calls) == 7  # 1 + 2 + 3 + 1 (the API error is a call that raised)


def test_each_document_gets_its_own_fresh_run_id_and_every_line_is_labelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD], "jd_002": [BAD, GOOD]})
    log = tmp_path / "attempts.jsonl"
    first = run_eval("m", "B", jds, labels, log_path=log)
    ids_first = [d.run_id for d in first.docs]
    assert len(set(ids_first)) == 2
    assert all(i.startswith("eval-B-") for i in ids_first)

    rows = read_log(log)
    assert {(x.setup, x.json_mode) for x in rows} == {("B", False)}
    assert {x.run_id for x in rows} == set(ids_first)


def test_a_second_run_is_scored_from_its_own_lines_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: dict[str, list[LLMResponse | LLMAPIError]] = {"jd_001": [BAD, GOOD, GOOD]}
    jds, labels, _ = make_world(tmp_path, monkeypatch, scripts)
    log = tmp_path / "attempts.jsonl"
    one = run_eval("m", "B", jds, labels, log_path=log)  # the same log file keeps growing
    two = run_eval("m", "B", jds, labels, log_path=log)
    assert one.first_try_valid_pct == 0.0 and two.first_try_valid_pct == 100.0
    assert one.docs[0].run_id != two.docs[0].run_id


@pytest.mark.parametrize(
    ("setup", "attempts", "native"), [("A", 1, False), ("B", 3, False), ("C", 1, True), ("D", 3, True)]
)
def test_setups_map_to_attempts_and_schema_mode(
    setup: Setup, attempts: int, native: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jds, labels, calls = make_world(tmp_path, monkeypatch, {"jd_001": [BAD] * 5})
    r = run_eval("m", setup, jds, labels, log_path=tmp_path / "a.jsonl")
    assert len(calls) == attempts  # the retry loop is off for A and C
    assert {mode for _, mode in calls} == ({JobPosting} if native else {None})
    assert (r.setup, r.json_mode, r.max_attempts) == (setup, native, attempts)
    assert {x.setup for x in read_log(tmp_path / "a.jsonl")} == {setup}


def test_limit_takes_the_first_n_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, calls = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD], "jd_002": [GOOD], "jd_003": [GOOD]})
    r = run_eval("m", "A", jds, labels, limit=2, log_path=tmp_path / "a.jsonl")
    assert [d.doc_id for d in r.docs] == ["jd_001", "jd_002"] and len(calls) == 2


def test_run_eval_without_documents_raises(tmp_path: Path) -> None:
    empty = tmp_path / "jds"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .txt job descriptions"):
        run_eval("m", "A", empty, tmp_path / "labels")
    with pytest.raises(FileNotFoundError, match="data directory not found"):
        run_eval("m", "A", tmp_path / "nope", tmp_path / "labels")


def test_skipped_labels_show_up_in_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD]})
    (labels / "jd_001.json").write_text(json.dumps({**VALID, "title": "TODO"}), encoding="utf-8")
    r = run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl")
    assert r.labels_skipped == ["jd_001"] and r.accuracy_raw is None


UNSUPPORTED = LLMAPIError(
    "schema mode unsupported by m: NotFoundError: No endpoints found", model="m", retryable=False, status_code=404,
    schema_unsupported=True,
)


def test_a_schema_mode_run_rejected_everywhere_is_reported_as_not_supported_with_no_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [UNSUPPORTED], "jd_002": [UNSUPPORTED]})
    (labels / "jd_001.json").write_text(json.dumps(VALID), encoding="utf-8")
    r = run_eval("m", "C", jds, labels, log_path=tmp_path / "a.jsonl")
    assert [d.status for d in r.docs] == ["unsupported", "unsupported"]
    assert (r.not_supported, r.n_unsupported, r.n_scored) == (True, 2, 0)
    assert (r.coverage_pct, r.accuracy_raw, r.end_to_end_raw, r.end_to_end_normalized) == (None, None, None, None)
    assert summary_row(r) == "| m | C | 2 (0 scored) | not supported by m | - | - | - | - | - | - |\n"
    assert report_notes(r) == ["setup C: schema mode is not supported by m (all 2 doc(s) rejected)"]


def test_a_schema_mode_run_stops_after_two_unsupported_docs_in_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[Any]] = {n: [UNSUPPORTED] for n in ("jd_001", "jd_002", "jd_003", "jd_004")}
    jds, labels, calls = make_world(tmp_path, monkeypatch, scripts)
    r = run_eval("m", "D", jds, labels, log_path=tmp_path / "a.jsonl")
    assert [name for name, _ in calls] == ["jd_001", "jd_002"]  # jd_003 and jd_004 were never sent
    assert (r.n, r.not_supported, r.n_unsupported) == (2, True, 2)
    assert summary_row(r).startswith("| m | D | 2 (0 scored) | not supported by m |")


def test_one_unsupported_doc_between_good_ones_does_not_stop_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[Any]] = {
        "jd_001": [UNSUPPORTED], "jd_002": [GOOD], "jd_003": [UNSUPPORTED], "jd_004": [GOOD],
    }
    jds, labels, calls = make_world(tmp_path, monkeypatch, scripts)
    r = run_eval("m", "C", jds, labels, log_path=tmp_path / "a.jsonl")
    assert len(calls) == 4 and (r.not_supported, r.n_unsupported, r.n) == (False, 2, 4)


def test_plain_setups_never_stop_early_on_unsupported_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts: dict[str, list[Any]] = {n: [UNSUPPORTED] for n in ("jd_001", "jd_002", "jd_003")}
    jds, labels, calls = make_world(tmp_path, monkeypatch, scripts)
    r = run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl")
    assert len(calls) == 3 and r.not_supported is False


def test_a_partly_rejected_run_keeps_its_numbers_and_says_how_many_were_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD], "jd_002": [UNSUPPORTED]})
    r = run_eval("m", "D", jds, labels, log_path=tmp_path / "a.jsonl")
    assert (r.not_supported, r.n_unsupported, r.n_scored) == (False, 1, 1)
    assert r.final_valid_pct == 100.0
    assert "setup D: 1 doc(s) were rejected because schema mode is unsupported" in report_notes(r)


def test_a_plain_api_error_in_schema_mode_is_api_error_not_unsupported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    down = LLMAPIError("AuthenticationError: bad key", model="m", retryable=False, status_code=401)
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [down]})
    r = run_eval("m", "C", jds, labels, log_path=tmp_path / "a.jsonl")
    assert (r.docs[0].status, r.not_supported, r.n_unsupported) == ("api_error", False, 0)


def test_the_not_supported_flag_is_only_for_schema_mode_setups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [UNSUPPORTED]})
    r = run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl")  # cannot happen for real; the flag stays off
    assert r.not_supported is False


# --- pacing and resume ------------------------------------------------------------------


def test_pause_is_slept_between_documents_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD], "jd_002": [GOOD], "jd_003": [GOOD]})
    waits: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", waits.append)
    run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl", pause=1.5)
    assert waits == [1.5, 1.5]  # not before the first document, not after the last


def test_no_pause_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD], "jd_002": [GOOD]})
    waits: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", waits.append)
    run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl")
    assert waits == []


def test_next_run_id_numbers_reruns_and_does_not_confuse_doc_ids_that_share_a_prefix() -> None:
    base = "eval-B-abc123-jd_001"
    lines = [
        line(base), line(f"{base}-r2"), line("eval-B-abc123-jd_0011"), line("eval-B-abc123-jd_001-x"),
        line("eval-D-abc123-jd_001"), line("eval-B-zzz999-jd_001"),
    ]
    assert next_run_id([], base) == base
    assert next_run_id(lines[:1], base) == f"{base}-r2"
    assert next_run_id(lines, base) == f"{base}-r3"


def test_resume_skips_finished_docs_reruns_the_rest_and_reports_all_of_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts: dict[str, list[Any]] = {
        "jd_001": [GOOD],
        "jd_002": [BAD, BAD, BAD],  # ExtractionFailed: finished (invalid), not retried on resume
        "jd_003": [LLMAPIError("AuthenticationError: bad key", model="m", retryable=False, status_code=401)],
        "jd_004": [BAD, KeyboardInterrupt()],  # the run is cut short in the middle of this doc
    }
    jds, labels, calls = make_world(tmp_path, monkeypatch, scripts)
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run_eval("m", "B", jds, labels, log_path=log, eval_id="abc123")
    assert len(calls) == 1 + 3 + 1 + 2

    scripts["jd_003"].append(GOOD)  # jd_001 and jd_002 have empty scripts: a call for them would raise IndexError
    scripts["jd_004"].append(GOOD)
    waits: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", waits.append)
    r = run_eval("m", "B", jds, labels, log_path=log, eval_id="abc123", resume=True, pause=0.5)

    assert len(calls) == 7 + 2  # only jd_003 and jd_004 were sent again
    assert waits == [0.5]  # between those two; nothing for the skipped docs
    assert [d.status for d in r.docs] == ["valid", "invalid", "valid", "valid"]
    assert [d.run_id for d in r.docs] == [
        "eval-B-abc123-jd_001", "eval-B-abc123-jd_002", "eval-B-abc123-jd_003-r2", "eval-B-abc123-jd_004-r2",
    ]
    assert r.eval_id == "abc123" and (r.n, r.n_scored) == (4, 4)
    assert r.docs[0].raw_posting == posting() and r.docs[0].posting == posting()  # rebuilt from the log
    assert r.first_try_valid_pct == 75.0 and r.final_valid_pct == 75.0
    assert r.avg_calls == pytest.approx((1 + 3 + 1 + 1) / 4)  # the superseded runs' lines are not counted
    assert r.api_error_count == 0


def test_resume_rebuilds_the_normalized_posting_and_disagreements(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    messy = reply({**VALID, "skills": [{"name": "postgres", "required": True}], "location": {"city": "Mumbai", "country": "India", "mode": "hybrid"}})
    jds, labels, _ = make_world(tmp_path, monkeypatch, {"jd_001": [messy]})
    log = tmp_path / "attempts.jsonl"
    first = run_eval("m", "B", jds, labels, log_path=log, eval_id="abc123")
    again = run_eval("m", "B", jds, labels, log_path=log, eval_id="abc123", resume=True)  # nothing left to send
    assert again.docs == first.docs
    assert again.docs[0].posting is not None and again.docs[0].posting.skills[0].name == "PostgreSQL"
    assert again.docs[0].disagreements == first.docs[0].disagreements == 1  # the header says Pune, the model Mumbai


def test_resume_of_a_setup_that_never_ran_just_runs_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, calls = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD]})
    r = run_eval("m", "C", jds, labels, log_path=tmp_path / "a.jsonl", eval_id="abc123", resume=True)
    assert len(calls) == 1 and r.docs[0].run_id == "eval-C-abc123-jd_001"


def test_without_resume_the_same_eval_id_does_not_skip_anything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, labels, calls = make_world(tmp_path, monkeypatch, {"jd_001": [GOOD, GOOD]})
    run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl", eval_id="abc123")
    run_eval("m", "A", jds, labels, log_path=tmp_path / "a.jsonl", eval_id="abc123")
    assert len(calls) == 2


def test_check_resume_needs_the_run_in_the_log_and_the_same_model(tmp_path: Path) -> None:
    log = tmp_path / "a.jsonl"
    with pytest.raises(ValueError, match="no eval run with id 'abc123'"):
        check_resume("abc123", "m", log)
    append_log(line("eval-B-abc123-jd_001"), log)
    check_resume("abc123", "m", log)
    with pytest.raises(ValueError, match="was made with m, not other"):
        check_resume("abc123", "other", log)
    with pytest.raises(ValueError, match="no eval run with id 'abc12'"):
        check_resume("abc12", "m", log)  # an id is matched whole, not as a prefix


# --- constraint violations under schema mode ------------------------------------------


def error_text(**overrides: Any) -> str:
    """The real retry-feedback text for a payload that breaks the schema (no model involved)."""
    with pytest.raises(ValidationError) as exc:
        JobPosting.model_validate_json(json.dumps({**VALID, **overrides}))
    return format_validation_errors(exc.value)


@pytest.mark.parametrize(
    ("overrides", "kind"),
    [
        ({"title": "x"}, "string_length"),
        ({"title": "x" * 101}, "string_length"),
        ({"skills": []}, "list_size"),
        ({"seniority": "wizard"}, "enum"),
        ({"salary_mentioned": "yes"}, "type"),
        ({"min_years_experience": "3"}, "type"),
        ({"min_years_experience": 31}, "number_bound"),
        ({"min_years_experience": -1}, "number_bound"),
        ({"bogus": 1}, "extra_field"),
        ({"skills": [{"name": "A", "required": True}, {"name": "a", "required": True}]}, None),  # custom rule
        ({"seniority": "intern", "min_years_experience": 2}, None),  # custom rule
    ],
)
def test_schema_constraint_names_the_violated_schema_rule(overrides: dict[str, Any], kind: str | None) -> None:
    kinds = {schema_constraint(e) for e in error_text(**overrides).splitlines()}
    assert kinds == {kind}


def test_schema_constraint_ignores_invalid_json_and_a_missing_key() -> None:
    assert schema_constraint("(root): Invalid JSON: expected value at line 1 column 1 (got: '```json')") is None
    assert schema_constraint("company: Field required (got: {'title': 'x'})") == "required_field"
    assert schema_constraint("no separator here") is None


def test_constraint_violations_are_counted_per_rejected_reply_in_schema_mode_runs() -> None:
    dupes = [{"name": "A", "required": True}, {"name": "a", "required": True}]
    logs = [
        line("r1", 1, ok=False, error=error_text(title="x", seniority="wizard", skills=[]), setup="D"),
        line("r1", 2, ok=False, error=error_text(min_years_experience=31, bogus=1), setup="D"),
        line("r1", 3, setup="D"),
        line("r2", 1, ok=False, error=error_text(skills=dupes), setup="D"),  # custom rule: not a schema constraint
        line("r2", 2, ok=False, error_type="JSONDecodeError", error="(root): Invalid JSON: expected value (got: 'x')", setup="D"),
        line("r3", 1, ok=False, error=error_text(title="x"), setup="D"),
    ]
    docs = [doc("r1"), doc("r2", "invalid"), doc("r3", "invalid")]
    r = build_report(
        model="m", setup="D", max_attempts=3, docs=docs, logs=logs, labels=Labels(by_doc={}, skipped=[]), run_date="d"
    )
    assert r.constraint_violations is not None
    assert r.constraint_violations.n_outputs == 3  # r1 attempt 1, r1 attempt 2, r3 attempt 1
    assert r.constraint_violations.by_constraint == {
        "string_length": 2, "enum": 1, "list_size": 1, "number_bound": 1, "extra_field": 1,
    }
    # most frequent first, ties alphabetical
    detail = "string_length 2, enum 1, extra_field 1, list_size 1, number_bound 1"
    assert f"setup D: constraint violations under schema mode: 3 ({detail})" in report_notes(r)


def test_constraint_violations_are_reported_even_when_zero_and_never_outside_schema_mode() -> None:
    clean = build_report(
        model="m", setup="C", max_attempts=1, docs=[doc("r1")], logs=[line("r1", setup="C")],
        labels=Labels(by_doc={}, skipped=[]), run_date="d",
    )
    assert clean.constraint_violations is not None and clean.constraint_violations.n_outputs == 0
    assert "setup C: constraint violations under schema mode: 0" in report_notes(clean)
    assert report().constraint_violations is None  # setup B
    assert not any("constraint violations" in note for note in report_notes(report()))


# --- plan and notes -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("A", ["A"]), ("b", ["B"]), (" d ", ["D"]), ("all", ["A", "B", "C", "D"]), ("ALL", ["A", "B", "C", "D"])],
)
def test_parse_setups(value: str, expected: list[str]) -> None:
    assert parse_setups(value) == expected


@pytest.mark.parametrize("value", ["", "E", "AB", "A,B", "none"])
def test_parse_setups_rejects_anything_else(value: str) -> None:
    with pytest.raises(ValueError, match="unknown setup"):
        parse_setups(value)


def test_call_range_is_docs_times_attempts_per_setup(tmp_path: Path) -> None:
    jds = tmp_path / "jds"
    jds.mkdir()
    for i in range(4):
        (jds / f"jd_{i}.txt").write_text("x", encoding="utf-8")
    assert plan_eval("m", ["A"], jds).call_range() == (4, 4)
    assert plan_eval("m", ["B"], jds).call_range() == (4, 12)
    assert plan_eval("m", ["A", "B", "C", "D"], jds).call_range() == (16, 32)
    assert plan_eval("m", ["D"], jds, limit=2).call_range() == (2, 6)


def test_plan_counts_label_files_and_defaults_to_the_sibling_folder(tmp_path: Path) -> None:
    (tmp_path / "jds").mkdir()
    (tmp_path / "jds" / "jd_1.txt").write_text("x", encoding="utf-8")
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "jd_1.json").write_text("{}", encoding="utf-8")
    plan = plan_eval("m", ["B"], tmp_path / "jds")
    assert (plan.n_labels, plan.labels_dir) == (1, tmp_path / "labels")
    assert plan_eval("m", ["B"], tmp_path / "jds", tmp_path / "elsewhere").n_labels == 0


def test_report_notes_only_mention_what_happened() -> None:
    assert report_notes(report([doc("r1")], [line("r1")])) == []
    notes = report_notes(report())  # 1 doc without a response, 4 API error lines
    assert notes == [
        "setup B: 4 API error line(s) in the log (not counted as validation attempts)",
        "setup B: 1 doc(s) got no model response and are left out of the validity numbers",
    ]


# --- temperature and routed providers ---------------------------------------


def _routed(run_id: str, provider: str | None, temperature: float | None) -> AttemptLog:
    base = line(run_id)
    return base.model_copy(update={"provider": provider, "temperature": temperature})


def test_report_records_providers_and_the_temperature_sent() -> None:
    docs = [doc("r1"), doc("r2"), doc("r3")]
    logs = [_routed("r1", "Together", 0.0), _routed("r2", "Together", 0.0), _routed("r3", None, 0.0)]
    r = report(docs, logs)
    assert r.providers == {"Together": 2, "(not reported)": 1}
    assert r.temperature == 0.0


def test_report_temperature_is_none_when_not_sent_or_mixed() -> None:
    docs = [doc("r1"), doc("r2")]
    assert report(docs, [line("r1"), line("r2")]).temperature is None
    mixed = [_routed("r1", "A", 0.0), _routed("r2", "A", 0.7)]
    assert report(docs, mixed).temperature is None

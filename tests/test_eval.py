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
    failure_breakdown,
    first_error_loc,
    load_labels,
    parse_setups,
    plan_eval,
    report_notes,
    run_eval,
    score_fields,
    setup_args,
    validity_stats,
)
from soa.llm import LLMAPIError, LLMResponse, Message
from soa.logger import AttemptLog, Setup, read_log
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
) -> AttemptLog:
    if not ok:
        error_type = error_type or "ValidationError"
        error = error or "seniority: Input should be 'intern', 'junior', 'mid', 'senior' or 'lead' (got: 'x')"
    return AttemptLog(
        run_id=run_id, doc_id=f"doc-{run_id}", attempt=attempt, model="m", setup=setup, json_mode=False, ok=ok,
        error_type=error_type, error=error, raw_output="{}", latency_ms=latency, prompt_tokens=100,
        completion_tokens=20, cost_usd=cost, cost_known=known,
    )


def api_line(run_id: str, attempt: int = 1) -> AttemptLog:
    """What agent._log_api_error writes: no output, no tokens, cost 0.0 and cost_known False."""
    return AttemptLog(
        run_id=run_id, doc_id=f"doc-{run_id}", attempt=attempt, model="m", setup="B", json_mode=False, ok=False,
        error_type="APIError", error="RateLimitError: slow down", raw_output="", latency_ms=50, prompt_tokens=0,
        completion_tokens=0, cost_usd=0.0, cost_known=False,
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
        latency_ms=500, model="m",
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
        if isinstance(item, LLMAPIError):
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

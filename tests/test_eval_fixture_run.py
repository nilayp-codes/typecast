"""One whole eval run over fixture JDs, fixture labels and a scripted model; every number below is hand-computed.

Documents (setup B, three attempts, llm.call mocked):
  jd_001  valid first try; the model leaves the location empty, the header tool fills it (raw != normalized)
  jd_002  seniority 'wizard', then valid on the retry; its gold label says senior (the model said mid)
  jd_003  three invalid replies -> ExtractionFailed; labelled, but there is no posting to score
  jd_004  a non-retryable API error: no model response at all
  jd_005  a fenced reply (JSONDecodeError), then valid; its label file is still a TODO template
  jd_006  valid first try, but the model says Mumbai where the header says Pune (a tool disagreement)
"""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from soa import agent, llm
from soa.cli import app
from soa.eval import EvalReport, run_eval, summary_row, write_results
from soa.llm import LLMAPIError, LLMResponse, Message

HEADER = "Backend Engineer\nAcme\nPune, Maharashtra, India\nHybrid\nFull-time\nAbout the job\n"
PUNE: dict[str, Any] = {"city": "Pune", "country": "India", "mode": "hybrid"}


def posting(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": "Backend Engineer", "company": "Acme", "seniority": "mid", "min_years_experience": 3,
        "skills": [{"name": "PostgreSQL", "required": True}, {"name": "Python", "required": True}],
        "location": PUNE, "salary_mentioned": False,
    }
    return {**base, **overrides}


def reply(obj: dict[str, Any] | str) -> LLMResponse:
    text = obj if isinstance(obj, str) else json.dumps(obj)
    return LLMResponse(
        text=text, prompt_tokens=400, completion_tokens=100, cost_usd=0.002, cost_known=True, latency_ms=500, model="m"
    )


def skills(*names: str) -> list[dict[str, Any]]:
    return [{"name": n, "required": True} for n in names]


BAD = reply(posting(seniority="wizard"))
API_DOWN = LLMAPIError("AuthenticationError: bad key", model="m", retryable=False, status_code=401)

MODEL_SCRIPTS: dict[str, list[LLMResponse | LLMAPIError]] = {
    "jd_001": [reply(posting(
        skills=skills("postgres", "Python"), location={"city": None, "country": None, "mode": "unspecified"},
    ))],
    "jd_002": [BAD, reply(posting())],
    "jd_003": [BAD, BAD, BAD],
    "jd_004": [API_DOWN],
    "jd_005": [reply("```json\n" + json.dumps(posting()) + "\n```"), reply(posting())],
    "jd_006": [reply(posting(
        title="backend engineer", company=None, skills=skills("Python", "Go"), location={**PUNE, "city": "Mumbai"},
    ))],
}
LABELS: dict[str, dict[str, Any]] = {
    "jd_001": posting(),
    "jd_002": posting(seniority="senior"),
    "jd_003": posting(),
    "jd_005": {**posting(), "title": "TODO"},
    "jd_006": posting(skills=skills("Python", "SQL")),
}


def build_world(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    jds, labels = root / "data" / "jds", root / "data" / "labels"
    jds.mkdir(parents=True)
    labels.mkdir()
    for name in MODEL_SCRIPTS:
        (jds / f"{name}.txt").write_text(f"{HEADER}ref {name}\n", encoding="utf-8")
    for name, label in LABELS.items():
        (labels / f"{name}.json").write_text(json.dumps(label), encoding="utf-8")
    queues = {name: list(items) for name, items in MODEL_SCRIPTS.items()}

    def fake(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
        name = next(n for n in queues if f"ref {n}" in messages[1].content)
        item = queues[name].pop(0)
        if isinstance(item, LLMAPIError):
            raise item
        return item

    monkeypatch.setattr(llm, "call", fake)
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    return jds, labels


def fixture_run(root: Path, monkeypatch: pytest.MonkeyPatch) -> EvalReport:
    jds, labels = build_world(root, monkeypatch)
    return run_eval("m", "B", jds, labels, log_path=root / "attempts.jsonl", run_date="2026-10-17")


def test_validity_cost_and_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = fixture_run(tmp_path, monkeypatch)
    assert [d.status for d in r.docs] == ["valid", "valid", "invalid", "api_error", "valid", "valid"]
    assert (r.n, r.n_scored, r.n_api_failed) == (6, 5, 1)
    assert r.first_try_valid_pct == pytest.approx(40.0)  # jd_001, jd_006 of 5
    assert r.final_valid_pct == pytest.approx(80.0)  # all but jd_003
    assert r.avg_calls == pytest.approx(9 / 5)  # 1 + 2 + 3 + 2 + 1
    assert r.total_cost_usd == pytest.approx(0.018)  # 9 responses x 0.002
    assert r.avg_latency_ms == pytest.approx(900.0)
    assert (r.api_error_count, r.disagreement_count) == (1, 1)
    assert (r.n_labelled, r.n_labelled_with_output, r.labels_skipped) == (4, 3, ["jd_005"])


def test_failure_breakdown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = fixture_run(tmp_path, monkeypatch)
    assert r.failures.by_error_type == {"ValidationError": 4, "JSONDecodeError": 1}
    assert r.failures.by_first_loc == {"seniority": 4, "(root)": 1}


def test_raw_field_accuracy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    acc = fixture_run(tmp_path, monkeypatch).accuracy_raw
    assert acc is not None and acc.n == 3  # jd_001, jd_002, jd_006
    third, two_thirds = 100 / 3, 200 / 3
    assert (acc.title, acc.company, acc.seniority) == (100.0, pytest.approx(two_thirds), pytest.approx(two_thirds))
    assert (acc.mode, acc.salary_mentioned, acc.min_years_experience) == (pytest.approx(two_thirds), 100.0, 100.0)
    assert (acc.city, acc.country) == (pytest.approx(third), pytest.approx(two_thirds))
    assert acc.skills_precision == pytest.approx(two_thirds)  # tp 4, fp 2, fn 2
    assert acc.skills_recall == pytest.approx(two_thirds) and acc.skills_f1 == pytest.approx(two_thirds)
    assert acc.overall == pytest.approx(100 * 6.666667 / 9, rel=1e-5)


def test_normalized_field_accuracy_is_better_where_the_tools_helped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = fixture_run(tmp_path, monkeypatch)
    raw, norm = r.accuracy_raw, r.accuracy_normalized
    assert raw is not None and norm is not None
    assert (norm.mode, norm.country) == (100.0, 100.0)
    assert norm.city == pytest.approx(200 / 3)  # jd_006 stays Mumbai: the model value is kept
    assert norm.skills_precision == pytest.approx(500 / 6) and norm.skills_f1 == pytest.approx(500 / 6)  # tp 5, fp 1, fn 1
    assert norm.overall == pytest.approx(783.3333333 / 9, rel=1e-6)
    assert norm.overall > raw.overall
    assert (norm.title, norm.company, norm.seniority) == (raw.title, raw.company, raw.seniority)  # tools don't touch these


def test_every_line_in_the_log_is_tagged_with_the_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_run(tmp_path, monkeypatch)
    rows = [json.loads(x) for x in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 10  # 9 model responses + 1 APIError line
    assert {(x["setup"], x["json_mode"]) for x in rows} == {("B", False)}
    assert sum(x["error_type"] == "APIError" for x in rows) == 1


def test_the_saved_report_round_trips_and_the_row_is_honest_about_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r = fixture_run(tmp_path, monkeypatch)
    path = write_results(r, tmp_path / "results")
    assert EvalReport.model_validate_json(path.read_text(encoding="utf-8")) == r
    assert summary_row(r) == "| m | B | 6 (5 scored) | 40.0% | 80.0% | 1.80 | 87.0% | $0.0180 |\n"


def test_the_same_run_through_the_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    jds, _ = build_world(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOA_LOG_PATH", str(tmp_path / "attempts.jsonl"))
    result = CliRunner().invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--yes"])
    assert result.exit_code == 0, result.output
    out = result.stdout_bytes.decode("utf-8").replace("\r\n", "\n")
    assert "| m | B | 6 (5 scored) | 40.0% | 80.0% | 1.80 | 87.0% | $0.0180 |" in out
    assert "estimated model calls: 6 to 18 (real API calls)" in out
    assert "labels skipped (invalid or still TODO): jd_005" in out

"""extract(): retry loop, error feedback, API backoff, totals. llm.call is mocked as a response sequence (rule 13)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from soa import agent, llm
from soa.agent import RETRY_PROMPT_TEMPLATE, ExtractionFailed, extract
from soa.llm import LLMAPIError, LLMResponse, Message
from soa.logger import AttemptLog
from soa.schemas import JobPosting

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": None,
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}],
    "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
    "salary_mentioned": False,
}
GOOD = json.dumps(VALID)
BAD_ENUM = json.dumps({**VALID, "seniority": "wizard"})
BAD_TYPE = json.dumps({**VALID, "min_years_experience": "3"})
NOT_JSON = "Sure! Here is the JSON you asked for."


def resp(
    text: str, *, prompt: int = 100, completion: int = 40, cost: float = 0.002, latency: int = 850, known: bool = True
) -> LLMResponse:
    return LLMResponse(
        text=text, prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost,
        cost_known=known, cost_source="provider" if known else "none", latency_ms=latency, model="m",
    )


def api_error(*, retryable: bool = True, status: int = 429, message: str = "RateLimitError: slow down") -> LLMAPIError:
    return LLMAPIError(message, model="m", retryable=retryable, status_code=status)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep so no test waits; returns the waits that were requested."""
    waits: list[float] = []
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: waits.append(seconds))
    return waits


def script(monkeypatch: pytest.MonkeyPatch, items: list[LLMResponse | LLMAPIError]) -> list[list[Message]]:
    """llm.call returns (or raises) `items` in order. Returns the messages each call received."""
    queue = list(items)
    calls: list[list[Message]] = []

    def fake(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
        calls.append(list(messages))
        item = queue.pop(0)
        if isinstance(item, LLMAPIError):
            raise item
        return item

    monkeypatch.setattr(llm, "call", fake)
    return calls


def read_log(path: Path) -> list[AttemptLog]:
    return [AttemptLog.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]


def run(log: Path, max_attempts: int = 3) -> agent.ExtractionResult:
    return extract("some jd", "m", max_attempts, doc_id="jd_007", run_id="run1", log_path=log)


# --- native (schema) mode -------------------------------------------------


@pytest.mark.parametrize(("json_mode", "expected"), [(True, JobPosting), (False, None)])
def test_json_mode_selects_response_model_on_every_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float],
    json_mode: bool, expected: type[BaseModel] | None,
) -> None:
    queue = [resp(BAD_ENUM), resp(GOOD)]
    seen: list[type[BaseModel] | None] = []

    def fake(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
        seen.append(response_model)
        return queue.pop(0)

    monkeypatch.setattr(llm, "call", fake)
    result = extract("some jd", "m", 3, json_mode, doc_id="jd_007", run_id="run1", log_path=tmp_path / "a.jsonl")
    assert result.attempts == 2
    assert seen == [expected, expected]  # the retry stays in the same mode


def test_setup_and_json_mode_are_written_to_every_log_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    script(monkeypatch, [api_error(), resp(BAD_ENUM), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    extract("some jd", "m", 3, True, doc_id="jd_007", run_id="run1", log_path=log, setup="D")
    rows = read_log(log)
    assert [r.error_type for r in rows] == ["APIError", "ValidationError", None]  # API error line included
    assert {(r.setup, r.json_mode) for r in rows} == {("D", True)}


def test_setup_defaults_to_none_and_json_mode_to_false_in_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    script(monkeypatch, [resp(GOOD)])
    log = tmp_path / "a.jsonl"
    run(log)
    assert {(r.setup, r.json_mode) for r in read_log(log)} == {(None, False)}


def test_schema_mode_rejection_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]) -> None:
    calls = script(monkeypatch, [api_error(retryable=False, status=400, message="schema mode unsupported by m")])
    log = tmp_path / "a.jsonl"
    with pytest.raises(LLMAPIError, match="schema mode unsupported by m"):
        extract("some jd", "m", 3, True, doc_id="jd_007", run_id="run1", log_path=log)
    assert len(calls) == 1 and sleeps == []  # no backoff, no second call
    assert [x.error_type for x in read_log(log)] == ["APIError"]


# --- validation retries ---------------------------------------------------


def test_valid_first_try(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]) -> None:
    calls = script(monkeypatch, [resp(GOOD)])
    log = tmp_path / "a.jsonl"
    result = run(log)
    assert result.attempts == 1
    assert result.api_retries == 0
    assert result.posting.title == "Backend Engineer"
    assert len(calls) == 1
    assert [x.ok for x in read_log(log)] == [True]
    assert sleeps == []


def test_invalid_then_valid_feeds_first_error_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(BAD_ENUM), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    result = run(log)
    assert result.attempts == 2
    first = read_log(log)[0]
    assert first.ok is False and first.error is not None and "seniority" in first.error
    assert [m.role for m in calls[0]] == ["system", "user"]
    assert [m.role for m in calls[1]] == ["system", "user", "assistant", "user"]
    assert calls[1][2].content == BAD_ENUM  # the rejected reply, verbatim
    assert first.error in calls[1][3].content  # the first error text, verbatim
    assert calls[1][3].content == RETRY_PROMPT_TEMPLATE.format(errors=first.error)
    assert "Keep every field that had no error unchanged." in calls[1][3].content
    assert [x.attempt for x in read_log(log)] == [1, 2]


def test_non_json_reply_is_retried_with_its_decode_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(NOT_JSON), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    assert run(log).attempts == 2
    first = read_log(log)[0]
    assert first.error_type == "JSONDecodeError"
    assert first.error is not None and first.error in calls[1][3].content


def test_invalid_three_times_raises_extraction_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(BAD_ENUM), resp(BAD_TYPE), resp(NOT_JSON)])
    log = tmp_path / "a.jsonl"
    with pytest.raises(ExtractionFailed) as caught:
        run(log)
    lines = read_log(log)
    assert len(lines) == 3
    assert all(not x.ok for x in lines)
    assert [x.attempt for x in lines] == [1, 2, 3]
    assert len(calls) == 3
    assert len(calls[2]) == 6  # system, user, then two (assistant, user) pairs
    assert lines[0].error is not None and lines[1].error is not None
    assert lines[0].error in calls[2][3].content and lines[1].error in calls[2][5].content
    assert caught.value.doc_id == "jd_007"
    assert len(caught.value.attempts) == 3
    assert caught.value.last_error == lines[-1].error
    assert caught.value.last_error in str(caught.value)


def test_max_attempts_one_is_the_no_retry_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(BAD_ENUM), resp(GOOD)])  # the valid reply must never be requested
    log = tmp_path / "a.jsonl"
    with pytest.raises(ExtractionFailed) as caught:
        run(log, max_attempts=1)
    assert len(calls) == 1
    assert [m.role for m in calls[0]] == ["system", "user"]  # no retry message was built or sent
    lines = read_log(log)
    assert len(lines) == 1 and lines[0].ok is False and lines[0].attempt == 1
    assert len(caught.value.attempts) == 1
    assert caught.value.last_error == lines[0].error
    assert sleeps == []


def test_max_attempts_below_one_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = script(monkeypatch, [])
    with pytest.raises(ValueError):
        run(tmp_path / "a.jsonl", max_attempts=0)
    assert calls == []


# --- post-processing ------------------------------------------------------

JD_HEADER = "Backend Engineer\nAcme\nBengaluru, Karnataka, India\nOn-site\nFull-time\nAbout the job\nPython and k8s.\n"
RAW = json.dumps({
    **VALID,
    "skills": [{"name": "postgres", "required": False}, {"name": "PostgreSQL", "required": True}],
    "location": {"city": None, "country": None, "mode": "unspecified"},
})


def run_text(log: Path, text: str, **kwargs: Any) -> agent.ExtractionResult:
    return extract(text, "m", doc_id="jd_007", run_id="run1", log_path=log, **kwargs)


def test_extract_returns_raw_and_normalized_postings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(RAW)])
    result = run_text(tmp_path / "a.jsonl", JD_HEADER)
    assert [(s.name, s.required) for s in result.raw_posting.skills] == [("postgres", False), ("PostgreSQL", True)]
    assert [(s.name, s.required) for s in result.posting.skills] == [("PostgreSQL", True)]
    assert result.raw_posting.location.city is None
    assert (result.posting.location.city, result.posting.location.country) == ("Bengaluru", "India")
    assert result.posting.location.mode.value == "onsite"
    assert len(calls) == 1  # post-processing makes no LLM call
    assert result.attempts == 1
    assert result.disagreements == []  # the model gave null: that is a fill, not a disagreement


def test_extract_reports_tool_disagreements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    pune = json.dumps({**VALID, "location": {"city": "Pune", "country": "India", "mode": "hybrid"}})
    script(monkeypatch, [resp(pune), resp(pune)])
    result = run_text(tmp_path / "a.jsonl", JD_HEADER)
    assert [(d.field, d.model_value, d.tool_value) for d in result.disagreements] == [
        ("location.city", "Pune", "Bengaluru"),
        ("location.mode", "hybrid", "onsite"),
    ]
    assert result.posting.location.city == "Pune"  # the model's value is kept
    off = run_text(tmp_path / "b.jsonl", JD_HEADER, normalize=False)
    assert off.disagreements == []


def test_extract_normalize_false_returns_the_model_output_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    script(monkeypatch, [resp(RAW)])
    result = run_text(tmp_path / "a.jsonl", JD_HEADER, normalize=False)
    assert result.posting == result.raw_posting
    assert [s.name for s in result.posting.skills] == ["postgres", "PostgreSQL"]


def test_extract_raises_tool_response_error_when_postprocessing_breaks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    from soa import tools

    script(monkeypatch, [resp(RAW)])
    monkeypatch.setattr(tools, "merge_skills", lambda skills: [])
    with pytest.raises(tools.ToolResponseError):
        run_text(tmp_path / "a.jsonl", JD_HEADER)


# --- API errors -----------------------------------------------------------


def test_retryable_api_error_then_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [api_error(), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    result = run(log)
    assert result.attempts == 1  # the API error is not a validation attempt
    assert result.api_retries == 1
    assert sleeps == [2.0]
    assert len(calls) == 2 and calls[0] == calls[1]
    lines = read_log(log)
    assert [(x.error_type, x.ok, x.attempt) for x in lines] == [("APIError", False, 1), (None, True, 1)]
    assert lines[0].raw_output == "" and lines[0].prompt_tokens == 0 and lines[0].cost_known is False
    assert lines[0].error is not None and "slow down" in lines[0].error


def test_backoff_is_exponential_and_exhaustion_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [api_error(), api_error(), api_error(), api_error()])
    log = tmp_path / "a.jsonl"
    with pytest.raises(LLMAPIError):
        run(log)
    assert sleeps == [2.0, 4.0, 8.0]
    assert len(calls) == 4  # first call + 3 retries
    assert [x.error_type for x in read_log(log)] == ["APIError"] * 4


def test_non_retryable_api_error_is_logged_and_raised_at_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [api_error(retryable=False, status=401, message="AuthenticationError: bad key")])
    log = tmp_path / "a.jsonl"
    with pytest.raises(LLMAPIError) as caught:
        run(log)
    assert caught.value.retryable is False
    assert sleeps == []
    assert len(calls) == 1
    lines = read_log(log)
    assert len(lines) == 1 and lines[0].error_type == "APIError"


def test_api_error_between_validation_attempts_does_not_consume_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(BAD_ENUM), api_error(), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    result = run(log, max_attempts=2)
    assert result.attempts == 2
    assert result.api_retries == 1
    assert [(x.error_type, x.attempt) for x in read_log(log)] == [("ValidationError", 1), ("APIError", 2), (None, 2)]
    assert calls[1] == calls[2]  # the retry re-sends the same conversation


def test_api_key_values_are_kept_out_of_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-supersecretvalue")
    script(monkeypatch, [api_error(retryable=False, status=401, message="AuthenticationError: bad sk-or-v1-supersecretvalue")])
    log = tmp_path / "a.jsonl"
    with pytest.raises(LLMAPIError):
        run(log)
    text = log.read_text(encoding="utf-8")
    assert "supersecretvalue" not in text and "***" in text


# --- totals ---------------------------------------------------------------


def test_tokens_cost_and_latency_are_summed_over_all_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    script(monkeypatch, [
        resp(BAD_ENUM, prompt=100, completion=40, cost=0.002, latency=800),
        resp(BAD_TYPE, prompt=200, completion=50, cost=0.003, latency=900),
        resp(GOOD, prompt=300, completion=60, cost=0.004, latency=1000),
    ])
    result = run(tmp_path / "a.jsonl")
    assert result.attempts == 3
    assert result.total_tokens == 100 + 40 + 200 + 50 + 300 + 60
    assert result.cost_usd == pytest.approx(0.009)
    assert result.latency_ms == 800 + 900 + 1000
    assert result.cost_known is True


def test_cost_known_is_false_if_any_call_cost_is_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    script(monkeypatch, [resp(BAD_ENUM, known=False, cost=0.0), resp(GOOD)])
    result = run(tmp_path / "a.jsonl")
    assert result.cost_known is False
    assert result.cost_usd == pytest.approx(0.002)


# --- count-aware retry feedback --------------------------------------------


def _too_many_skills(count: int) -> str:
    skills = [{"name": f"Skill{i}", "required": i < 5} for i in range(count)]
    return json.dumps({**VALID, "skills": skills})


def test_skills_over_cap_retry_line_states_count_and_fix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(_too_many_skills(39)), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    run(log)
    feedback = calls[1][3].content
    assert "skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones" in feedback
    assert "(got:" not in feedback.split("skills: has 39 items")[1].splitlines()[0]
    first = read_log(log)[0]  # the logged error keeps pydantic's wording (eval.py classifies it)
    assert first.error is not None and first.error.startswith("skills: List should have at most 30 items")


def test_other_errors_keep_the_plain_retry_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sleeps: list[float]
) -> None:
    calls = script(monkeypatch, [resp(BAD_ENUM), resp(GOOD)])
    log = tmp_path / "a.jsonl"
    run(log)
    first = read_log(log)[0]
    assert first.error is not None and first.error in calls[1][3].content
    assert "remove" not in calls[1][3].content

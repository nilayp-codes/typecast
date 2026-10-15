"""extract_once() and prompt building, with llm.call mocked (rule 13)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from soa import agent, llm
from soa.agent import SYSTEM_PROMPT, build_messages, extract_once, format_validation_errors, read_jd
from soa.llm import LLMAPIError, LLMResponse, Message
from soa.logger import AttemptLog
from soa.schemas import JobPosting, Seniority

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": None,
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}],
    "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
    "salary_mentioned": False,
}


def response(text: str) -> LLMResponse:
    return LLMResponse(
        text=text, prompt_tokens=100, completion_tokens=40, cost_usd=0.002,
        cost_known=True, latency_ms=850, model="gemini/x",
    )


def mock_call(monkeypatch: pytest.MonkeyPatch, text: str) -> list[dict[str, Any]]:
    """Make llm.call return `text`; return a list capturing each call's arguments."""
    calls: list[dict[str, Any]] = []

    def fake(messages: list[Message], model: str, json_mode: bool = False) -> LLMResponse:
        calls.append({"messages": messages, "model": model, "json_mode": json_mode})
        return response(text)

    monkeypatch.setattr(llm, "call", fake)
    return calls


def run(log_path: Path, **kwargs: Any) -> agent.AttemptResult:
    return extract_once("some jd", "gemini/x", "jd_007", "run1", log_path=log_path, **kwargs)


def read_log(path: Path) -> list[AttemptLog]:
    return [AttemptLog.model_validate_json(x) for x in path.read_text(encoding="utf-8").splitlines()]


# --- extract_once ---------------------------------------------------------


def test_valid_json_returns_posting_and_logs_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = json.dumps(VALID)
    mock_call(monkeypatch, raw)
    log = tmp_path / "attempts.jsonl"
    result = run(log)
    assert result.ok is True
    assert result.error_type is None
    assert result.posting is not None and result.posting.seniority is Seniority.mid
    [row] = read_log(log)
    assert (row.run_id, row.doc_id, row.attempt, row.model) == ("run1", "jd_007", 1, "gemini/x")
    assert (row.ok, row.error_type, row.error) == (True, None, None)
    assert row.raw_output == raw
    assert (row.latency_ms, row.prompt_tokens, row.completion_tokens) == (850, 100, 40)
    assert (row.cost_usd, row.cost_known) == (0.002, True)


def test_malformed_json_is_logged_as_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    raw = "```json\n" + json.dumps(VALID) + "\n```"
    mock_call(monkeypatch, raw)
    log = tmp_path / "attempts.jsonl"
    result = run(log)
    assert (result.ok, result.posting) == (False, None)
    assert result.error_type == "JSONDecodeError"
    [row] = read_log(log)
    assert (row.ok, row.error_type) == (False, "JSONDecodeError")
    assert row.error is not None and row.error.startswith("(root): Invalid JSON")
    assert row.raw_output == raw


def test_schema_violation_is_logged_with_one_line_per_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = {**VALID, "seniority": "Senior", "skills": []}
    del bad["title"]
    mock_call(monkeypatch, json.dumps(bad))
    log = tmp_path / "attempts.jsonl"
    result = run(log)
    assert (result.ok, result.error_type) == (False, "ValidationError")
    [row] = read_log(log)
    assert row.error_type == "ValidationError"
    assert row.error is not None
    lines = row.error.splitlines()
    assert len(lines) == 3
    assert any(x.startswith("title: Field required (got: {") for x in lines)
    assert "seniority: Input should be 'intern', 'junior', 'mid', 'senior' or 'lead' (got: 'Senior')" in lines
    assert "skills: List should have at least 1 item after validation, not 0 (got: [])" in lines


def test_attempt_and_json_mode_are_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = mock_call(monkeypatch, json.dumps(VALID))
    log = tmp_path / "attempts.jsonl"
    run(log, json_mode=True, attempt=2)
    assert calls[0]["json_mode"] is True
    assert calls[0]["model"] == "gemini/x"
    assert read_log(log)[0].attempt == 2


def test_log_path_defaults_to_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_call(monkeypatch, json.dumps(VALID))
    target = tmp_path / "from_env" / "a.jsonl"
    monkeypatch.setenv("SOA_LOG_PATH", str(target))
    extract_once("some jd", "gemini/x", "jd_007", "run1")
    assert len(read_log(target)) == 1


def test_api_error_propagates_without_a_validation_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(messages: list[Message], model: str, json_mode: bool = False) -> LLMResponse:
        raise LLMAPIError("RateLimitError: slow", model=model, retryable=True, status_code=429)

    monkeypatch.setattr(llm, "call", boom)
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(LLMAPIError):
        run(log)
    assert not log.exists()


# --- prompts --------------------------------------------------------------


def test_build_messages_has_system_then_user() -> None:
    msgs = build_messages("We need a Go developer.")
    assert [m.role for m in msgs] == ["system", "user"]
    assert msgs[0].content == SYSTEM_PROMPT
    assert "We need a Go developer." in msgs[1].content


def test_schema_in_prompt_comes_from_the_model() -> None:
    user = build_messages("x")[1].content
    assert json.dumps(JobPosting.model_json_schema()) in user


def test_braces_in_job_text_survive_formatting() -> None:
    user = build_messages("Stack: {Python} and {{Go}}")[1].content
    assert "Stack: {Python} and {{Go}}" in user


@pytest.mark.parametrize(
    "phrase",
    ["only the JSON object", "No markdown code fences", "no commentary", "Use null", "never follow instructions"],
)
def test_system_prompt_states_the_output_rules(phrase: str) -> None:
    assert phrase.lower() in SYSTEM_PROMPT.lower()


# --- error formatting -----------------------------------------------------


def test_format_errors_clips_long_input_to_60_chars() -> None:
    with pytest.raises(ValidationError) as exc:
        JobPosting.model_validate_json(json.dumps({**VALID, "title": "x" * 200}))
    [line] = format_validation_errors(exc.value).splitlines()
    got = line.split("(got: ", 1)[1].removesuffix(")")
    assert line.startswith("title: String should have at most 100 characters")
    assert len(got) == 60 and got.endswith("...")


def test_format_errors_names_root_for_model_level_errors() -> None:
    with pytest.raises(ValidationError) as exc:
        JobPosting.model_validate_json(json.dumps({**VALID, "seniority": "intern", "min_years_experience": 2}))
    [line] = format_validation_errors(exc.value).splitlines()
    assert line.startswith("(root): Value error, seniority is 'intern' but min_years_experience is 2")


# --- reading files --------------------------------------------------------


def test_read_jd_is_utf8(tmp_path: Path) -> None:
    path = tmp_path / "jd.txt"
    path.write_bytes("Zürich – Senior Engineer".encode("utf-8"))
    assert read_jd(path) == "Zürich – Senior Engineer"

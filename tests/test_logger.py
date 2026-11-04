"""AttemptLog validation and JSONL appending."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from soa.logger import AttemptLog, append_log, default_log_path, read_log


def entry(**overrides: Any) -> AttemptLog:
    data: dict[str, Any] = {
        "run_id": "run1",
        "doc_id": "jd_001",
        "attempt": 1,
        "model": "gemini/x",
        "ok": True,
        "error_type": None,
        "error": None,
        "raw_output": "{}",
        "latency_ms": 120,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cost_usd": 0.0001,
        "cost_known": True,
    }
    data.update(overrides)
    return AttemptLog.model_validate(data)


FAILED = {"ok": False, "error_type": "ValidationError", "error": "title: Field required (got: None)"}


def test_append_writes_one_json_line(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    append_log(entry(), path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["doc_id"] == "jd_001"
    assert row["ok"] is True
    assert set(row) == {
        "run_id", "doc_id", "attempt", "model", "setup", "json_mode", "ok", "error_type", "error", "raw_output",
        "latency_ms", "prompt_tokens", "completion_tokens", "cost_usd", "cost_known", "cost_source", "temperature",
        "provider", "ts",
    }
    assert row["temperature"] is None and row["provider"] is None  # not sent / not reported
    assert (row["setup"], row["json_mode"]) == (None, False)  # non-eval run, plain prompt


@pytest.mark.parametrize("setup", ["A", "B", "C", "D", None])
def test_setup_and_json_mode_round_trip(tmp_path: Path, setup: str | None) -> None:
    path = tmp_path / "attempts.jsonl"
    append_log(entry(setup=setup, json_mode=True), path)
    [row] = read_log(path)
    assert (row.setup, row.json_mode) == (setup, True)


@pytest.mark.parametrize("bad", ["E", "b", "", "AB"])
def test_setup_must_be_a_to_d(bad: str) -> None:
    with pytest.raises(ValidationError):
        entry(setup=bad)


def test_json_mode_must_be_a_bool() -> None:
    with pytest.raises(ValidationError):
        entry(json_mode="yes")


def test_a_log_line_from_before_phase_6_still_parses() -> None:
    old = entry().model_dump(mode="json")
    del old["setup"], old["json_mode"], old["cost_source"]
    row = AttemptLog.model_validate_json(json.dumps(old))
    assert (row.setup, row.json_mode, row.cost_source) == (None, False, None)


@pytest.mark.parametrize("source", ["provider", "estimate", "none", None])
def test_cost_source_round_trips(source: str | None) -> None:
    assert AttemptLog.model_validate_json(entry(cost_source=source).model_dump_json()).cost_source == source


def test_cost_source_must_be_a_known_value() -> None:
    with pytest.raises(ValidationError):
        entry(cost_source="guess")


def test_read_log_returns_lines_in_order_and_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    append_log(entry(doc_id="jd_001"), path)
    append_log(entry(doc_id="jd_002", **FAILED), path)
    with path.open("a", encoding="utf-8") as f:
        f.write("\n")
    assert [r.doc_id for r in read_log(path)] == ["jd_001", "jd_002"]


def test_read_log_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_log(tmp_path / "nope.jsonl") == []


def test_read_log_raises_on_a_corrupt_line(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    append_log(entry(), path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"run_id": "x"}\n')
    with pytest.raises(ValidationError):
        read_log(path)


def test_append_adds_lines_and_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    first, second = entry(), entry(attempt=2, **FAILED)
    append_log(first, path)
    append_log(second, path)
    rows = [AttemptLog.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [first, second]


def test_append_creates_missing_parent_folders(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "logs" / "attempts.jsonl"
    append_log(entry(), path)
    assert path.is_file()


def test_append_is_utf8_with_lf_endings(tmp_path: Path) -> None:
    path = tmp_path / "attempts.jsonl"
    append_log(entry(raw_output="Zürich – 日本語"), path)
    raw = path.read_bytes()
    assert b"\r" not in raw
    assert raw.endswith(b"\n")
    assert AttemptLog.model_validate_json(raw.decode("utf-8")).raw_output == "Zürich – 日本語"


def test_ts_defaults_to_utc_now() -> None:
    before = datetime.now(UTC)
    assert before <= entry().ts <= datetime.now(UTC)


def test_naive_ts_is_rejected() -> None:
    with pytest.raises(ValidationError):
        entry(ts=datetime(2026, 10, 15))


def test_success_with_error_fails() -> None:
    with pytest.raises(ValidationError, match="ok=True requires error_type and error to be null"):
        entry(error_type="ValidationError")


@pytest.mark.parametrize("missing", [{"error": None}, {"error_type": None}])
def test_failure_without_error_fails(missing: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="ok=False requires both error_type and error"):
        entry(**{**FAILED, **missing})


def test_failure_with_error_passes() -> None:
    assert entry(**FAILED).error_type == "ValidationError"


def test_attempt_starts_at_one() -> None:
    with pytest.raises(ValidationError):
        entry(attempt=0)


def test_default_log_path_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOA_LOG_PATH", raising=False)
    assert default_log_path() == Path("logs/attempts.jsonl")
    monkeypatch.setenv("SOA_LOG_PATH", "elsewhere/a.jsonl")
    assert default_log_path() == Path("elsewhere/a.jsonl")

"""`soa extract` and `soa eval` through typer's CliRunner, with llm.call mocked (rule 13)."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from soa import agent, cli, llm
from soa.cli import app
from soa.llm import LLMAPIError, LLMResponse, Message

runner = CliRunner()

JD = "Backend Engineer\nAcme\nBengaluru, Karnataka, India\nOn-site\nFull-time\nAbout the job\nPython and Postgres.\n"
VALID: dict[str, Any] = {
    "title": "Développeur Backend",  # non-ASCII on purpose
    "company": "Acme",
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "postgres", "required": False}, {"name": "PostgreSQL", "required": True}],
    "location": {"city": None, "country": None, "mode": "unspecified"},
    "salary_mentioned": False,
}
GOOD = json.dumps(VALID, ensure_ascii=False)
BAD = json.dumps({**VALID, "seniority": "wizard"})


def resp(text: str, *, known: bool = True) -> LLMResponse:
    return LLMResponse(
        text=text, prompt_tokens=1000, completion_tokens=240, cost_usd=0.0009,
        cost_known=known, latency_ms=2100, model="m",
    )


def script(monkeypatch: pytest.MonkeyPatch, items: list[LLMResponse | LLMAPIError]) -> list[str]:
    """llm.call returns (or raises) `items` in order; returns the model string of each call."""
    queue = list(items)
    models: list[str] = []

    def fake(messages: list[Message], model: str, json_mode: bool = False) -> LLMResponse:
        models.append(model)
        item = queue.pop(0)
        if isinstance(item, LLMAPIError):
            raise item
        return item

    monkeypatch.setattr(llm, "call", fake)
    return models


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Run in an empty folder: no .env, a private log file, no real sleeping, no model/attempt defaults."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOA_LOG_PATH", str(tmp_path / "attempts.jsonl"))
    monkeypatch.delenv("SOA_MODEL_PRIMARY", raising=False)
    monkeypatch.delenv("SOA_MAX_ATTEMPTS", raising=False)
    monkeypatch.setattr(agent.time, "sleep", lambda seconds: None)
    return tmp_path


@pytest.fixture
def jd_file(isolated: Path) -> Path:
    path = isolated / "jd_007.txt"
    path.write_text(JD, encoding="utf-8")
    return path


def utf8(result: Any) -> str:
    """The raw stdout bytes decoded as UTF-8 (raises if invalid), with Windows CRLF newlines normalized."""
    decoded: str = result.stdout_bytes.decode("utf-8")
    return decoded.replace("\r\n", "\n")


def posting_json(output: str) -> dict[str, Any]:
    start = output.index("\n{\n") + 1
    parsed: dict[str, Any] = json.loads(output[start:])
    return parsed


# --- extract: success -----------------------------------------------------


def test_success_exits_0_with_header_and_pretty_json(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 0, result.output
    out = utf8(result)
    assert out.splitlines()[0] == "✓ Valid after 1 attempt  (1,240 tokens · $0.0009 · 2.1s)"
    assert '\n  "title": "Développeur Backend"' in out  # pretty-printed, UTF-8 kept
    data = posting_json(out)
    assert [s["name"] for s in data["skills"]] == ["PostgreSQL"]  # normalized: duplicates merged
    assert data["skills"][0]["required"] is True
    assert data["location"] == {"city": "Bengaluru", "country": "India", "mode": "onsite"}  # filled from the header


def test_retry_success_lists_each_failed_attempt(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(BAD), resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 0
    lines = utf8(result).splitlines()
    assert lines[0] == "✓ Valid after 2 attempts  (2,480 tokens · $0.0018 · 4.2s)"
    assert lines[1].startswith("  attempt 1 ✗ seniority: Input should be")


def test_tool_disagreements_are_shown(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    pune = json.dumps({**VALID, "location": {"city": "Pune", "country": "India", "mode": "hybrid"}})
    script(monkeypatch, [resp(pune)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    out = utf8(result)
    assert "  tool disagreement  location.city: model 'Pune', header tool 'Bengaluru' (model value kept)" in out
    assert posting_json(out)["location"]["city"] == "Pune"


def test_no_normalize_prints_the_raw_posting(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m", "--no-normalize"])
    assert result.exit_code == 0
    data = posting_json(utf8(result))
    assert [s["name"] for s in data["skills"]] == ["postgres", "PostgreSQL"]  # as the model wrote them
    assert data["location"]["city"] is None  # not filled from the header
    assert "tool disagreement" not in utf8(result)


def test_unknown_cost_is_shown_as_na(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD, known=False)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert utf8(result).splitlines()[0] == "✓ Valid after 1 attempt  (1,240 tokens · n/a · 2.1s)"


# --- extract: failures and exit codes -------------------------------------


def test_extraction_failed_exits_1_and_shows_every_attempt(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(BAD), resp("not json at all"), resp(BAD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 1
    lines = utf8(result).splitlines()
    assert lines[0] == "✗ Failed after 3 attempts  (3,720 tokens · $0.0027 · 6.3s)"
    attempts = [line for line in lines if line.startswith("  attempt ")]
    assert [line.split()[1] for line in attempts] == ["1", "2", "3"]
    assert len(lines) == 1 + 3  # header + one line per attempt: no posting JSON is printed
    assert all(not line.startswith("{") for line in lines)


def test_missing_file_exits_2(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    models = script(monkeypatch, [])
    result = runner.invoke(app, ["extract", str(isolated / "nope.txt"), "--model", "m"])
    assert result.exit_code == 2
    assert "file not found" in result.stderr
    assert models == []  # no LLM call was attempted


def test_empty_and_non_utf8_files_exit_2(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    script(monkeypatch, [])
    empty = isolated / "empty.txt"
    empty.write_text("  \n", encoding="utf-8")
    binary = isolated / "latin1.txt"
    binary.write_bytes("Développeur".encode("latin-1"))
    assert runner.invoke(app, ["extract", str(empty), "--model", "m"]).exit_code == 2
    result = runner.invoke(app, ["extract", str(binary), "--model", "m"])
    assert result.exit_code == 2 and "not valid UTF-8" in result.stderr


def test_utf8_bom_is_accepted(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    script(monkeypatch, [resp(GOOD)])
    path = isolated / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbf" + JD.encode("utf-8"))
    assert runner.invoke(app, ["extract", str(path), "--model", "m"]).exit_code == 0


def test_non_retryable_api_error_exits_2_and_never_prints_the_key(
    monkeypatch: pytest.MonkeyPatch, jd_file: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-supersecretvalue")
    err = LLMAPIError("AuthenticationError: bad key sk-or-v1-supersecretvalue", model="m", retryable=False, status_code=401)
    script(monkeypatch, [err])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 2
    assert "supersecretvalue" not in result.output and "supersecretvalue" not in result.stderr
    assert "API error from m" in result.stderr and "***" in result.stderr


def test_no_model_configured_exits_2(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    models = script(monkeypatch, [])
    result = runner.invoke(app, ["extract", str(jd_file)])
    assert result.exit_code == 2 and "SOA_MODEL_PRIMARY" in result.stderr
    assert models == []


# --- extract: defaults from the environment -------------------------------


def test_model_and_max_attempts_default_from_env(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    monkeypatch.setenv("SOA_MODEL_PRIMARY", "env/model")
    monkeypatch.setenv("SOA_MAX_ATTEMPTS", "1")
    models = script(monkeypatch, [resp(BAD), resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file)])
    assert result.exit_code == 1  # one attempt only
    assert models == ["env/model"]


def test_cli_flags_override_env(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    monkeypatch.setenv("SOA_MODEL_PRIMARY", "env/model")
    monkeypatch.setenv("SOA_MAX_ATTEMPTS", "1")
    models = script(monkeypatch, [resp(BAD), resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "flag/model", "--max-attempts", "2"])
    assert result.exit_code == 0
    assert models == ["flag/model", "flag/model"]


def test_bad_max_attempts_in_env_exits_2(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    monkeypatch.setenv("SOA_MAX_ATTEMPTS", "lots")
    script(monkeypatch, [])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 2 and "SOA_MAX_ATTEMPTS" in result.stderr


# --- console encoding -----------------------------------------------------


class NarrowStream:
    """A stdout stand-in that cannot be switched to UTF-8 (no `reconfigure`) and only knows cp1252."""

    encoding = "cp1252"

    def write(self, text: str) -> int:
        return len(text)


def test_glyphs_fall_back_to_ascii_when_the_stream_cannot_carry_them(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "stdout", NarrowStream())
    assert cli.pick_glyphs() == cli.ASCII_GLYPHS
    header = cli.header_line(True, 2, agent.Totals(total_tokens=10, cost_usd=0.0, cost_known=True, latency_ms=0), 0, cli.ASCII_GLYPHS)
    assert header.startswith("OK Valid after 2 attempts") and header.isascii()


def test_glyphs_fall_back_when_reconfigure_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stuck(NarrowStream):
        def reconfigure(self, **kwargs: Any) -> None:
            raise OSError("cannot change encoding")

    monkeypatch.setattr(cli.sys, "stdout", Stuck())
    assert cli.pick_glyphs() == cli.ASCII_GLYPHS


"""`soa extract` and `soa eval` through typer's CliRunner, with llm.call mocked (rule 13)."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

from soa import agent, cli, llm
from soa.cli import app
from soa.llm import LLMAPIError, LLMResponse, Message
from soa.schemas import JobPosting

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


def script(
    monkeypatch: pytest.MonkeyPatch,
    items: list[LLMResponse | LLMAPIError],
    modes: list[type[BaseModel] | None] | None = None,
) -> list[str]:
    """llm.call returns (or raises) `items` in order; returns the model string of each call.

    Pass a list as `modes` to also collect the `response_model` of each call.
    """
    queue = list(items)
    models: list[str] = []
    modes = [] if modes is None else modes

    def fake(messages: list[Message], model: str, response_model: type[BaseModel] | None = None) -> LLMResponse:
        models.append(model)
        modes.append(response_model)
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


def status(result: Any) -> str:
    """The raw stderr bytes (header, attempt lines, disagreements, errors) decoded as UTF-8, CRLF normalized."""
    decoded: str = result.stderr_bytes.decode("utf-8")
    return decoded.replace("\r\n", "\n")


def posting_json(result: Any) -> dict[str, Any]:
    """stdout must be exactly one JSON document and nothing else."""
    parsed: dict[str, Any] = json.loads(utf8(result))
    return parsed


# --- extract: success -----------------------------------------------------


def test_success_exits_0_with_header_and_pretty_json(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 0, result.output
    assert status(result).splitlines()[0] == "✓ Valid after 1 attempt  (1,240 tokens · $0.0009 · 2.1s)"
    assert '\n  "title": "Développeur Backend"' in utf8(result)  # pretty-printed, UTF-8 kept
    data = posting_json(result)  # stdout parses as JSON: the status lines are not in it
    assert [s["name"] for s in data["skills"]] == ["PostgreSQL"]  # normalized: duplicates merged
    assert data["skills"][0]["required"] is True
    assert data["location"] == {"city": "Bengaluru", "country": "India", "mode": "onsite"}  # filled from the header


def test_json_mode_flag_requests_schema_mode_for_jobposting(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    modes: list[type[BaseModel] | None] = []
    script(monkeypatch, [resp(GOOD), resp(GOOD)], modes)
    assert runner.invoke(app, ["extract", str(jd_file), "--model", "m", "--json-mode"]).exit_code == 0
    assert runner.invoke(app, ["extract", str(jd_file), "--model", "m"]).exit_code == 0
    assert modes == [JobPosting, None]


def test_schema_mode_unsupported_exits_2_with_the_reason(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    err = LLMAPIError("schema mode unsupported by m: BadRequestError: nope", model="m", retryable=False, status_code=400)
    script(monkeypatch, [err])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m", "--json-mode"])
    assert result.exit_code == 2
    assert "schema mode unsupported by m" in status(result) and "retries exhausted" not in status(result)


def test_retry_success_lists_each_failed_attempt(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(BAD), resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 0
    lines = status(result).splitlines()
    assert lines[0] == "✓ Valid after 2 attempts  (2,480 tokens · $0.0018 · 4.2s)"
    assert lines[1].startswith("  attempt 1 ✗ seniority: Input should be")
    assert posting_json(result)["seniority"] == "mid"


def test_tool_disagreements_are_shown(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    pune = json.dumps({**VALID, "location": {"city": "Pune", "country": "India", "mode": "hybrid"}})
    script(monkeypatch, [resp(pune)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert "  tool disagreement  location.city: model 'Pune', header tool 'Bengaluru' (model value kept)" in status(result)
    assert "tool disagreement" not in utf8(result)
    assert posting_json(result)["location"]["city"] == "Pune"


def test_no_normalize_prints_the_raw_posting(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m", "--no-normalize"])
    assert result.exit_code == 0
    data = posting_json(result)
    assert [s["name"] for s in data["skills"]] == ["postgres", "PostgreSQL"]  # as the model wrote them
    assert data["location"]["city"] is None  # not filled from the header
    assert "tool disagreement" not in status(result)


def test_unknown_cost_is_shown_as_na(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(GOOD, known=False)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert status(result).splitlines()[0] == "✓ Valid after 1 attempt  (1,240 tokens · n/a · 2.1s)"


# --- extract: failures and exit codes -------------------------------------


def test_extraction_failed_exits_1_and_shows_every_attempt(monkeypatch: pytest.MonkeyPatch, jd_file: Path) -> None:
    script(monkeypatch, [resp(BAD), resp("not json at all"), resp(BAD)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 1
    lines = status(result).splitlines()
    assert lines[0] == "✗ Failed after 3 attempts  (3,720 tokens · $0.0027 · 6.3s)"
    attempts = [line for line in lines if line.startswith("  attempt ")]
    assert [line.split()[1] for line in attempts] == ["1", "2", "3"]
    assert len(lines) == 1 + 3  # header + one line per attempt
    assert result.stdout_bytes == b""  # a failed run prints no JSON


def test_library_log_warnings_do_not_leak_to_stderr(
    monkeypatch: pytest.MonkeyPatch, jd_file: Path
) -> None:
    # pytest's log capture normally hides this: with no handler on the root logger, Python's fallback handler
    # would print every WARNING (validation failures, tool disagreements) to stderr
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    pune = json.dumps({**VALID, "location": {"city": "Pune", "country": "India", "mode": "hybrid"}})
    script(monkeypatch, [resp(BAD), resp(pune)])
    result = runner.invoke(app, ["extract", str(jd_file), "--model", "m"])
    assert result.exit_code == 0
    err = status(result)
    # stderr holds exactly the CLI's own status lines, each once (city and mode both disagree with the header);
    # a leaking logging handler would repeat them
    assert err.count("attempt 1 ✗") == 1 and err.count("tool disagreement") == 2
    assert "WARNING" not in err and "output failed validation" not in err
    assert posting_json(result)["location"]["city"] == "Pune"


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
    monkeypatch.setattr(cli.sys, "stderr", NarrowStream())
    assert cli.pick_glyphs() == cli.ASCII_GLYPHS
    header = cli.header_line(True, 2, agent.Totals(total_tokens=10, cost_usd=0.0, cost_known=True, latency_ms=0), 0, cli.ASCII_GLYPHS)
    assert header.startswith("OK Valid after 2 attempts") and header.isascii()


def test_glyphs_fall_back_when_reconfigure_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stuck(NarrowStream):
        def reconfigure(self, **kwargs: Any) -> None:
            raise OSError("cannot change encoding")

    monkeypatch.setattr(cli.sys, "stderr", Stuck())
    assert cli.pick_glyphs() == cli.ASCII_GLYPHS


# --- eval stub ------------------------------------------------------------


def make_data(root: Path, n_docs: int, n_labels: int) -> Path:
    jds = root / "data" / "jds"
    labels = root / "data" / "labels"
    jds.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n_docs):
        (jds / f"jd_{i:03d}.txt").write_text(JD, encoding="utf-8")
    for i in range(n_labels):
        (labels / f"jd_{i:03d}.json").write_text("{}", encoding="utf-8")
    return jds


def plain_label() -> str:
    """A label that validates (no duplicate skills)."""
    return json.dumps({**VALID, "skills": [{"name": "PostgreSQL", "required": True}]}, ensure_ascii=False)


def out(result: Any) -> str:
    return utf8(result)


def test_eval_shows_the_plan_and_an_estimate_and_asks_before_calling(
    monkeypatch: pytest.MonkeyPatch, isolated: Path
) -> None:
    jds = make_data(isolated, n_docs=4, n_labels=0)
    models = script(monkeypatch, [])
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds)], input="n\n")
    text = out(result)
    assert result.exit_code == 1  # declined
    assert "documents: 4 in" in text and "field accuracy: skipped (no labels in" in text
    assert "setup B (retry on, native off): 4 to 12 calls" in text
    assert "estimated model calls: 4 to 12 (real API calls)" in text
    assert "Make these model calls now?" in text
    assert models == [] and not (isolated / "results").exists()  # nothing sent, nothing written


def test_eval_runs_after_a_yes_and_saves_results(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=3, n_labels=0)
    models = script(monkeypatch, [resp(GOOD)] * 3)
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds)], input="y\n")
    assert result.exit_code == 0, result.output
    assert len(models) == 3
    summary = (isolated / "results" / "summary.md").read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("| model | setup | n |") and len(summary) == 3
    assert summary[2].startswith("| m | B | 3 | 100.0% | 100.0% | 1.00 | n/a | $0.0027 |")
    assert summary[2] in out(result)  # the same row is printed
    assert len(list((isolated / "results").glob("*_m_B.json"))) == 1


def test_eval_yes_skips_the_prompt(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=1, n_labels=0)
    script(monkeypatch, [resp(GOOD)])
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--yes"])  # no input given
    assert result.exit_code == 0 and "Make these model calls now?" not in out(result)


def test_eval_without_yes_and_no_terminal_input_aborts(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=1, n_labels=0)
    models = script(monkeypatch, [])
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds)])  # stdin is empty
    assert result.exit_code == 1 and models == []


def test_eval_all_runs_the_four_setups_and_writes_four_rows(
    monkeypatch: pytest.MonkeyPatch, isolated: Path
) -> None:
    jds = make_data(isolated, n_docs=2, n_labels=0)
    modes: list[type[BaseModel] | None] = []
    script(monkeypatch, [resp(GOOD)] * 8, modes)
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--setup", "all", "--yes"])
    assert result.exit_code == 0, result.output
    assert "estimated model calls: 8 to 16 (real API calls)" in out(result)  # A 2, B 2-6, C 2, D 2-6
    assert modes == [None] * 4 + [JobPosting] * 4  # A, B plain; C, D schema mode
    names = sorted(p.name for p in (isolated / "results").glob("*.json"))
    assert [n.rsplit("_", 1)[1] for n in names] == ["A.json", "B.json", "C.json", "D.json"]
    rows = (isolated / "results" / "summary.md").read_text(encoding="utf-8").splitlines()[2:]
    assert [r.split(" | ")[1] for r in rows] == ["A", "B", "C", "D"]


def test_eval_setup_is_case_insensitive_and_validated(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=1, n_labels=0)
    script(monkeypatch, [])
    ok = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--setup", "d"], input="n\n")
    assert "setup D (retry on, native on)" in out(ok)
    bad = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--setup", "E"])
    assert bad.exit_code == 2 and "unknown setup 'E'" in bad.stderr


def test_eval_scores_field_accuracy_when_labels_exist(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=2, n_labels=0)
    (isolated / "data" / "labels" / "jd_000.json").write_text(plain_label(), encoding="utf-8")
    (isolated / "data" / "labels" / "jd_001.json").write_text('{"title": "TODO"}', encoding="utf-8")
    script(monkeypatch, [resp(GOOD)] * 2)
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--yes"])
    assert result.exit_code == 0, result.output
    assert "field accuracy: 2 label files in" in out(result)
    assert "labels skipped (invalid or still TODO): jd_001" in out(result)
    row = next(x for x in out(result).splitlines() if x.startswith("| m | B"))
    cells = [c.strip() for c in row.strip("| ").split("|")]
    assert cells[6] != "n/a" and cells[6].endswith("%")  # the field acc (norm) cell has a number
    [report_file] = (isolated / "results").glob("*.json")
    data = json.loads(report_file.read_text(encoding="utf-8"))
    assert (data["n_labelled"], data["n_labelled_with_output"]) == (1, 1)


def test_eval_reports_docs_without_a_model_response(monkeypatch: pytest.MonkeyPatch, isolated: Path) -> None:
    jds = make_data(isolated, n_docs=2, n_labels=0)
    err = LLMAPIError("AuthenticationError: bad key", model="m", retryable=False, status_code=401)
    script(monkeypatch, [err, resp(GOOD)])
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), "--yes"])
    assert result.exit_code == 0, result.output
    text = out(result)
    assert "| m | B | 2 (1 scored) | 100.0% | 100.0% |" in text
    assert "setup B: 1 API error line(s) in the log" in text
    assert "setup B: 1 doc(s) got no model response and are left out of the validity numbers" in text


def test_eval_labels_dir_overrides_the_sibling_folder(isolated: Path) -> None:
    jds = make_data(isolated, n_docs=2, n_labels=1)  # the sibling folder holds 1 label
    other = isolated / "gold"
    other.mkdir()
    for name in ("a", "b", "c"):
        (other / f"{name}.json").write_text("{}", encoding="utf-8")
    args = ["eval", "--model", "m", "--data-dir", str(jds), "--labels-dir", str(other)]
    result = runner.invoke(app, args, input="n\n")
    assert "field accuracy: 3 label files in" in out(result) and "gold" in out(result)


def test_eval_labels_dir_missing_means_no_field_accuracy(isolated: Path) -> None:
    jds = make_data(isolated, n_docs=2, n_labels=1)
    args = ["eval", "--model", "m", "--data-dir", str(jds), "--labels-dir", str(isolated / "nope")]
    assert "field accuracy: skipped (no labels in" in out(runner.invoke(app, args, input="n\n"))


def test_eval_limit_is_shown_in_the_plan_and_caps_the_estimate(isolated: Path) -> None:
    jds = make_data(isolated, n_docs=5, n_labels=2)
    result = runner.invoke(app, ["eval", "--model", "m", "--limit", "3", "--data-dir", str(jds)], input="n\n")
    assert "documents: 3 in" in out(result) and "(limit 3)" in out(result)
    assert "estimated model calls: 3 to 9" in out(result)
    assert "field accuracy: 2 label files in" in out(result)


def test_eval_missing_data_dir_exits_2(isolated: Path) -> None:
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(isolated / "nowhere")])
    assert result.exit_code == 2 and "data directory not found" in result.stderr


def test_eval_empty_data_dir_exits_2(isolated: Path) -> None:
    (isolated / "empty").mkdir()
    result = runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(isolated / "empty")])
    assert result.exit_code == 2 and "no .txt job descriptions" in result.stderr


def test_eval_default_data_dir_is_data_jds(isolated: Path) -> None:
    make_data(isolated, n_docs=1, n_labels=0)  # relative to the working directory
    result = runner.invoke(app, ["eval", "--model", "m"], input="n\n")
    assert "documents: 1 in data" in out(result).replace("\\", "/")


def test_eval_without_a_model_exits_2(isolated: Path) -> None:
    make_data(isolated, n_docs=1, n_labels=0)
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2 and "SOA_MODEL_PRIMARY" in result.stderr


def test_eval_no_longer_takes_the_old_setup_flags(isolated: Path) -> None:
    jds = make_data(isolated, n_docs=1, n_labels=0)
    for flag in ("--json-mode", "--no-retry"):
        assert runner.invoke(app, ["eval", "--model", "m", "--data-dir", str(jds), flag]).exit_code == 2


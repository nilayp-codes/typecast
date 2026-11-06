"""Typer CLI: `soa extract` and `soa eval`. Output format: docs/DESIGN.md section 2.

Exit codes: 0 success; 1 ExtractionFailed (or post-processing broke), or `soa eval` declined at the prompt;
2 bad input or config, or an API error.
stdout carries only the result JSON of `soa extract`, so it can be piped; the header, attempt lines, tool
disagreements and errors go to stderr. API key values are never printed.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, NamedTuple, NoReturn

import typer
from dotenv import find_dotenv, load_dotenv

from soa import eval as soa_eval
from soa.agent import (
    DEFAULT_MAX_ATTEMPTS,
    AttemptFailure,
    ExtractionFailed,
    ExtractionResult,
    Totals,
    extract,
)
from soa.errors import redact_keys
from soa.llm import LLMAPIError
from soa.logger import default_log_path
from soa.tools import Disagreement, ToolResponseError

EXIT_FAILED = 1
EXIT_BAD_INPUT = 2

# Tracebacks must never print local variables (litellm frames hold `api_key`). Typer's default is already False;
# it is set explicitly so a changed default cannot leak a key.
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Structured Output Agent.",
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main() -> None:
    """Structured Output Agent."""
    # Library warnings are already on screen as attempt / disagreement lines and in the JSONL log. Without a
    # handler, Python's fallback handler would print each of them again on stderr.
    soa_logger = logging.getLogger("soa")
    if not any(isinstance(h, logging.NullHandler) for h in soa_logger.handlers):
        soa_logger.addHandler(logging.NullHandler())


class Glyphs(NamedTuple):
    ok: str
    fail: str
    dot: str


UNICODE_GLYPHS = Glyphs("✓", "✗", "·")
ASCII_GLYPHS = Glyphs("OK", "FAIL", "-")


def fail(message: str, code: int = EXIT_BAD_INPUT) -> NoReturn:
    typer.echo(f"error: {redact_keys(message)}", err=True)
    raise typer.Exit(code)


def load_env() -> None:
    """Load `.env` from the working directory (or a parent). Variables already set are not overridden."""
    path = find_dotenv(usecwd=True)
    if path:  # load_dotenv("") would fall back to searching from this file's folder
        load_dotenv(path)


def status(line: str) -> None:
    """One status line (header, attempt, disagreement) on stderr, keeping stdout for the JSON."""
    typer.echo(line, err=True)


def pick_glyphs() -> Glyphs:
    """UTF-8 glyphs when stderr (where status lines go) can carry them, otherwise OK/FAIL.

    Switches stdout/stderr to UTF-8 where possible.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (OSError, ValueError):
                pass  # the stream keeps its encoding; the encode check below decides whether glyphs are safe
    try:
        "".join(UNICODE_GLYPHS).encode(sys.stderr.encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return ASCII_GLYPHS
    return UNICODE_GLYPHS


def resolve_model(cli_value: str | None) -> str:
    model = cli_value or os.environ.get("SOA_MODEL_PRIMARY", "")
    if not model:
        fail("no model: pass --model or set SOA_MODEL_PRIMARY (in .env)")
    return model


def resolve_max_attempts(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    raw = os.environ.get("SOA_MAX_ATTEMPTS", "").strip()
    if not raw:
        return DEFAULT_MAX_ATTEMPTS
    if not raw.isdigit() or int(raw) < 1:
        fail(f"SOA_MAX_ATTEMPTS must be a whole number >= 1, got {raw!r}")
    return int(raw)


def read_input(path: Path) -> str:
    """The file as UTF-8 (a leading BOM is dropped). Missing, unreadable, non-UTF-8 or empty files exit 2."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        fail(f"file not found: {path}")
    except UnicodeDecodeError:
        fail(f"{path} is not valid UTF-8")
    except OSError as exc:
        fail(f"cannot read {path}: {exc.strerror or exc}")
    if not text.strip():
        fail(f"{path} is empty")
    return text


def header_line(ok: bool, attempts: int, totals: Totals, api_retries: int, g: Glyphs) -> str:
    cost = f"${totals.cost_usd:.4f}" if totals.cost_known else "n/a"
    parts = [f"{totals.total_tokens:,} tokens", cost, f"{totals.latency_ms / 1000:.1f}s"]
    if api_retries:
        parts.append(f"{api_retries} API {'retry' if api_retries == 1 else 'retries'}")
    noun = "attempt" if attempts == 1 else "attempts"
    verdict = f"{g.ok} Valid after" if ok else f"{g.fail} Failed after"
    return f"{verdict} {attempts} {noun}  ({f' {g.dot} '.join(parts)})"


def failure_lines(failures: list[AttemptFailure], g: Glyphs) -> list[str]:
    """One block per rejected attempt: `attempt N ✗ <first error>`, further errors aligned underneath."""
    lines: list[str] = []
    for failure in failures:
        first, *rest = failure.error.splitlines() or [failure.error_type]
        prefix = f"  attempt {failure.attempt} {g.fail} "
        lines.append(prefix + first)
        lines.extend(" " * len(prefix) + line for line in rest)
    return lines


def disagreement_lines(disagreements: list[Disagreement]) -> list[str]:
    return [
        f"  tool disagreement  {d.field}: model {d.model_value!r}, header tool {d.tool_value!r} (model value kept)"
        for d in disagreements
    ]


def report_success(result: ExtractionResult, g: Glyphs, raw: bool) -> None:
    totals = Totals(
        total_tokens=result.total_tokens, cost_usd=result.cost_usd,
        cost_known=result.cost_known, latency_ms=result.latency_ms,
    )
    status(header_line(True, result.attempts, totals, result.api_retries, g))
    for line in failure_lines(result.failed_attempts, g) + disagreement_lines(result.disagreements):
        status(line)
    posting = result.raw_posting if raw else result.posting
    typer.echo(posting.model_dump_json(indent=2))


def report_failure(exc: ExtractionFailed, g: Glyphs) -> None:
    status(header_line(False, len(exc.attempts), exc.totals, 0, g))
    for line in failure_lines(exc.failures, g):
        status(line)


FileArg = Annotated[Path, typer.Argument(help="Job description text file (UTF-8).")]
ModelOpt = Annotated[str | None, typer.Option("--model", help="LiteLLM model string. Default: SOA_MODEL_PRIMARY.")]
JsonModeOpt = Annotated[bool, typer.Option("--json-mode", help="Use the provider's JSON-schema mode (built from JobPosting).")]


@app.command("extract")
def extract_cmd(
    file: FileArg,
    model: ModelOpt = None,
    json_mode: JsonModeOpt = False,
    no_normalize: Annotated[
        bool, typer.Option("--no-normalize", help="Skip tool post-processing; print the model's output as is.")
    ] = False,
    max_attempts: Annotated[
        int | None, typer.Option("--max-attempts", min=1, help="Default: SOA_MAX_ATTEMPTS, else 3.")
    ] = None,
) -> None:
    """Extract one job description into validated JSON."""
    load_env()
    glyphs = pick_glyphs()
    chosen = resolve_model(model)
    allowed = resolve_max_attempts(max_attempts)
    text = read_input(file)
    try:
        result = extract(text, chosen, allowed, json_mode, doc_id=file.stem, normalize=not no_normalize)
    except ExtractionFailed as exc:
        report_failure(exc, glyphs)
        raise typer.Exit(EXIT_FAILED) from exc
    except LLMAPIError as exc:
        retry_note = " (retries exhausted)" if exc.retryable else ""
        fail(f"API error from {exc.model}{retry_note}: {' '.join(str(exc).split())}")
    except ToolResponseError as exc:
        fail(f"post-processing failed (a bug in the tools; try --no-normalize): {exc}", EXIT_FAILED)
    report_success(result, glyphs, raw=no_normalize)


SetupOpt = Annotated[
    str,
    typer.Option(
        "--setup",
        help="A plain/no retry, B plain/retry, C schema mode/no retry, D schema mode/retry, or all.",
    ),
]
LabelsDirOpt = Annotated[
    Path | None,
    typer.Option("--labels-dir", help="Folder of gold label .json files. Default: labels/ beside --data-dir."),
]


PauseOpt = Annotated[
    float | None,
    typer.Option(
        "--pause", min=0, help=f"Seconds to wait between documents. Default: SOA_EVAL_PAUSE, else {soa_eval.DEFAULT_PAUSE_SECONDS:g}."
    ),
]
ResumeOpt = Annotated[
    str | None,
    typer.Option("--resume", help="Eval id of an earlier run: skip documents it already finished, add to its report."),
]


def resolve_pause(cli_value: float | None) -> float:
    """--pause, else SOA_EVAL_PAUSE, else the default."""
    if cli_value is not None:
        return cli_value
    raw = os.environ.get("SOA_EVAL_PAUSE", "").strip()
    if not raw:
        return soa_eval.DEFAULT_PAUSE_SECONDS
    try:
        seconds = float(raw)
    except ValueError:
        seconds = -1.0
    if not seconds >= 0:  # also rejects nan
        fail(f"SOA_EVAL_PAUSE must be a number of seconds >= 0, got {raw!r}")
    return seconds


def run_plan(plan: soa_eval.EvalPlan, results_dir: Path, eval_id: str) -> None:
    """Run each setup of the plan, save its results, then print the summary table and notes."""
    reports: list[soa_eval.EvalReport] = []
    for setup in plan.setups:
        typer.echo(f"running {soa_eval.setup_label(setup)} ...")
        try:
            report = soa_eval.run_eval(
                plan.model, setup, plan.data_dir, plan.labels_dir, plan.limit,
                pause=plan.pause, eval_id=eval_id, resume=plan.resume is not None,
            )
        except ToolResponseError as exc:
            fail(f"post-processing failed (a bug in the tools): {exc}", EXIT_FAILED)
        typer.echo(f"saved {soa_eval.write_results(report, results_dir, resumed=plan.resume is not None)}")
        reports.append(report)
    typer.echo("")
    typer.echo(soa_eval.SUMMARY_HEADER + "".join(soa_eval.summary_row(r) for r in reports), nl=False)
    for note in (n for r in reports for n in soa_eval.report_notes(r)):
        typer.echo(note)


@app.command("eval")
def eval_cmd(
    model: ModelOpt = None,
    setup: SetupOpt = "B",
    limit: Annotated[int | None, typer.Option("--limit", min=1, help="Run at most N documents.")] = None,
    data_dir: Annotated[Path, typer.Option("--data-dir", help="Folder of job description .txt files.")] = (
        soa_eval.DEFAULT_DATA_DIR
    ),
    labels_dir: LabelsDirOpt = None,
    results_dir: Annotated[Path, typer.Option("--results-dir", help="Where reports and summary.md go.")] = (
        soa_eval.RESULTS_DIR
    ),
    yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation before making model calls.")] = False,
    pause: PauseOpt = None,
    resume: ResumeOpt = None,
) -> None:
    """Run one or all eval setups over a folder of job descriptions. Makes REAL model calls."""
    load_env()
    pick_glyphs()
    chosen = resolve_model(model)
    seconds = resolve_pause(pause)
    try:
        if resume:
            soa_eval.check_resume(resume, chosen, default_log_path())
        plan = soa_eval.plan_eval(
            chosen, soa_eval.parse_setups(setup), data_dir, labels_dir, limit, pause=seconds, resume=resume
        )
    except (ValueError, FileNotFoundError) as exc:
        fail(str(exc))
    eval_id = resume or soa_eval.new_eval_id()
    for line in plan.describe():
        typer.echo(line)
    typer.echo(f"eval id: {eval_id} (if the run stops, continue it with: soa eval --resume {eval_id} ...)")
    if not yes:
        typer.confirm("Make these model calls now?", abort=True)
    run_plan(plan, results_dir, eval_id)

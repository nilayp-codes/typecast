"""Phase 2 baseline: single-shot extraction (no retries, json_mode off) over data/jds/*.

Makes REAL API calls: a deliberate exception to RULES.md rule 14. Never prints or records API keys.

    uv run python scripts/baseline.py --check   # load .env, show what is set, no API call
    uv run python scripts/baseline.py --probe   # ONE call on jd_007; stops with the error if it fails
    uv run python scripts/baseline.py           # all JDs, one shared run_id, then write the report
"""

import argparse
import os
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from soa.agent import extract_once, read_jd
from soa.llm import LLMAPIError
from soa.logger import AttemptLog, default_log_path
from soa.schemas import JobPosting

JD_DIR = Path("data/jds")
RESULTS_DIR = Path("results")
PROBE_DOC = "jd_007"
PAUSE_SECONDS = 5
KEY_NAMES = ("GEMINI_API_KEY", "OPENROUTER_API_KEY")
WRONG_TYPE_SUFFIXES = ("_type", "_parsing", "_from_float", "_from_str")
CONSTRAINT_TYPES = {
    "too_long", "too_short", "string_too_long", "string_too_short",
    "less_than_equal", "greater_than_equal", "less_than", "greater_than",
}


class Row(BaseModel):
    doc_id: str
    ok: bool
    error_type: str
    first_loc: str
    total_tokens: int | None
    latency_ms: int | None
    kinds: list[str]
    example: str


def redact(text: str) -> str:
    """Remove any API key value from text before it is printed or saved."""
    for name in KEY_NAMES:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, "***")
    return text


def check_env() -> str:
    loaded = load_dotenv()
    model = os.environ.get("SOA_MODEL_PRIMARY", "")
    print(f".env loaded: {loaded}")
    print(f"SOA_MODEL_PRIMARY: {model or '(not set)'}")
    for name in KEY_NAMES:
        print(f"{name} set: {bool(os.environ.get(name))}")
    print(f"log path: {default_log_path()}")
    if not model:
        sys.exit("SOA_MODEL_PRIMARY is not set. Fill it in .env and retry.")
    return model


def call_one(doc_id: str, model: str, run_id: str) -> tuple[bool, str]:
    """One extraction. Returns (ok, error text) where error text covers API and validation failures."""
    text = read_jd(JD_DIR / f"{doc_id}.txt")
    try:
        result = extract_once(text, model, doc_id, run_id)
    except LLMAPIError as exc:
        return False, "APIError: " + redact(" ".join(str(exc).split()))
    return result.ok, "" if result.ok else f"{result.error_type}: {result.error}"


def probe(model: str) -> None:
    ok, detail = call_one(PROBE_DOC, model, run_id=f"probe-{uuid.uuid4().hex[:8]}")
    if detail.startswith("APIError"):
        sys.exit(f"PROBE FAILED on {PROBE_DOC}, nothing else was run:\n{detail}")
    print(f"probe call on {PROBE_DOC} reached the model. validation ok: {ok}")
    if detail:
        print(detail)


def run_all(model: str, run_id: str) -> dict[str, str]:
    """Run every JD. Returns {doc_id: APIError text} for docs that hit an API error."""
    api_errors: dict[str, str] = {}
    docs = sorted(p.stem for p in JD_DIR.glob("jd_*.txt"))
    for i, doc_id in enumerate(docs):
        if i:
            time.sleep(PAUSE_SECONDS)
        ok, detail = call_one(doc_id, model, run_id)
        print(f"[{i + 1}/{len(docs)}] {doc_id}: {'ok' if ok else detail.splitlines()[0][:160]}")
        if detail.startswith("APIError"):
            api_errors[doc_id] = detail
    return api_errors


def generic_loc(loc: tuple[int | str, ...]) -> str:
    return ".".join("*" if isinstance(p, int) else p for p in loc) or "(root)"


def classify_json_error(raw: str) -> str:
    stripped = raw.strip()
    if not stripped:
        return "empty output"
    if stripped.startswith("```"):
        return "markdown fences"
    if not stripped.startswith("{"):
        return "prose around JSON"
    return "truncation or malformed JSON"


def classify_validation(raw: str) -> list[str]:
    """Re-validate the logged raw output to get Pydantic's error type codes."""
    try:
        JobPosting.model_validate_json(raw)
    except ValidationError as exc:
        return [classify_one(e["type"], e["msg"], generic_loc(e["loc"])) for e in exc.errors()]
    return ["(re-validation passed unexpectedly)"]


def classify_one(kind: str, msg: str, loc: str) -> str:
    if kind == "missing":
        return "missing field"
    if kind == "enum":
        return "enum mismatch"
    if kind == "extra_forbidden":
        return "extra field"
    if kind in CONSTRAINT_TYPES:
        return f"constraint: {kind} on {loc}"
    if kind == "value_error" and "duplicate skill names" in msg:
        return "duplicate skills"
    if kind == "value_error" and "'intern'" in msg:
        return "intern rule"
    if kind.endswith(WRONG_TYPE_SUFFIXES):
        return f"wrong JSON type ({kind} on {loc})"
    return f"other ({kind} on {loc})"


def row_from_log(log: AttemptLog) -> Row:
    first = (log.error or "").splitlines()[0] if log.error else ""
    total = log.prompt_tokens + log.completion_tokens
    if log.ok:
        return Row(doc_id=log.doc_id, ok=True, error_type="-", first_loc="-", total_tokens=total,
                   latency_ms=log.latency_ms, kinds=[], example="")
    if log.error_type == "JSONDecodeError":
        kinds = [classify_json_error(log.raw_output)]
    else:
        kinds = classify_validation(log.raw_output)
    return Row(doc_id=log.doc_id, ok=False, error_type=log.error_type or "?", first_loc=first.split(":", 1)[0],
               total_tokens=total, latency_ms=log.latency_ms, kinds=kinds, example=first)


def build_rows(run_id: str, api_errors: dict[str, str]) -> list[Row]:
    lines = default_log_path().read_text(encoding="utf-8").splitlines()
    logs = [AttemptLog.model_validate_json(x) for x in lines]
    rows = {x.doc_id: row_from_log(x) for x in logs if x.run_id == run_id}
    for doc_id, detail in api_errors.items():
        rows[doc_id] = Row(doc_id=doc_id, ok=False, error_type="APIError", first_loc="-", total_tokens=None,
                           latency_ms=None, kinds=["API error"], example=detail[:300])
    return [rows[k] for k in sorted(rows)]


def render_table(rows: list[Row]) -> str:
    out = ["| doc_id | ok | error_type | first error loc | total tokens | latency (ms) |", "|---|---|---|---|---|---|"]
    for r in rows:
        tokens = "-" if r.total_tokens is None else f"{r.total_tokens:,}"
        latency = "-" if r.latency_ms is None else f"{r.latency_ms:,}"
        out.append(f"| {r.doc_id} | {'yes' if r.ok else 'no'} | {r.error_type} | {r.first_loc} | {tokens} | {latency} |")
    return "\n".join(out)


def render_groups(rows: list[Row]) -> str:
    groups: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for kind in sorted(set(r.kinds)):
            groups[kind].append(r.doc_id)
    if not groups:
        return "No failures."
    out = ["| failure kind | docs | count |", "|---|---|---|"]
    for kind, docs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append(f"| {kind} | {', '.join(docs)} | {len(docs)} |")
    return "\n".join(out)


def git_head() -> str:
    done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False)
    return done.stdout.strip() or "unknown"


def render_report(rows: list[Row], model: str, run_id: str) -> str:
    passed = sum(r.ok for r in rows)
    tokens = sum(r.total_tokens or 0 for r in rows)
    failures = "\n".join(f"- {r.doc_id}: {r.example}" for r in rows if not r.ok)
    return f"""# Baseline: {model}, single shot, no retries

- date: {date.today().isoformat()}
- model: `{model}` (results are tied to this exact string)
- run_id: `{run_id}`
- json_mode: off; retries: none; pause between calls: {PAUSE_SECONDS}s
- code at run time: commit `{git_head()}` (prompt frozen, `619be12`)
- first-try valid: {passed}/{len(rows)}
- total tokens (prompt + completion): {tokens:,}

## Per document

{render_table(rows)}

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

{render_groups(rows)}

## First error line per failed document

{failures or 'none'}
"""


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="load .env and report, no API call")
    parser.add_argument("--probe", action="store_true", help=f"one call on {PROBE_DOC}, then stop")
    args = parser.parse_args()
    model = check_env()
    if args.check:
        return
    if args.probe:
        probe(model)
        return
    run_id = f"baseline-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
    api_errors = run_all(model, run_id)
    rows = build_rows(run_id, api_errors)
    report = render_report(rows, model, run_id)
    path = RESULTS_DIR / f"{date.today().isoformat()}_baseline_{model.replace('/', '_')}.md"
    RESULTS_DIR.mkdir(exist_ok=True)
    path.write_text(report, encoding="utf-8", newline="\n")
    print(f"\n{report}\nwrote {path}")


if __name__ == "__main__":
    main()

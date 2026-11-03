"""Phase 2 baseline: single-shot extraction (setup A) over data/jds/*.

Makes REAL API calls: a deliberate exception to RULES.md rule 14. Never prints or records API keys.

    uv run python scripts/baseline.py --check                 # load .env, show what is set, no API call
    uv run python scripts/baseline.py --probe [--model M]     # ONE call on jd_007; stops if it fails
    uv run python scripts/baseline.py [--model M] [--start jd_005] [--limit 3]
                                                              # JDs in order, one run_id, then the report

Setup A (ARCHITECTURE.md "Eval setups"): plain prompt, json_mode off, no validation retry. The setup is tagged in the
run_id (`baseline-A-<date>-<hash>`); `AttemptLog` gets a real `setup` field in Phase 6.
API errors are handled by this script only: 503 / per-minute 429 wait and retry the same doc (max 3 times);
402 (no credit) or a daily-quota 429 stop the run cleanly. Validation failures are never retried.
"""

import argparse
import os
import re
import subprocess
import sys
import json
import time
import urllib.request
import uuid
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from soa.agent import extract_once, read_jd
from soa.llm import LLMAPIError, LLMResponse
from soa.logger import AttemptLog, default_log_path
from soa.schemas import JobPosting

JD_DIR = Path("data/jds")
RESULTS_DIR = Path("results")
MODELS_URL = "https://openrouter.ai/api/v1/models"  # public list with per-token prices; no key needed
PROBE_DOC = "jd_007"
PAUSE_SECONDS = 5
RETRY_WAITS = (20, 40, 60)  # seconds before API-error retry 1, 2, 3
SETUP = "A"
SETUP_TEXT = "setup A: plain prompt, no retry"
KEY_NAMES = ("OPENROUTER_API_KEY",)
DAILY_MARKERS = ("per-day", "per_day", "per day", "daily")
CREDIT_MARKERS = ("insufficient credit", "more credits", "payment required", "out of credit")
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
    cost_usd: float = 0.0
    cost_known: bool = True
    api_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_estimated: bool = False  # cost_usd was computed from OpenRouter list prices because LiteLLM has none


class Outcome(BaseModel):
    """What one doc's call(s) came to: a model response (valid or not), or an API error."""

    doc_id: str
    ok: bool
    detail: str  # "" when ok; "<kind> | APIError: ..." for API errors; else "<error_type>: <error>"
    api_error: bool
    api_retries: int
    response: LLMResponse | None


class RunState(BaseModel):
    api_errors: dict[str, str] = Field(default_factory=dict)
    retries: dict[str, int] = Field(default_factory=dict)
    attempted: list[str] = Field(default_factory=list)
    stopped: str | None = None


def redact(text: str) -> str:
    """Remove any API key value from text before it is printed or saved."""
    for name in KEY_NAMES:
        value = os.environ.get(name, "")
        if len(value) >= 8:
            text = text.replace(value, "***")
    return text


def check_env(cli_model: str | None) -> str:
    loaded = load_dotenv()
    primary = os.environ.get("SOA_MODEL_PRIMARY", "")
    model = cli_model or primary
    print(f".env loaded: {loaded}")
    print(f"SOA_MODEL_PRIMARY: {primary or '(not set)'}")
    print(f"SOA_MODEL_SECONDARY: {os.environ.get('SOA_MODEL_SECONDARY') or '(not set)'}")
    print(f"model for this run: {model or '(not set)'}")
    for name in KEY_NAMES:
        print(f"{name} set: {bool(os.environ.get(name))}")
    print(f"log path: {default_log_path()}")
    if not model:
        sys.exit("No model: pass --model or set SOA_MODEL_PRIMARY in .env.")
    return model


def classify_api_error(exc: LLMAPIError) -> str:
    """'no-credit' and 'daily-quota' stop the run; 'transient' waits and retries; else 'other'."""
    msg = str(exc).lower()
    if exc.status_code == 402 or any(m in msg for m in CREDIT_MARKERS):
        return "no-credit"
    if exc.status_code == 429 or "ratelimiterror" in msg:
        return "daily-quota" if any(m in msg for m in DAILY_MARKERS) else "transient"
    if exc.status_code == 503 or "serviceunavailable" in msg:
        return "transient"
    return "other"


def call_once(doc_id: str, model: str, run_id: str) -> Outcome:
    """One extraction attempt. API errors are returned (api_error=True), never raised."""
    text = read_jd(JD_DIR / f"{doc_id}.txt")
    try:
        result = extract_once(text, model, doc_id, run_id)
    except LLMAPIError as exc:
        detail = f"{classify_api_error(exc)} | APIError: " + redact(" ".join(str(exc).split()))
        return Outcome(doc_id=doc_id, ok=False, detail=detail, api_error=True, api_retries=0, response=None)
    detail = "" if result.ok else f"{result.error_type}: {result.error}"
    return Outcome(doc_id=doc_id, ok=result.ok, detail=detail, api_error=False, api_retries=0,
                   response=result.response)


def api_kind(outcome: Outcome) -> str:
    return outcome.detail.split(" | ", 1)[0]


def call_with_waits(doc_id: str, model: str, run_id: str) -> Outcome:
    """Re-send the same doc after 503 / per-minute 429 (waits in RETRY_WAITS). Anything else returns at once."""
    outcome = call_once(doc_id, model, run_id)
    for n, wait in enumerate(RETRY_WAITS, start=1):
        if not outcome.api_error or api_kind(outcome) != "transient":
            break
        print(f"    {doc_id}: transient API error, waiting {wait}s (retry {n}/{len(RETRY_WAITS)})", flush=True)
        time.sleep(wait)
        outcome = call_once(doc_id, model, run_id)
        outcome.api_retries = n
    return outcome


def probe(model: str) -> None:
    outcome = call_once(PROBE_DOC, model, run_id=f"probe-{SETUP}-{uuid.uuid4().hex[:8]}")
    if outcome.response is None:
        sys.exit(f"PROBE FAILED on {PROBE_DOC} with {model}, nothing else was run:\n{outcome.detail}")
    r = outcome.response
    print(f"probe {PROBE_DOC} model={model}")
    print(f"  ok={outcome.ok} tokens={r.prompt_tokens}+{r.completion_tokens}={r.prompt_tokens + r.completion_tokens} "
          f"latency_ms={r.latency_ms} cost_usd={r.cost_usd:.6f} cost_known={r.cost_known}")
    if outcome.detail:
        print("  validation: " + outcome.detail.splitlines()[0][:200])


def select_docs(start: str | None, limit: int | None) -> list[str]:
    docs = sorted(p.stem for p in JD_DIR.glob("jd_*.txt"))
    if start:
        if start not in docs:
            sys.exit(f"--start {start}: no such JD. Have {docs[0]}..{docs[-1]}")
        docs = docs[docs.index(start):]
    return docs[:limit] if limit else docs


def stop_reason(doc_id: str, outcome: Outcome) -> str | None:
    kind = api_kind(outcome)
    if kind == "no-credit":
        return f"stopped at {doc_id}: out of OpenRouter credit (HTTP 402). Add credit, then re-run with --start {doc_id}"
    if kind == "daily-quota":
        return f"stopped at {doc_id}: daily quota exhausted (429). Re-run with --start {doc_id} after it resets"
    if kind == "transient":
        return f"stopped at {doc_id}: still failing after {len(RETRY_WAITS)} waits. Re-run with --start {doc_id}"
    return None


def run_all(model: str, run_id: str, docs: list[str]) -> RunState:
    """Run docs in order. Stops cleanly on no-credit, daily-quota or exhausted waits (state.stopped says why)."""
    state = RunState()
    for i, doc_id in enumerate(docs):
        if i:
            time.sleep(PAUSE_SECONDS)
        outcome = call_with_waits(doc_id, model, run_id)
        state.attempted.append(doc_id)
        shown = "ok" if outcome.ok else outcome.detail.replace("\n", " ")[:160]
        print(f"[{i + 1}/{len(docs)}] {doc_id}: {shown}", flush=True)
        if outcome.api_retries:
            state.retries[doc_id] = outcome.api_retries
        if outcome.api_error:
            state.api_errors[doc_id] = outcome.detail.split(" | ", 1)[1]
            reason = stop_reason(doc_id, outcome)
            if reason:
                state.stopped = reason
                break
    return state


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
    tokens = {"prompt_tokens": log.prompt_tokens, "completion_tokens": log.completion_tokens}
    if log.ok:
        return Row(doc_id=log.doc_id, ok=True, error_type="-", first_loc="-", total_tokens=total,
                   latency_ms=log.latency_ms, kinds=[], example="",
                   cost_usd=log.cost_usd, cost_known=log.cost_known, **tokens)
    if log.error_type == "JSONDecodeError":
        kinds = [classify_json_error(log.raw_output)]
    else:
        kinds = classify_validation(log.raw_output)
    return Row(doc_id=log.doc_id, ok=False, error_type=log.error_type or "?", first_loc=first.split(":", 1)[0],
               total_tokens=total, latency_ms=log.latency_ms, kinds=kinds, example=first,
               cost_usd=log.cost_usd, cost_known=log.cost_known, **tokens)


def fetch_prices(model: str) -> tuple[float, float] | None:
    """(USD per prompt token, USD per completion token) from OpenRouter's public model list, or None."""
    slug = model.removeprefix("openrouter/")
    try:
        with urllib.request.urlopen(MODELS_URL, timeout=30) as response:
            listed = json.load(response)["data"]
    except (OSError, ValueError, KeyError) as exc:
        print(f"note: could not fetch OpenRouter prices ({type(exc).__name__}); costs stay unknown")
        return None
    for entry in listed:
        if entry.get("id") == slug:
            return float(entry["pricing"]["prompt"]), float(entry["pricing"]["completion"])
    return None


def apply_price_estimates(rows: list[Row], model: str) -> None:
    """LiteLLM has no price for these models (cost_known=False, cost 0). Fill in an estimate from list prices."""
    if all(r.cost_known for r in rows if r.total_tokens is not None):
        return
    prices = fetch_prices(model)
    if prices is None:
        return
    per_prompt, per_completion = prices
    for r in rows:
        if r.total_tokens is not None and not r.cost_known:
            r.cost_usd = r.prompt_tokens * per_prompt + r.completion_tokens * per_completion
            r.cost_estimated = True


def build_rows(run_id: str, state: RunState) -> list[Row]:
    lines = default_log_path().read_text(encoding="utf-8").splitlines()
    logs = [AttemptLog.model_validate_json(x) for x in lines]
    rows = {x.doc_id: row_from_log(x) for x in logs if x.run_id == run_id}
    for doc_id, detail in state.api_errors.items():
        rows[doc_id] = Row(doc_id=doc_id, ok=False, error_type="APIError", first_loc="-", total_tokens=None,
                           latency_ms=None, kinds=["API error"], example=detail)  # full text, not truncated
    for doc_id, n in state.retries.items():
        rows[doc_id].api_retries = n
    return [rows[k] for k in sorted(rows)]


def cost_flag(r: Row) -> str:
    return "False (cost estimated from list price)" if r.cost_estimated else str(r.cost_known)


def render_table(rows: list[Row]) -> str:
    out = ["| doc_id | ok | error_type | first error loc | total tokens | latency (ms) | cost (USD) | cost_known "
           "| api retries |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        got = r.total_tokens is not None
        tokens = f"{r.total_tokens:,}" if got and r.total_tokens is not None else "-"
        latency = f"{r.latency_ms:,}" if r.latency_ms is not None else "-"
        out.append(f"| {r.doc_id} | {'yes' if r.ok else 'no'} | {r.error_type} | {r.first_loc} | {tokens} | {latency} "
                   f"| {f'{r.cost_usd:.6f}' if got else '-'} | {cost_flag(r) if got else '-'} | {r.api_retries} |")
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


def render_report(rows: list[Row], state: RunState, model: str, run_id: str, planned: int) -> str:
    reached = [r for r in rows if r.error_type != "APIError"]
    passed = sum(r.ok for r in reached)
    pct = f"{100 * passed / len(reached):.1f}%" if reached else "n/a"
    tokens = sum(r.total_tokens or 0 for r in rows)
    cost = sum(r.cost_usd for r in reached)
    unknown = [r.doc_id for r in reached if not r.cost_known]
    if any(r.cost_estimated for r in reached):
        cost_note = (" (ESTIMATE: LiteLLM has no price for this model, so cost_known=False on every call; computed from "
                     "OpenRouter's list prices for the tokens used. The billed amount may differ by provider)")
    else:
        cost_note = f" (cost_known=False on {', '.join(unknown)}; counted as 0)" if unknown else " (cost_known=True on every call)"
    failures = "\n".join(f"- {r.doc_id}: {r.example}" for r in rows if not r.ok)
    status = f"**INCOMPLETE, {state.stopped}**" if state.stopped else "complete"
    waits = "/".join(str(w) for w in RETRY_WAITS)
    return f"""# Baseline: {model}, {SETUP_TEXT}

- date: {date.today().isoformat()}
- {SETUP_TEXT} (json_mode off; the model gets the prompt once, a validation failure is not retried)
- model: `{model}` (results are tied to this exact string)
- run_id: `{run_id}`
- API errors: the script waits {waits} s and re-sends the same doc (503, per-minute 429); pause between docs {PAUSE_SECONDS}s
- code at run time: commit `{git_head()}` (prompt frozen at `d90d70a`: the seniority text there added "Architect -> lead")
- JD set: {planned} docs planned (run on jd_001-jd_011 only). The JD set will grow to 30 for Phase 6, so this is a baseline on a small sample, not the final set
- run status: {status}
- docs attempted: {len(state.attempted)} of {planned} planned
- reached validation (got a model response): {len(reached)}; API errors with no response: {len(rows) - len(reached)}
- first-try valid: {passed}/{len(reached)} ({pct}) of docs that reached validation
- total tokens (prompt + completion): {tokens:,}
- total cost: ${cost:.6f}{cost_note}

## Per document

{render_table(rows)}

## Failures grouped by kind

A document can appear in more than one group (one group per distinct error kind).

{render_groups(rows)}

## First error line per failed document (API errors in full)

{failures or 'none'}
"""


def report_path(model: str, run_id: str) -> Path:
    base = f"{date.today().isoformat()}_baseline_{re.sub(r'[^A-Za-z0-9._-]+', '_', model)}"
    path = RESULTS_DIR / f"{base}.md"
    return path if not path.exists() else RESULTS_DIR / f"{base}_{run_id.rsplit('-', 1)[-1]}.md"


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="load .env and report, no API call")
    parser.add_argument("--probe", action="store_true", help=f"one call on {PROBE_DOC}, then stop")
    parser.add_argument("--model", help="LiteLLM model string (default: SOA_MODEL_PRIMARY)")
    parser.add_argument("--limit", type=int, help="run at most N docs")
    parser.add_argument("--start", help="first doc id to run, e.g. jd_005")
    args = parser.parse_args()
    model = check_env(args.model)
    if args.check:
        return
    if args.probe:
        probe(model)
        return
    docs = select_docs(args.start, args.limit)
    run_id = f"baseline-{SETUP}-{date.today().isoformat()}-{uuid.uuid4().hex[:6]}"
    state = run_all(model, run_id, docs)
    rows = build_rows(run_id, state)
    apply_price_estimates(rows, model)
    report = render_report(rows, state, model, run_id, len(docs))
    RESULTS_DIR.mkdir(exist_ok=True)
    path = report_path(model, run_id)
    path.write_text(report, encoding="utf-8", newline="\n")
    print(f"\n{report}\nwrote {path}")
    if state.stopped:
        sys.exit(f"STOPPED EARLY: {state.stopped}")


if __name__ == "__main__":
    main()

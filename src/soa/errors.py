"""Rendering of Pydantic validation errors, shared by agent.py (retry feedback) and tools.py (ToolError)."""

from pydantic import ValidationError
from pydantic_core import ErrorDetails

MAX_INPUT_CHARS = 60


def format_validation_errors(exc: ValidationError) -> str:
    """One line per error: `loc: msg (got: <input, max 60 chars>)` (RULES.md rule 11)."""
    return "\n".join(_error_line(err) for err in exc.errors())


def _error_line(err: ErrorDetails) -> str:
    loc = ".".join(str(part) for part in err["loc"]) or "(root)"
    got = repr(err["input"])
    if len(got) > MAX_INPUT_CHARS:
        got = got[: MAX_INPUT_CHARS - 3] + "..."
    return f"{loc}: {err['msg']} (got: {got})"

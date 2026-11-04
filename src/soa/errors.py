"""Error rendering shared by agent.py (retry feedback, logs), tools.py (ToolError) and cli.py (messages)."""

import os

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


def format_retry_feedback(exc: ValidationError) -> str:
    """Like format_validation_errors, but a list over its maximum states the count and the fix.

    `skills: has 39 items, maximum is 30 - remove 9, starting with required=false ones`. The logged error text stays
    pydantic's own wording (eval.py classifies constraints by it); only the retry prompt gets this line.
    """
    return "\n".join(_retry_line(err) for err in exc.errors())


def _retry_line(err: ErrorDetails) -> str:
    if err["type"] != "too_long" or not isinstance(err["input"], list):
        return _error_line(err)
    loc = ".".join(str(part) for part in err["loc"]) or "(root)"
    maximum = (err.get("ctx") or {}).get("max_length")
    if not isinstance(maximum, int):
        return _error_line(err)
    count = len(err["input"])
    hint = ", starting with required=false ones" if loc == "skills" else ""
    return f"{loc}: has {count} items, maximum is {maximum} - remove {count - maximum}{hint}"


def redact_keys(text: str) -> str:
    """Keep API key values out of logs and terminal output: mask any *_API_KEY env value found in `text`."""
    for name, value in os.environ.items():
        if name.endswith("_API_KEY") and len(value) >= 8:
            text = text.replace(value, "***")
    return text

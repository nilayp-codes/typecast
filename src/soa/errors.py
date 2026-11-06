"""Keep secrets out of logs and terminal output. Validation error lines live in `typecast_mcp.errors`."""

import os

SECRET_ENV_SUFFIXES = ("_KEY", "_TOKEN")  # *_API_KEY is covered by _KEY
MIN_SECRET_LENGTH = 8  # shorter values would also mask ordinary words


def redact_keys(text: str) -> str:
    """Keep secrets out of logs and terminal output: mask any *_KEY or *_TOKEN env value (8+ chars) found in `text`."""
    for name, value in os.environ.items():
        if name.endswith(SECRET_ENV_SUFFIXES) and len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, "***")
    return text

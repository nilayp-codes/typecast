"""Shared test helpers."""

import re

_CI = re.compile(r" \[\d+(?:\.\d+)?-\d+(?:\.\d+)?\]")


def without_ci(text: str) -> str:
    """Drop the ` [low-high]` 95% intervals a summary row carries, to compare the point figures alone."""
    return _CI.sub("", text)

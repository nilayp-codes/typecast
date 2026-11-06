"""Logic of the validator-only MCP server (docs/MCP.md): the job-posting schema and a JSON validator.

No LLM, no key, no network. This module does not import `mcp` or `litellm`, so it is usable (and testable) without the
SDK; `soa.mcp_server` wires these functions into an MCP server.
"""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from soa.errors import format_retry_feedback
from soa.schemas import JobPosting

MAX_POSTING_CHARS = 20_000  # a valid posting is under 3,000 characters; this only stops absurd payloads
MAX_ERROR_LINES = 20  # a 20 KB payload of unknown keys otherwise returns about 2,000 lines


class ValidationResult(BaseModel):
    """What `validate_job_posting` returns. `valid`, an empty `errors` and a present `posting` always go together."""

    model_config = ConfigDict(extra="forbid", strict=True)

    valid: bool
    errors: list[str]  # one line per error, as in the retry prompt; [] when valid
    posting: JobPosting | None  # the validated posting when valid, else None

    @model_validator(mode="after")
    def fields_agree(self) -> Self:
        if self.valid != (not self.errors) or self.valid != (self.posting is not None):
            raise ValueError("valid must be True exactly when errors is empty and posting is present")
        return self


def get_job_posting_schema() -> dict[str, Any]:
    """JSON Schema of a job posting, unchanged from what the extraction prompt uses."""
    return JobPosting.model_json_schema()


def validate_job_posting(posting: str) -> ValidationResult:
    """Validate JSON text against JobPosting. Bad input is a result (`valid=False` plus error lines), never a raise."""
    if len(posting) > MAX_POSTING_CHARS:
        line = f"(root): payload is {len(posting)} characters, maximum is {MAX_POSTING_CHARS}"
        return ValidationResult(valid=False, errors=[line], posting=None)
    try:
        # JSON mode on purpose: the models are strict, and only JSON-mode validation accepts enum strings.
        parsed = JobPosting.model_validate_json(posting)
    except ValidationError as exc:
        errors = cap_errors(format_retry_feedback(exc).splitlines())
        return ValidationResult(valid=False, errors=errors, posting=None)
    return ValidationResult(valid=True, errors=[], posting=parsed)


def cap_errors(lines: list[str]) -> list[str]:
    """At most MAX_ERROR_LINES lines, plus one line stating how many were left out. Only the MCP path caps."""
    if len(lines) <= MAX_ERROR_LINES:
        return lines
    return [*lines[:MAX_ERROR_LINES], f"... {len(lines) - MAX_ERROR_LINES} more errors omitted"]

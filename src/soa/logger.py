"""AttemptLog model and JSONL writer."""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from soa.llm import CostSource

DEFAULT_LOG_PATH = "logs/attempts.jsonl"
API_ERROR_TYPE = "APIError"  # error_type of a line for a failed provider call; not a validation attempt

# A = plain prompt, no retry; B = plain + retry; C = schema mode, no retry; D = schema mode + retry.
Setup = Literal["A", "B", "C", "D"]


class AttemptLog(BaseModel):
    """One LLM attempt at extracting one document. Written as one JSONL line (DESIGN.md section 3).

    `setup` is the eval setup (null outside eval runs) and `json_mode` says whether schema mode was requested.
    Both have defaults so log lines written before Phase 6 still parse (they were all plain-prompt runs).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    run_id: str
    doc_id: str
    attempt: int = Field(ge=1)
    model: str
    setup: Setup | None = None
    json_mode: bool = False
    ok: bool
    error_type: str | None
    error: str | None
    raw_output: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_known: bool
    cost_source: CostSource | None = None  # None only on lines written before the source was recorded
    ts: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def error_fields_match_ok(self) -> Self:
        """A success carries no error; a failure carries both error_type and error."""
        has_error = self.error_type is not None and self.error is not None
        has_none = self.error_type is None and self.error is None
        if self.ok and not has_none:
            raise ValueError("ok=True requires error_type and error to be null")
        if not self.ok and not has_error:
            raise ValueError("ok=False requires both error_type and error to be set")
        return self


def default_log_path() -> Path:
    """Log file location: SOA_LOG_PATH from the environment, else logs/attempts.jsonl."""
    return Path(os.environ.get("SOA_LOG_PATH", DEFAULT_LOG_PATH))


def read_log(path: Path) -> list[AttemptLog]:
    """Every line of a JSONL log as an AttemptLog, in file order. A missing file is an empty log.

    A line that does not validate raises (a corrupt log must not be silently skipped).
    """
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [AttemptLog.model_validate_json(line) for line in lines if line.strip()]


def append_log(entry: AttemptLog, path: Path) -> None:
    """Append `entry` as one JSON line (UTF-8, LF), creating the parent folder if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(entry.model_dump_json() + "\n")

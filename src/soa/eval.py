"""Eval harness: run the dataset, compute metrics, write results. Phase 6; only the plan exists so far.

`plan_eval` is real (it reads the folder and the labels); `run_eval` is a stub until Phase 6.
"""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATA_DIR = Path("data/jds")


class EvalOptions(BaseModel):
    """What `soa eval` was asked to do. The setup letter follows DESIGN.md section 2."""

    model_config = ConfigDict(extra="forbid", strict=True)

    model: str
    json_mode: bool
    no_retry: bool
    limit: int | None = Field(default=None, ge=1)
    data_dir: Path
    labels_dir: Path | None = None  # None: the labels/ folder beside data_dir

    @property
    def setup(self) -> str:
        """A = no retry, B = default, C = no retry + native, D = native."""
        return {(True, False): "A", (False, False): "B", (True, True): "C", (False, True): "D"}[
            (self.no_retry, self.json_mode)
        ]


class EvalPlan(BaseModel):
    """What a run would cover: documents, and whether field accuracy can be computed."""

    model_config = ConfigDict(extra="forbid", strict=True)

    options: EvalOptions
    n_docs: int
    labels_dir: Path
    n_labels: int

    @property
    def field_accuracy(self) -> bool:
        return self.n_labels > 0

    def describe(self) -> list[str]:
        o = self.options
        retry = "off" if o.no_retry else "on"
        native = "on" if o.json_mode else "off"
        accuracy = f"{self.n_labels} label files in {self.labels_dir}" if self.field_accuracy else (
            f"skipped (no labels in {self.labels_dir})"
        )
        return [
            f"model: {o.model}   setup {o.setup} (retry {retry}, native {native})",
            f"documents: {self.n_docs} in {o.data_dir}" + (f" (limit {o.limit})" if o.limit else ""),
            f"field accuracy: {accuracy}",
        ]


def labels_dir_for(data_dir: Path) -> Path:
    """Gold labels sit beside the JD folder: data/jds -> data/labels (ARCHITECTURE.md layout)."""
    return data_dir.parent / "labels"


def plan_eval(options: EvalOptions) -> EvalPlan:
    """Count the documents and labels. Raises FileNotFoundError if the folder is missing or has no .txt files."""
    if not options.data_dir.is_dir():
        raise FileNotFoundError(f"data directory not found: {options.data_dir}")
    docs = sorted(options.data_dir.glob("*.txt"))
    if not docs:
        raise FileNotFoundError(f"no .txt job descriptions in {options.data_dir}")
    n_docs = min(len(docs), options.limit) if options.limit else len(docs)
    labels = options.labels_dir or labels_dir_for(options.data_dir)
    n_labels = len(list(labels.glob("*.json"))) if labels.is_dir() else 0
    return EvalPlan(options=options, n_docs=n_docs, labels_dir=labels, n_labels=n_labels)


def run_eval(plan: EvalPlan) -> None:
    """Phase 6. Not implemented yet."""
    raise NotImplementedError("soa eval is not implemented yet (Phase 6); the plan above is all it does today")

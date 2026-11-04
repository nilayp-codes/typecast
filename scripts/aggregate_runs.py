"""Aggregate repeated eval runs: mean and min-max per metric, per model and setup.

    uv run python scripts/aggregate_runs.py results/A.json results/B.json ... [--out results/aggregate.md]

Takes the `results/*.json` reports of several runs (any mix of models and setups; runs are grouped by model and
setup, so pass the repeats of each together). Prints a table and writes it to results/aggregate.md (this file is
derived from the reports and is rewritten each time). Cells read `mean (min-max)`, with full precision kept in the
reports and one decimal shown here. A report with no value for a metric (not supported, no labels) is left out of
that metric's mean; a metric with no values at all reads `n/a`. Exit code: 0 ok, 2 on an unreadable report or no input.
"""

import argparse
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from soa.eval import EvalReport

OUT_PATH = Path("results/aggregate.md")


def _norm(attribute: str) -> Callable[[EvalReport], float | None]:
    def get(report: EvalReport) -> float | None:
        accuracy = report.accuracy_normalized
        return None if accuracy is None else float(getattr(accuracy, attribute))

    return get


def _e2e(report: EvalReport) -> float | None:
    return None if report.end_to_end_normalized is None else report.end_to_end_normalized.overall


# (column, how to read it from a report, unit shown after each number)
METRICS: list[tuple[str, Callable[[EvalReport], float | None], str]] = [
    ("first-try", lambda r: r.first_try_valid_pct, "%"),
    ("final", lambda r: r.final_valid_pct, "%"),
    ("avg calls", lambda r: r.avg_calls, ""),
    ("field acc (norm)", _norm("overall"), "%"),
    ("coverage", lambda r: r.coverage_pct, "%"),
    ("e2e acc (norm)", _e2e, "%"),
    ("skills recall (norm)", _norm("skills_recall"), "%"),
    ("skills F1 (norm)", _norm("skills_f1"), "%"),
    ("cost", lambda r: r.total_cost_usd, "$"),
]


def _number(x: float, unit: str) -> str:
    if unit == "$":
        return f"${x:.4f}"
    if unit == "%":
        return f"{x:.1f}%"
    return f"{x:.2f}"


def format_cell(values: list[float], unit: str) -> str:
    """`mean (min-max)`; `n/a` with no values. `$` is a prefix with four decimals, `%` a suffix with one."""
    if not values:
        return "n/a"
    mean = sum(values) / len(values)
    return f"{_number(mean, unit)} ({_number(min(values), unit)}-{_number(max(values), unit)})"


def load_reports(paths: list[Path]) -> list[tuple[Path, EvalReport]]:
    """Each file as an EvalReport. Raises ValueError naming the file when one cannot be read or parsed."""
    loaded: list[tuple[Path, EvalReport]] = []
    for path in paths:
        try:
            loaded.append((path, EvalReport.model_validate_json(path.read_text(encoding="utf-8"))))
        except (OSError, ValidationError) as exc:
            raise ValueError(f"{path}: not a readable eval report ({type(exc).__name__})") from exc
    return loaded


def group_reports(loaded: list[tuple[Path, EvalReport]]) -> dict[tuple[str, str], list[tuple[Path, EvalReport]]]:
    groups: dict[tuple[str, str], list[tuple[Path, EvalReport]]] = defaultdict(list)
    for path, report in loaded:
        groups[(report.model, report.setup)].append((path, report))
    return dict(sorted(groups.items()))


def render(groups: dict[tuple[str, str], list[tuple[Path, EvalReport]]]) -> str:
    """The markdown: one table row per model and setup, then the files behind each row."""
    header = ["model", "setup", "runs", "n"] + [name for name, _, _ in METRICS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for (model, setup), members in groups.items():
        reports = [r for _, r in members]
        sizes = sorted({r.n for r in reports})
        cells = [model, setup, str(len(reports)), "/".join(str(s) for s in sizes)]
        for _, read, unit in METRICS:
            cells.append(format_cell([v for r in reports if (v := read(r)) is not None], unit))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Cells: mean (min-max) over the runs. Files:"]
    for (model, setup), members in groups.items():
        lines.append(f"- {model} {setup}: " + ", ".join(f"{p.name} ({r.eval_id})" for p, r in members))
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mean and min-max per metric over repeated eval runs.")
    parser.add_argument("reports", nargs="+", type=Path, help="results/*.json files")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    try:
        groups = group_reports(load_reports(args.reports))
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    text = render(groups)
    print(text, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

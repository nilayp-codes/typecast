"""scripts/aggregate_runs.py: grouping, mean and min-max, n/a handling, output file."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from test_eval import Labels, doc, line, posting, report

from soa.eval import EvalReport

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agg = load_script("aggregate_runs")


def saved(tmp_path: Path, name: str, r: EvalReport) -> Path:
    path = tmp_path / name
    path.write_text(r.model_dump_json(indent=2), encoding="utf-8")
    return path


def base() -> EvalReport:
    labels = Labels(by_doc={"doc-r1": posting()}, skipped=[])
    result: EvalReport = report([doc("r1")], [line("r1")], labels)
    return result


def test_mean_and_range_over_three_runs(tmp_path: Path) -> None:
    runs = [(80.0, 1.0, 0.002), (90.0, 1.5, 0.004), (100.0, 2.0, 0.006)]
    files = [
        saved(tmp_path, f"r{i}.json", base().model_copy(update={"first_try_valid_pct": a, "avg_calls": c, "total_cost_usd": k}))
        for i, (a, c, k) in enumerate(runs)
    ]
    out = tmp_path / "aggregate.md"
    assert agg.main([*map(str, files), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| m | B | 3 | 1 | 90.0% (80.0%-100.0%) |" in text  # first-try: mean 90, min 80, max 100
    assert "1.50 (1.00-2.00)" in text  # avg calls
    assert "$0.0040 ($0.0020-$0.0060)" in text  # cost


def test_runs_are_grouped_by_model_and_setup(tmp_path: Path) -> None:
    a = saved(tmp_path, "a.json", base())
    b = saved(tmp_path, "b.json", base().model_copy(update={"setup": "C"}))
    c = saved(tmp_path, "c.json", base().model_copy(update={"model": "other"}))
    out = tmp_path / "aggregate.md"
    assert agg.main([str(a), str(b), str(c), "--out", str(out)]) == 0
    rows = [x for x in out.read_text(encoding="utf-8").splitlines() if x.startswith("| ") and "---" not in x][1:]
    assert [r.split(" | ")[:3] for r in rows] == [["| m", "B", "1"], ["| m", "C", "1"], ["| other", "B", "1"]]


def test_a_metric_with_no_values_reads_na_and_none_values_are_left_out() -> None:
    assert agg.format_cell([], "%") == "n/a"
    assert agg.format_cell([50.0, 70.0], "%") == "60.0% (50.0%-70.0%)"


def test_unlabelled_runs_have_no_accuracy_cells(tmp_path: Path) -> None:
    f = saved(tmp_path, "a.json", report([doc("r1")], [line("r1")]))
    out = tmp_path / "aggregate.md"
    assert agg.main([str(f), "--out", str(out)]) == 0
    assert "n/a" in out.read_text(encoding="utf-8")


def test_an_unreadable_report_is_an_error_naming_the_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert agg.main([str(bad), "--out", str(tmp_path / "x.md")]) == 2
    assert "bad.json" in capsys.readouterr().out
    assert not (tmp_path / "x.md").exists()

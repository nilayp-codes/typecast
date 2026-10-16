"""results/<date>_<model>_<setup>.json and the results/summary.md row (fixtures from test_eval)."""

import json
from pathlib import Path

import pytest
from test_eval import VALID, Labels, doc, line, posting, report

from soa.eval import SUMMARY_HEADER, EvalReport, model_slug, summary_row, write_results


@pytest.mark.parametrize(
    ("model", "slug"),
    [
        ("openrouter/qwen/qwen-2.5-7b-instruct", "openrouter_qwen_qwen-2.5-7b-instruct"),
        ("openrouter/qwen/qwen3-235b-a22b-2507", "openrouter_qwen_qwen3-235b-a22b-2507"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("a:b c//d", "a_b_c_d"),
        ("/lead/trail/", "lead_trail"),
    ],
)
def test_model_slug(model: str, slug: str) -> None:
    assert model_slug(model) == slug


# --- summary row ----------------------------------------------------------------


def test_row_for_the_fixture_run_without_labels() -> None:
    # 5 docs, 4 scored; first-try 2/4, final 3/4, 7 responses / 4 docs, cost 7 x 0.001
    assert summary_row(report()) == "| m | B | 5 (4 scored) | 50.0% | 75.0% | 1.75 | n/a | $0.0070 |\n"


def test_row_shows_normalized_field_accuracy_and_plain_n_when_every_doc_was_scored() -> None:
    docs = [doc("r1")]
    r = report(docs, [line("r1")], Labels(by_doc={"doc-r1": posting()}, skipped=[]))
    assert summary_row(r) == "| m | B | 1 | 100.0% | 100.0% | 1.00 | 100.0% | $0.0010 |\n"


def test_row_uses_n_a_for_unknown_cost_and_for_unscored_runs() -> None:
    unknown = report([doc("r1")], [line("r1", known=False)])
    assert summary_row(unknown).endswith("| n/a |\n")
    nothing = report([doc("r4", "api_error")], [])
    assert summary_row(nothing) == "| m | B | 1 (0 scored) | n/a | n/a | n/a | n/a | $0.0000 |\n"


def test_a_pipe_in_the_model_name_does_not_break_the_table() -> None:
    r = report([doc("r1")], [line("r1")]).model_copy(update={"model": "a|b"})
    assert summary_row(r).startswith("| a\\|b | B |")


def test_rates_are_formatted_to_one_decimal_without_rounding_up_the_stored_value() -> None:
    logs = [line("r1"), line("r2"), line("r3", ok=False)]
    r = report([doc("r1"), doc("r2"), doc("r3", "invalid")], logs)
    assert r.final_valid_pct == pytest.approx(200 / 3)  # the report keeps full precision
    assert "| 66.7% |" in summary_row(r)


# --- files ------------------------------------------------------------------------


def test_write_results_saves_the_full_report_and_a_summary_row(tmp_path: Path) -> None:
    results = tmp_path / "results"  # created on demand
    r = report()
    path = write_results(r, results)
    assert path == results / "2026-10-17_m_B.json"
    assert EvalReport.model_validate_json(path.read_text(encoding="utf-8")) == r  # nothing lost
    summary = (results / "summary.md").read_text(encoding="utf-8")
    assert summary == SUMMARY_HEADER + summary_row(r)
    assert summary.splitlines()[0] == "| model | setup | n | first-try | final | avg calls | field acc (norm) | cost |"


def test_the_json_report_carries_the_documents_and_the_breakdown(tmp_path: Path) -> None:
    data = json.loads(write_results(report(), tmp_path).read_text(encoding="utf-8"))
    assert [d["status"] for d in data["docs"]] == ["valid", "valid", "invalid", "api_error", "valid"]
    assert data["failures"]["by_error_type"] == {"ValidationError": 4}
    assert data["docs"][0]["raw_posting"]["title"] == VALID["title"]
    assert data["n_api_failed"] == 1 and data["api_error_count"] == 4


def test_a_second_run_appends_a_row_and_never_overwrites_the_first_file(tmp_path: Path) -> None:
    first = write_results(report(), tmp_path)
    before = first.read_text(encoding="utf-8")
    second = write_results(report(), tmp_path)
    third = write_results(report(), tmp_path)
    assert [p.name for p in (first, second, third)] == ["2026-10-17_m_B.json", "2026-10-17_m_B_2.json", "2026-10-17_m_B_3.json"]
    assert first.read_text(encoding="utf-8") == before
    rows = (tmp_path / "summary.md").read_text(encoding="utf-8").splitlines()
    assert rows[:2] == SUMMARY_HEADER.splitlines() and len(rows) == 2 + 3  # one header, three rows


def test_different_setups_and_models_get_their_own_files(tmp_path: Path) -> None:
    base = report()
    other_setup = base.model_copy(update={"setup": "D"})
    other_model = base.model_copy(update={"model": "openrouter/qwen/qwen-2.5-7b-instruct"})
    names = {write_results(r, tmp_path).name for r in (base, other_setup, other_model)}
    assert names == {"2026-10-17_m_B.json", "2026-10-17_m_D.json", "2026-10-17_openrouter_qwen_qwen-2.5-7b-instruct_B.json"}


def test_an_existing_summary_without_a_trailing_newline_still_gets_a_clean_row(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text(SUMMARY_HEADER.rstrip("\n"), encoding="utf-8")
    write_results(report(), tmp_path)
    lines = (tmp_path / "summary.md").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("| model |") and lines[1].startswith("|---|") and lines[2].startswith("| m | B |")
    assert len(lines) == 3


def test_an_empty_summary_file_gets_the_header(tmp_path: Path) -> None:
    (tmp_path / "summary.md").write_text("", encoding="utf-8")
    write_results(report(), tmp_path)
    assert (tmp_path / "summary.md").read_text(encoding="utf-8").startswith(SUMMARY_HEADER)

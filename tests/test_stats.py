"""Confidence intervals: known Wilson values, seeded bootstrap, and how reports carry them."""

import pytest
from test_eval import doc, line, posting, report

from soa.eval import Labels, summary_row
from soa.stats import bootstrap_interval, wilson_interval


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (3, 3, 43.8, 100.0),  # textbook values for 3/3 and 0/10
        (0, 10, 0.0, 27.8),
        (26, 30, 70.3, 94.7),
        (30, 30, 88.6, 100.0),
        (15, 30, 33.2, 66.8),
    ],
)
def test_wilson_matches_known_values(successes: int, n: int, low: float, high: float) -> None:
    got_low, got_high = wilson_interval(successes, n)
    assert got_low == pytest.approx(low, abs=0.1)
    assert got_high == pytest.approx(high, abs=0.1)


@pytest.mark.parametrize(("successes", "n"), [(1, 0), (-1, 5), (6, 5)])
def test_wilson_rejects_impossible_counts(successes: int, n: int) -> None:
    with pytest.raises(ValueError, match="successes"):
        wilson_interval(successes, n)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def test_bootstrap_is_seeded_so_the_same_input_gives_the_same_interval() -> None:
    items = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    assert bootstrap_interval(items, mean) == bootstrap_interval(items, mean)
    assert bootstrap_interval(items, mean, seed=1) != bootstrap_interval(items, mean, seed=2)


def test_bootstrap_interval_brackets_the_estimate_and_stays_inside_the_data_range() -> None:
    items = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    low, high = bootstrap_interval(items, mean)
    assert low < mean(items) < high
    assert 10.0 <= low and high <= 80.0


def test_bootstrap_of_identical_items_is_a_point() -> None:
    assert bootstrap_interval([5.0, 5.0, 5.0], mean) == (5.0, 5.0)


def test_bootstrap_needs_items() -> None:
    with pytest.raises(ValueError, match="at least one"):
        bootstrap_interval([], mean)


def test_report_carries_wilson_for_validity_and_bootstrap_for_accuracy() -> None:
    docs = [doc("r1"), doc("r2")]
    labels = Labels(by_doc={"doc-r1": posting(), "doc-r2": posting(title="Something else")}, skipped=[])
    r = report(docs, [line("r1"), line("r2")], labels)
    assert r.intervals is not None
    assert r.intervals.first_try is not None and r.intervals.first_try.low == pytest.approx(34.2, abs=0.1)  # 2/2
    assert r.intervals.field_accuracy is not None and r.intervals.end_to_end is not None
    assert r.intervals.field_accuracy.low <= r.accuracy_normalized.overall <= r.intervals.field_accuracy.high
    assert r.intervals.end_to_end.low <= r.end_to_end_normalized.overall <= r.intervals.end_to_end.high
    assert r == report(docs, [line("r1"), line("r2")], labels)  # deterministic


def test_report_without_labels_has_validity_intervals_only() -> None:
    r = report([doc("r1")], [line("r1")])
    assert r.intervals is not None
    assert r.intervals.first_try is not None and r.intervals.final is not None
    assert r.intervals.field_accuracy is None and r.intervals.end_to_end is None


def test_summary_row_shows_the_interval_after_the_figure() -> None:
    row = summary_row(report([doc("r1")], [line("r1")]))
    assert "| 100.0% [20.7-100.0] | 100.0% [20.7-100.0] |" in row


def test_summary_row_of_a_report_from_before_intervals_is_unchanged() -> None:
    old = report([doc("r1")], [line("r1")]).model_copy(update={"intervals": None})
    assert summary_row(old).startswith("| m | B | 1 | 100.0% | 100.0% | 1.00 |")

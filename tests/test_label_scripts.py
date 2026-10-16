"""scripts/make_label_templates.py and scripts/check_labels.py (file-based, no LLM involved)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from soa.schemas import JobPosting, Location, Skill

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make = load_script("make_label_templates")
check = load_script("check_labels")

VALID: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": None,
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}],
    "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
    "salary_mentioned": False,
}


def leaves(node: Any) -> list[Any]:
    if isinstance(node, dict):
        return [x for v in node.values() for x in leaves(v)]
    if isinstance(node, list):
        return [x for v in node for x in leaves(v)]
    return [node]


# --- make_label_templates -------------------------------------------------


def test_template_has_every_schema_field_and_only_todo_values() -> None:
    data = json.loads(make.template_text())
    assert set(data) == set(JobPosting.model_fields)
    assert set(data["location"]) == set(Location.model_fields)
    assert set(data["skills"][0]) == set(Skill.model_fields)
    assert leaves(data) == ["TODO"] * len(leaves(data))  # nothing guessed


def test_template_fails_validation_on_purpose() -> None:
    with pytest.raises(ValidationError) as caught:
        JobPosting.model_validate_json(make.template_text())
    failing = {err["loc"][0] for err in caught.value.errors()}
    assert {"seniority", "min_years_experience", "salary_mentioned", "location", "skills"} <= failing


def test_templates_are_created_only_for_jds_without_a_label(tmp_path: Path) -> None:
    jds, labels = tmp_path / "jds", tmp_path / "labels"
    jds.mkdir()
    labels.mkdir()
    for stem in ("jd_001", "jd_002", "jd_003"):
        (jds / f"{stem}.txt").write_text("x", encoding="utf-8")
    (jds / "manifest.csv").write_text("not a jd", encoding="utf-8")
    done = json.dumps(VALID)
    (labels / "jd_002.json").write_text(done, encoding="utf-8")

    created, existing = make.make_templates(jds, labels)

    assert [p.name for p in created] == ["jd_001.json", "jd_003.json"]
    assert existing == 1
    assert (labels / "jd_002.json").read_text(encoding="utf-8") == done  # never overwritten
    assert sorted(p.name for p in labels.iterdir()) == ["jd_001.json", "jd_002.json", "jd_003.json"]
    assert make.make_templates(jds, labels) == ([], 3)  # second run changes nothing


def test_make_creates_the_labels_folder_and_reports_a_missing_jd_folder(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jds = tmp_path / "jds"
    jds.mkdir()
    (jds / "jd_001.txt").write_text("x", encoding="utf-8")
    assert make.main(["--jd-dir", str(jds), "--labels-dir", str(tmp_path / "new" / "labels")]) == 0
    assert (tmp_path / "new" / "labels" / "jd_001.json").is_file()
    assert "1 template(s) created" in capsys.readouterr().out
    assert make.main(["--jd-dir", str(tmp_path / "nope"), "--labels-dir", str(tmp_path)]) == 2


# --- check_labels ---------------------------------------------------------


def test_a_filled_label_is_ok(tmp_path: Path) -> None:
    path = tmp_path / "jd_001.json"
    path.write_text(json.dumps(VALID), encoding="utf-8")
    assert check.check_file(path) == ("ok", [])


def test_a_fresh_template_lists_every_todo_location(tmp_path: Path) -> None:
    path = tmp_path / "jd_001.json"
    path.write_text(make.template_text(), encoding="utf-8")
    status, where = check.check_file(path)
    assert status == "TODO"
    assert where == [
        "title", "company", "seniority", "min_years_experience", "skills.0.name", "skills.0.required",
        "location.city", "location.country", "location.mode", "salary_mentioned",
    ]


def test_a_partly_filled_label_still_counts_as_todo(tmp_path: Path) -> None:
    path = tmp_path / "jd_001.json"
    path.write_text(json.dumps({**VALID, "title": "TODO: check the company page"}), encoding="utf-8")
    assert check.check_file(path) == ("TODO", ["title"])  # valid for the schema, still not finished


def test_a_label_that_fails_validation_is_invalid_with_one_line_per_error(tmp_path: Path) -> None:
    path = tmp_path / "jd_001.json"
    path.write_text(json.dumps({**VALID, "seniority": "wizard", "min_years_experience": "3"}), encoding="utf-8")
    status, lines = check.check_file(path)
    assert status == "invalid"
    assert len(lines) == 2
    assert any(line.startswith("seniority: Input should be") for line in lines)
    assert any(line.startswith("min_years_experience:") and "(got: '3')" in line for line in lines)  # strict mode


def test_a_label_that_is_not_json_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "jd_001.json"
    path.write_text("{not json", encoding="utf-8")
    status, lines = check.check_file(path)
    assert status == "invalid" and lines[0].startswith("cannot read as JSON")


def test_check_main_exit_codes_and_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    labels = tmp_path / "labels"
    labels.mkdir()
    assert check.main(["--labels-dir", str(tmp_path / "nope")]) == 2
    assert check.main(["--labels-dir", str(labels)]) == 1  # no labels is not "ready"
    (labels / "jd_001.json").write_text(json.dumps(VALID), encoding="utf-8")
    assert check.main(["--labels-dir", str(labels)]) == 0
    (labels / "jd_002.json").write_text(make.template_text(), encoding="utf-8")
    (labels / "jd_003.json").write_text(json.dumps({**VALID, "seniority": "wizard"}), encoding="utf-8")
    capsys.readouterr()
    assert check.main(["--labels-dir", str(labels)]) == 1
    out = capsys.readouterr().out
    assert "ok      jd_001.json" in out and "TODO    jd_002.json" in out and "invalid jd_003.json" in out
    assert "3 label(s): 1 ok, 1 still with TODO, 1 invalid" in out

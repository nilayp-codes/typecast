"""scripts/make_blind_templates.py and scripts/label_agreement.py (file-based, no LLM involved)."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


blind = load_script("make_blind_templates")
agree = load_script("label_agreement")
make = load_script("make_label_templates")

LABEL: dict[str, Any] = {
    "title": "Backend Engineer",
    "company": "Acme",
    "seniority": "mid",
    "min_years_experience": 3,
    "skills": [{"name": "Python", "required": True}, {"name": "SQL", "required": False}],
    "location": {"city": "Pune", "country": "India", "mode": "hybrid"},
    "salary_mentioned": False,
}


def write(folder: Path, doc_id: str, **overrides: Any) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{doc_id}.json").write_text(json.dumps({**LABEL, **overrides}), encoding="utf-8")


# --- make_blind_templates ---------------------------------------------------


def jds(tmp_path: Path, *ids: str) -> Path:
    folder = tmp_path / "jds"
    folder.mkdir()
    for doc_id in ids:
        (folder / f"{doc_id}.txt").write_text("text", encoding="utf-8")
    return folder


def test_default_ids_are_the_five_blind_docs() -> None:
    assert blind.BLIND_IDS == ["jd_007", "jd_016", "jd_021", "jd_025", "jd_029"]


def test_templates_are_all_todo_and_identical_to_the_gold_templates(tmp_path: Path) -> None:
    jd_dir = jds(tmp_path, *blind.BLIND_IDS)
    out = tmp_path / "blind"
    assert blind.main(["--jd-dir", str(jd_dir), "--blind-dir", str(out)]) == 0
    assert sorted(p.stem for p in out.glob("*.json")) == sorted(blind.BLIND_IDS)
    for path in out.glob("*.json"):
        assert path.read_text(encoding="utf-8") == make.template_text()


def test_no_gold_value_can_reach_a_blind_template(tmp_path: Path) -> None:
    jd_dir = jds(tmp_path, "jd_007")
    gold = tmp_path / "labels"
    write(gold, "jd_007", title="SECRET GOLD TITLE")
    out = tmp_path / "blind"
    assert blind.main(["--jd-dir", str(jd_dir), "--blind-dir", str(out), "--ids", "jd_007"]) == 0
    assert "SECRET GOLD TITLE" not in (out / "jd_007.json").read_text(encoding="utf-8")


def test_an_existing_blind_label_is_never_overwritten(tmp_path: Path) -> None:
    jd_dir = jds(tmp_path, "jd_007")
    out = tmp_path / "blind"
    write(out, "jd_007", title="Mine")
    assert blind.main(["--jd-dir", str(jd_dir), "--blind-dir", str(out), "--ids", "jd_007"]) == 0
    assert json.loads((out / "jd_007.json").read_text(encoding="utf-8"))["title"] == "Mine"


def test_a_missing_jd_is_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jd_dir = jds(tmp_path)
    assert blind.main(["--jd-dir", str(jd_dir), "--blind-dir", str(tmp_path / "b"), "--ids", "jd_007"]) == 1
    assert "no JD file for jd_007" in capsys.readouterr().out


# --- label_agreement --------------------------------------------------------


def run_agreement(tmp_path: Path) -> tuple[int, str]:
    out = tmp_path / "agreement.md"
    code = agree.main(["--gold-dir", str(tmp_path / "gold"), "--blind-dir", str(tmp_path / "blind"), "--out", str(out)])
    return code, out.read_text(encoding="utf-8") if out.exists() else ""


def test_identical_labels_agree_fully(tmp_path: Path) -> None:
    for doc_id in ("jd_007", "jd_016"):
        write(tmp_path / "gold", doc_id)
        write(tmp_path / "blind", doc_id)
    code, text = run_agreement(tmp_path)
    assert code == 0
    assert "| overall | 100.0% |" in text and "| skills_f1 | 100.0% |" in text
    assert "jd_007: identical on every scored field" in text


def test_one_differing_field_and_one_missing_skill_are_scored_by_hand(tmp_path: Path) -> None:
    write(tmp_path / "gold", "jd_007")
    write(tmp_path / "blind", "jd_007", seniority="senior", skills=[{"name": "Python", "required": True}])
    write(tmp_path / "gold", "jd_016")
    write(tmp_path / "blind", "jd_016")
    code, text = run_agreement(tmp_path)
    assert code == 0
    assert "| seniority | 50.0% |" in text  # 1 of 2 docs
    assert "| title | 100.0% |" in text
    # skills: gold 4 (2+2), blind 3 (1+2): tp 3, fp 0, fn 1 -> precision 100, recall 75, F1 = 6/7 = 85.7
    assert "| skills_recall | 75.0% |" in text and "| skills_f1 | 85.7% |" in text
    assert "jd_007: seniority; skills only in gold: sql" in text


def test_todo_blind_labels_are_not_compared(tmp_path: Path) -> None:
    write(tmp_path / "gold", "jd_007")
    write(tmp_path / "gold", "jd_016")
    write(tmp_path / "blind", "jd_007")
    (tmp_path / "blind" / "jd_016.json").write_text(make.template_text(), encoding="utf-8")
    code, text = run_agreement(tmp_path)
    assert code == 0
    assert "Docs compared: 1 (jd_007)" in text
    assert "not compared (invalid or still TODO): jd_016" in text


def test_nothing_to_compare_is_exit_1_and_writes_no_file(tmp_path: Path) -> None:
    write(tmp_path / "gold", "jd_007")
    (tmp_path / "blind").mkdir()
    code, text = run_agreement(tmp_path)
    assert code == 1 and text == ""


def test_a_missing_folder_is_exit_2(tmp_path: Path) -> None:
    write(tmp_path / "gold", "jd_007")
    assert run_agreement(tmp_path)[0] == 2

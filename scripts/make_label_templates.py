"""Create a gold-label template for every job description that has none yet (Phase 6 hand-labelling).

    uv run python scripts/make_label_templates.py [--jd-dir data/jds] [--labels-dir data/labels]

Writes <labels-dir>/<jd stem>.json for each <jd-dir>/*.txt without one. Existing label files are never touched.
Nothing is guessed from the JD text: every value is the placeholder "TODO", chosen so the template FAILS validation
against JobPosting (bad enum, non-integer years, non-boolean flags) until a person has filled it in.
Fill the labels by hand (seniority rule: docs/DESIGN.md section 1), then run scripts/check_labels.py.
"""

import argparse
from pathlib import Path

from soa.schemas import JobPosting, Location, Skill

TODO = "TODO"
JD_DIR = Path("data/jds")
LABELS_DIR = Path("data/labels")


def template() -> JobPosting:
    """The placeholder posting. model_construct skips validation on purpose: it is not a valid JobPosting."""
    return JobPosting.model_construct(
        title=TODO,
        company=TODO,
        seniority=TODO,
        min_years_experience=TODO,
        skills=[Skill.model_construct(name=TODO, required=TODO)],
        location=Location.model_construct(city=TODO, country=TODO, mode=TODO),
        salary_mentioned=TODO,
    )


def template_text() -> str:
    # warnings=False: the serializer would otherwise warn that "TODO" is not a Seniority / WorkMode / int / bool
    return template().model_dump_json(indent=2, warnings=False) + "\n"


def make_templates(jd_dir: Path, labels_dir: Path) -> tuple[list[Path], int]:
    """Write a template for each JD without a label. Returns (files created, number of JDs already labelled)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    existing = 0
    for jd in sorted(jd_dir.glob("*.txt")):
        target = labels_dir / f"{jd.stem}.json"
        if target.exists():
            existing += 1
            continue
        with target.open("x", encoding="utf-8", newline="\n") as handle:  # "x": never overwrite
            handle.write(template_text())
        created.append(target)
    return created, existing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create TODO label templates for unlabelled job descriptions.")
    parser.add_argument("--jd-dir", type=Path, default=JD_DIR)
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    args = parser.parse_args(argv)
    if not args.jd_dir.is_dir():
        print(f"error: JD folder not found: {args.jd_dir}")
        return 2
    created, existing = make_templates(args.jd_dir, args.labels_dir)
    for path in created:
        print(f"created {path}")
    print(f"{len(created)} template(s) created, {existing} JD(s) already had a label")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

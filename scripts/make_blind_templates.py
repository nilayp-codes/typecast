"""Create blind-label templates for a second, independent labelling pass (label-quality check).

    uv run python scripts/make_blind_templates.py [--jd-dir data/jds] [--blind-dir data/labels_blind] [--ids jd_007 ...]

Writes <blind-dir>/<id>.json for each chosen JD (default: jd_007, jd_016, jd_021, jd_025, jd_029) as an all-"TODO"
template, exactly the same placeholder as scripts/make_label_templates.py. It reads NOTHING from data/labels/: no
gold value can reach a blind template, so the second labeller starts from the JD text alone. An existing file is never
touched. Fill each one by hand (rules: docs/DESIGN.md "Gold-label guide"), then run scripts/label_agreement.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_label_templates import template_text  # noqa: E402  (a sibling script, not a package)

JD_DIR = Path("data/jds")
BLIND_DIR = Path("data/labels_blind")
BLIND_IDS = ["jd_007", "jd_016", "jd_021", "jd_025", "jd_029"]


def make_blind_templates(jd_dir: Path, blind_dir: Path, ids: list[str]) -> tuple[list[Path], list[Path], list[str]]:
    """Write a TODO template per id. Returns (created, already there, ids with no JD file)."""
    blind_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    existing: list[Path] = []
    missing: list[str] = []
    for doc_id in ids:
        if not (jd_dir / f"{doc_id}.txt").is_file():
            missing.append(doc_id)
            continue
        target = blind_dir / f"{doc_id}.json"
        try:
            with target.open("x", encoding="utf-8", newline="\n") as handle:  # "x": never overwrite
                handle.write(template_text())
        except FileExistsError:
            existing.append(target)
            continue
        created.append(target)
    return created, existing, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create TODO templates for a blind second labelling pass.")
    parser.add_argument("--jd-dir", type=Path, default=JD_DIR)
    parser.add_argument("--blind-dir", type=Path, default=BLIND_DIR)
    parser.add_argument("--ids", nargs="+", default=BLIND_IDS, help="doc ids to label blind")
    args = parser.parse_args(argv)
    if not args.jd_dir.is_dir():
        print(f"error: JD folder not found: {args.jd_dir}")
        return 2
    created, existing, missing = make_blind_templates(args.jd_dir, args.blind_dir, args.ids)
    for path in created:
        print(f"created {path}")
    for path in existing:
        print(f"kept {path} (already exists, not touched)")
    for doc_id in missing:
        print(f"error: no JD file for {doc_id} in {args.jd_dir}")
    print(f"{len(created)} template(s) created, {len(existing)} already existed, {len(missing)} missing")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())

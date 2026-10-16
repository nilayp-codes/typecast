"""Validate the gold labels in data/labels/ against JobPosting and list the ones still holding a TODO.

    uv run python scripts/check_labels.py [--labels-dir data/labels]

Per file: ok, TODO (the placeholder from make_label_templates.py is still somewhere in it), or invalid (not
JSON, or JobPosting validation failed; one line per error, as in the retry feedback). Validation is strict, the
same as for model output. Exit code: 0 when every label is ok, 1 when any is not (or there are none), 2 when the
folder is missing.
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from soa.errors import format_validation_errors
from soa.schemas import JobPosting

LABELS_DIR = Path("data/labels")
TODO = "TODO"


def find_todos(node: object, where: str = "") -> list[str]:
    """Paths (like `skills.0.name`) of every string in parsed JSON that contains TODO. Raw JSON on purpose:
    the placeholder can sit in a value that is the wrong type for the model, so it must be found before validating."""
    if isinstance(node, str):
        return [where or "(root)"] if TODO in node else []
    if isinstance(node, dict):
        return [p for key, value in node.items() for p in find_todos(value, f"{where}.{key}" if where else str(key))]
    if isinstance(node, list):
        return [p for i, value in enumerate(node) for p in find_todos(value, f"{where}.{i}" if where else str(i))]
    return []


def check_file(path: Path) -> tuple[str, list[str]]:
    """(status, detail lines): status is "ok", "TODO" or "invalid"."""
    try:
        text = path.read_text(encoding="utf-8-sig")
        todos = find_todos(json.loads(text))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "invalid", [f"cannot read as JSON: {exc}"]
    if todos:
        return "TODO", todos
    try:
        JobPosting.model_validate_json(text)
    except ValidationError as exc:
        return "invalid", format_validation_errors(exc).splitlines()
    return "ok", []


def report(results: dict[Path, tuple[str, list[str]]]) -> None:
    for path, (status, detail) in results.items():
        print(f"{status:7} {path.name}")
        if status == "TODO":
            print(f"        TODO at: {', '.join(detail)}")
        else:
            for line in detail:
                print(f"        {line}")
    counts = [sum(1 for status, _ in results.values() if status == s) for s in ("ok", "TODO", "invalid")]
    print(f"{len(results)} label(s): {counts[0]} ok, {counts[1]} still with TODO, {counts[2]} invalid")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate gold labels and list the ones with TODO left.")
    parser.add_argument("--labels-dir", type=Path, default=LABELS_DIR)
    args = parser.parse_args(argv)
    if not args.labels_dir.is_dir():
        print(f"error: labels folder not found: {args.labels_dir}")
        return 2
    files = sorted(args.labels_dir.glob("*.json"))
    if not files:
        print(f"no label files in {args.labels_dir}; run scripts/make_label_templates.py first")
        return 1
    results = {path: check_file(path) for path in files}
    report(results)
    return 0 if all(status == "ok" for status, _ in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

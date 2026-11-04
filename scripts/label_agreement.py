"""Compare the blind second labels with the gold labels, with the same per-field metrics as the eval.

    uv run python scripts/label_agreement.py [--gold-dir data/labels] [--blind-dir data/labels_blind]
                                             [--out results/label_agreement.md]

For every doc that has a usable label in BOTH folders (valid JobPosting, no TODO left; see eval.load_labels), the gold
label is the reference and the blind label stands in for the model's output, scored by `eval.score_fields`: exact
match per field (title and company case-insensitive), skills precision / recall / F1 on casefolded names, and
`overall`. Skills precision and recall swap if the roles swap; F1 and the other fields do not. Also lists, per doc, the
fields that differ. Writes results/label_agreement.md (rewritten each time). Exit code: 0 ok, 1 if no doc could be
compared, 2 if a folder is missing.
"""

import argparse
from pathlib import Path

from soa.eval import FIELD_MATCHERS, FieldAccuracy, Labels, load_labels, score_fields
from soa.schemas import JobPosting

GOLD_DIR = Path("data/labels")
BLIND_DIR = Path("data/labels_blind")
OUT_PATH = Path("results/label_agreement.md")
FIELD_NAMES = [*FIELD_MATCHERS, "skills_precision", "skills_recall", "skills_f1", "overall"]


def compared_ids(gold: Labels, blind: Labels) -> list[str]:
    """Doc ids with a usable label in both folders, in name order."""
    return sorted(set(gold.by_doc) & set(blind.by_doc))


def differences(gold: JobPosting, blind: JobPosting) -> list[str]:
    """Human-readable differences for one doc: fields that do not match, then skills only one labeller listed."""
    found = [name for name, match in FIELD_MATCHERS.items() if not match(gold, blind)]
    gold_skills = {s.name.strip().casefold() for s in gold.skills}
    blind_skills = {s.name.strip().casefold() for s in blind.skills}
    if gold_skills - blind_skills:
        found.append("skills only in gold: " + ", ".join(sorted(gold_skills - blind_skills)))
    if blind_skills - gold_skills:
        found.append("skills only in blind: " + ", ".join(sorted(blind_skills - gold_skills)))
    return found


def render(gold: Labels, blind: Labels, ids: list[str], accuracy: FieldAccuracy) -> str:
    """The markdown report: per-field agreement, then per-doc differences, then what was skipped."""
    rows = [f"| {name} | {getattr(accuracy, name):.1f}% |" for name in FIELD_NAMES]
    lines = [
        "# Label agreement (blind second pass vs gold)", "",
        f"Docs compared: {accuracy.n} ({', '.join(ids)}). Gold = reference, blind = scored like model output.",
        "With this few docs one disagreement moves a field by 20 points: read it as a sanity check, not a rate.", "",
        "| field | agreement |", "|---|---|", *rows, "", "## Differences per doc", "",
    ]
    for doc_id in ids:
        diff = differences(gold.by_doc[doc_id], blind.by_doc[doc_id])
        lines.append(f"- {doc_id}: " + ("; ".join(diff) if diff else "identical on every scored field"))
    skipped = sorted(set(blind.skipped))
    if skipped:
        lines += ["", "Blind labels not compared (invalid or still TODO): " + ", ".join(skipped)]
    only_blind = sorted(set(blind.by_doc) - set(gold.by_doc))
    if only_blind:
        lines += ["", "Blind labels with no usable gold label: " + ", ".join(only_blind)]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-field agreement of blind labels with the gold labels.")
    parser.add_argument("--gold-dir", type=Path, default=GOLD_DIR)
    parser.add_argument("--blind-dir", type=Path, default=BLIND_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args(argv)
    for folder in (args.gold_dir, args.blind_dir):
        if not folder.is_dir():
            print(f"error: labels folder not found: {folder}")
            return 2
    gold, blind = load_labels(args.gold_dir), load_labels(args.blind_dir)
    ids = compared_ids(gold, blind)
    if not ids:
        print(f"no doc has a usable label in both {args.gold_dir} and {args.blind_dir}; fill the blind templates first")
        return 1
    accuracy = score_fields([(gold.by_doc[i], blind.by_doc[i]) for i in ids])
    text = render(gold, blind, ids, accuracy)
    print(text, end="")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

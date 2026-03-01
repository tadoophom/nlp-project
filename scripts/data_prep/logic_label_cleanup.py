"""Rule-based label cleanup for protein-disease relation datasets."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


NEGATIVE_PATTERNS = [
    r"\bno association\b",
    r"\bnot associated\b",
    r"\bno correlation\b",
    r"\bnot correlated\b",
    r"\bunrelated\b",
    r"\bindependent of\b",
    r"\bno significant (association|relationship|effect)\b",
    r"\bfailed to (show|demonstrate)\b",
]

POSITIVE_PATTERNS = [
    r"\bassociated with\b",
    r"\blinked to\b",
    r"\bcorrelated with\b",
    r"\bpredict(ed|ive)?\b",
    r"\bincrease(d)?\b",
    r"\belevated\b",
    r"\bupregulated\b",
]

INCIDENTAL_HINTS = [
    r"\breview\b",
    r"\bbackground\b",
    r"\bintroduction\b",
    r"\bobjective\b",
    r"\bmethod(s)?\b",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def rule_suggested_label(sentence: str) -> str | None:
    if has_any(sentence, NEGATIVE_PATTERNS):
        return "not_associated"
    if has_any(sentence, POSITIVE_PATTERNS):
        return "associated"
    if has_any(sentence, INCIDENTAL_HINTS):
        return "incidental"
    return None


def resolve_duplicate_labels(rows: list[dict]) -> tuple[list[dict], dict]:
    by_sentence: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = normalize(row["sentence"])
        by_sentence[key].append(row)

    cleaned = []
    stats = Counter()
    for key, group in by_sentence.items():
        if len(group) == 1:
            cleaned.append(group[0])
            stats["unique_kept"] += 1
            continue

        counts = Counter(item["label"] for item in group)
        sentence = group[0]["sentence"]
        suggestion = rule_suggested_label(sentence)

        if suggestion and suggestion in counts:
            chosen = suggestion
            stats["duplicates_resolved_by_rule"] += 1
        else:
            chosen = counts.most_common(1)[0][0]
            stats["duplicates_resolved_by_majority"] += 1

        cleaned.append({"sentence": sentence, "label": chosen})
        stats["duplicates_collapsed"] += len(group) - 1

    return cleaned, dict(stats)


def relabel_by_rules(rows: list[dict], strict: bool) -> tuple[list[dict], dict]:
    cleaned = []
    stats = Counter()

    for row in rows:
        sentence = row["sentence"]
        label = row["label"]
        suggestion = rule_suggested_label(sentence)

        if suggestion is None:
            cleaned.append(row)
            continue

        if label == suggestion:
            cleaned.append(row)
            continue

        if strict:
            # In strict mode, drop obvious contradictions instead of relabeling.
            cleaned.append({"sentence": sentence, "label": suggestion})
            stats["relabelled"] += 1
        else:
            cleaned.append({"sentence": sentence, "label": suggestion})
            stats["relabelled"] += 1

    return cleaned, dict(stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based dataset label cleanup")
    parser.add_argument("--input", required=True, help="Input JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--strict", action="store_true", help="Use strict contradiction handling")
    parser.add_argument(
        "--report",
        default="",
        help="Optional output report JSON path (default: <output>.report.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report) if args.report else output_path.with_suffix(".report.json")

    with open(input_path) as f:
        rows = json.load(f)

    original_size = len(rows)
    original_dist = Counter(row["label"] for row in rows)

    deduped, dedupe_stats = resolve_duplicate_labels(rows)
    relabelled, relabel_stats = relabel_by_rules(deduped, strict=args.strict)

    final_dist = Counter(row["label"] for row in relabelled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(relabelled, f, indent=2)

    report = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "original_size": original_size,
        "final_size": len(relabelled),
        "original_distribution": dict(original_dist),
        "final_distribution": dict(final_dist),
        "dedupe_stats": dedupe_stats,
        "relabel_stats": relabel_stats,
        "strict_mode": args.strict,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Input samples: {original_size}")
    print(f"Output samples: {len(relabelled)}")
    print(f"Original distribution: {dict(original_dist)}")
    print(f"Final distribution: {dict(final_dist)}")
    print(f"Saved cleaned data to {output_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

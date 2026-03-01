"""Combine base, expansion, and pseudo-labeled data into one training mix."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def normalize_sentence(text: str) -> str:
    return " ".join(text.split()).strip()


def main():
    parser = argparse.ArgumentParser(description="Build training mix")
    parser.add_argument("--base", required=True, help="Base train JSON")
    parser.add_argument("--pseudo", nargs="+", required=True, help="Pseudo labels JSON files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--pseudo-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pseudo-per-label", type=int, default=250)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    base_rows = load_json(args.base)
    pseudo_rows = []
    for pseudo_path in args.pseudo:
        pseudo_rows.extend(load_json(pseudo_path))

    base = [{"sentence": r["sentence"], "label": r["label"], "source": "base"} for r in base_rows]
    pseudo = [{
        "sentence": r["sentence"],
        "label": r["label"],
        "source": r.get("source", "pseudo"),
        "min_confidence": r.get("min_confidence", None),
    } for r in pseudo_rows]

    pseudo_by_label: dict[str, list[dict]] = {"associated": [], "not_associated": [], "incidental": []}
    for row in pseudo:
        if row["label"] in pseudo_by_label:
            pseudo_by_label[row["label"]].append(row)

    target_total = int(len(base) * max(args.pseudo_fraction, 0.0))
    selected_pseudo = []
    for label, rows in pseudo_by_label.items():
        rows = sorted(rows, key=lambda x: (x.get("min_confidence") or 0.0), reverse=True)
        keep = min(len(rows), args.max_pseudo_per_label, max(1, target_total // 3))
        selected_pseudo.extend(rows[:keep])

    if len(selected_pseudo) > target_total > 0:
        selected_pseudo = rng.sample(selected_pseudo, target_total)

    merged = []
    seen = set()
    duplicate_dropped = 0

    for row in base + selected_pseudo:
        sent = normalize_sentence(row["sentence"])
        if sent in seen:
            duplicate_dropped += 1
            continue
        seen.add(sent)
        merged.append({"sentence": sent, "label": row["label"]})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    report = {
        "base_count": len(base),
        "base_distribution": dict(Counter(r["label"] for r in base)),
        "pseudo_input_count": len(pseudo),
        "pseudo_selected_count": len(selected_pseudo),
        "pseudo_selected_distribution": dict(Counter(r["label"] for r in selected_pseudo)),
        "duplicate_dropped": duplicate_dropped,
        "output_count": len(merged),
        "output_distribution": dict(Counter(r["label"] for r in merged)),
        "pseudo_fraction": args.pseudo_fraction,
        "max_pseudo_per_label": args.max_pseudo_per_label,
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("report", report)
    print("saved_output", out_path)
    print("saved_report", report_path)


if __name__ == "__main__":
    main()

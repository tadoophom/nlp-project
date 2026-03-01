"""Export manual acceptance sheet from audit bucket files."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_json(path: str) -> list[dict]:
    with open(path) as f:
        rows = json.load(f)
    if isinstance(rows, list):
        return rows
    return []


def score_row(row: dict) -> float:
    if "assoc_expert_prob_associated" in row:
        return float(row.get("assoc_expert_prob_associated", 0.0))
    mean_probs = row.get("mean_probabilities", {})
    label = row.get("label", "")
    if label and isinstance(mean_probs, dict):
        return float(mean_probs.get(label, 0.0))
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Export manual acceptance sheet")
    parser.add_argument("--inputs", nargs="+", required=True, help="Audit JSON files")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        for row in read_json(path):
            sentence = row.get("sentence", "").strip()
            if not sentence:
                continue
            label = row.get("label", "")
            source_group = row.get("group", Path(path).stem)
            rec_score = score_row(row)
            recommended = "yes" if rec_score >= 0.75 else "review"
            rows.append(
                {
                    "source_file": path,
                    "source_group": source_group,
                    "sentence": sentence,
                    "suggested_label": label,
                    "confidence_hint": rec_score,
                    "recommended_accept": recommended,
                    "accept": "",
                    "reviewer_label": "",
                    "notes": "",
                }
            )

    rows.sort(key=lambda x: x["confidence_hint"], reverse=True)
    rows = rows[: args.top_k]

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "source_file",
            "source_group",
            "sentence",
            "suggested_label",
            "confidence_hint",
            "recommended_accept",
            "accept",
            "reviewer_label",
            "notes",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("rows_exported", len(rows))
    print("saved", out)


if __name__ == "__main__":
    main()

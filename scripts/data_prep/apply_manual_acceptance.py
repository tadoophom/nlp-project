"""Apply manually accepted rows from review sheet into a training mix."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def normalize(text: str) -> str:
    return " ".join(text.split()).strip()


def main():
    parser = argparse.ArgumentParser(description="Apply manual acceptance sheet")
    parser.add_argument("--base", required=True, help="Base training JSON")
    parser.add_argument("--sheet", required=True, help="Reviewed CSV sheet")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    with open(args.base) as f:
        base_rows = json.load(f)
    base = [{"sentence": normalize(r["sentence"]), "label": r["label"]} for r in base_rows]
    seen = {r["sentence"] for r in base}

    accepted = []
    with open(args.sheet) as f:
        reader = csv.DictReader(f)
        for row in reader:
            accept = str(row.get("accept", "")).strip().lower()
            if accept not in {"yes", "y", "1", "true"}:
                continue
            sentence = normalize(row.get("sentence", ""))
            if not sentence or sentence in seen:
                continue
            label = (row.get("reviewer_label") or row.get("suggested_label") or "").strip()
            if label not in {"associated", "not_associated", "incidental"}:
                continue
            accepted.append({"sentence": sentence, "label": label})
            seen.add(sentence)

    merged = base + accepted

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(merged, f, indent=2)

    report = {
        "base_count": len(base),
        "accepted_count": len(accepted),
        "output_count": len(merged),
        "accepted_distribution": dict(Counter(r["label"] for r in accepted)),
        "output_distribution": dict(Counter(r["label"] for r in merged)),
    }
    rep = Path(args.output_report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "w") as f:
        json.dump(report, f, indent=2)

    print("report", report)
    print("saved_output", out)
    print("saved_report", rep)


if __name__ == "__main__":
    main()

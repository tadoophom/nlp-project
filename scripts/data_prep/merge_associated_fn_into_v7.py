"""Merge only the 21 associated-labeled rows from the FN-targeted queue into v7 train (no-leak)."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def main():
    train_path = Path("data/splits/hfpef_v7_train_relabel3_noleak_large.json")
    eval_path = Path("data/splits/hfpef_v7_eval_relabel3_noleak_large.json")
    labeled_path = Path("data/labeling/97e4cdc9-53ab-4970-874e-effed3386077_labeled.csv")

    with open(train_path) as f:
        train = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    eval_sents = {r["sentence"].strip().lower() for r in eval_data}
    train_sents = {r["sentence"].strip().lower() for r in train}

    new_rows = []
    skipped_leak = 0
    skipped_dup = 0
    with open(labeled_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["manual_label"].strip()
            if label != "associated":
                continue
            sent = row["sentence"].strip()
            key = sent.lower()
            if key in eval_sents:
                skipped_leak += 1
                continue
            if key in train_sents:
                skipped_dup += 1
                continue
            new_rows.append({"sentence": sent, "label": label})

    print(f"v7_train: {len(train)}")
    print(f"new_associated: {len(new_rows)}")
    print(f"skipped_leak: {skipped_leak}")
    print(f"skipped_dup: {skipped_dup}")

    merged = train + new_rows
    from collections import Counter
    dist = Counter(r["label"] for r in merged)
    print(f"merged_total: {len(merged)}")
    print(f"merged_distribution: {dict(dist)}")

    out_path = Path("data/splits/hfpef_v7_train_plus_assoc_fn21_noleak.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"written: {out_path}")

    report = {
        "source_train": str(train_path),
        "source_labels": str(labeled_path),
        "v7_train_size": len(train),
        "new_associated_added": len(new_rows),
        "skipped_leak": skipped_leak,
        "skipped_dup": skipped_dup,
        "merged_total": len(merged),
        "merged_distribution": dict(dist),
    }
    report_path = out_path.with_suffix(".report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

"""Export flagged eval labels as a relabeling spreadsheet."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main():
    audit_path = Path("data/label_audit_v7_eval.json")
    eval_path = Path("data/splits/hfpef_v7_eval_relabel3_noleak_large.json")
    output_path = Path("data/labeling/eval_relabel_audit_88.csv")

    with open(audit_path) as f:
        audit = json.load(f)
    with open(eval_path) as f:
        eval_data = json.load(f)

    # Only include examples where majority disagrees with gold
    flagged = [a for a in audit if a["majority_pred"] != a["gold_label"]]
    flagged.sort(key=lambda x: x["suspicion_score"], reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "eval_index",
            "full_sentence",
            "current_label",
            "model_majority_vote",
            "models_agreeing_with_current",
            "total_models",
            "suspicion_score",
            "avg_prob_associated",
            "avg_prob_not_associated",
            "avg_prob_incidental",
            "new_label",
            "notes",
        ])

        for item in flagged:
            idx = item["index"]
            full_sentence = eval_data[idx]["sentence"]
            writer.writerow([
                idx,
                full_sentence,
                item["gold_label"],
                item["majority_pred"],
                item["n_agree_gold"],
                item["n_models"],
                f"{item['suspicion_score']:.3f}",
                f"{item['avg_probs']['associated']:.3f}",
                f"{item['avg_probs']['not_associated']:.3f}",
                f"{item['avg_probs']['incidental']:.3f}",
                "",  # new_label - to be filled in
                "",  # notes - to be filled in
            ])

    print(f"Exported {len(flagged)} flagged examples to {output_path}")

    # Breakdown
    from collections import Counter
    by_gold = Counter(item["gold_label"] for item in flagged)
    by_transition = Counter(
        f"{item['gold_label']} -> {item['majority_pred']}" for item in flagged
    )
    print(f"\nBy current label:")
    for label, count in by_gold.most_common():
        print(f"  {label}: {count}")
    print(f"\nBy transition (current -> model vote):")
    for transition, count in by_transition.most_common():
        print(f"  {transition}: {count}")


if __name__ == "__main__":
    main()

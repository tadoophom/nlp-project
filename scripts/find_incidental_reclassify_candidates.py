"""Find incidental sentences that may warrant reclassification to not_associated."""

from __future__ import annotations

import csv
from pathlib import Path

CSV_PATH = Path("data/labeling/hfpef_labeling_balanced_v3.csv")
OUTPUT_PATH = Path("/tmp/incidental_reclassify_candidates.csv")

NEGATION_PHRASES = [
    "not associated", "no significant", "no association", "no correlation",
    "not correlated", "no relationship", "failed to", "did not predict",
    "did not differ", "not significant", "no change", "no effect",
    "not predictive", "no evidence", "unrelated", "no difference",
    "not different", "but not in", "but not hfpef", "independent of",
]

HFPEF_TERMS = [
    "hfpef", "heart failure with preserved ejection fraction",
    "heart failure with preserved ef", "hf-pef", "hf with preserved ejection fraction",
]

DIASTOLIC_TERMS = ["diastolic dysfunction", "diastolic heart failure", "diastolic hf"]


def find_negation_phrase(text):
    text_lower = text.lower()
    for phrase in NEGATION_PHRASES:
        if phrase in text_lower:
            return phrase
    return None


def has_hfpef_or_diastolic(text):
    text_lower = text.lower()
    for term in HFPEF_TERMS + DIASTOLIC_TERMS:
        if term in text_lower:
            return True
    return False


def main():
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    label_counts = {}
    for row in rows:
        label = row["manual_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    print(f"Total rows: {len(rows)}")
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    print()

    incidental = [(i, row) for i, row in enumerate(rows) if row["manual_label"] == "incidental"]
    print(f"Incidental rows: {len(incidental)}")
    print()

    candidates = []
    for idx, row in incidental:
        sentence = row["sentence"]
        title = row.get("title", "")
        has_strong = row.get("has_strong_hfpef", "False") == "True"
        neg_phrase = find_negation_phrase(sentence)
        if neg_phrase is None:
            continue
        sentence_has_hfpef = has_hfpef_or_diastolic(sentence)
        title_has_hfpef = has_hfpef_or_diastolic(title)
        if sentence_has_hfpef or title_has_hfpef or has_strong:
            reason_parts = []
            if sentence_has_hfpef:
                reason_parts.append("sentence mentions HFpEF/diastolic")
            if title_has_hfpef:
                reason_parts.append("title mentions HFpEF/diastolic")
            if has_strong:
                reason_parts.append("has_strong_hfpef=True")
            candidates.append({
                "original_index": idx, "pmid": row["pmid"], "protein": row["protein"],
                "sentence": sentence, "title": title, "negation_phrase": neg_phrase,
                "has_strong_hfpef": has_strong, "reason": "; ".join(reason_parts),
            })

    print(f"Candidates for reclassification: {len(candidates)}")
    print("=" * 100)
    for c in candidates:
        print()
        oi = c["original_index"]
        pm = c["pmid"]
        pr = c["protein"]
        neg = c["negation_phrase"]
        reason = c["reason"]
        sent = c["sentence"][:200]
        ttl = c["title"][:120]
        print(f"[idx={oi}] PMID={pm} | Protein={pr}")
        print(f"  Negation: \"{neg}\"")
        print(f"  Reason:   {reason}")
        print(f"  Sentence: {sent}")
        print(f"  Title:    {ttl}")
        print("-" * 100)

    out_fields = ["original_index", "pmid", "protein", "sentence", "title",
                  "negation_phrase", "has_strong_hfpef", "reason"]
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(candidates)
    print(f"\nSaved {len(candidates)} candidates to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

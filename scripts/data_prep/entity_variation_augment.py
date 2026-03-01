"""Entity-variation augmentation for relation classification data."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


PROTEIN_POOL = [
    "TNF-alpha", "IL-6", "CRP", "BNP", "Troponin T", "ACE", "VEGF-A",
    "Endothelin-1", "TGF-beta1", "MMP-9", "Galectin-3", "ST2",
]

DISEASE_SYNONYMS = {
    "hfpef": ["heart failure with preserved ejection fraction", "HFpEF", "diastolic heart failure"],
    "heart failure with preserved ejection fraction": ["HFpEF", "diastolic heart failure"],
    "ischemic heart disease": ["coronary artery disease", "myocardial ischemia"],
    "arrhythmia": ["cardiac arrhythmia", "rhythm disorder"],
    "cardiomyopathy": ["myocardial disease", "cardiac myopathy"],
    "valvular disease": ["heart valve disease", "valve disorder"],
}

PROTEIN_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9\-]{1,11}\b")


def replace_disease_terms(sentence: str, rng: random.Random) -> str:
    out = sentence
    lower = sentence.lower()
    for term, options in DISEASE_SYNONYMS.items():
        if term in lower:
            replacement = rng.choice(options)
            out = re.sub(re.escape(term), replacement, out, flags=re.IGNORECASE)
            break
    return out


def replace_protein_token(sentence: str, rng: random.Random) -> str:
    matches = list(PROTEIN_TOKEN.finditer(sentence))
    if not matches:
        return sentence

    chosen = rng.choice(matches)
    original = chosen.group(0)
    replacement = rng.choice([p for p in PROTEIN_POOL if p.lower() != original.lower()])
    return sentence[:chosen.start()] + replacement + sentence[chosen.end():]


def augment_sentence(sentence: str, rng: random.Random) -> str:
    variant = replace_disease_terms(sentence, rng)
    variant = replace_protein_token(variant, rng)
    variant = re.sub(r"\s+", " ", variant).strip()
    return variant


def main():
    parser = argparse.ArgumentParser(description="Entity-variation data augmentation")
    parser.add_argument("--input", required=True, help="Input JSON dataset")
    parser.add_argument("--output", required=True, help="Output augmented JSON dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-associated", type=int, default=1000)
    parser.add_argument("--target-not-associated", type=int, default=600)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.input) as f:
        rows = json.load(f)

    by_label: dict[str, list[dict]] = {"associated": [], "not_associated": [], "incidental": []}
    for row in rows:
        label = row["label"]
        if label in by_label:
            by_label[label].append(row)

    seen = {row["sentence"] for row in rows}
    out = list(rows)
    stats = Counter()

    targets = {
        "associated": args.target_associated,
        "not_associated": args.target_not_associated,
    }

    for label, target in targets.items():
        source = by_label[label]
        if not source:
            continue
        current = len(source)
        needed = max(0, target - current)
        attempts = 0

        while needed > 0 and attempts < target * 20:
            attempts += 1
            row = rng.choice(source)
            aug = augment_sentence(row["sentence"], rng)
            if aug == row["sentence"]:
                continue
            if aug in seen:
                continue

            seen.add(aug)
            out.append({"sentence": aug, "label": label})
            needed -= 1
            stats[f"{label}_added"] += 1

    out_dist = Counter(row["label"] for row in out)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Input samples: {len(rows)}")
    print(f"Output samples: {len(out)}")
    print(f"Added samples: {dict(stats)}")
    print(f"Output distribution: {dict(out_dist)}")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

"""Mine uncertain/high-value candidates for manual or LLM-assisted labeling."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES


def iter_candidate_sentences(corpus_csv: Path):
    with open(corpus_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sent = row.get("evidence_sentence", "").strip()
            if len(sent) >= 30:
                yield sent


def main():
    parser = argparse.ArgumentParser(description="Mine uncertain candidates from corpus")
    parser.add_argument("--corpus", required=True, help="Corpus CSV path")
    parser.add_argument("--model", required=True, help="Classifier model path")
    parser.add_argument("--output", required=True, help="Output JSON candidates")
    parser.add_argument("--max-candidates", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    clf = PubMedBERTClassifier(model_path=Path(args.model))
    sentences = list(dict.fromkeys(iter_candidate_sentences(Path(args.corpus))))
    print(f"Candidate unique sentences: {len(sentences)}")

    candidates = []
    for i in range(0, len(sentences), args.batch_size):
        batch = sentences[i:i + args.batch_size]
        results = clf.classify_batch(batch, batch_size=args.batch_size)
        for sent, result in zip(batch, results):
            probs = [result["probabilities"][label] for label in LABEL_NAMES]
            probs_sorted = sorted(probs)
            margin = probs_sorted[-1] - probs_sorted[-2]
            confidence = result["confidence"]
            score = (1.0 - confidence) + (0.5 - margin)

            candidates.append({
                "sentence": sent,
                "predicted_label": result["label"],
                "confidence": confidence,
                "margin": margin,
                "uncertainty_score": score,
            })

    candidates.sort(key=lambda x: x["uncertainty_score"], reverse=True)
    candidates = candidates[:args.max_candidates]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2)

    print(f"Saved {len(candidates)} candidates to {out_path}")


if __name__ == "__main__":
    main()

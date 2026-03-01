"""Audit eval labels using multi-model disagreement and uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return -np.sum(p * np.log(p))


def main():
    parser = argparse.ArgumentParser(description="Audit eval labels via model ensemble disagreement")
    parser.add_argument("--eval", required=True, help="Eval split JSON")
    parser.add_argument("--models", nargs="+", required=True, help="Model checkpoint dirs")
    parser.add_argument("--output", required=True, help="Output JSON with flagged examples")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.eval) as f:
        rows = json.load(f)
    sentences = [r["sentence"] for r in rows]
    gold_labels = [r["label"] for r in rows]

    # Collect predictions from all models
    all_probs = []  # shape: (n_models, n_examples, n_classes)
    all_preds = []  # shape: (n_models, n_examples)

    for model_path in args.models:
        print(f"Running {model_path}...")
        clf = PubMedBERTClassifier(model_path=Path(model_path))
        pred_rows = clf.classify_batch(sentences, batch_size=args.batch_size)
        probs = np.array([[r["probabilities"][l] for l in LABEL_NAMES] for r in pred_rows])
        preds = [r["label"] for r in pred_rows]
        all_probs.append(probs)
        all_preds.append(preds)
        acc = accuracy_score(gold_labels, preds)
        print(f"  accuracy: {acc:.3f}")

    n_models = len(args.models)
    all_probs = np.array(all_probs)  # (n_models, n_examples, 3)

    # Compute per-example metrics
    flagged = []
    for i in range(len(sentences)):
        gold = gold_labels[i]
        model_predictions = [all_preds[m][i] for m in range(n_models)]
        n_agree_gold = sum(1 for p in model_predictions if p == gold)
        n_disagree = n_models - n_agree_gold

        # Average probability distribution across models
        avg_probs = all_probs[:, i, :].mean(axis=0)
        avg_entropy = entropy(avg_probs)

        # Most common model prediction
        from collections import Counter
        vote_counts = Counter(model_predictions)
        majority_pred, majority_count = vote_counts.most_common(1)[0]

        # Confidence in wrong answer: how confident are disagreeing models?
        wrong_confidences = []
        for m in range(n_models):
            if all_preds[m][i] != gold:
                # confidence of model m in its (wrong) prediction
                pred_label = all_preds[m][i]
                pred_idx = LABEL_NAMES.index(pred_label)
                wrong_confidences.append(all_probs[m, i, pred_idx])

        avg_wrong_conf = np.mean(wrong_confidences) if wrong_confidences else 0.0

        # Suspicion score: high when models confidently disagree with gold
        suspicion = (n_disagree / n_models) * (1 + avg_wrong_conf) + avg_entropy

        flagged.append({
            "index": i,
            "sentence": sentences[i][:200],
            "gold_label": gold,
            "majority_pred": majority_pred,
            "majority_count": majority_count,
            "n_agree_gold": n_agree_gold,
            "n_models": n_models,
            "avg_probs": {l: float(avg_probs[j]) for j, l in enumerate(LABEL_NAMES)},
            "avg_entropy": float(avg_entropy),
            "avg_wrong_confidence": float(avg_wrong_conf),
            "suspicion_score": float(suspicion),
            "model_predictions": model_predictions,
        })

    # Sort by suspicion score (highest first)
    flagged.sort(key=lambda x: x["suspicion_score"], reverse=True)

    # Print summary
    print(f"\n{'='*60}")
    print(f"LABEL AUDIT RESULTS ({len(sentences)} examples, {n_models} models)")
    print(f"{'='*60}")

    # Count how many have majority != gold
    n_majority_disagree = sum(1 for f in flagged if f["majority_pred"] != f["gold_label"])
    print(f"\nMajority vote disagrees with gold: {n_majority_disagree}/{len(sentences)}")

    # Show top suspicious
    print(f"\nTop 30 most suspicious labels:")
    print(f"{'idx':>4} {'gold':>15} {'majority':>15} {'agree':>6} {'susp':>6}  sentence")
    print("-" * 100)
    for f in flagged[:30]:
        print(f"{f['index']:4d} {f['gold_label']:>15} {f['majority_pred']:>15} "
              f"{f['n_agree_gold']:>3}/{f['n_models']:<3} {f['suspicion_score']:.3f}  "
              f"{f['sentence'][:60]}...")

    # Breakdown by gold label
    print(f"\nSuspicion by gold label:")
    for label in LABEL_NAMES:
        subset = [f for f in flagged if f["gold_label"] == label]
        avg_susp = np.mean([f["suspicion_score"] for f in subset])
        n_disagree = sum(1 for f in subset if f["majority_pred"] != f["gold_label"])
        print(f"  {label:>15}: avg_suspicion={avg_susp:.3f}, majority_disagrees={n_disagree}/{len(subset)}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(flagged, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

"""Weighted ensemble with per-model weight optimization."""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, ".")
from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)

    sentences = [r["sentence"] for r in eval_data]
    gold = [r["label"] for r in eval_data]

    # Get predictions from all models
    all_probs = []
    individual_results = []
    for model_path in args.models:
        clf = PubMedBERTClassifier(model_path=Path(model_path))
        preds = clf.classify_batch(sentences, batch_size=64)
        probs = np.array([[p["probabilities"][l] for l in LABEL_NAMES] for p in preds])
        all_probs.append(probs)

        pred_labels = [p["label"] for p in preds]
        acc = accuracy_score(gold, pred_labels)
        f1 = f1_score(gold, pred_labels, labels=LABEL_NAMES, average="macro")
        individual_results.append({"model": model_path, "accuracy": acc, "macro_f1": f1})
        print(f"  {Path(model_path).parent.name:45s} acc={acc:.4f} f1={f1:.4f}")
        del clf

    all_probs = np.stack(all_probs)  # (n_models, n_samples, n_classes)
    n_models = len(args.models)

    # Grid search over weights (step 0.1)
    weight_values = np.arange(0.0, 1.05, 0.1)
    best_acc = 0
    best_weights = None
    best_f1 = 0

    for raw_weights in product(weight_values, repeat=n_models):
        w = np.array(raw_weights)
        if w.sum() == 0:
            continue
        w = w / w.sum()
        weighted_probs = np.tensordot(w, all_probs, axes=([0], [0]))
        pred_indices = np.argmax(weighted_probs, axis=1)
        pred_labels = [LABEL_NAMES[i] for i in pred_indices]
        acc = accuracy_score(gold, pred_labels)
        f1 = f1_score(gold, pred_labels, labels=LABEL_NAMES, average="macro")
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_weights = w.tolist()

    # Equal weight baseline
    equal_probs = all_probs.mean(axis=0)
    equal_preds = [LABEL_NAMES[i] for i in np.argmax(equal_probs, axis=1)]
    equal_acc = accuracy_score(gold, equal_preds)
    equal_f1 = f1_score(gold, equal_preds, labels=LABEL_NAMES, average="macro")

    # Best weighted
    w = np.array(best_weights)
    weighted_probs = np.tensordot(w, all_probs, axes=([0], [0]))
    weighted_preds = [LABEL_NAMES[i] for i in np.argmax(weighted_probs, axis=1)]

    print(f"\nEqual-weight ensemble:    acc={equal_acc:.4f} f1={equal_f1:.4f}")
    print(f"Best weighted ensemble:   acc={best_acc:.4f} f1={best_f1:.4f}")
    print(f"Weights: {best_weights}")
    print(f"\n{classification_report(gold, weighted_preds, labels=LABEL_NAMES)}")

    result = {
        "individual": individual_results,
        "equal_weight": {"accuracy": equal_acc, "macro_f1": equal_f1},
        "best_weighted": {
            "accuracy": best_acc,
            "macro_f1": best_f1,
            "weights": dict(zip([str(p) for p in args.models], best_weights)),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

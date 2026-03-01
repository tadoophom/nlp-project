"""Evaluate a classifier checkpoint on a labeled split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on labeled split")
    parser.add_argument("--model", required=True, help="Model checkpoint directory")
    parser.add_argument("--eval", required=True, help="Labeled JSON split")
    parser.add_argument("--output", required=True, help="Output metrics JSON")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    with open(args.eval) as f:
        rows = json.load(f)
    sentences = [r["sentence"] for r in rows]
    y_true = [r["label"] for r in rows]

    clf = PubMedBERTClassifier(model_path=Path(args.model), max_length=args.max_length)
    pred_rows = clf.classify_batch(sentences, batch_size=args.batch_size)
    y_pred = [r["label"] for r in pred_rows]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    payload = {
        "model_path": args.model,
        "eval_split": args.eval,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_NAMES).tolist(),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_NAMES
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("accuracy", payload["accuracy"])
    print("macro_f1", payload["macro_f1"])
    print("weighted_f1", payload["weighted_f1"])
    print("saved", out)


if __name__ == "__main__":
    main()

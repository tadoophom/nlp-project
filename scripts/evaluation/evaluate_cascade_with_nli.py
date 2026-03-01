"""Evaluate a cascade: base classifier + NLI verifier on uncertain cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES
from scripts.evaluation.evaluate_nli_reformulation import predict_labels as nli_predict_labels
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Cascade evaluation with NLI verifier")
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--nli-model", required=True, help="Path to NLI model")
    parser.add_argument("--eval-data", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf-threshold", type=float, default=0.72)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    parser.add_argument("--output", default="", help="Optional metrics JSON output")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        data = json.load(f)
    sentences = [row["sentence"] for row in data]
    true_labels = [row["label"] for row in data]

    base = PubMedBERTClassifier(model_path=Path(args.base_model))
    base_results = base.classify_batch(sentences, batch_size=args.batch_size)

    nli_tokenizer = AutoTokenizer.from_pretrained(args.nli_model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(args.nli_model)
    nli_preds_all = nli_predict_labels(
        model=nli_model,
        tokenizer=nli_tokenizer,
        sentences=sentences,
        batch_size=args.batch_size,
        use_constraints=True,
    )

    final_preds = []
    uncertain_count = 0
    for idx, base_result in enumerate(base_results):
        probs = [base_result["probabilities"][label] for label in LABEL_NAMES]
        sorted_probs = sorted(probs)
        confidence = max(probs)
        margin = sorted_probs[-1] - sorted_probs[-2]

        if confidence < args.conf_threshold or margin < args.margin_threshold:
            final_preds.append(nli_preds_all[idx])
            uncertain_count += 1
        else:
            final_preds.append(base_result["label"])

    acc = accuracy_score(true_labels, final_preds)
    macro_f1 = f1_score(true_labels, final_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, final_preds, average="weighted", zero_division=0)
    report = classification_report(true_labels, final_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, final_preds, labels=LABEL_NAMES).tolist()

    print(f"Cascade accuracy: {acc:.4f}")
    print(f"Cascade macro F1: {macro_f1:.4f}")
    print(f"Cascade weighted F1: {weighted_f1:.4f}")
    print(f"Uncertain cases rerouted to NLI: {uncertain_count}/{len(sentences)}")

    payload = {
        "base_model": args.base_model,
        "nli_model": args.nli_model,
        "eval_data": args.eval_data,
        "conf_threshold": args.conf_threshold,
        "margin_threshold": args.margin_threshold,
        "uncertain_count": uncertain_count,
        "total": len(sentences),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_NAMES
        },
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

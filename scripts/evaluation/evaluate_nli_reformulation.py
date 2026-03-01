"""Evaluate NLI reformulation model on relation classification."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

LABELS = ["associated", "not_associated", "incidental"]
HYPOTHESES = {
    "associated": "The sentence states that the protein is associated with the disease.",
    "not_associated": "The sentence states that the protein is not associated with the disease.",
    "incidental": "The sentence mentions protein and disease but gives no association claim.",
}

NEG_PATTERNS = [
    r"\bno association\b", r"\bnot associated\b", r"\bno correlation\b", r"\bunrelated\b",
]
POS_PATTERNS = [
    r"\bassociated with\b", r"\blinked to\b", r"\bcorrelated with\b", r"\bpredict(ed|ive)?\b",
]


def constrained_scores(sentence: str, scores: dict[str, float]) -> dict[str, float]:
    out = dict(scores)
    if any(re.search(p, sentence, flags=re.IGNORECASE) for p in NEG_PATTERNS):
        out["not_associated"] *= 1.10
        out["associated"] *= 0.95
    if any(re.search(p, sentence, flags=re.IGNORECASE) for p in POS_PATTERNS):
        out["associated"] *= 1.06
    return out


def predict_labels(model, tokenizer, sentences: list[str], batch_size: int, use_constraints: bool) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    predictions = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # Flatten as (premise, hypothesis_k) pairs.
        pair_premise = []
        pair_hyp = []
        for sent in batch:
            for label in LABELS:
                pair_premise.append(sent)
                pair_hyp.append(HYPOTHESES[label])

        inputs = tokenizer(
            pair_premise,
            pair_hyp,
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()  # entailment prob

        for j, sent in enumerate(batch):
            start = j * len(LABELS)
            scores = {label: float(probs[start + k]) for k, label in enumerate(LABELS)}
            if use_constraints:
                scores = constrained_scores(sent, scores)
            best = max(scores.items(), key=lambda x: x[1])[0]
            predictions.append(best)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLI reformulation model")
    parser.add_argument("--model", required=True, help="Path to NLI model final directory")
    parser.add_argument("--eval-data", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="", help="Optional metrics JSON output")
    parser.add_argument("--no-constraints", action="store_true", help="Disable lexical constraints")
    args = parser.parse_args()

    with open(args.eval_data) as f:
        data = json.load(f)
    sentences = [row["sentence"] for row in data]
    true_labels = [row["label"] for row in data]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)

    pred_labels = predict_labels(
        model=model,
        tokenizer=tokenizer,
        sentences=sentences,
        batch_size=args.batch_size,
        use_constraints=not args.no_constraints,
    )

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=LABELS).tolist()

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(classification_report(true_labels, pred_labels, zero_division=0))

    payload = {
        "model": args.model,
        "eval_data": args.eval_data,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABELS
        },
        "constraints": not args.no_constraints,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()

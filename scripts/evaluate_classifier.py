"""Evaluate and compare rule-based vs BERT classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier, LABEL2ID
from src.nlp_utils import load_pipeline, classify_span


def evaluate_rule_based(sentences: list[str], labels: list[str], model_name: str = "en_core_web_sm"):
    """Evaluate rule-based classifier."""
    nlp = load_pipeline(model_name, use_context=True)
    predictions = []

    for sent in sentences:
        doc = nlp(sent)
        # Use first sentence span
        span = list(doc.sents)[0] if list(doc.sents) else doc[:]
        pred = classify_span(span)
        # Map to BERT labels
        pred_mapped = {"Positive": "positive", "Negative": "negative", "Neutral": "no_association"}.get(pred, "no_association")
        predictions.append(pred_mapped)

    return predictions


def evaluate_bert(sentences: list[str], model_path: str | None = None):
    """Evaluate BERT classifier."""
    classifier = PubMedBERTClassifier(model_path=model_path)
    predictions = []

    for label, _ in classifier.predict_batch(sentences):
        predictions.append(label)

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Compare classifiers")
    parser.add_argument("--data", required=True, help="Labeled JSON data")
    parser.add_argument("--bert-model", help="Path to fine-tuned BERT model")
    parser.add_argument("--output", help="Output comparison CSV")
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    sentences = [item["sentence"] for item in data]
    labels = [item["label"] for item in data]

    print(f"Evaluating on {len(sentences)} sentences\n")

    # Rule-based evaluation
    print("=" * 50)
    print("RULE-BASED CLASSIFIER")
    print("=" * 50)
    rule_preds = evaluate_rule_based(sentences, labels)
    print(classification_report(labels, rule_preds, target_names=list(LABEL2ID.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, rule_preds, labels=list(LABEL2ID.keys())))

    # BERT evaluation
    print("\n" + "=" * 50)
    print("PUBMEDBERT CLASSIFIER")
    print("=" * 50)
    bert_preds = evaluate_bert(sentences, model_path=args.bert_model)
    print(classification_report(labels, bert_preds, target_names=list(LABEL2ID.keys())))
    print("Confusion Matrix:")
    print(confusion_matrix(labels, bert_preds, labels=list(LABEL2ID.keys())))

    # Save comparison
    if args.output:
        df = pd.DataFrame({
            "sentence": sentences,
            "true_label": labels,
            "rule_pred": rule_preds,
            "bert_pred": bert_preds,
            "rule_correct": [t == p for t, p in zip(labels, rule_preds)],
            "bert_correct": [t == p for t, p in zip(labels, bert_preds)],
        })
        df.to_csv(args.output, index=False)
        print(f"\nComparison saved to {args.output}")

        # Summary
        rule_acc = sum(df["rule_correct"]) / len(df)
        bert_acc = sum(df["bert_correct"]) / len(df)
        print(f"\nSummary:")
        print(f"  Rule-based accuracy: {rule_acc:.1%}")
        print(f"  BERT accuracy: {bert_acc:.1%}")
        print(f"  Improvement: {(bert_acc - rule_acc):.1%}")


if __name__ == "__main__":
    main()

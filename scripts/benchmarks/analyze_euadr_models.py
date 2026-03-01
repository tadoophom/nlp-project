#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import extract, load_pipeline
from scripts.benchmarks.evaluate_euadr_relations import build_relations, collapse_not_associated


MODELS = {
    "rule_based": None,
    "pubmedbert": "models/pubmedbert-hfpef/final",
    "scibert_v4": "models/scibert-hfpef-v4/final",
}


def normalize_label(label: str) -> str:
    key = (label or "").strip().lower()
    if key == "associated":
        return "associated"
    if key == "not_associated":
        return "not_associated"
    if key == "incidental":
        return "incidental"
    return key


def rule_based_predict(text: str, ent1: str, ent2: str, nlp) -> str:
    hits = extract(text, [ent1, ent2], nlp)
    if not hits:
        return "associated"
    labels = {normalize_label(hit.get("classification", "")) for hit in hits}
    if "not_associated" in labels:
        return "not_associated"
    if "incidental" in labels:
        return "incidental"
    return "associated"


def evaluate_model(name: str, relations: list[dict], rule_nlp) -> dict:
    model_path = MODELS[name]
    classifier = None if model_path is None else PubMedBERTClassifier(model_path=model_path)

    y_true = [rel["label"] for rel in relations]
    y_pred = []
    for i, rel in enumerate(relations, start=1):
        if i % 100 == 0:
            print(f"{name}: processed {i}/{len(relations)}")
        if classifier is None:
            pred = rule_based_predict(rel["text"], rel["ent1"], rel["ent2"], rule_nlp)
        else:
            pred, _ = classifier.predict(rel["text"])
        y_pred.append(normalize_label(pred))

    accuracy = accuracy_score(y_true, y_pred)
    collapsed_pred = collapse_not_associated(y_pred)
    collapsed_accuracy = accuracy_score(y_true, collapsed_pred)

    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    present_labels = ["associated", "incidental"]
    macro_f1_present = float(np.mean([report.get(lbl, {}).get("f1-score", 0.0) for lbl in present_labels]))

    return {
        "name": name,
        "accuracy": accuracy,
        "collapsed_accuracy": collapsed_accuracy,
        "macro_f1_present": macro_f1_present,
        "report": report,
        "pred_counts": dict(Counter(y_pred)),
    }


def plot_model_metrics(metrics: list[dict], output_dir: Path):
    names = [m["name"] for m in metrics]
    accuracy = [m["accuracy"] for m in metrics]
    collapsed = [m["collapsed_accuracy"] for m in metrics]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, accuracy, width, label="Accuracy")
    plt.bar(x + width / 2, collapsed, width, label="Association-only Accuracy")
    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("EU-ADR Accuracy by Model")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "euadr_model_accuracy.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_per_class_f1(metrics: list[dict], output_dir: Path):
    labels = ["associated", "incidental"]
    names = [m["name"] for m in metrics]
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    for i, label in enumerate(labels):
        scores = [m["report"].get(label, {}).get("f1-score", 0.0) for m in metrics]
        plt.bar(x + (i - 0.5) * width, scores, width, label=label)

    plt.xticks(x, names)
    plt.ylim(0, 1)
    plt.ylabel("F1")
    plt.title("EU-ADR F1 by Label")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "euadr_f1_by_label.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_label_distribution(relations: list[dict], best_metrics: dict, output_dir: Path):
    gold_counts = Counter(rel["label"] for rel in relations)
    pred_counts = Counter(best_metrics["pred_counts"])
    labels = ["associated", "incidental", "not_associated"]
    gold = [gold_counts.get(lbl, 0) for lbl in labels]
    pred = [pred_counts.get(lbl, 0) for lbl in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width / 2, gold, width, label="Gold")
    plt.bar(x + width / 2, pred, width, label=f"Pred ({best_metrics['name']})")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title("EU-ADR Label Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = output_dir / "euadr_label_distribution.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default="data/benchmarks/euadr/euadr_bigbio_kb.train.jsonl")
    parser.add_argument("--output-dir", default="deliverable_email/2026-01-26")
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"EU-ADR JSONL not found: {jsonl_path}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EU-ADR relations from {jsonl_path}")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    relations = build_relations(jsonl_path, True, nlp, args.window_size)
    if args.limit:
        relations = relations[: args.limit]
    if not relations:
        print("No relations found.")
        return 1

    print(f"Evaluating {len(relations)} relations across models")
    rule_nlp = load_pipeline("en_core_web_sm", use_context=True)

    metrics = []
    for name in MODELS:
        metrics.append(evaluate_model(name, relations, rule_nlp))

    best = max(metrics, key=lambda m: m["collapsed_accuracy"])

    acc_path = plot_model_metrics(metrics, output_dir)
    f1_path = plot_per_class_f1(metrics, output_dir)
    dist_path = plot_label_distribution(relations, best, output_dir)

    summary_path = output_dir / "euadr_model_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "jsonl_path": str(jsonl_path),
                "rows": len(relations),
                "window_size": args.window_size,
                "metrics": metrics,
                "best_model": best["name"],
                "plots": {
                    "accuracy": str(acc_path),
                    "f1": str(f1_path),
                    "distribution": str(dist_path),
                },
            },
            handle,
            indent=2,
        )
    print(f"Saved metrics to {summary_path}")
    print(f"Plots: {acc_path}, {f1_path}, {dist_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


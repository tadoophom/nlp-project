import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import load_pipeline, classify_span


LABELS = ["associated", "not_associated", "incidental"]


def rule_predictions(sentences):
    label_map = {
        "Associated": "associated",
        "Not_Associated": "not_associated",
        "Incidental": "incidental",
    }
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    preds = []
    for sent in sentences:
        doc = nlp(sent)
        spans = list(doc.sents)
        span = spans[0] if spans else doc[:]
        pred = classify_span(span)
        preds.append(label_map.get(pred, "incidental"))
    return preds


def multilabel_predictions(sentences, model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    preds = []
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0]
        idx = probs.argmax().item()
        preds.append(LABELS[idx])
    return preds


def evaluate(true_labels, pred_labels):
    acc = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
    return acc, report


def main():
    data_path = Path("data/splits/hfpef_quality_relation.json")
    out_dir = Path("results/2026-02-02/hfpef_quality")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(data_path.read_text())
    sentences = [d["sentence"] for d in data]
    true_labels = [d["label"] for d in data]
    label_counts = Counter(true_labels)

    models = {}

    models["Rule-based"] = rule_predictions(sentences)

    pubmed = PubMedBERTClassifier(model_path="models/hfpef_quality/2026-02-02/pubmedbert/final")
    models["PubMedBERT"] = [pubmed.predict(s)[0] for s in sentences]

    scibert = PubMedBERTClassifier(model_path="models/hfpef_quality/2026-02-02/scibert_v4/final")
    models["SciBERT v4"] = [scibert.predict(s)[0] for s in sentences]

    models["SciBERT Multi-label"] = multilabel_predictions(
        sentences,
        "models/hfpef_quality/2026-02-02/scibert_multilabel/final",
    )

    rows = []
    for name, preds in models.items():
        acc, report = evaluate(true_labels, preds)
        rows.append(
            {
                "model": name,
                "accuracy": acc,
                "f1_weighted": report["weighted avg"]["f1-score"],
                "f1_macro": report["macro avg"]["f1-score"],
                "associated_f1": report["associated"]["f1-score"],
                "not_associated_f1": report["not_associated"]["f1-score"],
                "incidental_f1": report["incidental"]["f1-score"],
            }
        )

    metrics_path = out_dir / "model_metrics.json"
    metrics_path.write_text(json.dumps(rows, indent=2))

    table_path = out_dir / "model_metrics.csv"
    with table_path.open("w") as f:
        f.write(
            "model,accuracy,f1_weighted,f1_macro,associated_f1,not_associated_f1,incidental_f1\n"
        )
        for r in rows:
            f.write(
                f"{r['model']},{r['accuracy']:.4f},{r['f1_weighted']:.4f},{r['f1_macro']:.4f},"
                f"{r['associated_f1']:.4f},{r['not_associated_f1']:.4f},{r['incidental_f1']:.4f}\n"
            )

    stats_path = out_dir / "dataset_stats.txt"
    stats_path.write_text(
        "Dataset stats\n"
        f"total_sentences: {len(sentences)}\n"
        f"associated: {label_counts.get('associated', 0)}\n"
        f"not_associated: {label_counts.get('not_associated', 0)}\n"
        f"incidental: {label_counts.get('incidental', 0)}\n"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r["model"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    f1s = [r["f1_weighted"] for r in rows]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, accs, width, label="Accuracy")
    ax.bar(x + width / 2, f1s, width, label="F1 (weighted)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("HFpEF Quality Dataset: Model Metrics")
    fig.tight_layout()
    fig.savefig(out_dir / "model_metrics.png", dpi=150)


if __name__ == "__main__":
    main()

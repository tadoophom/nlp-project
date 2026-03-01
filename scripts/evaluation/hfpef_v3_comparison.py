"""Generate N-model comparison dashboard for HFpEF v3."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier

CLASSES = ["associated", "not_associated", "incidental"]
CLASS_DISPLAY = ["Associated", "Not Associated", "Incidental"]

# Default color palette (extends automatically if needed)
DEFAULT_COLORS = [
    "#e74c3c", "#3498db", "#e67e22", "#2ecc71", "#9b59b6",
    "#1abc9c", "#34495e", "#f39c12", "#d35400", "#7f8c8d",
]


def load_eval_data(path: str) -> tuple[list[str], list[str]]:
    with open(path) as f:
        data = json.load(f)
    return [d["sentence"] for d in data], [d["label"] for d in data]


def get_bert_predictions(model_path: str, sentences: list[str]) -> list[str]:
    clf = PubMedBERTClassifier(model_path=Path(model_path))
    results = clf.classify_batch(sentences)
    return [r["label"] for r in results]


def get_rule_predictions(sentences: list[str]) -> list[str]:
    from src.nlp_utils import load_pipeline, classify_span

    nlp = load_pipeline("en_core_web_sm", use_context=True)
    label_map = {
        "Associated": "associated",
        "Not_Associated": "not_associated",
        "Incidental": "incidental",
    }
    preds = []
    for sent in sentences:
        doc = nlp(sent)
        spans = list(doc.sents)
        span = spans[0] if spans else doc[:]
        pred = classify_span(span)
        preds.append(label_map.get(pred, "incidental"))
    return preds


def get_ensemble_predictions(model_paths: list[str], sentences: list[str]) -> list[str]:
    """Average softmax across models."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_probs = []

    for mp in model_paths:
        tokenizer = AutoTokenizer.from_pretrained(mp)
        model = AutoModelForSequenceClassification.from_pretrained(mp)
        model.eval().to(device)

        probs_list = []
        for i in range(0, len(sentences), 32):
            batch = sentences[i:i+32]
            inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
                probs_list.append(probs)
        all_probs.append(np.concatenate(probs_list, axis=0))
        del model

    avg_probs = np.stack(all_probs).mean(axis=0)
    pred_ids = avg_probs.argmax(axis=-1)
    label_names = list({"associated": 0, "not_associated": 1, "incidental": 2}.keys())
    return [label_names[i] for i in pred_ids]


def compute_all_metrics(true_labels, preds):
    acc = accuracy_score(true_labels, preds)
    report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_labels, preds, labels=CLASSES)
    macro_f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(true_labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": acc,
        "report": report,
        "cm": cm,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def main():
    parser = argparse.ArgumentParser(description="N-model comparison dashboard")
    parser.add_argument("--eval-data", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--models", nargs="*", default=None,
                        help="name:path pairs e.g. PubMedBERT:models/hfpef_v3/pubmedbert/final")
    parser.add_argument("--ensemble-models", nargs="*", default=None,
                        help="Model paths for ensemble averaging")
    parser.add_argument("--include-rule-based", action="store_true", default=True)
    parser.add_argument("--output", default="results/hfpef_v3_comparison.png")
    args = parser.parse_args()

    sentences, true_labels = load_eval_data(args.eval_data)
    label_counts = Counter(true_labels)
    print(f"Eval set: {len(sentences)} samples")
    print(f"Distribution: {dict(label_counts)}")

    model_configs = []
    if args.models:
        for spec in args.models:
            name, path = spec.split(":", 1)
            model_configs.append((name, path))
    else:
        model_configs = [
            ("PubMedBERT", "models/hfpef_v3/pubmedbert/final"),
            ("SciBERT", "models/hfpef_v3/scibert/final"),
        ]
        improved_dir = Path("models/hfpef_v3_improved")
        if improved_dir.exists():
            for d in sorted(improved_dir.iterdir()):
                final = d / "final"
                if final.exists():
                    model_configs.append((d.name, str(final)))

    results = {}
    names = []
    colors = {}

    if args.include_rule_based:
        print("  Rule-based (spaCy)...")
        rule_preds = get_rule_predictions(sentences)
        results["Rule-based"] = compute_all_metrics(true_labels, rule_preds)
        names.append("Rule-based")
        colors["Rule-based"] = DEFAULT_COLORS[0]
        print(f"    Accuracy: {results['Rule-based']['accuracy']:.1%}")

    color_idx = 1
    for name, path in model_configs:
        if not Path(path).exists():
            print(f"  {name}: SKIPPED (not found at {path})")
            continue
        print(f"  {name}...")
        preds = get_bert_predictions(path, sentences)
        results[name] = compute_all_metrics(true_labels, preds)
        names.append(name)
        colors[name] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
        color_idx += 1
        print(f"    Accuracy: {results[name]['accuracy']:.1%}")

    if args.ensemble_models:
        valid_ensemble = [p for p in args.ensemble_models if Path(p).exists()]
        if len(valid_ensemble) >= 2:
            print(f"  Ensemble ({len(valid_ensemble)} models)...")
            ens_preds = get_ensemble_predictions(valid_ensemble, sentences)
            results["Ensemble"] = compute_all_metrics(true_labels, ens_preds)
            names.append("Ensemble")
            colors["Ensemble"] = DEFAULT_COLORS[color_idx % len(DEFAULT_COLORS)]
            print(f"    Accuracy: {results['Ensemble']['accuracy']:.1%}")

    n_models = len(names)
    if n_models == 0:
        print("No models to compare.")
        return

    # Layout
    n_cm_cols = min(n_models, 4)
    n_cm_rows = math.ceil(n_models / n_cm_cols)
    total_rows = 2 + n_cm_rows

    fig = plt.figure(figsize=(max(18, n_models * 4), 5 * total_rows))
    fig.suptitle(
        f"HFpEF Protein-Disease Classification: {n_models}-Model Comparison",
        fontsize=16, fontweight="bold", y=0.99,
    )

    # Row 1: Accuracy | Per-class F1 | Dataset pie
    ax1 = fig.add_subplot(total_rows, 3, 1)
    accs = [results[n]["accuracy"] * 100 for n in names]
    bar_width = 0.6
    bars = ax1.bar(
        range(n_models), accs,
        color=[colors[n] for n in names],
        edgecolor="black", width=bar_width,
    )
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(names, fontsize=8, rotation=30, ha="right")
    ax1.set_ylabel("Accuracy (%)", fontweight="bold")
    ax1.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", fontweight="bold", fontsize=10,
        )
    ax1.set_title("Overall Accuracy", fontweight="bold")

    ax2 = fig.add_subplot(total_rows, 3, 2)
    x = np.arange(len(CLASSES))
    width = 0.8 / n_models
    for i, name in enumerate(names):
        f1s = [results[name]["report"][c]["f1-score"] for c in CLASSES]
        ax2.bar(
            x + i * width, f1s, width,
            label=name, color=colors[name], edgecolor="black",
        )
    ax2.set_ylabel("F1 Score", fontweight="bold")
    ax2.set_xticks(x + width * (n_models - 1) / 2)
    ax2.set_xticklabels(CLASS_DISPLAY)
    ax2.legend(fontsize=6, loc="upper right")
    ax2.set_ylim(0, 1.1)
    ax2.set_title("F1 Score by Class", fontweight="bold")

    ax3 = fig.add_subplot(total_rows, 3, 3)
    pie_values = [label_counts.get(c, 0) for c in CLASSES]
    pie_colors = ["#2ecc71", "#e74c3c", "#95a5a6"]
    wedges, texts, autotexts = ax3.pie(
        pie_values, labels=CLASS_DISPLAY, autopct="%1.0f%%",
        colors=pie_colors, startangle=90,
    )
    for t in autotexts:
        t.set_fontweight("bold")
    ax3.set_title(f"Eval Set (n={len(sentences)})", fontweight="bold")

    # Middle rows: Confusion matrices
    for i, name in enumerate(names):
        row = i // n_cm_cols
        col = i % n_cm_cols
        ax = fig.add_subplot(total_rows, n_cm_cols, n_cm_cols + row * n_cm_cols + col + 1)
        cm = results[name]["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(["Assoc", "Not", "Inc"], fontsize=9)
        ax.set_yticklabels(["Assoc", "Not", "Inc"], fontsize=9)
        for ii in range(3):
            for jj in range(3):
                text_color = "white" if cm[ii, jj] > cm.max() / 2 else "black"
                ax.text(
                    jj, ii, str(cm[ii, jj]),
                    ha="center", va="center",
                    color=text_color, fontweight="bold", fontsize=12,
                )
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Predicted")
        if col == 0:
            ax.set_ylabel("True")

    # Last row: Macro/Weighted F1 + Metrics table
    last_row_start = (1 + n_cm_rows) * 3 + 1

    ax7 = fig.add_subplot(total_rows, 3, last_row_start)
    metrics_x = np.arange(2)
    bar_w = 0.8 / n_models
    for i, name in enumerate(names):
        vals = [results[name]["macro_f1"], results[name]["weighted_f1"]]
        ax7.bar(
            metrics_x + i * bar_w, vals, bar_w,
            label=name, color=colors[name], edgecolor="black",
        )
    ax7.set_xticks(metrics_x + bar_w * (n_models - 1) / 2)
    ax7.set_xticklabels(["Macro F1", "Weighted F1"])
    ax7.set_ylim(0, 1.1)
    ax7.set_ylabel("F1 Score", fontweight="bold")
    ax7.legend(fontsize=6)
    ax7.set_title("Aggregate F1 Scores", fontweight="bold")

    # Metrics table
    ax8_idx = last_row_start + 1
    ax8 = fig.add_subplot(total_rows, 3, (ax8_idx, ax8_idx + 1))
    ax8.axis("off")

    col_w = max(12, 14)
    name_cols = "".join(f"{n:>{col_w}}" for n in names)
    header = f"{'Metric':<16}{name_cols}"
    lines = [header, "-" * (16 + col_w * n_models)]

    for metric_name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("Weighted F1", "weighted_f1")]:
        vals = "".join(f"{results[n][key]:>{col_w}.3f}" for n in names)
        lines.append(f"{metric_name:<16}{vals}")
    lines.append("")

    for cls, display in zip(CLASSES, CLASS_DISPLAY):
        lines.append(display)
        for metric_label, metric_key in [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1-score")]:
            vals = "".join(f"{results[n]['report'][cls][metric_key]:>{col_w}.3f}" for n in names)
            lines.append(f"  {metric_label:<14}{vals}")

    font_size = max(6, min(9, 120 // n_models))
    ax8.text(
        0.02, 0.95, "\n".join(lines),
        transform=ax8.transAxes, fontsize=font_size,
        verticalalignment="top", fontfamily="monospace",
    )
    ax8.set_title("Detailed Metrics", fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\nSaved dashboard to {output_path}")


if __name__ == "__main__":
    main()

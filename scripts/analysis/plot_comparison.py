"""Generate comparison plots for rule-based vs BERT classifier."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier, LABEL2ID
from src.nlp_utils import load_pipeline, classify_span


def load_data(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [d["sentence"] for d in data], [d["label"] for d in data]


def get_rule_predictions(sentences, labels):
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    predictions = []
    label_map = {"Positive": "positive", "Negative": "negative", "Neutral": "no_association"}
    
    for sent in sentences:
        doc = nlp(sent)
        span = list(doc.sents)[0] if list(doc.sents) else doc[:]
        pred = classify_span(span)
        predictions.append(label_map.get(pred, "no_association"))
    
    return predictions


def get_bert_predictions(sentences, model_path):
    classifier = PubMedBERTClassifier(model_path=model_path)
    predictions = []
    for label, _ in classifier.predict_batch(sentences):
        predictions.append(label)
    return predictions


def plot_accuracy_comparison(rule_acc, bert_acc, output_path):
    """Bar chart comparing overall accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    methods = ['Rule-based\n(spaCy + MedspaCy)', 'PubMedBERT\n(Fine-tuned)']
    accuracies = [rule_acc * 100, bert_acc * 100]
    colors = ['#e74c3c', '#27ae60']
    
    bars = ax.bar(methods, accuracies, color=colors, width=0.6, edgecolor='black', linewidth=1.2)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Classification Accuracy: Rule-based vs PubMedBERT', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    
    # Add improvement annotation
    improvement = bert_acc - rule_acc
    ax.annotate(f'+{improvement*100:.1f}%', 
                xy=(1, bert_acc*100), xytext=(1.3, (rule_acc + bert_acc)/2 * 100),
                fontsize=12, color='#27ae60', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#27ae60'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_f1_comparison(rule_report, bert_report, output_path):
    """Grouped bar chart comparing F1 scores per class."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classes = ['positive', 'negative', 'no_association']
    class_labels = ['Positive\nAssociation', 'Negative\nAssociation', 'No\nAssociation']
    
    rule_f1 = [rule_report[c]['f1-score'] for c in classes]
    bert_f1 = [bert_report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rule_f1, width, label='Rule-based', color='#e74c3c', edgecolor='black')
    bars2 = ax.bar(x + width/2, bert_f1, width, label='PubMedBERT', color='#27ae60', edgecolor='black')
    
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', fontsize=11)
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_matrices(true_labels, rule_preds, bert_preds, output_path):
    """Side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    classes = ['positive', 'negative', 'no_association']
    class_labels = ['Positive', 'Negative', 'No Assoc.']
    
    # Rule-based confusion matrix
    cm_rule = confusion_matrix(true_labels, rule_preds, labels=classes)
    sns.heatmap(cm_rule, annot=True, fmt='d', cmap='Reds', ax=axes[0],
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'shrink': 0.8})
    axes[0].set_title('Rule-based (spaCy)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('True', fontsize=11)
    
    # BERT confusion matrix
    cm_bert = confusion_matrix(true_labels, bert_preds, labels=classes)
    sns.heatmap(cm_bert, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=class_labels, yticklabels=class_labels,
                cbar_kws={'shrink': 0.8})
    axes[1].set_title('PubMedBERT (Fine-tuned)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('True', fontsize=11)
    
    plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_precision_recall(rule_report, bert_report, output_path):
    """Precision vs Recall scatter plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    classes = ['positive', 'negative', 'no_association']
    markers = ['o', 's', '^']
    
    for cls, marker in zip(classes, markers):
        # Rule-based
        ax.scatter(rule_report[cls]['recall'], rule_report[cls]['precision'],
                   marker=marker, s=150, c='#e74c3c', edgecolors='black', linewidths=1.5,
                   label=f'{cls} (Rule)' if cls == classes[0] else '')
        # BERT
        ax.scatter(bert_report[cls]['recall'], bert_report[cls]['precision'],
                   marker=marker, s=150, c='#27ae60', edgecolors='black', linewidths=1.5,
                   label=f'{cls} (BERT)' if cls == classes[0] else '')
    
    # Add labels
    for cls, marker in zip(classes, markers):
        ax.annotate(cls[:3], (rule_report[cls]['recall'] + 0.02, rule_report[cls]['precision'] + 0.02),
                    fontsize=8, color='#e74c3c')
        ax.annotate(cls[:3], (bert_report[cls]['recall'] + 0.02, bert_report[cls]['precision'] + 0.02),
                    fontsize=8, color='#27ae60')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision vs Recall by Class', fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=10, label='Rule-based'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#27ae60', markersize=10, label='PubMedBERT'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_summary_dashboard(rule_acc, bert_acc, rule_report, bert_report, output_path):
    """Combined dashboard with all metrics."""
    fig = plt.figure(figsize=(14, 10))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Accuracy comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Rule-based', 'PubMedBERT']
    accuracies = [rule_acc * 100, bert_acc * 100]
    colors = ['#e74c3c', '#27ae60']
    bars = ax1.bar(methods, accuracies, color=colors, width=0.5, edgecolor='black')
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.set_ylim(0, 105)
    
    # 2. F1 by class (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    classes = ['positive', 'negative', 'no_association']
    x = np.arange(len(classes))
    width = 0.35
    rule_f1 = [rule_report[c]['f1-score'] for c in classes]
    bert_f1 = [bert_report[c]['f1-score'] for c in classes]
    ax2.bar(x - width/2, rule_f1, width, label='Rule-based', color='#e74c3c', edgecolor='black')
    ax2.bar(x + width/2, bert_f1, width, label='PubMedBERT', color='#27ae60', edgecolor='black')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score by Class', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Positive', 'Negative', 'No Assoc.'])
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, 1.1)
    
    # 3. Improvement metrics (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    rule_vals = [rule_acc, rule_report['macro avg']['f1-score'], rule_report['weighted avg']['f1-score']]
    bert_vals = [bert_acc, bert_report['macro avg']['f1-score'], bert_report['weighted avg']['f1-score']]
    improvements = [(b - r) * 100 for r, b in zip(rule_vals, bert_vals)]
    
    colors_imp = ['#27ae60' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax3.barh(metrics, improvements, color=colors_imp, edgecolor='black')
    ax3.set_xlabel('Improvement (%)')
    ax3.set_title('BERT Improvement over Rule-based', fontweight='bold')
    ax3.axvline(x=0, color='black', linewidth=0.5)
    for bar, imp in zip(bars, improvements):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 f'+{imp:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    # 4. Key findings (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    findings = [
        f"Overall Accuracy: {rule_acc*100:.1f}% → {bert_acc*100:.1f}%",
        f"Negative Detection: {rule_report['negative']['f1-score']:.0%} → {bert_report['negative']['f1-score']:.0%}",
        f"False Negative Rate: Reduced by {(1-bert_report['negative']['recall'])/(1-rule_report['negative']['recall']+0.001)*100:.0f}%",
        "",
        "Key Advantages of PubMedBERT:",
        "• Understands biomedical context",
        "• Detects negation semantically",
        "• Handles uncertain language",
    ]
    ax4.text(0.1, 0.9, '\n'.join(findings), transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Key Findings', fontweight='bold')
    
    plt.suptitle('PubMedBERT vs Rule-based Classification\nHFpEF Protein-Disease Relations',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labeled.json")
    parser.add_argument("--bert-model", default="models/pubmedbert-hfpef/final")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    sentences, labels = load_data(Path(args.data))
    print(f"Loaded {len(sentences)} samples")
    
    print("Getting rule-based predictions...")
    rule_preds = get_rule_predictions(sentences, labels)
    
    print("Getting BERT predictions...")
    bert_preds = get_bert_predictions(sentences, args.bert_model)
    
    # Calculate metrics
    rule_report = classification_report(labels, rule_preds, output_dict=True, zero_division=0)
    bert_report = classification_report(labels, bert_preds, output_dict=True, zero_division=0)
    
    rule_acc = rule_report['accuracy']
    bert_acc = bert_report['accuracy']
    
    print(f"\nRule-based accuracy: {rule_acc:.1%}")
    print(f"BERT accuracy: {bert_acc:.1%}")
    print(f"Improvement: {(bert_acc - rule_acc):.1%}")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_comparison(rule_acc, bert_acc, output_dir / "accuracy_comparison.png")
    plot_f1_comparison(rule_report, bert_report, output_dir / "f1_comparison.png")
    plot_confusion_matrices(labels, rule_preds, bert_preds, output_dir / "confusion_matrices.png")
    plot_precision_recall(rule_report, bert_report, output_dir / "precision_recall.png")
    plot_summary_dashboard(rule_acc, bert_acc, rule_report, bert_report, output_dir / "summary_dashboard.png")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()

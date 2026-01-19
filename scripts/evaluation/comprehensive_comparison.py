"""Comprehensive 3-model comparison with dashboard and protein analysis."""
import json
import csv
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import load_pipeline, classify_span


def load_data(path: Path):
    with open(path) as f:
        data = json.load(f)
    return [d["sentence"] for d in data], [d["label"] for d in data]


def get_rule_predictions(sentences):
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
    confidences = []
    for label, conf in classifier.predict_batch(sentences):
        predictions.append(label)
        confidences.append(conf)
    return predictions, confidences


def create_comprehensive_dashboard(results, output_path):
    """Create single comprehensive dashboard with all comparisons."""
    fig = plt.figure(figsize=(20, 16))
    
    models = list(results.keys())
    colors = {'Rule-based': '#e74c3c', 'PubMedBERT (v1)': '#3498db', 'SciBERT (v4)': '#27ae60'}
    
    # Grid: 3 rows, 3 cols
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Overall Accuracy (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    accs = [results[m]['accuracy'] * 100 for m in models]
    bars = ax1.bar(models, accs, color=[colors[m] for m in models], edgecolor='black', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title('Overall Accuracy', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. Macro F1 (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    f1s = [results[m]['report']['macro avg']['f1-score'] for m in models]
    bars = ax2.bar(models, f1s, color=[colors[m] for m in models], edgecolor='black', linewidth=1.2)
    for bar, f1 in zip(bars, f1s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{f1:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Macro F1', fontsize=11)
    ax2.set_title('Macro F1 Score', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(axis='x', rotation=15)
    
    # 3. Dataset info (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    info_text = """DATASET EVOLUTION
    
Initial:     290 samples
+ Negatives: 500 from CaseOLAP  
+ Positives: ~300 from CaseOLAP
+ No Assoc:  ~300 from CaseOLAP
─────────────────────────
Final:       1,481 samples

Class Distribution:
• positive:       518 (35%)
• negative:       548 (37%)
• no_association: 415 (28%)
"""
    ax3.text(0.1, 0.95, info_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax3.set_title('Dataset Summary', fontsize=13, fontweight='bold')
    
    # 4-6. Per-class F1 comparison (middle row)
    classes = ['positive', 'negative', 'no_association']
    class_titles = ['Positive Association F1', 'Negative Association F1', 'No Association F1']
    
    for idx, (cls, title) in enumerate(zip(classes, class_titles)):
        ax = fig.add_subplot(gs[1, idx])
        f1_vals = [results[m]['report'][cls]['f1-score'] for m in models]
        bars = ax.bar(models, f1_vals, color=[colors[m] for m in models], edgecolor='black')
        for bar, f1 in zip(bars, f1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{f1:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis='x', rotation=20, labelsize=9)
    
    # 7-9. Confusion matrices (bottom row)
    class_labels = ['Pos', 'Neg', 'NoA']
    
    for idx, model in enumerate(models):
        ax = fig.add_subplot(gs[2, idx])
        cm = results[model]['confusion_matrix']
        cmap = 'Reds' if 'Rule' in model else ('Blues' if 'PubMed' in model else 'Greens')
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                    xticklabels=class_labels, yticklabels=class_labels,
                    cbar=False, annot_kws={'size': 12})
        ax.set_title(f'{model}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
    
    plt.suptitle('HFpEF Protein-Disease Classification: 3-Model Comparison\n'
                 'Rule-based vs PubMedBERT (Initial) vs SciBERT (Final)',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def analyze_protein_filtering(corpus_path, models_config, output_path):
    """Analyze which proteins each model would filter out."""
    # Load corpus with protein mentions
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        return None
    
    df = pd.read_csv(corpus_path)
    if 'protein' not in df.columns or 'sentence' not in df.columns:
        print("Corpus missing required columns")
        return None
    
    results = {}
    for model_name, model_path in models_config.items():
        if model_path == 'rule':
            preds = get_rule_predictions(df['sentence'].tolist())
        else:
            preds, _ = get_bert_predictions(df['sentence'].tolist(), model_path)
        
        df[f'{model_name}_pred'] = preds
        
        # Count proteins that would be filtered (negative or no_association)
        filtered = df[df[f'{model_name}_pred'].isin(['negative', 'no_association'])]
        protein_counts = Counter(filtered['protein'])
        
        results[model_name] = {
            'total_filtered': len(filtered),
            'proteins_affected': len(protein_counts),
            'top_filtered': protein_counts.most_common(20)
        }
    
    # Save comparison
    with open(output_path, 'w') as f:
        f.write("PROTEIN FILTERING ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        for model, data in results.items():
            f.write(f"{model}:\n")
            f.write(f"  Sentences filtered: {data['total_filtered']}\n")
            f.write(f"  Unique proteins affected: {data['proteins_affected']}\n")
            f.write(f"  Top filtered proteins:\n")
            for protein, count in data['top_filtered'][:10]:
                f.write(f"    - {protein}: {count} mentions\n")
            f.write("\n")
    
    print(f"Saved: {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/labeled.json")
    parser.add_argument("--pubmedbert", default="models/pubmedbert-hfpef/final")
    parser.add_argument("--scibert", default="models/scibert-hfpef-v4/final")
    parser.add_argument("--corpus", default="data/hfpef_corpus.csv")
    parser.add_argument("--output", default="deliverable_email")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading labeled data...")
    sentences, labels = load_data(Path(args.data))
    print(f"Loaded {len(sentences)} samples")
    
    # Get predictions from all 3 models
    print("\nGetting Rule-based predictions...")
    rule_preds = get_rule_predictions(sentences)
    
    print("Getting PubMedBERT predictions...")
    pubmed_preds, _ = get_bert_predictions(sentences, args.pubmedbert)
    
    print("Getting SciBERT predictions...")
    scibert_preds, _ = get_bert_predictions(sentences, args.scibert)
    
    # Calculate metrics
    classes = ['positive', 'negative', 'no_association']
    results = {}
    
    for name, preds in [('Rule-based', rule_preds), 
                         ('PubMedBERT (v1)', pubmed_preds), 
                         ('SciBERT (v4)', scibert_preds)]:
        report = classification_report(labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(labels, preds, labels=classes)
        results[name] = {
            'accuracy': report['accuracy'],
            'report': report,
            'confusion_matrix': cm,
            'predictions': preds
        }
        print(f"\n{name}: {report['accuracy']:.1%} accuracy")
    
    # Create comprehensive dashboard
    print("\nGenerating comprehensive dashboard...")
    create_comprehensive_dashboard(results, output_dir / "comprehensive_dashboard.png")
    
    # Save detailed metrics
    metrics_file = output_dir / "detailed_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("DETAILED MODEL COMPARISON METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        for model in results:
            f.write(f"\n{model}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy: {results[model]['accuracy']:.1%}\n")
            f.write(f"Macro F1: {results[model]['report']['macro avg']['f1-score']:.3f}\n")
            f.write(f"Weighted F1: {results[model]['report']['weighted avg']['f1-score']:.3f}\n\n")
            
            f.write("Per-class metrics:\n")
            for cls in classes:
                r = results[model]['report'][cls]
                f.write(f"  {cls:15} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}\n")
            f.write("\n")
    
    print(f"Saved: {metrics_file}")
    
    # Comparison table
    comparison_csv = output_dir / "model_comparison.csv"
    rows = []
    for model in results:
        row = {
            'Model': model,
            'Accuracy': f"{results[model]['accuracy']:.1%}",
            'Macro_F1': f"{results[model]['report']['macro avg']['f1-score']:.3f}",
            'Positive_F1': f"{results[model]['report']['positive']['f1-score']:.2f}",
            'Negative_F1': f"{results[model]['report']['negative']['f1-score']:.2f}",
            'NoAssoc_F1': f"{results[model]['report']['no_association']['f1-score']:.2f}",
        }
        rows.append(row)
    
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)
    print(f"Saved: {comparison_csv}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

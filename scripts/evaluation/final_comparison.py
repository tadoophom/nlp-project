"""Generate final comparison dashboard with all models."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import load_pipeline, classify_span
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_rule_predictions(sentences, nlp):
    label_map = {"Positive": "positive", "Negative": "negative", "Neutral": "no_association"}
    preds = []
    for sent in sentences:
        doc = nlp(sent)
        spans = list(doc.sents)
        span = spans[0] if spans else doc[:]
        pred = classify_span(span)
        preds.append(label_map.get(pred, "no_association"))
    return preds


def get_multilabel_predictions(sentences, model_path):
    """Get predictions from multi-label model."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    labels = ['positive', 'negative', 'no_association']
    preds = []
    confs = []
    multi_preds = []
    
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0]
        
        # Multi-label: threshold at 0.5
        active = [labels[i] for i, p in enumerate(probs) if p > 0.5]
        multi_preds.append(active if active else ['no_association'])
        
        # Single label (highest prob) for comparison
        idx = probs.argmax().item()
        preds.append(labels[idx])
        confs.append(probs[idx].item())
    
    return preds, confs, multi_preds


def main():
    # Load data
    with open("data/labeled.json") as f:
        data = json.load(f)
    
    sentences = [d['sentence'] for d in data]
    true_labels = [d['label'] for d in data]
    
    print(f"Evaluating on {len(sentences)} samples...")
    
    # Get predictions from all models
    print("  Rule-based...")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    rule_preds = get_rule_predictions(sentences, nlp)
    
    print("  PubMedBERT...")
    pubmed = PubMedBERTClassifier(model_path="models/pubmedbert-hfpef/final")
    pubmed_preds = [pubmed.predict(s)[0] for s in sentences]
    
    print("  SciBERT v4...")
    scibert = PubMedBERTClassifier(model_path="models/scibert-hfpef-v4/final")
    scibert_preds = [scibert.predict(s)[0] for s in sentences]
    
    print("  SciBERT multi-label...")
    ml_preds, ml_confs, ml_multi = get_multilabel_predictions(sentences, "models/scibert-multilabel")
    
    # Calculate metrics
    models = {
        'Rule-based': rule_preds,
        'PubMedBERT': pubmed_preds,
        'SciBERT v4': scibert_preds,
        'SciBERT Multi-label': ml_preds
    }
    
    results = {}
    for name, preds in models.items():
        acc = accuracy_score(true_labels, preds)
        report = classification_report(true_labels, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(true_labels, preds, labels=['positive', 'negative', 'no_association'])
        results[name] = {'accuracy': acc, 'report': report, 'cm': cm}
        print(f"  {name}: {acc:.1%}")
    
    # Count multi-label cases
    multi_count = sum(1 for m in ml_multi if len(m) > 1)
    print(f"\nMulti-label detections: {multi_count} sentences with multiple labels")
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 14))
    
    # Title
    fig.suptitle('HFpEF Protein-Disease Classification: 4-Model Comparison', fontsize=16, fontweight='bold', y=0.98)
    
    # 1. Accuracy comparison bar chart
    ax1 = fig.add_subplot(3, 4, 1)
    names = list(results.keys())
    accs = [results[n]['accuracy'] * 100 for n in names]
    colors = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
    bars = ax1.bar(names, accs, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, acc in zip(bars, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.1f}%', 
                ha='center', fontweight='bold', fontsize=11)
    ax1.set_title('Overall Accuracy', fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    
    # 2. F1 scores by class
    ax2 = fig.add_subplot(3, 4, 2)
    classes = ['positive', 'negative', 'no_association']
    x = np.arange(len(classes))
    width = 0.2
    
    for i, (name, color) in enumerate(zip(names, colors)):
        f1s = [results[name]['report'][c]['f1-score'] for c in classes]
        ax2.bar(x + i*width, f1s, width, label=name.replace(' ', '\n'), color=color, edgecolor='black')
    
    ax2.set_ylabel('F1 Score', fontweight='bold')
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(['Positive', 'Negative', 'No Assoc'])
    ax2.legend(loc='lower right', fontsize=7)
    ax2.set_ylim(0, 1.1)
    ax2.set_title('F1 Score by Class', fontweight='bold')
    
    # 3. Dataset composition
    ax3 = fig.add_subplot(3, 4, 3)
    label_counts = pd.Series(true_labels).value_counts()
    ax3.pie(label_counts.values, labels=label_counts.index, autopct='%1.0f%%',
            colors=['#2ecc71', '#e74c3c', '#95a5a6'], startangle=90)
    ax3.set_title(f'Dataset (n={len(sentences)})', fontweight='bold')
    
    # 4. Improvement trajectory
    ax4 = fig.add_subplot(3, 4, 4)
    ax4.plot(range(len(names)), accs, 'o-', markersize=10, linewidth=2, color='#2c3e50')
    for i, (name, acc) in enumerate(zip(names, accs)):
        ax4.annotate(f'{acc:.1f}%', (i, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8)
    ax4.set_ylabel('Accuracy (%)', fontweight='bold')
    ax4.set_ylim(30, 105)
    ax4.set_title('Model Progression', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5-8. Confusion matrices
    for i, (name, color) in enumerate(zip(names, colors)):
        ax = fig.add_subplot(3, 4, 5 + i)
        cm = results[name]['cm']
        im = ax.imshow(cm, cmap='Blues')
        
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(['Pos', 'Neg', 'No'], fontsize=9)
        ax.set_yticklabels(['Pos', 'Neg', 'No'], fontsize=9)
        
        for ii in range(3):
            for jj in range(3):
                color_text = 'white' if cm[ii, jj] > cm.max()/2 else 'black'
                ax.text(jj, ii, str(cm[ii, jj]), ha='center', va='center', 
                       color=color_text, fontweight='bold')
        
        ax.set_title(f'{name}', fontweight='bold', fontsize=10)
        ax.set_xlabel('Predicted')
        if i == 0:
            ax.set_ylabel('True')
    
    # 9. Accuracy gains
    ax9 = fig.add_subplot(3, 4, 9)
    ax9.axis('off')
    
    gain1 = (results['PubMedBERT']['accuracy']-results['Rule-based']['accuracy'])*100
    gain2 = (results['SciBERT v4']['accuracy']-results['PubMedBERT']['accuracy'])*100
    gain3 = (results['SciBERT Multi-label']['accuracy']-results['SciBERT v4']['accuracy'])*100
    total = (results['SciBERT Multi-label']['accuracy']-results['Rule-based']['accuracy'])*100
    
    improvement_text = f"""Rule-based     {results['Rule-based']['accuracy']*100:.1f}%
      ↓ +{gain1:.1f}%
PubMedBERT     {results['PubMedBERT']['accuracy']*100:.1f}%
      ↓ +{gain2:.1f}%
SciBERT v4     {results['SciBERT v4']['accuracy']*100:.1f}%
      ↓ +{gain3:.1f}%
Multi-label    {results['SciBERT Multi-label']['accuracy']*100:.1f}%

Total: +{total:.1f}%"""
    ax9.text(0.1, 0.5, improvement_text, transform=ax9.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax9.set_title('Accuracy Gains', fontweight='bold')
    
    # 10. What failed
    ax10 = fig.add_subplot(3, 4, 10)
    ax10.axis('off')
    
    fail_text = f"""Rule-based:
  Missed negations, 0% on
  methodology sentences

PubMedBERT:
  0% on no_association
  (methodology sentences)

SciBERT fixed both."""
    ax10.text(0.1, 0.5, fail_text, transform=ax10.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax10.set_title('Why Models Failed', fontweight='bold')
    
    # 11. Dataset
    ax11 = fig.add_subplot(3, 4, 11)
    ax11.axis('off')
    
    details_text = """Training: 1,481 sentences

positive:       1,257 (85%)
negative:          80 (5%)
no_association:   144 (10%)

Held-out test: 100 sentences
Held-out accuracy: 96%"""
    ax11.text(0.1, 0.5, details_text, transform=ax11.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax11.set_title('Dataset', fontweight='bold')
    
    # 12. Filtering results
    ax12 = fig.add_subplot(3, 4, 12)
    ax12.axis('off')
    
    filter_text = f"""Proteins analyzed: 88
Kept: 73 (positive evidence)
Removed: 15

  4 - negative findings
 11 - methodology only

Review queue: 60 sentences
(low confidence)"""
    ax12.text(0.1, 0.5, filter_text, transform=ax12.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace')
    ax12.set_title('Protein Filtering', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = "deliverable_email/final_comparison_dashboard.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved to {output_path}")
    
    # Save metrics to text
    with open("deliverable_email/final_metrics.txt", "w") as f:
        f.write("HFpEF Protein-Disease Classification Results\n")
        f.write("=" * 50 + "\n\n")
        
        for name in names:
            f.write(f"{name}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {results[name]['accuracy']:.1%}\n")
            for cls in classes:
                r = results[name]['report'][cls]
                f.write(f"  {cls}: P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}\n")
            f.write("\n")
        
        f.write(f"Multi-label detections: {multi_count} sentences\n")
        f.write("\nSample multi-label cases:\n")
        for i, (sent, labels) in enumerate(zip(sentences, ml_multi)):
            if len(labels) > 1:
                f.write(f"  {labels}: {sent[:80]}...\n")
                if i > 5:
                    break
    
    print("Saved metrics to deliverable_email/final_metrics.txt")


if __name__ == "__main__":
    main()

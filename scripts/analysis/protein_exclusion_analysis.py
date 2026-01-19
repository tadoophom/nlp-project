"""Analyze which proteins each model would exclude from CaseOLAP results."""
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import load_pipeline, classify_span


def get_rule_prediction(sentence, nlp):
    label_map = {"Positive": "positive", "Negative": "negative", "Neutral": "no_association"}
    doc = nlp(sentence)
    span = list(doc.sents)[0] if list(doc.sents) else doc[:]
    pred = classify_span(span)
    return label_map.get(pred, "no_association")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/hfpef_corpus.csv")
    parser.add_argument("--pubmedbert", default="models/pubmedbert-hfpef/final")
    parser.add_argument("--scibert", default="models/scibert-hfpef-v4/final")
    parser.add_argument("--output", default="deliverable_email")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    # Load corpus
    print("Loading corpus...")
    df = pd.read_csv(args.corpus, nrows=args.limit)
    
    # Get sentences with evidence
    df = df[df['evidence_sentence'].notna() & (df['evidence_sentence'] != '')]
    sentences = df['evidence_sentence'].tolist()
    proteins = df['protein'].tolist()
    print(f"Analyzing {len(sentences)} sentences from {len(set(proteins))} proteins")
    
    # Load models
    print("\nLoading models...")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    pubmed_clf = PubMedBERTClassifier(model_path=args.pubmedbert)
    scibert_clf = PubMedBERTClassifier(model_path=args.scibert)
    
    # Get predictions
    print("Running predictions...")
    results = []
    for i, (sent, protein) in enumerate(zip(sentences, proteins)):
        if i % 100 == 0:
            print(f"  {i}/{len(sentences)}")
        
        rule_pred = get_rule_prediction(sent, nlp)
        pubmed_pred, pubmed_conf = pubmed_clf.predict(sent)
        scibert_pred, scibert_conf = scibert_clf.predict(sent)
        
        results.append({
            'protein': protein,
            'sentence': sent[:200],
            'rule_pred': rule_pred,
            'pubmed_pred': pubmed_pred,
            'scibert_pred': scibert_pred,
        })
    
    df_results = pd.DataFrame(results)
    
    # Analyze exclusions (negative or no_association = excluded)
    exclusion_types = ['negative', 'no_association']
    
    stats = {}
    for model in ['rule', 'pubmed', 'scibert']:
        col = f'{model}_pred'
        excluded = df_results[df_results[col].isin(exclusion_types)]
        kept = df_results[df_results[col] == 'positive']
        
        excluded_proteins = set(excluded['protein'].unique())
        kept_proteins = set(kept['protein'].unique())
        
        stats[model] = {
            'total_sentences': len(df_results),
            'excluded_sentences': len(excluded),
            'kept_sentences': len(kept),
            'exclusion_rate': len(excluded) / len(df_results) * 100,
            'unique_proteins_excluded': len(excluded_proteins - kept_proteins),
            'top_excluded': Counter(excluded['protein']).most_common(15),
        }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Exclusion rates
    ax1 = axes[0, 0]
    models = ['Rule-based', 'PubMedBERT', 'SciBERT']
    rates = [stats['rule']['exclusion_rate'], stats['pubmed']['exclusion_rate'], stats['scibert']['exclusion_rate']]
    colors = ['#e74c3c', '#3498db', '#27ae60']
    bars = ax1.bar(models, rates, color=colors, edgecolor='black')
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{rate:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Exclusion Rate (%)', fontsize=11)
    ax1.set_title('Sentence Exclusion Rate by Model', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    
    # 2. Kept vs Excluded breakdown
    ax2 = axes[0, 1]
    x = np.arange(3)
    width = 0.35
    kept = [stats['rule']['kept_sentences'], stats['pubmed']['kept_sentences'], stats['scibert']['kept_sentences']]
    excluded = [stats['rule']['excluded_sentences'], stats['pubmed']['excluded_sentences'], stats['scibert']['excluded_sentences']]
    
    ax2.bar(x - width/2, kept, width, label='Kept (Positive)', color='#27ae60', edgecolor='black')
    ax2.bar(x + width/2, excluded, width, label='Excluded (Neg/NoAssoc)', color='#e74c3c', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel('Number of Sentences', fontsize=11)
    ax2.set_title('Sentence Classification Breakdown', fontsize=13, fontweight='bold')
    ax2.legend()
    
    # 3. Unique proteins fully excluded
    ax3 = axes[1, 0]
    unique_excluded = [stats['rule']['unique_proteins_excluded'], 
                       stats['pubmed']['unique_proteins_excluded'], 
                       stats['scibert']['unique_proteins_excluded']]
    bars = ax3.bar(models, unique_excluded, color=colors, edgecolor='black')
    for bar, val in zip(bars, unique_excluded):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha='center', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Number of Proteins', fontsize=11)
    ax3.set_title('Proteins Completely Excluded\n(All mentions classified as negative/no_association)', 
                  fontsize=12, fontweight='bold')
    
    # 4. Explanation text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
WHY MODELS DIFFER IN EXCLUSIONS

RULE-BASED (High False Positive Rate):
• Cannot detect semantic negation
• Marks method descriptions as "positive"
• Example: "We studied BNP in HFpEF patients"
  → Rule: Positive | SciBERT: No Association

PubMedBERT v1 (Moderate):
• Better context understanding
• Limited training data (290 samples)
• Struggles with "no_association" class

SciBERT v4 (Accurate):
• Trained on 1,481 balanced samples
• Best biomedical model (92.8% CV)
• Correctly identifies:
  - Study methodology sentences
  - Negative findings
  - True positive associations

PRACTICAL IMPACT:
• Rule-based: Over-reports protein associations
• SciBERT: Filters noise, keeps real findings
"""
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax4.set_title('Interpretation', fontsize=13, fontweight='bold')
    
    plt.suptitle('Protein Exclusion Analysis: Impact on CaseOLAP Results',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'protein_exclusion_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'protein_exclusion_analysis.png'}")
    
    # Save detailed exclusion data
    exclusion_report = output_dir / 'protein_exclusion_report.txt'
    with open(exclusion_report, 'w') as f:
        f.write("PROTEIN EXCLUSION ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        for model, name in [('rule', 'Rule-based'), ('pubmed', 'PubMedBERT v1'), ('scibert', 'SciBERT v4')]:
            s = stats[model]
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total sentences analyzed: {s['total_sentences']}\n")
            f.write(f"Sentences excluded: {s['excluded_sentences']} ({s['exclusion_rate']:.1f}%)\n")
            f.write(f"Sentences kept: {s['kept_sentences']}\n")
            f.write(f"Proteins fully excluded: {s['unique_proteins_excluded']}\n")
            f.write(f"\nTop excluded proteins:\n")
            for protein, count in s['top_excluded'][:10]:
                f.write(f"  {protein}: {count} mentions excluded\n")
            f.write("\n\n")
    
    print(f"Saved: {exclusion_report}")
    
    # Save CSV
    df_results.to_csv(output_dir / 'protein_predictions.csv', index=False)
    print(f"Saved: {output_dir / 'protein_predictions.csv'}")


if __name__ == "__main__":
    main()

"""Generate CSV of excluded proteins with explanations for each model."""
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd

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


def get_exclusion_reason(label, sentence):
    """Generate human-readable explanation for exclusion."""
    if label == "negative":
        # Check for common negative patterns
        sent_lower = sentence.lower()
        if "no significant" in sent_lower:
            return "Negative finding: no significant association/difference reported"
        elif "not associated" in sent_lower or "not significantly" in sent_lower:
            return "Negative finding: explicitly states no association"
        elif "failed to" in sent_lower:
            return "Negative finding: study failed to show association"
        elif "did not" in sent_lower:
            return "Negative finding: study did not demonstrate effect"
        elif "no correlation" in sent_lower or "no relationship" in sent_lower:
            return "Negative finding: no correlation/relationship found"
        elif "unclear" in sent_lower or "unknown" in sent_lower:
            return "Inconclusive: relationship unclear or unknown"
        else:
            return "Negative finding: sentence indicates lack of association"
    
    elif label == "no_association":
        sent_lower = sentence.lower()
        if any(x in sent_lower for x in ["we aimed", "we sought", "we investigated", "we examined"]):
            return "Methodology: describes study objectives, not findings"
        elif any(x in sent_lower for x in ["patients were", "subjects were", "enrolled", "recruited"]):
            return "Methodology: describes patient enrollment/study design"
        elif any(x in sent_lower for x in ["methods", "retrospective", "prospective", "cohort"]):
            return "Methodology: describes study methods"
        elif any(x in sent_lower for x in ["measured", "analyzed", "calculated", "assessed"]):
            return "Methodology: describes measurement/analysis procedures"
        else:
            return "No claim: sentence does not make association claim"
    
    return "Included: positive association"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/hfpef_corpus.csv")
    parser.add_argument("--pubmedbert", default="models/pubmedbert-hfpef/final")
    parser.add_argument("--scibert", default="models/scibert-hfpef-v4/final")
    parser.add_argument("--output", default="deliverable_email/excluded_proteins.csv")
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    
    # Load corpus
    print("Loading corpus...")
    df = pd.read_csv(args.corpus, nrows=args.limit)
    df = df[df['evidence_sentence'].notna() & (df['evidence_sentence'] != '')]
    print(f"Analyzing {len(df)} sentences")
    
    # Load models
    print("Loading models...")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    pubmed_clf = PubMedBERTClassifier(model_path=args.pubmedbert)
    scibert_clf = PubMedBERTClassifier(model_path=args.scibert)
    
    # Process each sentence
    print("Processing sentences...")
    rows = []
    
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  {i}/{len(df)}")
        
        protein = row['protein']
        sentence = row['evidence_sentence']
        pmid = row.get('pmid', '')
        
        # Get predictions
        rule_pred = get_rule_prediction(sentence, nlp)
        pubmed_pred, pubmed_conf = pubmed_clf.predict(sentence)
        scibert_pred, scibert_conf = scibert_clf.predict(sentence)
        
        # Only include if at least one model excludes it
        if rule_pred != 'positive' or pubmed_pred != 'positive' or scibert_pred != 'positive':
            rows.append({
                'protein': protein,
                'pmid': pmid,
                'sentence': sentence[:300] + ('...' if len(sentence) > 300 else ''),
                'rule_based_prediction': rule_pred,
                'rule_based_excluded': 'Yes' if rule_pred != 'positive' else 'No',
                'rule_based_reason': get_exclusion_reason(rule_pred, sentence) if rule_pred != 'positive' else '',
                'pubmedbert_prediction': pubmed_pred,
                'pubmedbert_confidence': f"{pubmed_conf:.2f}",
                'pubmedbert_excluded': 'Yes' if pubmed_pred != 'positive' else 'No',
                'pubmedbert_reason': get_exclusion_reason(pubmed_pred, sentence) if pubmed_pred != 'positive' else '',
                'scibert_prediction': scibert_pred,
                'scibert_confidence': f"{scibert_conf:.2f}",
                'scibert_excluded': 'Yes' if scibert_pred != 'positive' else 'No',
                'scibert_reason': get_exclusion_reason(scibert_pred, sentence) if scibert_pred != 'positive' else '',
            })
    
    # Save to CSV
    output_path = Path(args.output)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"\nSaved {len(rows)} excluded sentences to {output_path}")
    
    # Summary stats
    rule_excluded = sum(1 for r in rows if r['rule_based_excluded'] == 'Yes')
    pubmed_excluded = sum(1 for r in rows if r['pubmedbert_excluded'] == 'Yes')
    scibert_excluded = sum(1 for r in rows if r['scibert_excluded'] == 'Yes')
    
    print(f"\nExclusion counts:")
    print(f"  Rule-based: {rule_excluded}")
    print(f"  PubMedBERT: {pubmed_excluded}")
    print(f"  SciBERT: {scibert_excluded}")


if __name__ == "__main__":
    main()

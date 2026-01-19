"""
Create a held-out test set from sentences NOT in training data.

PURPOSE:
The 97% accuracy was measured on data that overlaps with training.
This script extracts NEW sentences the model has never seen,
so we can measure true generalization performance.

USAGE:
    uv run python scripts/create_holdout_test.py
    # Manually label the output file
    uv run python scripts/evaluate_holdout.py
"""
import json
import random
from pathlib import Path

import pandas as pd


def create_holdout_test(
    corpus_path: Path,
    labeled_path: Path,
    output_path: Path,
    n_samples: int = 100
):
    # Load existing labeled sentences
    with open(labeled_path) as f:
        labeled = json.load(f)
    labeled_sents = {item['sentence'].strip().lower() for item in labeled}
    print(f"Training data has {len(labeled_sents)} unique sentences")
    
    # Load corpus
    corpus = pd.read_csv(corpus_path)
    corpus = corpus[corpus['evidence_sentence'].notna()]
    print(f"Corpus has {len(corpus)} sentences")
    
    # Find sentences NOT in training data
    corpus['sent_lower'] = corpus['evidence_sentence'].str.strip().str.lower()
    new_sentences = corpus[~corpus['sent_lower'].isin(labeled_sents)]
    print(f"Found {len(new_sentences)} sentences not in training data")
    
    # Sample randomly
    if len(new_sentences) < n_samples:
        print(f"Warning: Only {len(new_sentences)} available, using all")
        sample = new_sentences
    else:
        sample = new_sentences.sample(n=n_samples, random_state=42)
    
    # Prepare output with empty label column for manual annotation
    output = []
    for _, row in sample.iterrows():
        output.append({
            'sentence': row['evidence_sentence'],
            'protein': row.get('protein', ''),
            'pmid': row.get('pmid', ''),
            'manual_label': '',  # TO BE FILLED: positive, negative, no_association
        })
    
    # Save
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved {len(output)} sentences to {output_path}")
    print("\nNEXT STEPS:")
    print("1. Open the file and fill 'manual_label' with: positive, negative, or no_association")
    print("2. Run: uv run python scripts/evaluate_holdout.py")


if __name__ == "__main__":
    create_holdout_test(
        corpus_path=Path("data/hfpef_corpus.csv"),
        labeled_path=Path("data/labeled.json"),
        output_path=Path("data/holdout_test.json"),
        n_samples=100
    )

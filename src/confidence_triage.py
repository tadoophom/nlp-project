"""
Confidence-based triage system.

Instead of binary include/exclude decisions, creates three buckets:
- INCLUDE: High confidence positive (>= high_threshold)
- EXCLUDE: High confidence negative/no_assoc (>= high_threshold)  
- REVIEW: Low confidence predictions requiring human verification
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bert_classifier import PubMedBERTClassifier


@dataclass
class TriageResult:
    sentence: str
    protein: str
    prediction: str
    confidence: float
    bucket: str  # 'include', 'exclude', 'review'


class ConfidenceTriage:
    def __init__(
        self,
        model_path: str = "models/scibert-hfpef-v4/final",
        high_threshold: float = 0.85,
        low_threshold: float = 0.65
    ):
        """
        Args:
            high_threshold: Above this, auto-decide
            low_threshold: Below this, always review
            Between thresholds: review if not positive
        """
        self.classifier = PubMedBERTClassifier(model_path=model_path)
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
    
    def triage_sentence(self, sentence: str, protein: str = "") -> TriageResult:
        """Classify and assign to appropriate bucket."""
        label, conf = self.classifier.predict(sentence)
        
        if conf >= self.high_threshold:
            bucket = 'include' if label == 'positive' else 'exclude'
        elif conf < self.low_threshold:
            bucket = 'review'
        else:
            # Medium confidence: include positive, review others
            bucket = 'include' if label == 'positive' else 'review'
        
        return TriageResult(
            sentence=sentence,
            protein=protein,
            prediction=label,
            confidence=conf,
            bucket=bucket
        )
    
    def triage_dataframe(self, df: pd.DataFrame) -> dict:
        """
        Triage all rows in a DataFrame.
        
        Returns dict with 'include', 'exclude', 'review' DataFrames.
        """
        results = {'include': [], 'exclude': [], 'review': []}
        
        for _, row in df.iterrows():
            sentence = row.get('evidence_sentence', '')
            if not sentence:
                continue
            
            result = self.triage_sentence(sentence, row.get('protein', ''))
            
            row_dict = row.to_dict()
            row_dict['triage_prediction'] = result.prediction
            row_dict['triage_confidence'] = result.confidence
            row_dict['triage_bucket'] = result.bucket
            
            results[result.bucket].append(row_dict)
        
        return {
            'include': pd.DataFrame(results['include']),
            'exclude': pd.DataFrame(results['exclude']),
            'review': pd.DataFrame(results['review'])
        }
    
    def generate_review_queue(self, df: pd.DataFrame, output_path: str):
        """Generate CSV of samples needing human review."""
        triaged = self.triage_dataframe(df)
        review_df = triaged['review']
        
        if len(review_df) == 0:
            print("No samples need review")
            return
        
        # Sort by confidence (lowest first - most uncertain)
        review_df = review_df.sort_values('triage_confidence')
        
        # Add empty column for human label
        review_df['human_label'] = ''
        
        # Select relevant columns
        cols = ['protein', 'evidence_sentence', 'triage_prediction', 
                'triage_confidence', 'human_label']
        cols = [c for c in cols if c in review_df.columns]
        
        review_df[cols].to_csv(output_path, index=False)
        
        print(f"Review queue: {len(review_df)} samples saved to {output_path}")
        print(f"  Confidence range: {review_df['triage_confidence'].min():.2f} - {review_df['triage_confidence'].max():.2f}")
        
        return review_df


def triage_corpus(
    corpus_path: str,
    output_dir: str,
    model_path: str = "models/scibert-hfpef-v4/final"
):
    """Convenience function to triage a full corpus."""
    df = pd.read_csv(corpus_path)
    df = df[df['evidence_sentence'].notna()]
    
    triage = ConfidenceTriage(model_path=model_path)
    results = triage.triage_dataframe(df)
    
    output = Path(output_dir)
    output.mkdir(exist_ok=True)
    
    for bucket, bucket_df in results.items():
        if len(bucket_df) > 0:
            bucket_df.to_csv(output / f"{bucket}.csv", index=False)
    
    # Summary
    total = len(df)
    print(f"\nTriage Summary:")
    print(f"  Include: {len(results['include'])} ({len(results['include'])/total:.1%})")
    print(f"  Exclude: {len(results['exclude'])} ({len(results['exclude'])/total:.1%})")
    print(f"  Review:  {len(results['review'])} ({len(results['review'])/total:.1%})")

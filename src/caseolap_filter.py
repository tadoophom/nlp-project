"""
CaseOLAP Pipeline Integration - Automatic Protein Filtering.

PURPOSE:
Integrates the SciBERT classifier directly into the CaseOLAP pipeline
to automatically filter out proteins with weak/negative associations.

HOW IT WORKS:
1. Takes CaseOLAP output (protein rankings with evidence sentences)
2. Classifies each evidence sentence
3. Filters out proteins where evidence is negative/no_association
4. Returns cleaned protein rankings with confidence scores

USAGE:
    from src.caseolap_filter import CaseOLAPFilter
    
    # Initialize once
    filter = CaseOLAPFilter()
    
    # Filter CaseOLAP results
    filtered = filter.filter_dataframe(raw_caseolap_df)
    
    # Or filter single protein
    is_valid, reason = filter.validate_protein("P12345", evidence_sentences)
"""
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .bert_classifier import PubMedBERTClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering a single protein."""
    protein_id: str
    included: bool
    positive_mentions: int
    negative_mentions: int
    no_association_mentions: int
    avg_confidence: float
    reason: str


class CaseOLAPFilter:
    """
    Filter CaseOLAP protein results using SciBERT classification.
    
    Removes proteins where the evidence sentences indicate:
    - Negative findings (no association found)
    - Methodology only (no claim about association)
    """
    
    def __init__(
        self,
        model_path: str = "models/scibert-hfpef-v4/final",
        confidence_threshold: float = 0.6,
        require_positive_majority: bool = True
    ):
        """
        Args:
            model_path: Path to trained SciBERT model
            confidence_threshold: Minimum confidence to trust prediction
            require_positive_majority: If True, protein needs >50% positive mentions
        """
        self.classifier = PubMedBERTClassifier(model_path=model_path)
        self.confidence_threshold = confidence_threshold
        self.require_positive_majority = require_positive_majority
        logger.info(f"CaseOLAPFilter initialized with model: {model_path}")
    
    def classify_sentence(self, sentence: str) -> tuple[str, float]:
        """Classify a single evidence sentence."""
        return self.classifier.predict(sentence)
    
    def validate_protein(
        self,
        protein_id: str,
        evidence_sentences: list[str]
    ) -> FilterResult:
        """
        Validate a protein based on its evidence sentences.
        
        Returns FilterResult with decision and reasoning.
        """
        if not evidence_sentences:
            return FilterResult(
                protein_id=protein_id,
                included=False,
                positive_mentions=0,
                negative_mentions=0,
                no_association_mentions=0,
                avg_confidence=0.0,
                reason="No evidence sentences provided"
            )
        
        # Classify all sentences
        positive = 0
        negative = 0
        no_association = 0
        confidences = []
        
        for sent in evidence_sentences:
            label, conf = self.classify_sentence(sent)
            confidences.append(conf)
            
            if label == 'positive':
                positive += 1
            elif label == 'negative':
                negative += 1
            else:
                no_association += 1
        
        total = len(evidence_sentences)
        avg_conf = sum(confidences) / len(confidences)
        
        # Decision logic
        if self.require_positive_majority:
            included = positive > (negative + no_association)
        else:
            included = positive > 0 and negative == 0
        
        # Generate reason
        if included:
            reason = f"Included: {positive}/{total} positive mentions ({positive/total:.0%})"
        else:
            if negative > 0:
                reason = f"Excluded: {negative}/{total} negative findings"
            elif no_association >= positive:
                reason = f"Excluded: {no_association}/{total} methodology/no claim"
            else:
                reason = f"Excluded: insufficient positive evidence ({positive}/{total})"
        
        return FilterResult(
            protein_id=protein_id,
            included=included,
            positive_mentions=positive,
            negative_mentions=negative,
            no_association_mentions=no_association,
            avg_confidence=avg_conf,
            reason=reason
        )
    
    def filter_dataframe(
        self,
        df: pd.DataFrame,
        protein_col: str = 'protein',
        sentence_col: str = 'evidence_sentence',
        score_col: str = 'protein_score'
    ) -> pd.DataFrame:
        """
        Filter a CaseOLAP results DataFrame.
        
        Args:
            df: CaseOLAP output with protein rankings
            protein_col: Column name for protein IDs
            sentence_col: Column name for evidence sentences
            score_col: Column name for CaseOLAP scores
        
        Returns:
            Filtered DataFrame with only valid positive associations,
            plus new columns: association_label, association_confidence, filter_reason
        """
        logger.info(f"Filtering {len(df)} rows...")
        
        # Group by protein
        results = []
        
        for protein_id, group in df.groupby(protein_col):
            sentences = group[sentence_col].dropna().tolist()
            
            # Validate protein
            filter_result = self.validate_protein(protein_id, sentences)
            
            if filter_result.included:
                # Keep all rows for this protein, add metadata
                for _, row in group.iterrows():
                    row_dict = row.to_dict()
                    
                    # Classify individual sentence
                    if pd.notna(row.get(sentence_col)):
                        label, conf = self.classify_sentence(row[sentence_col])
                        row_dict['association_label'] = label
                        row_dict['association_confidence'] = conf
                    
                    row_dict['filter_reason'] = filter_result.reason
                    results.append(row_dict)
            else:
                logger.info(f"Excluded protein {protein_id}: {filter_result.reason}")
        
        filtered_df = pd.DataFrame(results)
        
        logger.info(f"Kept {len(filtered_df)} rows ({len(filtered_df)/len(df):.1%})")
        logger.info(f"Unique proteins: {df[protein_col].nunique()} -> {filtered_df[protein_col].nunique()}")
        
        return filtered_df
    
    def generate_filter_report(
        self,
        df: pd.DataFrame,
        output_path: str,
        protein_col: str = 'protein',
        sentence_col: str = 'evidence_sentence'
    ):
        """
        Generate detailed report of filtering decisions.
        
        Creates CSV with per-protein filtering decisions and reasons.
        """
        report_rows = []
        
        for protein_id, group in df.groupby(protein_col):
            sentences = group[sentence_col].dropna().tolist()
            result = self.validate_protein(protein_id, sentences)
            
            report_rows.append({
                'protein': protein_id,
                'included': result.included,
                'total_mentions': len(sentences),
                'positive_mentions': result.positive_mentions,
                'negative_mentions': result.negative_mentions,
                'no_association_mentions': result.no_association_mentions,
                'avg_confidence': f"{result.avg_confidence:.2f}",
                'reason': result.reason
            })
        
        report_df = pd.DataFrame(report_rows)
        report_df.to_csv(output_path, index=False)
        
        # Summary stats
        included = report_df['included'].sum()
        excluded = len(report_df) - included
        
        logger.info(f"\nFilter Report saved to {output_path}")
        logger.info(f"Proteins included: {included}")
        logger.info(f"Proteins excluded: {excluded}")
        
        return report_df


def filter_caseolap_file(
    input_path: str,
    output_path: str,
    report_path: Optional[str] = None,
    model_path: str = "models/scibert-hfpef-v4/final"
):
    """
    Convenience function to filter a CaseOLAP CSV file.
    
    USAGE:
        python -c "from src.caseolap_filter import filter_caseolap_file; \
                   filter_caseolap_file('data/caseolap.csv', 'data/filtered.csv')"
    """
    # Load
    df = pd.read_csv(input_path)
    
    # Filter
    filter = CaseOLAPFilter(model_path=model_path)
    filtered = filter.filter_dataframe(df)
    
    # Save
    filtered.to_csv(output_path, index=False)
    print(f"Saved filtered results to {output_path}")
    
    # Optional report
    if report_path:
        filter.generate_filter_report(df, report_path)

#!/usr/bin/env python3
"""
validate_on_biored.py - Validate sentiment analysis on BioRED benchmark.

Extracts gene-disease relations from BioRED and evaluates our classifier.
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nlp_utils import load_pipeline, extract


BIORED_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "biored" / "BioRED"

BIORED_TO_OURS = {
    "Positive_Correlation": "Associated",
    "Negative_Correlation": "Not_Associated",
    "Association": "Associated",
}


def load_biored_gene_disease_relations(split: str = "Test") -> List[Dict]:
    """Load gene-disease relations from BioRED."""
    filepath = BIORED_DIR / f"{split}.BioC.JSON"
    with open(filepath) as f:
        data = json.load(f)

    relations = []
    for doc in data["documents"]:
        pmid = doc["id"]

        passages = doc.get("passages", [])
        title = passages[0]["text"] if passages else ""
        abstract = passages[1]["text"] if len(passages) > 1 else ""
        full_text = f"{title} {abstract}"

        entity_map = {}
        for passage in passages:
            for ann in passage.get("annotations", []):
                eid = ann["infons"].get("identifier", ann["id"])
                etype = ann["infons"].get("type", "")
                etext = ann.get("text", "")
                entity_map[eid] = {"type": etype, "text": etext}

        for rel in doc.get("relations", []):
            rel_type = rel["infons"].get("type", "")
            if rel_type not in BIORED_TO_OURS:
                continue

            e1_id = rel["infons"].get("entity1", "")
            e2_id = rel["infons"].get("entity2", "")

            e1 = entity_map.get(e1_id, {})
            e2 = entity_map.get(e2_id, {})

            is_gene_disease = (
                (e1.get("type") == "GeneOrGeneProduct" and e2.get("type") == "DiseaseOrPhenotypicFeature") or
                (e2.get("type") == "GeneOrGeneProduct" and e1.get("type") == "DiseaseOrPhenotypicFeature")
            )

            if not is_gene_disease:
                continue

            gene = e1 if e1.get("type") == "GeneOrGeneProduct" else e2
            disease = e2 if e2.get("type") == "DiseaseOrPhenotypicFeature" else e1

            relations.append({
                "pmid": pmid,
                "text": full_text,
                "gene": gene.get("text", ""),
                "disease": disease.get("text", ""),
                "biored_label": rel_type,
                "our_label": BIORED_TO_OURS[rel_type],
                "novel": rel["infons"].get("novel", ""),
            })

    return relations


def run_validation(relations: List[Dict], sample_size: int = None) -> Dict:
    """Run our classifier on BioRED relations and compute metrics."""
    print("Loading NLP pipeline...")
    nlp = load_pipeline("en_core_web_sm", use_context=True)

    if sample_size and len(relations) > sample_size:
        import random
        random.seed(42)
        relations = random.sample(relations, sample_size)

    print(f"Evaluating on {len(relations)} gene-disease relations...")

    results = []
    for i, rel in enumerate(relations):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(relations)}...")

        text = rel["text"]
        gene = rel["gene"]

        extractions = extract(text, [gene], nlp)

        if extractions:
            pred_label = extractions[0]["classification"]
            pred_conf = extractions[0]["confidence"]
        else:
            pred_label = "Associated"
            pred_conf = 0.5

        results.append({
            **rel,
            "pred_label": pred_label,
            "pred_conf": pred_conf,
            "correct": pred_label == rel["our_label"],
        })

    return compute_metrics(results)


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute precision, recall, F1 for each class."""
    y_true = [r["our_label"] for r in results]
    y_pred = [r["pred_label"] for r in results]

    classes = ["Associated", "Not_Associated", "Incidental"]
    metrics = {"overall": {}, "per_class": {}, "results": results}

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    metrics["overall"]["accuracy"] = correct / len(results) if results else 0
    metrics["overall"]["total"] = len(results)
    metrics["overall"]["correct"] = correct

    for cls in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics["per_class"][cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(1 for t in y_true if t == cls),
        }

    metrics["confusion"] = {
        "true": Counter(y_true),
        "pred": Counter(y_pred),
    }

    return metrics


def print_report(metrics: Dict):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("BIORED VALIDATION REPORT")
    print("=" * 60)

    print(f"\nOverall Accuracy: {metrics['overall']['accuracy']:.1%}")
    print(f"Total samples: {metrics['overall']['total']}")
    print(f"Correct: {metrics['overall']['correct']}")

    print("\nPer-class metrics:")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 58)
    for cls in ["Associated", "Not_Associated", "Incidental"]:
        m = metrics["per_class"].get(cls, {})
        print(f"{cls:<12} {m.get('precision', 0):<12.3f} {m.get('recall', 0):<12.3f} {m.get('f1', 0):<12.3f} {m.get('support', 0):<10}")

    print("\nLabel distribution:")
    print(f"  Gold:      {dict(metrics['confusion']['true'])}")
    print(f"  Predicted: {dict(metrics['confusion']['pred'])}")

    errors = [r for r in metrics["results"] if not r["correct"]]
    if errors:
        print(f"\nSample errors ({min(5, len(errors))} of {len(errors)}):")
        for err in errors[:5]:
            print(f"  Gene: {err['gene'][:30]}")
            print(f"  Gold: {err['our_label']}, Pred: {err['pred_label']}")
            print(f"  Text: {err['text'][:100]}...")
            print()


def main():
    print("Loading BioRED gene-disease relations...")
    relations = load_biored_gene_disease_relations("Test")
    print(f"Found {len(relations)} gene-disease relations")

    label_dist = Counter(r["our_label"] for r in relations)
    print(f"Label distribution: {dict(label_dist)}")

    metrics = run_validation(relations, sample_size=200)
    print_report(metrics)

    output_path = Path(__file__).parent.parent / "data" / "benchmarks" / "biored_validation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "overall": metrics["overall"],
            "per_class": metrics["per_class"],
            "confusion": {
                "true": dict(metrics["confusion"]["true"]),
                "pred": dict(metrics["confusion"]["pred"]),
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return metrics


if __name__ == "__main__":
    main()

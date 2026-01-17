"""Active learning: find uncertain samples for labeling."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier


def find_uncertain_samples(
    corpus_path: Path,
    model_path: Path,
    existing_labeled: Path,
    output_path: Path,
    min_conf: float = 0.4,
    max_conf: float = 0.75,
    limit: int = 100,
):
    """Find sentences where BERT is uncertain for human labeling."""
    # Load existing labeled to exclude
    existing = set()
    if existing_labeled.exists():
        with open(existing_labeled) as f:
            for item in json.load(f):
                existing.add(item["sentence"][:80])
    
    # Load classifier
    classifier = PubMedBERTClassifier(model_path=str(model_path))
    
    # Collect candidate sentences
    candidates = []
    with open(corpus_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get("evidence_sentence", "").strip()
            if not sentence or len(sentence) < 30 or sentence[:80] in existing:
                continue
            # Must mention HFpEF
            if "hfpef" not in sentence.lower() and "preserved ejection" not in sentence.lower():
                continue
            candidates.append(sentence)
    
    print(f"Found {len(candidates)} candidate sentences")
    
    # Classify and find uncertain ones
    uncertain = []
    batch_size = 32
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i+batch_size]
        results = classifier.predict_batch(batch)
        
        for sent, (label, conf) in zip(batch, results):
            if min_conf <= conf <= max_conf:
                uncertain.append({
                    "sentence": sent,
                    "predicted_label": label,
                    "confidence": round(conf, 3),
                    "manual_label": "",  # To be filled by human
                })
        
        if len(uncertain) >= limit:
            break
        
        print(f"Processed {min(i+batch_size, len(candidates))}/{len(candidates)}, found {len(uncertain)} uncertain")
    
    # Sort by confidence (most uncertain first)
    uncertain.sort(key=lambda x: abs(x["confidence"] - 0.5))
    uncertain = uncertain[:limit]
    
    # Save for labeling
    with open(output_path, "w") as f:
        json.dump(uncertain, f, indent=2)
    
    print(f"\nSaved {len(uncertain)} uncertain samples to {output_path}")
    print("\nLabel distribution of uncertain samples:")
    from collections import Counter
    dist = Counter(s["predicted_label"] for s in uncertain)
    for label, count in dist.items():
        print(f"  {label}: {count}")
    
    print("\nNext steps:")
    print(f"1. Open {output_path}")
    print("2. Fill in 'manual_label' field with: positive, negative, or no_association")
    print("3. Run: uv run python scripts/merge_labels.py")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/hfpef_corpus.csv")
    parser.add_argument("--model", default="models/pubmedbert-hfpef/final")
    parser.add_argument("--existing", default="data/labeled.json")
    parser.add_argument("--output", default="data/uncertain_for_labeling.json")
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--max-conf", type=float, default=0.75)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    
    find_uncertain_samples(
        Path(args.corpus),
        Path(args.model),
        Path(args.existing),
        Path(args.output),
        args.min_conf,
        args.max_conf,
        args.limit,
    )


if __name__ == "__main__":
    main()

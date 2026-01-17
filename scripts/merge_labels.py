"""Merge newly labeled samples into main labeled.json."""
import json
from pathlib import Path


def merge_labels(new_labels_path: Path, existing_path: Path):
    """Merge new labels into existing labeled dataset."""
    with open(existing_path) as f:
        existing = json.load(f)
    
    existing_sents = {item["sentence"][:80] for item in existing}
    
    with open(new_labels_path) as f:
        new_samples = json.load(f)
    
    added = 0
    skipped = 0
    
    for sample in new_samples:
        label = sample.get("manual_label", "").strip().lower()
        if not label:
            skipped += 1
            continue
        
        # Normalize label
        if label in ["positive", "positive_association"]:
            label = "positive"
        elif label in ["negative", "negative_association"]:
            label = "negative"
        elif label in ["no_association", "neutral", "none"]:
            label = "no_association"
        else:
            print(f"Unknown label '{label}', skipping")
            skipped += 1
            continue
        
        if sample["sentence"][:80] in existing_sents:
            skipped += 1
            continue
        
        existing.append({
            "sentence": sample["sentence"],
            "label": label,
        })
        existing_sents.add(sample["sentence"][:80])
        added += 1
    
    # Save
    with open(existing_path, "w") as f:
        json.dump(existing, f, indent=2)
    
    print(f"Added {added} new samples, skipped {skipped}")
    print(f"Total samples now: {len(existing)}")
    
    # Show distribution
    from collections import Counter
    dist = Counter(item["label"] for item in existing)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default="data/uncertain_for_labeling.json")
    parser.add_argument("--existing", default="data/labeled.json")
    args = parser.parse_args()
    
    merge_labels(Path(args.new), Path(args.existing))


if __name__ == "__main__":
    main()

"""Convert CSV training data to JSON format for BERT training."""
import csv
import json
from pathlib import Path
from collections import Counter


def convert_csv_to_json(csv_path: Path, json_path: Path) -> dict:
    """Convert CSV with manual labels to JSON training format."""
    samples = []
    label_counts = Counter()
    
    # Map CSV labels to our standard labels
    label_map = {
        "positive_association": "positive",
        "negative_association": "negative", 
        "no_association": "no_association",
    }
    
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            manual_label = row.get("manual_label", "").strip()
            if manual_label not in label_map:
                continue  # Skip unlabeled rows
                
            sentence = row.get("sentence", "").strip()
            if not sentence:
                continue
                
            label = label_map[manual_label]
            samples.append({
                "sentence": sentence,
                "label": label,
            })
            label_counts[label] += 1
    
    with open(json_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    return dict(label_counts)


if __name__ == "__main__":
    csv_path = Path(__file__).parent.parent / "pubmedbert_training_samples.csv"
    json_path = Path(__file__).parent.parent / "data" / "labeled.json"
    json_path.parent.mkdir(exist_ok=True)
    
    counts = convert_csv_to_json(csv_path, json_path)
    total = sum(counts.values())
    
    print(f"Converted {total} labeled samples to {json_path}")
    print("\nLabel distribution:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count} ({count/total*100:.1f}%)")

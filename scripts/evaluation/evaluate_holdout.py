"""
Evaluate model on held-out test set.

PURPOSE:
Measures TRUE accuracy on data the model has never seen.
This is the real test of generalization.

USAGE:
    uv run python scripts/evaluation/evaluate_holdout.py
"""
import json
from pathlib import Path

from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import PubMedBERTClassifier


def evaluate_holdout(
    test_path: Path,
    model_path: Path,
):
    # Load test data
    with open(test_path) as f:
        test_data = json.load(f)
    
    # Filter to only labeled samples
    labeled = [d for d in test_data if d.get('manual_label')]
    if not labeled:
        print("ERROR: No labels found. Please fill 'manual_label' field first.")
        return
    
    print(f"Evaluating on {len(labeled)} labeled samples")
    
    # Load model
    classifier = PubMedBERTClassifier(model_path=str(model_path))
    
    label_map = {
        "associated": "associated",
        "not_associated": "not_associated",
        "incidental": "incidental",
        "positive": "associated",
        "negative": "not_associated",
        "no_association": "incidental",
    }

    # Get predictions
    sentences = [d['sentence'] for d in labeled]
    true_labels = [label_map.get(d['manual_label'], d['manual_label']) for d in labeled]
    
    pred_labels = []
    for sent in sentences:
        label, conf = classifier.predict(sent)
        pred_labels.append(label_map.get(label, label))
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=0)
    
    print(f"\n{'='*50}")
    print(f"HELD-OUT TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"\n{report}")
    
    # Show misclassifications
    print(f"\n{'='*50}")
    print("MISCLASSIFICATIONS:")
    print(f"{'='*50}")
    for i, (true, pred, sent) in enumerate(zip(true_labels, pred_labels, sentences)):
        if true != pred:
            print(f"\n[{i+1}] True: {true}, Predicted: {pred}")
            print(f"    {sent[:150]}...")


if __name__ == "__main__":
    evaluate_holdout(
        test_path=Path("data/splits/holdout.json"),
        model_path=Path("models/scibert-hfpef-v4/final"),
    )

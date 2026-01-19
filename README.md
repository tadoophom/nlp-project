# HFpEF Protein-Disease Classification

SciBERT-based classifier for filtering CaseOLAP protein results. Identifies whether scientific sentences describe positive, negative, or no association between proteins and HFpEF.

## Results

| Model | Accuracy |
|-------|----------|
| Rule-based | 40.2% |
| PubMedBERT | 71.9% |
| SciBERT v4 | 97.0% |
| Multi-label | 97.8% |

Held-out validation: **98%**

## Project Structure

```
├── src/
│   ├── bert_classifier.py    # Core SciBERT classifier
│   ├── caseolap_filter.py    # CaseOLAP pipeline integration
│   ├── explainability.py     # Classification explanations
│   └── nlp_utils.py          # Rule-based baseline
│
├── scripts/
│   ├── training/
│   │   └── train_bert.py     # Model training
│   └── evaluation/
│       ├── evaluate_holdout.py    # Validation
│       └── final_comparison.py    # Generate dashboard
│
├── models/                   # Trained models (git-ignored)
├── data/                     # Training data (git-ignored)
└── deliverable_email/        # Reports and visualizations
```

## Usage

```bash
# Train model
uv run python scripts/training/train_bert.py --data data/labeled.json --output models/new-model

# Evaluate
uv run python scripts/evaluation/evaluate_holdout.py

# Generate comparison dashboard
uv run python scripts/evaluation/final_comparison.py
```

## Classification

```python
from src.bert_classifier import PubMedBERTClassifier

clf = PubMedBERTClassifier(model_path="models/scibert-hfpef-v4/final")
label, confidence = clf.predict("BNP is elevated in HFpEF patients.")
# ('positive', 0.99)
```

## CaseOLAP Integration

```python
from src.caseolap_filter import CaseOLAPFilter

filter = CaseOLAPFilter()
filtered_df = filter.filter_dataframe(caseolap_results)
# Removes proteins with negative/weak evidence
```

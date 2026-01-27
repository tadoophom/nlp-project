# HFpEF Protein-Disease Classification

SciBERT-based classifier for filtering CaseOLAP protein results. Identifies whether scientific sentences describe associated, not_associated, or incidental relationships between proteins and HFpEF.

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
├── data/                     # Datasets and benchmarks (git-ignored)
│   ├── annotation/           # Manual labeling materials
│   ├── benchmarks/           # Benchmark datasets + reports
│   ├── corpus/               # HFpEF corpus exports
│   ├── raw/                  # Raw inputs
│   ├── review/               # Review queues
│   └── splits/               # Train/holdout splits
└── deliverable_email/        # Date-stamped email attachments
```

## Usage

```bash
# Train model
uv run python scripts/training/train_bert.py --data data/splits/train.json --output models/new-model

# Evaluate
uv run python scripts/evaluation/evaluate_holdout.py

# Generate comparison dashboard
uv run python scripts/evaluation/final_comparison.py
```

## BioRED Benchmark Download

The BioRED files are not stored in the repository. Download them to the path expected by `scripts/validate_on_biored.py`.

```bash
mkdir -p data/benchmarks/biored
curl -L ftp://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/BIORED.zip -o data/benchmarks/biored/BIORED.zip
unzip -o data/benchmarks/biored/BIORED.zip -d data/benchmarks/biored
```

Then run:

```bash
uv run python scripts/validate_on_biored.py
```

## Classification

```python
from src.bert_classifier import PubMedBERTClassifier

clf = PubMedBERTClassifier(model_path="models/scibert-hfpef-v4/final")
label, confidence = clf.predict("BNP is elevated in HFpEF patients.")
# ('associated', 0.99)
```

## CaseOLAP Integration

```python
from src.caseolap_filter import CaseOLAPFilter

filter = CaseOLAPFilter()
filtered_df = filter.filter_dataframe(caseolap_results)
# Removes proteins with not_associated or incidental evidence
```

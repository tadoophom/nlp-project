# HFpEF Protein-Disease Relation Classifier

A SciBERT-based classifier for identifying protein-disease associations in biomedical literature, specifically for Heart Failure with Preserved Ejection Fraction (HFpEF).

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Rule-based (spaCy) | 40.2% | 0.265 |
| PubMedBERT v1 | 71.9% | 0.685 |
| **SciBERT v4** | **97.0%** | **0.969** |

## Project Structure

```
├── src/                    # Core library code
│   ├── bert_classifier.py  # SciBERT classifier wrapper
│   ├── nlp_utils.py        # Rule-based NLP utilities
│   └── ...
├── scripts/                # Training and evaluation scripts
│   ├── train_bert.py       # Model training
│   ├── evaluate_classifier.py
│   ├── comprehensive_comparison.py
│   └── ...
├── data/                   # Training data (gitignored)
│   ├── labeled.json        # 1,481 labeled sentences
│   └── hfpef_corpus.csv    # Full corpus
├── models/                 # Trained models (gitignored)
│   ├── pubmedbert-hfpef/   # Initial model
│   └── scibert-hfpef-v4/   # Final model (97% accuracy)
└── streamlit_app/          # Interactive demo app
```

## Usage

### Training
```bash
uv run python scripts/train_bert.py \
    --data data/labeled.json \
    --output models/scibert-new \
    --lr 1e-5 --epochs 8
```

### Evaluation
```bash
uv run python scripts/evaluate_classifier.py \
    --data data/labeled.json \
    --bert-model models/scibert-hfpef-v4/final
```

### Inference
```python
from src.bert_classifier import PubMedBERTClassifier

clf = PubMedBERTClassifier(model_path="models/scibert-hfpef-v4/final")
label, confidence = clf.predict("BNP is elevated in HFpEF patients.")
# label: "positive", confidence: 0.95
```

## Classes

- **positive**: Sentence indicates protein-disease association
- **negative**: Sentence explicitly negates association
- **no_association**: Methodology, study design, or no claim about relationship

## Dependencies

```bash
uv sync
```

"""Fine-tune PubMedBERT with k-fold cross-validation and hyperparameter tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import RelationDataset, LABEL2ID, ID2LABEL, load_labeled_data

# Best biomedical models to try
BIOMEDICAL_MODELS = {
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-v1.1",
    "scibert": "allenai/scibert_scivocab_uncased",
    "biomedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "pubmedbert-large": "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract",
}


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    return {cls: total / (n_classes * count) for cls, count in counts.items()}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.class_weights is not None:
            weight = torch.tensor(
                [self.class_weights[i] for i in range(len(self.class_weights))],
                device=logits.device, dtype=logits.dtype
            )
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_fold(
    train_sents, train_labels, val_sents, val_labels,
    model_name: str, output_dir: Path, fold: int,
    epochs: int, batch_size: int, learning_rate: float,
    use_class_weights: bool = True,
):
    """Train a single fold."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABEL2ID), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    
    train_dataset = RelationDataset(train_sents, train_labels, tokenizer)
    val_dataset = RelationDataset(val_sents, val_labels, tokenizer)
    
    fold_dir = output_dir / f"fold_{fold}"
    
    training_args = TrainingArguments(
        output_dir=str(fold_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_dir=str(fold_dir / "logs"),
        logging_steps=10,
        seed=42 + fold,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )
    
    class_weights = compute_class_weights(train_labels) if use_class_weights else None
    
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    
    trainer.train()
    
    # Evaluate
    preds_output = trainer.predict(val_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    
    return {
        "accuracy": accuracy_score(val_labels, preds),
        "f1_macro": f1_score(val_labels, preds, average="macro"),
        "f1_weighted": f1_score(val_labels, preds, average="weighted"),
        "predictions": preds.tolist(),
        "labels": val_labels,
    }


def cross_validate(
    data_path: Path,
    output_dir: Path,
    model_key: str = "pubmedbert",
    n_folds: int = 5,
    epochs: int = 8,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
):
    """Run k-fold cross-validation."""
    sentences, labels = load_labeled_data(data_path)
    print(f"Loaded {len(sentences)} samples")
    print(f"Label distribution: {Counter(labels)}")
    
    model_name = BIOMEDICAL_MODELS.get(model_key, model_key)
    print(f"\nModel: {model_name}")
    print(f"Hyperparameters: lr={learning_rate}, epochs={epochs}, batch_size={batch_size}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    all_results = []
    all_preds = []
    all_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(sentences, labels)):
        print(f"\n{'='*50}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        train_sents = [sentences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_sents = [sentences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print(f"Train: {len(train_sents)}, Val: {len(val_sents)}")
        
        result = train_fold(
            train_sents, train_labels, val_sents, val_labels,
            model_name, output_dir, fold,
            epochs, batch_size, learning_rate,
        )
        
        all_results.append(result)
        all_preds.extend(result["predictions"])
        all_labels.extend(result["labels"])
        
        print(f"Fold {fold + 1} - Acc: {result['accuracy']:.3f}, F1: {result['f1_weighted']:.3f}")
    
    # Aggregate results
    print(f"\n{'='*50}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*50}")
    
    accs = [r["accuracy"] for r in all_results]
    f1s = [r["f1_weighted"] for r in all_results]
    f1_macros = [r["f1_macro"] for r in all_results]
    
    print(f"Accuracy: {np.mean(accs):.3f} (+/- {np.std(accs):.3f})")
    print(f"F1 Weighted: {np.mean(f1s):.3f} (+/- {np.std(f1s):.3f})")
    print(f"F1 Macro: {np.mean(f1_macros):.3f} (+/- {np.std(f1_macros):.3f})")
    
    # Overall classification report
    print("\nOverall Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=list(LABEL2ID.keys())))
    
    # Save results
    results_summary = {
        "model": model_name,
        "n_folds": n_folds,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        "metrics": {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "f1_weighted_mean": float(np.mean(f1s)),
            "f1_weighted_std": float(np.std(f1s)),
            "f1_macro_mean": float(np.mean(f1_macros)),
            "f1_macro_std": float(np.std(f1_macros)),
        },
        "fold_results": [
            {"fold": i, "accuracy": r["accuracy"], "f1_weighted": r["f1_weighted"]}
            for i, r in enumerate(all_results)
        ],
    }
    
    with open(output_dir / "cv_results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'cv_results.json'}")
    
    return results_summary


def compare_models(data_path: Path, output_dir: Path, models: list[str] = None):
    """Compare multiple models using cross-validation."""
    if models is None:
        models = ["pubmedbert", "biobert", "scibert"]
    
    results = {}
    for model_key in models:
        print(f"\n{'#'*60}")
        print(f"# EVALUATING: {model_key}")
        print(f"{'#'*60}")
        
        model_dir = output_dir / model_key
        result = cross_validate(data_path, model_dir, model_key=model_key)
        results[model_key] = result
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':<15} {'F1 Weighted':<15} {'F1 Macro':<15}")
    print("-" * 65)
    
    for model_key, result in results.items():
        m = result["metrics"]
        print(f"{model_key:<20} {m['accuracy_mean']:.3f} +/- {m['accuracy_std']:.3f}   "
              f"{m['f1_weighted_mean']:.3f} +/- {m['f1_weighted_std']:.3f}   "
              f"{m['f1_macro_mean']:.3f} +/- {m['f1_macro_std']:.3f}")
    
    # Find best
    best = max(results.items(), key=lambda x: x[1]["metrics"]["f1_weighted_mean"])
    print(f"\nBest model: {best[0]} (F1: {best[1]['metrics']['f1_weighted_mean']:.3f})")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to labeled JSON")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="pubmedbert", 
                        choices=list(BIOMEDICAL_MODELS.keys()) + ["compare"],
                        help="Model to use or 'compare' to try multiple")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()
    
    if args.model == "compare":
        compare_models(Path(args.data), Path(args.output))
    else:
        cross_validate(
            Path(args.data), Path(args.output),
            model_key=args.model,
            n_folds=args.folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )


if __name__ == "__main__":
    main()

"""Fine-tune PubMedBERT for protein-disease relation classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import (
    RelationDataset,
    LABEL2ID,
    ID2LABEL,
    MODEL_NAME,
    load_labeled_data,
)


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}


def compute_class_weights(labels: list[int]) -> dict[int, float]:
    """Compute class weights for imbalanced data."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(counts)
    weights = {cls: total / (n_classes * count) for cls, count in counts.items()}
    return weights


def train(
    data_path: Path,
    output_dir: Path,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.15,
    seed: int = 42,
    use_class_weights: bool = True,
):
    """Fine-tune PubMedBERT on labeled data."""
    sentences, labels = load_labeled_data(data_path)
    print(f"Loaded {len(sentences)} labeled sentences")
    
    # Handle class imbalance
    from collections import Counter
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")
    
    # Check if we can stratify (need at least 2 samples per class)
    can_stratify = all(count >= 2 for count in label_counts.values())
    
    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=val_split, random_state=seed,
        stratify=labels if can_stratify else None
    )
    print(f"Train: {len(train_sents)}, Val: {len(val_sents)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = RelationDataset(train_sents, train_labels, tokenizer)
    val_dataset = RelationDataset(val_sents, val_labels, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    # Custom trainer with class weights for imbalanced data
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
    
    class_weights = compute_class_weights(train_labels) if use_class_weights else None
    if class_weights:
        print(f"Using class weights: {class_weights}")
    
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

    # Save final model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")

    # Evaluation on validation set
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, target_names=list(LABEL2ID.keys())))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, preds))

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PubMedBERT")
    parser.add_argument("--data", required=True, help="Path to labeled JSON data")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--no-class-weights", action="store_true", 
                        help="Disable class weighting for imbalanced data")
    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        use_class_weights=not args.no_class_weights,
    )


if __name__ == "__main__":
    main()

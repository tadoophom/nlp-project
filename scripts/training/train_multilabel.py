"""Fine-tune a multi-label SciBERT/PubMedBERT classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bert_classifier import LABEL2ID, MODEL_NAME, get_model_name, load_labeled_data


class MultiLabelDataset(Dataset):
    def __init__(self, sentences: List[str], labels: List[List[int]], tokenizer, max_length: int = 256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


def to_multilabel(labels: List[int], num_labels: int) -> List[List[int]]:
    out = []
    for y in labels:
        vec = [0] * num_labels
        if 0 <= y < num_labels:
            vec[y] = 1
        out.append(vec)
    return out


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (logits > 0).astype(int)
    true = labels.astype(int)
    exact = (preds == true).all(axis=1).mean()
    return {"exact_match": exact}


def train(
    data_path: Path,
    output_dir: Path,
    model_name: str = MODEL_NAME,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.15,
    seed: int = 42,
):
    sentences, labels = load_labeled_data(data_path)
    num_labels = len(LABEL2ID)
    ml_labels = to_multilabel(labels, num_labels)

    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, ml_labels, test_size=val_split, random_state=seed, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    train_dataset = MultiLabelDataset(train_sents, train_labels, tokenizer)
    val_dataset = MultiLabelDataset(val_sents, val_labels, tokenizer)

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
        metric_for_best_model="exact_match",
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    class MultiLabelTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune multi-label classifier")
    parser.add_argument("--data", required=True, help="Path to labeled JSON data")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="HuggingFace model name or key (e.g., pubmedbert, scibert)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        model_name=get_model_name(args.model),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()

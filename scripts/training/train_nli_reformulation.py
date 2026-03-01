"""Train an NLI reformulation model for relation classification."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import get_model_name

LABELS = ["associated", "not_associated", "incidental"]
HYPOTHESES = {
    "associated": "The sentence states that the protein is associated with the disease.",
    "not_associated": "The sentence states that the protein is not associated with the disease.",
    "incidental": "The sentence mentions protein and disease but gives no association claim.",
}


@dataclass
class PairRow:
    premise: str
    hypothesis: str
    label: int  # 1 entailment, 0 not_entailment


class NLIDataset(Dataset):
    def __init__(self, rows: list[PairRow], tokenizer, max_length: int = 256):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row.premise,
            row.hypothesis,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row.label, dtype=torch.long)
        return item


def build_pairs(data: list[dict]) -> list[PairRow]:
    rows = []
    for item in data:
        sentence = item["sentence"]
        true_label = item["label"]
        for label in LABELS:
            rows.append(PairRow(
                premise=sentence,
                hypothesis=HYPOTHESES[label],
                label=1 if label == true_label else 0,
            ))
    return rows


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary", zero_division=0),
    }


def class_weights(rows: list[PairRow]) -> torch.Tensor:
    counts = {0: 0, 1: 0}
    for row in rows:
        counts[row.label] += 1
    total = counts[0] + counts[1]
    weights = {
        0: total / (2 * max(counts[0], 1)),
        1: total / (2 * max(counts[1], 1)),
    }
    return torch.tensor([weights[0], weights[1]], dtype=torch.float32)


def train(args: argparse.Namespace):
    with open(args.data) as f:
        data = json.load(f)

    rows = build_pairs(data)
    labels = [row.label for row in rows]

    train_rows, val_rows = train_test_split(
        rows,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels,
    )

    model_name = get_model_name(args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "not_entailment", 1: "entailment"},
        label2id={"not_entailment": 0, "entailment": 1},
        ignore_mismatched_sizes=True,
    )

    train_dataset = NLIDataset(train_rows, tokenizer, max_length=args.max_length)
    val_dataset = NLIDataset(val_rows, tokenizer, max_length=args.max_length)

    weights = class_weights(train_rows)

    training_args = TrainingArguments(
        output_dir=str(Path(args.output)),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(Path(args.output) / "logs"),
        logging_steps=25,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels_local = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(logits.device))
            loss = loss_fn(logits, labels_local)
            return (loss, outputs) if return_outputs else loss

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    final_path = Path(args.output) / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    print(f"Saved NLI model to {final_path}")
    print(f"Train pairs: {len(train_rows)} | Val pairs: {len(val_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NLI reformulation model")
    parser.add_argument("--data", required=True, help="Sentence-label JSON dataset")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--model", default="pubmedbert", help="Model key or HF model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

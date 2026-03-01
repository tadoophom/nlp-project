"""Train a two-stage binary cascade for 3-class relation classification.

Stage 1: not_associated vs rest (associated + incidental)
Stage 2: associated vs incidental (only on examples predicted as 'rest')

This directly targets the hard associated/incidental boundary
with a dedicated binary classifier.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import (
    LABEL2ID,
    ID2LABEL,
    BIOMEDICAL_MODELS,
    get_model_name,
    preprocess_sentence,
)


class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=256):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.sentences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=-1)
        targets_oh = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_oh).sum(dim=-1)
        focal_weight = (1 - pt) ** self.gamma
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets_oh * log_probs).sum(dim=-1)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce).mean()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": acc, "f1": f1}


def train_stage(
    sentences, labels, label_names, output_dir, model_key, pretrain_path,
    epochs, batch_size, lr, seed, patience, max_length, stage_name,
):
    print(f"\n{'='*60}")
    print(f"Training {stage_name}")
    print(f"{'='*60}")

    model_name = get_model_name(model_key)
    print(f"Model: {model_key} ({model_name})")
    print(f"Samples: {len(sentences)}, Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    can_stratify = all(c >= 2 for c in np.unique(labels, return_counts=True)[1])
    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.15, random_state=seed,
        stratify=labels if can_stratify else None,
    )
    print(f"Train: {len(train_sents)}, Val: {len(val_sents)}")

    load_from = str(pretrain_path) if pretrain_path and pretrain_path.exists() else model_name
    if pretrain_path and pretrain_path.exists():
        print(f"Warm-starting from {pretrain_path}")

    tokenizer_source = str(pretrain_path) if pretrain_path and pretrain_path.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from, num_labels=2, ignore_mismatched_sizes=True,
    )

    train_ds = BinaryDataset(train_sents, train_labels, tokenizer, max_length)
    val_ds = BinaryDataset(val_sents, val_labels, tokenizer, max_length)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
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

    # Class weights
    from collections import Counter
    counts = Counter(train_labels)
    total = len(train_labels)
    weight_tensor = torch.tensor([total / (2 * counts[i]) for i in range(2)])
    loss_fn = FocalLoss(alpha=weight_tensor, gamma=2.0)

    class CustomTrainer(Trainer):
        def __init__(self, loss_fn=None, **kwargs):
            super().__init__(**kwargs)
            self.loss_fn = loss_fn

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = self.loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        loss_fn=loss_fn,
    )

    trainer.train()

    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    print(f"Saved to {final_path}")

    # Val evaluation
    predictions = trainer.predict(val_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    print(f"\nValidation ({stage_name}):")
    print(classification_report(val_labels, preds, target_names=label_names))

    return final_path


def evaluate_cascade(stage1_path, stage2_path, eval_path, max_length=256, batch_size=64):
    """Run the full cascade and evaluate."""
    print(f"\n{'='*60}")
    print("CASCADE EVALUATION")
    print(f"{'='*60}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    with open(eval_path) as f:
        rows = json.load(f)
    sentences = [preprocess_sentence(r["sentence"]) for r in rows]
    gold_labels = [r["label"] for r in rows]

    # Stage 1: not_associated vs rest
    tok1 = AutoTokenizer.from_pretrained(str(stage1_path))
    model1 = AutoModelForSequenceClassification.from_pretrained(str(stage1_path))
    model1.eval()

    stage1_preds = []
    stage1_probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        inputs = tok1(batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model1(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).tolist()
        stage1_preds.extend(preds)
        stage1_probs.extend(probs.numpy())

    # Stage 2: associated vs incidental (only for stage1 == "rest" i.e. label 0)
    tok2 = AutoTokenizer.from_pretrained(str(stage2_path))
    model2 = AutoModelForSequenceClassification.from_pretrained(str(stage2_path))
    model2.eval()

    rest_indices = [i for i, p in enumerate(stage1_preds) if p == 0]
    rest_sentences = [sentences[i] for i in rest_indices]

    stage2_preds = {}
    stage2_probs = {}
    for i in range(0, len(rest_sentences), batch_size):
        batch = rest_sentences[i:i+batch_size]
        indices = rest_indices[i:i+batch_size]
        inputs = tok2(batch, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            logits = model2(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1).tolist()
        for j, idx in enumerate(indices):
            stage2_preds[idx] = preds[j]
            stage2_probs[idx] = probs[j].numpy()

    # Combine predictions
    final_preds = []
    for i in range(len(sentences)):
        if stage1_preds[i] == 1:
            final_preds.append("not_associated")
        else:
            s2_pred = stage2_preds[i]
            final_preds.append("associated" if s2_pred == 0 else "incidental")

    # Metrics
    acc = accuracy_score(gold_labels, final_preds)
    macro_f1 = f1_score(gold_labels, final_preds, average="macro", zero_division=0)
    label_names = ["associated", "not_associated", "incidental"]

    print(f"\nCascade accuracy: {acc:.4f}")
    print(f"Cascade macro F1: {macro_f1:.4f}")
    print(f"\nStage 1 routed {len(rest_indices)} to Stage 2, "
          f"{len(sentences) - len(rest_indices)} directly to not_associated")
    print(classification_report(gold_labels, final_preds, labels=label_names))
    print("Confusion matrix:")
    print(confusion_matrix(gold_labels, final_preds, labels=label_names))

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "stage1_rest_count": len(rest_indices),
        "stage1_not_assoc_count": len(sentences) - len(rest_indices),
        "confusion_matrix": confusion_matrix(gold_labels, final_preds, labels=label_names).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train binary cascade classifier")
    parser.add_argument("--data", required=True, help="Training data JSON (3-class)")
    parser.add_argument("--eval", required=True, help="Eval data JSON (3-class)")
    parser.add_argument("--output-dir", required=True, help="Base output directory")
    parser.add_argument("--model", default="pubmedbert", choices=list(BIOMEDICAL_MODELS.keys()))
    parser.add_argument("--pretrain-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    all_sentences = [preprocess_sentence(r["sentence"]) for r in data]
    all_labels_str = [r["label"] for r in data]

    output_base = Path(args.output_dir)
    pretrain = Path(args.pretrain_path) if args.pretrain_path else None

    # Stage 1: not_associated (1) vs rest (0)
    stage1_labels = [1 if l == "not_associated" else 0 for l in all_labels_str]
    stage1_path = train_stage(
        sentences=all_sentences,
        labels=stage1_labels,
        label_names=["rest", "not_associated"],
        output_dir=output_base / "stage1_not_assoc_vs_rest",
        model_key=args.model,
        pretrain_path=pretrain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        max_length=args.max_length,
        stage_name="Stage 1: not_associated vs rest",
    )

    # Clean checkpoints
    for ckpt in (output_base / "stage1_not_assoc_vs_rest").glob("checkpoint-*"):
        import shutil
        shutil.rmtree(ckpt)

    # Stage 2: associated (0) vs incidental (1) — only assoc+incidental examples
    stage2_sents = [s for s, l in zip(all_sentences, all_labels_str) if l != "not_associated"]
    stage2_labels = [0 if l == "associated" else 1 for l in all_labels_str if l != "not_associated"]
    stage2_path = train_stage(
        sentences=stage2_sents,
        labels=stage2_labels,
        label_names=["associated", "incidental"],
        output_dir=output_base / "stage2_assoc_vs_incidental",
        model_key=args.model,
        pretrain_path=pretrain,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        patience=args.patience,
        max_length=args.max_length,
        stage_name="Stage 2: associated vs incidental",
    )

    # Clean checkpoints
    for ckpt in (output_base / "stage2_assoc_vs_incidental").glob("checkpoint-*"):
        import shutil
        shutil.rmtree(ckpt)

    # Evaluate cascade
    results = evaluate_cascade(stage1_path, stage2_path, args.eval, args.max_length, args.batch_size)

    results_path = output_base / "cascade_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

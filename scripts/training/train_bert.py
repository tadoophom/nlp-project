"""Fine-tune PubMedBERT for protein-disease relation classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import (
    RelationDataset,
    LABEL2ID,
    ID2LABEL,
    MODEL_NAME,
    BIOMEDICAL_MODELS,
    get_model_name,
    load_labeled_data,
)


class FocalLoss(torch.nn.Module):
    """Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1 - pt) ** self.gamma

        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(targets_one_hot * log_probs).sum(dim=-1)

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        return (focal_weight * ce).mean()


class LabelSmoothingLoss(torch.nn.Module):
    """Cross-entropy with asymmetric label smoothing.

    Smooths associated<->incidental more than not_associated,
    reflecting the genuine ambiguity at that boundary.
    """

    def __init__(self, smoothing: float = 0.1, weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        # Asymmetric smooth matrix: rows=true label, cols=soft target
        # associated(0) leaks more toward incidental(2)
        # incidental(2) leaks more toward associated(0)
        # not_associated(1) leaks uniformly but less
        s = smoothing
        self.soft_targets = torch.tensor([
            [1 - s,       s * 0.2, s * 0.8],   # associated -> mostly leaks to incidental
            [s * 0.5,     1 - s,   s * 0.5],   # not_associated -> leaks uniformly
            [s * 0.8,     s * 0.2, 1 - s],     # incidental -> mostly leaks to associated
        ])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        soft = self.soft_targets.to(logits.device)[targets]  # (B, C)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(soft * log_probs).sum(dim=-1)
        if self.weight is not None:
            w = self.weight.to(logits.device)[targets]
            loss = loss * w
        return loss.mean()


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
    model_key: str = "scibert",
    epochs: int = 8,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    val_split: float = 0.15,
    seed: int = 42,
    use_class_weights: bool = True,
    loss_type: str = "weighted_ce",
    focal_gamma: float = 2.0,
    patience: int = 3,
    pretrain_path: Path | None = None,
    class_weight_scale_associated: float = 1.0,
    class_weight_scale_not_associated: float = 1.0,
    class_weight_scale_incidental: float = 1.0,
    max_length: int = 256,
    rdrop_alpha: float = 0.0,
    label_smoothing: float = 0.0,
):
    """Fine-tune a biomedical BERT model on labeled data."""
    model_name = get_model_name(model_key)
    print(f"Using model: {model_key} ({model_name})")

    sentences, labels = load_labeled_data(data_path)
    print(f"Loaded {len(sentences)} labeled sentences")

    from collections import Counter
    label_counts = Counter(labels)
    print(f"Label distribution: {dict(label_counts)}")

    can_stratify = all(count >= 2 for count in label_counts.values())

    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=val_split, random_state=seed,
        stratify=labels if can_stratify else None
    )
    print(f"Train: {len(train_sents)}, Val: {len(val_sents)}")

    # 2-stage: load from pretrained checkpoint if available, else from HF hub
    load_from = str(pretrain_path) if pretrain_path and pretrain_path.exists() else model_name
    if pretrain_path and pretrain_path.exists():
        print(f"Loading pre-trained checkpoint from {pretrain_path}")

    tokenizer_source = str(pretrain_path) if pretrain_path and pretrain_path.exists() else model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        load_from,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = RelationDataset(train_sents, train_labels, tokenizer, max_length=max_length)
    val_dataset = RelationDataset(val_sents, val_labels, tokenizer, max_length=max_length)

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

    class CustomTrainer(Trainer):
        def __init__(self, loss_fn=None, rdrop_alpha: float = 0.0, **kwargs):
            super().__init__(**kwargs)
            self.loss_fn = loss_fn
            self.rdrop_alpha = rdrop_alpha

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.loss_fn(logits, labels)

            if self.rdrop_alpha > 0 and model.training:
                outputs2 = model(**inputs)
                logits2 = outputs2.logits
                loss2 = self.loss_fn(logits2, labels)

                p = F.log_softmax(logits, dim=-1)
                q = F.log_softmax(logits2, dim=-1)
                p_prob = p.exp()
                q_prob = q.exp()
                kl = F.kl_div(p, q_prob, reduction="batchmean") + F.kl_div(q, p_prob, reduction="batchmean")
                loss = (loss + loss2) / 2 + self.rdrop_alpha * kl / 2

            return (loss, outputs) if return_outputs else loss

    class_weights = compute_class_weights(train_labels) if use_class_weights else None
    if class_weights:
        weight_scales = {
            LABEL2ID["associated"]: class_weight_scale_associated,
            LABEL2ID["not_associated"]: class_weight_scale_not_associated,
            LABEL2ID["incidental"]: class_weight_scale_incidental,
        }
        class_weights = {
            cls: class_weights[cls] * weight_scales.get(cls, 1.0)
            for cls in class_weights
        }
    if class_weights:
        print(f"Using class weights: {class_weights}")

    loss_desc = f"Loss: {loss_type}"
    if loss_type == "focal":
        loss_desc += f" (gamma={focal_gamma})"
    if label_smoothing > 0:
        loss_desc += f" + label_smoothing={label_smoothing}"
    if rdrop_alpha > 0:
        loss_desc += f" + rdrop_alpha={rdrop_alpha}"
    print(loss_desc)

    weight_tensor = None
    if class_weights:
        weight_tensor = torch.tensor([class_weights[i] for i in range(len(class_weights))])

    if label_smoothing > 0:
        loss_fn = LabelSmoothingLoss(smoothing=label_smoothing, weight=weight_tensor)
    elif loss_type == "focal":
        loss_fn = FocalLoss(alpha=weight_tensor, gamma=focal_gamma)
    elif weight_tensor is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
        loss_fn=loss_fn,
        rdrop_alpha=rdrop_alpha,
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
    present_labels = sorted(set(val_labels) | set(preds))
    target_names = [ID2LABEL[i] for i in present_labels]
    print("\nClassification Report:")
    print(classification_report(val_labels, preds, labels=present_labels, target_names=target_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(val_labels, preds, labels=present_labels))

    return trainer


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a biomedical BERT model")
    parser.add_argument("--data", required=True, help="Path to labeled JSON data")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--model", default="scibert",
                        choices=list(BIOMEDICAL_MODELS.keys()),
                        help="Model to fine-tune (default: scibert)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--no-class-weights", action="store_true",
                        help="Disable class weighting for imbalanced data")
    parser.add_argument("--loss", choices=["weighted_ce", "focal"], default="weighted_ce",
                        help="Loss function (default: weighted_ce)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Gamma for focal loss (default: 2.0)")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--pretrain-path", type=str, default=None,
                        help="Path to pre-trained checkpoint for 2-stage training")
    parser.add_argument("--class-weight-scale-associated", type=float, default=1.0)
    parser.add_argument("--class-weight-scale-not-associated", type=float, default=1.0)
    parser.add_argument("--class-weight-scale-incidental", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rdrop-alpha", type=float, default=0.0,
                        help="R-Drop KL divergence weight (0 = disabled)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Asymmetric label smoothing (0 = disabled)")
    args = parser.parse_args()

    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        model_key=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        val_split=args.val_split,
        use_class_weights=not args.no_class_weights,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        patience=args.patience,
        pretrain_path=Path(args.pretrain_path) if args.pretrain_path else None,
        class_weight_scale_associated=args.class_weight_scale_associated,
        class_weight_scale_not_associated=args.class_weight_scale_not_associated,
        class_weight_scale_incidental=args.class_weight_scale_incidental,
        max_length=args.max_length,
        rdrop_alpha=args.rdrop_alpha,
        label_smoothing=args.label_smoothing,
    )


if __name__ == "__main__":
    main()

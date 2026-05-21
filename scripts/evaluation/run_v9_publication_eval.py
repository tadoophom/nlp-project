from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


LABEL_TO_ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
IDX_ASSOC = LABEL_TO_ID["associated"]


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return json.load(handle)


def prediction_to_payload(row: dict) -> dict:
    return {
        "label": row["label"],
        "probs": {label: float(row["probabilities"][label]) for label in LABEL_NAMES},
    }


def predictions_to_probs(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in rows],
        dtype=np.float64,
    )


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(y_true, y_pred, labels=LABEL_NAMES, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABEL_NAMES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABEL_NAMES, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_NAMES).tolist(),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABEL_NAMES
        },
    }


def scale_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
    scaled = np.exp(log_probs / temperature)
    return scaled / scaled.sum(axis=1, keepdims=True)


def predict_with_thresholds(probabilities: np.ndarray, thresholds: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pred_ids = (probabilities / thresholds).argmax(axis=1)
    return pred_ids, [ID_TO_LABEL[int(idx)] for idx in pred_ids]


def fit_temperature_and_thresholds(
    probabilities: np.ndarray,
    y_true: list[str],
    min_associated_recall: float,
) -> dict:
    y_ids = np.asarray([LABEL_TO_ID[label] for label in y_true], dtype=np.int64)
    temperature_grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    threshold_grid = {
        "associated": [0.30, 0.35, 0.40, 0.45, 0.50],
        "not_associated": [0.35, 0.40, 0.45, 0.50, 0.55],
        "incidental": [0.20, 0.25, 0.30, 0.35, 0.40],
    }

    best_temperature = None
    for temperature in temperature_grid:
        scaled = scale_temperature(probabilities, temperature)
        nll = float(log_loss(y_ids, scaled, labels=list(range(len(LABEL_NAMES)))))
        if best_temperature is None or nll < best_temperature["nll"]:
            best_temperature = {"temperature": float(temperature), "nll": nll}

    scaled_val = scale_temperature(probabilities, best_temperature["temperature"])
    best_threshold_result = None
    for assoc_thr, not_thr, incidental_thr in product(
        threshold_grid["associated"],
        threshold_grid["not_associated"],
        threshold_grid["incidental"],
    ):
        thresholds = np.asarray([assoc_thr, not_thr, incidental_thr], dtype=np.float64)
        _, y_pred = predict_with_thresholds(scaled_val, thresholds)
        current_metrics = metrics(y_true, y_pred)
        if current_metrics["per_class"]["associated"]["recall"] < min_associated_recall:
            continue
        candidate = {
            "thresholds": {
                "associated": float(assoc_thr),
                "not_associated": float(not_thr),
                "incidental": float(incidental_thr),
            },
            "metrics": current_metrics,
        }
        if best_threshold_result is None:
            best_threshold_result = candidate
            continue
        current_key = (candidate["metrics"]["accuracy"], candidate["metrics"]["macro_f1"])
        best_key = (best_threshold_result["metrics"]["accuracy"], best_threshold_result["metrics"]["macro_f1"])
        if current_key > best_key:
            best_threshold_result = candidate

    if best_threshold_result is None:
        thresholds = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        _, y_pred = predict_with_thresholds(scaled_val, thresholds)
        best_threshold_result = {
            "thresholds": {label: 1.0 for label in LABEL_NAMES},
            "metrics": metrics(y_true, y_pred),
        }

    return {
        "temperature": best_temperature,
        "thresholds": best_threshold_result["thresholds"],
        "val_metrics": best_threshold_result["metrics"],
        "min_associated_recall": float(min_associated_recall),
    }


def majority_vote(sample_predictions: dict[str, dict]) -> str:
    label_counts = Counter(prediction["label"] for prediction in sample_predictions.values())
    max_count = max(label_counts.values())
    candidates = [label for label, count in label_counts.items() if count == max_count]
    if len(candidates) == 1:
        return candidates[0]
    mean_probs = {
        label: statistics.mean(prediction["probs"][label] for prediction in sample_predictions.values())
        for label in LABEL_NAMES
    }
    return max(candidates, key=lambda label: (mean_probs[label], -LABEL_TO_ID[label]))


def build_per_sample(rows: list[dict], model_predictions: dict[str, list[dict]]) -> list[dict]:
    per_sample = []
    for index, row in enumerate(rows):
        sample_predictions = {
            model_name: model_predictions[model_name][index]
            for model_name in model_predictions
        }
        majority_label = majority_vote(sample_predictions)
        per_sample.append(
            {
                "index": index,
                "sentence": row["sentence"],
                "gold_label": row["label"],
                "model_predictions": sample_predictions,
                "majority_vote": majority_label,
                "correct": majority_label == row["label"],
            }
        )
    return per_sample


def build_summary(rows: list[dict], model_predictions: dict[str, list[dict]]) -> dict:
    y_true = [row["label"] for row in rows]
    per_model = {
        model_name: metrics(y_true, [prediction["label"] for prediction in predictions])
        for model_name, predictions in model_predictions.items()
    }
    majority_predictions = [
        majority_vote({model_name: predictions[index] for model_name, predictions in model_predictions.items()})
        for index in range(len(rows))
    ]
    accuracies = [model_metrics["accuracy"] for model_metrics in per_model.values()]
    return {
        "per_model": per_model,
        "majority_vote": metrics(y_true, majority_predictions),
        "cross_seed_mean_acc": float(statistics.mean(accuracies)),
        "cross_seed_std_acc": float(statistics.stdev(accuracies)),
    }


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the six v9 train109 models and calibrate them.")
    parser.add_argument("--eval-file", default="data/splits/hfpef_v8_eval_relabel4_noleak_large.json")
    parser.add_argument("--train-file", default="data/hfpef_v9_train_autocorrect109.json")
    parser.add_argument("--model-root", default="models/hfpef_v9_train109")
    parser.add_argument("--publication-output", default="logs/v9_publication_eval.json")
    parser.add_argument("--calibrated-output", default="logs/v9_train_calibrated_eval.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--min-associated-recall", type=float, default=0.45)
    args = parser.parse_args()

    eval_path = ROOT / args.eval_file
    train_path = ROOT / args.train_file
    model_root = ROOT / args.model_root
    publication_output = ROOT / args.publication_output
    calibrated_output = ROOT / args.calibrated_output

    eval_rows = load_rows(eval_path)
    train_rows = load_rows(train_path)
    model_paths = sorted(model_root.glob("*/final"))
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found under {model_root}")

    print(f"[setup] eval={eval_path} samples={len(eval_rows)}")
    print(f"[setup] train={train_path} samples={len(train_rows)}")
    print(f"[setup] models={len(model_paths)}")

    train_indices = np.arange(len(train_rows))
    train_labels = [row["label"] for row in train_rows]
    split_train_indices, val_indices = train_test_split(
        train_indices,
        test_size=args.val_fraction,
        random_state=args.split_seed,
        shuffle=True,
        stratify=train_labels,
    )
    val_rows = [train_rows[int(index)] for index in val_indices]
    print(
        f"[split] seed={args.split_seed} train={len(split_train_indices)} val={len(val_indices)} "
        f"val_fraction={args.val_fraction}"
    )

    eval_sentences = [row["sentence"] for row in eval_rows]
    val_sentences = [row["sentence"] for row in val_rows]
    eval_predictions_by_model: dict[str, list[dict]] = {}
    eval_probabilities_by_model: dict[str, np.ndarray] = {}
    val_probabilities_by_model: dict[str, np.ndarray] = {}

    for index, model_path in enumerate(model_paths, start=1):
        model_name = model_path.parent.name
        print(f"[model {index}/{len(model_paths)}] loading {model_name}")
        classifier = PubMedBERTClassifier(model_path=model_path, max_length=args.max_length, device="cpu")

        print(f"[model {index}/{len(model_paths)}] eval inference on {len(eval_sentences)} samples")
        raw_eval_predictions = classifier.classify_batch(eval_sentences, batch_size=args.batch_size)
        eval_predictions = [prediction_to_payload(row) for row in raw_eval_predictions]
        eval_predictions_by_model[model_name] = eval_predictions
        eval_probabilities_by_model[model_name] = predictions_to_probs(raw_eval_predictions)
        eval_metrics = metrics(
            [row["label"] for row in eval_rows],
            [prediction["label"] for prediction in eval_predictions],
        )
        print(
            f"[model {index}/{len(model_paths)}] eval accuracy={eval_metrics['accuracy']:.4f} "
            f"macro_f1={eval_metrics['macro_f1']:.4f}"
        )

        print(f"[model {index}/{len(model_paths)}] val inference on {len(val_sentences)} samples")
        raw_val_predictions = classifier.classify_batch(val_sentences, batch_size=args.batch_size)
        val_probabilities_by_model[model_name] = predictions_to_probs(raw_val_predictions)

        del raw_eval_predictions
        del raw_val_predictions
        del classifier
        gc.collect()

    publication_per_sample = build_per_sample(eval_rows, eval_predictions_by_model)
    publication_payload = {
        "eval_file": args.eval_file,
        "models": list(eval_predictions_by_model.keys()),
        "per_sample": publication_per_sample,
        "summary": build_summary(eval_rows, eval_predictions_by_model),
    }
    save_json(publication_output, publication_payload)
    print(f"[save] publication predictions -> {publication_output}")

    calibrated_predictions_by_model: dict[str, list[dict]] = {}
    calibration_by_model: dict[str, dict] = {}
    val_labels = [row["label"] for row in val_rows]
    for index, model_name in enumerate(eval_predictions_by_model, start=1):
        print(f"[calibration {index}/{len(eval_predictions_by_model)}] fitting {model_name}")
        calibration = fit_temperature_and_thresholds(
            val_probabilities_by_model[model_name],
            val_labels,
            min_associated_recall=args.min_associated_recall,
        )
        calibration_by_model[model_name] = calibration
        scaled_eval_probabilities = scale_temperature(
            eval_probabilities_by_model[model_name],
            calibration["temperature"]["temperature"],
        )
        threshold_array = np.asarray(
            [calibration["thresholds"][label] for label in LABEL_NAMES],
            dtype=np.float64,
        )
        _, calibrated_labels = predict_with_thresholds(scaled_eval_probabilities, threshold_array)
        calibrated_predictions_by_model[model_name] = [
            {
                "label": calibrated_labels[row_index],
                "probs": {
                    label: float(scaled_eval_probabilities[row_index, LABEL_TO_ID[label]])
                    for label in LABEL_NAMES
                },
            }
            for row_index in range(len(eval_rows))
        ]
        print(
            f"[calibration {index}/{len(eval_predictions_by_model)}] "
            f"temperature={calibration['temperature']['temperature']:.2f} "
            f"thresholds={calibration['thresholds']}"
        )

    calibrated_payload = {
        "eval_file": args.eval_file,
        "train_file": args.train_file,
        "split": {
            "seed": args.split_seed,
            "train_size": int(len(split_train_indices)),
            "val_size": int(len(val_indices)),
            "val_fraction": args.val_fraction,
        },
        "models": list(calibrated_predictions_by_model.keys()),
        "per_sample": build_per_sample(eval_rows, calibrated_predictions_by_model),
        "summary": build_summary(eval_rows, calibrated_predictions_by_model),
        "calibration": {"per_model": calibration_by_model},
    }
    save_json(calibrated_output, calibrated_payload)
    print(f"[save] train-calibrated predictions -> {calibrated_output}")
    print("[done] evaluation and calibration complete")


if __name__ == "__main__":
    main()

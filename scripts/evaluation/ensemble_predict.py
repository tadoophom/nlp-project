"""Ensemble prediction from multiple fine-tuned models."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import LABEL2ID, ID2LABEL, preprocess_sentence


LABEL_NAMES = list(LABEL2ID.keys())


def load_model(model_path: str):
    """Load a fine-tuned model and tokenizer."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, tokenizer, device


def get_softmax_probs(model, tokenizer, device, sentences: list[str], batch_size: int = 32) -> np.ndarray:
    """Get softmax probabilities for all sentences. Returns (N, C) array."""
    all_probs = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        inputs = tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=256, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


def ensemble_average(all_probs: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(all_probs, axis=0)
    return stacked.mean(axis=0)


def ensemble_majority(all_preds: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(all_preds, axis=0)
    result = np.zeros(stacked.shape[1], dtype=int)
    for i in range(stacked.shape[1]):
        counts = Counter(stacked[:, i])
        result[i] = counts.most_common(1)[0][0]
    return result


def scale_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    log_p = np.log(np.clip(probs, 1e-10, 1.0))
    scaled = log_p / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def fit_calibration(calib_probs: np.ndarray, calib_ids: np.ndarray, min_assoc_recall: float = 0.45):
    """Find best temperature and thresholds on calibration set."""
    from sklearn.metrics import log_loss

    best_temp = 1.0
    best_nll = float("inf")
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]:
        scaled = scale_temperature(calib_probs, t)
        nll = log_loss(calib_ids, scaled, labels=[0, 1, 2])
        if nll < best_nll:
            best_nll = nll
            best_temp = t

    scaled = scale_temperature(calib_probs, best_temp)

    best_acc = 0.0
    best_f1 = 0.0
    best_thresholds = {"associated": 1.0, "not_associated": 1.0, "incidental": 1.0}
    for a_thr in [0.30, 0.35, 0.40, 0.45, 0.50]:
        for n_thr in [0.35, 0.40, 0.45, 0.50, 0.55]:
            for i_thr in [0.20, 0.25, 0.30, 0.35, 0.40]:
                thr = np.array([a_thr, n_thr, i_thr])
                preds = (scaled / thr).argmax(axis=1)

                assoc_mask = calib_ids == 0
                if assoc_mask.sum() > 0:
                    assoc_recall = (preds[assoc_mask] == 0).mean()
                    if assoc_recall < min_assoc_recall:
                        continue

                acc = (preds == calib_ids).mean()
                pred_labels = [LABEL_NAMES[p] for p in preds]
                true_labels = [LABEL_NAMES[t] for t in calib_ids]
                mf1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

                if acc > best_acc or (acc == best_acc and mf1 > best_f1):
                    best_acc = acc
                    best_f1 = mf1
                    best_thresholds = {"associated": a_thr, "not_associated": n_thr, "incidental": i_thr}

    return {"temperature": best_temp, "thresholds": best_thresholds}


def apply_calibration(probs: np.ndarray, temperature: float, thresholds: dict) -> np.ndarray:
    """Apply learned temperature and thresholds to get predictions."""
    scaled = scale_temperature(probs, temperature)
    thr = np.array([thresholds["associated"], thresholds["not_associated"], thresholds["incidental"]])
    return (scaled / thr).argmax(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Ensemble predictions from multiple models")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--calib-data", default=None,
                        help="Train data for calibration (required for proper calibration)")
    parser.add_argument("--method", choices=["average", "majority", "both"], default="both")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.eval_data) as f:
        eval_data = json.load(f)
    sentences = [preprocess_sentence(d["sentence"]) for d in eval_data]
    true_labels = [d["label"] for d in eval_data]
    true_ids = np.array([LABEL2ID[l] for l in true_labels])

    print(f"Eval set: {len(sentences)} samples")
    print(f"Models: {len(args.models)}")

    all_probs = []
    all_preds = []
    individual_results = []
    for model_path in args.models:
        name = Path(model_path).name or Path(model_path).parent.name
        print(f"  Loading {name} from {model_path}...")
        model, tokenizer, device = load_model(model_path)
        probs = get_softmax_probs(model, tokenizer, device, sentences)
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=-1))

        acc = accuracy_score(true_ids, probs.argmax(axis=-1))
        print(f"    Individual accuracy: {acc:.1%}")
        individual_results.append({"model": model_path, "accuracy": float(acc)})

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def evaluate(pred_ids, method_name):
        pred_labels = [ID2LABEL[i] for i in pred_ids]
        acc = accuracy_score(true_labels, pred_labels)
        macro_f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)
        weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES).tolist()
        print(f"\n--- {method_name} ---")
        print(f"Accuracy:    {acc:.1%}")
        print(f"Macro F1:    {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(classification_report(true_labels, pred_labels, zero_division=0))
        return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "confusion_matrix": cm}

    results = {"individual": individual_results}

    avg_probs = ensemble_average(all_probs)

    if args.method in ("average", "both"):
        avg_preds = avg_probs.argmax(axis=-1)
        results["average"] = evaluate(avg_preds, "Softmax Average Ensemble")

    if args.method in ("majority", "both"):
        maj_preds = ensemble_majority(all_preds)
        results["majority"] = evaluate(maj_preds, "Majority Vote Ensemble")

    # Calibrated evaluation
    if args.calib_data:
        print("\n--- Calibrated Ensemble (train-calibrated) ---")
        with open(args.calib_data) as f:
            calib_data = json.load(f)
        calib_sents = [preprocess_sentence(d["sentence"]) for d in calib_data]
        calib_ids = np.array([LABEL2ID[d["label"]] for d in calib_data])

        calib_probs_list = []
        for model_path in args.models:
            model, tokenizer, device = load_model(model_path)
            probs = get_softmax_probs(model, tokenizer, device, calib_sents)
            calib_probs_list.append(probs)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        calib_probs = ensemble_average(calib_probs_list)

        params = fit_calibration(calib_probs, calib_ids)
        print(f"  Temperature: {params['temperature']}")
        print(f"  Thresholds:  {params['thresholds']}")

        eval_preds = apply_calibration(avg_probs, params["temperature"], params["thresholds"])
        calib_result = evaluate(eval_preds, "Train-Calibrated Ensemble")
        calib_result["temperature"] = params["temperature"]
        calib_result["thresholds"] = params["thresholds"]
        results["calibrated"] = calib_result
    else:
        print("\n--- Calibrated Ensemble (eval-fit, for reference only) ---")
        params = fit_calibration(avg_probs, true_ids)
        eval_preds = apply_calibration(avg_probs, params["temperature"], params["thresholds"])
        calib_result = evaluate(eval_preds, "Eval-Fit Calibrated (overfitted)")
        calib_result["temperature"] = params["temperature"]
        calib_result["thresholds"] = params["thresholds"]
        calib_result["warning"] = "calibrated on eval set - overfitted"
        results["calibrated"] = calib_result

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

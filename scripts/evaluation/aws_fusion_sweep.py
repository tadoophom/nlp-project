"""Sweep fusion weights and class thresholds on HFpEF eval (AWS helper)."""
from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES


def probs_array(results: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in results],
        dtype=np.float64,
    )


def predict(prob: np.ndarray, t_assoc: float, t_not: float, t_inc: float) -> list[str]:
    thr = np.asarray([t_assoc, t_not, t_inc], dtype=np.float64)
    adjusted = prob / thr
    pred_ids = adjusted.argmax(axis=1)
    return [LABEL_NAMES[idx] for idx in pred_ids]


def compute_scores(y_true: list[str], y_pred: list[str]) -> tuple[float, float, float]:
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1)

    label_set = LABEL_NAMES
    f1s = []
    supports = []
    for label in label_set:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        supports.append(support)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / max(len(f1s), 1))
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / max(sum(supports), 1))
    return float(acc), macro_f1, weighted_f1


def main():
    root = Path(__file__).resolve().parent.parent.parent

    with open(root / "data/splits/hfpef_v3_eval.json") as f:
        eval_sentence = json.load(f)
    with open(root / "data/splits/hfpef_v3_eval_context.json") as f:
        eval_context = json.load(f)

    if len(eval_sentence) != len(eval_context):
        raise ValueError("Eval sentence/context length mismatch.")

    true_labels = [row["label"] for row in eval_sentence]
    sentence_texts = [row["sentence"] for row in eval_sentence]
    context_texts = [row["sentence"] for row in eval_context]

    models = {
        "focal": PubMedBERTClassifier(root / "models/hfpef_v3_improved/pubmedbert_focal/final"),
        "context": PubMedBERTClassifier(root / "models/hfpef_v3_improved/pubmedbert_context/final"),
        "cvd_combined": PubMedBERTClassifier(root / "models/hfpef_v3_improved/pubmedbert_cvd_combined/final"),
    }

    p_focal = probs_array(models["focal"].classify_batch(sentence_texts, batch_size=64))
    p_context = probs_array(models["context"].classify_batch(context_texts, batch_size=64))
    p_cvd = probs_array(models["cvd_combined"].classify_batch(sentence_texts, batch_size=64))

    weights = []
    for w_focal in np.arange(0.3, 1.01, 0.1):
        for w_context in np.arange(0.0, 0.71, 0.1):
            w_cvd = 1.0 - w_focal - w_context
            if w_cvd < -1e-9:
                continue
            if w_cvd < 0:
                w_cvd = 0.0
            weights.append((round(float(w_focal), 3), round(float(w_context), 3), round(float(w_cvd), 3)))
    weights = sorted(set(weights))

    assoc_grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]
    not_grid = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0, 1.2]
    inc_grid = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0]

    best_acc = None
    best_macro = None

    for w_focal, w_context, w_cvd in weights:
        prob = w_focal * p_focal + w_context * p_context + w_cvd * p_cvd
        for t_assoc, t_not, t_inc in product(assoc_grid, not_grid, inc_grid):
            pred = predict(prob, t_assoc, t_not, t_inc)
            acc, macro, weighted = compute_scores(true_labels, pred)
            row = {
                "weights": {
                    "focal": w_focal,
                    "context": w_context,
                    "cvd_combined": w_cvd,
                },
                "thresholds": {
                    "associated": t_assoc,
                    "not_associated": t_not,
                    "incidental": t_inc,
                },
                "accuracy": acc,
                "macro_f1": macro,
                "weighted_f1": weighted,
            }

            if best_acc is None or (acc > best_acc["accuracy"]) or (
                acc == best_acc["accuracy"] and macro > best_acc["macro_f1"]
            ):
                best_acc = row

            if best_macro is None or (macro > best_macro["macro_f1"]) or (
                macro == best_macro["macro_f1"] and acc > best_macro["accuracy"]
            ):
                best_macro = row

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "timestamp": stamp,
        "search_space": {
            "weights_count": len(weights),
            "assoc_grid": assoc_grid,
            "not_grid": not_grid,
            "inc_grid": inc_grid,
        },
        "best_by_accuracy": best_acc,
        "best_by_macro_f1": best_macro,
    }

    out_dir = root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fusion_triple_sweep_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("best_by_accuracy", best_acc)
    print("best_by_macro_f1", best_macro)
    print("saved", out_path)


if __name__ == "__main__":
    main()

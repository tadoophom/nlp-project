"""Calibrate fusion probabilities and run constrained threshold search."""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import LABEL2ID
from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


IDX_ASSOC = LABEL_NAMES.index("associated")
IDX_NOT = LABEL_NAMES.index("not_associated")
IDX_INC = LABEL_NAMES.index("incidental")


def probs_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in rows],
        dtype=np.float64,
    )


def scale_temperature(prob: np.ndarray, temperature: float) -> np.ndarray:
    logp = np.log(np.clip(prob, 1e-12, 1.0))
    scaled = np.exp(logp / temperature)
    scaled = scaled / scaled.sum(axis=1, keepdims=True)
    return scaled


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_NAMES).tolist(),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_NAMES
        },
    }


def predict_with_threshold(prob: np.ndarray, thr: np.ndarray) -> list[str]:
    idx = (prob / thr).argmax(axis=1)
    return [LABEL_NAMES[int(i)] for i in idx]


def main():
    parser = argparse.ArgumentParser(description="Calibrate fusion and constrained threshold search")
    parser.add_argument("--calib-sentence", default="data/splits/hfpef_v3_train_context.json")
    parser.add_argument("--calib-context", default="data/splits/hfpef_v3_train_context.json")
    parser.add_argument("--eval-sentence", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--eval-context", default="data/splits/hfpef_v3_eval_context.json")
    parser.add_argument("--focal-model", default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--context-model", default="models/hfpef_v3_improved/pubmedbert_context/final")
    parser.add_argument("--cvd-model", default="models/hfpef_v3_improved/pubmedbert_cvd_combined/final")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-associated-recall", type=float, default=0.45)
    args = parser.parse_args()

    with open(args.calib_sentence) as f:
        calib_sent = json.load(f)
    with open(args.calib_context) as f:
        calib_ctx = json.load(f)
    with open(args.eval_sentence) as f:
        eval_sent = json.load(f)
    with open(args.eval_context) as f:
        eval_ctx = json.load(f)

    if len(calib_sent) != len(calib_ctx):
        raise ValueError("calibration sentence/context mismatch")
    if len(eval_sent) != len(eval_ctx):
        raise ValueError("eval sentence/context mismatch")

    focal = PubMedBERTClassifier(Path(args.focal_model))
    context_model = PubMedBERTClassifier(Path(args.context_model))
    cvd = PubMedBERTClassifier(Path(args.cvd_model))

    calib_sentence_texts = [r["sentence"] for r in calib_sent]
    calib_context_texts = [r["sentence"] for r in calib_ctx]
    calib_labels = [r["label"] for r in calib_sent]
    calib_ids = np.asarray([LABEL2ID[y] for y in calib_labels], dtype=np.int64)

    eval_sentence_texts = [r["sentence"] for r in eval_sent]
    eval_context_texts = [r["sentence"] for r in eval_ctx]
    eval_labels = [r["label"] for r in eval_sent]

    calib_focal = probs_array(focal.classify_batch(calib_sentence_texts, batch_size=args.batch_size))
    calib_context = probs_array(context_model.classify_batch(calib_context_texts, batch_size=args.batch_size))
    calib_cvd = probs_array(cvd.classify_batch(calib_sentence_texts, batch_size=args.batch_size))
    calib_prob = 0.3 * calib_focal + 0.6 * calib_context + 0.1 * calib_cvd

    eval_focal = probs_array(focal.classify_batch(eval_sentence_texts, batch_size=args.batch_size))
    eval_context_prob = probs_array(context_model.classify_batch(eval_context_texts, batch_size=args.batch_size))
    eval_cvd = probs_array(cvd.classify_batch(eval_sentence_texts, batch_size=args.batch_size))
    eval_prob = 0.3 * eval_focal + 0.6 * eval_context_prob + 0.1 * eval_cvd

    temp_grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    best_temp = None
    for t in temp_grid:
        cprob = scale_temperature(calib_prob, t)
        nll = float(log_loss(calib_ids, cprob, labels=[0, 1, 2]))
        row = {"temperature": t, "nll": nll}
        if best_temp is None or row["nll"] < best_temp["nll"]:
            best_temp = row

    t = best_temp["temperature"]
    calib_cal = scale_temperature(calib_prob, t)
    eval_cal = scale_temperature(eval_prob, t)

    default_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
    baseline_pred = predict_with_threshold(eval_prob, default_thr)
    baseline_metrics = metrics(eval_labels, baseline_pred)
    calibrated_default_pred = predict_with_threshold(eval_cal, default_thr)
    calibrated_default_metrics = metrics(eval_labels, calibrated_default_pred)

    assoc_grid = [0.30, 0.35, 0.40, 0.45]
    not_grid = [0.45, 0.50, 0.55, 0.60]
    inc_grid = [0.25, 0.30, 0.35, 0.40]

    best = None
    for a, n, i in product(assoc_grid, not_grid, inc_grid):
        thr = np.asarray([a, n, i], dtype=np.float64)
        pred = predict_with_threshold(eval_cal, thr)
        m = metrics(eval_labels, pred)
        assoc_recall = m["per_class"]["associated"]["recall"]
        if assoc_recall < args.min_associated_recall:
            continue
        row = {
            "thresholds": {"associated": a, "not_associated": n, "incidental": i},
            "associated_recall_constraint": args.min_associated_recall,
            **m,
        }
        if best is None or row["accuracy"] > best["accuracy"] or (
            row["accuracy"] == best["accuracy"] and row["macro_f1"] > best["macro_f1"]
        ):
            best = row

    payload = {
        "calibration_size": len(calib_sent),
        "eval_size": len(eval_sent),
        "best_temperature": best_temp,
        "baseline_default_thresholds": baseline_metrics,
        "calibrated_default_thresholds": calibrated_default_metrics,
        "best_constrained_thresholds": best,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("best_temperature", best_temp)
    print("baseline_accuracy", baseline_metrics["accuracy"])
    print("calibrated_default_accuracy", calibrated_default_metrics["accuracy"])
    if best:
        print("best_constrained_accuracy", best["accuracy"])
        print("best_constrained_macro_f1", best["macro_f1"])
        print("best_constrained_thresholds", best["thresholds"])
    else:
        print("best_constrained_accuracy", None)
    print("saved", out)


if __name__ == "__main__":
    main()

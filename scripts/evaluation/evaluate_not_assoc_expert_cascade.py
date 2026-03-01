"""Evaluate fusion + not_associated expert cascade on HFpEF eval."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


IDX_ASSOC = LABEL_NAMES.index("associated")
IDX_NOT = LABEL_NAMES.index("not_associated")
IDX_INC = LABEL_NAMES.index("incidental")


def probs_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in rows],
        dtype=np.float64,
    )


def compute(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABEL_NAMES).tolist()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm,
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_NAMES
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate not_associated expert cascade")
    parser.add_argument("--expert", required=True, help="Path to joblib expert model")
    parser.add_argument("--output", required=True, help="Output JSON metrics path")
    parser.add_argument("--eval-sentence", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--eval-context", default="data/splits/hfpef_v3_eval_context.json")
    parser.add_argument("--focal-model", default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--context-model", default="models/hfpef_v3_improved/pubmedbert_context/final")
    parser.add_argument("--cvd-model", default="models/hfpef_v3_improved/pubmedbert_cvd_combined/final")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.eval_sentence) as f:
        eval_sentence = json.load(f)
    with open(args.eval_context) as f:
        eval_context = json.load(f)
    if len(eval_sentence) != len(eval_context):
        raise ValueError("Sentence/context eval length mismatch.")

    y_true = [r["label"] for r in eval_sentence]
    sentence_texts = [r["sentence"] for r in eval_sentence]
    context_texts = [r["sentence"] for r in eval_context]

    focal = PubMedBERTClassifier(Path(args.focal_model))
    context_model = PubMedBERTClassifier(Path(args.context_model))
    cvd = PubMedBERTClassifier(Path(args.cvd_model))
    expert = joblib.load(args.expert)

    p_focal = probs_array(focal.classify_batch(sentence_texts, batch_size=args.batch_size))
    p_context = probs_array(context_model.classify_batch(context_texts, batch_size=args.batch_size))
    p_cvd = probs_array(cvd.classify_batch(sentence_texts, batch_size=args.batch_size))

    base_prob = 0.3 * p_focal + 0.6 * p_context + 0.1 * p_cvd
    base_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
    base_pred = [LABEL_NAMES[idx] for idx in (base_prob / base_thr).argmax(axis=1)]
    base_metrics = compute(y_true, base_pred)

    assoc_prob = base_prob[:, IDX_ASSOC]
    not_prob = base_prob[:, IDX_NOT]
    inc_prob = base_prob[:, IDX_INC]
    abs_gap = np.abs(not_prob - inc_prob)

    assoc_max_grid = [0.35, 0.4, 0.45, 0.5]
    gap_grid = [0.03, 0.05, 0.08, 0.1, 0.12, 0.15]

    best_acc = None
    best_macro = None

    expert_preds_all = expert.predict(sentence_texts)
    for assoc_max, gap in product(assoc_max_grid, gap_grid):
        pred = list(base_pred)
        rerouted = 0
        for i in range(len(pred)):
            if pred[i] == "associated":
                continue
            if assoc_prob[i] >= assoc_max:
                continue
            if abs_gap[i] > gap:
                continue
            exp = expert_preds_all[i]
            if exp in {"not_associated", "incidental"}:
                pred[i] = exp
                rerouted += 1

        row = {
            "assoc_max": assoc_max,
            "not_inc_gap": gap,
            "rerouted": rerouted,
            **compute(y_true, pred),
        }

        if best_acc is None or (row["accuracy"] > best_acc["accuracy"]) or (
            row["accuracy"] == best_acc["accuracy"] and row["macro_f1"] > best_acc["macro_f1"]
        ):
            best_acc = row
        if best_macro is None or (row["macro_f1"] > best_macro["macro_f1"]) or (
            row["macro_f1"] == best_macro["macro_f1"] and row["accuracy"] > best_macro["accuracy"]
        ):
            best_macro = row

    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "baseline_fusion": base_metrics,
        "best_by_accuracy": best_acc,
        "best_by_macro_f1": best_macro,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("baseline_fusion", base_metrics)
    print("best_by_accuracy", best_acc)
    print("best_by_macro_f1", best_macro)
    print("saved", out)


if __name__ == "__main__":
    main()


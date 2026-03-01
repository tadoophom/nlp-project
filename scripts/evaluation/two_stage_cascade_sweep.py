"""Sweep a two-stage cascade for HFpEF relation classification.

Stage 1:
  Detect `associated` using a weighted score threshold.

Stage 2:
  For remaining samples, decide `not_associated` vs `incidental`
  using a weighted not-vs-incidental ratio threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


ROOT = Path(__file__).resolve().parent.parent.parent

IDX_ASSOC = LABEL_NAMES.index("associated")
IDX_NOT = LABEL_NAMES.index("not_associated")
IDX_INC = LABEL_NAMES.index("incidental")


@dataclass(frozen=True)
class Weights:
    focal: float
    context: float
    cvd_combined: float


def probs_array(results: list[dict]) -> np.ndarray:
    rows = [[row["probabilities"][label] for label in LABEL_NAMES] for row in results]
    return np.asarray(rows, dtype=np.float64)


def weighted_score(weights: Weights, p_focal: np.ndarray, p_context: np.ndarray, p_cvd: np.ndarray) -> np.ndarray:
    return (
        weights.focal * p_focal
        + weights.context * p_context
        + weights.cvd_combined * p_cvd
    )


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1)

    f1s = []
    supports = []
    for label in LABEL_NAMES:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        supports.append(support)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)

    macro_f1 = float(sum(f1s) / len(f1s))
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / max(sum(supports), 1))
    return {"accuracy": float(accuracy), "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def weights_grid() -> list[Weights]:
    out: set[tuple[float, float, float]] = set()
    focal_grid = np.arange(0.2, 0.81, 0.1)
    context_grid = np.arange(0.1, 0.81, 0.1)
    for w_focal in focal_grid:
        for w_context in context_grid:
            w_cvd = 1.0 - w_focal - w_context
            if w_cvd < -1e-9:
                continue
            if w_cvd < 0:
                w_cvd = 0.0
            out.add((round(float(w_focal), 3), round(float(w_context), 3), round(float(w_cvd), 3)))

    # Keep combinations compact and biased around known good region.
    ordered = sorted(out, key=lambda x: (abs(x[0] - 0.3) + abs(x[1] - 0.6), -x[2]))
    trimmed = ordered[:40]
    return [Weights(*row) for row in trimmed]


def run():
    with open(ROOT / "data/splits/hfpef_v3_eval.json") as f:
        eval_sentence = json.load(f)
    with open(ROOT / "data/splits/hfpef_v3_eval_context.json") as f:
        eval_context = json.load(f)

    if len(eval_sentence) != len(eval_context):
        raise ValueError("Eval sentence/context length mismatch.")

    y_true = [row["label"] for row in eval_sentence]
    sentence_texts = [row["sentence"] for row in eval_sentence]
    context_texts = [row["sentence"] for row in eval_context]

    models = {
        "focal": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_focal/final"),
        "context": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_context/final"),
        "cvd_combined": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_cvd_combined/final"),
    }

    p_focal = probs_array(models["focal"].classify_batch(sentence_texts, batch_size=64))
    p_context = probs_array(models["context"].classify_batch(context_texts, batch_size=64))
    p_cvd = probs_array(models["cvd_combined"].classify_batch(sentence_texts, batch_size=64))

    stage1_weights = weights_grid()
    stage2_weights = weights_grid()
    assoc_thresholds = [round(v, 2) for v in np.arange(0.2, 0.71, 0.05)]
    not_ratio_thresholds = [round(v, 2) for v in np.arange(0.3, 0.81, 0.05)]

    best_acc = None
    best_macro = None

    for w1 in stage1_weights:
        s1 = weighted_score(w1, p_focal, p_context, p_cvd)
        assoc_score = s1[:, IDX_ASSOC]

        for w2 in stage2_weights:
            s2 = weighted_score(w2, p_focal, p_context, p_cvd)
            not_score = s2[:, IDX_NOT]
            inc_score = s2[:, IDX_INC]
            denom = not_score + inc_score + 1e-12
            not_ratio = not_score / denom

            for t_assoc, t_not_ratio in product(assoc_thresholds, not_ratio_thresholds):
                pred = np.where(
                    assoc_score >= t_assoc,
                    "associated",
                    np.where(not_ratio >= t_not_ratio, "not_associated", "incidental"),
                )
                row = {
                    "stage1_weights": {
                        "focal": w1.focal,
                        "context": w1.context,
                        "cvd_combined": w1.cvd_combined,
                    },
                    "stage2_weights": {
                        "focal": w2.focal,
                        "context": w2.context,
                        "cvd_combined": w2.cvd_combined,
                    },
                    "thresholds": {
                        "associated_score": t_assoc,
                        "not_ratio": t_not_ratio,
                    },
                }
                row.update(compute_metrics(y_true, pred.tolist()))

                if best_acc is None or (row["accuracy"] > best_acc["accuracy"]) or (
                    row["accuracy"] == best_acc["accuracy"] and row["macro_f1"] > best_acc["macro_f1"]
                ):
                    best_acc = row

                if best_macro is None or (row["macro_f1"] > best_macro["macro_f1"]) or (
                    row["macro_f1"] == best_macro["macro_f1"] and row["accuracy"] > best_macro["accuracy"]
                ):
                    best_macro = row

    # Fixed baseline from previous best fusion sweep for direct reference.
    baseline = {
        "weights": {"focal": 0.3, "context": 0.6, "cvd_combined": 0.1},
        "thresholds": {"associated": 0.4, "not_associated": 0.55, "incidental": 0.3},
    }
    p_fused = (
        baseline["weights"]["focal"] * p_focal
        + baseline["weights"]["context"] * p_context
        + baseline["weights"]["cvd_combined"] * p_cvd
    )
    thr = np.asarray(
        [
            baseline["thresholds"]["associated"],
            baseline["thresholds"]["not_associated"],
            baseline["thresholds"]["incidental"],
        ],
        dtype=np.float64,
    )
    fused_pred = [LABEL_NAMES[idx] for idx in (p_fused / thr).argmax(axis=1)]
    baseline_metrics = compute_metrics(y_true, fused_pred)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = ROOT / "logs" / f"two_stage_cascade_sweep_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "search_space": {
            "stage1_weight_count": len(stage1_weights),
            "stage2_weight_count": len(stage2_weights),
            "associated_thresholds": assoc_thresholds,
            "not_ratio_thresholds": not_ratio_thresholds,
        },
        "baseline_fusion_reference": {
            **baseline,
            **baseline_metrics,
        },
        "best_by_accuracy": best_acc,
        "best_by_macro_f1": best_macro,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("baseline_fusion_reference", payload["baseline_fusion_reference"])
    print("best_by_accuracy", best_acc)
    print("best_by_macro_f1", best_macro)
    print("saved", out_path)


if __name__ == "__main__":
    run()

"""Sweep lexical cue calibration on top of best fusion probabilities."""
from __future__ import annotations

import json
import re
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


ROOT = Path(__file__).resolve().parent.parent.parent

NEG_PATTERNS = [
    r"\bno association\b",
    r"\bnot associated\b",
    r"\bno correlation\b",
    r"\bunrelated\b",
    r"\bnot linked\b",
    r"\bdid not\b",
]

POS_PATTERNS = [
    r"\bassociated with\b",
    r"\blinked to\b",
    r"\bcorrelated with\b",
    r"\bpredict(ed|ive)?\b",
]


def probs_array(results: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in results],
        dtype=np.float64,
    )


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    n = len(y_true)
    acc = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(n, 1)

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
    return {"accuracy": float(acc), "macro_f1": macro_f1, "weighted_f1": weighted_f1}


def has_neg(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in NEG_PATTERNS)


def has_pos(text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in POS_PATTERNS)


def run():
    with open(ROOT / "data/splits/hfpef_v3_eval.json") as f:
        eval_sentence = json.load(f)
    with open(ROOT / "data/splits/hfpef_v3_eval_context.json") as f:
        eval_context = json.load(f)

    if len(eval_sentence) != len(eval_context):
        raise ValueError("Eval sentence/context length mismatch.")

    y_true = [row["label"] for row in eval_sentence]
    sentences = [row["sentence"] for row in eval_sentence]
    contexts = [row["sentence"] for row in eval_context]

    models = {
        "focal": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_focal/final"),
        "context": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_context/final"),
        "cvd": PubMedBERTClassifier(ROOT / "models/hfpef_v3_improved/pubmedbert_cvd_combined/final"),
    }

    p_focal = probs_array(models["focal"].classify_batch(sentences, batch_size=64))
    p_context = probs_array(models["context"].classify_batch(contexts, batch_size=64))
    p_cvd = probs_array(models["cvd"].classify_batch(sentences, batch_size=64))

    base = 0.3 * p_focal + 0.6 * p_context + 0.1 * p_cvd
    base_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
    base_pred = [LABEL_NAMES[idx] for idx in (base / base_thr).argmax(axis=1)]
    base_metrics = compute_metrics(y_true, base_pred)

    neg_flags = np.asarray([has_neg(text) for text in sentences], dtype=bool)
    pos_flags = np.asarray([has_pos(text) for text in sentences], dtype=bool)

    neg_boost_not = [1.0, 1.05, 1.1, 1.15, 1.2]
    neg_decay_assoc = [1.0, 0.98, 0.95, 0.9]
    pos_boost_assoc = [1.0, 1.03, 1.06, 1.1]
    assoc_thr = [0.3, 0.35, 0.4, 0.45, 0.5]
    not_thr = [0.45, 0.5, 0.55, 0.6, 0.65]
    inc_thr = [0.25, 0.3, 0.35, 0.4]

    best_acc = None
    best_macro = None

    idx_assoc = LABEL_NAMES.index("associated")
    idx_not = LABEL_NAMES.index("not_associated")

    for n_boost, n_decay, p_boost in product(neg_boost_not, neg_decay_assoc, pos_boost_assoc):
        adjusted = base.copy()

        if neg_flags.any():
            adjusted[neg_flags, idx_not] *= n_boost
            adjusted[neg_flags, idx_assoc] *= n_decay
        if pos_flags.any():
            adjusted[pos_flags, idx_assoc] *= p_boost

        for ta, tn, ti in product(assoc_thr, not_thr, inc_thr):
            thr = np.asarray([ta, tn, ti], dtype=np.float64)
            pred = [LABEL_NAMES[idx] for idx in (adjusted / thr).argmax(axis=1)]
            row = {
                "cue_multipliers": {
                    "neg_boost_not_associated": n_boost,
                    "neg_decay_associated": n_decay,
                    "pos_boost_associated": p_boost,
                },
                "thresholds": {
                    "associated": ta,
                    "not_associated": tn,
                    "incidental": ti,
                },
            }
            row.update(compute_metrics(y_true, pred))

            if best_acc is None or (row["accuracy"] > best_acc["accuracy"]) or (
                row["accuracy"] == best_acc["accuracy"] and row["macro_f1"] > best_acc["macro_f1"]
            ):
                best_acc = row
            if best_macro is None or (row["macro_f1"] > best_macro["macro_f1"]) or (
                row["macro_f1"] == best_macro["macro_f1"] and row["accuracy"] > best_macro["accuracy"]
            ):
                best_macro = row

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "logs" / f"fusion_lexical_sweep_{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "baseline_fusion": {
            "weights": {"focal": 0.3, "context": 0.6, "cvd_combined": 0.1},
            "thresholds": {"associated": 0.4, "not_associated": 0.55, "incidental": 0.3},
            **base_metrics,
        },
        "search_space": {
            "neg_boost_not": neg_boost_not,
            "neg_decay_assoc": neg_decay_assoc,
            "pos_boost_assoc": pos_boost_assoc,
            "assoc_thr": assoc_thr,
            "not_thr": not_thr,
            "inc_thr": inc_thr,
            "neg_sentence_count": int(neg_flags.sum()),
            "pos_sentence_count": int(pos_flags.sum()),
        },
        "best_by_accuracy": best_acc,
        "best_by_macro_f1": best_macro,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)

    print("baseline_fusion", payload["baseline_fusion"])
    print("best_by_accuracy", best_acc)
    print("best_by_macro_f1", best_macro)
    print("saved", out)


if __name__ == "__main__":
    run()

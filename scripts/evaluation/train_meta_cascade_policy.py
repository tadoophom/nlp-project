"""Train meta-policy classifiers on ensemble and expert features."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
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


def entropy_rows(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def one_hot_pred(labels: list[str], choices: list[str]) -> np.ndarray:
    out = np.zeros((len(labels), len(choices)), dtype=np.float64)
    idx = {c: i for i, c in enumerate(choices)}
    for r, label in enumerate(labels):
        if label in idx:
            out[r, idx[label]] = 1.0
    return out


def extract_features(
    sentences: list[str],
    contexts: list[str],
    focal: PubMedBERTClassifier,
    context_model: PubMedBERTClassifier,
    cvd: PubMedBERTClassifier,
    not_expert,
    assoc_expert,
    batch_size: int,
):
    r_focal = focal.classify_batch(sentences, batch_size=batch_size)
    r_context = context_model.classify_batch(contexts, batch_size=batch_size)
    r_cvd = cvd.classify_batch(sentences, batch_size=batch_size)

    p_focal = probs_array(r_focal)
    p_context = probs_array(r_context)
    p_cvd = probs_array(r_cvd)
    p_fused = 0.3 * p_focal + 0.6 * p_context + 0.1 * p_cvd

    fused_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
    fused_pred_idx = (p_fused / fused_thr).argmax(axis=1)
    fused_pred = [LABEL_NAMES[int(i)] for i in fused_pred_idx]

    not_pred = list(not_expert.predict(sentences))
    not_probs_full = not_expert.predict_proba(sentences)
    not_classes = list(not_expert.classes_)
    not_p_not = not_probs_full[:, not_classes.index("not_associated")] if "not_associated" in not_classes else np.zeros(len(sentences))
    not_p_inc = not_probs_full[:, not_classes.index("incidental")] if "incidental" in not_classes else np.zeros(len(sentences))

    assoc_pred = list(assoc_expert.predict(sentences))
    assoc_probs_full = assoc_expert.predict_proba(sentences)
    assoc_classes = list(assoc_expert.classes_)
    assoc_p_assoc = assoc_probs_full[:, assoc_classes.index("associated")] if "associated" in assoc_classes else np.zeros(len(sentences))
    assoc_p_inc = assoc_probs_full[:, assoc_classes.index("incidental")] if "incidental" in assoc_classes else np.zeros(len(sentences))

    sorted_fused = np.sort(p_fused, axis=1)
    fused_margin = sorted_fused[:, -1] - sorted_fused[:, -2]
    fused_max = p_fused.max(axis=1)
    fused_entropy = entropy_rows(p_fused)

    feats = np.column_stack(
        [
            p_fused,
            p_focal,
            p_context,
            p_cvd,
            fused_max,
            fused_margin,
            fused_entropy,
            one_hot_pred(fused_pred, LABEL_NAMES),
            one_hot_pred(not_pred, ["not_associated", "incidental"]),
            not_p_not,
            not_p_inc,
            one_hot_pred(assoc_pred, ["associated", "incidental"]),
            assoc_p_assoc,
            assoc_p_inc,
        ]
    )

    return feats, fused_pred


def train_policy(x_train, y_train, c_values: list[float], scoring: str, class_weight):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best = None

    for c in c_values:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        max_iter=4000,
                        solver="lbfgs",
                        class_weight=class_weight,
                    ),
                ),
            ]
        )
        scores = cross_val_score(model, x_train, y_train, cv=cv, scoring=scoring)
        row = {"C": c, "cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
        if best is None or row["cv_mean"] > best["cv_mean"]:
            best = row

    final = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=best["C"],
                    max_iter=4000,
                    solver="lbfgs",
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    final.fit(x_train, y_train)
    return final, best


def main():
    parser = argparse.ArgumentParser(description="Train meta cascade policy")
    parser.add_argument("--train-sentence", default="data/splits/hfpef_v3_train.json")
    parser.add_argument("--train-context", default="data/splits/hfpef_v3_train_context.json")
    parser.add_argument("--eval-sentence", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--eval-context", default="data/splits/hfpef_v3_eval_context.json")
    parser.add_argument("--focal-model", default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--context-model", default="models/hfpef_v3_improved/pubmedbert_context/final")
    parser.add_argument("--cvd-model", default="models/hfpef_v3_improved/pubmedbert_cvd_combined/final")
    parser.add_argument("--not-expert", required=True)
    parser.add_argument("--assoc-expert", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.train_sentence) as f:
        train_sentence = json.load(f)
    with open(args.train_context) as f:
        train_context = json.load(f)
    with open(args.eval_sentence) as f:
        eval_sentence = json.load(f)
    with open(args.eval_context) as f:
        eval_context = json.load(f)

    if len(train_sentence) != len(train_context):
        raise ValueError("Train sentence/context length mismatch.")
    if len(eval_sentence) != len(eval_context):
        raise ValueError("Eval sentence/context length mismatch.")

    train_sent = [r["sentence"] for r in train_sentence]
    train_ctx = [r["sentence"] for r in train_context]
    y_train = [r["label"] for r in train_sentence]

    eval_sent = [r["sentence"] for r in eval_sentence]
    eval_ctx = [r["sentence"] for r in eval_context]
    y_eval = [r["label"] for r in eval_sentence]

    focal = PubMedBERTClassifier(Path(args.focal_model))
    context_model = PubMedBERTClassifier(Path(args.context_model))
    cvd = PubMedBERTClassifier(Path(args.cvd_model))
    not_expert = joblib.load(args.not_expert)
    assoc_expert = joblib.load(args.assoc_expert)

    x_train, _ = extract_features(
        train_sent, train_ctx, focal, context_model, cvd, not_expert, assoc_expert, args.batch_size
    )
    x_eval, fused_eval_pred = extract_features(
        eval_sent, eval_ctx, focal, context_model, cvd, not_expert, assoc_expert, args.batch_size
    )

    c_values = [0.25, 0.5, 1.0, 2.0, 4.0]

    acc_policy, acc_cv = train_policy(
        x_train=x_train,
        y_train=y_train,
        c_values=c_values,
        scoring="accuracy",
        class_weight=None,
    )
    macro_policy, macro_cv = train_policy(
        x_train=x_train,
        y_train=y_train,
        c_values=c_values,
        scoring="f1_macro",
        class_weight="balanced",
    )

    acc_pred = list(acc_policy.predict(x_eval))
    macro_pred = list(macro_policy.predict(x_eval))

    baseline_metrics = compute_metrics(y_eval, fused_eval_pred)
    acc_metrics = compute_metrics(y_eval, acc_pred)
    macro_metrics = compute_metrics(y_eval, macro_pred)

    out_model_dir = Path(args.output_model_dir)
    out_model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(acc_policy, out_model_dir / "meta_policy_accuracy.joblib")
    joblib.dump(macro_policy, out_model_dir / "meta_policy_macrof1.joblib")

    payload = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "train_size": len(y_train),
        "eval_size": len(y_eval),
        "feature_dim": int(x_train.shape[1]),
        "baseline_fusion": baseline_metrics,
        "policy_accuracy_optimized": {
            "cv_selection": acc_cv,
            **acc_metrics,
        },
        "policy_macrof1_optimized": {
            "cv_selection": macro_cv,
            **macro_metrics,
        },
        "label_to_id": LABEL2ID,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("baseline_fusion", baseline_metrics)
    print("policy_accuracy_optimized", payload["policy_accuracy_optimized"])
    print("policy_macrof1_optimized", payload["policy_macrof1_optimized"])
    print("saved", out_path)
    print("saved_models", out_model_dir)


if __name__ == "__main__":
    main()

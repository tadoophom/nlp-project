"""Build a targeted correction set focused on associated/incidental confusion."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np

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


def entropy(prob: np.ndarray) -> np.ndarray:
    p = np.clip(prob, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description="Build targeted correction set")
    parser.add_argument("--sentence-data", required=True, help="JSON split with sentence and optional label")
    parser.add_argument("--context-data", required=True, help="Aligned context JSON split")
    parser.add_argument("--focal-model", default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--context-model", default="models/hfpef_v3_improved/pubmedbert_context/final")
    parser.add_argument("--cvd-model", default="models/hfpef_v3_improved/pubmedbert_cvd_combined/final")
    parser.add_argument("--assoc-expert", required=True)
    parser.add_argument("--not-expert", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(args.sentence_data) as f:
        sent_rows = json.load(f)
    with open(args.context_data) as f:
        ctx_rows = json.load(f)
    if len(sent_rows) != len(ctx_rows):
        raise ValueError("sentence/context length mismatch")

    sentences = [r["sentence"] for r in sent_rows]
    contexts = [r["sentence"] for r in ctx_rows]
    gold = [r.get("label", "") for r in sent_rows]

    focal = PubMedBERTClassifier(Path(args.focal_model))
    context_model = PubMedBERTClassifier(Path(args.context_model))
    cvd = PubMedBERTClassifier(Path(args.cvd_model))
    assoc_expert = joblib.load(args.assoc_expert)
    not_expert = joblib.load(args.not_expert)

    p_focal = probs_array(focal.classify_batch(sentences, batch_size=args.batch_size))
    p_context = probs_array(context_model.classify_batch(contexts, batch_size=args.batch_size))
    p_cvd = probs_array(cvd.classify_batch(sentences, batch_size=args.batch_size))
    p_fused = 0.3 * p_focal + 0.6 * p_context + 0.1 * p_cvd
    fused_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
    fused_idx = (p_fused / fused_thr).argmax(axis=1)
    fused_pred = [LABEL_NAMES[int(i)] for i in fused_idx]

    assoc_pred = list(assoc_expert.predict(sentences))
    assoc_probs = assoc_expert.predict_proba(sentences)
    assoc_classes = list(assoc_expert.classes_)
    assoc_p_assoc = assoc_probs[:, assoc_classes.index("associated")] if "associated" in assoc_classes else np.zeros(len(sentences))

    not_pred = list(not_expert.predict(sentences))
    not_probs = not_expert.predict_proba(sentences)
    not_classes = list(not_expert.classes_)
    not_p_not = not_probs[:, not_classes.index("not_associated")] if "not_associated" in not_classes else np.zeros(len(sentences))

    ent = entropy(p_fused)
    assoc_inc_gap = np.abs(p_fused[:, IDX_ASSOC] - p_fused[:, IDX_INC])
    not_inc_gap = np.abs(p_fused[:, IDX_NOT] - p_fused[:, IDX_INC])

    candidates = []
    for i, sentence in enumerate(sentences):
        pred = fused_pred[i]
        label = gold[i]
        base_assoc = float(p_fused[i, IDX_ASSOC])
        base_not = float(p_fused[i, IDX_NOT])
        base_inc = float(p_fused[i, IDX_INC])

        focus = pred in {"associated", "incidental"}
        if not focus:
            continue

        miscls = label in LABEL_NAMES and pred != label
        expert_flip_assoc = pred == "incidental" and assoc_pred[i] == "associated"
        expert_flip_not = pred == "associated" and not_pred[i] == "not_associated"
        ambiguous_assoc_inc = assoc_inc_gap[i] <= 0.08

        if not (miscls or expert_flip_assoc or expert_flip_not or ambiguous_assoc_inc):
            continue

        score = (
            (2.0 if miscls else 0.0)
            + (1.2 if expert_flip_assoc else 0.0)
            + (1.2 if expert_flip_not else 0.0)
            + (0.8 if ambiguous_assoc_inc else 0.0)
            + float(ent[i])
            + (0.5 - float(assoc_inc_gap[i]))
        )

        suggestion = pred
        if expert_flip_assoc and assoc_p_assoc[i] >= 0.62:
            suggestion = "associated"
        elif expert_flip_not and not_p_not[i] >= 0.70:
            suggestion = "not_associated"

        candidates.append(
            {
                "idx": i,
                "sentence": sentence,
                "gold_label": label,
                "fused_pred": pred,
                "suggested_label": suggestion,
                "fused_assoc_prob": base_assoc,
                "fused_not_prob": base_not,
                "fused_inc_prob": base_inc,
                "assoc_expert_pred": assoc_pred[i],
                "assoc_expert_prob_associated": float(assoc_p_assoc[i]),
                "not_expert_pred": not_pred[i],
                "not_expert_prob_not_associated": float(not_p_not[i]),
                "fused_entropy": float(ent[i]),
                "assoc_inc_gap": float(assoc_inc_gap[i]),
                "not_inc_gap": float(not_inc_gap[i]),
                "is_misclassified": bool(miscls),
                "score": float(score),
                "accept": "",
                "reviewer_label": "",
                "review_notes": "",
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    keep = candidates[: args.top_k]

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(keep, f, indent=2)

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(keep[0].keys()) if keep else [
            "idx",
            "sentence",
            "gold_label",
            "fused_pred",
            "suggested_label",
            "accept",
            "reviewer_label",
            "review_notes",
        ])
        writer.writeheader()
        for row in keep:
            writer.writerow(row)

    print("candidate_count", len(candidates))
    print("selected_count", len(keep))
    print("saved_json", out_json)
    print("saved_csv", out_csv)


if __name__ == "__main__":
    main()

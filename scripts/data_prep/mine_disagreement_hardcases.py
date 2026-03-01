"""Mine high-priority hard cases from model disagreement and uncertainty."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


def load_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list of rows.")
        return data

    if path.suffix.lower() == ".csv":
        with open(path) as f:
            return list(csv.DictReader(f))

    raise ValueError(f"Unsupported input format: {path}")


def get_text(row: dict, key: str) -> str:
    value = row.get(key, "")
    if value is None:
        return ""
    return str(value).strip()


def probs_array(results: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in results],
        dtype=np.float64,
    )


def entropy(prob: np.ndarray) -> float:
    p = np.clip(prob, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def main():
    parser = argparse.ArgumentParser(description="Mine disagreement hard cases")
    parser.add_argument("--input", required=True, help="Input JSON/CSV data with sentence field")
    parser.add_argument("--sentence-key", default="sentence", help="Field name for sentence text")
    parser.add_argument("--context-input", default="", help="Optional aligned context JSON/CSV input")
    parser.add_argument("--context-key", default="sentence", help="Field name for context text")
    parser.add_argument("--label-key", default="label", help="Optional gold-label field")
    parser.add_argument("--focal-model", required=True, help="Path to focal sentence model")
    parser.add_argument("--context-model", required=True, help="Path to context model")
    parser.add_argument("--cvd-model", required=True, help="Path to cvd-combined model")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--top-k", type=int, default=500, help="Top hard cases to keep")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=30)
    args = parser.parse_args()

    input_rows = load_rows(Path(args.input))
    context_rows = load_rows(Path(args.context_input)) if args.context_input else input_rows
    if len(input_rows) != len(context_rows):
        raise ValueError("Input rows and context rows must be aligned and equal length.")

    records = []
    for idx, (row, ctx_row) in enumerate(zip(input_rows, context_rows)):
        sentence = get_text(row, args.sentence_key)
        if len(sentence) < args.min_length:
            continue
        context = get_text(ctx_row, args.context_key)
        if not context:
            context = sentence

        records.append({
            "idx": idx,
            "sentence": sentence,
            "context": context,
            "label": row.get(args.label_key, ""),
        })

    print(f"Loaded rows: {len(input_rows)}")
    print(f"Filtered rows: {len(records)}")

    sentences = [row["sentence"] for row in records]
    contexts = [row["context"] for row in records]

    focal = PubMedBERTClassifier(model_path=Path(args.focal_model))
    context_model = PubMedBERTClassifier(model_path=Path(args.context_model))
    cvd = PubMedBERTClassifier(model_path=Path(args.cvd_model))

    focal_res = focal.classify_batch(sentences, batch_size=args.batch_size)
    context_res = context_model.classify_batch(contexts, batch_size=args.batch_size)
    cvd_res = cvd.classify_batch(sentences, batch_size=args.batch_size)

    p_focal = probs_array(focal_res)
    p_context = probs_array(context_res)
    p_cvd = probs_array(cvd_res)

    out = []
    for i, row in enumerate(records):
        preds = {
            "focal": focal_res[i]["label"],
            "context": context_res[i]["label"],
            "cvd_combined": cvd_res[i]["label"],
        }
        confs = {
            "focal": float(focal_res[i]["confidence"]),
            "context": float(context_res[i]["confidence"]),
            "cvd_combined": float(cvd_res[i]["confidence"]),
        }
        labels = list(preds.values())
        unique_labels = len(set(labels))
        disagreement = unique_labels - 1

        mean_prob = (p_focal[i] + p_context[i] + p_cvd[i]) / 3.0
        mean_pred = LABEL_NAMES[int(mean_prob.argmax())]
        ent = entropy(mean_prob)

        top2 = np.sort(mean_prob)[-2:]
        margin = float(top2[1] - top2[0])
        mean_conf = float(mean_prob.max())

        # Score prioritizes disagreement + low confidence + high entropy + low margin.
        hard_score = (
            1.2 * disagreement
            + (1.0 - mean_conf)
            + 0.7 * ent
            + (0.5 - margin)
        )

        out.append({
            "idx": row["idx"],
            "label": row["label"],
            "sentence": row["sentence"],
            "context": row["context"],
            "predictions": preds,
            "confidences": confs,
            "ensemble_mean_pred": mean_pred,
            "ensemble_mean_confidence": mean_conf,
            "ensemble_entropy": ent,
            "ensemble_margin": margin,
            "disagreement_count": disagreement,
            "hard_score": float(hard_score),
        })

    out.sort(key=lambda x: x["hard_score"], reverse=True)
    trimmed = out[: args.top_k]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trimmed, f, indent=2)

    print(f"Saved {len(trimmed)} hard cases to {out_path}")
    if trimmed:
        print("Top hard case:", trimmed[0]["idx"], trimmed[0]["hard_score"], trimmed[0]["predictions"])


if __name__ == "__main__":
    main()

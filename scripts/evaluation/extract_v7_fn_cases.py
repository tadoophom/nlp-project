"""Extract associated-as-incidental FN cases from v7 baseline model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np
from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES


def apply_temperature_and_thresholds(logits_row, temperature, thresholds):
    scaled = logits_row / temperature
    probs = torch.softmax(torch.tensor(scaled), dim=-1).numpy()
    scores = {LABEL_NAMES[i]: float(probs[i]) / thresholds[LABEL_NAMES[i]] for i in range(len(LABEL_NAMES))}
    return max(scores, key=scores.get), {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}


def main():
    with open("data/splits/hfpef_v7_eval_relabel3_noleak_large.json") as f:
        eval_data = json.load(f)

    print(f"eval_size: {len(eval_data)}")

    model_path = Path("models/hfpef_v7/pubmedbert_focal_large_base_20260224/final")
    clf = PubMedBERTClassifier(model_path=model_path)
    clf._load_model()

    sentences = [r["sentence"] for r in eval_data]

    # Get raw logits
    all_logits = []
    for i in range(0, len(sentences), 64):
        batch = sentences[i:i + 64]
        inputs = clf._tokenizer(batch, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(clf.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = clf._model(**inputs)
            all_logits.append(outputs.logits.cpu().numpy())
    all_logits = np.concatenate(all_logits, axis=0)

    # v7 best: temperature=0.7, thresholds: assoc=0.45, not=0.45, incid=0.3
    temperature = 0.7
    thresholds = {"associated": 0.45, "not_associated": 0.45, "incidental": 0.3}

    fn_cases = []
    all_errors = []
    for i, row in enumerate(eval_data):
        calibrated, probs = apply_temperature_and_thresholds(all_logits[i], temperature, thresholds)
        if row["label"] != calibrated:
            all_errors.append({
                "index": i,
                "sentence": row["sentence"],
                "true_label": row["label"],
                "calibrated_pred": calibrated,
                "prob_associated": round(probs["associated"], 4),
                "prob_not_associated": round(probs["not_associated"], 4),
                "prob_incidental": round(probs["incidental"], 4),
            })
        if row["label"] == "associated" and calibrated == "incidental":
            fn_cases.append({
                "index": i,
                "sentence": row["sentence"],
                "true_label": row["label"],
                "calibrated_pred": calibrated,
                "prob_associated": round(probs["associated"], 4),
                "prob_not_associated": round(probs["not_associated"], 4),
                "prob_incidental": round(probs["incidental"], 4),
            })

    print(f"total_errors: {len(all_errors)}")
    print(f"fn_associated_as_incidental: {len(fn_cases)}")

    out_dir = Path("logs")
    with open(out_dir / "v7_associated_fn_as_incidental.json", "w") as f:
        json.dump(fn_cases, f, indent=2)
    with open(out_dir / "v7_all_errors.json", "w") as f:
        json.dump(all_errors, f, indent=2)

    for c in fn_cases:
        print(f"IDX={c['index']} P(a)={c['prob_associated']} P(i)={c['prob_incidental']}")
        print(f"  {c['sentence'][:250]}")
        print()


if __name__ == "__main__":
    main()

"""Mine high-precision associated hard positives from corpus candidates."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


IDX_ASSOC = LABEL_NAMES.index("associated")
IDX_NOT = LABEL_NAMES.index("not_associated")
IDX_INC = LABEL_NAMES.index("incidental")

POSITIVE_PATTERNS = [
    r"\bassociated with\b",
    r"\bcorrelated with\b",
    r"\blinked to\b",
    r"\bpredict(ed|ive|or)\b",
    r"\bincreased risk\b",
    r"\belevated\b",
    r"\bmarker of\b",
]

NEGATIVE_PATTERNS = [
    r"\bno association\b",
    r"\bnot associated\b",
    r"\bno correlation\b",
    r"\bnot linked\b",
    r"\bunrelated\b",
    r"\bno significant\b",
]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= 30]


def iter_corpus_sentences(path: Path):
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = (row.get("evidence_sentence") or "").strip()
            if len(ev) >= 30:
                yield ev
            abstract = (row.get("abstract") or "").strip()
            if abstract and abstract.lower() != "no abstract available":
                for sent in split_sentences(abstract):
                    yield sent


def probs_array(rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [[row["probabilities"][label] for label in LABEL_NAMES] for row in rows],
        dtype=np.float64,
    )


def has_positive_cue(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in POSITIVE_PATTERNS)


def has_negative_cue(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in NEGATIVE_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description="Build associated hard-positive expansion set")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--existing-train", required=True)
    parser.add_argument("--assoc-expert", required=True)
    parser.add_argument("--focal-model", required=True)
    parser.add_argument("--context-model", required=True)
    parser.add_argument("--cvd-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--max-sentences", type=int, default=35000)
    parser.add_argument("--max-add", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--min-assoc-expert-prob", type=float, default=0.76)
    parser.add_argument("--min-base-assoc-prob", type=float, default=0.18)
    parser.add_argument("--max-base-assoc-prob", type=float, default=0.54)
    parser.add_argument("--max-base-not-prob", type=float, default=0.30)
    parser.add_argument("--audit-size", type=int, default=60)
    parser.add_argument("--allow-no-cue", action="store_true")
    args = parser.parse_args()

    with open(args.existing_train) as f:
        existing = json.load(f)
    blocked = {" ".join(r["sentence"].split()).strip() for r in existing}

    candidates = []
    seen = set()
    for sent in iter_corpus_sentences(Path(args.corpus)):
        sent = re.sub(r"\s+", " ", sent).strip()
        if len(sent) < args.min_length:
            continue
        if sent in blocked or sent in seen:
            continue
        seen.add(sent)
        if has_negative_cue(sent):
            continue
        candidates.append(sent)
        if len(candidates) >= args.max_sentences:
            break

    print("candidate_sentences", len(candidates))

    focal = PubMedBERTClassifier(Path(args.focal_model))
    context_model = PubMedBERTClassifier(Path(args.context_model))
    cvd = PubMedBERTClassifier(Path(args.cvd_model))
    assoc_expert = joblib.load(args.assoc_expert)

    accepted = []
    near_hits = []

    for i in range(0, len(candidates), args.batch_size):
        batch = candidates[i:i + args.batch_size]
        r_focal = focal.classify_batch(batch, batch_size=args.batch_size)
        r_context = context_model.classify_batch(batch, batch_size=args.batch_size)
        r_cvd = cvd.classify_batch(batch, batch_size=args.batch_size)
        p_focal = probs_array(r_focal)
        p_context = probs_array(r_context)
        p_cvd = probs_array(r_cvd)
        base_prob = 0.3 * p_focal + 0.6 * p_context + 0.1 * p_cvd
        base_thr = np.asarray([0.4, 0.55, 0.3], dtype=np.float64)
        base_pred_idx = (base_prob / base_thr).argmax(axis=1)

        expert_pred = assoc_expert.predict(batch)
        expert_probs = assoc_expert.predict_proba(batch)
        expert_classes = list(assoc_expert.classes_)
        assoc_idx = expert_classes.index("associated")

        for j, sent in enumerate(batch):
            assoc_prob = float(base_prob[j, IDX_ASSOC])
            not_prob = float(base_prob[j, IDX_NOT])
            inc_prob = float(base_prob[j, IDX_INC])
            pred = LABEL_NAMES[int(base_pred_idx[j])]
            expert_assoc = float(expert_probs[j, assoc_idx])
            exp_label = expert_pred[j]
            cue = has_positive_cue(sent)

            detail = {
                "sentence": sent,
                "label": "associated",
                "source": "associated_hard_positive",
                "base_pred": pred,
                "base_assoc_prob": assoc_prob,
                "base_not_prob": not_prob,
                "base_inc_prob": inc_prob,
                "assoc_expert_label": str(exp_label),
                "assoc_expert_prob_associated": expert_assoc,
                "has_positive_cue": cue,
            }

            cue_ok = cue or args.allow_no_cue
            accept = (
                pred == "incidental"
                and exp_label == "associated"
                and cue_ok
                and expert_assoc >= args.min_assoc_expert_prob
                and args.min_base_assoc_prob <= assoc_prob <= args.max_base_assoc_prob
                and not_prob <= args.max_base_not_prob
            )

            if accept:
                accepted.append(detail)
            elif pred == "incidental" and exp_label == "associated" and cue:
                near_hits.append(detail)

    accepted = sorted(
        accepted,
        key=lambda x: (x["assoc_expert_prob_associated"], x["base_assoc_prob"]),
        reverse=True,
    )[: args.max_add]

    audit_pool = sorted(
        accepted + near_hits,
        key=lambda x: (x["assoc_expert_prob_associated"], x["base_assoc_prob"]),
        reverse=True,
    )[: max(args.audit_size * 3, args.audit_size)]
    audit = audit_pool[: args.audit_size]

    out_rows = [{"sentence": r["sentence"], "label": "associated", "source": r["source"]} for r in accepted]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_rows, f, indent=2)

    report = {
        "candidate_sentences": len(candidates),
        "accepted_count": len(accepted),
        "accepted_distribution": dict(Counter(r["label"] for r in out_rows)),
        "min_assoc_expert_prob": args.min_assoc_expert_prob,
        "base_assoc_range": [args.min_base_assoc_prob, args.max_base_assoc_prob],
        "max_base_not_prob": args.max_base_not_prob,
        "positive_cue_patterns": len(POSITIVE_PATTERNS),
        "audit_count": len(audit),
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    audit_path = Path(args.output_audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print("report", report)
    print("saved_output", out_path)
    print("saved_report", report_path)
    print("saved_audit", audit_path)


if __name__ == "__main__":
    main()

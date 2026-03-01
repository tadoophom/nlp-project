"""Build pseudo-labels with strict and soft acceptance tiers plus audit samples."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import LABEL_NAMES, PubMedBERTClassifier


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


def main():
    parser = argparse.ArgumentParser(description="Precision-controlled pseudo-label mining")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--focal-model", required=True)
    parser.add_argument("--context-model", required=True)
    parser.add_argument("--cvd-model", required=True)
    parser.add_argument("--existing-train", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--output-audit", required=True)
    parser.add_argument("--strict-min-confidence", type=float, default=0.95)
    parser.add_argument("--soft-min-confidence", type=float, default=0.80)
    parser.add_argument("--soft-min-member-confidence", type=float, default=0.65)
    parser.add_argument("--soft-min-margin", type=float, default=0.10)
    parser.add_argument("--label-min-prob-associated", type=float, default=0.78)
    parser.add_argument("--label-min-prob-not-associated", type=float, default=0.78)
    parser.add_argument("--label-min-prob-incidental", type=float, default=0.84)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sentences", type=int, default=30000)
    parser.add_argument("--max-per-label", type=int, default=300)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--audit-per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    label_min_prob = {
        "associated": args.label_min_prob_associated,
        "not_associated": args.label_min_prob_not_associated,
        "incidental": args.label_min_prob_incidental,
    }

    blocked = set()
    if args.existing_train:
        with open(args.existing_train) as f:
            rows = json.load(f)
        blocked = {" ".join(r["sentence"].split()).strip() for r in rows}

    candidates = []
    seen = set()
    for sent in iter_corpus_sentences(Path(args.corpus)):
        sent = re.sub(r"\s+", " ", sent).strip()
        if len(sent) < args.min_length:
            continue
        if sent in blocked or sent in seen:
            continue
        seen.add(sent)
        candidates.append(sent)
        if len(candidates) >= args.max_sentences:
            break

    print("candidate_sentences", len(candidates))

    focal = PubMedBERTClassifier(model_path=Path(args.focal_model))
    context_model = PubMedBERTClassifier(model_path=Path(args.context_model))
    cvd = PubMedBERTClassifier(model_path=Path(args.cvd_model))

    accepted = []
    accepted_by_label = Counter()
    accepted_by_tier = Counter()
    audit_groups: dict[str, list[dict]] = defaultdict(list)

    for i in range(0, len(candidates), args.batch_size):
        batch = candidates[i:i + args.batch_size]
        r_focal = focal.classify_batch(batch, batch_size=args.batch_size)
        r_context = context_model.classify_batch(batch, batch_size=args.batch_size)
        r_cvd = cvd.classify_batch(batch, batch_size=args.batch_size)

        p_focal = probs_array(r_focal)
        p_context = probs_array(r_context)
        p_cvd = probs_array(r_cvd)
        p_mean = (p_focal + p_context + p_cvd) / 3.0

        for j, sent in enumerate(batch):
            labels = [r_focal[j]["label"], r_context[j]["label"], r_cvd[j]["label"]]
            votes = Counter(labels)
            top_label, top_votes = votes.most_common(1)[0]
            mean_probs = {label: float(p_mean[j, idx]) for idx, label in enumerate(LABEL_NAMES)}
            sorted_probs = sorted(mean_probs.values(), reverse=True)
            margin = sorted_probs[0] - sorted_probs[1]
            min_member_conf = min(
                r_focal[j]["confidence"],
                r_context[j]["confidence"],
                r_cvd[j]["confidence"],
            )

            tier = None
            if top_votes == 3:
                if min_member_conf >= args.strict_min_confidence and mean_probs[top_label] >= label_min_prob[top_label]:
                    tier = "strict_consensus"
            elif top_votes == 2:
                if (
                    mean_probs[top_label] >= args.soft_min_confidence
                    and min_member_conf >= args.soft_min_member_confidence
                    and margin >= args.soft_min_margin
                    and mean_probs[top_label] >= label_min_prob[top_label]
                ):
                    tier = "soft_majority"

            detail = {
                "sentence": sent,
                "label": top_label,
                "votes": dict(votes),
                "mean_probabilities": mean_probs,
                "mean_margin": float(margin),
                "min_member_confidence": float(min_member_conf),
                "focal_confidence": float(r_focal[j]["confidence"]),
                "context_confidence": float(r_context[j]["confidence"]),
                "cvd_confidence": float(r_cvd[j]["confidence"]),
            }

            if tier and accepted_by_label[top_label] < args.max_per_label:
                accepted_by_label[top_label] += 1
                accepted_by_tier[tier] += 1
                accepted.append({
                    **detail,
                    "source": "precision_controlled_pseudo",
                    "acceptance_tier": tier,
                })
                if tier == "soft_majority":
                    audit_groups[f"accepted_soft_{top_label}"].append(detail)
            else:
                if top_votes >= 2 and mean_probs[top_label] >= (args.soft_min_confidence - 0.05):
                    audit_groups[f"rejected_near_cutoff_{top_label}"].append(detail)

    audit = []
    for group, rows in audit_groups.items():
        if not rows:
            continue
        take = min(len(rows), args.audit_per_group)
        for row in rng.sample(rows, take):
            audit.append({"group": group, **row})

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(accepted, f, indent=2)

    report = {
        "candidate_sentences": len(candidates),
        "pseudo_count": len(accepted),
        "label_distribution": dict(Counter(r["label"] for r in accepted)),
        "tier_distribution": dict(accepted_by_tier),
        "strict_min_confidence": args.strict_min_confidence,
        "soft_min_confidence": args.soft_min_confidence,
        "soft_min_member_confidence": args.soft_min_member_confidence,
        "soft_min_margin": args.soft_min_margin,
        "max_per_label": args.max_per_label,
        "label_min_prob": label_min_prob,
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

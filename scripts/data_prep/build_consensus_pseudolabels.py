"""Build high-confidence consensus pseudo labels from corpus sentences."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier


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


def main():
    parser = argparse.ArgumentParser(description="Consensus pseudo-label mining")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--focal-model", required=True)
    parser.add_argument("--context-model", required=True)
    parser.add_argument("--cvd-model", required=True)
    parser.add_argument("--existing-train", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--min-confidence", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-sentences", type=int, default=20000)
    parser.add_argument("--max-per-label", type=int, default=350)
    parser.add_argument("--min-length", type=int, default=30)
    args = parser.parse_args()

    blocked = set()
    if args.existing_train:
        with open(args.existing_train) as f:
            rows = json.load(f)
        blocked = {r["sentence"].strip() for r in rows}

    unique = []
    seen = set()
    for sent in iter_corpus_sentences(Path(args.corpus)):
        sent = re.sub(r"\s+", " ", sent).strip()
        if len(sent) < args.min_length:
            continue
        if sent in seen or sent in blocked:
            continue
        seen.add(sent)
        unique.append(sent)
        if len(unique) >= args.max_sentences:
            break

    print("candidate_sentences", len(unique))

    focal = PubMedBERTClassifier(model_path=Path(args.focal_model))
    context_model = PubMedBERTClassifier(model_path=Path(args.context_model))
    cvd = PubMedBERTClassifier(model_path=Path(args.cvd_model))

    out = []
    kept_by_label = Counter()
    for i in range(0, len(unique), args.batch_size):
        batch = unique[i:i + args.batch_size]
        r_focal = focal.classify_batch(batch, batch_size=args.batch_size)
        r_context = context_model.classify_batch(batch, batch_size=args.batch_size)
        r_cvd = cvd.classify_batch(batch, batch_size=args.batch_size)

        for sent, a, b, c in zip(batch, r_focal, r_context, r_cvd):
            labels = [a["label"], b["label"], c["label"]]
            if not (labels[0] == labels[1] == labels[2]):
                continue

            min_conf = min(a["confidence"], b["confidence"], c["confidence"])
            if min_conf < args.min_confidence:
                continue

            label = labels[0]
            if kept_by_label[label] >= args.max_per_label:
                continue

            kept_by_label[label] += 1
            out.append({
                "sentence": sent,
                "label": label,
                "source": "consensus_pseudo",
                "min_confidence": float(min_conf),
                "focal_confidence": float(a["confidence"]),
                "context_confidence": float(b["confidence"]),
                "cvd_confidence": float(c["confidence"]),
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    conf_buckets = defaultdict(int)
    for row in out:
        c = row["min_confidence"]
        if c >= 0.99:
            conf_buckets["0.99-1.00"] += 1
        elif c >= 0.97:
            conf_buckets["0.97-0.99"] += 1
        else:
            conf_buckets["0.95-0.97"] += 1

    report = {
        "candidate_sentences": len(unique),
        "pseudo_count": len(out),
        "label_distribution": dict(Counter(r["label"] for r in out)),
        "confidence_buckets": dict(conf_buckets),
        "min_confidence": args.min_confidence,
        "max_per_label": args.max_per_label,
    }

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("report", report)
    print("saved_output", out_path)
    print("saved_report", report_path)


if __name__ == "__main__":
    main()


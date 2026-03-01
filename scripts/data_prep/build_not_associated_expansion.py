"""Build a high-precision not_associated expansion set."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


NEG_PATTERNS = [
    r"\bno association\b",
    r"\bnot associated\b",
    r"\bno correlation\b",
    r"\bunrelated\b",
    r"\bnot linked\b",
    r"\bdid not\b",
    r"\bno significant\b",
    r"\bindependent of\b",
    r"\bno effect\b",
]

POS_PATTERNS = [
    r"\bassociated with\b",
    r"\blinked to\b",
    r"\bcorrelated with\b",
    r"\bpredict(ed|ive)?\b",
]

DISEASE_TERMS = [
    "hfpef",
    "heart failure with preserved ejection fraction",
    "cva",
    "cerebrovascular",
    "stroke",
    "ihd",
    "ischemic heart disease",
    "coronary heart disease",
    "chd",
    "arrhythmia",
    "arr",
    "cardiomyopathy",
    "cm",
    "valvular",
    "vd",
]


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if len(c.strip()) >= 30]


def score_negative(sentence: str) -> float:
    low = sentence.lower()
    neg_hits = sum(1 for p in NEG_PATTERNS if re.search(p, low))
    if neg_hits == 0:
        return -1.0

    disease_hits = sum(1 for t in DISEASE_TERMS if t in low)
    if disease_hits == 0:
        return -1.0

    pos_hits = sum(1 for p in POS_PATTERNS if re.search(p, low))
    score = 1.5 * neg_hits + 0.8 * disease_hits - 0.9 * pos_hits
    return score


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
    parser = argparse.ArgumentParser(description="Build not_associated expansion dataset")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output-train", required=True)
    parser.add_argument("--output-added", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--max-corpus-add", type=int, default=400)
    parser.add_argument("--max-relabel", type=int, default=120)
    parser.add_argument("--min-score", type=float, default=2.0)
    args = parser.parse_args()

    with open(args.base_train) as f:
        base_rows = json.load(f)
    rows = [{"sentence": r["sentence"], "label": r["label"]} for r in base_rows]
    seen = {r["sentence"].strip() for r in rows}

    relabel_candidates = []
    for idx, row in enumerate(rows):
        if row["label"] != "incidental":
            continue
        score = score_negative(row["sentence"])
        if score >= args.min_score:
            relabel_candidates.append((score, idx))
    relabel_candidates.sort(reverse=True)
    relabel_selected = relabel_candidates[: args.max_relabel]

    changed = []
    for score, idx in relabel_selected:
        prev = rows[idx]["label"]
        rows[idx]["label"] = "not_associated"
        changed.append({
            "sentence": rows[idx]["sentence"],
            "old_label": prev,
            "new_label": "not_associated",
            "score": score,
            "source": "relabel_from_incidental_rule",
        })

    corpus_scored = []
    for sent in iter_corpus_sentences(Path(args.corpus)):
        sent = re.sub(r"\s+", " ", sent).strip()
        if sent in seen:
            continue
        score = score_negative(sent)
        if score >= args.min_score:
            corpus_scored.append((score, sent))

    corpus_scored.sort(reverse=True, key=lambda x: x[0])
    corpus_added = []
    for score, sent in corpus_scored:
        if len(corpus_added) >= args.max_corpus_add:
            break
        if sent in seen:
            continue
        seen.add(sent)
        rows.append({"sentence": sent, "label": "not_associated"})
        corpus_added.append({
            "sentence": sent,
            "label": "not_associated",
            "score": score,
            "source": "corpus_negative_rule",
        })

    out_train = Path(args.output_train)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with open(out_train, "w") as f:
        json.dump(rows, f, indent=2)

    added_payload = {
        "relabelled": changed,
        "corpus_added": corpus_added,
    }
    out_added = Path(args.output_added)
    out_added.parent.mkdir(parents=True, exist_ok=True)
    with open(out_added, "w") as f:
        json.dump(added_payload, f, indent=2)

    report = {
        "base_count": len(base_rows),
        "base_distribution": dict(Counter(r["label"] for r in base_rows)),
        "relabelled_count": len(changed),
        "corpus_added_count": len(corpus_added),
        "output_count": len(rows),
        "output_distribution": dict(Counter(r["label"] for r in rows)),
    }
    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    print("report", report)
    print("saved_train", out_train)
    print("saved_added", out_added)
    print("saved_report", out_report)


if __name__ == "__main__":
    main()


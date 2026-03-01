from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


HF_STRONG_TERMS = re.compile(
    r"\b("
    r"hfpef|hf-pef|hfnef|"
    r"heart failure with preserved ejection fraction|"
    r"heart failure with normal ejection fraction|"
    r"hf with preserved ef|"
    r"hf with normal ef"
    r")\b",
    re.I,
)

POSITIVE_RESULT_CUES = re.compile(
    r"\b("
    r"associated\s+with|significantly\s+associated|"
    r"correlated\s+with|significantly\s+correlated|"
    r"predict(?:ive|or)\s+of|independent\s+predictor|"
    r"increased|decreased|elevated|reduced|higher|lower"
    r")\b",
    re.I,
)


def norm_sentence(sentence: str) -> str:
    return re.sub(r"\s+", " ", (sentence or "").strip()).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a negative-query labeling pool with a corpus-derived candidate pool")
    parser.add_argument("--neg-labeling", required=True, help="CSV labeling file from scripts/build_labeling_pool_from_pmids.py")
    parser.add_argument("--corpus-candidates", required=True, help="CSV candidates file from scripts/build_labeling_pool_from_corpus.py")
    parser.add_argument("--out-labeling", required=True, help="Output merged labeling CSV path")
    parser.add_argument("--total", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    neg_path = Path(args.neg_labeling)
    corpus_path = Path(args.corpus_candidates)
    out_path = Path(args.out_labeling)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    neg_rows = []
    with neg_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            title = (row.get("title") or "").strip()
            neg_rows.append(
                {
                    "pmid": (row.get("pmid") or "").strip(),
                    "protein": (row.get("protein") or "").strip(),
                    "sentence": sentence,
                    "title": title,
                    "has_negative_result_cue": "True",
                    "has_strong_hfpef": "True" if (HF_STRONG_TERMS.search(sentence) or HF_STRONG_TERMS.search(title)) else "False",
                    "has_positive_result_cue": "True" if POSITIVE_RESULT_CUES.search(sentence) else "False",
                    "source": "neg_query",
                    "manual_label": "",
                }
            )

    corpus_rows = []
    with corpus_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            corpus_rows.append(row)

    seen = set()
    for row in neg_rows:
        seen.add((row["pmid"], row["protein"], norm_sentence(row["sentence"])))

    eligible = []
    for row in corpus_rows:
        pmid = (row.get("pmid") or "").strip()
        protein = (row.get("protein") or "").strip()
        sentence = (row.get("sentence") or "").strip()
        key = (pmid, protein, norm_sentence(sentence))
        if key in seen:
            continue
        seen.add(key)
        eligible.append(row)

    strong_nonneg = [r for r in eligible if r.get("has_strong_hfpef") == "True" and r.get("has_negative_result_cue") != "True"]
    other_nonneg = [r for r in eligible if r.get("has_strong_hfpef") != "True" and r.get("has_negative_result_cue") != "True"]
    rest = [r for r in eligible if r not in strong_nonneg and r not in other_nonneg]

    random.shuffle(strong_nonneg)
    random.shuffle(other_nonneg)
    random.shuffle(rest)

    selected = []
    selected.extend(neg_rows)

    if len(selected) > args.total:
        selected = selected[: args.total]

    remaining = args.total - len(selected)
    strong_quota = min(remaining // 2, len(strong_nonneg))
    selected.extend(strong_nonneg[:strong_quota])

    remaining = args.total - len(selected)
    selected.extend(other_nonneg[:remaining])

    remaining = args.total - len(selected)
    if remaining:
        selected.extend(rest[:remaining])

    selected = selected[: args.total]

    out_fields = [
        "pmid",
        "protein",
        "sentence",
        "title",
        "has_negative_result_cue",
        "has_strong_hfpef",
        "has_positive_result_cue",
        "source",
        "manual_label",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in selected:
            if "source" not in row:
                row = {
                    "pmid": (row.get("pmid") or "").strip(),
                    "protein": (row.get("protein") or "").strip(),
                    "sentence": (row.get("sentence") or "").strip(),
                    "title": (row.get("title") or "").strip(),
                    "has_negative_result_cue": str(row.get("has_negative_result_cue") or ""),
                    "has_strong_hfpef": str(row.get("has_strong_hfpef") or ""),
                    "has_positive_result_cue": str(row.get("has_positive_result_cue") or ""),
                    "source": "caseolap_corpus",
                    "manual_label": "",
                }
            writer.writerow(row)

    counts = Counter()
    for row in selected:
        counts["total"] += 1
        counts[f"source:{row.get('source', 'caseolap_corpus')}"] += 1
        if row.get("has_negative_result_cue") == "True":
            counts["neg_cue"] += 1
        if row.get("has_strong_hfpef") == "True":
            counts["strong_hfpef"] += 1

    print("out:", str(out_path))
    print("rows:", counts["total"])
    print("neg_cue:", counts.get("neg_cue", 0))
    print("strong_hfpef:", counts.get("strong_hfpef", 0))
    print("sources:", {k: v for k, v in counts.items() if k.startswith("source:")})


if __name__ == "__main__":
    main()


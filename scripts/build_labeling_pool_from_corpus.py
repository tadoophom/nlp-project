from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path


HF_TERMS = re.compile(
    r"\b("
    r"hfpef|hf-pef|hfnef|"
    r"heart failure with preserved ejection fraction|"
    r"heart failure with normal ejection fraction|"
    r"diastolic heart failure|"
    r"diastolic dysfunction|"
    r"preserved ejection fraction|"
    r"normal ejection fraction|"
    r"preserved ef|"
    r"normal ef|"
    r"hf with preserved ef|"
    r"hf with normal ef"
    r")\b",
    re.I,
)

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

METHODS = re.compile(
    r"\b("
    r"objective|aim|aims|method|methods|protocol|trial|randomized|double-blind|placebo|"
    r"cohort|cross-sectional|registry|study design|participants|enrolled|baseline|follow-up|"
    r"we investigated|we evaluated|we examined|we studied"
    r")\b",
    re.I,
)

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

NEGATIVE_RESULT_CUES = re.compile(
    r"\b("
    r"not\s+(?:independently\s+)?associated|"
    r"no\s+(?:statistically\s+)?significant\s+(?:difference|differences|association|associations|"
    r"correlation|correlations|relationship|relationships|change|changes|effect|effects)|"
    r"not\s+(?:statistically\s+)?significant(?:ly)?\s+(?:different|associated|correlated)|"
    r"no\s+association|lack\s+of\s+association|"
    r"no\s+(?:significant\s+)?correlation|"
    r"no\s+(?:clear\s+)?relationship|no\s+relation|"
    r"not\s+correlated|"
    r"did\s+not\s+(?:correlate|differ|predict|associate)|"
    r"does\s+not\s+(?:correlate|differ|predict|associate)|"
    r"failed\s+to\s+(?:correlate|predict|associate)|"
    r"not\s+predict(?:ive|or)|not\s+an?\s+(?:independent\s+)?predictor|"
    r"no\s+predictive\s+value|"
    r"no\s+difference|no\s+differences|not\s+different|did\s+not\s+differ|"
    r"no\s+evidence\s+of\s+(?:association|correlation|difference)|"
    r"unrelated"
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


def _normalise_synonym(term: str) -> str:
    term = term.replace("_", " ").strip()
    term = re.sub(r"\s+", " ", term).lower()
    if len(term) < 3:
        return ""
    if term.isdigit():
        return ""
    return term


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HFpEF protein sentence candidates from an existing corpus CSV")
    parser.add_argument("--corpus-csv", required=True, help="Path to corpus CSV produced by src.corpus_pipeline")
    parser.add_argument("--out-candidates", required=True, help="CSV output path for candidates")
    parser.add_argument("--out-labeling", required=True, help="CSV output path for labeling subset")
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means no limit")
    parser.add_argument("--labeling-neg", type=int, default=600, help="Target count of negative-result cue sentences")
    parser.add_argument("--labeling-strong", type=int, default=450, help="Target count of strong HFpEF mention sentences")
    parser.add_argument("--labeling-other", type=int, default=450, help="Target count of remaining sentences")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    corpus_path = Path(args.corpus_csv)
    out_candidates = Path(args.out_candidates)
    out_labeling = Path(args.out_labeling)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    out_labeling.parent.mkdir(parents=True, exist_ok=True)

    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, str, str]] = set()

    with corpus_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            protein = (row.get("protein") or "").strip()
            if not abstract or not protein:
                continue

            abstract_text = abstract.replace("\n", " ").strip()
            if not HF_TERMS.search(abstract_text) and not HF_TERMS.search(title):
                continue

            terms_raw = (row.get("protein_terms_used") or "").split(";")
            terms_norm = [_normalise_synonym(t) for t in terms_raw]
            terms_norm = [t for t in terms_norm if t]
            if not terms_norm:
                continue

            for sent in SENT_SPLIT.split(abstract_text):
                sentence = sent.strip()
                if not sentence:
                    continue
                if METHODS.search(sentence):
                    continue
                sent_norm = re.sub(r"\s+", " ", sentence).lower()
                if not any(t in sent_norm for t in terms_norm):
                    continue
                key = (pmid, protein, sent_norm)
                if key in seen:
                    continue
                seen.add(key)
                has_neg_cue = bool(NEGATIVE_RESULT_CUES.search(sentence))
                has_strong_hf = bool(HF_STRONG_TERMS.search(sentence) or HF_STRONG_TERMS.search(title))
                candidates.append(
                    {
                        "pmid": pmid,
                        "protein": protein,
                        "sentence": sentence,
                        "title": title,
                        "has_negative_result_cue": has_neg_cue,
                        "has_strong_hfpef": has_strong_hf,
                        "has_positive_result_cue": bool(POSITIVE_RESULT_CUES.search(sentence)),
                    }
                )
                if args.max_candidates and len(candidates) >= args.max_candidates:
                    break
            if args.max_candidates and len(candidates) >= args.max_candidates:
                break

    random.shuffle(candidates)

    fields = [
        "pmid",
        "protein",
        "sentence",
        "title",
        "has_negative_result_cue",
        "has_strong_hfpef",
        "has_positive_result_cue",
    ]
    with out_candidates.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)

    neg = [c for c in candidates if c["has_negative_result_cue"]]
    strong = [c for c in candidates if (c["has_strong_hfpef"] and not c["has_negative_result_cue"])]
    other = [c for c in candidates if (not c["has_negative_result_cue"] and not c["has_strong_hfpef"])]

    random.shuffle(neg)
    random.shuffle(strong)
    random.shuffle(other)

    target_total = args.labeling_neg + args.labeling_strong + args.labeling_other

    selected: list[dict[str, object]] = []
    selected.extend(neg[: args.labeling_neg])
    selected.extend(strong[: args.labeling_strong])
    selected.extend(other[: args.labeling_other])

    used = {(r["pmid"], r["protein"], re.sub(r"\s+", " ", str(r["sentence"])).lower()) for r in selected}
    if len(selected) < target_total:
        for cand in candidates:
            key = (cand["pmid"], cand["protein"], re.sub(r"\s+", " ", str(cand["sentence"])).lower())
            if key in used:
                continue
            used.add(key)
            selected.append(cand)
            if len(selected) >= target_total:
                break

    out_fields = fields + ["manual_label"]
    with out_labeling.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in selected:
            out = dict(row)
            out["manual_label"] = ""
            writer.writerow(out)

    counts = Counter()
    counts["candidates_total"] = len(candidates)
    counts["candidates_negative_cue"] = len(neg)
    counts["candidates_strong_hfpef"] = len([c for c in candidates if c["has_strong_hfpef"]])
    counts["labeling_total"] = len(selected)
    counts["labeling_negative_cue"] = len([c for c in selected if c["has_negative_result_cue"]])

    for k in ["candidates_total", "candidates_negative_cue", "candidates_strong_hfpef", "labeling_total", "labeling_negative_cue"]:
        print(f"{k}: {counts[k]}")


if __name__ == "__main__":
    main()

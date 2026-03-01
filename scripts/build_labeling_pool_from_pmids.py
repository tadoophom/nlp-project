"""Build HFpEF protein sentence candidates from PubMed PMIDs."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable, List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pubmed_fetch import fetch_abstracts


def load_pmids(path: Path) -> List[str]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return [str(p) for p in data]
    if isinstance(data, dict):
        if len(data) == 1:
            value = next(iter(data.values()))
            if isinstance(value, list):
                return [str(p) for p in value]
        pmids: List[str] = []
        for value in data.values():
            if isinstance(value, list):
                pmids.extend(str(p) for p in value)
        return pmids
    raise ValueError("Unsupported PMIDs JSON format")


def build_synonym_map(path: Path) -> Dict[str, str]:
    syn_to_protein: Dict[str, str] = {}
    syn_counts: Dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if not parts:
                continue
            protein = parts[0]
            for syn in parts[1:]:
                syn = syn.replace("_", " ").strip()
                syn_norm = re.sub(r"\s+", " ", syn).lower()
                if len(syn_norm) < 3:
                    continue
                if syn_norm.isdigit():
                    continue
                syn_counts[syn_norm] = syn_counts.get(syn_norm, 0) + 1
                if syn_norm not in syn_to_protein:
                    syn_to_protein[syn_norm] = protein

    for syn, count in list(syn_counts.items()):
        if count != 1:
            syn_to_protein.pop(syn, None)
    return syn_to_protein


def compile_chunks(terms: Iterable[str], chunk_size: int) -> List[re.Pattern[str]]:
    terms = list(terms)
    chunks = [terms[i:i + chunk_size] for i in range(0, len(terms), chunk_size)]
    return [
        re.compile(r"\b(?:" + "|".join(re.escape(s) for s in chunk) + r")\b", re.I)
        for chunk in chunks
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HFpEF protein sentence candidates from PubMed PMIDs")
    parser.add_argument("--pmid-json", required=True, help="Path to cellpmids.json")
    parser.add_argument("--entities-file", required=True, help="Path to entities_full.txt")
    parser.add_argument("--out-candidates", required=True, help="CSV output path for candidates")
    parser.add_argument("--out-labeling", required=True, help="CSV output path for labeling subset")
    parser.add_argument("--target-candidates", type=int, default=15000)
    parser.add_argument("--target-labeling", type=int, default=1500)
    parser.add_argument("--labeling-neg-target", type=int, default=0, help="Minimum number of negative-result cue sentences to include in labeling subset")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=10, help="Log progress every N batches")
    parser.add_argument("--require-negative-result-cue", action="store_true", help="Only keep candidates that contain a negative-result cue")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    hf_terms = re.compile(
        r"\b(hfpef|hf-pef|hfnef|heart failure with preserved ejection fraction|heart failure with normal ejection fraction|diastolic heart failure|diastolic dysfunction|preserved ejection fraction|normal ejection fraction|preserved ef|normal ef|hf with preserved ef|hf with normal ef)\b",
        re.I,
    )

    methods = re.compile(
        r"\b(objective|aim|aims|method|methods|protocol|trial|randomized|double-blind|placebo|cohort|cross-sectional|registry|we investigated|we evaluated|we examined|we studied|study design|participants|enrolled|baseline|follow-up)\b",
        re.I,
    )

    negative_result_cues = re.compile(
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

    sent_split = re.compile(r"(?<=[.!?])\s+")

    pmids = load_pmids(Path(args.pmid_json))
    random.shuffle(pmids)

    syn_to_protein = build_synonym_map(Path(args.entities_file))
    regex_chunks = compile_chunks(syn_to_protein.keys(), 200)

    candidates = []

    for batch_idx, i in enumerate(range(0, len(pmids), args.batch_size), start=1):
        if len(candidates) >= args.target_candidates:
            break
        batch = pmids[i:i + args.batch_size]
        articles = fetch_abstracts(batch)
        for article in articles:
            pmid = str(article.get("pmid") or "")
            title = str(article.get("title") or "")
            abstract = str(article.get("abstract") or "")
            if not abstract:
                continue
            text = abstract.replace("\n", " ").strip()
            if not hf_terms.search(text):
                continue
            for sent in sent_split.split(text):
                sentence = sent.strip()
                if not sentence:
                    continue
                if methods.search(sentence):
                    continue
                has_negative_result_cue = bool(negative_result_cues.search(sentence))
                if args.require_negative_result_cue and not has_negative_result_cue:
                    continue
                hf_in_sentence = bool(hf_terms.search(sentence))
                sent_norm = re.sub(r"\s+", " ", sentence).lower()
                proteins = set()
                for rgx in regex_chunks:
                    for match in rgx.finditer(sent_norm):
                        syn = match.group(0).lower()
                        protein = syn_to_protein.get(syn)
                        if protein:
                            proteins.add(protein)
                if not proteins:
                    continue
                candidates.append({
                    "pmid": pmid,
                    "protein": ";".join(sorted(proteins)),
                    "sentence": sentence,
                    "title": title,
                    "hf_in_sentence": "yes" if hf_in_sentence else "no",
                    "has_negative_result_cue": "yes" if has_negative_result_cue else "no",
                })
                if len(candidates) >= args.target_candidates:
                    break
            if len(candidates) >= args.target_candidates:
                break
        if batch_idx % args.log_every == 0:
            print(f"batches: {batch_idx}, candidates: {len(candidates)}")

    out_candidates = Path(args.out_candidates)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    candidate_fields = ["pmid", "protein", "sentence", "title", "hf_in_sentence", "has_negative_result_cue"]
    with out_candidates.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow(row)

    random.shuffle(candidates)
    if args.labeling_neg_target and args.labeling_neg_target > 0:
        neg = [c for c in candidates if c.get("has_negative_result_cue") == "yes"]
        other = [c for c in candidates if c.get("has_negative_result_cue") != "yes"]
        random.shuffle(neg)
        random.shuffle(other)
        target_neg = min(args.labeling_neg_target, args.target_labeling)
        subset = neg[:target_neg]
        if len(subset) < args.target_labeling:
            subset.extend(other[: args.target_labeling - len(subset)])
    else:
        subset = candidates[: args.target_labeling]

    out_labeling = Path(args.out_labeling)
    out_labeling.parent.mkdir(parents=True, exist_ok=True)
    with out_labeling.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields + ["manual_label"])
        writer.writeheader()
        for row in subset:
            row_out = dict(row)
            row_out["manual_label"] = ""
            writer.writerow(row_out)

    print("candidates:", len(candidates))
    print("negative cue candidates:", sum(1 for c in candidates if c.get("has_negative_result_cue") == "yes"))
    print("labeling subset:", len(subset))


if __name__ == "__main__":
    main()

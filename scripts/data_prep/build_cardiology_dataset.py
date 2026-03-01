"""Build broader cardiology training set via PubMed fetch + pseudo-labeling.

Fetches non-HFpEF cardiology abstracts, extracts protein-disease co-mention
sentences, and pseudo-labels them with the best existing model.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import spacy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmed_fetch import search_pubmed_advanced, fetch_abstracts
from src.bert_classifier import preprocess_sentence, LABEL2ID
from src.pubmedbert_classifier import PubMedBERTClassifier

ROOT = Path(__file__).resolve().parent.parent.parent

DISEASE_MESH = [
    "Cardiomyopathies",
    "Heart Failure",
    "Myocardial Ischemia",
]

DISEASE_KEYWORDS = [
    "dilated cardiomyopathy",
    "hypertrophic cardiomyopathy",
    "restrictive cardiomyopathy",
    "heart failure with reduced ejection fraction",
    "HFrEF",
    "ischemic heart disease",
    "myocardial infarction",
    "arrhythmogenic cardiomyopathy",
]

EXCLUDE_KEYWORDS = [
    "HFpEF",
    "heart failure with preserved ejection fraction",
]

SECTION_HEADERS = re.compile(
    r'^(INTRODUCTION|BACKGROUND|METHODS|RESULTS|DISCUSSION|CONCLUSION|OBJECTIVE|PURPOSE|AIM|AIMS):?\s+',
    re.IGNORECASE,
)


def load_protein_names(entities_path: Path, caseolap_path: Path, top_n: int = 200) -> list[tuple[str, list[str]]]:
    """Load top N proteins from CaseOLAP with their synonym lists."""
    # Load scores and sort
    scores = {}
    with open(caseolap_path) as f:
        import csv
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            uid, score = row[0], float(row[1])
            scores[uid] = score

    top_uids = sorted(scores, key=scores.get, reverse=True)[:top_n]
    top_set = set(top_uids)

    # Load names
    proteins = []
    with open(entities_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if not parts:
                continue
            uid = parts[0]
            if uid not in top_set:
                continue
            names = [n.replace("_", " ") for n in parts[1:] if len(n) > 2]
            if names:
                proteins.append((uid, names))

    return proteins


def extract_sentences(abstract: str, nlp) -> list[str]:
    """Split abstract into sentences using spaCy."""
    doc = nlp(abstract)
    sents = []
    for sent in doc.sents:
        text = sent.text.strip()
        text = preprocess_sentence(text)
        if len(text) >= 30:
            sents.append(text)
    return sents


def has_protein_mention(sentence: str, protein_names: list[str]) -> bool:
    """Check if sentence mentions any protein name."""
    lower = sentence.lower()
    for name in protein_names:
        if name.lower() in lower:
            return True
    return False


def has_disease_mention(sentence: str) -> bool:
    """Check if sentence mentions a cardiology disease term."""
    lower = sentence.lower()
    terms = DISEASE_KEYWORDS + [
        "cardiomyopathy", "heart failure", "myocardial",
        "cardiac", "ventricular dysfunction",
    ]
    for term in terms:
        if term.lower() in lower:
            return True
    return False


def has_hfpef_mention(sentence: str) -> bool:
    """Check if sentence mentions HFpEF (should be excluded)."""
    lower = sentence.lower()
    return "hfpef" in lower or "preserved ejection fraction" in lower


def fetch_cardiology_abstracts(
    protein_names: list[tuple[str, list[str]]],
    retmax: int = 20,
) -> list[dict]:
    """Fetch non-HFpEF cardiology abstracts for each protein."""
    all_records = {}  # pmid -> record (deduplicate)

    disease_clause = " OR ".join(
        [f'"{kw}"[Title/Abstract]' for kw in DISEASE_KEYWORDS]
        + [f'"{m}"[MeSH Terms]' for m in DISEASE_MESH]
    )
    exclude_clause = " OR ".join(f'"{kw}"[Title/Abstract]' for kw in EXCLUDE_KEYWORDS)

    total = len(protein_names)
    for i, (uid, names) in enumerate(protein_names):
        # Use first 3 names to keep query manageable
        name_terms = names[:3]
        protein_clause = " OR ".join(f'"{n}"[Title/Abstract]' for n in name_terms)

        raw = f"({disease_clause}) AND ({protein_clause}) NOT ({exclude_clause})"

        ids, query = search_pubmed_advanced(
            keywords=[], mesh_terms=[], retmax=retmax, raw_query=raw,
        )

        if ids:
            records = fetch_abstracts(ids)
            for rec in records:
                if rec["pmid"] not in all_records:
                    all_records[rec["pmid"]] = rec

        if (i + 1) % 20 == 0 or i == total - 1:
            print(f"  Fetched {i + 1}/{total} proteins, {len(all_records)} unique abstracts so far")

    return list(all_records.values())


def extract_comention_sentences(
    records: list[dict],
    protein_list: list[tuple[str, list[str]]],
    nlp,
) -> list[str]:
    """Extract sentences with protein-disease co-mentions from abstracts."""
    all_names = []
    for _, names in protein_list:
        all_names.extend(names)

    sentences = set()
    for rec in records:
        abstract = rec.get("abstract", "")
        if not abstract or abstract == "No abstract available":
            continue

        for sent in extract_sentences(abstract, nlp):
            if has_hfpef_mention(sent):
                continue
            if has_protein_mention(sent, all_names) and has_disease_mention(sent):
                sentences.add(sent)

    return list(sentences)


def pseudo_label(
    sentences: list[str],
    model_path: Path,
    confidence_threshold: float = 0.90,
    max_per_class: int = 2000,
    batch_size: int = 32,
) -> list[dict]:
    """Pseudo-label sentences using the best existing model."""
    clf = PubMedBERTClassifier(model_path=model_path)
    results = clf.classify_batch(sentences, batch_size=batch_size)

    # Show confidence distribution for diagnostics
    confs = [r["confidence"] for r in results]
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        n = sum(1 for c in confs if c >= threshold)
        print(f"    >= {threshold:.2f}: {n} sentences")

    by_class: dict[str, list[dict]] = {label: [] for label in LABEL2ID}

    for sent, res in zip(sentences, results):
        if res["confidence"] < confidence_threshold:
            continue
        label = res["label"]
        if len(by_class[label]) >= max_per_class:
            continue
        by_class[label].append({
            "sentence": sent,
            "label": label,
        })

    labeled = []
    for items in by_class.values():
        labeled.extend(items)

    return labeled


def main():
    parser = argparse.ArgumentParser(description="Build broader cardiology training dataset")
    parser.add_argument("--retmax", type=int, default=20, help="Max PubMed results per protein")
    parser.add_argument("--confidence", type=float, default=0.90, help="Pseudo-labeling confidence threshold")
    parser.add_argument("--max-per-class", type=int, default=2000, help="Max samples per class")
    parser.add_argument("--top-proteins", type=int, default=200, help="Top N proteins from CaseOLAP")
    parser.add_argument("--output", type=str, default="data/splits/cardiology_broad_train.json")
    parser.add_argument("--model-path", type=str,
                        default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--cache-dir", type=str, default="data/cache",
                        help="Directory to cache fetched abstracts")
    args = parser.parse_args()

    entities_path = ROOT / "data" / "caseolap" / "aktan_caseolap_2026-02-04" / "entities_full.txt"
    caseolap_path = ROOT / "data" / "caseolap" / "aktan_caseolap_2026-02-04" / "caseolap.csv"
    model_path = ROOT / args.model_path

    print("Loading protein names...")
    proteins = load_protein_names(entities_path, caseolap_path, top_n=args.top_proteins)
    print(f"  {len(proteins)} proteins loaded")

    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "cardiology_abstracts.json"

    if cache_file.exists():
        print(f"Loading cached abstracts from {cache_file}...")
        with open(cache_file) as f:
            records = json.load(f)
        print(f"  {len(records)} cached abstracts loaded")
    else:
        print("Fetching cardiology abstracts from PubMed...")
        records = fetch_cardiology_abstracts(proteins, retmax=args.retmax)
        print(f"  {len(records)} unique abstracts fetched")
        with open(cache_file, "w") as f:
            json.dump(records, f)
        print(f"  Cached to {cache_file}")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("Extracting co-mention sentences...")
    sentences = extract_comention_sentences(records, proteins, nlp)
    print(f"  {len(sentences)} co-mention sentences extracted")

    if not sentences:
        print("No sentences found. Exiting.")
        return

    print(f"Pseudo-labeling with threshold={args.confidence}...")
    labeled = pseudo_label(
        sentences, model_path,
        confidence_threshold=args.confidence,
        max_per_class=args.max_per_class,
    )

    dist = Counter(d["label"] for d in labeled)
    print(f"  {len(labeled)} labeled sentences")
    print(f"  Distribution: {dict(dist)}")

    # Verify no HFpEF leakage
    leaked = sum(1 for d in labeled if has_hfpef_mention(d["sentence"]))
    if leaked:
        print(f"  WARNING: {leaked} sentences mention HFpEF -- removing")
        labeled = [d for d in labeled if not has_hfpef_mention(d["sentence"])]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()

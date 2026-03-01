"""Build cross-disease CVD training set from 6 primary heart disease categories.

Strategy: fetch abstracts per-disease (not per-protein) with large retmax,
then locally match protein mentions against the full UniProt synonym list.
Extracts sentences with surrounding context windows for richer classification.

Disease categories (from CaseOLAP):
  CVA  - Cerebrovascular Accident / Stroke
  IHD  - Ischemic Heart Disease
  CM   - Cardiomyopathy
  CHD  - Congenital Heart Disease
  ARR  - Arrhythmia
  VD   - Valvular Disease
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import spacy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmed_fetch import search_pubmed_advanced, fetch_abstracts
from src.bert_classifier import preprocess_sentence, LABEL2ID
from src.pubmedbert_classifier import PubMedBERTClassifier

ROOT = Path(__file__).resolve().parent.parent.parent

DISEASE_CATEGORIES = {
    "CVA": {
        "mesh": ["Stroke", "Cerebrovascular Disorders"],
        "keywords": ["stroke", "cerebrovascular accident", "cerebral infarction",
                     "ischemic stroke", "hemorrhagic stroke", "transient ischemic attack"],
    },
    "IHD": {
        "mesh": ["Myocardial Ischemia", "Coronary Artery Disease"],
        "keywords": ["ischemic heart disease", "coronary artery disease",
                     "myocardial infarction", "acute coronary syndrome", "angina pectoris"],
    },
    "CM": {
        "mesh": ["Cardiomyopathies"],
        "keywords": ["cardiomyopathy", "dilated cardiomyopathy",
                     "hypertrophic cardiomyopathy", "restrictive cardiomyopathy",
                     "arrhythmogenic cardiomyopathy"],
    },
    "CHD": {
        "mesh": ["Heart Defects, Congenital"],
        "keywords": ["congenital heart disease", "congenital heart defect",
                     "ventricular septal defect", "atrial septal defect",
                     "tetralogy of fallot"],
    },
    "ARR": {
        "mesh": ["Arrhythmias, Cardiac"],
        "keywords": ["arrhythmia", "atrial fibrillation", "ventricular tachycardia",
                     "long QT syndrome", "cardiac arrhythmia", "supraventricular tachycardia"],
    },
    "VD": {
        "mesh": ["Heart Valve Diseases"],
        "keywords": ["valvular disease", "aortic stenosis", "mitral regurgitation",
                     "aortic regurgitation", "mitral stenosis", "valvular heart disease"],
    },
}

EXCLUDE_KEYWORDS = ["HFpEF", "heart failure with preserved ejection fraction"]


def load_all_protein_names(entities_path: Path) -> dict[str, list[str]]:
    """Load full UniProt protein list with synonyms. Returns uid -> [names]."""
    proteins = {}
    with open(entities_path) as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            uid = parts[0]
            names = [n.replace("_", " ") for n in parts[1:] if len(n) > 2]
            if names:
                proteins[uid] = names
    return proteins


def build_name_index(proteins: dict[str, list[str]]) -> dict[str, str]:
    """Build lowercase name -> uid index for fast matching."""
    index = {}
    for uid, names in proteins.items():
        for name in names:
            lower = name.lower()
            if len(lower) > 3:
                index[lower] = uid
    return index


def find_protein_mentions(text: str, name_index: dict[str, str]) -> list[str]:
    """Find which proteins are mentioned in text. Returns list of UIDs."""
    lower = text.lower()
    found = set()
    for name, uid in name_index.items():
        if name in lower:
            found.add(uid)
    return list(found)


def has_disease_mention(text: str, category: str) -> bool:
    """Check if text mentions any term from a disease category."""
    lower = text.lower()
    cat = DISEASE_CATEGORIES[category]
    for term in cat["keywords"]:
        if term.lower() in lower:
            return True
    for mesh in cat["mesh"]:
        if mesh.lower() in lower:
            return True
    return False


def has_hfpef_mention(text: str) -> bool:
    lower = text.lower()
    return "hfpef" in lower or "preserved ejection fraction" in lower


def fetch_disease_abstracts(category: str, retmax: int = 5000) -> list[dict]:
    """Fetch abstracts for a single disease category."""
    cat = DISEASE_CATEGORIES[category]

    mesh_clause = " OR ".join(f'"{m}"[MeSH Terms]' for m in cat["mesh"])
    kw_clause = " OR ".join(f'"{k}"[Title/Abstract]' for k in cat["keywords"])
    exclude_clause = " OR ".join(f'"{k}"[Title/Abstract]' for k in EXCLUDE_KEYWORDS)

    raw = f"({mesh_clause} OR {kw_clause}) NOT ({exclude_clause})"

    ids, query = search_pubmed_advanced(
        keywords=[], mesh_terms=[], retmax=retmax, raw_query=raw,
    )

    if not ids:
        print(f"    No results for {category}")
        return []

    # Fetch in batches of 200 (NCBI limit)
    all_records = []
    for i in range(0, len(ids), 200):
        batch_ids = ids[i:i + 200]
        records = fetch_abstracts(batch_ids)
        all_records.extend(records)

    return all_records


def extract_sentences_with_context(
    abstract: str,
    nlp,
    window: int = 1,
) -> list[dict]:
    """Split abstract into sentences and return each with surrounding context.

    Returns list of dicts with keys:
        - sentence: the target sentence (preprocessed)
        - context: target sentence with surrounding sentences concatenated
        - sent_index: position in abstract
    """
    doc = nlp(abstract)
    raw_sents = []
    for sent in doc.sents:
        text = sent.text.strip()
        cleaned = preprocess_sentence(text)
        if len(cleaned) >= 30:
            raw_sents.append(cleaned)

    results = []
    for i, sent in enumerate(raw_sents):
        parts = []
        for j in range(max(0, i - window), min(len(raw_sents), i + window + 1)):
            parts.append(raw_sents[j])
        context = " ".join(parts)
        results.append({
            "sentence": sent,
            "context": context,
            "sent_index": i,
        })

    return results


def build_dataset_for_category(
    category: str,
    records: list[dict],
    name_index: dict[str, str],
    nlp,
    window: int = 1,
) -> list[dict]:
    """Extract protein-disease co-mention sentences with context for one category."""
    sentences = []
    seen = set()

    for rec in records:
        abstract = rec.get("abstract", "")
        if not abstract or abstract == "No abstract available":
            continue

        sent_data = extract_sentences_with_context(abstract, nlp, window=window)
        for sd in sent_data:
            if has_hfpef_mention(sd["sentence"]):
                continue
            if sd["sentence"] in seen:
                continue

            protein_uids = find_protein_mentions(sd["sentence"], name_index)
            if not protein_uids:
                continue
            if not has_disease_mention(sd["sentence"], category):
                continue

            seen.add(sd["sentence"])
            sentences.append({
                "sentence": sd["sentence"],
                "context": sd["context"],
                "disease_category": category,
                "pmid": rec.get("pmid", ""),
                "protein_uids": protein_uids,
            })

    return sentences


def pseudo_label(
    data: list[dict],
    model_path: Path,
    confidence_threshold: float,
    max_per_class: int,
    use_context: bool = True,
    batch_size: int = 32,
) -> list[dict]:
    """Pseudo-label extracted sentences."""
    clf = PubMedBERTClassifier(model_path=model_path)

    texts = [d["context" if use_context else "sentence"] for d in data]
    results = clf.classify_batch(texts, batch_size=batch_size)

    confs = [r["confidence"] for r in results]
    for threshold in [0.50, 0.60, 0.70, 0.80, 0.90]:
        n = sum(1 for c in confs if c >= threshold)
        print(f"    >= {threshold:.2f}: {n} sentences")

    by_class: dict[str, list[dict]] = {label: [] for label in LABEL2ID}

    for d, res in zip(data, results):
        if res["confidence"] < confidence_threshold:
            continue
        label = res["label"]
        if len(by_class[label]) >= max_per_class:
            continue
        by_class[label].append({
            "sentence": d["context" if use_context else "sentence"],
            "label": label,
            "disease_category": d["disease_category"],
            "confidence": res["confidence"],
        })

    labeled = []
    for items in by_class.values():
        labeled.extend(items)

    return labeled


def main():
    parser = argparse.ArgumentParser(description="Build multi-disease CVD training dataset")
    parser.add_argument("--retmax", type=int, default=5000,
                        help="Max PubMed results per disease category")
    parser.add_argument("--confidence", type=float, default=0.50,
                        help="Pseudo-labeling confidence threshold")
    parser.add_argument("--max-per-class", type=int, default=3000,
                        help="Max samples per class")
    parser.add_argument("--window", type=int, default=1,
                        help="Context window size (sentences before/after)")
    parser.add_argument("--categories", nargs="*",
                        default=list(DISEASE_CATEGORIES.keys()),
                        help="Disease categories to include")
    parser.add_argument("--no-context", action="store_true",
                        help="Use single sentences instead of context windows")
    parser.add_argument("--output", type=str,
                        default="data/splits/cvd_broad_train.json")
    parser.add_argument("--model-path", type=str,
                        default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    args = parser.parse_args()

    entities_path = ROOT / "data" / "caseolap" / "aktan_caseolap_2026-02-04" / "entities_full.txt"
    model_path = ROOT / args.model_path

    print("Loading full UniProt protein list...")
    proteins = load_all_protein_names(entities_path)
    print(f"  {len(proteins)} proteins loaded")

    print("Building name index...")
    name_index = build_name_index(proteins)
    print(f"  {len(name_index)} unique name entries")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    cache_dir = ROOT / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_extracted = []

    for category in args.categories:
        print(f"\n=== {category}: {DISEASE_CATEGORIES[category]['mesh'][0]} ===")

        cache_file = cache_dir / f"cvd_{category.lower()}_abstracts.json"

        if cache_file.exists():
            print(f"  Loading cached abstracts from {cache_file}...")
            with open(cache_file) as f:
                records = json.load(f)
            print(f"  {len(records)} cached abstracts")
        else:
            print(f"  Fetching from PubMed (retmax={args.retmax})...")
            records = fetch_disease_abstracts(category, retmax=args.retmax)
            print(f"  {len(records)} abstracts fetched")
            with open(cache_file, "w") as f:
                json.dump(records, f)

        print(f"  Extracting co-mention sentences (window={args.window})...")
        extracted = build_dataset_for_category(
            category, records, name_index, nlp, window=args.window,
        )
        print(f"  {len(extracted)} co-mention sentences")
        all_extracted.extend(extracted)

    # Deduplicate across categories
    seen = set()
    deduped = []
    for d in all_extracted:
        if d["sentence"] not in seen:
            seen.add(d["sentence"])
            deduped.append(d)
    print(f"\nTotal unique sentences: {len(deduped)} (from {len(all_extracted)} raw)")

    cat_dist = Counter(d["disease_category"] for d in deduped)
    print(f"Per-category: {dict(cat_dist)}")

    if not deduped:
        print("No sentences found. Exiting.")
        return

    print(f"\nPseudo-labeling with threshold={args.confidence}...")
    use_context = not args.no_context
    labeled = pseudo_label(
        deduped, model_path,
        confidence_threshold=args.confidence,
        max_per_class=args.max_per_class,
        use_context=use_context,
    )

    dist = Counter(d["label"] for d in labeled)
    print(f"  {len(labeled)} labeled sentences")
    print(f"  Distribution: {dict(dist)}")

    # Strip metadata for training format, keep sentence+label
    output_data = [{"sentence": d["sentence"], "label": d["label"]} for d in labeled]

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {output_path}")

    # Also save full metadata version
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(labeled, f, indent=2)
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()

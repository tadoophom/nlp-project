"""Data augmentation, benchmark conversion, and semi-supervised expansion."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import LABEL2ID, preprocess_sentence

ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 1a. Minority-class augmentation for not_associated
# ---------------------------------------------------------------------------

NEGATION_SYNONYMS = [
    "no association", "not associated", "no relationship", "no correlation",
    "no significant association", "no link", "not correlated", "not related",
    "no connection", "unrelated", "failed to show association",
]

def _swap_negation_phrase(sentence: str) -> str | None:
    """Swap one negation phrase for another."""
    for phrase in NEGATION_SYNONYMS:
        if phrase.lower() in sentence.lower():
            replacement = random.choice([p for p in NEGATION_SYNONYMS if p != phrase])
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            return pattern.sub(replacement, sentence, count=1)
    return None


def _clause_reorder(sentence: str) -> str | None:
    """Swap clauses around a comma or semicolon midpoint."""
    for sep in ["; ", ", "]:
        parts = sentence.split(sep, 1)
        if len(parts) == 2 and len(parts[0].split()) >= 3 and len(parts[1].split()) >= 3:
            return parts[1].rstrip(".") + sep + parts[0].lstrip() + "."
    return None


def _entity_dropout(sentence: str, nlp) -> str | None:
    """Replace one biomedical entity with a generic placeholder."""
    doc = nlp(sentence)
    ents = [e for e in doc.ents if e.label_ in ("CHEMICAL", "DISEASE", "ORG", "GPE") or len(e.text) > 3]
    if not ents:
        return None
    ent = random.choice(ents)
    replacement = f"[{ent.label_}]" if ent.label_ else "[ENTITY]"
    return sentence[:ent.start_char] + replacement + sentence[ent.end_char:]


def augment_minority(data: list[dict], target_count: int = 270, seed: int = 42) -> list[dict]:
    """Augment not_associated samples to reach target_count."""
    random.seed(seed)

    minority = [d for d in data if d["label"] == "not_associated"]
    other = [d for d in data if d["label"] != "not_associated"]

    if len(minority) >= target_count:
        return data

    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None

    augmented = list(minority)
    attempts = 0
    seen = {d["sentence"] for d in data}

    while len(augmented) < target_count and attempts < target_count * 10:
        attempts += 1
        source = random.choice(minority)
        sent = source["sentence"]

        aug_fn = random.choice([_swap_negation_phrase, _clause_reorder])
        new_sent = aug_fn(sent)

        if new_sent is None and nlp is not None:
            new_sent = _entity_dropout(sent, nlp)

        if new_sent and new_sent not in seen:
            seen.add(new_sent)
            augmented.append({"sentence": new_sent, "label": "not_associated"})

    print(f"Augmented not_associated: {len(minority)} -> {len(augmented)}")
    return other + augmented


# ---------------------------------------------------------------------------
# 1b. Benchmark conversion (ChemProt + DDI -> 2-class pretrain data)
# ---------------------------------------------------------------------------

def load_benchmark_csv(path: Path) -> list[dict]:
    """Load benchmark CSV with columns: text, label (associated/not_associated)."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            if label not in ("associated", "not_associated"):
                continue
            text = row.get("text", "").strip()
            if not text or len(text) < 20:
                continue
            rows.append({"sentence": text, "label": label})
    return rows


def convert_benchmarks(
    chemprot_path: Path,
    ddi_path: Path,
    max_per_class: int = 3000,
    seed: int = 42,
) -> list[dict]:
    """Convert ChemProt + DDI to unified pretrain format (2-class only)."""
    random.seed(seed)

    all_rows = []
    for path, name in [(chemprot_path, "chemprot"), (ddi_path, "ddi")]:
        rows = load_benchmark_csv(path)
        print(f"  {name}: {len(rows)} usable rows")
        all_rows.extend(rows)

    counts = Counter(r["label"] for r in all_rows)
    print(f"  Combined: {dict(counts)}")

    # Balance classes by downsampling the majority
    by_label = {}
    for r in all_rows:
        by_label.setdefault(r["label"], []).append(r)

    balanced = []
    for label, items in by_label.items():
        if len(items) > max_per_class:
            items = random.sample(items, max_per_class)
        balanced.extend(items)

    random.shuffle(balanced)
    counts = Counter(r["label"] for r in balanced)
    print(f"  After balancing (max {max_per_class}/class): {dict(counts)}")
    return balanced


# ---------------------------------------------------------------------------
# 1c. Semi-supervised pseudo-labeling from corpus
# ---------------------------------------------------------------------------

def _sentence_split(text: str) -> list[str]:
    """Split text into sentences using basic regex."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) >= 30]


def pseudo_label_corpus(
    corpus_path: Path,
    model_path: Path,
    confidence_threshold: float = 0.85,
    max_samples: int = 2000,
) -> list[dict]:
    """Run model on corpus sentences, keep high-confidence predictions."""
    from src.pubmedbert_classifier import PubMedBERTClassifier

    clf = PubMedBERTClassifier(model_path=model_path)

    sentences = []
    with open(corpus_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ev = row.get("evidence_sentence", "").strip()
            if ev and len(ev) >= 30:
                sentences.append(preprocess_sentence(ev))

            abstract = row.get("abstract", "").strip()
            if abstract:
                for sent in _sentence_split(abstract):
                    sent = preprocess_sentence(sent)
                    if len(sent) >= 30:
                        sentences.append(sent)

    # Deduplicate and shuffle for diversity
    sentences = list(dict.fromkeys(sentences))
    random.shuffle(sentences)
    print(f"  Corpus sentences to classify: {len(sentences)}")

    # Process in batches with early stopping once we have enough
    pseudo = []
    batch_size = 64
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        results = clf.classify_batch(batch, batch_size=batch_size)
        for sent, result in zip(batch, results):
            if result["confidence"] >= confidence_threshold:
                pseudo.append({"sentence": sent, "label": result["label"]})
        if len(pseudo) >= max_samples:
            pseudo = pseudo[:max_samples]
            break
        if (i // batch_size) % 50 == 0:
            print(f"    Processed {i + len(batch)}/{len(sentences)}, collected {len(pseudo)} so far")

    counts = Counter(r["label"] for r in pseudo)
    print(f"  Pseudo-labeled (conf >= {confidence_threshold}): {len(pseudo)}")
    print(f"  Distribution: {dict(counts)}")
    return pseudo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Augment and expand training data")
    parser.add_argument("--train-data", default="data/splits/hfpef_v3_train.json",
                        help="Base training data")
    parser.add_argument("--augment", action="store_true",
                        help="1a: Augment not_associated minority class")
    parser.add_argument("--augment-target", type=int, default=270,
                        help="Target count for not_associated")
    parser.add_argument("--benchmarks", action="store_true",
                        help="1b: Convert benchmark data for pretraining")
    parser.add_argument("--chemprot-csv", default="data/benchmarks/chemprot/chemprot_relations.csv")
    parser.add_argument("--ddi-csv", default="data/benchmarks/ddi/ddi_relations.csv")
    parser.add_argument("--pseudo-label", action="store_true",
                        help="1c: Semi-supervised pseudo-labeling")
    parser.add_argument("--corpus", default="data/corpus/aktan_caseolap_2026-02-05_corpus_top500_retmax50.csv")
    parser.add_argument("--model-path", default="models/hfpef_v3/pubmedbert/final",
                        help="Model for pseudo-labeling")
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.all:
        args.augment = True
        args.benchmarks = True
        args.pseudo_label = True

    with open(args.train_data) as f:
        base_data = json.load(f)

    base_counts = Counter(d["label"] for d in base_data)
    print(f"Base training data: {len(base_data)} samples")
    print(f"Distribution: {dict(base_counts)}\n")

    output_dir = ROOT / "data" / "splits"

    # 1a: Augment minority class
    if args.augment:
        print("=== 1a: Augmenting not_associated class ===")
        augmented = augment_minority(base_data, target_count=args.augment_target, seed=args.seed)
        aug_path = output_dir / "hfpef_v3_train_augmented.json"
        with open(aug_path, "w") as f:
            json.dump(augmented, f, indent=2)
        aug_counts = Counter(d["label"] for d in augmented)
        print(f"Saved {len(augmented)} samples to {aug_path}")
        print(f"Distribution: {dict(aug_counts)}\n")

    # 1b: Convert benchmarks
    if args.benchmarks:
        print("=== 1b: Converting benchmark data ===")
        benchmark_data = convert_benchmarks(
            Path(args.chemprot_csv), Path(args.ddi_csv), seed=args.seed,
        )
        bench_path = output_dir / "benchmark_pretrain.json"
        with open(bench_path, "w") as f:
            json.dump(benchmark_data, f, indent=2)
        print(f"Saved {len(benchmark_data)} samples to {bench_path}\n")

    # 1c: Semi-supervised pseudo-labeling
    if args.pseudo_label:
        print("=== 1c: Semi-supervised pseudo-labeling ===")
        # Combine base + augmented if available
        if args.augment:
            combined = augmented
        else:
            combined = base_data

        pseudo = pseudo_label_corpus(
            Path(args.corpus), Path(args.model_path),
            confidence_threshold=args.confidence,
        )

        semi_data = combined + pseudo
        semi_path = output_dir / "hfpef_v3_train_semi.json"

        # Deduplicate by sentence
        seen = set()
        deduped = []
        for d in semi_data:
            if d["sentence"] not in seen:
                seen.add(d["sentence"])
                deduped.append(d)

        with open(semi_path, "w") as f:
            json.dump(deduped, f, indent=2)
        semi_counts = Counter(d["label"] for d in deduped)
        print(f"Saved {len(deduped)} samples to {semi_path}")
        print(f"Distribution: {dict(semi_counts)}\n")

    print("Done.")


if __name__ == "__main__":
    main()

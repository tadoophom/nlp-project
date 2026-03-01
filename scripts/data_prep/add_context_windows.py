"""Add surrounding sentence context to existing labeled data.

Traces each labeled sentence back to its source abstract in the corpus CSV,
finds its position, and concatenates neighboring sentences as context.

Input:  data/splits/hfpef_v3_train_augmented.json  (sentence + label)
        data/splits/hfpef_v3_eval.json
        data/corpus/hfpef_corpus.csv                (has abstracts)

Output: data/splits/hfpef_v3_train_context.json     (context + label)
        data/splits/hfpef_v3_eval_context.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import spacy

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.bert_classifier import preprocess_sentence

ROOT = Path(__file__).resolve().parent.parent.parent


def load_abstracts_from_corpus(corpus_path: Path) -> dict[str, str]:
    """Load PMID -> abstract mapping from corpus CSV."""
    abstracts = {}
    with open(corpus_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("pmid", "")
            abstract = row.get("abstract", "")
            if pmid and abstract and abstract != "No abstract available":
                abstracts[pmid] = abstract
    return abstracts


def split_abstract_to_sentences(abstract: str, nlp) -> list[str]:
    """Split abstract into preprocessed sentences."""
    doc = nlp(abstract)
    sents = []
    for sent in doc.sents:
        text = preprocess_sentence(sent.text.strip())
        if len(text) >= 20:
            sents.append(text)
    return sents


def build_sentence_to_context(abstracts: dict[str, str], nlp, window: int = 1) -> dict[str, str]:
    """Build a mapping from sentence -> context (with surrounding sentences).

    For each sentence in every abstract, stores the context window.
    """
    sent_to_context = {}

    for pmid, abstract in abstracts.items():
        sents = split_abstract_to_sentences(abstract, nlp)
        for i, sent in enumerate(sents):
            if sent in sent_to_context:
                continue
            parts = []
            for j in range(max(0, i - window), min(len(sents), i + window + 1)):
                parts.append(sents[j])
            sent_to_context[sent] = " ".join(parts)

    return sent_to_context


def fuzzy_match_sentence(sentence: str, sent_to_context: dict[str, str]) -> str | None:
    """Try exact match first, then substring match."""
    if sentence in sent_to_context:
        return sent_to_context[sentence]

    # Substring match: labeled sentence might be slightly different
    for stored_sent, context in sent_to_context.items():
        if sentence in stored_sent or stored_sent in sentence:
            return context

    # Token overlap match for near-matches
    sent_tokens = set(sentence.lower().split())
    best_overlap = 0
    best_context = None
    for stored_sent, context in sent_to_context.items():
        stored_tokens = set(stored_sent.lower().split())
        overlap = len(sent_tokens & stored_tokens) / max(len(sent_tokens), 1)
        if overlap > best_overlap and overlap > 0.85:
            best_overlap = overlap
            best_context = context

    return best_context


def add_context_to_dataset(
    data: list[dict],
    sent_to_context: dict[str, str],
) -> list[dict]:
    """Add context to each labeled sentence."""
    result = []
    matched = 0
    unmatched = 0

    for item in data:
        sentence = item["sentence"]
        context = fuzzy_match_sentence(sentence, sent_to_context)

        if context:
            result.append({
                "sentence": context,
                "label": item["label"],
            })
            matched += 1
        else:
            # Fall back to original sentence if no context found
            result.append({
                "sentence": sentence,
                "label": item["label"],
            })
            unmatched += 1

    print(f"  Matched: {matched}, Unmatched (using original): {unmatched}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Add context windows to labeled data")
    parser.add_argument("--window", type=int, default=1,
                        help="Number of sentences before/after to include")
    parser.add_argument("--corpus", type=str,
                        default="data/corpus/hfpef_corpus.csv")
    args = parser.parse_args()

    corpus_path = ROOT / args.corpus

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print(f"Loading abstracts from {corpus_path}...")
    abstracts = load_abstracts_from_corpus(corpus_path)
    print(f"  {len(abstracts)} abstracts loaded")

    print(f"Building sentence-to-context index (window={args.window})...")
    sent_to_context = build_sentence_to_context(abstracts, nlp, window=args.window)
    print(f"  {len(sent_to_context)} sentences indexed")

    datasets = [
        ("data/splits/hfpef_v3_train_augmented.json", "data/splits/hfpef_v3_train_context.json"),
        ("data/splits/hfpef_v3_eval.json", "data/splits/hfpef_v3_eval_context.json"),
    ]

    for input_rel, output_rel in datasets:
        input_path = ROOT / input_rel
        output_path = ROOT / output_rel

        if not input_path.exists():
            print(f"  Skipping {input_rel} (not found)")
            continue

        print(f"\nProcessing {input_rel}...")
        with open(input_path) as f:
            data = json.load(f)
        print(f"  {len(data)} samples")

        result = add_context_to_dataset(data, sent_to_context)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()

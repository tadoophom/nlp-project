"""Utility for labeling protein-disease relation sentences.

Supports:
1. Manual labeling via CLI
2. LLM pre-labeling (Claude/GPT) with manual verification
3. Export to JSON format for training
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

VALID_LABELS = ["positive", "negative", "no_association"]


def extract_sentences_from_corpus(corpus_path: Path) -> List[Dict[str, str]]:
    """Extract unique sentences from corpus CSV."""
    df = pd.read_csv(corpus_path)
    sentences = []
    seen = set()

    for _, row in df.iterrows():
        sent = row.get("evidence_sentence")
        if pd.isna(sent) or not sent.strip():
            continue
        if sent in seen:
            continue
        seen.add(sent)
        sentences.append({
            "sentence": sent.strip(),
            "protein": row.get("protein", ""),
            "pmid": row.get("pmid", ""),
            "current_label": row.get("relation", ""),
        })

    return sentences


def prelabel_with_llm(
    sentences: List[Dict[str, str]],
    provider: str = "anthropic",
    batch_size: int = 10,
) -> List[Dict[str, str]]:
    """Use LLM to pre-label sentences (requires API key in env)."""
    prompt_template = """Classify each sentence's protein-disease relationship:

Labels:
- positive: The sentence describes a biological association (biomarker, correlation, mechanism, therapeutic target)
- negative: The sentence explicitly negates or refutes an association
- no_association: The sentence mentions both but doesn't describe a meaningful relationship

Sentences:
{sentences}

Return JSON array with format: [{{"sentence": "...", "label": "positive|negative|no_association", "reason": "brief explanation"}}]
Only return the JSON, no other text."""

    results = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        numbered = "\n".join(f"{j+1}. {s['sentence']}" for j, s in enumerate(batch))
        prompt = prompt_template.format(sentences=numbered)

        try:
            if provider == "anthropic":
                import anthropic

                client = anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
            else:  # openai
                import openai

                client = openai.OpenAI()
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content

            # Parse JSON from response
            start = content.find("[")
            end = content.rfind("]") + 1
            parsed = json.loads(content[start:end])

            for j, item in enumerate(parsed):
                if j < len(batch):
                    batch[j]["llm_label"] = item.get("label", "")
                    batch[j]["llm_reason"] = item.get("reason", "")

        except Exception as e:
            print(f"LLM error on batch {i//batch_size}: {e}")
            for item in batch:
                item["llm_label"] = ""
                item["llm_reason"] = ""

        results.extend(batch)
        print(f"Processed {min(i + batch_size, len(sentences))}/{len(sentences)}")

    return results


def interactive_label(sentences: List[Dict[str, str]], output_path: Path) -> None:
    """Interactively label sentences via CLI."""
    labeled = []

    # Load existing if resuming
    if output_path.exists():
        with open(output_path) as f:
            labeled = json.load(f)
        print(f"Resuming from {len(labeled)} labeled sentences")

    labeled_sents = {item["sentence"] for item in labeled}
    remaining = [s for s in sentences if s["sentence"] not in labeled_sents]

    print(f"\n{len(remaining)} sentences remaining to label")
    print("Labels: [p]ositive, [n]egative, [x] no_association, [s]kip, [q]uit\n")

    for i, item in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]")
        print(f"Protein: {item.get('protein', 'N/A')}")
        print(f"PMID: {item.get('pmid', 'N/A')}")
        print(f"Current: {item.get('current_label', 'N/A')}")
        if item.get("llm_label"):
            print(f"LLM suggestion: {item['llm_label']} - {item.get('llm_reason', '')}")
        print(f"\nSentence: {item['sentence']}")

        while True:
            choice = input("\nLabel [p/n/x/s/q]: ").strip().lower()
            if choice == "p":
                label = "positive"
                break
            elif choice == "n":
                label = "negative"
                break
            elif choice == "x":
                label = "no_association"
                break
            elif choice == "s":
                label = None
                break
            elif choice == "q":
                save_labeled(labeled, output_path)
                print(f"Saved {len(labeled)} labels to {output_path}")
                return
            else:
                print("Invalid choice")

        if label:
            labeled.append({"sentence": item["sentence"], "label": label})

        # Auto-save every 10 labels
        if len(labeled) % 10 == 0:
            save_labeled(labeled, output_path)
            print(f"(Auto-saved {len(labeled)} labels)")

    save_labeled(labeled, output_path)
    print(f"\nDone! Saved {len(labeled)} labels to {output_path}")


def save_labeled(data: List[Dict], path: Path) -> None:
    """Save labeled data to JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def verify_llm_labels(sentences: List[Dict[str, str]], output_path: Path) -> None:
    """Quick verification mode for LLM pre-labeled sentences."""
    labeled = []

    if output_path.exists():
        with open(output_path) as f:
            labeled = json.load(f)

    labeled_sents = {item["sentence"] for item in labeled}
    remaining = [s for s in sentences if s["sentence"] not in labeled_sents and s.get("llm_label")]

    print(f"\n{len(remaining)} pre-labeled sentences to verify")
    print("Press Enter to accept LLM label, or type new label [p/n/x], [s]kip, [q]uit\n")

    for i, item in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}]")
        print(f"Sentence: {item['sentence']}")
        print(f"LLM: {item['llm_label']} - {item.get('llm_reason', '')}")

        choice = input("Accept or override [Enter/p/n/x/s/q]: ").strip().lower()

        if choice == "":
            label = item["llm_label"]
        elif choice == "p":
            label = "positive"
        elif choice == "n":
            label = "negative"
        elif choice == "x":
            label = "no_association"
        elif choice == "s":
            continue
        elif choice == "q":
            break
        else:
            continue

        labeled.append({"sentence": item["sentence"], "label": label})

        if len(labeled) % 10 == 0:
            save_labeled(labeled, output_path)

    save_labeled(labeled, output_path)
    print(f"\nSaved {len(labeled)} labels to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Label sentences for BERT training")
    parser.add_argument("--corpus", required=True, help="Path to corpus CSV")
    parser.add_argument("--output", required=True, help="Output JSON path for labels")
    parser.add_argument("--prelabel", choices=["anthropic", "openai"], help="Use LLM for pre-labeling")
    parser.add_argument("--verify", action="store_true", help="Verification mode for pre-labeled data")
    parser.add_argument("--limit", type=int, help="Limit number of sentences")
    args = parser.parse_args()

    sentences = extract_sentences_from_corpus(Path(args.corpus))
    print(f"Extracted {len(sentences)} unique sentences from corpus")

    if args.limit:
        sentences = sentences[: args.limit]

    if args.prelabel:
        print(f"Pre-labeling with {args.prelabel}...")
        sentences = prelabel_with_llm(sentences, provider=args.prelabel)

        # Save intermediate results
        temp_path = Path(args.output).with_suffix(".prelabeled.json")
        with open(temp_path, "w") as f:
            json.dump(sentences, f, indent=2)
        print(f"Pre-labeled data saved to {temp_path}")

    output_path = Path(args.output)

    if args.verify:
        # Load pre-labeled if exists
        temp_path = Path(args.output).with_suffix(".prelabeled.json")
        if temp_path.exists():
            with open(temp_path) as f:
                sentences = json.load(f)
        verify_llm_labels(sentences, output_path)
    else:
        interactive_label(sentences, output_path)


if __name__ == "__main__":
    main()

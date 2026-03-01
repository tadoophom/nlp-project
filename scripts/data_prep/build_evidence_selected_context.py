"""Build evidence-selected context variants from context-window datasets."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DISEASE_TERMS = [
    "hfpef",
    "heart failure",
    "preserved ejection fraction",
    "cardiomyopathy",
    "arrhythmia",
    "valvular",
    "ischemic",
    "stroke",
    "coronary",
]
POS_CUES = ["associated", "linked", "correlated", "predict", "elevated", "increase"]
NEG_CUES = ["no association", "not associated", "no correlation", "unrelated", "independent"]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def anchor_tokens(sentence: str) -> set[str]:
    toks = re.findall(r"[A-Za-z0-9\-]+", sentence)
    out = set()
    for t in toks:
        if len(t) > 2 and (t.isupper() or any(ch.isdigit() for ch in t) or "-" in t):
            out.add(t.lower())
    return out


def score_sentence(sent: str, anchors: set[str]) -> float:
    lower = sent.lower()
    score = 0.0
    score += sum(1.2 for term in DISEASE_TERMS if term in lower)
    score += sum(1.0 for tok in anchors if tok in lower)
    score += sum(0.8 for cue in POS_CUES if cue in lower)
    score += sum(1.0 for cue in NEG_CUES if cue in lower)
    return score


def select_context(target_sentence: str, context_text: str, top_k: int) -> str:
    sents = split_sentences(context_text)
    if len(sents) <= 1:
        return target_sentence

    anchors = anchor_tokens(target_sentence)
    scored = [(score_sentence(s, anchors), s) for s in sents]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [target_sentence]
    for _, sent in scored:
        if normalize(sent) == normalize(target_sentence):
            continue
        selected.append(sent)
        if len(selected) >= top_k + 1:
            break
    return " ".join(selected)


def transform(input_path: Path, output_path: Path, top_k: int, target_path: Path | None):
    with open(input_path) as f:
        rows = json.load(f)

    targets = None
    if target_path:
        with open(target_path) as f:
            target_rows = json.load(f)
        if len(target_rows) != len(rows):
            raise ValueError("Target dataset length does not match context dataset length.")
        targets = [row["sentence"] for row in target_rows]

    out = []
    for idx, row in enumerate(rows):
        target = targets[idx] if targets is not None else row.get("sentence", "")
        context = row["sentence"]
        selected = select_context(target, context, top_k=top_k)
        out.append({"sentence": selected, "label": row["label"]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {len(out)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build evidence-selected context dataset")
    parser.add_argument("--input", required=True, help="Input context JSON path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=2, help="Top extra context sentences")
    parser.add_argument(
        "--target-data",
        default="",
        help="Optional base sentence dataset aligned by index to context dataset",
    )
    args = parser.parse_args()

    target_path = Path(args.target_data) if args.target_data else None
    transform(Path(args.input), Path(args.output), args.top_k, target_path)


if __name__ == "__main__":
    main()

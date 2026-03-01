"""Build entity-marked relation datasets with simple protein/disease tagging."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DISEASE_PATTERNS = [
    r"heart failure with preserved ejection fraction",
    r"hfpef",
    r"heart failure",
    r"cardiomyopathy",
    r"arrhythmia",
    r"ischemic heart disease",
    r"coronary heart disease",
    r"valvular disease",
    r"stroke",
]

PROTEIN_TOKEN = re.compile(r"\b[A-Z][A-Z0-9\-]{1,14}\b")


def mark_first(patterns: list[str], text: str, start_tag: str, end_tag: str) -> tuple[str, bool]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            s, e = m.span()
            return text[:s] + start_tag + text[s:e] + end_tag + text[e:], True
    return text, False


def mark_protein(text: str) -> tuple[str, bool]:
    banned = {"HFPEF", "HFRF", "HF", "BP", "NYHA", "ECG", "LV", "RV"}
    for m in PROTEIN_TOKEN.finditer(text):
        token = m.group(0)
        if token in banned:
            continue
        s, e = m.span()
        return text[:s] + "[PROT]" + text[s:e] + "[/PROT]" + text[e:], True
    return text, False


def transform(sentence: str) -> tuple[str, bool, bool]:
    out, has_dis = mark_first(DISEASE_PATTERNS, sentence, "[DIS]", "[/DIS]")
    out, has_prot = mark_protein(out)
    return out, has_prot, has_dis


def convert(rows: list[dict], require_both_markers: bool) -> list[dict]:
    out = []
    for r in rows:
        sent = r["sentence"]
        marked, has_prot, has_dis = transform(sent)
        if require_both_markers and not (has_prot and has_dis):
            marked = sent
        out.append({"sentence": marked, "label": r["label"]})
    return out


def main():
    parser = argparse.ArgumentParser(description="Build entity-marked datasets")
    parser.add_argument("--train-in", required=True)
    parser.add_argument("--eval-in", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--eval-out", required=True)
    parser.add_argument("--require-both-markers", action="store_true")
    args = parser.parse_args()

    with open(args.train_in) as f:
        train_rows = json.load(f)
    with open(args.eval_in) as f:
        eval_rows = json.load(f)

    train_out_rows = convert(train_rows, args.require_both_markers)
    eval_out_rows = convert(eval_rows, args.require_both_markers)

    train_out = Path(args.train_out)
    train_out.parent.mkdir(parents=True, exist_ok=True)
    with open(train_out, "w") as f:
        json.dump(train_out_rows, f, indent=2)

    eval_out = Path(args.eval_out)
    eval_out.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_out, "w") as f:
        json.dump(eval_out_rows, f, indent=2)

    print("train_in", len(train_rows), "train_out", len(train_out_rows))
    print("eval_in", len(eval_rows), "eval_out", len(eval_out_rows))
    print("saved_train", train_out)
    print("saved_eval", eval_out)


if __name__ == "__main__":
    main()

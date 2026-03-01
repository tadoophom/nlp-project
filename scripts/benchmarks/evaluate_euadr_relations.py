#!/usr/bin/env python3

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.bert_classifier import PubMedBERTClassifier
from src.nlp_utils import load_pipeline


DEFAULT_JSONL = Path("data/benchmarks/euadr/euadr_bigbio_kb.train.jsonl")


def to_text(value) -> str:
    if isinstance(value, list):
        parts = [v.strip() for v in value if isinstance(v, str) and v.strip()]
        return " ".join(parts).strip()
    if isinstance(value, str):
        return value.strip()
    return ""


def passage_records(doc: dict) -> list[dict]:
    records = []
    for passage in doc.get("passages") or []:
        offsets = passage.get("offsets") or []
        start, end = (offsets[0] if offsets else (None, None))
        records.append(
            {
                "type": passage.get("type", ""),
                "start": start,
                "end": end,
                "text": to_text(passage.get("text")),
            }
        )
    return records


def passage_for_offset(passages: list[dict], offset: int | None) -> dict | None:
    if offset is None:
        return passages[0] if passages else None
    for passage in passages:
        start = passage.get("start")
        end = passage.get("end")
        if start is None or end is None:
            continue
        if start <= offset <= end:
            return passage
    return passages[0] if passages else None


def entity_view(entity: dict, passages: list[dict]) -> dict:
    offsets = entity.get("offsets") or []
    offset = offsets[0][0] if offsets and offsets[0] else None
    passage = passage_for_offset(passages, offset)
    passage_start = passage.get("start") if passage else None
    passage_text = passage.get("text") if passage else ""
    rel_offset = None
    if offset is not None and passage_start is not None:
        rel_offset = offset - passage_start
        if passage_text:
            rel_offset = max(0, min(rel_offset, len(passage_text) - 1))
    return {
        "id": entity.get("id", ""),
        "text": to_text(entity.get("text")),
        "type": entity.get("type", ""),
        "offset": offset,
        "passage": passage,
        "passage_start": passage_start,
        "rel_offset": rel_offset,
    }


def document_text(passages: list[dict]) -> str:
    parts = [p.get("text", "") for p in passages if p.get("text")]
    return " ".join(parts).strip()


def pick_sentence(text: str, ent1: str, ent2: str, nlp) -> str:
    if not text:
        return text
    ent1_l = ent1.lower()
    ent2_l = ent2.lower()
    doc = nlp(text)
    sentences = list(doc.sents)
    for sent in sentences:
        s = sent.text
        s_l = s.lower()
        if ent1_l in s_l and ent2_l in s_l:
            return s.strip()
    for sent in sentences:
        s = sent.text
        s_l = s.lower()
        if ent1_l in s_l or ent2_l in s_l:
            return s.strip()
    return sentences[0].text.strip() if sentences else text.strip()


def window_for_entity(entity: dict, window_size: int) -> str:
    passage = entity.get("passage") or {}
    text = passage.get("text") or ""
    rel_offset = entity.get("rel_offset")
    if not text or rel_offset is None:
        return text
    start = max(rel_offset - window_size, 0)
    end = min(rel_offset + window_size, len(text))
    return text[start:end].strip()


def relation_text(e1: dict, e2: dict, doc_text: str, nlp, window_size: int) -> str:
    p1 = e1.get("passage") or {}
    p2 = e2.get("passage") or {}
    same_passage = p1 and p2 and p1.get("start") == p2.get("start")
    if same_passage:
        passage_text = p1.get("text") or doc_text
        r1 = e1.get("rel_offset")
        r2 = e2.get("rel_offset")
        if passage_text and r1 is not None and r2 is not None:
            start = max(min(r1, r2) - window_size, 0)
            end = min(max(r1, r2) + window_size, len(passage_text))
            base = passage_text[start:end].strip()
        else:
            base = passage_text
    else:
        w1 = window_for_entity(e1, window_size)
        w2 = window_for_entity(e2, window_size)
        base = " ".join(part for part in (w1, w2) if part).strip() or doc_text
    sentence = pick_sentence(base, e1.get("text", ""), e2.get("text", ""), nlp)
    return sentence or base


def map_label(label: str, na_as_associated: bool) -> str | None:
    key = (label or "").strip().upper()
    if key == "PA":
        return "associated"
    if key == "NA":
        return "associated" if na_as_associated else "not_associated"
    if key == "SA":
        return "incidental"
    if key == "FA":
        return "not_associated"
    return None


def build_relations(jsonl_path: Path, na_as_associated: bool, nlp, window_size: int):
    relations = []
    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            passages = passage_records(doc)
            doc_text = document_text(passages)
            entities = {
                ent.get("id"): entity_view(ent, passages)
                for ent in (doc.get("entities") or [])
                if ent.get("id") is not None
            }
            for rel in doc.get("relations") or []:
                label = map_label(rel.get("type", ""), na_as_associated)
                if not label:
                    continue
                e1 = entities.get(rel.get("arg1_id"))
                e2 = entities.get(rel.get("arg2_id"))
                if not e1 or not e2:
                    continue
                sentence = relation_text(e1, e2, doc_text, nlp, window_size)
                relations.append(
                    {
                        "doc_id": doc.get("id", ""),
                        "rel_id": rel.get("id", ""),
                        "label": label,
                        "text": sentence,
                        "ent1": e1.get("text", ""),
                        "ent2": e2.get("text", ""),
                    }
                )
    return relations


def evaluate(relations: list[dict], model_path: str):
    classifier = PubMedBERTClassifier(model_path=model_path)
    true_labels = [r["label"] for r in relations]
    pred_labels = []
    for i, rel in enumerate(relations, start=1):
        if i % 50 == 0:
            print(f"  processed {i}/{len(relations)}")
        pred, _ = classifier.predict(rel["text"])
        pred_labels.append(pred)
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=0)
    print("\nAccuracy: {:.1%}".format(accuracy))
    print(report)
    print("Gold label counts:", dict(Counter(true_labels)))
    print("Pred label counts:", dict(Counter(pred_labels)))
    return accuracy, true_labels, pred_labels


def collapse_not_associated(labels: list[str]) -> list[str]:
    collapsed = []
    for label in labels:
        collapsed.append("associated" if label == "not_associated" else label)
    return collapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(DEFAULT_JSONL))
    parser.add_argument("--model-path", default="models/scibert-hfpef-v4/final")
    parser.add_argument("--na-as-associated", action="store_true", default=True)
    parser.add_argument("--na-as-not-associated", dest="na_as_associated", action="store_false")
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--collapse-not-associated", action="store_true", default=True)
    parser.add_argument("--no-collapse-not-associated", dest="collapse_not_associated", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"EU-ADR JSONL not found: {jsonl_path}")
        return 1
    print(f"Loading EU-ADR relations from {jsonl_path}")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    relations = build_relations(jsonl_path, args.na_as_associated, nlp, args.window_size)
    if not relations:
        print("No relations found.")
        return 1
    print(f"Evaluating {len(relations)} relations")
    accuracy, true_labels, pred_labels = evaluate(relations, args.model_path)
    collapsed_accuracy = None
    if args.collapse_not_associated:
        collapsed_pred = collapse_not_associated(pred_labels)
        collapsed_accuracy = accuracy_score(true_labels, collapsed_pred)
        collapsed_report = classification_report(true_labels, collapsed_pred, zero_division=0)
        print("\nAccuracy (collapse not_associated -> associated): {:.1%}".format(collapsed_accuracy))
        print(collapsed_report)
        print("Collapsed pred counts:", dict(Counter(collapsed_pred)))
    output_path = Path("data/benchmarks/euadr_relations_evaluation.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "jsonl_path": str(jsonl_path),
                "rows": len(relations),
                "accuracy": accuracy,
                "collapsed_accuracy": collapsed_accuracy,
                "gold_counts": dict(Counter(true_labels)),
                "pred_counts": dict(Counter(pred_labels)),
                "na_as_associated": args.na_as_associated,
                "window_size": args.window_size,
                "collapse_not_associated": args.collapse_not_associated,
            },
            handle,
            indent=2,
        )
    print(f"Saved metrics to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

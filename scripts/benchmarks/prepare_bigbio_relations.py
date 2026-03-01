#!/usr/bin/env python3

import argparse
import csv
import random
import sys
from pathlib import Path

from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.nlp_utils import load_pipeline


DATASET_CONFIGS = {
    "chemprot": {
        "repo": "bigbio/chemprot",
        "subset": "chemprot_bigbio_kb",
        "default_split": "test",
        "default_output": Path("data/benchmarks/chemprot/chemprot_relations.csv"),
        "default_negative_ratio": 1,
    },
    "ddi": {
        "repo": "bigbio/ddi_corpus",
        "subset": "ddi_corpus_bigbio_kb",
        "default_split": "test",
        "default_output": Path("data/benchmarks/ddi/ddi_relations.csv"),
        "default_negative_ratio": 2,
    },
}


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


def has_comention(text: str, ent1: str, ent2: str) -> bool:
    if not text or not ent1 or not ent2:
        return False
    t = text.lower()
    return ent1.lower() in t and ent2.lower() in t


def map_chemprot_label(label: str) -> str:
    key = (label or "").strip().lower()
    return "not_associated" if key == "cpr:0" else "associated"


def map_ddi_label(label: str) -> str:
    key = (label or "").strip().lower()
    if key in {"mechanism", "effect", "advise", "int"}:
        return "associated"
    return "associated"


def relation_pairs(relations: list[dict]) -> set[tuple[str, str]]:
    pairs = set()
    for rel in relations:
        a = rel.get("arg1_id")
        b = rel.get("arg2_id")
        if not a or not b:
            continue
        pair = tuple(sorted((a, b)))
        pairs.add(pair)
    return pairs


def all_entity_pairs(entity_ids: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for i, a in enumerate(entity_ids):
        for b in entity_ids[i + 1 :]:
            pairs.append((a, b))
    return pairs


def build_records(
    doc: dict,
    dataset: str,
    nlp,
    window_size: int,
    negative_ratio: int,
    rng,
    require_comention: bool,
) -> list[dict]:
    passages = passage_records(doc)
    doc_text = document_text(passages)
    entities = {
        ent.get("id"): entity_view(ent, passages)
        for ent in (doc.get("entities") or [])
        if ent.get("id") is not None
    }
    mapper = map_chemprot_label if dataset == "chemprot" else map_ddi_label
    positives = []
    for rel in doc.get("relations") or []:
        e1 = entities.get(rel.get("arg1_id"))
        e2 = entities.get(rel.get("arg2_id"))
        if not e1 or not e2:
            continue
        text = relation_text(e1, e2, doc_text, nlp, window_size)
        if require_comention and not has_comention(text, e1.get("text", ""), e2.get("text", "")):
            continue
        positives.append(
            {
                "doc_id": doc.get("id", ""),
                "rel_id": rel.get("id", ""),
                "label": mapper(rel.get("type", "")),
                "source_label": rel.get("type", ""),
                "text": text,
                "ent1": e1.get("text", ""),
                "ent2": e2.get("text", ""),
                "dataset": dataset,
            }
        )
    if dataset not in {"ddi", "chemprot"} or negative_ratio <= 0 or not positives:
        return positives
    rel_pairs = relation_pairs(doc.get("relations") or [])
    entity_ids = [eid for eid in entities.keys() if eid]
    candidate_pairs = [pair for pair in all_entity_pairs(entity_ids) if pair not in rel_pairs]
    if not candidate_pairs:
        return positives
    negatives_needed = min(len(candidate_pairs), negative_ratio * len(positives))
    sampled_pairs = rng.sample(candidate_pairs, negatives_needed)
    negatives = []
    for idx, (a, b) in enumerate(sampled_pairs, start=1):
        e1 = entities.get(a)
        e2 = entities.get(b)
        if not e1 or not e2:
            continue
        text = relation_text(e1, e2, doc_text, nlp, window_size)
        if require_comention and not has_comention(text, e1.get("text", ""), e2.get("text", "")):
            continue
        negatives.append(
            {
                "doc_id": doc.get("id", ""),
                "rel_id": f"neg-{idx}",
                "label": "not_associated",
                "source_label": "no_relation",
                "text": text,
                "ent1": e1.get("text", ""),
                "ent2": e2.get("text", ""),
                "dataset": dataset,
            }
        )
    return positives + negatives


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), required=True)
    parser.add_argument("--split")
    parser.add_argument("--output")
    parser.add_argument("--window-size", type=int, default=240)
    parser.add_argument("--negative-ratio", type=int)
    parser.add_argument("--require-comention", action="store_true", default=True)
    parser.add_argument("--allow-noncomention", dest="require_comention", action="store_false")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = DATASET_CONFIGS[args.dataset]
    split = args.split or cfg["default_split"]
    output_path = Path(args.output) if args.output else cfg["default_output"]
    negative_ratio = (
        args.negative_ratio if args.negative_ratio is not None else cfg.get("default_negative_ratio", 0)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.dataset} split={split} from {cfg['repo']}")
    dataset = load_dataset(cfg["repo"], cfg["subset"], trust_remote_code=True)
    if split not in dataset:
        print(f"Split not found: {split}")
        return 1
    docs = dataset[split]
    if args.limit:
        docs = docs.select(range(min(args.limit, len(docs))))
    print(f"Processing {len(docs)} documents")
    nlp = load_pipeline("en_core_web_sm", use_context=True)
    rng = random.Random(args.seed)
    rows = []
    for i, doc in enumerate(docs, start=1):
        if i % 100 == 0:
            print(f"  processed {i}/{len(docs)}")
        rows.extend(
            build_records(
                doc,
                dataset=args.dataset,
                nlp=nlp,
                window_size=args.window_size,
                negative_ratio=negative_ratio,
                rng=rng,
                require_comention=args.require_comention,
            )
        )
    if not rows:
        print("No relation rows created.")
        return 1
    print(f"Writing {len(rows)} rows to {output_path}")
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["doc_id", "rel_id", "label", "source_label", "text", "ent1", "ent2", "dataset"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

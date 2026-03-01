"""Benchmark research-inspired context-window methods on HFpEF eval data.

Methods implemented:
  M1 baseline_sentence
  M2 raw_context
  M3 sentence_context_fusion
  M4 position_aware_context
  M5 entity_aware_topk_context
  M6 two_pass_rerank
  M7 adaptive_context_budget
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pubmedbert_classifier import PubMedBERTClassifier, LABEL_NAMES

ROOT = Path(__file__).resolve().parent.parent.parent

DISEASE_TERMS = {
    "hfpef",
    "heart failure",
    "preserved ejection fraction",
    "cardiomyopathy",
    "arrhythmia",
    "valvular",
    "ischemic",
    "stroke",
    "coronary",
    "myocardial",
}

POSITIVE_CUES = {
    "associated", "association", "linked", "correlated", "predict", "predictive",
    "increase", "increased", "elevated", "upregulated", "related",
}
NEGATIVE_CUES = {
    "no association", "not associated", "unrelated", "independent", "no correlation",
    "not linked", "did not", "no effect", "failed",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "from", "by",
    "with", "as", "is", "are", "was", "were", "be", "been", "this", "that", "these",
    "those", "it", "its", "their", "our", "we", "they", "he", "she", "his", "her",
    "but", "if", "then", "than", "also", "may", "might", "can", "could", "would",
    "should", "into", "about", "between", "across", "during", "among",
}


@dataclass
class Example:
    sentence: str
    context: str
    label: str


def split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text.strip())
    return [c.strip() for c in chunks if c.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def extract_anchor_tokens(sentence: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9\-]+", sentence)
    out: set[str] = set()
    for tok in tokens:
        lower = tok.lower()
        if len(lower) <= 2:
            continue
        if lower in STOPWORDS:
            continue
        # Proxy for biomedical entity-like tokens.
        if tok.isupper() or any(ch.isdigit() for ch in tok) or "-" in tok:
            out.add(lower)
    # Fall back to content words if no entity-like token found.
    if not out:
        for tok in tokens:
            lower = tok.lower()
            if len(lower) > 4 and lower not in STOPWORDS:
                out.add(lower)
            if len(out) >= 4:
                break
    return out


def cue_score(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for cue in POSITIVE_CUES:
        if cue in lower:
            score += 0.8
    for cue in NEGATIVE_CUES:
        if cue in lower:
            score += 1.0
    return score


def sentence_entity_score(sentence: str, anchor_tokens: set[str]) -> float:
    lower = sentence.lower()
    score = 0.0
    disease_hits = sum(1 for term in DISEASE_TERMS if term in lower)
    score += 1.2 * disease_hits

    token_hits = 0
    for tok in anchor_tokens:
        if tok in lower:
            token_hits += 1
    score += 1.0 * token_hits
    score += cue_score(sentence)

    # Lightweight de-noising.
    if lower.startswith(("background", "methods", "objective", "conclusion")):
        score -= 0.5
    return score


def find_target_index(context_sents: list[str], target_sentence: str) -> int:
    target_norm = normalize(target_sentence)
    for idx, sent in enumerate(context_sents):
        sent_norm = normalize(sent)
        if target_norm == sent_norm:
            return idx
        if target_norm in sent_norm or sent_norm in target_norm:
            return idx
    # Fallback: highest token overlap.
    target_tokens = set(re.findall(r"[A-Za-z0-9\-]+", target_norm))
    best_idx = 0
    best = 0.0
    for idx, sent in enumerate(context_sents):
        sent_tokens = set(re.findall(r"[A-Za-z0-9\-]+", normalize(sent)))
        overlap = len(target_tokens & sent_tokens) / max(len(target_tokens), 1)
        if overlap > best:
            best = overlap
            best_idx = idx
    return best_idx


def build_position_aware_context(target_sentence: str, context_text: str) -> str:
    sents = split_sentences(context_text)
    if len(sents) <= 1:
        return target_sentence
    target_idx = find_target_index(sents, target_sentence)
    target = sents[target_idx]
    non_target = [s for i, s in enumerate(sents) if i != target_idx]
    # Lost-in-the-middle mitigation: keep target sentence at both edges.
    ordered = [target] + non_target + [target]
    return " ".join(ordered)


def build_entity_aware_context(target_sentence: str, context_text: str, top_k: int = 2) -> str:
    sents = split_sentences(context_text)
    if not sents:
        return target_sentence
    anchor_tokens = extract_anchor_tokens(target_sentence)
    scored = []
    for sent in sents:
        score = sentence_entity_score(sent, anchor_tokens)
        scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[str] = []
    selected.append(target_sentence)
    for _, sent in scored:
        if normalize(sent) == normalize(target_sentence):
            continue
        selected.append(sent)
        if len(selected) >= top_k + 1:
            break
    return " ".join(selected)


def build_adaptive_context(target_sentence: str, context_text: str) -> str:
    sents = split_sentences(context_text)
    if len(sents) <= 1:
        return target_sentence

    anchor_tokens = extract_anchor_tokens(target_sentence)
    scored = [(sentence_entity_score(sent, anchor_tokens), sent) for sent in sents]
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    if best_score < 1.5:
        return target_sentence

    # Keep only the strongest non-noisy evidence sentence.
    selected = [target_sentence]
    for score, sent in scored:
        if normalize(sent) == normalize(target_sentence):
            continue
        if score >= 1.5:
            selected.append(sent)
        if len(selected) == 3:
            break
    return " ".join(selected)


def probs_array(batch_results: list[dict]) -> np.ndarray:
    mat = []
    for row in batch_results:
        mat.append([row["probabilities"][label] for label in LABEL_NAMES])
    return np.asarray(mat, dtype=np.float64)


def predict_with_thresholds(prob_mat: np.ndarray, thresholds: dict[str, float]) -> list[str]:
    thr = np.asarray([thresholds[label] for label in LABEL_NAMES], dtype=np.float64)
    adjusted = prob_mat / thr
    pred_ids = adjusted.argmax(axis=1)
    return [LABEL_NAMES[idx] for idx in pred_ids]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
            }
            for label in LABEL_NAMES
        },
    }


def load_examples(sentence_path: Path, context_path: Path) -> list[Example]:
    with open(sentence_path) as f:
        sentence_data = json.load(f)
    with open(context_path) as f:
        context_data = json.load(f)

    if len(sentence_data) != len(context_data):
        raise ValueError("Sentence and context eval sets have mismatched lengths.")

    examples = []
    for base, ctx in zip(sentence_data, context_data):
        if base["label"] != ctx["label"]:
            raise ValueError("Label mismatch between sentence and context eval sets.")
        examples.append(Example(
            sentence=base["sentence"],
            context=ctx["sentence"],
            label=base["label"],
        ))
    return examples


def run(args):
    sentence_model = PubMedBERTClassifier(model_path=Path(args.sentence_model), max_length=args.max_length)
    context_model_path = Path(args.context_model) if args.context_model else None
    context_model = (
        PubMedBERTClassifier(model_path=context_model_path, max_length=args.max_length)
        if context_model_path and context_model_path.exists()
        else sentence_model
    )

    examples = load_examples(Path(args.eval_sentence), Path(args.eval_context))
    y_true = [x.label for x in examples]
    sentence_texts = [x.sentence for x in examples]
    raw_context_texts = [x.context for x in examples]
    pos_context_texts = [build_position_aware_context(x.sentence, x.context) for x in examples]
    ent_context_texts = [build_entity_aware_context(x.sentence, x.context, top_k=args.entity_top_k) for x in examples]
    adaptive_context_texts = [build_adaptive_context(x.sentence, x.context) for x in examples]

    print(f"Loaded {len(examples)} eval samples")
    print(f"Sentence model: {args.sentence_model}")
    if context_model is sentence_model:
        print("Context model: sentence model fallback")
    else:
        print(f"Context model: {args.context_model}")

    sent_results = sentence_model.classify_batch(sentence_texts, batch_size=args.batch_size)
    raw_ctx_results = context_model.classify_batch(raw_context_texts, batch_size=args.batch_size)
    pos_ctx_results = context_model.classify_batch(pos_context_texts, batch_size=args.batch_size)
    ent_ctx_results = context_model.classify_batch(ent_context_texts, batch_size=args.batch_size)
    adaptive_ctx_results = context_model.classify_batch(adaptive_context_texts, batch_size=args.batch_size)

    p_sent = probs_array(sent_results)
    p_raw = probs_array(raw_ctx_results)
    p_pos = probs_array(pos_ctx_results)
    p_ent = probs_array(ent_ctx_results)
    p_adaptive = probs_array(adaptive_ctx_results)

    thresholds = {
        "associated": args.threshold_associated,
        "not_associated": args.threshold_not_associated,
        "incidental": args.threshold_incidental,
    }

    method_probs: dict[str, np.ndarray] = {}
    method_preds: dict[str, list[str]] = {}

    # M1
    method_probs["method1_sentence"] = p_sent
    method_preds["method1_sentence"] = predict_with_thresholds(p_sent, thresholds)

    # M2
    method_probs["method2_raw_context"] = p_raw
    method_preds["method2_raw_context"] = predict_with_thresholds(p_raw, thresholds)

    # M3
    p_m3 = (1.0 - args.alpha_context) * p_sent + args.alpha_context * p_raw
    method_probs["method3_fusion"] = p_m3
    method_preds["method3_fusion"] = predict_with_thresholds(p_m3, thresholds)

    # M4
    p_m4 = (1.0 - args.alpha_context) * p_sent + args.alpha_context * p_pos
    method_probs["method4_position_aware"] = p_m4
    method_preds["method4_position_aware"] = predict_with_thresholds(p_m4, thresholds)

    # M5
    p_m5 = (1.0 - args.alpha_context) * p_sent + args.alpha_context * p_ent
    method_probs["method5_entity_aware"] = p_m5
    method_preds["method5_entity_aware"] = predict_with_thresholds(p_m5, thresholds)

    # M6: two-pass rerank on uncertain examples only.
    p_m6 = p_sent.copy()
    conf = p_sent.max(axis=1)
    top2 = np.sort(p_sent, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    uncertain = (conf < args.uncertain_conf) | (margin < args.uncertain_margin)
    p_m6[uncertain] = (
        (1.0 - args.rerank_alpha_context) * p_sent[uncertain]
        + args.rerank_alpha_context * p_ent[uncertain]
    )
    method_probs["method6_two_pass_rerank"] = p_m6
    method_preds["method6_two_pass_rerank"] = predict_with_thresholds(p_m6, thresholds)

    # M7: adaptive context budget + uncertainty-aware fusion weight.
    p_m7 = np.zeros_like(p_sent)
    for idx in range(len(examples)):
        c = conf[idx]
        if c < args.uncertain_conf:
            alpha = args.adaptive_alpha_high
        elif c < args.uncertain_conf + 0.1:
            alpha = args.adaptive_alpha_mid
        else:
            alpha = args.adaptive_alpha_low
        p_m7[idx] = (1.0 - alpha) * p_sent[idx] + alpha * p_adaptive[idx]
    method_probs["method7_adaptive_budget"] = p_m7
    method_preds["method7_adaptive_budget"] = predict_with_thresholds(p_m7, thresholds)

    metrics = {}
    for name, preds in method_preds.items():
        metrics[name] = compute_metrics(y_true, preds)

    # Summarize.
    ranking = sorted(
        (
            (name, m["accuracy"], m["macro_f1"], m["weighted_f1"])
            for name, m in metrics.items()
        ),
        key=lambda x: (x[2], x[1]),
        reverse=True,
    )
    print("\nMethod ranking (by macro F1):")
    for name, acc, macro, weighted in ranking:
        print(f"  {name:28s} acc={acc:.4f} macro_f1={macro:.4f} weighted_f1={weighted:.4f}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.output_metrics) if args.output_metrics else out_dir / f"context_window_research_methods_{stamp}.json"
    preds_path = Path(args.output_predictions) if args.output_predictions else out_dir / f"context_window_research_predictions_{stamp}.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    preds_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": stamp,
        "eval_sentence": args.eval_sentence,
        "eval_context": args.eval_context,
        "sentence_model": args.sentence_model,
        "context_model": str(context_model_path) if context_model_path else None,
        "used_context_model": context_model is not sentence_model,
        "params": {
            "alpha_context": args.alpha_context,
            "entity_top_k": args.entity_top_k,
            "thresholds": thresholds,
            "uncertain_conf": args.uncertain_conf,
            "uncertain_margin": args.uncertain_margin,
            "rerank_alpha_context": args.rerank_alpha_context,
            "adaptive_alpha_high": args.adaptive_alpha_high,
            "adaptive_alpha_mid": args.adaptive_alpha_mid,
            "adaptive_alpha_low": args.adaptive_alpha_low,
        },
        "metrics": metrics,
        "ranking_by_macro_f1": [
            {
                "method": name,
                "accuracy": acc,
                "macro_f1": macro,
                "weighted_f1": weighted,
            }
            for name, acc, macro, weighted in ranking
        ],
    }
    with open(metrics_path, "w") as f:
        json.dump(payload, f, indent=2)

    with open(preds_path, "w") as f:
        for idx, ex in enumerate(examples):
            row = {
                "idx": idx,
                "label": ex.label,
                "sentence": ex.sentence,
                "context": ex.context,
                "predictions": {name: method_preds[name][idx] for name in method_preds},
            }
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved metrics to: {metrics_path}")
    print(f"Saved predictions to: {preds_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate context-window research methods")
    parser.add_argument("--eval-sentence", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--eval-context", default="data/splits/hfpef_v3_eval_context.json")
    parser.add_argument("--sentence-model", default="models/hfpef_v3_improved/pubmedbert_focal/final")
    parser.add_argument("--context-model", default="")
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--alpha-context", type=float, default=0.3)
    parser.add_argument("--entity-top-k", type=int, default=2)

    parser.add_argument("--threshold-associated", type=float, default=1.0)
    parser.add_argument("--threshold-not-associated", type=float, default=1.0)
    parser.add_argument("--threshold-incidental", type=float, default=1.0)

    parser.add_argument("--uncertain-conf", type=float, default=0.70)
    parser.add_argument("--uncertain-margin", type=float, default=0.15)
    parser.add_argument("--rerank-alpha-context", type=float, default=0.45)

    parser.add_argument("--adaptive-alpha-high", type=float, default=0.50)
    parser.add_argument("--adaptive-alpha-mid", type=float, default=0.35)
    parser.add_argument("--adaptive-alpha-low", type=float, default=0.20)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output-metrics", default="")
    parser.add_argument("--output-predictions", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

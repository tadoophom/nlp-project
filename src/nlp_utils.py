"""NLP utilities for keyword polarity detection."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
import re
import warnings

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher

_bert_classifier = None

def _imp(name):
    try:
        return __import__(name)
    except ImportError:
        return None

medspacy = _imp("medspacy")

CUSTOM_NEG_TRIGGERS = set()

def set_custom_negation_triggers(triggers: List[str]) -> None:
    global CUSTOM_NEG_TRIGGERS
    CUSTOM_NEG_TRIGGERS = {t.strip().lower() for t in triggers if t and t.strip()}

try:
    from spacy import displacy
except ImportError:
    displacy = None


@lru_cache(maxsize=8)
def load_pipeline(name: str, gpu: bool = False, use_context: bool = True) -> Language:
    """Load a spaCy pipeline, falling back to blank English if unavailable."""
    if gpu:
        spacy.require_gpu()
    try:
        nlp = spacy.load(name, disable=["ner"])
    except OSError:
        warnings.warn(
            f"Model '{name}' not found; falling back to lightweight 'blank' English.",
            RuntimeWarning,
        )
        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
    if use_context and medspacy and "context" not in nlp.pipe_names:
        nlp.add_pipe("medspacy_context", config={"rules": "default"}, last=True)
    return nlp


def _is_neg(span) -> bool:
    """Determine whether a token/phrase is negated."""
    if hasattr(span._, "is_negated"):
        return bool(span._.is_negated)

    tok = span.root

    def neg(t):
        return (
            t.dep_ == "neg"
            or t.morph.get("Polarity") == ["Neg"]
            or t.morph.get("PronType") == ["Neg"]
        )

    if neg(tok):
        return True
    for ancestor in getattr(tok, "ancestors", []):
        if neg(ancestor) or any(neg(child) for child in getattr(ancestor, "children", [])):
            return True
    if any(neg(child) for child in getattr(tok, "children", [])):
        return True

    sent = span.sent if hasattr(span, "sent") else None
    if sent is not None:
        left_start = max(sent.start, span.start - 6)
        window_tokens = [t.text.lower() for t in span.doc[left_start:span.start]]
        window_text = " ".join(window_tokens)
        lex_triggers = {"no", "denies", "denied", "without", "lacks", "lack", "negative for", "absence of"} | CUSTOM_NEG_TRIGGERS
        trigger_indices = [i for i, tok in enumerate(window_tokens) if tok in lex_triggers]
        phrase_trigger_set = set(["negative for", "absence of"]) | CUSTOM_NEG_TRIGGERS
        phrase_trigger = any(phr in window_text for phr in phrase_trigger_set)
        if trigger_indices or phrase_trigger:
            last_trigger_idx = trigger_indices[-1] if trigger_indices else 0
            blockers = {"but", "however", "yet", "although", "though", "except", "has", "have", "had", "is", "are", "was", "were"}
            if any(tok in blockers for tok in window_tokens[last_trigger_idx + 1 :]):
                return False
            if (len(window_tokens) - 1) - last_trigger_idx <= 3 or phrase_trigger:
                return True
    return False


def classify_span(span):
    """Return Associated / Not_Associated / Incidental for a span."""
    if hasattr(span._, "is_negated") and span._.is_negated:
        return "Not_Associated"
    if (
        hasattr(span._, "is_uncertain") and span._.is_uncertain
    ) or (
        hasattr(span._, "is_hypothetical") and span._.is_hypothetical
    ):
        return "Incidental"
    if _is_neg(span):
        return "Not_Associated"
    return "Associated"

def _confidence_for(span) -> float:
    """Confidence score in [0.5, 1.0] based on available signals."""
    base = 0.75
    if hasattr(span._, "is_negated"):
        base += 0.1
        if span._.is_negated:
            base -= 0.2
    sent = getattr(span, "sent", None)
    if sent is not None:
        left = span.start - max(sent.start, span.start - 4)
        base += max(0, 0.05 * (3 - left))
    return float(max(0.5, min(1.0, base)))

def extract(text: str, terms: List[str], nlp: Language, **kwargs) -> List[Dict[str, Any]]:
    """Extract keyword occurrences and contextual metadata."""
    doc = nlp(text)
    normalized_terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not normalized_terms:
        return []

    single_terms = {t for t in normalized_terms if " " not in t}
    phrase_terms = [t for t in normalized_terms if " " in t]

    hits: List[Dict[str, Any]] = []
    seen = set()

    if phrase_terms:
        pm = PhraseMatcher(nlp.vocab, attr="LOWER")
        pm.add("KW", [nlp.make_doc(t) for t in phrase_terms])
        for match_id, start, end in pm(doc):
            span = doc[start:end]
            sent = span.sent
            sent_index = 0
            for i, s in enumerate(doc.sents):
                if span.start >= s.start and span.start < s.end:
                    sent_index = i
                    break
            keyword = span.text.lower()
            token_index = span.start
            key = (sent_index, token_index, keyword)
            if key in seen:
                continue
            seen.add(key)
            rec = {
                "keyword": keyword,
                "sentence": sent.text.strip(),
                "sent_index": sent_index,
                "token_index": token_index,
                "pos": getattr(span.root, "pos_", ""),
                "dep": getattr(span.root, "dep_", ""),
                "classification": classify_span(span),
                "confidence": _confidence_for(span),
            }
            model_name = kwargs.get("model_name")
            if model_name:
                rec["model"] = str(model_name)
            hits.append(rec)

    if single_terms:
        for sent_index, sent in enumerate(doc.sents):
            for token in sent:
                token_text_lower = token.text.lower()
                lemma_lower = token.lemma_.lower() if getattr(token, "lemma_", None) else token_text_lower
                if lemma_lower in single_terms or token_text_lower in single_terms:
                    span = doc[token.i : token.i + 1]
                    keyword = lemma_lower if lemma_lower in single_terms else token_text_lower
                    token_index = token.i
                    key = (sent_index, token_index, keyword)
                    if key in seen:
                        continue
                    seen.add(key)
                    rec = {
                        "keyword": keyword,
                        "sentence": sent.text.strip(),
                        "sent_index": sent_index,
                        "token_index": token_index,
                        "pos": getattr(token, "pos_", ""),
                        "dep": getattr(token, "dep_", ""),
                        "classification": classify_span(span),
                        "confidence": _confidence_for(span),
                    }
                    model_name = kwargs.get("model_name")
                    if model_name:
                        rec["model"] = str(model_name)
                    hits.append(rec)

    return hits


def render_dependency_svg(sentence: str, nlp: Language) -> str:
    """Render dependency tree as SVG."""
    if displacy is None:
        return f"<pre>{sentence}</pre>"
    try:
        doc = nlp(sentence)
        svg = displacy.render(doc, style="dep", options={"distance": 110})
    except Exception:
        return f"<pre>{sentence}</pre>"

    def _add_title(match: re.Match) -> str:
        idx = int(match.group(1))
        token = doc[idx]
        return match.group(0) + f"<title>{token.lemma_}|{token.pos_}|{token.dep_}</title>"

    return re.sub(r"<g class='token' id='token-(\d+)'>", _add_title, svg)


def detect_sections(text: str) -> List[Dict[str, Any]]:
    """Detect common clinical sections using regex. Returns list of {start, title}."""
    patterns = [
        ("Chief Complaint", r"^(?:cc|chief complaint)\s*[:\-]", re.I | re.M),
        ("HPI", r"^(?:hpi|history of present illness)\s*[:\-]", re.I | re.M),
        ("ROS", r"^(?:ros|review of systems)\s*[:\-]", re.I | re.M),
        ("PMH", r"^(?:pmh|past medical history)\s*[:\-]", re.I | re.M),
        ("Medications", r"^(?:medications?)\s*[:\-]", re.I | re.M),
        ("Allergies", r"^(?:allerg(?:y|ies))\s*[:\-]", re.I | re.M),
        ("Exam", r"^(?:physical exam|exam)\s*[:\-]", re.I | re.M),
        ("Assessment", r"^(?:assessment)\s*[:\-]", re.I | re.M),
        ("Plan", r"^(?:plan)\s*[:\-]", re.I | re.M),
    ]
    hits: List[Dict[str, Any]] = []
    for title, pat, flags in patterns:
        for m in re.finditer(pat, text, flags):
            hits.append({"start": m.start(), "title": title})
    hits.sort(key=lambda d: d["start"])
    return hits


def scrub_phi(text: str) -> str:
    """Redact likely PHI: emails, phones, dates, MRNs, and names."""
    t = text
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    t = re.sub(r"(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", "[PHONE]", t)
    t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", t)
    t = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", "[DATE]", t, flags=re.I)
    t = re.sub(r"\b(?:MRN|ID)[:#\s]*\d{5,}\b", "[ID]", t, flags=re.I)
    t = re.sub(r"\b([A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b", "[NAME]", t)
    return t


def load_bert_classifier(model_path: Optional[str] = None) -> "PubMedBERTClassifier":
    """Load or return cached BERT classifier."""
    global _bert_classifier
    if _bert_classifier is None:
        from .bert_classifier import PubMedBERTClassifier
        _bert_classifier = PubMedBERTClassifier(model_path=model_path)
    return _bert_classifier


def classify_with_bert(sentence: str, model_path: Optional[str] = None) -> Tuple[str, float]:
    """Classify a sentence using PubMedBERT. Returns (label, confidence)."""
    classifier = load_bert_classifier(model_path)
    label, conf = classifier.predict(sentence)
    # Map BERT labels to existing pipeline labels
    label_map = {
        "associated": "Associated",
        "not_associated": "Not_Associated",
        "incidental": "Incidental",
        "positive": "Associated",
        "negative": "Not_Associated",
        "no_association": "Incidental",
    }
    return label_map.get(label, "Incidental"), conf


def classify_span_hybrid(
    span,
    sentence: str,
    model_path: Optional[str] = None,
    bert_threshold: float = 0.7,
) -> Tuple[str, float, str]:
    """Hybrid classification using rule-based + BERT.
    
    Returns (classification, confidence, method).
    Uses BERT when confidence is high, falls back to rules otherwise.
    """
    bert_label, bert_conf = classify_with_bert(sentence, model_path)
    
    if bert_conf >= bert_threshold:
        return bert_label, bert_conf, "bert"
    
    # Fall back to rule-based
    rule_label = classify_span(span)
    rule_conf = _confidence_for(span)
    return rule_label, rule_conf, "rule"

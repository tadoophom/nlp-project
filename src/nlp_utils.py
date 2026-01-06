"""
nlp_utils.py – shared NLP utilities for the clinical keyword polarity suite.

This module centralises loading of spaCy pipelines, negation detection,
keyword extraction and dependency tree rendering. By separating these
utilities from the Streamlit application, we enable headless API
services and reuse in CLI contexts without pulling in Streamlit as a
dependency. Functions in this module are pure and do not depend on
global state outside of the loaded spaCy pipelines.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Dict, Any
import re
import warnings

import spacy
from spacy.language import Language
from spacy.matcher import PhraseMatcher

# Optional dependencies
def _imp(name):
    try:
        return __import__(name)
    except ModuleNotFoundError:
        return None

medspacy = _imp("medspacy")

# Customisable negation triggers (extended via UI)
CUSTOM_NEG_TRIGGERS = set()

def set_custom_negation_triggers(triggers: List[str]) -> None:
    """Set additional lower-cased negation cue words/phrases used by heuristics."""
    global CUSTOM_NEG_TRIGGERS
    CUSTOM_NEG_TRIGGERS = {t.strip().lower() for t in triggers if t and t.strip()}

try:
    # Lazy import for dependency tree rendering
    from spacy import displacy  # type: ignore
except Exception:
    displacy = None  # type: ignore


@lru_cache(maxsize=8)
def load_pipeline(name: str, gpu: bool = False, use_context: bool = True) -> Language:
    """
    Load a spaCy pipeline with optional GPU and medspaCy context integration.
    Falls back to a lightweight English pipeline with a sentencizer when
    the requested model is unavailable.
    """
    if gpu:
        try:
            spacy.require_gpu()
        except Exception:
            pass
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
        try:
            nlp.add_pipe("medspacy_context", config={"rules": "default"}, last=True)  # type: ignore
        except Exception as exc:
            warnings.warn(f"Failed to add medspaCy context: {exc}")
    return nlp


def _is_neg(span) -> bool:
    """
    Determine whether a token/phrase is negated.

    Uses medspaCy when available. Otherwise, applies dependency/morph features
    and lexical-window heuristics for common clinical negations (e.g., "no",
    "denies", "without", "absence of").
    """
    if hasattr(span._, "is_negated"):
        return bool(span._.is_negated)

    tok = span.root

    def neg(t):
        return (
            t.dep_ == "neg"
            or t.morph.get("Polarity") == ["Neg"]
            or t.morph.get("PronType") == ["Neg"]
        )

    # Dependency/morph based
    if neg(tok):
        return True
    for ancestor in getattr(tok, "ancestors", []):
        if neg(ancestor) or any(neg(child) for child in getattr(ancestor, "children", [])):
            return True
    if any(neg(child) for child in getattr(tok, "children", [])):
        return True

    # Lexical-window heuristics within the sentence (5 tokens to the left)
    sent = span.sent if hasattr(span, "sent") else None
    if sent is not None:
        left_start = max(sent.start, span.start - 6)
        window_tokens = [t.text.lower() for t in span.doc[left_start:span.start]]
        window_text = " ".join(window_tokens)
        lex_triggers = {"no", "denies", "denied", "without", "lacks", "lack", "negative for", "absence of"} | CUSTOM_NEG_TRIGGERS
        # Find index of latest trigger token (single-token triggers only) within the window
        trigger_indices = [i for i, tok in enumerate(window_tokens) if tok in lex_triggers]
        phrase_trigger_set = set(["negative for", "absence of"]) | CUSTOM_NEG_TRIGGERS
        phrase_trigger = any(phr in window_text for phr in phrase_trigger_set)
        if trigger_indices or phrase_trigger:
            last_trigger_idx = trigger_indices[-1] if trigger_indices else 0
            # Blockers that break negation scope (coordination or positive assertions)
            blockers = {"but", "however", "yet", "although", "though", "except", "has", "have", "had", "is", "are", "was", "were"}
            # If any blocker appears after the trigger, do not apply negation
            if any(tok in blockers for tok in window_tokens[last_trigger_idx + 1 :]):
                return False
            # Require proximity: trigger within 3 tokens of the target
            if (len(window_tokens) - 1) - last_trigger_idx <= 3 or phrase_trigger:
                return True
    return False


def classify_span(span):
    """Return Positive / Negative / Neutral using ConText if present, else heuristics."""
    if hasattr(span._, "is_negated") and span._.is_negated:
        return "Negative"
    if (
        hasattr(span._, "is_uncertain") and span._.is_uncertain
    ) or (
        hasattr(span._, "is_hypothetical") and span._.is_hypothetical
    ):
        return "Neutral"
    # Heuristic negation
    if _is_neg(span):
        return "Negative"
    return "Positive"

def _confidence_for(span) -> float:
    """Naive confidence score in [0.5, 1.0] based on available signals."""
    # Start moderate, boost when ConText is present, penalize when negated heuristically
    base = 0.75
    if hasattr(span._, "is_negated"):
        base += 0.1
        if span._.is_negated:
            base -= 0.2
    # proximity to negation cue
    sent = getattr(span, "sent", None)
    if sent is not None:
        left = span.start - max(sent.start, span.start - 4)
        base += max(0, 0.05 * (3 - left))
    return float(max(0.5, min(1.0, base)))

def extract(text: str, terms: List[str], nlp: Language, **kwargs) -> List[Dict[str, Any]]:
    """
    Extract keyword occurrences (single- and multi-word) and contextual metadata.

    Returns lowercase keys suitable for API/tests. When ``model_name`` is passed
    in ``kwargs``, a lowercase ``model`` field is included for UI use.
    """
    doc = nlp(text)
    normalized_terms = [t.strip().lower() for t in terms if t and t.strip()]
    if not normalized_terms:
        return []

    single_terms = {t for t in normalized_terms if " " not in t}
    phrase_terms = [t for t in normalized_terms if " " in t]

    hits: List[Dict[str, Any]] = []
    seen = set()  # dedupe by (sent_index, token_index, keyword)

    # Phrase matches (multi-word)
    if phrase_terms:
        pm = PhraseMatcher(nlp.vocab, attr="LOWER")
        pm.add("KW", [nlp.make_doc(t) for t in phrase_terms])
        for match_id, start, end in pm(doc):
            span = doc[start:end]
            sent = span.sent
            # compute sentence index
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

    # Single-token matches (lemma-based)
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
    """
    Render dependency tree as SVG. If unavailable (no parser), degrade gracefully
    by returning a simple HTML-wrapped sentence with token tooltips omitted.
    """
    if displacy is None:
        return f"<pre>{sentence}</pre>"
    try:
        doc = nlp(sentence)
        svg = displacy.render(doc, style="dep", options={"distance": 110})
    except Exception:
        # Parser not available or rendering failed
        return f"<pre>{sentence}</pre>"

    def _add_title(match: re.Match) -> str:
        idx = int(match.group(1))
        token = doc[idx]
        return match.group(0) + f"<title>{token.lemma_}|{token.pos_}|{token.dep_}</title>"

    return re.sub(r"<g class='token' id='token-(\d+)'>", _add_title, svg)


def detect_sections(text: str) -> List[Dict[str, Any]]:
    """Detect common clinical sections using lightweight regex heuristics.
    Returns a list of dicts: {start, title} sorted by start.
    """
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
    """Redact likely PHI: emails, phones, dates, MRNs, and person-like names.
    Conservative to avoid over-scrubbing. Pure regex approach.
    """
    t = text
    # Emails
    t = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", t)
    # Phones
    t = re.sub(r"(?:(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})", "[PHONE]", t)
    # Dates (simple)
    t = re.sub(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE]", t)
    t = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{2,4}\b", "[DATE]", t, flags=re.I)
    # MRN/ID-like numbers
    t = re.sub(r"\b(?:MRN|ID)[:#\s]*\d{5,}\b", "[ID]", t, flags=re.I)
    # Person-like names (two+ Capitalized tokens)
    t = re.sub(r"\b([A-Z][a-z]+\s+){1,3}[A-Z][a-z]+\b", "[NAME]", t)
    return t

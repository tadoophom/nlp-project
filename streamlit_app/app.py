"""Clinical Keyword Polarity Analyzer"""

from __future__ import annotations
import sys
import re
import json
import zipfile
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
import plotly.express as px
from spacy.language import Language

from src.nlp_utils import (
    load_pipeline,
    extract,
    render_dependency_svg,
    set_custom_negation_triggers,
    detect_sections,
)
from src.database import init_db, insert_feedback, get_feedback_summary

MODELS = [
    ("en_core_web_sm", "spaCy small"),
    ("en_core_web_md", "spaCy medium"),
    ("en_core_sci_md", "SciSpaCy clinical"),
]

CODE_MAP = {
    "procalcitonin": ("SNOMED", "704427003"),
    "pneumonia": ("SNOMED", "233604007"),
    "sepsis": ("SNOMED", "91302008"),
    "fever": ("SNOMED", "386661006"),
    "troponin": ("SNOMED", "105000003"),
    "chest pain": ("SNOMED", "29857009"),
    "aspirin": ("RxNorm", "1191"),
    "lactate": ("SNOMED", "83036002"),
    "hypotension": ("SNOMED", "45007003"),
    "antibiotics": ("SNOMED", "255631004"),
}

PRESETS = {
    "Pneumonia management": {
        "text": """A 58-year-old male presents with productive cough, fever, and dyspnea for 3 days. Chest X-ray shows right lower lobe consolidation. Procalcitonin is elevated at 2.4 ng/mL. Blood cultures are pending. Patient denies recent hospitalization or antibiotic use. No evidence of pleural effusion. Started on ceftriaxone and azithromycin for community-acquired pneumonia. Will reassess in 48-72 hours for clinical response.""",
        "keywords": ["pneumonia", "procalcitonin", "ceftriaxone", "fever", "consolidation"],
    },
    "Acute coronary syndrome": {
        "text": """67-year-old female with hypertension and diabetes presents with crushing substernal chest pain radiating to left arm for 2 hours. ECG shows ST-elevation in leads V2-V4. Troponin I elevated at 4.2 ng/mL. Patient denies cocaine use. No contraindications to thrombolysis. Aspirin and heparin administered. Cardiology consulted for emergent catheterization. No evidence of heart failure on exam.""",
        "keywords": ["troponin", "st-elevation", "chest pain", "aspirin", "catheterization"],
    },
    "Sepsis evaluation": {
        "text": """72-year-old nursing home resident with altered mental status, fever 39.2°C, and hypotension. Lactate elevated at 4.1 mmol/L. Urinalysis shows pyuria. Patient meets sepsis criteria. No meningismus on exam. Blood and urine cultures obtained. Broad-spectrum antibiotics initiated within 1 hour. Fluid resuscitation ongoing. Vasopressors not yet required.""",
        "keywords": ["sepsis", "lactate", "hypotension", "antibiotics", "cultures"],
    },
    "Stroke assessment": {
        "text": """78-year-old right-handed male with sudden onset left-sided weakness and slurred speech. Last known well 2 hours ago. NIHSS score is 14. CT head shows no hemorrhage. CTA demonstrates right MCA occlusion. Patient is within thrombolysis window. No contraindications identified. tPA administered. Transferred to interventional suite for thrombectomy consideration.""",
        "keywords": ["nihss", "tpa", "thrombectomy", "mca occlusion", "hemorrhage"],
    },
}

init_db()

st.set_page_config(page_title="Clinical Keyword Polarity", layout="wide")
st.title("Clinical Keyword Polarity Analyzer")
st.caption("Detect positive, negative, and neutral mentions of clinical terms")

# Session state
if "_doc_count" not in st.session_state:
    st.session_state._doc_count = 0
if "_hit_history" not in st.session_state:
    st.session_state._hit_history = []
if "_kw_colors" not in st.session_state:
    st.session_state._kw_colors = {}

# URL params
_qp = st.query_params if hasattr(st, "query_params") else {}


@st.cache_resource
def get_pipeline(name: str) -> Language:
    return load_pipeline(name, gpu=False)


# Preset loader
st.subheader("Quick Start")
col1, col2 = st.columns([3, 1])
with col1:
    preset = st.selectbox("Load a preset case", ["(none)"] + list(PRESETS.keys()))
with col2:
    if st.button("Load", use_container_width=True) and preset != "(none)":
        st.session_state.text = PRESETS[preset]["text"]
        st.session_state.keywords = ", ".join(PRESETS[preset]["keywords"])
        st.rerun()

# Text input
st.subheader("Clinical Text")
text = st.text_area(
    "Paste clinical note",
    value=st.session_state.get("text", ""),
    height=180,
    placeholder="Paste clinical note here...",
    label_visibility="collapsed",
)

# Keywords
st.subheader("Keywords")
url_kw = _qp.get("k", "")
default_kw = st.session_state.get("keywords", url_kw)
keywords_input = st.text_input(
    "Keywords to analyze (comma-separated)",
    value=default_kw,
    placeholder="fever, cough, pneumonia",
    label_visibility="collapsed",
)
keywords = [k.strip().lower() for k in keywords_input.split(",") if k.strip()]

# Settings
st.subheader("Settings")
col1, col2 = st.columns(2)

with col1:
    available_models = [m for m, _ in MODELS]
    url_model = _qp.get("m", "").split(",") if _qp.get("m") else []
    default_models = [m for m in url_model if m in available_models] or ["en_core_web_sm"]
    model_choices = st.multiselect(
        "Models",
        available_models,
        default=default_models,
        format_func=lambda x: dict(MODELS).get(x, x),
        max_selections=3,
    )

with col2:
    c1, c2 = st.columns(2)
    with c1:
        expand_synonyms = st.checkbox("Expand synonyms", help="WordNet expansion")
        show_codes = st.checkbox("Clinical codes", value=True)
    with c2:
        show_sections = st.checkbox("Detect sections", value=True)
        show_temporal = st.checkbox("Temporal extraction", value=True)

# Advanced options
with st.expander("Advanced options"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Keyword colors**")
        palette = ["#EF476F", "#06D6A0", "#118AB2", "#FFD166", "#8338EC"]
        for i, kw in enumerate(keywords[:5]):
            st.session_state._kw_colors.setdefault(kw, palette[i % len(palette)])
            st.session_state._kw_colors[kw] = st.color_picker(
                kw, st.session_state._kw_colors.get(kw, palette[0]), key=f"color_{kw}"
            )
    
    with col2:
        st.markdown("**Custom negation triggers**")
        custom_neg = st.text_area(
            "Additional triggers",
            placeholder="ruled out, unlikely, no evidence of",
            height=100,
            label_visibility="collapsed",
        )
        triggers = [t.strip() for t in custom_neg.split(",") if t.strip()]
        set_custom_negation_triggers(triggers)

st.divider()

# Analyze button
if st.button("Analyze", type="primary", use_container_width=True):
    if not text.strip():
        st.error("Enter text")
        st.stop()
    if not keywords:
        st.error("Enter keywords")
        st.stop()
    if not model_choices:
        st.error("Select at least one model")
        st.stop()
    
    # Synonym expansion
    terms = list(set(keywords))
    if expand_synonyms:
        try:
            from nltk.corpus import wordnet as wn
            expanded = set(terms)
            for term in keywords:
                for syn in wn.synsets(term):
                    for lemma in syn.lemma_names():
                        expanded.add(lemma.replace("_", " ").lower())
            terms = list(expanded)
        except Exception:
            pass
    
    results = []
    for model in model_choices:
        with st.spinner(f"Analyzing with {model}..."):
            nlp = get_pipeline(model)
            results.extend(extract(text, terms, nlp, model_name=model))
    
    if not results:
        st.warning("No occurrences found")
        st.stop()
    
    df = pd.DataFrame(results)
    st.session_state._doc_count += 1
    st.session_state._hit_history.extend(results)
    st.session_state.results = df
    
    # Section detection
    if show_sections:
        headers = detect_sections(text) + [{"start": len(text) + 1, "title": "END"}]
        def get_section(sentence: str) -> str:
            pos = text.find(sentence[:40])
            if pos < 0:
                return "Unknown"
            current = "Unknown"
            for h in headers:
                if pos < h["start"]:
                    return current
                current = h["title"]
            return current
        df["section"] = df["sentence"].map(get_section)
    
    # Clinical codes
    if show_codes:
        df["code_system"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[0])
        df["code"] = df["keyword"].map(lambda k: CODE_MAP.get(k, ("", ""))[1])
    
    # Temporal extraction
    if show_temporal:
        temporal_pat = re.compile(
            r"\b(?:today|yesterday|tomorrow|last\s+\w+|next\s+\w+|"
            r"\d{1,2}/\d{1,2}/\d{2,4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|"
            r"\d+\s+(?:days?|weeks?|months?|years?)\s+(?:ago|prior))\b",
            re.I
        )
        df["temporal"] = df["sentence"].map(
            lambda s: ", ".join(sorted(set(m.group(0) for m in temporal_pat.finditer(s))))
        )
    
    # Results
    st.divider()
    st.subheader("Results")
    
    # Metrics
    cols = st.columns(4)
    cols[0].metric("Total", len(df))
    for i, cls in enumerate(["Positive", "Negative", "Neutral"]):
        cols[i + 1].metric(cls, len(df[df["classification"] == cls]))
    
    # Chart
    pivot = df.pivot_table(
        index="model", columns="classification", values="keyword", aggfunc="count", fill_value=0
    )
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    
    fig = px.bar(
        pivot.reset_index().melt("model", var_name="Classification", value_name="Count"),
        x="model", y="Count", color="Classification",
        color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#f59e0b"},
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Occurrences
    COLORS = {"Positive": "#dcfce7", "Negative": "#fee2e2", "Neutral": "#fef3c7"}
    
    def highlight(row):
        kw = row["keyword"]
        color = st.session_state._kw_colors.get(kw, COLORS[row["classification"]])
        return re.sub(
            rf"(?i)\b({re.escape(kw)})\b",
            rf"<mark style='background:{color};padding:2px;border-radius:3px;'>\1</mark>",
            row["sentence"]
        )
    
    st.subheader("Occurrences")
    for idx, row in df.iterrows():
        with st.expander(f"{row['keyword']} — {row['classification']} ({row['model']})"):
            st.markdown(highlight(row), unsafe_allow_html=True)
            st.caption(f"Confidence: {row['confidence']:.2f}")
            if show_sections:
                st.caption(f"Section: {row.get('section', 'N/A')}")
            if show_codes and row.get("code"):
                st.caption(f"Code: {row['code_system']} {row['code']}")
            if show_temporal and row.get("temporal"):
                st.caption(f"Temporal: {row['temporal']}")
            
            c1, c2 = st.columns(2)
            if c1.button("✓ Correct", key=f"ok_{idx}"):
                insert_feedback(row["keyword"], row["sentence"], row["classification"], True)
                st.toast("Recorded")
            if c2.button("✗ Wrong", key=f"bad_{idx}"):
                insert_feedback(row["keyword"], row["sentence"], row["classification"], False)
                st.toast("Recorded")
    
    # Review queue
    st.subheader("Review Queue")
    low_conf = df[df["confidence"] < 0.7]
    if len(low_conf):
        st.warning(f"{len(low_conf)} items need review (confidence < 0.7)")
        st.dataframe(low_conf[["keyword", "classification", "confidence", "sentence"]])
    else:
        st.success("No low-confidence items")
    
    # Feedback dashboard
    st.subheader("Feedback Dashboard")
    try:
        agg, recent = get_feedback_summary()
        if agg:
            agg_df = pd.DataFrame(agg, columns=["keyword", "classification", "correct", "incorrect"])
            st.dataframe(agg_df)
        else:
            st.caption("No feedback recorded yet")
    except Exception:
        st.caption("No feedback recorded yet")
    
    # Session analytics
    st.subheader("Session Analytics")
    all_hits = pd.DataFrame(st.session_state._hit_history)
    if not all_hits.empty:
        counts = all_hits["classification"].value_counts()
        fig2 = px.pie(values=counts.values, names=counts.index, title="All-session distribution")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Documents analyzed: {st.session_state._doc_count}")
    
    # Exports
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.download_button("CSV", df.to_csv(index=False), "results.csv", mime="text/csv")
    
    with c2:
        def to_fhir(rec):
            return {
                "resourceType": "Observation",
                "code": {"text": rec["keyword"]},
                "valueString": rec["sentence"],
                "interpretation": [{"text": rec["classification"]}],
            }
        bundle = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": to_fhir(r)} for r in df.to_dict("records")]}
        st.download_button("FHIR Bundle", json.dumps(bundle, indent=2), "bundle.json", mime="application/json")
    
    with c3:
        if st.button("Generate dependency graphs ZIP"):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                for i, row in df.iterrows():
                    nlp = get_pipeline(row.get("model", model_choices[0]))
                    if "parser" in nlp.pipe_names:
                        svg = render_dependency_svg(row["sentence"], nlp)
                        zf.writestr(f"{i}_{row['keyword']}.svg", svg)
            st.download_button("Download ZIP", buf.getvalue(), "dep_graphs.zip", mime="application/zip")

"""BERT vs Rule-based Classifier Comparison"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BERT vs Rule-based", layout="wide")
st.title("BERT vs Rule-based Classification")
st.caption("Compare PubMedBERT and rule-based approaches for protein-disease relation classification")

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "pubmedbert-hfpef" / "final"
BERT_AVAILABLE = MODEL_PATH.exists()


@st.cache_resource
def load_bert_classifier():
    """Load BERT classifier (cached)."""
    from src.bert_classifier import PubMedBERTClassifier
    return PubMedBERTClassifier(model_path=str(MODEL_PATH))


@st.cache_resource
def load_spacy_pipeline():
    """Load spaCy pipeline (cached)."""
    from src.nlp_utils import load_pipeline
    return load_pipeline("en_core_web_sm", use_context=True)


def classify_rule_based(sentence: str) -> tuple[str, float]:
    """Classify using rule-based approach."""
    from src.nlp_utils import classify_span, _confidence_for
    nlp = load_spacy_pipeline()
    doc = nlp(sentence)
    span = list(doc.sents)[0] if list(doc.sents) else doc[:]
    label = classify_span(span)
    conf = _confidence_for(span)
    return label, conf


def classify_bert(sentence: str) -> tuple[str, float]:
    """Classify using PubMedBERT."""
    classifier = load_bert_classifier()
    label, conf = classifier.predict(sentence)
    label_map = {"positive": "Positive", "negative": "Negative", "no_association": "Neutral"}
    return label_map.get(label, "Neutral"), conf


# Check BERT availability
if not BERT_AVAILABLE:
    st.warning(
        "PubMedBERT model not found. Train the model first:\n"
        "```bash\nuv run python scripts/train_bert.py --data data/labeled.json --output models/pubmedbert-hfpef\n```"
    )

# Sidebar with model info
with st.sidebar:
    st.subheader("Model Information")
    
    st.markdown("**Rule-based (spaCy)**")
    st.caption("Uses MedspaCy negation detection with dependency parsing")
    
    st.markdown("**PubMedBERT**")
    if BERT_AVAILABLE:
        st.success("Model loaded")
        st.caption("Fine-tuned on HFpEF protein-disease relations")
    else:
        st.error("Model not available")
    
    st.divider()
    st.markdown("**Performance Summary**")
    st.metric("Rule-based Accuracy", "70.0%")
    st.metric("BERT Accuracy", "93.8%", delta="+23.8%")

# Main content
tab1, tab2, tab3 = st.tabs(["Single Sentence", "Batch Analysis", "Performance Metrics"])

with tab1:
    st.subheader("Classify a Single Sentence")
    
    example_sentences = [
        "Elevated BNP levels are strongly associated with worse outcomes in HFpEF patients.",
        "No significant association was found between SPARC and HFpEF progression.",
        "This retrospective study included 250 patients with HFpEF.",
        "TNF-alpha did not show prognostic value in the HFpEF cohort.",
        "The role of adiponectin in HFpEF remains unclear.",
    ]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_example = st.selectbox("Load example", ["(custom)"] + example_sentences)
    with col2:
        if st.button("Load", use_container_width=True) and selected_example != "(custom)":
            st.session_state.compare_sentence = selected_example
    
    sentence = st.text_area(
        "Enter sentence to classify",
        value=st.session_state.get("compare_sentence", ""),
        height=100,
        placeholder="Enter a sentence about protein-disease relationship...",
    )
    
    if st.button("Classify", type="primary", use_container_width=True, disabled=not sentence.strip()):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Rule-based")
            rule_label, rule_conf = classify_rule_based(sentence)
            color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}[rule_label]
            st.markdown(f"**Classification:** :{color}[{rule_label}]")
            st.progress(rule_conf, text=f"Confidence: {rule_conf:.1%}")
        
        with col2:
            st.markdown("### PubMedBERT")
            if BERT_AVAILABLE:
                bert_label, bert_conf = classify_bert(sentence)
                color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}[bert_label]
                st.markdown(f"**Classification:** :{color}[{bert_label}]")
                st.progress(bert_conf, text=f"Confidence: {bert_conf:.1%}")
            else:
                st.error("BERT model not available")
        
        # Agreement check
        if BERT_AVAILABLE:
            st.divider()
            if rule_label == bert_label:
                st.success(f"Both classifiers agree: **{rule_label}**")
            else:
                st.warning(f"Disagreement: Rule-based says **{rule_label}**, BERT says **{bert_label}**")
                st.info(
                    "Disagreements often occur with:\n"
                    "- Semantic negation (e.g., 'remains unclear')\n"
                    "- Method descriptions\n"
                    "- Uncertain findings"
                )

with tab2:
    st.subheader("Batch Classification")
    st.caption("Compare classifications for multiple sentences")
    
    batch_input = st.text_area(
        "Enter sentences (one per line)",
        height=200,
        placeholder="Enter multiple sentences, one per line...",
    )
    
    if st.button("Classify Batch", type="primary", disabled=not batch_input.strip() or not BERT_AVAILABLE):
        sentences = [s.strip() for s in batch_input.strip().split("\n") if s.strip()]
        
        results = []
        progress = st.progress(0, text="Classifying...")
        
        for i, sent in enumerate(sentences):
            rule_label, rule_conf = classify_rule_based(sent)
            bert_label, bert_conf = classify_bert(sent)
            
            results.append({
                "sentence": sent[:100] + "..." if len(sent) > 100 else sent,
                "rule_label": rule_label,
                "rule_conf": rule_conf,
                "bert_label": bert_label,
                "bert_conf": bert_conf,
                "agree": rule_label == bert_label,
            })
            progress.progress((i + 1) / len(sentences), text=f"Classifying {i+1}/{len(sentences)}...")
        
        progress.empty()
        df = pd.DataFrame(results)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sentences", len(df))
        col2.metric("Agreement Rate", f"{df['agree'].mean():.1%}")
        col3.metric("BERT High Confidence", f"{(df['bert_conf'] > 0.9).mean():.1%}")
        
        # Results table
        st.dataframe(
            df.style.apply(
                lambda row: [
                    "" if row["agree"] else "background-color: #fff3cd"
                    for _ in row
                ],
                axis=1
            ),
            use_container_width=True,
            column_config={
                "rule_conf": st.column_config.ProgressColumn("Rule Conf", max_value=1),
                "bert_conf": st.column_config.ProgressColumn("BERT Conf", max_value=1),
            }
        )
        
        # Disagreement analysis
        disagreements = df[~df["agree"]]
        if len(disagreements) > 0:
            st.subheader(f"Disagreements ({len(disagreements)})")
            for _, row in disagreements.iterrows():
                with st.expander(row["sentence"]):
                    col1, col2 = st.columns(2)
                    col1.metric("Rule-based", row["rule_label"], f"{row['rule_conf']:.1%}")
                    col2.metric("PubMedBERT", row["bert_label"], f"{row['bert_conf']:.1%}")
        
        # Download
        st.download_button(
            "Download Results (CSV)",
            df.to_csv(index=False),
            "batch_comparison.csv",
            mime="text/csv"
        )

with tab3:
    st.subheader("Performance Metrics")
    st.caption("Based on evaluation of 290 labeled sentences")
    
    # Accuracy comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = go.Figure(data=[
            go.Bar(
                x=["Rule-based", "PubMedBERT"],
                y=[70.0, 93.8],
                marker_color=["#e74c3c", "#27ae60"],
                text=["70.0%", "93.8%"],
                textposition="outside",
            )
        ])
        fig_acc.update_layout(
            title="Overall Accuracy",
            yaxis_title="Accuracy (%)",
            yaxis_range=[0, 105],
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # F1 by class
        classes = ["Positive", "Negative", "No Assoc."]
        rule_f1 = [0.81, 0.53, 0.00]
        bert_f1 = [0.95, 0.99, 0.86]
        
        fig_f1 = go.Figure(data=[
            go.Bar(name="Rule-based", x=classes, y=rule_f1, marker_color="#e74c3c"),
            go.Bar(name="PubMedBERT", x=classes, y=bert_f1, marker_color="#27ae60"),
        ])
        fig_f1.update_layout(
            title="F1 Score by Class",
            yaxis_title="F1 Score",
            yaxis_range=[0, 1.1],
            barmode="group",
            height=400,
        )
        st.plotly_chart(fig_f1, use_container_width=True)
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        cm_rule = [[186, 2, 0], [27, 17, 0], [57, 1, 0]]
        fig_cm_rule = px.imshow(
            cm_rule,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=["Positive", "Negative", "No Assoc."],
            y=["Positive", "Negative", "No Assoc."],
            color_continuous_scale="Reds",
            text_auto=True,
        )
        fig_cm_rule.update_layout(title="Rule-based", height=350)
        st.plotly_chart(fig_cm_rule, use_container_width=True)
    
    with col2:
        cm_bert = [[174, 1, 13], [0, 44, 0], [4, 0, 54]]
        fig_cm_bert = px.imshow(
            cm_bert,
            labels=dict(x="Predicted", y="True", color="Count"),
            x=["Positive", "Negative", "No Assoc."],
            y=["Positive", "Negative", "No Assoc."],
            color_continuous_scale="Greens",
            text_auto=True,
        )
        fig_cm_bert.update_layout(title="PubMedBERT", height=350)
        st.plotly_chart(fig_cm_bert, use_container_width=True)
    
    # Key insights
    st.subheader("Key Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Rule-based Limitations:**
        - Never predicts "No Association" (0% recall)
        - Relies on explicit negation words
        - Misses semantic negation patterns
        - High false positive rate for positive class
        """)
    
    with col2:
        st.markdown("""
        **PubMedBERT Strengths:**
        - Understands biomedical context
        - Detects semantic negation
        - Handles uncertain language
        - Balanced performance across classes
        """)
    
    # Improvement metrics
    st.subheader("Improvement Summary")
    improvements = pd.DataFrame({
        "Metric": ["Accuracy", "Macro F1", "Weighted F1", "Negative Detection"],
        "Rule-based": ["70.0%", "0.45", "0.61", "0%"],
        "PubMedBERT": ["93.8%", "0.93", "0.94", "86%"],
        "Improvement": ["+23.8%", "+48.7%", "+33.2%", "+86%"],
    })
    st.dataframe(improvements, use_container_width=True, hide_index=True)

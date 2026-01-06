"""Manual Review - Verify and correct keyword classifications"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Manual Review", layout="wide")
st.title("Manual Review")
st.caption("Verify and correct automated classifications")

if "results" not in st.session_state or st.session_state.results.empty:
    st.warning("No results to review. Run analysis first.")
    if st.button("Go to Analysis"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state.results.copy()

if "reviews" not in st.session_state:
    st.session_state.reviews = {}

# Progress
st.subheader("Progress")
reviewed = len(st.session_state.reviews)
total = len(df)
col1, col2, col3 = st.columns(3)
col1.metric("Total", total)
col2.metric("Reviewed", reviewed)
col3.metric("Remaining", total - reviewed)
st.progress(reviewed / total if total else 0)

st.divider()

# Filter options
st.subheader("Filters")
col1, col2 = st.columns(2)
with col1:
    show_only = st.radio("Show", ["All", "Unreviewed only", "Reviewed only"], horizontal=True)
with col2:
    filter_class = st.multiselect("Classification", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])

st.divider()

# Review items
st.subheader("Items")
for idx, row in df.iterrows():
    review_key = f"{idx}_{row['keyword']}"
    existing = st.session_state.reviews.get(review_key)
    
    # Apply filters
    if show_only == "Unreviewed only" and existing:
        continue
    if show_only == "Reviewed only" and not existing:
        continue
    if row["classification"] not in filter_class:
        continue
    
    status = "✓" if existing else "○"
    
    with st.expander(f"{status} {row['keyword']} — {row['classification']} (conf: {row['confidence']:.2f})"):
        st.markdown(f"**Sentence:** {row['sentence']}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            label = st.radio(
                "Your label:",
                ["Positive", "Negative", "Neutral"],
                index=["Positive", "Negative", "Neutral"].index(
                    existing["label"] if existing else row["classification"]
                ),
                key=f"label_{review_key}",
                horizontal=True,
            )
        
        with col2:
            if st.button("Save review", key=f"save_{review_key}", use_container_width=True):
                st.session_state.reviews[review_key] = {
                    "original": row["classification"],
                    "label": label,
                    "keyword": row["keyword"],
                    "sentence": row["sentence"][:100],
                }
                st.rerun()

# Summary
if st.session_state.reviews:
    st.divider()
    st.subheader("Summary")
    
    review_df = pd.DataFrame(st.session_state.reviews.values())
    
    col1, col2 = st.columns(2)
    
    with col1:
        agreement = (review_df["original"] == review_df["label"]).mean()
        st.metric("Agreement Rate", f"{agreement:.0%}")
        
        disagreements = review_df[review_df["original"] != review_df["label"]]
        if len(disagreements):
            st.markdown("**Disagreements:**")
            st.dataframe(disagreements[["keyword", "original", "label"]])
    
    with col2:
        if len(review_df) >= 3:
            fig, ax = plt.subplots(figsize=(5, 4))
            labels = ["Positive", "Negative", "Neutral"]
            present_labels = [l for l in labels if l in review_df["label"].values or l in review_df["original"].values]
            cm = confusion_matrix(
                review_df["label"], 
                review_df["original"],
                labels=present_labels
            )
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=present_labels, yticklabels=present_labels, ax=ax, cmap="Blues")
            ax.set_xlabel("Model")
            ax.set_ylabel("Manual")
            st.pyplot(fig)
    
    st.divider()
    st.subheader("Export")
    st.download_button(
        "Download Reviews (CSV)",
        review_df.to_csv(index=False),
        "manual_reviews.csv",
        mime="text/csv",
    )

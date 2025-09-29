"""
Manual Review Page for Keyword Classifications
Allows manual verification and correction of automated classifications
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import re
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Manual Review", layout="wide")

st.title("Manual Review of Keyword Classifications")

# Check if analysis results are available
if 'analysis_results' not in st.session_state or st.session_state.analysis_results.empty:
    st.error("No analysis results found. Please run the analysis first from the main page.")
    if st.button("Go to Main Analysis"):
        st.switch_page("app.py")
    st.stop()

# Get the analysis results
df = st.session_state.analysis_results.copy()

# Initialize manual review session state
if 'manual_reviews' not in st.session_state:
    st.session_state.manual_reviews = {}

if 'current_review_index' not in st.session_state:
    st.session_state.current_review_index = 0

if 'review_completed' not in st.session_state:
    st.session_state.review_completed = False

# Create unique IDs for each result
df['review_id'] = df.index

st.write(f"Total items to review: {len(df)}")

# Progress tracking with visual indicators
total_items = len(df)
reviewed_items = len(st.session_state.manual_reviews)
progress = reviewed_items / total_items if total_items > 0 else 0

st.progress(progress)
col_prog1, col_prog2, col_prog3 = st.columns(3)
with col_prog1:
    st.metric("Total Items", total_items)
with col_prog2:
    st.metric("Reviewed", reviewed_items)
with col_prog3:
    st.metric("Remaining", total_items - reviewed_items)

st.write(f"Progress: {reviewed_items}/{total_items} ({progress:.1%}) completed")

# Show current item status
current_review_id = df.iloc[st.session_state.current_review_index]['review_id'] if st.session_state.current_review_index < len(df) else None
is_current_reviewed = current_review_id in st.session_state.manual_reviews if current_review_id is not None else False

if is_current_reviewed:
    st.success(f"Item {st.session_state.current_review_index + 1} has been reviewed")
else:
    st.info(f"Item {st.session_state.current_review_index + 1} needs review")

# Navigation
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Previous", disabled=st.session_state.current_review_index <= 0):
        st.session_state.current_review_index = max(0, st.session_state.current_review_index - 1)
        st.rerun()

with col2:
    if st.button("Next", disabled=st.session_state.current_review_index >= len(df) - 1):
        st.session_state.current_review_index = min(len(df) - 1, st.session_state.current_review_index + 1)
        st.rerun()

with col3:
    jump_to = st.number_input("Jump to item:", min_value=1, max_value=len(df), value=st.session_state.current_review_index + 1) - 1
    if st.button("Jump"):
        st.session_state.current_review_index = jump_to
        st.rerun()

with col4:
    if st.button("Skip to Next Unreviewed"):
        # Find next unreviewed item
        for i in range(st.session_state.current_review_index + 1, len(df)):
            if df.iloc[i]['review_id'] not in st.session_state.manual_reviews:
                st.session_state.current_review_index = i
                st.rerun()
                break
        else:
            st.info("All remaining items have been reviewed!")

# Quick overview of review status
if st.checkbox("Show Review Overview", value=False):
    st.subheader("Review Status Overview")
    overview_data = []
    for idx, row in df.iterrows():
        review_id = row['review_id']
        is_reviewed = review_id in st.session_state.manual_reviews
        overview_data.append({
            'Item #': idx + 1,
            'Keyword': row['keyword'],
            'Model Classification': row['classification'],
            'Status': 'Reviewed' if is_reviewed else 'Pending',
            'Manual Classification': st.session_state.manual_reviews.get(review_id, {}).get('manual_classification', '') if is_reviewed else ''
        })
    
    overview_df = pd.DataFrame(overview_data)
    
    # Color code the status
    def highlight_status(val):
        color = 'lightgreen' if val == 'Reviewed' else 'lightcoral'
        return f'background-color: {color}'
    
    styled_df = overview_df.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

# Current item
if st.session_state.current_review_index < len(df):
    current_item = df.iloc[st.session_state.current_review_index]
    review_id = current_item['review_id']
    
    st.divider()
    st.subheader(f"Review Item {st.session_state.current_review_index + 1} of {len(df)}")
    
    # Display the item details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Sentence Context:**")
        sentence_text = current_item.get('sentence', 'No sentence context available')
        keyword = current_item.get('keyword', '')
        
        # Create highlighted version of the sentence
        if sentence_text and keyword:
            # Case-insensitive highlighting
            import re
            # Escape special regex characters in keyword
            escaped_keyword = re.escape(keyword)
            # Create pattern for case-insensitive matching
            pattern = re.compile(f'({escaped_keyword})', re.IGNORECASE)
            # Replace with highlighted version
            highlighted_sentence = pattern.sub(r'<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">\1</mark>', sentence_text)
            
            # Display highlighted sentence
            st.markdown(highlighted_sentence, unsafe_allow_html=True)
            
            # Also show plain text version for reference
            with st.expander("Plain text version", expanded=False):
                st.text_area(
                    "Sentence",
                    value=sentence_text,
                    height=100,
                    disabled=True,
                    label_visibility="collapsed",
                )
        else:
            st.text_area(
                "Sentence",
                value=sentence_text,
                height=150,
                disabled=True,
                label_visibility="collapsed",
            )
        
        st.write("**Keyword Found:**")
        st.write(f"**'{current_item['keyword']}'** at token position {current_item.get('token_index', 'N/A')}")
        
        st.write("**Additional Context:**")
        st.write(f"**POS Tag:** {current_item.get('pos', 'N/A')}")
        st.write(f"**Dependency:** {current_item.get('dep', 'N/A')}")
        st.write(f"**Sentence Index:** {current_item.get('sent_index', 'N/A')}")
    
    with col2:
        st.write("**Model Classification:**")
        classification_color = {
            'Positive': 'green',
            'Negative': 'red', 
            'Neutral': 'orange'
        }.get(current_item['classification'], 'gray')
        
        st.markdown(f"<h3 style='color: {classification_color};'>{current_item['classification']}</h3>", unsafe_allow_html=True)
        
        if 'confidence' in current_item:
            confidence_val = current_item['confidence']
            confidence_color = 'green' if confidence_val > 0.7 else 'orange' if confidence_val > 0.4 else 'red'
            st.markdown(f"**Confidence:** <span style='color: {confidence_color};'>{confidence_val:.3f}</span>", unsafe_allow_html=True)
        
        if 'model' in current_item:
            st.write(f"**Model:** {current_item['model']}")
        
        # Show keyword highlighting example
        st.write("**Keyword in Context:**")
        if sentence_text and keyword:
            # Show a snippet around the keyword
            keyword_pos = sentence_text.lower().find(keyword.lower())
            if keyword_pos != -1:
                start = max(0, keyword_pos - 30)
                end = min(len(sentence_text), keyword_pos + len(keyword) + 30)
                snippet = sentence_text[start:end]
                if start > 0:
                    snippet = "..." + snippet
                if end < len(sentence_text):
                    snippet = snippet + "..."
                
                # Highlight in snippet
                highlighted_snippet = re.sub(f'({re.escape(keyword)})', 
                                            r'<mark style="background-color: yellow; padding: 2px 4px; border-radius: 3px;">\1</mark>', 
                                            snippet, flags=re.IGNORECASE)
                st.markdown(highlighted_snippet, unsafe_allow_html=True)
    
    st.divider()
    
    # Manual review section
    st.subheader("Manual Review")
    
    # Get current manual classification if exists
    current_manual = st.session_state.manual_reviews.get(review_id, {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        manual_classification = st.selectbox(
            "Your Classification:",
            options=["Positive", "Negative", "Neutral"],
            index=["Positive", "Negative", "Neutral"].index(current_manual.get('manual_classification', current_item['classification']))
        )
    
    with col2:
        confidence_level = st.selectbox(
            "Your Confidence:",
            options=["High", "Medium", "Low"],
            index=["High", "Medium", "Low"].index(current_manual.get('confidence_level', 'Medium'))
        )
    
    # Comments
    comments = st.text_area(
        "Comments (optional):",
        value=current_manual.get('comments', ''),
        height=100,
        placeholder="Add any notes about this classification..."
    )
    
    # Save review
    if st.button("Save Review", type="primary"):
        sentence_snippet = current_item.get('sentence', '')[:100] + "..." if len(current_item.get('sentence', '')) > 100 else current_item.get('sentence', '')
        
        st.session_state.manual_reviews[review_id] = {
            'manual_classification': manual_classification,
            'confidence_level': confidence_level,
            'comments': comments,
            'original_classification': current_item['classification'],
            'keyword': current_item['keyword'],
            'sentence_snippet': sentence_snippet,
            'sent_index': current_item.get('sent_index', 'N/A'),
            'token_index': current_item.get('token_index', 'N/A')
        }
        
        # Auto-advance to next unreviewed item
        for i in range(st.session_state.current_review_index + 1, len(df)):
            if df.iloc[i]['review_id'] not in st.session_state.manual_reviews:
                st.session_state.current_review_index = i
                break
        else:
            # All items reviewed
            st.session_state.review_completed = True
        
        st.success("Review saved!")
        st.rerun()

# Review summary
if reviewed_items > 0:
    st.divider()
    st.subheader("Review Summary")
    
    # Create summary dataframe
    review_data = []
    for review_id, review in st.session_state.manual_reviews.items():
        review_data.append({
            'Review ID': review_id,
            'Keyword': review['keyword'],
            'Original': review['original_classification'],
            'Manual': review['manual_classification'],
            'Confidence': review['confidence_level'],
            'Agreement': review['original_classification'] == review['manual_classification'],
            'Sentence Snippet': review['sentence_snippet'],
            'Sent Index': review.get('sent_index', 'N/A'),
            'Token Index': review.get('token_index', 'N/A')
        })
    
    review_df = pd.DataFrame(review_data)
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        agreement_rate = review_df['Agreement'].mean()
        st.metric("Agreement Rate", f"{agreement_rate:.1%}")
    
    with col2:
        st.metric("Items Reviewed", f"{len(review_df)}")
    
    with col3:
        disagreements = (~review_df['Agreement']).sum()
        st.metric("Disagreements", f"{disagreements}")
    
    with col4:
        high_confidence = (review_df['Confidence'] == 'High').sum()
        st.metric("High Confidence", f"{high_confidence}")
    
    # Show disagreements
    if disagreements > 0:
        st.subheader("Disagreements")
        disagreement_df = review_df[~review_df['Agreement']]
        st.dataframe(disagreement_df, use_container_width=True)
    
    # Export reviews
    st.subheader("Export Reviews")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_reviews = review_df.to_csv(index=False)
        st.download_button(
            label="Download Review Data (CSV)",
            data=csv_reviews,
            file_name="manual_review_results.csv",
            mime="text/csv"
        )
    
    with col2:
        if st.button("Generate Confusion Matrix", type="primary"):
            if len(review_df) > 0:
                st.session_state.generate_confusion_matrix = True
                st.rerun()

# Generate confusion matrix if requested
if st.session_state.get('generate_confusion_matrix', False) and reviewed_items > 0:
    st.divider()
    st.subheader("Confusion Matrix")
    
    # Prepare data for confusion matrix
    review_data = []
    for review_id, review in st.session_state.manual_reviews.items():
        review_data.append({
            'original': review['original_classification'],
            'manual': review['manual_classification']
        })
    
    cm_df = pd.DataFrame(review_data)
    
    if len(cm_df) > 0:
        # Get unique labels actually present in the data
        all_labels = ["Positive", "Negative", "Neutral"]
        actual_labels = sorted(list(set(cm_df['manual'].tolist() + cm_df['original'].tolist())))
        
        # Create confusion matrix with actual labels
        cm = confusion_matrix(cm_df['manual'], cm_df['original'], labels=actual_labels)
        
        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=actual_labels, yticklabels=actual_labels, ax=ax)
        ax.set_xlabel('Model Prediction')
        ax.set_ylabel('Manual Review (Ground Truth)')
        ax.set_title('Confusion Matrix: Manual Review vs Model Prediction')
        
        st.pyplot(fig)
        
        # Classification report with actual labels only
        st.subheader("Classification Report")
        
        # Only generate report if we have more than one class
        if len(actual_labels) > 1:
            report = classification_report(cm_df['manual'], cm_df['original'], 
                                         labels=actual_labels, target_names=actual_labels, output_dict=True)
            
            # Convert to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
        else:
            st.info(f"Only one class present in the data: {actual_labels[0]}. Cannot generate classification report.")
        
        # Show summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            accuracy = (cm_df['manual'] == cm_df['original']).mean()
            st.metric("Accuracy", f"{accuracy:.3f}")
        
        with col2:
            total_agreements = (cm_df['manual'] == cm_df['original']).sum()
            st.metric("Agreements", f"{total_agreements}/{len(cm_df)}")
        
        with col3:
            total_disagreements = (cm_df['manual'] != cm_df['original']).sum()
            st.metric("Disagreements", f"{total_disagreements}/{len(cm_df)}")
        
        # Download confusion matrix data
        col1, col2 = st.columns(2)
        with col1:
            cm_csv = pd.DataFrame(cm, index=actual_labels, columns=actual_labels).to_csv()
            st.download_button(
                label="Download Confusion Matrix (CSV)",
                data=cm_csv,
                file_name="confusion_matrix.csv",
                mime="text/csv"
            )
        
        with col2:
            if len(actual_labels) > 1:
                report_csv = report_df.to_csv()
                st.download_button(
                    label="Download Classification Report (CSV)",
                    data=report_csv,
                    file_name="classification_report.csv",
                    mime="text/csv"
                )
            else:
                # Create a simple summary CSV for single class
                summary_data = {
                    'Class': actual_labels[0],
                    'Count': len(cm_df),
                    'Accuracy': accuracy
                }
                summary_csv = pd.DataFrame([summary_data]).to_csv(index=False)
                st.download_button(
                    label="Download Summary (CSV)",
                    data=summary_csv,
                    file_name="classification_summary.csv",
                    mime="text/csv"
                )

# Completion message
if st.session_state.get('review_completed', False):
    st.success("All items have been reviewed! You can now generate the confusion matrix above.")

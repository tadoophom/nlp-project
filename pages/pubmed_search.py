"""
PubMed Searst.title("PubMed Search")h Page - Dedicated page for searching and selecting PubMed articles
"""

import streamlit as st
import pandas as pd
import re
import io
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from corpus_pipeline import load_protein_entries, build_corpus


def _entries_to_dataframe(entries) -> pd.DataFrame:
    """Convert ProteinEntry objects to a DataFrame for preview purposes."""
    records: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.raw_row:
            records.append(entry.raw_row)
        else:
            row: Dict[str, Any] = {"identifier": entry.identifier}
            if entry.score is not None:
                row["score"] = entry.score
            records.append(row)
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)

# NLP utilities for abstract sentiment/polarity analysis
try:
    from nlp_utils import load_pipeline, extract as nlp_extract, render_dependency_svg, set_custom_negation_triggers
except Exception:
    load_pipeline = None
    nlp_extract = None
    render_dependency_svg = None
    set_custom_negation_triggers = None

# Feedback DB (optional)
try:
    from database import insert_feedback
except Exception:
    insert_feedback = None

# Import PubMed functions
try:
    from pubmed_fetch import search_pubmed, search_pubmed_advanced, fetch_abstracts
except ImportError:
    search_pubmed = search_pubmed_advanced = fetch_abstracts = None


def highlight_terms_in_text(text: str, keywords: List[str], mesh_terms: List[str]) -> str:
    """
    Highlight search terms in text using HTML styling.
    Keywords get blue highlighting, MeSH terms get green highlighting.
    """
    if not text:
        return text
    
    highlighted_text = text
    
    # Highlight keywords in blue
    for keyword in keywords:
        if keyword.strip():
            # Handle multi-word terms and single words
            keyword_clean = keyword.strip()
            # For multi-word terms, don't use word boundaries on the entire phrase
            if ' ' in keyword_clean:
                # Multi-word: exact phrase matching
                pattern = re.escape(keyword_clean)
            else:
                # Single word: use word boundaries
                pattern = r'\b' + re.escape(keyword_clean) + r'\b'
            
            highlighted_text = re.sub(
                pattern, 
                f'<span style="background-color: #E3F2FD; color: #1976D2; font-weight: bold; padding: 1px 3px; border-radius: 3px;">{keyword_clean}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )
    
    # Highlight MeSH terms in green
    for mesh_term in mesh_terms:
        if mesh_term.strip():
            # Handle multi-word terms and single words
            mesh_clean = mesh_term.strip()
            # For multi-word terms, don't use word boundaries on the entire phrase
            if ' ' in mesh_clean:
                # Multi-word: exact phrase matching
                pattern = re.escape(mesh_clean)
            else:
                # Single word: use word boundaries
                pattern = r'\b' + re.escape(mesh_clean) + r'\b'
                
            highlighted_text = re.sub(
                pattern,
                f'<span style="background-color: #E8F5E8; color: #2E7D32; font-weight: bold; padding: 1px 3px; border-radius: 3px;">{mesh_clean}</span>',
                highlighted_text,
                flags=re.IGNORECASE
            )
    
    return highlighted_text

st.set_page_config(page_title="PubMed Search", layout="wide")
st.title("PubMed Search")
st.write("Search for academic articles from PubMed database")

st.divider()

# Initialize session state for selected articles
if "selected_pubmed_articles" not in st.session_state:
    st.session_state.selected_pubmed_articles = []
if "pubmed_search_results" not in st.session_state:
    st.session_state.pubmed_search_results = []

# Sidebar for search parameters
with st.sidebar:
    st.header("Search Parameters")
    
    # Add explanation about search types
    with st.expander("Search Help", expanded=False):
        st.markdown("""
        **Keywords**: Search in article titles and abstracts using natural language
        - Example: `procalcitonin, sepsis, antibiotic therapy`
        
        **MeSH Terms**: Search using controlled medical vocabulary
        - Example: `Procalcitonin, Anti-Bacterial Agents, Pneumonia`
        
        **Boolean Logic Options**:
        - **OR (default)**: Find articles with ANY of your terms (broader search)
        - **AND**: Find articles with ALL your terms (narrower search)
        - **Keywords only**: Search only in titles/abstracts
        - **MeSH only**: Search only assigned subject headings
        
        **Date Filters**: 
        - **Specific year**: `2020` finds articles published in 2020
        - **Date range**: `2020-2023` finds articles published between those years
        
        **Exclusions**: Use NOT logic to exclude unwanted terms
        - Useful for excluding pediatric studies, specific populations, etc.
        
        **Best Practice**: Use both keywords and MeSH for comprehensive results!
        """)
    
    st.subheader("Search Fields")
    pubmed_keywords = st.text_input(
        "Keywords (searches title/abstract)", 
        placeholder="e.g., procalcitonin, sepsis, antibiotic therapy",
        help="Natural language terms that will be searched in article titles and abstracts"
    )
    pubmed_mesh = st.text_input(
        "MeSH terms (controlled vocabulary)", 
        placeholder="e.g., Procalcitonin, Anti-Bacterial Agents, Pneumonia",
        help="Medical Subject Headings - standardized terms assigned by indexers"
    )
    
    # Advanced search options
    with st.expander("Advanced Search Options", expanded=False):
        st.subheader("Boolean Logic")
        search_logic = st.radio(
            "How to combine keywords and MeSH terms:",
            ["OR (broader search - default)", "AND (narrower search)", "Keywords only", "MeSH terms only"],
            index=0,
            help="OR finds articles with ANY of your terms, AND finds articles with ALL terms"
        )

        # Optional: direct PubMed query to mix AND/OR in one expression
        use_raw_query = st.checkbox(
            "Use custom PubMed boolean query",
            value=False,
            help="Write an advanced PubMed query; overrides the above logic."
        )
        raw_query = ""
        if use_raw_query:
            raw_query = st.text_area(
                "Custom query",
                placeholder=(
                    "Example: (procalcitonin[Title/Abstract] AND sepsis[Title/Abstract]) "
                    "OR \"Anti-Bacterial Agents\"[MeSH Terms]"
                ),
                height=80,
            )
        
        # Boolean Builder – compose an advanced AND/OR query without writing syntax
        st.subheader("Boolean Builder (no syntax needed)")
        builder_enabled = st.checkbox(
            "Build an advanced query for me",
            value=False,
            help="Describe your terms; we compose a PubMed query using AND/OR and field tags."
        )
        builder_raw_query = ""
        if builder_enabled:
            group_count = st.number_input("Number of groups", min_value=1, max_value=3, value=2, step=1)
            groups = []
            for i in range(int(group_count)):
                st.markdown(f"**Group {i+1}**")
                terms_str = st.text_input(
                    f"Group {i+1} terms (comma-separated)",
                    key=f"builder_terms_{i}",
                    placeholder="e.g., procalcitonin, PCT"
                )
                field = st.selectbox(
                    f"Field for Group {i+1}",
                    ["Keywords", "MeSH Terms"],
                    key=f"builder_field_{i}",
                    help="Keywords searches Title/Abstract in PubMed"
                )
                within_logic = st.radio(
                    f"Within-group logic (Group {i+1})",
                    ["OR", "AND"],
                    horizontal=True,
                    key=f"builder_within_{i}"
                )
                terms = [t.strip() for t in terms_str.split(",") if t.strip()]
                if terms:
                    suffix = "[Title/Abstract]" if field == "Keywords" else "[MeSH Terms]"
                    # Build group query
                    group_terms = [f"{t}{suffix}" for t in terms]
                    group_query = "(" + f" {within_logic} ".join(group_terms) + ")"
                    groups.append(group_query)
                # Joiners between groups
                if i < int(group_count) - 1:
                    joiner = st.selectbox(
                        f"Join Group {i+1} with Group {i+2} using",
                        ["AND", "OR"],
                        key=f"builder_joiner_{i}"
                    )
                    groups.append(joiner)
            # Compose final raw query
            # groups list alternates: group_query, joiner, group_query, ...
            builder_raw_query = " ".join(groups).strip()
            if builder_raw_query:
                with st.expander("Composed Query Preview", expanded=False):
                    st.code(builder_raw_query)

        st.subheader("Publication Date Filters")
        col1, col2 = st.columns(2)
        with col1:
            use_date_filter = st.checkbox("Filter by publication date", value=False)
        with col2:
            if use_date_filter:
                date_filter_type = st.radio("Date filter type:", ["Specific year", "Date range"], index=0)
        
        if use_date_filter:
            if date_filter_type == "Specific year":
                pub_year = st.number_input("Publication year:", min_value=1900, max_value=2025, value=2020, step=1)
                date_query = f"{pub_year}[PDAT]"
            else:  # Date range
                col1, col2 = st.columns(2)
                with col1:
                    start_year = st.number_input("From year:", min_value=1900, max_value=2025, value=2020, step=1)
                with col2:
                    end_year = st.number_input("To year:", min_value=1900, max_value=2025, value=2025, step=1)
                date_query = f"{start_year}:{end_year}[PDAT]"
        else:
            date_query = ""
        
        st.subheader("Exclusion Terms")
        exclude_keywords = st.text_input(
            "Exclude keywords (NOT logic)", 
            placeholder="e.g., pediatric, children",
            help="Articles containing these terms will be excluded from results"
        )
        exclude_mesh = st.text_input(
            "Exclude MeSH terms", 
            placeholder="e.g., Child, Infant",
            help="Articles with these MeSH terms will be excluded"
        )
        
        st.subheader("Additional Filters")
        col1, col2 = st.columns(2)
        with col1:
            article_types = st.multiselect(
                "Article types (optional):",
                ["Clinical Trial", "Meta-Analysis", "Review", "Systematic Review", "Case Reports", "Randomized Controlled Trial"],
                help="Filter by specific publication types"
            )
        with col2:
            language_filter = st.selectbox(
                "Language:",
                ["Any language", "English", "Spanish", "French", "German"],
                index=0
            )
    
    # Show what will be searched
    if pubmed_keywords.strip() or pubmed_mesh.strip() or use_date_filter or exclude_keywords.strip() or exclude_mesh.strip():
        st.info("**Search Strategy Preview:**")
        
        # Main search terms
        if pubmed_keywords.strip():
            kw_list = [k.strip() for k in pubmed_keywords.split(",") if k.strip()]
            st.write(f"Keywords ({len(kw_list)}): {', '.join(kw_list)}")
        if pubmed_mesh.strip():
            mesh_list = [m.strip() for m in pubmed_mesh.split(",") if m.strip()]
            st.write(f"MeSH terms ({len(mesh_list)}): {', '.join(mesh_list)}")
        
        # Boolean logic
        if pubmed_keywords.strip() and pubmed_mesh.strip():
            if search_logic == "OR (broader search - default)":
                st.write("**Logic**: Keywords OR MeSH terms (broader search)")
            elif search_logic == "AND (narrower search)":
                st.write("**Logic**: Keywords AND MeSH terms (narrower search)")
        elif search_logic == "Keywords only" and pubmed_keywords.strip():
            st.write("**Logic**: Keywords only")
        elif search_logic == "MeSH terms only" and pubmed_mesh.strip():
            st.write("**Logic**: MeSH terms only")
        
        # Date filters
        if use_date_filter:
            if date_filter_type == "Specific year":
                st.write(f"**Date filter**: Published in {pub_year}")
            else:
                st.write(f"**Date filter**: Published between {start_year}-{end_year}")
        
        # Exclusions
        if exclude_keywords.strip():
            excl_kw = [k.strip() for k in exclude_keywords.split(",") if k.strip()]
            st.write(f"**Exclude keywords**: {', '.join(excl_kw)}")
        if exclude_mesh.strip():
            excl_mesh = [m.strip() for m in exclude_mesh.split(",") if m.strip()]
            st.write(f"**Exclude MeSH**: {', '.join(excl_mesh)}")
        
        # Additional filters
        if article_types:
            st.write(f"**Article types**: {', '.join(article_types)}")
        if language_filter != "Any language":
            st.write(f"**Language**: {language_filter}")
    
    st.subheader("Search Options")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        pubmed_retmax = st.number_input("Max results", min_value=1, max_value=100, value=20)
    with col2:
        try_full_text = st.checkbox("Try to get full text (slower)", value=False, 
                                   help="Attempts to retrieve full text from PMC and open access sources")
    with col3:
        st.write("")  # Spacer

    # Analysis model selection for sentiment step
    analysis_model = st.selectbox(
        "Analysis model",
        options=["en_core_web_sm", "en_core_web_md"],
        index=0,
        help="spaCy model used for abstract analysis and dependency trees"
    )
    
    search_button = st.button("Search PubMed", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if search_button:
        if not search_pubmed or not fetch_abstracts:
            st.error("PubMed functionality not available. Please check pubmed_fetch module.")
        else:
            # Allow Boolean Builder or custom raw query to satisfy input requirements
            has_builder_query = 'builder_enabled' in locals() and builder_enabled and bool(builder_raw_query)
            has_raw_query = 'use_raw_query' in locals() and use_raw_query and bool(raw_query.strip())
            has_basic_terms = bool(pubmed_keywords.strip()) or bool(pubmed_mesh.strip())

            keywords_only_block = (
                search_logic == "Keywords only" and
                not pubmed_keywords.strip() and
                not has_builder_query and
                not has_raw_query
            )
            mesh_only_block = (
                search_logic == "MeSH terms only" and
                not pubmed_mesh.strip() and
                not has_builder_query and
                not has_raw_query
            )
            no_terms_block = (
                not has_basic_terms and
                not has_builder_query and
                not has_raw_query
            )

            if keywords_only_block:
                st.error("Please enter keywords for keywords-only search, or use the Boolean Builder/custom query.")
                st.stop()
            if mesh_only_block:
                st.error("Please enter MeSH terms for MeSH-only search, or use the Boolean Builder/custom query.")
                st.stop()
            if no_terms_block:
                st.error("Provide Keywords, MeSH terms, or use the Boolean Builder/custom query.")
                st.stop()
            with st.spinner("Searching PubMed..."):
                try:
                    # Prepare search parameters
                    keywords_list = [k.strip() for k in pubmed_keywords.split(",") if k.strip()]
                    mesh_list = [m.strip() for m in pubmed_mesh.split(",") if m.strip()]
                    exclude_kw_list = [k.strip() for k in exclude_keywords.split(",") if k.strip()] if exclude_keywords.strip() else []
                    exclude_mesh_list = [m.strip() for m in exclude_mesh.split(",") if m.strip()] if exclude_mesh.strip() else []
                    
                    # Map search logic
                    logic_mapping = {
                        "OR (broader search - default)": "OR",
                        "AND (narrower search)": "AND", 
                        "Keywords only": "keywords_only",
                        "MeSH terms only": "mesh_only"
                    }
                    search_logic_param = logic_mapping.get(search_logic, "OR")
                    
                    # Use advanced search if any advanced options are set
                    use_advanced = (
                        use_date_filter or 
                        exclude_keywords.strip() or 
                        exclude_mesh.strip() or 
                        article_types or 
                        language_filter != "Any language" or 
                        search_logic != "OR (broader search - default)" or
                        ("raw_query" in locals() and use_raw_query and raw_query.strip()) or
                        ("builder_raw_query" in locals() and builder_enabled and builder_raw_query)
                    )
                    
                    if use_advanced and search_pubmed_advanced:
                        ids, actual_query = search_pubmed_advanced(
                            keywords=keywords_list,
                            mesh_terms=mesh_list,
                            retmax=int(pubmed_retmax),
                            search_logic=search_logic_param,
                            date_query=date_query if use_date_filter else "",
                            exclude_keywords=exclude_kw_list,
                            exclude_mesh=exclude_mesh_list,
                            article_types=article_types,
                            language=language_filter,
                            raw_query=(
                                builder_raw_query if ("builder_raw_query" in locals() and builder_enabled and builder_raw_query)
                                else (raw_query if ("raw_query" in locals() and use_raw_query and raw_query.strip()) else None)
                            ),
                        )
                        # Show the actual query used
                        with st.expander("Actual PubMed Query Used", expanded=False):
                            st.code(actual_query)
                    else:
                        # Use simple search for basic queries
                        ids = search_pubmed(keywords_list, mesh_list, int(pubmed_retmax))
                        actual_query = "Basic search (no advanced options)"
                    
                    if ids:
                        st.success(f"Found {len(ids)} articles!")
                        with st.spinner("Fetching article details..." + (" (including full text where available)" if try_full_text else "")):
                            records = fetch_abstracts(ids, try_full_text=try_full_text)
                            
                            # Show full text statistics
                            if try_full_text:
                                full_text_count = sum(1 for r in records if r.get('has_full_text'))
                                if full_text_count > 0:
                                    st.success(f"Retrieved full text for {full_text_count} out of {len(records)} articles!")
                                else:
                                    st.info("No full text available for these articles (abstracts only)")
                            
                            st.session_state.pubmed_search_results = records
                            # Store search terms for highlighting
                            st.session_state.search_keywords = keywords_list
                            st.session_state.search_mesh_terms = mesh_list
                    else:
                        st.warning("No articles found matching the search criteria.")
                        st.session_state.pubmed_search_results = []
                        
                except Exception as e:
                    st.error(f"Search failed: {str(e)}")
                    st.session_state.pubmed_search_results = []

    # Display search results
    if st.session_state.pubmed_search_results:
        st.subheader(f"Search Results ({len(st.session_state.pubmed_search_results)} articles)")
        
        # Show highlighting legend
        if st.session_state.get('search_keywords') or st.session_state.get('search_mesh_terms'):
            st.markdown("**Highlighting Legend:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.session_state.get('search_keywords'):
                    keywords_str = ", ".join(st.session_state.get('search_keywords', []))
                    st.markdown(f'<span style="background-color: #E3F2FD; color: #1976D2; font-weight: bold; padding: 2px 6px; border-radius: 3px;">Keywords</span> ({keywords_str})', unsafe_allow_html=True)
            with col2:
                if st.session_state.get('search_mesh_terms'):
                    mesh_str = ", ".join(st.session_state.get('search_mesh_terms', []))
                    st.markdown(f'<span style="background-color: #E8F5E8; color: #2E7D32; font-weight: bold; padding: 2px 6px; border-radius: 3px;">MeSH Terms</span> ({mesh_str})', unsafe_allow_html=True)
            st.write("")  # Add some spacing
        
        # Create DataFrame for easier handling
        df = pd.DataFrame(st.session_state.pubmed_search_results)
        
        # Selection interface
        st.write("**Select articles to add to your analysis:**")
        
        # Track selections without page refresh
        for idx, row in df.iterrows():
            col_check, col_content = st.columns([1, 10])
            
            # Check if this article is already selected
            is_selected = any(a.get('pmid') == row.get('pmid') for a in st.session_state.selected_pubmed_articles)
            
            with col_check:
                # Use session state to track selection
                checkbox_key = f"select_{row.get('pmid', idx)}"
                if st.checkbox("", value=is_selected, key=checkbox_key):
                    # Add to selection if not already there
                    if not is_selected:
                        st.session_state.selected_pubmed_articles.append(row.to_dict())
                else:
                    # Remove from selection if it was there
                    if is_selected:
                        st.session_state.selected_pubmed_articles = [
                            a for a in st.session_state.selected_pubmed_articles 
                            if a.get('pmid') != row.get('pmid')
                        ]
            
            with col_content:
                # Get search terms for highlighting
                search_keywords = st.session_state.get('search_keywords', [])
                search_mesh_terms = st.session_state.get('search_mesh_terms', [])
                
                # Article preview with full text indicator and highlighted title
                title_prefix = "[TEXT]" if row.get('has_full_text') else "[ABSTRACT]"
                highlighted_title = highlight_terms_in_text(row.get('title', 'No title'), search_keywords, search_mesh_terms)
                
                # Use markdown to render HTML highlighting in expander header
                with st.expander(f"{title_prefix} {row.get('title', 'No title')}", expanded=False):
                    # Show highlighted title separately inside expander
                    st.markdown(f"**Title:** {highlighted_title}", unsafe_allow_html=True)
                    st.write(f"**Authors:** {row.get('authors', 'N/A')}")
                    st.write(f"**Journal:** {row.get('journal', 'N/A')}")
                    st.write(f"**Year:** {row.get('year', 'N/A')}")
                    st.write(f"**PMID:** {row.get('pmid', 'N/A')}")
                    if row.get('doi'):
                        st.write(f"**DOI:** {row.get('doi')}")
                    
                    # Show match indicators
                    if search_keywords or search_mesh_terms:
                        st.markdown("**Search term matches:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            if search_keywords:
                                st.markdown('<span style="background-color: #E3F2FD; color: #1976D2; font-weight: bold; padding: 2px 6px; border-radius: 3px; margin-right: 5px;">Keywords</span>', unsafe_allow_html=True)
                        with col2:
                            if search_mesh_terms:
                                st.markdown('<span style="background-color: #E8F5E8; color: #2E7D32; font-weight: bold; padding: 2px 6px; border-radius: 3px;">MeSH Terms</span>', unsafe_allow_html=True)
                    
                    # Show full text availability
                    if row.get('has_full_text'):
                        st.success(f"Full text available from: {row.get('full_text_source', 'Unknown source')}")
                        if row.get('full_text_url'):
                            st.markdown(f"[View full text]({row.get('full_text_url')})", unsafe_allow_html=False)
                    elif row.get('full_text_source') and 'PDF' in row.get('full_text_source', ''):
                        pdf_url = row.get('full_text_url') or row.get('full_text_source').split(': ', 1)[1]
                        st.info(f"Open access PDF available")
                        if pdf_url:
                            st.markdown(f"[Open PDF]({pdf_url})", unsafe_allow_html=False)
                    else:
                        st.warning("Only abstract available")
                    
                    if row.get('abstract'):
                        highlighted_abstract = highlight_terms_in_text(row.get('abstract'), search_keywords, search_mesh_terms)
                        st.markdown(f"**Abstract:** {highlighted_abstract}", unsafe_allow_html=True)
                    else:
                        st.write("*No abstract available*")
                    
                    # Show preview of full text if available
                    if row.get('full_text') and row.get('has_full_text'):
                        with st.expander("View Full Text", expanded=False):
                            highlighted_full = highlight_terms_in_text(row.get('full_text', ''), search_keywords, search_mesh_terms)
                            st.markdown(highlighted_full, unsafe_allow_html=True)
        
        # Batch sentiment analysis over abstracts
        st.divider()
        st.subheader("Batch Sentiment Analysis (Abstracts)")
        st.caption("Analyze each abstract with your keyword list and export.")

        analysis_kw_default = (
            ", ".join(st.session_state.get("search_keywords", []))
            if st.session_state.get("search_keywords") else ""
        )
        analysis_keywords_raw = st.text_input(
            "Analysis keywords (comma-separated)",
            value=analysis_kw_default,
            placeholder="e.g., procalcitonin, sepsis, antibiotic therapy"
        )
        analysis_keywords = [k.strip().lower() for k in analysis_keywords_raw.split(",") if k.strip()]

        ca1, ca2, ca3 = st.columns(3)
        with ca1:
            run_analysis = st.button("Run Analysis on All Results")
        with ca2:
            run_and_export = st.button("Analyze + Download CSV")
        with ca3:
            run_selected = st.button("Analyze Selected Only")

        def _summarize_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
            per_kw: Dict[str, Dict[str, int]] = {}
            for h in hits:
                kw = h.get("keyword")
                cls = h.get("classification", "Neutral")
                if not kw:
                    continue
                per_kw.setdefault(kw, {"Positive": 0, "Negative": 0, "Neutral": 0})
                if cls in per_kw[kw]:
                    per_kw[kw][cls] += 1
                else:
                    per_kw[kw]["Neutral"] += 1

            parts: List[str] = []
            kw_labels: Dict[str, str] = {}
            for kw, counts in per_kw.items():
                label = (
                    "Negative" if counts.get("Negative", 0) > 0 else
                    "Positive" if counts.get("Positive", 0) > 0 else
                    "Neutral"
                )
                kw_labels[kw] = label
                parts.append(
                    f"{kw}: {label} (P{counts.get('Positive',0)}/N{counts.get('Negative',0)}/U{counts.get('Neutral',0)})"
                )

            totals = {"Positive": 0, "Negative": 0, "Neutral": 0}
            for lbl in kw_labels.values():
                totals[lbl] += 1
            overall = max(totals, key=totals.get) if sum(totals.values()) else "Neutral"

            return {
                "sentiment_per_keyword": kw_labels,
                "sentiment_counts": totals,
                "sentiment_overall": overall,
                "sentiment_summary": "; ".join(parts),
            }

        if (run_analysis or run_and_export or run_selected):
            if not analysis_keywords:
                st.error("Please provide analysis keywords.")
            elif not load_pipeline or not nlp_extract:
                st.error("NLP utilities not available; cannot run analysis.")
            else:
                with st.spinner("Analyzing abstracts…"):
                    nlp = load_pipeline(analysis_model, gpu=False)
                    # Sync custom negation triggers from main app if available
                    if set_custom_negation_triggers is not None:
                        trig = st.session_state.get("custom_neg_triggers", None)
                        if trig:
                            set_custom_negation_triggers(trig)
                    # Choose source: selected articles or all results
                    source_rows = (
                        st.session_state.selected_pubmed_articles
                        if run_selected and st.session_state.get("selected_pubmed_articles")
                        else st.session_state.pubmed_search_results
                    )
                    enriched_rows = []
                    for row in source_rows:
                        abs_text = row.get("abstract", "") or ""
                        hits = nlp_extract(abs_text, analysis_keywords, nlp)
                        summary = _summarize_hits(hits)
                        new_row = dict(row)
                        new_row.update(summary)
                        new_row["analysis_keywords"] = ", ".join(analysis_keywords)
                        new_row["hits"] = hits  # persist per-article details
                        enriched_rows.append(new_row)

                df_preview = pd.DataFrame(enriched_rows)
                st.dataframe(df_preview[[
                    c for c in ["pmid", "title", "year", "journal", "analysis_keywords", "sentiment_overall", "sentiment_summary"]
                    if c in df_preview.columns
                ]], use_container_width=True)

                st.session_state.pubmed_search_results = enriched_rows

                if run_and_export:
                    csv = df_preview.to_csv(index=False)
                    st.download_button(
                        label="Download CSV (with sentiment)",
                        data=csv,
                        file_name="pubmed_search_results_with_sentiment.csv",
                        mime="text/csv",
                    )

                # Detailed per-article view mirroring the main analysis
                st.divider()
                st.subheader("Per-Article Analysis Details")
                st.caption("Inspect hits, sentences, and labels for each abstract.")

                COLOUR = {"Positive": "#e6ffed", "Negative": "#ffeef0", "Neutral": "#fff8e1"}
                # Reuse keyword colors from main app if available
                kw_colors = st.session_state.get("_kw_colors", {})

                for i, art in enumerate(enriched_rows, 1):
                    with st.expander(f"{i}. {art.get('title','No title')} — {art.get('sentiment_overall','Neutral')} ", expanded=False):
                        st.write(f"PMID: {art.get('pmid','N/A')} | Year: {art.get('year','N/A')} | Journal: {art.get('journal','N/A')}")
                        if art.get("sentiment_summary"):
                            st.write(f"Summary: {art['sentiment_summary']}")
                        hits = art.get("hits", [])
                        if hits:
                            df_hits = pd.DataFrame(hits)
                            # Keep a consistent column order if available
                            cols = [c for c in ["keyword","classification","sentence","pos","dep","sent_index","token_index"] if c in df_hits.columns]
                            st.dataframe(df_hits[cols] if cols else df_hits, use_container_width=True, height=240)

                            # Sentence view with classification highlighting
                            with st.expander("Sentence View (colored by classification)", expanded=False):
                                # Legend
                                st.markdown(
                                    "<div>"
                                    "<span style='background:#e6ffed;padding:2px 6px;border-radius:4px;margin-right:6px;'>Positive</span>"
                                    "<span style='background:#ffeef0;padding:2px 6px;border-radius:4px;margin-right:6px;'>Negative</span>"
                                    "<span style='background:#fff8e1;padding:2px 6px;border-radius:4px;'>Neutral</span>"
                                    "</div>",
                                    unsafe_allow_html=True,
                                )
                                for idx_h, h in enumerate(hits):
                                    sent = h.get("sentence", "")
                                    kw = h.get("keyword", "")
                                    cls = h.get("classification", "Neutral")
                                    color = COLOUR.get(cls, "#fff")
                                    # Highlight keyword within sentence (case-insensitive)
                                    if kw:
                                        try:
                                            pattern = re.compile(re.escape(kw), flags=re.I)
                                            highlighted = pattern.sub(lambda m: f"<mark style='background:#fff59d;padding:0 2px;border-radius:2px;'>{m.group(0)}</mark>", sent)
                                        except Exception:
                                            highlighted = sent
                                    else:
                                        highlighted = sent
                                    edge = kw_colors.get(kw, "#ddd") if isinstance(kw_colors, dict) else "#ddd"
                                    html = (
                                        f"<div style='background:{color};padding:8px;border-radius:6px;margin:6px 0;"
                                        f"border-left: 6px solid {edge};'>"
                                        f"<b style='color:{edge}'>{kw}</b> — <i>{cls}</i><br/>{highlighted}"
                                        f"</div>"
                                    )
                                    st.markdown(html, unsafe_allow_html=True)

                                    # Feedback buttons per hit (if DB available)
                                    if insert_feedback is not None:
                                        c1, c2 = st.columns(2)
                                        if c1.button("👍 Correct", key=f"hit_ok_{i}_{idx_h}"):
                                            insert_feedback(keyword=kw, sentence=sent, classification=cls, correct_label=True)
                                            st.success("Feedback recorded as correct")
                                        if c2.button("👎 Incorrect", key=f"hit_bad_{i}_{idx_h}"):
                                            insert_feedback(keyword=kw, sentence=sent, classification=cls, correct_label=False)
                                            st.success("Feedback recorded as incorrect")

                            # Optional dependency trees for first N unique sentences
                            show_trees = st.checkbox("Show dependency trees (first N sentences)", key=f"trees_{i}")
                            if show_trees and render_dependency_svg is not None:
                                max_trees = st.number_input("Max trees per article", min_value=1, max_value=10, value=3, step=1, key=f"max_trees_{i}")
                                rendered = set()
                                count = 0
                                for h in hits:
                                    s = h.get("sentence", "")
                                    if not s or s in rendered:
                                        continue
                                    rendered.add(s)
                                    svg = render_dependency_svg(s, nlp)
                                    st.components.v1.html(svg, height=280, scrolling=False)
                                    count += 1
                                    if count >= int(max_trees):
                                        break
                        else:
                            st.info("No keyword hits found in this abstract with the provided analysis keywords.")

                # Download all hit-level rows across articles
                def _flatten_hits(rows: List[Dict[str, Any]]) -> pd.DataFrame:
                    flat: List[Dict[str, Any]] = []
                    for r in rows:
                        meta = {k: r.get(k) for k in ["pmid", "title", "year", "journal"]}
                        for h in r.get("hits", []) or []:
                            rec = dict(meta)
                            rec.update({
                                "keyword": h.get("keyword"),
                                "classification": h.get("classification"),
                                "sentence": h.get("sentence"),
                                "pos": h.get("pos"),
                                "dep": h.get("dep"),
                                "sent_index": h.get("sent_index"),
                                "token_index": h.get("token_index"),
                            })
                            flat.append(rec)
                    return pd.DataFrame(flat)

                flat_df = _flatten_hits(enriched_rows)
                if not flat_df.empty:
                    csv_hits = flat_df.to_csv(index=False)
                    st.download_button(
                        label="Download Hit-Level Analysis CSV",
                        data=csv_hits,
                        file_name="pubmed_hit_level_analysis.csv",
                        mime="text/csv",
                    )

                # Download Everything ZIP (preview + hit-level)
                df_enriched = pd.DataFrame(enriched_rows)
                csv_overview = df_enriched.to_csv(index=False).encode("utf-8")
                csv_hits_bytes = (flat_df.to_csv(index=False).encode("utf-8") if not flat_df.empty else b"")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("pubmed_overview.csv", csv_overview)
                    if csv_hits_bytes:
                        zf.writestr("pubmed_hit_level_analysis.csv", csv_hits_bytes)
                st.download_button(
                    label="Download Everything (ZIP)",
                    data=buf.getvalue(),
                    file_name="pubmed_analysis_bundle.zip",
                    mime="application/zip",
                )

        # Download all results (plain)
        if st.button("Download All Results as CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="pubmed_search_results.csv",
                mime="text/csv"
            )

with col2:
    # Selection summary
    st.subheader("Selected Articles")
    
    if st.session_state.selected_pubmed_articles:
        st.success(f"{len(st.session_state.selected_pubmed_articles)} article(s) selected")
        
        # Show selected articles
        for i, article in enumerate(st.session_state.selected_pubmed_articles, 1):
            with st.expander(f"{i}. {article.get('title', 'No title')[:50]}...", expanded=False):
                st.write(f"**PMID:** {article.get('pmid', 'N/A')}")
                st.write(f"**Authors:** {article.get('authors', 'N/A')}")
                st.write(f"**Journal:** {article.get('journal', 'N/A')}")
                if st.button(f"Remove", key=f"remove_{article.get('pmid', i)}"):
                    st.session_state.selected_pubmed_articles = [
                        a for a in st.session_state.selected_pubmed_articles 
                        if a.get('pmid') != article.get('pmid')
                    ]
                    st.rerun()
        
        # Export selected articles
        st.divider()
        st.subheader("Export Options")
        
        # Export format selection
        export_format = st.radio(
            "Choose export format:",
            ["Title + Abstract", "Full Article Text", "Abstract Only", "Full Text Only (if available)", "Custom Format"],
            index=0
        )
        
        # Custom format options
        if export_format == "Custom Format":
            include_title = st.checkbox("Include Title", value=True)
            include_authors = st.checkbox("Include Authors", value=True)
            include_journal = st.checkbox("Include Journal", value=True)
            include_year = st.checkbox("Include Year", value=True)
            include_pmid = st.checkbox("Include PMID", value=False)
            include_doi = st.checkbox("Include DOI (if available)", value=False)
            include_abstract = st.checkbox("Include Abstract", value=True)
            include_full_text_note = st.checkbox("Include Full Text Note", value=True)
            
            # Custom separator
            separator = st.text_input("Section separator:", value="\n\n")
        
        # Preview export text
        if st.button("Preview Export Text"):
            preview_text = ""
            
            for i, article in enumerate(st.session_state.selected_pubmed_articles, 1):
                if export_format == "Title + Abstract":
                    article_text = f"Article {i}:\nTitle: {article.get('title', 'No title')}\nAuthors: {article.get('authors', 'No authors listed')}\nJournal: {article.get('journal', 'No journal listed')}\nYear: {article.get('year', 'No year listed')}\nAbstract: {article.get('abstract', 'No abstract available')}"
                elif export_format == "Abstract Only":
                    article_text = f"Abstract {i}: {article.get('abstract', 'No abstract available')}"
                elif export_format == "Full Text Only (if available)":
                    if article.get('has_full_text'):
                        article_text = f"Article {i} - Full Text:\nTitle: {article.get('title', 'No title')}\n\n{article.get('full_text', 'No full text available')}"
                    else:
                        article_text = f"Article {i}: Full text not available. Abstract: {article.get('abstract', 'No abstract available')}"
                elif export_format == "Full Article Text":
                    article_text = f"Article {i}:\n"
                    article_text += f"Title: {article.get('title', 'No title')}\n"
                    article_text += f"Authors: {article.get('authors', 'No authors listed')}\n"
                    article_text += f"Journal: {article.get('journal', 'No journal listed')}\n"
                    article_text += f"Year: {article.get('year', 'No year listed')}\n"
                    article_text += f"PMID: {article.get('pmid', 'No PMID')}\n"
                    if article.get('doi'):
                        article_text += f"DOI: {article.get('doi')}\n"
                    if article.get('has_full_text'):
                        article_text += f"Full Text: {article.get('full_text', 'No full text available')}"
                    else:
                        article_text += f"Abstract: {article.get('abstract', 'No abstract available')}"
                    if article.get('full_text_note'):
                        article_text += article.get('full_text_note')
                elif export_format == "Custom Format":
                    article_text = f"Article {i}:\n"
                    if include_title:
                        article_text += f"Title: {article.get('title', 'No title')}\n"
                    if include_authors:
                        article_text += f"Authors: {article.get('authors', 'No authors listed')}\n"
                    if include_journal:
                        article_text += f"Journal: {article.get('journal', 'No journal listed')}\n"
                    if include_year:
                        article_text += f"Year: {article.get('year', 'No year listed')}\n"
                    if include_pmid:
                        article_text += f"PMID: {article.get('pmid', 'No PMID')}\n"
                    if include_doi and article.get('doi'):
                        article_text += f"DOI: {article.get('doi')}\n"
                    if include_abstract:
                        article_text += f"Abstract: {article.get('abstract', 'No abstract available')}"
                    if include_full_text_note and article.get('full_text_note'):
                        article_text += article.get('full_text_note')
                
                preview_text += article_text + "\n\n"
            
            st.text_area("Export Preview:", value=preview_text, height=300)
        
        if st.button("Export Selected to Analysis", type="primary"):
            # Generate export text based on selected format
            export_text = ""
            
            for i, article in enumerate(st.session_state.selected_pubmed_articles, 1):
                if export_format == "Title + Abstract":
                    article_text = f"Title: {article.get('title', 'No title')}\nAuthors: {article.get('authors', 'No authors listed')}\nJournal: {article.get('journal', 'No journal listed')}\nYear: {article.get('year', 'No year listed')}\nAbstract: {article.get('abstract', 'No abstract available')}"
                elif export_format == "Abstract Only":
                    article_text = article.get('abstract', 'No abstract available')
                elif export_format == "Full Text Only (if available)":
                    if article.get('has_full_text'):
                        article_text = f"Title: {article.get('title', 'No title')}\n\n{article.get('full_text', 'No full text available')}"
                    else:
                        article_text = f"Title: {article.get('title', 'No title')}\nAbstract: {article.get('abstract', 'No abstract available')} [Note: Full text not available]"
                elif export_format == "Full Article Text":
                    article_text = f"Title: {article.get('title', 'No title')}\n"
                    article_text += f"Authors: {article.get('authors', 'No authors listed')}\n"
                    article_text += f"Journal: {article.get('journal', 'No journal listed')}\n"
                    article_text += f"Year: {article.get('year', 'No year listed')}\n"
                    article_text += f"PMID: {article.get('pmid', 'No PMID')}\n"
                    if article.get('doi'):
                        article_text += f"DOI: {article.get('doi')}\n"
                    if article.get('has_full_text'):
                        article_text += f"Full Text: {article.get('full_text', 'No full text available')}"
                    else:
                        article_text += f"Abstract: {article.get('abstract', 'No abstract available')}"
                    if article.get('full_text_note'):
                        article_text += article.get('full_text_note')
                elif export_format == "Custom Format":
                    article_text = ""
                    if include_title:
                        article_text += f"Title: {article.get('title', 'No title')}\n"
                    if include_authors:
                        article_text += f"Authors: {article.get('authors', 'No authors listed')}\n"
                    if include_journal:
                        article_text += f"Journal: {article.get('journal', 'No journal listed')}\n"
                    if include_year:
                        article_text += f"Year: {article.get('year', 'No year listed')}\n"
                    if include_pmid:
                        article_text += f"PMID: {article.get('pmid', 'No PMID')}\n"
                    if include_doi and article.get('doi'):
                        article_text += f"DOI: {article.get('doi')}\n"
                    if include_abstract:
                        article_text += f"Abstract: {article.get('abstract', 'No abstract available')}"
                    if include_full_text_note and article.get('full_text_note'):
                        article_text += article.get('full_text_note')
                
                if export_format == "Custom Format":
                    export_text += article_text + separator
                else:
                    export_text += article_text + "\n\n"
            
            # Store selected articles and formatted text for use in main analysis
            st.session_state.selected_articles_for_analysis = st.session_state.selected_pubmed_articles
            st.session_state.pubmed_export_text = export_text.strip()
            st.session_state.pubmed_export_format = export_format
            
            st.success("Articles exported! Go to the main Analysis page to process them.")
            st.balloons()
        
        if st.button("Clear All Selections"):
            st.session_state.selected_pubmed_articles = []
            st.rerun()
        
        # Download selected articles
        if st.button("Download Selected as CSV"):
            selected_df = pd.DataFrame(st.session_state.selected_pubmed_articles)
            csv = selected_df.to_csv(index=False)
            st.download_button(
                label="Download Selected CSV",
                data=csv,
                file_name="selected_pubmed_articles.csv",
                mime="text/csv"
            )
    else:
        st.info("No articles selected yet. Search and select articles from the left.")

# Instructions
st.divider()
with st.expander("How to use this page", expanded=False):
    st.markdown("""
    1. **Search**: Enter keywords and/or MeSH terms in the sidebar and click "Search PubMed"
       - **Keywords**: Natural language terms (searched in titles/abstracts)
       - **MeSH Terms**: Controlled medical vocabulary (assigned by indexers)
       - **Advanced Options**: Use boolean logic, date filters, and exclusions for precise searches
    2. **Select**: Check the boxes next to articles you want to include in your analysis
    3. **Review**: View selected articles in the right panel
    4. **Export**: Click "Export Selected to Analysis" to send articles to the main analysis page
    5. **Analyze**: Go back to the main Analysis page to process your selected articles
    
    **Search Tips:**
    - **Basic Search**: Use both keywords AND MeSH terms for comprehensive results
    - **Year Filtering**: Find recent papers with "2023" or range "2020-2023" 
    - **Boolean Logic**: 
      - OR (default): Broader search, finds ANY matching terms
      - AND: Narrower search, requires ALL terms
    - **Exclusions**: Remove unwanted results (e.g., exclude "pediatric, children")
    - **Article Types**: Filter by study design (Clinical Trial, Meta-Analysis, etc.)
    - **Language**: Restrict to specific languages
    - **Full Text**: Enable to get complete articles when legally available
    
    **Example Advanced Search:**
    - Keywords: `procalcitonin, sepsis`
    - MeSH: `Procalcitonin, Anti-Bacterial Agents`
    - Year: `2020-2023`
    - Exclude: `pediatric, children`
    - Type: `Clinical Trial, Meta-Analysis`
    """)

st.divider()
st.header("Disease–Protein Relation Corpus")
st.write(
    "Combine PubMed retrieval with the built-in polarity model to spot positive, negative, or "
    "neutral co-mentions between a disease concept and a protein panel. Upload your own panel or "
    "try the bundled HFpEF sample to generate an annotated corpus directly inside the app."
)
st.caption(
    "Tip: If your list uses UniProt accessions (e.g. P51606), the tool will auto-expand them to "
    "gene symbols and preferred names via the UniProt API before querying PubMed."
)

col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Protein table (CSV or XLSX)",
        type=["csv", "xlsx"],
        key="relation_protein_file",
        help="Provide a spreadsheet with at least one column of protein identifiers.",
    )
with col_sample:
    use_sample = st.checkbox(
        "Use HFpEF sample",
        value=False,
        help="Loads the repository sample_aktan.xlsx protein list",
    )

protein_preview = None
protein_source_path: Path | None = None
uploaded_bytes: bytes | None = None
uploaded_suffix = ""

if use_sample:
    sample_path = Path("sample_aktan.xlsx")
    if sample_path.exists():
        protein_source_path = sample_path
        try:
            protein_preview = pd.read_excel(sample_path)
        except Exception:
            try:
                entries_preview = load_protein_entries(sample_path)
            except Exception as exc:  # pragma: no cover - interactive warning only
                st.error(f"Unable to read sample_aktan.xlsx: {exc}")
            else:
                protein_preview = _entries_to_dataframe(entries_preview)
    else:
        st.error("sample_aktan.xlsx not found in the project directory.")
elif uploaded_file is not None:
    uploaded_bytes = uploaded_file.getvalue()
    uploaded_suffix = Path(uploaded_file.name).suffix.lower() or ".csv"
    st.session_state["relation_uploaded_bytes"] = uploaded_bytes
    st.session_state["relation_uploaded_suffix"] = uploaded_suffix
    tmp_preview_path: Path | None = None
    try:
        buffer = io.BytesIO(uploaded_bytes)
        if uploaded_suffix == ".csv":
            protein_preview = pd.read_csv(buffer)
        else:
            protein_preview = pd.read_excel(buffer)
    except Exception:
        try:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_suffix)
            tmp_file.write(uploaded_bytes)
            tmp_file.flush()
            tmp_file.close()
            tmp_preview_path = Path(tmp_file.name)
            entries_preview = load_protein_entries(tmp_preview_path)
        except Exception as exc:  # pragma: no cover - interactive warning only
            st.error(f"Could not parse uploaded file: {exc}")
        else:
            protein_preview = _entries_to_dataframe(entries_preview)
    finally:
        if tmp_preview_path and tmp_preview_path.exists():
            tmp_preview_path.unlink(missing_ok=True)
else:
    st.info("Upload a protein table or enable the sample to build the relation corpus.")

if protein_preview is not None and not protein_preview.empty:
    st.caption("Preview (first five rows)")
    st.dataframe(protein_preview.head())

    preview_columns = [str(col) for col in protein_preview.columns]
    lower_columns = [col.lower() for col in preview_columns]

    if not preview_columns:
        st.warning("No columns detected in the protein table.")
    else:
        try:
            identifier_index = lower_columns.index("protein")
        except ValueError:
            identifier_index = 0
        identifier_column = st.selectbox(
            "Protein identifier column",
            preview_columns,
            index=identifier_index,
            help="Column containing the protein or biomarker identifier that will be queried in PubMed.",
        )

        score_options = ["(none)"] + preview_columns
        default_score_index = 0
        if "hfpef" in lower_columns:
            default_score_index = lower_columns.index("hfpef") + 1
        score_selection = st.selectbox(
            "Score/weight column (optional)",
            score_options,
            index=default_score_index,
            help="Use to prioritise proteins by score; enables top-N and score threshold filters.",
        )
        if score_selection == "(none)":
            score_column = None
        else:
            score_column = score_selection

        col_filters = st.columns(2)
        with col_filters[0]:
            if score_column:
                top_n = st.number_input(
                    "Limit to top N proteins",
                    min_value=1,
                    value=25,
                    step=1,
                    help="Sorts by the selected score column before taking the first N proteins.",
                )
            else:
                top_n = None
        with col_filters[1]:
            if score_column:
                min_score = st.number_input(
                    "Minimum score",
                    value=0.0,
                    step=0.01,
                    help="Discard proteins below this score before querying PubMed.",
                )
            else:
                min_score = None

        st.subheader("PubMed query configuration")
        default_disease_terms = "HFpEF, Heart Failure with Preserved Ejection Fraction" if use_sample else ""
        disease_keywords_input = st.text_input(
            "Disease keywords (comma separated)",
            value=default_disease_terms,
            help="Used in the Title/Abstract field for each protein query.",
        )
        disease_mesh_input = st.text_input(
            "Disease MeSH terms (optional)",
            value="Heart Failure, Diastolic" if use_sample else "",
            help="Repeatable MeSH descriptors for the disease concept.",
        )
        extra_keywords_input = st.text_input(
            "Additional keywords (optional)",
            placeholder="e.g., biomarker, plasma",
        )
        extra_mesh_input = st.text_input(
            "Additional MeSH terms (optional)",
            placeholder="e.g., Biomarkers, Blood Proteins",
        )

        logic_label = st.selectbox(
            "Boolean logic",
            [
                "Disease AND protein (recommended)",
                "All groups AND",
                "Any group OR",
                "Disease OR protein",
            ],
            index=0,
        )
        logic_map = {
            "Disease AND protein (recommended)": "disease_and_protein",
            "All groups AND": "all_and",
            "Any group OR": "all_or",
            "Disease OR protein": "disease_or_protein",
        }
        logic_value = logic_map[logic_label]

        retmax = st.number_input(
            "PubMed results per protein",
            min_value=5,
            max_value=200,
            step=5,
            value=50,
            help="Upper limit of PubMed abstracts to fetch for each protein.",
        )

        spacy_model = st.selectbox(
            "spaCy model for sentiment",
            ["en_core_web_sm", "en_core_web_md"],
            index=0,
            help="Model used to detect polarity cues in the retrieved abstracts.",
        )

        run_button = st.button("Build relation corpus", type="primary")

        disease_keywords = [term.strip() for term in disease_keywords_input.split(",") if term.strip()]
        disease_mesh = [term.strip() for term in disease_mesh_input.split(",") if term.strip()]
        extra_keywords = [term.strip() for term in extra_keywords_input.split(",") if term.strip()]
        extra_mesh = [term.strip() for term in extra_mesh_input.split(",") if term.strip()]

        if run_button:
            if not disease_keywords:
                st.error("Add at least one disease keyword before building the corpus.")
            else:
                path_for_loader: Path | None = None
                tmp_path: Path | None = None
                try:
                    if use_sample and protein_source_path is not None:
                        path_for_loader = protein_source_path
                    else:
                        bytes_payload = st.session_state.get("relation_uploaded_bytes")
                        suffix = st.session_state.get("relation_uploaded_suffix", ".csv")
                        if not bytes_payload:
                            st.error("Upload a protein table to continue.")
                            raise RuntimeError("Missing uploaded protein bytes")
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                        tmp_file.write(bytes_payload)
                        tmp_file.flush()
                        tmp_file.close()
                        tmp_path = Path(tmp_file.name)
                        path_for_loader = tmp_path

                    with st.spinner("Loading protein panel..."):
                        proteins = load_protein_entries(
                            Path(path_for_loader),
                            identifier_column=identifier_column,
                            score_column=score_column,
                            top_n=int(top_n) if (score_column and top_n) else None,
                            min_score=float(min_score) if (score_column and min_score is not None) else None,
                        )

                    if not proteins:
                        st.warning("No proteins available after applying filters.")
                    else:
                        with st.spinner("Querying PubMed and classifying relations..."):
                            rows = build_corpus(
                                protein_entries=proteins,
                                disease_keywords=disease_keywords,
                                disease_mesh=disease_mesh,
                                additional_keywords=extra_keywords,
                                additional_mesh=extra_mesh,
                                logic=logic_value,
                                spaCy_model=spacy_model,
                                retmax=int(retmax),
                            )

                        if not rows:
                            st.warning(
                                "No co-mentions detected. Try expanding the keyword list or increasing the PubMed limit."
                            )
                        else:
                            result_df = pd.DataFrame(rows)
                            st.session_state["relation_results_df"] = result_df
                            st.success(
                                f"Captured {len(result_df)} article rows across {len(proteins)} protein(s)."
                            )
                except Exception as exc:  # pragma: no cover - interactive warning only
                    st.error(f"Failed to build corpus: {exc}")
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink(missing_ok=True)

results_df = st.session_state.get("relation_results_df")
if results_df is not None and not results_df.empty:
    st.subheader("Relation corpus preview")
    st.dataframe(results_df)
    csv_export = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download relation corpus",
        csv_export,
        "relation_corpus.csv",
        mime="text/csv",
    )

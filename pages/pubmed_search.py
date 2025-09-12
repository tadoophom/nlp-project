"""
PubMed Searst.title("PubMed Search")h Page - Dedicated page for searching and selecting PubMed articles
"""

import streamlit as st
import pandas as pd
import re
from typing import List, Dict, Any

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
    
    search_button = st.button("Search PubMed", type="primary")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    if search_button:
        if not search_pubmed or not fetch_abstracts:
            st.error("PubMed functionality not available. Please check pubmed_fetch module.")
        elif (not pubmed_keywords.strip() and not pubmed_mesh.strip()) or \
             (search_logic == "Keywords only" and not pubmed_keywords.strip()) or \
             (search_logic == "MeSH terms only" and not pubmed_mesh.strip()):
            if search_logic == "Keywords only":
                st.error("Please enter keywords for keywords-only search.")
            elif search_logic == "MeSH terms only":
                st.error("Please enter MeSH terms for MeSH-only search.")
            else:
                st.error("Please enter at least one of: Keywords OR MeSH terms (or both for best results).")
        else:
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
                        search_logic != "OR (broader search - default)"
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
                            language=language_filter
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
                    elif row.get('full_text_source') and 'PDF' in row.get('full_text_source', ''):
                        st.info(f"Open access PDF available: {row.get('full_text_source').split(': ', 1)[1] if ': ' in row.get('full_text_source', '') else row.get('full_text_source')}")
                    else:
                        st.warning("Only abstract available")
                    
                    if row.get('abstract'):
                        highlighted_abstract = highlight_terms_in_text(row.get('abstract'), search_keywords, search_mesh_terms)
                        st.markdown(f"**Abstract:** {highlighted_abstract}", unsafe_allow_html=True)
                    else:
                        st.write("*No abstract available*")
                    
                    # Show preview of full text if available
                    if row.get('full_text') and row.get('has_full_text'):
                        with st.expander("Full Text Preview", expanded=False):
                            preview_text = row.get('full_text', '')[:1000] + "..." if len(row.get('full_text', '')) > 1000 else row.get('full_text', '')
                            highlighted_preview = highlight_terms_in_text(preview_text, search_keywords, search_mesh_terms)
                            st.markdown(highlighted_preview, unsafe_allow_html=True)
        
        # Download all results
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

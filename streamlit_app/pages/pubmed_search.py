"""PubMed Search - Advanced search with MeSH terms, boolean logic, and filters"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import pandas as pd
import re
from typing import List

from src.pubmed_fetch import search_pubmed, search_pubmed_advanced, fetch_abstracts

st.set_page_config(page_title="PubMed Search", layout="wide")
st.title("PubMed Search")


def highlight_terms(text: str, keywords: List[str], mesh_terms: List[str]) -> str:
    if not text:
        return text
    result = text
    for kw in keywords:
        if kw.strip():
            pattern = re.escape(kw.strip())
            result = re.sub(
                rf"(?i)\b({pattern})\b",
                r'<span style="background:#E3F2FD;color:#1976D2;font-weight:bold;padding:1px 3px;border-radius:3px;">\1</span>',
                result
            )
    for mesh in mesh_terms:
        if mesh.strip():
            pattern = re.escape(mesh.strip())
            result = re.sub(
                rf"(?i)\b({pattern})\b",
                r'<span style="background:#E8F5E9;color:#2E7D32;font-weight:bold;padding:1px 3px;border-radius:3px;">\1</span>',
                result
            )
    return result


# Search Terms
st.subheader("Search Terms")
col1, col2 = st.columns(2)
with col1:
    keywords = st.text_input(
        "Keywords (title/abstract)",
        placeholder="procalcitonin, sepsis, antibiotic",
        help="Natural language terms searched in titles and abstracts"
    )
with col2:
    mesh_terms = st.text_input(
        "MeSH Terms",
        placeholder="Procalcitonin, Anti-Bacterial Agents",
        help="Controlled medical vocabulary terms"
    )

# Boolean Logic
st.subheader("Boolean Logic")
col1, col2 = st.columns(2)
with col1:
    search_logic = st.radio(
        "Combine terms:",
        ["OR (broader)", "AND (narrower)", "Keywords only", "MeSH only"],
        horizontal=True,
    )
with col2:
    use_custom_query = st.checkbox("Use custom boolean query")

custom_query = ""
if use_custom_query:
    custom_query = st.text_area(
        "Custom PubMed query",
        placeholder='(procalcitonin[Title/Abstract] AND sepsis[MeSH]) OR "bacterial infection"[Title/Abstract]',
        height=80,
    )

# Boolean Builder
use_builder = st.checkbox("Use boolean query builder (no syntax needed)")
builder_query = ""
if use_builder:
    num_groups = st.number_input("Number of term groups", 1, 4, 2)
    groups = []
    joiners = []
    
    for i in range(int(num_groups)):
        st.markdown(f"**Group {i+1}**")
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            terms = st.text_input(f"Terms", key=f"grp_{i}", placeholder="term1, term2")
        with c2:
            field = st.selectbox(f"Field", ["Title/Abstract", "MeSH Terms"], key=f"field_{i}")
        with c3:
            within_logic = st.radio(f"Logic", ["OR", "AND"], key=f"logic_{i}", horizontal=True)
        
        if terms.strip():
            term_list = [t.strip() for t in terms.split(",") if t.strip()]
            suffix = "[Title/Abstract]" if field == "Title/Abstract" else "[MeSH Terms]"
            group_parts = [f'"{t}"{suffix}' for t in term_list]
            groups.append("(" + f" {within_logic} ".join(group_parts) + ")")
        
        if i < int(num_groups) - 1:
            joiners.append(st.selectbox(f"Join Group {i+1} → {i+2}", ["AND", "OR"], key=f"join_{i}"))
    
    if groups:
        parts = []
        for i, g in enumerate(groups):
            parts.append(g)
            if i < len(joiners):
                parts.append(joiners[i])
        builder_query = " ".join(parts)
        st.code(builder_query, language=None)

# Filters
st.subheader("Filters")
col1, col2, col3 = st.columns(3)

with col1:
    use_date = st.checkbox("Date filter")
    date_query = ""
    if use_date:
        date_type = st.radio("Type", ["Year", "Range"], horizontal=True)
        if date_type == "Year":
            year = st.number_input("Year", 1990, 2025, 2023)
            date_query = f"{year}[PDAT]"
        else:
            c1, c2 = st.columns(2)
            start = c1.number_input("From", 1990, 2025, 2020)
            end = c2.number_input("To", 1990, 2025, 2025)
            date_query = f"{start}:{end}[PDAT]"

with col2:
    article_types = st.multiselect(
        "Article types",
        ["Clinical Trial", "Meta-Analysis", "Review", "Systematic Review", "Case Reports", "Randomized Controlled Trial"]
    )
    language = st.selectbox("Language", ["Any", "English", "Spanish", "French", "German"])

with col3:
    exclude_kw = st.text_input("Exclude keywords", placeholder="pediatric, children")
    exclude_mesh = st.text_input("Exclude MeSH", placeholder="Child, Infant")

# Options
st.subheader("Options")
col1, col2, col3 = st.columns(3)
with col1:
    max_results = st.number_input("Max results", 5, 100, 20)
with col2:
    try_full_text = st.checkbox("Retrieve full text", help="Slower - gets full text from PMC/Europe PMC")

st.divider()
search_btn = st.button("Search PubMed", type="primary", use_container_width=True)

# Execute search
if search_btn:
    has_terms = keywords.strip() or mesh_terms.strip()
    has_custom = use_custom_query and custom_query.strip()
    has_builder = use_builder and builder_query.strip()
    
    if not (has_terms or has_custom or has_builder):
        st.error("Enter keywords, MeSH terms, or a custom query")
        st.stop()
    
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    mesh_list = [m.strip() for m in mesh_terms.split(",") if m.strip()]
    excl_kw = [k.strip() for k in exclude_kw.split(",") if k.strip()] if exclude_kw else []
    excl_mesh = [m.strip() for m in exclude_mesh.split(",") if m.strip()] if exclude_mesh else []
    
    logic_map = {
        "OR (broader)": "OR",
        "AND (narrower)": "AND",
        "Keywords only": "keywords_only",
        "MeSH only": "mesh_only",
    }
    logic_param = logic_map[search_logic]
    
    use_advanced = (
        use_date or excl_kw or excl_mesh or article_types or
        language != "Any" or search_logic != "OR (broader)" or
        has_custom or has_builder
    )
    
    raw_query = builder_query if has_builder else (custom_query if has_custom else None)
    
    with st.spinner("Searching PubMed..."):
        if use_advanced:
            ids, actual_query = search_pubmed_advanced(
                keywords=kw_list,
                mesh_terms=mesh_list,
                retmax=max_results,
                search_logic=logic_param,
                date_query=date_query if use_date else "",
                exclude_keywords=excl_kw,
                exclude_mesh=excl_mesh,
                article_types=article_types,
                language=language if language != "Any" else "",
                raw_query=raw_query,
            )
        else:
            ids = search_pubmed(kw_list, mesh_list, max_results)
            actual_query = "Basic search"
    
    if not ids:
        st.warning("No results found")
        st.stop()
    
    st.success(f"Found {len(ids)} articles")
    
    with st.expander("Query used"):
        st.code(actual_query)
    
    with st.spinner(f"Fetching abstracts{' and full text' if try_full_text else ''}..."):
        records = fetch_abstracts(ids, try_full_text=try_full_text)
    
    if try_full_text:
        full_count = sum(1 for r in records if r.get("has_full_text"))
        st.info(f"Full text available for {full_count}/{len(records)} articles")
    
    st.session_state.pubmed_results = records
    st.session_state.search_kw = kw_list
    st.session_state.search_mesh = mesh_list

# Display results
if "pubmed_results" in st.session_state:
    records = st.session_state.pubmed_results
    kw_list = st.session_state.get("search_kw", [])
    mesh_list = st.session_state.get("search_mesh", [])
    
    if "selected_pmids" not in st.session_state:
        st.session_state.selected_pmids = set()
    
    st.divider()
    st.subheader(f"Results ({len(records)})")
    
    if kw_list or mesh_list:
        st.markdown(
            '<span style="background:#E3F2FD;padding:2px 6px;border-radius:3px;margin-right:10px;">Keywords</span>'
            '<span style="background:#E8F5E9;padding:2px 6px;border-radius:3px;">MeSH Terms</span>',
            unsafe_allow_html=True
        )
    
    for rec in records:
        pmid = rec["pmid"]
        is_selected = pmid in st.session_state.selected_pmids
        
        col1, col2 = st.columns([0.05, 0.95])
        with col1:
            if st.checkbox("", value=is_selected, key=f"sel_{pmid}", label_visibility="collapsed"):
                st.session_state.selected_pmids.add(pmid)
            else:
                st.session_state.selected_pmids.discard(pmid)
        
        with col2:
            prefix = "📄" if rec.get("has_full_text") else "📋"
            title_hl = highlight_terms(rec["title"], kw_list, mesh_list)
            
            with st.expander(f"{prefix} {rec['title']} ({rec['year']})"):
                st.markdown(f"**Title:** {title_hl}", unsafe_allow_html=True)
                st.caption(f"PMID: {pmid} | {rec['journal']} | {rec['authors']}")
                
                if rec.get("doi"):
                    st.caption(f"DOI: {rec['doi']}")
                
                if rec.get("has_full_text"):
                    st.success(f"Full text: {rec.get('full_text_source')}")
                    if rec.get("full_text_url"):
                        st.markdown(f"[View full text]({rec['full_text_url']})")
                
                abstract_hl = highlight_terms(rec["abstract"], kw_list, mesh_list)
                st.markdown(f"**Abstract:** {abstract_hl}", unsafe_allow_html=True)
                
                if rec.get("has_full_text") and rec.get("full_text"):
                    with st.expander("Full text"):
                        full_hl = highlight_terms(rec["full_text"], kw_list, mesh_list)
                        st.markdown(full_hl, unsafe_allow_html=True)
                
                if st.button("Use for analysis", key=f"use_{pmid}"):
                    st.session_state.text = f"{rec['title']}\n\n{rec['abstract']}"
                    st.switch_page("app.py")
    
    # Selected summary
    selected = [r for r in records if r["pmid"] in st.session_state.selected_pmids]
    if selected:
        st.divider()
        st.subheader(f"Selected: {len(selected)} articles")
        
        if st.button("Export selected to analysis", type="primary"):
            combined = "\n\n---\n\n".join(
                f"**{r['title']}** ({r['year']})\n{r['abstract']}" for r in selected
            )
            st.session_state.text = combined
            st.switch_page("app.py")
    
    # Downloads
    st.divider()
    df = pd.DataFrame(records)
    c1, c2 = st.columns(2)
    c1.download_button("Download all (CSV)", df.to_csv(index=False), "pubmed_results.csv", mime="text/csv")
    if selected:
        sel_df = pd.DataFrame(selected)
        c2.download_button("Download selected (CSV)", sel_df.to_csv(index=False), "pubmed_selected.csv", mime="text/csv")

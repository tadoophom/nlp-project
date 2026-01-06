"""
Fetch PubMed abstracts using keywords and/or MeSH terms, and save as CSV.
Enhanced with full text retrieval where legally available.
"""
import argparse
import csv
import requests
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UNPAYWALL_API = "https://api.unpaywall.org/v2/"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"


def _clean_whitespace(text: str) -> str:
    """Collapse excess whitespace while preserving single spaces."""
    return " ".join(text.split())


def _extract_full_text_from_xml(xml_content: str) -> Optional[str]:
    """Extract readable text from PMC/Europe PMC full-text XML."""

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError:
        return None

    def _tag_name(element: ET.Element) -> str:
        return element.tag.split("}")[-1] if "}" in element.tag else element.tag

    parts: List[str] = []
    seen_titles: set[str] = set()
    seen_paragraphs: set[str] = set()

    for elem in root.iter():
        tag = _tag_name(elem)
        if tag == "title":
            title_text = (elem.text or "").strip()
            if title_text:
                cleaned = _clean_whitespace(title_text)
                if cleaned and cleaned not in seen_titles:
                    parts.append(cleaned)
                    seen_titles.add(cleaned)
        elif tag == "p":
            paragraph = " ".join(_clean_whitespace(txt) for txt in elem.itertext())
            paragraph = _clean_whitespace(paragraph)
            if paragraph and paragraph not in seen_paragraphs:
                parts.append(paragraph)
                seen_paragraphs.add(paragraph)

    if not parts:
        fallback = _clean_whitespace(" ".join(elem.strip() for elem in root.itertext()))
        if fallback:
            parts.append(fallback)

    if not parts:
        return None

    return "\n\n".join(parts).strip()


def get_pmc_id_from_pmid(pmid: str) -> Optional[str]:
    """Get PMC ID from PMID if available."""
    try:
        params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "xml"
        }
        r = requests.get(BASE_URL + "efetch.fcgi", params=params)
        r.raise_for_status()
        
        root = ET.fromstring(r.text)
        # Look for PMC ID in article IDs
        for article_id in root.findall(".//ArticleId"):
            if article_id.get("IdType") == "pmc":
                return article_id.text
        return None
    except Exception:
        return None


def try_pmc_full_text(pmid: str) -> Tuple[Optional[str], Optional[str]]:
    """Try to get full text from PMC if available.

    Returns the retrieved text and a link to the article on PMC when successful.
    """
    pmc_id = get_pmc_id_from_pmid(pmid)
    if not pmc_id:
        return None, None
    
    try:
        # Try to get full text XML from PMC
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml"
        }
        r = requests.get(BASE_URL + "efetch.fcgi", params=params)
        r.raise_for_status()
        text = _extract_full_text_from_xml(r.text)
        if text:
            url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/"
            return text, url
    except Exception:
        return None, None
    return None, None


def try_unpaywall(doi: str, email: str = "research@example.com") -> Optional[str]:
    """Try to find open access PDF URL via Unpaywall."""
    try:
        url = f"{UNPAYWALL_API}{doi}?email={email}"
        r = requests.get(url)
        r.raise_for_status()
        
        data = r.json()
        if data.get("is_oa"):  # Is open access
            # Try to get the best open access location
            for location in data.get("oa_locations", []):
                if location.get("url_for_pdf"):
                    return location["url_for_pdf"]
                elif location.get("url"):
                    return location["url"]
        
        return None
    except Exception:
        return None


def try_europe_pmc_full_text(pmid: str, doi: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Attempt to retrieve open access full text from Europe PMC.

    Returns the article text and a browser-friendly Europe PMC URL when found.
    """

    identifiers: List[str] = []
    if pmid:
        identifiers.append(pmid)
    if doi:
        identifiers.append(f"DOI:{doi}")

    for identifier in identifiers:
        try:
            search_resp = requests.get(
                EUROPE_PMC_API + "search",
                params={
                    "query": f"EXT_ID:{identifier}",
                    "format": "json",
                    "resultType": "core",
                },
                timeout=15,
            )
            search_resp.raise_for_status()
            data = search_resp.json()
            results = data.get("resultList", {}).get("result", [])
            for result in results:
                pmcid = result.get("pmcid")
                if not pmcid:
                    continue
                full_resp = requests.get(
                    EUROPE_PMC_API + f"{pmcid}/fullTextXML",
                    params={"format": "xml"},
                    timeout=20,
                )
                full_resp.raise_for_status()
                text = _extract_full_text_from_xml(full_resp.text)
                if text:
                    url = f"https://europepmc.org/article/PMC/{pmcid}"
                    return text, url
        except Exception:
            continue
    return None, None


def search_pubmed(keywords: List[str], mesh_terms: List[str], retmax: int = 100):
    query = ""
    if keywords:
        query += " OR ".join([f'{kw}[Title/Abstract]' for kw in keywords])
    if mesh_terms:
        if query:
            query += " OR "
        query += " OR ".join([f'{mt}[MeSH Terms]' for mt in mesh_terms])
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": retmax,
        "retmode": "json"
    }
    r = requests.get(BASE_URL + "esearch.fcgi", params=params)
    r.raise_for_status()
    ids = r.json()["esearchresult"]["idlist"]
    return ids


def search_pubmed_advanced(
    keywords: List[str],
    mesh_terms: List[str],
    retmax: int = 100,
    search_logic: str = "OR",
    date_query: str = "",
    exclude_keywords: List[str] | None = None,
    exclude_mesh: List[str] | None = None,
    article_types: List[str] | None = None,
    language: str = "",
    raw_query: str | None = None,
):
    """
    Advanced PubMed search with boolean logic, date filters, exclusions, and
    support for a raw PubMed query string.
    
    Args:
        keywords: List of keywords to search in title/abstract
        mesh_terms: List of MeSH terms
        retmax: Maximum number of results
        search_logic: "OR", "AND", "keywords_only", "mesh_only" (ignored when raw_query provided)
        date_query: Date filter string (e.g., "2020[PDAT]" or "2020:2023[PDAT]")
        exclude_keywords: Keywords to exclude
        exclude_mesh: MeSH terms to exclude
        article_types: Publication types to filter by
        language: Language filter
        raw_query: Full PubMed query string. If provided, overrides composed
            keyword/MeSH logic and is wrapped with additional filters.
    """
    query_parts: List[str] = []

    # Main search terms with logic, unless a raw query is provided
    main_query = ""
    if raw_query and raw_query.strip():
        # Trust caller to provide valid PubMed syntax; just wrap in parentheses
        main_query = raw_query.strip()
    else:
        if search_logic == "keywords_only" and keywords:
            main_query = " OR ".join([f"{kw}[Title/Abstract]" for kw in keywords])
        elif search_logic == "mesh_only" and mesh_terms:
            main_query = " OR ".join([f"{mt}[MeSH Terms]" for mt in mesh_terms])
        elif search_logic == "AND" and keywords and mesh_terms:
            kw_part = "(" + " OR ".join([f"{kw}[Title/Abstract]" for kw in keywords]) + ")"
            mesh_part = "(" + " OR ".join([f"{mt}[MeSH Terms]" for mt in mesh_terms]) + ")"
            main_query = f"{kw_part} AND {mesh_part}"
        else:  # OR logic (default)
            if keywords:
                main_query += " OR ".join([f"{kw}[Title/Abstract]" for kw in keywords])
            if mesh_terms:
                if main_query:
                    main_query += " OR "
                main_query += " OR ".join([f"{mt}[MeSH Terms]" for mt in mesh_terms])

    if main_query:
        # Avoid double-wrapping if the caller already included parentheses at outer level
        mp = main_query.strip()
        query_parts.append(mp if (mp.startswith("(") and mp.endswith(")")) else f"({mp})")
    
    # Date filter
    if date_query:
        query_parts.append(date_query)
    
    # Article type filters
    if article_types:
        type_mapping = {
            "Clinical Trial": "Clinical Trial[PT]",
            "Meta-Analysis": "Meta-Analysis[PT]",
            "Review": "Review[PT]",
            "Systematic Review": "Systematic Review[PT]",
            "Case Reports": "Case Reports[PT]",
            "Randomized Controlled Trial": "Randomized Controlled Trial[PT]"
        }
        type_filters = [type_mapping.get(at, f"{at}[PT]") for at in article_types]
        if type_filters:
            query_parts.append("(" + " OR ".join(type_filters) + ")")
    
    # Language filter
    if language and language != "Any language":
        query_parts.append(f"{language.lower()}[LA]")
    
    # Exclusions
    exclude_parts: List[str] = []
    if exclude_keywords:
        exclude_parts.extend([f'{kw}[Title/Abstract]' for kw in exclude_keywords])
    if exclude_mesh:
        exclude_parts.extend([f'{mt}[MeSH Terms]' for mt in exclude_mesh])
    
    # Combine query parts
    final_query = " AND ".join(query_parts)
    
    # Add exclusions with NOT
    if exclude_parts:
        exclusion_query = " OR ".join(exclude_parts)
        final_query = f"{final_query} NOT ({exclusion_query})"
    
    params = {
        "db": "pubmed",
        "term": final_query,
        "retmax": retmax,
        "retmode": "json"
    }
    
    r = requests.get(BASE_URL + "esearch.fcgi", params=params)
    r.raise_for_status()
    result = r.json()["esearchresult"]
    ids = result["idlist"]
    
    # Return both IDs and the actual query used for debugging
    return ids, final_query


def fetch_abstracts(ids: List[str], try_full_text: bool = False):
    if not ids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    r = requests.get(BASE_URL + "efetch.fcgi", params=params)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    records = []
    for article in root.findall(".//PubmedArticle"):
        # Basic article info
        title = article.findtext(".//ArticleTitle", "")
        abstract = article.findtext(".//AbstractText", "")
        pmid = article.findtext(".//PMID", "")
        
        # Extract authors
        authors = []
        for author in article.findall(".//Author"):
            last_name = author.findtext(".//LastName", "")
            fore_name = author.findtext(".//ForeName", "")
            if last_name and fore_name:
                authors.append(f"{fore_name} {last_name}")
            elif last_name:
                authors.append(last_name)
        
        authors_str = ", ".join(authors) if authors else "No authors listed"
        
        # Extract journal info
        journal = article.findtext(".//Journal/Title", "")
        if not journal:
            journal = article.findtext(".//Journal/ISOAbbreviation", "")
        if not journal:
            journal = "No journal listed"
        
        # Extract publication year
        year = article.findtext(".//PubDate/Year", "")
        if not year:
            year = article.findtext(".//DateCreated/Year", "")
        if not year:
            year = "No year listed"
        
        # Extract DOI if available
        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break
        
        # Extract MeSH terms
        mesh = [m.text for m in article.findall(".//MeshHeading/DescriptorName")]
        
        # Try to get structured abstract
        structured_abstract = ""
        abstract_sections = article.findall(".//Abstract/AbstractText")
        if len(abstract_sections) > 1:
            # Has structured abstract
            sections = []
            for section in abstract_sections:
                label = section.get("Label", "")
                text = section.text or ""
                if label and text:
                    sections.append(f"{label}: {text}")
                elif text:
                    sections.append(text)
            structured_abstract = " ".join(sections)
        
        # Use structured abstract if available, otherwise use regular abstract
        final_abstract = structured_abstract if structured_abstract else abstract
        if not final_abstract:
            final_abstract = "No abstract available"
        
        # Try to get full text if requested
        full_text = ""
        full_text_source = ""
        full_text_url = ""
        if try_full_text and pmid:
            # Try PMC first
            pmc_text, pmc_url = try_pmc_full_text(pmid)
            if pmc_text:
                full_text = pmc_text
                full_text_source = "PMC (PubMed Central)"
                full_text_url = pmc_url or ""
            else:
                europe_text, europe_url = try_europe_pmc_full_text(pmid, doi or None)
                if europe_text:
                    full_text = europe_text
                    full_text_source = "Europe PMC (Open Access)"
                    full_text_url = europe_url or ""

        if try_full_text and not full_text and doi:
            # Try Unpaywall for open access link
            unpaywall_url = try_unpaywall(doi)
            if unpaywall_url:
                full_text_source = f"Open Access PDF: {unpaywall_url}"
                full_text = f"[Full text PDF available at: {unpaywall_url}]"
                full_text_url = unpaywall_url

        # Create full text note
        if full_text and full_text_source in {"PMC (PubMed Central)", "Europe PMC (Open Access)"}:
            full_text_note = f"\n\n[FULL TEXT RETRIEVED from {full_text_source}]"
        elif full_text_source.startswith("Open Access PDF"):
            full_text_note = f"\n\n[FULL TEXT PDF AVAILABLE: {full_text_source.split(': ', 1)[1]}]"
        else:
            full_text_note = (
                "\n\n[NOTE: Only abstract available through PubMed API. "
                "For full text, try institutional access or check if paper is open access.]"
            )
        
        records.append({
            "pmid": pmid,
            "title": title,
            "abstract": final_abstract,
            "authors": authors_str,
            "journal": journal,
            "year": year,
            "doi": doi,
            "mesh_terms": ";".join(mesh),
            "full_text": full_text if full_text else "",
            "full_text_source": full_text_source,
            "full_text_note": full_text_note,
            "full_text_url": full_text_url,
            "has_full_text": bool(full_text and full_text_source in {"PMC (PubMed Central)", "Europe PMC (Open Access)"})
        })
    return records


def save_csv(records, out_file):
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pmid", "title", "authors", "journal", "year", "doi", "abstract", "full_text", "full_text_source", "mesh_terms", "full_text_note", "has_full_text"])
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts by keyword/MeSH and save as CSV.")
    parser.add_argument("--keywords", type=str, default="", help="Comma-separated keywords")
    parser.add_argument("--mesh", type=str, default="", help="Comma-separated MeSH terms")
    parser.add_argument("--retmax", type=int, default=100, help="Max results")
    parser.add_argument("--out", type=str, default="pubmed_dataset.csv", help="Output CSV file")
    parser.add_argument("--full-text", action="store_true", help="Try to retrieve full text where available")
    args = parser.parse_args()
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    mesh_terms = [m.strip() for m in args.mesh.split(",") if m.strip()]
    ids = search_pubmed(keywords, mesh_terms, args.retmax)
    print(f"Found {len(ids)} articles.")
    records = fetch_abstracts(ids, try_full_text=args.full_text)
    
    # Count full text availability
    full_text_count = sum(1 for r in records if r.get('has_full_text'))
    print(f"Retrieved {full_text_count} articles with full text out of {len(records)} total articles.")
    
    save_csv(records, args.out)
    print(f"Saved {len(records)} records to {args.out}")

if __name__ == "__main__":
    main()

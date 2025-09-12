"""
Enhanced PubMed fetcher with full text capabilities where legally available.
Tries multiple sources: PMC, Europe PMC, Unpaywall, and arXiv.
"""
import requests
import json
from typing import List, Optional, Dict
from xml.etree import ElementTree as ET

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UNPAYWALL_API = "https://api.unpaywall.org/v2/"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/"


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


def try_pmc_full_text(pmid: str) -> Optional[str]:
    """Try to get full text from PMC if available."""
    pmc_id = get_pmc_id_from_pmid(pmid)
    if not pmc_id:
        return None
    
    try:
        # Try to get full text XML from PMC
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml"
        }
        r = requests.get(BASE_URL + "efetch.fcgi", params=params)
        r.raise_for_status()
        
        # Parse XML and extract text content
        root = ET.fromstring(r.text)
        
        # Extract all text from body sections
        text_parts = []
        
        # Get abstract if available
        abstract = root.findtext(".//abstract")
        if abstract:
            text_parts.append(f"Abstract: {abstract}")
        
        # Get full body text
        for section in root.findall(".//sec"):
            title = section.findtext(".//title")
            if title:
                text_parts.append(f"\n{title}\n")
            
            # Get all paragraph text
            for p in section.findall(".//p"):
                if p.text:
                    text_parts.append(p.text)
        
        if text_parts:
            return "\n".join(text_parts)
        
        return None
    except Exception:
        return None


def try_unpaywall(doi: str, email: str = "research@example.com") -> Optional[str]:
    """Try to find open access version via Unpaywall."""
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


def try_europe_pmc_full_text(pmid: str) -> Optional[str]:
    """Try to get full text from Europe PMC."""
    try:
        # First check if full text is available
        url = f"{EUROPE_PMC_API}search?query=ext_id:{pmid}&resulttype=core&format=json"
        r = requests.get(url)
        r.raise_for_status()
        
        data = r.json()
        if data.get("resultList", {}).get("result"):
            result = data["resultList"]["result"][0]
            if result.get("hasTextMinedTerms") == "Y" or result.get("isOpenAccess") == "Y":
                # Try to get full text
                pmcid = result.get("pmcid")
                if pmcid:
                    full_text_url = f"{EUROPE_PMC_API}{pmcid}/fullTextXML"
                    r = requests.get(full_text_url)
                    if r.status_code == 200:
                        # Parse XML and extract text
                        root = ET.fromstring(r.text)
                        text_parts = []
                        
                        # Extract body content
                        for p in root.findall(".//p"):
                            if p.text:
                                text_parts.append(p.text)
                        
                        if text_parts:
                            return "\n".join(text_parts)
        
        return None
    except Exception:
        return None


def enhanced_fetch_abstracts(ids: List[str], try_full_text: bool = True) -> List[Dict]:
    """Enhanced version that tries to get full text when available."""
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
        if try_full_text and pmid:
            # Try PMC first
            full_text = try_pmc_full_text(pmid)
            if full_text:
                full_text_source = "PMC"
            elif doi:
                # Try Unpaywall
                unpaywall_url = try_unpaywall(doi)
                if unpaywall_url:
                    full_text_source = f"Open Access: {unpaywall_url}"
                else:
                    # Try Europe PMC
                    full_text = try_europe_pmc_full_text(pmid)
                    if full_text:
                        full_text_source = "Europe PMC"
        
        # Create availability note
        if full_text:
            availability_note = f"\n\n[FULL TEXT AVAILABLE from {full_text_source}]"
        else:
            availability_note = (
                "\n\n[NOTE: Only abstract available through PubMed API. "
                "For full text, try institutional access or check if paper is open access.]"
            )
        
        record = {
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
            "availability_note": availability_note,
            "has_full_text": bool(full_text)
        }
        
        records.append(record)
    
    return records


# Update the original fetch_abstracts to use enhanced version
def fetch_abstracts(ids: List[str]):
    """Wrapper for backward compatibility."""
    return enhanced_fetch_abstracts(ids, try_full_text=True)


if __name__ == "__main__":
    # Test with a known open access paper
    test_ids = ["33301246"]  # This should be an open access COVID paper
    results = enhanced_fetch_abstracts(test_ids)
    
    for result in results:
        print(f"Title: {result['title']}")
        print(f"Has full text: {result['has_full_text']}")
        if result['has_full_text']:
            print(f"Full text source: {result['full_text_source']}")
            print(f"Full text preview: {result['full_text'][:500]}...")
        print("---")

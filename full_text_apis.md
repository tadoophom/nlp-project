# Full Text Academic Paper APIs

## Free/Open Access APIs

### 1. Europe PMC API
- **URL**: https://europepmc.org/docs/EBI_Europe_PMC_Web_Service_Reference.pdf
- **What it provides**: Full text for open access papers
- **Usage**: Can get full text XML/HTML for papers with open access licenses
- **Example**: 
  ```
  https://www.ebi.ac.uk/europepmc/webservices/rest/PMC123456/fullTextXML
  ```

### 2. NCBI PMC (PubMed Central) API
- **URL**: https://www.ncbi.nlm.nih.gov/pmc/tools/oa-service/
- **What it provides**: Full text XML for open access papers in PMC
- **Usage**: Only works for papers in PMC with open access
- **Example**:
  ```
  https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id=PMC123456
  ```

### 3. arXiv API
- **URL**: https://arxiv.org/help/api
- **What it provides**: Full text PDFs for preprints
- **Usage**: Great for physics, math, CS, biology preprints
- **Example**:
  ```python
  import arxiv
  search = arxiv.Search(query="machine learning")
  for paper in search.results():
      paper.download_pdf(filename=f"{paper.entry_id}.pdf")
  ```

### 4. bioRxiv/medRxiv API
- **URL**: https://api.biorxiv.org/
- **What it provides**: Full text PDFs for biology/medicine preprints
- **Example**:
  ```
  https://api.biorxiv.org/details/biorxiv/2019-12-11
  ```

### 5. DOAJ (Directory of Open Access Journals) API
- **URL**: https://doaj.org/api/v2/docs
- **What it provides**: Metadata and links to open access papers
- **Usage**: Can find open access papers and get direct links

## Commercial/Restricted APIs

### 6. Crossref API
- **URL**: https://api.crossref.org/
- **What it provides**: Metadata + links to publisher sites
- **Usage**: Free for metadata, but full text requires publisher access
- **Note**: Some papers have "license-URL" that might indicate open access

### 7. Semantic Scholar API
- **URL**: https://api.semanticscholar.org/
- **What it provides**: Abstracts + some full text for open access papers
- **Usage**: Good for finding related papers and abstracts

### 8. Microsoft Academic API (Discontinued)
- **Status**: No longer available as of 2021

### 9. Publisher-Specific APIs
- **Springer Nature**: https://dev.springernature.com/
- **Elsevier Scopus**: https://dev.elsevier.com/
- **Wiley**: Limited API access
- **Note**: Require institutional subscriptions for full text

## Legal Alternatives

### 10. Sci-Hub (Not recommended)
- **Status**: Copyright violation, not legal
- **Note**: While it exists, using it violates copyright laws

### 11. Unpaywall API
- **URL**: https://unpaywall.org/products/api
- **What it provides**: Finds free/open versions of papers
- **Usage**: Legal way to find open access versions
- **Example**:
  ```
  https://api.unpaywall.org/v2/10.1038/nature12373?email=your@email.com
  ```

## Implementation Strategy

### For Your Application:
1. **Start with PubMed Central**: Check if papers are in PMC with open access
2. **Use Unpaywall**: Find legal open access versions
3. **Try Europe PMC**: Good for European papers
4. **arXiv for preprints**: If dealing with recent research
5. **Fall back to abstracts**: When full text isn't available

### Sample Implementation:
```python
def get_full_text(pmid, doi=None):
    # 1. Try PMC first
    pmc_text = try_pmc_full_text(pmid)
    if pmc_text:
        return pmc_text
    
    # 2. Try Unpaywall
    if doi:
        unpaywall_url = try_unpaywall(doi)
        if unpaywall_url:
            return download_from_url(unpaywall_url)
    
    # 3. Try Europe PMC
    europe_pmc_text = try_europe_pmc(pmid)
    if europe_pmc_text:
        return europe_pmc_text
    
    # 4. Fall back to abstract only
    return get_abstract_only(pmid)
```

## Legal Considerations
- **Copyright**: Most academic papers are copyrighted
- **Fair Use**: Research purposes may qualify for fair use
- **Institutional Access**: Universities often have subscriptions
- **Open Access**: Only these are freely available
- **Preprints**: Usually freely available

## Limitations
- Only ~30% of papers are open access
- Most recent papers are behind paywalls
- Publisher APIs require institutional subscriptions
- Rate limits on free APIs
- Legal restrictions on redistribution

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.pubmed_fetch import EUROPE_PMC_API, fetch_abstracts


class MockResponse:
    """Minimal mock for requests.Response used in tests."""

    def __init__(self, *, text: str = "", status_code: int = 200, json_data: Dict[str, Any] | None = None):
        self.text = text
        self.status_code = status_code
        self._json_data = json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP error {self.status_code}")

    def json(self) -> Dict[str, Any]:
        if self._json_data is None:
            raise ValueError("JSON not available")
        return self._json_data


def test_fetch_abstracts_uses_europe_pmc_full_text(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify Europe PMC fallback populates full text details when PMC is unavailable."""

    abstract_xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>Sample Study</ArticleTitle>
            <Abstract>
              <AbstractText>Abstract body.</AbstractText>
            </Abstract>
            <Journal>
              <Title>Test Journal</Title>
            </Journal>
            <ArticleIdList>
              <ArticleId IdType=\"doi\">10.1000/test</ArticleId>
            </ArticleIdList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """.strip()

    pmid_lookup_xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType=\"pubmed\">12345</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>
    """.strip()

    europe_search_payload = {
        "resultList": {
            "result": [
                {
                    "pmcid": "PMC9999999",
                    "title": "Sample Study",
                }
            ]
        }
    }

    europe_full_text_xml = """
    <article>
      <body>
        <sec>
          <title>Introduction</title>
          <p>Introduction paragraph.</p>
        </sec>
        <sec>
          <title>Methods</title>
          <p>Methods paragraph.</p>
        </sec>
      </body>
    </article>
    """.strip()

    call_state = {"efetch": 0}

    def mock_get(url: str, params: Dict[str, Any] | None = None, timeout: int | None = None, **_: Any) -> MockResponse:
        if "efetch.fcgi" in url:
            if call_state["efetch"] == 0:
                call_state["efetch"] += 1
                return MockResponse(text=abstract_xml)
            return MockResponse(text=pmid_lookup_xml)
        if url == EUROPE_PMC_API + "search":
            return MockResponse(text=json.dumps(europe_search_payload), json_data=europe_search_payload)
        if url == EUROPE_PMC_API + "PMC9999999/fullTextXML":
            return MockResponse(text=europe_full_text_xml)
        raise AssertionError(f"Unexpected URL called: {url}")

    monkeypatch.setattr("src.pubmed_fetch.requests.get", mock_get)

    records = fetch_abstracts(["12345"], try_full_text=True)

    assert len(records) == 1
    record = records[0]

    assert record["full_text_source"] == "Europe PMC (Open Access)"
    assert record["has_full_text"] is True
    assert "FULL TEXT RETRIEVED from Europe PMC (Open Access)" in record["full_text_note"]
    assert "Introduction paragraph." in record["full_text"]
    assert "Methods paragraph." in record["full_text"]
    assert record["full_text_url"] == "https://europepmc.org/article/PMC/PMC9999999"

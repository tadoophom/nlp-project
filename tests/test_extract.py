"""
Test suite for NLP extraction and API endpoints.

These tests verify that the keyword extractor correctly detects negation
and that the FastAPI service produces consistent results. Run with
pytest: ``pytest -q``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

from src import nlp_utils
from src.api import app


def test_extract_positive_and_negative():
    nlp = nlp_utils.load_pipeline("en_core_web_sm", gpu=False)
    text = "He denied having asthma. He has asthma."
    results = nlp_utils.extract(text, ["asthma"], nlp)
    assert len(results) == 2
    classifications = [r["classification"] for r in results]
    assert "Negative" in classifications
    assert "Positive" in classifications


def test_api_analyze_endpoint():
    client = TestClient(app)
    payload = {"text": "Patient denies fever but has cough.", "keywords": ["fever", "cough"]}
    response = client.post("/analyze", json=payload)
    assert response.status_code == 200
    body = response.json()
    hits = body.get("hits")
    assert isinstance(hits, list)
    # Should contain 2 hits: fever (Negative) and cough (Positive)
    kw_map = {h["keyword"]: h["classification"] for h in hits}
    assert kw_map.get("fever") == "Negative"
    assert kw_map.get("cough") == "Positive"
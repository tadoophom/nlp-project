"""
api.py – FastAPI service for the clinical keyword polarity suite.

This module exposes a simple RESTful interface around the NLP utilities
implemented in ``nlp_utils.py``. Clients can submit text and keywords and
receive structured JSON describing each keyword hit, including its
sentence context, part‑of‑speech tag, dependency relation and basic
polarity classification.

The FastAPI application is intended to be used headlessly in pipelines
and microservices. The Streamlit UI defined in ``app.py`` builds on top
of the same utilities for interactive exploration.
"""

from __future__ import annotations

from typing import List

from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field

from nlp_utils import load_pipeline, extract


class AnalysisRequest(BaseModel):
    text: str = Field(..., description="The raw text to analyse.")
    keywords: List[str] = Field(..., description="A list of keywords to search for.")
    model: str = Field("en_core_web_sm", description="spaCy model identifier.")
    use_gpu: bool = Field(False, description="Whether to use GPU acceleration.")


class KeywordHit(BaseModel):
    keyword: str
    sentence: str
    sent_index: int
    token_index: int
    pos: str
    dep: str
    classification: str


class AnalysisResponse(BaseModel):
    hits: List[KeywordHit]


app = FastAPI(title="Clinical Keyword Polarity API", version="0.1.0")


def get_nlp(req: AnalysisRequest):
    """Dependency that lazily loads and caches the spaCy pipeline."""
    return load_pipeline(req.model, gpu=req.use_gpu)


@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}


@app.get("/models", summary="List available spaCy models")
async def models():
    # Minimal list; frontends can map to labels
    return {
        "models": [
            "en_core_web_sm",
            "en_core_web_md",
            "en_core_sci_md",
            "es_core_news_md",
            "fr_core_news_md",
        ]
    }


@app.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze(req: AnalysisRequest, nlp=Depends(get_nlp)):
    """
    Analyse a document for occurrences of the supplied keywords and return
    structured information about each hit.

    The caller should provide a plain text string and a list of lower‑case
    keywords. An optional spaCy model name and GPU flag may also be
    specified. The response contains an array of hit objects with the
    original keyword (lemma), its surrounding sentence, positional indices
    and a naive polarity classification.
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text must not be empty")
    if not req.keywords:
        raise HTTPException(status_code=400, detail="Keywords must be provided")
    # Normalise keywords
    terms = [k.strip().lower() for k in req.keywords if k.strip()]
    hits = extract(text, terms, nlp)
    return AnalysisResponse(hits=hits)
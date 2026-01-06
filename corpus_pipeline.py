"""Utility module/CLI for building disease-protein corpora with sentiment labels.

This script orchestrates PubMed querying using keyword/MeSH combinations and
performs sentence-level polarity detection between a disease concept and a
protein list. Results are exported as a CSV corpus that can drive downstream
analysis or manual review.
"""
from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zipfile import ZipFile

from nlp_utils import extract, load_pipeline
from pubmed_fetch import fetch_abstracts, search_pubmed_advanced
import requests


@dataclass
class ProteinEntry:
    """Representation of a protein row loaded from the user-provided table."""

    identifier: str
    score: Optional[float] = None
    raw_row: Optional[Dict[str, str]] = None


def _format_term(term: str) -> str:
    term = term.strip()
    if not term:
        return term
    # Always wrap with quotes unless already quoted to preserve multi-word terms
    if not (term.startswith("\"") and term.endswith("\"")):
        term = term.replace("\"", "")
        term = f'"{term}"'
    return term


def _build_field_clauses(terms: Sequence[str], field_tag: str) -> List[str]:
    clauses: List[str] = []
    for term in terms:
        formatted = _format_term(term)
        if formatted:
            clauses.append(f"{formatted}[{field_tag}]")
    return clauses


def compose_query(
    *,
    disease_keywords: Sequence[str],
    disease_mesh: Sequence[str],
    protein_terms: Sequence[str],
    additional_keywords: Sequence[str],
    additional_mesh: Sequence[str],
    logic: str,
) -> str:
    """Compose a PubMed raw query string using disease and protein groups."""

    disease_group_parts = []
    disease_kw = _build_field_clauses(disease_keywords, "Title/Abstract")
    disease_mesh_clauses = _build_field_clauses(disease_mesh, "MeSH Terms")
    if disease_kw:
        disease_group_parts.append(" OR ".join(disease_kw))
    if disease_mesh_clauses:
        disease_group_parts.append(" OR ".join(disease_mesh_clauses))
    disease_group = (
        f"({ ' OR '.join(disease_group_parts) })" if disease_group_parts else ""
    )

    protein_group_parts = []
    protein_kw = _build_field_clauses(protein_terms, "Title/Abstract")
    if protein_kw:
        protein_group_parts.append(" OR ".join(protein_kw))
    protein_group = (
        f"({ ' OR '.join(protein_group_parts) })" if protein_group_parts else ""
    )

    extra_group_parts = []
    extra_kw = _build_field_clauses(additional_keywords, "Title/Abstract")
    extra_mesh_clauses = _build_field_clauses(additional_mesh, "MeSH Terms")
    if extra_kw:
        extra_group_parts.append(" OR ".join(extra_kw))
    if extra_mesh_clauses:
        extra_group_parts.append(" OR ".join(extra_mesh_clauses))
    extra_group = (
        f"({ ' OR '.join(extra_group_parts) })" if extra_group_parts else ""
    )

    groups = [g for g in [disease_group, protein_group, extra_group] if g]
    if not groups:
        raise ValueError("At least one of disease, protein or extra group must be provided")

    logic = logic.lower()
    if logic == "disease_and_protein":
        if not disease_group or not protein_group:
            raise ValueError("disease_and_protein logic requires both disease and protein inputs")
        base = f"{disease_group} AND {protein_group}"
        if extra_group:
            return f"({base}) AND {extra_group}"
        return base
    if logic == "all_and":
        return " AND ".join(groups)
    if logic == "all_or":
        return " OR ".join(groups)
    if logic == "disease_or_protein":
        if not disease_group or not protein_group:
            raise ValueError("disease_or_protein logic requires both disease and protein inputs")
        combo = f"({disease_group}) OR ({protein_group})"
        if extra_group:
            return f"({combo}) AND {extra_group}" if extra_group else combo
        return combo
    raise ValueError(
        "Unsupported logic value. Choose from disease_and_protein, all_and, all_or, disease_or_protein."
    )


def _excel_column_index(cell_ref: str) -> int:
    col = "".join(ch for ch in cell_ref if ch.isalpha())
    idx = 0
    for char in col:
        idx = idx * 26 + (ord(char.upper()) - ord("A") + 1)
    return idx - 1


def _read_xlsx_without_dependencies(path: Path) -> Tuple[List[str], List[List[str]]]:
    with ZipFile(path) as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            import xml.etree.ElementTree as ET

            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            for si in root.findall("s:si", ns):
                text = ""
                t_el = si.find("s:t", ns)
                if t_el is not None and t_el.text:
                    text = t_el.text
                else:
                    # Handle rich text runs if present
                    parts = [node.text or "" for node in si.findall("s:r/s:t", ns)]
                    text = "".join(parts)
                shared_strings.append(text)
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")
        import xml.etree.ElementTree as ET

        sheet_root = ET.fromstring(sheet_xml)
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        rows: List[Dict[int, str]] = []
        max_col_index = 0
        for row in sheet_root.findall(".//s:row", ns):
            values: Dict[int, str] = {}
            for c in row.findall("s:c", ns):
                cell_ref = c.get("r", "")
                idx = _excel_column_index(cell_ref)
                max_col_index = max(max_col_index, idx)
                value_el = c.find("s:v", ns)
                if value_el is None:
                    continue
                text = value_el.text or ""
                if c.get("t") == "s":
                    try:
                        text = shared_strings[int(text)]
                    except (IndexError, ValueError):
                        text = ""
                values[idx] = text
            rows.append(values)
        if not rows:
            return [], []
        headers = [rows[0].get(i, "") for i in range(max_col_index + 1)]
        data_rows: List[List[str]] = []
        for row in rows[1:]:
            data_rows.append([row.get(i, "") for i in range(max_col_index + 1)])
        return headers, data_rows


def load_protein_entries(
    table_path: Path,
    *,
    identifier_column: Optional[str] = None,
    score_column: Optional[str] = None,
    top_n: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[ProteinEntry]:
    """Load protein identifiers and optional scores from CSV or XLSX."""

    if not table_path.exists():
        raise FileNotFoundError(f"Protein table not found: {table_path}")

    rows: List[Dict[str, str]] = []
    headers: List[str] = []
    if table_path.suffix.lower() == ".csv":
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append({k: (v or "").strip() for k, v in row.items()})
            headers = reader.fieldnames or []
    else:
        df = None
        try:
            import pandas as pd  # type: ignore
        except ModuleNotFoundError:
            pd = None  # type: ignore[assignment]
        else:
            try:
                df = pd.read_excel(table_path)
            except (ImportError, ValueError):
                df = None
        if df is not None:
            headers = [str(col) for col in df.columns]
            rows = [
                {
                    str(col): (
                        ""
                        if (
                            val is None
                            or (isinstance(val, float) and math.isnan(val))
                        )
                        else str(val)
                    )
                    for col, val in row.items()
                }
                for row in df.to_dict("records")
            ]
        else:
            headers, plain_rows = _read_xlsx_without_dependencies(table_path)
            for cells in plain_rows:
                row = {headers[i]: cells[i] for i in range(len(headers)) if headers[i]}
                rows.append({k: v.strip() for k, v in row.items()})

    if not rows:
        return []

    identifier_column = identifier_column or (headers[0] if headers else None)
    if not identifier_column:
        raise ValueError("Could not infer identifier column from table header")
    if identifier_column not in (headers or []):
        raise ValueError(f"Identifier column '{identifier_column}' not found in {headers}")

    if score_column and score_column not in (headers or []):
        raise ValueError(f"Score column '{score_column}' not present in table")

    entries: List[ProteinEntry] = []
    for row in rows:
        identifier = row.get(identifier_column, "").strip()
        if not identifier:
            continue
        score: Optional[float] = None
        if score_column:
            raw_score = row.get(score_column, "").strip()
            if raw_score:
                try:
                    score = float(raw_score)
                except ValueError:
                    score = None
        elif len(headers) > 1:
            # Try to interpret second column as score when not explicitly provided
            secondary_header = headers[1]
            raw_score = row.get(secondary_header, "").strip()
            if raw_score:
                try:
                    score = float(raw_score)
                except ValueError:
                    score = None
        entries.append(ProteinEntry(identifier=identifier, score=score, raw_row=row))

    if min_score is not None:
        entries = [entry for entry in entries if (entry.score is not None and entry.score >= min_score)]

    if top_n is not None and top_n > 0:
        entries = sorted(entries, key=lambda e: (e.score is not None, e.score), reverse=True)[:top_n]

    return entries


UNIPROT_RE = re.compile(r"^[A-NR-Z][0-9][A-Z0-9]{3}[0-9]$|^[OPQ][0-9][A-Z0-9]{3}[0-9]$", re.I)
UNIPROT_API = "https://rest.uniprot.org/uniprotkb/{}?format=json"


def _is_uniprot(identifier: str) -> bool:
    return bool(UNIPROT_RE.match(identifier.strip()))


@lru_cache(maxsize=512)
def _fetch_uniprot_metadata(accession: str, *, timeout: float = 10.0) -> Optional[Dict[str, object]]:
    try:
        response = requests.get(UNIPROT_API.format(accession), timeout=timeout)
        if response.status_code != 200:
            return None
        return response.json()
    except Exception:
        return None


def expand_protein_terms(identifier: str) -> List[str]:
    """Return a list of synonyms for a protein identifier.

    Falls back to the raw identifier when no metadata is available.
    """

    base = identifier.strip()
    if not base:
        return []

    synonyms: List[str] = [base]
    if not _is_uniprot(base):
        return synonyms

    meta = _fetch_uniprot_metadata(base)
    if not meta:
        return synonyms

    seen = {base.lower()}
    def _add(term: Optional[str]) -> None:
        if not term:
            return
        cleaned = term.strip()
        if not cleaned:
            return
        lowered = cleaned.lower()
        if lowered in seen:
            return
        seen.add(lowered)
        synonyms.append(cleaned)

    # UniProt entry name
    _add(meta.get("uniProtkbId"))

    # Gene symbols and synonyms
    for gene in meta.get("genes", []) or []:
        if isinstance(gene, dict):
            _add(gene.get("geneName", {}).get("value"))
            for syn in gene.get("synonyms", []) or []:
                if isinstance(syn, dict):
                    _add(syn.get("value"))

    # Recommended and alternative protein names
    protein_desc = meta.get("proteinDescription", {}) or {}
    rec_name = protein_desc.get("recommendedName", {}) if isinstance(protein_desc, dict) else {}

    if isinstance(rec_name, dict):
        _add(rec_name.get("fullName", {}).get("value") if isinstance(rec_name.get("fullName"), dict) else rec_name.get("value"))

    for alt in protein_desc.get("alternativeNames", []) or []:
        if isinstance(alt, dict):
            alt_full = alt.get("fullName", {})
            if isinstance(alt_full, dict):
                _add(alt_full.get("value"))
            else:
                _add(alt_full)

    return synonyms


def _normalise_terms(values: Iterable[str]) -> List[str]:
    return [v.strip().lower() for v in values if v and v.strip()]


def detect_relation(
    text: str,
    *,
    disease_terms: Sequence[str],
    protein_terms: Sequence[str],
    nlp_model,
    model_name: str,
) -> Dict[str, Optional[str]]:
    """Detect relation polarity between disease and protein mentions within text."""

    terms = list(disease_terms) + list(protein_terms)
    if not terms:
        return {
            "relation": "Missing terms",
            "evidence_sentence": None,
            "protein_polarity": None,
            "disease_polarity": None,
            "protein_confidence": None,
            "disease_confidence": None,
            "co_mentions": 0,
        }

    hits = extract(text, terms, nlp_model, model_name=model_name)
    if not hits:
        return {
            "relation": "No mentions",
            "evidence_sentence": None,
            "protein_polarity": None,
            "disease_polarity": None,
            "protein_confidence": None,
            "disease_confidence": None,
            "co_mentions": 0,
        }

    disease_set = set(_normalise_terms(disease_terms))
    protein_set = set(_normalise_terms(protein_terms))

    by_sentence: Dict[int, Dict[str, List[Dict[str, object]]]] = {}
    for hit in hits:
        keyword = str(hit.get("keyword", "")).lower()
        sent_idx = int(hit.get("sent_index", -1))
        if sent_idx < 0:
            continue
        bucket = by_sentence.setdefault(sent_idx, {"disease": [], "protein": []})
        if keyword in disease_set:
            bucket["disease"].append(hit)
        if keyword in protein_set:
            bucket["protein"].append(hit)

    overlapping = [bucket for bucket in by_sentence.values() if bucket["disease"] and bucket["protein"]]
    if not overlapping:
        return {
            "relation": "No co-mention",
            "evidence_sentence": None,
            "protein_polarity": None,
            "disease_polarity": None,
            "protein_confidence": None,
            "disease_confidence": None,
            "co_mentions": 0,
        }

    def pick_priority_bucket() -> Dict[str, List[Dict[str, object]]]:
        # Negative polarity takes precedence, then positive, else first bucket
        for bucket in overlapping:
            if any(hit.get("classification") == "Negative" for hit in bucket["disease"] + bucket["protein"]):
                return bucket
        for bucket in overlapping:
            if any(hit.get("classification") == "Positive" for hit in bucket["disease"] + bucket["protein"]):
                return bucket
        return overlapping[0]

    selected = pick_priority_bucket()
    combined = selected["disease"] + selected["protein"]
    polarity_values = [str(hit.get("classification")) for hit in combined]
    if any(value == "Negative" for value in polarity_values):
        relation = "Negative"
    elif any(value == "Positive" for value in polarity_values):
        relation = "Positive"
    else:
        relation = "Neutral"

    sentence = str(selected["disease"][0].get("sentence")) if selected["disease"] else str(selected["protein"][0].get("sentence"))

    # Derive representative polarity/confidence for reporting
    protein_hit = selected["protein"][0] if selected["protein"] else None
    disease_hit = selected["disease"][0] if selected["disease"] else None

    return {
        "relation": relation,
        "evidence_sentence": sentence,
        "protein_polarity": str(protein_hit.get("classification")) if protein_hit else None,
        "disease_polarity": str(disease_hit.get("classification")) if disease_hit else None,
        "protein_confidence": float(protein_hit.get("confidence", 0.0)) if protein_hit else None,
        "disease_confidence": float(disease_hit.get("confidence", 0.0)) if disease_hit else None,
        "co_mentions": len(overlapping),
    }


def build_corpus(
    *,
    protein_entries: Sequence[ProteinEntry],
    disease_keywords: Sequence[str],
    disease_mesh: Sequence[str],
    additional_keywords: Sequence[str],
    additional_mesh: Sequence[str],
    logic: str,
    spaCy_model: str,
    retmax: int,
) -> List[Dict[str, object]]:
    """Build corpus rows for provided proteins."""

    nlp_model = load_pipeline(spaCy_model, gpu=False, use_context=True)
    results: List[Dict[str, object]] = []

    for entry in protein_entries:
        protein_terms = expand_protein_terms(entry.identifier)
        if not protein_terms:
            continue
        query = compose_query(
            disease_keywords=disease_keywords,
            disease_mesh=disease_mesh,
            protein_terms=protein_terms,
            additional_keywords=additional_keywords,
            additional_mesh=additional_mesh,
            logic=logic,
        )
        ids, actual_query = search_pubmed_advanced(
            keywords=[],
            mesh_terms=[],
            retmax=retmax,
            search_logic="OR",
            raw_query=query,
        )
        if not ids:
            continue
        articles = fetch_abstracts(ids)
        expanded_disease_terms = list(disease_keywords) + list(disease_mesh)

        for article in articles:
            abstract_text = article.get("abstract") or ""
            relation = detect_relation(
                abstract_text,
                disease_terms=expanded_disease_terms,
                protein_terms=protein_terms,
                nlp_model=nlp_model,
                model_name=spaCy_model,
            )
            row = {
                "protein": entry.identifier,
                "protein_terms_used": ";".join(protein_terms),
                "protein_score": entry.score,
                "pmid": article.get("pmid"),
                "title": article.get("title"),
                "abstract": abstract_text,
                "journal": article.get("journal"),
                "year": article.get("year"),
                "mesh_terms": article.get("mesh_terms"),
                "full_text_source": article.get("full_text_source"),
                "query": actual_query,
                "relation": relation["relation"],
                "evidence_sentence": relation["evidence_sentence"],
                "protein_polarity": relation["protein_polarity"],
                "disease_polarity": relation["disease_polarity"],
                "protein_confidence": relation["protein_confidence"],
                "disease_confidence": relation["disease_confidence"],
                "co_mentions": relation["co_mentions"],
            }
            results.append(row)
    return results


def export_corpus(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a disease-protein sentiment corpus from PubMed abstracts.")
    parser.add_argument("--protein-file", required=True, help="Path to CSV/XLSX table containing protein identifiers")
    parser.add_argument("--identifier-column", help="Column name for protein identifiers (defaults to first column)")
    parser.add_argument("--score-column", help="Optional column to carry score or weight values")
    parser.add_argument("--top-n", type=int, help="Limit to top N proteins based on score (requires score column)")
    parser.add_argument("--min-score", type=float, help="Minimum score threshold for proteins")
    parser.add_argument("--disease-keyword", action="append", dest="disease_keywords", default=[], help="Disease keyword (Title/Abstract). Repeat for multiple terms.")
    parser.add_argument("--disease-mesh", action="append", dest="disease_mesh", default=[], help="Disease MeSH term. Repeat for multiple terms.")
    parser.add_argument("--extra-keyword", action="append", dest="extra_keywords", default=[], help="Additional keyword applied to all queries")
    parser.add_argument("--extra-mesh", action="append", dest="extra_mesh", default=[], help="Additional MeSH term applied to all queries")
    parser.add_argument("--logic", default="disease_and_protein", choices=["disease_and_protein", "all_and", "all_or", "disease_or_protein"], help="Boolean combination to use for composed query")
    parser.add_argument("--retmax", type=int, default=50, help="Maximum PubMed results per protein")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model name for polarity detection")
    parser.add_argument("--output", required=True, help="Output CSV path for the corpus")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_path = Path(args.output)
    protein_path = Path(args.protein_file)
    proteins = load_protein_entries(
        protein_path,
        identifier_column=args.identifier_column,
        score_column=args.score_column,
        top_n=args.top_n,
        min_score=args.min_score,
    )
    if not proteins:
        raise SystemExit("No protein entries found after applying filters")

    rows = build_corpus(
        protein_entries=proteins,
        disease_keywords=args.disease_keywords,
        disease_mesh=args.disease_mesh,
        additional_keywords=args.extra_keywords,
        additional_mesh=args.extra_mesh,
        logic=args.logic,
        spaCy_model=args.spacy_model,
        retmax=args.retmax,
    )

    export_corpus(rows, output_path)
    print(f"Exported {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

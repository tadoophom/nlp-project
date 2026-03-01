from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pubmed_fetch import search_pubmed_advanced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search PubMed for HFpEF abstracts with negative-result cues and export PMIDs as JSON")
    parser.add_argument("--out-pmid-json", required=True, help="Output JSON path for PMID list")
    parser.add_argument("--retmax", type=int, default=5000, help="Maximum PubMed results")
    parser.add_argument("--date-query", default="", help="Optional PubMed date query, e.g. 2015:2026[PDAT]")
    return parser.parse_args()


def main() -> None:
    hf_terms = [
        "HFpEF",
        "HF-PEF",
        "HFNEF",
        "heart failure with preserved ejection fraction",
        "heart failure with normal ejection fraction",
        "diastolic heart failure",
        "diastolic dysfunction",
        "preserved ejection fraction",
        "normal ejection fraction",
        "preserved EF",
        "normal EF",
        "HF with preserved EF",
        "HF with normal EF",
    ]

    negative_cues = [
        "not associated",
        "no association",
        "no significant association",
        "not significantly associated",
        "no correlation",
        "no significant correlation",
        "does not correlate",
        "did not correlate",
        "no significant difference",
        "no differences",
        "no difference",
        "did not differ",
        "not predictive",
        "did not predict",
        "failed to predict",
        "no effect",
        "no significant effect",
    ]

    hf_ta = " OR ".join([f"\"{t}\"[Title/Abstract]" for t in hf_terms])
    hf_mesh = "\"Heart Failure, Diastolic\"[MeSH Terms]"
    hf_query = f"({hf_ta}) OR ({hf_mesh})"
    neg_query = " OR ".join([f"\"{t}\"[Title/Abstract]" for t in negative_cues])
    raw_query = f"({hf_query}) AND ({neg_query})"

    args = parse_args()
    ids, actual_query = search_pubmed_advanced(
        keywords=[],
        mesh_terms=[],
        retmax=args.retmax,
        search_logic="OR",
        date_query=args.date_query,
        raw_query=raw_query,
    )

    out_path = Path(args.out_pmid_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([str(p) for p in ids], indent=2))

    print("pmids:", len(ids))
    print("query:", actual_query)
    print("out:", str(out_path))


if __name__ == "__main__":
    main()

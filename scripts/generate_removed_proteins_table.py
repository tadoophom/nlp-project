"""
Generate a table of removed proteins with their functions and rationale.

Compares baseline CaseOLAP rankings with filtered rankings to identify
proteins removed during sentiment filtering, then queries UniProt for
protein names and functions.

Usage:
    python scripts/generate_removed_proteins_table.py
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

DATA_DIR = Path(__file__).parent.parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
BASELINE_CSV = PLOTS_DIR / "baseline_rankings.csv"
FILTERED_CSV = PLOTS_DIR / "filtered_rankings.csv"
SENTIMENT_CSV = PLOTS_DIR / "sentiment_summary.csv"
OUTPUT_CSV = PLOTS_DIR / "removed_proteins_table.csv"
OUTPUT_MD = PLOTS_DIR / "removed_proteins_table.md"


def load_proteins_from_csv(path: Path, limit: int = 100) -> List[Tuple[str, float]]:
    proteins = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= limit:
                break
            protein_id = row.get("protein", "").strip()
            score = float(row.get("caseolap_score", 0))
            proteins.append((protein_id, score))
    return proteins


def load_sentiment_summary(path: Path) -> Dict[str, Dict]:
    summary = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            protein_id = row.get("protein", "").strip()
            summary[protein_id] = {
                "negative": int(row.get("Negative", 0)),
                "no_comention": int(row.get("No co-mention", 0)),
                "no_mentions": int(row.get("No mentions", 0)),
                "positive": int(row.get("Positive", 0)),
                "sentiment_label": row.get("sentiment_label", "neutral"),
            }
    return summary


def fetch_uniprot_info(protein_id: str) -> Dict[str, str]:
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        protein_name = "Unknown"
        if "proteinDescription" in data:
            desc = data["proteinDescription"]
            if "recommendedName" in desc and "fullName" in desc["recommendedName"]:
                protein_name = desc["recommendedName"]["fullName"].get("value", "Unknown")
            elif "submissionNames" in desc and desc["submissionNames"]:
                protein_name = desc["submissionNames"][0].get("fullName", {}).get("value", "Unknown")
        
        gene_name = ""
        if "genes" in data and data["genes"]:
            gene_data = data["genes"][0]
            if "geneName" in gene_data:
                gene_name = gene_data["geneName"].get("value", "")
        
        function = "No function annotation available"
        if "comments" in data:
            for comment in data["comments"]:
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function = texts[0].get("value", function)
                        break
        
        return {
            "protein_name": protein_name,
            "gene_name": gene_name,
            "function": function,
        }
    except Exception as e:
        return {
            "protein_name": "Error fetching",
            "gene_name": "",
            "function": f"Error: {str(e)}",
        }


def determine_removal_rationale(sentiment_data: Dict) -> str:
    if not sentiment_data:
        return "Not analyzed in sentiment corpus"
    
    label = sentiment_data.get("sentiment_label", "neutral")
    positive = sentiment_data.get("positive", 0)
    negative = sentiment_data.get("negative", 0)
    no_comention = sentiment_data.get("no_comention", 0)
    no_mentions = sentiment_data.get("no_mentions", 0)
    
    if label == "neutral":
        if positive == 0 and negative == 0:
            if no_comention > 0 or no_mentions > 0:
                return "Neutral only: No positive/negative disease-protein co-mentions found"
            return "Neutral only: No mentions in corpus"
        return f"Neutral classification (pos={positive}, neg={negative})"
    
    return f"Retained (label={label})"


def main():
    print("Loading rankings...")
    
    baseline = load_proteins_from_csv(BASELINE_CSV, limit=100)
    baseline_proteins = {p[0]: p[1] for p in baseline}
    
    filtered = load_proteins_from_csv(FILTERED_CSV, limit=100)
    filtered_proteins = set(p[0] for p in filtered)
    
    sentiment_summary = {}
    if SENTIMENT_CSV.exists():
        sentiment_summary = load_sentiment_summary(SENTIMENT_CSV)
    
    removed = []
    for protein_id, score in baseline:
        if protein_id not in filtered_proteins:
            removed.append((protein_id, score))
    
    print(f"Found {len(removed)} removed proteins")
    
    results = []
    for i, (protein_id, score) in enumerate(removed):
        print(f"Fetching info for {protein_id} ({i+1}/{len(removed)})...")
        uniprot_info = fetch_uniprot_info(protein_id)
        sentiment_data = sentiment_summary.get(protein_id, {})
        rationale = determine_removal_rationale(sentiment_data)
        
        results.append({
            "uniprot_id": protein_id,
            "gene_name": uniprot_info["gene_name"],
            "protein_name": uniprot_info["protein_name"],
            "caseolap_score": round(score, 4),
            "positive_mentions": sentiment_data.get("positive", 0),
            "negative_mentions": sentiment_data.get("negative", 0),
            "no_comention": sentiment_data.get("no_comention", 0),
            "sentiment_label": sentiment_data.get("sentiment_label", "not_analyzed"),
            "removal_rationale": rationale,
            "function": uniprot_info["function"][:200] + "..." if len(uniprot_info["function"]) > 200 else uniprot_info["function"],
        })
        
        time.sleep(0.2)
    
    results.sort(key=lambda x: x["caseolap_score"], reverse=True)
    
    print(f"Writing {OUTPUT_CSV}...")
    fieldnames = [
        "uniprot_id", "gene_name", "protein_name", "caseolap_score",
        "positive_mentions", "negative_mentions", "no_comention",
        "sentiment_label", "removal_rationale", "function"
    ]
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Writing {OUTPUT_MD}...")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# Removed Proteins Table\n\n")
        f.write("This table lists proteins that were excluded from the filtered rankings after sentiment analysis.\n")
        f.write("Proteins are removed when they have no positive or negative disease-protein associations in the literature.\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total proteins removed from top-100**: {len(results)}\n")
        f.write(f"- **Primary reason**: Neutral-only mentions (no clear positive/negative association with HFpEF)\n\n")
        f.write("## Removed Proteins Details\n\n")
        
        f.write("| UniProt ID | Gene | Protein Name | CaseOLAP Score | Pos | Neg | Rationale |\n")
        f.write("|------------|------|--------------|----------------|-----|-----|------------|\n")
        
        for r in results:
            protein_name = r["protein_name"][:40] + "..." if len(r["protein_name"]) > 40 else r["protein_name"]
            rationale = r["removal_rationale"][:50] + "..." if len(r["removal_rationale"]) > 50 else r["removal_rationale"]
            f.write(f"| {r['uniprot_id']} | {r['gene_name']} | {protein_name} | {r['caseolap_score']} | {r['positive_mentions']} | {r['negative_mentions']} | {rationale} |\n")
        
        f.write("\n## Protein Functions\n\n")
        for r in results:
            f.write(f"### {r['uniprot_id']} ({r['gene_name']})\n\n")
            f.write(f"**{r['protein_name']}**\n\n")
            f.write(f"- **CaseOLAP Score**: {r['caseolap_score']}\n")
            f.write(f"- **Sentiment**: {r['sentiment_label']} (Pos: {r['positive_mentions']}, Neg: {r['negative_mentions']})\n")
            f.write(f"- **Removal Rationale**: {r['removal_rationale']}\n")
            f.write(f"- **Function**: {r['function']}\n\n")
    
    print(f"Generated: {OUTPUT_CSV}")
    print(f"Generated: {OUTPUT_MD}")


if __name__ == "__main__":
    main()

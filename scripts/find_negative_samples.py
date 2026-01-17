"""Find potential negative/no_association samples from corpus for labeling."""
import csv
import json
import re
from pathlib import Path

# Patterns indicating negative association
NEGATIVE_PATTERNS = [
    r"no significant (association|difference|effect|benefit|change|correlation)",
    r"not (associated|correlated|related|significant|observed)",
    r"did not (show|demonstrate|affect|improve|reduce)",
    r"failed to (show|demonstrate|reach)",
    r"no (benefit|effect|difference|association|correlation)",
    r"lack of (association|correlation|effect|benefit)",
    r"absence of (association|correlation|effect)",
    r"unlikely to (be|have)",
    r"no.*mortality (benefit|reduction|difference)",
    r"not.*predictor",
]

# Patterns indicating no_association (methods, study design, no claims)
NO_ASSOC_PATTERNS = [
    r"^(this|we|the) (retrospective|prospective|single-center|multicenter) study",
    r"^patients (were|who) (included|enrolled|recruited|randomized)",
    r"^(methods|objective|aim|background|purpose):",
    r"search terms included",
    r"were (included|enrolled|randomized|divided|compared)",
    r"data (were|was) (collected|extracted|analyzed)",
]


def find_candidates(corpus_path: Path, existing_labeled: Path) -> dict:
    """Find sentences likely to be negative or no_association."""
    # Load existing labeled sentences to exclude
    existing = set()
    if existing_labeled.exists():
        with open(existing_labeled) as f:
            for item in json.load(f):
                existing.add(item["sentence"].strip()[:100])
    
    negative_candidates = []
    no_assoc_candidates = []
    
    with open(corpus_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = row.get("evidence_sentence", "").strip()
            if not sentence or sentence[:100] in existing:
                continue
            
            sent_lower = sentence.lower()
            
            # Check for negative patterns
            for pattern in NEGATIVE_PATTERNS:
                if re.search(pattern, sent_lower):
                    negative_candidates.append({
                        "sentence": sentence,
                        "protein": row.get("protein", ""),
                        "pmid": row.get("pmid", ""),
                        "pattern": pattern,
                    })
                    break
            else:
                # Check for no_association patterns
                for pattern in NO_ASSOC_PATTERNS:
                    if re.search(pattern, sent_lower):
                        no_assoc_candidates.append({
                            "sentence": sentence,
                            "protein": row.get("protein", ""),
                            "pmid": row.get("pmid", ""),
                            "pattern": pattern,
                        })
                        break
    
    return {
        "negative": negative_candidates,
        "no_association": no_assoc_candidates,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Path to corpus CSV")
    parser.add_argument("--existing", default="data/labeled.json")
    parser.add_argument("--output", default="data/candidates_for_labeling.json")
    parser.add_argument("--limit", type=int, default=50, help="Max candidates per class")
    args = parser.parse_args()
    
    candidates = find_candidates(Path(args.corpus), Path(args.existing))
    
    # Deduplicate and limit
    seen = set()
    for key in candidates:
        unique = []
        for c in candidates[key]:
            if c["sentence"][:100] not in seen:
                seen.add(c["sentence"][:100])
                unique.append(c)
        candidates[key] = unique[:args.limit]
    
    print(f"Found {len(candidates['negative'])} negative candidates")
    print(f"Found {len(candidates['no_association'])} no_association candidates")
    
    # Save for review
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(candidates, f, indent=2)
    print(f"Saved to {output_path}")
    
    # Print samples
    print("\n=== NEGATIVE CANDIDATES (sample) ===")
    for c in candidates["negative"][:5]:
        print(f"- [{c['pattern']}] {c['sentence'][:120]}...")
    
    print("\n=== NO_ASSOCIATION CANDIDATES (sample) ===")
    for c in candidates["no_association"][:5]:
        print(f"- [{c['pattern']}] {c['sentence'][:120]}...")


if __name__ == "__main__":
    main()

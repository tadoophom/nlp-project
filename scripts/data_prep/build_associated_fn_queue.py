"""Build targeted labeling queue for associated sentences the v7 model misclassifies as incidental.

The v7 model handles explicit "X was associated with Y" well, but fails on:
1. Mechanistic language: protects, leads to, arises from, impairs, improves
2. Aim/hypothesis framing: aimed to, hypothesized, investigated whether
3. Implicit association: found in, involved in, suggested to be involved
4. Prognostic/predictive: prognostic marker, predictor of, predictive value
5. Biomarker listing: biomarkers include X, Y, Z
6. Therapeutic impact: improved function, led to reduction, favourable effects
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

# --- Protein detection ---
PROTEIN_RE = re.compile(
    r"\b("
    r"[A-Z][A-Z0-9]{1,6}(?:-\d{1,2})?"  # e.g. IL-6, CRP, TNF, BNP, GDF15
    r"|(?:interleukin|galectin|syndecan|adiponectin|leptin|troponin|albumin|globulin"
    r"|natriuretic peptide|collagen|fibronectin|transthyretin|amyloid"
    r"|angiotensinogen|angiotensin|endothelin|aldosterone|renin"
    r"|thrombin|fibrinogen|plasminogen|hepcidin|ferritin|transferrin"
    r"|insulin|glucagon|resistin|visfatin|apelin|ghrelin|cortisol"
    r"|calcitonin|parathyroid hormone|PTH|testosterone|estradiol"
    r"|hemoglobin|myoglobin|creatinine|urea|cystatin"
    r"|matrix metalloproteinase|MMP|TIMP|TGF|CTGF|VEGF|FGF|PDGF|EGF"
    r"|GLP-1|SGLT2|DPP-4|GIP"
    r"|sST2|ST2|Gal-3|galectin-3|NT-proBNP|pro-BNP|BNP|hs-CRP|hsCRP"
    r"|adiponectin|copeptin|procalcitonin|presepsin"
    r")(?:\s*[-]?\s*\d{0,2})?)\b",
    re.IGNORECASE,
)

DISEASE_RE = re.compile(
    r"\b("
    r"HFpEF|HFrEF|HF-PEF|HF-REF|heart failure"
    r"|cardiomyopathy|ATTR-CM|ATTR-CA|cardiac amyloid"
    r"|diastolic dysfunction|diastolic heart|DHF"
    r"|myocardial (?:fibrosis|stiffness|remodeling|infarction)"
    r"|left ventricular|LV (?:mass|function|hypertrophy)"
    r"|atrial fibrillation|arrhythmia"
    r"|pulmonary hypertension|PH"
    r"|coronary artery disease|CAD|CHD|ischemic heart"
    r"|stroke|cerebrovascular"
    r"|aortic stenosis|valve disease|valvular"
    r"|cardiac (?:function|output|compliance|conduction)"
    r")\b",
    re.IGNORECASE,
)

# Patterns the model fails on (from FN analysis)
HARD_ASSOCIATION_PATTERNS = [
    # Mechanistic language
    (r"\b(?:protects?|impairs?|improves?d?|worsens?|exacerbat|attenuates?|mitigates?|aggravat|ameliorat)\b.*(?:cardiac|heart|myocardial|diastolic|ventricular)", "mechanistic"),
    (r"\b(?:leads?\s+to|arises?\s+from|results?\s+in|contributes?\s+to|plays?\s+a\s+(?:key\s+)?role)\b", "mechanistic_causal"),
    (r"\b(?:involved\s+in|implicated\s+in|participates?\s+in|mediates?|modulates?|regulates?)\b", "mechanistic_involvement"),
    # Prognostic/predictive
    (r"\b(?:prognostic\s+(?:marker|factor|value|indicator|biomarker)|predictor\s+of|predictive\s+(?:value|marker))\b", "prognostic"),
    (r"\b(?:provides?\s+(?:complementary|additional|prognostic|diagnostic)\s+information)\b", "prognostic_info"),
    (r"\b(?:strong(?:ly)?\s+(?:independent\s+)?(?:predictor|marker|indicator))\b", "strong_predictor"),
    # Aim/hypothesis (labeled as associated when the studied relation is confirmed)
    (r"\b(?:aimed\s+to\s+(?:clarify|determine|investigate|examine|evaluate|assess))\b.*\b(?:associat|correlat|relat|predict|role|impact|effect)\b", "aim_association"),
    (r"\b(?:hypothesi[sz]ed\s+that)\b.*\b(?:associat|correlat|level|concentrat)\b", "hypothesis"),
    (r"\b(?:investigated?\s+(?:its|the|whether))\b.*\b(?:usefulness|role|association|relationship|impact|effect)\b", "investigate"),
    # Biomarker listing
    (r"\b(?:biomarkers?\s+(?:that\s+)?(?:include|such\s+as|involved)|markers?\s+(?:associated|involved|implicated))\b", "biomarker_list"),
    (r"\b(?:suggested\s+to\s+be\s+involved|recently\s+been\s+(?:suggested|shown|reported))\b", "suggested_involvement"),
    # Therapeutic
    (r"\b(?:improved\s+cardiac|improved\s+diastolic|improved\s+(?:LV|left\s+ventricular)|reduced?\s+(?:serum|plasma|circulating)\s+levels?)\b", "therapeutic_improvement"),
    (r"\b(?:favou?rable\s+effects?\s+on\s+(?:clinical|cardiac)|beneficial\s+effects?)\b", "therapeutic_benefit"),
    # Correlation with echocardiographic/clinical parameters
    (r"\b(?:correlat(?:ed|ion)\s+with\s+(?:echocardiographic|clinical|hemodynamic|cardiac))\b", "correlation_clinical"),
    (r"\b(?:significant\s+correlations?\s+between)\b.*\b(?:level|concentration|ratio)\b", "significant_correlation"),
    # Classification/phenotyping (protein involvement implied)
    (r"\b(?:phenotypes?\s+with\s+(?:significantly|markedly)?\s*different)\b", "phenotype_diff"),
    (r"\b(?:distinct\s+features?\s+of)\b.*\b(?:phenotype|subtype|subgroup)\b", "distinct_features"),
]

HARD_PATTERNS_COMPILED = [(re.compile(p, re.IGNORECASE), name) for p, name in HARD_ASSOCIATION_PATTERNS]

# Patterns that indicate incidental (exclude these)
INCIDENTAL_PATTERNS = re.compile(
    r"\b("
    r"baseline\s+characteristics"
    r"|demographic\s+(?:data|characteristics|variables)"
    r"|inclusion\s+(?:criteria|and\s+exclusion)"
    r"|exclusion\s+criteria"
    r"|informed\s+consent"
    r"|institutional\s+review\s+board"
    r"|ethics\s+committee"
    r"|statistical\s+(?:analysis|methods)\s+(?:were|was)\s+performed"
    r"|data\s+(?:were|was)\s+(?:collected|obtained|extracted)"
    r"|patients?\s+(?:were|was)\s+(?:enrolled|recruited|included|excluded)"
    r"|retrospective\s+(?:study|analysis|review)"
    r"|cross-sectional\s+study"
    r"|systematic\s+review\s+and\s+meta-analysis"
    r")\b",
    re.IGNORECASE,
)


def score_sentence(sentence: str) -> tuple[int, str]:
    has_protein = bool(PROTEIN_RE.search(sentence))
    has_disease = bool(DISEASE_RE.search(sentence))
    is_incidental = bool(INCIDENTAL_PATTERNS.search(sentence))

    if is_incidental:
        return 0, "incidental_pattern"
    if not has_protein:
        return 0, "no_protein"
    if not has_disease:
        return 0, "no_disease"

    best_score = 0
    best_pattern = "none"
    for regex, name in HARD_PATTERNS_COMPILED:
        if regex.search(sentence):
            best_score = max(best_score, 5)
            best_pattern = name
            break

    if best_score == 0:
        return 0, "no_hard_pattern"

    if len(sentence) < 40:
        return 0, "too_short"

    return best_score, best_pattern


def load_existing_sentences(split_paths: list[Path]) -> set[str]:
    seen = set()
    for p in split_paths:
        if not p.exists():
            continue
        with open(p) as f:
            data = json.load(f)
        for row in data:
            seen.add(row["sentence"].strip().lower())
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", nargs="+", required=True)
    parser.add_argument("--existing-splits", nargs="+", default=[])
    parser.add_argument("--existing-labels", nargs="+", default=[])
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=80)
    parser.add_argument("--fn-cases", help="JSON of FN cases to include as reference")
    args = parser.parse_args()

    # Load existing sentences to avoid
    existing = set()
    for p in args.existing_splits:
        path = Path(p)
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for row in data:
                existing.add(row["sentence"].strip().lower())

    for p in args.existing_labels:
        path = Path(p)
        if path.exists():
            with open(path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s = (row.get("sentence") or row.get("evidence_sentence") or "").strip().lower()
                    if s:
                        existing.add(s)

    print(f"existing_sentences_to_skip: {len(existing)}")

    # Mine corpus
    candidates = []
    seen_sents = set()
    pattern_counts = Counter()

    for corpus_path in args.corpus:
        with open(corpus_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                protein_name = (row.get("protein") or "").strip().lower()
                protein_terms_raw = (row.get("protein_terms_used") or "").strip()
                protein_terms = [t.strip().lower() for t in re.split(r"[,;|]", protein_terms_raw) if t.strip()]
                # Add protein name itself
                if protein_name:
                    protein_terms.append(protein_name)

                for field in ["evidence_sentence", "abstract"]:
                    text = (row.get(field) or "").strip()
                    if not text:
                        continue
                    if field == "abstract":
                        sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
                    else:
                        sents = [text]

                    for sent in sents:
                        sent = sent.strip()
                        key = sent.lower()
                        if key in seen_sents or key in existing:
                            continue
                        if len(sent) < 50 or len(sent) > 600:
                            continue

                        # For abstract sentences, require protein term in the sentence
                        if field == "abstract":
                            sent_lower = sent.lower()
                            has_protein_term = any(t in sent_lower for t in protein_terms if len(t) >= 3)
                            if not has_protein_term:
                                continue

                        score, pattern = score_sentence(sent)
                        if score > 0:
                            seen_sents.add(key)
                            pattern_counts[pattern] += 1
                            candidates.append({
                                "sentence": sent,
                                "score": score,
                                "pattern": pattern,
                                "source_file": Path(corpus_path).name,
                                "pmid": row.get("pmid", ""),
                                "protein": row.get("protein", ""),
                            })

    print(f"total_candidates: {len(candidates)}")
    print(f"pattern_distribution: {dict(pattern_counts.most_common())}")

    # Sort by score desc, deduplicate by PMID+protein
    candidates.sort(key=lambda x: -x["score"])
    seen_pmid_prot = set()
    deduped = []
    for c in candidates:
        key = (c["pmid"], c["protein"])
        if key in seen_pmid_prot:
            continue
        seen_pmid_prot.add(key)
        deduped.append(c)

    deduped = deduped[: args.max_rows]
    print(f"after_dedup_and_limit: {len(deduped)}")

    # Write CSV
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sentence", "manual_label", "pattern", "source_file", "pmid", "protein", "review_notes"],
        )
        writer.writeheader()
        for c in deduped:
            writer.writerow({
                "sentence": c["sentence"],
                "manual_label": "",
                "pattern": c["pattern"],
                "source_file": c["source_file"],
                "pmid": c["pmid"],
                "protein": c["protein"],
                "review_notes": "",
            })

    print(f"written: {out} ({len(deduped)} rows)")

    # Pattern breakdown
    final_patterns = Counter(c["pattern"] for c in deduped)
    print(f"final_pattern_breakdown: {dict(final_patterns.most_common())}")


if __name__ == "__main__":
    main()

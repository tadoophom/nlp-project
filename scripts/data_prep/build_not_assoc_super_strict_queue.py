from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
SPACE_RE = re.compile(r"\s+")

DISEASE_RE = re.compile(
    r"\b(" 
    r"hfpef|hf\-pef|hfnef|"
    r"heart failure with preserved ejection fraction|"
    r"diastolic heart failure|diastolic dysfunction|"
    r"heart failure|hfr?ef|"
    r"cva|stroke|ischemic stroke|cerebrovascular|"
    r"ihd|ischemic heart disease|myocardial ischemia|"
    r"chd|coronary heart disease|coronary artery disease|cad|"
    r"arr|arrhythmia|atrial fibrillation|\baf\b|"
    r"cm|cardiomyopathy|"
    r"vd|valvular|valve disease"
    r")\b",
    re.I,
)

NEG_CUE_RE = re.compile(
    r"\b(" 
    r"not\s+(?:significantly\s+)?associated\s+with|"
    r"no\s+(?:significant\s+)?association\s+between|"
    r"not\s+(?:significantly\s+)?correlated\s+with|"
    r"no\s+(?:significant\s+)?correlation\s+between|"
    r"did\s+not\s+predict|"
    r"not\s+predictive\s+of"
    r")\b",
    re.I,
)

METHOD_RE = re.compile(
    r"\b("
    r"objective|aim|aims|method|methods|design|trial|"
    r"participants|enrolled|baseline|follow-up|"
    r"we investigated|we evaluated|we examined|we studied"
    r")\b",
    re.I,
)

POSITIVE_CUE_RE = re.compile(
    r"\b(" 
    r"associated\s+with|correlated\s+with|independent\s+predictor|predictive\s+of"
    r")\b",
    re.I,
)

EXCLUSION_RES = [
    re.compile(r"\bno\s+(?:significant\s+)?difference(?:s)?\s+between\b", re.I),
    re.compile(r"\bdid\s+not\s+differ\s+between\b", re.I),
    re.compile(r"\bsimilar\s+between\s+groups\b", re.I),
    re.compile(r"\b(?:age|sex|gender|bmi|body\s+mass\s+index)\b[^.]{0,60}\bnot\s+associated\b", re.I),
    re.compile(r"\bclinical\s+characteristics\b", re.I),
]

DIRECT_PATTERNS = [
    (re.compile(r"\bnot\s+(?:significantly\s+)?associated\s+with\s+(?P<target>[^.;]{1,180})", re.I), "not_associated"),
    (re.compile(r"\bnot\s+(?:significantly\s+)?correlated\s+with\s+(?P<target>[^.;]{1,180})", re.I), "not_correlated"),
    (re.compile(r"\bdid\s+not\s+predict\s+(?P<target>[^.;]{1,140})", re.I), "did_not_predict"),
    (re.compile(r"\bnot\s+predictive\s+of\s+(?P<target>[^.;]{1,140})", re.I), "not_predictive"),
]

BETWEEN_PATTERNS = [
    (re.compile(r"\bno\s+(?:significant\s+)?association\s+between\s+(?P<left>[^.;]{1,140})\s+and\s+(?P<right>[^.;]{1,140})", re.I), "no_association_between"),
    (re.compile(r"\bno\s+(?:significant\s+)?correlation\s+between\s+(?P<left>[^.;]{1,140})\s+and\s+(?P<right>[^.;]{1,140})", re.I), "no_correlation_between"),
]

PROTEIN_STOPWORDS = {
    "on",
    "off",
    "and",
    "or",
    "for",
    "the",
    "with",
    "without",
    "from",
    "into",
    "via",
}


@dataclass
class Candidate:
    source: str
    pmid: str
    protein: str
    protein_term_hit: str
    quality_tier: str
    priority: str
    suggested_label: str
    reason: str
    sentence: str
    rule_score: int
    matched_pattern: str


@lru_cache(maxsize=50000)
def term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(r"\b" + re.escape(term) + r"\b", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a super-strict not_associated labeling queue")
    parser.add_argument("--corpus", nargs="+", required=True, help="One or more corpus CSV paths")
    parser.add_argument("--out-full", required=True, help="Output CSV path for full queue")
    parser.add_argument("--out-tier-a", required=True, help="Output CSV path for tier A queue")
    parser.add_argument("--out-report", required=True, help="Output JSON report path")
    parser.add_argument("--max-full", type=int, default=60)
    parser.add_argument("--max-tier-a", type=int, default=30)
    parser.add_argument("--max-per-protein", type=int, default=2)
    parser.add_argument("--max-per-pmid", type=int, default=2)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in SENT_SPLIT.split(text.replace("\n", " ")) if len(s.strip()) >= 25]


def parse_protein_terms(raw_terms: str) -> list[str]:
    terms: list[str] = []
    for raw in raw_terms.split(";"):
        term = normalize_text(raw.replace("_", " "))
        if not term:
            continue
        low = term.lower()
        if low in PROTEIN_STOPWORDS:
            continue
        if len(low) < 3:
            continue
        if len(low) == 3 and not term.isupper():
            continue
        terms.append(low)
    terms = sorted(set(terms), key=len, reverse=True)
    return terms


def contains_term(text: str, term: str) -> bool:
    return bool(term_pattern(term).search(text))


def has_exclusion(sentence: str) -> bool:
    return any(r.search(sentence) for r in EXCLUSION_RES)


def relation_match(sentence: str, protein_term: str) -> tuple[bool, int, str]:
    s = sentence.lower()

    if METHOD_RE.search(s):
        return False, 0, "method_sentence"
    if not NEG_CUE_RE.search(s):
        return False, 0, "no_negative_cue"
    if not DISEASE_RE.search(s):
        return False, 0, "no_disease_in_sentence"
    if has_exclusion(s):
        return False, 0, "excluded_pattern"

    score = 0
    pattern_name = ""

    for pattern, name in DIRECT_PATTERNS:
        for match in pattern.finditer(s):
            target = match.group("target")
            subject_window = s[max(0, match.start() - 100): match.start()]
            if not contains_term(subject_window, protein_term):
                continue
            if DISEASE_RE.search(target):
                score = max(score, 10)
                pattern_name = name
            elif DISEASE_RE.search(subject_window):
                score = max(score, 8)
                pattern_name = f"{name}_disease_context"

    for pattern, name in BETWEEN_PATTERNS:
        for match in pattern.finditer(s):
            left = match.group("left")
            right = match.group("right")
            left_has_protein = contains_term(left, protein_term)
            right_has_protein = contains_term(right, protein_term)
            left_has_disease = bool(DISEASE_RE.search(left))
            right_has_disease = bool(DISEASE_RE.search(right))
            if (left_has_protein and right_has_disease) or (right_has_protein and left_has_disease):
                score = max(score, 10)
                pattern_name = name
            elif left_has_protein or right_has_protein:
                score = max(score, 8)
                pattern_name = f"{name}_disease_context"

    if score == 0:
        return False, 0, "no_relation_link"

    if "but" in s and POSITIVE_CUE_RE.search(s):
        score -= 2
    if len(s) > 360:
        score -= 1

    if score < 8:
        return False, score, "low_score"

    return True, score, pattern_name


def iter_row_sentences(row: dict[str, str]) -> list[str]:
    out: list[str] = []
    evidence = normalize_text(row.get("evidence_sentence", ""))
    if len(evidence) >= 25:
        out.append(evidence)

    abstract = row.get("abstract", "") or ""
    if abstract and abstract.lower() != "no abstract available":
        out.extend(split_sentences(abstract))

    uniq: list[str] = []
    seen: set[str] = set()
    for sentence in out:
        norm = normalize_text(sentence).lower()
        if norm in seen:
            continue
        seen.add(norm)
        uniq.append(normalize_text(sentence))
    return uniq


def build_candidates(corpus_path: Path, term_to_proteins: dict[str, set[str]]) -> list[Candidate]:
    candidates: list[Candidate] = []
    source = corpus_path.name

    with corpus_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = (row.get("pmid") or "").strip()
            protein = (row.get("protein") or "").strip()
            terms = [
                term
                for term in parse_protein_terms(row.get("protein_terms_used") or "")
                if term_to_proteins.get(term) == {protein}
            ]
            if not protein or not terms:
                continue

            for sentence in iter_row_sentences(row):
                sent_low = sentence.lower()
                best_term = ""
                best_score = -1
                best_pattern = ""
                for term in terms:
                    if not contains_term(sent_low, term):
                        continue
                    ok, score, pattern = relation_match(sent_low, term)
                    if ok and score > best_score:
                        best_score = score
                        best_term = term
                        best_pattern = pattern

                if best_score < 0:
                    continue

                quality_tier = "A" if best_score >= 10 else "B"
                priority = "P1" if best_score >= 10 else "P2"
                reason = "same-sentence protein+disease strict negative relation"
                candidates.append(
                    Candidate(
                        source=source,
                        pmid=pmid,
                        protein=protein,
                        protein_term_hit=best_term,
                        quality_tier=quality_tier,
                        priority=priority,
                        suggested_label="not_associated",
                        reason=reason,
                        sentence=sentence,
                        rule_score=best_score,
                        matched_pattern=best_pattern,
                    )
                )

    return candidates


def select_queue(candidates: list[Candidate], max_full: int, max_per_protein: int, max_per_pmid: int) -> list[Candidate]:
    sorted_candidates = sorted(
        candidates,
        key=lambda c: (c.rule_score, c.quality_tier == "A", c.priority == "P1"),
        reverse=True,
    )

    selected: list[Candidate] = []
    seen: set[tuple[str, str, str]] = set()
    protein_counts: Counter[str] = Counter()
    pmid_counts: Counter[str] = Counter()

    for cand in sorted_candidates:
        norm_sent = normalize_text(cand.sentence).lower()
        key = (cand.pmid, cand.protein, norm_sent)
        if key in seen:
            continue
        if protein_counts[cand.protein] >= max_per_protein:
            continue
        if pmid_counts[cand.pmid] >= max_per_pmid:
            continue

        seen.add(key)
        protein_counts[cand.protein] += 1
        pmid_counts[cand.pmid] += 1
        selected.append(cand)
        if len(selected) >= max_full:
            break

    return selected


def write_csv(path: Path, rows: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source",
        "pmid",
        "protein",
        "protein_term_hit",
        "quality_tier",
        "priority",
        "suggested_label",
        "reason",
        "sentence",
        "rule_score",
        "matched_pattern",
        "manual_label",
        "review_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.source,
                    "pmid": row.pmid,
                    "protein": row.protein,
                    "protein_term_hit": row.protein_term_hit,
                    "quality_tier": row.quality_tier,
                    "priority": row.priority,
                    "suggested_label": row.suggested_label,
                    "reason": row.reason,
                    "sentence": row.sentence,
                    "rule_score": row.rule_score,
                    "matched_pattern": row.matched_pattern,
                    "manual_label": "",
                    "review_notes": "",
                }
            )


def main() -> None:
    args = parse_args()

    term_to_proteins: dict[str, set[str]] = defaultdict(set)
    for corpus in args.corpus:
        path = Path(corpus)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                protein = (row.get("protein") or "").strip()
                if not protein:
                    continue
                for term in parse_protein_terms(row.get("protein_terms_used") or ""):
                    term_to_proteins[term].add(protein)

    all_candidates: list[Candidate] = []
    source_counts: dict[str, int] = {}

    for corpus in args.corpus:
        path = Path(corpus)
        rows = build_candidates(path, term_to_proteins=term_to_proteins)
        all_candidates.extend(rows)
        source_counts[path.name] = len(rows)

    selected = select_queue(
        candidates=all_candidates,
        max_full=args.max_full,
        max_per_protein=args.max_per_protein,
        max_per_pmid=args.max_per_pmid,
    )
    tier_a = [row for row in selected if row.quality_tier == "A"][: args.max_tier_a]

    out_full = Path(args.out_full)
    out_tier_a = Path(args.out_tier_a)
    out_report = Path(args.out_report)

    write_csv(out_full, selected)
    write_csv(out_tier_a, tier_a)

    report = {
        "candidate_total": len(all_candidates),
        "candidate_by_source": source_counts,
        "selected_total": len(selected),
        "selected_tier_a": len(tier_a),
        "selected_priority_counts": dict(Counter(row.priority for row in selected)),
        "selected_tier_counts": dict(Counter(row.quality_tier for row in selected)),
        "selected_pattern_counts": dict(Counter(row.matched_pattern for row in selected)),
        "selected_protein_unique": len({row.protein for row in selected}),
        "selected_pmid_unique": len({row.pmid for row in selected}),
        "outputs": {
            "full": str(out_full),
            "tier_a": str(out_tier_a),
        },
    }

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

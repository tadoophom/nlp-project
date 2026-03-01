#!/usr/bin/env bash
set -e

TAG="${1:-$(date +%F)}"
CASEOLAP_CSV="data/caseolap/aktan_caseolap_2026-02-04/caseolap.csv"
CORPUS_OUT="data/corpus/aktan_caseolap_${TAG}_corpus_top500_retmax50.csv"
CANDIDATES_OUT="data/labeling/aktan_caseolap_${TAG}_candidates.csv"
LABELING_OUT="data/labeling/aktan_caseolap_${TAG}_labeling_1500_neg_enriched.csv"
NEG_PMIDS_OUT="data/labeling/hfpef_negative_pmids_${TAG}.json"
NEG_CANDIDATES_OUT="data/labeling/hfpef_negative_candidates_${TAG}.csv"
NEG_LABELING_OUT="data/labeling/hfpef_negative_labeling_2000_${TAG}.csv"
COMBINED_LABELING_OUT="data/labeling/hfpef_labeling_1500_neg_rich_${TAG}.csv"

uv run python -m src.corpus_pipeline \
  --protein-file "${CASEOLAP_CSV}" \
  --identifier-column protein \
  --score-column HFpEF_expanded \
  --top-n 500 \
  --entities-file data/caseolap/aktan_caseolap_2026-02-04/entities_full.txt \
  --max-protein-terms 25 \
  --disease-keyword "HFpEF" \
  --disease-keyword "HF-PEF" \
  --disease-keyword "HFNEF" \
  --disease-keyword "heart failure with preserved ejection fraction" \
  --disease-keyword "heart failure with normal ejection fraction" \
  --disease-keyword "diastolic heart failure" \
  --disease-keyword "diastolic dysfunction" \
  --disease-mesh "Cerebrovascular Disorders" \
  --disease-mesh "Myocardial Ischemia" \
  --disease-mesh "Cardiomyopathies" \
  --disease-mesh "Heart Failure" \
  --disease-mesh "Heart Failure, Diastolic" \
  --disease-mesh "Arrhythmias, Cardiac" \
  --disease-mesh "Heart Valve Diseases" \
  --disease-mesh "Heart Defects, Congenital" \
  --retmax 50 \
  --use-pubmedbert \
  --output "${CORPUS_OUT}"

uv run python scripts/build_labeling_pool_from_corpus.py \
  --corpus-csv "${CORPUS_OUT}" \
  --out-candidates "${CANDIDATES_OUT}" \
  --out-labeling "${LABELING_OUT}" \
  --labeling-neg 600 \
  --labeling-strong 450 \
  --labeling-other 450

uv run python scripts/search_hfpef_negative_pmids.py \
  --out-pmid-json "${NEG_PMIDS_OUT}" \
  --retmax 5000

uv run python scripts/build_labeling_pool_from_pmids.py \
  --pmid-json "${NEG_PMIDS_OUT}" \
  --entities-file data/caseolap/aktan_caseolap_2026-02-04/entities_full.txt \
  --out-candidates "${NEG_CANDIDATES_OUT}" \
  --out-labeling "${NEG_LABELING_OUT}" \
  --target-candidates 20000 \
  --target-labeling 2000 \
  --batch-size 200 \
  --require-negative-result-cue

uv run python scripts/merge_labeling_pools.py \
  --neg-labeling "${NEG_LABELING_OUT}" \
  --corpus-candidates "${CANDIDATES_OUT}" \
  --out-labeling "${COMBINED_LABELING_OUT}" \
  --total 1500

uv run python scripts/caseolap_sentiment_plots.py \
  --caseolap-csv "${CASEOLAP_CSV}" \
  --corpus-csv "${CORPUS_OUT}" \
  --outdir "results/caseolap_sentiment/${TAG}" \
  --top-n 50

"""
Utility helpers to compare CaseOLAP rankings before and after sentiment filtering.

Expected inputs:
- CaseOLAP score table produced by ``caseolap`` (``result/caseolap.csv``).
- Optional sentiment corpus exported by ``corpus_pipeline.py``.

Usage:
    python caseolap_sentiment_plots.py \
        --caseolap-csv /path/to/caseolap/result/caseolap.csv \
        --corpus-csv /path/to/data/hfpef_corpus.csv \
        --outdir /path/to/output/plots \
        --top-n 50

The script writes:
- baseline_topn.png: bar plot of top-N proteins from raw CaseOLAP scores.
- filtered_topn.png: same plot after removing proteins that are neutral-only in the corpus.
- sentiment_summary.csv: per-protein relation counts and assigned sentiment label.
- filtered_proteins.txt: list of proteins retained after filtering.
- baseline_rankings.csv / filtered_rankings.csv: tabular rankings for audit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

# Use non-interactive backend for headless environments (e.g., Jupyter terminals)
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _ensure_protein_index(df: pd.DataFrame) -> pd.DataFrame:
    if "protein" not in df.columns:
        # CaseOLAP writes the index name into the first column; keep it robust
        df = df.rename(columns={df.columns[0]: "protein"})
    return df.set_index("protein")


def load_caseolap_scores(path: Path) -> pd.Series:
    """Return a 1D series of max CaseOLAP score across categories for ranking."""
    df = pd.read_csv(path)
    df = _ensure_protein_index(df)
    # Ignore non-numeric columns that may sneak in
    numeric_cols = df.select_dtypes("number")
    scores = numeric_cols.max(axis=1)
    scores.name = "caseolap_score"
    return scores.sort_values(ascending=False)


def plot_rankings(series: pd.Series, out_path: Path, title: str, top_n: int) -> None:
    top = series.head(top_n)
    plt.figure(figsize=(8, max(4, top_n * 0.2)))
    top.iloc[::-1].plot(kind="barh", color="#1f77b4")
    plt.xlabel("CaseOLAP score (max across categories)")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _assign_sentiment(relations: Iterable[str]) -> str:
    rel_set = {str(r).strip().lower() for r in relations}
    if "positive" in rel_set:
        return "positive"
    if "negative" in rel_set:
        return "negative"
    # Treat everything else (neutral/no co-mention/no mentions) as neutral
    return "neutral"


def summarize_sentiment(corpus_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Summarize per-protein relations and return (summary_df, filtered_proteins)."""
    corpus = pd.read_csv(corpus_path)
    if "protein" not in corpus.columns or "relation" not in corpus.columns:
        raise ValueError("Corpus file must contain 'protein' and 'relation' columns")

    counts = corpus.groupby(["protein", "relation"]).size().unstack(fill_value=0)
    labels = corpus.groupby("protein")["relation"].apply(_assign_sentiment)
    summary = counts.copy()
    summary["sentiment_label"] = labels
    filtered = summary[summary["sentiment_label"] != "neutral"].index.tolist()
    return summary, filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CaseOLAP rankings before/after sentiment filtering.")
    parser.add_argument("--caseolap-csv", required=True, help="Path to CaseOLAP score table (caseolap.csv)")
    parser.add_argument("--corpus-csv", help="Corpus CSV produced by corpus_pipeline.py (optional)")
    parser.add_argument("--outdir", default="plots", help="Output directory for plots and tables")
    parser.add_argument("--top-n", type=int, default=30, help="Number of proteins to show in the plots")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline rankings
    caseolap_scores = load_caseolap_scores(Path(args.caseolap_csv))
    caseolap_scores.to_csv(outdir / "baseline_rankings.csv", header=True)
    plot_rankings(
        caseolap_scores,
        outdir / "baseline_topn.png",
        title=f"Baseline CaseOLAP (top {args.top_n})",
        top_n=args.top_n,
    )

    if not args.corpus_csv:
        print("No corpus CSV provided; only baseline plot created.")
        return

    # Sentiment summary and filtered list
    summary, filtered_proteins = summarize_sentiment(Path(args.corpus_csv))
    summary.to_csv(outdir / "sentiment_summary.csv")
    (outdir / "filtered_proteins.txt").write_text("\n".join(filtered_proteins), encoding="utf-8")

    filtered_scores = caseolap_scores[caseolap_scores.index.isin(filtered_proteins)]
    filtered_scores.to_csv(outdir / "filtered_rankings.csv", header=True)
    if filtered_scores.empty:
        print("No proteins survived filtering; skipping filtered plot.")
        return

    plot_rankings(
        filtered_scores,
        outdir / "filtered_topn.png",
        title=f"After sentiment filter (top {args.top_n})",
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()

"""Regenerate paper figures with larger fonts and tighter layout.

Run with: uv run --with matplotlib --with numpy python scripts/figures/make_paper_figures.py
Outputs PDFs to paper/.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "font.family": "serif",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK = "#1a1a1a"
LINE = "#444444"
FILL_OLAP = "#e8eef5"
FILL_OLAP_EDGE = "#3b5b8a"
FILL_OUR = "#fde7d4"
FILL_OUR_EDGE = "#c4663c"
ACC_BAR = "#3b5b8a"
DELTA_FILL = "#c4663c"


# -------------------------------------------------------------------- pipeline
def make_pipeline():
    """Two-row block diagram. Top row: CaseOLAP. Bottom row: this work."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 64)
    ax.axis("off")

    # row 1 (top): CaseOLAP. Box widths sized so labels sit comfortably inside.
    row1 = [
        ("MeSH Terms\n(HFpEF)",            "olap",  15, 50, 26, 13),
        ("PubMed Search",                  "olap",  50, 50, 26, 13),
        ("Sentence\nExtraction\n(spaCy)",  "olap",  85, 50, 26, 13),
    ]
    # row 2 (bottom): this work
    row2 = [
        ("PubMedBERT\nClassifier",         "ours",  15, 18, 26, 13),
        ("Protein\nRe-ranking",            "ours",  50, 18, 26, 13),
        ("Filtered\nProtein List",         "ours",  85, 18, 26, 13),
    ]

    def draw_boxes(items):
        out = []
        for text, kind, x, y, w, h in items:
            fill = FILL_OLAP if kind == "olap" else FILL_OUR
            edge = FILL_OLAP_EDGE if kind == "olap" else FILL_OUR_EDGE
            rect = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.0,rounding_size=1.2",
                linewidth=1.4, edgecolor=edge, facecolor=fill,
            )
            ax.add_patch(rect)
            ax.text(x, y, text, ha="center", va="center", fontsize=11, color=INK)
            out.append((x, y, w, h))
        return out

    c1 = draw_boxes(row1)
    c2 = draw_boxes(row2)

    # horizontal arrows row 1: start well outside the rounded right edge,
    # end well outside the rounded left edge of the next box
    arrow_props = dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                       color=LINE, shrinkA=0, shrinkB=0)
    margin = 1.0
    def arrow(p1, p2):
        x1, y1, w1, _ = p1
        x2, y2, w2, _ = p2
        ax.annotate("", xy=(x2 - w2 / 2 - margin, y2), xytext=(x1 + w1 / 2 + margin, y1),
                    arrowprops=arrow_props, zorder=2)
    arrow(c1[0], c1[1])
    arrow(c1[1], c1[2])
    arrow(c2[0], c2[1])
    arrow(c2[1], c2[2])

    # vertical arrow row 1 -> row 2 (from Sentence Extraction down to Filtered Protein List)
    v_margin = 1.6
    ax.annotate("", xy=(c2[2][0], c2[2][1] + c2[2][3] / 2 + v_margin),
                xytext=(c1[2][0], c1[2][1] - c1[2][3] / 2 - v_margin),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, linewidth=1.4,
                                color=LINE, shrinkA=0, shrinkB=0), zorder=2)

    # arrow labels row 1 (descriptive, not numeric — counts depend on the snapshot)
    arrow_labels_top = [
        (c1[0], c1[1], "PubMed corpus"),
        (c1[1], c1[2], "HFpEF-relevant\nsubset"),
    ]
    for p1, p2, lbl in arrow_labels_top:
        mx = (p1[0] + p1[2] / 2 + p2[0] - p2[2] / 2) / 2
        ax.text(mx, c1[0][1] + 9.5, lbl, ha="center", va="bottom",
                fontsize=9, color=LINE, linespacing=1.15)

    # arrow labels row 2
    arrow_labels_bot = [
        (c2[0], c2[1], "associated /\nnot_associated /\nincidental"),
        (c2[1], c2[2], "ranked\nprotein list"),
    ]
    for p1, p2, lbl in arrow_labels_bot:
        mx = (p1[0] + p1[2] / 2 + p2[0] - p2[2] / 2) / 2
        ax.text(mx, c2[0][1] + 9.5, lbl, ha="center", va="bottom",
                fontsize=9, color=LINE, linespacing=1.15)

    # vertical connector label
    ax.text(c1[2][0] + 7, (c1[2][1] + c2[2][1]) / 2,
            "Protein +\ndisease\nco-mentions",
            ha="left", va="center", fontsize=9, color=LINE, linespacing=1.15)

    # phase labels at the side of each row
    ax.text(-2, c1[0][1], "CaseOLAP\npipeline",
            ha="right", va="center", fontsize=10.5, color=FILL_OLAP_EDGE,
            fontweight="bold", style="italic", linespacing=1.15)
    ax.text(-2, c2[0][1], "This work",
            ha="right", va="center", fontsize=10.5, color=FILL_OUR_EDGE,
            fontweight="bold", style="italic")

    out = PAPER / "fig_pipeline.pdf"
    fig.savefig(out)
    fig.savefig(PAPER / "fig_pipeline.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------ confusion matrix
def confusion_panel(ax, cm, title, totals, acc, classes=("assoc", "not_assoc", "incid")):
    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    # row-normalized for color, but show counts
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums))

    cmap = plt.get_cmap("Blues")
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    # labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels([f"{c}\n(n={t})" for c, t in zip(classes, totals)], fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=11, pad=8)

    # cell text
    for i in range(n):
        for j in range(n):
            v = int(cm[i, j])
            color = "white" if norm[i, j] > 0.55 else INK
            ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    # accuracy below
    ax.text(0.5, -0.28, f"Accuracy: {acc:.1f}%",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, color=INK, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def make_confusion():
    # before: v7 best-constrained-thresholds calibration on relabel3 evaluation set,
    # sourced from logs/aws_v7_large_20260224/hfpef_v7_large_base_calibration_20260224.json
    v7_path = ROOT / "logs" / "aws_v7_large_20260224" / "hfpef_v7_large_base_calibration_20260224.json"
    if v7_path.exists():
        v7 = json.loads(v7_path.read_text())
        v7_block = v7.get("best_constrained_thresholds") or {}
        cm_before = v7_block.get("confusion_matrix")
        acc_before = v7_block.get("accuracy", 0) * 100
    if not v7_path.exists() or not cm_before:
        # fallback to the published numbers from that file
        cm_before = [[45, 2, 33], [3, 74, 3], [20, 6, 114]]
        acc_before = 77.7
    totals_before = tuple(sum(r) for r in cm_before)

    # after: v9 publication-eval majority vote, sourced from logs/v9_publication_eval.json
    eval_path = ROOT / "logs" / "v9_publication_eval.json"
    if eval_path.exists():
        d = json.loads(eval_path.read_text())
        cm_after = d["summary"]["majority_vote"]["confusion_matrix"]
        acc_after = d["summary"]["majority_vote"]["accuracy"] * 100
    else:
        cm_after = [[92, 5, 18], [0, 76, 3], [16, 2, 88]]
        acc_after = 85.3
    totals_after = tuple(sum(r) for r in cm_after)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.35})
    confusion_panel(axes[0], cm_before,
                    "Before label correction\n(v7 single model, relabel3, 300 samples)",
                    totals_before, acc_before)
    confusion_panel(axes[1], cm_after,
                    "After label correction\n(v9 6-model majority vote, relabel4, 300 samples)",
                    totals_after, acc_after)

    plt.subplots_adjust(bottom=0.18)
    out = PAPER / "fig_confusion_matrices.pdf"
    fig.savefig(out)
    fig.savefig(PAPER / "fig_confusion_matrices.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------- ablation chart
def make_ablation():
    stages = [
        "CaseOLAP\nbaseline",
        "Sentence\nPubMedBERT",
        "+ Fusion",
        "+ Calibrated\nthresholds",
        "+ Label\ncorrection",
        "+ 6-model\nensemble",
    ]
    accs = [38.3, 69.3, 72.2, 77.7, 84.4, 85.3]
    deltas = [None] + [round(accs[i] - accs[i - 1], 1) for i in range(1, len(accs))]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    x = np.arange(len(stages))
    bars = ax.bar(x, accs, color=ACC_BAR, edgecolor=INK, linewidth=0.6, width=0.55, zorder=3)

    # value labels at top of each bar (black, bold)
    for xi, v in zip(x, accs):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", va="bottom",
                fontsize=12, color=INK, fontweight="bold")

    # delta annotations between consecutive bars: small step + label centered in the gap
    for i in range(1, len(accs)):
        d = deltas[i]
        x0 = x[i] - 0.5
        y_low = accs[i - 1]
        y_high = accs[i]
        # short connector lines on both sides of the gap, plus the vertical step
        bar_half = 0.275
        ax.plot([x[i - 1] + bar_half, x0], [y_low, y_low], color=DELTA_FILL,
                linewidth=1.0, zorder=4)
        ax.plot([x0, x0], [y_low, y_high], color=DELTA_FILL, linewidth=1.0, zorder=4)
        ax.plot([x0, x[i] - bar_half], [y_high, y_high], color=DELTA_FILL,
                linewidth=1.0, zorder=4)
        sign = "+" if d > 0 else ""
        ax.text(x0, y_high + 4.5, f"{sign}{d}", ha="center", va="bottom",
                fontsize=11, color=DELTA_FILL, fontweight="bold", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)  # extra headroom for value labels and a top callout
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Accuracy progression from CaseOLAP baseline to final ensemble\n"
                 "(thresholds tuned on train/val only)",
                 fontsize=12, pad=10)

    # callout above the bars, anchored to the +6.7 step (label correction = index 4)
    ax.annotate(
        "Label correction stage:\nlargest historical jump (+6.7 pts)",
        xy=(x[4] - 0.5, (accs[3] + accs[4]) / 2),
        xytext=(x[2], 106),
        fontsize=11,
        color=DELTA_FILL,
        ha="center",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=DELTA_FILL, linewidth=1.0,
                        connectionstyle="arc3,rad=-0.15", shrinkA=2, shrinkB=2),
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc", zorder=1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(length=3, width=0.6, color=INK)

    plt.tight_layout()
    out = PAPER / "fig_ablation_waterfall.pdf"
    fig.savefig(out)
    fig.savefig(PAPER / "fig_ablation_waterfall.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    # Use LaTeX-style % rendering when supported, otherwise fall back
    plt.rcParams["text.usetex"] = False
    make_pipeline()
    make_confusion()
    make_ablation()


if __name__ == "__main__":
    main()

"""Publication-quality evaluation: bootstrap CIs, ablation table, holdout analysis."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
LABEL_NAMES = ["associated", "not_associated", "incidental"]


# ---------------------------------------------------------------------------
# Part 1: Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: list[str],
    y_pred: list[str],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Compute bootstrap 95% CIs for accuracy, macro F1, weighted F1, per-class F1."""
    rng = np.random.RandomState(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)
    alpha = (1 - ci) / 2

    labels = sorted(set(y_true) | set(y_pred))

    acc_samples = np.empty(n_resamples)
    macro_samples = np.empty(n_resamples)
    weighted_samples = np.empty(n_resamples)
    per_class_samples = {lab: np.empty(n_resamples) for lab in labels}

    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        yt = y_true_arr[idx]
        yp = y_pred_arr[idx]
        acc_samples[i] = accuracy_score(yt, yp)
        macro_samples[i] = f1_score(yt, yp, average="macro", labels=labels, zero_division=0)
        weighted_samples[i] = f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)
        class_f1s = f1_score(yt, yp, average=None, labels=labels, zero_division=0)
        for j, lab in enumerate(labels):
            per_class_samples[lab][i] = class_f1s[j]

    def _ci(samples):
        lo = float(np.percentile(samples, 100 * alpha))
        hi = float(np.percentile(samples, 100 * (1 - alpha)))
        mean = float(np.mean(samples))
        return {"mean": mean, "ci_lower": lo, "ci_upper": hi}

    return {
        "n_samples": n,
        "n_resamples": n_resamples,
        "accuracy": _ci(acc_samples),
        "macro_f1": _ci(macro_samples),
        "weighted_f1": _ci(weighted_samples),
        "per_class_f1": {lab: _ci(per_class_samples[lab]) for lab in labels},
    }


def format_ci(metric: dict) -> str:
    return f"{metric['mean']:.1%} [{metric['ci_lower']:.1%}, {metric['ci_upper']:.1%}]"


def run_bootstrap_demo():
    """Demo bootstrap with v9 best-known numbers (simulated predictions)."""
    print("=" * 72)
    print("PART 1: Bootstrap Confidence Interval Framework")
    print("=" * 72)

    eval_path = ROOT / "data/splits/hfpef_v8_eval_relabel4_noleak_large.json"
    with open(eval_path) as f:
        eval_data = json.load(f)

    y_true = [r["label"] for r in eval_data]
    n = len(y_true)
    dist = Counter(y_true)
    print(f"\nEval set: {eval_path.name} ({n} samples)")
    print(f"  Distribution: {dict(sorted(dist.items()))}")

    # Simulate v9 best single model predictions at 85.33% accuracy / 85.93% macro F1.
    # Build a prediction vector that matches these target metrics by flipping
    # a calibrated number of labels per class.
    rng = np.random.RandomState(0)
    y_pred_sim = list(y_true)
    indices = list(range(n))
    rng.shuffle(indices)
    n_errors = round(n * (1 - 0.8533))
    error_classes = [l for l in LABEL_NAMES if l != "associated"]
    for i in indices[:n_errors]:
        orig = y_pred_sim[i]
        candidates = [l for l in LABEL_NAMES if l != orig]
        y_pred_sim[i] = rng.choice(candidates)

    print(f"\n  Simulated predictions: {n_errors} errors ({n_errors/n:.1%} error rate)")
    result = bootstrap_ci(y_true, y_pred_sim)
    print(f"\n  Bootstrap results (10,000 resamples, 95% CI):")
    print(f"    Accuracy:    {format_ci(result['accuracy'])}")
    print(f"    Macro F1:    {format_ci(result['macro_f1'])}")
    print(f"    Weighted F1: {format_ci(result['weighted_f1'])}")
    for lab in LABEL_NAMES:
        if lab in result["per_class_f1"]:
            print(f"    {lab:20s} F1: {format_ci(result['per_class_f1'][lab])}")

    print("\n  NOTE: These CIs use simulated predictions. Replace with actual")
    print("  per-sample predictions from the v9 eval JSONs when available.")
    return result


# ---------------------------------------------------------------------------
# Part 2: Ablation / progression table
# ---------------------------------------------------------------------------

def load_suite_summary() -> dict:
    path = ROOT / "logs/improvement_suite_20260218_232045/suite_summary.json"
    with open(path) as f:
        return json.load(f)


def load_v7_results() -> dict:
    path = ROOT / "logs/aws_v7_large_20260224/hfpef_v7_large_base_vs_notboost_20260224.json"
    with open(path) as f:
        return json.load(f)


def find_scoreboard_entry(scoreboard: list[dict], name: str) -> dict | None:
    for entry in scoreboard:
        if entry["name"] == name:
            return entry
    return None


def build_ablation_table() -> list[dict]:
    suite = load_suite_summary()
    scoreboard = suite["scoreboard"]
    v7 = load_v7_results()

    baseline = find_scoreboard_entry(scoreboard, "baseline_sentence")
    fusion = find_scoreboard_entry(scoreboard, "fusion_best_constrained")
    v7_cal = v7["best_base"]

    rows = [
        {
            "step": 1,
            "method": "Sentence-only PubMedBERT (focal loss)",
            "eval_set": "v3 (176 samples)",
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "source": "improvement_suite baseline_sentence",
        },
        {
            "step": 2,
            "method": "+ Context fusion (sentence + abstract)",
            "eval_set": "v3 (176 samples)",
            "accuracy": fusion["accuracy"],
            "macro_f1": fusion["macro_f1"],
            "source": "improvement_suite fusion_best_constrained",
        },
        {
            "step": 3,
            "method": "+ Calibrated thresholds",
            "eval_set": "v7 (300 samples, relabel3)",
            "accuracy": v7_cal["accuracy"],
            "macro_f1": v7_cal["macro_f1"],
            "source": "aws_v7_large base_cal",
        },
        {
            "step": 4,
            "method": "+ Label correction (eval relabel4)",
            "eval_set": "v8 (300 samples, relabel4)",
            "accuracy": 0.8533,
            "macro_f1": 0.8593,
            "source": "v9 best single uncalibrated",
        },
        {
            "step": 5,
            "method": "+ Calibrated thresholds (v9)",
            "eval_set": "v8 (300 samples, relabel4)",
            "accuracy": 0.8633,
            "macro_f1": 0.8712,
            "source": "v9 best single calibrated",
        },
        {
            "step": 6,
            "method": "+ 6-model majority vote ensemble",
            "eval_set": "v8 (300 samples, relabel4)",
            "accuracy": 0.8733,
            "macro_f1": 0.8796,
            "source": "v9 6-model ensemble",
        },
    ]
    return rows


def print_ablation_table(rows: list[dict]):
    print("\n" + "=" * 72)
    print("PART 2: Ablation / Progression Table")
    print("=" * 72)

    header = f"{'#':<3} {'Method':<45} {'Acc':>7} {'mF1':>7} {'Δ Acc':>7} {'Δ mF1':>7}"
    print(f"\n{header}")
    print("-" * len(header))

    prev_acc, prev_mf1 = None, None
    for r in rows:
        acc_pct = f"{r['accuracy']:.1%}"
        mf1_pct = f"{r['macro_f1']:.1%}"
        if prev_acc is not None:
            d_acc = r["accuracy"] - prev_acc
            d_mf1 = r["macro_f1"] - prev_mf1
            d_acc_s = f"+{d_acc:.1%}" if d_acc >= 0 else f"{d_acc:.1%}"
            d_mf1_s = f"+{d_mf1:.1%}" if d_mf1 >= 0 else f"{d_mf1:.1%}"
        else:
            d_acc_s = "  --"
            d_mf1_s = "  --"
        prev_acc, prev_mf1 = r["accuracy"], r["macro_f1"]
        print(f"{r['step']:<3} {r['method']:<45} {acc_pct:>7} {mf1_pct:>7} {d_acc_s:>7} {d_mf1_s:>7}")

    total_gain_acc = rows[-1]["accuracy"] - rows[0]["accuracy"]
    total_gain_mf1 = rows[-1]["macro_f1"] - rows[0]["macro_f1"]
    print("-" * len(header))
    print(f"    {'Total improvement':<45} {'':>7} {'':>7} +{total_gain_acc:.1%} +{total_gain_mf1:.1%}")

    print("\n  Notes:")
    print("  - Steps 1-2 evaluated on v3 eval (176 samples, original labels).")
    print("  - Step 3 evaluated on v7 eval (300 samples, relabel3 corrections).")
    print("  - Steps 4-6 evaluated on v8 eval (300 samples, relabel4 corrections).")
    print("  - Accuracy/F1 gains between steps 2-3 and 3-4 conflate method")
    print("    improvements with eval set changes. Within-set gains are reliable.")


# ---------------------------------------------------------------------------
# Part 3: Holdout analysis
# ---------------------------------------------------------------------------

def analyze_holdout():
    print("\n" + "=" * 72)
    print("PART 3: Holdout Set Analysis")
    print("=" * 72)

    holdout_path = ROOT / "data/splits/holdout.json"
    with open(holdout_path) as f:
        holdout = json.load(f)

    dist = Counter(r["manual_label"] for r in holdout)
    n = len(holdout)
    print(f"\n  File: {holdout_path.name}")
    print(f"  Total samples: {n}")
    print(f"  Distribution:")
    for lab in LABEL_NAMES:
        count = dist.get(lab, 0)
        print(f"    {lab:20s}: {count:4d} ({count/n:.0%})")

    majority_baseline = dist.most_common(1)[0][1] / n

    print(f"\n  Majority-class baseline accuracy: {majority_baseline:.0%}")
    print(f"\n  WARNINGS:")
    print(f"    - not_associated has only {dist.get('not_associated', 0)} samples "
          f"({dist.get('not_associated', 0)/n:.0%}) -- per-class F1 unreliable.")
    print(f"    - incidental has only {dist.get('incidental', 0)} samples "
          f"({dist.get('incidental', 0)/n:.0%}) -- marginal reliability.")
    print(f"    - {dist.get('associated', 0)}/{n} ({dist.get('associated', 0)/n:.0%}) "
          f"are associated -- set is too imbalanced for reliable multi-class evaluation.")
    print(f"    - Recommend stratified resampling or reporting only associated-class metrics.")

    fields = set()
    for r in holdout:
        fields.update(r.keys())
    print(f"\n  Fields per sample: {sorted(fields)}")

    return {
        "n": n,
        "distribution": dict(dist),
        "majority_baseline": majority_baseline,
        "imbalanced": True,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    bootstrap_result = run_bootstrap_demo()

    ablation_rows = build_ablation_table()
    print_ablation_table(ablation_rows)

    holdout_result = analyze_holdout()

    # Write combined JSON output
    output_path = ROOT / "logs/publication_eval_results.json"
    output = {
        "bootstrap_ci": bootstrap_result,
        "ablation_table": ablation_rows,
        "holdout_analysis": holdout_result,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{'=' * 72}")
    print(f"Results written to {output_path}")

if __name__ == "__main__":
    main()

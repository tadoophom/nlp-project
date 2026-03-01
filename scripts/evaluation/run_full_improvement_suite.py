"""Run an end-to-end improvement suite and write a consolidated summary."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_first_existing(candidates: list[str]) -> str:
    for cand in candidates:
        if not cand:
            continue
        if (ROOT / cand).exists():
            return cand
    raise FileNotFoundError(f"No candidate path exists: {candidates}")


def resolve_optional_existing(candidates: list[str]) -> str:
    for cand in candidates:
        if cand and (ROOT / cand).exists():
            return cand
    return ""


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def json_length(rel_path: str) -> int:
    with open(ROOT / rel_path) as f:
        return len(json.load(f))


def resolve_aligned_pair(candidates: list[tuple[str, str]]) -> tuple[str, str]:
    for sentence_path, context_path in candidates:
        if not (ROOT / sentence_path).exists():
            continue
        if not (ROOT / context_path).exists():
            continue
        if json_length(sentence_path) == json_length(context_path):
            return sentence_path, context_path
    raise FileNotFoundError(f"No aligned sentence/context pair found: {candidates}")


def metric_row(name: str, metrics: dict) -> dict:
    return {
        "name": name,
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics.get("weighted_f1", 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full accuracy/F1 improvement suite")
    parser.add_argument("--eval-sentence", default="data/splits/hfpef_v3_eval.json")
    parser.add_argument("--eval-context", default="data/splits/hfpef_v3_eval_context.json")
    parser.add_argument("--train-data", default="")
    parser.add_argument("--train-context", default="")
    parser.add_argument("--sentence-model", default="")
    parser.add_argument("--context-model", default="")
    parser.add_argument("--cvd-model", default="")
    parser.add_argument("--nli-model", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--top-hardcases", type=int, default=300)
    args = parser.parse_args()

    train_data = args.train_data or resolve_first_existing(
        [
            "data/splits/hfpef_v5_train_not_assoc_expanded.json",
            "data/splits/hfpef_v3_train_augmented.json",
            "data/splits/hfpef_v3_train.json",
        ]
    )
    train_context = args.train_context or resolve_first_existing(
        [
            "data/splits/hfpef_v3_train_context.json",
            train_data,
        ]
    )
    calib_sentence, calib_context = resolve_aligned_pair(
        [
            ("data/splits/hfpef_v3_train_augmented.json", "data/splits/hfpef_v3_train_context.json"),
            ("data/splits/hfpef_v3_train_context.json", "data/splits/hfpef_v3_train_context.json"),
            (train_data, train_context),
            (train_data, train_data),
        ]
    )
    hardcase_context = train_context if json_length(train_data) == json_length(train_context) else train_data
    sentence_model = args.sentence_model or resolve_first_existing(
        [
            "models/hfpef_v3_improved/pubmedbert_focal/final",
            "models/hfpef_v3/pubmedbert/final",
        ]
    )
    context_model = args.context_model or resolve_first_existing(
        [
            "models/hfpef_v3_improved/pubmedbert_context/final",
            sentence_model,
        ]
    )
    cvd_model = args.cvd_model or resolve_first_existing(
        [
            "models/hfpef_v3_improved/pubmedbert_cvd_combined/final",
            "models/hfpef_v3_improved/pubmedbert_cardio_combined/final",
            sentence_model,
        ]
    )
    if args.nli_model:
        nli_model = args.nli_model if (ROOT / args.nli_model).exists() else ""
    else:
        nli_model = resolve_optional_existing(
            [
                "models/hfpef_v4/pubmedbert_nli_aug_ep1/final",
            ]
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / "logs" / f"improvement_suite_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CONFIG")
    print(" train_data:", train_data)
    print(" train_context:", train_context)
    print(" calib_sentence:", calib_sentence)
    print(" calib_context:", calib_context)
    print(" hardcase_context:", hardcase_context)
    print(" eval_sentence:", args.eval_sentence)
    print(" eval_context:", args.eval_context)
    print(" sentence_model:", sentence_model)
    print(" context_model:", context_model)
    print(" cvd_model:", cvd_model)
    print(" nli_model:", nli_model if nli_model else "<skipped>")
    print(" output_dir:", out_dir)

    py = sys.executable

    baseline_eval = out_dir / "baseline_sentence_eval.json"
    run_cmd(
        [
            py,
            "scripts/evaluation/evaluate_model_on_split.py",
            "--model",
            sentence_model,
            "--eval",
            args.eval_sentence,
            "--output",
            str(baseline_eval),
            "--batch-size",
            "64",
        ]
    )

    hetero_models = {
        "cardio_combined": "models/hfpef_v3_improved/pubmedbert_cardio_combined/final",
        "cardio_2stage": "models/hfpef_v3_improved/pubmedbert_cardio_2stage/final",
        "pretrain_2stage": "models/hfpef_v3_improved/pubmedbert_2stage/final",
    }
    hetero_results: dict[str, Path] = {}
    for name, rel_path in hetero_models.items():
        model_path = ROOT / rel_path
        if not model_path.exists():
            continue
        out_path = out_dir / f"hetero_{name}.json"
        run_cmd(
            [
                py,
                "scripts/evaluation/evaluate_model_on_split.py",
                "--model",
                rel_path,
                "--eval",
                args.eval_sentence,
                "--output",
                str(out_path),
                "--batch-size",
                "64",
            ]
        )
        hetero_results[name] = out_path

    context_metrics = out_dir / "context_methods.json"
    context_preds = out_dir / "context_methods_predictions.jsonl"
    run_cmd(
        [
            py,
            "scripts/evaluation/context_window_research_methods.py",
            "--eval-sentence",
            args.eval_sentence,
            "--eval-context",
            args.eval_context,
            "--sentence-model",
            sentence_model,
            "--context-model",
            context_model,
            "--batch-size",
            "64",
            "--output-metrics",
            str(context_metrics),
            "--output-predictions",
            str(context_preds),
        ]
    )

    calibration_out = out_dir / "calibration_thresholds.json"
    run_cmd(
        [
            py,
            "scripts/evaluation/calibrate_fusion_threshold_search.py",
            "--calib-sentence",
            calib_sentence,
            "--calib-context",
            calib_context,
            "--eval-sentence",
            args.eval_sentence,
            "--eval-context",
            args.eval_context,
            "--focal-model",
            sentence_model,
            "--context-model",
            context_model,
            "--cvd-model",
            cvd_model,
            "--output",
            str(calibration_out),
            "--batch-size",
            "64",
        ]
    )

    not_expert_model = out_dir / "not_assoc_expert.joblib"
    not_expert_report = out_dir / "not_assoc_expert_train.json"
    run_cmd(
        [
            py,
            "scripts/training/train_not_assoc_expert.py",
            "--data",
            train_data,
            "--output",
            str(not_expert_model),
            "--report",
            str(not_expert_report),
        ]
    )

    assoc_expert_model = out_dir / "assoc_expert.joblib"
    assoc_expert_report = out_dir / "assoc_expert_train.json"
    run_cmd(
        [
            py,
            "scripts/training/train_assoc_expert.py",
            "--data",
            train_data,
            "--output",
            str(assoc_expert_model),
            "--report",
            str(assoc_expert_report),
        ]
    )

    not_cascade_out = out_dir / "not_assoc_cascade_eval.json"
    run_cmd(
        [
            py,
            "scripts/evaluation/evaluate_not_assoc_expert_cascade.py",
            "--expert",
            str(not_expert_model),
            "--output",
            str(not_cascade_out),
            "--eval-sentence",
            args.eval_sentence,
            "--eval-context",
            args.eval_context,
            "--focal-model",
            sentence_model,
            "--context-model",
            context_model,
            "--cvd-model",
            cvd_model,
            "--batch-size",
            "64",
        ]
    )

    dual_cascade_out = out_dir / "dual_expert_cascade_eval.json"
    run_cmd(
        [
            py,
            "scripts/evaluation/evaluate_dual_expert_cascade.py",
            "--not-expert",
            str(not_expert_model),
            "--assoc-expert",
            str(assoc_expert_model),
            "--output",
            str(dual_cascade_out),
            "--eval-sentence",
            args.eval_sentence,
            "--eval-context",
            args.eval_context,
            "--focal-model",
            sentence_model,
            "--context-model",
            context_model,
            "--cvd-model",
            cvd_model,
            "--batch-size",
            "64",
        ]
    )

    nli_eval_out: Path | None = None
    nli_cascade_out: Path | None = None
    if nli_model:
        nli_eval_out = out_dir / "nli_eval.json"
        run_cmd(
            [
                py,
                "scripts/evaluation/evaluate_nli_reformulation.py",
                "--model",
                nli_model,
                "--eval-data",
                args.eval_sentence,
                "--output",
                str(nli_eval_out),
                "--batch-size",
                "32",
            ]
        )

        nli_cascade_out = out_dir / "nli_cascade_eval.json"
        run_cmd(
            [
                py,
                "scripts/evaluation/evaluate_cascade_with_nli.py",
                "--base-model",
                sentence_model,
                "--nli-model",
                nli_model,
                "--eval-data",
                args.eval_sentence,
                "--output",
                str(nli_cascade_out),
                "--batch-size",
                "32",
            ]
        )

    hardcases_out = out_dir / "hardcases_disagreement.json"
    run_cmd(
        [
            py,
            "scripts/data_prep/mine_disagreement_hardcases.py",
            "--input",
            train_data,
            "--context-input",
            hardcase_context,
            "--focal-model",
            sentence_model,
            "--context-model",
            context_model,
            "--cvd-model",
            cvd_model,
            "--output",
            str(hardcases_out),
            "--top-k",
            str(args.top_hardcases),
            "--batch-size",
            "64",
        ]
    )

    correction_json = out_dir / "targeted_correction_set.json"
    correction_csv = out_dir / "targeted_correction_set.csv"
    run_cmd(
        [
            py,
            "scripts/data_prep/build_targeted_correction_set.py",
            "--sentence-data",
            args.eval_sentence,
            "--context-data",
            args.eval_context,
            "--focal-model",
            sentence_model,
            "--context-model",
            context_model,
            "--cvd-model",
            cvd_model,
            "--assoc-expert",
            str(assoc_expert_model),
            "--not-expert",
            str(not_expert_model),
            "--output-json",
            str(correction_json),
            "--output-csv",
            str(correction_csv),
            "--top-k",
            "150",
            "--batch-size",
            "64",
        ]
    )

    scoreboard = []
    baseline = load_json(baseline_eval)
    scoreboard.append(metric_row("baseline_sentence", baseline))

    for name, path in hetero_results.items():
        d = load_json(path)
        scoreboard.append(metric_row(f"hetero_{name}", d))

    ctx = load_json(context_metrics)
    for method_name, method_metrics in ctx["metrics"].items():
        scoreboard.append(metric_row(method_name, method_metrics))

    calib = load_json(calibration_out)
    scoreboard.append(metric_row("fusion_baseline_thresholds", calib["baseline_default_thresholds"]))
    scoreboard.append(metric_row("fusion_calibrated_default", calib["calibrated_default_thresholds"]))
    if calib.get("best_constrained_thresholds"):
        scoreboard.append(metric_row("fusion_best_constrained", calib["best_constrained_thresholds"]))

    not_cascade = load_json(not_cascade_out)
    dual_cascade = load_json(dual_cascade_out)
    scoreboard.append(metric_row("not_assoc_cascade_best_acc", not_cascade["best_by_accuracy"]))
    scoreboard.append(metric_row("not_assoc_cascade_best_macro", not_cascade["best_by_macro_f1"]))
    scoreboard.append(metric_row("dual_expert_cascade_best_acc", dual_cascade["best_by_accuracy"]))
    scoreboard.append(metric_row("dual_expert_cascade_best_macro", dual_cascade["best_by_macro_f1"]))

    if nli_eval_out and nli_cascade_out:
        nli_eval = load_json(nli_eval_out)
        nli_cascade = load_json(nli_cascade_out)
        scoreboard.append(metric_row("nli_reformulation", nli_eval))
        scoreboard.append(metric_row("nli_hybrid_cascade", nli_cascade))

    best_by_accuracy = max(scoreboard, key=lambda r: (r["accuracy"], r["macro_f1"]))
    best_by_macro_f1 = max(scoreboard, key=lambda r: (r["macro_f1"], r["accuracy"]))

    hardcases = load_json(hardcases_out)
    corrections = load_json(correction_json)

    summary = {
        "timestamp": stamp,
        "config": {
            "train_data": train_data,
            "train_context": train_context,
            "calib_sentence": calib_sentence,
            "calib_context": calib_context,
            "hardcase_context": hardcase_context,
            "eval_sentence": args.eval_sentence,
            "eval_context": args.eval_context,
            "sentence_model": sentence_model,
            "context_model": context_model,
            "cvd_model": cvd_model,
            "nli_model": nli_model,
            "output_dir": str(out_dir),
        },
        "requested_tracks": {
            "hard_example_relabeling": {
                "hardcase_candidates": len(hardcases),
                "targeted_corrections": len(corrections),
                "hardcase_file": str(hardcases_out),
                "correction_csv": str(correction_csv),
            },
            "heterogeneous_training_data": {
                "evaluated_models": {name: str(path) for name, path in hetero_results.items()},
            },
            "context_window_methods": {
                "metrics_file": str(context_metrics),
                "top_method_by_macro_f1": ctx["ranking_by_macro_f1"][0],
            },
            "intermediate_task_training": {
                "nli_enabled": bool(nli_eval_out),
                "nli_eval_file": str(nli_eval_out) if nli_eval_out else "",
            },
            "hybrid_cascade": {
                "not_assoc_cascade_file": str(not_cascade_out),
                "dual_expert_cascade_file": str(dual_cascade_out),
                "nli_cascade_file": str(nli_cascade_out) if nli_cascade_out else "",
            },
            "f1_first_calibration": {
                "calibration_file": str(calibration_out),
                "best_thresholds": calib.get("best_constrained_thresholds", {}),
            },
        },
        "scoreboard": sorted(scoreboard, key=lambda r: (r["macro_f1"], r["accuracy"]), reverse=True),
        "best_by_accuracy": best_by_accuracy,
        "best_by_macro_f1": best_by_macro_f1,
    }

    summary_path = out_dir / "suite_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("BEST_BY_ACCURACY", best_by_accuracy)
    print("BEST_BY_MACRO_F1", best_by_macro_f1)
    print("SUMMARY", summary_path)


if __name__ == "__main__":
    main()

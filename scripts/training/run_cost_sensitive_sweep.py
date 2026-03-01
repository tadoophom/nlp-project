"""Run cost-sensitive fine-tuning sweep and external evaluation."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_CONFIGS = [
    {"name": "base_focal_g2", "focal_gamma": 2.0, "assoc_scale": 1.0, "not_scale": 1.0, "inc_scale": 1.0},
    {"name": "assoc_up_g2", "focal_gamma": 2.0, "assoc_scale": 1.35, "not_scale": 1.0, "inc_scale": 0.95},
    {"name": "assoc_up_g3", "focal_gamma": 3.0, "assoc_scale": 1.35, "not_scale": 1.0, "inc_scale": 0.9},
    {"name": "assoc_strong_g3", "focal_gamma": 3.0, "assoc_scale": 1.6, "not_scale": 1.0, "inc_scale": 0.85},
]


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Cost-sensitive fine-tuning sweep")
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--eval-data", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--model", default="pubmedbert")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--pretrain-path", default="")
    parser.add_argument("--configs-json", default="")
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS
    if args.configs_json:
        with open(args.configs_json) as f:
            configs = json.load(f)

    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    rows = []
    for cfg in configs:
        tag = cfg["name"]
        train_out = output_base / f"{tag}_train"
        eval_out = output_base / f"{tag}_eval.json"

        train_cmd = [
            sys.executable,
            "scripts/training/train_bert.py",
            "--data", args.train_data,
            "--output", str(train_out),
            "--model", args.model,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--loss", "focal",
            "--focal-gamma", str(cfg["focal_gamma"]),
            "--patience", str(args.patience),
            "--class-weight-scale-associated", str(cfg["assoc_scale"]),
            "--class-weight-scale-not-associated", str(cfg["not_scale"]),
            "--class-weight-scale-incidental", str(cfg["inc_scale"]),
        ]
        if args.pretrain_path:
            train_cmd.extend(["--pretrain-path", args.pretrain_path])
        run_cmd(train_cmd)

        eval_cmd = [
            sys.executable,
            "scripts/evaluation/evaluate_model_on_split.py",
            "--model", str(train_out / "final"),
            "--eval", args.eval_data,
            "--output", str(eval_out),
            "--batch-size", "64",
        ]
        run_cmd(eval_cmd)

        with open(eval_out) as f:
            eval_metrics = json.load(f)

        row = {
            "config": cfg,
            "model_path": str(train_out / "final"),
            "eval_path": str(eval_out),
            "accuracy": float(eval_metrics["accuracy"]),
            "macro_f1": float(eval_metrics["macro_f1"]),
            "weighted_f1": float(eval_metrics["weighted_f1"]),
            "associated_recall": float(eval_metrics["per_class"]["associated"]["recall"]),
            "incidental_precision": float(eval_metrics["per_class"]["incidental"]["precision"]),
        }
        rows.append(row)
        print("RESULT", row)

    best_acc = max(rows, key=lambda r: (r["accuracy"], r["macro_f1"]))
    best_macro = max(rows, key=lambda r: (r["macro_f1"], r["accuracy"]))

    summary = {
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "runs": rows,
        "best_by_accuracy": best_acc,
        "best_by_macro_f1": best_macro,
    }

    summary_path = output_base / "cost_sensitive_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print("saved_summary", summary_path)


if __name__ == "__main__":
    main()

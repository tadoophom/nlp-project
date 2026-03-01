"""Sweep learning rates for a given model and pick the best by eval F1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_LRS = [1e-5, 2e-5, 3e-5]


def run_train(
    data: str,
    output: str,
    model: str,
    lr: float,
    loss: str = "focal",
    focal_gamma: float = 2.0,
    epochs: int = 8,
    patience: int = 3,
    pretrain_path: str | None = None,
) -> dict | None:
    """Run train_bert.py and parse the results."""
    cmd = [
        sys.executable, "scripts/training/train_bert.py",
        "--data", data,
        "--output", output,
        "--model", model,
        "--lr", str(lr),
        "--loss", loss,
        "--focal-gamma", str(focal_gamma),
        "--epochs", str(epochs),
        "--patience", str(patience),
    ]
    if pretrain_path:
        cmd.extend(["--pretrain-path", pretrain_path])

    print(f"\n{'='*60}")
    print(f"Training {model} with lr={lr}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"FAILED: {result.stderr}")
        return None

    # Parse F1 from trainer output - look for eval_f1 in logs
    best_f1 = None
    for line in result.stdout.split("\n"):
        if "'f1'" in line or '"f1"' in line:
            import re
            match = re.search(r"'f1':\s*([\d.]+)", line) or re.search(r'"f1":\s*([\d.]+)', line)
            if match:
                f1 = float(match.group(1))
                if best_f1 is None or f1 > best_f1:
                    best_f1 = f1

    # Also check the trainer_state.json for the best metric
    state_path = Path(output) / "trainer_state.json"
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        if "best_metric" in state:
            best_f1 = state["best_metric"]

    return {"lr": lr, "f1": best_f1, "output": output}


def main():
    parser = argparse.ArgumentParser(description="Sweep learning rates")
    parser.add_argument("--data", required=True, help="Training data JSON")
    parser.add_argument("--output-base", required=True, help="Base output directory")
    parser.add_argument("--model", required=True, help="Model key (pubmedbert, scibert, etc.)")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS,
                        help="Learning rates to sweep")
    parser.add_argument("--loss", default="focal", choices=["weighted_ce", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--pretrain-path", type=str, default=None)
    args = parser.parse_args()

    results = []
    for lr in args.lrs:
        output_dir = f"{args.output_base}/lr_{lr:.0e}"
        r = run_train(
            data=args.data,
            output=output_dir,
            model=args.model,
            lr=lr,
            loss=args.loss,
            focal_gamma=args.focal_gamma,
            epochs=args.epochs,
            patience=args.patience,
            pretrain_path=args.pretrain_path,
        )
        if r:
            results.append(r)

    if not results:
        print("\nNo successful runs.")
        return

    print(f"\n{'='*60}")
    print(f"LR Sweep Results for {args.model}")
    print(f"{'='*60}")
    print(f"{'LR':>10} {'F1':>10} {'Output':>40}")
    print("-" * 62)
    for r in sorted(results, key=lambda x: x["f1"] or 0, reverse=True):
        f1_str = f"{r['f1']:.4f}" if r["f1"] else "N/A"
        print(f"{r['lr']:>10.0e} {f1_str:>10} {r['output']:>40}")

    best = max(results, key=lambda x: x["f1"] or 0)
    print(f"\nBest: lr={best['lr']:.0e} F1={best['f1']:.4f}")
    print(f"Model at: {best['output']}/final")

    # Save summary
    summary_path = Path(args.output_base) / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({"model": args.model, "results": results, "best": best}, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""Train an associated vs incidental expert classifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Train associated expert")
    parser.add_argument("--data", required=True, help="JSON dataset with sentence/label")
    parser.add_argument("--output", required=True, help="Path to save joblib model")
    parser.add_argument("--report", required=True, help="Path to save metrics JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.data) as f:
        rows = json.load(f)

    filtered = [r for r in rows if r["label"] in {"associated", "incidental"}]
    x = [r["sentence"] for r in filtered]
    y = [r["label"] for r in filtered]

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=args.seed,
        stratify=y if len(set(y)) > 1 else None,
    )

    clf = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000)),
            ("lr", LogisticRegression(max_iter=2500, class_weight="balanced")),
        ]
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_val)

    acc = accuracy_score(y_val, pred)
    macro = f1_score(y_val, pred, average="macro", zero_division=0)
    weighted = f1_score(y_val, pred, average="weighted", zero_division=0)
    report = classification_report(y_val, pred, output_dict=True, zero_division=0)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_path)

    payload = {
        "data": args.data,
        "train_size": len(x_train),
        "val_size": len(x_val),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "per_class": {
            "associated": {
                "precision": float(report["associated"]["precision"]),
                "recall": float(report["associated"]["recall"]),
                "f1": float(report["associated"]["f1-score"]),
            },
            "incidental": {
                "precision": float(report["incidental"]["precision"]),
                "recall": float(report["incidental"]["recall"]),
                "f1": float(report["incidental"]["f1-score"]),
            },
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("accuracy", payload["accuracy"])
    print("macro_f1", payload["macro_f1"])
    print("weighted_f1", payload["weighted_f1"])
    print("saved_model", out_path)
    print("saved_report", report_path)


if __name__ == "__main__":
    main()

"""
Generate a confusion matrix from a CSV file with true and predicted labels.
"""
import argparse
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix from CSV labels.")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--pred_col", type=str, default="predicted_label", help="Predicted label column name")
    parser.add_argument("--true_col", type=str, default="true_label", help="True label column name")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    y_true = df[args.true_col]
    y_pred = df[args.pred_col]
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()

"""Compare all 9 trained model/imbalance-method variants against the two
baselines from `src/baselines.py`, on a common set of metrics.

For every `models/<model>_<method>.pkl` bundle ({"model", "preprocessor"}),
scores `data/processed/test.csv` and computes:

- ROC-AUC, PR-AUC
- Precision/recall at a 0.5 probability threshold
- Effort-aware recall@20%: the fraction of actually-buggy test commits caught
  in the top 20% of commits ranked by predicted probability (Yang et al.,
  2016; see `src.baselines.recall_at_k`)

The same metrics are computed for the majority-class and size-ranking
baselines, using the same train/test split and metric functions, so the
whole comparison lives in one table: `reports/model_comparison.csv`, printed
sorted by PR-AUC descending.

Run as: python -m src.evaluate
"""
import sys
from pathlib import Path

# Allow `python src/evaluate.py` as well as `python -m src.evaluate`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from src.baselines import (
    LABEL_COLUMN,
    TEST_FILE,
    TRAIN_FILE,
    majority_class_baseline,
    recall_at_k,
    size_ranking_baseline,
)

MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
COMPARISON_FILE = REPORTS_DIR / "model_comparison.csv"

CLASSIFICATION_THRESHOLD = 0.5


def compute_metrics(y_true, scores):
    """ROC-AUC, PR-AUC, precision/recall at 0.5, and recall@20% effort."""
    preds = (scores >= CLASSIFICATION_THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision_at_0.5": precision_score(y_true, preds, zero_division=0),
        "recall_at_0.5": recall_score(y_true, preds, zero_division=0),
        "recall_at_20pct": recall_at_k(y_true, scores),
    }


def score_model_variant(bundle_path, X_test_raw):
    """Load a `{model, preprocessor}` bundle and score `X_test_raw` with it."""
    bundle = joblib.load(bundle_path)
    X_test = bundle["preprocessor"].transform(X_test_raw)
    return bundle["model"].predict_proba(X_test)[:, 1]


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    X_test_raw = test_df.drop(columns=[LABEL_COLUMN])
    y_test = test_df[LABEL_COLUMN].to_numpy()

    rows = []

    for bundle_path in sorted(MODELS_DIR.glob("*.pkl")):
        model_name, method = bundle_path.stem.rsplit("_", 1)
        scores = score_model_variant(bundle_path, X_test_raw)
        rows.append(
            {"name": bundle_path.stem, "model": model_name, "imbalance_method": method}
            | compute_metrics(y_test, scores)
        )

    baselines = {
        "majority_class": majority_class_baseline(train_df[LABEL_COLUMN], len(test_df)),
        "size_ranking": size_ranking_baseline(test_df),
    }
    for name, scores in baselines.items():
        rows.append(
            {"name": name, "model": name, "imbalance_method": "-"}
            | compute_metrics(y_test, scores)
        )

    comparison = pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_FILE, index=False)

    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(comparison.to_string(index=False))
    print(f"\nSaved: {COMPARISON_FILE}")

    return comparison


if __name__ == "__main__":
    main()

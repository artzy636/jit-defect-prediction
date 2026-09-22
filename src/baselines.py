"""Two baselines for the ApacheJIT just-in-time defect prediction task.

1. Majority-class classifier: always predicts the train split's majority
   label. A trained model earns its complexity only if it beats this floor.

2. Unsupervised size-ranking baseline: ranks commits by total change size
   `la + ld` (lines added + lines deleted), with no training at all. Yang et
   al. (2016), "Deep Learning for Just-In-Time Defect Prediction" (ICSE),
   found that this trivial heuristic can beat trained supervised models on
   effort-aware metrics, because commit size correlates strongly with both
   review effort and defect risk - a model needs to clear this bar too, not
   just beat random guessing.

Both baselines are evaluated on data/processed/test.csv. The majority-class
baseline is fit on data/processed/train.csv (it only needs the train label
distribution); the size-ranking baseline needs no fitting - its score is a
fixed function of a commit's own `la`/`ld` counts, so it's computed directly
on the test split with nothing learned from train.

Run as: python -m src.baselines
"""
import sys
from pathlib import Path

# Allow `python src/baselines.py` as well as `python -m src.baselines`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

LABEL_COLUMN = "label"
INSPECTION_EFFORT = 0.2


def recall_at_k(y_true, scores, k_frac=INSPECTION_EFFORT):
    """Recall among the top `k_frac` fraction of commits ranked by `scores`.

    This is the effort-aware metric from Yang et al. (2016): if you inspect
    only the top `k_frac` of commits (by descending score), what fraction of
    all actually-buggy commits in the test split did you catch? Ties are
    broken by a stable sort on the original row order, so a baseline that
    assigns identical scores to most rows (e.g. the majority classifier)
    still gets a fixed, reproducible top-k slice rather than a random one.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n_top = int(np.ceil(len(scores) * k_frac))
    order = np.argsort(-scores, kind="stable")
    top_idx = order[:n_top]
    total_positive = y_true.sum()
    return y_true[top_idx].sum() / total_positive


def evaluate(name, y_true, scores, note=None):
    metrics = {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "recall_at_20pct": recall_at_k(y_true, scores),
    }
    print(name)
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  Recall@20% effort: {metrics['recall_at_20pct']:.4f}")
    if note:
        print(f"  Note: {note}")
    print()
    return metrics


def majority_class_baseline(train_labels, n_test):
    """Score every test commit with the train split's majority-class rate.

    `DummyClassifier(strategy="most_frequent")` always predicts the label
    that's most common in `train_labels`, so `predict_proba` for the
    positive class is constant across every test row (0.0 if the majority
    class is 0, as it is here since buggy commits are the minority).
    """
    clf = DummyClassifier(strategy="most_frequent")
    clf.fit(np.zeros((len(train_labels), 1)), train_labels)
    return clf.predict_proba(np.zeros((n_test, 1)))[:, 1]


def size_ranking_baseline(df):
    """Unsupervised score: total change size `la + ld` (Yang et al., 2016)."""
    return (df["la"] + df["ld"]).to_numpy()


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    y_test = test_df[LABEL_COLUMN].to_numpy()

    results = {}

    majority_label = int(train_df[LABEL_COLUMN].mode()[0])
    majority_scores = majority_class_baseline(train_df[LABEL_COLUMN], len(test_df))
    results["majority_class"] = evaluate(
        "Majority-class classifier",
        y_test,
        majority_scores,
        note=(
            f"always predicts the train majority class ({majority_label}), so "
            "every test commit gets the same score - ROC-AUC/PR-AUC/recall@20% "
            "are degenerate here (constant-score ties), not a meaningful ranking."
        ),
    )

    size_scores = size_ranking_baseline(test_df)
    results["size_ranking"] = evaluate(
        "Unsupervised size-ranking baseline (la + ld)",
        y_test,
        size_scores,
    )

    return results


if __name__ == "__main__":
    main()

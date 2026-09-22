"""Chronological train/test split for the ApacheJIT dataset.

Splits data/raw/apachejit_total.csv by commit date (`author_date`), not
randomly: the oldest ~80% of commits become the training split and the most
recent ~20% become the test split.

Why chronological, not random: McIntosh & Kamei (2018, "Are Fix-Inducing
Changes a Moving Target? A Longitudinal Case Study of Just-In-Time Defect
Prediction", IEEE TSE) show that JIT defect models drift over time - the
relationship between commit features and bugginess shifts as a project
evolves. A random split scatters future commits into the training set, so
the model can pick up on time-correlated patterns (e.g. later conventions,
churn trends) that would never be available at real prediction time, when
you only ever have the past to train on. That leakage inflates evaluation
scores relative to real deployment, where a model trained today only ever
predicts on commits that haven't happened yet. Splitting by date keeps the
test split strictly in the training split's future, matching how the model
would actually be used.

Run as: python -m src.split
"""
import sys
from pathlib import Path

# Allow `python src/split.py` as well as `python -m src.split`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DATA_FILE = RAW_DIR / "apachejit_total.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"

TRAIN_FRACTION = 0.8

MANUAL_INSTRUCTIONS = f"""
{DATA_FILE} not found.

Download the ApacheJIT dataset first:
  python -m src.download_data
"""


def chronological_split(df, train_fraction=TRAIN_FRACTION):
    """Sort `df` by `author_date` and split into (train, test) by position.

    The oldest `train_fraction` of rows (by date) go to train; the remaining,
    most recent rows go to test. Ties on `author_date` keep their original
    relative order (stable sort), so the split is deterministic.
    """
    ordered = df.sort_values("author_date", kind="stable").reset_index(drop=True)
    split_idx = int(len(ordered) * train_fraction)
    return ordered.iloc[:split_idx].copy(), ordered.iloc[split_idx:].copy()


def _describe(name, split_df):
    dates = pd.to_datetime(split_df["author_date"], unit="s")
    counts = split_df["label"].value_counts().reindex([0, 1], fill_value=0)
    positive_rate = counts[1] / counts.sum() * 100

    print(f"{name}: {len(split_df):,} commits")
    print(f"  date range: {dates.min().date()} -> {dates.max().date()}")
    print(f"  label==1 (buggy): {counts[1]:,} ({positive_rate:.2f}%)")
    print(f"  label==0 (clean): {counts[0]:,} ({100 - positive_rate:.2f}%)")


def main():
    if not DATA_FILE.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    df["label"] = df["buggy"].astype(int)

    train_df, test_df = chronological_split(df)

    print(
        f"Full dataset: {len(df):,} commits, split "
        f"{TRAIN_FRACTION:.0%}/{1 - TRAIN_FRACTION:.0%} by date\n"
    )
    _describe("Train (oldest)", train_df)
    print()
    _describe("Test (most recent)", test_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)
    print(f"\nSaved: {TRAIN_FILE}, {TEST_FILE}")


if __name__ == "__main__":
    main()

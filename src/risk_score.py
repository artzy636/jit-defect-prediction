"""Single-commit risk scoring for the ApacheJIT just-in-time defect model.

Wraps `models/final_calibrated_model.pkl` (see `src/calibration.py`) and
`src.interpret.explain_commit` into one user-facing call: a 0-100 risk
score (the calibrated probability of the commit being buggy, as a
percentage), a coarse risk band, and the top 3 features driving that score,
written as short plain-language reasons.

Run as: python -m src.risk_score --commit-id <commit_id>
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

# Allow `python src/risk_score.py` as well as `python -m src.risk_score`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from src.interpret import LABEL_COLUMN, MODEL_FILE, TEST_FILE, commit_row, explain_commit

# (inclusive upper bound, band name), checked in ascending order.
BANDS = [
    (24, "Low"),
    (49, "Medium"),
    (74, "High"),
    (100, "Critical"),
]

# Short plain-language label for each feature LinearExplainer can attribute
# to (src.preprocessing.OUTPUT_FEATURES); direction ("up"/"down") is derived
# from the sign of that commit's SHAP contribution, not hardcoded here.
FEATURE_DESCRIPTIONS = {
    "la": "Large change",
    "ld": "Large deletion",
    "ns": "Many subsystems touched",
    "nd": "Many directories touched",
    "ent": "Changes spread across many files",
    "ndev": "Many recent contributors",
    "age": "Files not recently touched",
    "nuc": "Frequently-changed files",
    "aexp": "Experienced author",
    "asexp": "Author familiar with this subsystem",
}


def _band(score):
    for upper_bound, name in BANDS:
        if score <= upper_bound:
            return name
    raise AssertionError(f"score {score} out of the expected 0-100 range")


def _reason(feature_name, contribution):
    description = FEATURE_DESCRIPTIONS.get(feature_name, feature_name)
    direction = "up" if contribution > 0 else "down"
    return f"{description} ({feature_name}) - pushes risk {direction}"


@lru_cache(maxsize=1)
def _load_bundle():
    return joblib.load(MODEL_FILE)


def risk_score(commit_features):
    """Score one commit: `{"score": int, "band": str, "top_reasons": [str, ...]}`.

    `commit_features` follows the same contract as
    `src.interpret.commit_row` (a dict / Series / single-row DataFrame of
    raw ApacheJIT feature values). `score` is
    `round(100 * calibrated_probability)` from
    `models/final_calibrated_model.pkl`; `band` buckets it into Low (0-24),
    Medium (25-49), High (50-74), or Critical (75-100); `top_reasons` turns
    `explain_commit()`'s top-3 SHAP contributions into short sentences.
    """
    bundle = _load_bundle()
    row = commit_row(commit_features)
    X = bundle["preprocessor"].transform(row)
    probability = bundle["model"].predict_proba(X)[0, 1]

    score = round(100 * probability)
    top_reasons = [
        _reason(name, contribution) for name, contribution in explain_commit(commit_features)
    ]

    return {"score": score, "band": _band(score), "top_reasons": top_reasons}


def main():
    parser = argparse.ArgumentParser(description="Risk-score a commit from data/processed/test.csv")
    parser.add_argument("--commit-id", required=True, help="commit_id to look up in data/processed/test.csv")
    args = parser.parse_args()

    test_df = pd.read_csv(TEST_FILE)
    matches = test_df[test_df["commit_id"] == args.commit_id]
    if matches.empty:
        raise SystemExit(f"commit_id not found in {TEST_FILE}: {args.commit_id}")

    commit = matches.drop(columns=[LABEL_COLUMN]).iloc[0]
    result = risk_score(commit)

    print(f"commit_id: {args.commit_id}")
    print(f"score:     {result['score']}/100")
    print(f"band:      {result['band']}")
    print("top_reasons:")
    for reason in result["top_reasons"]:
        print(f"  - {reason}")


if __name__ == "__main__":
    main()

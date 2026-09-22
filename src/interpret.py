"""SHAP interpretability for the calibrated ApacheJIT defect-prediction model.

`models/final_calibrated_model.pkl` is a `{"model", "preprocessor"}` bundle
where `"model"` is a `CalibratedClassifierCV` (isotonic) wrapping a frozen
`LogisticRegression` (see `src/calibration.py`). Isotonic regression is a
monotonic remapping of the logistic regression's `decision_function`
(log-odds) output onto a calibrated probability - it doesn't change which
features drove that raw score up or down, only the scale of the final
number. So SHAP values here are computed with `shap.LinearExplainer` on the
underlying `LogisticRegression` directly, in log-odds space: additivity
holds exactly (`base_value + sum(shap_values) == decision_function(x)`), and
because isotonic calibration is monotonic, a feature's sign/ranking in
log-odds space carries over to its effect on the final calibrated
probability, even though the two scales aren't linearly related.

Run as: python -m src.interpret
"""
import sys
from functools import lru_cache
from pathlib import Path

# Allow `python src/interpret.py` as well as `python -m src.interpret`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import SEED
from src.preprocessing import INPUT_FEATURES

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")

MODEL_FILE = MODELS_DIR / "final_calibrated_model.pkl"
FIGURE_FILE = FIGURES_DIR / "shap_summary.png"

LABEL_COLUMN = "label"

# Background sample used by LinearExplainer to estimate each feature's
# expected value; drawn once, deterministically, from the training split.
BACKGROUND_SAMPLES = 100

# Palette (matching notebooks/01_eda.ipynb / src/calibration.py) -----------
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "text.color": INK,
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "font.size": 11,
})


def _logistic_regression(calibrated_model):
    """Pull the frozen `LogisticRegression` out of a fitted
    `CalibratedClassifierCV`. `src/calibration.py` fits it with
    `FrozenEstimator(base_model)` and no cross-validation, so there is
    exactly one `_CalibratedClassifier`, wrapping the frozen estimator,
    wrapping the logistic regression untouched by calibration.
    """
    calibrated_classifier = calibrated_model.calibrated_classifiers_[0]
    return calibrated_classifier.estimator.estimator


@lru_cache(maxsize=1)
def _load_explainer():
    """Build (once) the `LinearExplainer`, preprocessor, and feature names
    shared by `main()`'s global summary and by `explain_commit()`.
    """
    bundle = joblib.load(MODEL_FILE)
    preprocessor = bundle["preprocessor"]
    logreg = _logistic_regression(bundle["model"])

    train_df = pd.read_csv(TRAIN_FILE)
    X_train_raw = train_df.drop(columns=[LABEL_COLUMN])
    X_train = preprocessor.transform(X_train_raw)
    background = X_train.sample(n=BACKGROUND_SAMPLES, random_state=SEED)

    explainer = shap.LinearExplainer(logreg, background)
    feature_names = list(preprocessor.get_feature_names_out())
    return explainer, preprocessor, feature_names


def commit_row(commit_features):
    """Normalize one commit's raw ApacheJIT feature values - a dict, a
    `pandas.Series`, or a single-row `DataFrame` - into a single-row
    `DataFrame` ready for `JITFeaturePreprocessor.transform`.

    The input must cover at least `src.preprocessing.INPUT_FEATURES` (`la`,
    `ld`, `ns`, `nd`, `nf`, `ent`, `ndev`, `age`, `nuc`, `aexp`, `arexp`,
    `asexp`); extra columns such as `commit_id` are ignored, so a raw row
    from `data/processed/test.csv` works directly.
    """
    if isinstance(commit_features, dict):
        row = pd.DataFrame([commit_features])
    elif isinstance(commit_features, pd.Series):
        row = commit_features.to_frame().T
    elif isinstance(commit_features, pd.DataFrame):
        if len(commit_features) != 1:
            raise ValueError("commit_row expects exactly one commit's features")
        row = commit_features
    else:
        raise TypeError(f"Unsupported commit_features type: {type(commit_features)}")

    # A Series/DataFrame built from a mixed-type raw commit row (strings
    # like `commit_id` alongside numeric features) can end up with the
    # numeric feature columns boxed as dtype "object", which breaks
    # `np.log1p` inside `preprocessor.transform`. Coerce just those columns.
    numeric_cols = [c for c in INPUT_FEATURES if c in row.columns]
    row = row.copy()
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric)
    return row


def explain_commit(commit_features):
    """Top-3 features pushing a single commit's score up or down.

    `commit_features` follows the same contract as `commit_row()`.

    Returns up to 3 `(feature_name, contribution)` pairs, sorted by
    `abs(contribution)` descending. `contribution` is that feature's signed
    SHAP value in log-odds space (positive pushes the score toward
    "buggy", negative toward "clean") - see the module docstring for why
    log-odds, not the calibrated probability, is what `LinearExplainer` can
    attribute additively here.
    """
    row = commit_row(commit_features)
    explainer, preprocessor, feature_names = _load_explainer()
    X = preprocessor.transform(row)
    shap_values = explainer(X).values[0]

    contributions = sorted(
        zip(feature_names, (float(v) for v in shap_values)),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    return contributions[:3]


def plot_summary(shap_values, path):
    """Global beeswarm summary plot (every test commit, every feature)."""
    fig = plt.figure(figsize=(8, 5.5))
    shap.plots.beeswarm(shap_values, show=False, max_display=len(shap_values.feature_names))
    plt.gca().set_title(
        "SHAP feature impact - final_calibrated_model (log-odds scale)",
        color=INK, fontweight="bold", loc="left",
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    explainer, preprocessor, feature_names = _load_explainer()

    test_df = pd.read_csv(TEST_FILE)
    X_test_raw = test_df.drop(columns=[LABEL_COLUMN])
    X_test = preprocessor.transform(X_test_raw)

    shap_values = explainer(X_test)

    plot_summary(shap_values, FIGURE_FILE)
    print(f"Saved: {FIGURE_FILE}")

    mean_abs = pd.Series(
        np.abs(shap_values.values).mean(axis=0), index=feature_names
    ).sort_values(ascending=False)
    print("\nGlobal feature importance (mean |SHAP| on log-odds score):")
    for name, val in mean_abs.items():
        print(f"  {name:8s} {val:.4f}")

    example_id = test_df.loc[0, "commit_id"]
    print(f"\nexplain_commit example (commit {example_id}):")
    for name, contribution in explain_commit(X_test_raw.iloc[0]):
        print(f"  {name:8s} {contribution:+.4f}")

    return shap_values


if __name__ == "__main__":
    main()

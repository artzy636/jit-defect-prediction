"""Post-hoc probability calibration for the SMOTE-trained logistic regression.

`src/train.py`'s logreg_smote variant is fit on a 1:1 SMOTE-resampled training
set, so its `predict_proba` output is shaped by that artificial 50/50 balance,
not by the true ~16% buggy rate - a commit it scores 0.7 has no guarantee of
being buggy 70% of the time. This script checks that directly with a
reliability diagram and Brier score, then fixes it with isotonic regression
via `CalibratedClassifierCV`.

Pipeline:
1. Load `models/logreg_smote.pkl` ({model, preprocessor}) and score
   `data/processed/test.csv` -> reliability diagram + Brier score ("before").
2. Take the most recent `CALIBRATION_FRACTION` of `data/processed/train.csv`
   (chronological, same reasoning as `src/split.py`) as a calibration slice.
   Wrap the already-fitted model in `sklearn.frozen.FrozenEstimator` and fit
   `CalibratedClassifierCV(method="isotonic")` on that slice - this only fits
   the isotonic recalibration; the underlying `LogisticRegression` and
   `JITFeaturePreprocessor` are frozen. Note this slice isn't held out from
   the base model's *training* data (train.py already fit it on the full
   split), only from the test split - which is what matters for an honest
   before/after Brier comparison there.
3. Re-score test.csv with the calibrated model -> reliability diagram +
   Brier score ("after"), plotted alongside "before" for direct comparison.
4. Save the calibrated {model, preprocessor} bundle to
   `models/final_calibrated_model.pkl`.

Run as: python -m src.calibration
"""
import sys
from pathlib import Path

# Allow `python src/calibration.py` as well as `python -m src.calibration`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
MODELS_DIR = Path("models")
FIGURES_DIR = Path("reports/figures")

BASE_MODEL_FILE = MODELS_DIR / "logreg_smote.pkl"
CALIBRATED_MODEL_FILE = MODELS_DIR / "final_calibrated_model.pkl"
FIGURE_FILE = FIGURES_DIR / "reliability_diagram.png"

LABEL_COLUMN = "label"
N_BINS = 10
CALIBRATION_FRACTION = 0.2

# Palette (fixed roles, matching notebooks/01_eda.ipynb) ---------------------
BLUE = "#2a78d6"    # before calibration (raw logreg_smote)
ORANGE = "#eb6834"  # after calibration (isotonic)
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY_INK,
    "text.color": INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "axes.grid": False,
    "font.size": 11,
})


def style_axes(ax, y_grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    if y_grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def calibration_slice(train_df, fraction=CALIBRATION_FRACTION):
    """Most recent `fraction` of `train_df`, ordered by `author_date`."""
    ordered = train_df.sort_values("author_date", kind="stable").reset_index(drop=True)
    split_idx = int(len(ordered) * (1 - fraction))
    return ordered.iloc[split_idx:].copy()


def score(bundle, X_raw):
    """Score `X_raw` with a `{model, preprocessor}` bundle."""
    X = bundle["preprocessor"].transform(X_raw)
    return bundle["model"].predict_proba(X)[:, 1]


def plot_reliability(ax_curve, ax_hist, y_true, before, after):
    prob_true_b, prob_pred_b = calibration_curve(y_true, before, n_bins=N_BINS, strategy="uniform")
    prob_true_a, prob_pred_a = calibration_curve(y_true, after, n_bins=N_BINS, strategy="uniform")

    ax_curve.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, color=MUTED, label="Perfectly calibrated", zorder=2)
    ax_curve.plot(prob_pred_b, prob_true_b, marker="o", markersize=7, linewidth=2, color=BLUE,
                  label="Before (raw logreg_smote)", zorder=3)
    ax_curve.plot(prob_pred_a, prob_true_a, marker="o", markersize=7, linewidth=2, color=ORANGE,
                  label="After (isotonic calibrated)", zorder=3)

    style_axes(ax_curve)
    ax_curve.set_xlim(0, 1)
    ax_curve.set_ylim(0, 1)
    ax_curve.set_ylabel("Observed frequency (buggy)")
    ax_curve.set_title("Reliability diagram - logreg_smote", color=INK, fontweight="bold", loc="left")
    ax_curve.legend(frameon=False, loc="upper left")

    bins = np.linspace(0, 1, N_BINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    bar_width = (bins[1] - bins[0]) * 0.42
    hist_before, _ = np.histogram(before, bins=bins)
    hist_after, _ = np.histogram(after, bins=bins)
    ax_hist.bar(centers - bar_width / 2, hist_before, width=bar_width, color=BLUE, zorder=3)
    ax_hist.bar(centers + bar_width / 2, hist_after, width=bar_width, color=ORANGE, zorder=3)

    style_axes(ax_hist)
    ax_hist.set_xlim(0, 1)
    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Commits")
    ax_hist.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    X_test_raw = test_df.drop(columns=[LABEL_COLUMN])
    y_test = test_df[LABEL_COLUMN].to_numpy()

    bundle = joblib.load(BASE_MODEL_FILE)
    preprocessor = bundle["preprocessor"]

    probs_before = score(bundle, X_test_raw)
    brier_before = brier_score_loss(y_test, probs_before)

    calib_df = calibration_slice(train_df)
    X_calib_raw = calib_df.drop(columns=[LABEL_COLUMN])
    y_calib = calib_df[LABEL_COLUMN].to_numpy()
    X_calib = preprocessor.transform(X_calib_raw)

    calibrated_model = CalibratedClassifierCV(FrozenEstimator(bundle["model"]), method="isotonic")
    calibrated_model.fit(X_calib, y_calib)
    calibrated_bundle = {"model": calibrated_model, "preprocessor": preprocessor}

    probs_after = score(calibrated_bundle, X_test_raw)
    brier_after = brier_score_loss(y_test, probs_after)

    print(f"Test: {len(test_df):,} commits, {y_test.mean():.2%} buggy")
    print(
        f"Calibration slice: {len(calib_df):,} commits, the most recent "
        f"{CALIBRATION_FRACTION:.0%} of train.csv ({y_calib.mean():.2%} buggy)\n"
    )
    print(f"Brier score  before calibration: {brier_before:.4f}")
    print(f"Brier score  after  calibration: {brier_after:.4f}")
    print(f"Relative improvement: {(brier_before - brier_after) / brier_before:+.1%}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_curve, ax_hist) = plt.subplots(2, 1, figsize=(7, 8), height_ratios=[3, 1], sharex=True)
    plot_reliability(ax_curve, ax_hist, y_test, probs_before, probs_after)
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=150, facecolor=SURFACE)
    print(f"\nSaved: {FIGURE_FILE}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated_bundle, CALIBRATED_MODEL_FILE)
    print(f"Saved: {CALIBRATED_MODEL_FILE}")

    return {"brier_before": brier_before, "brier_after": brier_after}


if __name__ == "__main__":
    main()

"""Train Logistic Regression, Random Forest, and XGBoost on the ApacheJIT
just-in-time defect prediction training split, each combined with three
class-imbalance strategies:

1. none        - no resampling, rely on `class_weight="balanced"` only
                  (XGBoost has no `class_weight`, so it gets an equivalent
                  `scale_pos_weight` computed from the train label counts).
2. undersample - random undersampling of the majority class to 1:1.
3. smote       - SMOTE oversampling of the minority class to 1:1.

That's 3 models x 3 imbalance strategies = 9 variants. Each variant fits its
own `JITFeaturePreprocessor` on the (un-resampled) training split only -
resampling happens after preprocessing, so synthetic/duplicated rows never
influence the scaler's statistics. Every variant is saved to `models/` as a
`{model, preprocessor}` bundle, filename-encoded as `models/<model>_<method>.pkl`.

Run as: python -m src.train
"""
import sys
import time
from pathlib import Path

# Allow `python src/train.py` as well as `python -m src.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import SEED
from src.preprocessing import JITFeaturePreprocessor

PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.csv"
MODELS_DIR = Path("models")

LABEL_COLUMN = "label"

IMBALANCE_METHODS = ["none", "undersample", "smote"]


def make_model(name, y_train):
    """Construct an unfitted estimator for `name`, balanced for `y_train`."""
    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced", random_state=SEED, max_iter=1000
        )
    if name == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=SEED)
    if name == "xgboost":
        # XGBoost has no `class_weight`; `scale_pos_weight` is its equivalent
        # for a binary task - the ratio of negative to positive train counts.
        neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
        return XGBClassifier(
            scale_pos_weight=neg / pos,
            random_state=SEED,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown model: {name}")


def resample(method, X_train, y_train):
    """Apply `method` to already-preprocessed features. "none" is a no-op:
    class imbalance is instead handled by the model's `class_weight`/
    `scale_pos_weight`, set in `make_model`."""
    if method == "none":
        return X_train, y_train
    if method == "undersample":
        sampler = RandomUnderSampler(random_state=SEED)
    elif method == "smote":
        sampler = SMOTE(random_state=SEED)
    else:
        raise ValueError(f"Unknown imbalance method: {method}")
    return sampler.fit_resample(X_train, y_train)


def train_variant(model_name, method, X_train_raw, y_train):
    preprocessor = JITFeaturePreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)

    X_res, y_res = resample(method, X_train, y_train)

    # Balancing strategy for "none" is baked into the estimator
    # (class_weight/scale_pos_weight); resampled variants are already
    # balanced 1:1, so the estimator gets no extra class weighting.
    model_y = y_train if method == "none" else y_res
    model = make_model(model_name, model_y)

    start = time.perf_counter()
    model.fit(X_res, y_res)
    elapsed = time.perf_counter() - start

    filename = MODELS_DIR / f"{model_name}_{method}.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "preprocessor": preprocessor}, filename)

    print(f"{model_name:14s} + {method:11s}  {elapsed:7.2f}s  -> {filename}")
    return elapsed


def main():
    train_df = pd.read_csv(TRAIN_FILE)
    X_train_raw = train_df.drop(columns=[LABEL_COLUMN])
    y_train = train_df[LABEL_COLUMN].to_numpy()

    print(f"Train: {len(train_df):,} rows, {y_train.mean():.2%} buggy\n")

    total_start = time.perf_counter()
    for model_name in ["logreg", "random_forest", "xgboost"]:
        for method in IMBALANCE_METHODS:
            train_variant(model_name, method, X_train_raw, y_train)
    total_elapsed = time.perf_counter() - total_start

    print(f"\nTotal training time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

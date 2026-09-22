"""Feature preprocessing for ApacheJIT just-in-time defect prediction.

Implements the transform decided in notebooks/01_eda.ipynb:

- Log1p the right-skewed size/diffusion count features (la, ld, ns, nd, nf).
- Drop `arexp` (keep `aexp`) and `nf` (keep `nd`) - the EDA's correlation
  matrix flags aexp/arexp at 0.94 and nd/nf at 0.81, both above the |r| > 0.8
  threshold, so each pair is collapsed to one feature.
- Standard-scale everything that's left.

Usage (fit only on the training split, never on validation/test):

    prep = JITFeaturePreprocessor()
    X_train_t = prep.fit_transform(X_train)
    X_test_t = prep.transform(X_test)  # reuses X_train's log/scale statistics

`transform()` never recomputes statistics - it only applies whatever
`StandardScaler` was produced by the single `fit()` call, so it cannot leak
test-split statistics into the training features. Calling `transform()`
before `fit()` raises `NotFittedError` instead of silently scaling with
default (unfit) statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# Right-skewed size/diffusion counts: log1p before scaling.
LOG_FEATURES = ["la", "ld", "ns", "nd", "nf"]

# Scaled as-is (not log-transformed).
PASSTHROUGH_FEATURES = ["ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"]

# Dropped for collinearity with a feature already kept, per the EDA
# correlation matrix (notebooks/01_eda.ipynb): aexp/arexp = 0.94, nd/nf = 0.81.
DROPPED_FEATURES = ["arexp", "nf"]

INPUT_FEATURES = LOG_FEATURES + PASSTHROUGH_FEATURES
OUTPUT_FEATURES = [f for f in INPUT_FEATURES if f not in DROPPED_FEATURES]


class JITFeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Scikit-learn transformer for the ApacheJIT commit-level features.

    Selects `OUTPUT_FEATURES` by column name (ignoring identifier/label
    columns such as `commit_id`, `project`, `buggy`, `year`), log1p-transforms
    the skewed size/diffusion columns, and standard-scales the result.
    """

    def fit(self, X, y=None):
        engineered = self._engineer(X)
        self.scaler_ = StandardScaler().fit(engineered)
        self.feature_names_in_ = np.array(engineered.columns, dtype=object)
        return self

    def transform(self, X):
        check_is_fitted(self, "scaler_")
        engineered = self._engineer(X)
        scaled = self.scaler_.transform(engineered)
        return pd.DataFrame(scaled, columns=self.feature_names_in_, index=engineered.index)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "scaler_")
        return self.feature_names_in_.copy()

    @staticmethod
    def _engineer(X):
        missing = [c for c in INPUT_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected ApacheJIT feature columns: {missing}")

        engineered = X[INPUT_FEATURES].drop(columns=DROPPED_FEATURES).copy()
        present_log_features = [c for c in LOG_FEATURES if c in engineered.columns]
        engineered[present_log_features] = np.log1p(engineered[present_log_features])
        return engineered

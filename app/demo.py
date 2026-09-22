"""Streamlit demo: JIT defect risk score for a single commit.

Loads `models/final_calibrated_model.pkl` and a 200-row random sample of
`data/processed/test.csv` once at startup (not the full ~21k rows), so the
app stays fast. Pick a commit from that sample, or paste a commit's raw
ApacheJIT feature values, and see its calibrated risk score, risk band,
top 3 SHAP reasons (via `src.risk_score.risk_score`), and percentile among
the sampled commits.

Run as: streamlit run app/demo.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.config import SEED
from src.interpret import MODEL_FILE, TEST_FILE
from src.preprocessing import INPUT_FEATURES
from src.risk_score import risk_score

SAMPLE_SIZE = 200

# Mirrors the band names produced by src.risk_score._band / BANDS.
BAND_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}

st.set_page_config(page_title="JIT Defect Risk", page_icon="\U0001F6A6", layout="centered")


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_FILE)


@st.cache_data
def load_scored_sample():
    """A fixed 200-row random sample of the test set, scored once."""
    df = pd.read_csv(TEST_FILE)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=SEED).reset_index(drop=True)

    bundle = load_bundle()
    X = bundle["preprocessor"].transform(sample[INPUT_FEATURES])
    probabilities = bundle["model"].predict_proba(X)[:, 1]
    scores = np.round(100 * probabilities).astype(int)

    return sample, scores


def percentile_of(score, scores):
    return 100 * float((scores <= score).mean())


def parse_pasted_commit(text):
    """Parse pasted commit metadata into a `{feature: value}` dict.

    Accepts a JSON object, or `key=value` / `key: value` pairs separated by
    commas or newlines. Extra fields (e.g. `commit_id`) are ignored; raises
    `ValueError` with a human-readable message if any of
    `src.preprocessing.INPUT_FEATURES` is missing or unparsable.
    """
    text = text.strip()
    if not text:
        raise ValueError("Paste some commit metadata first.")

    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Could not parse as JSON: {e}")
    else:
        data = {}
        for entry in re.split(r"[,\n]+", text):
            entry = entry.strip()
            if not entry:
                continue
            if "=" in entry:
                key, _, value = entry.partition("=")
            elif ":" in entry:
                key, _, value = entry.partition(":")
            else:
                raise ValueError(f"Could not parse {entry!r} - expected key=value or key: value")
            data[key.strip()] = value.strip()

    missing = [f for f in INPUT_FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

    try:
        return {f: float(data[f]) for f in INPUT_FEATURES}
    except (TypeError, ValueError) as e:
        raise ValueError(f"Non-numeric feature value: {e}")


def render_result(result, scores):
    score, band = result["score"], result["band"]
    color = BAND_COLORS[band]

    st.markdown(
        f"""
        <div style="text-align:center; margin: 1.5rem 0 0.5rem;">
            <div style="font-size:5.5rem; font-weight:800; color:{color}; line-height:1;">{score}</div>
            <div style="font-size:1.25rem; font-weight:600; color:{color};">{band} risk</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"{percentile_of(score, scores):.0f}th percentile among {len(scores)} sampled test commits"
    )

    st.subheader("Top reasons")
    for reason in result["top_reasons"]:
        st.markdown(f"- {reason}")


def main():
    st.title("JIT Defect Risk Score")
    st.write(
        "Pick a sample commit or paste a commit's ApacheJIT features to see "
        "its predicted risk of introducing a bug."
    )

    sample, scores = load_scored_sample()

    mode = st.radio("Commit source", ["Pick from sample", "Paste commit metadata"], horizontal=True)

    commit_features = None

    if mode == "Pick from sample":
        commit_id = st.selectbox("Sample commit", sample["commit_id"])
        commit_features = sample.loc[sample["commit_id"] == commit_id].iloc[0]
    else:
        example_row = sample.iloc[0]
        example = ", ".join(f"{f}={example_row[f]}" for f in INPUT_FEATURES)
        text = st.text_area(
            "Commit metadata",
            placeholder=(
                'JSON, e.g. {"la": 10, "ld": 2, "ns": 1, ...}\n'
                f"or key=value pairs, e.g.\n{example}"
            ),
            height=160,
        )
        if text.strip():
            try:
                commit_features = parse_pasted_commit(text)
            except ValueError as e:
                st.error(str(e))

    if commit_features is not None:
        with st.spinner("Scoring..."):
            result = risk_score(commit_features)
        render_result(result, scores)


if __name__ == "__main__":
    main()

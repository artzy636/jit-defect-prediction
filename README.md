# JIT Defect Prediction

This project implements a just-in-time (JIT) defect prediction pipeline on the ApacheJIT dataset, aiming to predict whether a software commit is likely to introduce a defect at the time it is made. It covers data ingestion and preprocessing, feature engineering, model training and evaluation (including handling of class imbalance), probability calibration, SHAP-based interpretability, and a Streamlit demo for interactive exploration of predictions.

## Setup

Requires Python 3.11+ (developed and tested on Python 3.14.0).

```powershell
# From the repository root
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS/Linux, activate with `source venv/bin/activate` instead. If PowerShell blocks the activation script with an execution-policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

`requirements.txt` pins `ipykernel`, which provides the Python kernel used to run `notebooks/01_eda.ipynb` from VS Code's Jupyter extension. If you'd rather run the notebook from a browser via the classic `jupyter notebook` / `jupyter lab` command, also install a frontend: `pip install jupyterlab`.

## Running the pipeline

Run these from the repository root, in order, with the virtual environment activated. Each step consumes the output of the one before it. All scripts are deterministic (`src/config.py` fixes `SEED = 42`), so re-running the pipeline reproduces the same split, models, and metrics.

1. **Download the raw dataset** — fetches `apachejit_total.csv` and `apachejit_train.csv` (ApacheJIT, Zenodo record [5907002](https://zenodo.org/records/5907002)) into `data/raw/`. Only `apachejit_total.csv` is used downstream; this pipeline derives its own chronological split rather than using the release's pre-made train file.
   ```powershell
   python -m src.download_data
   ```
2. **Verify the download** — asserts row counts and label balance against known-good values and prints a column/dtype summary.
   ```powershell
   python -m src.verify_data
   ```
3. **Chronological train/test split** — sorts commits by `author_date` and writes `data/processed/train.csv` (oldest 80%) and `data/processed/test.csv` (most recent 20%), so the test split is strictly in the future relative to training, matching real deployment.
   ```powershell
   python -m src.split
   ```
4. **Exploratory data analysis** — open [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb) in VS Code (or Jupyter) and run all cells. Confirms class balance and date coverage, and identifies the correlated feature pairs (`aexp`/`arexp`, `nd`/`nf`) that `src/preprocessing.py` collapses. Saves `reports/figures/class_balance.png` and `reports/figures/commits_per_year.png`.
5. **Train all model variants** — 3 models (logistic regression, random forest, XGBoost) × 3 imbalance strategies (none, random undersampling, SMOTE) = 9 `{model, preprocessor}` bundles saved to `models/`.
   ```powershell
   python -m src.train
   ```
6. **Evaluate and compare** — scores all 9 variants plus the majority-class and size-ranking baselines against `data/processed/test.csv`; writes `reports/model_comparison.csv` (see below).
   ```powershell
   python -m src.evaluate
   ```
7. **Calibrate probabilities** — fits isotonic calibration on top of `logreg_smote` (whose raw `predict_proba` is skewed by SMOTE's artificial 50/50 balance), saves `models/final_calibrated_model.pkl` and `reports/figures/reliability_diagram.png`.
   ```powershell
   python -m src.calibration
   ```
8. **SHAP interpretability** — computes global feature importance for the calibrated model via `shap.LinearExplainer`, saves `reports/figures/shap_summary.png`.
   ```powershell
   python -m src.interpret
   ```
9. **Risk-score a single commit** — looks up a commit by ID in `data/processed/test.csv` and prints its calibrated 0–100 risk score, risk band, and top 3 SHAP-based reasons.
   ```powershell
   python -m src.risk_score --commit-id <commit_id>
   ```
10. **Interactive demo** — pick a sample commit or paste raw commit metadata and see its risk score, band, and reasons live.
    ```powershell
    streamlit run app/demo.py
    ```

## Model comparison

Full results in [reports/model_comparison.csv](reports/model_comparison.csv), sorted by PR-AUC descending:

| Model | Imbalance method | ROC-AUC | PR-AUC | Precision@0.5 | Recall@0.5 | Recall@20% effort |
|---|---|---|---|---|---|---|
| logreg | smote | 0.8032 | 0.4436 | 0.3151 | 0.7879 | 0.5251 |
| logreg | none | 0.8030 | 0.4432 | 0.3146 | 0.7908 | 0.5239 |
| logreg | undersample | 0.8022 | 0.4418 | 0.3136 | 0.7896 | 0.5228 |
| xgboost | none | 0.7935 | 0.4283 | 0.3139 | 0.7762 | 0.5030 |
| xgboost | undersample | 0.7945 | 0.4275 | 0.3054 | 0.8074 | 0.4976 |
| random_forest | none | 0.7923 | 0.4250 | 0.4032 | 0.5171 | 0.5007 |
| xgboost | smote | 0.7939 | 0.4241 | 0.3513 | 0.6821 | 0.4944 |
| random_forest | undersample | 0.7927 | 0.4178 | 0.3068 | 0.8146 | 0.5033 |
| random_forest | smote | 0.7861 | 0.4056 | 0.3425 | 0.6887 | 0.4907 |
| size_ranking (baseline) | – | 0.7772 | 0.4038 | 0.1635 | 1.0000 | 0.4933 |
| majority_class (baseline) | – | 0.5000 | 0.1635 | 0.0000 | 0.0000 | 0.3035 |

All 9 trained variants comfortably beat both baselines on PR-AUC and ROC-AUC. The three logistic regression variants are effectively tied for the best PR-AUC and outperform random forest and XGBoost; `logreg_smote` is the one carried forward into calibration, interpretability, and the risk-scoring CLI/demo. Random forest reaches the highest precision@0.5 but at a large recall cost, and the unsupervised size-ranking baseline (commit size alone) is a surprisingly strong reference point per Yang et al. (2016).

## Limitations

### Label noise (SZZ)

The `buggy` label in ApacheJIT is derived by tracing bug-fix commits back to the earlier commit(s) that introduced the fixed defect — the SZZ algorithm. SZZ is known to systematically mislabel a substantial share of commits: Fan et al. (2021), *"The Impact of Mislabeled Changes by SZZ on Just-in-Time Defect Prediction"* (IEEE Transactions on Software Engineering, 47(8), 1559–1586), found that a large fraction of SZZ-identified bug-inducing changes are noise, and that this noise measurably distorts JIT defect prediction performance and feature importance. Every metric in `reports/model_comparison.csv`, and every score produced by `src/risk_score.py`, is trained and evaluated against these SZZ-derived labels — they should be read as estimates that inherit this labeling noise, not as ground truth.

### Generalizability

The dataset covers 15 GitHub repositories spanning 14 Apache Software Foundation projects (Hadoop is split across three repos: hadoop, hadoop-hdfs, hadoop-mapreduce) (2003–2019), each selected for being predominantly Java codebases with a long, well-instrumented commit history. This limits how far the results should be expected to transfer:

- **Ecosystem**: the model has only ever seen ASF projects, with ASF-specific code review norms, contributor tenure, and governance. It has not seen closed-source repositories, other open-source foundations, or projects without a comparable review process.
- **Language**: results are specific to Java-dominant codebases; JIT features like `ns`/`nd`/`ent` and the log1p/scaling choices in `src/preprocessing.py` were tuned on this dataset and may behave differently for other languages, build systems, or commit conventions.
- **Time period**: all commits fall between 2003 and 2019, predating many now-common workflows (squash-merge pull requests, monorepos, AI-assisted commits). A model trained here may not transfer cleanly to codebases with substantially different commit granularity or style today.

"""Load the ApacheJIT dataset and verify its row counts and label balance.

Run as: python -m src.verify_data
"""
import random
import sys
from pathlib import Path

# Allow `python src/verify_data.py` as well as `python -m src.verify_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import SEED

random.seed(SEED)

RAW_DIR = Path("data/raw")
DATA_FILE = RAW_DIR / "apachejit_total.csv"
LABEL_SOURCE_COLUMN = "buggy"  # True/False -> label 1/0

EXPECTED_TOTAL_ROWS = 106674
EXPECTED_POSITIVE = 28239
EXPECTED_NEGATIVE = 78435

MANUAL_INSTRUCTIONS = f"""
{DATA_FILE} not found.

Download the ApacheJIT dataset first:
  python -m src.download_data

Or manually:
  1. Open https://zenodo.org/records/5907002 in a browser.
  2. In the "Files" section, download apachejit_total.csv.
  3. Move it into: {RAW_DIR.resolve()}
"""


def main():
    if not DATA_FILE.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)
    df["label"] = df[LABEL_SOURCE_COLUMN].astype(int)

    total_rows = len(df)
    positive_count = int((df["label"] == 1).sum())
    negative_count = int((df["label"] == 0).sum())

    assert total_rows == EXPECTED_TOTAL_ROWS, (
        f"Expected {EXPECTED_TOTAL_ROWS} total rows, got {total_rows}"
    )
    assert positive_count == EXPECTED_POSITIVE, (
        f"Expected {EXPECTED_POSITIVE} label==1 rows, got {positive_count}"
    )
    assert negative_count == EXPECTED_NEGATIVE, (
        f"Expected {EXPECTED_NEGATIVE} label==0 rows, got {negative_count}"
    )

    print(f"Loaded {DATA_FILE} (seed={SEED})")
    print(f"Total rows: {total_rows}")
    print(f"label==1 (buggy):  {positive_count}")
    print(f"label==0 (clean):  {negative_count}")
    print()
    print("Column / dtype summary:")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
    })
    print(summary.to_string())
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()

"""Download the ApacheJIT dataset (Zenodo record 5907002) into data/raw/.

Run as: python -m src.download_data
"""
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Allow `python src/download_data.py` as well as `python -m src.download_data`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SEED

random.seed(SEED)

ZENODO_RECORD_ID = "5907002"
RECORD_API_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}"
RAW_DIR = Path("data/raw")
WANTED_FILES = ["apachejit_total.csv", "apachejit_train.csv"]
USER_AGENT = "jit-defect-prediction/1.0"
TIMEOUT_SECONDS = 60
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 3

MANUAL_INSTRUCTIONS = f"""
Automatic download failed.

Please download the ApacheJIT dataset manually:
  1. Open https://zenodo.org/records/{ZENODO_RECORD_ID} in a browser.
  2. In the "Files" section, download:
{chr(10).join(f"       - {name}" for name in WANTED_FILES)}
  3. Move the downloaded file(s) into: {RAW_DIR.resolve()}
  4. Re-run this script (it will skip files already present) or
     run src/verify_data.py directly to check the data.
"""


def _fetch_json(url):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url, dest):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        with open(tmp_dest, "wb") as out_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
    tmp_dest.replace(dest)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        record = _fetch_json(RECORD_API_URL)
        files_by_name = {f["key"]: f for f in record["files"]}

        for filename in WANTED_FILES:
            if filename not in files_by_name:
                raise KeyError(f"'{filename}' not found in Zenodo record {ZENODO_RECORD_ID}")

            file_info = files_by_name[filename]
            download_url = file_info["links"]["self"]
            expected_size = file_info["size"]
            dest = RAW_DIR / filename

            if dest.exists() and dest.stat().st_size == expected_size:
                print(f"{filename} already present in {RAW_DIR}, skipping.")
                continue

            print(f"Downloading {filename} ({expected_size:,} bytes)...")
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    _download_file(download_url, dest)
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as exc:
                    print(f"  attempt {attempt}/{MAX_ATTEMPTS} failed: {exc}")
                    if attempt == MAX_ATTEMPTS:
                        raise
                    time.sleep(RETRY_DELAY_SECONDS)

            actual_size = dest.stat().st_size
            if actual_size != expected_size:
                raise IOError(
                    f"Downloaded size for {filename} ({actual_size} bytes) "
                    f"does not match expected size ({expected_size} bytes)"
                )
            print(f"Saved to {dest}")

    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    print("Download complete.")


if __name__ == "__main__":
    main()

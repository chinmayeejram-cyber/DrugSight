"""
Step 1: Collect EGFR bioactivity data from ChEMBL (with progress output).
Target: EGFR -> CHEMBL203
Output: data/raw/egfr_raw.csv
"""
import os
import time
import requests
import pandas as pd

TARGET_ID = "CHEMBL203"  # Human EGFR
RAW_PATH = os.path.join("data", "raw", "egfr_raw.csv")
URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
PAGE = 1000


def get_page(offset, retries=5):
    params = {
        "target_chembl_id": TARGET_ID,
        "standard_type": "IC50",
        "standard_units": "nM",
        "standard_relation": "=",
        "limit": PAGE,
        "offset": offset,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"[collect] Problem at offset {offset} (try {attempt}/{retries}): {e}")
            time.sleep(3 * attempt)
    raise RuntimeError("ChEMBL download failed. Check your internet and try again.")


def fetch_egfr_bioactivity() -> pd.DataFrame:
    rows, offset = [], 0
    while True:
        data = get_page(offset)
        batch = data["activities"]
        rows.extend(batch)
        total = data["page_meta"]["total_count"]
        print(f"[collect] Downloaded {len(rows)} / {total}")
        if not data["page_meta"]["next"] or not batch:
            break
        offset += PAGE
    df = pd.DataFrame(rows)
    keep = ["molecule_chembl_id", "canonical_smiles", "standard_type",
            "standard_relation", "standard_value", "standard_units",
            "assay_chembl_id", "assay_type"]
    return df[[c for c in keep if c in df.columns]]


def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df = fetch_egfr_bioactivity()
    df.to_csv(RAW_PATH, index=False)
    print(f"[collect] Saved {len(df)} records -> {RAW_PATH}")


if __name__ == "__main__":
    main()
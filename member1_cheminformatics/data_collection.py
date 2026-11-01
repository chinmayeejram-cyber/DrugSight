"""
Step 1: Collect EGFR bioactivity data from ChEMBL.
Target: EGFR (Epidermal Growth Factor Receptor) -> CHEMBL203
Output: data/raw/egfr_raw.csv
"""
import os
import pandas as pd

TARGET_ID = "CHEMBL203"  # Human EGFR
RAW_PATH = os.path.join("data", "raw", "egfr_raw.csv")


def fetch_egfr_bioactivity() -> pd.DataFrame:
    # Imported here so the ChEMBL connection is only made when downloading
    from chembl_webresource_client.new_client import new_client

    activities = new_client.activity
    results = activities.filter(
        target_chembl_id=TARGET_ID,
        standard_type="IC50",
        standard_units="nM",
        standard_relation="=",
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "assay_type",
    )
    df = pd.DataFrame.from_records(list(results))
    print(f"[collect] Downloaded {len(df)} bioactivity records")
    return df


def main():
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df = fetch_egfr_bioactivity()
    df.to_csv(RAW_PATH, index=False)
    print(f"[collect] Saved -> {RAW_PATH}")


if __name__ == "__main__":
    main()
"""
Step 4: Prepare the final ML-ready dataset (hand this to Member 2).
Input : data/processed/egfr_descriptors.csv, egfr_fingerprints.csv
Output: data/final/ -> X_train.csv, X_test.csv, y_train.csv, y_test.csv, egfr_ml_dataset.csv
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

DESC_PATH = os.path.join("data", "processed", "egfr_descriptors.csv")
FP_PATH = os.path.join("data", "processed", "egfr_fingerprints.csv")
FINAL_DIR = os.path.join("data", "final")

DESC_COLS = ["MolWt", "LogP", "HBD", "HBA", "TPSA", "RotatableBonds",
             "NumRings", "AromaticRings", "HeavyAtoms", "FractionCSP3"]


def main():
    desc = pd.read_csv(DESC_PATH)
    fp = pd.read_csv(FP_PATH)
    assert (desc["molecule_chembl_id"].values == fp["molecule_chembl_id"].values).all()

    fp_cols = [c for c in fp.columns if c.startswith(("morgan_", "maccs_"))]
    X = pd.concat([desc[DESC_COLS], fp[fp_cols]], axis=1)
    y = desc["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(FINAL_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(FINAL_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(FINAL_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(FINAL_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(FINAL_DIR, "y_test.csv"), index=False)

    # Full dataset in one file (includes pIC50 for regression models)
    full = pd.concat([desc[["molecule_chembl_id", "smiles", "pIC50", "label", "Lipinski_Pass"]], X], axis=1)
    full.to_csv(os.path.join(FINAL_DIR, "egfr_ml_dataset.csv"), index=False)

    print(f"[dataset] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[dataset] Class balance (train): {y_train.value_counts().to_dict()}")
    print(f"[dataset] Saved to {FINAL_DIR}/")


if __name__ == "__main__":
    main()

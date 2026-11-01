"""
Step 2: Clean the raw ChEMBL data and process SMILES with RDKit.
Input : data/raw/egfr_raw.csv
Output: data/processed/egfr_clean.csv
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

RAW_PATH = os.path.join("data", "raw", "egfr_raw.csv")
CLEAN_PATH = os.path.join("data", "processed", "egfr_clean.csv")

ACTIVE_NM = 1000      # IC50 <= 1 uM  -> active
INACTIVE_NM = 10000   # IC50 >= 10 uM -> inactive (in between is dropped)

_chooser = rdMolStandardize.LargestFragmentChooser()
_uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(smiles):
    """SMILES -> canonical SMILES (largest fragment, neutralised). None if invalid."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _chooser.choose(mol)      # remove salts / counter-ions
        mol = _uncharger.uncharge(mol)  # neutralise charges
        return Chem.MolToSmiles(mol)    # canonical SMILES
    except Exception:
        return None


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["canonical_smiles", "standard_value"]).copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]
    print(f"[clean] {n0} -> {len(df)} after removing missing / invalid values")

    # Standardise SMILES with RDKit
    df["smiles"] = df["canonical_smiles"].apply(standardize_smiles)
    df = df.dropna(subset=["smiles"])
    print(f"[clean] {len(df)} records with valid SMILES")

    # One row per unique molecule (median IC50 of duplicates)
    df = (
        df.groupby("smiles", as_index=False)
        .agg(molecule_chembl_id=("molecule_chembl_id", "first"),
             ic50_nM=("standard_value", "median"))
    )
    print(f"[clean] {len(df)} unique compounds after removing duplicates")

    # pIC50 = -log10(IC50 in M) = 9 - log10(IC50 in nM)
    df["pIC50"] = 9 - np.log10(df["ic50_nM"])

    # Active / inactive labels (intermediate compounds removed)
    df = df[(df["ic50_nM"] <= ACTIVE_NM) | (df["ic50_nM"] >= INACTIVE_NM)].copy()
    df["label"] = (df["ic50_nM"] <= ACTIVE_NM).astype(int)  # 1 = active, 0 = inactive
    print(f"[clean] {len(df)} compounds kept | active={df['label'].sum()} "
          f"inactive={(df['label'] == 0).sum()}")
    return df.reset_index(drop=True)


def main():
    df = pd.read_csv(RAW_PATH)
    df = clean_dataset(df)
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"[clean] Saved -> {CLEAN_PATH}")


if __name__ == "__main__":
    main()
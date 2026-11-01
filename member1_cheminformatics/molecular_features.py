"""
Step 3: Molecular descriptors, fingerprints and Lipinski's Rule of Five.
Input : data/processed/egfr_clean.csv
Output: data/processed/egfr_descriptors.csv
        data/processed/egfr_fingerprints.csv
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys

RDLogger.DisableLog("rdApp.*")

CLEAN_PATH = os.path.join("data", "processed", "egfr_clean.csv")
DESC_PATH = os.path.join("data", "processed", "egfr_descriptors.csv")
FP_PATH = os.path.join("data", "processed", "egfr_fingerprints.csv")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)


def calc_descriptors(mol) -> dict:
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "RotatableBonds": Lipinski.NumRotatableBonds(mol),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
        "AromaticRings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
    }


def lipinski_rule_of_five(desc: dict) -> dict:
    """Lipinski: MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10.
    A compound passes if it has at most 1 violation."""
    violations = (
        int(desc["MolWt"] > 500)
        + int(desc["LogP"] > 5)
        + int(desc["HBD"] > 5)
        + int(desc["HBA"] > 10)
    )
    return {"Lipinski_Violations": violations, "Lipinski_Pass": int(violations <= 1)}


def morgan_fp(mol) -> np.ndarray:
    return np.array(_morgan_gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def maccs_fp(mol) -> np.ndarray:
    return np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.uint8)


def featurize(df: pd.DataFrame):
    desc_rows, morgan_rows, maccs_rows, keep = [], [], [], []
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        d = calc_descriptors(mol)
        d.update(lipinski_rule_of_five(d))
        desc_rows.append(d)
        morgan_rows.append(morgan_fp(mol))
        maccs_rows.append(maccs_fp(mol))
        keep.append(i)

    base = df.iloc[keep].reset_index(drop=True)
    desc_df = pd.concat([base, pd.DataFrame(desc_rows)], axis=1)

    ids = base[["molecule_chembl_id", "smiles", "pIC50", "label"]]
    morgan_df = pd.DataFrame(morgan_rows, columns=[f"morgan_{i}" for i in range(MORGAN_BITS)])
    maccs_df = pd.DataFrame(maccs_rows, columns=[f"maccs_{i}" for i in range(167)])
    fp_df = pd.concat([ids, morgan_df, maccs_df], axis=1)
    return desc_df, fp_df


def main():
    df = pd.read_csv(CLEAN_PATH)
    desc_df, fp_df = featurize(df)
    desc_df.to_csv(DESC_PATH, index=False)
    fp_df.to_csv(FP_PATH, index=False)
    print(f"[features] Descriptors -> {DESC_PATH} {desc_df.shape}")
    print(f"[features] Fingerprints -> {FP_PATH} {fp_df.shape}")
    print(f"[features] Lipinski pass: {desc_df['Lipinski_Pass'].sum()}/{len(desc_df)}")


if __name__ == "__main__":
    main()
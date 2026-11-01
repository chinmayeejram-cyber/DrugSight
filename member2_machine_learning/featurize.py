"""Shared featurization + data loading used by every model, the explainer and the web app.

Contract with member1 (cheminformatics):
    data/processed/dataset.csv   with columns:  smiles, label     (label: 1 = active, 0 = inactive)
    (optional extra columns: molecule_chembl_id, pIC50 - if `label` is missing it is derived from pIC50)
If dataset.csv does not exist yet, data/processed/demo_dataset.csv (toy data) is used so you can test the code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS_DIR = RESULTS / "models"
FIG_DIR = RESULTS / "figures"
PRED_DIR = RESULTS / "predictions"
for _d in (MODELS_DIR, FIG_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FP_BITS = 2048
FP_RADIUS = 2
MAX_ATOMS = 70
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)


# ----------------------------------------------------------------- molecules
def default_csv() -> Path:
    p = DATA_PROCESSED / "dataset.csv"
    return p if p.exists() else DATA_PROCESSED / "demo_dataset.csv"


def standardize(mol):
    """Keep the largest fragment (strips salts / counter-ions)."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    return max(frags, key=lambda m: m.GetNumHeavyAtoms()) if len(frags) > 1 else mol


def mol_from_smiles(smiles):
    """SMILES -> cleaned RDKit molecule, or None if invalid / empty."""
    if smiles is None or not str(smiles).strip():
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    return standardize(mol) if mol is not None and mol.GetNumAtoms() > 0 else None


# ----------------------------------------------------------------- fingerprints
def morgan_fp(mol) -> np.ndarray:
    return _GEN.GetFingerprintAsNumPy(mol).astype(np.float32)


def morgan_fp_with_bitinfo(mol):
    """Fingerprint + {bit: [(centre_atom, radius), ...]} so a bit can be mapped back to atoms."""
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = _GEN.GetFingerprintAsNumPy(mol, additionalOutput=ao).astype(np.float32)
    return fp, ao.GetBitInfoMap()


# ----------------------------------------------------------------- graphs (for the GCN)
def atom_features(atom) -> list:
    sym = atom.GetSymbol()
    return ([1.0 if sym == e else 0.0 for e in ELEMENTS] + [0.0 if sym in ELEMENTS else 1.0]
            + [1.0 if atom.GetDegree() == d else 0.0 for d in range(5)]
            + [1.0 if atom.GetTotalNumHs() == h else 0.0 for h in range(4)]
            + [float(atom.GetIsAromatic()), float(atom.IsInRing()), float(atom.GetFormalCharge())])


N_ATOM_FEATS = len(ELEMENTS) + 1 + 5 + 4 + 3


def mol_to_graph(mol, max_atoms: int = MAX_ATOMS):
    """(X, A_hat): padded node-feature matrix and symmetric-normalised adjacency (with self loops)."""
    n = mol.GetNumAtoms()
    if n == 0 or n > max_atoms:
        return None
    X = np.zeros((max_atoms, N_ATOM_FEATS), dtype=np.float32)
    A = np.zeros((max_atoms, max_atoms), dtype=np.float32)
    for atom in mol.GetAtoms():
        X[atom.GetIdx()] = atom_features(atom)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        A[i, j] = A[j, i] = 1.0
    A[:n, :n] += np.eye(n, dtype=np.float32)
    d = A.sum(1)
    inv_sqrt = np.zeros_like(d)
    nz = d > 0
    inv_sqrt[nz] = d[nz] ** -0.5
    return X, inv_sqrt[:, None] * A * inv_sqrt[None, :]


# ----------------------------------------------------------------- dataset + split
def load_dataset(csv=None, graphs: bool = True) -> dict:
    """Load, clean and featurize the dataset. Every model calls this so they all see identical molecules."""
    csv = Path(csv) if csv else default_csv()
    df = pd.read_csv(csv)
    df = df.rename(columns={"canonical_smiles": "smiles", "SMILES": "smiles", "Smiles": "smiles"})
    if "label" not in df.columns:
        if "pIC50" not in df.columns:
            raise ValueError("dataset needs a 'label' column (1 active / 0 inactive) or a 'pIC50' column")
        df = df[(df.pIC50 >= 6.0) | (df.pIC50 <= 5.0)].copy()
        df["label"] = (df.pIC50 >= 6.0).astype(int)
    df = df.dropna(subset=["smiles", "label"]).drop_duplicates(subset="smiles")

    smiles, ys, scaf, fps, gx, ga = [], [], [], [], [], []
    for smi, y in zip(df.smiles, df.label):
        mol = mol_from_smiles(smi)
        if mol is None or mol.GetNumAtoms() > MAX_ATOMS:
            continue
        smiles.append(Chem.MolToSmiles(mol))
        ys.append(int(y))
        scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        fps.append(morgan_fp(mol))
        if graphs:
            x, a = mol_to_graph(mol)
            gx.append(x); ga.append(a)
    print(f"[data] {len(smiles)} valid molecules from {csv.name} "
          f"({int(sum(ys))} active / {len(ys) - int(sum(ys))} inactive)")
    return {"smiles": smiles, "y": np.array(ys), "scaffold": scaf, "fp": np.stack(fps),
            "gx": np.stack(gx) if graphs else None, "ga": np.stack(ga) if graphs else None, "csv": str(csv)}


def scaffold_split(scaffolds, y, seed: int = 42):
    """80/10/10 split where molecules sharing a Bemis-Murcko scaffold stay together (tests 'new chemistry').
    Falls back to a stratified random split when the dataset has too few scaffolds. Deterministic."""
    from sklearn.model_selection import train_test_split
    n = len(y)
    groups = {}
    for i, s in enumerate(scaffolds):
        groups.setdefault(s, []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    tr, va, te = [], [], []
    for g in ordered:
        if len(tr) + len(g) <= 0.8 * n:
            tr += g
        elif len(va) + len(g) <= 0.1 * n:
            va += g
        else:
            te += g
    if len(va) >= 5 and len(te) >= 10 and len(set(y[te])) > 1 and len(set(y[va])) > 1:
        return np.array(tr), np.array(va), np.array(te), "scaffold"
    idx = np.arange(n)
    tr, rest = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.5, stratify=y[rest], random_state=seed)
    return tr, va, te, "random (too few scaffolds)"
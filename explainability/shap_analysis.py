"""Explainable AI for DrugSight: WHICH part of the molecule drives the prediction?

* explain_molecule()   - per-molecule explanation (SHAP on the tree model, or occlusion fallback),
                         mapped back to atoms so the structure can be coloured green (pushes towards
                         ACTIVE) / red (pushes towards INACTIVE).
* global_importance()  - which substructures matter most across the whole dataset.

Run the global analysis:  python -m explainability.shap_analysis
"""
from __future__ import annotations

import argparse
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

from member2_machine_learning.featurize import (FIG_DIR, MODELS_DIR, RESULTS, load_dataset, mol_from_smiles,
                                                morgan_fp_with_bitinfo, scaffold_split)

_EXPLAINERS = {}


# ---------------------------------------------------------------- helpers
def _environment(mol, centre: int, radius: int):
    """Atoms + bonds covered by one Morgan bit occurrence."""
    if radius == 0:
        return [centre], []
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, radius, centre))
    atoms = set()
    for b in bonds:
        bd = mol.GetBondWithIdx(b)
        atoms.update([bd.GetBeginAtomIdx(), bd.GetEndAtomIdx()])
    return sorted(atoms), bonds


def _fragment_smiles(mol, atoms, bonds, centre) -> str:
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=list(atoms), bondsToUse=list(bonds), rootedAtAtom=centre)
    except Exception:
        return ""


def _shap_values(model, X: np.ndarray):
    """SHAP values for the 'active' class, shape (n, n_features), plus the base value."""
    import shap
    key = id(model)
    if key not in _EXPLAINERS:
        _EXPLAINERS[key] = shap.TreeExplainer(model)
    ex = _EXPLAINERS[key]
    sv = ex.shap_values(X)
    if isinstance(sv, list):           # older SHAP + sklearn: [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                   # newer SHAP + sklearn: (n, features, classes)
        sv = sv[:, :, -1]
    base = float(np.ravel(ex.expected_value)[-1])
    return sv, base


def _occlusion_values(model, x: np.ndarray):
    """Fallback explanation without SHAP: switch each active bit off and measure the drop in P(active)."""
    base = float(model.predict_proba(x[None, :])[0, 1])
    on = np.where(x > 0)[0]
    batch = np.repeat(x[None, :], len(on), axis=0)
    batch[np.arange(len(on)), on] = 0
    drops = base - model.predict_proba(batch)[:, 1]
    vals = np.zeros_like(x)
    vals[on] = drops
    return vals, base


# ---------------------------------------------------------------- per-molecule explanation
def explain_molecule(mol, model, topk: int = 6, method: str = "shap") -> dict:
    """Return {'method', 'base_value', 'contributions': [...], 'atom_scores': {atom: score}}."""
    fp, info = morgan_fp_with_bitinfo(mol)
    used = method
    try:
        if method != "shap":
            raise ImportError
        vals, base = _shap_values(model, fp[None, :])
        vals = vals[0]
    except Exception:
        vals, base = _occlusion_values(model, fp)
        used = "occlusion"

    on_bits = [b for b in np.where(fp > 0)[0] if abs(vals[b]) > 1e-9]
    contributions, atom_scores = [], {}
    for b in sorted(on_bits, key=lambda b: -abs(vals[b])):
        occ = info.get(int(b), [])
        for centre, radius in occ:
            atoms, bonds = _environment(mol, centre, radius)
            share = float(vals[b]) / len(occ) / max(len(atoms), 1)
            for a in atoms:
                atom_scores[a] = atom_scores.get(a, 0.0) + share
        if len(contributions) < topk and occ:
            centre, radius = occ[0]
            atoms, bonds = _environment(mol, centre, radius)
            contributions.append({"bit": int(b), "value": float(vals[b]), "radius": int(radius),
                                  "fragment": _fragment_smiles(mol, atoms, bonds, centre) or "?",
                                  "n_occurrences": len(occ), "atoms": atoms})
    return {"method": used, "base_value": base, "contributions": contributions, "atom_scores": atom_scores}


def draw_explanation(mol, atom_scores: dict, size=(520, 400)) -> bytes:
    """PNG bytes: atoms coloured green (towards active) / red (towards inactive), stronger = bigger effect."""
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().addAtomIndices = False
    top = max((abs(v) for v in atom_scores.values()), default=0.0)
    atoms, colours, radii = [], {}, {}
    if top > 0:
        for a, s in atom_scores.items():
            t = float(abs(s) / top)
            if t < 0.15:
                continue
            base = (0.13, 0.70, 0.30) if s > 0 else (0.90, 0.22, 0.22)
            colours[int(a)] = tuple(float(1 - min(1.0, t) * (1 - c)) for c in base)
            radii[int(a)] = float(0.35 + 0.2 * t)
            atoms.append(int(a))
    drawer.DrawMolecule(mol, highlightAtoms=atoms, highlightAtomColors=colours, highlightAtomRadii=radii,
                        highlightBonds=[])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ---------------------------------------------------------------- global importance
def global_importance(model, X: np.ndarray, smiles, topn: int = 12, sample: int = 300, seed: int = 0):
    """Mean |SHAP| per fingerprint bit + an example substructure for each of the top bits."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample, len(X)), replace=False)
    sv, _ = _shap_values(model, X[idx])
    imp = np.abs(sv).mean(0)
    top = np.argsort(-imp)[:topn]
    rows, frag_mols, frag_atoms, legends = [], [], [], []
    for b in top:
        present = X[idx][:, b] > 0
        mean_when_present = float(sv[present, b].mean()) if present.any() else 0.0
        example, frag = None, ""
        for j in idx[present]:
            mol = mol_from_smiles(smiles[j])
            _, info = morgan_fp_with_bitinfo(mol)
            if int(b) in info:
                centre, radius = info[int(b)][0]
                atoms, bonds = _environment(mol, centre, radius)
                frag = _fragment_smiles(mol, atoms, bonds, centre)
                example = (mol, atoms)
                break
        rows.append({"bit": int(b), "mean_abs_shap": float(imp[b]), "mean_shap_when_present": mean_when_present,
                     "direction": "-> ACTIVE" if mean_when_present > 0 else "-> INACTIVE",
                     "example_fragment": frag, "molecules_with_bit": int(present.sum())})
        if example:
            frag_mols.append(example[0]); frag_atoms.append(example[1])
            legends.append(f"bit {b} {'+' if mean_when_present > 0 else '-'}")
    return pd.DataFrame(rows), frag_mols, frag_atoms, legends


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--sample", type=int, default=300)
    args = ap.parse_args(argv)

    path = next((MODELS_DIR / f for f in ("xgboost.joblib", "random_forest.joblib") if (MODELS_DIR / f).exists()), None)
    if path is None:
        raise SystemExit("Train xgboost_model or random_forest first.")
    model = joblib.load(path)
    data = load_dataset(args.csv, graphs=False)
    _, _, te, _ = scaffold_split(data["scaffold"], data["y"])
    X, smiles = data["fp"][te], [data["smiles"][i] for i in te]
    df, mols, atoms, legends = global_importance(model, X, smiles, sample=args.sample)
    df.to_csv(RESULTS / "shap_global.csv", index=False)

    labels = [f"bit {r.bit}: {r.example_fragment[:22]}" for r in df.itertuples()]
    colours = ["#2e9e4f" if d > 0 else "#d93a3a" for d in df.mean_shap_when_present]
    plt.figure(figsize=(7, 4.5))
    plt.barh(labels[::-1], df.mean_abs_shap[::-1], color=colours[::-1])
    plt.xlabel("mean |SHAP value|  (green = pushes towards ACTIVE, red = towards INACTIVE)")
    plt.title(f"Most important substructures ({path.stem})")
    plt.tight_layout(); plt.savefig(FIG_DIR / "shap_global.png", dpi=150); plt.close()
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 220), legends=legends,
                                   highlightAtomLists=atoms)
        img.save(FIG_DIR / "shap_substructures.png")
    print(df[["bit", "mean_abs_shap", "direction", "example_fragment"]].to_string(index=False))
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
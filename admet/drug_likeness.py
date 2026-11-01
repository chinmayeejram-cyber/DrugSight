"""Rule-based drug-likeness and ADMET screening (RDKit only - no training needed).

IMPORTANT for the presentation: these are well-known *rules of thumb and computed estimates*
(Lipinski, Veber, Egan, QED, ESOL solubility, PAINS/Brenk structural alerts), not lab measurements
or a trained ADMET model. Say so on the slide - judges respect that honesty.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


@lru_cache(maxsize=1)
def _catalogs():
    pains = FilterCatalogParams(); pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    brenk = FilterCatalogParams(); brenk.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(pains), FilterCatalog(brenk)


def physchem(mol) -> dict:
    heavy = mol.GetNumHeavyAtoms()
    arom = sum(a.GetIsAromatic() for a in mol.GetAtoms())
    return {"MolWt": round(Descriptors.MolWt(mol), 1),
            "LogP": round(Crippen.MolLogP(mol), 2),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 1),
            "H-bond donors": Lipinski.NumHDonors(mol),
            "H-bond acceptors": Lipinski.NumHAcceptors(mol),
            "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "Heavy atoms": heavy,
            "Aromatic proportion": round(arom / heavy, 2) if heavy else 0.0}


def rule_checks(p: dict) -> dict:
    """Each rule -> (passed?, human readable detail)."""
    lip_v = sum([p["MolWt"] > 500, p["LogP"] > 5, p["H-bond donors"] > 5, p["H-bond acceptors"] > 10])
    return {
        "Lipinski rule of 5": (lip_v <= 1, f"{lip_v} violation(s); MW<=500, LogP<=5, HBD<=5, HBA<=10"),
        "Veber (oral bioavailability)": (p["Rotatable bonds"] <= 10 and p["TPSA"] <= 140,
                                          "rotatable bonds <=10 and TPSA <=140"),
        "Egan (absorption)": (p["TPSA"] <= 131.6 and p["LogP"] <= 5.88, "TPSA <=131.6 and LogP <=5.88"),
        "Ghose (drug-like window)": (160 <= p["MolWt"] <= 480 and -0.4 <= p["LogP"] <= 5.6,
                                      "MW 160-480 and LogP -0.4 to 5.6"),
    }


def esol_logS(p: dict) -> float:
    """Delaney ESOL estimate of aqueous solubility (log mol/L)."""
    return round(0.16 - 0.63 * p["LogP"] - 0.0062 * p["MolWt"] + 0.066 * p["Rotatable bonds"]
                 - 0.74 * p["Aromatic proportion"], 2)


def solubility_class(logS: float) -> str:
    return "High" if logS > -4 else ("Moderate" if logS > -6 else "Low")


def structural_alerts(mol) -> dict:
    pains, brenk = _catalogs()
    return {"PAINS": sorted({e.GetDescription() for e in pains.GetMatches(mol)}),
            "Brenk (toxicophores / unstable groups)": sorted({e.GetDescription() for e in brenk.GetMatches(mol)})}


def admet_profile(mol) -> dict:
    """Full profile for one RDKit molecule."""
    p = physchem(mol)
    rules = rule_checks(p)
    alerts = structural_alerts(mol)
    logS = esol_logS(p)
    qed = round(float(QED.qed(mol)), 3)
    est = {
        "Absorption (GI, heuristic)": "High" if rules["Egan (absorption)"][0] else "Low",
        "Blood-brain barrier (heuristic)": "Possible" if (p["TPSA"] <= 90 and p["MolWt"] <= 450
                                                         and p["H-bond donors"] <= 3) else "Unlikely",
        "Solubility (ESOL logS)": f"{logS}  ({solubility_class(logS)})",
        "Drug-likeness (QED, 0-1)": qed,
    }
    n_fail = sum(not ok for ok, _ in rules.values())
    n_pains = len(alerts["PAINS"])
    if n_fail == 0 and n_pains == 0:
        light = "green"
    elif n_fail <= 1 and n_pains == 0:
        light = "orange"
    else:
        light = "red"
    return {"properties": p, "rules": rules, "alerts": alerts, "estimates": est, "qed": qed,
            "traffic_light": light}


def summary_row(mol) -> dict:
    """Flat dict for tables (virtual screening)."""
    prof = admet_profile(mol)
    p = prof["properties"]
    return {"MolWt": p["MolWt"], "LogP": p["LogP"], "TPSA": p["TPSA"], "QED": prof["qed"],
            "Lipinski_pass": prof["rules"]["Lipinski rule of 5"][0],
            "Veber_pass": prof["rules"]["Veber (oral bioavailability)"][0],
            "PAINS_alerts": len(prof["alerts"]["PAINS"]),
            "ESOL_logS": esol_logS(p), "ADMET_flag": prof["traffic_light"]}


def screen_smiles(smiles_list) -> pd.DataFrame:
    rows = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        rows.append(summary_row(m) if m is not None else {})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import json
    m = Chem.MolFromSmiles("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")  # gefitinib
    print(json.dumps(admet_profile(m), indent=2, default=str))
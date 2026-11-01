"""Input widgets: example picker, SMILES box, optional drawing editor, library upload."""
from __future__ import annotations

import pandas as pd
import streamlit as st

try:  # optional: pip install streamlit-ketcher
    from streamlit_ketcher import st_ketcher
except Exception:
    st_ketcher = None

EXAMPLES = {
    "Gefitinib - approved EGFR drug": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "Erlotinib - approved EGFR drug": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
    "Lapatinib - approved EGFR/HER2 drug": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "Aspirin - not a kinase inhibitor": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine - not a kinase inhibitor": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol - not a kinase inhibitor": "CC(=O)Nc1ccc(O)cc1",
}


def smiles_input(key: str = "main") -> str:
    """Return the SMILES the user typed / picked / drew (empty string if none)."""
    choice = st.selectbox("Try an example molecule", ["(enter my own)"] + list(EXAMPLES), key=f"{key}_ex")
    default = EXAMPLES.get(choice, "")
    smiles = st.text_input("SMILES string", value=default, key=f"{key}_smiles_{choice}",
                           placeholder="e.g. CC(=O)Oc1ccccc1C(=O)O")
    if st_ketcher is not None and st.checkbox("✏️ Draw the molecule instead", key=f"{key}_draw"):
        smiles = st_ketcher(smiles, key=f"{key}_ketcher") or smiles
    return (smiles or "").strip()


def library_upload(key: str = "lib"):
    """CSV with a 'smiles' column -> DataFrame (or None)."""
    up = st.file_uploader("Compound library (CSV with a column named `smiles`)", type="csv", key=key)
    if up is None:
        return None
    df = pd.read_csv(up)
    df.columns = [c.strip() for c in df.columns]
    col = next((c for c in df.columns if c.lower() in ("smiles", "canonical_smiles")), None)
    if col is None:
        st.error("The CSV needs a column named `smiles`.")
        return None
    return df.rename(columns={col: "smiles"})

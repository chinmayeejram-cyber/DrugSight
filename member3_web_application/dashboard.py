"""Dashboard pages: dataset overview, model comparison, explainability gallery."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from rdkit.Chem import Crippen, Descriptors

from member2_machine_learning.featurize import FIG_DIR, RESULTS, default_csv, mol_from_smiles


@st.cache_data(show_spinner=False)
def _dataset_stats(csv: str, max_mols: int = 3000):
    df = pd.read_csv(csv).rename(columns={"canonical_smiles": "smiles"})
    if "label" not in df.columns:
        return None
    df = df.dropna(subset=["smiles"]).sample(min(len(df), max_mols), random_state=0)
    rows = []
    for s, y in zip(df.smiles, df.label):
        m = mol_from_smiles(s)
        if m is not None:
            rows.append({"label": "active" if y == 1 else "inactive",
                         "MolWt": Descriptors.MolWt(m), "LogP": Crippen.MolLogP(m)})
    return pd.DataFrame(rows), int(len(pd.read_csv(csv)))


def render_overview():
    csv = default_csv()
    st.caption(f"Dataset: `{csv.name}`")
    res = _dataset_stats(str(csv))
    if res is None:
        st.info("Dataset has no `label` column.")
        return
    stats, total = res
    c1, c2, c3 = st.columns(3)
    c1.metric("Molecules in dataset", f"{total:,}")
    c2.metric("Active", f"{(stats.label == 'active').mean():.0%}")
    c3.metric("Inactive", f"{(stats.label == 'inactive').mean():.0%}")
    a, b = st.columns(2)
    a.write("**Molecular weight distribution**")
    a.bar_chart(pd.cut(stats.MolWt, bins=12).value_counts().sort_index().rename(lambda i: f"{int(i.left)}-{int(i.right)}"))
    b.write("**LogP (lipophilicity) distribution**")
    b.bar_chart(pd.cut(stats.LogP, bins=12).value_counts().sort_index().rename(lambda i: f"{i.left:.1f}..{i.right:.1f}"))


def render_models():
    f = RESULTS / "metrics.json"
    if not f.exists():
        st.info("No results yet. Run `python -m member2_machine_learning.train_all` first.")
        return
    m = json.load(open(f))
    st.write(f"Evaluated on **{m['n_test']} held-out molecules** (split: *{m['split_method']}*, "
             f"{m['test_active_fraction']:.0%} active).")
    table = pd.DataFrame(m["results"]).T
    st.dataframe(table.style.highlight_max(axis=0, color="#cdeccd"), width="stretch")
    st.bar_chart(table[["ROC_AUC", "PR_AUC", "F1"]])
    if (FIG_DIR / "roc_pr.png").exists():
        st.image(str(FIG_DIR / "roc_pr.png"))
    if (FIG_DIR / "confusion.png").exists():
        st.image(str(FIG_DIR / "confusion.png"))
    if m.get("gnn_history"):
        h = pd.DataFrame(m["gnn_history"]).set_index("epoch")
        st.write("**GCN training curve**")
        c1, c2 = st.columns(2)
        c1.line_chart(h[["train_loss"]]); c2.line_chart(h[["val_auc"]])


def render_explainability():
    st.write("Which molecular substructures matter most **across the whole dataset** (mean |SHAP value|):")
    shown = False
    for name, cap in (("shap_global.png", "Top substructures by importance"),
                      ("shap_substructures.png", "Example of each substructure (highlighted atoms)")):
        if (FIG_DIR / name).exists():
            st.image(str(FIG_DIR / name), caption=cap); shown = True
    if (RESULTS / "shap_global.csv").exists():
        st.dataframe(pd.read_csv(RESULTS / "shap_global.csv"), width="stretch")
    if not shown:
        st.info("Run `python -m explainability.shap_analysis` to generate these plots.")

"""Loads every trained model once and predicts P(active) for a molecule (used by the web app)."""
from __future__ import annotations

import joblib
import numpy as np

from member2_machine_learning.featurize import MODELS_DIR, morgan_fp, mol_to_graph

try:  # the app still works with only the tree models if torch is missing
    import torch
    from member2_machine_learning.gnn_model import GCN
except Exception:  # pragma: no cover
    torch = None
    GCN = None


class Predictor:
    def __init__(self):
        self.sk_models = {}   # name -> fitted sklearn-style model
        self.gcn = None
        for key, nice in (("random_forest", "Random Forest"), ("xgboost", "XGBoost")):
            f = MODELS_DIR / f"{key}.joblib"
            if f.exists():
                self.sk_models[nice] = joblib.load(f)
        f = MODELS_DIR / "gnn.pt"
        if torch is not None and f.exists():
            self.gcn = GCN()
            self.gcn.load_state_dict(torch.load(f, map_location="cpu"))
            self.gcn.eval()

    @property
    def names(self):
        return list(self.sk_models) + (["GCN (deep learning)"] if self.gcn is not None else [])

    @property
    def ready(self) -> bool:
        return bool(self.names)

    def explainer_model(self):
        """Tree model used for SHAP: XGBoost if available, else Random Forest."""
        return self.sk_models.get("XGBoost") or self.sk_models.get("Random Forest")

    def predict(self, mol) -> dict:
        out = {}
        if self.sk_models:
            x = morgan_fp(mol)[None, :]
            for name, m in self.sk_models.items():
                out[name] = float(m.predict_proba(x)[0, 1])
        if self.gcn is not None:
            g = mol_to_graph(mol)
            if g is not None:
                with torch.no_grad():
                    logit = self.gcn(torch.as_tensor(g[0])[None], torch.as_tensor(g[1])[None])
                out["GCN (deep learning)"] = float(torch.sigmoid(logit))
        if len(out) > 1:
            out["Ensemble"] = float(np.mean(list(out.values())))
        return out

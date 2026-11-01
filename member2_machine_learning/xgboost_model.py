"""XGBoost on Morgan fingerprints (usually the strongest classical model).
Run: python -m member2_machine_learning.xgboost_model"""
import argparse

import joblib
from xgboost import XGBClassifier

from member2_machine_learning.featurize import MODELS_DIR, load_dataset, scaffold_split
from member2_machine_learning.model_evaluation import compute_metrics, save_predictions


def build_model(scale_pos_weight: float = 1.0, seed: int = 42) -> XGBClassifier:
    return XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.4, min_child_weight=1, scale_pos_weight=scale_pos_weight,
                         eval_metric="auc", early_stopping_rounds=40, tree_method="hist",
                         n_jobs=-1, random_state=seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data = load_dataset(args.csv, graphs=False)
    X, y = data["fp"], data["y"]
    tr, va, te, method = scaffold_split(data["scaffold"], y, args.seed)
    print(f"[xgb] split={method}  train={len(tr)} val={len(va)} test={len(te)}")
    spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))
    model = build_model(spw, args.seed)
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
    p = model.predict_proba(X[te])[:, 1]
    print("[xgb] test metrics:", compute_metrics(y[te], p))
    joblib.dump(model, MODELS_DIR / "xgboost.joblib")
    save_predictions("xgboost", y[te], p, method, len(tr))


if __name__ == "__main__":
    main()

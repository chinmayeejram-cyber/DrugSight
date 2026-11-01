"""Random Forest baseline on Morgan fingerprints.   Run: python -m member2_machine_learning.random_forest"""
import argparse

import joblib
from sklearn.ensemble import RandomForestClassifier

from member2_machine_learning.featurize import MODELS_DIR, load_dataset, scaffold_split
from member2_machine_learning.model_evaluation import compute_metrics, save_predictions


def build_model(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, n_jobs=-1, class_weight="balanced_subsample",
                                  random_state=seed)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="default: data/processed/dataset.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    data = load_dataset(args.csv, graphs=False)
    tr, va, te, method = scaffold_split(data["scaffold"], data["y"], args.seed)
    print(f"[rf] split={method}  train={len(tr)} val={len(va)} test={len(te)}")
    model = build_model(args.seed).fit(data["fp"][tr], data["y"][tr])
    p = model.predict_proba(data["fp"][te])[:, 1]
    print("[rf] test metrics:", compute_metrics(data["y"][te], p))
    joblib.dump(model, MODELS_DIR / "random_forest.joblib")
    save_predictions("random_forest", data["y"][te], p, method, len(tr))


if __name__ == "__main__":
    main()
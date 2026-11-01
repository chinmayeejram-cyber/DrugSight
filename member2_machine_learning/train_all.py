"""One command to train and evaluate everything.
Run:  python -m member2_machine_learning.train_all [--csv data/processed/dataset.csv] [--epochs 30] [--skip-gnn]
"""
import argparse

from member2_machine_learning import gnn_model, model_evaluation, random_forest, xgboost_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-gnn", action="store_true")
    a = ap.parse_args()
    base = ["--csv", a.csv] if a.csv else []
    random_forest.main(base)
    xgboost_model.main(base)
    if not a.skip_gnn:
        gnn_model.main(base + ["--epochs", str(a.epochs)])
    model_evaluation.main()
    try:
        from explainability import shap_analysis
        shap_analysis.main(base)
    except Exception as e:  # explanation plots are nice-to-have
        print("[shap] skipped:", e)


if __name__ == "__main__":
    main()
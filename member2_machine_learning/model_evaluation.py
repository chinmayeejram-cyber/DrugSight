"""Compare all trained models on the SAME held-out test set.

Each model script saves its test-set probabilities with save_predictions(); this module reads them,
adds a simple average Ensemble, computes metrics and writes plots + results/metrics.json.

Run:  python -m member2_machine_learning.model_evaluation
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score,
                             roc_curve, precision_recall_curve)

from member2_machine_learning.featurize import FIG_DIR, PRED_DIR, RESULTS

DISPLAY = {"random_forest": "Random Forest", "xgboost": "XGBoost", "gnn": "GCN (deep learning)"}


def save_predictions(name: str, y_true, y_prob, split_method: str, n_train: int):
    np.savez(PRED_DIR / f"{name}.npz", y_true=np.asarray(y_true), y_prob=np.asarray(y_prob),
             split_method=split_method, n_train=n_train)


def compute_metrics(y, p, thr: float = 0.5) -> dict:
    pred = (p >= thr).astype(int)
    return {"ROC_AUC": round(float(roc_auc_score(y, p)), 3),
            "PR_AUC": round(float(average_precision_score(y, p)), 3),
            "Accuracy": round(float(accuracy_score(y, pred)), 3),
            "Precision": round(float(precision_score(y, pred, zero_division=0)), 3),
            "Recall": round(float(recall_score(y, pred, zero_division=0)), 3),
            "F1": round(float(f1_score(y, pred, zero_division=0)), 3),
            "MCC": round(float(matthews_corrcoef(y, pred)), 3)}


def main(argv=None):
    runs = {}
    for key, nice in DISPLAY.items():
        f = PRED_DIR / f"{key}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True)
            runs[nice] = (d["y_true"], d["y_prob"], str(d["split_method"]), int(d["n_train"]))
    if not runs:
        raise SystemExit("No predictions found - train at least one model first.")
    y = next(iter(runs.values()))[0]
    for name, (yt, _, _, _) in runs.items():
        assert np.array_equal(yt, y), f"{name} was evaluated on a different test set"
    probs = {name: v[1] for name, v in runs.items()}
    if len(probs) > 1:
        probs["Ensemble (average)"] = np.mean(list(probs.values()), axis=0)

    results = {name: compute_metrics(y, p) for name, p in probs.items()}
    first = next(iter(runs.values()))
    out = {"results": results, "n_test": int(len(y)), "n_train": first[3], "split_method": first[2],
           "test_active_fraction": round(float(np.mean(y)), 3)}
    old = RESULTS / "metrics.json"
    if old.exists():  # keep the GCN training curve if it exists
        prev = json.load(open(old))
        if "gnn_history" in prev:
            out["gnn_history"] = prev["gnn_history"]
    hist = RESULTS / "gnn_history.json"
    if hist.exists():
        out["gnn_history"] = json.load(open(hist))
    json.dump(out, open(old, "w"), indent=2)
    print(json.dumps(results, indent=2))

    # ---- ROC + PR curves
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, label=f"{name} ({results[name]['ROC_AUC']})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, label=f"{name} ({results[name]['PR_AUC']})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (held-out test set)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "roc_pr.png", dpi=150); plt.close()

    # ---- confusion matrices
    fig, axes = plt.subplots(1, len(probs), figsize=(3.6 * len(probs), 3.4))
    axes = np.atleast_1d(axes)
    for a, (name, p) in zip(axes, probs.items()):
        cm = confusion_matrix(y, (p >= 0.5).astype(int))
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, int(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a.set(title=name, xticks=[0, 1], yticks=[0, 1], xticklabels=["inactive", "active"],
              yticklabels=["inactive", "active"], xlabel="predicted", ylabel="true")
        a.title.set_fontsize(9)
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion.png", dpi=150); plt.close()
    print(f"Saved results/metrics.json and figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
"""Graph Convolutional Network (deep learning) in plain PyTorch - no PyTorch-Geometric needed.
Run: python -m member2_machine_learning.gnn_model --epochs 30"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from member2_machine_learning.featurize import (MODELS_DIR, N_ATOM_FEATS, RESULTS, load_dataset,
                                                scaffold_split)
from member2_machine_learning.model_evaluation import compute_metrics, save_predictions


class GCN(nn.Module):
    """3 graph-convolution layers -> mean pooling over real atoms -> MLP head (outputs a logit)."""

    def __init__(self, in_dim: int = N_ATOM_FEATS, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(in_dim, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, X, A):  # X: (B, N, F)   A: (B, N, N)
        mask = (X.abs().sum(-1, keepdim=True) > 0).float()  # 1 for real atoms, 0 for padding
        h = self.drop(torch.relu(torch.bmm(A, self.l1(X))))
        h = self.drop(torch.relu(torch.bmm(A, self.l2(h))))
        h = torch.relu(torch.bmm(A, self.l3(h))) * mask
        return self.head(h.sum(1) / mask.sum(1).clamp(min=1)).squeeze(-1)


def predict_proba(model, X, A, device="cpu", bs: int = 256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i + bs]).to(device)
            ab = torch.as_tensor(A[i:i + bs]).to(device)
            out.append(torch.sigmoid(model(xb, ab)).cpu().numpy())
    return np.concatenate(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_dataset(args.csv, graphs=True)
    y, GX, GA = data["y"], data["gx"], data["ga"]
    tr, va, te, method = scaffold_split(data["scaffold"], y, args.seed)
    print(f"[gnn] device={dev} split={method}  train={len(tr)} val={len(va)} test={len(te)}")

    model = GCN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    pos_w = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), dtype=torch.float32, device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ty = torch.tensor(y, dtype=torch.float32)

    best, history = -1.0, []
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(tr)
        total = 0.0
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(torch.as_tensor(GX[b]).to(dev), torch.as_tensor(GA[b]).to(dev)), ty[b].to(dev))
            loss.backward(); opt.step()
            total += loss.item() * len(b)
        val_auc = float(roc_auc_score(y[va], predict_proba(model, GX[va], GA[va], dev)))
        history.append({"epoch": ep, "train_loss": round(total / len(perm), 4), "val_auc": round(val_auc, 4)})
        print(f"[gnn] epoch {ep:02d}  loss {total / len(perm):.4f}  val AUC {val_auc:.3f}")
        if val_auc >= best:
            best = val_auc
            torch.save(model.state_dict(), MODELS_DIR / "gnn.pt")

    model.load_state_dict(torch.load(MODELS_DIR / "gnn.pt", map_location=dev))
    p = predict_proba(model, GX[te], GA[te], dev)
    print("[gnn] test metrics:", compute_metrics(y[te], p))
    json.dump(history, open(RESULTS / "gnn_history.json", "w"))
    save_predictions("gnn", y[te], p, method, len(tr))


if __name__ == "__main__":
    main()
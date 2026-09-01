"""Seed solution — the official FM baseline, expressed in the agent's contract.

This is the agent's starting point, not a target. It is a faithful port of
baseline.py (verified to reproduce valid primary 0.6015 exactly, see
fm_torch.py --compare), so any improvement the agent measures is a real
improvement over the official baseline rather than over a weakened copy.
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "kuairand-starter-kit"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "kuairand-starter-kit"))
from evaluate import evaluate


class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        out = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(out)


def run(D, seed=0):
    k, lr, l2, bs, epochs, patience = 16, 0.001, 1e-6, 8192, 40, 4

    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = torch.from_numpy(D["ytr"]).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]

    m = FM(D["dim"], k=k, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)          # matches baseline.py: bias is plain SGD
    lossf = nn.BCEWithLogitsLoss(reduction="mean")

    rng = np.random.default_rng(seed)
    best, best_state, bad, history = -1.0, None, 0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), bs):
            b_idx = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr[b_idx]), ytr[b_idx])
            loss.backward()
            opt.step(); opt_b.step()
            losses.append(loss.item())

        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        history.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                        "valid_primary": float(p), "secs": round(time.time() - t0, 1)})

        if p > best + 1e-5:
            best, bad = p, 0
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    m.load_state_dict(best_state)
    return m.predict(Xva), history

import sys, os, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "kuairand-starter-kit"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kuairand-starter-kit"))
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


def train_single_fm(D, model_seed, k=16, lr=0.001, l2=1e-6, bs=8192, epochs=40, patience=4):
    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = torch.from_numpy(D["ytr"]).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]

    m = FM(D["dim"], k=k, seed=model_seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)
    lossf = nn.BCEWithLogitsLoss(reduction="mean")

    rng = np.random.default_rng(model_seed)
    best, best_state, bad = -1.0, None, 0

    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            b_idx = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            loss = lossf(m(Xtr[b_idx]), ytr[b_idx])
            loss.backward()
            opt.step(); opt_b.step()

        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5:
            best, bad = p, 0
            best_state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    m.load_state_dict(best_state)
    return m.predict(Xva)


def run(D, seed=0):
    num_models = 5
    all_preds = []
    history = []

    t_start = time.time()
    for i in range(num_models):
        model_seed = seed + i * 100
        t0 = time.time()
        preds = train_single_fm(D, model_seed=model_seed)
        all_preds.append(preds)

        curr_avg_preds = np.mean(all_preds, axis=0)
        curr_primary = evaluate(D["uva"], D["yva"], curr_avg_preds)["primary"]
        history.append({
            "epoch": i + 1,
            "train_loss": 0.0,
            "valid_primary": float(curr_primary),
            "secs": round(time.time() - t0, 1)
        })

    final_preds = np.mean(all_preds, axis=0)
    return final_preds, history

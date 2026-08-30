import sys, os, time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def run(D, seed=0):
    k, lr, l2, bs, epochs, patience = 16, 0.001, 1e-6, 8192, 40, 4

    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = D["ytr"]
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]

    user_pos_dict = defaultdict(list)
    user_neg_dict = defaultdict(list)
    for idx, (u, y) in enumerate(zip(D["utr"], ytr)):
        if y > 0.5:
            user_pos_dict[u].append(idx)
        else:
            user_neg_dict[u].append(idx)

    valid_users = [u for u in user_pos_dict if len(user_pos_dict[u]) > 0 and len(user_neg_dict[u]) > 0]
    user_pos_arrs = [np.array(user_pos_dict[u], dtype=np.int32) for u in valid_users]
    user_neg_arrs = [np.array(user_neg_dict[u], dtype=np.int32) for u in valid_users]

    m = FM(D["dim"], k=k, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)

    rng = np.random.default_rng(seed)
    best, best_state, bad, history = -1.0, None, 0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()

        pos_indices, neg_indices = [], []
        for pos_arr, neg_arr in zip(user_pos_arrs, user_neg_arrs):
            n_pos, n_neg = len(pos_arr), len(neg_arr)
            k_samp = min(4, n_neg)
            pos_rep = np.repeat(pos_arr, k_samp)
            sampled_negs = rng.choice(neg_arr, size=(n_pos, k_samp), replace=True).ravel()
            pos_indices.append(pos_rep)
            neg_indices.append(sampled_negs)

        pos_idx_all = np.concatenate(pos_indices)
        neg_idx_all = np.concatenate(neg_indices)

        pair_perm = rng.permutation(len(pos_idx_all))
        pos_idx_all = pos_idx_all[pair_perm]
        neg_idx_all = neg_idx_all[pair_perm]

        losses = []
        for i in range(0, len(pos_idx_all), bs):
            b_pos = torch.from_numpy(pos_idx_all[i:i + bs]).long()
            b_neg = torch.from_numpy(neg_idx_all[i:i + bs]).long()

            opt.zero_grad(set_to_none=True)
            opt_b.zero_grad(set_to_none=True)

            s_pos = m(Xtr[b_pos])
            s_neg = m(Xtr[b_neg])

            loss = F.softplus(-(s_pos - s_neg)).mean()
            loss.backward()
            opt.step()
            opt_b.step()

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

import sys, os, time
import numpy as np
int_type = np.int64 if sys.platform == 'win32' else np.int64
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def sample_user_pairs(utr, ytr, rng, samples_per_pos=2):
    user_ids, u_idx = np.unique(utr, return_inverse=True)
    n_users = len(user_ids)
    pos_mask = (ytr == 1)

    user_pos = [[] for _ in range(n_users)]
    user_neg = [[] for _ in range(n_users)]

    for idx, (u, is_pos) in enumerate(zip(u_idx, pos_mask)):
        if is_pos:
            user_pos[u].append(idx)
        else:
            user_neg[u].append(idx)

    valid_users = [u for u in range(n_users) if len(user_pos[u]) > 0 and len(user_neg[u]) > 0]

    pos_list, neg_list = [], []
    for u in valid_users:
        p_arr = np.array(user_pos[u], dtype=np.int64)
        n_arr = np.array(user_neg[u], dtype=np.int64)
        p_rep = np.repeat(p_arr, samples_per_pos)
        n_samp = rng.choice(n_arr, size=len(p_rep), replace=True)
        pos_list.append(p_rep)
        neg_list.append(n_samp)

    return np.concatenate(pos_list), np.concatenate(neg_list)


def run(D, seed=0):
    k, lr, l2, bs, epochs, patience = 16, 0.001, 1e-6, 8192, 40, 5

    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = D["ytr"]
    utr = D["utr"]
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]

    m = FM(D["dim"], k=k, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)

    rng = np.random.default_rng(seed)
    best, best_state, bad, history = -1.0, None, 0, []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        pos_idx, neg_idx = sample_user_pairs(utr, ytr, rng, samples_per_pos=2)
        perm = rng.permutation(len(pos_idx))
        pos_idx, neg_idx = pos_idx[perm], neg_idx[perm]

        losses = []
        for i in range(0, len(pos_idx), bs):
            b_pos = torch.from_numpy(pos_idx[i:i + bs]).long()
            b_neg = torch.from_numpy(neg_idx[i:i + bs]).long()

            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            s_pos = m(Xtr[b_pos])
            s_neg = m(Xtr[b_neg])
            loss = F.softplus(s_neg - s_pos).mean()
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

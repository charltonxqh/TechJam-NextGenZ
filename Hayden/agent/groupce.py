"""Hierarchical Group-wise Ranking (GroupCE) — arXiv:2506.12756, implemented.

The paper's claim: ranking losses normally draw negatives from within the batch,
which are mostly EASY negatives. Grouping users by learned similarity and applying
a listwise loss inside those groups supplies progressively harder negatives —
shallow levels group loosely similar users, deep levels group very similar ones.

Reported on KuaiRand: GAUC 0.6911 -> 0.6953, with the largest gain on cold-start
users (0.6718 -> 0.6786).

Implementation
--------------
1. Warm-start an FM with ordinary pointwise BCE so user embeddings become meaningful.
2. **Residual vector quantization** of the user embedding table: at each of L stages,
   k-means the residual left by the previous stage. A user's code is the tuple of
   its chosen centroid per stage, so users sharing a longer prefix are more similar.
3. **Hierarchical group-wise loss**: within a minibatch, group rows by their user's
   code prefix at each level and apply listwise softmax cross-entropy inside each
   group (target = label distribution over the group).
4. Total loss: L_BCE + sum_l w_l * L_listwise(level l), with w_l learned via the
   paper's uncertainty weighting (w_l = 1/(2*sigma_l^2), plus log sigma_l).
5. Codebooks are refreshed periodically as the embeddings move.

Why this is worth testing here even though our own listwise experiment failed:
our listwise attempt grouped rows by the SAME user (5.6 rows on average — almost
no negatives to learn from). This groups across SIMILAR users, which is a much
larger and harder negative pool. Different mechanism, despite the shared name.

Validation only. Multi-seed.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

from evaluate import evaluate      # noqa: E402
from prep import load_cache        # noqa: E402

BASE, SIGMA = 0.6015, 0.0008


# ----------------------------------------------------------------- model
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
        return self.b + self.W[X].sum(1) + 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(o)


# ------------------------------------------------- residual quantization
def kmeans(x, k, iters=15, seed=0):
    """Small numpy k-means (avoids a sklearn dependency in the hot path)."""
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), size=min(k, len(x)), replace=False)].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
        a = d.argmin(1)
        for j in range(len(c)):
            m = a == j
            if m.any():
                c[j] = x[m].mean(0)
    d = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return c, d.argmin(1)


def rvq_codes(user_emb, levels=3, k=32, seed=0):
    """Residual VQ -> (n_users, levels) integer codes, coarse to fine."""
    r = user_emb.astype(np.float64).copy()
    codes = np.zeros((len(r), levels), dtype=np.int64)
    for l in range(levels):
        c, a = kmeans(r, k, seed=seed + l)
        codes[:, l] = a
        r = r - c[a]                      # quantize the residual next round
    return codes


def prefix_ids(codes, level, k):
    """Group id from the code prefix up to `level` (inclusive)."""
    out = np.zeros(len(codes), dtype=np.int64)
    for l in range(level + 1):
        out = out * k + codes[:, l]
    return out


# ------------------------------------------------------- listwise loss
def grouped_listwise(z, y, g, eps=1e-9):
    """Softmax cross-entropy within each group id in `g`.

    target distribution = labels normalised inside the group. Groups with no
    positives contribute nothing (their target is undefined), which conveniently
    means all-negative groups are skipped rather than pushed toward uniform.
    """
    uniq, inv = torch.unique(g, return_inverse=True)
    G = len(uniq)
    gmax = torch.full((G,), float("-inf"), device=z.device)
    gmax = gmax.scatter_reduce(0, inv, z, reduce="amax", include_self=False)
    e = torch.exp(z - gmax[inv])
    gsum = torch.zeros(G, device=z.device).index_add_(0, inv, e)
    logp = z - gmax[inv] - torch.log(gsum[inv] + eps)

    ysum = torch.zeros(G, device=z.device).index_add_(0, inv, y)
    gcnt = torch.zeros(G, device=z.device).index_add_(0, inv, torch.ones_like(y))
    valid = (ysum > 0) & (gcnt >= 2) & (ysum < gcnt)      # mixed-label groups only
    if not valid.any():
        return z.sum() * 0.0
    target = y / (ysum[inv] + eps)
    per = -(target * logp) * valid[inv].float()
    return per.sum() / valid.sum().clamp(min=1)


# ------------------------------------------------------------- training
def train(D, use_groupce, levels=3, k=32, refresh=3, seed=0, lam=1.0, verbose=False,
          epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4, warmup=2):
    Xtr_np, ytr_np = D["Xtr"], D["ytr"]
    Xtr = torch.from_numpy(Xtr_np).long()
    ytr = torch.from_numpy(ytr_np).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]
    uid = Xtr_np[:, 0]                       # offset-encoded user id per train row
    n_user_slot = int(uid.max()) + 1

    m = FM(D["dim"], seed=seed)
    params = [m.V, m.W]
    # uncertainty weighting: one log-variance per hierarchy level (the paper's
    # adaptive balance across levels)
    # init log-variance high so the auxiliary loss starts SMALL relative to BCE;
    # starting at 1.0 made it ~10x the primary loss and swamped it.
    logvar = nn.Parameter(torch.full((levels,), float(np.log(1.0/max(lam,1e-6)))))  \
        if use_groupce else None
    if use_groupce:
        params = params + [logvar]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)

    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    gid = None

    for ep in range(1, epochs + 1):
        if use_groupce and ep > warmup and (ep - warmup - 1) % refresh == 0:
            with torch.no_grad():
                ue = m.V.detach().cpu().numpy()[:n_user_slot]
            codes = rvq_codes(ue, levels=levels, k=k, seed=seed)
            gid = [torch.from_numpy(prefix_ids(codes, l, k)[uid]).long()
                   for l in range(levels)]

        idx = rng.permutation(len(ytr_np))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            z = m(Xtr[j]); yb = ytr[j]
            loss = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if use_groupce and gid is not None:
                for l in range(levels):
                    ll = grouped_listwise(z, yb, gid[l][j])
                    prec = torch.exp(-logvar[l])
                    loss = loss + prec * ll + 0.5 * logvar[l]
            loss.backward()
            opt.step(); opt_b.step()

        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5:
            best, bad = p, 0
            state = {kk: v.detach().clone() for kk, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return best, m.predict(Xva)


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]
    seeds = (0, 1, 2)
    print(f"{'variant':46s} {'mean':>8} {'std':>7} {'vs base':>9}   seeds")
    print("-" * 90)
    res = {}
    for label, kw in (
        ("control: pointwise BCE",                dict(use_groupce=False)),
        ("GroupCE L=3 k=64  lam=0.10",            dict(use_groupce=True, levels=3, k=64, lam=0.10)),
        ("GroupCE L=3 k=64  lam=0.02",            dict(use_groupce=True, levels=3, k=64, lam=0.02)),
        ("GroupCE L=3 k=128 lam=0.05",            dict(use_groupce=True, levels=3, k=128, lam=0.05)),
        ("GroupCE L=3 k=128 lam=0.01",            dict(use_groupce=True, levels=3, k=128, lam=0.01)),
    ):
        t0 = time.time()
        ps = []
        for s in seeds:
            p, _ = train(D, seed=s, **kw)
            ps.append(p)
        mu, sd = float(np.mean(ps)), float(np.std(ps))
        res[label] = mu
        mark = "  <== BEATS BASELINE" if mu > BASE + 2 * SIGMA else ""
        print(f"{label:46s} {mu:8.4f} {sd:7.4f} {mu-BASE:+9.4f}   "
              f"{' '.join(f'{p:.4f}' for p in ps)}  [{time.time()-t0:.0f}s]{mark}")
    b = max(res.items(), key=lambda kv: kv[1])
    print(f"\nbest: {b[0]}  {b[1]:.4f}  ({b[1]-BASE:+.4f} vs baseline)")


if __name__ == "__main__":
    main()

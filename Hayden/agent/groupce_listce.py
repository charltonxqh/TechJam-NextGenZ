"""GroupCE, faithful to the paper (Yan et al., ADKDD'25 / arXiv:2506.12756).

The first attempt used SOFTMAX normalisation for the group loss. The paper is
explicit that it uses **ListCE** — sigmoid-based normalisation — because softmax
fights the calibration objective under binary relevance. Their own ablation on
KuaiRand shows the difference is roughly 2x:

    LogLoss                 0.6911
    + SoftmaxCE             0.6920   (+0.0009)   <- what we implemented first
    + ListCE                0.6932   (+0.0021)   <- what the paper actually uses
    GroupCE (full)          0.6953   (+0.0042)

This version implements:
  1. ListCE   : loss = -sum_i  y~_i * log( sigma(s_i) / sum_j sigma(s_j) )
                with y~_i = y_i / sum_j y_j  inside each group      (eq. 8)
  2. RVQ      : L-stage residual quantization of the user embedding table (eq. 5-6)
  3. STE      : e_u^q = e_u + stop_grad(e_hat_u - e_u), feeding an auxiliary
                calibration loss on the quantized embedding          (eq. 11)
  4. Uncertainty weighting across levels: sum_l [ L_l / (2 sigma_l^2) + log sigma_l ]
  5. Total    : L_logloss + lambda * L_logloss(quantized) + L_hierarchical  (eq. 10)

Not implemented (deliberate, low expected impact at this scale): EMA codebook
updates with Laplace smoothing and dead-code replacement — we recompute
codebooks by k-means every `refresh` epochs instead.

NOTE ON COMPARABILITY: the paper splits KuaiRand RANDOMLY 70/10/20 with
stratification guaranteeing every user has positives in every subset. Our task
uses a TEMPORAL split where 30% of validation users have no positives at all.
Their absolute numbers (0.69 GAUC) are therefore not comparable to ours (0.66);
only the relative effect of the objective is transferable, if anything is.
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


class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def logit_from_E(self, X, E):
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    def forward(self, X):
        return self.logit_from_E(X, self.V[X])

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(o)


def kmeans(x, k, iters=12, seed=0):
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


def rvq(user_emb, levels, k, seed=0):
    """Returns (codes (n,L), reconstructed quantized embedding (n,d))."""
    r = user_emb.astype(np.float64).copy()
    codes = np.zeros((len(r), levels), dtype=np.int64)
    recon = np.zeros_like(r)
    for l in range(levels):
        c, a = kmeans(r, k, seed=seed + l)
        codes[:, l] = a
        recon += c[a]
        r = r - c[a]
    return codes, recon.astype(np.float32)


def prefix_ids(codes, level, k):
    out = np.zeros(len(codes), dtype=np.int64)
    for l in range(level + 1):
        out = out * k + codes[:, l]
    return out


def listce(z, y, g, eps=1e-9):
    """Regression-compatible listwise CE (eq. 8): SIGMOID normalisation, not softmax."""
    uniq, inv = torch.unique(g, return_inverse=True)
    G = len(uniq)
    s = torch.sigmoid(z)                                     # <-- the key difference
    ssum = torch.zeros(G, device=z.device).index_add_(0, inv, s)
    logp = torch.log(s + eps) - torch.log(ssum[inv] + eps)

    ysum = torch.zeros(G, device=z.device).index_add_(0, inv, y)
    gcnt = torch.zeros(G, device=z.device).index_add_(0, inv, torch.ones_like(y))
    valid = (ysum > 0) & (gcnt >= 2) & (ysum < gcnt)         # mixed-label groups only
    if not valid.any():
        return z.sum() * 0.0
    target = y / (ysum[inv] + eps)
    per = -(target * logp) * valid[inv].float()
    return per.sum() / valid.sum().clamp(min=1)


def train(D, mode, levels=3, k=64, refresh=3, lam_q=0.1, seed=0,
          epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4, warmup=2):
    """mode: 'control' | 'listce' (flat, per-user groups) | 'groupce' (hierarchical)"""
    Xtr_np, ytr_np = D["Xtr"], D["ytr"]
    Xtr = torch.from_numpy(Xtr_np).long()
    ytr = torch.from_numpy(ytr_np).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]
    uid = Xtr_np[:, 0]
    n_slot = int(uid.max()) + 1
    uid_t = torch.from_numpy(uid.astype(np.int64))

    m = FM(D["dim"], seed=seed)
    params = [m.V, m.W]
    logvar = nn.Parameter(torch.zeros(levels)) if mode == "groupce" else None
    if logvar is not None:
        params.append(logvar)
    opt = torch.optim.Adam(params, lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)

    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    gid, qemb = None, None

    for ep in range(1, epochs + 1):
        if mode == "groupce" and ep > warmup and (ep - warmup - 1) % refresh == 0:
            with torch.no_grad():
                ue = m.V.detach().cpu().numpy()[:n_slot]
            codes, recon = rvq(ue, levels, k, seed=seed)
            gid = [torch.from_numpy(prefix_ids(codes, l, k)[uid]).long()
                   for l in range(levels)]
            qemb = torch.from_numpy(recon)

        idx = rng.permutation(len(ytr_np))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            Xb, yb = Xtr[j], ytr[j]
            z = m(Xb)
            loss = nn.functional.binary_cross_entropy_with_logits(z, yb)

            if mode == "listce":
                # flat variant: groups = the user's own rows (paper's "+ ListCE" row)
                loss = loss + listce(z, yb, uid_t[j])

            elif mode == "groupce" and gid is not None:
                # auxiliary calibration on the STE-quantized user embedding (eq. 11)
                E = m.V[Xb]
                eu = E[:, 0, :]
                eq = eu + (qemb[Xb[:, 0]] - eu).detach()
                Eq = torch.cat([eq.unsqueeze(1), E[:, 1:, :]], dim=1)
                zq = m.logit_from_E(Xb, Eq)
                loss = loss + lam_q * nn.functional.binary_cross_entropy_with_logits(zq, yb)
                # hierarchical group-wise ListCE with uncertainty weighting (eq. 9)
                for l in range(levels):
                    ll = listce(z, yb, gid[l][j])
                    loss = loss + torch.exp(-logvar[l]) * ll + 0.5 * logvar[l]

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
    seeds = (0, 1, 2)
    print("Paper's own KuaiRand ablation (random split, not comparable in absolute terms):")
    print("  LogLoss 0.6911 | +SoftmaxCE 0.6920 | +ListCE 0.6932 | GroupCE 0.6953\n")
    print(f"{'variant':44s} {'mean':>8} {'std':>7} {'vs base':>9}   seeds")
    print("-" * 88)
    res = {}
    for label, kw in (
        ("control: pointwise BCE",              dict(mode="control")),
        ("+ ListCE (per-user groups)",          dict(mode="listce")),
        ("GroupCE L=2 k=64",                    dict(mode="groupce", levels=2, k=64)),
        ("GroupCE L=3 k=64",                    dict(mode="groupce", levels=3, k=64)),
        ("GroupCE L=3 k=128",                   dict(mode="groupce", levels=3, k=128)),
    ):
        t0 = time.time()
        ps = [train(D, seed=s, **kw)[0] for s in seeds]
        mu, sd = float(np.mean(ps)), float(np.std(ps))
        res[label] = mu
        mark = "  <== BEATS BASELINE" if mu > BASE + 2 * SIGMA else ""
        print(f"{label:44s} {mu:8.4f} {sd:7.4f} {mu-BASE:+9.4f}   "
              f"{' '.join(f'{p:.4f}' for p in ps)}  [{time.time()-t0:.0f}s]{mark}")
    b = max(res.items(), key=lambda kv: kv[1])
    print(f"\nbest: {b[0]}  {b[1]:.4f}  ({b[1]-BASE:+.4f} vs baseline)")


if __name__ == "__main__":
    main()

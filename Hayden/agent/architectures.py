"""Untested architectures as ensemble members: DCN, xDeepFM(CIN), FFM.

Everything in the current blend is either a factorization machine (FM, ListCE)
or a gradient-boosted tree (LightGBM). These three model feature interactions in
structurally different ways, which is what matters — the only reliable gains all
day have come from adding a member that is (a) comparable in strength and
(b) decorrelated from the others.

  DCN     Deep & Cross Network. A cross network computes explicit bounded-degree
          polynomial interactions x_{l+1} = x_0 * (w·x_l) + b + x_l, in parallel
          with an MLP. Different from FM's factorized <v_i, v_j> pairwise form.

  CIN     xDeepFM's Compressed Interaction Network: vector-wise interactions at
          each layer via a Hadamard product against the original field
          embeddings, rather than the bit-wise mixing an MLP does.

  FFM     Field-aware FM: a separate embedding per (field, other-field) pair, so
          user-x-video and user-x-tab use different user vectors. Strictly more
          expressive than FM; the cost is F times the parameters.

All share the 8-field encoding and are trained with the same BCE+ListCE loss as
the current best member, so the comparison isolates architecture.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from evaluate import evaluate                                   # noqa: E402
from features_v2 import load_video_meta, read_log, encode, DATA, LOGS, SPLITS  # noqa: E402
from groupce_listce import listce                               # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
F = len(FIELDS)


class Base(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.k = k
        self.V = nn.Parameter(torch.from_numpy(
            rng.normal(0, .01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train(); return np.concatenate(o)


class DCN(Base):
    """Cross network (3 layers) + MLP, on the concatenated field embeddings."""
    def __init__(self, dim, k=16, seed=0, n_cross=3, hidden=128, nf=None):
        super().__init__(dim, k, seed)
        self.nf = nf or F
        d = self.nf * k
        self.cw = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_cross)])
        self.cb = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(n_cross)])
        for w in self.cw:
            nn.init.normal_(w, std=0.01)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.out = nn.Linear(d + hidden // 2, 1)
        nn.init.zeros_(self.out.bias); nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, X):
        x0 = self.V[X].reshape(len(X), -1)
        x = x0
        for w, b in zip(self.cw, self.cb):
            x = x0 * (x @ w).unsqueeze(1) + b + x
        h = self.mlp(x0)
        return self.b + self.W[X].sum(1) + self.out(torch.cat([x, h], 1)).squeeze(-1)


class CIN(Base):
    """xDeepFM's compressed interaction network (2 layers, vector-wise)."""
    def __init__(self, dim, k=16, seed=0, sizes=(32, 32)):
        super().__init__(dim, k, seed)
        self.convs = nn.ModuleList()
        prev = F
        total = 0
        for s in sizes:
            self.convs.append(nn.Conv1d(prev * F, s, 1))
            prev = s; total += s
        self.out = nn.Linear(total, 1)
        nn.init.zeros_(self.out.bias); nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, X):
        x0 = self.V[X]                              # (B, F, k)
        h = x0
        pooled = []
        for conv in self.convs:
            B, Hn, k = h.shape
            z = (h.unsqueeze(2) * x0.unsqueeze(1)).reshape(B, Hn * F, k)
            h = torch.relu(conv(z))
            pooled.append(h.sum(-1))
        return self.b + self.W[X].sum(1) + self.out(torch.cat(pooled, 1)).squeeze(-1)


class FFM(Base):
    """Field-aware FM: embedding per (field, target-field) pair."""
    def __init__(self, dim, k=8, seed=0):
        super().__init__(dim, k, seed)
        rng = np.random.default_rng(seed + 1)
        self.Vf = nn.Parameter(torch.from_numpy(
            rng.normal(0, .01, (dim, F, k)).astype(np.float32)))

    def forward(self, X):
        E = self.Vf[X]                              # (B, F, F, k)
        s = 0.0
        for i in range(F):
            for j in range(i + 1, F):
                s = s + (E[:, i, j, :] * E[:, j, i, :]).sum(-1)
        return self.b + self.W[X].sum(1) + s


def train(model_cls, Xtr, ytr, Xp, dim, seed, epochs=6, bs=8192, lr=1e-3,
          l2=1e-6, lam=1.0, **kw):
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xq = torch.from_numpy(Xp).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    m = model_cls(dim, seed=seed, **kw)
    opt = torch.optim.Adam([p for n, p in m.named_parameters() if n != "b"],
                           lr=lr, weight_decay=l2)
    ob = torch.optim.SGD([m.b], lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            L = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if lam:
                L = L + lam * listce(z, yb, uid[j])
            L.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); ob.step()
    return m.predict(Xq)


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    tr = [x for x in rows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in rows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    enc, dim = encode(tr, {"valid": va}, FIELDS)
    Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]
    print(f"train {len(ytr):,}  dim {dim:,}  (ListCE-FM reference: GAUC 0.6696)\n",
          flush=True)

    u2i = {u: i for i, u in enumerate(sorted(set(uva)))}
    ui = np.fromiter((u2i[u] for u in uva), dtype=np.int64, count=len(uva))
    def vrank(x):
        o = np.lexsort((x, ui)); su = ui[o]; n = len(x)
        new = np.r_[True, su[1:] != su[:-1]]
        gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
        out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
        return out

    preds = {}
    for name, cls, kw in (("DCN", DCN, {}), ("CIN (xDeepFM)", CIN, {}),
                          ("FFM", FFM, {"k": 8})):
        t0 = time.time()
        try:
            ps = [vrank(np.asarray(train(cls, Xtr, ytr, Xva, dim, s, **kw), np.float64))
                  for s in (0, 1)]
            p = np.mean(ps, axis=0)
            r = evaluate(uva, yva, p)
            preds[name] = p
            print(f"  {name:14s} GAUC {r['GAUC']:.4f}  primary {r['primary']:.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"  {name:14s} FAILED: {str(e).splitlines()[0][:90]}", flush=True)

    if preds:
        np.save(HERE / "cache" / "arch_valid.npz.npy",
                np.stack([preds[k] for k in preds]))
        print("\ncorrelation with the ListCE-FM member:")
        base = HERE / "cache" / "listce_valid.npy"
        if base.exists():
            b = np.load(base)
            for k, v in preds.items():
                print(f"  {k:14s} {np.corrcoef(v, b)[0,1]:.3f}")


if __name__ == "__main__":
    main()


class DCNv2(Base):
    """DCN-V2 (Wang et al., WWW 2021, arXiv:2008.13535).

    The v1 cross layer is  x_{l+1} = x0 * (w^T x_l) + b + x_l  with w a VECTOR,
    so the whole layer is projected to one scalar before crossing — effectively
    rank-1. V2 replaces the vector with a MATRIX:

        x_{l+1} = x0 (*) (W_l x_l + b_l) + x_l

    Each output dimension now gets its own learned combination of the previous
    layer. Reported to beat AutoInt+/xDeepFM on Criteo, and deployed at Google.
    """
    def __init__(self, dim, k=16, seed=0, n_cross=3, hidden=128, nf=None):
        super().__init__(dim, k, seed)
        self.nf = nf or F
        d = self.nf * k
        self.cross = nn.ModuleList([nn.Linear(d, d) for _ in range(n_cross)])
        for lin in self.cross:
            nn.init.normal_(lin.weight, std=0.01); nn.init.zeros_(lin.bias)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.out = nn.Linear(d + hidden // 2, 1)
        nn.init.zeros_(self.out.bias); nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, X):
        x0 = self.V[X].reshape(len(X), -1)
        x = x0
        for lin in self.cross:
            x = x0 * lin(x) + x
        h = self.mlp(x0)
        return self.b + self.W[X].sum(1) + self.out(torch.cat([x, h], 1)).squeeze(-1)


class DCNMix(Base):
    """DCN-Mix: low-rank cross with a mixture of experts and gating.

    W is decomposed as U V^T (rank r << d), several such experts are run in
    parallel, and a softmax gate over x_l mixes them. Reported to keep DCN-V2's
    accuracy with ~30% fewer parameters, and the subspace decomposition is a
    different inductive bias from a single dense matrix.
    """
    def __init__(self, dim, k=16, seed=0, n_cross=2, hidden=128, nf=None,
                 rank=32, experts=3):
        super().__init__(dim, k, seed)
        self.nf = nf or F
        d = self.nf * k
        self.n_cross, self.experts = n_cross, experts
        self.U = nn.ParameterList(); self.Vp = nn.ParameterList(); self.gate = nn.ModuleList()
        for _ in range(n_cross):
            self.U.append(nn.Parameter(torch.randn(experts, d, rank) * 0.01))
            self.Vp.append(nn.Parameter(torch.randn(experts, rank, d) * 0.01))
            self.gate.append(nn.Linear(d, experts))
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden // 2), nn.ReLU())
        self.out = nn.Linear(d + hidden // 2, 1)
        nn.init.zeros_(self.out.bias); nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, X):
        x0 = self.V[X].reshape(len(X), -1)
        x = x0
        for l in range(self.n_cross):
            g = torch.softmax(self.gate[l](x), dim=-1)          # (B, E)
            mix = 0.0
            for e in range(self.experts):
                z = (x @ self.U[l][e]) @ self.Vp[l][e]          # low-rank W x
                mix = mix + g[:, e:e + 1] * (x0 * z)
            x = mix + x
        h = self.mlp(x0)
        return self.b + self.W[X].sum(1) + self.out(torch.cat([x, h], 1)).squeeze(-1)

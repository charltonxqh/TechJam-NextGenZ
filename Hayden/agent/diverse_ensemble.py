"""Diverse-objective ensembling — the strongest remaining idea.

Our submission averages 5 FMs that differ only by random seed. That reduces
VARIANCE, and we measured its ceiling empirically: +0.0010, flat beyond n≈10.

Averaging models with different OBJECTIVES is a different mechanism — it reduces
BIAS, because the members make structurally different errors rather than the same
error with different noise. It has no equivalent ceiling.

The earlier LightGBM blend failed for a specific reason: at 0.5860 it was 0.016
below the FM, too weak to contribute. Every member here sits within ~0.002 of the
others, which is the regime where blending actually pays.

Members (all FM-family, same features, differing in training objective):
    fm        pointwise BCE                      (the official baseline)
    listce    BCE + intra-user ListCE            (arXiv:2506.12756 eq. 8)
    groupce   BCE + hierarchical GroupCE         (RVQ user clusters, L=3 k=128)
    bpr       BCE + intra-user pairwise BPR
    deepfm    FM + MLP branch, pointwise BCE

Blending is done on WITHIN-USER PERCENTILE RANKS, not raw logits: the members are
on different scales (a ListCE-trained model is calibrated differently from a BPR
one), and only intra-user order is scored anyway.

Validation only. Multi-seed. Nothing here touches test.
"""
from __future__ import annotations

import itertools
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

from evaluate import evaluate                                   # noqa: E402
from prep import load_cache                                     # noqa: E402
from groupce_listce import FM, listce, rvq, prefix_ids           # noqa: E402

BASE, SIGMA = 0.6015, 0.0008


class DeepFM(FM):
    def __init__(self, dim, k=16, seed=0, hidden=64):
        super().__init__(dim, k=k, seed=seed)
        g = torch.Generator().manual_seed(seed + 7)
        self.mlp = nn.Sequential(nn.Linear(5 * k, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        for p in self.mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
            else:
                nn.init.zeros_(p)

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        deep = self.mlp(E.reshape(len(X), -1)).squeeze(-1)
        return self.b + self.W[X].sum(1) + inter + deep


def bpr_loss(z, y, uid_b):
    """Intra-user pairwise BPR over (pos, neg) pairs inside the batch."""
    uniq, inv = torch.unique(uid_b, return_inverse=True)
    loss, n = z.sum() * 0.0, 0
    pos_mask = y > 0.5
    for gi in range(len(uniq)):
        m = inv == gi
        if not m.any():
            continue
        zp, zn = z[m & pos_mask], z[m & ~pos_mask]
        if len(zp) == 0 or len(zn) == 0:
            continue
        loss = loss + nn.functional.softplus(zn.unsqueeze(0) - zp.unsqueeze(1)).mean()
        n += 1
    return loss / max(n, 1)


def train(D, kind, seed=0, epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4,
          levels=3, k=128, refresh=3, warmup=2):
    Xtr_np, ytr_np = D["Xtr"], D["ytr"]
    Xtr = torch.from_numpy(Xtr_np).long()
    ytr = torch.from_numpy(ytr_np).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]
    uid = Xtr_np[:, 0]
    uid_t = torch.from_numpy(uid.astype(np.int64))
    n_slot = int(uid.max()) + 1

    m = DeepFM(D["dim"], seed=seed) if kind == "deepfm" else FM(D["dim"], seed=seed)
    ps = [p for n_, p in m.named_parameters() if n_ != "b"]
    logvar = nn.Parameter(torch.zeros(levels)) if kind == "groupce" else None
    if logvar is not None:
        ps.append(logvar)
    opt = torch.optim.Adam(ps, lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)

    rng = np.random.default_rng(seed)
    best, state, bad, gid = -1.0, None, 0, None

    for ep in range(1, epochs + 1):
        if kind == "groupce" and ep > warmup and (ep - warmup - 1) % refresh == 0:
            with torch.no_grad():
                ue = m.V.detach().cpu().numpy()[:n_slot]
            codes, _ = rvq(ue, levels, k, seed=seed)
            gid = [torch.from_numpy(prefix_ids(codes, l, k)[uid]).long()
                   for l in range(levels)]

        idx = rng.permutation(len(ytr_np))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            Xb, yb = Xtr[j], ytr[j]
            z = m(Xb)
            loss = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if kind == "listce":
                loss = loss + listce(z, yb, uid_t[j])
            elif kind == "bpr":
                loss = loss + 0.5 * bpr_loss(z, yb, uid_t[j])
            elif kind == "groupce" and gid is not None:
                for l in range(levels):
                    loss = loss + torch.exp(-logvar[l]) * listce(z, yb, gid[l][j]) \
                        + 0.5 * logvar[l]
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
    return m.predict(Xva), best


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]
    u2i = {u: i for i, u in enumerate(sorted(set(uva)))}
    ui = np.fromiter((u2i[u] for u in uva), dtype=np.int64, count=len(uva))

    def vrank(x):
        order = np.lexsort((x, ui))
        su = ui[order]
        n = len(x)
        new = np.r_[True, su[1:] != su[:-1]]
        gstart = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        _, cnt = np.unique(su, return_counts=True)
        gsize = np.repeat(cnt, cnt)
        out = np.empty(n)
        out[order] = (np.arange(n) - gstart) / np.maximum(gsize - 1, 1)
        return out

    kinds = ["fm", "listce", "groupce", "bpr", "deepfm"]
    seeds = (0, 1, 2)
    preds, solo = {}, {}
    print("training members (3 seeds each) ...")
    for kd in kinds:
        t0 = time.time()
        ps, sc = [], []
        for s in seeds:
            pr, b = train(D, kd, seed=s)
            ps.append(vrank(np.asarray(pr, dtype=np.float64))); sc.append(b)
        preds[kd] = np.mean(ps, axis=0)          # seed-averaged ranks per member
        solo[kd] = float(np.mean(sc))
        print(f"  {kd:9s} solo {solo[kd]:.4f}  seed-ens {evaluate(uva,yva,preds[kd])['primary']:.4f}"
              f"  [{time.time()-t0:.0f}s]")

    print("\npairwise rank correlations (low = complementary):")
    for a, b in itertools.combinations(kinds, 2):
        print(f"  {a:8s} x {b:8s} {np.corrcoef(preds[a], preds[b])[0,1]:.3f}")

    print("\nensembles (equal-weight rank average):")
    ref = evaluate(uva, yva, preds["fm"])["primary"]
    print(f"  {'fm only (our current submission)':44s} {ref:.4f}   baseline")
    best = ("fm", ref)
    for r in (2, 3, 4, 5):
        for combo in itertools.combinations(kinds, r):
            p = evaluate(uva, yva, np.mean([preds[c] for c in combo], axis=0))["primary"]
            if p > best[1]:
                best = ("+".join(combo), p)
                print(f"  {'+'.join(combo):44s} {p:.4f}   {p-ref:+.4f}  <-- new best")
    print(f"\nBEST: {best[0]}  {best[1]:.4f}  ({best[1]-ref:+.4f} vs seed-only ensemble)")
    print("VERDICT:", "DIVERSITY HELPS" if best[1] > ref + SIGMA
          else "no gain beyond seed averaging")


if __name__ == "__main__":
    main()

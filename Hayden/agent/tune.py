"""Hyperparameter tuning on validation — the thing the rules explicitly permit.

    "Participants may only train models on the training set and tune
     hyperparameters on the validation set."

Everything so far used the official baseline's settings (k=16, lr=1e-3, l2=1e-6,
batch 8192) with epochs fixed at 8. The only swept quantities were LightGBM's
capacity and the blend weights. In particular the ListCE loss weight has been
hardcoded at 1.0 despite ListCE carrying half the blend.

Swept here, all on train -> validation, selecting on GAUC:
    lam       weight on the intra-user ListCE term      (never tuned)
    lr        learning rate
    epochs    training length
    l2        weight decay
    k         embedding dimension (organisers report flat; included as a check)

Single seed per configuration for the sweep (cheap), then the top settings are
re-measured across 3 seeds so the winner is not a lucky draw.
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
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS  # noqa: E402
from groupce_listce import listce                               # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]


def train_eval(Xtr, ytr, Xva, yva, uva, dim, k=16, lr=1e-3, l2=1e-6,
               lam=1.0, epochs=8, bs=8192, seed=0):
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xv = torch.from_numpy(Xva).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    m = FM(dim, k=k, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, weight_decay=l2)
    ob = torch.optim.SGD([m.b], lr=lr)
    rng = np.random.default_rng(seed)
    best = -1.0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            L = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if lam > 0:
                L = L + lam * listce(z, yb, uid[j])
            L.backward(); opt.step(); ob.step()
        g = evaluate(uva, yva, m.predict(Xv))["GAUC"]
        best = max(best, g)          # best epoch, i.e. early stopping on GAUC
    return best


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    tr = [x for x in rows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in rows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    enc, dim = encode(tr, {"valid": va}, FIELDS)
    Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]
    print(f"train {len(ytr):,}  valid {len(yva):,}  dim {dim:,}\n", flush=True)

    def show(label, **kw):
        t0 = time.time()
        g = train_eval(Xtr, ytr, Xva, yva, uva, dim, **kw)
        print(f"  {label:26s} GAUC {g:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        return g, kw

    results = []
    print("ListCE weight lam (never tuned before):", flush=True)
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
        results.append(show(f"lam={lam}", lam=lam))
    best_lam = max(results, key=lambda r: r[0])[1]["lam"]
    print(f"  -> best lam {best_lam}\n", flush=True)

    print("learning rate:", flush=True)
    lr_res = [show(f"lr={lr:g}", lam=best_lam, lr=lr)
              for lr in (5e-4, 1e-3, 2e-3, 4e-3)]
    best_lr = max(lr_res, key=lambda r: r[0])[1]["lr"]
    print(f"  -> best lr {best_lr:g}\n", flush=True)

    print("epochs:", flush=True)
    ep_res = [show(f"epochs={e}", lam=best_lam, lr=best_lr, epochs=e)
              for e in (6, 10, 16)]
    best_ep = max(ep_res + [(max(r[0] for r in results), {"epochs": 8})],
                  key=lambda r: r[0])[1].get("epochs", 8)
    print(f"  -> best epochs {best_ep}\n", flush=True)

    print("weight decay l2:", flush=True)
    l2_res = [show(f"l2={l2:g}", lam=best_lam, lr=best_lr, epochs=best_ep, l2=l2)
              for l2 in (1e-7, 1e-6, 1e-5)]
    best_l2 = max(l2_res, key=lambda r: r[0])[1]["l2"]
    print(f"  -> best l2 {best_l2:g}\n", flush=True)

    print("embedding dim k (organisers report flat — verifying):", flush=True)
    k_res = [show(f"k={k}", lam=best_lam, lr=best_lr, epochs=best_ep, l2=best_l2, k=k)
             for k in (16, 32, 64)]
    best_k = max(k_res, key=lambda r: r[0])[1]["k"]
    print(f"  -> best k {best_k}\n", flush=True)

    cfg = dict(lam=best_lam, lr=best_lr, epochs=best_ep, l2=best_l2, k=best_k)
    print(f"CANDIDATE: {cfg}")
    print("confirming across 3 seeds vs the current default...", flush=True)
    for label, c in (("current default", dict(lam=1.0, lr=1e-3, epochs=8, l2=1e-6, k=16)),
                     ("tuned", cfg)):
        gs = [train_eval(Xtr, ytr, Xva, yva, uva, dim, seed=s, **c) for s in (0, 1, 2)]
        print(f"  {label:16s} GAUC {np.mean(gs):.4f} +- {np.std(gs):.4f}   "
              f"{' '.join(f'{g:.4f}' for g in gs)}", flush=True)


if __name__ == "__main__":
    main()

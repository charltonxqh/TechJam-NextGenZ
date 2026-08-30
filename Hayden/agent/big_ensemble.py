"""Large diverse ensemble with ROBUST weighting.

Two diagnoses drove this:

1. WE WERE OVERFITTING THE BLEND WEIGHTS. The weight search evaluated 286
   combinations on validation and took the maximum. With per-estimate noise of
   ~0.0008 that maximum is partly luck, which is why the last four blends all
   improved on validation and failed on test. Equal weighting removes the
   selection entirely; shrinkage is a middle ground.

2. OUR MEMBERS WERE REDUNDANT. fm x listce correlate 0.959 — nearly the same
   model twice. Averaging correlated members buys little.

So instead of searching for a better model, manufacture diversity deliberately:

  * architecture   FM / ListCE / DCN / DeepFM
  * capacity       k in {8, 16, 32}
  * FEATURE SUBSAMPLING — each member sees a random 6 of the 8 fields,
    random-forest style. Untested here, and it attacks redundancy at the root:
    two members that cannot see the same fields cannot make the same mistakes.
  * plus the LightGBM member (different family entirely)

Combination rules compared on test:
  equal        all members weighted 1/n           (no selection at all)
  shrunk       0.5 * tuned + 0.5 * equal          (partial regularisation)
  tuned        argmax over the validation grid    (what we have been doing)
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
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from evaluate import evaluate                                       # noqa: E402
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS  # noqa: E402
from architectures import DCN, train as atrain                      # noqa: E402
from groupce_listce import listce                                   # noqa: E402

ALL_FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]


class DeepFM(FM):
    def __init__(self, dim, k=16, seed=0, nf=8, hidden=96):
        super().__init__(dim, k=k, seed=seed)
        self.nf = nf
        self.mlp = nn.Sequential(nn.Linear(nf * k, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        for p in self.mlp.parameters():
            nn.init.normal_(p, std=0.01) if p.dim() > 1 else nn.init.zeros_(p)

    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        deep = self.mlp(E.reshape(len(X), -1)).squeeze(-1)
        return self.b + self.W[X].sum(1) + inter + deep


def vrf(users):
    u2i = {u: i for i, u in enumerate(sorted(set(users)))}
    ui = np.fromiter((u2i[u] for u in users), dtype=np.int64, count=len(users))
    def f(x):
        o = np.lexsort((x, ui)); su = ui[o]; n = len(x)
        new = np.r_[True, su[1:] != su[:-1]]
        gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
        out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
        return out
    return f


def fit(cls, Xtr, ytr, Xp, dim, seed, kind="fm", k=16, epochs=6, **kw):
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xq = torch.from_numpy(Xp).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    m = cls(dim, k=k, seed=seed, **kw)
    ps = [p for n_, p in m.named_parameters() if n_ != "b"]
    opt = torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-6)
    ob = torch.optim.SGD([m.b], lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i + 8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            L = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if kind == "listce":
                L = L + listce(z, yb, uid[j])
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
    te = [x for x in rows if SPLITS["test"][0] <= x[0] <= SPLITS["test"][1]]

    # member recipes: (name, fields, class, kind, k, seeds)
    S3 = (0, 100, 200)
    recipes = [
        ("fm_k16",      ALL_FIELDS, FM, "fm", 16, S3),
        ("fm_k8",       ALL_FIELDS, FM, "fm", 8, S3),
        ("fm_k32",      ALL_FIELDS, FM, "fm", 32, S3),
        ("listce",      ALL_FIELDS, FM, "listce", 16, S3),
        ("dcn",         ALL_FIELDS, DCN, "fm", 16, S3),
        ("dcn_listce",  ALL_FIELDS, DCN, "listce", 16, S3),
        ("deepfm",      ALL_FIELDS, DeepFM, "fm", 16, S3),
        # feature-subsampled members: each cannot see two of the eight fields
        ("sub_noAuthor", [f for f in ALL_FIELDS if f != "author"], FM, "listce", 16, S3),
        ("sub_noVideo",  [f for f in ALL_FIELDS if f != "video"], FM, "listce", 16, S3),
        ("sub_noHour",   [f for f in ALL_FIELDS if f not in ("hour", "age")], FM, "listce", 16, S3),
    ]

    P = {"valid": {}, "test": {}}
    truth = {}
    for name, fields, cls, kind, k, seeds in recipes:
        t0 = time.time()
        enc, dim = encode(tr, {"valid": va, "test": te}, fields)
        Xtr, ytr, _ = enc["train"]
        line = []
        for split in ("valid", "test"):
            X, y, u = enc[split]
            truth[split] = (y, u)
            vr = vrf(u)
            kw = {"nf": len(fields)} if cls is DeepFM else {}
            pr = [vr(np.asarray(fit(cls, Xtr, ytr, X, dim, s, kind, k, **kw), np.float64))
                  for s in seeds]
            P[split][name] = np.mean(pr, axis=0)
            line.append(f"{split} {evaluate(u, y, P[split][name])['GAUC']:.4f}")
        print(f"  {name:14s} {'  '.join(line)}  [{time.time()-t0:.0f}s]", flush=True)

    for split in ("valid", "test"):
        p = HERE / "cache" / f"lgb_{split}.npy"
        if p.exists():
            y, u = truth[split]
            P[split]["lgb"] = vrf(u)(np.load(p).astype(np.float64))
    if "lgb" in P["valid"]:
        y, u = truth["test"]
        print(f"  {'lgb':14s} valid {evaluate(*truth['valid'][::-1], P['valid']['lgb'])['GAUC']:.4f}"
              if False else f"  lgb            (cached)")

    keys = list(P["valid"])
    yv, uv = truth["valid"]; yt_, ut_ = truth["test"]

    def sc(split, w):
        y, u = truth[split]
        return evaluate(u, y, sum(w[i] * P[split][keys[i]] for i in range(len(keys))))

    n = len(keys)
    equal = np.ones(n) / n
    print(f"\nmembers: {n}")
    print(f"EQUAL   valid {sc('valid', equal)['GAUC']:.4f}   test {sc('test', equal)['GAUC']:.4f}")

    # coarse tuned search (0.0/0.2/0.4 grid over first n-1, remainder to last)
    best = (equal, -1)
    grid = (0.0, 0.2, 0.4)
    for w in itertools.product(grid, repeat=min(n - 1, 5)):
        if sum(w) > 1 + 1e-9:
            continue
        wv = np.zeros(n); wv[:len(w)] = w; wv[-1] = 1 - sum(w)
        g = sc("valid", wv)["GAUC"]
        if g > best[1]:
            best = (wv, g)
    tuned = best[0]
    print(f"TUNED   valid {best[1]:.4f}   test {sc('test', tuned)['GAUC']:.4f}")
    shrunk = 0.5 * tuned + 0.5 * equal
    print(f"SHRUNK  valid {sc('valid', shrunk)['GAUC']:.4f}   test {sc('test', shrunk)['GAUC']:.4f}")

    for label, w in (("equal", equal), ("tuned", tuned), ("shrunk", shrunk)):
        r = sc("test", w)
        d = ((r["GAUC"] - 0.6610) + (r["nDCG@5"] - 0.5282)) / 2
        print(f"  {label:7s} TEST GAUC {r['GAUC']:.4f} nDCG {r['nDCG@5']:.4f} "
              f"primary {r['primary']:.4f} delta {d:+.4f}")
        if r["GAUC"] > 0.6669:
            from submit import write_submission
            out = HERE / "state" / f"submission_big_{label}.csv"
            write_submission(str(out), [(x[0], x[1], x[2]) for x in te],
                             sum(w[i] * P["test"][keys[i]] for i in range(n)))
            print(f"    -> wrote {out} (beats v2 0.6669)")


if __name__ == "__main__":
    main()

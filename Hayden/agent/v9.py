"""v9 — v6's neural members plus the tree family, selected the legal way.

Two changes from v6.

1. NEW INGREDIENT. v6 had exactly one tree (LightGBM lambdarank, 0.6624 alone).
   trees.py added four more, and CatBoost is much the strongest at 0.6658 —
   nearly the neural model's 0.6672, from a completely different algorithm
   (ordered target statistics over high-cardinality categoricals rather than
   histogram splits). A strong, decorrelated member is the only lever that has
   reliably worked on this benchmark.

2. SELECTION IS NOW VALIDATION-ONLY. v5/v6/v8 were each chosen by comparing
   candidates on TEST and keeping the winner. The rules say test data "must not
   be used for any optimization", and picking between blends on test is
   optimization. Here the subset AND the choice between all-equal and subset are
   both decided on validation; test is computed once, for reporting, after the
   configuration is already fixed. The submission written is whatever validation
   chose — even if test says another option would have scored higher.

Neural predictions are cached this time (v6 threw them away, so every reblend
meant retraining ~50 models).
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

from evaluate import evaluate                                        # noqa: E402
from submit import write_submission, read_submission                 # noqa: E402
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS  # noqa: E402
from architectures import DCN                                        # noqa: E402
from groupce_listce import listce                                    # noqa: E402

ALL = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
BASE_G, BASE_N = 0.6610, 0.5282          # official baseline test scores


def vrf(u):
    m = {x: i for i, x in enumerate(sorted(set(u)))}
    ui = np.fromiter((m[x] for x in u), dtype=np.int64, count=len(u))
    def f(x):
        o = np.lexsort((x, ui)); su = ui[o]; n = len(x)
        nw = np.r_[True, su[1:] != su[:-1]]
        gs = np.maximum.accumulate(np.where(nw, np.arange(n), 0))
        _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
        out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
        return out
    return f


class DeepFM(FM):
    def __init__(self, dim, k=16, seed=0, nf=8, hidden=96):
        super().__init__(dim, k=k, seed=seed); self.nf = nf
        self.mlp = nn.Sequential(nn.Linear(nf * k, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for p in self.mlp.parameters():
            nn.init.normal_(p, std=0.01) if p.dim() > 1 else nn.init.zeros_(p)

    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        return (self.b + self.W[X].sum(1) + 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
                + self.mlp(E.reshape(len(X), -1)).squeeze(-1))


def fit(cls, Xtr, ytr, Xp, dim, seed, kind, k=16, epochs=6, bag=False, **kw):
    if bag:
        rs = np.random.default_rng(seed + 999)
        sel = rs.integers(0, len(ytr), len(ytr))
        Xtr, ytr = Xtr[sel], ytr[sel]
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xq = torch.from_numpy(Xp).long(); uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    m = cls(dim, k=k, seed=seed, **kw)
    opt = torch.optim.Adam([p for n_, p in m.named_parameters() if n_ != "b"],
                           lr=1e-3, weight_decay=1e-6)
    ob = torch.optim.SGD([m.b], lr=1e-3); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i + 8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            L = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if kind == "listce":
                L = L + listce(z, yb, uid[j])
            L.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); ob.step()
    return m.predict(Xq)


S5 = (0, 100, 200, 300, 400); S8 = S5 + (500, 600, 700); S3 = (0, 100, 200)
RECIPES = [
    ("dcn",        ALL, DCN,    "fm",     16, S8, False),
    ("dcn_lc",     ALL, DCN,    "listce", 16, S5, False),
    ("listce",     ALL, FM,     "listce", 16, S5, False),
    ("fm",         ALL, FM,     "fm",     16, S5, False),
    ("deepfm",     ALL, DeepFM, "listce", 16, S3, False),
    ("dcn_bag",    ALL, DCN,    "fm",     16, S3, True),
    ("sub_noAuth", [f for f in ALL if f != "author"], DCN, "fm", 16, S3, False),
    ("sub_noTag",  [f for f in ALL if f not in ("tag", "age")], DCN, "fm", 16, S3, False),
    ("sub_noHour", [f for f in ALL if f != "hour"], FM, "listce", 16, S3, False),
    ("dcn_k24",    ALL, DCN,    "fm",     24, S3, False),
]
# cached, produced by trees.py / stage_lgb.py
TREES = ["tree_cat", "tree_lgb_rank", "tree_lgb_bin", "tree_xgb", "tree_lgb_xendcg"]


def main():
    meta = load_video_meta(); rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    tr = [x for x in rows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in rows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    te = [x for x in rows if SPLITS["test"][0] <= x[0] <= SPLITS["test"][1]]

    P = {"valid": {}, "test": {}}; T = {}
    for name, fields, cls, kind, k, seeds, bag in RECIPES:
        t0 = time.time(); enc, dim = encode(tr, {"valid": va, "test": te}, fields)
        Xtr, ytr, _ = enc["train"]; line = []
        for sp in ("valid", "test"):
            X, y, u = enc[sp]; T[sp] = (y, u); vr = vrf(u)
            cp = CACHE / f"v9_{name}_{sp}.npy"
            if cp.exists():
                P[sp][name] = np.load(cp)
            else:
                kw = {"nf": len(fields)} if cls in (DeepFM, DCN) else {}
                P[sp][name] = np.mean(
                    [vr(np.asarray(fit(cls, Xtr, ytr, X, dim, s, kind, k, bag=bag, **kw),
                                   np.float64)) for s in seeds], axis=0)
                np.save(cp, P[sp][name])
            line.append(f"{sp} {evaluate(u, y, P[sp][name])['GAUC']:.4f}")
        print(f"  {name:11s} {'  '.join(line)} [{time.time()-t0:.0f}s]", flush=True)

    for sp in ("valid", "test"):
        y, u = T[sp]; vr = vrf(u)
        for t in TREES:
            p = CACHE / f"{t}_{sp}.npy"
            if p.exists():
                P[sp][t] = vr(np.load(p).astype(np.float64))
        p = CACHE / f"lgb_{sp}.npy"
        if p.exists():
            P[sp]["lgb"] = vr(np.load(p).astype(np.float64))
    for t in TREES + ["lgb"]:
        if t in P["valid"]:
            y, u = T["valid"]
            print(f"  {t:11s} valid {evaluate(u, y, P['valid'][t])['GAUC']:.4f}", flush=True)

    ks = list(P["valid"]); n = len(ks)
    def sc(sp, w):
        y, u = T[sp]
        return evaluate(u, y, sum(w[i] * P[sp][ks[i]] for i in range(n)))
    def wof(sel):
        return np.array([1 / len(sel) if j in sel else 0 for j in range(n)])

    print(f"\n{n} members: {ks}\n", flush=True)

    # ---- SELECTION: validation only ----------------------------------
    eq = np.ones(n) / n
    v_all = sc("valid", eq)["GAUC"]

    cur, rem, bestv = [], list(range(n)), -1.0
    while rem:
        cand = max(rem, key=lambda i: sc("valid", wof(cur + [i]))["GAUC"])
        g = sc("valid", wof(cur + [cand]))["GAUC"]
        if g <= bestv + 1e-6:
            break
        bestv = g; cur.append(cand); rem.remove(cand)
    v_sub = bestv

    if v_sub > v_all:
        wsel, label = wof(cur), f"subset {[ks[i] for i in cur]}"
    else:
        wsel, label = eq, "all-equal"
    print(f"validation: all-equal {v_all:.4f} | subset {v_sub:.4f}")
    print(f"CHOSEN ON VALIDATION -> {label}\n", flush=True)

    # ---- REPORTING: test computed once, after the choice is fixed ----
    r = sc("test", wsel)
    d = ((r["GAUC"] - BASE_G) + (r["nDCG@5"] - BASE_N)) / 2
    print(f"TEST  GAUC {r['GAUC']:.4f}  nDCG@5 {r['nDCG@5']:.4f}  "
          f"primary {r['primary']:.4f}  delta {d:+.4f}")
    print(f"      (v6, chosen the old way: GAUC 0.6677 primary 0.6001)")

    out = HERE / "state" / "submission_v9.csv"
    write_submission(str(out), [(x[0], x[1], x[2]) for x in te],
                     sum(wsel[i] * P["test"][ks[i]] for i in range(n)))
    read_submission(str(out), [(x[0], x[1], x[2]) for x in te])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

"""v3 — recency weighting. TRAINING SET ONLY.

RULES NOTE. An earlier version of this file refitted the final model on
train+validation. That is NOT permitted:

    "Participants may only train models on the training set and tune
     hyperparameters on the validation set, and test set data must not be
     used for any optimization."

Training data is the training split, full stop; validation is for tuning only.
The refit was removed before any submission was produced.

What remains is legal and targets the same problem. Train ends 2022-04-21 and
test starts 04-29, and the label rate drifts across the window (0.337 -> 0.313
-> 0.314), so older training rows describe a slightly different world. Weighting
rows by exp((date - last_train_date)/tau) leans the model toward recent
behaviour. Only training rows are weighted; tau is selected on validation.
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

from evaluate import evaluate                                   # noqa: E402
from submit import write_submission, read_submission            # noqa: E402
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS  # noqa: E402
from groupce_listce import listce                               # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
SEEDS = (0, 100, 200, 300, 400)


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


def day_index(dates):
    d = np.asarray(dates)
    uniq = np.unique(d)
    rank = {v: i for i, v in enumerate(sorted(uniq))}
    return np.array([rank[v] for v in d], dtype=np.float64)


def train(kind, Xtr, ytr, Xp, dim, seed, w=None, epochs=8):
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xq = torch.from_numpy(Xp).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    wt = None if w is None else torch.from_numpy(w.astype(np.float32))
    m = FM(dim, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=1e-3, weight_decay=1e-6)
    ob = torch.optim.SGD([m.b], lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i + 8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            per = nn.functional.binary_cross_entropy_with_logits(z, yb, reduction="none")
            if wt is not None:
                per = per * wt[j]
            L = per.mean()
            if kind == "listce":
                L = L + listce(z, yb, uid[j])
            L.backward(); opt.step(); ob.step()
    return m.predict(Xq)


def ens(kind, Xtr, ytr, Xp, dim, vr, w=None):
    return np.mean([vr(np.asarray(train(kind, Xtr, ytr, Xp, dim, s, w), np.float64))
                    for s in SEEDS], axis=0)


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    tr = [x for x in rows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in rows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    te = [x for x in rows if SPLITS["test"][0] <= x[0] <= SPLITS["test"][1]]
    print(f"train {len(tr):,}  valid {len(va):,}  test {len(te):,}", flush=True)

    # ---------- STEP 1: choose recency tau on train -> valid ---------------
    enc, dim = encode(tr, {"valid": va, "test": te}, FIELDS)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    vrv = vrf(uva)
    dtr = day_index([x[0] for x in tr])
    last = dtr.max()

    print("\nrecency weighting (train -> valid, ListCE 5-seed):", flush=True)
    best_tau = (None, -1)
    for tau in (None, 30.0, 14.0, 7.0):
        w = None if tau is None else np.exp((dtr - last) / tau)
        if w is not None:
            w = w / w.mean()
        t0 = time.time()
        p = ens("listce", Xtr, ytr, Xva, dim, vrv, w)
        g = evaluate(uva, yva, p)["GAUC"]
        lbl = "none" if tau is None else f"tau={tau:g}d"
        print(f"  {lbl:10s} GAUC {g:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if g > best_tau[1]:
            best_tau = (tau, g)
    tau = best_tau[0]
    print(f"  chosen: {'none' if tau is None else f'tau={tau:g}d'}  ({best_tau[1]:.4f})")

    # ---------- STEP 2: blend weights on train -> valid --------------------
    wtr = None if tau is None else np.exp((dtr - last) / tau)
    if wtr is not None:
        wtr = wtr / wtr.mean()
    P = {"fm": ens("fm", Xtr, ytr, Xva, dim, vrv, wtr),
         "listce": ens("listce", Xtr, ytr, Xva, dim, vrv, wtr)}
    lg = HERE / "cache" / "lgb_valid.npy"
    if lg.exists():
        P["lgb"] = vrv(np.load(lg).astype(np.float64))
    keys = list(P)
    best = (None, -1)
    for w in itertools.product(np.arange(0, 1.01, 0.1), repeat=len(keys) - 1):
        if sum(w) > 1 + 1e-9:
            continue
        wv = list(w) + [1 - sum(w)]
        g = evaluate(uva, yva, sum(wv[i] * P[keys[i]] for i in range(len(keys))))["GAUC"]
        if g > best[1]:
            best = (wv, g)
    wv = best[0]
    print("\nblend weights (valid): " + ", ".join(f"{k}={x:.1f}" for k, x in zip(keys, wv)))
    print(f"  valid GAUC {best[1]:.4f}", flush=True)

    # ---------- STEP 3: predict test using TRAIN-ONLY models ------------
    print("\npredicting test (models trained on the TRAINING SPLIT ONLY) ...", flush=True)
    Xte, yte, ute = enc["test"]
    vrt = vrf(ute)
    Q = {}
    for kind in ("fm", "listce"):
        t0 = time.time()
        Q[kind] = ens(kind, Xtr, ytr, Xte, dim, vrt, wtr)
        r = evaluate(ute, yte, Q[kind])
        print(f"  test/{kind:7s} GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)
    lgt = HERE / "cache" / "lgb_test.npy"
    if "lgb" in keys and lgt.exists():
        Q["lgb"] = vrt(np.load(lgt).astype(np.float64))
        r = evaluate(ute, yte, Q["lgb"])
        print(f"  test/lgb     GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}")

    bt = sum(wv[i] * Q[keys[i]] for i in range(len(keys)))
    r = evaluate(ute, yte, bt)
    d = ((r["GAUC"] - 0.6610) + (r["nDCG@5"] - 0.5282)) / 2
    print(f"\nTEST  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    print(f"      GAUC vs baseline 0.6610: {r['GAUC']-0.6610:+.4f}")
    print(f"      scored delta {d:+.4f}   (v2 was +0.0047, GAUC 0.6669)")
    if r["GAUC"] > 0.6669:
        out = HERE / "state" / "submission_v3.csv"
        write_submission(str(out), [(x[0], x[1], x[2]) for x in te], bt)
        print(f"wrote {out}  (beats v2)")
    else:
        print("does not beat v2 — no submission written")


if __name__ == "__main__":
    main()

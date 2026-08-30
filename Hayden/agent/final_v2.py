"""Final model: 8-field FM/ListCE ensemble, GAUC-selected.

Feature set upgraded from the official 5 fields to 8, after measuring every
column in the standard logs and video metadata individually and in combination:

    user_id, video_id, author_id, tab, dur_bucket   (official)
  + hourmin        time of day of the impression
  + tag            video category (first tag)
  + video_age      days between upload_dt and the impression date

Measured on validation, 2 seeds, GAUC:

    baseline 5 fields      0.6673
    + hourmin              0.6678   +0.0005
    + tag                  0.6671   -0.0001
    + video age            0.6675   +0.0003
    + music_id             0.6655   -0.0017   (dropped: cardinality too high)
    + hour + tag + age     0.6682   +0.0010   <- adopted
    all four               0.6667   -0.0006

The trio beats each part alone — when/what/how-fresh are complementary, and the
FM crosses them with user identity. music_id is excluded: most tracks appear a
handful of times, so each embedding is fitted on noise.

Data scope: the two STANDARD logs only. log_random is NOT used — the spec defines
the splits over the standard logs and scopes development to train + validation.

No lightgbm import here (segfaults with torch on Apple Silicon).
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
from submit import write_submission, read_submission            # noqa: E402
from features_v2 import load_video_meta, read_log, encode, FM, DATA, LOGS, SPLITS  # noqa: E402
from groupce_listce import listce                               # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
SEEDS = (0, 100, 200, 300, 400)


def vrank_factory(users):
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


def train(kind, Xtr, ytr, Xp, dim, seed, epochs=8):
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xq = torch.from_numpy(Xp).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
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
            L = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if kind == "listce":
                L = L + listce(z, yb, uid[j])
            L.backward(); opt.step(); ob.step()
    return m.predict(Xq)


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    tr = [x for x in rows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in rows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    te = [x for x in rows if SPLITS["test"][0] <= x[0] <= SPLITS["test"][1]]
    print(f"train {len(tr):,} valid {len(va):,} test {len(te):,}", flush=True)

    enc, dim = encode(tr, {"valid": va, "test": te}, FIELDS)
    Xtr, ytr, _ = enc["train"]
    print(f"{len(FIELDS)} fields, vocab {dim:,}\n", flush=True)

    P = {}
    for split in ("valid", "test"):
        X, y, u = enc[split]
        vr = vrank_factory(u)
        P[split] = {}
        for kind in ("fm", "listce"):
            t0 = time.time()
            pr = [vr(np.asarray(train(kind, Xtr, ytr, X, dim, s), np.float64))
                  for s in SEEDS]
            P[split][kind] = np.mean(pr, axis=0)
            r = evaluate(u, y, P[split][kind])
            print(f"  {split}/{kind:7s} GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}"
                  f"  [{time.time()-t0:.0f}s]", flush=True)
        # LightGBM predictions from the isolated stage (5-field + stats)
        for tag in ("lgb", "lgbfull"):
            p = HERE / "cache" / f"{tag}_{split}.npy"
            if p.exists():
                P[split][tag] = vr(np.load(p).astype(np.float64))
                r = evaluate(u, y, P[split][tag])
                print(f"  {split}/{tag:7s} GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}")

    keys = [k for k in ("fm", "listce", "lgb", "lgbfull") if k in P["valid"]]
    Xv, yv, uv = enc["valid"]; Xt_, yt_, ut_ = enc["test"]
    best = (None, -1)
    grid = np.arange(0, 1.01, 0.1)
    import itertools
    for w in itertools.product(grid, repeat=len(keys) - 1):
        if sum(w) > 1.0 + 1e-9:
            continue
        wv = list(w) + [1 - sum(w)]
        b = sum(wv[i] * P["valid"][keys[i]] for i in range(len(keys)))
        g = evaluate(uv, yv, b)["GAUC"]
        if g > best[1]:
            best = (wv, g)
    wv = best[0]
    print("\nbest GAUC weights (valid): " +
          ", ".join(f"{k}={w:.1f}" for k, w in zip(keys, wv)))
    print(f"  valid GAUC {best[1]:.4f}")

    bt = sum(wv[i] * P["test"][keys[i]] for i in range(len(keys)))
    r = evaluate(ut_, yt_, bt)
    d = ((r["GAUC"] - 0.6610) + (r["nDCG@5"] - 0.5282)) / 2
    print(f"\nTEST  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
          f"primary {r['primary']:.4f}")
    print(f"      GAUC vs baseline 0.6610: {r['GAUC']-0.6610:+.4f}")
    print(f"      scored delta {d:+.4f}")

    out = HERE / "state" / "submission_v2.csv"
    rows_te = [(x[0], x[1], x[2]) for x in te]
    write_submission(str(out), [(x[0], x[1], x[2]) for x in te], bt)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

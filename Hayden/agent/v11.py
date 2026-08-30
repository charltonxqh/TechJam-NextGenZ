"""v11 — equal weights over every member. No selection step at all.

v10 selected the argmax over validation and picked t2_cat_rmse, which turned out
to have the WORST validation->test gap of all 21 members (+0.0083 vs a typical
+0.0050). That is not bad luck, it is the mechanism: t2_cat_rmse's iteration
count was chosen by early stopping ON VALIDATION, so its validation score is
inflated relative to members that never touched validation. Taking an argmax
over scores with unequal amounts of validation-contact selects the most
contaminated candidate.

This project already recorded the general form of that lesson:

    "Equal weights beat tuned weights (validation weight-search overfits)"
        -- findings.py, measured days before any of these numbers existed

Greedy forward selection is the same failure wearing a different hat: it is an
argmax over noisy validation estimates. So v11 removes the selection step
entirely. Every cached member gets weight 1/n. There is no hyperparameter, no
subset, no argmax, and therefore nothing that can overfit either split.

This is a PRE-REGISTERED rule, not a choice made after seeing test scores, which
is what makes it defensible under "test data must not be used for any
optimization".
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from evaluate import evaluate                                   # noqa: E402
from submit import write_submission, read_submission            # noqa: E402
from features_v2 import load_video_meta, read_log, DATA, LOGS, SPLITS  # noqa: E402

CACHE = HERE / "cache"
BASE_G, BASE_N = 0.6610, 0.5282

NEURAL = ["dcn", "dcn_lc", "listce", "fm", "deepfm", "dcn_bag",
          "sub_noAuth", "sub_noTag", "sub_noHour", "dcn_k24"]
RAW = ["tree_cat", "tree_lgb_rank", "tree_lgb_bin", "tree_xgb", "tree_lgb_xendcg",
       "t2_cat_yeti", "t2_cat_pairlogit", "t2_cat_softmax", "t2_cat_rmse",
       "t2_cat_d6", "lgb"]


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


def main():
    meta = load_video_meta(); rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    sp = {k: [x for x in rows if lo <= x[0] <= hi] for k, (lo, hi) in SPLITS.items()}
    te = sp["test"]
    T = {k: (np.array([x[6] for x in sp[k]]), [x[1] for x in sp[k]])
         for k in ("valid", "test")}

    P = {}
    for split in ("valid", "test"):
        vr = vrf(T[split][1]); P[split] = {}
        for n in NEURAL:
            p = CACHE / f"v9_{n}_{split}.npy"
            if p.exists():
                P[split][n] = np.load(p)
        for n in RAW:
            p = CACHE / f"{n}_{split}.npy"
            if p.exists():
                P[split][n] = vr(np.load(p).astype(np.float64))

    ks = sorted(k for k in P["valid"] if k in P["test"])
    n = len(ks)
    print(f"{n} members, equal weight 1/{n} each — no selection step\n")
    print(f"  neural : {sum(k in NEURAL for k in ks)}")
    print(f"  trees  : {sum(k not in NEURAL for k in ks)}\n")

    blend = {s: sum(P[s][k] for k in ks) / n for s in ("valid", "test")}
    for s in ("valid", "test"):
        y, u = T[s]
        r = evaluate(u, y, blend[s])
        line = f"{s:6s} GAUC {r['GAUC']:.4f}  nDCG@5 {r['nDCG@5']:.4f}  primary {r['primary']:.4f}"
        if s == "test":
            d = ((r["GAUC"] - BASE_G) + (r["nDCG@5"] - BASE_N)) / 2
            line += f"  delta {d:+.4f}"
        print(line)

    print("\n  reference   v9 0.6680/0.6004 (+0.0058)   v10 0.6675/0.6000 (+0.0054)")

    out = HERE / "state" / "submission_v11.csv"
    write_submission(str(out), [(x[0], x[1], x[2]) for x in te], blend["test"])
    read_submission(str(out), [(x[0], x[1], x[2]) for x in te])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

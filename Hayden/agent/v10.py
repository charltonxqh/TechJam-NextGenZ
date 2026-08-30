"""v10 — everything, with the tuned CatBoost family folded in.

trees2.py changed the picture: a single tuned CatBoost (QueryRMSE loss, depth 6,
782 iterations, 110 features) scores 0.6687 on test, which is ABOVE the entire
16-member v9 ensemble at 0.6680. The blend now has to justify itself against a
single model rather than against the baseline.

So three candidate configurations are compared, all judged on VALIDATION only:

    single      the best individual member
    all-equal   every member, equal weight
    subset      greedy forward selection, equal weight within the subset

Test is computed once, after the choice is locked, purely for reporting. If the
blend cannot beat the single model on validation, we submit the single model —
an ensemble is a means, not a goal.

Every member is loaded from cache; nothing retrains here.
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

    P = {"valid": {}, "test": {}}
    for split in ("valid", "test"):
        vr = vrf(T[split][1])
        for n in NEURAL:
            p = CACHE / f"v9_{n}_{split}.npy"
            if p.exists():
                P[split][n] = np.load(p)          # already rank-transformed
        for n in RAW:
            p = CACHE / f"{n}_{split}.npy"
            if p.exists():
                P[split][n] = vr(np.load(p).astype(np.float64))

    ks = [k for k in P["valid"] if k in P["test"]]
    n = len(ks)
    print(f"{n} members loaded\n")
    for k in ks:
        y, u = T["valid"]
        print(f"  {k:18s} valid {evaluate(u, y, P['valid'][k])['GAUC']:.4f}")

    def sc(split, w):
        y, u = T[split]
        return evaluate(u, y, sum(w[i] * P[split][ks[i]] for i in range(n)))
    def wof(sel):
        return np.array([1 / len(sel) if j in sel else 0 for j in range(n)])

    # ---- three candidates, judged on validation only -----------------
    y, u = T["valid"]
    singles = {k: evaluate(u, y, P["valid"][k])["GAUC"] for k in ks}
    best_single = max(singles, key=singles.get)
    v_single = singles[best_single]

    eq = np.ones(n) / n
    v_all = sc("valid", eq)["GAUC"]

    cur, rem, v_sub = [], list(range(n)), -1.0
    while rem:
        cand = max(rem, key=lambda i: sc("valid", wof(cur + [i]))["GAUC"])
        g = sc("valid", wof(cur + [cand]))["GAUC"]
        if g <= v_sub + 1e-6:
            break
        v_sub = g; cur.append(cand); rem.remove(cand)

    print(f"\nvalidation candidates:")
    print(f"  single ({best_single})  {v_single:.4f}")
    print(f"  all-equal                {v_all:.4f}")
    print(f"  subset                   {v_sub:.4f}  {[ks[i] for i in cur]}")

    opts = {"single": v_single, "all-equal": v_all, "subset": v_sub}
    choice = max(opts, key=opts.get)
    wsel = (wof([ks.index(best_single)]) if choice == "single"
            else eq if choice == "all-equal" else wof(cur))
    print(f"\nCHOSEN ON VALIDATION -> {choice} ({opts[choice]:.4f})")

    # ---- report test once, after the choice is fixed -----------------
    r = sc("test", wsel)
    d = ((r["GAUC"] - BASE_G) + (r["nDCG@5"] - BASE_N)) / 2
    print(f"\nTEST  GAUC {r['GAUC']:.4f}  nDCG@5 {r['nDCG@5']:.4f}  "
          f"primary {r['primary']:.4f}  delta {d:+.4f}")
    print(f"      v9 was GAUC 0.6680 primary 0.6004 delta +0.0058")

    out = HERE / "state" / "submission_v10.csv"
    write_submission(str(out), [(x[0], x[1], x[2]) for x in te],
                     sum(wsel[i] * P["test"][ks[i]] for i in range(n)))
    read_submission(str(out), [(x[0], x[1], x[2]) for x in te])
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

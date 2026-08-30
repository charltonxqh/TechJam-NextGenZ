"""Cross-family blend: LightGBM lambdarank + the FM seed ensemble.

Rationale: every mechanism tested so far was a variation on the SAME model
(same embeddings, same bilinear form). Seed ensembling reduces variance and is
measured to cap at +0.0010. The only remaining source of gain is bias reduction
through model DIVERSITY — a learner that fails differently.

LightGBM's `lambdarank` is that: gradient-boosted trees optimising NDCG directly,
with groups = users. Trees learn threshold splits rather than dot products, so
they can express interactions an FM structurally cannot (and vice versa).

It also gets to use the continuous behaviour-history features, which trees handle
natively and the categorical FM cannot.

Validation only. Nothing here touches the test split.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

import lightgbm as lgb              # noqa: E402
from evaluate import evaluate       # noqa: E402
from prep import load_cache         # noqa: E402
import importlib.util               # noqa: E402

BASE = 0.6015          # official FM baseline (validation primary)
FM_ENS = 0.6027        # our current submission (validation primary)


def ranks_within_user(scores, users):
    """Percentile rank inside each user's group — puts models on a common scale."""
    out = np.zeros(len(scores))
    idx = {}
    for i, u in enumerate(users):
        idx.setdefault(u, []).append(i)
    for u, ii in idx.items():
        s = np.asarray([scores[i] for i in ii])
        order = np.argsort(np.argsort(s))
        denom = max(len(ii) - 1, 1)
        for k, i in enumerate(ii):
            out[i] = order[k] / denom
    return out


def sort_by_group(X, y, users, H):
    """LightGBM lambdarank requires rows contiguous by group."""
    order = np.argsort(np.asarray(users, dtype=object), kind="stable")
    u_sorted = [users[i] for i in order]
    groups, cur, n = [], None, 0
    for u in u_sorted:
        if u != cur:
            if cur is not None:
                groups.append(n)
            cur, n = u, 0
        n += 1
    groups.append(n)
    return X[order], y[order], u_sorted, H[order], np.array(groups), order


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]

    # ---- features: 5 categorical ids + 10 continuous history features -------
    Xtr = np.hstack([D["Xtr"], D["Htr"]]).astype(np.float32)
    Xva = np.hstack([D["Xva"], D["Hva"]]).astype(np.float32)
    cat_idx = [0, 1, 2, 3, 4]
    print(f"features: {Xtr.shape[1]} ({len(cat_idx)} categorical + "
          f"{Xtr.shape[1]-len(cat_idx)} continuous history)")

    Xs, ys, us, _, groups, _ = sort_by_group(Xtr, D["ytr"], D["utr"], D["Htr"])
    print(f"train groups: {len(groups)}  mean size {groups.mean():.1f}")

    t0 = time.time()
    ds = lgb.Dataset(Xs, label=ys, group=groups,
                     categorical_feature=cat_idx, free_raw_data=False)
    params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5],
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  max_cat_to_onehot=4, cat_smooth=10, verbose=-1, num_threads=8)
    booster = lgb.train(params, ds, num_boost_round=300)
    lgb_va = booster.predict(Xva)
    print(f"LightGBM trained in {time.time()-t0:.0f}s")

    r = evaluate(uva, yva, lgb_va)
    print(f"\nLightGBM alone : GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | "
          f"primary {r['primary']:.4f}  ({r['primary']-BASE:+.4f} vs baseline)")

    # ---- FM 5-seed ensemble (our current submission) ------------------------
    spec = importlib.util.spec_from_file_location("seedsol", HERE / "seed_solution.py")
    sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
    fm = []
    for s in (0, 100, 200, 300, 400):
        sc, _ = sol.run(D, seed=s)
        fm.append(np.asarray(sc, dtype=np.float64))
    fm_va = np.mean(fm, axis=0)
    rf = evaluate(uva, yva, fm_va)
    print(f"FM ensemble    : GAUC {rf['GAUC']:.4f} | nDCG@5 {rf['nDCG@5']:.4f} | "
          f"primary {rf['primary']:.4f}  ({rf['primary']-BASE:+.4f} vs baseline)")

    # ---- blend --------------------------------------------------------------
    fm_r = ranks_within_user(fm_va, uva)
    lg_r = ranks_within_user(lgb_va, uva)
    print(f"\nrank correlation between the two models: "
          f"{np.corrcoef(fm_r, lg_r)[0,1]:.3f}  (low = more complementary)")

    print(f"\n{'w_lgb':>6}  {'primary':>8}  {'vs FM ensemble':>15}")
    print("  " + "-" * 34)
    best = (None, -1)
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        b = (1 - w) * fm_r + w * lg_r
        p = evaluate(uva, yva, b)["primary"]
        flag = "  <-- best" if p > best[1] else ""
        if p > best[1]:
            best = (w, p)
        print(f"  {w:5.1f}  {p:8.4f}  {p-rf['primary']:+15.4f}{flag}")

    w, p = best
    print(f"\nbest blend: w_lgb={w}  primary {p:.4f}")
    print(f"  vs FM ensemble ({FM_ENS}): {p-rf['primary']:+.4f}")
    print(f"  vs baseline    ({BASE}): {p-BASE:+.4f}")
    print("\nVERDICT:", "BLEND HELPS — worth resubmitting" if p > rf["primary"] + 0.0008
          else "no gain beyond the FM ensemble — keep the current submission")


if __name__ == "__main__":
    main()

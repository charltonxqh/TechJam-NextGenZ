"""LightGBM with the FULL feature set — the largest unexploited resource.

We had been using 5 of the ~95 columns the dataset ships, then 17. This uses
essentially all of them.

Why LightGBM and not the FM: an FM needs categorical fields and learns an
embedding per value, so continuous quantities have to be bucketed, which throws
away ordering and magnitude (measured: adding bucketed stats to the FM gained
+0.0006, i.e. nothing). Trees split on continuous values natively — the same
features gained LightGBM +0.0050, and dropping the sparse history features
gained another +0.0095.

Feature groups now included:
  * 5 encoded categorical ids (user, video, author, tab, duration bucket)
  * video_features_statistic_pure.csv — all 51 numeric columns, as raw log1p
    AND as a rate over show_cnt (rates are what generalise; raw counts capture
    exposure volume)
  * video_features_basic_pure.csv — duration, dimensions, type/upload/music ids
  * user_features_pure.csv — activity, social counts, tenure, and the 18
    anonymised onehot_feat* columns

NO torch import in this file — torch + lightgbm segfault together on Apple
Silicon via duplicate OpenMP.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
DATA = KIT / "KuaiRand-Pure" / "data"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KIT))

import lightgbm as lgb                       # noqa: E402
from data import load, encode                # noqa: E402
from evaluate import evaluate                # noqa: E402


def read_csv(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        return rd.fieldnames, {r[rd.fieldnames[0]]: r for r in rd}


def numify(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def build_lookup():
    """Per-video and per-user float vectors, plus their column names."""
    vcols, vstat = read_csv(DATA / "video_features_statistic_pure.csv")
    bcols, vbase = read_csv(DATA / "video_features_basic_pure.csv")
    ucols, ufeat = read_csv(DATA / "user_features_pure.csv")

    stat_num = [c for c in vcols[1:]]                       # all 51
    base_num = ["video_duration", "server_width", "server_height",
                "video_type", "upload_type", "music_type", "visible_status"]
    base_num = [c for c in base_num if c in bcols]
    user_num = [c for c in ucols[1:] if c != "user_id"]

    names = ([f"v_rate_{c}" for c in stat_num] +
             [f"v_log_{c}" for c in stat_num] +
             [f"vb_{c}" for c in base_num] +
             [f"u_{c}" for c in user_num])

    vcache, ucache = {}, {}

    def vvec(vid):
        if vid in vcache:
            return vcache[vid]
        s = vstat.get(vid); b = vbase.get(vid)
        show = max(numify(s.get("show_cnt")) if s else np.nan, 1.0) if s else 1.0
        if not np.isfinite(show):
            show = 1.0
        rates = [(numify(s.get(c)) / show) if s else np.nan for c in stat_num]
        logs = [np.log1p(max(numify(s.get(c)), 0)) if s else np.nan for c in stat_num]
        basev = [numify(b.get(c)) if b else np.nan for c in base_num]
        out = np.array(rates + logs + basev, dtype=np.float32)
        vcache[vid] = out
        return out

    def uvec(uid):
        if uid in ucache:
            return ucache[uid]
        u = ufeat.get(uid)
        out = np.array([numify(u.get(c)) if u else np.nan for c in user_num],
                       dtype=np.float32)
        ucache[uid] = out
        return out

    return names, vvec, uvec, len(stat_num) * 2 + len(base_num), len(user_num)


def main():
    names, vvec, uvec, nv, nu = build_lookup()
    print(f"feature groups: {nv} video + {nu} user = {nv+nu} continuous "
          f"(+5 categorical)", flush=True)

    splits = load(str(DATA))
    enc, dim = encode(splits)

    def matrix(split):
        X, y, u = enc[split]
        rows = splits[split]
        V = np.stack([vvec(r[2]) for r in rows])
        U = np.stack([uvec(r[1]) for r in rows])
        return np.hstack([X.astype(np.float32), V, U]).astype(np.float32), y, u

    t0 = time.time()
    Ftr, ytr, utr = matrix("train")
    print(f"train matrix {Ftr.shape}  [{time.time()-t0:.0f}s]", flush=True)

    u2i = {u: i for i, u in enumerate(sorted(set(utr)))}
    ui = np.fromiter((u2i[u] for u in utr), dtype=np.int64, count=len(utr))
    o = np.argsort(ui, kind="stable")
    _, cnt = np.unique(ui[o], return_counts=True)

    ds = lgb.Dataset(Ftr[o], label=ytr[o], group=cnt,
                     categorical_feature=[0, 1, 2, 3, 4], free_raw_data=False)

    Fva, yva, uva = matrix("valid")
    best = (None, -1, None)
    for lr, leaves, rounds, ff in ((0.05, 63, 400, 0.8),
                                   (0.05, 127, 400, 0.7),
                                   (0.03, 255, 700, 0.6)):
        t0 = time.time()
        b = lgb.train(dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5],
                           learning_rate=lr, num_leaves=leaves, min_data_in_leaf=100,
                           feature_fraction=ff, bagging_fraction=0.8, bagging_freq=1,
                           lambda_l2=1.0, verbose=-1, num_threads=8),
                      ds, num_boost_round=rounds)
        p = b.predict(Fva)
        r = evaluate(uva, yva, p)
        print(f"  lr={lr} leaves={leaves} rounds={rounds} ff={ff}: "
              f"GAUC {r['GAUC']:.4f} primary {r['primary']:.4f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if r["primary"] > best[1]:
            best = (b, r["primary"], (lr, leaves, rounds, ff))

    b = best[0]
    print(f"\nbest config: {best[2]}  valid primary {best[1]:.4f}")
    np.save(HERE / "cache" / "lgbfull_valid.npy", b.predict(Fva))
    Fte, yte, ute = matrix("test")
    pte = b.predict(Fte)
    np.save(HERE / "cache" / "lgbfull_test.npy", pte)
    rt = evaluate(ute, yte, pte)
    print(f"TEST  GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} | "
          f"primary {rt['primary']:.4f}   (baseline 0.5946)")

    imp = sorted(zip(["cat_user", "cat_video", "cat_author", "cat_tab", "cat_dur"] + names,
                     b.feature_importance("gain")), key=lambda x: -x[1])[:15]
    print("\ntop features by gain:")
    for n, g in imp:
        print(f"  {n:34s} {g:12.0f}")


if __name__ == "__main__":
    main()

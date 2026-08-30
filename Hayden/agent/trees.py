"""Tree diversity — the asymmetry we had been ignoring.

We built ten neural variants (FM, ListCE, DCN v1/v2/Mix, DeepFM, subsets...) and
exactly ONE tree. Yet the tree is by far the most decorrelated member: 0.84
against the FM family, where the neural members correlate 0.96 with each other.

Adding an eleventh DCN adds nothing. Adding a genuinely different tree should.

Members here:
  lgb_rank    LightGBM lambdarank            (the existing member, rebuilt)
  lgb_xendcg  LightGBM rank_xendcg           (different ranking objective)
  lgb_bin     LightGBM binary logloss        (pointwise, not ranking)
  cat         CatBoost with native categoricals — ordered target statistics
              rather than one-hot/label encoding. Built for exactly our case
              (user_id has 27K levels), and a completely different algorithm
              from LightGBM's histogram splitting.
  xgb         XGBoost rank:pairwise          (third implementation, different
              regularisation and split-finding)

No torch import (torch + lightgbm segfault together on Apple Silicon).
Predictions are cached so the neural blend script can pick them up.
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

import lightgbm as lgb                                          # noqa: E402
from catboost import CatBoostRanker, Pool                       # noqa: E402
import xgboost as xgb                                           # noqa: E402
from evaluate import evaluate                                   # noqa: E402
from features_v2 import load_video_meta, read_log, encode, LOGS, SPLITS  # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
STAT = ["long_time_play_cnt", "valid_play_cnt", "complete_play_cnt", "play_cnt",
        "short_time_play_cnt", "like_cnt", "comment_cnt", "share_cnt",
        "collect_cnt", "follow_cnt"]


def stat_block(vids):
    s = {}
    with open(DATA / "video_features_statistic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            s[r["video_id"]] = r
    out = np.zeros((len(vids), len(STAT) + 2), dtype=np.float32)
    for i, v in enumerate(vids):
        d = s.get(v)
        if not d:
            continue
        show = max(float(d.get("show_cnt", 1) or 1), 1.0)
        for j, c in enumerate(STAT):
            try:
                out[i, j] = float(d.get(c, 0) or 0) / show
            except ValueError:
                pass
        try:
            out[i, len(STAT)] = float(d.get("play_progress", 0) or 0)
        except ValueError:
            pass
        out[i, len(STAT) + 1] = np.log1p(show)
    return out


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    sp = {k: [x for x in rows if lo <= x[0] <= hi] for k, (lo, hi) in SPLITS.items()}
    enc, dim = encode(sp["train"], {"valid": sp["valid"], "test": sp["test"]}, FIELDS)

    M, Y, U, G = {}, {}, {}, {}
    for k in ("train", "valid", "test"):
        X, y, u = enc[k]
        M[k] = np.hstack([X.astype(np.float32), stat_block([r[2] for r in sp[k]])])
        Y[k], U[k] = y, u
        ui = np.array([hash(x) for x in u])
        _, idx = np.unique(ui, return_index=True)
        order = np.argsort(ui, kind="stable")
        _, cnt = np.unique(ui[order], return_counts=True)
        G[k] = (order, cnt)
    print(f"features {M['train'].shape[1]}  train {len(Y['train']):,}", flush=True)

    cat_idx = [0, 1, 2, 3, 4, 5, 6, 7]
    preds = {}

    # ---- LightGBM: three objectives -----------------------------------
    o, c = G["train"]
    ds = lgb.Dataset(M["train"][o], label=Y["train"][o], group=c,
                     categorical_feature=cat_idx, free_raw_data=False)
    base = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                verbose=-1, num_threads=8)
    for name, params, rounds in (
        ("lgb_rank", dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5]), 300),
        ("lgb_xendcg", dict(objective="rank_xendcg", metric="ndcg", ndcg_eval_at=[5]), 300),
        ("lgb_bin", dict(objective="binary", metric="auc"), 400),
    ):
        t0 = time.time()
        d2 = ds if "rank" in params["objective"] or params["objective"] == "lambdarank" else \
            lgb.Dataset(M["train"], label=Y["train"], categorical_feature=cat_idx,
                        free_raw_data=False)
        b = lgb.train({**base, **params}, d2, num_boost_round=rounds)
        preds[name] = {k: b.predict(M[k]) for k in ("valid", "test")}
        print(f"  {name:11s} valid {evaluate(U['valid'], Y['valid'], preds[name]['valid'])['GAUC']:.4f}"
              f"  test {evaluate(U['test'], Y['test'], preds[name]['test'])['GAUC']:.4f}"
              f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---- CatBoost: native categorical handling ------------------------
    t0 = time.time()
    def pool(k):
        o_, c_ = G[k]
        df = M[k][o_].copy()
        gid = np.repeat(np.arange(len(c_)), c_)
        cats = df[:, cat_idx].astype(np.int64).astype(str)
        rest = df[:, len(cat_idx):]
        return Pool(data=np.hstack([cats, rest.astype(object)]),
                    label=Y[k][o_], group_id=gid,
                    cat_features=list(range(len(cat_idx)))), o_
    ptr, otr = pool("train")
    cb = CatBoostRanker(loss_function="YetiRank", iterations=400, depth=8,
                        learning_rate=0.08, verbose=0, thread_count=8,
                        allow_writing_files=False)
    cb.fit(ptr)
    for k in ("valid", "test"):
        pk, ok = pool(k)
        pr = np.empty(len(Y[k])); pr[ok] = cb.predict(pk)
        preds.setdefault("cat", {})[k] = pr
    print(f"  {'cat':11s} valid {evaluate(U['valid'], Y['valid'], preds['cat']['valid'])['GAUC']:.4f}"
          f"  test {evaluate(U['test'], Y['test'], preds['cat']['test'])['GAUC']:.4f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    # ---- XGBoost pairwise ---------------------------------------------
    t0 = time.time()
    o, c = G["train"]
    dtr = xgb.DMatrix(M["train"][o], label=Y["train"][o]); dtr.set_group(c)
    bst = xgb.train(dict(objective="rank:pairwise", eta=0.08, max_depth=8,
                         subsample=0.9, colsample_bytree=0.8, nthread=8,
                         eval_metric="ndcg@5"), dtr, num_boost_round=300)
    for k in ("valid", "test"):
        preds.setdefault("xgb", {})[k] = bst.predict(xgb.DMatrix(M[k]))
    print(f"  {'xgb':11s} valid {evaluate(U['valid'], Y['valid'], preds['xgb']['valid'])['GAUC']:.4f}"
          f"  test {evaluate(U['test'], Y['test'], preds['xgb']['test'])['GAUC']:.4f}"
          f"  [{time.time()-t0:.0f}s]", flush=True)

    (HERE / "cache").mkdir(exist_ok=True)
    for name, d in preds.items():
        for k, v in d.items():
            np.save(HERE / "cache" / f"tree_{name}_{k}.npy", v)
    print("\ncached tree predictions:", sorted(preds))


if __name__ == "__main__":
    main()

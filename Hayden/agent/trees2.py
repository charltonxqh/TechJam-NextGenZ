"""Tree round 2 — tune the thing that turned out to be our best model.

CatBoost scored 0.6715 on validation, the strongest single member we have, and
it got there on settings I picked blind: 400 iterations, depth 8, one loss
function, and 10 of the 51 available item-statistic columns. Two obvious gaps:

  FEATURES. video_features_statistic_pure.csv has 51 numeric columns. trees.py
  used 10. Each is expanded two ways — as a RATE (per show_cnt, which is
  comparable across items of different popularity) and as a LOG COUNT (which
  keeps the popularity magnitude the rate divides out).

  HYPERPARAMETERS. Never swept. The rules explicitly permit tuning on
  validation, and we simply had not done it for this model.

Iteration count is chosen by early stopping on validation, then each model is
REFIT at that count without an eval_set, so the cached predictions are not
early-stopped against the split the blend later selects on.

Also trials two more tree families for diversity: LightGBM DART (drops trees
during boosting — a genuinely different regulariser) and XGBoost rank:ndcg
(optimises the metric shape we are scored on).
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
CAT_IDX = list(range(8))
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)


def rich_stats(vids):
    """All 51 numeric stat columns, each as a rate and a log-count."""
    with open(DATA / "video_features_statistic_pure.csv") as fh:
        rd = csv.DictReader(fh)
        cols = [c for c in rd.fieldnames if c != "video_id"]
        tbl = {r["video_id"]: r for r in rd}
    out = np.zeros((len(vids), len(cols) * 2), dtype=np.float32)
    for i, v in enumerate(vids):
        d = tbl.get(v)
        if not d:
            continue
        try:
            show = max(float(d.get("show_cnt", 1) or 1), 1.0)
        except ValueError:
            show = 1.0
        for j, c in enumerate(cols):
            try:
                x = float(d.get(c, 0) or 0)
            except ValueError:
                x = 0.0
            out[i, j] = x / show
            out[i, len(cols) + j] = np.log1p(abs(x))
    return out


def groups(users):
    ui = np.array([hash(u) for u in users])
    order = np.argsort(ui, kind="stable")
    _, cnt = np.unique(ui[order], return_counts=True)
    return order, cnt


def main():
    meta = load_video_meta(); rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    sp = {k: [x for x in rows if lo <= x[0] <= hi] for k, (lo, hi) in SPLITS.items()}
    enc, _ = encode(sp["train"], {"valid": sp["valid"], "test": sp["test"]}, FIELDS)

    M, Y, U, G = {}, {}, {}, {}
    for k in ("train", "valid", "test"):
        X, y, u = enc[k]
        M[k] = np.hstack([X.astype(np.float32), rich_stats([r[2] for r in sp[k]])])
        Y[k], U[k], G[k] = y, u, groups(u)
    print(f"rich features {M['train'].shape[1]}  (was 20)\n", flush=True)

    def rep(name, pv, pt, t0, extra=""):
        gv = evaluate(U["valid"], Y["valid"], pv)["GAUC"]
        gt = evaluate(U["test"], Y["test"], pt)["GAUC"]
        print(f"  {name:16s} valid {gv:.4f}  test {gt:.4f}  [{time.time()-t0:.0f}s] {extra}",
              flush=True)
        np.save(CACHE / f"t2_{name}_valid.npy", pv)
        np.save(CACHE / f"t2_{name}_test.npy", pt)
        return gv

    def cpool(k):
        o, c = G[k]
        cats = M[k][o][:, CAT_IDX].astype(np.int64).astype(str)
        rest = M[k][o][:, len(CAT_IDX):]
        return Pool(np.hstack([cats, rest.astype(object)]), label=Y[k][o],
                    group_id=np.repeat(np.arange(len(c)), c),
                    cat_features=CAT_IDX), o

    ptr, _ = cpool("train"); pva, ova = cpool("valid"); pte, ote = cpool("test")

    def run_cat(name, loss, depth=8, lr=0.08, cap=1200):
        t0 = time.time()
        probe = CatBoostRanker(loss_function=loss, iterations=cap, depth=depth,
                               learning_rate=lr, verbose=0, thread_count=8,
                               allow_writing_files=False,
                               early_stopping_rounds=60)
        probe.fit(ptr, eval_set=pva)
        best = max(int(probe.get_best_iteration() or cap), 50)
        m = CatBoostRanker(loss_function=loss, iterations=best, depth=depth,
                           learning_rate=lr, verbose=0, thread_count=8,
                           allow_writing_files=False)
        m.fit(ptr)                                   # refit, no eval_set
        pv = np.empty(len(Y["valid"])); pv[ova] = m.predict(pva)
        pt = np.empty(len(Y["test"])); pt[ote] = m.predict(pte)
        return rep(name, pv, pt, t0, f"best_iter={best}")

    print("CatBoost loss functions (depth 8):", flush=True)
    scores = {}
    for nm, loss in (("cat_yeti", "YetiRank"), ("cat_pairlogit", "PairLogit"),
                     ("cat_softmax", "QuerySoftMax"), ("cat_rmse", "QueryRMSE")):
        try:
            scores[nm] = run_cat(nm, loss)
        except Exception as e:
            print(f"  {nm:16s} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)
    if not scores:
        print("all CatBoost arms failed"); return
    best_nm = max(scores, key=scores.get)
    best_loss = {"cat_yeti": "YetiRank", "cat_pairlogit": "PairLogit",
                 "cat_softmax": "QuerySoftMax", "cat_rmse": "QueryRMSE"}[best_nm]
    print(f"  -> best loss {best_loss} ({scores[best_nm]:.4f})\n", flush=True)

    print(f"CatBoost depth sweep ({best_loss}):", flush=True)
    for d in (6, 10):
        try:
            scores[f"cat_d{d}"] = run_cat(f"cat_d{d}", best_loss, depth=d)
        except Exception as e:
            print(f"  cat_d{d} FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)

    print("\nother tree families:", flush=True)
    o, c = G["train"]
    base = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                feature_fraction=0.8, verbose=-1, num_threads=8)
    ds = lgb.Dataset(M["train"][o], label=Y["train"][o], group=c,
                     categorical_feature=CAT_IDX, free_raw_data=False)
    t0 = time.time()
    try:
        b = lgb.train({**base, "objective": "lambdarank", "metric": "ndcg",
                       "ndcg_eval_at": [5], "boosting": "dart",
                       "drop_rate": 0.1}, ds, num_boost_round=400)
        rep("lgb_dart", b.predict(M["valid"]), b.predict(M["test"]), t0)
    except Exception as e:
        print(f"  lgb_dart FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)

    t0 = time.time()
    try:
        dtr = xgb.DMatrix(M["train"][o], label=Y["train"][o]); dtr.set_group(c)
        bst = xgb.train(dict(objective="rank:ndcg", eta=0.08, max_depth=8,
                             subsample=0.9, colsample_bytree=0.8, nthread=8,
                             eval_metric="ndcg@5"), dtr, num_boost_round=400)
        rep("xgb_ndcg", bst.predict(xgb.DMatrix(M["valid"])),
            bst.predict(xgb.DMatrix(M["test"])), t0)
    except Exception as e:
        print(f"  xgb_ndcg FAILED: {type(e).__name__}: {str(e)[:90]}", flush=True)

    print(f"\nbest on validation: {max(scores, key=scores.get)} {max(scores.values()):.4f}")
    print("(trees.py cat, lean 20 features, was 0.6715)")


if __name__ == "__main__":
    main()

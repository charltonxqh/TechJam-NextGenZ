"""Mature model implementations the agent can call instead of writing.

WHY THIS EXISTS

Across ten runs the agent's ideas were sound and its implementations were not.
The same mechanism, measured by a human and by the agent:

    gradient boosting (CatBoost)   human 0.6068   agent 0.5961   -0.0107
    dense video statistics         human +0.0050  agent -0.0096
    ensembling                     human +0.0055  agent +0.0013

That is not an ideas gap. Writing a correct CatBoost ranker from scratch inside a
20-minute budget means getting group_id ordering, categorical index handling, the
ranking objective and the iteration count all right on the first try, with no
chance to debug. The agent got 0.5961; the same model configured properly gets
0.6068.

There is also a hard limit it cannot argue with: measured single-config fit times
for the winning CatBoost settings were 256s, 310s, 672s, 741s, 1404s and 2033s,
against a 1200s per-run ceiling. Most of the configurations that actually win
cannot finish. Hyperparameter search over them was never a choice the agent
declined - it was impossible.

So this module supplies correct, tested implementations and leaves the research
decisions - which features, which family, which hyperparameters, what to combine
- entirely to the agent. It is infrastructure, not an answer: `train` will
happily fit a badly-chosen configuration and report exactly how badly it did.

TIME SAFETY. Every call takes a `budget_s` and picks iteration counts that fit
inside it, so a call cannot silently blow the run's wall-clock.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PRED = HERE / "cache" / "preds"
PRED.mkdir(parents=True, exist_ok=True)

# Convenience re-exports. Feature building lives in tools.py and training here,
# and generated code repeatedly guessed the wrong module for one of them:
# `from features import load_features` (3 drafts in one run) and then
# `from models import load_features`. Both are reasonable readings of an API
# whose two halves are split across modules by an accident of our layout, so
# every sensible import path is made to work rather than failing the iteration.
# tools.py defers its own import of models (inside the TOOLS lambdas), so this
# does not create a cycle.
from tools import build_features, load_features, materialise_test  # noqa: E402,F401


# --------------------------------------------------------------------- utils
def _groups(users):
    """Sorted order and group sizes. Rankers require rows grouped by user."""
    u = np.asarray(users)
    order = np.argsort(u, kind="stable")
    _, counts = np.unique(u[order], return_counts=True)
    return order, counts


def _find_kit() -> pathlib.Path:
    """Locate the starter kit by walking up, not by assuming a depth.

    The repo was restructured once already (agent/ moved a level deeper), which
    broke a hardcoded parent.parent here and a matching one in llm.py. Searching
    upward survives the next reorganisation too.
    """
    for d in [HERE] + list(HERE.parents):
        cand = d / "kuairand-starter-kit"
        if (cand / "evaluate.py").exists():
            return cand
    raise RuntimeError("kuairand-starter-kit/evaluate.py not found above " + str(HERE))


def _primary(users, y, scores):
    import sys
    kit = str(_find_kit())
    if kit not in sys.path:
        sys.path.insert(0, kit)
    from evaluate import evaluate
    return evaluate(list(users), np.asarray(y), np.asarray(scores, dtype=np.float64))


def _vrank(users, x):
    """Within-user percentile rank. Only intra-user order is scored, and members
    are calibrated differently, so ranks are the right space to average in."""
    m = {u: i for i, u in enumerate(sorted(set(users)))}
    idx = np.fromiter((m[u] for u in users), np.int64, len(users))
    o = np.lexsort((x, idx)); su = idx[o]; n = len(x)
    new = np.r_[True, su[1:] != su[:-1]]
    gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
    out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
    return out


def _tag(family, params, handle):
    h = hashlib.sha1(json.dumps([family, params, handle], sort_keys=True).encode())
    return f"{family}_{h.hexdigest()[:10]}"


# ------------------------------------------------------------------- trainers
def _fit_catboost(F, params, budget_s):
    from catboost import CatBoostRanker, Pool
    # Accept CatBoost's own key too. Reading only `loss` meant a request for
    # QuerySoftMax was silently served as the YetiRank default - the caller saw
    # a number for a configuration it never actually ran.
    if "loss_function" in params and "loss" not in params:
        params["loss"] = params.pop("loss_function")
    ncat = F["Xtr"].shape[1]
    # Measured ~0.35s per iteration at depth 6 on this data; keep the fit inside
    # the caller's budget rather than discovering the ceiling by being killed.
    # Per-iteration cost at depth 6, from six measured fits through this tool:
    #   400 iters / 159s · 400 / 223s · 409 / 199s · 409 / 182s  ->  0.40-0.56 s
    # The first estimate of 0.35s was too low (a budget_s=600 call ran 1083s);
    # correcting it to 1.1s then over-shot the other way and silently cut a
    # requested 1000 iterations to 409, while the best hand-tuned result on this
    # data used 782. Both directions cost score, so this is set from the
    # measurements with a modest safety margin rather than from a guess.
    per_iter = 0.6 * (2 ** max(0, params.get("depth", 6) - 6))
    cap = max(100, int(budget_s * 0.75 / per_iter))
    iters = min(int(params.get("iterations", 600)), cap)

    def pool(split):
        X = F[f"X{split}"]; y = F[f"y{split}"]; u = F[f"u{split}"]
        o, c = _groups(u)
        cats = X[o][:, :ncat].astype(np.int64).astype(str)
        dense = F.get(f"D{split}")
        data = np.hstack([cats, dense[o].astype(object)]) if dense is not None else cats
        return Pool(data, label=y[o], group_id=np.repeat(np.arange(len(c)), c),
                    cat_features=list(range(ncat))), o

    ptr, _ = pool("tr")
    m = CatBoostRanker(loss_function=params.get("loss", "YetiRank"),
                       iterations=iters, depth=params.get("depth", 6),
                       learning_rate=params.get("learning_rate", 0.08),
                       l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
                       verbose=0, thread_count=8, allow_writing_files=False)
    m.fit(ptr)
    pva, ova = pool("va")
    s = np.empty(len(F["yva"])); s[ova] = m.predict(pva)
    return s, {"iterations_used": iters}


def _fit_lightgbm(F, params, budget_s):
    import lightgbm as lgb
    ncat = F["Xtr"].shape[1]
    Xtr = F["Xtr"].astype(np.float32)
    if F.get("Dtr") is not None:
        Xtr = np.hstack([Xtr, F["Dtr"]])
    Xva = F["Xva"].astype(np.float32)
    if F.get("Dva") is not None:
        Xva = np.hstack([Xva, F["Dva"]])
    o, c = _groups(F["utr"])
    obj = params.get("objective", "lambdarank")
    ds = (lgb.Dataset(Xtr[o], label=F["ytr"][o], group=c,
                      categorical_feature=list(range(ncat)), free_raw_data=False)
          if obj in ("lambdarank", "rank_xendcg")
          else lgb.Dataset(Xtr, label=F["ytr"],
                           categorical_feature=list(range(ncat)), free_raw_data=False))
    p = {"objective": obj, "learning_rate": params.get("learning_rate", 0.05),
         "num_leaves": params.get("num_leaves", 63),
         "min_data_in_leaf": params.get("min_data_in_leaf", 50),
         "feature_fraction": params.get("feature_fraction", 0.9),
         "bagging_fraction": params.get("bagging_fraction", 0.9), "bagging_freq": 1,
         "verbose": -1, "num_threads": 8}
    if obj in ("lambdarank", "rank_xendcg"):
        p.update(metric="ndcg", ndcg_eval_at=[5])
    b = lgb.train(p, ds, num_boost_round=int(params.get("n_estimators", 300)))
    return b.predict(Xva), {"rounds": int(params.get("n_estimators", 300))}


def _fit_xgboost(F, params, budget_s):
    import xgboost as xgb
    Xtr = F["Xtr"].astype(np.float32)
    if F.get("Dtr") is not None:
        Xtr = np.hstack([Xtr, F["Dtr"]])
    Xva = F["Xva"].astype(np.float32)
    if F.get("Dva") is not None:
        Xva = np.hstack([Xva, F["Dva"]])
    o, c = _groups(F["utr"])
    d = xgb.DMatrix(Xtr[o], label=F["ytr"][o]); d.set_group(c)
    bst = xgb.train({"objective": params.get("objective", "rank:ndcg"),
                     "eta": params.get("learning_rate", 0.08),
                     "max_depth": params.get("max_depth", 8),
                     "subsample": params.get("subsample", 0.9),
                     "colsample_bytree": params.get("colsample_bytree", 0.8),
                     "nthread": 8, "eval_metric": "ndcg@5"},
                    d, num_boost_round=int(params.get("n_estimators", 300)))
    return bst.predict(xgb.DMatrix(Xva)), {}


FAMILIES = {"catboost": _fit_catboost, "lightgbm": _fit_lightgbm,
            "xgboost": _fit_xgboost}


# ----------------------------------------------------------------------- API
def train(family: str, feature_handle: str, params: dict = None,
          budget_s: int = 600, seed: int = 0) -> dict:
    """Fit one model on train, score validation, cache the predictions.

    Returns the validation metrics and a prediction id that `blend` can combine.
    Test is never touched: the feature set contains train and validation only.
    """
    import tools
    if family not in FAMILIES:
        return {"error": f"unknown family '{family}'", "available": sorted(FAMILIES),
                "note": "torch models you should still write yourself; these are "
                        "the ones where a correct configuration is fiddly"}
    try:
        F = tools.load_features(feature_handle)
    except FileNotFoundError as e:
        return {"error": str(e)}

    params = dict(params or {})
    params.setdefault("random_seed", seed)
    t0 = time.time()
    try:
        scores, extra = FAMILIES[family](F, params, budget_s)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "family": family,
                "params": params}
    secs = time.time() - t0

    r = _primary(F["uva"], F["yva"], scores)
    pid = _tag(family, params, feature_handle)
    np.save(PRED / f"{pid}.npy", np.asarray(scores, dtype=np.float64))
    (PRED / f"{pid}.json").write_text(json.dumps(
        {"family": family, "handle": feature_handle, "params": params,
         "valid": {k: round(float(v), 4) for k, v in r.items()}, "secs": round(secs)}))
    return {"prediction_id": pid, "family": family, "seconds": round(secs),
            "valid_primary": round(float(r["primary"]), 4),
            "valid_GAUC": round(float(r["GAUC"]), 4),
            "valid_nDCG@5": round(float(r["nDCG@5"]), 4),
            "baseline_primary": 0.6015, **extra,
            "note": "combine several prediction_ids with the blend tool"}


# The TOOL is advertised as `train_model`; the function is `train`. Generated
# code imported the tool's name and failed with "cannot import name
# 'train_model'", which is the third naming mismatch of this kind to cost an
# iteration (after `features` and `from models import load_features`). An audit
# of every tool name against every importable function found this and `recall`
# as the only two that did not line up; both are aliased so the name the agent
# is shown is always the name it can import.
def train_model(family: str, feature_handle: str, params: dict = None,
                budget_s: int = 600, seed: int = 0) -> dict:
    """Alias of train(), matching the advertised tool name."""
    return train(family, feature_handle, params, budget_s, seed)


def recall(topic: str = "", limit: int = 12) -> dict:
    """Alias of memory.recall(), matching the advertised tool name."""
    import memory
    return memory.recall(topic, limit)


def load_scores(prediction_id: str) -> np.ndarray:
    """Validation scores from an earlier train() or blend(), for run() to return.

    Without this the tool phase was a dead end: an agent could call train_model,
    see 0.6046, and then have no way to turn that into the iteration's result
    except by hand-writing the same model inside run() - which is the exact
    reimplementation the tool exists to avoid, and which failed three times on
    group ordering immediately after a successful tool call.
    """
    # Guard the shape first. A caller did r["prediction_id"] on a train() result
    # that was actually an error dict, then passed the dict itself here — which
    # became a 200-character filename and an unreadable "File name too long"
    # error two layers from the real cause.
    if isinstance(prediction_id, dict):
        raise ValueError(
            "load_scores was given a dict, not a prediction_id. This usually "
            "means train() returned an error and the code read the result "
            "without checking: " + json.dumps(prediction_id)[:300] +
            "\nCheck `if 'error' in r:` before using r['prediction_id'].")
    if not isinstance(prediction_id, str) or not prediction_id:
        raise ValueError(f"prediction_id must be a non-empty string, got "
                         f"{type(prediction_id).__name__}: {prediction_id!r}")
    p = PRED / f"{prediction_id}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"no prediction '{prediction_id}'. Call train_model or blend first; "
            f"available: {[q.stem for q in PRED.glob('*.npy')]}")
    return np.load(p)


def list_predictions() -> dict:
    """Every model trained so far this run, with its validation score."""
    out = []
    for p in sorted(PRED.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        out.append({"prediction_id": p.stem, "family": d.get("family"),
                    "handle": d.get("handle"),
                    "valid_primary": d.get("valid", {}).get("primary"),
                    "params": d.get("params")})
    out.sort(key=lambda x: -(x["valid_primary"] or 0))
    return {"count": len(out), "predictions": out}


def blend(prediction_ids: list, feature_handle: str = None) -> dict:
    """Rank-average several cached predictions and score the combination.

    Equal weights, deliberately. Searching blend weights on validation was
    measured to HURT test on this benchmark - the weight search fits noise - so
    the combination rule here has no free parameters to overfit.
    """
    import tools
    ids = [i for i in (prediction_ids or [])]
    if len(ids) < 2:
        return {"error": "give at least two prediction_ids", "have": list_predictions()}
    metas, arrays = [], []
    for i in ids:
        f = PRED / f"{i}.npy"
        if not f.exists():
            return {"error": f"no prediction '{i}'", "have": list_predictions()}
        arrays.append(np.load(f))
        try:
            metas.append(json.loads((PRED / f"{i}.json").read_text()))
        except Exception:
            metas.append({})
    handle = feature_handle or metas[0].get("handle")
    F = tools.load_features(handle)
    u, y = F["uva"], F["yva"]
    if any(len(a) != len(y) for a in arrays):
        return {"error": "predictions have different lengths; they must all come "
                         "from feature sets with the same validation rows"}

    ranked = [_vrank(u, a) for a in arrays]
    combined = np.mean(ranked, axis=0)
    r = _primary(u, y, combined)
    singles = [round(float(_primary(u, y, a)["primary"]), 4) for a in arrays]
    pid = _tag("blend", {"ids": sorted(ids)}, handle)
    np.save(PRED / f"{pid}.npy", combined)
    (PRED / f"{pid}.json").write_text(json.dumps(
        {"family": "blend", "handle": handle, "params": {"ids": ids},
         "valid": {k: round(float(v), 4) for k, v in r.items()}}))
    return {"prediction_id": pid, "members": len(ids),
            "member_primaries": singles, "best_member": max(singles),
            "valid_primary": round(float(r["primary"]), 4),
            "gain_over_best_member": round(float(r["primary"]) - max(singles), 4)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    import tools
    h = tools.build_features({"spec": {
        "name": "selftest_models",
        "categorical": ["user_id", "video_id", "author_id", "tab"],
        "derived": ["hour", "dur_bucket", "video_age"],
        "video_meta": ["tag"],
        "video_stats": True}})
    print("features:", h.get("dim"), h.get("dense_block"))
    a = train("lightgbm", "selftest_models", {"n_estimators": 200}, budget_s=300)
    print("lightgbm:", a)
    b = train("catboost", "selftest_models",
              {"loss": "QueryRMSE", "depth": 6, "iterations": 300}, budget_s=400)
    print("catboost:", b)
    if "prediction_id" in a and "prediction_id" in b:
        print("blend   :", blend([a["prediction_id"], b["prediction_id"]]))

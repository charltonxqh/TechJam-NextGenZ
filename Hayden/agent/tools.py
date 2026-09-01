"""Tools the agent can call — its hands, as opposed to its prompt.

Until now the agent had none. It received one prompt containing facts WE had
measured, wrote one Python function, and that was the whole interaction. Two
consequences, both visible in the results:

  1. It could not reach the raw data. `prep.py` handed it 5 pre-encoded columns,
     so every feature that actually moved the score this project -- hourmin, tag,
     video age, the 51 columns of video_features_statistic_pure.csv -- was
     structurally unreachable. A human found those, not the agent.

  2. The facts in context.py are OUR conclusions, some with instructions
     attached ("SIM has no problem to solve here"). The README claims the design
     is "facts, not answers"; that line is an answer. Handing the agent findings
     it did not make weakens exactly the thing Innovation is scored on -- what
     the agent identified as worth trying, and why.

These tools replace both. The agent asks its own questions of the data and
builds its own features, so a finding it reports is one it actually made.

    list_columns()          what exists, across all four raw CSVs
    inspect(question)       EDA over train+valid: distributions, coverage, rates
    build_features(spec)    construct an encoded design matrix from chosen raw
                            columns; caches it and returns a handle the
                            generated code loads with load_features(handle)
    search_papers(query)    arXiv search, so published methods enter the loop by
                            the agent's choice rather than ours

SPLIT SAFETY. Every tool that touches the log filters to train+valid dates before
anything else. The test window (20220429-20220508) is dropped at read time, in
one place -- `_read_rows` -- so a tool cannot return test-derived numbers even by
accident. build_features writes only train/valid matrices; there is no code path
that produces a test array.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
DATA = KIT / "KuaiRand-Pure" / "data"
FCACHE = HERE / "cache" / "features"

TRAIN = (20220408, 20220421)
VALID = (20220422, 20220428)
TEST_START = 20220429           # never read past here

LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
VIDEO_BASIC = "video_features_basic_pure.csv"
VIDEO_STATS = "video_features_statistic_pure.csv"
USER_FEAT = "user_features_pure.csv"

# outcome columns: legal as training targets, illegal as model inputs
OUTCOMES = {"is_click", "is_like", "is_follow", "is_comment", "is_forward",
            "is_hate", "long_view", "is_profile_enter", "play_time_ms",
            "profile_stay_time", "comment_stay_time"}


# --------------------------------------------------------------------------
# reading — the single choke point where test is excluded
# --------------------------------------------------------------------------
_rows_cache = None


def _read_rows():
    """All train+valid log rows as dicts. Test dates never enter this process."""
    global _rows_cache
    if _rows_cache is not None:
        return _rows_cache
    out = []
    for fn in LOGS:
        with open(DATA / fn) as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                if d >= TEST_START:          # <- the guard
                    continue
                r["_split"] = "train" if d <= TRAIN[1] else "valid"
                out.append(r)
    _rows_cache = out
    return out


def _side_table(fn, key):
    with open(DATA / fn) as fh:
        return {r[key]: r for r in csv.DictReader(fh)}


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# tool: list_columns
# --------------------------------------------------------------------------
def list_columns() -> dict:
    """Every column available, by file, with a usable/outcome marker."""
    out = {}
    for fn in (LOGS[0], VIDEO_BASIC, VIDEO_STATS, USER_FEAT):
        with open(DATA / fn) as fh:
            cols = next(csv.reader(fh))
        out[fn] = [{"name": c, "role": "OUTCOME (target only, never an input)"
                    if c in OUTCOMES else "feature"} for c in cols]
    out["_note"] = ("Outcome columns are observed only after the impression. Use "
                    "them as auxiliary training targets if you like; using one as "
                    "a model input is leakage and will be rejected.")
    return out


# --------------------------------------------------------------------------
# tool: inspect
# --------------------------------------------------------------------------
def inspect(question: str = "") -> dict:
    """Answer an EDA question over train+valid. `question` selects a report."""
    q = (question or "").lower()
    if not q.strip():
        return {"supported_questions": [
            "group sizes / rows per user", "label rate / positives",
            "cold start / unseen users", "cardinality of each column",
            "available columns"],
            "hint": 'call inspect({"question": "group sizes"})'}
    rows = _read_rows()
    tr = [r for r in rows if r["_split"] == "train"]
    va = [r for r in rows if r["_split"] == "valid"]

    if "group" in q or "per user" in q or "history" in q:
        out = {}
        for name, rs in (("train", tr), ("valid", va)):
            c = {}
            for r in rs:
                c[r["user_id"]] = c.get(r["user_id"], 0) + 1
            v = np.array(sorted(c.values()))
            out[name] = {"users": len(v), "rows_per_user_mean": round(float(v.mean()), 2),
                         "median": int(np.median(v)), "p90": int(np.quantile(v, .9)),
                         "max": int(v.max())}
        return out

    if "label" in q or "rate" in q or "positive" in q:
        out = {}
        for name, rs in (("train", tr), ("valid", va)):
            y = np.array([_num(r["long_view"]) for r in rs])
            byu = {}
            for r in rs:
                byu.setdefault(r["user_id"], []).append(_num(r["long_view"]))
            allneg = sum(1 for v in byu.values() if sum(v) == 0)
            allpos = sum(1 for v in byu.values() if sum(v) == len(v))
            out[name] = {"rows": len(y), "long_view_rate": round(float(y.mean()), 4),
                         "users": len(byu),
                         "all_negative_users_pct": round(100 * allneg / len(byu), 1),
                         "all_positive_users_pct": round(100 * allpos / len(byu), 1)}
        return out

    if "cold" in q or "unseen" in q or "overlap" in q:
        tu = {r["user_id"] for r in tr}; ti = {r["video_id"] for r in tr}
        return {"valid_users_unseen_in_train_pct":
                round(100 * np.mean([r["user_id"] not in tu for r in va]), 2),
                "valid_videos_unseen_in_train_pct":
                round(100 * np.mean([r["video_id"] not in ti for r in va]), 2)}

    if "cardinal" in q or "unique" in q or "distinct" in q:
        cols = [c for c in tr[0] if c not in OUTCOMES and not c.startswith("_")]
        return {c: len({r[c] for r in tr}) for c in cols}

    if "column" in q or "available" in q:
        return list_columns()

    return {"error": "unrecognised question",
            "supported": ["group sizes / rows per user", "label rate / positives",
                          "cold start / unseen users", "cardinality of each column",
                          "available columns"]}


# --------------------------------------------------------------------------
# tool: build_features
# --------------------------------------------------------------------------
def build_features(spec: dict) -> dict:
    """Build an encoded design matrix from raw columns.

    spec = {
      "name": "v1",
      "categorical": ["user_id","video_id","author_id","tab"],   # log columns
      "derived":     ["hour","video_age","dur_bucket"],          # computed below
      "video_meta":  ["tag"],                # from video_features_basic
      "video_stats": ["long_time_play_cnt","show_cnt"],  # as rate + log-count
    }
    Returns a handle. Generated code calls load_features(handle) to get
    Xtr/ytr/utr/Xva/yva/uva/dim.
    """
    # Be forgiving about shape. The model reaches for reasonable-looking
    # variants (extra_fields=[...], video_stats=True) and every rejection costs
    # it a tool call out of a small budget, so accept the obvious synonyms
    # instead of making it guess our exact schema.
    if not isinstance(spec, dict):
        return {"error": "spec must be an object",
                "example": {"name": "v1", "categorical": ["user_id", "video_id"],
                            "derived": ["hour"], "video_stats": True}}
    spec = dict(spec)
    if "spec" in spec and isinstance(spec["spec"], dict):     # double-wrapped
        spec = spec["spec"]

    def _listify(v):
        if v is None or v is False:
            return []
        if v is True:
            return True                     # sentinel: "all of them"
        return list(v) if isinstance(v, (list, tuple)) else [v]

    name = spec.get("name") or "unnamed"
    # Accept every synonym the model has actually used. It reached for
    # {"features": [...], "dense": [...]} twice in one run; the schema silently
    # ignored both, built a ZERO-COLUMN feature set, returned a valid-looking
    # handle, and the generated code then failed on 8192x0 matmuls with no way
    # to see why. Two iterations were lost to that.
    cats = _listify(spec.get("categorical") or spec.get("fields")
                    or spec.get("extra_fields") or spec.get("features")
                    or spec.get("categorical_features")
                    or spec.get("columns"))
    derived = _listify(spec.get("derived"))
    vmeta = _listify(spec.get("video_meta") or spec.get("video_metadata"))
    # `dense_block` is included because that is the key this function RETURNS
    # for the dense matrix, so it is the name the caller sees first and the
    # obvious thing to pass back in. Accepting only "video_stats" while
    # reporting "dense_block" is an inconsistency in this API, not a mistake by
    # the caller.
    _vs = next((spec[k] for k in ("video_stats", "include_video_stats", "dense",
                                  "dense_features", "stats", "video_statistics",
                                  "dense_block", "dense_columns")
                if k in spec), None)
    vstats = _listify(_vs)

    ALL_DERIVED = ["hour", "dur_bucket", "video_age"]
    if cats is True:
        cats = ["user_id", "video_id", "author_id", "tab"]
    if derived is True:
        derived = list(ALL_DERIVED)
    if vmeta is True:
        vmeta = ["tag", "music_id", "video_type", "upload_type"]
    if vstats is True:                       # "give me all the statistics"
        with open(DATA / VIDEO_STATS) as fh:
            vstats = [c for c in next(csv.reader(fh)) if c != "video_id"]

    # A field the log doesn't have but we can derive: route it automatically
    # rather than erroring (the model often lists 'hourmin' or 'tag' as
    # categorical without knowing which file they live in).
    LOGCOLS = set(next(csv.reader(open(DATA / LOGS[0]))))
    moved = []
    for c in list(cats):
        if c in ("hour", "hourmin"):
            cats.remove(c); derived.append("hour"); moved.append(f"{c}->derived hour")
        elif c in ("video_age", "age", "upload_dt"):
            cats.remove(c); derived.append("video_age"); moved.append(f"{c}->derived video_age")
        elif c == "dur_bucket":
            cats.remove(c); derived.append("dur_bucket"); moved.append(f"{c}->derived")
        elif c not in LOGCOLS:
            cats.remove(c); vmeta.append(c); moved.append(f"{c}->video_meta")
    derived = list(dict.fromkeys(derived)); vmeta = list(dict.fromkeys(vmeta))

    bad = [c for c in cats if c in OUTCOMES]
    if bad:
        return {"error": f"outcome columns cannot be model inputs: {bad}"}

    rows = _read_rows()
    basic = _side_table(VIDEO_BASIC, "video_id") if (vmeta or "video_age" in derived) else {}
    stats = _side_table(VIDEO_STATS, "video_id") if vstats else {}

    # ---- derive per-row categorical values ---------------------------
    def derive(r):
        vals = {}
        if "hour" in derived:
            vals["hour"] = str(int(_num(r.get("hourmin"), 0)) // 100)
        if "dur_bucket" in derived:
            vals["dur_bucket"] = str(int(min(_num(r.get("duration_ms")) // 10000, 20)))
        if "video_age" in derived:
            b = basic.get(r["video_id"])
            age = -1
            if b:
                try:
                    up = dt.date.fromisoformat(b["upload_dt"])
                    d = int(r["date"])
                    age = (dt.date(d // 10000, d // 100 % 100, d % 100) - up).days
                except Exception:
                    age = -1
            vals["video_age"] = str(min(max(age, -1), 365) // 7)     # weekly buckets
        for m in vmeta:
            b = basic.get(r["video_id"])
            vals[m] = (b or {}).get(m, "UNK")
        return vals

    fields = cats + [d for d in derived] + vmeta
    if not fields:
        # Never hand back a usable-looking handle for an empty feature set.
        return {"error": "no feature columns were recognised in your spec, so "
                         "there is nothing to build",
                "you_sent": sorted(spec),
                "expected_keys": {
                    "name": "str",
                    "categorical": "log columns, e.g. ['user_id','video_id','tab']",
                    "derived": "any of ['hour','dur_bucket','video_age']",
                    "video_meta": "video_features_basic columns, e.g. ['tag','music_id']",
                    "video_stats": "video_features_statistic columns, or true for all 51",
                    "aggregates": "statistics computed from the training log itself: "
                                  "video_count, video_duration_stats, video_target_loo, "
                                  "user_video_duration_gap - or true for all"},
                "example": {"name": "f1",
                            "categorical": ["user_id", "video_id", "author_id", "tab"],
                            "derived": ["hour", "video_age"],
                            "video_meta": ["tag"],
                            "video_stats": True}}
    tr = [r for r in rows if r["_split"] == "train"]
    va = [r for r in rows if r["_split"] == "valid"]

    # ---- vocabulary from TRAIN only; unseen -> UNK slot ---------------
    vocab, offset, dim = {}, {}, 0
    tr_vals = [derive(r) for r in tr]
    va_vals = [derive(r) for r in va]
    for f in fields:
        seen = sorted({(tr_vals[i].get(f) if f not in r else r[f])
                       for i, r in enumerate(tr)})
        vocab[f] = {v: j for j, v in enumerate(seen)}
        offset[f] = dim
        dim += len(seen) + 1                       # +1 UNK

    def encode(rs, vals):
        X = np.zeros((len(rs), len(fields)), dtype=np.int32)
        for i, r in enumerate(rs):
            for j, f in enumerate(fields):
                v = vals[i].get(f) if f not in r else r[f]
                X[i, j] = offset[f] + vocab[f].get(v, len(vocab[f]))
        return X

    Xtr, Xva = encode(tr, tr_vals), encode(va, va_vals)
    ytr = np.array([_num(r["long_view"]) for r in tr], dtype=np.float32)
    yva = np.array([_num(r["long_view"]) for r in va], dtype=np.float32)
    utr = [r["user_id"] for r in tr]; uva = [r["user_id"] for r in va]

    # ---- optional dense block from video statistics -------------------
    # ---- aggregate features computed FROM THE TRAINING LOG ------------
    # Statistics of the data itself (counts, means, standard deviations),
    # rather than columns read out of a file.
    #
    # ONE CONSTRAINT DOMINATES THE DESIGN: a per-USER aggregate is constant
    # inside that user's group, and both metrics only see intra-user order, so
    # a user's mean watch time or impression count cannot move the score at
    # all. Only per-VIDEO aggregates and USER x VIDEO interactions vary within
    # a group, so those are what is built here. User-level statistics appear
    # only as the reference point of an interaction term.
    #
    # Every statistic is computed on TRAIN rows only. Validation rows look up
    # values fitted on train; train rows use LEAVE-ONE-OUT so a row never
    # contributes to the statistic used to predict it.
    aggs = _listify(spec.get("aggregates") if "aggregates" in spec
                    else spec.get("aggregate_features"))
    if aggs is True:
        # video_target_loo is deliberately NOT in the default set. Measured
        # against an identical control it cost -0.0169 (~20 sigma) while the
        # other three were each within noise:
        #     control 0.6040 | count 0.6044 | duration 0.6045
        #     user-video gap 0.6045 | target LOO 0.5871
        # Leave-one-out target encoding is invertible: gradient boosting can
        # solve the LOO formula for the row's own label, so training fit rises
        # and validation collapses. It stays available by explicit request, for
        # anyone who wants to pair it with out-of-fold encoding instead.
        aggs = ["video_count", "video_duration_stats", "user_video_duration_gap"]
    Atr = Ava = None
    agg_names: list = []
    if aggs:
        from collections import defaultdict
        v_n = defaultdict(int); v_dur = defaultdict(list)
        v_pos = defaultdict(float); v_users = defaultdict(set)
        u_dur = defaultdict(list)
        for r in tr:
            vid = r["video_id"]; d = _num(r.get("duration_ms"))
            v_n[vid] += 1; v_dur[vid].append(d); v_users[vid].add(r["user_id"])
            v_pos[vid] += _num(r.get("long_view"))
            u_dur[r["user_id"]].append(d)
        v_mean = {k: float(np.mean(x)) for k, x in v_dur.items()}
        v_std = {k: float(np.std(x)) for k, x in v_dur.items()}
        u_mean = {k: float(np.mean(x)) for k, x in u_dur.items()}
        prior = float(np.mean([_num(r.get("long_view")) for r in tr]))
        PRIOR_W = 20.0            # smoothing: a video seen twice should not
                                  # swing its rate to 0 or 1

        def build_agg(rs, loo: bool):
            cols = []
            if "video_count" in aggs:
                cols += ["v_log_count", "v_log_users"]
            if "video_duration_stats" in aggs:
                cols += ["v_dur_mean", "v_dur_std"]
            if "video_target_loo" in aggs:
                cols += ["v_rate_smoothed"]
            if "user_video_duration_gap" in aggs:
                cols += ["uv_dur_gap", "uv_dur_ratio"]
            M = np.zeros((len(rs), len(cols)), dtype=np.float32)
            for i, r in enumerate(rs):
                vid = r["video_id"]; uid = r["user_id"]
                n = v_n.get(vid, 0); j = 0
                if "video_count" in aggs:
                    M[i, j] = np.log1p(max(n - (1 if loo else 0), 0)); j += 1
                    M[i, j] = np.log1p(len(v_users.get(vid, ()))); j += 1
                if "video_duration_stats" in aggs:
                    M[i, j] = v_mean.get(vid, 0.0) / 1e4; j += 1
                    M[i, j] = v_std.get(vid, 0.0) / 1e4; j += 1
                if "video_target_loo" in aggs:
                    s = v_pos.get(vid, 0.0); c = float(n)
                    if loo:                       # remove this row's own label
                        s -= _num(r.get("long_view")); c -= 1.0
                    M[i, j] = (s + PRIOR_W * prior) / (c + PRIOR_W); j += 1
                if "user_video_duration_gap" in aggs:
                    um = u_mean.get(uid); vd = _num(r.get("duration_ms"))
                    M[i, j] = 0.0 if um is None else (vd - um) / 1e4; j += 1
                    M[i, j] = 0.0 if not um else float(vd / max(um, 1.0)); j += 1
            return M, cols

        Atr, agg_names = build_agg(tr, loo=True)
        Ava, _ = build_agg(va, loo=False)

    Dtr = Dva = None
    if vstats:
        def dense(rs):
            M = np.zeros((len(rs), len(vstats) * 2), dtype=np.float32)
            for i, r in enumerate(rs):
                s = stats.get(r["video_id"])
                if not s:
                    continue
                show = max(_num(s.get("show_cnt"), 1.0), 1.0)
                for j, c in enumerate(vstats):
                    x = _num(s.get(c))
                    M[i, j] = x / show
                    M[i, len(vstats) + j] = np.log1p(abs(x))
            return M
        Dtr, Dva = dense(tr), dense(va)

    # aggregates ride in the same dense block the models already consume
    if Atr is not None:
        Dtr = Atr if Dtr is None else np.hstack([Dtr, Atr])
        Dva = Ava if Dva is None else np.hstack([Dva, Ava])
        vstats = list(vstats) + agg_names

    FCACHE.mkdir(parents=True, exist_ok=True)
    p = FCACHE / f"{name}.npz"
    payload = dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva,
                   utr=np.array(utr), uva=np.array(uva),
                   fields=np.array(fields), dim=np.array(dim))
    if Dtr is not None:
        payload.update(Dtr=Dtr, Dva=Dva, dense_cols=np.array(vstats))
    np.savez_compressed(p, **payload)

    out = {"handle": name, "path": str(p), "fields": fields, "dim": int(dim),
           "train_rows": int(len(ytr)), "valid_rows": int(len(yva)),
           "dense_block": None if Dtr is None else list(Dtr.shape),
           "usage": f'load_features("{name}") -> dict with '
                    f'Xtr,ytr,utr,Xva,yva,uva,dim' +
                    (",Dtr,Dva" if Dtr is not None else "")}
    if moved:
        out["auto_routed"] = moved      # tell it what we fixed, so it learns
    return out


# Set to True by finalize.py ONLY, after the run has converged, so the winning
# code can be scored on test. It is False for the entire agent run: during the
# loop, load_features cannot return test rows because _read_rows never reads
# them. finalize is a separate process started by a human after convergence.
FINALIZE_TEST = False


def load_features(handle: str) -> dict:
    """Called by generated code to load what build_features produced.

    During the run this returns train+valid. finalize.py flips FINALIZE_TEST and
    the VALID SLOT is then filled with the test split — mirroring the trick
    finalize already uses for the default D, so a solution written against
    load_features can be scored on test without being rewritten.
    """
    p = FCACHE / f"{handle}.npz"
    if not p.exists():
        # The solution referenced a handle that was never built - it skipped the
        # build_features tool call. A bare FileNotFoundError gives the repair
        # loop nothing to act on, so say exactly what went wrong and what exists.
        avail = sorted(q.stem for q in FCACHE.glob("*.npz"))
        raise FileNotFoundError(
            f"No feature set named '{handle}'. A handle only exists after you call "
            f"the build_features TOOL during the investigation phase; it cannot be "
            f"created from inside run(). "
            + (f"Available handles: {avail}. " if avail else "No handles have been "
               "built in this run. ")
            + "Either use one of those, or use the default D that run(D, seed) is "
              "given, which needs no handle.")
    z = np.load(p, allow_pickle=False)
    D = {k: z[k] for k in z.files}
    D["dim"] = int(D["dim"])
    D["utr"] = list(D["utr"]); D["uva"] = list(D["uva"])
    D["fields"] = list(D["fields"])
    if FINALIZE_TEST:
        te = materialise_test(handle)
        D["Xva"], D["yva"], D["uva"] = te["X"], te["y"], te["u"]
        if te.get("Dense") is not None:
            D["Dva"] = te["Dense"]
    return D


def _read_test_rows():
    """Test rows. Reachable ONLY from materialise_test, which only finalize calls."""
    out = []
    with open(DATA / LOGS[1]) as fh:
        for r in csv.DictReader(fh):
            if int(r["date"]) >= TEST_START:
                out.append(r)
    return out


def materialise_test(handle: str) -> dict:
    """Encode the test split with the SAME vocabulary as the stored feature set.

    build_features derives its vocabulary from train rows only, sorted, so it is
    fully reproducible - we can rebuild the identical mapping here without
    having stored it, and any test value unseen in train lands in the UNK slot
    exactly as a validation-time unknown would.
    """
    z = np.load(FCACHE / f"{handle}.npz", allow_pickle=False)
    fields = [str(f) for f in z["fields"]]
    vstats = [str(c) for c in z["dense_cols"]] if "dense_cols" in z.files else []

    rows = _read_rows()                       # train+valid, for the vocabulary
    tr = [r for r in rows if r["_split"] == "train"]
    te = _read_test_rows()
    basic = _side_table(VIDEO_BASIC, "video_id")
    stats = _side_table(VIDEO_STATS, "video_id") if vstats else {}

    def derive(r):
        vals = {}
        if "hour" in fields:
            vals["hour"] = str(int(_num(r.get("hourmin"), 0)) // 100)
        if "dur_bucket" in fields:
            vals["dur_bucket"] = str(int(min(_num(r.get("duration_ms")) // 10000, 20)))
        if "video_age" in fields:
            b = basic.get(r["video_id"]); age = -1
            if b:
                try:
                    up = dt.date.fromisoformat(b["upload_dt"]); d = int(r["date"])
                    age = (dt.date(d // 10000, d // 100 % 100, d % 100) - up).days
                except Exception:
                    age = -1
            vals["video_age"] = str(min(max(age, -1), 365) // 7)
        for f in fields:
            if f in ("hour", "dur_bucket", "video_age") or f in r:
                continue
            vals[f] = (basic.get(r["video_id"]) or {}).get(f, "UNK")
        return vals

    tr_vals = [derive(r) for r in tr]
    vocab, offset, dim = {}, {}, 0
    for f in fields:
        seen = sorted({(tr_vals[i].get(f) if f not in r else r[f])
                       for i, r in enumerate(tr)})
        vocab[f] = {v: j for j, v in enumerate(seen)}
        offset[f] = dim
        dim += len(seen) + 1

    te_vals = [derive(r) for r in te]
    X = np.zeros((len(te), len(fields)), dtype=np.int32)
    for i, r in enumerate(te):
        for j, f in enumerate(fields):
            v = te_vals[i].get(f) if f not in r else r[f]
            X[i, j] = offset[f] + vocab[f].get(v, len(vocab[f]))

    Dense = None
    if vstats:
        Dense = np.zeros((len(te), len(vstats) * 2), dtype=np.float32)
        for i, r in enumerate(te):
            s = stats.get(r["video_id"])
            if not s:
                continue
            show = max(_num(s.get("show_cnt"), 1.0), 1.0)
            for j, c in enumerate(vstats):
                x = _num(s.get(c))
                Dense[i, j] = x / show
                Dense[i, len(vstats) + j] = np.log1p(abs(x))

    assert dim == int(z["dim"]), (
        f"vocabulary drift: rebuilt dim {dim} != stored {int(z['dim'])}")
    return {"X": X, "y": np.array([_num(r["long_view"]) for r in te], np.float32),
            "u": [r["user_id"] for r in te], "Dense": Dense}


# --------------------------------------------------------------------------
# tool: search_papers
# --------------------------------------------------------------------------
def search_papers(query: str, max_results: int = 5) -> dict:
    """arXiv search. No API key; lets the agent find published methods itself."""
    url = ("http://export.arxiv.org/api/query?"
           + urllib.parse.urlencode({"search_query": f"all:{query}",
                                     "start": 0, "max_results": max_results,
                                     "sortBy": "relevance"}))
    # arXiv rate-limits and occasionally stalls; one retry keeps a slow response
    # from costing the agent a whole tool call.
    last = None
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=40) as fh:
                root = ET.fromstring(fh.read())
            break
        except Exception as e:
            last = e
            if attempt == 0:
                import time as _t
                _t.sleep(3)
    else:
        return {"error": f"{type(last).__name__}: {last}",
                "hint": "arXiv was unreachable; proceed without it."}
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in root.findall("a:entry", ns):
        out.append({
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "summary": " ".join(e.findtext("a:summary", "", ns).split())[:600],
            "link": e.findtext("a:id", "", ns),
        })
    return {"query": query, "results": out}


# --------------------------------------------------------------------------
# registry — what gets advertised to the model
# --------------------------------------------------------------------------
TOOLS = {
    "list_columns": {
        "fn": list_columns,
        "args": {},
        "desc": "List every column in every raw CSV, marking which are outcomes "
                "(legal as training targets, illegal as model inputs).",
    },
    "inspect": {
        "fn": inspect,
        "args": {"question": "str"},
        "desc": "Run an EDA query over train+valid. Supports: group sizes / rows "
                "per user, label rate and all-positive/all-negative user shares, "
                "cold-start overlap, per-column cardinality, available columns.",
    },
    "build_features": {
        "fn": build_features,
        "args": {"spec": "dict with keys name, categorical, derived, video_meta, "
                         "video_stats"},
        "desc": "Build an encoded design matrix from any raw columns you choose, "
                "including derived fields (hour, dur_bucket, video_age), video "
                "metadata (tag, music_id) and a dense block from the 51 video "
                "statistics columns. Returns a handle; your generated code calls "
                "load_features(handle).",
    },
    "search_papers": {
        "fn": search_papers,
        "args": {"query": "str", "max_results": "int"},
        "desc": "Search arXiv for published methods. Use it before proposing a "
                "mechanism so the hypothesis is grounded in literature.",
    },
    "train_model": {
        "fn": lambda family, feature_handle, params=None, budget_s=600, seed=0:
            __import__("models").train(family, feature_handle, params, budget_s, seed),
        "args": {"family": "catboost | lightgbm | xgboost",
                 "feature_handle": "str, from build_features",
                 "params": "dict of hyperparameters",
                 "budget_s": "int seconds, the fit is sized to fit inside it"},
        "desc": "Fit a gradient-boosting model on train, score it on validation, "
                "and cache the predictions. These are the real CatBoost / "
                "LightGBM / XGBoost libraries; the tool only handles the data "
                "plumbing (group ordering for rankers, categorical indices) that "
                "is easy to get wrong in one attempt. YOU choose the family, the "
                "features and every hyperparameter. Returns a prediction_id.",
    },
    "blend": {
        "fn": lambda prediction_ids, feature_handle=None:
            __import__("models").blend(prediction_ids, feature_handle),
        "args": {"prediction_ids": "list of ids from train_model",
                 "feature_handle": "optional"},
        "desc": "Rank-average several cached predictions and score the result. "
                "Equal weights, no tunable coefficients - searching blend weights "
                "on validation was measured to hurt on this benchmark.",
    },
    "list_predictions": {
        "fn": lambda: __import__("models").list_predictions(),
        "args": {},
        "desc": "Every model trained so far this run with its validation score, "
                "so you can pick which ones to blend.",
    },
    "recall": {
        "fn": lambda topic="", limit=12: __import__("memory").recall(topic, limit),
        "args": {"topic": "str", "limit": "int"},
        "desc": "Search long-term memory for what has already been measured on "
                "this benchmark. Each result carries who measured it (agent or "
                "human), how many seeds, and a confidence rating. Check here "
                "BEFORE proposing a mechanism - re-running a measured dead end "
                "wastes an iteration and the convergence budget.",
    },
}


def describe() -> str:
    """Tool documentation block for the prompt."""
    lines = ["## Tools you can call",
             "",
             "Reply with a JSON object {\"tool\": name, \"args\": {...}} to call one.",
             "Call as many as you need before proposing a change; results come back",
             "and you may call again. When ready, reply with the change proposal",
             "instead of a tool call.",
             ""]
    for n, t in TOOLS.items():
        lines.append(f"- {n}({', '.join(t['args']) or ''})")
        lines.append(f"    {t['desc']}")
    lines += ["",
              "The test split (20220429-20220508) is excluded inside every tool at "
              "read time. You cannot reach it and should not try."]
    return "\n".join(lines)


def call(name: str, args: dict) -> dict:
    if name not in TOOLS:
        return {"error": f"unknown tool {name}", "available": list(TOOLS)}
    args = dict(args or {})
    # build_features takes a single `spec` object, but the model often passes the
    # spec's keys at the top level instead. That is a reasonable reading of the
    # signature, and rejecting it costs a tool call out of a small budget, so
    # rewrap rather than error.
    if name == "build_features" and "spec" not in args:
        args = {"spec": args}
    try:
        return TOOLS[name]["fn"](**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}",
                "expected": TOOLS[name]["args"]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        print("columns:", {k: len(v) for k, v in list_columns().items()
                           if k != "_note"})
        print("groups :", json.dumps(inspect("group sizes"), indent=None))
        print("labels :", json.dumps(inspect("label rate"), indent=None))
        print("cold   :", json.dumps(inspect("cold start"), indent=None))
        r = build_features({"name": "selftest",
                            "categorical": ["user_id", "video_id", "author_id", "tab"],
                            "derived": ["hour", "dur_bucket", "video_age"],
                            "video_meta": ["tag"],
                            "video_stats": ["long_time_play_cnt", "show_cnt"]})
        print("build  :", json.dumps(r, indent=None))
        D = load_features("selftest")
        print("loaded :", D["Xtr"].shape, D["Xva"].shape, "dim", D["dim"],
              "dense", D["Dtr"].shape)
    else:
        print(describe())

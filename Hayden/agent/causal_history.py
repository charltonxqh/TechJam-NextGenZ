"""Causal per-user history features, and the question of what the history may contain.

The other team's gain comes from features we never built: exponentially-decayed
per-user interaction rates, per-tab decayed rates, the previous outcome, a
short-window rate, and time since last interaction. Each row's feature is
computed from that user's OWN EARLIER OUTCOMES, walking forward in time.

We had ruled this family out, and the reasoning was wrong in an instructive way.
Our EDA measured that only 3.4% of validation rows involve a previously-seen
AUTHOR and 1.6% a repeat VIDEO, and concluded behavioural history was a dead end.
That is correct for user x ITEM history and irrelevant here: these features are
per USER over TIME, and every user has a history, so coverage is ~100%.

THE QUESTION THIS SCRIPT ANSWERS. Their write-up says the features use
"train-only history, no leakage into valid/test", but their code flattens
('train','valid','test') into one list and walks it, so a validation row's
history includes that user's EARLIER VALIDATION rows — that is, labels from the
evaluation period. Those are two materially different features, so both are built
here and measured separately:

    TRAIN_ONLY  state is updated only by train rows. A valid/test row sees the
                user's training history and nothing else. Unambiguously legal:
                at serving time you would have exactly this.

    ALL_CAUSAL  state is updated by every row, including valid/test. Strictly
                causal in time — a row never sees its own or any later label —
                but a validation row's feature is built from other validation
                rows' answers.

If the two score the same, the feature is legitimate and we simply missed it. If
ALL_CAUSAL is much higher, the difference is the evaluation period's labels
leaking in, and only TRAIN_ONLY is usable.
"""
from __future__ import annotations

import collections
import csv
import math
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
DATA = KIT / "KuaiRand-Pure" / "data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from data import SPLITS                      # noqa: E402
from evaluate import evaluate                # noqa: E402

LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
HALFLIFE_MS = 2.5 * 24 * 3600 * 1000         # 2.5 days, their winning value
TAB_HALFLIFE_MS = 3.0 * 24 * 3600 * 1000
ALPHA = 0.5                                  # Laplace smoothing, their retuned value
K = 10                                       # short-window length

FEATS = ["prior_rate", "decay_rate", "decay_act", "tab_decay_rate",
         "last1", "lastk_rate", "gap_days", "seq_pos"]


def load_rows():
    rows = []
    for fn in LOGS:
        with open(DATA / fn) as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                split = ("train" if d <= SPLITS["train"][1]
                         else "valid" if d <= SPLITS["valid"][1] else "test")
                rows.append({
                    "split": split, "user": r["user_id"], "video": r["video_id"],
                    "tab": r["tab"], "y": int(r["long_view"]),
                    "t": int(r["time_ms"]), "i": len(rows),
                })
    return rows


def build(rows, update_from):
    """Walk each user's rows forward in time, reading state before each row.

    update_from: which splits may contribute their label to the running state.
    """
    n = len(rows)
    out = {k: np.zeros(n, dtype=np.float32) for k in FEATS}
    by_user = collections.defaultdict(list)
    for idx, r in enumerate(rows):
        by_user[r["user"]].append(idx)

    for _, idxs in by_user.items():
        idxs.sort(key=lambda i: (rows[i]["t"], rows[i]["i"]))
        pos = tot = 0.0
        dpos = dtot = 0.0
        tab_pos = collections.defaultdict(float)
        tab_tot = collections.defaultdict(float)
        window = collections.deque(maxlen=K)
        last_t = None
        last_decay_t = None
        for step, i in enumerate(idxs):
            r = rows[i]; t = r["t"]

            # decay the accumulators to this row's timestamp
            if last_decay_t is not None:
                f = 0.5 ** ((t - last_decay_t) / HALFLIFE_MS)
                dpos *= f; dtot *= f
                ft = 0.5 ** ((t - last_decay_t) / TAB_HALFLIFE_MS)
                for k in list(tab_pos):
                    tab_pos[k] *= ft; tab_tot[k] *= ft
            last_decay_t = t

            # ---- READ state as it stood strictly before this row ----
            out["prior_rate"][i] = (pos + ALPHA) / (tot + 2 * ALPHA)
            out["decay_rate"][i] = (dpos + ALPHA) / (dtot + 2 * ALPHA)
            out["decay_act"][i] = math.log1p(dtot)
            tb = r["tab"]
            out["tab_decay_rate"][i] = ((tab_pos[tb] + ALPHA) /
                                        (tab_tot[tb] + 2 * ALPHA))
            out["last1"][i] = window[-1] if window else -1.0
            out["lastk_rate"][i] = ((sum(window) + ALPHA) /
                                    (len(window) + 2 * ALPHA)) if window else -1.0
            out["gap_days"][i] = ((t - last_t) / 86_400_000.0) if last_t is not None else -1.0
            out["seq_pos"][i] = step
            last_t = t

            # ---- only now fold this row's own outcome into the state ----
            if r["split"] in update_from:
                y = r["y"]
                pos += y; tot += 1.0
                dpos += y; dtot += 1.0
                tab_pos[tb] += y; tab_tot[tb] += 1.0
                window.append(y)
    return out


def score_with(rows, hist, tag):
    """Train LightGBM on base ids + these history features; report validation."""
    import lightgbm as lgb
    import tools

    F = tools.load_features("ch_base")
    ntr = len(F["ytr"]); nva = len(F["yva"])
    tr_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    va_idx = [i for i, r in enumerate(rows) if r["split"] == "valid"]
    assert len(tr_idx) == ntr and len(va_idx) == nva, "row alignment mismatch"

    Htr = np.column_stack([hist[k][tr_idx] for k in FEATS])
    Hva = np.column_stack([hist[k][va_idx] for k in FEATS])
    Xtr = np.hstack([F["Xtr"].astype(np.float32), F["Dtr"], Htr])
    Xva = np.hstack([F["Xva"].astype(np.float32), F["Dva"], Hva])

    u = np.array([hash(x) for x in F["utr"]])
    o = np.argsort(u, kind="stable")
    _, c = np.unique(u[o], return_counts=True)
    ds = lgb.Dataset(Xtr[o], label=F["ytr"][o], group=c,
                     categorical_feature=list(range(F["Xtr"].shape[1])),
                     free_raw_data=False)
    b = lgb.train({"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [5],
                   "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 50,
                   "feature_fraction": 0.9, "verbose": -1, "num_threads": 8},
                  ds, num_boost_round=400)
    r = evaluate(F["uva"], F["yva"], b.predict(Xva))
    print(f"  {tag:34s} GAUC {r['GAUC']:.4f}  nDCG@5 {r['nDCG@5']:.4f}  "
          f"primary {r['primary']:.4f}", flush=True)
    return r["primary"]


def main():
    import tools
    tools.build_features({"spec": {
        "name": "ch_base",
        "categorical": ["user_id", "video_id", "author_id", "tab"],
        "derived": ["hour", "dur_bucket", "video_age"],
        "video_meta": ["tag"], "video_stats": True}})

    t0 = time.time()
    rows = load_rows()
    print(f"loaded {len(rows):,} rows in {time.time()-t0:.0f}s\n", flush=True)

    # how predictive is each history feature on its own, before any model?
    va = [i for i, r in enumerate(rows) if r["split"] == "valid"]
    uv = [rows[i]["user"] for i in va]
    yv = np.array([rows[i]["y"] for i in va], float)

    for name, upd in (("TRAIN_ONLY", {"train"}),
                      ("ALL_CAUSAL", {"train", "valid", "test"})):
        h = build(rows, upd)
        print(f"── {name} — each feature alone, no model", flush=True)
        for k in ("decay_rate", "prior_rate", "tab_decay_rate", "lastk_rate"):
            r = evaluate(uv, yv, h[k][va].astype(np.float64))
            print(f"     {k:16s} GAUC {r['GAUC']:.4f}  primary {r['primary']:.4f}",
                  flush=True)
        score_with(rows, h, f"{name} + full model")
        print(flush=True)

    print("  reference: FM baseline 0.6015 · our best agent run 0.6051", flush=True)


if __name__ == "__main__":
    main()

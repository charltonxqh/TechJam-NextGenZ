"""One-time data preparation.

Loading + encoding the CSVs takes ~6s. Every experiment subprocess would pay
that, so we do it once and cache to .npz.

CRITICAL: the cache contains TRAIN and VALID only. The test split is
deliberately excluded so that no experiment can read it even by accident —
the agent physically cannot peek. Test is loaded once, separately, at
finalisation time (finalize.py).
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
KIT = next(p / "kuairand-starter-kit" for p in [_HERE, *_HERE.parents]
           if (p / "kuairand-starter-kit").is_dir())   # search upward: layout-independent
sys.path.insert(0, str(KIT))

from data import load, encode, FIELDS  # noqa: E402

CACHE = pathlib.Path(__file__).resolve().parent / "cache"
DATA_DIR = KIT / "KuaiRand-Pure" / "data"


AUX_COLS = ["is_click", "play_time_ms", "is_like", "is_profile_enter",
            "duration_ms"]
LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
SPLIT_DATES = {"train": (20220408, 20220421), "valid": (20220422, 20220428)}


def _load_aux(data_dir) -> dict:
    """Auxiliary supervision signals, in exactly the row order data.load() produces.

    data.load() reads the two standard logs in a fixed order and filters by date
    while preserving file order, so replicating that here keeps every array
    aligned 1:1 with Xtr / Xva. Test dates are skipped entirely.
    """
    import csv
    rows = {k: [] for k in SPLIT_DATES}
    for f in LOGS:
        with open(os.path.join(str(data_dir), f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                for name, (lo, hi) in SPLIT_DATES.items():
                    if lo <= d <= hi:
                        rows[name].append([float(r.get(c, 0) or 0) for c in AUX_COLS])
                        break
    return {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}


HIST_COLS = [
    "ua_cnt", "ua_pos", "ua_rate",     # user x author  (this video's author)
    "ud_cnt", "ud_pos", "ud_rate",     # user x duration bucket
    "ut_cnt", "ut_pos", "ut_rate",     # user x tab
    "uv_cnt",                          # times this user saw this exact video before
]


def _history_features(Xtr, ytr, Xva, prior=5.0):
    """User x item-attribute behaviour history — the features the workshop calls core.

    Why these shapes specifically: the metric only sees ordering WITHIN a user, so
    any per-user aggregate (e.g. "this user's overall long-view rate") is a constant
    across their candidates and provably invisible — we measured exactly that
    (per-user offsets scored -0.0001). History only helps if it varies BETWEEN a
    user's own candidates, which means it must be crossed with an item attribute.

    Leakage control:
      * train rows use leave-one-out counts (the row's own label is subtracted)
      * valid rows use the full train history, which is strictly earlier in time
    """
    import collections
    U, V, A, T, Db = 0, 1, 2, 3, 4          # column order in the encoded matrix

    def key_counts(col):
        c = collections.defaultdict(lambda: [0.0, 0.0])   # [count, positives]
        for i in range(len(ytr)):
            k = (Xtr[i, U], Xtr[i, col])
            c[k][0] += 1.0
            c[k][1] += ytr[i]
        return c

    auth, dur, tab = key_counts(A), key_counts(Db), key_counts(T)
    vid = collections.Counter((int(Xtr[i, U]), int(Xtr[i, V])) for i in range(len(ytr)))
    gmean = float(ytr.mean())

    def rows(X, y, loo):
        out = np.zeros((len(X), len(HIST_COLS)), dtype=np.float32)
        for i in range(len(X)):
            u = X[i, U]
            j = 0
            for c, col in ((auth, A), (dur, Db), (tab, T)):
                n, p = c.get((u, X[i, col]), (0.0, 0.0))
                if loo:                                   # remove this row's own contribution
                    n, p = n - 1.0, p - float(y[i])
                out[i, j] = n
                out[i, j + 1] = p
                out[i, j + 2] = (p + prior * gmean) / (n + prior)
                j += 3
            nv = vid.get((int(u), int(X[i, V])), 0)
            out[i, j] = nv - 1 if loo else nv
        return out

    return rows(Xtr, ytr, True), rows(Xva, np.zeros(len(Xva), np.float32), False)


def build(data_dir=None, force=False) -> pathlib.Path:
    CACHE.mkdir(exist_ok=True)
    out = CACHE / "trainvalid.npz"
    if out.exists() and not force:
        return out

    data_dir = data_dir or DATA_DIR
    splits = load(str(data_dir))
    enc, dim = encode(splits)
    aux = _load_aux(data_dir)

    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]

    # user ids are strings; store as an index + vocab to keep the npz compact
    uniq = sorted({*utr, *uva})
    u2i = {u: i for i, u in enumerate(uniq)}

    print("computing user x item behaviour-history features ...")
    hist_tr, hist_va = _history_features(Xtr, ytr, Xva)

    assert len(aux["train"]) == len(ytr), (
        f"aux/train misalignment: {len(aux['train'])} vs {len(ytr)}")
    assert len(aux["valid"]) == len(yva), (
        f"aux/valid misalignment: {len(aux['valid'])} vs {len(yva)}")

    np.savez_compressed(
        out,
        Xtr=Xtr, ytr=ytr, utr=np.array([u2i[u] for u in utr], dtype=np.int32),
        Xva=Xva, yva=yva, uva=np.array([u2i[u] for u in uva], dtype=np.int32),
        users=np.array(uniq), dim=np.array([dim]), fields=np.array(FIELDS),
        aux_tr=aux["train"], aux_va=aux["valid"], aux_cols=np.array(AUX_COLS),
        hist_tr=hist_tr, hist_va=hist_va, hist_cols=np.array(HIST_COLS),
    )
    print(f"cached -> {out}  ({out.stat().st_size/1e6:.1f} MB)  dim={dim}")
    print(f"  train {Xtr.shape}  valid {Xva.shape}   (test deliberately excluded)")
    print(f"  aux {aux['train'].shape} cols={AUX_COLS}")
    for i, c in enumerate(AUX_COLS):
        col = aux["train"][:, i]
        print(f"    {c:18s} mean={col.mean():12.3f}  nonzero={100*(col!=0).mean():5.1f}%")
    return out


def load_cache():
    """Returns dict with Xtr,ytr,utr,Xva,yva,uva (user ids as ORIGINAL strings), dim."""
    p = CACHE / "trainvalid.npz"
    if not p.exists():
        build()
    z = np.load(p, allow_pickle=False)
    users = z["users"]
    D = {
        "Xtr": z["Xtr"], "ytr": z["ytr"], "utr": users[z["utr"]].tolist(),
        "Xva": z["Xva"], "yva": z["yva"], "uva": users[z["uva"]].tolist(),
        "dim": int(z["dim"][0]), "fields": z["fields"].tolist(),
    }
    if "aux_tr" in z.files:
        cols = z["aux_cols"].tolist()
        D["aux_cols"] = cols
        # expose each auxiliary signal by name, train split only (they are
        # supervision, not features — using them at prediction time would leak)
        for i, c in enumerate(cols):
            D[f"aux_{c}"] = z["aux_tr"][:, i]
    if "hist_tr" in z.files:
        D["hist_cols"] = z["hist_cols"].tolist()
        D["Htr"] = z["hist_tr"]          # (N_train, 10) float32
        D["Hva"] = z["hist_va"]          # (N_valid, 10) float32
    return D


if __name__ == "__main__":
    build(force="--force" in sys.argv)

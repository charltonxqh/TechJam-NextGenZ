"""Feature audit round 2 — the columns and the file we never used.

Missed until now:
  hourmin    time of day of the impression (log column 4)
  tag        video category (video_features_basic)
  upload_dt  -> video AGE in days at impression time
  music_id   background music identity
  log_random_4_22_to_5_08_pure.csv — 1,186,060 rows of randomly-exposed
             impressions. MORE rows than our entire training split.

Each is tested as an added FM field, then log_random is tested as training
augmentation.

⚠️ RULES NOTE on log_random. It is part of KuaiRand-Pure, so "no external
training data" does not exclude it. But it spans 2022-04-22 to 05-08, which
overlaps the validation AND hidden-test windows, and the starter kit describes
it as "额外的无偏验证集" — an additional unbiased *validation* set. Training on it
is legal by the letter and questionable in spirit. We measure the effect and
report it; whether to use it in a submission is Hayden's call, not ours.
"""
from __future__ import annotations

import csv
import datetime as dt
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
DATA = KIT / "KuaiRand-Pure" / "data"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))
from evaluate import evaluate       # noqa: E402

SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428),
          "test": (20220429, 20220508)}
LOGS = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"


def load_video_meta():
    v = {}
    with open(DATA / "video_features_basic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            try:
                up = dt.date.fromisoformat(r["upload_dt"])
            except Exception:
                up = None
            v[r["video_id"]] = (r.get("author_id", "UNK"), r.get("tag", "UNK"),
                                r.get("music_id", "UNK"), up)
    return v


def read_log(path, meta, keep_dates=None):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"])
            if keep_dates and not (keep_dates[0] <= d <= keep_dates[1]):
                continue
            vid = r["video_id"]
            author, tag, music, up = meta.get(vid, ("UNK", "UNK", "UNK", None))
            try:
                age = (dt.date(d // 10000, d // 100 % 100, d % 100) - up).days if up else -1
            except Exception:
                age = -1
            rows.append((d, r["user_id"], vid, author, r["tab"],
                         float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0,
                         int(r["hourmin"]) // 100, tag, music, age))
    return rows


def encode(train_rows, other, fields):
    """fields: list of tuple indices to use. Returns X per split + dim."""
    edges = np.quantile([x[5] for x in train_rows], np.linspace(0, 1, 11)[1:-1])
    age_edges = np.quantile([max(x[10], 0) for x in train_rows], np.linspace(0, 1, 9)[1:-1])

    def raw(x):
        return {
            "user": x[1], "video": x[2], "author": x[3], "tab": x[4],
            "dur": str(int(np.searchsorted(edges, x[5]))),
            "hour": str(x[7]),
            "tag": x[8].split(",")[0] if x[8] else "UNK",
            "music": x[9],
            "age": str(int(np.searchsorted(age_edges, max(x[10], 0)))),
        }

    vocabs = {f: {} for f in fields}
    for x in train_rows:
        d = raw(x)
        for f in fields:
            if d[f] not in vocabs[f]:
                vocabs[f][d[f]] = len(vocabs[f])
    unk = {f: len(v) for f, v in vocabs.items()}
    dims = [len(vocabs[f]) + 1 for f in fields]
    offs = np.cumsum([0] + dims[:-1])

    def enc(rows):
        X = np.empty((len(rows), len(fields)), dtype=np.int64)
        y = np.empty(len(rows), dtype=np.float32)
        u = []
        for n, x in enumerate(rows):
            d = raw(x)
            for i, f in enumerate(fields):
                X[n, i] = vocabs[f].get(d[f], unk[f]) + offs[i]
            y[n] = x[6]; u.append(x[1])
        return X, y, u

    out = {"train": enc(train_rows)}
    for k, rws in other.items():
        out[k] = enc(rws)
    return out, int(sum(dims))


class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(rng.normal(0, .01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim)); self.b = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        E = self.V[X]; S = E.sum(1)
        return self.b + self.W[X].sum(1) + .5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i+bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train(); return np.concatenate(o)


def run(enc, dim, seed=0, epochs=12):
    Xtr, ytr, _ = enc["train"]; Xva, yva, uva = enc["valid"]
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xv = torch.from_numpy(Xva).long()
    m = FM(dim, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=1e-3, weight_decay=1e-6)
    ob = torch.optim.SGD([m.b], lr=1e-3); rng = np.random.default_rng(seed)
    best, bp = -1, None
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i+8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            nn.functional.binary_cross_entropy_with_logits(m(Xt[j]), yt[j]).backward()
            opt.step(); ob.step()
        p = m.predict(Xv); r = evaluate(uva, yva, p)
        if r["GAUC"] > best:
            best, bp = r["GAUC"], r
    return bp


if __name__ == "__main__":
    meta = load_video_meta()
    t0 = time.time()
    allrows = []
    for f in LOGS:
        allrows += read_log(DATA / f, meta)
    tr = [x for x in allrows if SPLITS["train"][0] <= x[0] <= SPLITS["train"][1]]
    va = [x for x in allrows if SPLITS["valid"][0] <= x[0] <= SPLITS["valid"][1]]
    print(f"loaded standard: train {len(tr):,} valid {len(va):,}  [{time.time()-t0:.0f}s]",
          flush=True)

    BASE = ["user", "video", "author", "tab", "dur"]
    trials = [
        ("baseline 5 fields", BASE),
        ("+ hourmin", BASE + ["hour"]),
        ("+ tag", BASE + ["tag"]),
        ("+ video age", BASE + ["age"]),
        ("+ music_id", BASE + ["music"]),
        ("+ hour + tag + age", BASE + ["hour", "tag", "age"]),
        ("ALL new fields", BASE + ["hour", "tag", "age", "music"]),
    ]
    print(f"\n{'variant':26s} {'GAUC':>8} {'nDCG@5':>8} {'primary':>8}   vs base GAUC")
    print("-" * 70)
    ref = None
    for name, fields in trials:
        e, dim = encode(tr, {"valid": va}, fields)
        rs = [run(e, dim, seed=s) for s in (0, 1)]
        g = float(np.mean([r["GAUC"] for r in rs]))
        n = float(np.mean([r["nDCG@5"] for r in rs]))
        p = float(np.mean([r["primary"] for r in rs]))
        if ref is None:
            ref = g
        mark = "  <== " if g > ref + 0.0016 else ""
        print(f"{name:26s} {g:8.4f} {n:8.4f} {p:8.4f}   {g-ref:+.4f}{mark}", flush=True)

    # ---- log_random augmentation -----------------------------------------
    print("\n--- training augmentation with log_random (1.19M extra rows) ---")
    t0 = time.time()
    rnd = read_log(DATA / RANDOM_LOG, meta)
    print(f"log_random rows: {len(rnd):,}  [{time.time()-t0:.0f}s]", flush=True)
    best_fields = BASE + ["hour", "tag", "age"]
    for label, extra in (("standard only", []), ("+ log_random", rnd)):
        e, dim = encode(tr + extra, {"valid": va}, best_fields)
        rs = [run(e, dim, seed=s) for s in (0, 1)]
        g = float(np.mean([r["GAUC"] for r in rs]))
        p = float(np.mean([r["primary"] for r in rs]))
        print(f"  {label:16s} train rows {len(tr)+len(extra):>9,}  "
              f"GAUC {g:.4f}  primary {p:.4f}", flush=True)

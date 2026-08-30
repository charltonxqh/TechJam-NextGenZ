"""Watch time as a training signal — the discarded magnitude.

`long_view` is a THRESHOLD on watch time. Binarising it destroys ordering
information that the metric actually rewards:

    watched 95% of the video  -> label 1
    watched 51%               -> label 1     (same label, very different signal)
    watched 49%               -> label 0
    watched  2%               -> label 0     (same label, very different signal)

GAUC and nDCG@5 score the ORDER of items inside a user's group. Two positives
are interchangeable to the loss but not to the ranking: if the model can learn
that 95% beats 51%, it orders the group better. Nothing we have built uses this
— every model trains on the binary label.

play_time_ms is dense (86% nonzero) and sits in the log next to long_view. It is
an outcome, so it is available for TRAINING rows only and is never fed as an
input feature — it is used purely as a target. That is legitimate: we train on
the training set and predict from features alone.

Four arms, same architecture (DCN), same seeds, so the only difference is the
target:

  ctrl      BCE on binary long_view                        (what we do today)
  soft      BCE on a SOFT label = clipped watch ratio      (magnitude in-band)
  aux       BCE on long_view + lam * MSE on log1p(play_ms) (multi-task)
  auxratio  BCE on long_view + lam * MSE on watch ratio    (multi-task, bounded)

Three seeds per arm because single-seed deltas on this benchmark have been wrong
by 2-4x (sigma = 0.0008).
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

from evaluate import evaluate                                   # noqa: E402
from features_v2 import load_video_meta, encode, LOGS, SPLITS    # noqa: E402
from architectures import DCN                                    # noqa: E402

FIELDS = ["user", "video", "author", "tab", "dur", "hour", "tag", "age"]
PT = 11          # index of play_time_ms in our extended row tuple


def read_log_pt(path, meta):
    """read_log + play_time_ms appended at index 11 (encode ignores extras)."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            d = int(r["date"]); vid = r["video_id"]
            author, tag, music, up = meta.get(vid, ("UNK", "UNK", "UNK", None))
            try:
                age = (dt.date(d // 10000, d // 100 % 100, d % 100) - up).days if up else -1
            except Exception:
                age = -1
            rows.append((d, r["user_id"], vid, author, r["tab"],
                         float(r["duration_ms"]), 1 if r["long_view"] != "0" else 0,
                         int(r["hourmin"]) // 100, tag, music, age,
                         float(r.get("play_time_ms", 0) or 0)))
    return rows


def vrank(users, x):
    u2i = {u: i for i, u in enumerate(sorted(set(users)))}
    idx = np.fromiter((u2i[u] for u in users), np.int64, len(users))
    o = np.lexsort((x, idx)); su = idx[o]; n = len(x)
    new = np.r_[True, su[1:] != su[:-1]]
    gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
    out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
    return out


def run(arm, Xtr, ytr, soft, aux, Xq, dim, seed, lam=0.3, epochs=6, k=16):
    Xt = torch.from_numpy(Xtr).long()
    yt = torch.from_numpy(ytr).float()
    st = torch.from_numpy(soft).float()
    at = torch.from_numpy(aux).float()
    m = DCN(dim, k=k, seed=seed, nf=len(FIELDS))
    head = nn.Linear(1, 1)                       # tiny aux head on the logit
    nn.init.ones_(head.weight); nn.init.zeros_(head.bias)
    ps = [p for n_, p in m.named_parameters() if n_ != "b"] + list(head.parameters())
    opt = torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-6)
    ob = torch.optim.SGD([m.b], lr=1e-3)
    rng = np.random.default_rng(seed)
    bce = nn.functional.binary_cross_entropy_with_logits
    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i + 8192]).long()
            opt.zero_grad(set_to_none=True); ob.zero_grad(set_to_none=True)
            z = m(Xt[j])
            if arm == "ctrl":
                L = bce(z, yt[j])
            elif arm == "soft":
                L = bce(z, st[j])
            else:                                 # aux / auxratio
                L = bce(z, yt[j]) + lam * nn.functional.mse_loss(
                    head(z.unsqueeze(-1)).squeeze(-1), at[j])
            L.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step(); ob.step()
    return m.predict(torch.from_numpy(Xq).long())


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log_pt(DATA / f, meta)
    sp = {k: [x for x in rows if lo <= x[0] <= hi] for k, (lo, hi) in SPLITS.items()}
    enc, dim = encode(sp["train"], {"valid": sp["valid"], "test": sp["test"]}, FIELDS)
    Xtr, ytr, _ = enc["train"]

    pt = np.array([x[PT] for x in sp["train"]], np.float64)
    dur = np.array([x[5] for x in sp["train"]], np.float64)
    ratio = np.clip(pt / np.maximum(dur, 1.0), 0, 3.0)
    print(f"play_time nonzero {np.mean(pt > 0):.1%}   "
          f"ratio mean {ratio.mean():.3f}  p50 {np.median(ratio):.3f}  "
          f"p90 {np.quantile(ratio, .9):.3f}", flush=True)
    print(f"corr(ratio, long_view) = {np.corrcoef(ratio, ytr)[0,1]:.3f}\n", flush=True)

    soft = np.clip(ratio / 2.0, 0.02, 0.98).astype(np.float32)   # soft label
    logpt = ((np.log1p(pt) - np.log1p(pt).mean()) / (np.log1p(pt).std() + 1e-6)).astype(np.float32)
    rat_n = ((ratio - ratio.mean()) / (ratio.std() + 1e-6)).astype(np.float32)

    SEEDS = (0, 100, 200)
    out = {}
    # One trained model per (arm, seed); predict BOTH splits from it. Training
    # separately per split would burn double the compute and — worse — report
    # validation and test from different models.
    Xcat = np.vstack([enc["valid"][0], enc["test"][0]])
    nva = len(enc["valid"][0])
    for arm in ("ctrl", "soft", "aux", "auxratio"):
        aux_t = logpt if arm == "aux" else rat_n
        t0 = time.time()
        acc = {"valid": [], "test": []}
        for s in SEEDS:
            z = np.asarray(run(arm, Xtr, ytr.astype(np.float32), soft,
                               aux_t, Xcat, dim, s), np.float64)
            for split, sl in (("valid", slice(0, nva)), ("test", slice(nva, None))):
                acc[split].append(vrank(enc[split][2], z[sl]))
        line = []
        for split in ("valid", "test"):
            _, y, u = enc[split]
            p = np.mean(acc[split], axis=0)
            out.setdefault(arm, {})[split] = p
            line.append(f"{split} {evaluate(u, y, p)['GAUC']:.4f}")
        print(f"  {arm:9s} {'  '.join(line)}  [{time.time()-t0:.0f}s]", flush=True)

    (HERE / "cache").mkdir(exist_ok=True)
    for arm, d in out.items():
        for k, v in d.items():
            np.save(HERE / "cache" / f"wt_{arm}_{k}.npy", v)
    print("\ncached:", sorted(out))


if __name__ == "__main__":
    main()

"""Final submission: rank-blended ensemble.

Validated recipe (verified on two disjoint seed sets, both 0.6042 on validation):

    0.10 * FM      5-seed rank-ensemble   (pointwise BCE, the official baseline)
    0.70 * ListCE  5-seed rank-ensemble   (BCE + intra-user ListCE, arXiv:2506.12756)
    0.20 * LightGBM lambdarank            (+ platform item statistics)

Three independent, individually-measured contributions:
  * rank-averaging instead of logit-averaging      +0.0004
  * ListCE members instead of plain FM             +0.0005 (replicated on held-out seeds)
  * 20% weight on the decorrelated LightGBM        +0.0008

Blending is on WITHIN-USER PERCENTILE RANKS: members are calibrated differently and
only intra-user order is scored.

This is the only script that touches the test split.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KIT))

import lightgbm as lgb                                  # noqa: E402
from data import load, encode                           # noqa: E402
from evaluate import evaluate                           # noqa: E402
from guards import verify_integrity                     # noqa: E402
from submit import write_submission, read_submission    # noqa: E402
from groupce_listce import FM, listce                   # noqa: E402
import torch, torch.nn as nn                            # noqa: E402

SEEDS = (0, 100, 200, 300, 400)
W = {"fm": 0.10, "listce": 0.70, "lgb": 0.20}
BASE_V, BASE_T = 0.6015, 0.5946
STAT_NUM = ["long_time_play_cnt", "valid_play_cnt", "complete_play_cnt", "play_cnt",
            "short_time_play_cnt", "like_cnt", "comment_cnt", "share_cnt",
            "collect_cnt", "follow_cnt"]


def vrank_factory(users):
    u2i = {u: i for i, u in enumerate(sorted(set(users)))}
    ui = np.fromiter((u2i[u] for u in users), dtype=np.int64, count=len(users))

    def vrank(x):
        o = np.lexsort((x, ui)); su = ui[o]; n = len(x)
        new = np.r_[True, su[1:] != su[:-1]]
        gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
        _, c = np.unique(su, return_counts=True)
        sz = np.repeat(c, c)
        out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
        return out
    return vrank


def train_member(kind, Xtr, ytr, Xpred, uva_dummy, dim, seed):
    """kind: 'fm' (pointwise) or 'listce'. Trains to a fixed epoch budget."""
    Xt = torch.from_numpy(Xtr).long(); yt = torch.from_numpy(ytr).float()
    Xp = torch.from_numpy(Xpred).long()
    uid = torch.from_numpy(Xtr[:, 0].astype(np.int64))
    m = FM(dim, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=1e-3, weight_decay=1e-6)
    opt_b = torch.optim.SGD([m.b], lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(7):                        # baseline peaks ~epoch 7
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), 8192):
            j = torch.from_numpy(idx[i:i + 8192]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            z = m(Xt[j]); yb = yt[j]
            loss = nn.functional.binary_cross_entropy_with_logits(z, yb)
            if kind == "listce":
                loss = loss + listce(z, yb, uid[j])
            loss.backward(); opt.step(); opt_b.step()
    return m.predict(Xp)


def stat_block(vids):
    stat = {}
    with open(KIT / "KuaiRand-Pure" / "data" / "video_features_statistic_pure.csv") as fh:
        for r in csv.DictReader(fh):
            stat[r["video_id"]] = r
    out = np.zeros((len(vids), len(STAT_NUM) + 2), dtype=np.float32)
    for i, v in enumerate(vids):
        s = stat.get(v)
        if not s:
            continue
        show = max(float(s.get("show_cnt", 1) or 1), 1.0)
        for j, c in enumerate(STAT_NUM):
            try:
                out[i, j] = float(s.get(c, 0) or 0) / show
            except ValueError:
                pass
        try:
            out[i, len(STAT_NUM)] = float(s.get("play_progress", 0) or 0)
        except ValueError:
            pass
        out[i, len(STAT_NUM) + 1] = np.log1p(show)
    return out


def main():
    verify_integrity()
    print("integrity OK\n")
    splits = load(str(KIT / "KuaiRand-Pure" / "data"))
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    Xtr = Xtr.astype(np.int64); Xva = Xva.astype(np.int64); Xte = Xte.astype(np.int64)

    for split, X, y, u, base in (("valid", Xva, yva, uva, BASE_V),
                                 ("test",  Xte, yte, ute, BASE_T)):
        vrank = vrank_factory(u)
        parts = {}
        for kind in ("fm", "listce"):
            t0 = time.time()
            pr = [vrank(np.asarray(train_member(kind, Xtr, ytr, X, u, dim, s), np.float64))
                  for s in SEEDS]
            parts[kind] = np.mean(pr, axis=0)
            print(f"  {split}/{kind:7s} {evaluate(u, y, parts[kind])['primary']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        # LightGBM lambdarank with platform statistics
        vid_tr = [r[2] for r in splits["train"]]
        vid_x = [r[2] for r in splits[split]]
        Ftr = np.hstack([Xtr, stat_block(vid_tr)]).astype(np.float32)
        Fx = np.hstack([X, stat_block(vid_x)]).astype(np.float32)
        u2i = {uu: i for i, uu in enumerate(sorted(set(utr)))}
        ui = np.fromiter((u2i[uu] for uu in utr), dtype=np.int64, count=len(utr))
        o = np.argsort(ui, kind="stable"); _, c = np.unique(ui[o], return_counts=True)
        ds = lgb.Dataset(Ftr[o], label=ytr[o], group=c,
                         categorical_feature=[0, 1, 2, 3, 4], free_raw_data=False)
        b = lgb.train(dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[5],
                           learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                           feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                           verbose=-1, num_threads=8), ds, num_boost_round=300)
        parts["lgb"] = vrank(b.predict(Fx).astype(np.float64))
        print(f"  {split}/lgb     {evaluate(u, y, parts['lgb'])['primary']:.4f}")

        blend = sum(W[k] * parts[k] for k in W)
        r = evaluate(u, y, blend)
        print(f"  {split.upper():5s} BLEND  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} "
              f"| primary {r['primary']:.4f}   ({r['primary']-base:+.4f} vs baseline)\n")
        if split == "test":
            d = ((r["GAUC"] - 0.6610) + (r["nDCG@5"] - 0.5282)) / 2
            print(f"  SCORED DELTA: {d:+.4f}\n")
            out = HERE / "state" / "submission_blend.csv"
            write_submission(str(out), splits["test"], blend)
            read_submission(str(out), splits["test"])
            print(f"  wrote {out} — passes official validator")


if __name__ == "__main__":
    main()

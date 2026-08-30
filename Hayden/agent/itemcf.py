"""Collaborative filtering — a signal source the FM captures only implicitly.

Every model we have built is a FEATURE model: it maps (user_id, video_id, tab,
duration, ...) through embeddings to a score. The user embedding is the only
thing carrying "who this person is", and it is fitted from ~43 training rows.

A NEIGHBOURHOOD model computes something structurally different: for a
candidate video i, how similar is i to the videos this user already long-viewed
in TRAINING? That is not a per-user constant (it varies across the candidates
inside a user's group), so unlike a user bias it is visible to GAUC/nDCG.

This is the Netflix Prize lesson: neighbourhood models and latent-factor models
make different errors and blend well, even when the neighbourhood model is
weaker alone.

Crucially this does NOT need repeat exposure. Our coverage note killed
user x author history because only 3.4% of rows involve a seen author. ItemCF
needs no repeat: it needs the user's training items and an item-item similarity,
both of which exist for nearly every user.

Three scorers, all fitted on TRAIN ONLY:
  cf_cos    cosine item-item CF over the user x item long_view matrix
  cf_exp    same but over the EXPOSURE matrix (every impression, not just
            positives) — denser, captures "what gets shown together"
  cf_svd    truncated SVD of the long_view matrix, score = user . item
            (a latent-factor CF, i.e. ALS-like, with no side features at all)

First it prints coverage, because if users have no training positives the whole
idea is dead on arrival and we should know that before reading any score.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from evaluate import evaluate                                   # noqa: E402
from features_v2 import load_video_meta, read_log, DATA, LOGS, SPLITS  # noqa: E402

U, V, LBL = 1, 2, 6      # row tuple indices: user_id, video_id, long_view


def build(rows, ui, vi, positives_only):
    r, c = [], []
    for x in rows:
        if positives_only and x[LBL] == 0:
            continue
        a, b = ui.get(x[U]), vi.get(x[V])
        if a is not None and b is not None:
            r.append(a); c.append(b)
    M = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)),
                      shape=(len(ui), len(vi)))
    M.sum_duplicates(); M.data[:] = 1.0     # binary
    return M


def cf_scores(M, users, items, ui, vi, topk=200):
    """score(u,i) = sum_j in P_u cosine(i, j), excluding j == i."""
    norm = np.sqrt(np.asarray(M.multiply(M).sum(0)).ravel()) + 1e-6
    Mn = M @ sp.diags(1.0 / norm)                     # column-normalised
    S = (Mn.T @ Mn).toarray().astype(np.float32)      # item x item cosine
    np.fill_diagonal(S, 0.0)
    if topk and topk < S.shape[0]:                    # keep top-k neighbours
        cut = np.partition(S, -topk, axis=1)[:, -topk][:, None]
        S[S < cut] = 0.0
    P = (M @ S)                                       # user x item affinity
    cnt = np.asarray(M.sum(1)).ravel() + 1e-6
    P = P / cnt[:, None]
    out = np.zeros(len(users), np.float64)
    for n, (u, v) in enumerate(zip(users, items)):
        a, b = ui.get(u), vi.get(v)
        if a is not None and b is not None:
            out[n] = P[a, b]
    return out


def svd_scores(M, users, items, ui, vi, k=64, seed=0):
    U_, s, Vt = randomized_svd(M, n_components=k, random_state=seed)
    F = U_ * s
    out = np.zeros(len(users), np.float64)
    for n, (u, v) in enumerate(zip(users, items)):
        a, b = ui.get(u), vi.get(v)
        if a is not None and b is not None:
            out[n] = float(F[a] @ Vt[:, b])
    return out


def vrank(users, x):
    u2i = {u: i for i, u in enumerate(sorted(set(users)))}
    idx = np.fromiter((u2i[u] for u in users), np.int64, len(users))
    o = np.lexsort((x, idx)); su = idx[o]; n = len(x)
    new = np.r_[True, su[1:] != su[:-1]]
    gs = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    _, c = np.unique(su, return_counts=True); sz = np.repeat(c, c)
    out = np.empty(n); out[o] = (np.arange(n) - gs) / np.maximum(sz - 1, 1)
    return out


def main():
    meta = load_video_meta()
    rows = []
    for f in LOGS:
        rows += read_log(DATA / f, meta)
    sp_ = {k: [x for x in rows if lo <= x[0] <= hi] for k, (lo, hi) in SPLITS.items()}
    tr = sp_["train"]

    ui = {u: i for i, u in enumerate(sorted({x[U] for x in tr}))}
    vi = {v: i for i, v in enumerate(sorted({x[V] for x in rows}))}
    print(f"train users {len(ui):,}  items {len(vi):,}", flush=True)

    # --- coverage: does the premise hold? -----------------------------
    pos = {}
    for x in tr:
        if x[LBL]:
            pos.setdefault(x[U], 0)
            pos[x[U]] += 1
    for k in ("valid", "test"):
        us = [x[U] for x in sp_[k]]
        have = np.mean([pos.get(u, 0) > 0 for u in us])
        npos = np.mean([pos.get(u, 0) for u in us])
        print(f"  {k}: {have:.1%} of rows have a user with >=1 train positive; "
              f"mean {npos:.1f} train positives per row's user", flush=True)

    Mpos = build(tr, ui, vi, True)
    Mexp = build(tr, ui, vi, False)
    print(f"  long_view matrix nnz {Mpos.nnz:,}   exposure matrix nnz {Mexp.nnz:,}\n",
          flush=True)

    preds = {}
    for name, fn in (("cf_cos", lambda k: cf_scores(Mpos, [x[U] for x in sp_[k]],
                                                    [x[V] for x in sp_[k]], ui, vi)),
                     ("cf_exp", lambda k: cf_scores(Mexp, [x[U] for x in sp_[k]],
                                                    [x[V] for x in sp_[k]], ui, vi)),
                     ("cf_svd", lambda k: svd_scores(Mpos, [x[U] for x in sp_[k]],
                                                     [x[V] for x in sp_[k]], ui, vi))):
        t0 = time.time(); line = []
        for k in ("valid", "test"):
            y = np.array([x[LBL] for x in sp_[k]])
            u = [x[U] for x in sp_[k]]
            p = fn(k)
            preds.setdefault(name, {})[k] = vrank(u, p)
            line.append(f"{k} {evaluate(u, y, p)['GAUC']:.4f}")
        print(f"  {name:8s} {'  '.join(line)}  [{time.time()-t0:.0f}s]", flush=True)

    (HERE / "cache").mkdir(exist_ok=True)
    for name, d in preds.items():
        for k, v in d.items():
            np.save(HERE / "cache" / f"cf_{name}_{k}.npy", v)
    print("\ncached:", sorted(preds))


if __name__ == "__main__":
    main()

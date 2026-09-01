"""Faithful rebuild of the other team's FM + BPR base, to test whether it reproduces.

Two of their headline mechanisms did not reproduce on our gradient-boosting
pipeline: shrinking tree capacity to num_leaves=2 came out flat (0.6013 vs 0.6014
at 63 leaves), and their causal history features scored 0.6005, below our own
baseline. Both were tested on OUR base, which is a different model from theirs.

Their base is the organisers' FM trained with a PAIRWISE BPR loss rather than
pointwise logloss, and that difference matters more than we credited. BPR samples
a (positive, negative) pair from WITHIN one user and pushes the positive above the
negative. That is a direct optimisation of intra-user ordering — which is exactly
and only what GAUC measures. Pointwise logloss instead spends capacity on
calibrating absolute probabilities, which the metric discards entirely.

We had tested "intra-user pairwise BPR" earlier and measured +0.0005 over three
seeds, i.e. nothing. But we tested it on the five base fields alone. Their result
comes from BPR *plus* the bucketed causal history features, and the pair is what
they claim compounds.

Reproduced here as closely as the write-up and code allow:
  * the organisers' own FM class from baseline.py, unmodified
  * BPR pairs sampled uniformly per user, one positive and one negative each
  * the gradient of -log sigmoid(z_pos - z_neg)
  * causal history features (train-only history), quantile-bucketed to 20 levels
  * early stopping on validation primary, patience 4
  * a 5-seed ensemble averaging sigmoid-transformed scores

Arms measured separately so the source of any gain is attributable:
  A  pointwise logloss, base fields          — reproduces the official baseline
  B  BPR,               base fields          — isolates the loss change
  C  BPR,               base + history       — their full base
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(KIT))

from baseline import FM, sigmoid            # noqa: E402  the organisers' own FM
from evaluate import evaluate               # noqa: E402
import causal_history as ch                 # noqa: E402

N_BUCKETS = 20            # their retuned value
EPOCHS, BS, PATIENCE = 40, 8192, 4
LR, K, L2 = 0.001, 16, 1e-6


def bucketize(v, n=N_BUCKETS):
    """Quantile-bucket a continuous column into n levels (FM needs categoricals)."""
    finite = v[np.isfinite(v)]
    edges = np.quantile(finite, np.linspace(0, 1, n + 1)[1:-1]) if len(finite) else []
    return np.searchsorted(edges, v).astype(np.int64)


def build_matrices(rows, hist, use_history):
    """One shared embedding table; every field offset into it."""
    split_of = np.array([r["split"] for r in rows])
    cols, names = [], []

    for key in ("user", "video", "tab"):
        vals = [r[key] for r in rows]
        vocab = {v: i for i, v in enumerate(sorted(set(vals)))}
        cols.append(np.fromiter((vocab[v] for v in vals), np.int64, len(vals)))
        names.append(key)

    if use_history:
        for f in ("decay_rate", "decay_act", "tab_decay_rate",
                  "last1", "lastk_rate", "gap_days"):
            cols.append(bucketize(hist[f].astype(np.float64)))
            names.append(f)

    dim = 0
    out = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for j, c in enumerate(cols):
        out[:, j] = c + dim
        dim += int(c.max()) + 1

    y = np.array([r["y"] for r in rows], dtype=np.float32)
    u = [r["user"] for r in rows]
    idx = {s: np.where(split_of == s)[0] for s in ("train", "valid", "test")}
    return out, y, u, dim, idx, names


def bpr_index(y, users):
    """Per-user positive and negative row lists, flattened."""
    order = np.argsort(np.array([hash(x) for x in users]), kind="stable")
    pos, neg = {}, {}
    for i in order:
        (pos if y[i] > 0 else neg).setdefault(users[i], []).append(i)
    keep = [u for u in pos if u in neg]
    pos_flat = np.concatenate([np.array(pos[u]) for u in keep])
    neg_flat = np.concatenate([np.array(neg[u]) for u in keep])
    pos_len = np.array([len(pos[u]) for u in keep])
    neg_len = np.array([len(neg[u]) for u in keep])
    return (pos_flat, np.r_[0, np.cumsum(pos_len)[:-1]], pos_len,
            neg_flat, np.r_[0, np.cumsum(neg_len)[:-1]], neg_len, len(keep))


def bpr_step(m, Xpos, Xneg):
    """Gradient of -log sigmoid(z_pos - z_neg), matching their implementation."""
    B = len(Xpos)
    zpos, Epos, Spos = m.logits(Xpos)
    zneg, Eneg, Sneg = m.logits(Xneg)
    d = zpos - zneg
    gpos = ((sigmoid(d) - 1) / B).astype(np.float32)
    X = np.concatenate([Xpos, Xneg]); E = np.concatenate([Epos, Eneg])
    S = np.concatenate([Spos, Sneg]); g = np.concatenate([gpos, -gpos])
    gV = np.zeros_like(m.V); gW = np.zeros_like(m.W)
    np.add.at(gW, X, g[:, None])
    np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
    gV += m.l2 * m.V; gW += m.l2 * m.W
    m.t += 1
    b1, b2, eps = 0.9, 0.999, 1e-8
    for P, G, M, Vv in ((m.V, gV, m.mV, m.vV), (m.W, gW, m.mW, m.vW)):
        M *= b1; M += (1 - b1) * G
        Vv *= b2; Vv += (1 - b2) * (G * G)
        P -= m.lr * (M / (1 - b1 ** m.t)) / (np.sqrt(Vv / (1 - b2 ** m.t)) + eps)
    m.b -= m.lr * g.sum()


def train_arm(X, y, u, dim, idx, seed, use_bpr):
    tr, va = idx["train"], idx["valid"]
    m = FM(dim, k=K, lr=LR, l2=L2, seed=seed)
    uva = [u[i] for i in va]
    best, best_scores, bad = -1.0, None, 0
    rng = np.random.default_rng(seed)

    if use_bpr:
        pf, ps, pl, nf, ns, nl, nu = bpr_index(y[tr], [u[i] for i in tr])
        steps = max(1, len(tr) // BS)

    for _ in range(EPOCHS):
        if use_bpr:
            for _ in range(steps):
                pick = rng.integers(0, nu, size=BS)
                pr = pf[ps[pick] + (rng.random(BS) * pl[pick]).astype(np.int64)]
                nr = nf[ns[pick] + (rng.random(BS) * nl[pick]).astype(np.int64)]
                bpr_step(m, X[tr][pr], X[tr][nr])
        else:
            order = rng.permutation(len(tr))
            for i in range(0, len(order), BS):
                j = tr[order[i:i + BS]]
                m.step(X[j], y[j])
        s = m.predict(X[va])
        p = evaluate(uva, y[va], s)["primary"]
        if p > best:
            best, best_scores, bad = p, s, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best, best_scores


def main():
    rows = ch.load_rows()
    hist = ch.build(rows, {"train"})          # train-only history: the legal variant
    print(f"loaded {len(rows):,} rows\n", flush=True)

    for label, use_hist, use_bpr in (
            ("A  pointwise, base fields", False, False),
            ("B  BPR,       base fields", False, True),
            ("C  BPR,       base + history", True, True)):
        X, y, u, dim, idx, names = build_matrices(rows, hist, use_hist)
        t0 = time.time()
        scores, primaries = [], []
        for seed in (0, 1, 2):
            p, s = train_arm(X, y, u, dim, idx, seed, use_bpr)
            primaries.append(p); scores.append(sigmoid(s))
        uva = [u[i] for i in idx["valid"]]
        ens = evaluate(uva, y[idx["valid"]], np.mean(scores, axis=0))
        print(f"{label:30s} fields={len(names):2d}  "
              f"single-seed mean {np.mean(primaries):.4f}  "
              f"3-seed ensemble {ens['primary']:.4f}  "
              f"(GAUC {ens['GAUC']:.4f})  [{time.time()-t0:.0f}s]", flush=True)

    print("\n  official FM baseline (valid): 0.6015", flush=True)
    print("  our best agent run    (valid): 0.6051", flush=True)
    print("  their iter27 claim    (test) : 0.63889", flush=True)


if __name__ == "__main__":
    main()

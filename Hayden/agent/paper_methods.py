"""Methods taken from the surveyed papers and the starter kit's untested list.

Each variant changes ONE thing about how the FM is trained, so the measured
delta is attributable. All evaluated on validation only, 3 seeds each, against
the official baseline (0.6015).

Variants
  baseline      pointwise BCE on the binary long_view label (the official model)

  graded        Train on WATCH RATIO (play_time / duration, clipped to [0,1])
                as a soft target instead of the binary label.
                Rationale: long_view is a thresholded version of watch time, so
                the binary label throws away the magnitude. A soft target keeps
                it. Starter-kit untested direction #4; never tested by us
                (we only ever used play_time as a separate auxiliary head).

  censored      CWM's core idea (KDD 2024). A play that reaches the end of the
                video is RIGHT-CENSORED: the user might have watched longer had
                the video been longer, so squared error against the observed
                value is the wrong likelihood. Use a one-sided loss on those
                rows - only penalise UNDER-prediction.

  ips           Exposure-bias correction, motivated by MMRF (arXiv:2405.01847).
                Training data comes from a deployed recommender, so popular
                items are over-exposed. Weight each row by 1/pop^alpha, using
                item exposure frequency in TRAIN as a propensity proxy.
                Rules-safe: does not touch log_random (whose dates overlap test).

  coldboost     Up-weight users with short training history. Paper
                arXiv:2506.12756 reports its largest gains on cold-start users
                (GAUC 0.6718 -> 0.6786), and our own EDA found short-history
                users are our weakest segment (0.5940 vs 0.6071).

  graded+ens    Whichever of the above wins, combined with the 5-seed ensemble
                that is our current submission.
"""
from __future__ import annotations

import sys
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

from evaluate import evaluate      # noqa: E402
from prep import load_cache        # noqa: E402

BASE = 0.6015
SIGMA = 0.0008


class FM(nn.Module):
    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        return self.b + self.W[X].sum(1) + 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(o)


def train(D, target, weights=None, censored=None, seed=0,
          epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4):
    """One FM. `target` is the training signal (binary or soft), `weights` optional
    per-row weights, `censored` a bool mask for one-sided loss rows."""
    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = torch.from_numpy(np.asarray(target, dtype=np.float32))
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]
    w = None if weights is None else torch.from_numpy(np.asarray(weights, np.float32))
    cen = None if censored is None else torch.from_numpy(np.asarray(censored, np.bool_))

    m = FM(D["dim"], seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0

    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            z = m(Xtr[j])
            t = ytr[j]
            per = nn.functional.binary_cross_entropy_with_logits(z, t, reduction="none")
            if cen is not None:
                # one-sided: on censored rows only penalise under-prediction
                mask = cen[j] & (torch.sigmoid(z) > t)
                per = torch.where(mask, torch.zeros_like(per), per)
            if w is not None:
                per = per * w[j]
            per.mean().backward()
            opt.step(); opt_b.step()

        p = evaluate(uva, yva, m.predict(Xva))["primary"]
        if p > best + 1e-5:
            best, bad = p, 0
            state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    return m.predict(Xva), best


def variants(D):
    y = D["ytr"]
    pt = D["aux_play_time_ms"]
    dur = np.maximum(D["aux_duration_ms"], 1.0)
    ratio = np.clip(pt / dur, 0.0, 1.0)

    # ---- exposure propensity: item frequency in train ----------------------
    vid = D["Xtr"][:, 1]
    cnt = np.bincount(vid, minlength=int(vid.max()) + 1).astype(np.float32)
    pop = cnt[vid]
    ips = (pop.mean() / pop) ** 0.5                      # alpha = 0.5
    ips = ips / ips.mean()

    # ---- cold-start upweighting -------------------------------------------
    u = D["Xtr"][:, 0]
    ucnt = np.bincount(u, minlength=int(u.max()) + 1).astype(np.float32)
    hist = ucnt[u]
    cold = (hist.mean() / np.maximum(hist, 1.0)) ** 0.3
    cold = cold / cold.mean()

    censored = pt >= dur * 0.95                          # play reached the end

    return {
        "baseline (binary BCE)":            dict(target=y),
        "graded: watch-ratio soft target":  dict(target=ratio),
        "graded 50/50 blend with label":    dict(target=0.5 * ratio + 0.5 * y),
        "censored one-sided (CWM idea)":    dict(target=ratio, censored=censored),
        "IPS exposure debias (a=0.5)":      dict(target=y, weights=ips),
        "cold-start upweight (a=0.3)":      dict(target=y, weights=cold),
    }


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]
    seeds = (0, 1, 2)
    print(f"censored rows: {100*np.mean(D['aux_play_time_ms'] >= D['aux_duration_ms']*0.95):.1f}%")
    print(f"watch ratio: mean {np.clip(D['aux_play_time_ms']/np.maximum(D['aux_duration_ms'],1),0,1).mean():.3f}\n")
    print(f"{'variant':38s} {'mean':>8} {'std':>7} {'vs base':>9}   seeds")
    print("-" * 82)

    results = {}
    for name, kw in variants(D).items():
        t0 = time.time()
        ps = []
        for s in seeds:
            _, p = train(D, seed=s, **kw)
            ps.append(p)
        mu, sd = float(np.mean(ps)), float(np.std(ps))
        results[name] = (mu, sd)
        mark = "  <== BEATS BASELINE" if mu > BASE + 2 * SIGMA else ""
        print(f"{name:38s} {mu:8.4f} {sd:7.4f} {mu-BASE:+9.4f}   "
              f"{' '.join(f'{p:.4f}' for p in ps)}  [{time.time()-t0:.0f}s]{mark}")

    print()
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"best: {best[0]}  {best[1][0]:.4f} ({best[1][0]-BASE:+.4f} vs baseline)")


if __name__ == "__main__":
    main()

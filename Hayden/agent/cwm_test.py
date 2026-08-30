"""CWM done properly — censored watch-time regression as an AUXILIARY task.

The earlier attempt (paper_methods.py) was vacuous: it applied the one-sided
rule to a watch RATIO clipped to [0,1] predicted through a sigmoid. Since
sigmoid(z) < 1 and the censored rows have target ~1.0, "prediction exceeds
observation" was unreachable and the censoring never fired once — proven by the
result being identical to the uncensored variant at every seed.

Two corrections here:

1. **Unbounded target.** Regress log1p(play_time_ms) with a linear head, so the
   model CAN over-predict and the one-sided rule becomes meaningful.

2. **Auxiliary, not primary.** We measured that ranking directly by watch time
   is far worse than ranking by P(long_view) (-0.0444), because long_view is a
   duration-dependent rule, not a ratio threshold. So watch time should shape
   the shared representation, not replace the ranking signal. Main task stays
   binary BCE on long_view.

The censoring idea (Zhao et al., KDD 2024): a play that reaches the end of the
video is right-censored — the user might have watched longer given the chance.
Squared error against the observed value therefore punishes the model for
correctly predicting a higher latent watch time. The fix is a one-sided loss:
on censored rows, only penalise UNDER-prediction.

Validation only. 3 seeds per variant.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

from evaluate import evaluate     # noqa: E402
from prep import load_cache       # noqa: E402

BASE, SIGMA = 0.6015, 0.0008


class FMAux(nn.Module):
    """FM with a shared embedding table and a second linear head for watch time."""

    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.V = nn.Parameter(torch.from_numpy(
            rng.normal(0, 0.01, (dim, k)).astype(np.float32)))
        self.W = nn.Parameter(torch.zeros(dim))
        self.b = nn.Parameter(torch.zeros(()))
        self.Wt = nn.Parameter(torch.zeros(dim))     # watch-time head
        self.bt = nn.Parameter(torch.zeros(()))

    def _inter(self, X):
        E = self.V[X]
        S = E.sum(1)
        return 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))

    def forward(self, X):
        inter = self._inter(X)
        return (self.b + self.W[X].sum(1) + inter,      # long_view logit
                self.bt + self.Wt[X].sum(1) + inter)    # watch-time prediction

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        o = [self(X[i:i + bs])[0].cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(o)


def train(D, mode, lam=0.2, seed=0, epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4):
    """mode: 'none' | 'plain' (MSE) | 'censored' (one-sided MSE on censored rows)"""
    Xtr = torch.from_numpy(D["Xtr"]).long()
    ytr = torch.from_numpy(D["ytr"]).float()
    Xva = torch.from_numpy(D["Xva"]).long()
    uva, yva = D["uva"], D["yva"]

    pt = D["aux_play_time_ms"].astype(np.float64)
    dur = np.maximum(D["aux_duration_ms"].astype(np.float64), 1.0)
    t_raw = np.log1p(pt)
    t = (t_raw - t_raw.mean()) / (t_raw.std() + 1e-9)         # standardise
    wt = torch.from_numpy(t.astype(np.float32))
    cen = torch.from_numpy((pt >= 0.95 * dur).astype(np.bool_))

    m = FMAux(D["dim"], seed=seed)
    opt = torch.optim.Adam([m.V, m.W, m.Wt], lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b, m.bt], lr=lr)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0

    for _ in range(epochs):
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            z, tz = m(Xtr[j])
            loss = nn.functional.binary_cross_entropy_with_logits(z, ytr[j])
            if mode != "none":
                resid = tz - wt[j]
                sq = resid ** 2
                if mode == "censored":
                    # right-censored: the true watch time is >= observed, so an
                    # over-prediction is not evidence of error. Only penalise
                    # under-prediction on those rows.
                    sq = torch.where(cen[j] & (resid > 0), torch.zeros_like(sq), sq)
                loss = loss + lam * sq.mean()
            loss.backward()
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
    return best


def main():
    D = load_cache()
    pt = D["aux_play_time_ms"].astype(np.float64)
    dur = np.maximum(D["aux_duration_ms"].astype(np.float64), 1.0)
    print(f"censored (play >= 0.95*duration): {100*np.mean(pt >= 0.95*dur):.1f}% of rows\n")

    seeds = (0, 1, 2)
    print(f"{'variant':44s} {'mean':>8} {'std':>7} {'vs base':>9}   seeds")
    print("-" * 88)
    out = {}
    for label, kw in (
        ("no auxiliary head (control)",            dict(mode="none")),
        ("aux plain MSE on log watch time",        dict(mode="plain", lam=0.2)),
        ("aux CENSORED one-sided (CWM), lam=0.2",  dict(mode="censored", lam=0.2)),
        ("aux CENSORED one-sided (CWM), lam=0.5",  dict(mode="censored", lam=0.5)),
    ):
        t0 = time.time()
        ps = [train(D, seed=s, **kw) for s in seeds]
        mu, sd = float(np.mean(ps)), float(np.std(ps))
        out[label] = mu
        mark = "  <== BEATS BASELINE" if mu > BASE + 2 * SIGMA else ""
        print(f"{label:44s} {mu:8.4f} {sd:7.4f} {mu-BASE:+9.4f}   "
              f"{' '.join(f'{p:.4f}' for p in ps)}  [{time.time()-t0:.0f}s]{mark}")

    b = max(out.items(), key=lambda kv: kv[1])
    print(f"\nbest: {b[0]}  {b[1]:.4f} ({b[1]-BASE:+.4f})")
    # sanity: censored and plain MUST differ, or the mask is vacuous again
    if abs(out.get("aux plain MSE on log watch time", 0)
           - out.get("aux CENSORED one-sided (CWM), lam=0.2", 1)) < 1e-6:
        print("WARNING: censored == plain to 6dp — the one-sided mask is not firing.")


if __name__ == "__main__":
    main()

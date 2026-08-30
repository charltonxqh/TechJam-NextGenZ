"""Platform-wide item statistics as features — the largest untapped signal source.

We had been using 5 of the 95 columns the dataset ships. `video_features_
statistic_pure.csv` carries 52 aggregate statistics per video, measured over the
whole Kuaishou platform rather than over our 1.14M training rows.

Why this is different from the feature engineering the organisers already ruled
out: they added more *categorical* fields (music_id, video_type, upload_type,
user buckets) — re-encodings of information the model could already reach. These
are *external measurements*. A video appears ~150 times in our training split, so
the `video_id` embedding must estimate item quality from ~150 noisy observations;
`long_time_play_cnt / show_cnt` measures the same quantity over millions of plays.

Measured alone as a ranking score on validation:
    long_time_play_cnt/show   GAUC 0.6385   (item popularity: 0.6308)
    valid_play_cnt/show       GAUC 0.6249
    play_progress             GAUC 0.6076

⚠️ LEAKAGE CAVEAT — read before using in a submission.
   These statistics are a static file with no timestamp. If they were aggregated
   over a window that includes the hidden-test period, using them leaks future
   information. They are shipped as a feature file by the dataset authors and
   using them is conventional, but the risk is real and unverifiable from our
   side. We therefore measure BOTH:
     (a) provided statistics  — possible leakage, flagged
     (b) train-only item rates — computed from our training split alone,
                                 unambiguously legal
   If (b) captures most of (a)'s gain, prefer (b) and avoid the question entirely.
"""
from __future__ import annotations

import csv
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KIT))

from data import load                 # noqa: E402
from evaluate import evaluate         # noqa: E402
from prep import load_cache           # noqa: E402

BASE, SIGMA = 0.6015, 0.0008
NB = 20                                # quantile buckets per continuous feature

STAT_FEATURES = [
    ("long_time_play_cnt", "show_cnt"),
    ("valid_play_cnt", "show_cnt"),
    ("complete_play_cnt", "show_cnt"),
    ("play_cnt", "show_cnt"),
    ("play_progress", None),
]


def build_features(mode):
    """mode: 'provided' | 'trainonly' | 'both' -> (extra_tr, extra_va) int arrays."""
    splits = load(str(KIT / "KuaiRand-Pure" / "data"))
    vid_tr = [r[2] for r in splits["train"]]
    vid_va = [r[2] for r in splits["valid"]]
    y_tr = np.array([r[6] for r in splits["train"]], dtype=np.float64)

    cols = []

    if mode in ("provided", "both"):
        stat = {}
        with open(KIT / "KuaiRand-Pure" / "data" / "video_features_statistic_pure.csv") as fh:
            for r in csv.DictReader(fh):
                stat[r["video_id"]] = r

        def ratio(vids, num, den):
            out = np.zeros(len(vids))
            for i, v in enumerate(vids):
                s = stat.get(v)
                if not s:
                    continue
                try:
                    n = float(s.get(num, 0) or 0)
                    d = float(s.get(den, 1) or 1) if den else 1.0
                    out[i] = n / max(d, 1.0)
                except ValueError:
                    pass
            return out

        for num, den in STAT_FEATURES:
            cols.append((f"stat:{num}/{den or '1'}", ratio(vid_tr, num, den),
                         ratio(vid_va, num, den)))

    if mode in ("trainonly", "both"):
        # item long-view rate estimated from TRAIN ONLY, smoothed. Leakage-free.
        pos, cnt = {}, {}
        for v, y in zip(vid_tr, y_tr):
            cnt[v] = cnt.get(v, 0) + 1
            pos[v] = pos.get(v, 0.0) + y
        g = y_tr.mean()
        prior = 20.0

        def rate(vids, loo):
            out = np.zeros(len(vids))
            for i, v in enumerate(vids):
                c, p = cnt.get(v, 0), pos.get(v, 0.0)
                if loo:
                    c, p = c - 1, p - y_tr[i]
                out[i] = (p + prior * g) / (c + prior)
            return out

        cols.append(("train:item_longview_rate", rate(vid_tr, True), rate(vid_va, False)))
        cols.append(("train:item_impressions",
                     np.array([cnt.get(v, 0) for v in vid_tr], dtype=np.float64),
                     np.array([cnt.get(v, 0) for v in vid_va], dtype=np.float64)))

    # bucket each continuous column into NB quantiles (FM needs categoricals)
    names, tr_list, va_list = [], [], []
    for name, a, b in cols:
        edges = np.unique(np.quantile(a, np.linspace(0, 1, NB + 1)[1:-1]))
        tr_list.append(np.searchsorted(edges, a))
        va_list.append(np.searchsorted(edges, b))
        names.append(name)
    return names, np.stack(tr_list, 1).astype(np.int64), np.stack(va_list, 1).astype(np.int64)


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


def train(Xtr_np, ytr_np, Xva_np, uva, yva, dim, seed=0,
          epochs=40, bs=8192, lr=1e-3, l2=1e-6, patience=4):
    Xtr = torch.from_numpy(Xtr_np).long()
    ytr = torch.from_numpy(ytr_np).float()
    Xva = torch.from_numpy(Xva_np).long()
    m = FM(dim, seed=seed)
    opt = torch.optim.Adam([m.V, m.W], lr=lr, weight_decay=l2)
    opt_b = torch.optim.SGD([m.b], lr=lr)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    for _ in range(epochs):
        idx = rng.permutation(len(ytr_np))
        for i in range(0, len(idx), bs):
            j = torch.from_numpy(idx[i:i + bs]).long()
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            nn.functional.binary_cross_entropy_with_logits(m(Xtr[j]), ytr[j]).backward()
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
    return best, m.predict(Xva)


def assemble(D, extra_tr, extra_va):
    """Append bucketed extra fields to the encoded matrix with fresh offsets."""
    Xtr, Xva, off = D["Xtr"], D["Xva"], D["dim"]
    tr, va = [Xtr], [Xva]
    for c in range(extra_tr.shape[1]):
        a, b = extra_tr[:, c], extra_va[:, c]
        n = int(max(a.max(), b.max())) + 1
        tr.append((a + off).reshape(-1, 1))
        va.append((np.clip(b, 0, n - 1) + off).reshape(-1, 1))
        off += n
    return (np.hstack(tr).astype(np.int64), np.hstack(va).astype(np.int64), off)


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]
    seeds = (0, 1, 2)
    print(f"{'variant':46s} {'mean':>8} {'std':>7} {'vs base':>9}   seeds")
    print("-" * 90)

    ps = [train(D["Xtr"].astype(np.int64), D["ytr"], D["Xva"].astype(np.int64),
                uva, yva, D["dim"], seed=s)[0] for s in seeds]
    print(f"{'control: 5 fields (official baseline)':46s} {np.mean(ps):8.4f} "
          f"{np.std(ps):7.4f} {np.mean(ps)-BASE:+9.4f}   "
          f"{' '.join(f'{p:.4f}' for p in ps)}")

    for mode, label in (("trainonly", "+ train-only item rate (LEAKAGE-FREE)"),
                        ("provided", "+ platform statistics (leakage risk)"),
                        ("both", "+ both")):
        t0 = time.time()
        names, etr, eva = build_features(mode)
        Xtr2, Xva2, dim2 = assemble(D, etr, eva)
        ps = [train(Xtr2, D["ytr"], Xva2, uva, yva, dim2, seed=s)[0] for s in seeds]
        mu, sd = float(np.mean(ps)), float(np.std(ps))
        mark = "  <== BEATS BASELINE" if mu > BASE + 2 * SIGMA else ""
        print(f"{label:46s} {mu:8.4f} {sd:7.4f} {mu-BASE:+9.4f}   "
              f"{' '.join(f'{p:.4f}' for p in ps)}  [{time.time()-t0:.0f}s]{mark}")
        print(f"{'':4s}fields added: {', '.join(names)}")


if __name__ == "__main__":
    main()

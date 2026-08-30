"""PyTorch port of the official numpy FM baseline.

Purpose: infrastructure only. This must reproduce baseline.py's numbers before
anything is built on top of it — otherwise every later result is unverifiable.

Parity is achieved by matching the numpy version exactly:
  * same init RNG (numpy default_rng(seed)) -> identical starting weights
  * same shuffle RNG -> identical batch composition and order
  * same optimizer split: Adam for V/W, plain SGD for the bias b
      (baseline.py line 69 updates b with `b -= lr * g.sum()`, NOT Adam)
  * L2 added to the gradient before the Adam moments (== torch weight_decay)
  * mean-reduction BCE, matching g = (sigmoid(z) - y) / B

Device: CPU by default for the parity check (MPS reduction order differs
slightly). --device mps is available for the larger models in later phases.

Usage:
    python3 fm_torch.py                  # train + report
    python3 fm_torch.py --compare        # run numpy and torch, report the gap
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn

from data import load, encode
from evaluate import evaluate


class FMTorch(nn.Module):
    """Factorization Machine. Mirrors baseline.py's FM.logits() exactly."""

    def __init__(self, dim, k=16, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)                      # same RNG as numpy FM
        V0 = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.V = nn.Parameter(torch.from_numpy(V0))
        self.W = nn.Parameter(torch.zeros(dim, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def forward(self, X):
        E = self.V[X]                                          # (B,F,k)
        S = E.sum(1)                                           # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter

    @torch.no_grad()
    def predict(self, X, bs=200_000):
        self.eval()
        out = [self(X[i:i + bs]).cpu().numpy() for i in range(0, len(X), bs)]
        self.train()
        return np.concatenate(out)


def train_fm_torch(enc, dim, k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
                   patience=4, seed=0, device='cpu', verbose=True):
    """Train and return (model, history). Selection is on VALID only — never test."""
    dev = torch.device(device)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']

    Xtr_t = torch.from_numpy(Xtr).long().to(dev)
    ytr_t = torch.from_numpy(ytr).float().to(dev)
    Xva_t = torch.from_numpy(Xva).long().to(dev)

    model = FMTorch(dim, k=k, seed=seed).to(dev)

    # Adam for V/W (weight_decay == numpy's `gV += l2 * V` before the moments).
    # Bias uses plain SGD to match baseline.py line 69.
    opt = torch.optim.Adam([model.V, model.W], lr=lr, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=l2)
    opt_b = torch.optim.SGD([model.b], lr=lr)
    lossf = nn.BCEWithLogitsLoss(reduction='mean')   # matches g = (sigmoid(z)-y)/B

    rng = np.random.default_rng(seed)                # same shuffle stream as numpy
    best, best_state, bad = -1.0, None, 0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(ytr))
        losses = []
        for i in range(0, len(idx), bs):
            batch = torch.from_numpy(idx[i:i + bs]).long().to(dev)
            opt.zero_grad(set_to_none=True); opt_b.zero_grad(set_to_none=True)
            loss = lossf(model(Xtr_t[batch]), ytr_t[batch])
            loss.backward()
            opt.step(); opt_b.step()
            losses.append(loss.item())

        va = evaluate(uva, yva, model.predict(Xva_t))
        history.append({'epoch': ep, 'train_loss': float(np.mean(losses)),
                        'valid_GAUC': va['GAUC'], 'valid_nDCG@5': va['nDCG@5'],
                        'valid_primary': va['primary'], 'secs': time.time() - t0})
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")

        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = {k_: v.detach().clone() for k_, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break

    model.load_state_dict(best_state)
    return model, history


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'mps'])
    ap.add_argument('--compare', action='store_true',
                    help='also run the numpy FM and report the parity gap')
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    enc, dim = encode(splits)
    Xte, yte, ute = enc['test']

    print(f"\n=== torch FM (seed={a.seed}, device={a.device}) ===")
    t0 = time.time()
    model, hist = train_fm_torch(enc, dim, k=a.k, lr=a.lr, epochs=a.epochs,
                                 seed=a.seed, device=a.device)
    torch_secs = time.time() - t0

    Xva, yva, uva = enc['valid']
    dev = torch.device(a.device)
    tv = evaluate(uva, yva, model.predict(torch.from_numpy(Xva).long().to(dev)))
    tt = evaluate(ute, yte, model.predict(torch.from_numpy(Xte).long().to(dev)))
    print(f"  valid  GAUC {tv['GAUC']:.4f} | nDCG@5 {tv['nDCG@5']:.4f} | primary {tv['primary']:.4f}")
    print(f"  test   GAUC {tt['GAUC']:.4f} | nDCG@5 {tt['nDCG@5']:.4f} | primary {tt['primary']:.4f}")
    print(f"  {torch_secs:.1f}s, {len(hist)} epochs")

    if a.compare:
        from baseline import run_fm
        print(f"\n=== numpy FM (seed={a.seed}) ===")
        t0 = time.time()
        npres = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, verbose=False)
        np_secs = time.time() - t0
        nv, nt = npres['valid'], npres['test']
        print(f"  valid  GAUC {nv['GAUC']:.4f} | nDCG@5 {nv['nDCG@5']:.4f} | primary {nv['primary']:.4f}")
        print(f"  test   GAUC {nt['GAUC']:.4f} | nDCG@5 {nt['nDCG@5']:.4f} | primary {nt['primary']:.4f}")
        print(f"  {np_secs:.1f}s")

        SIGMA = 0.0008                       # published per-seed std
        print(f"\n=== PARITY (tolerance: 1 sigma = {SIGMA}) ===")
        ok = True
        for split, tr, nr in (('valid', tv, nv), ('test', tt, nt)):
            for m in ('GAUC', 'nDCG@5', 'primary'):
                d = tr[m] - nr[m]
                flag = 'OK ' if abs(d) <= SIGMA else 'FAIL'
                if abs(d) > SIGMA:
                    ok = False
                print(f"  {flag} {split:5s} {m:8s} torch {tr[m]:.4f}  numpy {nr[m]:.4f}  delta {d:+.4f}")
        print(f"\n  {'PARITY PASSED' if ok else 'PARITY FAILED — do not build on this'}")
        print(f"  speed: torch {torch_secs:.1f}s vs numpy {np_secs:.1f}s")

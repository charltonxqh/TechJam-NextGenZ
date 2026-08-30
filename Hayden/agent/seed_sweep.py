"""How far can seed ensembling go?

Trains N models once, then evaluates the ensemble of the first n for increasing
n. This answers, empirically, where the curve flattens — i.e. the ceiling of
variance reduction for this model on this data.

No LLM calls. Pure compute.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

from evaluate import evaluate      # noqa: E402
from prep import load_cache        # noqa: E402
import importlib.util              # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BASE = 0.6015          # official FM baseline, validation primary


def load_seed_model():
    """The single-FM trainer from the seed solution (one model, one seed)."""
    spec = importlib.util.spec_from_file_location("seedsol", HERE / "seed_solution.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    D = load_cache()
    uva, yva = D["uva"], D["yva"]
    sol = load_seed_model()

    print(f"training {N} independent FMs (seeds 0..{N-1}) ...")
    preds = []
    t0 = time.time()
    for s in range(N):
        scores, _ = sol.run(D, seed=s)
        preds.append(np.asarray(scores, dtype=np.float64))
        p = evaluate(uva, yva, preds[-1])["primary"]
        print(f"  seed {s:2d}  single-model primary {p:.4f}   [{time.time()-t0:5.0f}s]")

    P = np.vstack(preds)
    singles = [evaluate(uva, yva, P[i])["primary"] for i in range(N)]
    print(f"\nsingle-model: mean {np.mean(singles):.4f}  std {np.std(singles):.4f}  "
          f"(published sigma 0.0008)")

    print(f"\n{'n':>4}  {'primary':>8}  {'vs baseline':>12}  {'gain vs n=1':>12}")
    print("  " + "-" * 42)
    base1 = np.mean(singles)
    curve = []
    for n in (1, 2, 3, 5, 8, 10, 15, 20):
        if n > N:
            break
        # average over disjoint-ish subsets where possible, else the first n
        ens = evaluate(uva, yva, P[:n].mean(axis=0))["primary"]
        curve.append((n, ens))
        print(f"  {n:>3}  {ens:8.4f}  {ens-BASE:+12.4f}  {ens-base1:+12.4f}")

    # extrapolate the 1 - 1/n variance-reduction law from the largest n measured
    n_last, p_last = curve[-1]
    k = (p_last - base1) / (1 - 1 / n_last)
    print(f"\nfitted asymptote (n -> infinity): {base1 + k:.4f}  "
          f"({base1 + k - BASE:+.4f} vs baseline)")
    print(f"  i.e. seed ensembling alone cannot exceed roughly {base1+k-BASE:+.4f}")
    remaining = (base1 + k) - p_last
    print(f"  remaining headroom beyond n={n_last}: {remaining:+.5f}")


if __name__ == "__main__":
    main()

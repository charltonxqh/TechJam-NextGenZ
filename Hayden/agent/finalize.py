"""Finalisation — the ONLY place the test split is ever loaded.

The agent develops on train+valid and never sees test (it is not even in the
data cache). This script runs once, at the end, on the validation-best solution
the agent selected, and produces the submission.

    python finalize.py [--solution agent/state/best_solution.py]

Steps
  1. verify integrity (scorer unmodified)
  2. load train+valid+test
  3. retrain the winning solution and predict BOTH valid and test
  4. confirm the valid score matches what the agent recorded (catches
     non-determinism or a solution that only worked by accident)
  5. write submission.csv and validate it with the official submit.py checker
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
KIT = HERE.parent / "kuairand-starter-kit"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KIT))

from data import load, encode          # noqa: E402
from evaluate import evaluate          # noqa: E402
from guards import verify_integrity, sanity_bounds   # noqa: E402
from submit import write_submission, read_submission  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solution", default=str(HERE / "state" / "best_solution.py"))
    ap.add_argument("--data_dir", default=str(KIT / "KuaiRand-Pure" / "data"))
    ap.add_argument("--out", default=str(HERE / "state" / "submission.csv"))
    ap.add_argument("--seeds", type=int, default=1,
                    help="average scores over N seeds (rank-averaged)")
    a = ap.parse_args()

    verify_integrity()
    print("integrity OK — evaluate.py unmodified\n")

    print("loading all splits (test is touched HERE and nowhere else) ...")
    splits = load(a.data_dir)
    enc, dim = encode(splits)
    Xtr, ytr, utr = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]

    # The solution's contract only exposes train+valid, so to score test we
    # hand it a D whose "valid" slot is the test matrix, and run it twice.
    spec = importlib.util.spec_from_file_location("winner", a.solution)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # A winning solution may ignore D entirely and call tools.load_features(),
    # which returns train+valid by design so the agent cannot reach test during
    # the run. Flip the flag per call so such a solution is evaluated on the same
    # split we are asking D about; otherwise it would return validation-length
    # scores while we score test, and the submission would be misaligned.
    import tools

    def score(X, y, u, seed, split):
        tools.FINALIZE_TEST = (split == "test")
        try:
            D = {"Xtr": Xtr, "ytr": ytr, "utr": utr, "Xva": X, "yva": y, "uva": u,
                 "dim": dim, "fields": enc and []}
            s, _ = mod.run(D, seed=seed)
        finally:
            tools.FINALIZE_TEST = False
        s = np.asarray(s, dtype=np.float64)
        if len(s) != len(y):
            raise SystemExit(
                f"ALIGNMENT FAILURE on {split}: solution returned {len(s)} scores "
                f"for {len(y)} rows. If it calls load_features(), the handle's "
                f"cached split does not match the split being scored.")
        return s

    def ranks(x):
        o = np.empty(len(x), dtype=np.float64)
        o[np.argsort(np.argsort(x))] = np.arange(len(x))
        return o

    t0 = time.time()
    va_runs, te_runs = [], []
    for s in range(a.seeds):
        print(f"  seed {s} ...", flush=True)
        va_runs.append(score(Xva, yva, uva, s, "valid"))
        te_runs.append(score(Xte, yte, ute, s, "test"))

    # rank-average across seeds (only order matters, and ranks are scale-free)
    va = np.mean([ranks(v) for v in va_runs], axis=0) if a.seeds > 1 else va_runs[0]
    te = np.mean([ranks(v) for v in te_runs], axis=0) if a.seeds > 1 else te_runs[0]

    rv = evaluate(uva, yva, va)
    rt = evaluate(ute, yte, te)

    BASE_V, BASE_T = 0.6015, 0.5946
    print(f"\n{'='*66}\nFINAL ({a.seeds} seed(s), {time.time()-t0:.0f}s)\n{'='*66}")
    print(f"  VALID  GAUC {rv['GAUC']:.4f} | nDCG@5 {rv['nDCG@5']:.4f} | "
          f"primary {rv['primary']:.4f}   (baseline {BASE_V}, {rv['primary']-BASE_V:+.4f})")
    print(f"  TEST   GAUC {rt['GAUC']:.4f} | nDCG@5 {rt['nDCG@5']:.4f} | "
          f"primary {rt['primary']:.4f}   (baseline {BASE_T}, {rt['primary']-BASE_T:+.4f})")
    print(f"\n  scored delta (mean of per-metric deltas vs official baseline):")
    d = ((rt["GAUC"] - 0.6610) + (rt["nDCG@5"] - 0.5282)) / 2
    print(f"    {d:+.4f}")

    for w in sanity_bounds(rv["primary"]):
        print(f"  WARNING: {w}")

    # cross-check against what the agent believed
    sp = HERE / "state" / "state.json"
    if sp.exists():
        claimed = json.loads(sp.read_text()).get("best_primary")
        if claimed is not None:
            claimed = float(claimed)      # state.json serialises np.float32 as a string
            gap = float(rv["primary"]) - claimed
            flag = "OK" if abs(gap) < 0.003 else "MISMATCH"
            print(f"  {flag}: agent recorded valid {claimed:.4f}, reproduced {rv['primary']:.4f} "
                  f"({gap:+.4f})")

    write_submission(a.out, splits["test"], te)
    print(f"\n  wrote {a.out}")
    read_submission(a.out, splits["test"])          # official validator
    print("  submission.csv passes the official format/alignment check")

    (HERE / "state" / "final_results.json").write_text(json.dumps({
        "valid": rv, "test": rt, "scored_delta": d, "seeds": a.seeds,
        "baseline_valid": BASE_V, "baseline_test": BASE_T,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

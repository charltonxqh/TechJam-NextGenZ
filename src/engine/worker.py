"""Subprocess entry point — runs ONE candidate solution in isolation.

Invoked by runner.py as:
    python worker.py <candidate.py> <out_dir> <seed>

The candidate module must define:

    def run(D, seed=0) -> (valid_scores, history)

        D: dict with Xtr (N,F) int32, ytr (N,) float32, utr (list[str]),
                     Xva, yva, uva, dim (int), fields (list[str])
        returns
            valid_scores : array of length len(yva) — any real numbers,
                           only the ordering WITHIN each user matters
            history      : list of per-epoch dicts, each with at least
                           {'epoch', 'train_loss', 'valid_primary'}

Writes scores.npy + result.json into out_dir. Any exception is caught and
serialised so the parent can classify and recover rather than crash.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_kit = next(p / "kuairand-starter-kit" for p in [HERE, *HERE.parents]
            if (p / "kuairand-starter-kit").is_dir())   # search upward: layout-independent
sys.path.insert(0, str(_kit))

from prep import load_cache  # noqa: E402


def main() -> int:
    cand_path, out_dir, seed = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"ok": False, "seed": seed}
    t0 = time.time()
    try:
        D = load_cache()

        spec = importlib.util.spec_from_file_location("candidate", cand_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                      # SyntaxError surfaces here

        if not hasattr(mod, "run"):
            raise AttributeError("candidate module defines no run(D, seed) function")

        scores, history = mod.run(D, seed=seed)

        scores = np.asarray(scores, dtype=np.float64).ravel()
        if len(scores) != len(D["yva"]):
            raise ValueError(
                f"run() returned {len(scores)} scores but the valid split has "
                f"{len(D['yva'])} rows — they must align 1:1"
            )
        if not np.all(np.isfinite(scores)):
            raise ValueError(
                f"{int((~np.isfinite(scores)).sum())} NaN/Inf scores — "
                f"a submission with these would be rejected"
            )

        np.save(out_dir / "scores.npy", scores)
        res.update(ok=True, history=list(history or []), secs=time.time() - t0)

    except Exception as e:                                # noqa: BLE001
        res.update(
            ok=False,
            error_type=type(e).__name__,
            error=str(e)[:2000],
            traceback=traceback.format_exc()[-4000:],
            secs=time.time() - t0,
        )

    (out_dir / "result.json").write_text(json.dumps(res, default=str))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

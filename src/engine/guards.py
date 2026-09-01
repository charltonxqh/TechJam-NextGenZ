"""Anti-self-deception guards.

The signature failure of autonomous ML agents is optimising hard against their
own bug — producing a beautiful number that means nothing. These checks make
the two worst versions of that structurally impossible rather than merely
discouraged.
"""
from __future__ import annotations

import hashlib
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
KIT = next(p / "kuairand-starter-kit" for p in [_HERE, *_HERE.parents]
           if (p / "kuairand-starter-kit").is_dir())   # search upward: layout-independent
CACHE = pathlib.Path(__file__).resolve().parent / "cache"

# sha256 of the pristine organiser-provided scorer, recorded at setup.
EVALUATE_SHA = "PIN_ON_FIRST_RUN"
_PIN_FILE = pathlib.Path(__file__).resolve().parent / "state" / "evaluate.sha256"


def _sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_integrity() -> None:
    """Raise if the scorer changed or the test split leaked into the cache."""
    ev = KIT / "evaluate.py"
    if not ev.exists():
        raise RuntimeError(f"evaluate.py missing at {ev}")

    digest = _sha(ev)
    _PIN_FILE.parent.mkdir(exist_ok=True)
    if _PIN_FILE.exists():
        pinned = _PIN_FILE.read_text().strip()
        if digest != pinned:
            raise RuntimeError(
                "evaluate.py HAS BEEN MODIFIED.\n"
                f"  pinned : {pinned}\n  now    : {digest}\n"
                "The scorer is the pinned task definition. Every number produced "
                "after this point would be meaningless. Halting."
            )
    else:
        _PIN_FILE.write_text(digest)

    # The cache must never contain test data — the agent cannot peek at what
    # is not there.
    import numpy as np
    p = CACHE / "trainvalid.npz"
    if p.exists():
        keys = set(np.load(p, allow_pickle=False).files)
        leaked = {k for k in keys if "te" in k.lower() and k not in ("fields",)}
        if leaked:
            raise RuntimeError(f"Test data leaked into the cache: {sorted(leaked)}")


def sanity_bounds(primary: float) -> list[str]:
    """Scores that are impossible or implausible — treat as bugs, not results."""
    warn = []
    if primary < 0.4834:
        warn.append(f"primary {primary:.4f} is BELOW random (0.4834) — this is a bug, "
                    f"not a result.")
    elif primary < 0.5807:
        # A hard floor at "worse than random" is too generous to catch a broken
        # model: one iteration scored 0.4962, which is indistinguishable from
        # random yet sat above the 0.4834 line and was reported as an ordinary
        # result. Item popularity needs no model at all and reaches 0.5807, so
        # anything below that is a training failure rather than a weak idea.
        warn.append(f"primary {primary:.4f} is below ITEM POPULARITY (0.5807), which "
                    f"uses no model at all. A trained model scoring this is almost "
                    f"certainly not learning — suspect initialisation, learning "
                    f"rate, or score/label misalignment rather than the mechanism.")
    if primary > 0.8484:
        warn.append(f"primary {primary:.4f} EXCEEDS the oracle ceiling (0.8484) — "
                    f"impossible. There is a leak or an evaluation error.")
    if primary > 0.68:
        warn.append(f"primary {primary:.4f} is a very large jump over the 0.6015 "
                    f"baseline. Verify across seeds before believing it.")
    return warn


if __name__ == "__main__":
    verify_integrity()
    print("integrity OK")
    print(f"  evaluate.py sha256 = {_sha(KIT / 'evaluate.py')[:16]}...")
    for v in (0.40, 0.60, 0.70, 0.90):
        print(f"  sanity({v}) -> {sanity_bounds(v) or 'ok'}")

"""Which aggregate feature is hurting?

Adding all four aggregates together dropped CatBoost from 0.6040 to 0.5824. A
drop that large is not a weak feature, it is a broken one, and the prime suspect
is `video_target_loo`: leave-one-out target encoding is a well-known trap with
gradient boosting, because the model can invert the LOO arithmetic and recover
the row's own label. That inflates training fit and destroys validation.

Each group is measured separately against the same no-aggregate control.
"""
from __future__ import annotations

import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # engine modules live one level up
sys.path.insert(0, str(HERE))

import tools      # noqa: E402
import models     # noqa: E402

BASE = {"categorical": ["user_id", "video_id", "author_id", "tab"],
        "derived": ["hour", "dur_bucket", "video_age"],
        "video_meta": ["tag"],
        "video_stats": True}

ARMS = [
    ("control        ", []),
    ("count only     ", ["video_count"]),
    ("duration stats ", ["video_duration_stats"]),
    ("user-video gap ", ["user_video_duration_gap"]),
    ("target LOO     ", ["video_target_loo"]),
    ("all but target ", ["video_count", "video_duration_stats",
                         "user_video_duration_gap"]),
]

PARAMS = {"loss": "QueryRMSE", "depth": 6, "iterations": 300}


def main():
    print(f"{'arm':17s} {'valid primary':>14s} {'GAUC':>8s} {'secs':>6s}", flush=True)
    print("-" * 50, flush=True)
    for name, aggs in ARMS:
        spec = dict(BASE)
        spec["name"] = "abl_" + name.strip().replace(" ", "_").replace("-", "_")
        if aggs:
            spec["aggregates"] = aggs
        b = tools.build_features({"spec": spec})
        if "error" in b:
            print(f"{name} BUILD FAILED: {b['error'][:60]}", flush=True)
            continue
        r = models.train("catboost", spec["name"], PARAMS, budget_s=400)
        if "error" in r:
            print(f"{name} TRAIN FAILED: {r['error'][:60]}", flush=True)
            continue
        print(f"{name} {r['valid_primary']:>14.4f} {r['valid_GAUC']:>8.4f} "
              f"{r['seconds']:>6d}", flush=True)


if __name__ == "__main__":
    main()

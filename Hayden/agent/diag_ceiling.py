"""Is the agent's STARTING REPRESENTATION the thing capping it?

Hypothesis. The agent is handed D: five pre-encoded categorical fields, on which
the organisers report that model capacity is flat (k=8/16/32 -> 0.5895/0.5902/
0.5887) and extra static fields are flat (5 -> 13 fields, 0.5950 -> 0.5940). If
the FM at 0.6015 is already near the best ANY model can do on that
representation, then every single-change proposal that stays inside D is
competing for a few ten-thousandths, and the agent's repeated convergence at
exactly 0.6015 is a property of the problem we handed it rather than a failure
of its reasoning.

Test: train strong models on the DEFAULT five fields alone, then the same models
with a richer feature set. If the five-field ceiling sits at ~0.60 while the rich
one reaches ~0.605, the binding constraint is the representation, and the fix is
to change what the agent starts from - not to keep editing its prompt.
"""
from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tools    # noqa: E402
import models   # noqa: E402


def main():
    # ---- arm A: the five official fields, exactly what D gives the agent ----
    a = tools.build_features({"spec": {
        "name": "diag_5field",
        "categorical": ["user_id", "video_id", "author_id", "tab"],
        "derived": ["dur_bucket"]}})
    print(f"5-field set: dim {a.get('dim')}, dense {a.get('dense_block')}", flush=True)

    # ---- arm B: everything the tools can reach ----
    b = tools.build_features({"spec": {
        "name": "diag_rich",
        "categorical": ["user_id", "video_id", "author_id", "tab"],
        "derived": ["hour", "dur_bucket", "video_age"],
        "video_meta": ["tag"],
        "video_stats": True,
        "aggregates": True}})
    print(f"rich set   : dim {b.get('dim')}, dense {b.get('dense_block')}\n", flush=True)

    trials = [
        ("catboost", {"loss": "QueryRMSE", "depth": 6, "iterations": 400}),
        ("catboost", {"loss": "YetiRank", "depth": 6, "iterations": 400}),
        ("lightgbm", {"objective": "lambdarank", "num_leaves": 63,
                      "n_estimators": 300}),
    ]

    print(f"{'model':28s} {'5 fields':>10s} {'rich':>10s} {'gain':>8s}", flush=True)
    print("-" * 60, flush=True)
    best5 = bestr = 0.0
    for family, params in trials:
        row = []
        for handle in ("diag_5field", "diag_rich"):
            t0 = time.time()
            r = models.train(family, handle, params, budget_s=420)
            row.append(r.get("valid_primary") if "error" not in r else None)
            if "error" in r:
                print(f"  {family} on {handle} FAILED: {r['error'][:70]}", flush=True)
        if row[0] and row[1]:
            best5 = max(best5, row[0]); bestr = max(bestr, row[1])
            name = f"{family} {params.get('loss') or params.get('objective')}"
            print(f"{name:28s} {row[0]:10.4f} {row[1]:10.4f} "
                  f"{row[1]-row[0]:+8.4f}", flush=True)

    print("-" * 60, flush=True)
    print(f"{'BEST':28s} {best5:10.4f} {bestr:10.4f} {bestr-best5:+8.4f}", flush=True)
    print(f"\n  FM baseline on the same 5 fields: 0.6015")
    print(f"  best on 5 fields              : {best5:.4f}  "
          f"({best5-0.6015:+.4f} vs baseline)")
    print(f"  best with richer features     : {bestr:.4f}  "
          f"({bestr-0.6015:+.4f} vs baseline)")
    if best5 - 0.6015 < 0.002 <= bestr - 0.6015:
        print("\n  => The five-field representation is the binding constraint.")
        print("     No single change inside D can clear the convergence threshold;")
        print("     the agent must leave D, which takes several coupled changes.")


if __name__ == "__main__":
    main()

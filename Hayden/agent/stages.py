"""Pipeline-stage coverage — stopping the agent circling two stages out of five.

The challenge defines the MLE loop as five stages, and says improvements "are not
limited to the model itself... every upstream and downstream module is fair game".

Across seven runs, every single attempt this agent made classified as either
MODEL or LOSS:

    Replaced the standard FM model with a Field-aware FM        MODEL
    Implementation of a DeepFM architecture                     MODEL
    Replaced the neural FM with a CatBoost classifier           MODEL
    Replaced pointwise BCE with a listwise ListCE loss          LOSS
    Replaced BCE with a Softmax-normalized ranking loss         LOSS
    Expanded feature set to include author_id, tag, music_id    FEATURES
    ...

Nothing ever touched TRAINING (sampling, weighting, schedule), TUNING
(hyperparameters, which the rules explicitly permit and which took CatBoost from
0.6044 to 0.6075 when a human did it), or ENSEMBLING (which produced the single
largest human-measured gain in this project, +0.005).

That is not the agent being incurious - nothing told it those stages existed as
options, and the prompt handed it a single model to edit. This module makes the
stage explicit: every proposal declares which stage it targets, coverage is
tracked, and untouched stages are surfaced as unexplored territory.

Classification is keyword-based on the change summary. It is approximate, and
deliberately so: it exists to answer "what have we not tried", not to be a
taxonomy.
"""
from __future__ import annotations

import json
import pathlib
import re

STATE = pathlib.Path(__file__).resolve().parent / "state"
FILE = STATE / "stages.json"

# Ordered roughly as the loop runs, matching the challenge's Figure 1.
STAGES = {
    "features": {
        "desc": "what the model sees: which raw columns, how they are encoded, "
                "bucketing, crosses, derived fields, dense blocks",
        "cues": ("feature", "field", "column", "encod", "bucket", "cross",
                 "embedding dim", "tag", "music_id", "author_id", "hourmin",
                 "statistic", "video_age", "dense"),
    },
    "model": {
        "desc": "the architecture: FM, DeepFM, DCN, gradient boosting, "
                "attention, depth and width",
        "cues": ("architect", "deepfm", "dcn", "ffm", "catboost", "lightgbm",
                 "xgboost", "mlp", "layer", "network", "model", "tree", "cin",
                 "attention", "transformer"),
    },
    "loss": {
        "desc": "the training objective: pointwise, pairwise, listwise, "
                "multi-task heads, auxiliary targets",
        "cues": ("loss", "objective", "bce", "bpr", "listce", "listnet",
                 "softmax", "pairwise", "listwise", "pointwise", "multi-task",
                 "auxiliary", "cross-entropy", "hinge"),
    },
    "training": {
        "desc": "HOW it is trained: batching, negative sampling, sample or user "
                "weighting, epochs, early stopping, recency weighting, "
                "matching train and eval group-size distributions",
        "cues": ("batch", "sampl", "weight", "epoch", "early stop", "schedule",
                 "learning rate schedul", "curriculum", "recency", "group size",
                 "iterator", "shuffl"),
    },
    "tuning": {
        "desc": "hyperparameter search on VALIDATION - explicitly permitted by "
                "the rules, and optuna is installed. Untuned CatBoost measured "
                "0.6044; tuned it measured 0.6075.",
        "cues": ("hyperparameter", "tune", "tuning", "optuna", "grid search",
                 "sweep", "learning rate", "depth", "num_leaves", "iterations",
                 "regularis", "regulariz", "weight decay"),
    },
    "ensembling": {
        "desc": "combining several models: seed averaging, rank averaging, "
                "blending different families. The largest human-measured gain in "
                "this project came from here.",
        "cues": ("ensemble", "blend", "average", "averaging", "seed", "bagging",
                 "stack", "rank-averag", "combin"),
    },
    "evaluation": {
        "desc": "the scoring path itself: how raw scores are turned into a "
                "ranking - per-user normalisation, calibration, tie-breaking, "
                "rank transforms",
        "cues": ("calibrat", "normalis", "normaliz", "rank transform", "tie",
                 "percentile", "per-user offset", "post-process", "shift"),
    },
}


def classify(summary: str) -> str:
    """Best-guess stage for a change summary. Ties go to the earlier stage."""
    t = (summary or "").lower()
    best, best_hits = "model", 0
    for name, spec in STAGES.items():
        hits = sum(1 for c in spec["cues"] if c in t)
        if hits > best_hits:
            best, best_hits = name, hits
    return best if best_hits else "model"


def load() -> dict:
    if FILE.exists():
        try:
            return json.loads(FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {k: {"tried": 0, "best_delta": None} for k in STAGES}


def save(d: dict) -> None:
    STATE.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(d, indent=2))


def record(summary: str, delta: float, stage: str | None = None) -> str:
    st = stage if stage in STAGES else classify(summary)
    d = load()
    e = d.setdefault(st, {"tried": 0, "best_delta": None})
    e["tried"] += 1
    e["positive"] = e.get("positive", 0) + (1 if delta > 0.0008 else 0)
    if e["best_delta"] is None or delta > e["best_delta"]:
        e["best_delta"] = round(float(delta), 4)
    save(d)
    return st


def as_prompt_section() -> str:
    d = load()
    tried = {k: v for k, v in d.items() if v.get("tried")}
    untried = [k for k in STAGES if not d.get(k, {}).get("tried")]

    # Rank by what each stage has actually RETURNED, not by how often it was
    # tried. Listing raw counts was not enough: after "model: tried 5x, best
    # -0.0001" appeared in the prompt, the agent chose model four times in the
    # next five iterations anyway. Counts read as neutral bookkeeping; a stage
    # ordered last and labelled EXHAUSTED reads as evidence.
    def rank(item):
        name, e = item
        n = e.get("tried", 0)
        if not n:
            return (0, 0)                       # untried first: unknown, not bad
        bd = e.get("best_delta") or 0.0
        return (1 if bd <= 0.0008 else 0, -bd)  # then by best measured delta

    lines = ["## Pipeline stage coverage — ordered by what each has RETURNED",
             "",
             "Improvements are not limited to the model. Every stage below is",
             "something you can change, and the ordering is evidence, not opinion:",
             ""]
    for name, e in sorted(d.items(), key=rank):
        spec = STAGES.get(name)
        if not spec:
            continue
        n = e.get("tried", 0)
        bd = e.get("best_delta")
        pos = e.get("positive", 0)
        # `positive` is only counted from the run that introduced it, so
        # backfilled stages legitimately read 0 while having a positive
        # best_delta. Reporting "0 positive ... has produced real gains" in one
        # line is self-contradictory; best_delta is the field with history
        # behind it, so the verdict rests on that alone.
        if not n:
            mark = "NEVER TRIED — unknown, and the only unmeasured variance left"
        elif bd is not None and bd > 0.0008:
            mark = (f"{n} attempt(s), best {bd:+.4f}"
                    + (f" ({pos} positive)" if pos else "")
                    + "  <- has produced real gains")
        else:
            mark = (f"EXHAUSTED: {n} attempt(s), best {bd:+.4f}. "
                    f"That is {n} iteration(s) spent here for nothing.")
        lines.append(f"  {name:<12} {mark}")
        lines.append(f"       {spec['desc']}")
    if untried:
        lines += ["",
                  f"NEVER TRIED: {', '.join(untried)}.",
                  "Across nine runs of this agent, every accepted improvement came",
                  "from a stage with one or fewer prior attempts, and every stage",
                  "with 4+ attempts has returned nothing. Repeating an exhausted",
                  "stage is the single most reliable way to waste an iteration."]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        # Classify what previous runs actually did, so coverage starts honest.
        n = 0
        for sp in sorted(STATE.glob("archive_*/state.json")) + [STATE / "state.json"]:
            if not sp.exists():
                continue
            try:
                s = json.loads(sp.read_text())
            except json.JSONDecodeError:
                continue
            for a in s.get("attempts", []):
                cs = a.get("change_summary")
                if cs and cs != "N/A":
                    d = a.get("delta")
                    record(cs, float(d) if isinstance(d, (int, float)) else 0.0)
                    n += 1
        print(f"backfilled {n} past attempts")
    print(as_prompt_section())

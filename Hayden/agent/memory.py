"""Long-term memory — what survives a run, with provenance and confidence.

The previous store (findings.json) was a flat append-only list of
{mechanism, measured_delta, verdict}. Four failures showed up in practice:

1. NO PROVENANCE. Nothing recorded whether a finding came from the agent or
   from a human. Over one session a human added CatBoost, collaborative-filtering
   and DCN-V2 results to the same file the agent reads as "measurements from
   previous runs of this agent". Innovation is graded on what the AGENT
   identified and why, so silently mixing the two overstates the agent and is
   simply not true. Every entry now carries `source`.

2. NO CORRECTIONS. We once recorded "intra-user listwise loss is harmful,
   -0.0019". That was a softmax-vs-sigmoid implementation bug; re-measured
   correctly it is +0.0000. The false finding had already propagated into the
   run log, the README and the agent's own priors, and the store had no way to
   express "this entry was superseded". Entries can now be superseded, and a
   superseded entry is shown WITH its correction so the mistake stays visible
   instead of being quietly deleted.

3. DUMPED, NOT QUERIED. Every finding entered every prompt. That grows without
   bound and crowds out the actual task. `recall(topic)` returns the relevant
   subset, and is exposed to the agent as a tool.

4. NO CONFIDENCE. A single-seed +0.002 and a 3-seed-confirmed +0.002 were stored
   identically, on a benchmark where single-seed deltas were measured wrong by
   2-4x. `seeds` and a derived confidence are now first-class.
"""
from __future__ import annotations

import json
import pathlib
import re
import time

STATE = pathlib.Path(__file__).resolve().parent / "state"
FILE = STATE / "memory.json"
SIGMA = 0.0008

SOURCE_AGENT = "agent"      # produced by an autonomous iteration
SOURCE_HUMAN = "human"      # produced by a person running scripts by hand
SOURCE_KIT = "organisers"   # published in the starter kit


def _now() -> str:
    return time.strftime("%Y-%m-%d")


def load() -> list:
    if FILE.exists():
        try:
            return json.loads(FILE.read_text())
        except json.JSONDecodeError:
            pass
    return []


def save(items: list) -> None:
    STATE.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(items, indent=2))


def confidence(delta: float, seeds: int) -> str:
    """How much weight this measurement deserves.

    Replication matters more than size here: several apparent +0.002 gains on
    this benchmark averaged to +0.0005 or less once re-run across seeds.
    """
    if seeds >= 3 and abs(delta) > 2 * SIGMA:
        return "high (replicated, >2 sigma)"
    if seeds >= 3:
        return "medium (replicated, within noise)"
    if abs(delta) > 2 * SIGMA:
        return "low (single seed, unreplicated - historically wrong by 2-4x)"
    return "none (single seed, within noise - indistinguishable from zero)"


def record(mechanism: str, delta: float, kept: bool, *, source: str = SOURCE_AGENT,
           seeds: int = 1, iteration: int | None = None, tags: list | None = None,
           note: str = "") -> dict:
    """Append one measurement. `source` is required to be honest about who ran it."""
    items = load()
    entry = {
        "id": f"m{len(items) + 1:03d}",
        "mechanism": mechanism[:400],
        "delta": round(float(delta), 4),
        "seeds": int(seeds),
        "verdict": "POSITIVE" if kept else ("NEGATIVE" if delta < -0.001 else "NEUTRAL"),
        "confidence": confidence(delta, seeds),
        "source": source,
        "iteration": iteration,
        "date": _now(),
        "tags": tags or _autotag(mechanism),
        "superseded_by": None,
        "note": note,
    }
    items.append(entry)
    save(items)
    return entry


def supersede(old_id: str, reason: str, new_id: str | None = None) -> bool:
    """Mark an entry as corrected. The entry stays - the mistake is evidence."""
    items = load()
    for it in items:
        if it["id"] == old_id:
            it["superseded_by"] = {"by": new_id, "reason": reason, "date": _now()}
            save(items)
            return True
    return False


_TAGS = {
    "loss": ("loss", "bpr", "listwise", "pairwise", "listce", "softmax", "objective",
             "ranking loss", "cross-entropy"),
    "features": ("feature", "field", "hourmin", "tag", "age", "statistic", "history",
                 "column", "embedding dim"),
    "architecture": ("fm", "dcn", "deepfm", "cin", "ffm", "mlp", "cross", "catboost",
                     "lightgbm", "xgboost", "tree", "boosting"),
    "ensembling": ("ensemble", "blend", "seed", "average", "rank-averag", "bagging"),
    "tuning": ("hyperparameter", "learning rate", "weight decay", "epochs", "depth",
               "iterations", "tune"),
    "multitask": ("multi-task", "auxiliary", "is_click", "play_time", "watch time"),
}


def _autotag(text: str) -> list:
    t = text.lower()
    return [k for k, words in _TAGS.items() if any(w in t for w in words)] or ["other"]


def recall(topic: str = "", limit: int = 12) -> dict:
    """Return the findings relevant to a topic, not the whole store.

    Negative results are always included when they match, because their whole
    purpose is to stop the agent re-running something already measured.
    """
    items = load()
    if not items:
        return {"findings": [], "note": "memory is empty"}
    # Direct id lookup. The agent reads an id in one result and naturally tries
    # recall("m013") to pull the full entry; keyword matching returns nothing for
    # that, which wastes a tool call out of a small budget.
    ids = {it["id"].lower() for it in items}
    if topic.strip().lower() in ids:
        hits = [it for it in items if it["id"].lower() == topic.strip().lower()]
    elif not topic.strip():
        hits = items
    else:
        q = set(re.findall(r"[a-z0-9_]+", topic.lower()))
        def score(it):
            hay = set(re.findall(r"[a-z0-9_]+", (it["mechanism"] + " " +
                                                 " ".join(it["tags"])).lower()))
            return len(q & hay)
        hits = [it for it in items if score(it)]
        hits.sort(key=lambda it: (-score(it), it["id"]))
    hits = hits[:limit]
    return {
        "topic": topic or "(all)",
        "returned": len(hits),
        "of_total": len(items),
        "findings": [{
            "id": it["id"],
            "mechanism": it["mechanism"],
            "measured": f"{it['delta']:+.4f} over {it['seeds']} seed(s)",
            "verdict": it["verdict"],
            "confidence": it["confidence"],
            "measured_by": it["source"],
            **({"CORRECTED": it["superseded_by"]} if it.get("superseded_by") else {}),
        } for it in hits],
    }


def stats() -> dict:
    items = load()
    by_source, by_verdict = {}, {}
    for it in items:
        by_source[it["source"]] = by_source.get(it["source"], 0) + 1
        by_verdict[it["verdict"]] = by_verdict.get(it["verdict"], 0) + 1
    return {"total": len(items), "by_source": by_source, "by_verdict": by_verdict,
            "superseded": sum(1 for it in items if it.get("superseded_by"))}


def as_prompt_section(topic: str = "", limit: int = 14) -> str:
    """Compact block for the prompt. Provenance is shown, never hidden."""
    r = recall(topic, limit)
    if not r["findings"]:
        return ""
    lines = [f"## Memory ({r['returned']} of {r['of_total']} findings"
             + (f", matching '{topic}'" if topic else "") + ")",
             "",
             "`measured_by` says who ran it. Findings marked `human` were produced by",
             "a person working outside the loop - treat them as background, not as",
             "your own prior work.",
             ""]
    for f in r["findings"]:
        lines.append(f"- [{f['id']}] {f['mechanism']}")
        lines.append(f"    {f['measured']}  verdict {f['verdict']}  "
                     f"confidence: {f['confidence']}  by: {f['measured_by']}")
        if "CORRECTED" in f:
            lines.append(f"    /!\\ SUPERSEDED: {f['CORRECTED']['reason']}")
    lines += ["",
              "Measurement discipline: sigma is 0.0008 per seed. Single-seed deltas on",
              "this benchmark have been wrong by 2-4x. Do not build a hypothesis on a",
              "finding whose confidence is 'low' or 'none' without re-measuring it."]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def migrate_from_findings() -> int:
    """One-time import of findings.py DEFAULT, with provenance assigned honestly."""
    import findings
    if load():
        return 0
    # Everything a human ran by hand this session, identified by mechanism text.
    human_markers = ("catboost", "item-item collaborative", "dcn-v2", "dcn-mix",
                     "lightgbm lambdarank with platform item statistics")
    out = []
    for i, f in enumerate(findings.DEFAULT, 1):
        mech = f["mechanism"]
        src = SOURCE_HUMAN if any(m in mech.lower() for m in human_markers) else SOURCE_AGENT
        raw = str(f.get("measured_delta", ""))
        # Some legacy entries quote an ABSOLUTE score ("valid 0.6715 alone")
        # rather than a delta. A real delta on this benchmark is a few
        # thousandths; anything above 0.1 is a score, so convert it against the
        # official baseline instead of storing it as an improvement of +0.67.
        m = re.search(r"([+-]?\d*\.\d+)", raw)
        delta = float(m.group(1)) if m else 0.0
        if abs(delta) > 0.1:
            delta = round(delta - 0.6015, 4)      # vs official valid primary
        seeds = 3 if ("3 seeds" in raw or "seeds" in raw) else 1
        out.append({
            "id": f"m{i:03d}", "mechanism": mech, "delta": delta, "seeds": seeds,
            "verdict": ("POSITIVE" if "POSITIVE" in f.get("verdict", "")
                        else "NEGATIVE" if "NEGATIVE" in f.get("verdict", "")
                        else "NEUTRAL"),
            "confidence": confidence(delta, seeds),
            "source": src, "iteration": None, "date": _now(),
            "tags": _autotag(mech), "superseded_by": None,
            "note": f.get("verdict", "")[:300],
        })
    save(out)
    return len(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        print(f"imported {migrate_from_findings()} findings")
    print(json.dumps(stats(), indent=2))
    print()
    print(as_prompt_section("catboost tree boosting", limit=4))

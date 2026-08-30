"""Rich experiment feedback — the agent's eyes.

Why this file matters more than its size suggests: an agent that is told
`primary = 0.6015` can only propose generic tweaks. An agent told *where* the
model is losing, whether it overfit, and what regressed against the previous
best can propose a targeted next hypothesis. Same LLM, very different quality.

Everything here is computed on TRAIN + VALID only. The test split is never
touched — see guards.py.
"""
from __future__ import annotations

import collections
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'kuairand-starter-kit'))
from evaluate import evaluate  # noqa: E402  (pinned scorer — never modified)

# Reference rungs on VALID (measured in Run-Log iteration 0).
REF = {"random": 0.4834, "pop": 0.5807, "baseline_fm": 0.6015, "oracle": 0.8484}
SIGMA = 0.0008          # published per-seed std — the noise floor for any claim


# ---------------------------------------------------------------- segments
def build_segments(splits) -> dict:
    """Per-user metadata used to slice validation performance.

    Returns {user_id: {...}} for users present in valid.
    """
    train_counts = collections.Counter(x[1] for x in splits["train"])
    valid_users = collections.defaultdict(list)
    for x in splits["valid"]:
        valid_users[x[1]].append(x[6])

    seg = {}
    for u, labels in valid_users.items():
        h = train_counts.get(u, 0)
        npos, n = sum(labels), len(labels)
        seg[u] = {
            "history": h,
            "cold": h == 0,
            "hist_bucket": ("cold" if h == 0 else "short" if h < 15
                            else "medium" if h < 50 else "long"),
            "group_size": n,
            "size_bucket": "tiny" if n <= 2 else "small" if n <= 5 else "medium" if n <= 12 else "large",
            "composition": ("all_neg" if npos == 0 else "all_pos" if npos == n
                            else "discriminative"),
        }
    return seg


def _slice(users, labels, scores, keep) -> dict | None:
    idx = [i for i, u in enumerate(users) if u in keep]
    if len(idx) < 50:
        return None
    r = evaluate([users[i] for i in idx], [labels[i] for i in idx], [scores[i] for i in idx])
    return {"users": r["users"], "rows": r["rows"], "GAUC": r["GAUC"],
            "nDCG@5": r["nDCG@5"], "primary": r["primary"]}


def segment_report(users, labels, scores, seg, key) -> dict:
    """Metrics broken down by one segment key ('hist_bucket', 'size_bucket', ...)."""
    groups = collections.defaultdict(set)
    for u, meta in seg.items():
        groups[meta[key]].add(u)
    out = {}
    for name, keep in groups.items():
        r = _slice(users, labels, scores, keep)
        if r:
            out[name] = r
    return out


# ------------------------------------------------------- training dynamics
def training_dynamics(history: list) -> dict:
    """Overfitting / convergence signals from the per-epoch log."""
    if not history:
        return {}
    # The history contract asks for {"epoch", "train_loss", "valid_primary"} per
    # epoch, but generated code does not always honour it — a tree model has no
    # per-epoch train loss to report, and one run omitted the key entirely and
    # took the whole agent down with a KeyError. Diagnostics must never be the
    # thing that kills a run: a malformed history is a degraded report, not a
    # crash.
    def _f(x):
        """Coerce to float or None. Generated code reports numbers as strings,
        numpy scalars, or formatted text ('0.6015') interchangeably; every one of
        those has reached this function, and arithmetic on the string forms
        raised TypeError mid-run."""
        if isinstance(x, bool) or x is None:
            return None
        if isinstance(x, (int, float, np.generic)):
            v = float(x)
            return v if np.isfinite(v) else None
        try:
            v = float(str(x).strip())
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    history = [h for h in history
               if isinstance(h, dict) and _f(h.get("valid_primary")) is not None]
    if not history:
        return {"note": "history had no usable per-epoch entries"}
    prim = [_f(h["valid_primary"]) for h in history]
    loss = [_f(h.get("train_loss")) for h in history]
    have_loss = all(x is not None for x in loss)
    best_i = int(np.argmax(prim))
    d = {
        "epochs_run": len(history),
        "best_epoch": history[best_i].get("epoch", best_i),
        "best_valid_primary": prim[best_i],
        "final_valid_primary": prim[-1],
        "valid_drop_after_best": round(prim[best_i] - prim[-1], 5),
    }
    if not have_loss:
        d["note"] = ("no per-epoch train_loss reported, so overfitting could not "
                     "be assessed; include train_loss in history to get it")
        d["overfitting"] = False
        return d
    d["train_loss_first"], d["train_loss_last"] = loss[0], loss[-1]
    # Overfitting: train loss still falling while valid has turned over.
    still_learning = loss[-1] < loss[best_i] - 1e-4
    turned_over = prim[-1] < prim[best_i] - SIGMA
    d["overfitting"] = bool(still_learning and turned_over)
    # Report the OBSERVATION only. Suggesting remedies here caused the agent to
    # spend iterations turning the knob we named (weight decay) instead of
    # reasoning about mechanisms — the same "facts, not conclusions" principle
    # that governs context.py applies to feedback.
    if d["overfitting"]:
        d["hint"] = (f"Observation: validation peaked at epoch {d['best_epoch']} of "
                     f"{len(history)} while training loss continued to fall "
                     f"({loss[best_i]:.4f} -> {loss[-1]:.4f}).")
    elif best_i == len(history) - 1:
        d["hint"] = ("Observation: validation was still improving at the final epoch — "
                     "training stopped at the epoch cap, not at convergence.")
    return d


# ------------------------------------------------------ prediction health
def prediction_health(scores, users) -> dict:
    """Catches degenerate outputs that a metric alone can hide."""
    s = np.asarray(scores, dtype=np.float64)
    out = {
        "mean": float(s.mean()), "std": float(s.std()),
        "min": float(s.min()), "max": float(s.max()),
        "nan_or_inf": int((~np.isfinite(s)).sum()),
        "unique_frac": round(len(np.unique(s)) / len(s), 4),
    }
    # Within-user variation is the only thing the metric can see.
    byu = collections.defaultdict(list)
    for u, v in zip(users, s):
        byu[u].append(v)
    flat = sum(1 for v in byu.values() if len(v) > 1 and np.std(v) < 1e-9)
    multi = sum(1 for v in byu.values() if len(v) > 1)
    out["users_with_constant_scores"] = flat
    out["users_with_constant_scores_pct"] = round(100 * flat / max(multi, 1), 2)

    warn = []
    if out["nan_or_inf"]:
        warn.append(f"FATAL: {out['nan_or_inf']} NaN/Inf scores — submission would be rejected.")
    if out["std"] < 1e-8:
        warn.append("FATAL: model output is constant — it ranks nothing.")
    if out["unique_frac"] < 0.01:
        warn.append(f"Only {out['unique_frac']*100:.1f}% of scores are distinct — heavy ties "
                    f"blunt the ranking (ties get averaged rank in GAUC).")
    if out["users_with_constant_scores_pct"] > 5:
        warn.append(f"{out['users_with_constant_scores_pct']}% of multi-impression users get "
                    f"identical scores across all their items — no ordering signal for them.")
    out["warnings"] = warn
    return out


# --------------------------------------------------------------- regression
def compare_to(prev_scores, users, labels, seg, scores) -> dict:
    """Which segments improved / regressed against the previous best."""
    if prev_scores is None:
        return {}
    out = {}
    for key in ("hist_bucket", "size_bucket", "composition"):
        now = segment_report(users, labels, scores, seg, key)
        was = segment_report(users, labels, prev_scores, seg, key)
        deltas = {n: round(now[n]["primary"] - was[n]["primary"], 5)
                  for n in now if n in was}
        out[key] = deltas
    gained = [f"{k}:{n} {d:+.4f}" for k, v in out.items() for n, d in v.items() if d > SIGMA]
    lost = [f"{k}:{n} {d:+.4f}" for k, v in out.items() for n, d in v.items() if d < -SIGMA]
    out["_summary"] = {"improved": gained, "regressed": lost}
    if gained and lost:
        out["_hint"] = ("This change TRADES populations rather than lifting everything: "
                        f"gained on [{', '.join(gained[:3])}] but lost on [{', '.join(lost[:3])}].")
    return out


# ------------------------------------------------------------------ report
def diagnose(users, labels, scores, seg, history=None, prev_scores=None,
             best_primary=None) -> dict:
    """Full feedback bundle for one experiment. Valid split only."""
    r = evaluate(users, labels, scores)
    prim = r["primary"]
    rep = {
        "headline": {
            "GAUC": round(r["GAUC"], 5),
            "nDCG@5": round(r["nDCG@5"], 5),
            "primary": round(prim, 5),
            "vs_baseline_fm": round(prim - REF["baseline_fm"], 5),
            "vs_current_best": (round(prim - best_primary, 5) if best_primary is not None else None),
            "pct_of_headroom": round(100 * (prim - REF["baseline_fm"])
                                     / (REF["oracle"] - REF["baseline_fm"]), 2),
        },
        "reference_rungs_valid": REF,
        "noise_floor_sigma": SIGMA,
        "training": training_dynamics(history or []),
        "health": prediction_health(scores, users),
        "segments": {
            "by_user_history": segment_report(users, labels, scores, seg, "hist_bucket"),
            "by_group_size": segment_report(users, labels, scores, seg, "size_bucket"),
            "by_composition": segment_report(users, labels, scores, seg, "composition"),
        },
        "regression_vs_previous_best": compare_to(prev_scores, users, labels, seg, scores),
    }

    # Plain-language verdict the agent can act on directly.
    notes = []
    if best_primary is not None:
        d = prim - best_primary
        if d > 2 * SIGMA:
            notes.append(f"IMPROVED over best by {d:+.4f} (> 2 sigma) — real.")
        elif d > SIGMA:
            notes.append(f"Improved by {d:+.4f}, between 1 and 2 sigma — confirm across seeds before trusting.")
        elif abs(d) <= SIGMA:
            notes.append(f"WITHIN NOISE ({d:+.4f}, sigma={SIGMA}) — no evidence this helped.")
        else:
            notes.append(f"WORSE by {d:+.4f} — revert.")
    if prim < REF["random"]:
        notes.append("BELOW RANDOM — treat as a bug, not a result. Check the pipeline before believing it.")
    if prim > REF["oracle"]:
        notes.append("ABOVE THE ORACLE CEILING — impossible. There is a leak or an evaluation bug.")
    worst = None
    bh = rep["segments"]["by_user_history"]
    if bh:
        worst = min(bh.items(), key=lambda kv: kv[1]["primary"])
        notes.append(f"Weakest user segment: {worst[0]} history "
                     f"(primary {worst[1]['primary']:.4f} over {worst[1]['users']} users).")
    notes += rep["health"]["warnings"]
    if rep["training"].get("hint"):
        notes.append(rep["training"]["hint"])
    if rep["regression_vs_previous_best"].get("_hint"):
        notes.append(rep["regression_vs_previous_best"]["_hint"])
    rep["notes"] = notes
    return rep


def format_for_llm(rep: dict) -> str:
    """Compact human/LLM-readable rendering. Keeps context small."""
    h = rep["headline"]
    L = [
        f"VALID  GAUC {h['GAUC']:.4f} | nDCG@5 {h['nDCG@5']:.4f} | primary {h['primary']:.4f}",
        f"  vs FM baseline (0.6015): {h['vs_baseline_fm']:+.4f}   "
        f"({h['pct_of_headroom']:.1f}% of headroom to oracle 0.8484)",
    ]
    if h["vs_current_best"] is not None:
        L.append(f"  vs current best:        {h['vs_current_best']:+.4f}   (noise sigma = {rep['noise_floor_sigma']})")
    t = rep["training"]
    if t and t.get("epochs_run"):
        line = f"TRAINING  best epoch {t.get('best_epoch')}/{t.get('epochs_run')}"
        # train_loss is absent whenever the run reported no per-epoch loss (a
        # tree model has none). Formatting it unconditionally crashed the run.
        lo, hi = t.get("train_loss_first"), t.get("train_loss_last")
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            line += f" | train loss {lo:.4f} -> {hi:.4f}"
        line += f" | overfitting={t.get('overfitting')}"
        if t.get("note"):
            line += f" | {t['note']}"
        L.append(line)
    for title, key in (("BY USER HISTORY", "by_user_history"),
                       ("BY GROUP SIZE", "by_group_size"),
                       ("BY COMPOSITION", "by_composition")):
        s = rep["segments"].get(key) or {}
        if s:
            L.append(title + "  " + " | ".join(
                f"{n}: {v['primary']:.4f} (n={v['users']})" for n, v in sorted(s.items())))
    reg = rep.get("regression_vs_previous_best") or {}
    if reg.get("_summary"):
        g, b = reg["_summary"]["improved"], reg["_summary"]["regressed"]
        if g: L.append("IMPROVED SEGMENTS  " + ", ".join(g[:5]))
        if b: L.append("REGRESSED SEGMENTS " + ", ".join(b[:5]))
    if rep["notes"]:
        L.append("NOTES")
        L += [f"  - {n}" for n in rep["notes"]]
    return "\n".join(L)

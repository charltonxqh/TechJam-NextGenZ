"""What the agent knows and sees each iteration.

Design principle: give the agent FACTS and CONSTRAINTS, never conclusions.

We know (from working through the metric definition) that a within-user
listwise objective should beat pointwise logloss. That conclusion is
deliberately NOT written here. Instead we state the underlying fact — both
metrics are invariant to adding a per-user constant — and let the agent derive
what to do. If we hardcoded the answer, the autonomy and innovation claims
would be fiction; the agent would just be pressing play on our idea.

Everything asserted below is either from the official spec/starter kit or was
measured by us in eda.py. Nothing is guessed.
"""
from __future__ import annotations

import json

SYSTEM = """You are an autonomous ML research engineer competing on a recommender-system \
benchmark. You work in a loop: form ONE hypothesis, implement it, measure it on validation, \
and learn from the result.

You are judged on:
  * the converged validation-best score (a delta over the official baseline)
  * the QUALITY OF YOUR REASONING - what you chose to try and why. A well-argued
    hypothesis that fails scores better than an unexplained tweak that works.
  * robustness - recovering from your own errors rather than stalling.

Be a scientist, not a hyperparameter sweeper. State a mechanism, predict an outcome,
say what would falsify it, then test exactly that. Change ONE thing per iteration so
the measurement is attributable."""


TASK = """## Task

Within-user ranking of logged impressions on KuaiRand-Pure (short-video feed).

  label      : long_view (0/1)
  metrics    : GAUC and nDCG@5;  primary = mean(GAUC, nDCG@5)
  ranking    : each user's own impressions are ranked against each other.
               There is NO cross-user comparison and NO retrieval from a catalogue.
  splits     : train 20220408-0421 (1,141,112 rows) / valid 20220422-0428 (124,909 rows)
               A hidden test split exists but you can never see it. You develop on
               train, you measure on valid.

## A mathematical property of the scoring you should reason about

Both metrics depend ONLY on the ordering of scores within a single user's group.
Scores are compared only inside a user, never across users. The absolute values
are never used - only the relative order.

The general form: applying ANY strictly increasing function f_u to every score of
user u leaves both GAUC and nDCG@5 exactly unchanged. That covers, and rules out
as no-ops:

    adding a per-user constant          x + c_u
    multiplying by a per-user scale     s_u * x        (s_u > 0)
    per-user z-scoring                  (x - mean_u) / std_u
    per-user min-max or rank scaling
    any per-user calibration or temperature

Each of these is a strictly increasing per-user map, so each is provably
incapable of changing either metric by more than floating-point noise. Do not
spend an iteration measuring one - the answer is 0.0000 by construction. Only a
transform that changes the ORDER of two items belonging to the SAME user can
move the score.

Consider carefully what that implies about which parts of a model's output are
measured and which are discarded, and whether the current training objective
spends its capacity accordingly.

## Reference scores (validation)

  random scoring      0.4834      <- sanity floor; below this means a bug
  item popularity     0.5807
  OFFICIAL BASELINE   0.6015      <- the number to beat (FM, k=16, lr=0.001)
  oracle ceiling      0.8484      <- perfect ranking. NOT 1.0, because 30.3% of valid
                                     users have no positive label (their nDCG is 0 for
                                     any model) and 11.9% are all-positive (always 1).

So the realistic headroom is ~0.247, not ~0.40. A gain of +0.01 is substantial here.
Per-seed noise is sigma = 0.0008: any delta under ~0.002 is not evidence."""


RULED_OUT = """## Already tested - do NOT spend an iteration re-testing these

Measured by the organisers on this exact benchmark:
  - Larger embedding dimension k: k=8/16/32 gave 0.5895/0.5902/0.5887. Flat.
    Model CAPACITY is not the bottleneck.
  - More static feature fields: 5 fields -> 13 fields gave 0.5940 vs 0.5950.
    Flat to slightly worse. Naive feature engineering does not help here.

Measured by us (exploratory data analysis on the training data):
  - Grouping by (user, date) instead of (user): 28.1% of train groups become
    singletons, and in valid the median (user,date) group size is 1. Evaluation
    groups by user_id alone. Use (user) if you need groups.
  - Duration vs label is a shallow inverted-U (rate 0.281 at 7.9s, peaks 0.376 at
    ~104s, falls to 0.318 at 287s), spread 0.273-0.376 against a 0.337 base rate.
    A linear duration term would mis-model this; the existing 10-way bucketing
    already fits the shape.
  - Auxiliary signals is_follow / is_comment / is_forward / is_hate have positive
    rates of 0.0004-0.0026. Far too sparse to carry gradient. Do not build
    multi-task heads on them.

## Hyperparameter tuning is a poor use of an iteration

Observed in an earlier run of this agent: after one genuinely novel change, two
consecutive iterations were spent adjusting constants of that same change
(weight decay 1e-6 -> 1e-3; pair-sampling density 2 -> 8). Both lost ground, and
under the convergence rule two such iterations put the whole run one step from
being terminated.

Constants are not mechanisms. Unless you have a specific argument that one
particular constant is the binding constraint - and can say why - propose a
DIFFERENT MECHANISM instead: a different objective, a different source of
signal, a different way of using the data. There are many untried mechanisms
here and very little to gain from the constants of the one you just wrote."""


FACTS = """## Measured facts about the data (from our EDA - trust these)

  User history length (train): mean 43.5, median 31, p90 97, max 809.
    NOTE: tens of interactions per user, NOT hundreds. Long-sequence retrieval
    architectures (e.g. SIM) have no problem to solve here.
  Group sizes: train 43.5 rows/user average; valid 5.6; test 7.1.
    The training and evaluation group-size distributions differ by ~8x.
  User composition: train is 92.7% label-discriminative, valid only 57.8%
    (30.3% all-negative, 11.9% all-positive). This is a consequence of group size.
  Cold users: 1.9% of valid users never appear in train (3.3% in test); their
    user_id maps to an UNK slot.
  Label drift: long_view rate falls 0.337 (train) -> 0.313 (valid) -> 0.314 (test).
  Usable auxiliary signals: is_click 46.3% positive, play_time_ms (dense numeric).
  Feature fields available in the encoded matrix (5 columns, all categorical ids):
    user_id, video_id, author_id, tab, dur_bucket

## The baseline model (what you are beating)

A Factorization Machine over those 5 fields: score = b + sum(w_i) + sum over all
pairs of <v_i, v_j>. k=16, Adam lr=0.001, L2 1e-6, batch 8192, up to 40 epochs with
early stopping (patience 4). It trains with POINTWISE binary cross-entropy against
the 0/1 long_view label, on shuffled individual rows. It reaches valid primary 0.6015,
peaking at epoch 7 and early-stopping at epoch 11."""


CONTRACT = """## Code contract

Return a complete, self-contained Python module defining exactly:

    def run(D, seed=0):
        # D["Xtr"] int32 (1141112, 5)  encoded ids, offset into one embedding table
        # D["ytr"] float32             long_view 0/1
        # D["utr"] list[str]           user id per row
        # D["Xva"], D["yva"], D["uva"] same, 124,909 validation rows
        # D["dim"] int                 vocabulary size for the shared table
        # D["fields"]                  user_id, video_id, author_id, tab, dur_bucket
        # aux targets, TRAIN ONLY, aligned to Xtr (labels, never inputs):
        #   D["aux_is_click"] 46.3% pos · D["aux_play_time_ms"] 86% nonzero
        #   D["aux_is_like"] 1.9% · D["aux_is_profile_enter"] 2.5%
        return valid_scores, history

  valid_scores : 1-D, len == len(D["yva"]), same order as D["Xva"]. NaN/Inf rejected.
  history      : [{"epoch": int, "train_loss": float, "valid_primary": float}, ...]

## Richer features and ready-made models (optional, via tools)

    # investigation phase:
    build_features({"name":"f1", "categorical":["user_id","video_id","author_id","tab"],
                    "derived":["hour","dur_bucket","video_age"],
                    "video_meta":["tag"], "video_stats":true, "aggregates":true})
    train_model({"family":"catboost", "feature_handle":"f1",
                 "params":{"loss":"QueryRMSE","depth":6,"iterations":600},
                 "budget_s":500})     -> {"prediction_id":"catboost_ab12",
                                          "valid_primary":0.6040}
    blend({"prediction_ids":["catboost_ab12","lightgbm_cd34"]})

    # then in run():
    from models import train, blend, load_scores
    r = train("catboost", "f1", {...}, budget_s=500, seed=seed)
    if "error" in r: raise RuntimeError(r["error"])   # ALWAYS check
    return load_scores(r["prediction_id"]), [{"epoch":1, "train_loss":0.0,
                                              "valid_primary": r["valid_primary"]}]

train_model runs the real CatBoost/LightGBM/XGBoost and handles group ordering
and categorical indices; hand-writing that scored 0.5961 where the tool scored
0.6040. The handle must be built by the TOOL first. Ranking objective key is
`loss` for catboost, `objective` for lightgbm/xgboost.

If you use load_features(handle) output F, use it for EVERYTHING - never mix F
and D, their encodings differ and indices will be out of range.

## Non-negotiables

  * Embeddings init at std 0.01. torch's nn.Embedding default of N(0,1) makes an
    8-field FM produce logits with std ~21, which saturates the sigmoid and the
    model never trains - this is the most common cause of a dead run here.
        nn.init.normal_(self.V.weight, std=0.01); nn.init.zeros_(self.W.weight)
  * Any strictly increasing PER-USER transform of the final scores - adding a
    constant, scaling, z-scoring, calibration - cannot change either metric.
    Do not spend an iteration measuring one.
  * Never fabricate a feature column (np.random, zeros, hashed ids). Get it from
    build_features or choose a different mechanism.
  * A result at or below item popularity (0.5807) means the model did not train.
    Fix the implementation; the mechanism was not tested.

Installed: numpy, scipy, pandas, sklearn, torch 2.13, lightgbm 4.7, catboost 1.2,
xgboost 3.4, recbole 1.2, implicit 0.7, optuna 4.9, stdlib. Nothing else.
Hyperparameter search inside run() is allowed and encouraged - tuning on
validation is permitted, and 20-50 configurations fit in the budget.
NOTE: torch and lightgbm/catboost in the SAME process segfault here (duplicate
OpenMP). Use one family per run.
Wall-clock limit per run: 20 minutes. The baseline trains in 8 seconds.

Rules:
  - NEVER modify or reimplement evaluate.py. Import it.
  - You cannot access the test split. Do not try.
  - Select on validation only.
  - ONE conceptual change per iteration."""


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string",
                       "description": "The mechanism you are testing, in one or two sentences. State WHY it should work, not just what you are changing."},
        "reasoning": {"type": "string",
                      "description": "Your derivation. Reference the specific facts or metric properties that led you here."},
        "prediction": {"type": "string",
                       "description": "What validation primary you expect, and roughly why."},
        "falsifier": {"type": "string",
                      "description": "What observation would prove this hypothesis wrong."},
        "stage": {"type": "string",
                  "description": "Which pipeline stage this change targets: features, model, loss, training, tuning, ensembling, or evaluation."},
        "change_summary": {"type": "string",
                           "description": "The one thing that differs from the current best."},
        "code": {"type": "string",
                 "description": "The complete Python module. Must define run(D, seed=0)."},
    },
    "required": ["hypothesis", "reasoning", "prediction", "falsifier", "stage", "change_summary", "code"],
}


def build_prompt(best_code: str, best_primary: float, memory: str,
                 last_feedback: str, iteration: int, budget: dict,
                 tree: str = "", incumbent: float | None = None,
                 drafting: bool = False, draft_index: int = 0,
                 draft_total: int = 0) -> str:
    """Assemble the per-iteration prompt."""
    import findings
    import tools
    # FACTS is deliberately NOT included any more. It contained numbers we
    # measured, several with a conclusion attached ("SIM has no problem to solve
    # here"), which is an answer rather than a fact and makes any finding the
    # agent reports partly ours. tools.inspect() returns the same numbers on
    # request, so the agent derives them by asking.
    # aliased: the `memory` parameter of this function is the WITHIN-RUN history,
    # which is a different thing from the cross-run store.
    import memory as store
    import stages
    # Only the high-signal slice of the store goes in the prompt; the agent pulls
    # the rest on demand with the recall() tool. Negatives are what stop it
    # re-running measured dead ends, so they are never trimmed away silently.
    parts = [TASK, RULED_OUT, tools.describe(),
             stages.as_prompt_section(),
             store.as_prompt_section(limit=14), CONTRACT]

    if drafting:
        parts.append(DRAFT + f"\n\nThis is draft {draft_index} of {draft_total}.")
    if tree:
        parts.append(tree)

    # A draft is independent by definition, so the "you are extending a weaker
    # branch" notice does not apply to it.
    if (not drafting) and incumbent is not None and incumbent > best_primary + 1e-9:
        # We are deliberately expanding a node that is NOT the best. Say so
        # plainly, or the agent reads its starting point as a regression it
        # caused and spends the iteration undoing it.
        parts.append(f"""## You are expanding a non-incumbent branch

The solution below scores {best_primary:.4f}, which is {incumbent - best_primary:.4f}
BELOW the current best of {incumbent:.4f}. That is deliberate, not a mistake, and
not something you did.

This branch was kept because a weaker first step can be the foundation of a
stronger mechanism - switching model family, or adding a feature block, often
costs score until the accompanying loss function and hyperparameters catch up.
Your job this iteration is to take the NEXT step along THIS branch and make it
pay off. Do not revert it to the incumbent; the incumbent is safe and is what
gets submitted if nothing beats it.""")

    parts.append(f"""## Where you are

Iteration {iteration}. Validation primary of the solution you are extending: {best_primary:.4f}
(official baseline 0.6015, oracle ceiling 0.8484).

Budget: {budget['iterations_left']} iterations left, {budget['minutes_left']:.0f} minutes left.
STOPPING RULE: if validation primary does not improve by more than 0.002 across
3 consecutive iterations, the run is declared converged and ENDS - locking in
whatever the best-so-far is. You have had {budget['stall_count']} consecutive
non-improving iterations. Spend iterations on ideas with a real mechanism behind
them; a run of weak ideas ends the run early.""")

    if memory:
        parts.append("## What you have already tried\n\n" + memory)

    if last_feedback:
        parts.append("## Result of your last attempt\n\n" + last_feedback)

    parts.append("## Current best solution (your starting point)\n\n```python\n"
                 + best_code + "\n```")

    if drafting:
        parts.append("""## Now

Build the representation with build_features, then TRAIN IT WITH train_model and
check the score before you write any code. Both are tool calls. Only once a
configuration measures well should run() reproduce it via train/load_scores.

Do not hand-write a gradient-boosting model. The last draft built a good feature
set, then wrote its own LightGBM and scored 0.5798; the same data through
train_model scores 0.6014. Hand-writing the plumbing - group ordering,
categorical indices, iteration counts - inside one attempt with no debugger is
where drafts die, not in the choice of model.

You may change the feature set, the model family and the objective all at once -
that is what a draft is for, and on this benchmark it is the only way past the
baseline. Return JSON with the required fields.""")
    else:
        parts.append("""## Now

You may first call tools to investigate - inspect the data, list columns, build a
feature set, or search the literature. Reply with {"tool": name, "args": {...}} and
the result comes back for you to act on.

When you are ready, propose and implement ONE change. Return JSON with the required
fields. The code must be the COMPLETE module, not a diff.""")

    return "\n\n".join(parts)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "description": "Tool name to call."},
        "args": {"type": "string",
                 "description": "JSON object of arguments, as a string."},
        "why": {"type": "string",
                "description": "What you expect to learn and how it changes your "
                               "next hypothesis."},
    },
    "required": ["tool", "args", "why"],
}


def gather(llm, prompt: str, max_calls: int = 6, log=print) -> str:
    """Let the agent call tools before it proposes anything.

    Returns a transcript of (call, result) pairs to append to the prompt. The
    agent decides what to ask; we only cap how many questions it gets, so the
    investigation is its own rather than a script we wrote for it.
    """
    import json as _json
    import tools

    transcript, convo = [], prompt
    for i in range(max_calls):
        ask = (convo + "\n\n## Investigate (optional)\n\n"
               "Call ONE tool now, or reply with tool=\"none\" to stop investigating "
               "and go straight to your proposal.")
        try:
            r = llm.ask_json(ask, TOOL_SCHEMA, max_output_tokens=4000)
        except Exception as e:                    # investigation is best-effort
            log(f"    tool phase ended ({type(e).__name__})")
            break
        name = (r.get("tool") or "none").strip()
        if name in ("none", "", "stop"):
            break
        try:
            args = _json.loads(r.get("args") or "{}")
        except _json.JSONDecodeError:
            args = {}
        out = tools.call(name, args)
        text = _json.dumps(out)[:2500]
        log(f"    tool {i+1}: {name}({args}) -> {text[:110]}")
        transcript.append(f"### {name}({args})\nwhy: {r.get('why','')}\n{text}")
        convo = prompt + "\n\n## Your investigation so far\n\n" + "\n\n".join(transcript)

    return "\n\n".join(transcript)


def format_memory(attempts: list, limit: int = 12) -> str:
    """Compact history. Failures are kept in full - they are what prevents repeats."""
    if not attempts:
        return ""
    lines = []
    for a in attempts[-limit:]:
        status = (f"KEPT   primary {a['primary']:.4f} ({a['delta']:+.4f})" if a.get("kept")
                  else f"REVERTED primary {a['primary']:.4f} ({a['delta']:+.4f})" if a.get("primary") is not None
                  else f"FAILED  [{a.get('failure_class')}] {(a.get('error') or '')[:110]}")
        lines.append(f"- iter {a['iteration']}: {a['change_summary']}\n"
                     f"    hypothesis: {a['hypothesis'][:200]}\n"
                     f"    outcome:    {status}")
        if a.get("lesson"):
            lines.append(f"    lesson:     {a['lesson']}")
    return "\n".join(lines)


DRAFT = """## You are DRAFTING an independent solution, not editing one

This is a draft step. Write a COMPLETE solution from scratch. You are not
constrained by, and should not try to preserve, any existing solution.

Draft steps exist because of a measured property of this benchmark. The official
FM baseline is a tuned local optimum for the five pre-encoded fields in D: on
that representation NO stronger model beats it (measured on validation, same
protocol) —

    FM baseline, 5 fields                 0.6015
    CatBoost QueryRMSE, 5 fields          0.5956
    CatBoost YetiRank,  5 fields          0.5944
    LightGBM lambdarank, 5 fields         0.5994

    CatBoost QueryRMSE, richer features   0.6039

Escaping it needs the feature set AND the model changed TOGETHER; either alone
is worse than doing nothing. A single-change edit therefore cannot get there,
which is why this step is allowed to change everything at once.

So choose a representation and a model as one decision. Build the feature set
you want with the build_features tool, then train on it. Make this draft
DIFFERENT from the other drafts in this run — a different model family, a
different feature set, a different objective. Diversity across drafts is the
point; the best one becomes the solution the rest of the run improves."""

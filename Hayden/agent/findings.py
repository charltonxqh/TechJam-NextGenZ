"""Cross-run findings — knowledge that survives the end of a run.

Within one run the agent remembers what it tried. But state resets between
runs, so verified discoveries were being lost and rediscovered from scratch:

    run 1  found intra-user pairwise BPR      -> +0.0020
    run 2  found intra-user pairwise BPR      -> +0.0018   (rediscovered)
    run 3  found DeepFM (an MLP over the
           same embeddings)                   -> +0.0017

Three runs, two distinct mechanisms, and **nobody ever tried them together**.
That matters because the benchmark's convergence rule needs >0.002 of gain
across 3 iterations and each single mechanism yields only ~0.002 — so gains
must STACK for a run to survive.

This module carries verified findings into the next run's context. It is
evidence the agent produced itself, not a solution we are handing it; the
combination is proposed as an untested question, not an answer.
"""
from __future__ import annotations

import json
import pathlib

STATE = pathlib.Path(__file__).resolve().parent / "state"
FILE = STATE / "findings.json"

# Seeded from runs 1-3 (see docs/RUN-LOG.md). Each entry is a measurement the
# agent made, not a claim we invented.
DEFAULT = [
    {
        "mechanism": "Intra-user pairwise ranking loss (BPR over (pos, neg) pairs from "
                     "the same user) replacing pointwise BCE on shuffled rows",
        "measured_delta": "+0.0005 averaged over 3 seeds "
                          "(single-seed runs showed +0.0020/+0.0018/+0.0016, which "
                          "did not survive replication)",
        "verdict": "WITHIN NOISE (sigma=0.0008). Not a demonstrated improvement.",
    },
    {
        "mechanism": "Intra-user listwise cross-entropy over each user's impression "
                     "group, added to pointwise BCE. NOTE: normalisation matters. "
                     "SOFTMAX normalisation measured -0.0019; SIGMOID normalisation "
                     "(ListCE, arXiv:2506.12756 eq. 8) measured +0.0000.",
        "measured_delta": "softmax -0.0019 / ListCE-sigmoid +0.0000",
        "verdict": "The earlier NEGATIVE verdict was an implementation artefact. "
                   "Softmax fights the calibration objective under binary relevance; "
                   "sigmoid normalisation does not. Correctly implemented, listwise "
                   "ranking is NEUTRAL here, not harmful.",
    },
    {
        "mechanism": "GroupCE: residual vector quantization of user embeddings into "
                     "hierarchical codes, ListCE within same-prefix user groups, "
                     "uncertainty-weighted across levels, plus a straight-through "
                     "calibration loss on the quantized embedding (arXiv:2506.12756)",
        "measured_delta": "+0.0003 at best (L=3, k=128), within noise",
        "verdict": "NEUTRAL on this benchmark. Caveat: at k=128/L=3 the deepest "
                   "groups are near-singletons contributing no gradient, so the "
                   "apparent improvement with larger k may just be the auxiliary "
                   "loss vanishing toward the control.",
    },
    {
        "mechanism": "DeepFM (MLP branch beside the bilinear term) trained with "
                     "intra-user pairwise loss - i.e. both positive-looking ideas combined",
        "measured_delta": "-0.0000 averaged over 3 seeds",
        "verdict": "NEUTRAL. The two ideas do not stack.",
    },
    {
        "mechanism": "Joint objective L = L_BCE + L_BPR",
        "measured_delta": "-0.0019",
        "verdict": "NEGATIVE",
    },
    {
        "mechanism": "Raising embedding weight decay 1e-6 -> 1e-3",
        "measured_delta": "-0.0044",
        "verdict": "NEGATIVE",
    },
    {
        "mechanism": "Increasing BPR pair-sampling density (2 -> 8 pairs per positive)",
        "measured_delta": "-0.0012",
        "verdict": "NEGATIVE - more pairs of the same kind add no information",
    },
    {
        "mechanism": "Per-user empirical log-odds added as fixed logit offsets",
        "measured_delta": "-0.0001",
        "verdict": "NEUTRAL - a clean control: per-user constants are invisible to "
                   "the metric, exactly as the invariance property predicts",
    },
    {
        "mechanism": "Multi-task: auxiliary BCE head on is_click sharing the embedding "
                     "table (two independent implementations)",
        "measured_delta": "-0.0005 and -0.0001",
        "verdict": "NEUTRAL, replicated. Auxiliary click supervision adds nothing.",
    },
    {
        "mechanism": "Feature fields hourmin (time of day) + tag (video category) + "
                     "video_age (days from upload_dt). Added to the official 5 fields.",
        "measured_delta": "+0.0010 GAUC together; individually +0.0005/-0.0001/+0.0003",
        "verdict": "POSITIVE and adopted. The trio beats each alone - when/what/how-fresh "
                   "are complementary. music_id was tested and REJECTED (-0.0017): its "
                   "cardinality is so high that each embedding is fitted on a handful of rows.",
    },
    {
        "mechanism": "Rank-averaging ensemble members (within-user percentile ranks) "
                     "instead of averaging raw logits",
        "measured_delta": "+0.0004",
        "verdict": "POSITIVE and adopted. Members are calibrated differently; only "
                   "intra-user order is scored.",
    },
    {
        "mechanism": "LightGBM lambdarank with platform item statistics "
                     "(long_time_play_cnt/show_cnt etc.), groups = users",
        "measured_delta": "+0.0050 to LightGBM from the statistics; a further +0.0095 "
                          "from REMOVING the sparse behaviour-history features",
        "verdict": "POSITIVE as an ENSEMBLE MEMBER. Alone it is ~0.6630 GAUC (below the "
                   "FM), but it correlates only 0.757 with the FM family and carries 0.3 "
                   "blend weight. Trees use continuous features natively; the FM cannot.",
    },
    {
        "mechanism": "CatBoost (YetiRank) with NATIVE high-cardinality categorical "
                     "handling — ordered target statistics rather than the label "
                     "encoding LightGBM gets — over the 8 id fields + 12 platform "
                     "item statistics",
        "measured_delta": "valid 0.6715 alone, the single strongest member measured on "
                          "this benchmark — above the 8-seed DCN's 0.6710 and far above "
                          "LightGBM lambdarank's 0.6680",
        "verdict": "POSITIVE and adopted. Greedy validation selection picks it FIRST, "
                   "ahead of every neural model. The tree family had been explored with "
                   "exactly one member while the neural family had ten; that asymmetry "
                   "was hiding the best ingredient. Other tree objectives were weaker: "
                   "lgb_bin 0.6689, xgb rank:pairwise 0.6681, lgb rank_xendcg 0.6675.",
    },
    {
        "mechanism": "Item-item collaborative filtering (cosine over the user x item "
                     "long_view matrix), exposure-matrix CF, and truncated-SVD latent CF",
        "measured_delta": "cf_cos test 0.5785, cf_svd 0.5545, cf_exp 0.5535 "
                          "(popularity = 0.5807, random = 0.4834)",
        "verdict": "NEGATIVE. Coverage is NOT the problem — 93.8% of test rows have a "
                   "user with >=1 training positive, mean 19 positives. The neighbourhood "
                   "signal simply is not there: CF lands at popularity level, so it "
                   "recovers item popularity rather than user taste. Distinct from the "
                   "author-history dead end, which failed on coverage instead.",
    },
    {
        "mechanism": "DCN-V2 full-matrix cross layer (arXiv:2008.13535) and DCN-Mix "
                     "low-rank mixture-of-experts cross, vs DCN v1's vector cross",
        "measured_delta": "dcnv2 0.6670, dcnmix 0.6668, dcnv2_x2 0.6666, dcnv2_lc 0.6653, "
                          "vs DCN v1 0.6672",
        "verdict": "NEUTRAL-to-NEGATIVE. The published upgrades do not transfer here, "
                   "consistent with every other capacity increase on this dataset.",
    },
    {
        "mechanism": "Hyperparameter tuning on validation (lam, lr, epochs, l2, k)",
        "measured_delta": "+0.0002 over the default, 3 seeds - within noise",
        "verdict": "NEUTRAL. The baseline's hyperparameters are already optimal. "
                   "lr=1e-3 sits at a clean optimum; k=16 beats 32 and 64; epochs 6 = 10 "
                   "= 16. A lam=0.5 spike on one seed did NOT replicate.",
    },
    {
        "mechanism": "Recency weighting of training rows, exp((date-last)/tau)",
        "measured_delta": "+0.0006 on validation (tau=14d), -0.0006 on test",
        "verdict": "NEGATIVE. The validation gain did not transfer - textbook "
                   "hyperparameter overfitting across the temporal gap.",
    },
]

# Everything below is measured fact, not suggestion.
CEILING_NOTE = """## Why the score is hard to move (measured)

Feature importance from LightGBM trained on 144 features (5 ids + 51 video
statistics as rates + 51 as log-counts + 7 video basics + 30 user attributes):

    cat_user                        274,093
    cat_tab                          75,452
    cat_video                        73,293
    cat_author                       65,777
    v_rate_long_time_play_user_num   18,538   <- best of 139 added features
    u_follow_user_num                   966
    u_register_days                     900   <- user attributes are noise-level

The four identity columns carry almost all the signal. Every external feature
file combined contributes a rounding error. Adding information is not the lever.

Every compute-scaling knob is at or past its optimum:
    training epochs   6 = 10 = 16          (identical)
    seed ensembling   caps at +0.0010, flat beyond n=10
    embedding dim k   16 > 32 > 64
    LightGBM rounds   300 > 600
    LightGBM leaves   63 > 127 > 255

More capacity and more training make it worse, not better. 1.14M rows over 27K
users cannot support a larger model.

What HAS worked, every time, is adding an ENSEMBLE MEMBER that is comparable in
strength and decorrelated from the others. That is the only reliable lever
found in 25+ measured mechanisms."""

_OLD = [
    {
        "mechanism": "User x item-attribute behaviour-history features (user x author, "
                     "user x duration-bucket, user x tab counts/positives/rates; "
                     "available as D['Htr'] / D['Hva'])",
        "measured_delta": "each feature scored ALONE on validation: best is user x tab "
                          "positives at 0.5240, vs random 0.4834 and popularity 0.5807",
        "verdict": "WEAK - see the coverage note below before building on these",
    },
]

# A structural observation that explains the cluster of null results above, and
# which the agent should factor into its next hypothesis.
CONTEXT_NOTE = """## Why the objective-function direction appears exhausted

Every attempt to replace or augment the pointwise objective with a ranking
objective - pairwise, listwise, and joint - has measured at or below zero once
averaged over seeds. That is surprising if you reason only from the metric's
invariance property.

A likely explanation, worth taking into account: the baseline ALREADY selects its
final weights by early stopping on validation primary, which is itself a ranking
metric. So the pointwise-trained model is model-SELECTED for ranking even though
it is TRAINED for calibration. That selection step appears to absorb most of what
switching the training objective was supposed to deliver.

Treat "align the loss with the metric" as tested and unproductive on this
benchmark. Signal has to come from somewhere else: information the model does not
currently use, or structure in the data it does not currently exploit."""

COVERAGE_NOTE = """## Why the behaviour-history features are weak (measured, not assumed)

The obvious history feature - "has this user engaged with this video's AUTHOR
before?" - is empty almost everywhere in this dataset:

    user has seen this author before   3.4% of validation rows
    user long-viewed this author        0.7%
    user has seen this exact video      1.6%

With 27K users, 7.6K items and ~43 interactions per user, repeat exposure to the
same author barely happens. The only history features with real coverage cross the
user against `tab` (6 values) or `dur_bucket` (10 values), which are far too coarse
to express taste - and they score ~0.52 alone, barely above random.

So behaviour history is a core feature in industrial recommenders, but the sampling
that produced this benchmark destroyed the repeat-exposure structure it depends on.
Treat per-user history as a measured dead end here unless you can construct a
variant with materially better coverage."""

OPEN_QUESTIONS = [
    "Training groups average 43.5 rows per user; evaluation groups average 5.6. "
    "The training objective therefore sees a systematically different problem shape "
    "from the one it is scored on. Re-weighting or re-sampling training users to "
    "match the evaluation distribution has been proposed once but never ran "
    "successfully - it remains unmeasured.",
    "No run has tried ensembling across model families. LightGBM is installed and "
    "its 'lambdarank' objective is a gradient-boosted ranker with completely "
    "different inductive bias from an FM; rank-averaging two dissimilar models is a "
    "standard way to gain when neither alone can be improved further.",
    "No run has tried averaging the SAME model across several random seeds and "
    "rank-averaging the results, which reduces variance rather than bias.",
    "play_time_ms (dense, 86% nonzero) has never been used as a regression target, "
    "only is_click as a classification target.",
]


def load() -> list:
    if FILE.exists():
        try:
            return json.loads(FILE.read_text())
        except json.JSONDecodeError:
            pass
    return list(DEFAULT)


def save(findings: list) -> None:
    STATE.mkdir(exist_ok=True)
    FILE.write_text(json.dumps(findings, indent=2))


def record(mechanism: str, delta: float, kept: bool) -> None:
    """Append a measurement from the current run."""
    f = load()
    f.append({
        "mechanism": mechanism[:300],
        "measured_delta": f"{delta:+.4f}",
        "verdict": "POSITIVE" if kept else ("NEGATIVE" if delta < -0.001 else "NEUTRAL"),
    })
    save(f)


def as_prompt_section() -> str:
    f = load()
    lines = ["## Measured results from previous runs of this agent",
             "",
             "These are measurements, not suggestions. Each was produced by running the",
             "code and scoring it on validation.",
             ""]
    for x in f:
        lines.append(f"- {x['mechanism']}")
        lines.append(f"    measured: {x['measured_delta']}   verdict: {x['verdict']}")
    lines += ["", CONTEXT_NOTE, "", COVERAGE_NOTE, "", CEILING_NOTE, "", "## Known-unmeasured", ""]
    lines += [f"- {q}" for q in OPEN_QUESTIONS]
    lines += ["",
              "Measurement discipline: single-seed deltas on this benchmark have been",
              "wrong by 2-4x. Several apparent gains of +0.002 averaged to +0.0005 or",
              "less over three seeds. Per-seed sigma is 0.0008. Do not trust a small",
              "delta from one run, and do not build a subsequent hypothesis on one."]
    return "\n".join(lines)


if __name__ == "__main__":
    print(as_prompt_section())

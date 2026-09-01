# All Experiments — complete results table

Every mechanism tested, with source and measurement. Validation split, 3 seeds
unless noted. Baseline = 0.6015, per-seed σ = 0.0008.

Related: [[Paper-References]] · [[Run-Log]] · [[Critical-Review]]

---

## Results

| # | mechanism | source | result | Δ vs baseline |
|---|---|---|---|---|
| 1 | **5-seed logit ensemble** | agent | **0.6027** | **+0.0012 ✅** |
| 2 | LightGBM blend (w=0.2) | cross-family | 0.6023 | +0.0004 (noise) |
| 3 | baseline control | — | 0.6014 | −0.0001 |
| 4 | IPS exposure debias (α=0.5) | MMRF 2405.01847 | 0.6011 | −0.0004 |
| 5 | CWM censored aux (λ=0.2) | CWM KDD 2024 | 0.6010 | −0.0005 |
| 6 | pairwise BPR loss | agent | 0.6005* | +0.0005 (noise) |
| 7 | aux plain MSE on watch time | — | 0.6006 | −0.0009 |
| 8 | CWM censored aux (λ=0.5) | CWM | 0.6005 | −0.0010 |
| 9 | cold-start upweight (α=0.3) | 2506.12756 | 0.6002 | −0.0013 |
| 10 | multi-task on `is_click` ×2 | — | ~0.6013 | −0.0005, −0.0001 |
| 11 | per-user logit offsets | agent | 0.6014 | −0.0001 |
| 12 | multi-task on `play_time_ms` | — | 0.6001 | −0.0014 |
| 13 | joint BCE + BPR | agent | 0.5996 | −0.0019 |
| 14 | listwise **softmax** ⚠️ | agent | 0.5996 | −0.0019 |
| 14b | listwise **ListCE / sigmoid** ✅ | arXiv:2506.12756 | 0.6015 | +0.0000 |
| 14c | **GroupCE** (RVQ + hierarchical ListCE + STE) | arXiv:2506.12756 | 0.6018 | +0.0003 |
| 15 | per-user loss weight 1/N_u | agent | 0.5947 | −0.0068 |
| 16 | weight decay 1e-6 → 1e-3 | agent | 0.5971 | −0.0044 |
| 17 | LightGBM lambdarank alone | cross-family | 0.5860 | −0.0155 |
| 18 | 50/50 watch-ratio blend | kit #4 | 0.5828 | −0.0187 |
| 19 | watch-ratio soft target | kit #4 | 0.5571 | −0.0444 |
| 20 | rank-normalise before ensembling | agent | 0.5845 | −0.0182 |
| 21 | behaviour-history features (best alone) | workshop | 0.5240 | −0.0775 |
| 22 | larger k / more static fields | organizers | flat | ~0 |

\* single-seed runs of BPR showed +0.0020/+0.0018/+0.0016; the 3-seed mean was +0.0005.

**One of twenty-two mechanisms produced a replicable gain.**

---

## ⚠️ A retracted finding — normalisation, not the method

We originally recorded *"listwise loss: −0.0019, NEGATIVE"* and used it to argue the
whole ranking-objective family was exhausted here. **That was our implementation, not
the method.**

Our version used **softmax** normalisation. The paper (Yan et al., ADKDD'25) specifies
**ListCE** — *sigmoid*-based normalisation — precisely because softmax fights the
calibration objective under binary relevance. Their own ablation on KuaiRand:

| objective | GAUC | gain |
|---|---|---|
| LogLoss | 0.6911 | — |
| + SoftmaxCE | 0.6920 | +0.0009 |
| + **ListCE** | 0.6932 | +0.0021 |
| **GroupCE** | 0.6953 | +0.0042 |

Re-measured on our data with the correct loss:

| | primary | Δ |
|---|---|---|
| listwise, softmax | 0.5996 | −0.0019 |
| listwise, **ListCE sigmoid** | 0.6015 | **+0.0000** |

**Switching normalisation recovered +0.0019.** Correctly implemented, listwise ranking
is *neutral* on this benchmark, not harmful. The original claim was wrong and had
propagated into the agent's priors, the run log and the README.

**Comparability, finally settled.** The paper splits KuaiRand **randomly** 70/10/20 with
stratification guaranteeing every user has positives in every subset. We use a
**temporal** split in which 30% of validation users have no positives at all and 3.3% of
test users are unseen. Their 0.6911 baseline and our 0.6610 are measuring different
problems — which resolves the discrepancy that made an earlier "~0.66 is the ceiling"
claim look both overconfident and, later, refuted.

**Scale calibration.** GroupCE's headline gain is **+0.0042 GAUC** on an easier split.
Our seed ensemble gives **+0.0028 GAUC**. Published ADKDD work operates at the same order
of magnitude we do; nobody is finding +0.05 on this data.

## Findings worth more than the table

### 1. `long_view` is NOT a watch-ratio threshold
Ranking by predicted watch ratio costs **−0.0444** — one of the largest effects measured.
The label encodes a *duration-dependent* rule, so the intuition that "the binary label
discards magnitude" is wrong here. Mean watch ratio (0.334) coincidentally almost equals
the long_view rate (0.337), which makes the wrong hypothesis look plausible.

### 2. CWM's censoring correction is real — the thing it corrects is not
| | primary |
|---|---|
| no auxiliary head | 0.6014 |
| aux plain MSE | 0.6006 |
| **aux censored one-sided (λ=0.2)** | **0.6010** |
| aux censored (λ=0.5) | 0.6005 |

The one-sided loss recovers **+0.0004** over naive MSE with **std 0.0000** — the paper's
mechanism is genuinely measurable. But auxiliary watch-time supervision is net negative
against no auxiliary head at all, and raising λ makes it worse. Censoring makes a bad
idea less bad.

⚠️ **A first attempt at this was vacuous** and produced a false negative: the one-sided
rule was applied to a watch ratio clipped to [0,1] predicted through a sigmoid, so
"prediction exceeds observation" was unreachable and the mask never fired. Detected only
because the censored and uncensored arms were identical to 4 dp at every seed. The
corrected version uses an unbounded target through a linear head, and the test script now
carries an automatic vacuity check.

### 3. Within-user ranking is largely immune to exposure bias
IPS propensity weighting: **−0.0004**, pure noise. Popularity bias determines *which*
items a user was shown; the metric only compares items they were already shown. Correcting
exposure does not change intra-set ordering.

### 4. Model diversity exists but cannot be cashed in
FM and LightGBM have a within-user rank correlation of **0.664** — genuinely
complementary. Blending still yields only +0.0004, because LightGBM is too far behind
(0.586 vs 0.602). Diversity pays only when the weaker model is close enough to contribute.

### 5. Behaviour history is impossible here, not merely unhelpful
| coverage | |
|---|---|
| user has seen this video's author before | **3.4%** |
| user long-viewed that author | 0.7% |
| user has seen this exact video | 1.6% |

27K users over 7.6K items with ~43 interactions each. The sampling that produced this
benchmark destroyed the repeat-exposure structure history features depend on.

### 6. Seed ensembling caps at +0.0010 — measured, not assumed
20 independent models, evaluating ensembles of the first *n*:

    n= 1  0.6015      n= 8  0.6022      n=15  0.6026
    n= 2  0.6019      n=10  0.6023      n=20  0.6025
    n= 5  0.6021

Fitted asymptote **0.6025 (+0.0010)**; remaining headroom beyond n=20 is **+0.00004**.
n=20 scoring below n=15 shows the measurement itself is in the noise. More seeds will not
help. Single-model std measured **0.0006** against the published σ of 0.0008.

---

## Honest caveat on the submitted number

The submission measured +0.0012 (valid) / +0.0021 (test), but the *expected* effect of
seed ensembling is **+0.0010**. Our submitted score sits on the fortunate side of that
distribution. Report +0.0010 as the method's effect size.

---

## Why the score cannot go much higher — the decisive evidence

### 1. The signal is almost entirely in identity

LightGBM trained on **144 features** (5 categorical ids + 51 video statistics as
rates + 51 as log-counts + 7 video basics + 30 user attributes), feature importance
by gain:

    cat_user                        274,093
    cat_tab                          75,452
    cat_video                        73,293
    cat_author                       65,777
    v_rate_long_time_play_user_num   18,538   <- best of 139 added features
    u_follow_user_num                   966
    u_register_days                     900
    u_fans_user_num                     798   <- user attributes are noise-level

The four identity columns dominate. The best external feature contributes 7% of
`cat_user`; the entire `user_features_pure.csv` file is three orders of magnitude
below it. **Who the user is and which video it is already encodes nearly everything
the data expresses.** The organisers reached this conclusion with 13 fields; we
confirmed it with 144, including every column the dataset ships.

### 2. Capacity is not the constraint (re-confirmed on trees)

    63 leaves,  400 rounds   GAUC 0.6680   <- best
    127 leaves, 400 rounds   GAUC 0.6676
    255 leaves, 700 rounds   GAUC 0.6666   (5x the compute, worse)

Same finding the organisers reported for embedding dimension, now reproduced in a
completely different model family.

### 3. Optimising GAUC directly changes nothing

Blend weights re-tuned to maximise GAUC rather than primary:

| | test GAUC | test primary |
|---|---|---|
| primary-optimised | **0.6657** | **0.5985** |
| GAUC-optimised | 0.6655 | 0.5982 |

The two objectives select the same solution to within noise.

### 4. The temporal split costs a fixed ~0.006

Validation GAUC reached 0.6719 (four-way blend, the best number produced). Test came
back at 0.6655 — a **−0.0064** gap seen consistently across every model. Validation is
Apr 22–28; test is Apr 29–May 8. Behaviour drifts, the label rate declines across the
window, and 3.3% of test users never appear in training.

To reach **test** GAUC 0.68 would require validation GAUC ≈ 0.686. Every lever
available on this feature set tops out around 0.672.

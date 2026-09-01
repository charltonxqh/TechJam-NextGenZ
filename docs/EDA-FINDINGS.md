# EDA Findings (Phase 0)

Script: `kuairand-starter-kit/eda.py` (read-only, ~6s). Log: [[Run-Log]] · Plan: [[Refined-Strategy]]

**Validation that the analysis is sound:** the oracle ceiling computed here reproduces the published figures exactly — valid 0.8484, test 0.8645 — as does the test user composition (27.1 / 9.2 / 63.7%). So `evaluate.py` is being driven correctly.

---

## 1. ❌ The README's history-length claim is wrong

> README: *"KuaiRand 每用户在 train 里有上百到上千条交互"* (hundreds to thousands per user)

**Actual train history length: mean 43.5, median 31, p90 97, p99 207, max 809.**

| split | users | rows/user mean | median | p90 | max |
|---|---|---|---|---|---|
| train | 26,210 | 43.5 | 31 | 97 | 809 |
| valid | 22,377 | 5.6 | 4 | 12 | 74 |
| test | 23,875 | 7.1 | 5 | 16 | 109 |

It's **tens, not hundreds-to-thousands** — off by roughly an order of magnitude.

**Consequence:** the DIN/sequence phase is **downgraded, not cancelled**. A median history of 31 is enough for target-attention to do something, but this is not the regime DIN was designed for (Alibaba's setting had hundreds-to-thousands, and SIM exists specifically to handle *thousands*). **SIM is now firmly off the table** — there is no long-sequence retrieval problem to solve. Expect a modest gain from DIN at best, and sequence modeling should no longer outrank cheaper ideas.

## 2. ✅ Group-granularity question answered — no ablation needed

I had planned an ablation on `(user)` vs `(user, date)` grouping for the listwise loss. The data settles it outright:

| grouping | train mean size | singletons (zero listwise gradient) |
|---|---|---|
| `(user)` | 43.5 | — |
| `(user, date)` | 5.8 | **28.1% of train groups** |

And in eval, `(user, date)` is degenerate: median size **1**, with **54.7% (valid) / 55.7% (test)** singletons.

Decisively: **group by `(user)` over the whole split.** This also structurally matches evaluation — `evaluate.py` groups by `user_id` alone with no date key (verified in source: `byu[u].append(...)`). `(user, date)` would throw away 28% of the training signal *and* mismatch the eval structure. **One planned ablation eliminated before costing an iteration.**

## 3. ⚠️ Correction to my own earlier reasoning

In [[Refined-Strategy]] I argued that listwise loss "automatically concentrates on the ~63.7% discriminative users" because all-same-label groups yield zero gradient. **That benefit is much smaller than I implied**, and I should correct it:

| split | all-neg | all-pos | discriminative |
|---|---|---|---|
| **train** | 5.1% | 2.3% | **92.7%** |
| valid | 30.3% | 11.9% | 57.8% |
| test | 27.1% | 9.2% | 63.7% |

The 27/9/64 split is a property of **evaluation** groups (tiny, ~5–7 rows, so often all-same by chance), not of **training** groups (43 rows, so almost always mixed). In training only **7.3%** of users would give zero gradient — so listwise "focusing" saves very little.

**The main argument is unaffected and still stands:** shift-invariance means the objective stops spending capacity on per-user base rates that the metric provably ignores. That was always the load-bearing claim; the gradient-focusing point was a secondary flourish and it does not survive contact with the data.

## 4. Multi-task: only 3 of 8 auxiliary signals are usable

Train positive rates:

| signal | rate | verdict |
|---|---|---|
| `is_click` | **0.4634** | ✅ dense, closely related — best auxiliary task |
| `long_view` (target) | 0.3366 | — |
| `play_time_ms` | dense numeric (median 4,970ms, mean 23,260ms) | ✅ the CWM/censored-regression direction |
| `is_profile_enter` | 0.0254 | marginal |
| `is_like` | 0.0187 | marginal |
| `is_comment` | 0.0026 | ❌ too sparse |
| `is_follow` | 0.0010 | ❌ too sparse |
| `is_forward` | 0.0010 | ❌ too sparse |
| `is_hate` | 0.0004 | ❌ too sparse |
| `is_rand` | 0.0000 | all-zero in standard log (as expected) |

**Plan change:** the multi-task phase targets **`is_click` + `play_time_ms` only**. The spec and README both list follow/comment/forward as candidates, but at 0.1–0.26% positive rate they cannot carry useful gradient. Dropping them removes most of the complexity from that phase.

## 5. ⬇️ Duration coupling is weaker than expected — demote that idea

I proposed finer/continuous duration treatment as an early cheap win. The data says it isn't one:

| dur bucket | median duration | long_view rate |
|---|---|---|
| 0 | 7.9s | 0.281 |
| 1 | 15.3s | 0.273 |
| 3 | 39.7s | 0.367 |
| 6 | 104.4s | **0.376** |
| 9 | 287.3s | 0.318 |

The relationship is **non-monotonic** (inverted-U, peaking mid-range) and **shallow** — the entire spread is 0.273→0.376 against a 0.3366 base rate. The existing 10-way quantile bucketing already captures a smooth curve like this well; a finer or continuous treatment has little left to extract, and a *linear* duration term would actively mis-model the inverted-U.

**Plan change: demote duration engineering** out of the early phases. The censored-watch-time idea (Phase 5) is unaffected — it attacks a different mechanism (truncation of observed watch time), not the duration→label shape.

## 6. Cold users are a small but real gap

| split | rows w/ unseen user | distinct users unseen |
|---|---|---|
| valid | 1.59% | 1.9% |
| test | **3.62%** | **3.3%** |

Videos are effectively fully covered (0.01% unseen). So ~3.3% of test users hit the `user_id` UNK slot and get no personalization at all. Small, but it's a population where item-side/content signal is the *only* thing available — worth remembering if we later add content features.

## 7. Distribution drift is mild but consistently downward

| split | long_view rate | mean duration |
|---|---|---|
| train | 0.3366 | 97,880ms |
| valid | 0.3133 | 102,820ms |
| test | 0.3135 | 107,210ms |

By date, the label rate drifts down across the whole window (train 0.336→0.315, valid 0.319→0.290, test ~0.30–0.32). Tab mix also shifts: tab `1` grows 73.2%→76.7% while tab `0` shrinks 13.1%→8.4%.

**This is another point for the listwise loss:** a global downward drift in base rate is exactly the kind of per-split shift a shift-invariant, within-user objective is immune to, while pointwise logloss would be systematically miscalibrated by it.

---

## Net effect on the plan

| idea | before | after |
|---|---|---|
| Listwise loss | #1 | **#1, unchanged** (main argument intact; drift adds support) |
| Group granularity ablation | planned | **resolved — group by `(user)`, no iteration spent** |
| Duration engineering | early cheap win | **demoted** — effect is shallow & non-monotonic |
| Sequence / DIN | major phase | **downgraded** — histories are ~31 median, not hundreds. SIM dropped entirely |
| Multi-task | 6 aux signals | **narrowed to `is_click` + `play_time_ms`** |

Revised order: **listwise loss → multi-task (click + watch-time) → DIN (modest expectations) → censored watch-time → architecture swap.**

Headroom on valid: FM 0.6015 → oracle 0.8484, so **0.247 of realistic room.**

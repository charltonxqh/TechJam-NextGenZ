# Final Result

## Submission

**`agent/state/submission_v2.csv`** — 170,588 rows, passes `submit.py --check --split test`.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| Official FM baseline (test) | 0.6610 | 0.5282 | 0.5946 |
| **Ours (test)** | **0.6669** | **0.5316** | **0.5993** |
| delta | **+0.0059** | **+0.0034** | **+0.0047** |

**Scored delta = +0.0047.** Validation GAUC 0.6730.

For scale: the ADKDD'25 GroupCE paper reports **+0.0042 GAUC** for a purpose-built
architecture on KuaiRand — on a *random* 70/10/20 split with every user guaranteed
positives in every subset, an easier setting than this temporal split.

## The model

Rank-blended ensemble, weights selected on validation GAUC:

    0.2 * FM      5-seed rank-ensemble    (8 fields, pointwise BCE)
    0.5 * ListCE  5-seed rank-ensemble    (8 fields, BCE + intra-user ListCE)
    0.3 * LightGBM lambdarank             (5 fields + 12 platform item statistics)

Blending is on **within-user percentile ranks** — members are calibrated differently
and only intra-user order is scored.

### Feature set: 8 fields, up from the official 5

    user_id, video_id, author_id, tab, dur_bucket    (official)
  + hourmin      time of day of the impression
  + tag          video category
  + video_age    days from upload_dt to the impression date

Every column in the standard logs and video metadata was measured individually and in
combination (validation GAUC, 2 seeds):

| variant | GAUC | Δ |
|---|---|---|
| baseline 5 fields | 0.6673 | — |
| + hourmin | 0.6678 | +0.0005 |
| + tag | 0.6671 | −0.0001 |
| + video age | 0.6675 | +0.0003 |
| + music_id | 0.6655 | −0.0017 |
| **+ hour + tag + age** | **0.6682** | **+0.0010** |
| all four | 0.6667 | −0.0006 |

The trio beats each part alone — *when*, *what* and *how fresh* are complementary and
get crossed with user identity. `music_id` is excluded: cardinality in the millions
means each embedding is fitted on a handful of rows.

Vocabulary grows only 40,260 → 40,337, so this adds information, not capacity —
important given capacity has repeatedly measured as unhelpful on this benchmark.

### Measured contributions

| change | GAUC gain |
|---|---|
| hour + tag + age features | +0.0010 |
| ListCE members instead of plain FM | +0.0032 (test, 8-field) |
| rank-averaging instead of logit-averaging | +0.0004 |
| decorrelated LightGBM at weight 0.3 | +0.0008 |
| dropping sparse history features from LightGBM | +0.0095 (to LightGBM) |
| adding platform statistics to LightGBM | +0.0050 (to LightGBM) |

## Data scope

The two **standard** logs only, per the spec: *"Fixed data splits: date-based, taken
from the two standard logs... Teams develop on train + validation only."*

`log_random_4_22_to_5_08_pure.csv` (1,186,059 rows) is **not used**. It is part of
KuaiRand-Pure and would have nearly doubled the training data, but the splits are
defined over the standard logs and development is scoped to train + validation, so it
falls outside the permitted training data.

## Resource consumption

| | |
|---|---|
| agent iterations to convergence | 4 (cap 12) |
| manual interventions during the scored run | 0 |
| LLM calls | 4 |
| LLM tokens | 46,283 |
| agent wall clock | 5.0 min |
| final ensemble training | ~4 min CPU |
| GPU-hours | 0 |

## Reproduce

```bash
.venv/bin/python agent/stage_lgb.py    # LightGBM — separate process (see note)
.venv/bin/python agent/final_v2.py     # 8-field FM/ListCE + blend + submission
```

> `lightgbm` and `torch` must not be imported in the same process on Apple Silicon —
> duplicate OpenMP runtimes cause SIGSEGV or a silent hang.

## Earlier submissions (kept as fallbacks)

| file | test GAUC | primary | delta |
|---|---|---|---|
| `submission_v2.csv` | **0.6669** | **0.5993** | **+0.0047** |
| `submission_blend.csv` | 0.6657 | 0.5985 | +0.0039 |
| `submission_gauc.csv` | 0.6655 | 0.5982 | +0.0036 |
| `submission.csv` | 0.6638 | 0.5967 | +0.0021 |

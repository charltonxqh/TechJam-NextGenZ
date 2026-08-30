# Refined Strategy — Decisions & Core Insight

**This is the live plan.** Supersedes [[Architecture-Plan]] (archived). Spec: [[Docs]] · Evidence: [[EDA-Findings]] · Log: [[Run-Log]] · Scoring strategy: [[Maximizing-Score]]

> ⚠️ **Read [[Maximizing-Score]] before iteration 1.** It identifies a convergence trap that changes iteration ordering: three consecutive iterations without >0.002 total improvement **ends the run**, locking in whatever validation-best exists at that point. The 50-iteration cap is a red herring — real runs end at 8–20 iterations.

Status: **Phase 0 complete** (env, data, baselines reproduced, EDA done — see [[Run-Log]]). No model code written yet; the PyTorch port and Iteration 1 are next.

---

## 1. The Core Insight (why loss function is the big lever)

The organizers guessed "change the loss function" is the top untested direction. Working through *why* makes it much stronger than a guess, and tells us exactly which loss to use.

**Both scored metrics are pure intra-user ranking metrics.**
- GAUC: per-user AUC, averaged over users with `0 < npos < n_impressions`, weighted by positive count.
- nDCG@5: per-user, ranks within that user's impression list.

Neither metric can see anything except the *ordering of scores inside one user's group*. Formally: adding any per-user constant `c_u` to every score of user `u` changes nothing. The README already noticed a symptom of this ("pure user-side first-order terms contribute exactly 0"), but the consequence for **training** is bigger than the consequence for features:

**Pointwise logloss spends a large share of its capacity and gradient signal fitting each user's base positive rate — a quantity that provably cannot affect the score.**

That's the leak. The FM is being asked to output calibrated P(long_view) per row, which requires modeling per-user propensity; then evaluation discards exactly that component. Capacity and gradient budget are being burned on a nuisance parameter.

**Listwise softmax within the user's impression group is exactly invariant to `c_u`** (softmax is shift-invariant). So every parameter update goes into the thing that is actually scored.

Two more properties fall out for free:
- ~~A group whose labels are all-0 or all-1 produces zero gradient, so training automatically concentrates on the ~63.7% discriminative users.~~ **RETRACTED after EDA — see [[EDA-Findings]] §3.** The 27/9/64 composition is a property of *evaluation* groups (~5–7 rows, often all-same by chance), not *training* groups (~43 rows, almost always mixed). In train, 92.7% of users are already discriminative, so this "focusing" effect saves only ~7.3% — negligible. The shift-invariance argument above is the load-bearing one and is unaffected.
- Eval groups are tiny (~7 impressions/user: 170,588 test rows / 23,875 users). Full listwise softmax over a whole group is trivially cheap — **no negative sampling needed**, unlike typical listwise setups.
- **Added post-EDA:** the label rate drifts downward train→valid→test (0.337 → 0.313 → 0.314). A shift-invariant within-user objective is structurally immune to that global drift; pointwise logloss is systematically miscalibrated by it. Another point for listwise.

**→ Iteration 1 is: same FM, same 5 features, same capacity — swap pointwise BCE for within-user listwise softmax cross-entropy.** Nothing else changes, so the measured delta is attributable to the objective alone.

BPR (pairwise) is the fallback if listwise underperforms, but listwise is strictly better-motivated here: it uses the whole group at once and directly matches nDCG's list structure, whereas BPR decomposes into independent pairs and needs sampling choices.

## 2. Training group granularity — ✅ RESOLVED by EDA, no iteration needed

Listwise loss requires choosing what a "group" is. **[[EDA-Findings]] §2 settles it: group by `(user)` over the whole split.**

`(user, date)` is degenerate — 28.1% singletons in train, and **median size 1** in eval with ~55% singletons. It would discard a quarter of the training signal *and* mismatch evaluation, which groups by `user_id` alone with no date key (verified in `evaluate.py` source). The planned ablation is cancelled — answered from data instead of costing an iteration.

Also resolved: the README's "hundreds to thousands of interactions per user" is **wrong** — actual median is 31, mean 43.5. See §4 below.

## 3. Duration ↔ label coupling — ⬇️ DEMOTED after EDA

I proposed finer/continuous duration treatment as an early cheap win. **[[EDA-Findings]] §5 shows it isn't one.** The duration→`long_view` relationship is non-monotonic (inverted-U, peaking ~104s) and shallow (0.273→0.376 spread against a 0.3366 base rate). The existing 10-way quantile bucketing already fits a curve of that shape well, and a *linear* duration term would actively mis-model the inverted-U.

Demoted out of the early phases. The Phase-5 censored-watch-time idea is unaffected — it attacks truncation of observed watch time, a different mechanism.

## 4. Sequence modeling — ⬇️ DOWNGRADED after EDA

Median user history is **31**, not "hundreds to thousands" as the README claims ([[EDA-Findings]] §1). DIN target-attention remains worth trying, but with modest expectations — this is not the long-sequence regime it was designed for. **SIM is dropped entirely**: there is no long-sequence retrieval problem here to justify it.

## 5. Free win: seed averaging

Per-seed std is 0.0008. Averaging scores across ~5 seeds reduces variance and typically nets a small but reliable gain, at ~40s per seed for the FM (trivial wall-clock). Apply once a strong configuration is found, not during exploration.

---

## Decisions (you said I decide — these are my calls)

| Question | Decision | Reason |
|---|---|---|
| Framework | **Hand-rolled PyTorch. No RecBole.** | RecBole's data layer expects its own atomic-file format and its evaluation is built for full-catalog retrieval; we need "score these exact logged rows, hand the array to `evaluate.py`." Adapting it fights the framework. Each model we need (listwise FM, DIN attention, MMoE heads) is 50–150 lines of plain PyTorch. Under a time limit, control beats reuse here. |
| numpy vs torch | **Port FM to PyTorch in Phase 0**, verified to reproduce the numpy baseline within seed noise before anything is built on it. | Phases 1–4 all need custom losses/architectures; hand-deriving numpy gradients for listwise softmax, attention, and multi-task heads is slow and error-prone. The numpy-only constraint is a barrier-to-entry convenience, not a rule — the resource policy explicitly allows any library. |
| Device | **Apple Silicon MPS backend**, CPU fallback. | This is an M-series Mac. Data is small (1.14M rows) so CPU is survivable, but DIN/multi-task phases benefit materially. |
| `evaluate.py` | **Never modified.** Import it, never edit it. | It is the pinned scoring definition. Also the sanity gate: `--model random` must give primary ≈ 0.475 ±0.001 or the harness is broken. |
| Experiment tracking | **Local JSON run-log + Obsidian narrative. No wandb/MLflow.** | The graded deliverable is a per-iteration log (hypothesis / diff / metrics / errors). A local file satisfies it directly; a hosted tracker adds an account+setup dependency mid-hackathon for no scoring benefit. |
| Notebooks | **No. Plain scripts.** | Everything is driven via Bash; scripts stay reproducible end-to-end and satisfy the "steps to reproduce" deliverable more directly. |
| Bonus datasets (1k / 27k) | **Skip unless KuaiRand-Pure is comfortably beaten and time remains.** | Pure determines 100% of the primary score; 1k/27k are 11.7M/322M interactions — a different engineering problem (out-of-core data handling) that would eat the whole budget for bonus points only. |

## Environment — ✅ INSTALLED

`/Users/hayden/TechJamHackathon/.venv` (Python 3.12.14), pinned in `requirements.txt`:

| package | version | why |
|---|---|---|
| torch | 2.13.0 | **MPS available** — Phases 1–4 (custom losses, attention, multi-task heads) |
| numpy | 2.5.2 | required by the kit |
| lightgbm | 4.7.0 | `lambdarank` = off-the-shelf listwise ranker → independent comparison for Iteration 1 (smoke-tested) |
| scikit-learn | 1.9.0 | metric cross-checks |
| pandas / scipy / matplotlib | 3.0.5 / 1.18.1 / 3.11.1 | our EDA + report charts only — never in the scored path |

`libomp` installed via Homebrew (LightGBM dependency on macOS).

**Claude Code skill — ✅ BUILT:** `TechJamHackathon/.claude/skills/rec-iteration/SKILL.md`, encoding the iteration protocol plus the hard rules (never modify `evaluate.py`; select on valid only; don't train on `log_random`; one change per iteration) and the list of already-ruled-out ideas.

Not decoration: **Autonomy is a scored criterion, measured by number of manual interventions.** Encoding the loop makes each iteration a reproducible invocation rather than an ad-hoc conversation, and guarantees every iteration emits the run-log fields the deliverable requires.

**Deliberately not installed:** RecBole (fights our I/O contract), wandb (account dependency, no scoring benefit), CWM's repo (pins `torch==1.6.0`), and any marketplace ML plugin — all 29 ML-relevant ones in the official marketplace are vendor/cloud platform integrations (SageMaker, DataRobot, BigQuery…), useless for a 45MB local dataset that trains in 15s.

---

## Phase order (current — post-EDA)

- **0. Setup + reproduce + EDA** — ✅ **DONE**, see [[Run-Log]]. Env built, all three baselines reproduced (FM valid 0.6015 / test 0.5953), EDA complete.
- **0b. PyTorch port** — ⏭️ *next*. Reimplement the FM in torch, verify parity with the numpy version within seed noise before building on it. Infra, not a scored hypothesis.
- **1. Listwise softmax loss** — the core insight in §1. Same FM, same 5 features, same k; only the objective changes. Group by `(user)` (§2 — settled, no ablation needed). *Compare against LightGBM `lambdarank` as an off-the-shelf listwise reference point.*
- **2. Multi-task** — auxiliary heads on **`is_click` (46.3%) and `play_time_ms`** only. The sparse signals (`is_like` 1.9%, `is_follow`/`is_comment`/`is_forward`/`is_hate` ≤0.26%) are dropped per [[EDA-Findings]] §4. Watch for seesaw; keep only if valid primary improves.
- **3. Sequence / DIN** — ⬇️ downgraded (§4). Target-attention over causal prior interactions, modest expectations at median history 31. SIM dropped.
- **4. Censored watch-time regression** — CWM idea reimplemented cleanly (not its `torch==1.6.0` repo). Stretch.
- **5. Architecture swap** (DeepFM / DCN) — last; capacity was shown not to be the bottleneck.
- **6. Seed averaging** on the winning config (§5), at the very end.

**Dropped from the earlier plan:** group-granularity ablation (§2 — resolved from data), duration engineering as an early phase (§3 — demoted), SIM (§4), and four sparse auxiliary tasks ([[EDA-Findings]] §4).

---

## ⚠️ One rules question I want your call on

`log_random_4_22_to_5_08_pure.csv` (1.18M rows, randomized exposure) is part of KuaiRand-Pure, so the "no external training data" rule does **not** forbid training on it. But its date range **overlaps the hidden test window (4/29–5/08)**. Training on it would give the model knowledge of user behaviour *during the test period* — legal by the letter of the rules, arguably against their spirit, and potentially bad-looking to judges.

The organizers framed it as an *unbiased validation* set, not a training set.

**My recommendation: use it for validation/diagnostics only** (checking whether we're overfitting to biased traffic), and do not train on it. If you want to consider training on it, that's worth asking the organizers explicitly rather than deciding unilaterally.

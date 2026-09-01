# Autonomous ML Research Agent for KuaiRand-Pure
**TechJam 2026, Track 2** 
· Required benchmark: KuaiRand-Pure 
· Metrics: GAUC / nDCG@5

## 1. How the solution addresses the problem statement
The challenge asks for an agent that runs the machine learning loop in Figure 1 by
itself: read the problem, inspect the data, build features, train and tune a
model, evaluate it, reflect on the result, then repeat. The agent writes the code
for every stage.

Three requirements from the brief are addressed explicitly:
**Reproduce the official baseline.** 
Every run starts by running `seed_solution.py`, which is the Factorization
Machine baseline, and measuring its score. The agent therefore starts from a
measured 0.6015, not from a number we assumed. Every later improvement is compared
against this measured baseline.

**Iterate autonomously across the full stack.** 
Each proposal must say which of seven pipeline stages it targets: features, model,
loss, training, tuning, ensembling or evaluation. We feed back how many times each
stage has been tried and what it returned, so the agent can see which parts of the
pipeline have paid off and which have not.

**Robust operation.** 
Every candidate runs in a separate subprocess, so it cannot damage the main loop.
When code fails, the agent sorts the failure into one of eight types. Each type
has its own fix instruction. The agent gets two repair attempts. If both fail, it
drops the idea and writes the failure to the log. The diagnostic code is also
wrapped in error handling. A bad result from generated code should cost one
iteration, not the whole run.

## 2. Results
Official baseline (hidden test): GAUC 0.6610 · nDCG@5 0.5282 · primary 0.5946.
Scored delta is the mean of the two per-metric improvements:
`Δ = ½[(GAUC_agent − 0.6610) + (nDCG_agent − 0.5282)]`

### Validation-best scores and absolute delta

The submission is the validation-best checkpoint at the point the run stopped,
scored once on the hidden test split.

| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| **agent, validation-best** | **0.6723** | **0.5379** | **0.6051** |
| agent, hidden test | 0.6657 | 0.5304 | 0.5981 |
| official baseline, hidden test | 0.6610 | 0.5282 | 0.5946 |
| **absolute delta over baseline** | **+0.0047** | **+0.0022** | n/a |
| **scored delta** `½(ΔGAUC + ΔnDCG@5)` | | | **+0.0035** |

The winning solution reproduced its recorded validation score to four decimal
places on re-run (0.6051 recorded, 0.6051 reproduced).

### The submission

**`submission_draft.csv`. Δ = +0.0035, from a single run with zero manual
interventions.** 170,588 rows, passes the official `submit.py --check` validator.


### Resource consumption to reach the converged result

For the proposed submission's run:

| | |
|---|---|
| iterations used | **12** of the 50 cap |
| agent wall-clock | **120.6 min** of the 6 h ceiling |
| LLM tokens (input + output) | **418,825** across 67 calls |
| GPU-hours | **0** (CPU only, no GPU used) |
| manual interventions | **0** |
| accepted changes | 2 of 12 (iterations 1 and 8) |

Of the 402,598 prompt tokens, **106,251 were cache hits**, roughly a quarter of
the prompt volume served from cache rather than recomputed.
Most of the wall-clock time is spent training models, not waiting for the LLM. The
agent's candidates are real gradient-boosting fits taking about 200 seconds each,
while the baseline trains in 8 seconds. The run stopped at iteration 12 because it
had converged, well before the 50-iteration cap or the 6-hour limit.

---
## 3. The measurements that shaped the design
### 3.1 The starting point is a tuned local optimum, the run's binding constraint

The agent starts with `D`, which holds five pre-encoded categorical fields. We
assumed a good agent could beat the FM baseline using those fields. It cannot, and
the problem is not the agent. We trained the strongest models we had on **exactly
those five fields**, using the same protocol. Validation primary scores:

| model | 5 fields (what `D` gives) | richer features |
|---|---|---|
| FM baseline | **0.6015** | n/a |
| CatBoost QueryRMSE | 0.5956 | 0.6039 |
| CatBoost YetiRank | 0.5944 | 0.6033 |
| LightGBM lambdarank | 0.5994 | 0.6014 |
| **best** | **0.5994** | **0.6039** |

**Nothing beats the FM on its own representation.** We then checked whether the FM
itself could be pushed further on that representation, and it is already flat in
both directions: widening the embedding gains nothing
(`k = 8/16/32 → 0.5895 / 0.5902 / 0.5887`), and neither does adding fields
(`5 → 13 fields → 0.5950 vs 0.5940`). It is a genuine local optimum. The FM has
already extracted almost all the signal these five fields contain, so no simple
modification to that representation improves it.

One run went twelve iterations and accepted nothing, finishing at exactly the
baseline score. The agent was right. No single change to its starting point
improves anything. We had spent several rounds rewriting its prompt, but the
prompt was never the problem. The starting point was.

**The fix comes from AIDE**, whose policy is to *"first explore a set of diverse
initial solutions and continuously improve the best one"*. We had built AIDE's
improve and debug steps, but not its draft step. So the agent always started from
the current solution and made small changes to it. We added a drafting phase: for
the first few iterations, the agent writes complete solutions from scratch, and it
can change the features, the model and the objective at the same time. The result
improved straight away:

| run configuration | accepted changes | best validation |
|---|---|---|
| improve-only, 12 iterations | 0 / 12 | 0.6015 (= baseline) |
| **with 3-draft phase** | **2** | **0.6051** |

In the drafting run the incumbent was established on **iteration 1** at 0.6040
and improved to 0.6051 by iteration 8.

### 3.2 Node selection in the solution tree
A loop that only keeps improvements will never find a method whose first step
scores worse than the current best. On this benchmark, the path to the strongest
single model looked like this:
```
CatBoost, default settings, 5 fields    0.5954   ← BELOW the 0.6015 baseline
+ 20 platform-statistic features      0.6044
+ all 110 statistic features          0.6054
+ QueryRMSE ranking loss              0.6075   ← best measured
```

Step 1 loses 0.006 against the baseline. A loop that only keeps improvements stops
there and never reaches step 4. One of our runs did exactly this: it proposed
CatBoost, scored 0.5954, and threw it away. Under its own rules that was correct.

So instead of keeping only the best solution, we keep all of them in a tree. Every
candidate we have measured stays available for the agent to build on. To choose
which one to work on next, we score each node:

```
score(n) = value(n) + c·√(ln(ΣN+1)/(visits(n)+1)) − λ·depth(n)

value(n) = max(0, 1 − deficit(n)/MAX_DEFICIT),   deficit = best − primary(n)
c = 2.5,  λ = 0.05,  MAX_DEFICIT = 0.010
```

Two parameters were fixed:
- **`MAX_DEFICIT = 0.010`** sets how far below the best score a node can be and
  still be worth expanding. We first tried scaling it by the noise level σ, but
  that put the CatBoost first step 7.6σ down and excluded it. That is the exact
  branch this design exists to protect, so we used a fixed value instead.
- **`c = 2.5`** controls how often the agent explores a weaker branch instead of
  the best one. At c = 1.2 that CatBoost branch was chosen 17% of the time, which
  is too rare for a run that stops after three iterations without improvement. At
  c = 2.5 it is chosen 33% of the time.

A node is marked `dead` when it falls too far behind. We recalculate these flags
every time the best score changes. This means a branch that was closed because of
one bad implementation can open again later, and a node that has fallen too far
behind stops using up iterations.

### 3.3 Mechanism choices

Every run below is the same agent on the same benchmark with zero manual interventions; the only differences are the mechanisms enabled. 

| run configuration | accepted | valid best | test primary | Δ |
|---|---|---|---|---|
| improve-only loop, 12 iterations | 0 / 12 | 0.6015 | n/a | n/a |
| + solution tree, converged early | 1 | 0.6033 | 0.5966 | +0.0020 |
| + tree inherited across runs † | 1 | 0.6037 | 0.5982 | +0.0036 |
| + undeclared iteration floor † | 2 | 0.6036 | 0.5986 | +0.0040 |
| **+ drafting phase (the submission)** | **2** | **0.6051** | **0.5981** | **+0.0035** |

Read down the `accepted` column rather than the deltas: the improve-only loop
accepts **nothing in twelve iterations**. We tested these design choices to see which ones actually helped.


### 3.4 The agent re-runs anything that looks like a small win, because small wins here are usually luck
**The problem.** If you train the same model twice with different random seeds,
the score moves by about 0.0008 on its own. So a change that scores +0.0015 might
be a real improvement, or it might just be random variation.

This happened to us. One change scored **+0.0020, then +0.0018, then +0.0016** on
three single runs. Three times in a row it looked like a real, consistent gain. We
then ran it across three seeds and took the average: **+0.0005**. There was nothing
there. On this benchmark, single-seed results were wrong by a factor of two to
four.

**Change is only kept if the gain is big enough to be safe, or if it survives being re-run:**

```
gain > 0.0016  (2 standard deviations)  →  keep it
gain 0 to 0.0016                        →  re-run with 2 more seeds, keep only if the average still holds
gain ≤ 0                                →  reject
```

The middle case matters most. With a single fixed cutoff, the *same* change would
be accepted or rejected depending on luck. That is worse than it sounds, because a
rejection is written into memory as an idea that failed. Every later iteration
would then avoid something that actually works.

**Three apparent gains were rejected by this rule.** The submitted result passed
it. +0.0035 is more than four standard deviations, and the change behind it was
repeated before we kept it.

### 3.5 Feature results: one whole category is useless, and one feature was cheating
**Half the obvious features cannot work here, and we can prove it without testing them.**
The metrics only compare a user's videos *against each other*. Some features have the same value on every row belonging to one user. Examples are that user's average watch time, how many videos they were shown, and how active they are. These features cannot change the ordering of anything. The same value is added to every score in that user's group, so the ranking stays the same.

That rules out a whole family of natural-sounding features at no cost. The aggregate features we built are therefore all per-video or user × video:

```
v_log_count · v_log_users · v_dur_mean · v_dur_std      about the video
uv_dur_gap  · uv_dur_ratio                              this user vs this video
```

There is no standalone user statistic. A user's average duration appears only inside `uv_dur_gap`, meaning "how far is this video's length from what this person usually watches". That value *does* differ from video to video. *"pure user-side first-order terms contribute exactly 0"*.

**One feature, however, leaked the target.**

We added each video's historical long-view rate as a feature. We used a
leave-one-out calculation, which excludes the current row's own label, because we
thought this would avoid using the answer. It did not. The feature dropped the
validation score by 0.0169, about twenty times the noise level, and it was the
worst result in the whole project.

The reason is that leave-one-out does not actually hide the current label. For a
video with n training rows and k positive labels, the feature for row i is:

```
e_i = (k − y_i + w·p) / (n − 1 + w)
```

Rearranging this formula gives back `y_i` exactly, which is the label we are trying
to predict. The model also knows which video each row belongs to, so it has
everything it needs to do that rearranging. A deep enough gradient-boosting model
finds this shortcut, predicts the training labels almost perfectly, and learns
nothing that works on new data.

We found it by testing each aggregate separately:
| what was added | validation primary |
|---|---|
| nothing (control) | 0.6040 |
| video counts | 0.6044 |
| duration mean and standard deviation | 0.6045 |
| user × video duration gap | 0.6045 |
| **the leave-one-out rate** | **0.5871** |

Three of the four are harmless; one is a leak. It is now off by default and only available if asked for explicitly.

---


## 4. Architecture

The agent has to answer two questions. First, **what experiment should it run
next?** Second, **how does it avoid being fooled by its own bugs?** Every part
described below serves one of these two questions.

### 4.1 Three concerns, and what connects them

```
DECIDE     what to try next, and on top of which prior solution
   │       ├─ the loop, the convergence rule, the iteration budget
   │       ├─ what the agent is shown each iteration
   │       ├─ the LLM client, with key and model rotation
   │       └─ node selection over the tree of past solutions
   │
   ↓  a proposal: hypothesis · reasoning · prediction · falsifier · stage · code
   │
EXECUTE    run it without letting it break anything
   │       ├─ an isolated subprocess with a hard wall-clock limit
   │       ├─ eight failure classes, each with a recovery instruction
   │       └─ scoring, segment breakdown, prediction health, sanity bounds
   │
   ↓  a measured result, or a classified failure
   │
REMEMBER   write down what happened, so the next iteration is better informed
           ├─ the solution tree: every candidate, kept or not
           ├─ memory: findings, with who measured them and how many seeds
           ├─ stage coverage: which parts of the pipeline have paid off
           └─ the run log: the graded per-iteration record
```

The loop closes because what the agent remembers becomes the input to its next
decision. Each iteration can see every previous attempt, the whole solution tree,
and what each pipeline stage has returned so far.

### 4.2 What the agent can reach

The agent can only do what its tools allow. We give it eight tools and it decides
which ones to use.

| tool | what it lets the agent do |
|---|---|
| `list_columns` · `inspect` | ask its own questions about the raw data |
| `build_features` | build a feature set of its own design |
| `search_papers` | find published methods on arXiv |
| `train_model` · `blend` · `list_predictions` | fit and combine models without writing the plumbing by hand |
| `recall` | look up what has already been measured |

These tools mattered more than anything else we changed. In earlier versions the
agent was given one fixed feature set and no way to build another. Every run
stopped at the baseline score. Section 3.1 shows that this was a limit of the
search space, not a failure of the agent's reasoning.

We also learned that listing a tool is not enough. Three tools were installed and
described in the agent's prompt, but were never called in ten runs. We then added
a short code example showing how to call each one. After that, the agent used them
straight away.

### 4.3 Three design decisions

**Give the agent facts, not conclusions.** `context.py` tells the agent three
things: how the metric behaves, the reference scores (random, popularity,
baseline, oracle), and which ideas have already been tested and failed. It does
not tell the agent what to think. We removed an earlier block that contained
sentences like *"SIM has no problem to solve here"*. That is a conclusion, and the
judging criteria reward what the agent works out by itself. The agent can still
get the same numbers by calling `inspect()`.

**The test data is not present during the run.** Three parts of the code enforce
this:

1. `prep.py` caches only training and validation data.
2. `tools.py` removes test dates when it reads the raw files. This happens in one
   function, so there is only one place to check.
3. `guards.py` stops the run if an array with the test row count appears in the
   cache.

Test data is loaded once, in `finalize.py`, after the run has finished. This is
not a rule that the agent is asked to follow. The code enforces it. It worked in
practice: one iteration wrote code that tried to read `D["Xte"]` and failed with a
`KeyError`, because that key does not exist.

**Failures are recorded, not hidden.** When candidate code fails, the agent sorts
the failure into one of eight types: `syntax`, `alignment`, `numerical`, `oom`,
`timeout`, `import`, `contract`, `runtime`. Each type has a fixed instruction for
how to fix it, and that instruction is sent back to the model. The agent gets two
repair attempts. If both fail, the hypothesis is dropped and the failure is
written to the run log.

### 4.4 Guards against self deception

An ML agent can get a high score by exploiting a bug in its own code instead of by
learning something real. The number then looks good but means nothing. We added
five checks to prevent this:

1. `evaluate.py` is checksummed with SHA-256 every iteration. If the file has
   changed, the run stops.
2. The agent selects on validation only. Test data cannot be reached during the
   loop.
3. Every score is compared against the reference scores while the run is still
   going. A score below random (0.4834) or below item popularity (0.5807) is
   reported to the agent as a broken implementation, with the likely causes
   listed. It is not reported as an idea that failed.
4. Score arrays are checked for the correct length, and for NaN or infinity,
   before any result is trusted.
5. Small gains must be repeated with more seeds before they are accepted.

Check 3 was added after one run scored 0.4962, 0.4802 and 0.5802 on three
iterations. All three models had failed to train at all. Without this check, the
agent would have recorded those three mechanisms as tested and refuted, when in
fact they had never run.

---

## 5. What the agent chose to try, and why

Everything in this section is the agent's own reasoning, quoted directly from the
run log of the submitted run. Before writing any code, every iteration must state
three things: a hypothesis, a predicted score, and a falsifier, which is the result
that would prove the hypothesis wrong. A proposal that is missing any of the three
is rejected.

### It commits to a number, then is held to it

Iteration 1, before running anything:

> **Hypothesis:** *"Expanding the feature space to include video metadata,
> statistics, and temporal signals allows a non-linear ranking model to capture
> complex interactions that the FM baseline's limited categorical features miss."*
> **Prediction:** *"approximately 0.6040, exceeding the official baseline."*
> **Falsifier:** *"The validation score is less than 0.6015."*

It measured **0.6040**. The prediction was exact.

Iteration 9 went the other way, which matters more:

> **Hypothesis:** *"Ensembling two CatBoost rankers with different
> initializations and slightly varied objectives will reduce variance…"*
> **Prediction:** *"approximately 0.6075"*
> **Falsifier:** *"If the blended primary score is not higher than the
> single-model incumbent (0.6051), then the ensemble of these specific models
> adds no value."*

It measured **0.6045**, so its own falsifier fired. Iteration 10 opened by naming
that: *"The previous ensemble failed due to potential overfitting or divergence
in the two objectives."* It did not repeat the attempt.

### It targeted five different parts of the pipeline, not just the model

The brief asks what the agent chose to work on across the whole pipeline, not just
the model. Here are all twelve iterations, grouped by the stage each one declared:

| iteration | stage | what it proposed | result |
|---|---|---|---|
| 1 | features + model | rich feature set with a non-linear ranker | **0.6040 kept** |
| 2, 9 | ensembling | blending several rankers | failed / 0.6045 |
| 3, 4, 7 | loss | QueryRMSE → QuerySoftMax → QueryCrossEntropy | 0.6040 / failed |
| 5 | model | greater tree depth, finer interactions | failed |
| 6 | model | LightGBM lambdarank instead of CatBoost | 0.5659 |
| 8 | tuning | learning rate and depth on the richer feature set | **0.6051 kept** |
| 11 | training | subsample training groups to match validation | 0.6047 |
| 12 | features | reweight toward item-intrinsic features | 0.6047 |

### Two hypotheses that came from its own analysis of the data

**Iteration 11, the group-size mismatch.** It queried the data itself, then
proposed:

> *"The drastic difference in group size distributions between training (median
> 31) and validation (median 4) suggests the model is overfitting on dense
> training groups. By subsampling training data to match the validation group
> size distribution…"*

This is a real property of the benchmark. The agent found it by calling the data
inspection tool itself, and it does not appear anywhere in its prompt. It then
predicted an improvement for small-group users specifically. It measured 0.6047,
so the method does not work here, but the observation and the reasoning were
correct.

**Iteration 12, cold start.** It read the per-segment breakdown in its own
diagnostics and proposed:

> *"The regression on cold users suggests the current model relies too heavily on
> historical aggregates which are absent or unreliable for cold-start users."*

Again this came from the agent reading its own results, and again it made a
testable prediction about a specific group of users.

### It reached for published methods

During the submitted run the agent searched arXiv without being asked to. Its
query was *"learning to rank sparse interactions user grouping CatBoost"*. That
query combines three things: the task it was solving, the data property it had
just measured, and the model family it was using. It then tested three CatBoost
ranking objectives across iterations 3, 4 and 7: QueryRMSE, QuerySoftMax and
QueryCrossEntropy. These are the listwise and groupwise objectives named in the
learning-to-rank literature, not small adjustments to the baseline's pointwise
loss.

---

## 6. Development tools

| tool | use |
|---|---|
| **Gemini** | generate every hypothesis, every line of candidate code, and every decision, with no human or other model in the loop. |
| **VS Code** | editing and review |
| **Obsidian** | design notes, EDA findings and the experiment log kept across sessions |
| **git / GitHub** | version control; per-member folders merged into one repository |

## 7. APIs used

| API | use |
|---|---|
| **Google Gemini API** (`google-genai` 2.20.0) | **The model that runs the agent**. Primary `gemini-3.6-flash`, with automatic rotation over `gemini-3.7-flash`, `gemini-3.5-flash`, `gemini-flash-latest`, `gemini-3.1-flash-lite-preview` on quota exhaustion or overload |
| **arXiv API** (`export.arxiv.org`, no key) | the `search_papers` tool, so published methods enter the loop by the agent's choice |

## 8. Libraries and frameworks

| library | version | role |
|---|---|---|
| PyTorch | 2.13.0 | FM / DeepFM / DCN and all agent-authored neural models (CPU) |
| CatBoost | 1.2.10 | gradient boosting with native high-cardinality categorical handling |
| LightGBM | 4.7.0 | gradient boosting; `lambdarank`, `rank_xendcg` |
| XGBoost | 3.4.1 | gradient boosting; `rank:ndcg`, `rank:pairwise` |
| NumPy | 2.5.2 | arrays, encoding, rank transforms |
| SciPy | 1.18.1 | sparse matrices for the collaborative-filtering experiments |
| scikit-learn | 1.9.0 | truncated SVD, utilities |
| pandas | 3.0.5 | data handling |
| Optuna | 4.9.0 | hyperparameter search on validation |
| RecBole | 1.2.0 | installed and offered to the agent; **not used by it** (see Section 10) |
| implicit | 0.7.3 | ALS / BPR matrix factorisation; available, unused |
| google-genai | 2.20.0 | Gemini client |

## 9. Datasets and assets

| asset | detail |
|---|---|
| **KuaiRand-Pure** (Kuaishou) | the only training data. 1.4M interactions, 27K users, 7.6K items |
| `log_standard_4_08_to_4_21_pure.csv` | train, dates 20220408–20220421, 1,141,112 rows |
| `log_standard_4_22_to_5_08_pure.csv` | validation 20220422–20220428 (124,909 rows) and test 20220429–20220508 (170,588 rows), split by the `date` column |
| `video_features_basic_pure.csv` | video metadata: author, tag, music, upload date, duration |
| `video_features_statistic_pure.csv` | 51 platform statistics per video; used as rates (÷ `show_cnt`) and log-counts |
| `user_features_pure.csv` | user attributes; measured as noise-level in feature importance and unused |
| `log_random_4_22_to_5_08_pure.csv` | **deliberately unused** |

---

## 10. Limitations
**1. The feature representation is the binding constraint, and we only partly moved it.**
Section 3.1 established that no model beats the FM on the five fields the agent
starts from. Drafting let it build better representations, but the search over
feature space stayed shallow. LightGBM importance over 144 features placed
`user_id` at 274,093 against the best external feature at 18,538, a 15× gap. The
four identity columns carry nearly all the signal, and the aggregate features we
added measured +0.0004 to +0.0005 alone, inside the 0.0008 noise floor.

**2. Implementation quality, not idea quality, sets the ceiling.**
The same mechanisms, measured hand-written by us and written by the agent:

| mechanism | human | agent | gap |
|---|---|---|---|
| CatBoost | 0.6068 | 0.5961 | −0.0107 |
| dense video statistics | +0.0050 | −0.0096 | wrong model pairing |
| ensembling | +0.0055 | +0.0013 | −0.0042 |

To write a correct CatBoost ranker inside a 20-minute budget, the agent has to get
four things right on the first try, with no debugger: the group ordering, the
categorical column indices, the ranking objective and the iteration count. That is
hard. We fixed this with `models.py`, which sets up the libraries correctly and
leaves every research decision to the agent. The same algorithm scored 0.6040
through the tool and 0.5961 when the agent wrote it by hand.

**3. The agent does not use a tool just because we list it.**
This was the clearest finding of the project. Three capabilities were installed and
described in the agent's prompt, but it never called any of them. They stayed
unused until we added a short *worked code example* showing how to call them:

| capability | listed | used after listing | used after example |
|---|---|---|---|
| Optuna | ✓ | 0 uses in 10 runs | yes |
| `train_model` | ✓ | 0 uses | yes, reached 0.6046, the best single-model score any run produced |
| RecBole | ✓ | 0 uses | **no example written yet, still unused** |

RecBole is the biggest open item. It has roughly 90 ready-made model
implementations installed, and the agent has never used any of them. Most of its
models are from families we already measured as flat on this benchmark (DeepFM,
DCN and AutoInt all scored 0.597 to 0.600), so we expect it to give us correct
implementations rather than new ideas. We have not tested that.

**4. Our own tool API caused more failures than the agent's reasoning did.**
Once drafting let the agent get far enough to actually use the tools, one run of 7
attempts produced 4 failures. Three of them were the agent guessing which module
we had put a function in:

```
No module named 'features'                              (3 drafts in one run)
cannot import name 'load_features' from 'models'
cannot import name 'train_model' from 'models'
```

We then checked every tool name we advertise against every function the agent can
actually import. Only `train_model` and `recall` did not match. We added aliases
for both, and `features.py` now re-exports both halves, so any reasonable import
path works.

Two more API faults each cost a full iteration before we found them. First, a draft
read `r["prediction_id"]` from an error dictionary and passed that dictionary to
`load_scores`. The error appeared as *"File name too long"*, two layers away from
the real cause. Second, we had sized an iteration-count cap from a guessed cost
per iteration, and it quietly cut a requested 1000 iterations down to 409.

**5. The convergence rule is not the binding constraint.**
We tested this directly in an earlier development run, whose archive is not part
of this submission. Removing the rule and running 5× longer produced **2 accepted
changes in 24 iterations, both by iteration 5**, followed by 19 rejections and one
failure, using 70 minutes and 1.27M tokens for +0.0011. The agent runs out of
productive ideas after about five iterations, so the rule was correctly detecting
that the search had stopped producing results. The submitted run matches this
pattern: its two accepted changes came on iterations 1 and 8 of 12.

---

## 11. Citations

[1] C. Gao, S. Li, W. Lei, J. Jia, P. Chen, J. Zhang, et al., "KuaiRand: An unbiased sequential recommendation dataset with randomly exposed videos," in Proc. 31st ACM Int. Conf. Inf. Knowl. Manage., 2022, pp. 3953–3957.

[2] WecoAI, "AIDE: AI-driven exploration in the space of code," arXiv preprint arXiv:2502.13138, 2025.

[3] S. Rendle, "Factorization machines," in Proc. IEEE Int. Conf. Data Mining, 2010, pp. 995–1000.

[4] L. Kocsis and C. Szepesvári, "Bandit based Monte-Carlo planning," in Proc. 17th Eur. Conf. Machine Learning (ECML), 2006, pp. 282–293.

[5] Y. Yan, L. Li, and R. Choudhary, "Hierarchical group-wise ranking framework for recommendation models," arXiv preprint arXiv:2506.12756, 2025.

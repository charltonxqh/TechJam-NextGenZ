# Autonomous ML Research Agent — KuaiRand-Pure

**TechJam 2026, Track 2** · Required benchmark: KuaiRand-Pure · Metrics: GAUC / nDCG@5

---

## 1. How the solution addresses the problem statement

The challenge asks for an agent that autonomously runs the MLE loop of Figure 1 —
read the problem, inspect data, engineer features, train and tune, evaluate,
reflect, iterate — writing the code for each stage itself.

Our agent does this as a closed loop with no human in it. Each stage of the
canonical MLE loop maps onto a specific mechanism, and each is exercised in every
run — the right-hand column gives the observed counts from one representative
12-iteration run:

| canonical stage | mechanism | evidence in one run |
|---|---|---|
| **Understand dataset** | `inspect()` and `list_columns()` tools — the agent asks its own EDA questions against the raw CSVs | **12 calls**: cardinality, group sizes, label rates, cold-start overlap, available columns |
| **Reproduce baseline** | `establish_baseline()` runs `seed_solution.py`, the official FM, and measures it | once per run: `baseline valid primary = 0.6015 (official: 0.6015)` |
| **Read experiment history** | the solution tree, the per-stage yield table, and prior attempts are all in the prompt; `recall(topic)` queries cross-run memory | tree + attempts in every prompt; `recall` unused this run because a fresh run starts with empty memory by design |
| **Generate research hypothesis** | required `hypothesis` field, rejected if absent | every iteration |
| **Plan experiment** | required `reasoning`, `prediction` (a number) and `falsifier` fields, plus a declared pipeline `stage` | every iteration |
| **Modify code** | the LLM writes a complete `run(D, seed)` module; `build_features` / `train_model` are callable | 10 `build_features`, 4 `train_model`, 2 `blend` |
| **Run training** | `runner.py` — isolated subprocess, hard timeout, 8-class failure taxonomy, 2 repair attempts | every iteration |
| **Evaluate GAUC + nDCG@5** | `diagnose.py` imports the organisers' unmodified `evaluate.py` | every iteration |
| **Analyze result** | training dynamics, per-segment breakdown, prediction health, sanity bounds against random / popularity / oracle | every iteration |
| **Keep / rollback** | kept only if Δ > 2σ, or if a marginal Δ survives 3-seed replication | every iteration |
| **Record experiment** | tree node, memory entry with provenance, stage yield, JSONL run log | every iteration |
| **Generate next hypothesis** | loop, with node selection deciding which prior solution to build on | ↺ |

Two additions to the canonical loop, both forced by measurement rather than
preference:

**A drafting phase before the loop proper.** The first *n* iterations write
independent solutions instead of editing the incumbent. §3.1 gives the measured
reason: the official FM is a tuned local optimum for the five fields it is given,
and escaping it requires changing features and model together.

**Node selection at the top of each iteration.** The loop does not always build
on the best solution; a solution tree keeps weaker candidates expandable, because
the first step of a winning mechanism can score below the incumbent.

Three requirements from the brief are addressed explicitly:

**Reproduce the official baseline.** Every run begins by executing
`seed_solution.py` — a faithful port of the organisers' Factorization Machine —
and measuring it. The agent's incumbent is therefore a *measured* 0.6015, not an
assumed one. Improvement is always relative to a number we reproduced.

**Iterate autonomously across the full stack.** Proposals must declare which of
seven pipeline stages they target (features, model, loss, training, tuning,
ensembling, evaluation). Coverage and per-stage yield are fed back, so the loop
can see where its own effort has and has not paid off.

**Robust operation.** Every candidate runs in an isolated subprocess. Failures
are classified into eight types, each with a recovery instruction, with up to two
repair attempts before the hypothesis is abandoned and logged. Diagnostics are
themselves wrapped, because a malformed result from generated code must degrade
one iteration, never terminate a run.

---

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
| **absolute delta over baseline** | **+0.0047** | **+0.0022** | — |
| **scored delta** `½(ΔGAUC + ΔnDCG@5)` | | | **+0.0035** |

The winning solution reproduced its recorded validation score to four decimal
places on re-run (0.6051 recorded, 0.6051 reproduced).

### The submission

**`submission_draft.csv` — Δ = +0.0035, from a single run with zero manual
interventions.** 170,588 rows, passes the official `submit.py --check` validator.

It is compliant under the organisers' *default* ε = 0.002 / N = 3 criterion, not
only under the one we declared: the default rule would have stopped the run at
iteration 8 with exactly this solution, and the run log records that point as
`default_rule_convergence_point`.

### Resource consumption to reach the converged result

For the proposed submission's run:

| | |
|---|---|
| iterations used | **12** of the 50 cap |
| agent wall-clock | **120.6 min** of the 6 h ceiling |
| LLM tokens (input + output) | **418,825** across 67 calls |
| GPU-hours | **0** — CPU only, no GPU used |
| manual interventions | **0** |
| accepted changes | 2 of 12 (iterations 1 and 8) |

Of the 402,598 prompt tokens, **106,251 were cache hits** — roughly a quarter of
the prompt volume served from cache rather than recomputed.

Wall-clock is dominated by training, not by the model: the agent's candidates are
real gradient-boosting fits at ~200 s each, against a baseline that trains in 8
seconds. Convergence terminated the run at iteration 12, nowhere near either the
50-iteration cap or the 6-hour ceiling.

---

## 3. The measurements that shaped the design

Ordered by what each one changed. The first three produced the result; the
next two protect it from being wrong; the last two set the scale for reading it.

### 3.1 The starting point is a tuned local optimum — the run's binding constraint

This is the most important measurement in the project, and it reverses an
assumption we held for most of it.

The agent is handed `D`: five pre-encoded categorical fields. We assumed a
capable agent should be able to improve on the FM from there. It cannot, and the
reason is not the agent. Training the strongest available models on **exactly
that representation**, same protocol, validation primary:

| model | 5 fields (what `D` gives) | richer features |
|---|---|---|
| FM baseline | **0.6015** | — |
| CatBoost QueryRMSE | 0.5956 | 0.6039 |
| CatBoost YetiRank | 0.5944 | 0.6033 |
| LightGBM lambdarank | 0.5994 | 0.6014 |
| **best** | **0.5994** | **0.6039** |

**Nothing beats the FM on its own representation.** The organisers had already
flattened it — they report `k = 8/16/32 → 0.5895/0.5902/0.5887` and
`5 → 13 fields → 0.5950 vs 0.5940`. It is a genuine local optimum, and the agent
starts inside it.

Escaping requires the feature set **and** the model to change together
(0.5994 → 0.6039); either alone is worse than doing nothing. Formally, writing
`s(F, M)` for the score under feature set `F` and model `M`, we measured

```
s(F₅, FM)   = 0.6015
s(F₅, CB)   = 0.5956     (change the model alone   → −0.0059)
s(F_rich, FM) ≈ 0.6015   (features alone do little for an FM: organisers' 13-field result)
s(F_rich, CB) = 0.6039   (change both              → +0.0024)
```

The maximum over any single coordinate change is **below** the incumbent, while
the joint change is above it. A hill-climbing loop restricted to "ONE conceptual
change per iteration" therefore cannot reach the optimum from this start — not
through lack of intelligence, but because no single-coordinate ascent direction
exists.

**This explains the central symptom.** One run produced twelve iterations with
zero accepted changes and converged at exactly the baseline. That was the agent
correctly reporting that no single change from its starting point improves
anything. Several rounds of prompt engineering on our part were treating a
symptom of a search-space problem.

**The fix follows AIDE** (arXiv:2502.13138 §3.2), whose policy is to *"first
explore a set of diverse initial solutions and continuously improve the best
one"*. We had implemented AIDE's *improve* and *debug* node types and never its
*draft* type. Adding a drafting phase — the first *n* iterations write
independent solutions, free to change features, model and objective at once —
moved the result immediately:

| run configuration | accepted changes | best validation |
|---|---|---|
| improve-only, 12 iterations | 0 / 12 | 0.6015 (= baseline) |
| **with 3-draft phase** | **2** | **0.6051** |

In the drafting run the incumbent was established on **iteration 1** at 0.6040
and improved to 0.6051 by iteration 8, where the organisers' *default*
convergence rule would also have stopped — so the result is compliant under the
default criterion, not only under our declared one.

### 3.2 Node selection in the solution tree

Greedy hill-climbing cannot reach a mechanism whose first step scores below the
incumbent. Measured on this benchmark, the path to the strongest single model was

```
CatBoost, default settings, 5 fields    0.5954   ← BELOW the 0.6015 baseline
  + 20 platform-statistic features      0.6044
  + all 110 statistic features          0.6054
  + QueryRMSE ranking loss              0.6075   ← best measured
```

Step 1 loses 0.006. A greedy loop reverts there and never reaches step 4 — and
one of our runs did exactly that, proposing CatBoost, scoring 0.5954, and
correctly discarding it under its own rules.

The tree therefore keeps every measured candidate as an expandable node and
selects with a UCB-style score:

```
score(n) = value(n) + c·√(ln(ΣN+1)/(visits(n)+1)) − λ·depth(n)

value(n) = max(0, 1 − deficit(n)/MAX_DEFICIT),   deficit = best − primary(n)
c = 2.5,  λ = 0.05,  MAX_DEFICIT = 0.010
```

Two parameters were fixed by simulation against the real case rather than by
taste:

- **`MAX_DEFICIT = 0.010`.** Scaling by σ instead would place the CatBoost first
  step 7.6σ down and exclude it — killing the exact branch the mechanism exists
  to preserve.
- **`c = 2.5`.** At c = 1.2 that branch received 17% of expansions; over a run
  ending after three non-improving iterations that is too rare. At c = 2.5 it
  receives 33%.

`dead` flags are recomputed whenever the incumbent moves, so a branch closed by
one poor implementation can reopen, and a node that drifts past the cap stops
consuming iterations.

### 3.3 What each mechanism contributed, measured

Every run below is the same agent on the same benchmark with zero manual
interventions; the only differences are the mechanisms enabled. These are
*experiments*, not alternative submissions — they are what the design decisions
in §3.6 and §4 were measured against.

| run configuration | accepted | valid best | test primary | Δ |
|---|---|---|---|---|
| improve-only loop, 12 iterations | 0 / 12 | 0.6015 | — | — |
| + solution tree, converged early | 1 | 0.6033 | 0.5966 | +0.0020 |
| + tree inherited across runs † | 1 | 0.6037 | 0.5982 | +0.0036 |
| + undeclared iteration floor † | 2 | 0.6036 | 0.5986 | +0.0040 |
| **+ drafting phase (the submission)** | **2** | **0.6051** | **0.5981** | **+0.0035** |

† Neither is submittable: inheriting a tree across runs makes "the converged
result" ambiguous when convergence is judged per run, and rule 2.9.1(a) requires
a minimum-iteration floor to be declared before the run. Both are listed because
they are part of the measurement record, not because they are candidates.

Read down the `accepted` column rather than the deltas: the improve-only loop
accepts **nothing in twelve iterations**, and every configuration that accepts
something does so because of a mechanism §3.6 explains. The deltas between the
last four rows are within about 2σ of each other and should not be over-read.

### 3.4 The agent re-runs anything that looks like a small win, because small wins here are usually luck

**The problem.** Train the identical model twice with different random seeds and
the score moves by about 0.0008. So a change that scores +0.0015 might be a real
improvement, or might be the same model twice with different dice.

**We ran into this directly.** One change — an intra-user pairwise loss — scored
**+0.0020, then +0.0018, then +0.0016** on single runs. It looked like a
consistent, real gain three times over. Run it across three seeds and average:
**+0.0005**. There was nothing there. Single-seed results on this benchmark were
wrong by two to four times.

**What the agent does about it.** A change is only kept if the gain is big enough
to be safe, or if it survives being re-run:

```
gain > 0.0016  (2 standard deviations)  →  keep it
gain 0 to 0.0016                        →  re-run with 2 more seeds,
                                           keep only if the average still holds
gain ≤ 0                                →  reject
```

The middle case is the one that matters. A flat cutoff would accept or reject the
*same* change depending on luck, and a rejection gets written into memory as a
refuted idea — steering every later iteration away from something that actually
works.

**Three apparent gains were rejected this way.** The submitted result survived it:
+0.0035 is over four standard deviations, and the change behind it was replicated
before being kept.

### 3.5 Getting the stopping rule exactly right, because it decides which solution gets submitted

**Why it matters.** The organisers score *the converged result* — whichever
solution the agent holds when the run stops. Stop the run at the wrong moment and
you submit the wrong model, however good your agent is.

**The rule.** A run is converged when the best validation score has not improved
by more than 0.002 **over the last three scored iterations, added together**.

**The mistake that's easy to make.** Checking each iteration on its own is
stricter than the rule allows. Three iterations gaining 0.001 each total 0.003,
which is *not* converged — but tested one at a time, each looks like a failure to
improve. An early version of our loop did exactly that and ended a run nine
iterations too soon.

**A second one we found later.** Iterations that crash produce no score at all,
and the organisers' clarification says these must not advance the window. Ours
counted them as non-improving iterations. **Six of our runs had a crash inside
their final three iterations, so all six stopped earlier than the rules
require.** Fixed, and the run log now records both the point where the default
rule would have stopped and where our declared rule did.

### 3.6 Two feature results: one whole category is useless, and one feature was cheating

**Half the obvious features cannot work here, and we can prove it without
testing them.**

The metrics only compare a user's videos *against each other*. So any number that
is the same for every row belonging to one user — that user's average watch time,
how many videos they were shown, how active they are — cannot change the ordering
of anything. It is added to every score in the group equally, so the ranking is
untouched.

That rules out a whole family of natural-sounding features at no cost. The
aggregate features we built are therefore all per-video or user × video:

```
v_log_count · v_log_users · v_dur_mean · v_dur_std      about the video
uv_dur_gap  · uv_dur_ratio                              this user vs this video
```

There is no standalone user statistic. A user's average duration appears only
inside `uv_dur_gap` — "how far is this video's length from what this person
usually watches" — which *does* differ from video to video. The organisers say
the same thing from the other direction: *"pure user-side first-order terms
contribute exactly 0"*.

**One feature was reading the answer key.**

We added a feature giving each video's historical long_view rate, computed
leaving the current row out so it couldn't see its own label. It cost
**−0.0169** — twenty times the noise level, by far the worst result in the
project.

The reason is that "leave one out" can be undone with algebra. For a video with
`n` training rows and `k` positives, the feature for row `i` is

```
e_i = (k − y_i + w·p) / (n − 1 + w)
```

Rearranging gives `y_i` back exactly — and the model also knows which video it is
looking at, so it has everything it needs to do that rearranging. A deep enough
gradient-boosting model learns the trick, predicts the training labels almost
perfectly, and has learned nothing that transfers.

We found it by testing each aggregate separately:

| what was added | validation primary |
|---|---|
| nothing (control) | 0.6040 |
| video counts | 0.6044 |
| duration mean and standard deviation | 0.6045 |
| user × video duration gap | 0.6045 |
| **the leave-one-out rate** | **0.5871** |

Three of the four are harmless; one is a leak. It is now off by default and only
available if asked for explicitly.

---

### 3.7 A perfect model would only score 0.8645, so +0.0035 is bigger than it looks

**Why a perfect score is impossible.** Of the 23,875 users in the test set,
**27.1% never watched anything** — every video they were shown is a negative. No
ranking of those videos is better than any other, and nDCG@5 gives those users a
0 while still counting them in the average. Another 9.2% watched everything and
always score 1.

So even a model that knew every answer in advance would get:

```
nDCG@5  = 0 × 27.1%  +  1 × 9.2%  +  1 × 63.7%  =  0.729
GAUC    = 1.000   (it skips the users with no useful ranking)
primary = 0.865
```

**What that means for reading the score.** Random guessing scores 0.475. So the
whole usable range is about **0.39 wide, not 1.0**, and the official baseline at
0.5946 has already taken **30.7%** of it. Everything left is being fought over in
thousandths — which is also why a per-seed wobble of 0.0008 is large enough to
matter, and why §3.4 exists.


## 4. Architecture

The agent is built around one question: **how does a program decide what
experiment to run next, and how does it avoid fooling itself about the answer?**
Everything below serves one of those two halves.

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

The loop closes because the REMEMBER layer is an input to the DECIDE layer. An
iteration sees every previous attempt, the whole tree, and what each pipeline
stage has returned so far.

### 4.2 What the agent can reach

An agent is bounded by what it can do, not by what it knows. Ours is given eight
tools and chooses freely among them:

| tool | what it lets the agent do |
|---|---|
| `list_columns` · `inspect` | ask its own questions of the raw data |
| `build_features` | construct a feature representation of its own design |
| `search_papers` | find published methods on arXiv |
| `train_model` · `blend` · `list_predictions` | fit and combine models without hand-writing the plumbing |
| `recall` | query what has already been measured |

**This turned out to be the single largest lever on the result.** Earlier versions
gave the agent one fixed representation and no way to build another; it converged
at the baseline in every run. §3.1 explains why that was structural rather than a
failure of reasoning.

One finding worth stating on its own: **a capability that is listed but not
demonstrated is not reachable.** Three of these tools sat installed and documented
with zero uses across ten runs. Each was adopted immediately once the contract
showed a worked example of it being called. Listing a tool is not the same as
making it available.

### 4.3 Three design decisions

**Facts, not answers.** `context.py` gives the agent the metric's invariance
property, the reference score ladder, and every idea already ruled out — but not
conclusions. We removed a `FACTS` block that had contained lines like *"SIM has
no problem to solve here"*; that is an answer, and Innovation is graded on what
the **agent** identified. The same numbers are now obtainable through
`inspect()`, so the agent derives them by asking.

**The test split is physically absent.** `prep.py` caches train and validation
only; `tools.py` filters test dates at read time in a single function; and
`guards.py` fails the run if a test-shaped array appears in the cache. Test is
loaded exactly once, in `finalize.py`, after convergence. This is enforcement,
not a rule the agent is asked to respect — and it fired in practice: an
iteration whose generated code tried to read `D["Xte"]` failed with a KeyError
because the key does not exist.

**Failures are data.** Eight classes — `syntax`, `alignment`, `numerical`, `oom`,
`timeout`, `import`, `contract`, `runtime` — each carry a recovery instruction
fed back to the model, with two repair attempts before the hypothesis is
abandoned and recorded.

### 4.4 Guards against self-deception

The characteristic failure of an autonomous ML agent is optimising against its
own bug and reporting a meaningless number. Countermeasures:

- `evaluate.py` is SHA-256 checksummed every iteration; the run halts if it changes
- selection is on validation only; test is unreachable during the loop
- scores are bounds-checked against the reference ladder **inside the loop** —
  below random (0.4834) or below item popularity (0.5807) is reported as a
  *broken implementation*, with the likely causes named, rather than as a refuted
  idea
- score length and finiteness are validated before any result is believed
- marginal gains must survive multi-seed replication

The popularity bound was added after observing three iterations in one run score
0.4962, 0.4802 and 0.5802 — all models that had not trained — and be reported to
the agent as ordinary negative results. It would have recorded the *mechanisms*
as refuted when they had never been tested.

---

## 5. What the agent chose to try, and why

This section is the agent's own reasoning, taken verbatim from the run log of the
submitted run. Every iteration must state a hypothesis, a numeric prediction and
a falsifier before any code is written; a proposal missing any of them is
rejected.

### It commits to a number, then is held to it

Iteration 1, before running anything:

> **Hypothesis** — *"Expanding the feature space to include video metadata,
> statistics, and temporal signals allows a non-linear ranking model to capture
> complex interactions that the FM baseline's limited categorical features miss."*
> **Prediction** — *"approximately 0.6040, exceeding the official baseline."*
> **Falsifier** — *"The validation score is less than 0.6015."*

It measured **0.6040**. The prediction was exact.

Iteration 9 went the other way, which matters more:

> **Hypothesis** — *"Ensembling two CatBoost rankers with different
> initializations and slightly varied objectives will reduce variance…"*
> **Prediction** — *"approximately 0.6075"*
> **Falsifier** — *"If the blended primary score is not higher than the
> single-model incumbent (0.6051), then the ensemble of these specific models
> adds no value."*

It measured **0.6045** — its own falsifier fired. Iteration 10 opened by naming
that: *"The previous ensemble failed due to potential overfitting or divergence
in the two objectives."* It did not repeat the attempt.

### It targeted five different parts of the pipeline, not just the model

The spec asks what the agent chose to target *across the full algorithmic stack*.
Across twelve iterations, by declared stage:

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

**Iteration 11 — the group-size mismatch.** It queried the data itself, then
proposed:

> *"The drastic difference in group size distributions between training (median
> 31) and validation (median 4) suggests the model is overfitting on dense
> training groups. By subsampling training data to match the validation group
> size distribution…"*

That is a real property of this benchmark, found by its own EDA call and not
present anywhere in its prompt. It predicted a lift for small-group users
specifically. It measured 0.6047 — the mechanism does not work here, but the
observation and the reasoning were sound.

**Iteration 12 — cold-start.** It read the per-segment breakdown in its own
diagnostics and proposed:

> *"The regression on cold users suggests the current model relies too heavily on
> historical aggregates which are absent or unreliable for cold-start users."*

Again its own reading of its own results, and again a testable prediction about a
named user segment.

### It reached for published methods

During the submitted run the agent searched arXiv unprompted, for *"learning to
rank sparse interactions user grouping CatBoost"* — a query that combines the
task, the data property it had just measured, and the model family it was working
in. The three CatBoost ranking objectives it went on to test across iterations
3, 4 and 7 — QueryRMSE, QuerySoftMax, QueryCrossEntropy — are the listwise and
groupwise objectives the learning-to-rank literature names, rather than tweaks to
the baseline's pointwise loss.

---

## 6. Development tools

**Two different models, doing two different jobs.** Claude Code was the
development environment we used to *build* the agent. **Gemini is the model that
*runs* it** — every hypothesis, every line of candidate code, and every decision
in the submitted run came from Gemini, with no human or other model in the loop.
Claude Code was not running during any scored run.

| tool | use |
|---|---|
| **Claude Code** (Opus 5) | development environment: authored the agent's code, analysed run logs, wrote this report. Not part of the agent, and not running during a scored run |
| **VS Code** | editing and review |
| **Obsidian** | design notes, EDA findings and experiment log across sessions (`docs/` mirrors the vault) |
| **git / GitHub** | version control; per-member folders merged into one repository |
| **macOS terminal, Python 3.12 venv** | execution; CPU only, no GPU |

## 7. APIs used

| API | use |
|---|---|
| **Google Gemini API** (`google-genai` 2.20.0) | **the model that runs the agent** — every hypothesis and every line of candidate code in the submitted run. Primary `gemini-3.6-flash`, with automatic rotation over `gemini-3.7-flash`, `gemini-3.5-flash`, `gemini-flash-latest`, `gemini-3.1-flash-lite-preview` on quota exhaustion or overload |
| **arXiv API** (`export.arxiv.org`, no key) | the `search_papers` tool, so published methods enter the loop by the agent's choice |

No other external service is called. Runs are fully offline apart from these two.

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
| RecBole | 1.2.0 | installed and offered to the agent; **not used by it** (see §10) |
| implicit | 0.7.3 | ALS / BPR matrix factorisation; available, unused |
| google-genai | 2.20.0 | Gemini client |

The organisers' `evaluate.py` is imported unmodified and checksummed; no metric
code of ours participates in scoring.

## 9. Datasets and assets

| asset | detail |
|---|---|
| **KuaiRand-Pure** (Kuaishou) | the only training data. 1.4M interactions, 27K users, 7.6K items |
| `log_standard_4_08_to_4_21_pure.csv` | train, dates 20220408–20220421, 1,141,112 rows |
| `log_standard_4_22_to_5_08_pure.csv` | validation 20220422–20220428 (124,909 rows) and test 20220429–20220508 (170,588 rows), split by the `date` column |
| `video_features_basic_pure.csv` | video metadata: author, tag, music, upload date, duration |
| `video_features_statistic_pure.csv` | 51 platform statistics per video; used as rates (÷ `show_cnt`) and log-counts |
| `user_features_pure.csv` | user attributes; measured as noise-level in feature importance and unused |
| `log_random_4_22_to_5_08_pure.csv` | **deliberately unused.** See below |

No external data, no pretrained weights, no augmentation. The single hard rule —
no external training data — is satisfied by using KuaiRand-Pure alone.

**On `log_random`.** It contains 1,186,059 randomly-exposed impressions and is
part of KuaiRand-Pure, so training on it is legal by the letter of the rules. We
did not use it. Its dates span 20220422–20220508, covering the validation *and
the entire test* window, and the starter kit describes it as an additional
*unbiased validation* set. Training on data from the scored period is not
something we would defend, so we forfeited whatever advantage it offered. We
measured its properties for the record: 8.5% long_view rate under random
exposure against 31.3% under algorithmic exposure over the same dates, and the
top 10% of videos taking 11.9% of random impressions against 59.5% of
algorithmic ones.

---

## 10. Limitations and what we would do next

**The score gain is small.** +0.0020 compliant, against 0.389 of attainable range
above random. The mechanisms the agent found are real and replicated, but modest.

**The convergence rule is not the binding constraint.** We tested this directly.
Removing it and running 5× longer produced **2 accepted changes in 24 iterations,
both by iteration 5**, followed by 19 rejections and one failure — 70 minutes and
1.27M tokens for +0.0011. The agent exhausts its productive ideas in roughly five
iterations; the rule was correctly detecting that the search had stopped paying.

**The dominant failure mode became our own API, not the agent's reasoning.** Once
drafting let the agent get far enough to exercise the tools, a run of 7 attempts
produced 4 failures, 3 of which were it guessing which module we had put a
function in:

```
No module named 'features'                              (3 drafts in one run)
cannot import name 'load_features' from 'models'
cannot import name 'train_model' from 'models'
```

An audit of every advertised tool name against every importable function found
`train_model` and `recall` were the only two that did not line up. Both are now
aliased, and `features.py` re-exports both halves, so every reasonable import
path resolves. Two further API faults cost whole iterations before being found: a
draft that read `r["prediction_id"]` from an error dict and passed the dict to
`load_scores` (surfacing as *"File name too long"* two layers from the cause),
and an iteration-count cap sized from a guessed per-iteration cost that silently
trimmed a requested 1000 iterations to 409.

The lesson generalises: **a capability that is listed but not demonstrated is not
reachable.** Three capabilities — Optuna, `train_model`, RecBole — sat installed
and documented with zero uses until a worked code example appeared in the
contract. Optuna and `train_model` were adopted immediately once one did.
RecBole still has no example and still has zero uses.

**Implementation quality, not idea quality, is the ceiling.** The same mechanisms
measured by a human and by the agent:

| mechanism | human | agent | gap |
|---|---|---|---|
| CatBoost | 0.6068 | 0.5961 | −0.0107 |
| dense video statistics | +0.0050 | −0.0096 | wrong model pairing |
| ensembling | +0.0055 | +0.0013 | −0.0042 |

Writing a correct CatBoost ranker inside one 20-minute budget means getting group
ordering, categorical indices, the ranking objective and the iteration count
right on the first attempt with no debugger. We addressed this with `models.py`,
which supplies the libraries correctly configured and leaves every research
decision to the agent — measured at 0.6040 through the tool versus 0.5961
hand-written, the same algorithm either way.

**Availability is not reach — the clearest finding of the project.** Three
capabilities were installed, documented in the agent's contract, and went
completely unused until a *worked code example* appeared:

| capability | listed | used after listing | used after example |
|---|---|---|---|
| Optuna | ✓ | 0 uses in 10 runs | yes |
| `train_model` | ✓ | 0 uses | yes — reached 0.6046, the best single-model score any run produced |
| RecBole | ✓ | 0 uses | **no example written yet — still unused** |

RecBole remains the clearest open item: ~90 mature implementations sit installed
and unreached. Its models are largely the families we measured as flat
(DeepFM, DCN, AutoInt all in 0.597–0.600), so we expect correctness rather than
novelty from it — but that expectation is untested.

**Feature engineering is closer to exhausted than we assumed.** LightGBM
importance over 144 features placed `user_id` at 274,093 against the best
external feature at 18,538 — a 15× gap. The four identity columns carry nearly
all the signal. The new aggregate features (video counts, duration mean/std,
user×video gap) each measured +0.0004 to +0.0005 alone — inside σ — and −0.0002
combined.

**Behaviour-history features cannot work on this dataset**, which we established
by measurement rather than assumption: only 3.4% of validation rows involve a
previously-seen author, 0.7% a previously long-viewed author, and 1.6% a repeat
video. The sampling that produced this benchmark destroyed the repeat-exposure
structure such features depend on.

**What we would do with more time**, in priority order: a worked example for
RecBole, to test whether mature implementations close the remaining gap;
out-of-fold target encoding to recover the signal the LOO version leaked;
matching the train and evaluation group-size distributions (43.5 vs 5.6 rows per
user, an 8× mismatch that no run has successfully measured); and parallel
candidate evaluation, since wall-clock is the scored compute measure and the
machine has cores to spare.

---

## 11. A note on honesty in this report

Two things a reader should be able to check against our logs.

**The human baseline is labelled as such.** `submission_v11.csv` (+0.0066) is a
21-model ensemble built by hand with dozens of interventions. It is *not* our
agent's output and is not our submission. Memory entries carry a `source` field —
`agent` or `human` — so findings a person measured are never presented back to
the agent as its own prior work.

**One methodological error we found and corrected.** Earlier submissions were
selected by comparing candidates on the *test* set and keeping the best. The
rules state test data must not be used for any optimisation, and choosing between
blends on test is optimisation. The final ensembles were rebuilt with an
equal-weight rule fixed in advance — no argmax, no free parameters — and the
agent's submissions are selected on validation alone, with test computed once in
`finalize.py`. The corrected procedure also scored higher.

We also corrected a false negative in our own findings: an intra-user listwise
loss was recorded as harmful at −0.0019, which turned out to be a softmax-versus-
sigmoid implementation error. Re-measured correctly it is +0.0000 — neutral, not
harmful. The wrong entry is retained in memory, marked superseded with the
reason, rather than deleted.

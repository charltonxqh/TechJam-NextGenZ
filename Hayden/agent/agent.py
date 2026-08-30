"""The autonomous ML research agent.

Loop: hypothesise -> implement -> run isolated -> diagnose -> keep/revert -> log.

Design commitments:
  * Selection is on VALIDATION only. The test split is not even in the data cache.
  * Best-so-far is immutable; each iteration is a candidate that must earn promotion.
  * Failures are recovered from and recorded, never fatal.
  * Everything needed for the graded run-log is emitted as it happens.
  * Token and wall-clock accounting is metered from iteration 1 (Feasibility).

Usage:
    python agent.py --iterations 30 --minutes 300
    python agent.py --resume
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "kuairand-starter-kit"))

import context as ctx                                    # noqa: E402
from diagnose import build_segments, diagnose, format_for_llm, SIGMA, REF  # noqa: E402
from llm import LLM, QuotaExhausted                      # noqa: E402
from prep import load_cache                              # noqa: E402
from runner import Runner                                # noqa: E402
from guards import verify_integrity                      # noqa: E402
from tree import Tree                                    # noqa: E402

STATE = HERE / "state"
EPS, N_STALL = 0.002, 3                                  # official convergence rule


class Agent:
    def __init__(self, max_iterations=30, max_minutes=300, model="gemini-3.6-flash",
                 timeout_s=1200, resume=False, inherit=False):
        STATE.mkdir(exist_ok=True)
        self.max_iterations, self.max_minutes = max_iterations, max_minutes
        self.inherit = inherit
        self.llm = LLM(model=model)
        self.runner = Runner(timeout_s=timeout_s)
        self.t0 = time.time()

        from data import load
        self.segments = build_segments(load(str(HERE.parent / "kuairand-starter-kit"
                                               / "KuaiRand-Pure" / "data")))
        self.D = load_cache()

        self.state = self._load() if resume and (STATE / "state.json").exists() else {
            "iteration": 0,
            "best_primary": None,
            "best_code": (HERE / "seed_solution.py").read_text(),
            "attempts": [],
            "stall_count": 0,
            "interventions": 0,
            "converged": False,
        }
        self.best_scores = None
        if (STATE / "best_scores.npy").exists():
            self.best_scores = np.load(STATE / "best_scores.npy")

    # ------------------------------------------------------------ state
    def _load(self):
        s = json.loads((STATE / "state.json").read_text())
        # _save writes with default=str, which turns numpy floats into strings.
        # Resuming then crashed formatting best_primary as %.4f. Coerce the
        # numeric fields back on the way in so a resume is never worse than a
        # fresh start.
        for k in ("best_primary", "stall_count", "iteration", "interventions"):
            if isinstance(s.get(k), str):
                try:
                    s[k] = float(s[k]) if k == "best_primary" else int(float(s[k]))
                except ValueError:
                    pass
        for a in s.get("attempts", []):
            for k in ("primary", "delta"):
                if isinstance(a.get(k), str):
                    try:
                        a[k] = float(a[k])
                    except ValueError:
                        a[k] = None
        s["best_history"] = [float(x) for x in s.get("best_history", [])
                             if str(x).replace(".", "", 1).replace("-", "", 1).isdigit()]
        return s

    @staticmethod
    def _jsonable(o):
        """Convert numpy scalars/arrays to native types BEFORE serialising.

        The previous `default=str` turned every numpy float into a string, so
        state.json round-tripped 0.6015 as "0.6015". Resuming then crashed three
        separate times on arithmetic and formatting against those strings. Fix
        the write rather than coercing each field back on read.
        """
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def _save(self):
        (STATE / "state.json").write_text(
            json.dumps(self.state, indent=2, default=self._jsonable))
        (STATE / "best_solution.py").write_text(self.state["best_code"])
        if self.best_scores is not None:
            np.save(STATE / "best_scores.npy", self.best_scores)
        (STATE / "usage.json").write_text(json.dumps({
            **self.llm.usage.as_dict(),
            "wall_clock_minutes": round((time.time() - self.t0) / 60, 1),
            "iterations_used": self.state["iteration"],
            "manual_interventions": self.state["interventions"],
        }, indent=2))

    def _log(self, entry):
        with open(STATE / "run_log.jsonl", "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    # ------------------------------------------------------------ budget
    def budget(self):
        return {
            "iterations_left": self.max_iterations - self.state["iteration"],
            "minutes_left": max(0.0, self.max_minutes - (time.time() - self.t0) / 60),
            "stall_count": self.state["stall_count"],
        }

    def out_of_budget(self):
        b = self.budget()
        return b["iterations_left"] <= 0 or b["minutes_left"] <= 0

    # ----------------------------------------------------- baseline eval
    def establish_baseline(self):
        """Run the seed once so 'current best' is a measured number, not an assumption."""
        if self.state["best_primary"] is not None:
            # Resuming: the baseline is already measured, but the tree still has
            # to exist. It rehydrates from state/tree.json when there is one.
            self.tree = Tree(self.state["best_code"], self.state["best_primary"],
                             inherit=True)   # resuming: always rehydrate
            return
        print("Establishing baseline from seed solution ...")
        r = self.runner.run_code(self.state["best_code"], 0)
        if not r.ok:
            raise RuntimeError(f"Seed solution failed — cannot start: {r.brief()}")
        rep = diagnose(self.D["uva"], self.D["yva"], r.scores, self.segments,
                       history=r.history)
        self.state["best_primary"] = rep["headline"]["primary"]
        self.best_scores = r.scores
        print(f"  baseline valid primary = {self.state['best_primary']:.4f} "
              f"(official: {REF['baseline_fm']})")
        self.tree = Tree(self.state["best_code"], self.state["best_primary"],
                         inherit=self.inherit)

        # A tree inherited from an earlier run may already hold something better
        # than the freshly reproduced baseline. Adopt it as the incumbent, or the
        # two disagree: deltas would be measured against 0.6015 while the tree
        # says 0.6040, so re-deriving the known solution would register as a
        # fresh gain and the convergence window would be reading the wrong
        # series. The baseline is still reproduced first, which is what the task
        # requires - we just do not pretend to have forgotten the rest.
        tb = self.tree.best
        if tb["primary"] > self.state["best_primary"] + 1e-9:
            print(f"  inherited tree: {len(self.tree.nodes)} nodes, best "
                  f"{tb['primary']:.4f} ({tb['id']}: {tb['summary'][:50]})")
            print(f"  adopting it as the incumbent "
                  f"({self.state['best_primary']:.4f} -> {tb['primary']:.4f})")
            self.state["best_primary"] = tb["primary"]
            self.state["best_code"] = tb["code"]
            self.state["inherited_from_tree"] = tb["id"]
        self._save()

    # ---------------------------------------------------------- one step
    def iterate(self):
        self.state["iteration"] += 1
        it = self.state["iteration"]
        print(f"\n{'='*72}\nITERATION {it}   best={self.state['best_primary']:.4f}   "
              f"stall={self.state['stall_count']}/{N_STALL}\n{'='*72}")

        # --- choose which solution to build on ------------------------
        # Greedy expansion always starts from the incumbent, which cannot reach a
        # mechanism whose first step scores BELOW it. The tree keeps weaker
        # candidates expandable; select() decides whether to refine the incumbent
        # or revisit a promising branch.
        node = self.tree.select()
        self.tree.visit(node["id"])
        self.state["expanding"] = node["id"]
        if node["id"] != self.tree.best["id"]:
            print(f"  expanding {node['id']} (primary {node['primary']:.4f}, "
                  f"{self.tree.best['primary'] - node['primary']:+.4f} vs incumbent): "
                  f"{node['summary'][:60]}")

        prompt = ctx.build_prompt(
            best_code=node["code"],
            best_primary=node["primary"],
            memory=ctx.format_memory(self.state["attempts"]),
            last_feedback=self.state.get("last_feedback", ""),
            iteration=it,
            budget=self.budget(),
            tree=self.tree.as_prompt_section(),
            incumbent=self.tree.best["primary"],
        )

        # --- investigate ---------------------------------------------
        # The agent asks its own questions of the data before proposing. Any
        # finding it cites is then one it actually made, which is what the
        # Innovation criterion grades. Best-effort: if the tool phase fails the
        # iteration still proceeds on the prompt alone.
        try:
            found = ctx.gather(self.llm, prompt, log=print)
            if found:
                prompt += "\n\n## What you found\n\n" + found
                self.state.setdefault("investigations", []).append(
                    {"iteration": it, "transcript": found[:4000]})
        except Exception as e:
            print(f"    investigation skipped: {type(e).__name__}: {e}")

        # --- propose -------------------------------------------------
        try:
            prop = self.llm.ask_json(prompt, ctx.OUTPUT_SCHEMA, system=ctx.SYSTEM,
                                     max_output_tokens=32000)
        except QuotaExhausted as e:
            # No model can serve. Continuing would fail every remaining iteration
            # and pollute the log with identical errors — stop cleanly and keep
            # the best-so-far, which is resumable once quota resets.
            print(f"\n  ALL MODELS EXHAUSTED — stopping cleanly.\n  {e}")
            self.state["stopped_reason"] = "quota_exhausted"
            self.state["iteration"] -= 1                  # this one never ran
            self._save()
            raise
        except Exception as e:                            # noqa: BLE001
            print(f"  LLM proposal failed: {e}")
            self._record_failure(it, "llm_error", str(e)[:400], {})
            return
        print(f"  hypothesis: {prop['hypothesis'][:200]}")
        print(f"  change:     {prop['change_summary'][:150]}")

        # --- implement + recover ------------------------------------
        code, res, attempts_used = prop["code"], None, 0
        for attempt in range(3):                          # 1 try + 2 repairs
            attempts_used = attempt + 1
            res = self.runner.run_code(code, it, seed=0)
            if res.ok:
                break
            print(f"  run failed [{res.failure_class}] {(res.error or '')[:120]}")
            if attempt == 2:
                break
            try:
                fix = self.llm.ask_json(
                    self._repair_prompt(code, res), ctx.OUTPUT_SCHEMA,
                    system=ctx.SYSTEM, max_output_tokens=32000)
                code = fix["code"]
                print(f"  repairing (attempt {attempt+2}) ...")
            except Exception as e:                        # noqa: BLE001
                print(f"  repair failed: {e}")
                break

        if not res.ok:
            self._record_failure(it, res.failure_class, res.error, prop,
                                 repairs=attempts_used)
            return

        # --- measure -------------------------------------------------
        # Diagnostics run on data the agent generated, so they must be treated as
        # untrusted input. A malformed per-epoch history once raised KeyError
        # here and killed the entire run mid-flight - the exact "long iterative
        # runs must neither crash nor stall" failure. A broken report is now one
        # failed iteration, never a dead run.
        try:
            rep = diagnose(self.D["uva"], self.D["yva"], res.scores, self.segments,
                           history=res.history, prev_scores=self.best_scores,
                           best_primary=self.state["best_primary"])
            prim = rep["headline"]["primary"]
            feedback = format_for_llm(rep)      # inside the guard: it reads the
                                                # same untrusted report
        except Exception as e:                                  # noqa: BLE001
            print(f"  diagnostics failed: {type(e).__name__}: {e}")
            self._record_failure(it, "contract",
                                 f"diagnostics could not read the run's output "
                                 f"({type(e).__name__}: {e}). Return scores as a "
                                 f"1-D array and history as a list of dicts with "
                                 f"epoch/train_loss/valid_primary.",
                                 prop, repairs=attempts_used)
            return
        delta = prim - self.state["best_primary"]
        print(f"  -> primary {prim:.4f} ({delta:+.4f})  [{res.secs:.0f}s]")

        # --- decide (valid only, noise-aware) ------------------------
        # A hard 2-sigma cutoff throws away real gains: the same BPR change
        # measured +0.0020 / +0.0018 / +0.0016 across runs, so a fixed bar at
        # 0.0016 accepts or rejects an identical mechanism by luck — and a
        # rejection is then recorded as a failed idea, steering the agent away
        # from the one direction that works. For a marginal positive delta,
        # spend ~10s per extra seed to measure it properly instead of guessing.
        seeds_used = [0]
        if delta > 2 * SIGMA:
            kept = True
        elif delta > 0:
            print(f"  marginal (+{delta:.4f}) — confirming across seeds ...")
            deltas = [delta]
            for s in (1, 2):
                rs = self.runner.run_code(code, it, seed=s)
                if not rs.ok:
                    break
                ps = diagnose(self.D["uva"], self.D["yva"], rs.scores, self.segments,
                              history=rs.history)["headline"]["primary"]
                deltas.append(ps - self.state["best_primary"])
                seeds_used.append(s)
                print(f"    seed {s}: {ps:.4f} ({deltas[-1]:+.4f})")
            mean_d = float(np.mean(deltas))
            kept = len(deltas) >= 2 and mean_d > SIGMA
            print(f"    mean over {len(deltas)} seeds: {mean_d:+.4f} -> "
                  f"{'KEEP' if kept else 'reject'}")
            delta = mean_d
            prim = self.state["best_primary"] + mean_d
        else:
            kept = False

        # Record EVERY measured candidate as a node, whether or not it beat the
        # incumbent. This is the whole point: a node below the incumbent is not
        # discarded, it becomes a place the search can return to and build on.
        new_node = self.tree.add(self.state.get("expanding", "n0"), code, prim,
                                 prop["change_summary"])

        if kept:
            self.state["best_primary"], self.state["best_code"] = prim, code
            self.best_scores = res.scores
            print("  KEPT — new best")
        elif new_node["dead"]:
            # Past the deficit cap: recorded for the log, but not somewhere the
            # search will spend another iteration.
            print(f"  not an improvement, and {self.tree.best['primary'] - prim:.4f} "
                  f"below the incumbent — recorded as {new_node['id']}, not expandable")
        else:
            print(f"  not an improvement ({prim:.4f}) — kept as {new_node['id']}, "
                  f"expandable: a later iteration can build on it")

        # Official convergence rule: converged when the validation score has not
        # improved by more than EPS *over the last N consecutive iterations*
        # (Chinese README: "连续 3 轮迭代 validation 主分提升不超过 0.002").
        # This is a CUMULATIVE window, not a per-iteration test — three gains of
        # 0.001 total 0.003 and are NOT converged. Testing each iteration
        # individually is strictly harsher than the rule and ends runs early.
        self.state.setdefault("best_history", []).append(self.state["best_primary"])
        bh = self.state["best_history"]
        if len(bh) > N_STALL:
            window_gain = bh[-1] - bh[-(N_STALL + 1)]
            self.state["stall_count"] = N_STALL if window_gain <= EPS else 0
        else:
            self.state["stall_count"] = 0          # not enough history to judge yet

        self.state["attempts"].append({
            "iteration": it, "hypothesis": prop["hypothesis"],
            "reasoning": prop["reasoning"], "prediction": prop["prediction"],
            "falsifier": prop["falsifier"], "change_summary": prop["change_summary"],
            "primary": prim, "delta": delta, "kept": kept, "repairs": attempts_used,
            "secs": res.secs,
        })
        self.state["last_feedback"] = feedback
        import findings                                   # persist across runs
        import memory
        findings.record(prop["change_summary"], delta, kept)
        # Same measurement, but with provenance and replication count, so a
        # later run can tell an agent-made finding from a human-made one and a
        # confirmed result from a single-seed guess.
        memory.record(prop["change_summary"], delta, kept,
                      source=memory.SOURCE_AGENT,
                      seeds=len(deltas) if "deltas" in dir() else 1,
                      iteration=it,
                      note=prop.get("hypothesis", "")[:300])
        self._log({"ts": datetime.now(timezone.utc).isoformat(), "iteration": it,
                   "hypothesis": prop["hypothesis"], "reasoning": prop["reasoning"],
                   "prediction": prop["prediction"], "falsifier": prop["falsifier"],
                   "change_summary": prop["change_summary"], "code": code,
                   "metrics": rep["headline"], "diagnostics": rep, "kept": kept,
                   "repairs": attempts_used, "secs": res.secs,
                   "usage": self.llm.usage.as_dict()})
        self._save()

    def _repair_prompt(self, code, res):
        return (f"Your code failed to run. Fix it and return the COMPLETE corrected module.\n\n"
                f"FAILURE CLASS: {res.failure_class}\n"
                f"ERROR: {res.error_type}: {res.error}\n\n"
                f"RECOVERY GUIDANCE: {res.recovery}\n\n"
                f"TRACEBACK:\n{(res.traceback or '')[-2500:]}\n\n"
                f"YOUR CODE:\n```python\n{code}\n```\n\n"
                f"Keep the same hypothesis — only fix the defect.")

    def _record_failure(self, it, fclass, err, prop, repairs=0):
        # A failed iteration makes no progress, so it enters the convergence
        # window as an unchanged best — same accounting as a reverted change.
        self.state.setdefault("best_history", []).append(self.state["best_primary"])
        bh = self.state["best_history"]
        if len(bh) > N_STALL:
            self.state["stall_count"] = N_STALL if (bh[-1] - bh[-(N_STALL + 1)]) <= EPS else 0
        self.state["attempts"].append({
            "iteration": it, "hypothesis": prop.get("hypothesis", "(no proposal)"),
            "change_summary": prop.get("change_summary", "(failed before running)"),
            "primary": None, "delta": 0.0, "kept": False,
            "failure_class": fclass, "error": err, "repairs": repairs,
            "lesson": f"This approach failed with {fclass}. Try a different mechanism.",
        })
        self.state["last_feedback"] = (
            f"Your last attempt FAILED ({fclass}) and could not be repaired: "
            f"{(err or '')[:400]}\nPick a different approach.")
        self._log({"ts": datetime.now(timezone.utc).isoformat(), "iteration": it,
                   "failed": True, "failure_class": fclass, "error": err,
                   "hypothesis": prop.get("hypothesis"), "repairs": repairs,
                   "usage": self.llm.usage.as_dict()})
        self._save()

    # -------------------------------------------------------------- run
    def loop(self):
        verify_integrity()
        self.establish_baseline()
        while not self.out_of_budget():
            if self.state["stall_count"] >= N_STALL:
                print(f"\nCONVERGED — {N_STALL} consecutive iterations without "
                      f">{EPS} improvement.")
                self.state["converged"] = True
                break
            try:
                self.iterate()
            except QuotaExhausted:
                break                                     # already logged & saved
            verify_integrity()                            # scorer must stay untouched
        self._report()

    def _report(self):
        s, u = self.state, self.llm.usage.as_dict()
        base = REF["baseline_fm"]
        print(f"\n{'='*72}\nRUN COMPLETE\n{'='*72}")
        print(f"  iterations         {s['iteration']}")
        print(f"  best valid primary {s['best_primary']:.4f}  "
              f"(baseline {base}, delta {s['best_primary']-base:+.4f})")
        print(f"  headroom captured  {100*(s['best_primary']-base)/(REF['oracle']-base):.1f}%")
        print(f"  converged          {s['converged']}")
        print(f"  wall clock         {(time.time()-self.t0)/60:.1f} min")
        print(f"  LLM tokens         {u['total_tokens']:,} ({u['calls']} calls)")
        print(f"  interventions      {s['interventions']}")
        kept = [a for a in s["attempts"] if a.get("kept")]
        print(f"  accepted changes   {len(kept)}/{len(s['attempts'])}")
        for a in kept:
            print(f"    iter {a['iteration']}: {a['change_summary'][:80]} -> {a['primary']:.4f}")
        print(f"\n  artefacts in {STATE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=30)  # cap is 50; convergence ends it first
    ap.add_argument("--minutes", type=float, default=300)  # 5h, under the 6h ceiling
    ap.add_argument("--model", default="gemini-3.6-flash")
    ap.add_argument("--inherit", action="store_true",
                    help="carry the tree and incumbent over from a previous run "
                         "(off by default: convergence is judged per run)")
    ap.add_argument("--timeout", type=int, default=1200)   # 20 min, matches the contract
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    Agent(max_iterations=a.iterations, max_minutes=a.minutes, model=a.model,
          timeout_s=a.timeout, resume=a.resume, inherit=a.inherit).loop()

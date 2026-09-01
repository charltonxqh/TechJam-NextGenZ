"""Solution tree — so a mechanism that needs several steps can be reached.

The greedy loop keeps exactly one solution: the validation-best. Every other
candidate is discarded the moment it fails to improve. That is fine when good
mechanisms improve on their first attempt, and fatal when they do not.

They do not here. The path to the best model measured in this project, in order:

    CatBoost, 5 default fields, default params      0.5954   <- WORSE than FM
    + 20 platform-statistic features                0.6044
    + all 110 statistic features                    0.6054
    + QueryRMSE ranking loss                        0.6075   <- best measured

Step 1 loses 0.006 against the FM baseline. A greedy loop reverts there and can
never reach step 4. Run 4 of this agent proposed exactly that first step, scored
exactly 0.5954, and correctly discarded it under its own rules. Combined with the
organisers' convergence rule (3 iterations without +0.002 ends the run), a
conjunction that pays off only on the third step is structurally unreachable.

So: keep a POOL of solutions rather than a single best. A candidate that scored
below the incumbent stays in the pool as an expandable node. The agent can then
propose "tune the CatBoost node" instead of only ever "change the FM".

Selection follows AIDE (arXiv:2502.13138): explore promising-but-not-yet-winning
branches rather than committing greedily to the incumbent.

WHAT DOES NOT CHANGE
  * The reported best, and the submission, are still the validation-best node.
    Keeping a weak node explorable never promotes it.
  * The convergence rule is untouched: it reads best_primary, which only moves
    when a node actually beats the incumbent.
  * Selection remains validation-only.
"""
from __future__ import annotations

import json
import math
import pathlib

STATE = pathlib.Path(__file__).resolve().parent / "state"
FILE = STATE / "tree.json"

# A node this far below the incumbent is a dead end, not a scaffold. CatBoost's
# first attempt was 0.006 below; three times the per-seed sigma of 0.0008 is
# 0.0024, which would have excluded it, so the bound is set from the measured
# case rather than from sigma.
MAX_DEFICIT = 0.010


class Tree:
    def __init__(self, root_code: str, root_primary: float, inherit: bool = False):
        """inherit=False starts from an empty tree, ignoring any tree.json.

        The tree WITHIN a run is the mechanism - it is what let a branch that
        scored below baseline at iteration 1 become the best at iteration 3.
        Carrying it ACROSS runs is a different thing, and it is off by default:
        the organisers judge convergence per run, so a chain of runs that each
        start where the last stopped accumulates score while every individual
        run still looks converged. Opt in with --inherit when that is what you
        actually want, and disclose it.
        """
        self.nodes: list[dict] = []
        if inherit:
            self.load()
        elif FILE.exists():
            # Starting fresh still has to write to tree.json, which would
            # silently destroy the previous run's tree. Set it aside first -
            # discarding a run's search history as a side effect of not passing
            # a flag is not a reasonable default.
            import time as _t
            FILE.rename(FILE.with_name(f"tree.{_t.strftime('%Y%m%d-%H%M%S')}.json"))
        if not self.nodes:
            self.nodes.append({
                "id": "n0", "parent": None, "code": root_code,
                "primary": float(root_primary), "summary": "official FM baseline",
                "visits": 0, "depth": 0, "dead": False,
            })
            self.save()

    # ------------------------------------------------------------- io
    def load(self):
        if FILE.exists():
            try:
                self.nodes = json.loads(FILE.read_text())
            except json.JSONDecodeError:
                self.nodes = []

    def save(self):
        STATE.mkdir(exist_ok=True)
        FILE.write_text(json.dumps(self.nodes, indent=2))

    # ---------------------------------------------------------- access
    @property
    def best(self) -> dict:
        return max((n for n in self.nodes), key=lambda n: n["primary"])

    def get(self, nid: str) -> dict | None:
        return next((n for n in self.nodes if n["id"] == nid), None)

    def add(self, parent_id: str, code: str, primary: float, summary: str) -> dict:
        p = self.get(parent_id) or self.nodes[0]
        n = {
            "id": f"n{len(self.nodes)}", "parent": parent_id, "code": code,
            "primary": float(primary), "summary": summary[:200],
            "visits": 0, "depth": p["depth"] + 1,
            # Far enough below the incumbent that expanding it is not worth an
            # iteration. Recorded, never expanded.
            "dead": float(primary) < self.best["primary"] - MAX_DEFICIT,
        }
        self.nodes.append(n)
        self.save()
        return n

    # --------------------------------------------------------- select
    def _refresh_dead(self) -> None:
        """Re-evaluate every node's dead flag against the CURRENT incumbent.

        `dead` used to be computed once, when a node was added, and never
        revisited. That is wrong in both directions as the incumbent moves:

          * a node 0.0096 below a 0.6015 incumbent was marked alive; once the
            incumbent reached 0.6037 its deficit was 0.0118, past the cap, yet
            it stayed selectable and consumed a 9-minute iteration
          * conversely a branch killed early - because the agent's first
            implementation of it was poor - stayed dead forever, even across
            runs, closing off a model family on the strength of one bad attempt

        Recomputing makes the cap mean what it says at all times, and lets a
        branch come back if the incumbent it was measured against was itself
        weak.
        """
        best_p = max(n["primary"] for n in self.nodes)
        changed = False
        for n in self.nodes:
            now_dead = n["primary"] < best_p - MAX_DEFICIT
            if now_dead != n["dead"]:
                n["dead"] = now_dead
                changed = True
        if changed:
            self.save()

    def select(self) -> dict:
        self._refresh_dead()
        """Pick the node to expand next.

        UCB-style: value is how close the node is to the incumbent, exploration
        is the usual sqrt(log N / visits) bonus. A node that has never been
        expanded therefore gets tried before one already explored twice, which
        is what stops the search collapsing back onto the incumbent every turn.
        """
        live = [n for n in self.nodes if not n["dead"]]
        if not live:
            return self.nodes[0]
        total = sum(n["visits"] for n in live) + 1
        best_p = self.best["primary"]

        def score(n):
            # Value falls LINEARLY from 1 at the incumbent to 0 at MAX_DEFICIT.
            # Scaling by sigma instead (1/(1+deficit/sigma)) collapses far too
            # fast: the CatBoost first step sits ~7.6 sigma down and scored 0.12,
            # so it was picked once in twelve turns and the search still
            # effectively followed the incumbent. The cap, not the noise floor,
            # is the right yardstick for "how much worse is acceptable".
            deficit = max(0.0, best_p - n["primary"])
            value = max(0.0, 1.0 - deficit / MAX_DEFICIT)
            # c=2.5 chosen by simulating the case this whole module exists for:
            # incumbent FM 0.6015 against the untuned CatBoost 0.5954. At c=1.2
            # that branch got 17% of expansions, which over a run ended by three
            # non-improving iterations is too rare to find the tuning it needed.
            # c=2.5 gives it 33%; higher barely moves it and wastes iterations
            # on genuinely weak nodes.
            explore = 2.5 * math.sqrt(math.log(total + 1) / (n["visits"] + 1))
            # mild penalty for depth so the search does not run away down one
            # branch while shallow alternatives are unexplored
            return value + explore - 0.05 * n["depth"]

        return max(live, key=score)

    def visit(self, nid: str):
        n = self.get(nid)
        if n:
            n["visits"] += 1
            self.save()

    # ---------------------------------------------------------- prompt
    def as_prompt_section(self) -> str:
        if len(self.nodes) <= 1:
            return ""
        best = self.best
        lines = ["## Solution tree",
                 "",
                 "Every candidate measured so far is kept, including ones that scored",
                 "BELOW the incumbent. A weaker node is not a failure to avoid - it may",
                 "be the first step of a mechanism that only pays off after two or three",
                 "further changes, and this project has measured exactly that: CatBoost",
                 "scored 0.5954 untuned (below the 0.6015 baseline) and 0.6075 once it had",
                 "richer features and a ranking loss.",
                 ""]
        for n in sorted(self.nodes, key=lambda x: -x["primary"]):
            mark = " <- INCUMBENT" if n["id"] == best["id"] else (
                " (dead end)" if n["dead"] else "")
            lines.append(f"  {n['id']}  primary {n['primary']:.4f}  depth {n['depth']}  "
                         f"visits {n['visits']}{mark}")
            lines.append(f"       {n['summary']}")
        return "\n".join(lines)


if __name__ == "__main__":
    t = Tree("# fm", 0.6015)
    t.add("n0", "# catboost", 0.5954, "CatBoost untuned, 5 fields")
    t.add("n1", "# catboost+feats", 0.6044, "CatBoost + 20 statistic features")
    t.add("n2", "# catboost+all", 0.6075, "CatBoost + 110 features + QueryRMSE")
    print(t.as_prompt_section())
    print("\nselect ->", t.select()["id"], t.select()["summary"])
    print("best   ->", t.best["id"], t.best["primary"])

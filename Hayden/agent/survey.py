"""Literature / prior-solution survey, run BY the agent.

The organisers state this explicitly:

    "Prior solutions on GitHub are fair reading. Your agent can run that survey
     for you."

and the Innovation criterion rewards "originality in drawing on published
methods, papers, or public solutions". So instead of us reading GitHub and
hand-feeding conclusions into the prompt (which would be our insight, not the
agent's), the agent performs the survey itself with search grounding and the
result becomes part of its own context.

Run once per project — the result is cached, so it costs one API call, not one
per iteration.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FILE = HERE / "state" / "survey.md"

PROMPT = """You are surveying prior published work before starting an ML research task.
Search the web and report what is actually known.

TASK CONTEXT
  Dataset : KuaiRand-Pure (Kuaishou short-video feed), 1.14M training rows,
            27K users, 7.6K videos, ~43 interactions per user.
  Task    : within-user ranking of logged impressions. Rank each user's own
            impressions against each other; no retrieval from a catalogue.
  Label   : long_view (binary).
  Metrics : GAUC and nDCG@5; primary = mean of the two.
  Baseline: a Factorization Machine over 5 categorical fields
            (user_id, video_id, author_id, tab, duration_bucket), k=16,
            pointwise BCE. Validation primary 0.6015.

ALREADY MEASURED AND RULED OUT on this exact benchmark (do not re-suggest these):
  - larger embedding dimension (k=8/16/32 all flat)
  - more static categorical feature fields (5 -> 13 fields, flat/slightly worse)
  - pairwise BPR loss, listwise softmax loss, joint BCE+BPR (all <= 0)
  - multi-task heads on is_click and on play_time_ms (both ~0)
  - per-user loss weighting by 1/N_u (-0.0068)
  - user x author / user x tab / user x duration history features (too sparse:
    only 3.4% of rows involve an author the user has seen before)

SEARCH FOR AND REPORT:
1. Published results on KuaiRand specifically. What GAUC / nDCG do papers report
   for long_view or click prediction, and with which models? Is ~0.66 GAUC
   typical, or do stronger results exist?
2. Techniques that reliably improve GAUC on small-to-medium recommendation
   datasets (order 1M interactions) where capacity is NOT the bottleneck.
3. Any open-source implementations or competition write-ups for KuaiRand.
4. Ensembling and variance-reduction practice in recsys ranking competitions.

RULES FOR YOUR ANSWER
  - Report only what you actually found, with the source named.
  - If you cannot find real evidence for something, say so. Do not speculate
    and present it as a finding.
  - Be concrete: name methods, name numbers, name repositories.
  - Flag explicitly anything that contradicts the ruled-out list above, because
    that would be the most valuable thing you could find.
  - Keep it under 700 words."""


def run(llm=None, force: bool = False) -> str:
    """Perform the survey (or return the cached one)."""
    if FILE.exists() and not force:
        return FILE.read_text()
    if llm is None:
        from llm import LLM
        llm = LLM()
    text = llm.ask(PROMPT, max_output_tokens=24000, search=True)
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(text)
    return text


def as_prompt_section() -> str:
    if not FILE.exists():
        return ""
    return ("## Survey of prior published work (performed by you, with web search)\n\n"
            + FILE.read_text().strip())


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    print(run(force="--force" in sys.argv))

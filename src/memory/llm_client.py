"""
llm_client.py
--------------
Handles calls to an LLM (Google Gemini) that analyzes ONE iteration's
code change: given the hypothesis, the code diff, and the before/after
scores, it produces:
  1. a short plain-English summary of what the code change does
  2. a guess at WHY the score moved the way it did

This is kept in its own file, separate from memory_store.py, so that:
  - the memory module has no hard dependency on any specific LLM provider
  - if the API key is missing, the network is down, or the call fails,
    the rest of the system keeps running via a rule-based fallback
    (this directly supports the "Robust operation" requirement - a
    flaky API call must never crash a multi-hour run)

SECURITY:
  - The API key is NEVER hardcoded here. It is read from the
    GOOGLE_API_KEY environment variable.
  - Set it before running:
        export GOOGLE_API_KEY="your-key-here"      (Mac/Linux)
        setx GOOGLE_API_KEY "your-key-here"         (Windows)
  - Get a free-tier key from https://aistudio.google.com/apikey
  - Never commit a real key to git. If you already pasted one somewhere
    public (chat, code, commit), revoke it and generate a new one.
"""

import os
import json
import time
from typing import Optional
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a .env file in the current directory, if present
except ImportError:
    pass  # dotenv is optional - falls back to real environment variables

try:
    import requests
except ImportError:
    requests = None


GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-3.6-flash:generateContent"
)

# Keep the diff sent to the LLM bounded - the full diff already lives
# permanently in the run log; the LLM only needs enough to reason about
# what changed. This keeps token cost (Feasibility scoring) predictable.
MAX_DIFF_CHARS_IN_PROMPT = 3000


@dataclass
class CodeAnalysis:
    summary: str             # 1-sentence plain-English description of the code change
    likely_reason: str       # 1-sentence guess at why the score moved this way
    tokens_used: int = 0
    used_fallback: bool = False  # True if the LLM call failed/was skipped


class LLMClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 2,
        timeout: int = 20,
        model_endpoint: str = GEMINI_ENDPOINT,
    ):
        # Falls back to the environment variable if no key is passed in.
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.max_retries = max_retries
        self.timeout = timeout
        self.model_endpoint = model_endpoint

    def analyze_change(
        self,
        hypothesis: str,
        code_diff: str,
        primary_before: Optional[float],
        primary_after: Optional[float],
    ) -> CodeAnalysis:
        """
        Main entry point. Always returns a CodeAnalysis - never raises.
        If the API key is missing, requests isn't installed, or the call
        fails after retries, returns a rule-based fallback instead.
        """
        if not self.api_key:
            return self._fallback(hypothesis, primary_before, primary_after,
                                   reason_suffix="(no API key set)")
        if requests is None:
            return self._fallback(hypothesis, primary_before, primary_after,
                                   reason_suffix="(requests library not installed)")

        trimmed_diff = code_diff[:MAX_DIFF_CHARS_IN_PROMPT]
        prompt = self._build_prompt(hypothesis, trimmed_diff, primary_before, primary_after)

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._call_api(prompt)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(1.5 * (attempt + 1))  # simple linear backoff
                    continue

        # all retries exhausted - degrade gracefully instead of crashing the run
        return self._fallback(
            hypothesis, primary_before, primary_after,
            reason_suffix=f"(LLM call failed after retries: {last_error})"
        )

    # ------------------------------------------------------------------
    def _build_prompt(self, hypothesis, code_diff, primary_before, primary_after) -> str:
        return f"""You are reviewing one iteration of an autonomous ML experiment.

Hypothesis for this iteration: {hypothesis}

Code change applied:
{code_diff}

Validation primary score before this change: {primary_before}
Validation primary score after this change: {primary_after}

Respond with ONLY a JSON object, no other text, with exactly these two keys:
{{"summary": "one sentence describing what the code change actually does",
  "likely_reason": "one sentence on why the score likely moved this way"}}
"""

    def _call_api(self, prompt: str) -> CodeAnalysis:
        # This method can be called independently of ``analyze_change``.
        # The explicit guard both prevents an AttributeError and narrows the
        # optional import for static type checkers such as Pylance.
        http = requests
        if http is None:
            raise RuntimeError("The requests library is not installed")

        resp = http.post(
            f"{self.model_endpoint}?key={self.api_key}",
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=self.timeout,
        )
        if not resp.ok:
            # Surface the ACTUAL reason (bad model name, invalid key, quota,
            # etc.) instead of letting raise_for_status() hide it behind a
            # generic "HTTPError" class name.
            raise RuntimeError(
                f"HTTP {resp.status_code} from Gemini API: {resp.text[:300]}"
            )
        data = resp.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip()
        # models sometimes wrap JSON in markdown fences - strip those if present
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        parsed = json.loads(text.strip())

        tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
        return CodeAnalysis(
            summary=parsed.get("summary", "").strip(),
            likely_reason=parsed.get("likely_reason", "").strip(),
            tokens_used=tokens,
            used_fallback=False,
        )

    # ------------------------------------------------------------------
    def _fallback(self, hypothesis, primary_before, primary_after, reason_suffix="") -> CodeAnalysis:
        """
        Rule-based backup. Guarantees the pipeline never breaks just
        because the LLM call didn't work.
        """
        if primary_before is None or primary_after is None:
            direction = "score unavailable"
        elif primary_after > primary_before:
            direction = f"improved by {primary_after - primary_before:.4f}"
        elif primary_after < primary_before:
            direction = f"dropped by {primary_before - primary_after:.4f}"
        else:
            direction = "no measurable change"

        return CodeAnalysis(
            summary=hypothesis,  # best available substitute for a real summary
            likely_reason=f"{direction} {reason_suffix}".strip(),
            tokens_used=0,
            used_fallback=True,
        )

    # ------------------------------------------------------------------
    # CONSOLIDATION - lets the LLM decide what's worth remembering from
    # a batch of iterations, instead of a hard-coded "keep last N" rule.
    # ------------------------------------------------------------------
    def consolidate_learnings(self, iteration_summaries: list, prior_notes: str = "") -> str:
        """
        Given a batch of iteration summaries (short strings, one per
        iteration) plus whatever was distilled before, ask the LLM to
        write an updated, SHORT set of "lessons learned so far" - the
        kind of thing a human researcher would jot in their notebook.

        This is what lets memory scale to 50+ iterations without the
        prompt growing forever: instead of keeping every iteration's
        detail, we periodically compress a batch into a few sentences
        that capture what actually mattered (patterns, dead ends,
        things worth revisiting) - decided by the LLM, not a fixed rule.

        Falls back to a simple bullet-point concatenation if the LLM is
        unavailable - never crashes the run.
        """
        if not self.api_key or requests is None:
            return self._fallback_consolidate(iteration_summaries, prior_notes)

        # Keep a local, non-optional reference: Pylance cannot assume that a
        # module-level variable remains unchanged between the guard and use.
        http = requests

        batch_text = "\n".join(f"- {s}" for s in iteration_summaries)
        prompt = f"""You are maintaining a running research notebook for an
ML experiment. Below are the previously distilled notes, followed by a
new batch of iteration results.

Previous notes:
{prior_notes or "(none yet)"}

New iterations:
{batch_text}

Write an UPDATED set of notes (3-6 sentences max) that captures what
actually matters for future decisions: patterns that worked, dead ends
to avoid repeating, and anything surprising. Merge with the previous
notes rather than just appending - drop anything superseded or no
longer useful. Respond with ONLY the updated notes text, no preamble.
"""
        try:
            resp = http.post(
                f"{self.model_endpoint}?key={self.api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return text
        except Exception:
            return self._fallback_consolidate(iteration_summaries, prior_notes)

    def _fallback_consolidate(self, iteration_summaries: list, prior_notes: str) -> str:
        """Rule-based backup: just keep a running bullet list, capped in length."""
        combined = (prior_notes + "\n" if prior_notes else "") + \
            "\n".join(f"- {s}" for s in iteration_summaries)
        lines = combined.strip().split("\n")
        return "\n".join(lines[-10:])  # keep only the most recent 10 bullet lines

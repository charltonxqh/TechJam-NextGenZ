"""LLM client for the ML research agent (Google Gemini).

Single place where the model is called. Everything else in the agent talks to
`LLM.ask()` and never touches the SDK directly, so swapping provider or model
is a one-file change.

Tracks cumulative token usage — the Feasibility criterion (15%) requires
reporting total input+output tokens for the run, so this is metered from
iteration 1 rather than reconstructed afterwards.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass, field

from google import genai
from google.genai import types


def _load_env(path: str = None) -> None:
    """Minimal .env loader (no python-dotenv dependency).

    Searches upward from this file rather than assuming a fixed depth. The repo
    was restructured so that agent/ moved a level deeper (into Hayden/), and the
    hardcoded parent.parent then pointed at a directory with no .env — the run
    died at startup with "No API key found" even though the file was sitting two
    levels up. .env is gitignored, so it does not move when the tracked files do;
    walking up is the only lookup that survives a reorganisation.
    """
    if path:
        p = pathlib.Path(path)
    else:
        p = None
        for d in pathlib.Path(__file__).resolve().parents:
            cand = d / ".env"
            if cand.exists():
                p = cand
                break
        if p is None:
            return
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


class EmptyResponse(RuntimeError):
    """Model returned no text (usually: thinking consumed max_output_tokens)."""


class QuotaExhausted(RuntimeError):
    """Every model in the rotation has hit its daily free-tier quota."""


# Free tier is 20 requests/day PER MODEL, so rotating models buys more budget.
# Ordered by preference; all verified to serve on this key.
MODEL_ROTATION = [
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest",
    "gemini-3.1-flash-lite-preview",
]


@dataclass
class Usage:
    """Cumulative token accounting for the whole run (Feasibility deliverable)."""
    calls: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cached_tokens: int = 0
    seconds: float = 0.0
    per_call: list = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens + self.thinking_tokens

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "llm_seconds": round(self.seconds, 1),
        }


class LLM:
    """Thin wrapper with retries and usage metering."""

    def __init__(self, model: str = "gemini-3.6-flash", api_key: str = None,
                 max_retries: int = 4, temperature: float = 1.0,
                 timeout_s: int = 120):
        _load_env()
        # Free tier is 20 req/day per MODEL per KEY, so extra keys multiply budget.
        # Collect GEMINI_API_KEY, GEMINI_API_KEY_2, ... plus GOOGLE_API_KEY.
        self.keys = [api_key] if api_key else [
            v for k, v in sorted(os.environ.items())
            if (k.startswith("GEMINI_API_KEY") or k == "GOOGLE_API_KEY") and v.strip()
        ]
        # de-dupe, preserve order
        seen, ks = set(), []
        for k in self.keys:
            if k not in seen:
                seen.add(k); ks.append(k)
        self.keys = ks
        self.key_idx = 0
        key = self.keys[0] if self.keys else None
        if not key:
            raise RuntimeError("No API key found. Put GEMINI_API_KEY=... in .env")
        # Hard HTTP timeout. Without it, a quota-blocked model hangs forever
        # instead of erroring — which stalled the whole agent during setup.
        self.timeout_s = timeout_s
        self.client = genai.Client(
            api_key=key,
            http_options=types.HttpOptions(timeout=timeout_s * 1000),  # ms
        )
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.usage = Usage()
        self._tried = [model]
        self._dead: set[str] = set()      # quota-exhausted: gone for the day
        self._busy: set[str] = set()      # 503 right now: may recover later

    def _next_key(self) -> bool:
        """Move to the next API key, keeping the current (stronger) model.

        Quota is per key AND per model, so when a good model runs out on key A it
        is usually still available on key B. Switching keys first preserves model
        quality; only when every key is spent do we drop to a weaker model.
        """
        if self.key_idx + 1 >= len(self.keys):
            return False
        self.key_idx += 1
        self.client = genai.Client(
            api_key=self.keys[self.key_idx],
            http_options=types.HttpOptions(timeout=self.timeout_s * 1000),
        )
        self._dead.clear()          # fresh key: models are available again
        self._busy.clear()
        return True

    def _rotate_model(self, permanent: bool = False) -> bool:
        """Switch models. `permanent` marks the current one dead for the day.

        Quota exhaustion is permanent (resets tomorrow); a 503 is transient, so
        an overloaded model stays eligible for a later cycle rather than being
        blacklisted for the whole run.
        """
        (self._dead if permanent else self._busy).add(self.model)

        candidates = [m for m in MODEL_ROTATION if m not in self._dead
                      and m != self.model]
        fresh = [m for m in candidates if m not in self._busy]
        pick = fresh or candidates          # prefer untried; else recycle a busy one
        if not pick:
            return False
        if not fresh:
            self._busy.clear()              # full cycle done — give them another go
        self.model = pick[0]
        if self.model not in self._tried:
            self._tried.append(self.model)
        return True

    def ask(self, prompt: str, system: str = None, max_output_tokens: int = 16000,
            json_schema: dict = None, search: bool = False) -> str:
        """One call. Retries transient failures with backoff. Returns text.

        `search=True` enables Google Search grounding so the model can survey
        prior published solutions — the organisers explicitly allow this
        ("Prior solutions on GitHub are fair reading. Your agent can run that
        survey for you."). Grounding cannot be combined with a response schema,
        so a grounded call returns free text.
        """
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system,
        )
        if search:
            cfg.tools = [types.Tool(google_search=types.GoogleSearch())]
        elif json_schema is not None:
            cfg.response_mime_type = "application/json"
            cfg.response_schema = json_schema

        last = None
        attempt = 0
        # Rotations (switching key or model) are not retries of the same request,
        # so they must not consume the retry budget — otherwise a long rotation
        # gives up before it has tried everything available.
        max_rotations = len(self.keys) * len(MODEL_ROTATION) + 4
        rotations = 0
        while attempt < self.max_retries and rotations <= max_rotations:
            t0 = time.time()
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                self._record(resp, time.time() - t0)
                text = resp.text
                if not text:
                    # Thinking models can burn the whole budget on thoughts and return
                    # no text. Retrying repeats the same outcome — fail fast and tell
                    # the caller to raise max_output_tokens instead.
                    fin = getattr(resp.candidates[0], "finish_reason", "?") if resp.candidates else "?"
                    th = self.usage.per_call[-1]["thinking"]
                    raise EmptyResponse(
                        f"empty text (finish={fin}, thinking_tokens={th}). "
                        f"Raise max_output_tokens above the thinking budget."
                    )
                return text
            except EmptyResponse:
                raise                                    # never retry — deterministic
            except Exception as e:                       # noqa: BLE001 - retry transients
                last = e
                msg = str(e)
                # Non-retryable: bad request / auth / not found
                if any(s in msg for s in ("API_KEY_INVALID", "PERMISSION_DENIED", "NOT_FOUND", "400")):
                    raise
                # Two distinct model-level failures, both better solved by moving
                # to a different model than by retrying this one:
                #   429 PerDay  — daily free-tier quota (20/req/day PER MODEL);
                #                 will not recover today, so backoff is pointless
                #   503         — that model is overloaded right now; a sibling
                #                 model is usually serving fine
                model_level = ("PerDay" in msg or "RESOURCE_EXHAUSTED" in msg
                               or "503" in msg or "UNAVAILABLE" in msg)
                if model_level:
                    is_quota = "PerDay" in msg or "RESOURCE_EXHAUSTED" in msg
                    prev = self.model
                    if is_quota and self._next_key():
                        print(f"    [quota] key {self.key_idx} -> retrying on {self.model}")
                        rotations += 1
                        continue
                    if self._rotate_model(permanent=is_quota):
                        rotations += 1
                        print(f"    [{'quota' if is_quota else 'overloaded'}] "
                              f"{prev} -> {self.model}")
                        continue
                    raise QuotaExhausted(
                        f"All models unavailable or quota-exhausted "
                        f"(dead: {sorted(self._dead)}). Wait for the daily reset "
                        f"or use a paid key."
                    )
                attempt += 1
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"LLM failed after {self.max_retries} attempts: {last}")

    def ask_json(self, prompt: str, schema: dict, system: str = None,
                 max_output_tokens: int = 16000) -> dict:
        """Structured output. Falls back to brace-extraction if the model wraps it."""
        raw = self.ask(prompt, system=system, max_output_tokens=max_output_tokens,
                       json_schema=schema)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e > s:
                return json.loads(raw[s:e + 1])
            raise

    def _record(self, resp, secs: float) -> None:
        u = getattr(resp, "usage_metadata", None)
        p = int(getattr(u, "prompt_token_count", 0) or 0) if u else 0
        o = int(getattr(u, "candidates_token_count", 0) or 0) if u else 0
        t = int(getattr(u, "thoughts_token_count", 0) or 0) if u else 0
        c = int(getattr(u, "cached_content_token_count", 0) or 0) if u else 0
        self.usage.calls += 1
        self.usage.prompt_tokens += p
        self.usage.output_tokens += o
        self.usage.thinking_tokens += t
        self.usage.cached_tokens += c
        self.usage.seconds += secs
        self.usage.per_call.append({"prompt": p, "output": o, "thinking": t, "secs": round(secs, 1)})


# ---------------------------------------------------------------- self-test
CANDIDATES = [
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite-preview",
]

if __name__ == "__main__":
    import sys
    _load_env()
    print("Probing which models serve, and their latency:\n")
    working = []
    for m in CANDIDATES:
        try:
            # no retries during probing, and a budget above the thinking allowance
            llm = LLM(model=m, max_retries=1)
            t0 = time.time()
            out = llm.ask("Reply with exactly: OK", max_output_tokens=8000)
            dt = time.time() - t0
            u = llm.usage
            print(f"  OK    {m:32s} {dt:5.1f}s  reply={out.strip()[:20]!r:12s} "
                  f"tokens(p/o/t)={u.prompt_tokens}/{u.output_tokens}/{u.thinking_tokens}")
            working.append((m, dt))
        except Exception as e:                            # noqa: BLE001
            print(f"  FAIL  {m:32s} {str(e).splitlines()[0][:80]}")
    if working:
        print(f"\nUse: {working[0][0]}")
    else:
        print("\nNo model served — check the key.")
        sys.exit(1)

"""Isolated experiment execution with a recovery ladder.

Robustness is explicitly graded ("not judged by whether the agent hits a
failure, but by how it handles one"). So every candidate runs in a subprocess
with a hard timeout, failures are classified, and each class has a defined
recovery action the agent can act on instead of crashing.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
WORK = HERE / "runs"


@dataclass
class RunResult:
    ok: bool
    iteration: int
    seed: int
    secs: float = 0.0
    scores: np.ndarray | None = None
    history: list = field(default_factory=list)
    error_type: str | None = None
    error: str | None = None
    traceback: str | None = None
    failure_class: str | None = None
    recovery: str | None = None
    stdout: str = ""
    stderr: str = ""

    def brief(self) -> str:
        if self.ok:
            return f"ok in {self.secs:.1f}s, {len(self.history)} epochs"
        return f"{self.failure_class}: {self.error_type}: {(self.error or '')[:160]}"


# ------------------------------------------------------- failure taxonomy
def classify(res: dict, stderr: str, timed_out: bool, killed: bool) -> tuple[str, str]:
    """-> (failure_class, recovery instruction the agent can act on)"""
    if timed_out:
        return ("timeout",
                "Training exceeded the wall-clock limit. The FM baseline finishes in ~8s, "
                "so anything near the limit is pathological. Reduce epochs, reduce model "
                "size, or vectorise — do not simply raise the limit.")
    blob = f"{res.get('error_type','')} {res.get('error','')} {stderr}".lower()
    if killed or "killed" in blob or "out of memory" in blob or "cannot allocate" in blob:
        return ("oom",
                "Ran out of memory. Halve the batch size and retry; if it recurs, reduce "
                "embedding dimension or avoid materialising large dense intermediates.")
    if res.get("error_type") in ("SyntaxError", "IndentationError"):
        return ("syntax",
                "The generated code does not parse. Rewrite the whole file carefully; "
                "return only valid Python.")
    if res.get("error_type") in ("ImportError", "ModuleNotFoundError"):
        return ("import",
                "Unavailable import. Only numpy, torch, scipy, sklearn, lightgbm and the "
                "standard library are installed. Rewrite using those.")
    if res.get("error_type") == "AttributeError" and "run(" in (res.get("error") or ""):
        return ("contract",
                "The module must define run(D, seed=0) returning (valid_scores, history).")
    if "nan" in blob or "inf" in blob:
        return ("numerical",
                "Produced NaN/Inf. Lower the learning rate by 10x, clamp logits, and check "
                "for log(0) or division by zero.")
    if "align" in blob or "must align" in blob or "returned" in blob and "scores" in blob:
        return ("alignment",
                "Score array length must equal the number of validation rows, in the same "
                "order as D['Xva'].")
    return ("runtime",
            "Read the traceback, find the specific line, and fix that. Do not rewrite "
            "unrelated parts of the file.")


class Runner:
    """Runs candidate code safely. One instance per agent run."""

    def __init__(self, timeout_s: int = 600, workdir: pathlib.Path = None):
        self.timeout_s = timeout_s
        self.work = workdir or WORK
        self.work.mkdir(parents=True, exist_ok=True)

    def run_code(self, code: str, iteration: int, seed: int = 0) -> RunResult:
        d = self.work / f"iter_{iteration:03d}_seed{seed}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        cand = d / "candidate.py"
        cand.write_text(code)

        t0 = time.time()
        timed_out = killed = False
        try:
            p = subprocess.run(
                [PY, str(HERE / "worker.py"), str(cand), str(d), str(seed)],
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            out, err, rc = p.stdout, p.stderr, p.returncode
            killed = rc < 0                                 # signal (e.g. OOM kill)
        except subprocess.TimeoutExpired as e:
            timed_out = True
            out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        secs = time.time() - t0

        rj = d / "result.json"
        res = json.loads(rj.read_text()) if rj.exists() else {}

        if res.get("ok") and (d / "scores.npy").exists():
            return RunResult(ok=True, iteration=iteration, seed=seed, secs=secs,
                             scores=np.load(d / "scores.npy"),
                             history=res.get("history", []),
                             stdout=out[-3000:], stderr=err[-3000:])

        fclass, recovery = classify(res, err, timed_out, killed)
        return RunResult(
            ok=False, iteration=iteration, seed=seed, secs=secs,
            error_type=res.get("error_type") or ("Timeout" if timed_out else "ProcessFailure"),
            error=res.get("error") or (f"exceeded {self.timeout_s}s" if timed_out else err[-500:]),
            traceback=res.get("traceback"), failure_class=fclass, recovery=recovery,
            stdout=out[-3000:], stderr=err[-3000:],
        )

    def run_file(self, path: str | pathlib.Path, iteration: int, seed: int = 0) -> RunResult:
        return self.run_code(pathlib.Path(path).read_text(), iteration, seed)


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    r = Runner(timeout_s=120)

    print("=== 1. valid candidate (random scores) ===")
    ok_code = '''
import numpy as np
def run(D, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(len(D["yva"])), [{"epoch": 1, "train_loss": 0.7, "valid_primary": 0.48}]
'''
    print("  ", r.run_code(ok_code, 900).brief())

    print("=== 2. syntax error (recovery ladder) ===")
    print("  ", r.run_code("def run(D, seed=0)\n    return 1", 901).brief())

    print("=== 3. wrong output length (alignment guard) ===")
    print("  ", r.run_code(
        "import numpy as np\ndef run(D, seed=0):\n    return np.zeros(10), []", 902).brief())

    print("=== 4. NaN scores (health guard) ===")
    print("  ", r.run_code(
        "import numpy as np\ndef run(D, seed=0):\n"
        "    s = np.zeros(len(D['yva'])); s[0] = np.nan; return s, []", 903).brief())

    print("=== 5. timeout ===")
    r2 = Runner(timeout_s=5)
    print("  ", r2.run_code("import time\ndef run(D, seed=0):\n    time.sleep(60)", 904).brief())

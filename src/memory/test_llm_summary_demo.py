"""
test_llm_summary_demo.py
--------------------------
Feeds a handful of REALISTIC fake iterations through MemoryStore +
LLMClient, so you can see exactly what kind of feedback the LLM
analysis gives.

IMPORTANT: each sample below is the FULL training script for that
iteration - this matches the realistic scenario where the agent
architecture hands you a complete file every round, not a diff. The
memory module itself (via `memory.compute_code_diff`) is responsible
for turning "full code every time" into "full code once, diffs after
that" - you don't need to do that conversion yourself.

Run this to validate:
    python3 test_llm_summary_demo.py

Behavior depends on your environment:
  - If GOOGLE_API_KEY is set (directly or via a .env file) AND you have
    network access to Google's API:
        -> makes real Gemini calls, shows real analysis
  - If the key is missing OR the network call fails:
        -> automatically falls back to the rule-based summary
           (this is expected and fine - it proves the fallback works)

Either way, the script runs top to bottom without crashing - that's the
main thing to verify first. Then read the printed output to judge
whether the LLM's summaries/reasons are actually useful text, AND check
that iteration 1 stores the FULL script while iterations 2+ store only
a short diff (printed clearly at the end of each iteration).
"""

from records import IterationRecord, Metrics, ResourceUsage, FailureType
from memory_store import MemoryStore
from llm_client import LLMClient


# ---------------------------------------------------------------------
# Sample fake iterations - each one is the FULL script for that round,
# just like a real agent would hand over. Scores are hand-picked to
# tell a story: improve, improve, improve, DROP (overfit), recover.
# ---------------------------------------------------------------------

FULL_CODE_V1 = '''
import numpy as np

class FactorizationMachine:
    def __init__(self, n_features, k=16, lr=0.001):
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.01, (n_features, k))
        self.lr = lr

    def predict(self, x):
        linear = self.w0 + np.dot(self.w, x)
        interaction = 0.5 * np.sum(
            np.dot(x, self.V) ** 2 - np.dot(x ** 2, self.V ** 2)
        )
        return linear + interaction

    def fit(self, X, y, epochs=5):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = pred - yi
                self.w0 -= self.lr * error
                self.w -= self.lr * error * xi
'''

FULL_CODE_V2 = '''
import numpy as np

class FactorizationMachine:
    def __init__(self, n_features, k=16, lr=0.001):
        self.w0 = 0.0
        self.w = np.zeros(n_features)
        self.V = np.random.normal(0, 0.01, (n_features, k))
        self.lr = lr

    def predict(self, x):
        linear = self.w0 + np.dot(self.w, x)
        interaction = 0.5 * np.sum(
            np.dot(x, self.V) ** 2 - np.dot(x ** 2, self.V ** 2)
        )
        return linear + interaction

    def fit(self, X, y, epochs=5):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = pred - yi
                self.w0 -= self.lr * error
                self.w -= self.lr * error * xi


def add_avg_watch_time_feature(df, train_df):
    """New: user's historical average watch_time as a feature."""
    user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
    df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
    return df
'''

FULL_CODE_V3 = '''
import numpy as np
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    """Replaces the plain FM with a neural net on top of the same embeddings."""
    def __init__(self, num_features, embed_dim=16, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embed_dim)
        self.fm_linear = nn.Linear(num_features, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        fm_out = self.fm_linear(x.float())
        deep_out = self.mlp(emb.mean(dim=1))
        return torch.sigmoid(fm_out + deep_out)


def add_avg_watch_time_feature(df, train_df):
    user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
    df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
    return df
'''

FULL_CODE_V4 = '''
import numpy as np
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, num_features, embed_dim=64, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embed_dim)  # 64, was 16
        self.fm_linear = nn.Linear(num_features, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        fm_out = self.fm_linear(x.float())
        deep_out = self.mlp(emb.mean(dim=1))
        return torch.sigmoid(fm_out + deep_out)


def add_avg_watch_time_feature(df, train_df):
    user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
    df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
    return df
'''

FULL_CODE_V5 = '''
import numpy as np
import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, num_features, embed_dim=16, hidden=64):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embed_dim)  # reverted to 16
        self.fm_linear = nn.Linear(num_features, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        emb = self.embedding(x)
        fm_out = self.fm_linear(x.float())
        deep_out = self.mlp(emb.mean(dim=1))
        return torch.sigmoid(fm_out + deep_out)


def add_avg_watch_time_feature(df, train_df):
    user_avg = train_df.groupby('user_id')['play_time_ms'].mean()
    df['user_avg_watch_time'] = df['user_id'].map(user_avg).fillna(user_avg.mean())
    return df
'''


# Each tuple: (hypothesis, stage, full_code_for_this_iteration, primary_score)
SAMPLE_ITERATIONS = [
    (
        # Hypothesis (applied to current code alr)
        "Reproduce the official baseline FM to confirm the pipeline works end-to-end.",
        # Stage
        "model",
        # Code
        FULL_CODE_V1,
        # Simulated primary validation score.
        0.5946,
    ),
    (
        "Add user's historical average watch_time as a feature, since "
        "long_view likely correlates with past viewing behavior.",
        "features",
        FULL_CODE_V2,
        0.6010,
    ),
    (
        "Replace the linear FM with a DeepFM (embeddings + MLP) to "
        "capture nonlinear feature interactions.",
        "model",
        FULL_CODE_V3,
        0.6048,
    ),
    (
        "Increase embedding dimension from 16 to 64, expecting more "
        "capacity to help the model fit richer interactions.",
        "model",
        FULL_CODE_V4,
        0.5958,  # DROP - simulates overfitting from too much capacity
    ),
    (
        "Revert embedding dimension back to 16, add dropout(0.2) on the "
        "MLP layers instead to regularize without losing capacity.",
        "training",
        FULL_CODE_V5,
        0.6103,
    ),
]


def run_test():
    memory = MemoryStore(log_path="test_demo_run_log.jsonl")
    llm = LLMClient()  # reads GOOGLE_API_KEY from environment / .env automatically

    if not llm.api_key:
        print("NOTE: No GOOGLE_API_KEY found in environment.")
        print("      This run will use the rule-based FALLBACK summaries.")
        print("      To test real LLM output: set it in a .env file (see README).\n")
    else:
        print("GOOGLE_API_KEY found - will attempt real Gemini calls.\n")

    for i, (hypothesis, stage, full_code, score) in enumerate(SAMPLE_ITERATIONS, start=1):
        # This is the key step: hand over the FULL code, get back either
        # the full code (iteration 1) or a compact diff (iteration 2+).
        stored_code = memory.compute_code_diff(full_code.strip())

        record = IterationRecord(
            iteration=i,
            hypothesis=hypothesis,
            stage=stage,
            code_diff=stored_code,
            metrics=Metrics(gauc=score + 0.06, ndcg5=score - 0.06, primary=score),
            failure=FailureType.NONE,
            resource_usage=ResourceUsage(wall_clock_sec=1.0),
        )

        memory.add_with_analysis(record, llm)

        # every_n=3 here (not the default 10) just so this small 5-iteration
        # demo actually triggers consolidation - in the real run, use a
        # larger every_n (e.g. 10) matched to your iteration budget.
        if memory.consolidate_if_needed(llm, every_n=3):
            print(f">>> Consolidation triggered after iteration {i} <<<")
            print(memory.distilled_notes)
            print()

        print(f"--- Iteration {i} [{stage}] ---")
        print(f"Hypothesis:        {hypothesis}")
        print(f"Score:             {score:.4f}")
        print(f"Full code length:  {len(full_code.strip())} chars")
        print(f"Stored code length:{len(stored_code)} chars "
              f"({'FULL CODE (first iteration)' if i == 1 else 'DIFF ONLY'})")
        print(f"LLM summary:       {record.code_summary}")
        print(f"Likely reason:     {record.likely_reason}")
        print()

    print("=" * 60)
    print("DISTILLED NOTES (built by consolidate_if_needed):")
    print("=" * 60)
    print(memory.distilled_notes or "(none - not enough iterations to trigger consolidation)")
    print()

    print("=" * 60)
    print("COMPRESSED CONTEXT the agent would see on the NEXT iteration:")
    print("=" * 60)
    print(memory.get_prompt_context())

    memory.export_run_log_markdown("test_demo_run_log.md")
    print("\nFull report written to test_demo_run_log.md")


if __name__ == "__main__":
    run_test()
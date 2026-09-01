"""Compatibility shim: `from features import load_features` also works.

The tools are named build_features and load_features, so generated code
reasonably infers a module called `features`. All three drafts of one run failed
with `No module named 'features'` before writing a single line of model code -
the same wrong guess three times, which makes it a naming trap in our API rather
than a mistake on the model's part.

The real implementations live in tools.py (features) and models.py (training).
This module re-exports them so either import path works.
"""
from __future__ import annotations

from tools import (                     # noqa: F401
    build_features,
    load_features,
    materialise_test,
)

try:                                    # optional: training helpers
    from models import (                # noqa: F401
        train,
        train_model,
        blend,
        load_scores,
        list_predictions,
        recall,
    )
except Exception:                       # models pulls in the boosting libraries;
    pass                                # a torch-only candidate must not fail here

__all__ = ["build_features", "load_features", "materialise_test",
           "train", "train_model", "blend", "load_scores",
           "list_predictions", "recall"]

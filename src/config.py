"""Shared paths, constants and model configuration."""

import json
from pathlib import Path

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.15

# Hand-set baseline, used until src/10_tune_challenger.py has been run. Chosen
# to be conservative on noisy 13%-event-rate credit data: a high leaf floor
# stops the model carving out small high-default pockets that will not reappear
# in a later vintage.
_BASELINE = dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=50, reg_lambda=2.0)


def xgb_params():
    """Tuned parameters if the search has been run, baseline otherwise.

    The challenger and every ablation variant read from here, so the variants
    can only differ in features - which is what makes the ablation a valid
    attribution rather than a comparison of two things at once.
    """
    f = MODELS / "xgb_params.json"
    tuned = json.load(open(f))["tuned"] if f.exists() else _BASELINE
    return dict(n_estimators=3000, eval_metric="auc",
                early_stopping_rounds=50, enable_categorical=True,
                tree_method="hist", random_state=RANDOM_STATE,
                n_jobs=-1, **tuned)


def param_summary(params):
    """One-line record of the parameters a run used, for the console dump."""
    keys = ("max_depth", "learning_rate", "min_child_weight", "subsample",
            "colsample_bytree", "reg_lambda")
    return ", ".join(f"{k}={params[k]}" for k in keys if k in params)

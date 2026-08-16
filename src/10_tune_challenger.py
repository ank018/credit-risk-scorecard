"""
Project 1 - Credit Risk Scorecard
Step 8c: Optuna hyperparameter search for the XGBoost challenger.

Run:  python src/10_tune_challenger.py
In:   data/processed/abt.parquet
Out:  models/xgb_params.json
      reports/optuna_trials.csv
      reports/figures/optuna_history.png

The search optimises AUC on the *validation* slice - the same 15% carved out of
train for early stopping in src/08_challenger.py. Test and out-of-time are never
touched here. Tuning against either would select hyperparameters by the
performance of the comparison itself, which is the same error the early-stopping
fix corrected.

Note that the resulting parameters are tuned on the 2013-14 development
population. Whether they remain optimal on the 2015 window is not something the
search can know, and the gap between validation and out-of-time gain is itself
worth reading.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.15
N_TRIALS = 50

NON_FEATURES = ["id", "issue_d", "loan_status", "vintage", "split", "target",
                "grade", "sub_grade", "int_rate", "term"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state",
               "initial_list_status", "application_type"]

# Hand-set baseline from src/08_challenger.py, for comparison. reg_alpha and
# gamma are nominally 0 there; 1e-4 is the log-scale equivalent, small enough
# to be functionally identical.
BASELINE = dict(max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=50, reg_lambda=2.0,
                reg_alpha=1e-4, gamma=1e-4)

FIXED = dict(n_estimators=3000, eval_metric="auc", early_stopping_rounds=50,
             enable_categorical=True, tree_method="hist",
             random_state=RANDOM_STATE, n_jobs=-1)


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def build_splits():
    df = pd.read_parquet(PROCESSED / "abt.parquet")
    df = df[df["split"].isin(["dev", "oot"])].copy()
    df["target"] = df["target"].astype(int)
    df = derive_features(df)

    features = [c for c in df.columns if c not in NON_FEATURES]
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")

    dev = df[df["split"] == "dev"]
    oot = df[df["split"] == "oot"].copy()
    train, test = train_test_split(dev, test_size=TEST_SIZE,
                                   stratify=dev["target"],
                                   random_state=RANDOM_STATE)
    train_fit, val = train_test_split(train, test_size=VAL_SIZE,
                                      stratify=train["target"],
                                      random_state=RANDOM_STATE)
    return train_fit.copy(), val.copy(), test.copy(), oot, features


def fit_and_score(params, train_fit, val, features, eval_on=None):
    m = xgb.XGBClassifier(**{**FIXED, **params})
    m.fit(train_fit[features], train_fit["target"],
          eval_set=[(val[features], val["target"])], verbose=False)
    target = eval_on if eval_on is not None else val
    p = m.predict_proba(target[features])[:, 1]
    return m, roc_auc_score(target["target"], p)


def objective(trial, train_fit, val, features):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1,
                                             log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # Wide floor range: credit data at a 13% event rate punishes small
        # leaves, and the search should be free to say so.
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 200,
                                              log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
    }
    _, auc = fit_and_score(params, train_fit, val, features)
    return auc


if __name__ == "__main__":
    train_fit, val, test, oot, features = build_splits()
    print(f"tuning on {len(train_fit):,} train_fit / {len(val):,} val, "
          f"{len(features)} features")
    print(f"test ({len(test):,}) and oot ({len(oot):,}) are untouched\n")

    # Baseline for reference.
    m0, auc0 = fit_and_score(BASELINE, train_fit, val, features)
    print(f"hand-set baseline: val Gini {2*auc0-1:.4f} "
          f"(iter {m0.best_iteration})")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.enqueue_trial(BASELINE)   # start from the hand-set point

    print(f"\nrunning {N_TRIALS} trials...")
    study.optimize(lambda t: objective(t, train_fit, val, features),
                   n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_params
    print(f"\nbest val Gini: {2*study.best_value-1:.4f} "
          f"(trial {study.best_trial.number})")
    print("\nbest parameters:")
    for k, v in best.items():
        b = BASELINE.get(k)
        note = f"   (baseline {b})" if b is not None else ""
        print(f"  {k:<20} {v}{note}")

    trials = study.trials_dataframe()
    trials.to_csv(REPORTS / "optuna_trials.csv", index=False)

    # Honest comparison on the untouched windows, baseline vs tuned.
    print("\nheld-out comparison (never seen during the search):")
    rows = []
    for label, params in [("hand-set", BASELINE), ("tuned", best)]:
        m = xgb.XGBClassifier(**{**FIXED, **params})
        m.fit(train_fit[features], train_fit["target"],
              eval_set=[(val[features], val["target"])], verbose=False)
        g_train = gini(train_fit["target"],
                       m.predict_proba(train_fit[features])[:, 1])
        g_val = gini(val["target"], m.predict_proba(val[features])[:, 1])
        g_test = gini(test["target"], m.predict_proba(test[features])[:, 1])
        g_oot = gini(oot["target"], m.predict_proba(oot[features])[:, 1])
        rows.append({"params": label, "best_iter": m.best_iteration,
                     "train_gini": g_train, "val_gini": g_val,
                     "test_gini": g_test, "oot_gini": g_oot,
                     "train_minus_test": g_train - g_test})
    r = pd.DataFrame(rows)
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    d_val = r.loc[1, "val_gini"] - r.loc[0, "val_gini"]
    d_oot = r.loc[1, "oot_gini"] - r.loc[0, "oot_gini"]
    print(f"\n  val gain  {d_val:+.4f}   oot gain {d_oot:+.4f}")
    print("  If the out-of-time gain is much smaller than the validation gain,")
    print("  the search has fitted the 2013-14 population rather than found")
    print("  parameters that generalise to a later vintage.")

    with open(MODELS / "xgb_params.json", "w") as f:
        json.dump({"tuned": best, "baseline": BASELINE,
                   "n_trials": N_TRIALS,
                   "best_val_gini": 2 * study.best_value - 1}, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    vals = [2 * t.value - 1 for t in study.trials if t.value is not None]
    ax[0].plot(vals, ".", alpha=0.6)
    ax[0].plot(np.maximum.accumulate(vals), lw=2, label="best so far")
    ax[0].axhline(2 * auc0 - 1, ls="--", c="r", lw=1, label="hand-set baseline")
    ax[0].set_xlabel("trial")
    ax[0].set_ylabel("validation Gini")
    ax[0].set_title("Search history")
    ax[0].legend()

    imp = optuna.importance.get_param_importances(study)
    ax[1].barh(list(imp.keys())[::-1], list(imp.values())[::-1])
    ax[1].set_xlabel("importance")
    ax[1].set_title("Which hyperparameters mattered")
    plt.tight_layout()
    plt.savefig(FIGS / "optuna_history.png", dpi=140)

    print("\nsaved models/xgb_params.json")

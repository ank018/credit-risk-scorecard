"""
Project 1 - Credit Risk Scorecard
Step 8b: ablation - decomposing the challenger's lift.

Run:  python src/09_ablation.py
In:   data/processed/abt.parquet, models/{binning_process,logit_model}.pkl
Out:  reports/ablation.csv
      reports/figures/ablation.png

Five XGBoost variants, all identically parameterised, fitted on the same rows and
evaluated on the same held-out windows. Each removes one thing, so the difference
between two adjacent rows attributes lift to a specific cause:

  A  full 35 features                 - the headline challenger
  B  minus addr_state                 - isolates geography
  C  minus addr_state, purpose        - the plausibly fair-lending-safe variant
  D  champion's 12 characteristics    - isolates functional form: same inputs as
                                        the scorecard, non-linearity and
                                        interactions allowed
  E  champion's 12 + loan_amnt        - tests the specific IV-screening failure
                                        identified by SHAP

Read D against the champion: any gap there is what the scorecard gives up by
being additive and linear in WOE, using no extra information at all.
"""

from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.15

NON_FEATURES = ["id", "issue_d", "loan_status", "vintage", "split", "target",
                "grade", "sub_grade", "int_rate", "term"]
CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state",
               "initial_list_status", "application_type"]

PARAMS = dict(
    n_estimators=2000, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=50,
    reg_lambda=2.0, eval_metric="auc", early_stopping_rounds=50,
    enable_categorical=True, tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1,
)


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


def champion_reference(splits):
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    model, feats = obj["model"], obj["features"]

    out = {}
    for name, d in splits.items():
        w = bp.transform(d[bp.variable_names], metric="woe")
        w.columns = [f"woe_{c}" for c in w.columns]
        X = sm.add_constant(w[feats].astype(float), has_constant="add")
        out[name] = model.predict(X).values
    champ_feats = [f.replace("woe_", "") for f in feats]
    return out, champ_feats


def run_variant(label, cols, train_fit, val, test, oot):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(train_fit[cols], train_fit["target"],
          eval_set=[(val[cols], val["target"])], verbose=False)
    g_test = gini(test["target"], m.predict_proba(test[cols])[:, 1])
    g_oot = gini(oot["target"], m.predict_proba(oot[cols])[:, 1])
    print(f"  {label:<38} n_feat {len(cols):>2}  "
          f"iter {m.best_iteration:>4}  test {g_test:.4f}  oot {g_oot:.4f}")
    return {"variant": label, "n_features": len(cols),
            "best_iteration": m.best_iteration,
            "test_gini": g_test, "oot_gini": g_oot}


if __name__ == "__main__":
    train_fit, val, test, oot, all_features = build_splits()
    splits = {"train_fit": train_fit, "val": val, "test": test, "oot": oot}

    champ, champ_feats = champion_reference(splits)
    champ_test = gini(test["target"], champ["test"])
    champ_oot = gini(oot["target"], champ["oot"])
    print(f"champion (scorecard)                   n_feat 12"
          f"              test {champ_test:.4f}  oot {champ_oot:.4f}\n")

    variants = {
        "A. full challenger": all_features,
        "B. minus addr_state": [f for f in all_features if f != "addr_state"],
        "C. minus addr_state, purpose": [f for f in all_features
                                         if f not in ("addr_state", "purpose")],
        "D. champion characteristics only": champ_feats,
        "E. champion + loan_amnt": champ_feats + ["loan_amnt"],
    }

    rows = [run_variant(k, v, train_fit, val, test, oot)
            for k, v in variants.items()]

    r = pd.DataFrame(rows)
    r["lift_vs_champion"] = r["oot_gini"] - champ_oot
    full_lift = r.loc[0, "lift_vs_champion"]
    r["pct_of_full_lift"] = r["lift_vs_champion"] / full_lift

    r.to_csv(REPORTS / "ablation.csv", index=False)

    print("\nablation, out-of-time:")
    print(r[["variant", "n_features", "oot_gini", "lift_vs_champion",
             "pct_of_full_lift"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nattribution:")
    a, b, c, d, e = r["oot_gini"]
    print(f"  geography (A - B):                    {a - b:+.4f}")
    print(f"  geography + purpose (A - C):          {a - c:+.4f}")
    print(f"  fair-lending-safe lift (C - champion):{c - champ_oot:+.4f}"
          f"  = {(c - champ_oot) / full_lift:.0%} of headline")
    print(f"  functional form alone (D - champion): {d - champ_oot:+.4f}"
          f"  = {(d - champ_oot) / full_lift:.0%} of headline")
    print(f"  adding loan_amnt (E - D):             {e - d:+.4f}")
    print(f"  other discarded features (A - E):     {a - e:+.4f}")

    plt.figure(figsize=(9, 5))
    labels = ["champion"] + list(r["variant"])
    vals = [champ_oot] + list(r["oot_gini"])
    colors = ["tab:grey"] + ["tab:blue"] * len(r)
    plt.barh(range(len(vals)), vals, color=colors)
    plt.yticks(range(len(vals)), labels)
    plt.axvline(champ_oot, ls="--", c="k", lw=1)
    plt.xlabel("out-of-time Gini")
    plt.title("Ablation: where the challenger's lift comes from")
    plt.gca().invert_yaxis()
    plt.xlim(0.25, max(vals) * 1.05)
    plt.tight_layout()
    plt.savefig(FIGS / "ablation.png", dpi=140)
    print("\nwrote reports/ablation.csv and figures/ablation.png")

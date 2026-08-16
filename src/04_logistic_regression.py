"""
Project 1 - Credit Risk Scorecard
Step 4: logistic regression champion on WOE features, with coefficient sign check.

Run:  python src/04_logistic_regression.py
In:   data/processed/woe_{train,test,oot}.parquet, models/selected_features.json
Out:  models/logit_model.pkl
      reports/coefficients.csv, reports/sign_check.csv
      data/processed/pd_{train,test,oot}.parquet
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
MODELS = Path("models")

# Wald test significance level for retaining a characteristic.
ALPHA = 0.05


def load():
    with open(MODELS / "selected_features.json") as f:
        features = json.load(f)["features"]
    frames = {
        name: pd.read_parquet(PROCESSED / f"woe_{name}.parquet")
        for name in ("train", "test", "oot")
    }
    return features, frames


def univariate_direction(train, features):
    """Each feature's standalone relationship with default, before the model sees
    any of them together. This is the reference the multivariate coefficients get
    checked against."""
    dirs = {}
    for f in features:
        r = np.corrcoef(train[f].astype(float), train["target"])[0, 1]
        dirs[f] = np.sign(r)
    return dirs


def fit(train, features):
    X = sm.add_constant(train[features].astype(float))
    y = train["target"]
    model = sm.Logit(y, X).fit(disp=0)
    return model


def sign_check(model, dirs, features):
    """A characteristic that predicts one way on its own and the opposite way
    inside the model has had its effect redistributed by correlation with other
    features. Regardless of how well the model fits, a coefficient pointing
    against its own univariate direction cannot be defended to a credit committee
    and is a rejection at model review."""
    rows = []
    for f in features:
        coef = model.params[f]
        expected = dirs[f]
        actual = np.sign(coef)
        rows.append({
            "feature": f.replace("woe_", ""),
            "coefficient": coef,
            "std_err": model.bse[f],
            "p_value": model.pvalues[f],
            "univariate_dir": "+" if expected > 0 else "-",
            "model_dir": "+" if actual > 0 else "-",
            "sign_ok": expected == actual,
            "significant": model.pvalues[f] < ALPHA,
        })
    return pd.DataFrame(rows)


def report(check, model):
    print("\n  coefficient table:")
    show = check.copy()
    show["flag"] = np.where(~show["sign_ok"], "SIGN FLIP",
                     np.where(~show["significant"], f"p >= {ALPHA}", ""))
    print(show[["feature", "coefficient", "std_err", "p_value",
                "univariate_dir", "model_dir", "flag"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    flips = check.loc[~check["sign_ok"], "feature"].tolist()
    insig = check.loc[~check["significant"], "feature"].tolist()

    print(f"\n  intercept: {model.params['const']:.4f}")
    print(f"  pseudo R-sq (McFadden): {model.prsquared:.4f}")
    print(f"  log-likelihood: {model.llf:,.1f}")

    if flips:
        print(f"\n  !! SIGN FLIPS: {flips}")
        print("     Drop these and refit. A wrong-signed coefficient fails model")
        print("     review no matter what the AUC says.")
    else:
        print("\n  all coefficients agree with their univariate direction")

    if insig:
        print(f"\n  not significant at {ALPHA}: {insig}")
    return flips, insig


def evaluate(model, frames, features):
    print("\n  discrimination (AUC / Gini):")
    preds = {}
    for name, d in frames.items():
        X = sm.add_constant(d[features].astype(float), has_constant="add")
        p = model.predict(X)
        auc = roc_auc_score(d["target"], p)
        print(f"    {name:<6} AUC {auc:.4f}   Gini {2 * auc - 1:.4f}")
        out = pd.DataFrame({"pd": p.values, "target": d["target"].values,
                            "issue_d": d["issue_d"].values})
        out.to_parquet(PROCESSED / f"pd_{name}.parquet", index=False)
        preds[name] = out
    return preds


if __name__ == "__main__":
    features, frames = load()
    train = frames["train"]
    print(f"fitting on {len(train):,} accounts, {len(features)} characteristics")

    dirs = univariate_direction(train, features)
    model = fit(train, features)
    check = sign_check(model, dirs, features)
    flips, insig = report(check, model)

    # Refit without failures. Sign flips go first - they are a hard gate.
    # Insignificant characteristics are dropped as a second pass so the removal
    # of a flipped feature has a chance to restore significance elsewhere.
    drop = [f"woe_{f}" for f in flips]
    if drop:
        features = [f for f in features if f not in drop]
        print(f"\n  refitting without {len(drop)} flipped characteristic(s)...")
        dirs = univariate_direction(train, features)
        model = fit(train, features)
        check = sign_check(model, dirs, features)
        flips, insig = report(check, model)

    drop = [f"woe_{f}" for f in insig]
    if drop:
        features = [f for f in features if f not in drop]
        print(f"\n  refitting without {len(drop)} insignificant characteristic(s)...")
        model = fit(train, features)
        check = sign_check(model, univariate_direction(train, features), features)
        flips, insig = report(check, model)

    check.to_csv(REPORTS / "coefficients.csv", index=False)
    evaluate(model, frames, features)

    with open(MODELS / "logit_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features}, f)
    print(f"\nsaved models/logit_model.pkl  ({len(features)} characteristics)")

"""
Project 1 - Credit Risk Scorecard
Step 8: XGBoost challenger on raw features, SHAP attribution, champion comparison.

Run:  python src/08_challenger.py
In:   data/processed/abt.parquet, models/{binning_process,logit_model,scorecard}.pkl
Out:  models/xgb_model.pkl
      reports/challenger_comparison.csv, reports/shap_importance.csv
      reports/figures/{shap_beeswarm,shap_bar,roc_champion_challenger,
                       calibration_challenger}.png

The challenger is given the full 35-candidate feature set on raw values - not the
12 WOE-transformed characteristics the champion uses. Handing it the champion's
pre-screened, pre-binned inputs would not be a fair contest: the point is to find
out what a modern model does with everything the IV screen discarded.

Lending Club's own grade / sub_grade / int_rate are withheld from both models.
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
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features
from config import xgb_params, param_summary

warnings.filterwarnings("ignore", category=UserWarning)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_SIZE = 0.15        # carved out of train, for early stopping only
SHAP_SAMPLE = 8000     # beeswarm on the full OOT set is slow and unreadable

NON_FEATURES = ["id", "issue_d", "loan_status", "vintage", "split", "target",
                "grade", "sub_grade", "int_rate", "term"]

CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state",
               "initial_list_status", "application_type"]



def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def ks_statistic(y, p):
    order = np.argsort(p)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return np.abs(cum_bad - cum_good).max()


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
    # Early stopping needs a sample the model has not fitted on. Using `test`
    # here would select the number of trees by test performance, making the
    # reported test Gini optimistic - and unfair to the champion, which never
    # saw test at any point.
    train_fit, val = train_test_split(train, test_size=VAL_SIZE,
                                      stratify=train["target"],
                                      random_state=RANDOM_STATE)
    return train_fit.copy(), val.copy(), test.copy(), oot, features


def champion_predictions(splits, features_raw):
    """Re-score the champion on the same rows so the comparison is like for like."""
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
    return out


def fit_challenger(train_fit, val, features):
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(train_fit[features], train_fit["target"],
          eval_set=[(val[features], val["target"])], verbose=False)
    print(f"  best iteration: {m.best_iteration} of {PARAMS['n_estimators']}")
    return m


def compare(splits, champ, chal_pred):
    rows = []
    for name, d in splits.items():
        y = d["target"].values
        for label, p in [("champion (scorecard)", champ[name]),
                         ("challenger (XGBoost)", chal_pred[name])]:
            rows.append({
                "split": name, "model": label,
                "gini": gini(y, p), "ks": ks_statistic(y, p),
                "brier": brier_score_loss(y, p),
                "mean_pred": p.mean(), "observed": y.mean(),
                "cal_gap": y.mean() - p.mean(),
            })
    r = pd.DataFrame(rows)
    r.to_csv(REPORTS / "challenger_comparison.csv", index=False)
    print("\nchampion vs challenger:")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n  train_fit is in-sample for the challenger and val selected its")
    print("  tree count, so neither is a clean estimate. The champion was fitted")
    print("  on train_fit + val, so both are in-sample for it too.")
    print("  test and oot are held out for both models - quote those.")

    oot = r[r["split"] == "oot"].set_index("model")
    lift = (oot.loc["challenger (XGBoost)", "gini"]
            - oot.loc["champion (scorecard)", "gini"])
    print(f"\n  OOT Gini lift from the challenger: {lift:+.4f}")
    return r, lift


def shap_analysis(model, oot, features):
    sample = oot.sample(min(SHAP_SAMPLE, len(oot)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample[features])

    imp = (pd.DataFrame({"feature": features,
                         "mean_abs_shap": np.abs(values).mean(axis=0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp.to_csv(REPORTS / "shap_importance.csv", index=False)

    print("\n  top 15 characteristics by mean |SHAP| (OOT):")
    print(imp.head(15).to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    shap.summary_plot(values, sample[features], show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(FIGS / "shap_beeswarm.png", dpi=140, bbox_inches="tight")
    plt.close()

    shap.summary_plot(values, sample[features], plot_type="bar", show=False,
                      max_display=20)
    plt.tight_layout()
    plt.savefig(FIGS / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()
    return imp, values, sample


def overlap_with_champion(imp):
    """How much of the challenger's signal comes from characteristics the
    champion already uses. A large overlap means the lift is coming from
    interactions and non-linearity rather than from new information."""
    with open(MODELS / "logit_model.pkl", "rb") as f:
        champ_feats = [f.replace("woe_", "")
                       for f in pickle.load(f)["features"]]
    top15 = imp.head(15)["feature"].tolist()
    shared = [f for f in top15 if f in champ_feats]
    new = [f for f in top15 if f not in champ_feats]
    total = imp["mean_abs_shap"].sum()
    share = imp[imp["feature"].isin(champ_feats)]["mean_abs_shap"].sum() / total

    print(f"\n  of the challenger's top 15 drivers:")
    print(f"    {len(shared)} are on the scorecard: {shared}")
    print(f"    {len(new)} are not: {new}")
    print(f"  scorecard characteristics account for {share:.1%} of total |SHAP|")
    return shared, new, share


def plot_roc(splits, champ, chal_pred):
    y = splits["oot"]["target"].values
    plt.figure(figsize=(6, 6))
    for label, p in [("champion (scorecard)", champ["oot"]),
                     ("challenger (XGBoost)", chal_pred["oot"])]:
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, lw=1.5, label=f"{label}  Gini {gini(y, p):.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Out-of-time ROC: champion vs challenger")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "roc_champion_challenger.png", dpi=140)
    plt.close()


def plot_calibration(splits, champ, chal_pred):
    y = splits["oot"]["target"].values
    plt.figure(figsize=(6, 6))
    for label, p in [("champion", champ["oot"]), ("challenger", chal_pred["oot"])]:
        q = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame({"q": q, "p": p, "y": y}).groupby("q").mean()
        plt.plot(g["p"], g["y"], "o-", label=label)
    lim = [0, max(champ["oot"].max(), chal_pred["oot"].max()) * 1.05]
    plt.plot(lim, lim, "k--", lw=0.8, label="perfect")
    plt.xlabel("mean predicted PD")
    plt.ylabel("observed default rate")
    plt.title("Out-of-time calibration: champion vs challenger")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "calibration_challenger.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    train_fit, val, test, oot, features = build_splits()
    splits = {"train_fit": train_fit, "val": val, "test": test, "oot": oot}

    PARAMS = xgb_params()
    print(f"xgb params: {param_summary(PARAMS)}")
    print(f"challenger feature set: {len(features)} raw characteristics")
    print(f"  (champion uses 12 WOE-transformed)")

    print("\nfitting XGBoost...")
    model = fit_challenger(train_fit, val, features)

    chal_pred = {n: model.predict_proba(d[features])[:, 1]
                 for n, d in splits.items()}
    champ = champion_predictions(splits, features)

    comparison, lift = compare(splits, champ, chal_pred)

    print("\nSHAP attribution...")
    imp, values, sample = shap_analysis(model, oot, features)
    overlap_with_champion(imp)

    plot_roc(splits, champ, chal_pred)
    plot_calibration(splits, champ, chal_pred)

    with open(MODELS / "xgb_model.pkl", "wb") as f:
        pickle.dump({"model": model, "features": features,
                     "categorical": CATEGORICAL}, f)
    print("\nsaved models/xgb_model.pkl")
    print("wrote figures to reports/figures/")

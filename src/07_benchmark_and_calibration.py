"""
Project 1 - Credit Risk Scorecard
Step 7: incremental lift over the Lending Club benchmark, and intercept
recalibration for the out-of-time population.

Run:  python src/07_benchmark_and_calibration.py
In:   data/processed/abt.parquet, models/{binning_process,logit_model,scorecard}.pkl
Out:  reports/incremental_lift.csv, reports/recalibration.csv
      reports/figures/calibration_recalibrated.png

The train/test split is reproduced deterministically from the same seed used in
src/02_woe_binning.py, so sub_grade can be joined to each account without
carrying identifiers through every intermediate file.
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

RANDOM_STATE = 42
TEST_SIZE = 0.30


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1


def rebuild():
    """Reproduce train/test/oot with sub_grade attached."""
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    model, features = obj["model"], obj["features"]
    with open(MODELS / "scorecard.pkl", "rb") as f:
        card = pickle.load(f)

    df = pd.read_parquet(PROCESSED / "abt.parquet")
    df = df[df["split"].isin(["dev", "oot"])].copy()
    df["target"] = df["target"].astype(int)
    df = derive_features(df)

    dev = df[df["split"] == "dev"]
    oot = df[df["split"] == "oot"].copy()
    train, test = train_test_split(dev, test_size=TEST_SIZE,
                                   stratify=dev["target"],
                                   random_state=RANDOM_STATE)

    fitted = bp.variable_names
    out = {}
    for name, d in [("train", train.copy()), ("test", test.copy()), ("oot", oot)]:
        w = bp.transform(d[fitted], metric="woe")
        w.columns = [f"woe_{c}" for c in w.columns]
        X = sm.add_constant(w[features].astype(float), has_constant="add")
        d = d.reset_index(drop=True)
        d["pd"] = model.predict(X).values
        d["score"] = score_from_woe(w, features, model, card)
        out[name] = d
    return out, model, features, card


def score_from_woe(w, features, model, card):
    n = len(features)
    intercept = model.params["const"]
    factor, offset = card["factor"], card["offset"]
    total = np.zeros(len(w))
    for f in features:
        total += -(model.params[f] * w[f].astype(float).values
                   + intercept / n) * factor + offset / n
    return np.round(total)


def grade_rank(s):
    codes = sorted(s.dropna().unique())
    return s.map({g: i for i, g in enumerate(codes)})


def incremental_lift(data):
    """Three models fitted on train, all evaluated on the out-of-time window.

    The question is not whether the scorecard beats Lending Club's grade in
    isolation - grade is fitted on their full credit file and, because it sets
    the interest rate, partially causes the outcome it predicts. The question is
    whether the scorecard carries information grade does not already have."""
    train, oot = data["train"], data["oot"]
    for d in (train, oot):
        d["grade_rank"] = grade_rank(d["sub_grade"])
    train = train[train["grade_rank"].notna()]
    oot = oot[oot["grade_rank"].notna()]

    specs = {
        "grade only": ["grade_rank"],
        "scorecard only": ["score"],
        "grade + scorecard": ["grade_rank", "score"],
    }
    rows = []
    for label, cols in specs.items():
        m = sm.Logit(train["target"],
                     sm.add_constant(train[cols].astype(float))).fit(disp=0)
        p = m.predict(sm.add_constant(oot[cols].astype(float), has_constant="add"))
        rows.append({"model": label, "oot_gini": gini(oot["target"], p),
                     "n_oot": len(oot)})
        if label == "grade + scorecard":
            combined = m

    r = pd.DataFrame(rows)
    r["lift_vs_grade"] = r["oot_gini"] - r.loc[0, "oot_gini"]
    print("\nincremental lift (fitted on train, evaluated on OOT):")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n  combined model coefficients:")
    for k in ["grade_rank", "score"]:
        print(f"    {k:<12} {combined.params[k]:+.5f}  p={combined.pvalues[k]:.4g}")
    print("\n  A significant score coefficient alongside grade means the card")
    print("  carries information the incumbent assessment does not.")
    r.to_csv(REPORTS / "incremental_lift.csv", index=False)
    return r


def recalibrate(data, model, features):
    """Shift the intercept so predicted PD matches the observed rate on the new
    population. Discrimination is unchanged by construction - only the level
    moves. This is what a bank does when the rank ordering still holds but the
    through-the-door population has shifted; refitting the whole model would
    discard a card that is still working."""
    oot = data["oot"]
    observed = oot["target"].mean()
    predicted = oot["pd"].mean()

    # Solve for the intercept shift that matches the mean predicted PD to the
    # observed rate.
    logit_p = np.log(oot["pd"] / (1 - oot["pd"]))
    lo, hi = -2.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        m = 1 / (1 + np.exp(-(logit_p + mid)))
        if m.mean() < observed:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    adj = 1 / (1 + np.exp(-(logit_p + shift)))

    rows = [
        {"stage": "before", "mean_pd": predicted, "observed": observed,
         "gap": observed - predicted, "gini": gini(oot["target"], oot["pd"])},
        {"stage": "after", "mean_pd": adj.mean(), "observed": observed,
         "gap": observed - adj.mean(), "gini": gini(oot["target"], adj)},
    ]
    r = pd.DataFrame(rows)
    print("\nintercept recalibration on OOT:")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n  intercept shift: {shift:+.4f}")
    print("  Gini is unchanged - recalibration moves the level, not the ranking.")
    r.to_csv(REPORTS / "recalibration.csv", index=False)

    q = pd.qcut(oot["pd"], 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"q": q, "observed": oot["target"],
                      "before": oot["pd"], "after": adj}).groupby("q").mean()

    plt.figure(figsize=(6, 6))
    plt.plot(g["before"], g["observed"], "o-", label="before")
    plt.plot(g["after"], g["observed"], "s-", label="after recalibration")
    lim = [0, max(g.max()) * 1.05]
    plt.plot(lim, lim, "k--", lw=0.8, label="perfect")
    plt.xlabel("mean predicted PD")
    plt.ylabel("observed default rate")
    plt.title("OOT calibration before and after intercept shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "calibration_recalibrated.png", dpi=140)
    return shift


if __name__ == "__main__":
    data, model, features, card = rebuild()
    for name, d in data.items():
        print(f"  {name:<6} {len(d):>7,}  gini {gini(d['target'], d['pd']):.4f}")

    incremental_lift(data)
    shift = recalibrate(data, model, features)

    with open(MODELS / "recalibration.pkl", "wb") as f:
        pickle.dump({"intercept_shift": shift}, f)
    print("\nsaved models/recalibration.pkl")

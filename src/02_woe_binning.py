"""
Project 1 - Credit Risk Scorecard
Step 2: feature derivation, WOE binning with monotonic constraints, IV screening.

Run:  python src/02_woe_binning.py
In:   data/processed/abt.parquet
Out:  data/processed/woe_train.parquet, woe_test.parquet, woe_oot.parquet
      models/binning_process.pkl
      reports/iv_table.csv
      reports/figures/woe/<feature>.png
"""

from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from optbinning import BinningProcess

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features

warnings.filterwarnings("ignore", category=FutureWarning)

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
WOE_FIGS = REPORTS / "figures" / "woe"
MODELS = Path("models")
for p in (WOE_FIGS, MODELS):
    p.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30

# IV screening thresholds. Below the floor a feature carries too little signal to
# justify a place in a deliberately small scorecard. Above the ceiling, the usual
# cause is leakage rather than a genuinely excellent predictor - flag, inspect,
# do not auto-drop.
IV_FLOOR = 0.02
IV_CEILING = 0.50

# Characteristics retained despite failing the univariate IV floor, on ablation
# evidence rather than judgement. loan_amnt scores IV 0.0115 on its own but was
# measured at +0.0178 Gini in docs/challenger_analysis.md section 4: larger
# loans go to higher-income borrowers, so the risk of the amount cancels
# against the quality of the borrower unless income is held constant.
# Univariate screening cannot see that by construction.
FORCE_KEEP = ["loan_amnt"]

# Binning granularity. A 5% floor per bin and a hard cap of 8 bins keeps WOE
# values stable across the OOT window - thin bins swing on new populations - and
# keeps the resulting points table readable by a credit committee.
MIN_BIN_SIZE = 0.05
MAX_N_BINS = 8

# Not features: identifiers, the outcome, split assignment, and Lending Club's
# own risk grade (held back as a benchmark, see docs/target_definition.md).
NON_FEATURES = ["id", "issue_d", "loan_status", "vintage", "split", "target",
                "grade", "sub_grade", "int_rate", "term"]

CATEGORICAL = ["home_ownership", "verification_status", "purpose", "addr_state",
               "initial_list_status", "application_type"]


def make_splits(df):
    """Random stratified split inside dev; OOT is already time-separated."""
    dev = df[df["split"] == "dev"].copy()
    oot = df[df["split"] == "oot"].copy()

    train, test = train_test_split(
        dev, test_size=TEST_SIZE, stratify=dev["target"], random_state=RANDOM_STATE
    )
    for name, d in [("train", train), ("test", test), ("oot", oot)]:
        print(f"  {name:<6} {len(d):>7,}  bad rate {d['target'].mean():.4f}")
    return train, test, oot


def fit_binning(train, features):
    """Fit on train only. Bin edges are learned parameters - fitting them on
    data used for evaluation leaks the target into the boundaries."""
    categorical = [c for c in CATEGORICAL if c in features]
    numeric = [c for c in features if c not in categorical]

    # auto_asc_desc forces each numeric feature to a single monotonic direction.
    # A scorecard where risk rises, falls, then rises again across a
    # characteristic cannot be explained to a credit committee, and is usually
    # fitting noise in sparse bins.
    fit_params = {v: {"monotonic_trend": "auto_asc_desc"} for v in numeric}

    bp = BinningProcess(
        variable_names=features,
        categorical_variables=categorical,
        binning_fit_params=fit_params,
        min_prebin_size=0.02,
        min_bin_size=MIN_BIN_SIZE,
        max_n_bins=MAX_N_BINS,
        max_n_prebins=20,
    )
    bp.fit(train[features], train["target"])
    return bp


def iv_report(bp, features):
    s = bp.summary()
    s = s[["name", "dtype", "status", "n_bins", "iv", "js", "quality_score"]]
    s = s.sort_values("iv", ascending=False).reset_index(drop=True)

    s["flag"] = np.select(
        [s["iv"] < IV_FLOOR, s["iv"] > IV_CEILING],
        ["WEAK - drop", "HIGH - check leakage"],
        default="",
    )

    forced = s.loc[(s["iv"] < IV_FLOOR) & (s["name"].isin(FORCE_KEEP)),
                   "name"].tolist()
    s.loc[s["name"].isin(forced), "flag"] = "BELOW FLOOR - force kept"
    
    s.to_csv(REPORTS / "iv_table.csv", index=False)

    print("\n  IV ranking:")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    weak = s.loc[(s["iv"] < IV_FLOOR) & (~s["name"].isin(FORCE_KEEP)),
                 "name"].tolist()
    high = s.loc[s["iv"] > IV_CEILING, "name"].tolist()
    if high:
        print(f"\n  !! IV > {IV_CEILING}: {high}")
        print("     Inspect the WOE plot before keeping. A feature this strong is")
        print("     usually post-origination information that survived the filter.")
    if forced:
        print(f"\n  below the {IV_FLOOR} floor but force-kept: {forced}")
        print("     Retained on ablation evidence, not on IV. See FORCE_KEEP.")
    if weak:
        print(f"\n  IV < {IV_FLOOR} (drop): {weak}")
    return s, weak


def plot_woe(bp, features, top_n=20):
    """WOE plots are the artefact you actually read. Check that each trend runs
    in the direction business intuition says it should - higher FICO must mean
    lower risk - and that no bin is carrying an implausible spike."""
    for v in features[:top_n]:
        try:
            t = bp.get_binned_variable(v).binning_table
            t.build()
            t.plot(metric="woe", show_bin_labels=True,
                   savefig=str(WOE_FIGS / f"{v}.png"))
            plt.close("all")
        except Exception as e:
            print(f"  plot failed for {v}: {e}")


if __name__ == "__main__":
    print("loading abt...")
    df = pd.read_parquet(PROCESSED / "abt.parquet")
    df = df[df["split"].isin(["dev", "oot"])].copy()
    df["target"] = df["target"].astype(int)

    print("deriving features...")
    df = derive_features(df)
    features = [c for c in df.columns if c not in NON_FEATURES]
    print(f"  {len(features)} candidate features")

    print("splitting...")
    train, test, oot = make_splits(df)

    print("fitting binning on train only...")
    bp = fit_binning(train, features)

    summary, weak = iv_report(bp, features)

    print("\nwriting WOE plots...")
    ranked = summary["name"].tolist()
    plot_woe(bp, ranked)

    keep = [f for f in features if f not in weak]
    print(f"\n  {len(keep)} features retained after IV floor")

    print("transforming...")
    for name, d in [("train", train), ("test", test), ("oot", oot)]:
        # transform expects the full fitted schema; the IV screen is applied to
        # the output, not the input.
        w = bp.transform(d[features], metric="woe")
        w = w[keep]
        w.columns = [f"woe_{c}" for c in w.columns]
        w["target"] = d["target"].values
        w["issue_d"] = d["issue_d"].values
        w.to_parquet(PROCESSED / f"woe_{name}.parquet", index=False)
        print(f"  woe_{name}.parquet: {w.shape[0]:,} x {w.shape[1]}")

    with open(MODELS / "binning_process.pkl", "wb") as f:
        pickle.dump({"binning_process": bp, "features": keep}, f)
    print("\nsaved models/binning_process.pkl")

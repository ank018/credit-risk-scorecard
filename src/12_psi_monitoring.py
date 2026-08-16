"""
Project 1 - Credit Risk Scorecard
Step 10: Population Stability Index monitoring.

Run:  python src/12_psi_monitoring.py
In:   data/processed/abt.parquet, models/{binning_process,logit_model,scorecard}.pkl
Out:  reports/psi_score.csv, psi_characteristics.csv, psi_offcard.csv
      reports/figures/{psi_heatmap,psi_trend,score_shift}.png

PSI answers a question the out-of-time window cannot: has the population moved
since the card was built, and where? It needs no outcomes, which is why the
2016-17 vintages are carried unlabelled (docs/target_definition.md section 6).
Those accounts do not mature until 2019-20, so they have no bad flag - but their
characteristic distributions are fully observed and that is all PSI requires.

    PSI = sum over bins of (actual% - expected%) * ln(actual% / expected%)

Computed on the scorecard's own WOE bins rather than on deciles. The bins are
what the card uses: a population can shift substantially within a bin without
moving any score, and decile-based PSI would flag that as drift when it changes
no decision. A small shift across a bin boundary does move scores, and bin-based
PSI catches it.
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
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features
from config import PROCESSED, REPORTS, FIGS, MODELS

warnings.filterwarnings("ignore", category=UserWarning)

# Industry-standard PSI thresholds.
PSI_STABLE = 0.10      # below: no action
PSI_WARN = 0.25        # 0.10-0.25: investigate; above: card is out of population
EPS = 1e-6

N_SCORE_BANDS = 10

# Characteristics not on the card, monitored to illustrate what the challenger
# would additionally require (docs/challenger_analysis.md section 6).
OFF_CARD = ["addr_state", "purpose", "loan_amnt", "bc_limit_to_income",
            "num_actv_bc_tl"]


def psi(expected_counts, actual_counts):
    """PSI between two binned distributions. Empty bins are floored rather than
    dropped: a bin that empties out is a real signal, and dropping it would
    understate the drift."""
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    e = np.maximum(e / e.sum(), EPS)
    a = np.maximum(a / a.sum(), EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def flag(v):
    if v < PSI_STABLE:
        return "stable"
    if v < PSI_WARN:
        return "monitor"
    return "ACTION"


def load():
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(MODELS / "scorecard.pkl", "rb") as f:
        card = pickle.load(f)
    return bp, obj["model"], obj["features"], card


def build(bp, model, features, card):
    """Score every vintage, including the unlabelled monitoring population."""
    df = pd.read_parquet(PROCESSED / "abt.parquet")
    df = derive_features(df).reset_index(drop=True)

    idx = bp.transform(df[bp.variable_names], metric="indices")
    idx.columns = [f"bin_{c}" for c in idx.columns]

    w = bp.transform(df[bp.variable_names], metric="woe")
    w.columns = [f"woe_{c}" for c in w.columns]

    n = len(features)
    intercept = model.params["const"]
    factor, offset = card["factor"], card["offset"]
    score = np.zeros(len(df))
    for f in features:
        score += -(model.params[f] * w[f].astype(float).values
                   + intercept / n) * factor + offset / n
    df["score"] = np.round(score)

    logodds = intercept + sum(model.params[f] * w[f].astype(float).values
                              for f in features)
    df["pd"] = 1 / (1 + np.exp(-logodds))

    df["period"] = np.where(
        df["split"] == "dev", "dev 2013-14",
        np.where(df["split"] == "oot", "oot 2015",
                 df["issue_d"].dt.to_period("Q").astype(str)))
    return pd.concat([df, idx], axis=1)


def period_order(df):
    fixed = ["dev 2013-14", "oot 2015"]
    quarters = sorted([p for p in df["period"].unique() if p not in fixed
                       and p[0].isdigit()])
    return fixed + quarters


def score_psi(df, periods):
    base = df[df["period"] == "dev 2013-14"]
    edges = np.unique(np.percentile(base["score"],
                                    np.linspace(0, 100, N_SCORE_BANDS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    exp = pd.cut(base["score"], edges, labels=False).value_counts().sort_index()

    rows = []
    for p in periods:
        d = df[df["period"] == p]
        act = (pd.cut(d["score"], edges, labels=False)
               .value_counts().reindex(exp.index, fill_value=0).sort_index())
        rows.append({
            "period": p, "n": len(d),
            "psi": psi(exp, act),
            "mean_score": d["score"].mean(),
            "mean_pd": d["pd"].mean(),
            "pct_below_530": (d["score"] < 530).mean(),
        })
    r = pd.DataFrame(rows)
    r["flag"] = r["psi"].apply(flag)
    return r


def characteristic_psi(df, periods, chars, prefix="bin_"):
    base = df[df["period"] == "dev 2013-14"]
    out = {}
    for c in chars:
        col = f"{prefix}{c}"
        if col not in df.columns:
            continue
        exp = base[col].value_counts()
        vals = {}
        for p in periods:
            act = (df.loc[df["period"] == p, col]
                   .value_counts().reindex(exp.index, fill_value=0))
            vals[p] = psi(exp, act)
        out[c] = vals
    return pd.DataFrame(out).T


if __name__ == "__main__":
    bp, model, features, card = load()
    card_chars = [f.replace("woe_", "") for f in features]
    print(f"monitoring {len(card_chars)} card characteristics")

    df = build(bp, model, features, card)
    periods = period_order(df)
    print(f"periods: {len(periods)}  ({periods[0]} ... {periods[-1]})")
    print(f"accounts: {len(df):,}  "
          f"unlabelled monitoring: {df['target'].isna().sum():,}\n")

    # --- score-level PSI
    sp = score_psi(df, periods)
    sp.to_csv(REPORTS / "psi_score.csv", index=False)
    print("score PSI vs dev 2013-14 baseline:")
    print(sp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- characteristic-level PSI
    cp = characteristic_psi(df, periods, card_chars)
    cp.to_csv(REPORTS / "psi_characteristics.csv")
    print("\ncharacteristic PSI (card):")
    print(cp.round(4).to_string())

    latest = periods[-1]
    breaches = cp[cp[latest] >= PSI_STABLE][latest].sort_values(ascending=False)
    print(f"\n  at {latest}, above {PSI_STABLE}:")
    if breaches.empty:
        print("    none")
    for c, v in breaches.items():
        print(f"    {c:<24} {v:.4f}  {flag(v)}")

    # --- off-card characteristics, for the challenger comparison
    op = characteristic_psi(df, periods, OFF_CARD)
    op.to_csv(REPORTS / "psi_offcard.csv")
    print("\ncharacteristic PSI (not on the card):")
    print(op.round(4).to_string())
    print("\n  These do not affect the scorecard. They would all require")
    print("  monitoring under the challenger, which uses all 41.")

    # --- figures
    plt.figure(figsize=(12, 6))
    sns.heatmap(cp, annot=True, fmt=".3f", cmap="RdYlGn_r",
                vmin=0, vmax=PSI_WARN, cbar_kws={"label": "PSI"},
                linewidths=0.4)
    plt.title("Characteristic PSI vs development baseline")
    plt.tight_layout()
    plt.savefig(FIGS / "psi_heatmap.png", dpi=140)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    x = range(len(sp))
    ax[0].plot(x, sp["psi"], marker="o")
    ax[0].axhline(PSI_STABLE, ls="--", c="orange", lw=1, label="0.10 monitor")
    ax[0].axhline(PSI_WARN, ls="--", c="red", lw=1, label="0.25 action")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(sp["period"], rotation=45, ha="right")
    ax[0].set_ylabel("score PSI")
    ax[0].set_title("Score stability over time")
    ax[0].legend()

    ax[1].plot(x, sp["mean_pd"], marker="o", color="tab:red")
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels(sp["period"], rotation=45, ha="right")
    ax[1].set_ylabel("mean predicted PD")
    ax[1].set_title("Predicted risk of the through-the-door population")
    plt.tight_layout()
    plt.savefig(FIGS / "psi_trend.png", dpi=140)
    plt.close()

    plt.figure(figsize=(11, 5))
    for p, c in [("dev 2013-14", "tab:blue"), ("oot 2015", "tab:orange"),
                 (periods[-1], "tab:red")]:
        s = df.loc[df["period"] == p, "score"]
        plt.hist(s, bins=60, alpha=0.45, density=True, label=p, color=c)
    plt.axvline(530, ls="--", c="k", lw=1, label="cutoff 530")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.title("Score distribution drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "score_shift.png", dpi=140)
    print("\nwrote PSI reports and figures")

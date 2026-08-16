"""
Project 1 - Credit Risk Scorecard
Step 3: multicollinearity reduction via correlation clustering and VIF.

Run:  python src/03_feature_selection.py
In:   data/processed/woe_train.parquet, reports/iv_table.csv
Out:  models/selected_features.json
      reports/vif_table.csv, reports/correlation_pairs.csv
      reports/figures/correlation_heatmap.png
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from statsmodels.stats.outliers_influence import variance_inflation_factor

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")
FIGS.mkdir(parents=True, exist_ok=True)

# Correlation ceiling for clustering. Above this, two features are treated as
# measuring the same thing and only the stronger one survives.
CORR_THRESHOLD = 0.70

# VIF ceiling. 5 is the conventional line in credit scorecard work - stricter
# than the 10 often used elsewhere, because coefficient *signs* have to be
# defensible to a credit committee, not just the fit.
VIF_THRESHOLD = 5.0


def load():
    train = pd.read_parquet(PROCESSED / "woe_train.parquet")
    woe_cols = [c for c in train.columns if c.startswith("woe_")]
    iv = pd.read_csv(REPORTS / "iv_table.csv").set_index("name")["iv"]
    # map woe_fico -> fico for IV lookup
    iv_by_col = {c: iv.get(c.replace("woe_", ""), 0.0) for c in woe_cols}
    return train, woe_cols, iv_by_col


def correlation_report(train, woe_cols):
    corr = train[woe_cols].corr(method="spearman")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.7},
                annot=True, fmt=".2f", annot_kws={"size": 7})
    plt.title("Spearman correlation, WOE-transformed features (train)")
    plt.tight_layout()
    plt.savefig(FIGS / "correlation_heatmap.png", dpi=140)
    plt.close()

    pairs = (corr.where(~mask).stack().rename("corr").reset_index()
             .rename(columns={"level_0": "feature_a", "level_1": "feature_b"}))
    pairs["abs_corr"] = pairs["corr"].abs()
    pairs = pairs.sort_values("abs_corr", ascending=False)
    high = pairs[pairs["abs_corr"] > CORR_THRESHOLD]
    high.to_csv(REPORTS / "correlation_pairs.csv", index=False)

    print(f"\n  pairs above |r| = {CORR_THRESHOLD}:")
    if high.empty:
        print("    none")
    else:
        for _, r in high.iterrows():
            print(f"    {r.feature_a:<28} {r.feature_b:<28} {r['corr']:+.3f}")
    return corr


def cluster_select(corr, iv_by_col):
    """Group features that move together, keep the highest-IV member of each
    group. Choosing by IV rather than arbitrarily means the surviving feature is
    the one that discriminates best, and the dropped ones were carrying
    substantially the same information."""
    d = 1 - corr.abs()
    np.fill_diagonal(d.values, 0)
    link = hierarchy.linkage(squareform(d, checks=False), method="average")
    labels = hierarchy.fcluster(link, t=1 - CORR_THRESHOLD, criterion="distance")

    clusters = {}
    for col, lab in zip(corr.columns, labels):
        clusters.setdefault(lab, []).append(col)

    keep, dropped = [], []
    print("\n  correlation clusters:")
    for lab, members in sorted(clusters.items()):
        best = max(members, key=lambda c: iv_by_col[c])
        keep.append(best)
        if len(members) > 1:
            others = [m for m in members if m != best]
            dropped += others
            print(f"    cluster {lab}: keep {best} (IV {iv_by_col[best]:.4f})")
            for m in others:
                print(f"               drop {m} (IV {iv_by_col[m]:.4f})")
        else:
            print(f"    cluster {lab}: {best} (standalone)")
    return keep, dropped


def vif_prune(train, cols):
    """Iteratively remove the worst offender until every VIF clears the ceiling.
    One at a time: dropping a feature changes every other VIF, so batch removal
    over-prunes."""
    cols = list(cols)
    history = []
    while True:
        X = train[cols].astype(float).values
        vifs = pd.Series(
            [variance_inflation_factor(X, i) for i in range(len(cols))],
            index=cols,
        ).sort_values(ascending=False)
        worst, worst_vif = vifs.index[0], vifs.iloc[0]
        if worst_vif <= VIF_THRESHOLD:
            break
        print(f"    drop {worst:<28} VIF {worst_vif:.2f}")
        history.append({"feature": worst, "vif": worst_vif})
        cols.remove(worst)

    final = pd.DataFrame({"feature": vifs.index, "vif": vifs.values})
    final.to_csv(REPORTS / "vif_table.csv", index=False)
    print("\n  final VIF:")
    print(final.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return cols


if __name__ == "__main__":
    train, woe_cols, iv_by_col = load()
    print(f"starting from {len(woe_cols)} features")

    corr = correlation_report(train, woe_cols)
    keep, dropped = cluster_select(corr, iv_by_col)
    print(f"\n  {len(keep)} features after correlation clustering")

    print("\n  VIF pruning:")
    final = vif_prune(train, keep)

    # Order the final list by IV so the scorecard reads strongest-first.
    final = sorted(final, key=lambda c: iv_by_col[c], reverse=True)

    print(f"\nFINAL: {len(final)} features")
    for c in final:
        print(f"  {c.replace('woe_', ''):<28} IV {iv_by_col[c]:.4f}")

    with open(MODELS / "selected_features.json", "w") as f:
        json.dump({"features": final,
                   "dropped_correlation": dropped,
                   "corr_threshold": CORR_THRESHOLD,
                   "vif_threshold": VIF_THRESHOLD}, f, indent=2)
    print("\nsaved models/selected_features.json")

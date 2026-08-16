"""
Project 1 - Credit Risk Scorecard
Step 6: discrimination, calibration and score band performance.

Run:  python src/06_evaluation.py
In:   data/processed/scores_*.parquet, pd_*.parquet, abt.parquet
Out:  reports/performance_summary.csv, reports/score_bands.csv,
      reports/calibration.csv
      reports/figures/{roc,ks,calibration,bad_rate_by_band}.png
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

SPLITS = ("train", "test", "oot")
N_BANDS = 10


def ks_statistic(y, score):
    """Maximum separation between the cumulative good and bad distributions.
    Credit teams quote KS alongside Gini because it answers a different
    question: not 'how well does it rank' but 'where is the cut most effective'."""
    order = np.argsort(score)
    y = np.asarray(y)[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    diff = np.abs(cum_bad - cum_good)
    i = np.argmax(diff)
    return diff[i], np.sort(score)[i]


def load_split(name):
    s = pd.read_parquet(PROCESSED / f"scores_{name}.parquet")
    p = pd.read_parquet(PROCESSED / f"pd_{name}.parquet")
    s["pd"] = p["pd"].values
    return s


def benchmark_grade():
    """Lending Club's own sub-grade on the same out-of-time accounts. The
    scorecard was built without it (see docs/target_definition.md); this is the
    comparison that says whether the model adds anything over the lender's
    existing assessment."""
    abt = pd.read_parquet(PROCESSED / "abt.parquet")
    oot = abt[abt["split"] == "oot"].copy()
    oot["target"] = oot["target"].astype(int)
    codes = sorted(oot["sub_grade"].dropna().unique())
    rank = {g: i for i, g in enumerate(codes)}
    oot = oot[oot["sub_grade"].notna()]
    auc = roc_auc_score(oot["target"], oot["sub_grade"].map(rank))
    return auc, 2 * auc - 1, len(oot)


def performance_table(data):
    rows = []
    for name, d in data.items():
        auc = roc_auc_score(d["target"], d["pd"])
        ks, ks_at = ks_statistic(d["target"], d["score"])
        rows.append({
            "split": name, "n": len(d), "bad_rate": d["target"].mean(),
            "auc": auc, "gini": 2 * auc - 1, "ks": ks, "ks_at_score": ks_at,
            "brier": brier_score_loss(d["target"], d["pd"]),
            "mean_pred_pd": d["pd"].mean(),
        })
    return pd.DataFrame(rows)


def score_bands(data, train_scores):
    """Bands fixed on the training distribution, then applied unchanged. Cutting
    each split at its own deciles would hide exactly the population shift the
    out-of-time window exists to reveal."""
    edges = np.unique(np.percentile(train_scores, np.linspace(0, 100, N_BANDS + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    out = []
    for name, d in data.items():
        b = pd.cut(d["score"], bins=edges, labels=False, include_lowest=True)
        g = (d.assign(band=b).groupby("band")
               .agg(n=("target", "size"), bads=("target", "sum"),
                    bad_rate=("target", "mean"), mean_pd=("pd", "mean"),
                    min_score=("score", "min"), max_score=("score", "max")))
        g["split"] = name
        g["pop_pct"] = g["n"] / g["n"].sum()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


def calibration(data, n_bins=10):
    """Predicted PD against observed default rate. Discrimination and
    calibration fail independently: a model can rank perfectly and still be
    wrong about the level, which matters here because PD feeds pricing and
    provisioning, not just accept/decline."""
    rows = []
    for name, d in data.items():
        q = pd.qcut(d["pd"], n_bins, labels=False, duplicates="drop")
        g = (d.assign(q=q).groupby("q")
               .agg(n=("target", "size"), predicted=("pd", "mean"),
                    observed=("target", "mean")))
        g["split"] = name
        rows.append(g.reset_index())
    return pd.concat(rows, ignore_index=True)


def plot_roc(data):
    plt.figure(figsize=(6, 6))
    for name, d in data.items():
        fpr, tpr, _ = roc_curve(d["target"], d["pd"])
        auc = roc_auc_score(d["target"], d["pd"])
        plt.plot(fpr, tpr, lw=1.5, label=f"{name} (Gini {2*auc-1:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC by split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "roc.png", dpi=140)
    plt.close()


def plot_ks(d, name="oot"):
    order = np.argsort(d["score"].values)
    y = d["target"].values[order]
    s = d["score"].values[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    i = np.argmax(np.abs(cum_bad - cum_good))

    plt.figure(figsize=(8, 5))
    plt.plot(s, cum_bad, label="cumulative bads")
    plt.plot(s, cum_good, label="cumulative goods")
    plt.vlines(s[i], cum_good[i], cum_bad[i], color="r", lw=2,
               label=f"KS {abs(cum_bad[i]-cum_good[i]):.3f} @ {s[i]:.0f}")
    plt.xlabel("score")
    plt.ylabel("cumulative proportion")
    plt.title(f"KS separation ({name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "ks.png", dpi=140)
    plt.close()


def plot_calibration(cal):
    plt.figure(figsize=(6, 6))
    for name in SPLITS:
        c = cal[cal["split"] == name]
        plt.plot(c["predicted"], c["observed"], marker="o", lw=1.2, label=name)
    lim = [0, cal[["predicted", "observed"]].values.max() * 1.05]
    plt.plot(lim, lim, "k--", lw=0.8, label="perfect calibration")
    plt.xlabel("mean predicted PD")
    plt.ylabel("observed default rate")
    plt.title("Calibration by predicted-PD decile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "calibration.png", dpi=140)
    plt.close()


def plot_bands(bands):
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.27
    for i, name in enumerate(SPLITS):
        b = bands[bands["split"] == name].sort_values("band")
        ax.bar(b["band"] + (i - 1) * w, b["bad_rate"], width=w, label=name)
    ax.set_xlabel("score band (1 = lowest score, 10 = highest)")
    ax.set_ylabel("observed default rate")
    ax.set_title("Default rate by score band, bands fixed on train")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "bad_rate_by_band.png", dpi=140)
    plt.close()


if __name__ == "__main__":
    data = {name: load_split(name) for name in SPLITS}

    perf = performance_table(data)
    perf.to_csv(REPORTS / "performance_summary.csv", index=False)
    print("performance:")
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    auc_g, gini_g, n_g = benchmark_grade()
    scorecard_gini = perf.loc[perf["split"] == "oot", "gini"].iloc[0]
    print(f"\nbenchmark - Lending Club sub_grade on the same OOT accounts:")
    print(f"  sub_grade  Gini {gini_g:.4f}  (n={n_g:,})")
    print(f"  scorecard  Gini {scorecard_gini:.4f}")
    print(f"  difference {scorecard_gini - gini_g:+.4f}")

    bands = score_bands(data, data["train"]["score"].values)
    bands.to_csv(REPORTS / "score_bands.csv", index=False)
    print("\nscore bands (OOT):")
    b = bands[bands["split"] == "oot"].sort_values("band")
    print(b[["band", "min_score", "max_score", "n", "bad_rate", "mean_pd"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    mono = b["bad_rate"].is_monotonic_decreasing
    print(f"\n  bad rate monotonic across bands: {mono}")

    cal = calibration(data)
    cal.to_csv(REPORTS / "calibration.csv", index=False)
    print("\ncalibration gap (observed - predicted), by split:")
    for name in SPLITS:
        c = cal[cal["split"] == name]
        gap = (c["observed"] - c["predicted"]).mean()
        print(f"  {name:<6} {gap:+.4f}")

    plot_roc(data)
    plot_ks(data["oot"])
    plot_calibration(cal)
    plot_bands(bands)
    print("\nwrote figures to reports/figures/")

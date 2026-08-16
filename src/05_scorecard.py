"""
Project 1 - Credit Risk Scorecard
Step 5: scale the logistic model into a points-based scorecard.

Run:  python src/05_scorecard.py
In:   models/logit_model.pkl, models/binning_process.pkl
      data/processed/woe_{train,test,oot}.parquet
Out:  models/scorecard.pkl
      reports/points_table.csv, reports/score_summary.csv
      reports/figures/score_distribution.png
      data/processed/scores_{train,test,oot}.parquet
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED = Path("data/processed")
REPORTS = Path("reports")
FIGS = REPORTS / "figures"
MODELS = Path("models")

# Scaling parameters. These are a presentation choice, not a modelling one -
# they change the numbers on the card, never the rank ordering or the PDs.
# Chosen to sit in the range consumers recognise from bureau scores.
BASE_SCORE = 600      # score at which the odds equal BASE_ODDS
BASE_ODDS = 50        # 50:1 good:bad at the base score
PDO = 20              # points to double the odds


def scaling_constants():
    """factor = PDO / ln(2), offset = base - factor * ln(base_odds).

    Every PDO points added doubles the good:bad odds, which is what makes a
    scorecard readable: the difference between 640 and 660 means the same thing
    as the difference between 700 and 720."""
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return factor, offset


def build_points_table(bp, model, features, factor, offset):
    """Points per attribute.

        points_i = -(beta_i * WOE_i + intercept/n) * factor + offset/n

    The intercept and offset are spread evenly across the n characteristics so
    the per-attribute points sum to the total score. The leading minus converts
    the model's log-odds of *bad* into odds of *good*, so that a higher score
    means a better applicant - the direction everyone expects."""
    n = len(features)
    intercept = model.params["const"]
    rows = []

    for f in features:
        raw = f.replace("woe_", "")
        beta = model.params[f]
        t = bp.get_binned_variable(raw).binning_table.build()
        t = t.iloc[:-1]              # drop the Totals row
        t = t[t["Count"] > 0]        # drop the empty Special bin

        for _, r in t.iterrows():
            woe = r["WoE"]
            pts = -(beta * woe + intercept / n) * factor + offset / n
            rows.append({
                "characteristic": raw,
                "bin": str(r["Bin"]),
                "count": int(r["Count"]),
                "count_pct": r["Count (%)"],
                "event_rate": r["Event rate"],
                "woe": woe,
                "coefficient": beta,
                "points": round(pts),
            })

    pts = pd.DataFrame(rows)

    # Spread = how much a single characteristic can move an applicant's score.
    # A card where one characteristic dominates is fragile: drift in that one
    # feature moves the whole book.
    spread = (pts.groupby("characteristic")["points"]
                 .agg(min_points="min", max_points="max")
                 .assign(spread=lambda d: d.max_points - d.min_points)
                 .sort_values("spread", ascending=False))
    return pts, spread


def score_frame(d, features, model, factor, offset):
    """Score = offset + factor * ln(odds_good), computed from the WOE values."""
    n = len(features)
    intercept = model.params["const"]
    total = np.full(len(d), 0.0)
    for f in features:
        total += -(model.params[f] * d[f].astype(float).values
                   + intercept / n) * factor + offset / n
    return np.round(total)


def verify(scores, pd_series, factor, offset):
    """A score must map back to the PD it came from. If this does not hold, the
    points table and the model have diverged and the card is wrong."""
    odds_good = np.exp((scores - offset) / factor)
    implied_pd = 1 / (1 + odds_good)
    err = np.abs(implied_pd - pd_series).max()
    print(f"  max |implied PD - model PD|: {err:.6f}")
    if err > 0.005:
        print("  !! points table does not reconcile to the model")
    else:
        print("  points table reconciles to the model")


if __name__ == "__main__":
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    model, features = obj["model"], obj["features"]
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]

    factor, offset = scaling_constants()
    print(f"scaling: base {BASE_SCORE} @ {BASE_ODDS}:1 odds, PDO {PDO}")
    print(f"  factor {factor:.4f}, offset {offset:.4f}")

    points, spread = build_points_table(bp, model, features, factor, offset)
    points.to_csv(REPORTS / "points_table.csv", index=False)

    print(f"\n  points table: {len(points)} attributes across "
          f"{points['characteristic'].nunique()} characteristics")
    print("\n  score contribution by characteristic:")
    print(spread.to_string())

    print("\n  scoring:")
    summary = []
    for name in ("train", "test", "oot"):
        d = pd.read_parquet(PROCESSED / f"woe_{name}.parquet")
        s = score_frame(d, features, model, factor, offset)
        out = pd.DataFrame({"score": s, "target": d["target"].values,
                            "issue_d": d["issue_d"].values})
        out.to_parquet(PROCESSED / f"scores_{name}.parquet", index=False)

        summary.append({
            "split": name, "n": len(out), "bad_rate": out["target"].mean(),
            "min": s.min(), "p05": np.percentile(s, 5), "median": np.median(s),
            "p95": np.percentile(s, 95), "max": s.max(), "mean": s.mean(),
        })
        print(f"    {name:<6} median {np.median(s):.0f}  "
              f"range {s.min():.0f}-{s.max():.0f}")

    summary = pd.DataFrame(summary)
    summary.to_csv(REPORTS / "score_summary.csv", index=False)

    # Reconciliation check on train.
    print("\n  verifying:")
    pd_train = pd.read_parquet(PROCESSED / "pd_train.parquet")
    s_train = pd.read_parquet(PROCESSED / "scores_train.parquet")
    verify(s_train["score"].values, pd_train["pd"].values, factor, offset)

    # Distribution plot.
    plt.figure(figsize=(11, 5))
    for name, c in [("train", "tab:blue"), ("oot", "tab:orange")]:
        s = pd.read_parquet(PROCESSED / f"scores_{name}.parquet")["score"]
        plt.hist(s, bins=60, alpha=0.5, label=name, color=c, density=True)
    plt.axvline(BASE_SCORE, ls="--", c="k", lw=1,
                label=f"base score {BASE_SCORE} ({BASE_ODDS}:1)")
    plt.xlabel("score")
    plt.ylabel("density")
    plt.title("Scorecard distribution: development vs out-of-time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "score_distribution.png", dpi=140)

    with open(MODELS / "scorecard.pkl", "wb") as f:
        pickle.dump({"points": points, "factor": factor, "offset": offset,
                     "base_score": BASE_SCORE, "base_odds": BASE_ODDS,
                     "pdo": PDO, "features": features}, f)
    print("\nsaved models/scorecard.pkl")

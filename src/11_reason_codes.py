"""
Project 1 - Credit Risk Scorecard
Step 9: adverse action reason codes (ECOA / Regulation B).

Run:  python src/11_reason_codes.py
In:   models/{scorecard,logit_model,binning_process,xgb_model}.pkl
      data/processed/abt.parquet
Out:  reports/cutoff_analysis.csv
      reports/reason_codes_sample.csv
      reports/reason_code_frequency.csv
      reports/reason_code_agreement.csv
      reports/figures/reason_code_frequency.png

Under ECOA / Regulation B, an applicant declined on the basis of a credit model
must be told the specific principal reasons. "Your score was too low" does not
satisfy the requirement; the disclosure must identify which characteristics drove
the outcome.

For a points-based scorecard this is arithmetic on the published table rather than
a separate explanation model. Each applicant's score is a sum of attribute points;
the reasons are the characteristics where they lost the most points. Two
applicants with identical inputs receive identical reasons, the result is stable
until the card is rebuilt, and anyone holding the points table can reproduce it.
"""

from pathlib import Path
import json
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from features import derive_features
from config import PROCESSED, REPORTS, FIGS, MODELS

warnings.filterwarnings("ignore", category=UserWarning)

N_REASONS = 4          # Reg B allows up to four principal reasons
TARGET_DECLINE = 0.20  # policy cutoff used for illustration
SAMPLE_N = 25

# Shortfall reference for ranking reasons. 'mean' is the deployed convention:
# ranking against the best attainable attribute lets characteristics with wide
# point spreads dominate, and measurement showed four of eleven characteristics
# could then never appear in any disclosure. See docs/reason_codes.md section 4.
METHOD = "mean"

# Reason statements follow the convention used in bureau-score disclosures:
# each names the *factor*, not the direction. This is deliberate. A statement
# like "your income is too low" asserts a threshold the card does not define,
# whereas naming the factor is accurate for every applicant who lost points on
# it, whichever side of the distribution they sit on.
REASON_TEXT = {
    "fico": "Credit bureau score",
    "dti_with_loan": "Total debt obligations relative to income",
    "loan_to_income": "Amount requested relative to income",
    "annual_inc": "Level of income stated on the application",
    "avg_cur_bal": "Level of balances maintained across accounts",
    "total_bc_limit": "Amount of credit available on revolving accounts",
    "mo_sin_rcnt_tl": "Time since most recent account was opened",
    "mo_sin_old_rev_tl_op": "Length of time revolving accounts established",
    "inq_intensity": "Recent credit inquiries relative to accounts held",
    "percent_bc_gt_75": "Proportion of revolving accounts near their limit",
    "home_ownership": "Housing status",
}


def load_artifacts():
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(MODELS / "scorecard.pkl", "rb") as f:
        card = pickle.load(f)
    return bp, obj["model"], obj["features"], card


def build_oot(bp):
    df = pd.read_parquet(PROCESSED / "abt.parquet")
    df = df[df["split"] == "oot"].copy()
    df["target"] = df["target"].astype(int)
    df = derive_features(df).reset_index(drop=True)
    w = bp.transform(df[bp.variable_names], metric="woe")
    w.columns = [f"woe_{c}" for c in w.columns]
    return df, w


def attribute_points(w, features, model, card):
    """Per-characteristic points for every applicant, and the maximum attainable
    points for each characteristic across all its bins."""
    n = len(features)
    intercept = model.params["const"]
    factor, offset = card["factor"], card["offset"]

    pts = pd.DataFrame(index=w.index)
    for f in features:
        pts[f.replace("woe_", "")] = np.round(
            -(model.params[f] * w[f].astype(float).values + intercept / n)
            * factor + offset / n
        )

    table = card["points"]
    max_pts = table.groupby("characteristic")["points"].max()
    return pts, max_pts


def choose_cutoff(scores, target_decline):
    """Illustrative policy cutoff. A real cutoff comes from a loss-versus-volume
    optimisation with a funding cost and a loss-given-default assumption; this
    one simply declines a fixed share so the reason codes have a population to
    describe."""
    return int(np.percentile(scores, target_decline * 100))


def cutoff_table(scores, target):
    rows = []
    for q in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        c = int(np.percentile(scores, q * 100))
        declined = scores < c
        rows.append({
            "decline_rate": q, "cutoff_score": c,
            "n_declined": int(declined.sum()),
            "bad_rate_declined": target[declined].mean(),
            "bad_rate_approved": target[~declined].mean(),
            "bad_rate_overall": target.mean(),
        })
    return pd.DataFrame(rows)


def reason_codes(pts, max_pts, method="max"):
    """Rank characteristics by points forgone.

    method='max'  - shortfall against the best attainable attribute. This is the
                    common convention and answers "what most limited this
                    score". Its weakness is that characteristics with wide point
                    spreads dominate the ranking for nearly everyone.

    method='mean' - shortfall against the population-average points for that
                    characteristic. Answers "what makes this applicant worse
                    than typical", which discriminates better between applicants
                    but can name a characteristic where they are only slightly
                    below average.
    """
    reference = max_pts if method == "max" else pts.mean()
    shortfall = reference[pts.columns] - pts
    order = np.argsort(-shortfall.values, axis=1)[:, :N_REASONS]
    chars = np.array(pts.columns)
    return chars[order], np.take_along_axis(shortfall.values, order, axis=1)


def frequency(codes, label):
    flat = pd.Series(codes.ravel())
    f = (flat.value_counts(normalize=True).rename("share").reset_index()
         .rename(columns={"index": "characteristic"}))
    f["method"] = label
    return f


def shap_reasons(df, features_raw):
    """Top-4 characteristics by |SHAP| per applicant from the challenger, for
    comparison. Restricted to characteristics that exist on the card, so the
    comparison is about ranking rather than vocabulary."""
    import shap
    with open(MODELS / "xgb_model.pkl", "rb") as f:
        obj = pickle.load(f)
    m, feats = obj["model"], obj["features"]
    for c in ["home_ownership", "verification_status", "purpose", "addr_state",
              "initial_list_status", "application_type"]:
        if c in df.columns:
            df[c] = df[c].astype("category")
    vals = shap.TreeExplainer(m).shap_values(df[feats])
    keep = [i for i, f in enumerate(feats) if f in features_raw]
    sub = np.abs(vals[:, keep])
    names = np.array([feats[i] for i in keep])
    order = np.argsort(-sub, axis=1)[:, :N_REASONS]
    return names[order]


if __name__ == "__main__":
    bp, model, features, card = load_artifacts()
    print(f"scorecard: {len(features)} characteristics, "
          f"{len(card['points'])} attributes")

    df, w = build_oot(bp)
    pts, max_pts = attribute_points(w, features, model, card)
    scores = pts.sum(axis=1).values
    print(f"scored {len(scores):,} out-of-time applications")

    ct = cutoff_table(scores, df["target"])
    ct.to_csv(REPORTS / "cutoff_analysis.csv", index=False)
    print("\ncutoff analysis:")
    print(ct.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    cutoff = choose_cutoff(scores, TARGET_DECLINE)
    declined = scores < cutoff
    print(f"\npolicy cutoff {cutoff}: {declined.sum():,} declined "
          f"({declined.mean():.1%}), bad rate among declined "
          f"{df.loc[declined, 'target'].mean():.2%} vs "
          f"{df.loc[~declined, 'target'].mean():.2%} approved")

    d_pts = pts[declined].reset_index(drop=True)
    codes_max, gap_max = reason_codes(d_pts, max_pts, "max")
    codes_mean, gap_mean = reason_codes(d_pts, max_pts, "mean")
    codes, gaps = (codes_mean, gap_mean) if METHOD == "mean" else (codes_max, gap_max)
    reference = d_pts.mean() if METHOD == "mean" else max_pts
    print(f"\nranking reasons by shortfall against: {METHOD}")

    freq = pd.concat([frequency(codes_max, "max"),
                      frequency(codes_mean, "mean")], ignore_index=True)
    freq.to_csv(REPORTS / "reason_code_frequency.csv", index=False)

    print("\nreason code frequency among declined applicants:")
    piv = freq.pivot(index="characteristic", columns="method",
                     values="share").fillna(0).sort_values(METHOD, ascending=False)
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))

    dead = piv.index[piv["max"] < 0.005].tolist()
    if dead:
        print(f"\n  never reachable under 'max': {dead}")
        print("     Their point spreads are narrower than a typical partial")
        print("     shortfall on a wide characteristic, so they cannot rank in")
        print("     any applicant's top four. This is why 'mean' is deployed.")

    # Sample disclosures.
    idx = np.random.RandomState(42).choice(len(d_pts), SAMPLE_N, replace=False)
    rows = []
    for i in idx:
        for rank in range(N_REASONS):
            c = codes[i, rank]
            rows.append({
                "applicant": int(i),
                "score": int(d_pts.iloc[i].sum()),
                "rank": rank + 1,
                "characteristic": c,
                "points_awarded": int(d_pts.iloc[i][c]),
                "points_available": int(max_pts[c]),
                "points_typical": round(float(reference[c]), 1),
                "points_forgone": round(float(gaps[i, rank]), 1),
                "reason": REASON_TEXT.get(c, c),
                "method": METHOD,
            })
    sample = pd.DataFrame(rows)
    sample.to_csv(REPORTS / "reason_codes_sample.csv", index=False)

    print("\nexample disclosure:")
    first = sample[sample["applicant"] == sample["applicant"].iloc[0]]
    print(f"  Application declined. Score {first['score'].iloc[0]} "
          f"(cutoff {cutoff}).")
    print("  Principal reasons:")
    for _, r in first.iterrows():
        print(f"    {r['rank']}. {r['reason']}  "
              f"({r['points_awarded']} points, typical "
              f"{r['points_typical']}, available {r['points_available']})")

    # Agreement with the challenger's SHAP-derived reasons.
    print("\ncomparing against challenger SHAP attribution...")
    features_raw = [f.replace("woe_", "") for f in features]
    sub = df[declined].reset_index(drop=True).iloc[:5000]
    shap_codes = shap_reasons(sub, features_raw)
    overlap = [len(set(codes[i]) & set(shap_codes[i]))
               for i in range(len(shap_codes))]
    top1 = np.mean([codes[i][0] == shap_codes[i][0]
                    for i in range(len(shap_codes))])
    agree = pd.DataFrame({
        "metric": ["mean overlap of top-4", "top-1 exact match",
                   "at least 3 of 4 shared", "at least 2 of 4 shared"],
        "value": [np.mean(overlap), top1,
                  np.mean(np.array(overlap) >= 3),
                  np.mean(np.array(overlap) >= 2)],
    })
    agree.to_csv(REPORTS / "reason_code_agreement.csv", index=False)
    print(agree.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Persist everything the scoring service needs to reproduce these reasons:
    # the shortfall reference, the per-characteristic maxima, the cutoff and the
    # disclosure text. Without this the API would have to re-derive the
    # reference from data it does not have at serving time.
    with open(MODELS / "reason_reference.json", "w") as f:
        json.dump({
            "method": METHOD,
            "cutoff": int(cutoff),
            "n_reasons": N_REASONS,
            "reference_points": {k: round(float(v), 4)
                                 for k, v in reference.items()},
            "max_points": {k: int(v) for k, v in max_pts.items()
                           if k in d_pts.columns},
            "reason_text": REASON_TEXT,
        }, f, indent=2)
    print("\nsaved models/reason_reference.json")

    fig, ax = plt.subplots(figsize=(9, 5))
    piv.plot.barh(ax=ax)
    ax.set_xlabel("share of declined applicants naming this characteristic")
    ax.set_ylabel("")
    ax.set_title(f"Reason code frequency, cutoff {cutoff} "
                 f"({declined.mean():.0%} declined)")
    plt.tight_layout()
    plt.savefig(FIGS / "reason_code_frequency.png", dpi=140)
    print("\nwrote reason code reports and figure")

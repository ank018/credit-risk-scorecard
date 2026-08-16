"""
Project 1 - Credit Risk Scorecard
Step 1: bad definition, vintage profiling, analytical base table.

Run:  python src/01_build_base_table.py
In:   data/raw/accepted_2007_to_2018Q4.csv.gz
Out:  data/interim/vintage_profile.csv      <- bad rate by issue month
      data/processed/abt.parquet            <- modelling base table
      reports/figures/bad_rate_by_vintage.png
"""

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = Path("data/raw/accepted_2007_to_2018Q4.csv.gz")
INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
FIGS = Path("reports/figures")
for p in (INTERIM, PROCESSED, FIGS):
    p.mkdir(parents=True, exist_ok=True)

# --- Target definition -------------------------------------------------------
# Bad    = account reached charge-off / default within the 36-month term.
# Good   = account repaid in full.
# Indet. = still open or in early delinquency at the observation cut. Excluded,
#          but counted, because silently dropping accounts distorts the bad rate.
BAD = {"Charged Off", "Default"}
GOOD = {"Fully Paid"}
INDETERMINATE = {"Current", "In Grace Period", "Late (16-30 days)", "Late (31-120 days)"}

# Pre-2010 accounts underwritten under a different credit policy. Different
# population, so they are removed rather than relabelled.
POLICY_PREFIX = "Does not meet the credit policy"

# --- Window design -----------------------------------------------------------
# Observation point = issue_d. Performance window = the full 36-month term.
# A vintage is only eligible once its window has closed inside the data.
DEV_START, DEV_END = "2013-01-01", "2014-12-31"   # train/test drawn from here
OOT_START, OOT_END = "2015-01-01", "2015-12-31"   # last fully matured vintage
MON_START, MON_END = "2016-01-01", "2017-12-31"   # PSI only, no labels needed

# --- Columns -----------------------------------------------------------------
# Everything below is known at or before the application decision.
APPLICATION_FEATURES = [
    "loan_amnt", "term", "emp_length", "home_ownership", "annual_inc",
    "verification_status", "purpose", "addr_state", "dti", "delinq_2yrs",
    "earliest_cr_line", "fico_range_low", "fico_range_high", "inq_last_6mths",
    "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "initial_list_status",
    "application_type", "acc_now_delinq", "tot_coll_amt", "tot_cur_bal",
    "mort_acc", "pub_rec_bankruptcies", "collections_12_mths_ex_med",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_tl", "num_actv_bc_tl", "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "total_bc_limit", "avg_cur_bal",
]

# Known at origination, but they encode Lending Club's own risk model. Held out
# of the champion and kept only as a benchmark to score the scorecard against.
BENCHMARK = ["grade", "sub_grade", "int_rate"]

KEYS = ["id", "issue_d", "loan_status"]


def load():
    cols = KEYS + APPLICATION_FEATURES + BENCHMARK
    df = pd.read_csv(RAW, usecols=cols, low_memory=False)
    # issue_d ships as "Dec-2015" on this mirror and "Dec-15" on older ones.
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed")
    df["vintage"] = df["issue_d"].dt.to_period("M")
    df["term"] = df["term"].str.strip()
    return df


def apply_filters(df):
    """Population filters that apply to every split, labelled or not."""
    n0 = len(df)
    df = df[~df["loan_status"].str.startswith(POLICY_PREFIX, na=False)]
    print(f"  dropped {n0 - len(df):,} old-credit-policy accounts")
    n1 = len(df)
    df = df[df["term"] == "36 months"]
    print(f"  dropped {n1 - len(df):,} 60-month accounts")
    return df


def build_modelling_frame(df):
    """Dev and OOT only: matured vintages, so outcomes are fully observed."""
    indet = df["loan_status"].isin(INDETERMINATE).sum()
    print(f"  {indet:,} indeterminate across all vintages")
    m = df[df["loan_status"].isin(BAD | GOOD)].copy()
    m["target"] = m["loan_status"].isin(BAD).astype(int)
    d = m["issue_d"]
    m["split"] = pd.NA
    m.loc[d.between(DEV_START, DEV_END), "split"] = "dev"
    m.loc[d.between(OOT_START, OOT_END), "split"] = "oot"
    return m[m["split"].notna()].copy()


def build_monitoring_frame(df):
    """PSI compares distributions, not outcomes, so immature vintages are fine
    here - and must be kept whole. Filtering 2016-17 to terminal statuses keeps
    early charge-offs while dropping accounts still paying: survivorship bias,
    not drift."""
    m = df[df["issue_d"].between(MON_START, MON_END)].copy()
    m["split"] = "monitor"
    m["target"] = pd.NA
    print(f"  monitoring population: {len(m):,} accounts, unlabelled")
    return m


def profile(df):
    """Bad rate by vintage. This table is the week-1 deliverable - read it
    before modelling anything. Look for: volume ramp, seasonality, and whether
    the 2015 OOT bad rate sits meaningfully above the dev period."""
    g = (df.groupby("vintage")
           .agg(accounts=("target", "size"), bads=("target", "sum"))
           .assign(bad_rate=lambda x: x.bads / x.accounts))
    g.to_csv(INTERIM / "vintage_profile.csv")

    ax = g["bad_rate"].plot(figsize=(12, 5), marker=".", lw=1)
    for start, end, label, c in [
        (DEV_START, DEV_END, "dev", "tab:blue"),
        (OOT_START, OOT_END, "oot", "tab:orange"),
    ]:
        ax.axvspan(pd.Period(start, "M").ordinal, pd.Period(end, "M").ordinal,
                   alpha=0.12, color=c, label=label)
    ax.set_title("Bad rate by vintage (36-month accounts, matured)")
    ax.set_ylabel("bad rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / "bad_rate_by_vintage.png", dpi=140)

    print("\n  bad rate by split:")
    print(df.groupby("split", dropna=True)
            .agg(accounts=("target", "size"), bad_rate=("target", "mean"))
            .round(4).to_string())
    return g


if __name__ == "__main__":
    print("loading...")
    df = load()
    print(f"  {len(df):,} rows, {df['issue_d'].min():%b %Y} - {df['issue_d'].max():%b %Y}")

    print("applying filters...")
    df = apply_filters(df)

    model = build_modelling_frame(df)
    monitor = build_monitoring_frame(df)
    profile(model)

    abt = pd.concat([model, monitor], ignore_index=True)
    abt.to_parquet(PROCESSED / "abt.parquet", index=False)
    print(f"\nwrote abt.parquet: {len(abt):,} rows x {abt.shape[1]} cols")

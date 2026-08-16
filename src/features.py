"""Feature derivation shared across the pipeline and the scoring API.

Kept in one place so an applicant scored through the API passes through
identical transformations to those used in development.

Two kinds of work happen here:

  1. TYPE COERCION - raw fields that are unusable in their published form
     (a date, a string ladder, a two-column band).

  2. RATIO CONSTRUCTION - characteristics built by combining raw fields.

The second exists because univariate IV screening cannot see conditional
signal. `loan_amnt` scores IV 0.0115 on its own, because larger loans go to
higher-income borrowers and the risk of the amount cancels against the quality
of the borrower. Ablation (docs/challenger_analysis.md §4) measured its true
contribution at +0.0178 Gini - more than every interaction and non-linearity
XGBoost found across the 12 characteristics that survived the screen.

The fix is to encode the conditional relationship directly, as a ratio a credit
committee can reason about, rather than hoping the regression recovers it.

Division by zero produces inf, which is converted to NaN so that binning assigns
it a Missing bin with its own weight. This is deliberate: an applicant with zero
declared income is a meaningful category, not a value to impute.
"""

import numpy as np
import pandas as pd

TERM_MONTHS = 36  # population is filtered to 36-month accounts


def _safe_divide(numerator, denominator):
    """Ratio with inf mapped to NaN, for the WOE Missing bin to absorb."""
    out = numerator / denominator
    return out.replace([np.inf, -np.inf], np.nan)


def derive_features(df):
    # ------------------------------------------------------------------
    # 1. Type coercion
    # ------------------------------------------------------------------

    # Credit file age at application. The raw field is a date, which is not a
    # risk characteristic; file age at the observation point is.
    ecl = pd.to_datetime(df["earliest_cr_line"], format="mixed", errors="coerce")
    df["credit_hist_months"] = ((df["issue_d"] - ecl).dt.days / 30.44).round()
    df = df.drop(columns=["earliest_cr_line"])

    # "< 1 year" ... "10+ years" -> ordinal. Nulls stay null so binning assigns
    # them their own bin rather than an assumed value.
    emp_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
               "4 years": 4, "5 years": 5, "6 years": 6, "7 years": 7,
               "8 years": 8, "9 years": 9, "10+ years": 10}
    df["emp_length_num"] = df["emp_length"].map(emp_map)
    df = df.drop(columns=["emp_length"])

    # Some mirrors ship revol_util as "34.5%" rather than a float.
    if df["revol_util"].dtype == object:
        df["revol_util"] = (df["revol_util"].astype(str)
                            .str.rstrip("%").replace("nan", np.nan).astype(float))

    # FICO is published as a 4-point band; the endpoints are perfectly
    # collinear, so the midpoint is the usable value.
    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df = df.drop(columns=["fico_range_low", "fico_range_high"])

    # ------------------------------------------------------------------
    # 2. Ratio construction
    # ------------------------------------------------------------------

    # Loan-to-income. The standard bank characteristic for exposure relative to
    # capacity, and the direct remedy for the loan_amnt screening failure: a
    # $30k loan against $50k income is a different proposition from $8k against
    # the same income, and only the ratio says so.
    df["loan_to_income"] = _safe_divide(df["loan_amnt"], df["annual_inc"])

    # Debt service including the loan being applied for. The published `dti`
    # covers existing obligations only, which is a strange basis on which to
    # judge a new one. Monthly principal is approximated as loan_amnt / term:
    # `installment` is deliberately avoided because it is a function of
    # int_rate and would reintroduce Lending Club's own pricing.
    monthly_principal = df["loan_amnt"] / TERM_MONTHS
    monthly_income = df["annual_inc"] / 12
    df["dti_with_loan"] = df["dti"] + _safe_divide(monthly_principal,
                                                   monthly_income) * 100

    # Revolving capacity other lenders have already extended, relative to
    # income. A bureau-sourced second opinion on capacity, independent of the
    # applicant's own declaration.
    df["bc_limit_to_income"] = _safe_divide(df["total_bc_limit"],
                                            df["annual_inc"])

    # Balance carried against bankcard limit. Related to revol_util but
    # anchored on the bankcard line specifically; correlation clustering will
    # drop whichever of the two is weaker.
    df["bal_to_bc_limit"] = _safe_divide(df["revol_bal"],
                                         df["total_bc_limit"])

    # Accounts opened per year of credit history. Separates a thin file from an
    # aggressively expanding one - a distinction the raw account counts cannot
    # make, since 8 accounts over 20 years and 8 over 2 years are different
    # applicants.
    df["acct_velocity"] = _safe_divide(df["open_acc"],
                                       df["credit_hist_months"] / 12)

    # Inquiry intensity. Six inquiries against forty existing accounts is
    # routine; six against four is credit-seeking behaviour.
    df["inq_intensity"] = _safe_divide(df["inq_last_6mths"],
                                       df["total_acc"] + 1)

    return df

"""Feature derivation tests.

These matter more than they look. src/features.py runs in the pipeline and in
the scoring service, so a change here silently changes what a served applicant
is scored on. The ratios are also where division by zero can appear, and the
handling of that is a modelling decision (Missing bin, not imputation) rather
than an implementation detail.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import derive_features  # noqa: E402


def base_row(**overrides):
    row = {
        "issue_d": pd.Timestamp("2015-06-01"),
        "earliest_cr_line": "Jun-2005",
        "emp_length": "5 years",
        "revol_util": 42.0,
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "annual_inc": 60000.0,
        "loan_amnt": 12000.0,
        "dti": 15.0,
        "total_bc_limit": 20000.0,
        "revol_bal": 5000.0,
        "open_acc": 10.0,
        "total_acc": 24.0,
        "inq_last_6mths": 1.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_fico_is_band_midpoint():
    out = derive_features(base_row())
    assert out["fico"].iloc[0] == 702.0
    # endpoints are perfectly collinear, so they must not survive
    assert "fico_range_low" not in out.columns


def test_loan_to_income():
    out = derive_features(base_row(loan_amnt=15000.0, annual_inc=50000.0))
    assert out["loan_to_income"].iloc[0] == pytest.approx(0.30)


def test_dti_with_loan_exceeds_dti():
    """The published dti excludes the loan being decided on. Adding it must
    always increase the ratio, never decrease it."""
    out = derive_features(base_row(dti=15.0, loan_amnt=12000.0,
                                   annual_inc=60000.0))
    # 12000/36 = 333.33/mo against 5000/mo income = 6.67pp
    assert out["dti_with_loan"].iloc[0] == pytest.approx(21.67, abs=0.01)
    assert out["dti_with_loan"].iloc[0] > out["dti"].iloc[0]


def test_inq_intensity_uses_plus_one_denominator():
    """total_acc + 1 keeps a thin-file applicant finite rather than undefined."""
    out = derive_features(base_row(inq_last_6mths=3.0, total_acc=0.0))
    assert out["inq_intensity"].iloc[0] == pytest.approx(3.0)


def test_zero_income_becomes_missing_not_infinity():
    """Division by zero must produce NaN so binning assigns a Missing bin. An
    infinity would propagate into the WOE lookup; an imputed value would
    silently invent capacity the applicant does not have."""
    out = derive_features(base_row(annual_inc=0.0))
    assert np.isnan(out["loan_to_income"].iloc[0])
    assert not np.isinf(out["loan_to_income"].iloc[0])


def test_zero_bankcard_limit_becomes_missing():
    out = derive_features(base_row(total_bc_limit=0.0))
    assert np.isnan(out["bal_to_bc_limit"].iloc[0])


def test_emp_length_ladder_maps_to_ordinal():
    for text, expected in [("< 1 year", 0), ("1 year", 1), ("10+ years", 10)]:
        out = derive_features(base_row(emp_length=text))
        assert out["emp_length_num"].iloc[0] == expected


def test_unknown_emp_length_stays_null():
    """Nulls must not be filled - binning gives them their own weight."""
    out = derive_features(base_row(emp_length=None))
    assert pd.isna(out["emp_length_num"].iloc[0])


def test_credit_history_length():
    out = derive_features(base_row(issue_d=pd.Timestamp("2015-06-01"),
                                   earliest_cr_line="Jun-2005"))
    assert out["credit_hist_months"].iloc[0] == pytest.approx(120, abs=1)


def test_revol_util_percent_string_is_coerced():
    """Some Lending Club mirrors ship revol_util as '34.5%'."""
    df = base_row()
    df["revol_util"] = df["revol_util"].astype(object)
    df.loc[0, "revol_util"] = "34.5%"
    out = derive_features(df)
    assert out["revol_util"].iloc[0] == pytest.approx(34.5)


def test_derivation_is_row_independent():
    """No cross-row statistics anywhere, so scoring one application must give
    the same answer as scoring it inside a batch."""
    single = derive_features(base_row(loan_amnt=20000.0))
    batch = derive_features(pd.concat([base_row(loan_amnt=20000.0),
                                       base_row(loan_amnt=5000.0)],
                                      ignore_index=True))
    assert single["loan_to_income"].iloc[0] == batch["loan_to_income"].iloc[0]

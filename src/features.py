"""Feature derivation shared across the pipeline and the scoring API.

Kept in one place so that an applicant scored through the API passes through
identical transformations to those used in development.
"""

import numpy as np
import pandas as pd


def derive_features(df):
    """Construct the three characteristics that are not usable in raw form."""
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

    return df

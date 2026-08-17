"""Scoring service tests.

The invariants worth protecting are not "does it return 200". They are:

  - the points on the card sum to the score that is served
  - the score converts back to the PD it was derived from
  - the same application always gets the same answer
  - a decline always carries reasons

A regression in any of those produces a service that looks healthy and scores
wrongly, which is the failure mode that matters in credit.
"""

from pathlib import Path
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import app  # noqa: E402

STRONG = {
    "fico_range_low": 745, "fico_range_high": 749, "annual_inc": 112000,
    "loan_amnt": 9000, "dti": 9.4, "total_bc_limit": 48000,
    "avg_cur_bal": 19500, "percent_bc_gt_75": 0, "mo_sin_rcnt_tl": 26,
    "mo_sin_old_rev_tl_op": 238, "inq_last_6mths": 0, "total_acc": 31,
    "home_ownership": "MORTGAGE",
}

WEAK = {
    "fico_range_low": 620, "fico_range_high": 624, "annual_inc": 32000,
    "loan_amnt": 25000, "dti": 28.5, "total_bc_limit": 4500,
    "avg_cur_bal": 1200, "percent_bc_gt_75": 80, "mo_sin_rcnt_tl": 2,
    "mo_sin_old_rev_tl_op": 40, "inq_last_6mths": 4, "total_acc": 9,
    "home_ownership": "RENT",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# --- service ---------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["artifacts_loaded"] is True


def test_model_info_matches_the_documented_card(client):
    info = client.get("/model-info").json()
    assert len(info["characteristics"]) == 11
    assert info["n_attributes"] == 82
    assert info["scaling"] == {"base_score": 600, "base_odds": 50, "pdo": 20,
                               "factor": 28.8539, "offset": 487.1229}
    assert info["cutoff"] == 530
    assert info["reason_code_method"] == "mean"


def test_engineered_ratios_are_on_the_card(client):
    chars = client.get("/model-info").json()["characteristics"]
    for c in ("dti_with_loan", "loan_to_income", "inq_intensity"):
        assert c in chars


# --- decisions -------------------------------------------------------------

def test_strong_file_approves(client):
    d = client.post("/score", json=STRONG).json()
    assert d["decision"] == "APPROVE"
    assert d["score"] >= d["cutoff"]
    assert d["band"] >= 8
    assert d["reason_codes"] == []


def test_weak_file_declines_with_reasons(client):
    d = client.post("/score", json=WEAK).json()
    assert d["decision"] == "DECLINE"
    assert d["score"] < d["cutoff"]
    assert len(d["reason_codes"]) == 4
    assert [r["rank"] for r in d["reason_codes"]] == [1, 2, 3, 4]


def test_reasons_are_ordered_by_shortfall(client):
    """Rank 1 must be the largest gap against the population reference,
    otherwise the disclosure names the wrong principal reason."""
    d = client.post("/score", json=WEAK).json()
    gaps = [r["points_typical"] - r["points_awarded"] for r in d["reason_codes"]]
    assert gaps == sorted(gaps, reverse=True)


def test_reasons_carry_disclosure_text(client):
    d = client.post("/score", json=WEAK).json()
    for r in d["reason_codes"]:
        assert r["reason"] and r["reason"] != r["characteristic"]
        assert r["points_awarded"] <= r["points_available"]


def test_strong_outranks_weak(client):
    a = client.post("/score", json=STRONG).json()
    b = client.post("/score", json=WEAK).json()
    assert a["score"] > b["score"]
    assert a["probability_of_default"] < b["probability_of_default"]


# --- invariants ------------------------------------------------------------

def test_points_sum_to_score(client):
    """The served score must be the sum of the published attribute points. If
    these diverge, the points table no longer explains the decision."""
    for payload in (STRONG, WEAK):
        d = client.post("/score", json=payload).json()
        assert sum(d["points_breakdown"].values()) == d["score"]


def test_score_reconciles_to_pd(client):
    """Inverting score = offset + factor * ln(odds_good) must recover the
    score the model implies.

    Tolerance is in score points rather than PD because PD sensitivity varies
    hugely across the range - a point is worth ~0.009 PD at the cutoff and
    ~0.001 at the top of the card. The card awards integer points per
    attribute, so 11 attributes accumulate up to +/-5.5 points of rounding in
    the worst case and ~1 point typically. Three points is the tolerance that
    matters: larger than that means the points table and the model have
    diverged, which is the failure this guards against.
    """
    info = client.get("/model-info").json()["scaling"]
    for payload in (STRONG, WEAK):
        d = client.post("/score", json=payload).json()
        p = d["probability_of_default"]
        implied_score = info["offset"] + info["factor"] * np.log((1 - p) / p)
        assert abs(implied_score - d["score"]) <= 3


def test_calibrated_pd_is_higher(client):
    """The intercept shift corrects out-of-time under-prediction, so the
    calibrated figure must exceed the raw one."""
    d = client.post("/score", json=WEAK).json()
    assert d["calibrated_probability_of_default"] > d["probability_of_default"]


def test_one_characteristic_per_point_entry(client):
    d = client.post("/score", json=STRONG).json()
    assert len(d["points_breakdown"]) == 11


def test_scoring_is_deterministic(client):
    a = client.post("/score", json=WEAK).json()
    b = client.post("/score", json=WEAK).json()
    assert a["score"] == b["score"]
    assert a["points_breakdown"] == b["points_breakdown"]
    assert ([r["characteristic"] for r in a["reason_codes"]]
            == [r["characteristic"] for r in b["reason_codes"]])


def test_batch_matches_single(client):
    """Batch scoring must not change any answer - no cross-row statistics."""
    batch = client.post("/score/batch", json=[STRONG, WEAK]).json()
    single = [client.post("/score", json=p).json() for p in (STRONG, WEAK)]
    assert [b["score"] for b in batch] == [s["score"] for s in single]


def test_higher_loan_never_raises_the_score(client):
    """Monotonicity the card must respect: on identical income and file, asking
    for more cannot make an applicant look better."""
    small = client.post("/score", json={**WEAK, "loan_amnt": 6000}).json()
    large = client.post("/score", json={**WEAK, "loan_amnt": 30000}).json()
    assert large["score"] <= small["score"]


# --- validation ------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    {"fico_range_low": 900},          # above the FICO range
    {"annual_inc": -1},               # negative income
    {"loan_amnt": 0},                 # zero loan
    {"percent_bc_gt_75": 140},        # above 100%
    {"home_ownership": "CASTLE"},     # not an accepted category
])
def test_invalid_input_is_rejected(client, bad):
    r = client.post("/score", json={**STRONG, **bad})
    assert r.status_code == 422


def test_missing_field_is_rejected(client):
    payload = {k: v for k, v in STRONG.items() if k != "fico_range_low"}
    assert client.post("/score", json=payload).status_code == 422


def test_batch_size_is_capped(client):
    r = client.post("/score/batch", json=[STRONG] * 1001)
    assert r.status_code == 413


def test_application_id_is_echoed(client):
    d = client.post("/score", json={**STRONG, "application_id": "APP-42"}).json()
    assert d["application_id"] == "APP-42"

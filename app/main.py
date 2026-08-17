"""
Credit Risk Scorecard - scoring service.

Run locally:  uvicorn app.main:app --reload
Docs:         http://localhost:8000/docs

The service loads the same artefacts the pipeline produced and routes every
application through `src/features.py` and the fitted `BinningProcess`. An
applicant scored here therefore passes through byte-identical feature
derivation and bin edges to those used in development - which is the property
that makes a served score defensible.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
import json
import pickle
import sys

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import derive_features  # noqa: E402

MODELS = Path("models")
STATIC = Path(__file__).resolve().parent / "static"

# Score bands, cut on the development distribution. Band 1 is the riskiest.
BAND_EDGES = [-np.inf, 523, 530, 535, 540, 544, 548, 553, 560, 569, np.inf]

ARTIFACTS = {}


def load_artifacts():
    with open(MODELS / "binning_process.pkl", "rb") as f:
        bp = pickle.load(f)["binning_process"]
    with open(MODELS / "logit_model.pkl", "rb") as f:
        obj = pickle.load(f)
    with open(MODELS / "scorecard.pkl", "rb") as f:
        card = pickle.load(f)
    with open(MODELS / "reason_reference.json") as f:
        ref = json.load(f)
    try:
        with open(MODELS / "recalibration.pkl", "rb") as f:
            shift = pickle.load(f)["intercept_shift"]
    except FileNotFoundError:
        shift = 0.0

    return {
        "bp": bp,
        "model": obj["model"],
        "features": obj["features"],
        "card": card,
        "ref": ref,
        "intercept_shift": float(shift),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    ARTIFACTS.update(load_artifacts())
    yield
    ARTIFACTS.clear()


app = FastAPI(
    title="Credit Risk Scorecard",
    description="Bank-style PD scorecard: score, probability of default, "
                "risk band and ECOA adverse action reason codes.",
    version="1.0.0",
    lifespan=lifespan,
)


class Application(BaseModel):
    """Application-time fields. Only characteristics on the scorecard are
    required; the rest of the fitted schema is filled with nulls, which the
    binning assigns to Missing bins."""

    fico_range_low: float = Field(..., ge=300, le=850, examples=[680])
    fico_range_high: float = Field(..., ge=300, le=850, examples=[684])
    annual_inc: float = Field(..., ge=0, examples=[65000])
    loan_amnt: float = Field(..., gt=0, examples=[12000])
    dti: float = Field(..., ge=0, examples=[18.5])
    total_bc_limit: float = Field(..., ge=0, examples=[21000])
    avg_cur_bal: float = Field(..., ge=0, examples=[8400])
    percent_bc_gt_75: float = Field(..., ge=0, le=100, examples=[25.0])
    mo_sin_rcnt_tl: float = Field(..., ge=0, examples=[8])
    mo_sin_old_rev_tl_op: float = Field(..., ge=0, examples=[142])
    inq_last_6mths: float = Field(..., ge=0, examples=[1])
    total_acc: float = Field(..., ge=0, examples=[24])
    home_ownership: Literal["MORTGAGE", "RENT", "OWN", "OTHER", "NONE", "ANY"] = (
        Field(..., examples=["MORTGAGE"]))

    application_id: Optional[str] = Field(None, examples=["APP-0001"])


class ReasonCode(BaseModel):
    rank: int
    characteristic: str
    reason: str
    points_awarded: int
    points_typical: float
    points_available: int


class ScoreResponse(BaseModel):
    application_id: Optional[str]
    score: int
    band: int
    probability_of_default: float
    calibrated_probability_of_default: float
    decision: Literal["APPROVE", "DECLINE"]
    cutoff: int
    reason_codes: list[ReasonCode]
    points_breakdown: dict[str, int]
    model_version: str


def to_frame(a: Application) -> pd.DataFrame:
    """Build a single row matching the fitted schema. Fields the card does not
    use are left null rather than imputed - the binning gives nulls their own
    weight, which is the same treatment they received in development."""
    bp = ARTIFACTS["bp"]
    raw = a.model_dump(exclude={"application_id"})

    needed = set(bp.variable_names) | {
        "earliest_cr_line", "emp_length", "revol_util", "issue_d",
        "fico_range_low", "fico_range_high", "revol_bal", "open_acc",
    }
    row = {c: np.nan for c in needed}
    row.update(raw)
    row["issue_d"] = pd.Timestamp(datetime.now(timezone.utc).date())

    df = pd.DataFrame([row])
    df["issue_d"] = pd.to_datetime(df["issue_d"])
    return derive_features(df)


def score_application(a: Application) -> ScoreResponse:
    bp, model = ARTIFACTS["bp"], ARTIFACTS["model"]
    features, card, ref = (ARTIFACTS["features"], ARTIFACTS["card"],
                           ARTIFACTS["ref"])

    df = to_frame(a)
    w = bp.transform(df[bp.variable_names], metric="woe")
    w.columns = [f"woe_{c}" for c in w.columns]

    n = len(features)
    intercept = model.params["const"]
    factor, offset = card["factor"], card["offset"]

    points = {}
    for f in features:
        c = f.replace("woe_", "")
        points[c] = int(round(
            -(model.params[f] * float(w[f].iloc[0]) + intercept / n)
            * factor + offset / n
        ))
    score = int(sum(points.values()))

    logodds = intercept + sum(model.params[f] * float(w[f].iloc[0])
                              for f in features)
    pd_raw = 1 / (1 + np.exp(-logodds))
    pd_cal = 1 / (1 + np.exp(-(logodds + ARTIFACTS["intercept_shift"])))

    band = int(np.digitize(score, BAND_EDGES[1:-1]) + 1)
    cutoff = int(ref["cutoff"])
    decision = "DECLINE" if score < cutoff else "APPROVE"

    # Reasons are ranked by points forgone against the population reference -
    # arithmetic on the published table, not a separate explanation model.
    reasons: list[ReasonCode] = []
    if decision == "DECLINE":
        shortfall = {c: ref["reference_points"][c] - p
                     for c, p in points.items()
                     if c in ref["reference_points"]}
        ranked = sorted(shortfall.items(), key=lambda kv: -kv[1])
        for i, (c, gap) in enumerate(ranked[:ref["n_reasons"]], start=1):
            reasons.append(ReasonCode(
                rank=i,
                characteristic=c,
                reason=ref["reason_text"].get(c, c),
                points_awarded=points[c],
                points_typical=round(ref["reference_points"][c], 1),
                points_available=int(ref["max_points"][c]),
            ))

    return ScoreResponse(
        application_id=a.application_id,
        score=score,
        band=band,
        probability_of_default=round(float(pd_raw), 6),
        calibrated_probability_of_default=round(float(pd_cal), 6),
        decision=decision,
        cutoff=cutoff,
        reason_codes=reasons,
        points_breakdown=points,
        model_version=app.version,
    )


@app.get("/", include_in_schema=False)
def index():
    """Demo console. The OpenAPI docs live at /docs."""
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "artifacts_loaded": bool(ARTIFACTS)}


@app.get("/model-info")
def model_info():
    if not ARTIFACTS:
        raise HTTPException(503, "artifacts not loaded")
    card, ref = ARTIFACTS["card"], ARTIFACTS["ref"]
    return {
        "version": app.version,
        "characteristics": [f.replace("woe_", "")
                            for f in ARTIFACTS["features"]],
        "n_attributes": len(card["points"]),
        "scaling": {"base_score": card["base_score"],
                    "base_odds": card["base_odds"], "pdo": card["pdo"],
                    "factor": round(card["factor"], 4),
                    "offset": round(card["offset"], 4)},
        "cutoff": ref["cutoff"],
        "reason_code_method": ref["method"],
        "intercept_shift_applied": round(ARTIFACTS["intercept_shift"], 4),
        # Exposed so the console can render points awarded against what was
        # attainable, rather than against an arbitrary scale.
        "max_points": ref["max_points"],
        "reference_points": ref["reference_points"],
        "reason_text": ref["reason_text"],
    }


@app.post("/score", response_model=ScoreResponse)
def score(application: Application):
    if not ARTIFACTS:
        raise HTTPException(503, "artifacts not loaded")
    try:
        return score_application(application)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"scoring failed: {e}")


@app.post("/score/batch", response_model=list[ScoreResponse])
def score_batch(applications: list[Application]):
    if not ARTIFACTS:
        raise HTTPException(503, "artifacts not loaded")
    if len(applications) > 1000:
        raise HTTPException(413, "batch limited to 1000 applications")
    return [score_application(a) for a in applications]

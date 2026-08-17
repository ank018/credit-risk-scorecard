"""
Project 1 - Credit Risk Scorecard
Step 11: MLflow experiment tracking and model registry.

Run:  python src/13_mlflow_tracking.py
      mlflow ui            # then open http://localhost:5000

In:   reports/*.csv, models/*.pkl
Out:  mlruns/  (local tracking store)
      registered model "credit-scorecard-champion"

Two things happen here, and they are different in kind.

REGISTRY. The champion is packaged as an MLflow pyfunc model with its binning
process, coefficients and scaling bundled as artifacts. The result is loadable
with mlflow.pyfunc.load_model() and scores raw application data end to end -
which is what makes the registered version a deployable artefact rather than a
record that a model once existed.

TRACKING. Runs for the champion, the challenger, the five ablation variants and
the fifty Optuna trials are logged from the results persisted in reports/,
rather than by re-executing them under an active run. This is a deliberate
choice and worth being plain about: those runs cost roughly forty minutes of
compute and their outputs are already committed, so re-running them under
tracking would produce identical numbers at real cost. In a pipeline being
developed rather than reconstructed, mlflow.start_run() belongs inside each
training script. The metrics logged here are the ones those scripts printed and
wrote to disk; nothing is recomputed or estimated.
"""

from pathlib import Path
import json
import pickle
import sys

import numpy as np
import pandas as pd
import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from config import PROCESSED, REPORTS, MODELS

EXPERIMENT = "credit-risk-scorecard"
REGISTERED_NAME = "credit-scorecard-champion"
FIGS = REPORTS / "figures"


# --------------------------------------------------------------------------
# pyfunc wrapper
# --------------------------------------------------------------------------

class ScorecardModel(mlflow.pyfunc.PythonModel):
    """Serves the champion from raw application fields.

    The wrapper deliberately reuses src/features.py and the fitted
    BinningProcess rather than reimplementing either. A registered model that
    recomputes its own features is a second implementation waiting to diverge
    from the first.
    """

    def load_context(self, context):
        # features.py is shipped via code_paths, which MLflow prepends to
        # sys.path at load time - so this is a plain import, and it resolves to
        # the exact file that was logged rather than whatever happens to be on
        # the loading machine.
        from features import derive_features
        self._derive = derive_features

        with open(context.artifacts["binning"], "rb") as f:
            self.bp = pickle.load(f)["binning_process"]
        with open(context.artifacts["logit"], "rb") as f:
            obj = pickle.load(f)
        self.model, self.features = obj["model"], obj["features"]
        with open(context.artifacts["card"], "rb") as f:
            self.card = pickle.load(f)

    def predict(self, context, model_input, params=None):
        df = model_input.copy()
        # Numeric coercion happens here rather than through a logged signature.
        # Callers reasonably supply counts as integers and money as either, and
        # widening to float is lossless.
        for c in df.columns:
            if c != "home_ownership":
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        if "issue_d" not in df:
            df["issue_d"] = pd.Timestamp("2015-06-01")
        df["issue_d"] = pd.to_datetime(df["issue_d"])
        for c in set(self.bp.variable_names) | {
                "earliest_cr_line", "emp_length", "revol_util", "revol_bal",
                "open_acc", "fico_range_low", "fico_range_high"}:
            if c not in df:
                df[c] = np.nan

        df = self._derive(df)
        w = self.bp.transform(df[self.bp.variable_names], metric="woe")
        w.columns = [f"woe_{c}" for c in w.columns]

        n = len(self.features)
        b0 = self.model.params["const"]
        factor, offset = self.card["factor"], self.card["offset"]

        score = np.zeros(len(df))
        logodds = np.full(len(df), b0)
        for f in self.features:
            v = w[f].astype(float).values
            # Rounded per attribute, matching the published points table and the
            # scoring service. Summing unrounded contributions and rounding once
            # differs by a point or so, and the points table is the artefact of
            # record - two representations of the same model must not disagree.
            score += np.round(-(self.model.params[f] * v + b0 / n) * factor
                              + offset / n)
            logodds += self.model.params[f] * v

        return pd.DataFrame({
            "score": score.astype(int),
            "probability_of_default": 1 / (1 + np.exp(-logodds)),
        })


# --------------------------------------------------------------------------

def read(name, **kw):
    p = REPORTS / name
    return pd.read_csv(p, **kw) if p.exists() else None


def log_figures(names):
    for n in names:
        p = FIGS / n
        if p.exists():
            mlflow.log_artifact(str(p), artifact_path="figures")


def log_champion():
    perf = read("performance_summary.csv").set_index("split")
    coef = read("coefficients.csv")
    with open(MODELS / "scorecard.pkl", "rb") as f:
        card = pickle.load(f)
    with open(MODELS / "selected_features.json") as f:
        sel = json.load(f)

    with mlflow.start_run(run_name="champion-scorecard") as run:
        mlflow.set_tags({"model_type": "logistic_regression",
                         "role": "champion", "interpretable": "true"})
        mlflow.log_params({
            "n_characteristics": len(card["features"]),
            "n_attributes": len(card["points"]),
            "iv_floor": 0.02, "iv_ceiling": 0.50,
            "min_bin_size": 0.05, "max_n_bins": 8,
            "corr_threshold": sel["corr_threshold"],
            "vif_threshold": sel["vif_threshold"],
            "base_score": card["base_score"], "base_odds": card["base_odds"],
            "pdo": card["pdo"],
            "engineered_ratios": "dti_with_loan,loan_to_income,inq_intensity",
        })
        for split in ("train", "test", "oot"):
            r = perf.loc[split]
            for m in ("gini", "ks", "auc", "brier"):
                mlflow.log_metric(f"{split}_{m}", float(r[m]))
        mlflow.log_metric("oot_calibration_gap",
                          float(perf.loc["oot", "bad_rate"]
                                - perf.loc["oot", "mean_pred_pd"]))

        recal = read("recalibration.csv")
        if recal is not None:
            mlflow.log_metric("oot_gap_after_recalibration",
                              float(recal.loc[1, "gap"]))
        lift = read("incremental_lift.csv")
        if lift is not None:
            mlflow.log_metric("lc_grade_gini", float(lift.loc[0, "oot_gini"]))
            mlflow.log_metric("incremental_gini_over_grade",
                              float(lift.loc[2, "lift_vs_grade"]))

        for f in ("points_table.csv", "coefficients.csv", "iv_table.csv",
                  "vif_table.csv", "score_bands.csv", "calibration.csv"):
            if (REPORTS / f).exists():
                mlflow.log_artifact(str(REPORTS / f), artifact_path="tables")
        log_figures(["roc.png", "ks.png", "calibration.png",
                     "bad_rate_by_band.png", "score_distribution.png"])

        # Coefficient signs are a pass/fail gate, so record the outcome as a
        # metric rather than leaving it in a CSV nobody opens.
        mlflow.log_metric("coefficients_sign_ok", float(coef["sign_ok"].all()))

        sample = pd.DataFrame([{
            "fico_range_low": 680, "fico_range_high": 684,
            "annual_inc": 65000, "loan_amnt": 12000, "dti": 18.5,
            "total_bc_limit": 21000, "avg_cur_bal": 8400,
            "percent_bc_gt_75": 25.0, "mo_sin_rcnt_tl": 8,
            "mo_sin_old_rev_tl_op": 142, "inq_last_6mths": 1,
            "total_acc": 24, "home_ownership": "MORTGAGE",
        }])

        kw = dict(
            python_model=ScorecardModel(),
            artifacts={
                "binning": str(MODELS / "binning_process.pkl"),
                "logit": str(MODELS / "logit_model.pkl"),
                "card": str(MODELS / "scorecard.pkl"),
            },
            # Ships the feature derivation with the model and makes it
            # importable at load time. The registered model therefore carries
            # the same transformation code as the pipeline and the API - three
            # consumers, one implementation.
            code_paths=[str(Path(__file__).parent / "features.py")],
            # No signature is logged. MLflow enforces declared dtypes *before*
            # predict() runs, and rejects int64 where a double is declared - so
            # any payload mixing integers and floats fails on whichever column
            # was typed the other way, regardless of how the schema is written.
            # The model coerces numerics itself, and the FastAPI service, which
            # is the actual serving path, validates types *and* ranges through
            # Pydantic. The cost is no type checking at the registry boundary.
            input_example=sample,
            registered_model_name=REGISTERED_NAME,
        )
        try:
            mlflow.pyfunc.log_model(name="scorecard", **kw)
        except TypeError:            # mlflow < 3
            mlflow.pyfunc.log_model(artifact_path="scorecard", **kw)

        print(f"  champion logged: {run.info.run_id}")
        return run.info.run_id


def log_challenger():
    comp = read("challenger_comparison.csv")
    if comp is None:
        return
    chal = comp[comp["model"].str.contains("XGBoost")].set_index("split")
    with open(MODELS / "xgb_params.json") as f:
        params = json.load(f)

    with mlflow.start_run(run_name="challenger-xgboost"):
        mlflow.set_tags({"model_type": "xgboost", "role": "challenger",
                         "interpretable": "false"})
        mlflow.log_params({**params["tuned"], "n_features": 41,
                           "tuning": "optuna", "n_trials": params["n_trials"]})
        for split in chal.index:
            for m in ("gini", "ks", "brier"):
                mlflow.log_metric(f"{split}_{m}", float(chal.loc[split, m]))
        mlflow.log_metric("oot_calibration_gap",
                          float(chal.loc["oot", "cal_gap"]))

        champ = comp[comp["model"].str.contains("scorecard")].set_index("split")
        mlflow.log_metric("oot_gini_lift_vs_champion",
                          float(chal.loc["oot", "gini"]
                                - champ.loc["oot", "gini"]))
        for f in ("shap_importance.csv", "challenger_comparison.csv"):
            if (REPORTS / f).exists():
                mlflow.log_artifact(str(REPORTS / f), artifact_path="tables")
        log_figures(["shap_beeswarm.png", "shap_bar.png",
                     "roc_champion_challenger.png"])
        print("  challenger logged")


def log_ablation():
    abl = read("ablation.csv")
    if abl is None:
        return
    with mlflow.start_run(run_name="ablation-study"):
        mlflow.set_tags({"role": "analysis", "study": "ablation"})
        for _, r in abl.iterrows():
            with mlflow.start_run(run_name=r["variant"], nested=True):
                mlflow.log_params({"n_features": int(r["n_features"]),
                                   "best_iteration": int(r["best_iteration"])})
                mlflow.log_metric("test_gini", float(r["test_gini"]))
                mlflow.log_metric("oot_gini", float(r["oot_gini"]))
                mlflow.log_metric("lift_vs_champion",
                                  float(r["lift_vs_champion"]))
        a = abl.set_index("variant")["oot_gini"]
        mlflow.log_metric("geography_contribution",
                          float(a.iloc[0] - a.iloc[1]))
        mlflow.log_metric("functional_form_contribution",
                          float(a.iloc[3] - abl.loc[3, "oot_gini"]
                                + abl.loc[3, "lift_vs_champion"]))
        mlflow.log_artifact(str(REPORTS / "ablation.csv"),
                            artifact_path="tables")
        log_figures(["ablation.png"])
        print(f"  ablation logged: {len(abl)} variants")


def log_tuning():
    trials = read("optuna_trials.csv")
    if trials is None:
        return
    pcols = [c for c in trials.columns if c.startswith("params_")]
    with mlflow.start_run(run_name="optuna-search"):
        mlflow.set_tags({"role": "analysis", "study": "hyperparameter_search"})
        mlflow.log_param("n_trials", len(trials))
        mlflow.log_param("objective", "validation AUC")
        best = trials["value"].max()
        mlflow.log_metric("best_val_gini", float(2 * best - 1))
        for _, t in trials.iterrows():
            if pd.isna(t["value"]):
                continue
            with mlflow.start_run(run_name=f"trial-{int(t['number']):03d}",
                                  nested=True):
                mlflow.log_params({c.replace("params_", ""): t[c]
                                   for c in pcols})
                mlflow.log_metric("val_gini", float(2 * t["value"] - 1))
        mlflow.log_artifact(str(REPORTS / "optuna_trials.csv"),
                            artifact_path="tables")
        log_figures(["optuna_history.png"])
        print(f"  tuning logged: {len(trials)} trials")


def log_monitoring():
    psi = read("psi_score.csv")
    if psi is None:
        return
    chars = read("psi_characteristics.csv", index_col=0)
    with mlflow.start_run(run_name="psi-monitoring"):
        mlflow.set_tags({"role": "monitoring"})
        mlflow.log_param("baseline", "dev 2013-14")
        mlflow.log_param("psi_thresholds", "0.10 monitor / 0.25 action")
        # PSI over time is a series, so step-indexed metrics rather than one
        # value per period.
        for i, r in psi.iterrows():
            mlflow.log_metric("score_psi", float(r["psi"]), step=i)
            mlflow.log_metric("mean_predicted_pd", float(r["mean_pd"]), step=i)
            mlflow.log_metric("pct_below_cutoff",
                              float(r["pct_below_530"]), step=i)
        if chars is not None:
            last = chars.columns[-1]
            for c, v in chars[last].items():
                mlflow.log_metric(f"psi_{c}", float(v))
            mlflow.log_metric("max_characteristic_psi",
                              float(chars[last].max()))
        for f in ("psi_score.csv", "psi_characteristics.csv",
                  "psi_offcard.csv"):
            if (REPORTS / f).exists():
                mlflow.log_artifact(str(REPORTS / f), artifact_path="tables")
        log_figures(["psi_heatmap.png", "psi_trend.png", "score_shift.png"])
        print("  monitoring logged")


if __name__ == "__main__":
    mlflow.set_experiment(EXPERIMENT)
    print(f"experiment: {EXPERIMENT}\n")

    run_id = log_champion()
    log_challenger()
    log_ablation()
    log_tuning()
    log_monitoring()

    print(f"\nregistered: {REGISTERED_NAME}")
    print("\nload the champion with:")
    print(f'  m = mlflow.pyfunc.load_model("models:/{REGISTERED_NAME}/latest")')
    print("  m.predict(applications_df)")
    print("\nstart the UI with:  mlflow ui")

# Champion vs Challenger: XGBoost and the Interpretability Trade-off

Stage 8 of the build. A gradient boosted challenger is fitted against the
scorecard champion, attributed with SHAP, and the deployment decision argued on
the evidence.

Script: `src/08_challenger.py`.

---

## 1. Experimental design

The comparison is constructed so the challenger is not handicapped:

| | Champion | Challenger |
|---|---|---|
| Model | Logistic regression | XGBoost |
| Features | 12 WOE-transformed characteristics | **35 raw candidates** |
| Feature selection | IV screen + correlation + VIF | None — model selects |
| Missing values | WOE missing bin | Native handling |
| Categoricals | WOE by event rate | Native categorical support |

The challenger receives **every candidate characteristic**, including the 20 the IV
screen discarded and the 2 dropped for collinearity, on raw untransformed values.
Restricting it to the champion's pre-selected, pre-binned inputs would have made
the contest meaningless.

`grade`, `sub_grade` and `int_rate` are withheld from **both** models, so neither
has access to Lending Club's own risk assessment.

### Early stopping and split hygiene

An initial run used the test partition as the early-stopping `eval_set`, which
selects the tree count by test performance and makes the reported test Gini
optimistic — and unfair to the champion, which never saw test at any point.

This was corrected: a 15% validation slice is carved out of train for early
stopping, leaving test and out-of-time genuinely held out for both models.

The effect was small — challenger test Gini moved 0.3533 → 0.3513, and best
iteration 463 → 334 — but the corrected design is the one reported. Note that the
champion was fitted on train + validation while the challenger saw only train,
an asymmetry that runs *against* the challenger; the lift below is therefore
conservative.

**Only `test` and `oot` are quotable.** `train_fit` is in-sample for the
challenger, `val` selected its tree count, and both are in-sample for the
champion.

---

## 2. Results

| Split | Model | Gini | KS | Brier | Calibration gap |
|---|---|---:|---:|---:|---:|
| train_fit | Champion | 0.3102 | 0.2273 | 0.1106 | −0.0001 |
| train_fit | Challenger | 0.4900 | 0.3602 | 0.1038 | −0.0000 |
| val | Champion | 0.2913 | 0.2123 | 0.1110 | +0.0005 |
| val | Challenger | 0.3323 | 0.2372 | 0.1100 | +0.0010 |
| **test** | Champion | 0.2927 | 0.2107 | 0.1110 | −0.0002 |
| **test** | Challenger | **0.3513** | 0.2574 | 0.1094 | +0.0001 |
| **oot** | Champion | 0.3089 | 0.2233 | 0.1223 | +0.0182 |
| **oot** | Challenger | **0.3607** | 0.2613 | 0.1204 | +0.0158 |

**Out-of-time Gini lift: +0.0518** (0.3089 → 0.3607), a 17% relative improvement
in discrimination. KS improves from 22.3 to 26.1.

The lift is real and material. Any argument for shipping the champion has to
justify giving it up.

### Two observations that cut against the champion

**The challenger is better calibrated, not worse.** The expectation going in was
that raw gradient boosting probabilities would need Platt scaling or isotonic
regression. They did not: out-of-time calibration gap is +0.0158 against the
champion's +0.0182, and Brier is 0.1204 against 0.1223. Log-loss optimisation
without class weighting produces well-calibrated output on this data. The
"tree models rank well but predict badly" argument does not apply here and is not
used below.

**Both models under-predict out-of-time by a similar margin**, for the same
reason — both were fitted at a 13.19% bad rate and applied at 14.89%. This is a
population shift, not a model defect, and it affects the champion and challenger
alike.

### The observation that cuts for the champion

| Model | train_fit → test | Degradation |
|---|---|---:|
| Champion | 0.3102 → 0.2927 | 0.0175 |
| Challenger | 0.4900 → 0.3513 | **0.1387** |

The challenger fits the training sample eight times harder than it generalises.
It still generalises *better* in absolute terms, so this is not a
disqualification — but the stability profiles are entirely different, and that
difference matters for an artefact that must hold up for eighteen to twenty-four
months between rebuilds with no opportunity to observe outcomes in between.

---

## 3. SHAP attribution

Mean absolute SHAP on an 8,000-account out-of-time sample.

| Rank | Characteristic | mean \|SHAP\| | On the scorecard? | IV |
|---:|---|---:|---|---:|
| 1 | `fico` | 0.2387 | Yes | 0.1391 |
| 2 | `annual_inc` | 0.1976 | Yes | 0.0937 |
| 3 | `dti` | 0.1441 | Yes | 0.0533 |
| 4 | `loan_amnt` | 0.1338 | **No** | 0.0115 |
| 5 | `total_bc_limit` | 0.1289 | Yes | 0.0759 |
| 6 | `mo_sin_rcnt_tl` | 0.1155 | Yes | 0.0379 |
| 7 | `addr_state` | 0.1079 | **No** | 0.0154 |
| 8 | `inq_last_6mths` | 0.0935 | Yes | 0.0279 |
| 9 | `percent_bc_gt_75` | 0.0834 | Yes | 0.0380 |
| 10 | `purpose` | 0.0828 | **No** | 0.0191 |
| 11 | `mo_sin_old_rev_tl_op` | 0.0711 | Yes | 0.0375 |
| 12 | `avg_cur_bal` | 0.0686 | Yes | 0.0756 |
| 13 | `num_actv_bc_tl` | 0.0576 | **No** | 0.0044 |
| 14 | `home_ownership` | 0.0493 | Yes | 0.0411 |
| 15 | `pct_tl_nvr_dlq` | 0.0490 | **No** | 0.0008 |

Scorecard characteristics account for **63.0%** of total |SHAP|. The remaining
37% comes from characteristics the selection process discarded.

### Univariate IV is blind to conditional effects

`loan_amnt` has IV 0.0115 — dropped as too weak — and ranks **4th** by SHAP.
`pct_tl_nvr_dlq` has IV 0.0008, effectively zero, and ranks 15th.

The explanation for `loan_amnt` is the interesting one. In isolation, loan size
barely predicts default, because larger loans are extended to higher-income
borrowers: the risk of the larger amount and the quality of the borrower cancel
out. *Conditional on* income, DTI and credit limit, a larger loan is
unambiguously riskier. The signal only exists once the other characteristics are
held constant, and univariate IV screening cannot see it by construction.

This is the same phenomenon that produced the `revol_bal` sign flip in
`docs/model_development.md` §1, approached from the other direction. There, a
conditional relationship reversed a univariate one and the characteristic had to
be dropped. Here, a conditional relationship *creates* signal where the
univariate view saw none.

**The honest conclusion is that IV screening is a real limitation of the
scorecard methodology, not merely a simplification.** The ablation in §4
quantifies it: restoring `loan_amnt` alone is worth +0.0178 Gini, and the
discarded characteristics collectively account for roughly three quarters of the
challenger's advantage. Most of that is available to the champion through better
selection, not through abandoning the linear form.

---

## 4. Ablation: where the lift actually comes from

Five identically parameterised XGBoost variants, fitted on the same rows and
evaluated on the same held-out windows. Each removes one thing, so the difference
between adjacent rows attributes lift to a specific cause.

| Variant | Features | Test Gini | OOT Gini | Lift vs champion | % of headline |
|---|---:|---:|---:|---:|---:|
| Champion (scorecard) | 12 | 0.2927 | 0.3089 | — | — |
| A. Full challenger | 35 | 0.3513 | 0.3607 | +0.0518 | 100% |
| B. Minus `addr_state` | 34 | 0.3513 | **0.3612** | +0.0523 | 101% |
| C. Minus `addr_state`, `purpose` | 33 | 0.3443 | 0.3566 | +0.0477 | 92% |
| D. Champion characteristics only | 12 | 0.3087 | 0.3226 | +0.0137 | 26% |
| E. Champion + `loan_amnt` | 13 | 0.3261 | 0.3404 | +0.0316 | 61% |

### Decomposition

| Source of lift | Gini | Share |
|---|---:|---:|
| Functional form — non-linearity and interactions on the champion's own 12 inputs | +0.0137 | 26% |
| Adding `loan_amnt` alone | +0.0178 | 34% |
| The other 22 discarded characteristics | +0.0203 | 39% |
| Geography (`addr_state`) | −0.0005 | 0% |

**Approximately three quarters of the challenger's advantage is a feature
selection failure, not a functional form advantage.** The scorecard's
linear-additive structure costs 0.0137 Gini. The univariate IV screen costs
0.0381 — nearly three times as much.

### Geography contributes nothing, and that corrects an earlier claim

An earlier draft of this document argued that a material share of the
challenger's lift came from `addr_state`, and that the lift was therefore partly
unavailable to a lender after fair lending review. **The ablation does not support
that.**

Removing `addr_state` moved out-of-time Gini from 0.3607 to 0.3612 — it improved
slightly — and left test Gini unchanged to four decimal places. Despite ranking
7th of 35 by mean |SHAP|, geography adds no generalisable discrimination at all.

The gap between attribution and contribution is the lesson. **SHAP measures how
much the model used a characteristic, not whether using it helped.** The
challenger was fitting state-level pockets present in the 2013–14 development
sample that did not reappear in the 2015 window — plausibly because Lending Club's
geographic mix shifted materially as it expanded, which is exactly the kind of
drift the PSI monitor in stage 10 is built to detect.

The practical consequences:

- SHAP importance is not evidence that a characteristic earns its place. Ablation
  is. Any feature-importance ranking used to justify a model should be checked
  this way.
- The fair lending exposure is genuine as a compliance matter — a US lender would
  still need review before deploying a model that prices state of residence — but
  it is **free to remediate**. Dropping `addr_state` costs nothing, so it is not
  an argument for the champion.
- `purpose` does contribute (+0.0046). Removing both leaves **92% of the headline
  lift intact**.

> This is a portfolio exercise, not legal advice. The claim is that these
> characteristics would require fair lending review, not that any particular
> outcome of that review is certain.

### `loan_amnt`: the IV screen's most expensive mistake

Adding `loan_amnt` to the champion's 12 characteristics is worth **+0.0178 Gini** —
more than the entire non-linear machinery of variant D. One characteristic,
dropped at IV 0.0115 for falling under a 0.02 floor, carries more signal than
every interaction and non-linearity XGBoost can find across the 12 that survived.

Section 3 argued this from SHAP ranking. The ablation measures it. Univariate IV
screening cannot see conditional signal, and here that blindness is the single
largest identifiable component of the performance gap.
## 5. Reason codes: a structural difference, not a presentational one

Under ECOA / Reg B, a declined applicant must receive the specific principal
reasons for the decision. This is where the two models genuinely diverge, and it
is not a matter of explanation quality.

**The scorecard produces reasons as a by-product of its structure.** Each
applicant's score is a sum of 12 attribute points from a fixed 87-row table. The
reason for a decline is the set of characteristics where the applicant lost the
most points against the maximum attainable — arithmetic on a published table,
identical for two applicants with identical inputs, stable until the card is
rebuilt, and auditable by anyone holding the table.

**SHAP produces a local attribution, which is a different object.** It is a
per-prediction decomposition against a baseline, it depends on the background
dataset chosen, it varies between applicants with the same values on the
characteristic being explained (because it is conditional on everything else),
and it is not reproducible without the model artefact and the explainer
configuration. It is a good tool for understanding model behaviour. It is a
harder basis for a legally binding statement of principal reasons that must be
defensible to a regulator years after issuance.

Reason codes can be derived from SHAP and lenders do it. The point is that the
scorecard's reasons fall out of the artefact, while the challenger's have to be
constructed, validated and separately governed.

---

## 6. Monitoring surface

The champion's monitoring pack is complete when PSI has been computed for 12
characteristics and the score. Drift localises immediately to a named
characteristic and a named bin.

The challenger requires PSI on 35 characteristics, and the majority of its
advantage lives in **interactions between them**. There is no PSI for an
interaction. Individual feature distributions can each stay within tolerance while
the joint distribution the model relies on has shifted, and the standard
monitoring pack will not detect it.

This is the argument that carries the most weight with a model risk function. It
is not that the challenger cannot be monitored — it is that the standard early
warning apparatus is substantially less informative for it, so a problem is more
likely to be discovered through realised losses than through a monitoring flag.

---

## 7. Remediation options when drift appears

`docs/model_development.md` §3 demonstrates the champion's response to the 2015
population shift: an intercept shift of +0.1569 closed the calibration gap from
+0.0182 to 0.0000 with Gini unchanged to four decimal places. One parameter, ten
minutes, no refit, no revalidation of the characteristic set.

The challenger has no intercept to shift. Options are a post-hoc calibration layer
(Platt or isotonic) fitted on recent outcomes — which adds a second model artefact
with its own governance and drift behaviour — or a full refit, which means
retraining, revalidation, fresh SHAP analysis, fresh fair lending review and a new
approval cycle.

For a model that will need periodic recalibration as vintages shift, this
difference compounds across the model's life.

---

## 8. What would actually be deployed

**Champion in production, challenger in shadow — and a scorecard rebuild with a
corrected selection stage.** The ablation changes the emphasis of this section
substantially.

### What the ablation settled

The comfortable version of the champion's case was that the challenger's lift
depends on characteristics a lender cannot use and interactions nobody can
monitor. Only the second half survives. Geography contributes nothing, and 92% of
the lift remains after removing every fair-lending-exposed characteristic.

The uncomfortable version is the accurate one: **the scorecard's real deficit is
in feature selection, not in functional form.** Being linear and additive costs
0.0137 Gini. The univariate IV screen costs 0.0381.

That is a solvable problem, and it does not require giving up the scorecard.

### The case for the scorecard as the decisioning model

1. Reason codes fall out of the artefact rather than requiring a parallel
   explanation system (§5).
2. Monitoring localises drift to a named characteristic and bin; the challenger's
   interaction-driven advantage is not observable through PSI (§6).
3. Recalibration is a one-parameter intercept shift, demonstrated to preserve
   discrimination exactly (§7).
4. Coefficient signs are individually checkable against credit intuition — the
   check that caught `revol_bal`.
5. The functional-form gap is small. XGBoost given exactly the champion's inputs
   reaches 0.3226 against 0.3089: a linear-additive scorecard captures most of
   what is available from those 12 characteristics.

### The case against, stated fairly

1. It gives up 0.0518 Gini out-of-time, which is material on a large book.
2. The challenger is **better calibrated**, not worse.
3. The fair lending argument does not do the work it appeared to. Dropping
   `addr_state` is free.
4. Roughly 0.038 of the gap is attributable to a selection process that discards
   conditionally useful characteristics, and that is a methodological weakness
   rather than a deliberate trade-off.

### The recommended action

The ablation points at a specific remediation rather than a choice between two
models:

1. **Reinstate `loan_amnt`** and re-run the selection stage. Worth roughly
   +0.018 Gini, retaining the points table, the reason codes and the intercept
   recalibration path.
2. **Replace univariate IV screening with a multivariate selection step** —
   forward selection on WOE features against out-of-time performance, or using
   challenger SHAP rankings as a candidate generator with each addition validated
   by ablation rather than by importance.
3. **Verify every addition with an ablation, not an importance ranking.**
   `addr_state` ranked 7th by SHAP and contributed nothing; importance rankings
   are not evidence of contribution.
4. **Run the challenger in shadow.** It scores every application alongside the
   champion without making decisions. Cases where the two disagree sharply are
   where the champion's specification is failing, and they form the requirement
   list for the next rebuild.

A scorecard rebuilt this way would plausibly close half the gap while keeping
every operational property that makes it deployable. **That is a better outcome
than either shipping the challenger or defending the champion as it stands** —
and it is what the evidence supports.

### The summary worth remembering

The textbook answer is that banks ship logistic regression because it is
explainable. The evidence here gives a more specific and less flattering account:
the scorecard's linear form costs very little, its selection methodology costs a
great deal, and the challenger's most-attributed characteristic contributed
nothing at all. The interpretability argument for the champion holds — but it
holds on monitoring and remediation grounds, not because the challenger's lift is
illusory or unusable.

## 9. Limitations

1. **The recommended scorecard rebuild has not been carried out.** Reinstating
   `loan_amnt` and replacing the univariate IV screen is argued in §8 and
   supported by the ablation, but the rebuilt card is not in this repository. The
   champion's reported 0.309 Gini is therefore a floor for what the methodology
   can achieve, not a ceiling.
2. **Ablation variants were not repeated across seeds.** Differences below roughly
   0.005 Gini — including the −0.0005 attributed to geography — are within the
   range that seed variation could plausibly produce. The conclusion that
   `addr_state` contributes nothing is safe; a precise point estimate for it is
   not.
3. **Hyperparameters were set by judgement, not searched.** `max_depth=5`,
   `min_child_weight=50`, `learning_rate=0.05` were chosen to be conservative on
   noisy 13%-event-rate data. A tuned challenger would likely do better, making
   the reported lift a floor rather than a ceiling.
4. **One out-of-time window.** Whether the lift persists across multiple vintages
   is untested, and it is precisely the question that matters for the shadow
   deployment argument.
5. **SHAP computed on an 8,000-account sample** rather than the full 283,026
   out-of-time population, for tractability.

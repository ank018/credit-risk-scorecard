# Model Development, Scaling and Validation

Stages 5 to 7 of the build: logistic regression champion, conversion to a
points-based scorecard, and the validation pack.

Scripts: `src/04_logistic_regression.py`, `src/05_scorecard.py`,
`src/06_evaluation.py`, `src/07_benchmark_and_calibration.py`.

Development sample: 184,094 accounts (train), 78,898 (test), 283,026
(out-of-time). Bad definition and window design in `docs/target_definition.md`;
characteristic construction and selection in `docs/feature_selection.md`.

---

## 1. Champion model

Logistic regression on WOE-transformed characteristics, fitted with `statsmodels`
rather than `scikit-learn`. Two reasons: standard errors and Wald p-values are
required for the coefficient table, and `sklearn.LogisticRegression` applies L2
regularisation by default, which shrinks coefficients and weakens the sign check
below.

### The coefficient sign check

A characteristic that predicts one way on its own and the opposite way inside the
model has had its effect redistributed by correlation with other characteristics.
Such a coefficient cannot be defended at model review regardless of how well the
model fits, so the check is a gate, not a diagnostic: each coefficient's sign is
compared against that characteristic's univariate direction on the training
sample, and disagreements are dropped and the model refitted.

**Two characteristics failed: `revol_bal` and `revol_util`.**

| Characteristic | Univariate | In model | Coefficient | Std err |
|---|---|---|---:|---:|
| `revol_bal` | − | **+** | +0.2242 | 0.0992 |
| `revol_util` | − | **+** | +0.1415 | 0.0780 |

This is a suppression effect with a clear credit interpretation. Univariately, a
larger revolving balance and higher utilisation both signal higher risk. But the
model also contains `bal_to_bc_limit` — the ratio of balance to bankcard limit —
alongside `revol_bal` and `total_bc_limit`, its two components. Conditional on the
ratio and the limit, a large balance implies a large available line, which
indicates capacity rather than distress, and the sign reverses.

**This was predicted before fitting.** `docs/feature_selection.md` §4 flagged that
`revol_bal`, `total_bc_limit` and `bal_to_bc_limit` form an *X, Y, X/Y* structure
that VIF cannot detect, because VIF is a linear diagnostic and a ratio is not a
linear function of its components. All three had VIF below 5 (3.81, 4.48, 3.37)
and the instability appeared anyway.

The effect is real, not a coding error, and it would have passed silently on any
discrimination metric.

### Why the drop order matters

Sign flips are removed first and the model refitted; only then is the significance
screen applied. Correlated characteristics cannot be estimated precisely while
both are present, so a characteristic that looks insignificant alongside a flipped
one may become significant once it is gone.

The order mattered in the earlier build, where `revol_util` moved from p = 0.486
to p = 0.018 after `revol_bal` was dropped. In this build, removing both flips
left `mort_acc` (p = 0.403), `bal_to_bc_limit` (p = 0.099) and `loan_amnt`
(p = 0.906) still insignificant, and all three were then dropped.

### `loan_amnt` at p = 0.906

`loan_amnt` was **force-kept past the IV screen** on ablation evidence — it scored
IV 0.0115, below the 0.02 floor, but earlier ablation had valued it at +0.0178
Gini (`docs/feature_selection.md` §3).

The multivariate stage rejected it at p = 0.906.

Both decisions were correct, and the sequence is the point. Univariate IV was
wrong to reject loan size on its own terms, because the signal only exists
conditional on income. But once `loan_to_income` is in the model, that conditional
signal is already captured and the raw amount adds nothing. **The ratio absorbed
it entirely.** Subsequent ablation confirms this directly: adding `loan_amnt` to
the final characteristic set is now worth +0.0002 Gini, against +0.0178 before the
ratios existed.

The whole revolving-utilisation family also collapsed to a single survivor —
`revol_bal`, `revol_util` and `bal_to_bc_limit` all fell, leaving
`percent_bc_gt_75` to cover utilisation intensity alone.

### Final coefficients

Fitted on 184,094 training accounts. All 11 coefficients agree with their
univariate direction; all significant at α = 0.05.

| Characteristic | Coefficient | Std err | p-value |
|---|---:|---:|---:|
| `mo_sin_rcnt_tl` | −0.8148 | 0.0389 | < 0.0001 |
| `dti_with_loan` | −0.7010 | 0.0318 | < 0.0001 |
| `inq_intensity` | −0.6973 | 0.0379 | < 0.0001 |
| `fico` | −0.6363 | 0.0241 | < 0.0001 |
| `loan_to_income` | −0.5347 | 0.0415 | < 0.0001 |
| `total_bc_limit` | −0.4325 | 0.0330 | < 0.0001 |
| `percent_bc_gt_75` | −0.4239 | 0.0410 | < 0.0001 |
| `mo_sin_old_rev_tl_op` | −0.4005 | 0.0380 | < 0.0001 |
| `avg_cur_bal` | −0.3739 | 0.0365 | < 0.0001 |
| `home_ownership` | −0.2778 | 0.0437 | < 0.0001 |
| `annual_inc` | −0.1877 | 0.0306 | < 0.0001 |
| *intercept* | −1.8857 | | |

McFadden pseudo R² = 0.0474, log-likelihood = −68,392.6.

**Three of the eleven are engineered ratios**, and they occupy the 2nd, 3rd and 5th
largest coefficients. All coefficients are negative because WOE is oriented so a
higher weight indicates a lower-risk bin; the uniform sign is the expected result
for a correctly specified scorecard.

The low pseudo R² is characteristic of credit scoring on an accepted-applicant
population — see `docs/feature_selection.md` §3 on population truncation.

---

## 2. Scorecard scaling

The fitted model is converted to points so that the deployed artefact is a table
rather than a set of coefficients.

| Parameter | Value |
|---|---|
| Base score | 600 |
| Base odds | 50:1 good:bad at the base score |
| PDO (points to double the odds) | 20 |
| Derived factor | 28.8539 |
| Derived offset | 487.1229 |

```
factor = PDO / ln(2)
offset = base_score − factor × ln(base_odds)

points_i = −(β_i × WOE_i + intercept/n) × factor + offset/n
score    = Σ points_i = offset + factor × ln(odds_good)
```

The intercept and offset are distributed evenly across the *n* characteristics so
per-attribute points sum to the total score. The leading negative converts the
model's log-odds of *bad* into odds of *good*, so a higher score indicates a
better applicant.

These parameters are a **presentation choice, not a modelling one**. They change
the numbers printed on the card and nothing about rank ordering or predicted PD.
The value of the PDO convention is that score differences carry constant meaning:
the gap from 640 to 660 represents the same change in odds as 700 to 720.

**Resulting card: 82 attributes across 11 characteristics.**

### Reconciliation

Each score is converted back to an implied PD and compared against the model's own
prediction:

```
odds_good  = exp((score − offset) / factor)
implied_PD = 1 / (1 + odds_good)
```

Maximum absolute discrepancy: **0.004157**, consistent with rounding points to
whole numbers. This guards against the failure mode where the points table and the
model silently diverge, producing a card that looks correct and scores incorrectly.

### Score contribution by characteristic

| Characteristic | Min | Max | Spread |
|---|---:|---:|---:|
| `fico` | 42 | 71 | 29 |
| `avg_cur_bal` | 33 | 57 | 24 |
| `dti_with_loan` | 39 | 57 | 18 |
| `mo_sin_rcnt_tl` | 42 | 59 | 17 |
| `total_bc_limit` | 45 | 59 | 14 |
| `inq_intensity` | 41 | 52 | 11 |
| `loan_to_income` | 43 | 52 | 9 |
| `mo_sin_old_rev_tl_op` | 44 | 52 | 8 |
| `annual_inc` | 46 | 52 | 6 |
| `percent_bc_gt_75` | 47 | 53 | 6 |
| `home_ownership` | 48 | 51 | 3 |

No single characteristic dominates. This matters for stability: where one
characteristic drives most of the score, drift in that one feature moves the whole
book. Every characteristic on the card moves the score by at least 3 points, so
none is carrying data-collection and monitoring cost without being able to affect
a decision.

**`annual_inc` fell from 15 points of spread to 6** when the ratios were
introduced. Income now enters the card largely through `loan_to_income` rather
than standing alone, which is the more sensible credit representation: what matters
is income relative to the obligation, not income in isolation. This is the same
absorption effect that eliminated `loan_amnt`, operating partially rather than
completely.

### Score distribution

| Split | Median | Range |
|---|---:|---|
| Train | 544 | 489 – 616 |
| Test | 544 | 492 – 616 |
| OOT | 544 | 487 – 616 |

The distribution sits entirely below the 600 base score, because base odds were
set at 50:1 while the development population runs at approximately 6.6:1. This is
a presentation artefact of the chosen anchor, not a modelling result — setting base
odds to portfolio odds would centre the distribution on 600 without changing any
PD or ranking.

---

## 3. Validation

### Discrimination

| Split | n | Bad rate | AUC | Gini | KS | KS at score | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 184,094 | 13.19% | 0.6594 | 0.3187 | 0.2299 | 542 | 0.1103 |
| Test | 78,898 | 13.19% | 0.6527 | 0.3053 | 0.2182 | 543 | 0.1107 |
| **OOT** | 283,026 | 14.89% | 0.6607 | **0.3214** | **0.2337** | 541 | 0.1220 |

Train-to-test degradation is 0.013 Gini — negligible with 184k accounts and 11
parameters. There is no meaningful overfitting.

**Out-of-time discrimination exceeds in-time test discrimination** (0.3214 vs
0.3053). This is not an anomaly. The 2015 population has a higher bad rate (14.89%
vs 13.19%) because Lending Club loosened underwriting as volume grew. Looser
screening admits a wider range of applicant quality, and a wider range is easier
to rank — the model has more to separate. Discrimination holds up out-of-time even
though, as shown below, calibration does not.

### Default rate by score band

Bands are cut on the **training** score distribution and applied unchanged to test
and out-of-time. Cutting each split at its own deciles would force equal
populations into every band and conceal the population shift the out-of-time
window exists to reveal.

Out-of-time performance:

| Band | Score range | n | Observed bad rate | Mean predicted PD |
|---:|---|---:|---:|---:|
| 1 | 487 – 523 | 28,587 | 28.39% | 26.49% |
| 2 | 524 – 530 | 29,304 | 22.94% | 19.95% |
| 3 | 531 – 535 | 27,915 | 19.74% | 16.92% |
| 4 | 536 – 540 | 31,397 | 17.16% | 14.63% |
| 5 | 541 – 544 | 25,801 | 14.98% | 12.80% |
| 6 | 545 – 548 | 24,718 | 12.86% | 11.34% |
| 7 | 549 – 553 | 28,108 | 11.39% | 9.88% |
| 8 | 554 – 560 | 30,864 | 9.66% | 8.22% |
| 9 | 561 – 569 | 26,570 | 7.40% | 6.40% |
| 10 | 570 – 616 | 29,762 | **4.05%** | 3.98% |

**Monotonic across all ten bands with no reversals**, and a **7.0× spread** between
the worst and best band. Monotonicity matters more than the headline Gini for
operational use: a reversal in any region would make a cutoff placed there
indefensible.

Observed exceeds predicted in every band — the calibration issue below.

### Calibration

| Split | Mean gap (observed − predicted) |
|---|---:|
| Train | +0.0000 |
| Test | −0.0001 |
| **OOT** | **+0.0180** |

Train and test are calibrated essentially perfectly. Out-of-time, the model
predicts 13.09% against an observed 14.89% — it **under-predicts default risk by
1.8 percentage points**.

The cause is structural, not a modelling defect: the model was fitted on a
population with a 13.19% bad rate and applied to one running at 14.89%. Rank
ordering is unaffected; the level is wrong.

Notably, **the gap is unchanged from the pre-engineering build** (+0.0182 → +0.0180)
even though Gini improved by 0.0125. Better characteristics improve ranking; they
cannot fix a base rate that moved after fitting. Only recalibration does that.

This distinction matters operationally. PD feeds pricing and provisioning, not just
accept/decline decisions, so a model that ranks correctly but understates the level
will systematically under-price risk across the entire book.

### Recalibration

The correct response to stable discrimination with drifted calibration is an
**intercept adjustment, not a refit**. The characteristics still rank; only the
base level has moved, and refitting would discard a card that is working.

An intercept shift of **+0.1559** was solved for on the out-of-time population:

| | Mean predicted PD | Observed | Gap | Gini |
|---|---:|---:|---:|---:|
| Before | 0.1309 | 0.1489 | +0.0180 | 0.3214 |
| After | 0.1489 | 0.1489 | 0.0000 | 0.3214 |

**Gini is identical to four decimal places.** An intercept shift is a monotonic
transformation of predicted probability and cannot change rank ordering by
construction. The demonstration is the point: discrimination and calibration are
separate properties that fail independently and are remediated differently.

---

## 4. Benchmark against Lending Club's own grade

`grade`, `sub_grade` and `int_rate` were excluded from the characteristic set
(`docs/target_definition.md` §7) because they encode Lending Club's internal risk
model; a scorecard built on them would re-encode another model rather than assess
the applicant independently. They are used here as a benchmark.

All three models fitted on train, evaluated on the same 283,026 out-of-time
accounts:

| Model | OOT Gini | vs grade alone |
|---|---:|---:|
| Lending Club `sub_grade` only | 0.3573 | — |
| Scorecard only | 0.3213 | −0.0360 |
| **Grade + scorecard** | **0.3763** | **+0.0189** |

Combined model coefficients: `grade_rank` +0.05629 (p < 0.0001), `score` −0.02455
(p < 0.0001). The negative score coefficient is the correct direction.

### Reading this result

**The scorecard is 0.0360 Gini behind Lending Club's own grade in isolation**,
narrowed from 0.0485 before feature engineering. Three factors bear on the
comparison:

1. **Grade partially causes the outcome it predicts.** `sub_grade` sets the
   interest rate, which sets the monthly instalment, which affects whether the
   borrower can afford to pay. Part of its measured discrimination is a pricing
   feedback channel unavailable to any model that excludes it.
2. **Grade is fitted on more information.** Lending Club underwrites on the full
   credit report with employment and income verification, plus proprietary data
   not in the public file.
3. **The relevant question is incremental, not absolute.** A lender already has
   grade. What matters is whether a new model contributes information the existing
   assessment lacks.

On that third question the answer is affirmative and improving: **adding the
scorecard to grade lifts out-of-time Gini by 0.0189** (from +0.0157 before feature
engineering), with the score coefficient highly significant alongside grade.

---

## 5. Headline results

| Metric | Value |
|---|---|
| Out-of-time Gini | **0.321** |
| Out-of-time KS | **23.4** |
| Default rate, worst band → best band | **28.4% → 4.1%** (7.0×) |
| Band monotonicity | 10 of 10, no reversals |
| Incremental Gini over Lending Club grade | **+0.019** |
| Calibration gap, OOT before → after | +1.80pp → 0.00pp |
| Characteristics on the card | 11 |
| Attributes in the points table | 82 |

### Effect of feature engineering

| | Before ratios | After ratios |
|---|---:|---:|
| Characteristics | 12 | **11** |
| OOT Gini | 0.3089 | **0.3214** |
| OOT KS | 22.3 | **23.4** |
| Band spread | 6.5× | **7.0×** |
| Gap to LC grade | −0.0485 | **−0.0360** |
| Incremental over grade | +0.0157 | **+0.0189** |
| Gap to XGBoost challenger † | +0.0518 | **+0.0395** |

† Both columns compare against the **untuned** challenger, so the row isolates
what the ratios changed. The challenger was subsequently given a 50-trial Optuna
search, which widened the current gap to +0.0425
(`docs/challenger_analysis.md` §2).

**Better discrimination on fewer characteristics**, and the gap to the
gradient-boosted challenger closed by 24% before the challenger was tuned.

---

## 6. Limitations arising at this stage

1. **Calibration is not stable across vintages.** The intercept shift corrects the
   2015 population but is not a permanent fix; a deployed card would need scheduled
   recalibration triggered by monitoring, which is what the PSI work in stage 10
   supports.
2. **Base odds anchor is arbitrary.** 50:1 places the whole population below the
   base score. Harmless, but a production card would anchor on portfolio odds.
3. **Single out-of-time window.** Validation rests on one vintage year. A rolling
   out-of-time evaluation across several vintages would better estimate how quickly
   performance decays.
4. **No interaction terms.** Ablation attributes +0.0151 Gini to functional form
   (`docs/challenger_analysis.md` §4). Explicit interactions such as
   `fico × dti_with_loan` might recover part of it, at the cost of a points table
   that no longer decomposes cleanly into per-characteristic attributes — which
   would complicate the reason codes in stage 9.
5. **Accepted applicants only**, carried forward from `docs/target_definition.md`
   §8 — measured performance is optimistic relative to a genuine through-the-door
   application scorecard.

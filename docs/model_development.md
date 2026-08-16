# Model Development, Scaling and Validation

Stages 5 to 7 of the build: logistic regression champion, conversion to a
points-based scorecard, and the validation pack.

Scripts: `src/04_logistic_regression.py`, `src/05_scorecard.py`,
`src/06_evaluation.py`, `src/07_benchmark_and_calibration.py`.

Development sample: 184,094 accounts (train), 78,898 (test), 283,026
(out-of-time). Bad definition and window design in `docs/target_definition.md`;
characteristic selection in `docs/feature_selection.md`.

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
model fits, so the check is implemented as a gate rather than a diagnostic: each
coefficient's sign is compared against that characteristic's univariate direction
on the training sample, and disagreements are dropped and the model refitted.

**One characteristic failed: `revol_bal`.**

| | Univariate | In model |
|---|---|---|
| `revol_bal` direction | − (higher balance, higher risk) | + |
| Coefficient | | +0.4187 (se 0.0860) |

This is a suppression effect, and it has a clear credit interpretation.
Univariately, a larger revolving balance signals higher risk. But *conditional on*
`revol_util` and `total_bc_limit` — that is, holding utilisation and credit line
constant — a large balance implies a large available line, which indicates
capacity rather than distress. The model found the conditional relationship and
reversed the sign.

The effect is real, not a coding error, and it would have passed silently on any
discrimination metric. `revol_bal` was dropped and the model refitted on 12
characteristics.

### Why the drop order matters

In the first fit, `revol_util` was not significant (p = 0.486) and was a candidate
for removal. After `revol_bal` was dropped, `revol_util` became significant
(p = 0.018) and was retained.

The two characteristics are correlated; with both present, neither could be
estimated precisely. Removing sign flips first, refitting, and only then applying
the significance screen preserved a legitimate characteristic that a
simultaneous drop would have discarded.

### Final coefficients

Fitted on 184,094 training accounts. All 12 coefficients agree with their
univariate direction; all significant at α = 0.05.

| Characteristic | Coefficient | Std err | p-value |
|---|---:|---:|---:|
| `mo_sin_rcnt_tl` | −0.7422 | 0.0398 | < 0.0001 |
| `inq_last_6mths` | −0.7806 | 0.0443 | < 0.0001 |
| `dti` | −0.7282 | 0.0320 | < 0.0001 |
| `fico` | −0.6073 | 0.0249 | < 0.0001 |
| `annual_inc` | −0.4713 | 0.0289 | < 0.0001 |
| `percent_bc_gt_75` | −0.4355 | 0.0482 | < 0.0001 |
| `mo_sin_old_rev_tl_op` | −0.4217 | 0.0393 | < 0.0001 |
| `avg_cur_bal` | −0.3864 | 0.0389 | < 0.0001 |
| `total_bc_limit` | −0.2493 | 0.0317 | < 0.0001 |
| `home_ownership` | −0.2245 | 0.0474 | < 0.0001 |
| `revol_util` | −0.1658 | 0.0700 | 0.0178 |
| `mort_acc` | −0.0921 | 0.0460 | 0.0454 |
| *intercept* | −1.8856 | | |

McFadden pseudo R² = 0.0440, log-likelihood = −68,639.6.

All coefficients are negative because WOE is oriented so that a higher weight
indicates a lower-risk bin; the uniform sign is the expected result for a
correctly specified scorecard.

The low pseudo R² is characteristic of credit scoring on an accepted-applicant
population and should not be read as a poorly specified model — see
`docs/feature_selection.md` §3 on population truncation.

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
that per-attribute points sum to the total score. The leading negative converts
the model's log-odds of *bad* into odds of *good*, so a higher score indicates a
better applicant.

These parameters are a **presentation choice, not a modelling one**. They change
the numbers printed on the card and nothing about rank ordering or predicted PD.
The practical value of the PDO convention is that score differences carry constant
meaning: the gap from 640 to 660 represents the same change in odds as the gap
from 700 to 720.

**Resulting card: 87 attributes across 12 characteristics.**

### Reconciliation

Each score is converted back to an implied PD and compared against the model's own
prediction:

```
odds_good  = exp((score − offset) / factor)
implied_PD = 1 / (1 + odds_good)
```

Maximum absolute discrepancy: **0.004078**, consistent with rounding points to
whole numbers. This check guards against the failure mode where the points table
and the model silently diverge, producing a card that looks correct and scores
incorrectly.

### Score contribution by characteristic

| Characteristic | Min | Max | Spread |
|---|---:|---:|---:|
| `fico` | 38 | 65 | 27 |
| `avg_cur_bal` | 29 | 54 | 25 |
| `annual_inc` | 37 | 52 | 15 |
| `mo_sin_rcnt_tl` | 39 | 54 | 15 |
| `dti` | 37 | 51 | 14 |
| `inq_last_6mths` | 38 | 48 | 10 |
| `mo_sin_old_rev_tl_op` | 40 | 48 | 8 |
| `total_bc_limit` | 43 | 51 | 8 |
| `percent_bc_gt_75` | 42 | 49 | 7 |
| `home_ownership` | 44 | 47 | 3 |
| `revol_util` | 44 | 47 | 3 |
| `mort_acc` | 45 | 46 | 1 |

No single characteristic dominates the card, which matters for stability: where one
characteristic drives most of the score, drift in that one feature moves the entire
book.

`mort_acc` contributes a **one-point** spread. It is statistically significant
(p = 0.045) and practically irrelevant — it cannot alter a decision at any
cutoff. It is retained here for completeness, but a production card would drop it,
since a characteristic carrying data-collection and monitoring cost for one point
of movement does not earn its place.

### Score distribution

| Split | Median | Range |
|---|---:|---|
| Train | 544 | 492 – 613 |
| Test | 543 | 494 – 611 |
| OOT | 544 | 494 – 613 |

Theoretical card range is 476 – 612; the observed range is narrower because the
worst and best attributes rarely co-occur in the same applicant.

The distribution sits entirely below the 600 base score, because base odds were
set at 50:1 while the development population runs at approximately 6.6:1. This is
a presentation artefact of the chosen anchor, not a modelling result — setting
base odds to the population odds would centre the distribution on 600 without
changing any PD or ranking.

---

## 3. Validation

### Discrimination

| Split | n | Bad rate | AUC | Gini | KS | KS at score | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 184,094 | 13.19% | 0.6537 | 0.3074 | 0.2245 | 542 | 0.1107 |
| Test | 78,898 | 13.19% | 0.6464 | 0.2927 | 0.2107 | 542 | 0.1110 |
| **OOT** | 283,026 | 14.89% | 0.6544 | **0.3089** | **0.2234** | 543 | 0.1223 |

Train-to-test degradation is 0.015 Gini — negligible, as expected with 184k
accounts and 12 parameters. There is no meaningful overfitting to diagnose.

**Out-of-time discrimination exceeds in-time test discrimination** (0.3089 vs
0.2927). This is not an anomaly. The 2015 population has a higher bad rate
(14.89% vs 13.19%) because Lending Club loosened underwriting as origination
volume grew. Looser screening admits a wider range of applicant quality, and a
wider range is easier to rank — the model has more to separate. Discrimination
therefore holds up out-of-time even though, as shown below, calibration does not.

### Default rate by score band

Bands are cut on the **training** score distribution and applied unchanged to test
and out-of-time. Cutting each split at its own deciles would force equal
populations into every band and conceal the population shift the out-of-time
window exists to reveal.

Out-of-time performance:

| Band | Score range | n | Observed bad rate | Mean predicted PD |
|---:|---|---:|---:|---:|
| 1 | 494 – 524 | 30,063 | 27.64% | 25.42% |
| 2 | 525 – 530 | 27,507 | 22.26% | 19.71% |
| 3 | 531 – 535 | 28,941 | 19.64% | 16.92% |
| 4 | 536 – 539 | 26,008 | 17.01% | 14.86% |
| 5 | 540 – 544 | 32,434 | 15.33% | 13.00% |
| 6 | 545 – 548 | 24,944 | 13.02% | 11.34% |
| 7 | 549 – 553 | 27,995 | 11.81% | 9.88% |
| 8 | 554 – 559 | 26,868 | 9.71% | 8.33% |
| 9 | 560 – 568 | 28,213 | 7.73% | 6.61% |
| 10 | 569 – 613 | 30,053 | **4.25%** | 4.17% |

**Monotonic across all ten bands with no reversals**, and a 6.5× spread between the
worst and best band. Monotonicity matters more than the headline Gini for
operational use: a reversal in any region would make a cutoff placed there
indefensible.

Note that observed exceeds predicted in every band — the calibration issue below.

### Calibration

| Split | Mean gap (observed − predicted) |
|---|---:|
| Train | +0.0000 |
| Test | −0.0002 |
| **OOT** | **+0.0182** |

Train and test are calibrated essentially perfectly. Out-of-time, the model
predicts 13.07% against an observed 14.89% — it **under-predicts default risk by
1.8 percentage points**.

The cause is structural, not a modelling defect: the model was fitted on a
population with a 13.19% bad rate and applied to one running at 14.89%. Rank
ordering is unaffected; the level is wrong.

This distinction matters operationally. PD feeds pricing and provisioning, not just
accept/decline decisions, so a model that ranks correctly but understates the level
will systematically under-price risk across the entire book.

### Recalibration

The correct response to stable discrimination with drifted calibration is an
**intercept adjustment, not a refit**. The characteristics still rank; only the
base level has moved, and refitting would discard a card that is working.

An intercept shift of **+0.1569** was solved for on the out-of-time population:

| | Mean predicted PD | Observed | Gap | Gini |
|---|---:|---:|---:|---:|
| Before | 0.1307 | 0.1489 | +0.0182 | 0.3089 |
| After | 0.1489 | 0.1489 | 0.0000 | 0.3089 |

**Gini is identical to four decimal places.** An intercept shift is a monotonic
transformation of predicted probability and therefore cannot change rank ordering
by construction. The demonstration is the point: discrimination and calibration
are separate properties that fail independently and are remediated differently.

---

## 4. Benchmark against Lending Club's own grade

`grade` and `sub_grade` were excluded from the characteristic set (see
`docs/target_definition.md` §7) because they encode Lending Club's internal risk
model; a scorecard built on them would re-encode another model rather than assess
the applicant independently. They are used here instead as a benchmark.

All three models fitted on train, evaluated on the same 283,026 out-of-time
accounts:

| Model | OOT Gini | vs grade alone |
|---|---:|---:|
| Lending Club `sub_grade` only | 0.3573 | — |
| Scorecard only | 0.3088 | −0.0485 |
| **Grade + scorecard** | **0.3730** | **+0.0157** |

Combined model coefficients: `grade_rank` +0.05829 (p < 0.0001), `score` −0.02347
(p < 0.0001). The negative score coefficient is the correct direction — higher
score, lower risk.

### Reading this result

**The scorecard is 0.0485 Gini behind Lending Club's own grade in isolation.**
That is reported as measured. Three factors bear on the comparison:

1. **Grade partially causes the outcome it predicts.** `sub_grade` sets the
   interest rate, which sets the monthly instalment, which affects whether the
   borrower can afford to pay. It is not a pure ex-ante risk assessment; part of
   its measured discrimination is a pricing feedback channel unavailable to any
   model that excludes it.
2. **Grade is fitted on more information.** Lending Club underwrites on the full
   credit report with employment and income verification, plus proprietary data
   not present in the public file. The scorecard uses 12 characteristics from the
   published application data.
3. **The relevant question is incremental, not absolute.** A lender already has
   grade. What matters is whether a new model contributes information the existing
   assessment lacks.

On that third question the answer is affirmative: **adding the scorecard to grade
improves out-of-time Gini by 0.0157**, and the score coefficient remains highly
significant with grade in the model. The card carries independent signal.

---

## 5. Headline results

| Metric | Value |
|---|---|
| Out-of-time Gini | **0.309** |
| Out-of-time KS | **22.3** at score 543 |
| Default rate, worst band → best band | **27.6% → 4.3%** |
| Band monotonicity | 10 of 10, no reversals |
| Incremental Gini over Lending Club grade | **+0.016** |
| Calibration gap, OOT before → after | +1.82pp → 0.00pp |
| Characteristics on the card | 12 |
| Attributes in the points table | 87 |

---

## 6. Limitations arising at this stage

1. **Calibration is not stable across vintages.** The intercept shift corrects the
   2015 population but is not a permanent fix; a deployed card would need
   scheduled recalibration triggered by monitoring, which is what the PSI work in
   stage 10 supports.
2. **`mort_acc` earns one point of spread** and would be removed from a production
   card on parsimony grounds.
3. **Base odds anchor is arbitrary.** 50:1 places the whole population below the
   base score. Harmless, but a production card would anchor on portfolio odds.
4. **Single out-of-time window.** Validation rests on one vintage year. A rolling
   out-of-time evaluation across several vintages would give a better estimate of
   how quickly performance decays.
5. **Accepted applicants only**, carried forward from
   `docs/target_definition.md` §8 — measured performance is optimistic relative to
   a genuine through-the-door application scorecard.

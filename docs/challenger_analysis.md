# Champion vs Challenger: XGBoost and the Interpretability Trade-off

Stage 8 of the build. A gradient boosted challenger is fitted against the
scorecard champion, attributed with SHAP, decomposed by ablation, and the
deployment decision argued on the evidence.

Scripts: `src/08_challenger.py`, `src/09_ablation.py`.

---

## 1. Experimental design

The comparison is constructed so the challenger is not handicapped:

| | Champion | Challenger |
|---|---|---|
| Model | Logistic regression | XGBoost |
| Features | 11 WOE-transformed characteristics | **41 raw candidates** |
| Feature selection | IV screen + correlation + VIF + sign check | None — model selects |
| Missing values | WOE missing bin | Native handling |
| Categoricals | WOE by event rate | Native categorical support |

The challenger receives **every candidate characteristic**, including the six
engineered ratios and the 21 the IV screen discarded, on raw untransformed values.
Restricting it to the champion's pre-selected, pre-binned inputs would make the
contest meaningless.

`grade`, `sub_grade` and `int_rate` are withheld from **both** models, so neither
has access to Lending Club's own risk assessment.

### Split hygiene

An initial run used the test partition as the early-stopping `eval_set`, which
selects tree count by test performance and makes the reported test Gini
optimistic — and unfair to the champion, which never saw test at any point.

This was corrected: a 15% validation slice is carved out of train for early
stopping, leaving test and out-of-time genuinely held out for both models. The
effect was small (test Gini 0.3533 → 0.3513 on the pre-engineering build) but the
corrected design is the one reported.

The champion is fitted on train + validation while the challenger sees only train.
That asymmetry runs *against* the challenger, so the lift below is conservative.

**Only `test` and `oot` are quotable.** `train_fit` is in-sample for the
challenger, `val` selected its tree count, and both are in-sample for the champion.

---

## 2. Results

| Split | Model | Gini | KS | Brier | Calibration gap |
|---|---|---:|---:|---:|---:|
| train_fit | Champion | 0.3209 | 0.2326 | 0.1103 | −0.0001 |
| train_fit | Challenger | 0.4912 | 0.3616 | 0.1037 | −0.0000 |
| val | Champion | 0.3064 | 0.2199 | 0.1106 | +0.0005 |
| val | Challenger | 0.3360 | 0.2403 | 0.1099 | +0.0011 |
| **test** | Champion | 0.3053 | 0.2196 | 0.1107 | −0.0001 |
| **test** | Challenger | **0.3517** | 0.2551 | 0.1094 | +0.0001 |
| **oot** | Champion | 0.3214 | 0.2338 | 0.1220 | +0.0180 |
| **oot** | Challenger | **0.3609** | 0.2630 | 0.1205 | +0.0159 |

**Out-of-time Gini lift: +0.0395** (0.3214 → 0.3609), a 12% relative improvement.
KS improves from 23.4 to 26.3.

The lift is real and material. Any argument for shipping the champion has to
justify giving it up.

### Two observations that cut against the champion

**The challenger is better calibrated, not worse.** The expectation going in was
that raw gradient boosting probabilities would need Platt scaling or isotonic
regression. They did not: out-of-time calibration gap is +0.0159 against the
champion's +0.0180, and Brier is 0.1205 against 0.1220. Log-loss optimisation
without class weighting produces well-calibrated output on this data. The "tree
models rank well but predict badly" argument does not apply here and is not used
below.

**Both models under-predict out-of-time by a similar margin**, for the same reason —
both were fitted at a 13.19% bad rate and applied at 14.89%. This is a population
shift, not a model defect, and it affects both alike.

### The observation that cuts for the champion

| Model | train_fit → test | Degradation |
|---|---|---:|
| Champion | 0.3209 → 0.3053 | 0.0156 |
| Challenger | 0.4912 → 0.3517 | **0.1395** |

The challenger fits the training sample nine times harder than it generalises. It
still generalises *better* in absolute terms, so this is not a disqualification —
but the stability profiles are entirely different, and that matters for an artefact
that must hold up for eighteen to twenty-four months between rebuilds with no
opportunity to observe outcomes in between.

---

## 3. SHAP attribution

Mean absolute SHAP on an 8,000-account out-of-time sample.

| Rank | Characteristic | mean \|SHAP\| | On the card? |
|---:|---|---:|---|
| 1 | `fico` | 0.2353 | Yes |
| 2 | `dti_with_loan` | 0.1457 | Yes (engineered) |
| 3 | `mo_sin_rcnt_tl` | 0.1167 | Yes |
| 4 | `annual_inc` | 0.1151 | Yes |
| 5 | `addr_state` | 0.1086 | **No** |
| 6 | `loan_to_income` | 0.0879 | Yes (engineered) |
| 7 | `purpose` | 0.0816 | **No** |
| 8 | `inq_intensity` | 0.0779 | Yes (engineered) |
| 9 | `percent_bc_gt_75` | 0.0725 | Yes |
| 10 | `total_bc_limit` | 0.0684 | Yes |
| 11 | `mo_sin_old_rev_tl_op` | 0.0637 | Yes |
| 12 | `num_actv_bc_tl` | 0.0626 | **No** |
| 13 | `bc_limit_to_income` | 0.0551 | **No** (engineered, failed IV) |
| 14 | `avg_cur_bal` | 0.0546 | Yes |
| 15 | `home_ownership` | 0.0502 | Yes |

Scorecard characteristics account for **56.9%** of total |SHAP|.

**All three engineered ratios on the card rank in the challenger's top eight**, and
`dti_with_loan` is its second most important characteristic overall. The ratios
were not merely a convenience for the linear model; they are constructions the
challenger also finds useful.

`bc_limit_to_income` at 13th is the notable miss. It is one of the two engineered
ratios that **failed** the IV screen (0.0128, below the 0.02 floor), yet the
challenger ranks it above `avg_cur_bal` and `home_ownership`, both of which are on
the card. Univariate screening is still discarding conditionally useful
characteristics — the same limitation that originally hid `loan_amnt`.

---

## 4. Ablation: where the lift actually comes from

Five identically parameterised XGBoost variants, fitted on the same rows and
evaluated on the same held-out windows.

| Variant | Features | Test Gini | OOT Gini | Lift vs champion | % of headline |
|---|---:|---:|---:|---:|---:|
| Champion (scorecard) | 11 | 0.3053 | 0.3214 | — | — |
| A. Full challenger | 41 | 0.3517 | 0.3609 | +0.0395 | 100% |
| B. Minus `addr_state` | 40 | 0.3502 | **0.3612** | +0.0399 | 101% |
| C. Minus `addr_state`, `purpose` | 39 | 0.3431 | 0.3571 | +0.0358 | 90% |
| D. Champion characteristics only | 11 | 0.3213 | 0.3365 | +0.0151 | 38% |
| E. Champion + `loan_amnt` | 12 | 0.3220 | 0.3367 | +0.0153 | 39% |

### Decomposition

| Source of lift | Gini | Share |
|---|---:|---:|
| Functional form — non-linearity and interactions on the card's own 11 inputs | +0.0151 | 38% |
| The 30 discarded characteristics | +0.0242 | 61% |
| Adding `loan_amnt` | +0.0002 | 1% |
| Geography (`addr_state`) | −0.0003 | 0% |

### Geography contributes nothing — and this replicates

An earlier draft argued that a material share of the challenger's lift came from
`addr_state`, and that the lift was therefore partly unavailable to a lender after
fair lending review. **The ablation does not support that.**

Removing `addr_state` moved out-of-time Gini from 0.3609 to 0.3612 — it improved
slightly. Despite ranking **5th of 41** by mean |SHAP|, geography adds no
generalisable discrimination at all.

This has now been measured twice, on two different feature pools: −0.0005 before
feature engineering and −0.0003 after. The finding is stable, not an artefact of
one run.

The gap between attribution and contribution is the lesson. **SHAP measures how
much the model used a characteristic, not whether using it helped.** The challenger
was fitting state-level pockets present in the 2013–14 development sample that did
not reappear in the 2015 window — plausibly because Lending Club's geographic mix
shifted materially as it expanded, which is exactly the drift the PSI monitor in
stage 10 is built to detect.

Practical consequences:

- Feature importance is not evidence that a characteristic earns its place.
  Ablation is. Any importance ranking used to justify a model should be checked
  this way.
- The fair lending exposure is genuine as a compliance matter — a US lender would
  still need review before deploying a model that prices state of residence, since
  geography correlates with race and national origin and creates disparate impact
  exposure regardless of intent. But it is **free to remediate**. Dropping
  `addr_state` costs nothing, so it is not an argument for the champion.
- `purpose` does contribute (+0.0038). Removing both leaves **90% of the headline
  lift intact**.

> This is a portfolio exercise, not legal advice. The claim is that these
> characteristics would require fair lending review, not that any particular
> outcome of that review is certain.

### What feature engineering changed

The ablation was run twice, before and after the six ratios were introduced:

| | Before ratios | After ratios |
|---|---:|---:|
| Champion OOT Gini | 0.3089 | **0.3214** |
| Challenger OOT Gini | 0.3607 | 0.3609 |
| Headline lift | +0.0518 | **+0.0395** |
| Functional form share | 26% | **38%** |
| Value of adding `loan_amnt` | +0.0178 | **+0.0002** |

Three findings.

**The challenger gained +0.0002 from the ratios; the champion gained +0.0125.**
Gradient boosting already constructs these relationships internally — it splits on
loan amount within regions of income, approximating `loan_to_income` without being
told about it. A linear model cannot. **Feature engineering adds no information to
a tree ensemble; it changes the representation so an additive model can use
information that was present all along.**

**`loan_amnt` fell from +0.0178 to +0.0002.** It was worth more than every
interaction the challenger could find, until `loan_to_income` entered the card and
absorbed it completely. Same information, better encoding, raw version redundant.

**Functional form rose from 26% to 38% of the remaining lift.** Not because the
non-linearity got worse, but because the feature-selection component shrank. As the
selection gap closes, what remains is increasingly genuine interaction effect —
and closing it further requires interaction terms, not more ratios.

---

## 5. Reason codes: a structural difference, not a presentational one

Under ECOA / Reg B, a declined applicant must receive the specific principal
reasons for the decision. This is where the two models genuinely diverge, and it is
not a matter of explanation quality.

**The scorecard produces reasons as a by-product of its structure.** Each
applicant's score is a sum of 11 attribute points from a fixed 82-row table. The
reason for a decline is the set of characteristics where the applicant lost the
most points against the maximum attainable — arithmetic on a published table,
identical for two applicants with identical inputs, stable until the card is
rebuilt, and auditable by anyone holding the table.

**SHAP produces a local attribution, which is a different object.** It is a
per-prediction decomposition against a baseline; it depends on the background
dataset chosen; it varies between applicants with the same value on the
characteristic being explained, because it is conditional on everything else; and
it is not reproducible without the model artefact and the explainer configuration.
It is a good tool for understanding model behaviour and a harder basis for a
legally binding statement of principal reasons that must be defensible to a
regulator years after issuance.

Reason codes can be derived from SHAP and lenders do it. The point is that the
scorecard's reasons fall out of the artefact, while the challenger's must be
constructed, validated and separately governed.

---

## 6. Monitoring surface

The champion's monitoring pack is complete when PSI has been computed for 11
characteristics and the score. Drift localises immediately to a named
characteristic and a named bin.

The challenger requires PSI on 41 characteristics, and 38% of its advantage lives
in **interactions between them**. There is no PSI for an interaction. Individual
feature distributions can each stay within tolerance while the joint distribution
the model relies on has shifted, and the standard monitoring pack will not detect
it.

The `addr_state` result is a concrete instance. The challenger placed substantial
weight on geography — 5th by SHAP — and that weight contributed nothing
out-of-time, because the geographic mix moved between vintages. A univariate PSI on
`addr_state` would have flagged the distribution shift, but nothing in the standard
pack would reveal that the model's *use* of it had stopped working.

This is the argument that carries most weight with a model risk function. It is not
that the challenger cannot be monitored — it is that the standard early warning
apparatus is substantially less informative for it, so a problem is more likely to
surface through realised losses than through a monitoring flag.

---

## 7. Remediation options when drift appears

`docs/model_development.md` §3 demonstrates the champion's response to the 2015
population shift: an intercept shift of +0.1559 closed the calibration gap from
+0.0180 to 0.0000 with Gini unchanged to four decimal places. One parameter, no
refit, no revalidation of the characteristic set.

The challenger has no intercept to shift. Options are a post-hoc calibration layer
(Platt or isotonic) fitted on recent outcomes — a second model artefact with its
own governance and drift behaviour — or a full refit, meaning retraining,
revalidation, fresh SHAP analysis, fresh fair lending review and a new approval
cycle.

For a model needing periodic recalibration as vintages shift, this difference
compounds across the model's life.

---

## 8. What would actually be deployed

**Champion in production, challenger in shadow.**

### What the ablation settled

The comfortable version of the champion's case was that the challenger's lift
depends on characteristics a lender cannot use and interactions nobody can monitor.
Only the second half survives. Geography contributes nothing, and 90% of the lift
remains after removing every fair-lending-exposed characteristic.

The accurate version is that the scorecard's deficit was mostly in **feature
selection**, not functional form — and that has now been partly acted on. The six
engineered ratios closed the gap from +0.0518 to +0.0395, a 24% reduction, while
*reducing* the card from 12 characteristics to 11.

### The case for the scorecard as the decisioning model

1. Reason codes fall out of the artefact rather than requiring a parallel
   explanation system (§5).
2. Monitoring localises drift to a named characteristic and bin; the challenger's
   interaction-driven advantage is not observable through PSI (§6).
3. Recalibration is a one-parameter intercept shift, demonstrated to preserve
   discrimination exactly (§7).
4. Coefficient signs are individually checkable against credit intuition — the
   check that caught `revol_bal` and `revol_util`.
5. The functional-form gap is small in absolute terms. XGBoost given exactly the
   card's 11 inputs reaches 0.3365 against 0.3214: an additive linear model
   captures most of what those characteristics offer.
6. It reaches 0.3214 without using geography, which the challenger weights heavily
   and gains nothing from.

### The case against, stated fairly

1. It gives up 0.0395 Gini out-of-time, which is material on a large book.
2. The challenger is **better calibrated**, not worse.
3. The fair lending argument does not do the work it appeared to. Dropping
   `addr_state` is free.
4. Univariate IV screening still discards conditionally useful characteristics —
   `bc_limit_to_income` ranks 13th by SHAP after being rejected at IV 0.0128.

### Remaining work

1. **Address the residual selection gap.** The 30 discarded characteristics are
   worth +0.0242. `bc_limit_to_income` is the identified candidate; a
   forward-selection or ablation-validated procedure would find the rest.
2. **Test interaction terms.** Functional form is now 38% of the remaining lift.
   `fico × dti_with_loan` is the obvious candidate, at the cost of a points table
   that no longer decomposes cleanly per characteristic.
3. **Verify every addition by ablation, not importance.** `addr_state` ranked 5th
   by SHAP and contributed nothing, twice.
4. **Run the challenger in shadow.** Cases where the two disagree sharply are where
   the champion's specification is failing, and they form the requirement list for
   the next rebuild.

### The summary worth remembering

The textbook answer is that banks ship logistic regression because it is
explainable. The evidence here gives a more specific account: the scorecard's
linear form costs 0.0151 Gini, its selection methodology cost considerably more
until six engineered ratios recovered a quarter of the gap, and the challenger's
5th most-attributed characteristic contributed nothing at all. The interpretability
argument for the champion holds — but on monitoring and remediation grounds, not
because the challenger's lift is illusory or unusable.

---

## 9. Limitations

1. **Ablation variants were not repeated across seeds.** Differences below roughly
   0.005 Gini — including the −0.0003 attributed to geography — are within the
   range seed variation could produce. The conclusion that `addr_state` contributes
   nothing is safe, having replicated across two feature pools; a precise point
   estimate is not.
2. **Hyperparameters were set by judgement, not searched.** `max_depth=5`,
   `min_child_weight=50`, `learning_rate=0.05` were chosen to be conservative on
   noisy 13%-event-rate data. A tuned challenger would likely do better, making the
   reported lift a floor rather than a ceiling.
3. **One out-of-time window.** Whether the lift persists across multiple vintages
   is untested, and it is precisely the question that matters for the shadow
   deployment argument.
4. **No interaction terms were tested on the champion**, so the +0.0151 attributed
   to functional form is an upper bound on what the scorecard cannot capture.
5. **SHAP computed on an 8,000-account sample** rather than the full 283,026
   out-of-time population, for tractability.

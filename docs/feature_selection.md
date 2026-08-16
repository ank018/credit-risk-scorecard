# Feature Engineering, Transformation and Selection

Stages 3 and 4 of the build. This document records how candidate characteristics
were constructed, transformed, screened and reduced.

Fitted on the development training sample only: 184,094 accounts, 13.19% bad rate.
Scripts: `src/features.py`, `src/02_woe_binning.py`, `src/03_feature_selection.py`.

---

## 1. Feature engineering

The published Lending Club fields are raw quantities: a loan amount, an income, a
balance, a credit limit, an inquiry count. Credit risk is rarely about a raw
quantity. It is about **an obligation measured against a capacity** — and that is
a relationship between two fields, not a property of either one.

Six ratios were constructed on that principle. Each has a stated credit rationale;
none came from an automated sweep of pairwise combinations.

| Ratio | Definition | Rationale |
|---|---|---|
| `loan_to_income` | `loan_amnt / annual_inc` | Exposure against capacity. The standard bank characteristic. |
| `dti_with_loan` | `dti + (loan_amnt/36)/(annual_inc/12) × 100` | Debt service **including** the obligation being decided on. |
| `bal_to_bc_limit` | `revol_bal / total_bc_limit` | Balance carried against the bankcard line specifically. |
| `inq_intensity` | `inq_last_6mths / (total_acc + 1)` | Credit-seeking relative to an established file. |
| `acct_velocity` | `open_acc / (credit_hist_months/12)` | Accounts opened per year of history — thin file vs expanding file. |
| `bc_limit_to_income` | `total_bc_limit / annual_inc` | Revolving capacity other lenders have extended, relative to income. |

Two implementation notes. `dti_with_loan` approximates monthly principal as
`loan_amnt / 36` rather than using the published `installment`, because
`installment` is a function of `int_rate` and would reintroduce Lending Club's own
pricing (see `docs/target_definition.md` §7). And division by zero produces `inf`,
which is converted to `NaN` so that binning assigns it a Missing bin with its own
weight — an applicant with zero declared income is a meaningful category, not a
value to impute.

### Why this was necessary: the `loan_amnt` case

`loan_amnt` scores **IV 0.0115** on its own, below the 0.02 screening floor. Loan
size appears to carry almost no information about default.

It does not, because of a confound. Larger loans are extended to higher-income
borrowers — Lending Club will not lend $30,000 against a $35,000 income. Sorting
applicants by loan amount therefore also sorts them by income, and the two effects
run in opposite directions: a bigger loan is riskier, but a bigger loan also means
a better borrower. They cancel.

Hold income constant and the relationship is unambiguous. A $30,000 loan against
$50,000 of income is a materially different proposition from $8,000 against the
same income.

Expressed as a ratio, the signal becomes visible to univariate screening:

**`loan_amnt` IV 0.0115 → `loan_to_income` IV 0.0424.** A 269% increase from the
same underlying information, differently encoded.

### Results

**Four of six ratios cleared the IV floor.** Each of the four scored above the raw
field it was built from:

| Ratio | IV | Raw comparator | IV | Change |
|---|---:|---|---:|---:|
| `dti_with_loan` | 0.0733 | `dti` | 0.0533 | +38% |
| `bal_to_bc_limit` | 0.0471 | `revol_util` | 0.0206 | +129% |
| `loan_to_income` | 0.0424 | `loan_amnt` | 0.0115 | **+269%** |
| `inq_intensity` | 0.0359 | `inq_last_6mths` | 0.0279 | +29% |

**Two failed.** `acct_velocity` (IV 0.0183) and `bc_limit_to_income` (IV 0.0128)
both fell below the floor and were dropped. Both are reported here because a
record that lists only the ratios that worked is not evidence of method.

`bc_limit_to_income` is the more interesting failure: it ranks **13th of 41 by
mean |SHAP|** in the challenger (`docs/challenger_analysis.md` §3), so it carries
conditional signal that univariate screening again cannot see. The same limitation
that hid `loan_amnt` is still operating on this characteristic, and no override
was applied to it.

### What the engineering was worth

Measured downstream, holding everything else constant:

| | Before ratios | After ratios |
|---|---:|---:|
| Champion OOT Gini | 0.3089 | **0.3214** |
| Champion characteristics | 12 | **11** |
| Challenger OOT Gini | 0.3607 | 0.3609 |

**+0.0125 Gini for the champion, with one fewer characteristic. +0.0002 for the
challenger.**

The asymmetry is the point. Gradient boosting already constructs these
relationships internally — it splits on loan amount within regions of income,
which approximates `loan_to_income` without being told about it. A linear model
cannot. Feature engineering adds no information to a tree ensemble; it changes the
*representation* so that an additive model can use information that was present all
along.

---

## 2. Why Weight of Evidence

Every candidate is WOE-transformed before modelling. For a bin *i*:

```
WOE_i = ln( (goods_i / total_goods) / (bads_i / total_bads) )
```

Four properties make this standard in scorecard work:

1. **Linear in log-odds.** WOE expresses each bin on the scale logistic regression
   operates on, so a non-linear raw relationship becomes linear after binning —
   without splines or polynomial terms that would be hard to justify in review.
2. **Missing is a category, not a defect.** `mths_since_last_delinq` is null
   precisely when the applicant has never been delinquent; the null *is* the
   signal. **No imputation is performed anywhere in this pipeline.**
3. **Outlier-insensitive.** A declared income of $10m lands in the top bin and
   contributes the same weight as any other applicant there.
4. **Auditable.** Each bin is a stated range with a stated weight, readable by
   someone who does not write code.

### Monotonic constraints

Numeric characteristics are fitted with `monotonic_trend="auto_asc_desc"`, forcing
a single direction of risk.

An unconstrained fit often produces a WOE curve that falls, rises, then falls
again. That shape is usually noise in sparse bins, and it fails twice: it cannot
be explained to a credit committee ("risk rises with income between $40k and
$55k, then falls"), and it does not survive a population shift.

### Bin granularity

| Parameter | Value |
|---|---|
| `min_bin_size` | 0.05 — no bin below 5% of the training population |
| `max_n_bins` | 8 |
| `min_prebin_size` | 0.02 |

An initial unconstrained fit produced up to 17 bins on some characteristics.
Constraining to 8 cost very little IV but improved bin quality substantially:

| Feature | Bins | IV | `quality_score` |
|---|---|---|---|
| `fico` | 17 → 8 | 0.1413 → 0.1391 | 0.0136 → **0.5134** |
| `annual_inc` | 14 → 8 | 0.0945 → 0.0937 | 0.0447 → **0.3638** |

`quality_score` measures how statistically distinct adjacent bins are. The 17-bin
version was cutting the population into groups that were not reliably different
from each other. Coarsening did not discard signal so much as stop manufacturing
false precision — thin bins produce WOE values that swing on a new population,
which is the failure mode the out-of-time window exists to detect.

---

## 3. Information Value screening

```
IV = Σ_i (goods_i/total_goods − bads_i/total_bads) × WOE_i
```

Thresholds: **drop below 0.02**, **investigate above 0.50**.

**41 candidates → 20 retained.** Nothing exceeded the 0.50 ceiling, which is
itself a result: the exclusion list in `docs/target_definition.md` §7 removed the
post-origination fields effectively, and no surviving characteristic behaves like
an outcome variable.

### Documented override

`loan_amnt` was **retained despite failing the floor**, via an explicit
`FORCE_KEEP` list in `src/02_woe_binning.py`.

The justification is measured rather than intuitive: ablation
(`docs/challenger_analysis.md` §4) valued `loan_amnt` at +0.0178 Gini when added
to the champion's characteristics, more than every interaction and non-linearity
the challenger found across the surviving set. Overriding a documented threshold
with documented evidence is preferable to quietly lowering the floor to 0.01 and
leaving the reason unstated.

The override was subsequently **rejected by the multivariate stage** — see §6.
That is the correct sequence, not a contradiction: univariate IV was wrong to
reject `loan_amnt` on its own terms, and the regression was right to find it
redundant once a better encoding of the same information was available.

### Why the IV levels are modest

The strongest characteristic, `fico`, reaches IV 0.14. Textbook scorecards
routinely report bureau score IVs above 0.40.

**Lending Club only originates to applicants who passed its own screening.** There
are effectively no sub-660 FICO accounts, and applicants with severe derogatory
history were declined before appearing in this data. Within an accepted
population, characteristics that separate strongly through-the-door discriminate
far less.

This is directly visible: `delinq_2yrs` (IV 0.0016), `pub_rec` (0.0019) and
`pub_rec_bankruptcies` (0.0014) are near-zero not because delinquency history is
uninformative about credit risk, but because Lending Club had already used it to
decline the applicants where it mattered.

This is limitation 2 of `docs/target_definition.md` appearing in the numbers, and
it caps achievable performance.

---

## 4. Multicollinearity reduction

Correlated inputs inflate standard errors and destabilise coefficient signs. Since
the sign check in stage 5 is a pass/fail gate, collinearity is removed before
fitting rather than diagnosed after.

### Correlation clustering

Spearman correlation on WOE-transformed features, average-linkage hierarchical
clustering, cut at |r| = 0.70, keeping the highest-IV member of each cluster.

Four pairs breached the threshold:

| Kept | Dropped | r | Interpretation |
|---|---|---:|---|
| **`inq_intensity`** (0.0359) | `inq_last_6mths` (0.0279) | +0.963 | Engineered ratio displaces its raw parent |
| **`dti_with_loan`** (0.0733) | `dti` (0.0533) | +0.914 | Engineered ratio displaces its raw parent |
| `avg_cur_bal` (0.0756) | `tot_cur_bal` (0.0653) | +0.907 | Average and total balance, same construct scaled by account count |
| `mo_sin_old_rev_tl_op` (0.0375) | `credit_hist_months` (0.0244) | +0.879 | Both measure credit file age |

**Two of the four drops are engineered ratios displacing the raw fields they were
built from.** The ratios won on IV and then won the cluster, which is the outcome
the construction was designed to produce.

The fourth selection is worth noting: `credit_hist_months` is the more intuitive
characteristic, and `mo_sin_old_rev_tl_op` (months since oldest revolving account
opened) won on IV alone. They describe substantially the same thing, and the
survivor still maps cleanly to the standard ECOA reason "length of credit history"
for the adverse action codes in stage 9.

**20 → 16 characteristics.**

### Variance Inflation Factor

Iterative elimination at a **VIF ceiling of 5** — stricter than the 10 used outside
credit, because coefficient signs must be individually defensible. Removal is one
at a time, since dropping any feature changes every remaining VIF.

**No characteristics were removed.** Correlation clustering had already eliminated
the redundancy VIF would have caught. Highest values:

| Feature | VIF | Feature | VIF |
|---|---:|---|---:|
| `total_bc_limit` | 4.48 | `revol_util` | 2.60 |
| `revol_bal` | 3.81 | `loan_to_income` | 2.59 |
| `bal_to_bc_limit` | 3.37 | `mort_acc` | 2.19 |
| `annual_inc` | 3.02 | `avg_cur_bal` | 2.16 |
| `loan_amnt` | 2.79 | *(remainder below 2)* | |

### A known limitation of VIF here

The top three values are `total_bc_limit`, `revol_bal` and `bal_to_bc_limit` — a
ratio sitting alongside both of its components. The same structure exists for
`loan_amnt`, `annual_inc` and `loan_to_income`.

**VIF is a linear diagnostic, and a ratio is not a linear function of its
components.** A VIF below 5 therefore does not rule out the instability this
structure creates: a regression can satisfy the fit by pushing one component's
coefficient in a direction that makes no credit sense in isolation.

This was anticipated before fitting and is exactly what happened — see §6.

---

## 5. Characteristic set entering the model

**16 characteristics**, above the 10–15 target at this stage; the significance
screen in stage 5 reduces it further.

| # | Characteristic | IV | # | Characteristic | IV |
|---:|---|---:|---:|---|---:|
| 1 | `fico` | 0.1391 | 9 | `home_ownership` | 0.0411 |
| 2 | `annual_inc` | 0.0937 | 10 | `percent_bc_gt_75` | 0.0380 |
| 3 | `total_bc_limit` | 0.0759 | 11 | `mo_sin_rcnt_tl` | 0.0379 |
| 4 | `avg_cur_bal` | 0.0756 | 12 | `mo_sin_old_rev_tl_op` | 0.0375 |
| 5 | **`dti_with_loan`** | 0.0733 | 13 | **`inq_intensity`** | 0.0359 |
| 6 | `mort_acc` | 0.0494 | 14 | `revol_bal` | 0.0210 |
| 7 | **`bal_to_bc_limit`** | 0.0471 | 15 | `revol_util` | 0.0206 |
| 8 | **`loan_to_income`** | 0.0424 | 16 | `loan_amnt` | 0.0115 † |

† force-kept below the IV floor. **Bold** = engineered.

### Selection waterfall

| Step | Removed | Remaining |
|---|---:|---:|
| Raw fields + 6 engineered ratios | — | 41 |
| IV below 0.02 (1 override) | 21 | 20 |
| Correlation clustering, \|r\| > 0.70 | 4 | 16 |
| VIF above 5 | 0 | **16** |
| *Sign check + significance (stage 5)* | *5* | ***11*** |

---

## 6. Downstream outcome

Five of the 16 did not survive stage 5 (`docs/model_development.md` §1):

**Sign flips — `revol_bal` and `revol_util`.** Both raw revolving measures reversed
sign in the multivariate fit, precisely the instability §4 predicted from having
`revol_bal`, `total_bc_limit` and `bal_to_bc_limit` in the model together. Both
were dropped by the sign-check gate.

**Not significant — `mort_acc`, `bal_to_bc_limit`, `loan_amnt`.**

`loan_amnt` came out at **p = 0.906**. Having been force-kept past the IV screen on
ablation evidence, it was then found to contribute nothing once `loan_to_income`
was present — the ratio absorbs it entirely. The subsequent ablation confirms
this: adding `loan_amnt` to the final characteristic set is now worth +0.0002
Gini, against +0.0178 before the ratios existed.

The whole revolving-utilisation family collapsed to a single survivor. `revol_bal`,
`revol_util` and `bal_to_bc_limit` all fell; `percent_bc_gt_75` covers utilisation
intensity on its own.

**Final: 11 characteristics, OOT Gini 0.3214.**

---

## 7. Type coercion

Three characteristics were coerced rather than engineered:

- **`credit_hist_months`** — months between `earliest_cr_line` and `issue_d`. The
  raw field is a date, which is not a risk characteristic. *(Dropped in
  clustering; retained as an input to `acct_velocity`.)*
- **`emp_length_num`** — `"< 1 year"` … `"10+ years"` mapped to 0–10, nulls left
  null. *(Dropped on IV.)*
- **`fico`** — midpoint of `fico_range_low` and `fico_range_high`, which are a
  published 4-point band and perfectly collinear.

---

## 8. Leakage controls

Binning is fitted on the **training partition only** and applied unchanged to test
and out-of-time. Bin edges are learned parameters; fitting them on data used for
evaluation would leak the target distribution into the boundaries.

Ratio construction in `src/features.py` uses no cross-account statistics — every
value is computed from that applicant's own fields — so there is no path for
information to leak between rows or across splits.

The fitted `BinningProcess` is persisted to `models/binning_process.pkl` and is
the same object used at scoring time, so an applicant scored through the API
passes through byte-identical bin edges to those fitted in development.

---

## 9. Limitations

1. **Univariate screening remains the weakest step.** `bc_limit_to_income` was
   dropped at IV 0.0128 yet ranks 13th of 41 by SHAP in the challenger. The
   `loan_amnt` override was applied on evidence; no equivalent check was run for
   the other 20 discarded characteristics. A forward-selection or
   ablation-validated procedure would be a better design.
2. **VIF cannot detect ratio-component instability**, as §4 sets out. The sign
   check caught it after the fact; nothing caught it before fitting.
3. **Only six ratios were tried**, all encoding obligation-against-capacity. No
   interaction terms (e.g. `fico × dti_with_loan`) were tested; ablation attributes
   +0.0151 Gini to functional form that explicit interactions might partly recover.
4. **Correlation clustering selects on IV**, a univariate criterion, so a cluster
   member with weaker standalone IV but stronger conditional contribution would be
   discarded unseen.

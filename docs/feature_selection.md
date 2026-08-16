# Feature Transformation and Selection

Stages 3 and 4 of the build. This document records how candidate features were
transformed, screened and reduced to the final scorecard characteristic set.

Fitted on the development training sample only: 184,094 accounts, 13.19% bad rate.
Scripts: `src/02_woe_binning.py`, `src/03_feature_selection.py`.

---

## 1. Why Weight of Evidence

Every candidate feature is WOE-transformed before modelling. For a bin *i*:

```
WOE_i = ln( (goods_i / total_goods) / (bads_i / total_bads) )
```

Four properties make this the standard transform in scorecard work:

1. **Linear in log-odds.** WOE expresses each bin directly on the log-odds scale,
   which is the scale logistic regression operates on. A non-linear relationship
   between a raw characteristic and default risk becomes linear after binning,
   without polynomial terms or splines that would be hard to justify in review.
2. **Missing is a category, not a defect.** `mths_since_last_delinq` is null
   precisely when the applicant has never been delinquent — the null *is* the
   signal. WOE assigns missing its own bin with its own weight. **No imputation is
   performed anywhere in this pipeline**, because every imputation strategy here
   would destroy information.
3. **Outlier-insensitive.** Binning caps the influence of extreme values. A
   declared income of $10m lands in the top bin and contributes the same weight as
   any other applicant in it.
4. **Auditable.** Each bin is a stated range with a stated weight. The resulting
   points table can be read by someone who does not write code, which is a
   practical requirement for model risk review.

### Monotonic constraints

All numeric features are fitted with `monotonic_trend="auto_asc_desc"`, forcing a
single direction of risk across the characteristic.

An unconstrained fit will often produce a WOE curve that falls, rises, then falls
again. That shape is nearly always noise in sparse bins, and it fails on two
counts: it cannot be explained to a credit committee ("risk rises with income
between $40k and $55k, then falls again"), and it does not survive a population
shift. The constraint costs a small amount of in-sample IV and buys stability
plus explicability.

Categorical features are ordered by event rate rather than constrained, since
there is no natural ordering to preserve.

---

## 2. Bin granularity

| Parameter | Value | Reason |
|---|---|---|
| `min_bin_size` | 0.05 | No bin below 5% of the training population |
| `max_n_bins` | 8 | Hard ceiling per characteristic |
| `min_prebin_size` | 0.02 | Pre-binning granularity before optimisation |

An initial fit without size constraints produced up to 17 bins on some
characteristics. Constraining to 8 cost very little IV — `fico` moved from 0.1413
to 0.1391 — but improved bin quality substantially:

| Feature | Bins (unconstrained → constrained) | IV | `quality_score` |
|---|---|---|---|
| `fico` | 17 → 8 | 0.1413 → 0.1391 | 0.0136 → **0.5134** |
| `annual_inc` | 14 → 8 | 0.0945 → 0.0937 | 0.0447 → **0.3638** |
| `total_bc_limit` | 13 → 8 | 0.0784 → 0.0759 | 0.1866 → **0.2862** |

`quality_score` measures how statistically distinct adjacent bins are. The
17-bin version was cutting the population into groups that were not reliably
different from one another. Coarsening did not discard signal so much as stop
manufacturing false precision — thin bins produce WOE values that swing when
applied to a new population, which is the failure mode the 2015 out-of-time
window exists to detect.

---

## 3. Information Value screening

IV summarises a characteristic's total discriminatory power across its bins:

```
IV = Σ_i (goods_i/total_goods − bads_i/total_bads) × WOE_i
```

Thresholds applied: **drop below 0.02**, **investigate above 0.50**.

**35 candidates → 15 retained.** The 20 dropped:

`emp_length_num`, `purpose`, `addr_state`, `loan_amnt`, `total_acc`,
`verification_status`, `num_actv_bc_tl`, `mths_since_last_record`, `pub_rec`,
`delinq_2yrs`, `mths_since_last_delinq`, `pub_rec_bankruptcies`,
`pct_tl_nvr_dlq`, `initial_list_status`, `num_tl_90g_dpd_24m`, `open_acc`,
`acc_now_delinq`, `tot_coll_amt`, `application_type`,
`collections_12_mths_ex_med`.

Nothing exceeded the 0.50 ceiling. The absence of any leakage flag is itself a
result: the exclusion list in `docs/target_definition.md` removed the
post-origination fields effectively, and no surviving characteristic behaves
suspiciously like an outcome variable.

### Judgement call: `purpose`

`purpose` scored 0.0191, marginally under the floor, and was dropped. The 0.02
threshold is a convention rather than a rule, and loan purpose is a legitimate
scorecard characteristic with a clear business rationale — debt consolidation and
small business lending do not carry the same risk. It was excluded because it is
a categorical adding little discriminatory power to an already-tight feature
budget, but the decision is a close one and could reasonably go the other way.

### Why the IV levels are modest

The strongest characteristic, `fico`, reaches IV 0.14. Textbook credit scorecards
routinely report bureau score IVs above 0.40. The gap is a property of the data,
not the method.

**Lending Club only originates to applicants who passed its own screening.** The
population is truncated — there are effectively no sub-660 FICO accounts, and
applicants with severe derogatory history were already declined. Within an
accepted population, characteristics that would separate strongly
through-the-door discriminate far less. This is directly visible in the screen:
`delinq_2yrs` (IV 0.0016), `pub_rec` (0.0019) and `pub_rec_bankruptcies` (0.0014)
are near-zero not because delinquency history is uninformative about credit risk,
but because Lending Club had already used it to decline the applicants where it
mattered.

This is limitation 2 of `docs/target_definition.md` appearing in the numbers, and
it caps achievable performance. Correcting it would require reject inference on
the declined population, which is out of scope here.

---

## 4. Multicollinearity reduction

Correlated inputs inflate standard errors and destabilise coefficient signs. Since
the coefficient sign check in stage 5 is a pass/fail gate — a characteristic
pointing the wrong way is a rejected model regardless of AUC — collinearity is
removed before fitting rather than diagnosed after.

### Correlation clustering

Spearman correlation on WOE-transformed features, average-linkage hierarchical
clustering, cut at |r| = 0.70. The highest-IV member of each cluster survives.

Two pairs breached the threshold:

| Kept | Dropped | r | Interpretation |
|---|---|---:|---|
| `avg_cur_bal` (IV 0.0756) | `tot_cur_bal` (IV 0.0653) | +0.907 | Average and total current balance are the same construct scaled by account count |
| `mo_sin_old_rev_tl_op` (IV 0.0375) | `credit_hist_months` (IV 0.0244) | +0.879 | Both measure credit file age |

The second selection is worth noting. `credit_hist_months` (months since earliest
credit line) is the more intuitive characteristic; `mo_sin_old_rev_tl_op` (months
since oldest revolving account opened) won on IV alone. The two describe
substantially the same thing, and the surviving feature still maps cleanly to the
standard ECOA reason "length of credit history" for the adverse action codes in
stage 9, so the loss of interpretability is acceptable.

**15 → 13 features.**

### Variance Inflation Factor

Iterative elimination at a **VIF ceiling of 5**, stricter than the 10 commonly
used outside credit. Removal is one feature at a time, since dropping any feature
changes every remaining VIF and batch removal over-prunes.

**No features were removed at this stage** — correlation clustering had already
eliminated the redundancy. Final VIFs:

| Feature | VIF | Feature | VIF |
|---|---:|---|---:|
| `revol_bal` | 2.93 | `percent_bc_gt_75` | 1.87 |
| `total_bc_limit` | 2.69 | `annual_inc` | 1.69 |
| `revol_util` | 2.32 | `fico` | 1.67 |
| `mort_acc` | 2.19 | `mo_sin_old_rev_tl_op` | 1.26 |
| `avg_cur_bal` | 2.14 | `dti` | 1.20 |
| `home_ownership` | 1.91 | `mo_sin_rcnt_tl` | 1.18 |
| | | `inq_last_6mths` | 1.14 |

Every value sits below 3, comfortably inside the ceiling.

---

## 5. Final characteristic set

**13 characteristics**, within the 10–15 target for a deliberately parsimonious
scorecard.

| # | Characteristic | IV | Construct |
|---:|---|---:|---|
| 1 | `fico` | 0.1391 | Bureau score (band midpoint) |
| 2 | `annual_inc` | 0.0937 | Declared income |
| 3 | `total_bc_limit` | 0.0759 | Total bankcard credit limit |
| 4 | `avg_cur_bal` | 0.0756 | Average current balance across accounts |
| 5 | `dti` | 0.0533 | Debt-to-income ratio |
| 6 | `mort_acc` | 0.0494 | Number of mortgage accounts |
| 7 | `home_ownership` | 0.0411 | Own / mortgage / rent |
| 8 | `percent_bc_gt_75` | 0.0380 | % of bankcards above 75% utilisation |
| 9 | `mo_sin_rcnt_tl` | 0.0379 | Months since most recent account opened |
| 10 | `mo_sin_old_rev_tl_op` | 0.0375 | Months since oldest revolving account opened |
| 11 | `inq_last_6mths` | 0.0279 | Credit inquiries, last 6 months |
| 12 | `revol_bal` | 0.0210 | Revolving balance |
| 13 | `revol_util` | 0.0206 | Revolving utilisation |

The set spans the standard credit dimensions: external score, capacity
(`annual_inc`, `dti`), exposure (`total_bc_limit`, `avg_cur_bal`, `revol_bal`),
utilisation intensity (`revol_util`, `percent_bc_gt_75`), file maturity
(`mo_sin_old_rev_tl_op`, `mo_sin_rcnt_tl`), credit-seeking behaviour
(`inq_last_6mths`) and asset proxies (`mort_acc`, `home_ownership`). No single
dimension dominates, which is what makes the reason codes in stage 9 meaningful
rather than repetitive.

### Selection waterfall

| Step | Removed | Remaining |
|---|---:|---:|
| Candidate features after derivation | — | 35 |
| IV below 0.02 | 20 | 15 |
| Correlation clustering, \|r\| > 0.70 | 2 | 13 |
| VIF above 5 | 0 | **13** |

---

## 6. Derived features

Three characteristics were constructed rather than used raw:

- **`credit_hist_months`** — months between `earliest_cr_line` and `issue_d`. The
  raw field is a date, which is not a risk characteristic; file age at the
  observation point is. *(Subsequently dropped in correlation clustering.)*
- **`emp_length_num`** — `"< 1 year"` … `"10+ years"` mapped to 0–10. Nulls left
  null so that binning assigns them their own bin rather than an assumed value.
  *(Subsequently dropped on IV.)*
- **`fico`** — midpoint of `fico_range_low` and `fico_range_high`. The two
  endpoints are a published 4-point band and perfectly collinear; the midpoint is
  the usable value.

---

## 7. Leakage controls

Binning is fitted on the **training partition only** and applied unchanged to
test and out-of-time. Bin edges are learned parameters: fitting them on data
later used for evaluation would leak the target distribution into the boundaries
and inflate measured performance.

The fitted `BinningProcess` is persisted to `models/binning_process.pkl` and is
the same object used at scoring time, so an applicant scored through the API
passes through byte-identical bin edges to those fitted in development.

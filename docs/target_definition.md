# Target Definition and Performance Windows

Stage 1 of the build. This document fixes the modelling population, the
definition of a bad account, and the observation and performance windows.
Everything downstream — binning, the scorecard, the OOT evaluation, the PSI
monitor — depends on the decisions recorded here.

Data: Lending Club accepted loans, `accepted_2007_to_2018Q4.csv.gz`
(Kaggle mirror `wordsforthewise/lending-club`). 2,260,701 accounts,
June 2007 – December 2018.

---

## 1. Definition of bad

| Class | `loan_status` values | Treatment |
|---|---|---|
| **Bad** | `Charged Off`, `Default` | `target = 1` |
| **Good** | `Fully Paid` | `target = 0` |
| **Indeterminate** | `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)` | Excluded from modelling, counted below |
| **Out of scope** | any `Does not meet the credit policy…` | Removed entirely |

### Why this is a lifetime PD, not a 12-month PD

The conventional bank definition would be *90+ days past due within 12 months of
observation*. **That definition is not constructible on this dataset.** Lending
Club's public accepted-loans file publishes a single terminal `loan_status` per
account and no monthly delinquency series, so there is no way to establish when
an account crossed 90 DPD — only whether it ultimately charged off.

The definition adopted here therefore observes outcome over the **full 36-month
contractual term**. In regulatory terms this is closer to an **IFRS 9 lifetime
PD** than to a 12-month Basel PD. The distinction matters for how the output PDs
should be read: they are lifetime probabilities over a three-year horizon, and
are not directly comparable to a 12-month PD used for regulatory capital.

Lending Club charges off at approximately 120–150 days past due, so the bad flag
is a charge-off definition rather than a delinquency definition. This is
conservative — it captures terminal credit loss and excludes accounts that
cured after a delinquency episode.

### Indeterminate accounts

Accounts still open or in early delinquency at the December 2018 data cut have no
observed outcome and are excluded rather than assumed good. Assigning them to
either class would bias the bad rate: treating them as good understates risk,
and dropping them silently without reporting the count hides the size of the
assumption. The count is reported in the waterfall below.

---

## 2. Observation and performance windows

- **Observation point:** `issue_d`. All features are the application snapshot —
  applicant-declared attributes and credit bureau attributes as at origination.
- **Performance window:** the 36 months following origination.
- **Eligibility rule:** a vintage enters the modelling population only once its
  full performance window has closed inside the data.

```
        2013        2014        2015        2016        2017        2018
        |-----------|-----------|-----------|-----------|-----------|-----|
DEV     [====== observation ======]
        └── performance ─────────────────────────────┘ (matures by Dec 2017)

OOT                             [== obs ==]
                                └── performance ─────────────────────┘ (Dec 2018)

MONITOR                                     [===== observation =====]
                                            performance INCOMPLETE — unlabelled
```

| Split | Vintages | Accounts | Bad rate | Purpose |
|---|---|---|---|---|
| **Dev** | Jan 2013 – Dec 2014 | 262,992 | 13.19% | Train/test, split 70/30 stratified within |
| **OOT** | Jan 2015 – Dec 2015 | 283,026 | 14.89% | Out-of-time validation, held out until final evaluation |
| **Monitor** | Jan 2016 – Dec 2017 | 643,914 | n/a | PSI drift monitoring, unlabelled by design |

**Why 2015 is the boundary.** A 36-month loan issued in December 2015 completes
its term in December 2018, the last month in the extract. 2015 is therefore the
most recent vintage whose outcomes are fully observed, which makes it the natural
out-of-time window — the latest possible test of the model against a population
it was not fitted on.

**Why the split is by vintage, not random.** Credit models degrade as the applicant
population and the macro environment shift. A random split draws train and test
from the same population and cannot detect that degradation. The dev-to-OOT bad
rate movement documented in section 5 shows the effect is material here.

---

## 3. Population filters

### 36-month term only

60-month accounts are excluded, for two reasons:

1. **Window coherence.** A single term holds the performance window constant
   across every account. Mixing terms would mean the bad flag represents a
   36-month outcome for some accounts and a 60-month outcome for others, making
   the target internally inconsistent.
2. **Maturity.** No 60-month vintage issued after December 2013 matures inside a
   December 2018 extract, so the 60-month book contributes almost nothing to an
   out-of-time window.

The cost is roughly 650k accounts and a loss of generality: the scorecard applies
to 36-month originations only, and would need to be refitted or extended for the
60-month book.

### Pre-2013 vintages

Excluded from the modelling population. Beyond the maturity requirement, the
bureau attribute block (`mo_sin_*`, `num_tl_*`, `pct_tl_nvr_dlq`, and related)
only populates from approximately 2012. Including earlier vintages would
introduce structural missingness that reflects Lending Club's data collection
history rather than applicant behaviour, distorting the WOE bins.

### Old credit policy accounts

`Does not meet the credit policy…` statuses represent accounts underwritten under
a different, earlier credit policy. They are a different population, not a
different outcome, and are removed rather than relabelled.

---

## 4. Exclusion waterfall

| Step | Accounts removed | Remaining |
|---|---:|---:|
| Raw extract | — | 2,260,701 |
| Old credit policy | 2,749 | 2,257,952 |
| 60-month term | 650,636 | 1,607,316 |
| Indeterminate status | 586,548 | 1,020,768 |
| Outside dev/OOT vintage windows | 474,750 | **546,018** |

Overall bad rate across all labelled 36-month accounts: **16.00%**.

Final analytical base table: 546,018 labelled (dev + OOT) plus 643,914 unlabelled
monitoring accounts = **1,189,932 rows × 46 columns**.

---

## 5. Observed drift between dev and OOT

Bad rate rises from **13.19%** (2013–14) to **14.89%** (2015) — a 1.7pp absolute
increase, roughly 13% in relative terms.

This is consistent with Lending Club loosening underwriting standards as
origination volume scaled through 2015. It is a real population shift, not
sampling noise, and it has two consequences for this build:

- The out-of-time split is doing genuine work. Model performance measured on OOT
  should be expected to sit below in-time test performance, and that gap is the
  honest estimate of deployed performance.
- It motivates the PSI monitor in stage 10. Drift is present in this data, so the
  monitoring pack has something real to detect rather than being a formality.

---

## 6. The unlabelled monitoring population

The 2016–17 vintages are carried as an **unlabelled** population, deliberately.

PSI compares *distributions* — of features and of the model score — between a
development sample and a later population. It does not require outcomes. Because
36-month loans issued in 2016–17 do not mature until 2019–20, applying the bad
definition to them would retain only accounts that had already reached a terminal
status by December 2018.

That subset is not a random sample. Charge-offs resolve quickly; completing 36
payments does not. Filtering on terminal status therefore keeps early defaults
while discarding accounts still performing — **survivorship bias, not drift**.

The effect was measured directly during the build: filtering the 2016–17
population to labelled accounts produced an apparent bad rate of **19.91%** across
360,912 accounts, against 14.89% in the 2015 OOT window. Carrying the population
whole yields 643,914 accounts and no bad rate — which is the correct treatment,
because the inflated figure reflected the filter rather than any deterioration in
credit quality.

---

## 7. Feature scope at the observation point

Only attributes known at or before the origination decision are eligible.

### Excluded — post-origination outcome information

`out_prncp*`, `total_pymnt*`, `total_rec_*`, `recoveries`,
`collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d`,
`last_credit_pull_d`, `last_fico_range_*`, `pymnt_plan`, and all `hardship_*`,
`settlement_*` and `debt_settlement_*` fields.

These are populated during or after the performance window and would leak the
target directly.

### Excluded — Lending Club's own risk assessment

`grade`, `sub_grade`, `int_rate`, `installment`.

These are known at origination and are not leakage in the strict sense, but they
encode Lending Club's internal risk model. A scorecard built on them would be a
re-encoding of another model rather than an independent assessment of the
applicant. `installment` is a deterministic function of `loan_amnt`, `term` and
`int_rate`, so retaining it would reintroduce the interest rate indirectly.

`grade` is retained outside the feature set as a **benchmark**: the scorecard's
OOT Gini is reported alongside the Gini achievable from Lending Club's own grade
on the same accounts.

### Retained

Applicant-declared attributes (`annual_inc`, `emp_length`, `home_ownership`,
`purpose`, `dti`, `verification_status`, `application_type`, `addr_state`), loan
request (`loan_amnt`), and credit bureau attributes as at application
(`fico_range_low/high`, `earliest_cr_line`, `delinq_2yrs`, `inq_last_6mths`,
`open_acc`, `total_acc`, `revol_bal`, `revol_util`, `pub_rec`,
`pub_rec_bankruptcies`, `mths_since_last_delinq`, `mths_since_last_record`,
`acc_now_delinq`, `tot_cur_bal`, `tot_coll_amt`, `mort_acc`, `avg_cur_bal`,
`total_bc_limit`, `mo_sin_*`, `num_*`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`,
`collections_12_mths_ex_med`).

**Note on `fico_range_low/high`:** the bureau score is expected to dominate the IV
ranking. This is legitimate — lenders do use bureau score as a scorecard
characteristic — but a variant excluding it is worth fitting, mirroring the
distinction between an application scorecard and a bureau-score overlay, and
showing what incremental lift the application data provides over the score alone.

---

## 8. Known limitations

1. **Lifetime, not 12-month, PD.** Driven by data availability, as set out in
   section 1. Not directly comparable to a Basel 12-month PD.
2. **Accepted applicants only.** The dataset contains only funded loans, so the
   model is fitted on a population that already passed Lending Club's own
   screening. Applying it to a full through-the-door population would require
   reject inference, which is out of scope here. Reported performance is
   therefore optimistic relative to a genuine application scorecard.
3. **36-month originations only.** Does not generalise to the 60-month book.
4. **Single-lender, single-market.** US marketplace lending, 2013–2015. The
   scorecard is not portable to other products or geographies without refitting.
5. **No macroeconomic overlay.** The observation period is a benign part of the
   credit cycle. Performance under stress is untested.
6. **Charge-off timing.** A small number of accounts issued late in 2015 may
   charge off shortly after December 2018 and are recorded as good. The effect is
   marginal but biases the 2015 bad rate slightly downward.

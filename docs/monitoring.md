# Population Stability Monitoring

Stage 10 of the build. Detecting whether the population has moved away from the
one the scorecard was built on, and where.

Script: `src/12_psi_monitoring.py`. Baseline is the development window
(2013–14, 262,992 accounts); comparison periods run from the 2015 out-of-time
window through eight quarters of 2016–17.

---

## 1. What PSI is for, and why it needs no outcomes

The out-of-time window answers whether the card still ranks on a population whose
outcomes are known. It cannot answer whether *today's* applicants resemble the
development sample, because today's applicants have not defaulted yet.

Population Stability Index compares two binned distributions:

```
PSI = Σ over bins (actual% − expected%) × ln(actual% / expected%)
```

Only the distributions are required. This is why the 2016–17 vintages are carried
**unlabelled** (`docs/target_definition.md` §6): those accounts do not mature until
2019–20 and have no bad flag, but their characteristic distributions are fully
observed, which is all PSI needs. **643,914 accounts** that are useless for
validation are the entire basis of the monitoring pack.

Conventional thresholds are applied: below 0.10 stable, 0.10–0.25 investigate,
above 0.25 the card is operating outside the population it was built for.

### Binning choice

PSI is computed on the **scorecard's own WOE bins**, not on deciles of the raw
values. The bins are what the card uses. A population can shift substantially
within a bin without moving any score, and decile-based PSI would flag that as
drift when it changes no decision. A small shift across a bin boundary does move
scores, and bin-based PSI catches it.

---

## 2. Score-level stability

| Period | n | Score PSI | Mean score | Mean predicted PD | % below cutoff 530 | Flag |
|---|---:|---:|---:|---:|---:|---|
| dev 2013–14 | 262,992 | — | 545.28 | 13.20% | 18.98% | baseline |
| oot 2015 | 283,026 | 0.0006 | 545.67 | 13.09% | 18.67% | stable |
| 2016Q1 | 96,120 | 0.0070 | 546.76 | 12.74% | 17.51% | stable |
| 2016Q2 | 74,537 | 0.0072 | 546.82 | 12.63% | 16.51% | stable |
| 2016Q3 | 73,898 | 0.0093 | 546.46 | 12.66% | 15.80% | stable |
| 2016Q4 | 78,940 | 0.0168 | 547.26 | 12.39% | 14.86% | stable |
| 2017Q1 | 72,410 | 0.0373 | 548.73 | 11.95% | 13.48% | stable |
| 2017Q2 | 77,105 | 0.0394 | 548.87 | 11.92% | 13.68% | stable |
| 2017Q3 | 88,227 | 0.0360 | 548.74 | 11.99% | 14.02% | stable |
| 2017Q4 | 82,677 | 0.0937 | 550.85 | 11.29% | 11.79% | stable |

**Score PSI never breaches 0.10**, peaking at 0.0937 in the final quarter.

A monitoring pack that watched only the score would report green for eight
consecutive quarters. Section 3 shows that conclusion would be wrong.

---

## 3. Characteristic-level stability

PSI by characteristic against the development baseline:

| Characteristic | 2015 | 2016Q1 | 2016Q2 | 2016Q3 | 2016Q4 | 2017Q1 | 2017Q2 | 2017Q3 | 2017Q4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `percent_bc_gt_75` | 0.028 | 0.051 | 0.086 | 0.079 | 0.085 | 0.095 | **0.120** | **0.152** | **0.297** |
| `fico` | 0.003 | 0.007 | 0.008 | 0.004 | 0.005 | 0.014 | 0.026 | 0.043 | **0.133** |
| `mo_sin_old_rev_tl_op` | 0.012 | 0.035 | 0.045 | 0.058 | 0.059 | 0.057 | 0.066 | 0.083 | 0.083 |
| `total_bc_limit` | 0.005 | 0.031 | 0.008 | 0.003 | 0.013 | 0.044 | 0.026 | 0.023 | 0.061 |
| `inq_intensity` | 0.031 | 0.031 | 0.042 | 0.028 | 0.035 | 0.055 | 0.050 | 0.052 | 0.053 |
| `dti_with_loan` | 0.019 | 0.040 | 0.029 | 0.001 | 0.003 | 0.003 | 0.003 | 0.013 | 0.045 |
| `loan_to_income` | 0.002 | 0.003 | 0.005 | 0.025 | 0.039 | 0.026 | 0.046 | 0.042 | 0.035 |
| `home_ownership` | 0.004 | 0.009 | 0.013 | 0.014 | 0.008 | 0.007 | 0.006 | 0.008 | 0.016 |
| `annual_inc` | 0.003 | 0.015 | 0.004 | 0.004 | 0.009 | 0.018 | 0.010 | 0.006 | 0.013 |
| `mo_sin_rcnt_tl` | 0.004 | 0.013 | 0.004 | 0.007 | 0.008 | 0.004 | 0.002 | 0.003 | 0.009 |
| `avg_cur_bal` | 0.004 | 0.006 | 0.003 | 0.002 | 0.004 | 0.004 | 0.002 | 0.004 | 0.004 |

At 2017Q4: **`percent_bc_gt_75` at 0.2967 — ACTION**, `fico` at 0.1333 — monitor.

### The finding that justifies monitoring at both levels

**Score PSI stayed under 0.10 for every period while an individual characteristic
reached 0.30.**

Characteristics drifted substantially and in offsetting directions, and the
aggregate score distribution absorbed the movement. A pack that monitored only the
score — which is the cheaper and more common design — would have reported a stable
population throughout, while the proportion of applicants with near-limit revolving
accounts changed beyond the point where the card's bins remain representative.

This is the practical argument for characteristic-level monitoring, and here it is
a measured result rather than a stated principle.

### Escalation timing

`percent_bc_gt_75` drifted monotonically: 0.028 → 0.051 → 0.086 → 0.079 → 0.085 →
0.095 → 0.120 → 0.152 → 0.297.

It crossed 0.10 in **2017Q2**, two quarters before the sharp rise. A monitoring
process with a 0.10 investigation trigger would have escalated then, with two
quarters of lead time before the characteristic passed 0.25.

---

## 4. What the drift represents

Mean predicted PD falls from 13.20% to 11.29% across the monitoring window, and
the share of applicants below the 530 cutoff falls from 18.98% to 11.79%. **The
through-the-door population improved.**

This is consistent with Lending Club tightening underwriting through 2016–17
following the May 2016 governance crisis, when the CEO resigned over loan
misrepresentation, institutional investors withdrew, and the platform raised rates
and applied stricter criteria.

Two things worth drawing out.

**The monitoring pack detects this on unlabelled accounts.** These loans do not
mature until 2019–20. Waiting for outcomes would mean learning about a
population shift three years after it began; PSI and mean predicted PD identify it
from application data alone.

**Approval rate drifted without a policy change.** A cutoff of 530 declined 19.0%
of the development population and would decline 11.8% of the 2017Q4 population. The
policy was never altered — the population moved underneath it. That is a cutoff
review trigger in its own right, independent of any model performance question, and
it is invisible to anyone monitoring Gini alone.

### A caution on direction

PSI is directionless: it measures distance between distributions, not whether the
change is favourable. Here the drift is benign — a better population, a lower
predicted default rate. The same PSI values would be produced by a deterioration
of equal magnitude.

This matters for how a threshold breach should be actioned. `percent_bc_gt_75` at
0.30 requires investigation because the card's bins may no longer be
representative, **not** because risk has risen. Reading a PSI breach as bad news is
a common error.

---

## 5. Monitoring surface: champion versus challenger

Five characteristics that are **not** on the card were monitored for comparison:

| Characteristic | 2015 | 2016Q4 | 2017Q4 |
|---|---:|---:|---:|
| `purpose` | 0.001 | 0.038 | 0.076 |
| `bc_limit_to_income` | 0.004 | 0.004 | 0.054 |
| `addr_state` | 0.015 | 0.041 | 0.041 |
| `num_actv_bc_tl` | 0.006 | 0.015 | 0.032 |
| `loan_amnt` | 0.004 | 0.016 | 0.025 |

None of these affects the scorecard, so their drift is irrelevant to it. All of
them would require monitoring under the challenger, which uses all 41
characteristics.

**The champion's monitoring pack is 11 characteristics plus the score.** The
challenger's is 41 — and, as `docs/challenger_analysis.md` §6 sets out, 36% of its
advantage comes from *interactions* between characteristics, for which no PSI
equivalent exists. Every input can sit within tolerance while the joint
distribution the model depends on has moved.

The champion is not merely cheaper to monitor. It is monitorable in a way the
challenger is not.

### A correction

An earlier draft of `docs/challenger_analysis.md` §4 explained `addr_state`'s zero
measured contribution by suggesting Lending Club's geographic mix had shifted
materially between vintages.

**PSI does not support that.** `addr_state` peaks at 0.051 — comfortably stable —
and is flat across 2016–17. The marginal distribution of applicant states barely
moved.

The finding that geography contributes nothing to out-of-time performance stands
(measured three times, §4 of that document). The proposed mechanism was wrong. What
must have changed is the *relationship* between state and default rather than the
distribution of states, or the state-level structure the challenger fitted was
never generalisable signal in the first place. The second is more likely.

This is a useful demonstration in its own right: **univariate PSI cannot detect a
change in the relationship between a characteristic and the outcome.** It monitors
inputs, not the mapping from inputs to risk. Detecting that requires outcome data
and a performance monitor, which is a slower signal by construction.

---

## 6. What a production pack would add

The monitoring implemented here covers population stability. A complete pack has
three layers, and only the first is built:

1. **Population stability** — PSI on score and characteristics. *Implemented.*
2. **Performance monitoring** — Gini, KS and calibration on matured vintages, with
   a rolling out-of-time evaluation. Slower, because it waits for outcomes, but it
   is the only layer that detects a change in the input-to-outcome relationship.
3. **Override and decision monitoring** — approval rate, manual override rate,
   score distribution of overridden cases. Detects the model being worked around
   rather than the model failing.

Recommended actions on breach:

| Trigger | Action |
|---|---|
| Any characteristic PSI ≥ 0.10 | Investigate; review the bin distribution and confirm the WOE assignment remains representative |
| Any characteristic PSI ≥ 0.25 | Escalate to model owner; assess whether rebinning that characteristic is required |
| Score PSI ≥ 0.10 | Review cutoff and expected approval rate |
| Score PSI ≥ 0.25 | Card is out of population; rebuild |
| Approval rate moves > 5pp with no policy change | Cutoff review, independent of PSI |
| Calibration gap > 2pp on a matured vintage | Intercept recalibration (`docs/model_development.md` §3) |

---

## 7. Limitations

1. **Univariate PSI only.** No monitoring of the joint distribution or of
   correlations between characteristics. A shift in the relationship between two
   inputs would go undetected even with every individual PSI green.
2. **Cannot detect relationship change**, as §5 sets out. PSI monitors inputs, not
   the mapping from inputs to outcome.
3. **Fixed baseline.** All periods are compared against 2013–14. A rolling baseline
   would show quarter-on-quarter change, which is a different and sometimes more
   actionable signal than cumulative drift.
4. **No performance monitoring on matured vintages.** The 2016 vintages matured in
   2019 and would be observable in a longer extract; this data ends at 2018Q4.
5. **Thresholds are conventional, not derived.** 0.10 and 0.25 are the standard
   values. A calibrated threshold would relate PSI to realised performance
   degradation on this specific card, which requires more matured vintages than
   this dataset provides.
6. **Quarterly granularity.** Monthly monitoring would identify the onset of the
   `percent_bc_gt_75` drift more precisely.

# Adverse Action Reason Codes

Stage 9 of the build. Generating the specific principal reasons a declined
applicant must receive under ECOA / Regulation B, directly from the points table.

Script: `src/11_reason_codes.py`. Evaluated on the 283,026-account out-of-time
window.

---

## 1. The requirement

Under the Equal Credit Opportunity Act and its implementing Regulation B, an
applicant declined on the basis of a credit model must be given the **specific
principal reasons** for the decision. "Your score was too low" does not satisfy
the requirement — the disclosure must identify which characteristics drove the
outcome. Regulation B contemplates up to four principal reasons, which is the
number generated here.

The obligation is not merely to produce an explanation. It is to produce one that
is **specific, reproducible and defensible**, potentially years after issuance and
potentially to an examiner rather than to the applicant.

---

## 2. Why a scorecard makes this arithmetic

Each applicant's score is a sum of attribute points drawn from a fixed 82-row
table across 11 characteristics. The reasons are the characteristics where the
applicant forgave the most points.

That has four consequences, and together they are the strongest operational
argument for the champion (`docs/challenger_analysis.md` §5):

1. **Deterministic.** Two applicants with identical inputs receive identical
   reasons. No sampling, no background dataset, no explainer configuration.
2. **Stable.** Reasons change only when the card is rebuilt, and a rebuild is a
   governed event.
3. **Reproducible by a third party.** Anyone holding the published points table
   can verify a disclosure without the model artefact or any code.
4. **No second system.** The explanation is a property of the model, not a
   parallel component with its own drift behaviour and governance.

---

## 3. Cutoff selection

Reason codes require a decline population, which requires a cutoff. Cutoffs at
various decline rates on the out-of-time window:

| Decline rate | Cutoff score | Declined | Bad rate declined | Bad rate approved |
|---:|---:|---:|---:|---:|
| 5% | 518 | 13,585 | 30.91% | 14.08% |
| 10% | 523 | 26,300 | 28.58% | 13.48% |
| 15% | 527 | 40,708 | 26.99% | 12.85% |
| **20%** | **530** | **54,111** | **25.87%** | **12.29%** |
| 25% | 533 | 69,770 | 24.71% | 11.67% |
| 30% | 535 | 81,333 | 23.97% | 11.22% |
| 40% | 540 | 112,513 | 22.16% | 10.09% |

Portfolio bad rate is 14.89%.

**A cutoff of 530 is used throughout**, declining 19.1% of applicants. Those
declined would have defaulted at 25.87% against 12.29% among those approved — a
2.1× separation, which is what makes the line defensible as a business decision
rather than an arbitrary threshold.

This is an **illustrative** cutoff. A real one comes from optimising loss against
volume with an assumed funding cost and loss-given-default; the table above shows
the trade-off but does not price it. Note the shape: moving from 5% to 40%
declines only reduces the approved book's bad rate from 14.08% to 10.09%, so the
marginal value of each additional decline falls steeply.

---

## 4. Ranking convention, and a defect found by measuring it

The reasons are the characteristics with the largest **points shortfall**. What
the shortfall is measured *against* turns out to matter more than expected.

### Two conventions

**Against the maximum attainable** — the common convention. Answers "what most
limited this score".

**Against the population mean** — answers "what makes this applicant worse than
typical".

Both were computed on all 54,111 declined applicants. Share of applicants for whom
each characteristic appeared among the four reasons:

| Characteristic | Point spread | vs max | vs mean |
|---|---:|---:|---:|
| `mo_sin_rcnt_tl` | 17 | 0.183 | 0.129 |
| `fico` | 29 | 0.250 | 0.113 |
| `dti_with_loan` | 18 | 0.180 | 0.112 |
| `loan_to_income` | 9 | 0.021 | 0.111 |
| `percent_bc_gt_75` | 6 | **0.000** | 0.108 |
| `inq_intensity` | 11 | 0.028 | 0.097 |
| `total_bc_limit` | 14 | 0.204 | 0.082 |
| `avg_cur_bal` | 24 | 0.134 | 0.077 |
| `mo_sin_old_rev_tl_op` | 8 | **0.002** | 0.074 |
| `annual_inc` | 6 | **0.000** | 0.068 |
| `home_ownership` | 3 | **0.000** | 0.029 |

### The defect

**Under the maximum convention, four of eleven characteristics can never appear in
any disclosure.**

The cause is mechanical rather than statistical. `fico` spans 29 points;
`home_ownership` spans 3. Losing *every available point* on housing status costs
3 points — less than a typical *partial* shortfall on FICO. Narrow characteristics
therefore cannot rank in any applicant's top four, no matter what the applicant's
values are.

This is a property of the card's point spreads, not of any applicant. Under
Regulation B it is a genuine weakness: if housing status was in fact a binding
factor for a particular applicant, the convention is structurally incapable of
saying so.

### Resolution

**The population-mean convention is deployed.** Every characteristic appears for
between 2.9% and 12.9% of declined applicants, and the most common reason falls
from 25.0% to 12.9% of declines — a more discriminating set of disclosures with no
unreachable characteristics.

A third option was considered and rejected: normalising shortfall by each
characteristic's own range. That would also equalise appearance rates, but losing
all 3 points on `home_ownership` would register as a 100% shortfall and outrank a
20-point loss on `fico`. It equalises by making the disclosure less truthful. The
mean convention is the right correction because it measures the applicant against
the population rather than against an arbitrary ceiling.

The maximum convention remains implemented and is reported alongside, because the
degeneracy is a more useful finding than a clean result would have been — and any
lender adopting the common convention on a card with uneven point spreads has the
same latent problem.

### The cost of the mean convention

Ranking against the population mean introduces a weakness of its own: for an
applicant close to the cutoff, the fourth reason may rest on a very small
shortfall. The worked example in §5 declines at 527 against a cutoff of 530, and
its fourth reason reflects a shortfall of **0.8 points** against typical.

This is not an artefact of the method so much as a property of the applicant. A
marginal decline genuinely has no strong principal reasons — the applicant sits
slightly below average across the board rather than failing on anything in
particular. Regulation B nonetheless requires reasons, so something must be
disclosed.

A materiality floor — suppressing reasons below a minimum shortfall and
disclosing fewer than four — would be the natural remedy. Regulation B permits
fewer than four reasons where fewer apply. It is not implemented here; see §7.

---

## 5. Disclosure format

Reason statements name the **factor**, not a direction:

| Characteristic | Statement |
|---|---|
| `fico` | Credit bureau score |
| `dti_with_loan` | Total debt obligations relative to income |
| `loan_to_income` | Amount requested relative to income |
| `annual_inc` | Level of income stated on the application |
| `avg_cur_bal` | Level of balances maintained across accounts |
| `total_bc_limit` | Amount of credit available on revolving accounts |
| `mo_sin_rcnt_tl` | Time since most recent account was opened |
| `mo_sin_old_rev_tl_op` | Length of time revolving accounts established |
| `inq_intensity` | Recent credit inquiries relative to accounts held |
| `percent_bc_gt_75` | Proportion of revolving accounts near their limit |
| `home_ownership` | Housing status |

This follows the convention used in bureau-score disclosures, and the choice is
deliberate. A statement such as "your income is too low" asserts a threshold the
card does not define — the card assigns points across a range, it does not declare
a minimum. Naming the factor is accurate for every applicant who lost points on
it, whichever side of the distribution they sit on.

### Example

```
Application declined. Score 527 (cutoff 530).

Principal reasons:
  1. Length of time revolving accounts established
  2. Level of balances maintained across accounts
  3. Amount requested relative to income
  4. Amount of credit available on revolving accounts
```

The internal record retains the arithmetic behind each reason — points awarded,
population-typical points, and maximum available — so a disclosure can be
reconstructed and defended on request. Sample disclosures for 25 declined
applicants are in `reports/reason_codes_sample.csv`.

---

## 6. Comparison against SHAP-derived reasons

The same declined applicants were given reasons derived from the challenger's SHAP
values, restricted to characteristics present on the card so the comparison is
about ranking rather than vocabulary.

| Metric | Deployed (mean) | Max convention |
|---|---:|---:|
| Mean overlap of top-4 | 2.60 of 4 | 2.27 of 4 |
| Top-1 exact match | **33.7%** | 40.3% |
| At least 3 of 4 shared | 57.2% | 37.2% |
| At least 2 of 4 shared | 95.2% | 87.4% |

**For roughly two declined applicants in three, the leading reason differs
depending on which method generated it.**

The two metrics moved in opposite directions when the convention changed, which is
informative. Under the maximum convention `fico` led for a quarter of all declines,
and `fico` is also the challenger's strongest SHAP driver, so the two methods
agreed on first place largely by default. The mean convention distributes leading
reasons across all eleven characteristics, so the *sets* now overlap more (2.27 →
2.60) while the *ordering* agrees less (40.3% → 33.7%). The apparently higher top-1
agreement under the maximum convention was an artefact of concentration, not of
better correspondence between the methods.

This is the concrete version of the argument in `docs/challenger_analysis.md` §5.
SHAP-based reason codes are used in production by real lenders and are a legitimate
approach — the point is that they are **not a free substitution**. The two methods
send materially different disclosures to the same person, so adopting SHAP is a
decision that has to be made and defended, not a technical detail.

The two methods also differ in kind, not only in output. Points shortfall is a
property of a published table. SHAP is a local attribution against a baseline that
depends on the background sample, varies between applicants with identical values
on the characteristic being explained, and cannot be reproduced without the model
artefact and the explainer configuration.

---

## 7. Limitations

1. **The cutoff is illustrative.** A production cutoff requires a loss-versus-volume
   optimisation with funding cost and loss-given-default assumptions. §3 shows the
   trade-off without pricing it.
2. **Reason statements have not been legally reviewed.** The phrasing follows
   bureau-disclosure convention but would require compliance sign-off before use.
3. **No counterfactual guidance.** The disclosure names what cost the applicant
   points, not what change would have altered the outcome. Some lenders provide
   the latter; it is a different and harder computation, and can imply promises the
   model cannot keep.
4. **No materiality floor.** Under the mean convention a reason may rest on a
   shortfall of under one point, as in the §5 example. Regulation B permits
   disclosing fewer than four reasons where fewer apply, so suppressing
   immaterial ones would be defensible and arguably more truthful. Not
   implemented.
5. **Applicants near the cutoff receive reasons for a marginal decline.** An
   applicant at 529 against a cutoff of 530 is told four factors with no
   indication that their outcome was close. This is standard practice but is
   worth noting.
6. **The mean reference is computed on the declined population**, not the full
   through-the-door population. Using the full population would shift appearance
   rates somewhat; the declined-population reference answers "worse than other
   declined applicants", which is arguably not the right comparison.

> This is a portfolio exercise, not legal advice. The ECOA framing describes the
> requirement as generally understood; it is not a compliance opinion.

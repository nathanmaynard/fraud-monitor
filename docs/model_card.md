# Model card: account-opening fraud classifier

| | |
|---|---|
| Model | LightGBM binary classifier (`models/model.joblib`), 400 trees, class-weighted |
| Version | v0.2 — `customer_age` withheld from inputs (v0.1 included it; v0.3 proxy-removal tested and rejected; see §5) |
| Owner | Nathan Maynard |
| Training data | BAF `Base`, months 0–4 (fit), month 5 (validation / threshold) |
| Held-out evaluation | months 6–7, treated as post-deployment "production" |

## 1. Intended use

Score new bank-account applications for fraud risk at the point of application. The score feeds a single
threshold: above it the application is **held for manual review**, not auto-declined. Chosen on the
validation month for a 5% false-positive budget (threshold = 0.715).

**Out of scope:** transaction monitoring on existing accounts; any credit or affordability decision;
use on populations materially different from BAF's (which is itself synthetic — see §7).

## 2. Data

- [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) suite, Jesus et al., NeurIPS 2022.
  `Base` variant: 1,000,000 applications over 8 months, 11,029 fraud (1.10%).
- Feature layer in `sql/features.sql`: `-1` sentinels → NULL with explicit `*_missing` flags, velocity ratios
  (short-window / 4-week), phone-validity count. 34 columns in the feature table; 33 reach the model.
- EDA and hypothesis tests: `notebooks/01_eda.ipynb`.

## 3. Personal data and GDPR

| Feature(s) | Personal data? | Used by model? | Notes |
|---|---|---|---|
| `customer_age` | Yes — and a protected characteristic (Equality Act 2010) | **No** | Retained in the feature table for fairness monitoring only. See §5. |
| `income`, `employment_status`, `housing_status`, `credit_risk_score`, `proposed_credit_limit` | Yes | Yes | Lawful basis: legitimate interests, fraud prevention — GDPR Art. 6(1)(f), Recital 47. These correlate with age (proxies) — see §5. |
| `name_email_similarity`, `email_is_free`, `date_of_birth_distinct_emails_4w` | Derived | Yes | No raw name, email or DOB is stored in the feature table. |
| `device_os`, `session_length_in_minutes`, `keep_alive_session`, `device_distinct_emails_8w` | Yes (device / online identifiers) | Yes | |
| `foreign_request`, `source`, `payment_type` | Application metadata | Yes | |
| `velocity_*`, `zip_count_4w`, `bank_branch_count_8w` | Aggregates | Yes | Counts over windows; not individually identifying. |

- **Automated decision-making (Art. 22):** the model does not make a solely automated decision with legal effect —
  a score above threshold routes to a human reviewer. Per-application SHAP contributions are available so a
  reviewer (or the applicant, on request) can see which factors drove the score.
- **Retention:** feature rows are retained only as long as needed for model monitoring; the raw application is
  governed by the bank's onboarding retention schedule, not by this model.
- **Data minimisation:** age was removed from the inputs at a measured cost of 0.9 pt recall (§5). Removing the
  age *proxies* would cost substantially more; that trade-off is documented rather than hidden.

## 4. Performance

Validation = month 5 (time-ordered, never seen in fitting). Threshold fixed for FPR ≤ 5%.

| Model | AUC | Avg precision | Recall @ 5% FPR |
|---|---|---|---|
| Logistic regression | 0.883 | 0.158 | 51.5% |
| **LightGBM v0.2 (deployed)** | **0.894** | **0.186** | **52.9%** |
| LightGBM v0.1 (with age, for reference) | 0.894 | 0.189 | 53.8% |

Held-out production months at the deployed threshold:

| Month | Prevalence | Recall | Realised FPR |
|---|---|---|---|
| 6 | 1.34% | 54.8% | 5.8% |
| 7 | 1.47% | 49.8% | 3.5% |

Read this as: roughly half of fraudulent applications are caught for a review load of ~4–6% of all applications.
Realised FPR wobbles around the 5% budget month to month — the threshold should be re-fitted if it drifts persistently.

## 5. Fairness: the `customer_age` decision

**Finding.** Fraud rate in BAF rises monotonically with age (0.5% at 20 → 5.3% at 90). A model trained with
`customer_age` as an input (v0.1) wrongly held **11.5%** of legitimate applicants aged 50+ vs **3.7%** of
under-50s — a **3.1×** false-positive gap — while catching more of the 50+ fraud (67% vs 46%). Older
legitimate customers were paying, in friction, for the model's accuracy on older fraudsters.

**Decision.** Withhold `customer_age` from the model inputs; keep it in the feature table for monitoring.

**Measured effect (production months 6–7, deployed threshold):**

| | v0.1 with age | v0.2 without age |
|---|---|---|
| Recall @ 5% FPR (validation) | 53.8% | 52.9% |
| FPR, under-50 | 3.7% | 3.8% |
| FPR, 50+ | 11.5% | 9.5% |
| FPR ratio (50+ / under-50) | 3.11× | 2.48× |
| Recall, 50+ | 67.0% | 62.6% |
| Recall, under-50 | 46.2% | 47.0% |

**What this does and doesn't fix.** Removing age costs ~1 point of recall and cuts the disparity by about a
fifth. It does **not** eliminate it: income, proposed credit limit, credit risk score and housing status are
all age-correlated, and the EDA shows fraud in this dataset genuinely skews older. Exclusion is the necessary
first step (you cannot defend a model that takes age as a direct input), not a sufficient one.

**Alternatives considered.**
1. *Group-specific thresholds* to equalise FPR — same model, higher threshold for 50+. Explicit and auditable,
   but applying a different rule by age is itself direct age discrimination under the Equality Act unless
   justified as a proportionate means to a legitimate aim; rejected for now.
2. *Remove the proxies too* (`housing_status`, `income`, `credit_risk_score`, `proposed_credit_limit`,
   `employment_status`) — **measured, rejected.** `experiments/proxy_removal.py`, full table in
   [docs/experiments/proxy_removal.md](experiments/proxy_removal.md):

   | | Recall @ 5% FPR | FPR ratio | Recall, 50+ |
   |---|---|---|---|
   | v0.1 with age | 53.8% | 3.11× | 67.0% |
   | v0.2 no age (deployed) | 52.9% | 2.48× | 62.6% |
   | v0.3 no age, no proxies | 43.7% | 2.17× | 49.9% |

   Dropping the proxies costs **9.2 points of recall** (roughly one in six fraudulent applications that v0.2
   catches would now get through) to move the ratio by 0.31× — about twenty times worse a trade than removing
   age was. And the gap still does not close: the features that remain (`device_os`, missing previous-address
   history, phone validity, session behaviour) are behaviourally age-correlated as well. Deleting features is
   the wrong tool past this point.
3. *Reweighting / fairness-constrained training* (e.g. the constrained methods benchmarked in the BAF paper),
   or post-hoc score adjustment with a documented justification. This is the right next step; not implemented
   in this version.

**Ongoing control.** `monitor.py` computes the FPR ratio on every run; CI fails the build if it exceeds 3.0×.
The current 2.48× is within that gate but should be reported to model-risk governance as a known disparity.

## 6. Monitoring

Run by `fm monitor` on every retrain; outputs in `reports/`.

| Check | Method | Alert |
|---|---|---|
| Feature drift | PSI, training months vs production months, per feature | > 0.25 |
| Performance | Recall @ 5% FPR and realised FPR per month | Recall < 45% or FPR > 7% for 2 consecutive months |
| Fairness | FPR ratio, age ≥ 50 vs < 50, at deployed threshold | > 3.0× (CI gate) |

Current state: 5 features drifted (all rolling-window counts — `velocity_4w` PSI 2.9, `velocity_24h` 1.6,
`velocity_6h` 0.9, `zip_count_4w` 0.4, `date_of_birth_distinct_emails_4w` 0.3). Application volume declines
steadily across the eight months, so raw counts leave the training range; the derived velocity *ratios* stay in
the watch band (PSI 0.13–0.15). Recommended action: retrain on a rolling window and prefer ratio features over
raw counts in the next version.

## 7. Known limitations

- **BAF is synthetic** (CTGAN-generated from a real anonymised dataset, then perturbed). Relationships are
  realistic but absolute numbers do not transfer to a real book; the pipeline, not the AUC, is the deliverable.
- **Class imbalance** handled by class weights only; no resampling or focal loss tried.
- **Threshold selected on a single month** (month 5). A rolling multi-month validation would be more stable.
- **Fairness measured on one attribute** (age) at one cut (50). BAF also supports income-based analysis; not done.
- **No calibration** — scores are ranked, not probabilities. Fine for a threshold; not for expected-loss maths
  in the what-if tool, which currently uses the empirical confusion matrix rather than calibrated probabilities.

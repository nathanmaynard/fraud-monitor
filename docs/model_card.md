# Model card: account-opening fraud classifier

## Intended use
Score new bank-account applications for fraud risk. Output feeds a threshold-based block/review decision.
Not for use on existing-customer transactions.

## Data
- Bank Account Fraud (BAF) suite, Feedzai / NeurIPS 2022 — `Base` variant, 1M synthetic applications over 8 months, ~1.1% fraud.
- Train: months 0–4. Validation (threshold selection): month 5. Held-out "production": months 6–7.

## Personal data & GDPR
| Feature | Personal data? | Note |
|---|---|---|
| customer_age | yes | protected characteristic — used only for fairness monitoring, **not** as a model input? *(decide and document)* |
| income, employment_status, housing_status | yes | lawful basis: legitimate interest (fraud prevention, GDPR Art. 6(1)(f), Recital 47) |
| name_email_similarity, email_is_free | derived | no raw name/email stored |
| device_os, session_length | yes (device data) | |

- Retention: features retained for model monitoring only, TBD months.
- Right to explanation (Art. 22): per-decision SHAP contributions available; automated blocks are routed to manual review.

## Metrics
Fill from `reports/train_metrics.json` after training.
| Model | AUC | AP | Recall @ 5% FPR |
|---|---|---|---|
| logreg | | | |
| lgbm | | | |

## Monitoring
- Feature drift (PSI) vs training months; alert at PSI > 0.25.
- Recall @ 5% FPR and realised FPR per month.
- Fairness: FPR ratio between age ≥ 50 and < 50 groups; CI fails if > 3×.

## Known limitations
- BAF is synthetic (CTGAN-generated from real data); absolute numbers don't transfer to a real book.
- Class imbalance handled with class weights only; no resampling.

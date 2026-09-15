# fraud-monitor

Account-opening fraud detection on the [Bank Account Fraud (BAF)](https://github.com/feedzai/bank-account-fraud) dataset,
with post-deployment drift/fairness monitoring, a CI/CD pipeline, and an interactive **what-if threshold tool**
deployed to Cloud Run.

## What it does

| Layer | Where | Why it matters |
|---|---|---|
| SQL feature layer | `sql/features.sql` (DuckDB) | Same logic runs in a warehouse; `-1` sentinels handled explicitly |
| Data validation | `src/fraud_monitor/schema.py` (pandera) | Gate in CI — bad data fails the build before training |
| Models | logistic regression baseline vs LightGBM | Selected on **recall @ 5% FPR**, not accuracy (1% prevalence) |
| Time-based split | months 0–4 fit, 5 validate, 6–7 "production" | No leakage; mimics real deployment |
| Monitoring | `monitor.py` — PSI drift, per-month recall/FPR, age-group fairness | Model risk / compliance requirement |
| What-if tool | `app/streamlit_app.py` | Slide the threshold, set £ costs, see fraud caught vs customers blocked |
| CI/CD | `.github/workflows` | Lint → tests → synthetic pipeline → fairness gate → deploy to Cloud Run |

**Live app:** https://fraud-monitor-237612887168.europe-west2.run.app

## Quickstart

```bash
uv sync --extra dev
uv run fm all --synthetic        # smoke run, no data download
uv run streamlit run app/streamlit_app.py
```

With real data: put `Base.csv` in `data/raw/` (or `make data` with Kaggle creds) and run `uv run fm all`.

## Results (BAF `Base`, 1M applications, 1.2% fraud)

Validation = month 5 (time-ordered, out-of-sample). Threshold chosen for a 5% false-positive budget.

| Model | AUC | Avg precision | Recall @ 5% FPR |
|---|---|---|---|
| Logistic regression | 0.883 | 0.158 | 51.5% |
| **LightGBM** (selected) | **0.894** | **0.186** | **52.9%** |

Held-out "production" months 6–7: recall 50–55% at a realised FPR of 3.5–5.8%.
`customer_age` is deliberately **not** a model input — see below.

**Monitoring findings**
- Five features drifted (PSI > 0.25) in production months, all rolling-window counts — `velocity_4w` (PSI 2.9), `velocity_24h` (1.6), `velocity_6h` (0.9), `zip_count_4w`, `date_of_birth_distinct_emails_4w`. The derived velocity *ratios* stayed in the "watch" band (0.13–0.15), so ratio features are more robust to volume shifts than raw counts.
- **Fairness:** with `customer_age` as an input, applicants aged 50+ were wrongly held 3.1× more often than under-50s (FPR 11.5% vs 3.7%). Withholding age costs 0.9 pt of recall and cuts the gap to 2.5× — but not to parity, because income, credit limit and credit score are age proxies. The [model card](docs/model_card.md) documents the decision, the measured cost, and the alternatives considered.

## How AI tools were used

Built with Claude Code as a pair programmer. What it did and what I did:

- **Scaffold:** Claude generated the initial repo structure, SQL feature layer, training/monitoring
  code, tests, CI/CD workflows and Dockerfile from a description of the project. I reviewed each
  file before the first commit.
- **Bugs it caught / caused:** the first test run failed on PSI over a boolean column (its own
  code); it fixed it. LightGBM failed on Cloud Run with a missing `libgomp.so.1` — a Dockerfile
  dependency it had left out. Both are the kind of thing you only find by running the code.
- **EDA notebook:** Claude drafted the analysis and the takeaway text — then checked the text against
  the actual outputs and corrected several claims it had written from memory of the BAF paper
  (e.g. which feature separates fraud most strongly). The lesson: generated narrative needs
  verifying against generated numbers.
- **The age decision (§5 of the model card)** was mine. Claude laid out three options with
  trade-offs; I chose to withhold `customer_age`, and it ran the retrain to quantify the cost.
- **Infra:** Claude ran the GCP setup (project, Workload Identity Federation, bucket, secrets).
  I ran the deploy commands myself, as they publish a public endpoint.

## Docs
- [Model card](docs/model_card.md) — data, GDPR considerations, metrics, the age decision, limitations
- [EDA notebook](notebooks/01_eda.ipynb) · [SHAP explanations](notebooks/02_model_explain.ipynb)

## Data attribution
Bank Account Fraud (BAF) dataset suite — Jesus, S., Pombal, J., Alves, D., Cruz, A., Saleiro, P., Ribeiro, R.,
Gama, J., Bizarro, P. *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*,
NeurIPS 2022 Datasets and Benchmarks. Licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/);
used here for non-commercial, educational purposes. The data is not redistributed in this repository.

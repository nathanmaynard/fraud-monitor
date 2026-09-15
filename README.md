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
| Logistic regression | 0.883 | 0.157 | 51.0% |
| **LightGBM** (selected) | **0.894** | **0.189** | **53.8%** |

Held-out "production" months 6–7: recall 52–57% at a realised FPR of 3.8–6.0%.

**Monitoring findings**
- Five features drifted (PSI > 0.25) in production months, all rolling-window counts — `velocity_4w` (PSI 2.9), `velocity_24h` (1.6), `velocity_6h` (0.9), `zip_count_4w`, `date_of_birth_distinct_emails_4w`. The derived velocity *ratios* stayed in the "watch" band (0.13–0.15), so ratio features are more robust to volume shifts than raw counts.
- **Fairness:** at the deployed threshold, applicants aged 50+ have a false-positive rate of 11.5% vs 3.7% for under-50s (3.1×). The model also catches more of their fraud (67% vs 46%). See the [model card](docs/model_card.md) for the discussion and mitigation options.

## How AI tools were used
_Document here: what Claude Code / Copilot generated, what you changed and why. Interviewers ask._

## Docs
- [Model card](docs/model_card.md) — data, GDPR considerations, metrics, limitations

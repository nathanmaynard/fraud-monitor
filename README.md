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

## Results
_Fill in after training on real data — table from `reports/train_metrics.json`, drift chart from `reports/drift.csv`._

## How AI tools were used
_Document here: what Claude Code / Copilot generated, what you changed and why. Interviewers ask._

## Docs
- [Model card](docs/model_card.md) — data, GDPR considerations, metrics, limitations

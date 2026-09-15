"""Load the BAF Base.csv into DuckDB, or generate a small synthetic stand-in for tests/CI.

Real data: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
    kaggle datasets download -d sgpjesus/bank-account-fraud-dataset-neurips-2022 -p data/raw --unzip
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from . import config


def make_synthetic(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the BAF schema. Fraud signal is planted in a few columns and
    drifts in later months so the monitoring code has something to detect."""
    rng = np.random.default_rng(seed)
    month = rng.integers(0, 8, n)
    drift = (month >= 6).astype(float)  # months 6-7 behave differently

    df = pd.DataFrame(
        {
            "income": rng.choice(np.round(np.arange(0.1, 1.0, 0.1), 1), n),
            "name_email_similarity": rng.beta(2, 2, n),
            "prev_address_months_count": rng.integers(-1, 300, n),
            "current_address_months_count": rng.integers(-1, 400, n),
            "customer_age": rng.choice(np.arange(10, 91, 10), n),
            "days_since_request": rng.exponential(1.0, n),
            "intended_balcon_amount": rng.normal(10, 20, n),
            "payment_type": rng.choice(list("ABCDE"), n),
            "zip_count_4w": rng.integers(0, 3000, n),
            "velocity_6h": rng.normal(5000 + 2000 * drift, 3000, n),
            "velocity_24h": rng.normal(4800, 1500, n),
            "velocity_4w": rng.normal(4900, 900, n),
            "bank_branch_count_8w": rng.integers(0, 2000, n),
            "date_of_birth_distinct_emails_4w": rng.integers(0, 40, n),
            "employment_status": rng.choice(list("ABCDEFG"), n),
            "credit_risk_score": rng.integers(-150, 350, n),
            "email_is_free": rng.integers(0, 2, n),
            "housing_status": rng.choice(list("ABCDEFG"), n),
            "phone_home_valid": rng.integers(0, 2, n),
            "phone_mobile_valid": rng.integers(0, 2, n),
            "bank_months_count": rng.integers(-1, 33, n),
            "has_other_cards": rng.integers(0, 2, n),
            "proposed_credit_limit": rng.choice([200.0, 500.0, 1000.0, 1500.0, 2000.0], n),
            "foreign_request": rng.integers(0, 2, n),
            "source": rng.choice(["INTERNET", "TELEAPP"], n, p=[0.99, 0.01]),
            "session_length_in_minutes": rng.exponential(5, n),
            "device_os": rng.choice(["windows", "macintosh", "linux", "x11", "other"], n),
            "keep_alive_session": rng.integers(0, 2, n),
            "device_distinct_emails_8w": rng.integers(-1, 3, n),
            "device_fraud_count": np.zeros(n, dtype=int),
            "month": month,
        }
    )
    logit = (
        -5.0
        - 2.0 * df["name_email_similarity"]
        + 0.004 * df["credit_risk_score"]
        + 0.8 * df["email_is_free"]
        + 0.6 * df["foreign_request"]
        - 0.7 * df["phone_home_valid"]
        + 0.5 * drift
    )
    p = 1 / (1 + np.exp(-logit))
    df["fraud_bool"] = (rng.random(n) < p).astype(int)
    return df


def load_raw() -> pd.DataFrame:
    if config.RAW_CSV.exists():
        return pd.read_csv(config.RAW_CSV)
    raise FileNotFoundError(
        f"{config.RAW_CSV} not found. Download BAF from Kaggle (see data.py docstring) "
        "or use make_synthetic() for a smoke test."
    )


def to_duckdb(df: pd.DataFrame, path=config.DUCKDB_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute("CREATE OR REPLACE TABLE applications AS SELECT * FROM df")
    con.close()

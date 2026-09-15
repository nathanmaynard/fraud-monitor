from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
SQL_DIR = ROOT / "sql"

RAW_CSV = DATA_RAW / "Base.csv"
DUCKDB_PATH = DATA_PROCESSED / "baf.duckdb"
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
# Production-months slice only; this is what ships in the container for the what-if app.
APP_PARQUET = DATA_PROCESSED / "app_features.parquet"
MODEL_PATH = MODELS / "model.joblib"

TARGET = "fraud_bool"
MONTH = "month"
# Months 0-5 train, 6-7 held out as "production" for monitoring.
TRAIN_MONTHS = list(range(6))
PROD_MONTHS = [6, 7]

# Fraud teams fix a false-positive budget and maximise recall within it.
FPR_BUDGET = 0.05

# Protected attribute used for fairness checks (BAF paper uses age >= 50).
PROTECTED_ATTR = "customer_age"
PROTECTED_THRESHOLD = 50

# Columns kept in the feature table for monitoring/fairness but withheld from the model.
# customer_age is a protected characteristic; see docs/model_card.md for the decision and its cost.
EXCLUDE_FROM_MODEL = [PROTECTED_ATTR]

CATEGORICAL = ["payment_type", "employment_status", "housing_status", "source", "device_os"]

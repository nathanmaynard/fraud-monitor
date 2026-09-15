from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config, metrics
from .features import split_xy


def _baseline(numeric, categorical) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def _lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        verbose=-1,
    )


def train(
    feat: pd.DataFrame,
    model_path=config.MODEL_PATH,
    report_path=config.REPORTS / "train_metrics.json",
    exclude: list[str] | None = None,
):
    train_df = feat[feat[config.MONTH].isin(config.TRAIN_MONTHS)]
    # last training month is the validation slice (time-ordered, no leakage)
    val_month = max(config.TRAIN_MONTHS)
    fit = train_df[train_df[config.MONTH] != val_month]
    val = train_df[train_df[config.MONTH] == val_month]

    X_fit, y_fit = split_xy(fit, exclude)
    X_val, y_val = split_xy(val, exclude)
    categorical = [c for c in X_fit.columns if c in config.CATEGORICAL]
    numeric = [c for c in X_fit.columns if c not in categorical]

    models = {
        "logreg": _baseline(numeric, categorical),
        "lgbm": _lgbm(),
    }
    results = {}
    for name, m in models.items():
        m.fit(X_fit, y_fit)
        s = m.predict_proba(X_val)[:, 1]
        results[name] = metrics.summary(y_val, s, config.FPR_BUDGET)

    key = f"recall_at_fpr_{config.FPR_BUDGET}"
    best_name = max(results, key=lambda k: results[k][key])
    best = models[best_name]
    scores_val = best.predict_proba(X_val)[:, 1]
    thr = metrics.threshold_at_fpr(y_val, scores_val, config.FPR_BUDGET)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": best, "name": best_name, "threshold": thr, "features": list(X_fit.columns),
              "excluded": list(config.EXCLUDE_FROM_MODEL if exclude is None else exclude)}
    joblib.dump(bundle, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"validation": results, "selected": best_name, "threshold": thr}, indent=2))
    return best_name, results, thr

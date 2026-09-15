"""Post-deployment monitoring: drift, performance by month, fairness."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config, metrics
from .features import split_xy


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drifted."""
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.dtype.name in ("category", "object", "bool") or expected.dtype == object:
        cats = sorted(set(expected) | set(actual))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0)
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0)
    else:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0
        e = pd.Series(np.histogram(expected, edges)[0] / len(expected))
        a = pd.Series(np.histogram(actual, edges)[0] / len(actual))
    eps = 1e-6
    e, a = e + eps, a + eps
    return float(np.sum((a - e) * np.log(a / e)))


def feature_drift(train_feat: pd.DataFrame, prod_feat: pd.DataFrame) -> pd.DataFrame:
    X_tr, _ = split_xy(train_feat)
    X_pr, _ = split_xy(prod_feat)
    rows = [{"feature": c, "psi": psi(X_tr[c], X_pr[c])} for c in X_tr.columns]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False)
    out["status"] = pd.cut(out["psi"], [-1, 0.1, 0.25, np.inf], labels=["stable", "watch", "drifted"])
    return out


def performance_by_month(feat: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    rows = []
    for m, g in feat.groupby(config.MONTH):
        X, y = split_xy(g)
        s = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
        row = metrics.summary(y, s, config.FPR_BUDGET)
        row.update(metrics.confusion_at_threshold(y, s, bundle["threshold"]))
        row["month"] = m
        row["phase"] = "train" if m in config.TRAIN_MONTHS else "prod"
        rows.append(row)
    return pd.DataFrame(rows)


def fairness(feat: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """FPR / recall split by protected group at the deployed threshold.
    A large FPR gap means one group is being wrongly blocked more often."""
    X, y = split_xy(feat)
    s = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    group = np.where(feat[config.PROTECTED_ATTR] >= config.PROTECTED_THRESHOLD, "older", "younger")
    rows = []
    for gname in ["younger", "older"]:
        mask = group == gname
        conf = metrics.confusion_at_threshold(y[mask], s[mask], bundle["threshold"])
        conf["group"] = gname
        conf["n"] = int(mask.sum())
        rows.append(conf)
    df = pd.DataFrame(rows)
    df.attrs["fpr_ratio"] = float(df["fpr"].max() / max(df["fpr"].min(), 1e-9))
    return df


def run(feat: pd.DataFrame, model_path=config.MODEL_PATH, out_dir=config.REPORTS) -> dict:
    bundle = joblib.load(model_path)
    train_feat = feat[feat[config.MONTH].isin(config.TRAIN_MONTHS)]
    prod_feat = feat[feat[config.MONTH].isin(config.PROD_MONTHS)]

    drift = feature_drift(train_feat, prod_feat)
    perf = performance_by_month(feat, bundle)
    fair = fairness(prod_feat, bundle)

    out_dir.mkdir(parents=True, exist_ok=True)
    drift.to_csv(out_dir / "drift.csv", index=False)
    perf.to_csv(out_dir / "performance_by_month.csv", index=False)
    fair.to_csv(out_dir / "fairness.csv", index=False)

    key = f"recall_at_fpr_{config.FPR_BUDGET}"
    summary = {
        "n_features_drifted": int((drift["status"] == "drifted").sum()),
        "top_drifted": drift.head(5)["feature"].tolist(),
        "train_recall_at_fpr": float(perf[perf.phase == "train"][key].mean()),
        "prod_recall_at_fpr": float(perf[perf.phase == "prod"][key].mean()),
        "prod_fpr_at_threshold": float(perf[perf.phase == "prod"]["fpr"].mean()),
        "fairness_fpr_ratio": fair.attrs["fpr_ratio"],
    }
    (out_dir / "monitor_summary.json").write_text(json.dumps(summary, indent=2))
    return summary

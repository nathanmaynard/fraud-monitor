"""Experiment: what does it cost to remove the age *proxies* as well as age itself?

Trains three variants on the same time-ordered split and reports validation recall @ 5% FPR alongside
the production-months fairness ratio. Run: `uv run python experiments/proxy_removal.py`
Writes reports/experiments/proxy_removal.{csv,md}.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd

from fraud_monitor import config, metrics, monitor, train

PROXIES = ["housing_status", "income", "credit_risk_score", "proposed_credit_limit", "employment_status"]

VARIANTS = {
    "v0.1 with age": [],
    "v0.2 no age (deployed)": ["customer_age"],
    "v0.3 no age, no proxies": ["customer_age", *PROXIES],
}


def main():
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    for c in config.CATEGORICAL:
        feat[c] = feat[c].astype("category")
    prod = feat[feat[config.MONTH].isin(config.PROD_MONTHS)]
    out_dir = config.REPORTS / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, exclude in VARIANTS.items():
        mp = out_dir / f"{name.split()[0]}.joblib"
        best, results, thr = train.train(feat, model_path=mp, report_path=out_dir / f"{name.split()[0]}.json", exclude=exclude)
        bundle = joblib.load(mp)
        val = results[best]
        fair = monitor.fairness(prod, bundle)
        s = bundle["model"].predict_proba(prod[bundle["features"]])[:, 1]
        conf = metrics.confusion_at_threshold(prod[config.TARGET], s, thr)
        by = fair.set_index("group")
        rows.append({
            "variant": name,
            "n_features": len(bundle["features"]),
            "model": best,
            "val_auc": val["auc"],
            "val_recall_at_5pct_fpr": val[f"recall_at_fpr_{config.FPR_BUDGET}"],
            "prod_recall": conf["recall"],
            "prod_fpr": conf["fpr"],
            "fpr_under50": by.loc["younger", "fpr"],
            "fpr_50plus": by.loc["older", "fpr"],
            "fpr_ratio": fair.attrs["fpr_ratio"],
            "recall_50plus": by.loc["older", "recall"],
        })
        print(f"{name:28s} recall@5%FPR {rows[-1]['val_recall_at_5pct_fpr']:.3f}  FPR ratio {rows[-1]['fpr_ratio']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "proxy_removal.csv", index=False)
    md = df.assign(
        val_auc=df.val_auc.round(3),
        val_recall_at_5pct_fpr=(df.val_recall_at_5pct_fpr * 100).round(1).astype(str) + "%",
        prod_recall=(df.prod_recall * 100).round(1).astype(str) + "%",
        prod_fpr=(df.prod_fpr * 100).round(1).astype(str) + "%",
        fpr_under50=(df.fpr_under50 * 100).round(1).astype(str) + "%",
        fpr_50plus=(df.fpr_50plus * 100).round(1).astype(str) + "%",
        fpr_ratio=df.fpr_ratio.round(2).astype(str) + "×",
        recall_50plus=(df.recall_50plus * 100).round(1).astype(str) + "%",
    ).to_markdown(index=False)
    (out_dir / "proxy_removal.md").write_text(md + "\n")
    print(md)
    (out_dir / "proxy_removal_config.json").write_text(json.dumps({"proxies": PROXIES}))


if __name__ == "__main__":
    main()

"""Fraud-appropriate metrics. Accuracy is meaningless at ~1% prevalence."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def threshold_at_fpr(y_true, scores, fpr_budget: float) -> float:
    """Largest threshold whose FPR is <= budget."""
    fpr, _, thr = roc_curve(y_true, scores)
    ok = np.where(fpr <= fpr_budget)[0]
    return float(thr[ok[-1]]) if len(ok) else float("inf")


def recall_at_fpr(y_true, scores, fpr_budget: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = np.where(fpr <= fpr_budget)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def confusion_at_threshold(y_true, scores, thr: float) -> dict:
    y_true = np.asarray(y_true)
    pred = np.asarray(scores) >= thr
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    fn = int((~pred & (y_true == 1)).sum())
    tn = int((~pred & (y_true == 0)).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fpr": fp / max(fp + tn, 1),
    }


def expected_cost(conf: dict, cost_missed_fraud: float, cost_false_block: float) -> float:
    return conf["fn"] * cost_missed_fraud + conf["fp"] * cost_false_block


def summary(y_true, scores, fpr_budget: float) -> dict:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        f"recall_at_fpr_{fpr_budget}": recall_at_fpr(y_true, scores, fpr_budget),
        "prevalence": float(np.mean(y_true)),
        "n": len(y_true),
    }

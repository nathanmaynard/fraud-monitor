import numpy as np

from fraud_monitor import metrics
from fraud_monitor.monitor import psi


def test_recall_at_fpr_perfect_scores():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.9, 0.95])
    assert metrics.recall_at_fpr(y, s, 0.0) == 1.0


def test_threshold_respects_budget():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    s = rng.random(2000)
    thr = metrics.threshold_at_fpr(y, s, 0.05)
    assert metrics.confusion_at_threshold(y, s, thr)["fpr"] <= 0.05


def test_expected_cost():
    conf = {"fn": 2, "fp": 10, "tp": 0, "tn": 0}
    assert metrics.expected_cost(conf, 100, 5) == 250


def test_psi_identical_is_zero():
    import pandas as pd
    a = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(a, a) < 1e-3


def test_psi_detects_shift():
    import pandas as pd
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(2, 1, 5000))
    assert psi(a, b) > 0.25

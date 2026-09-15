import numpy as np

from fraud_monitor import policy


def _sample():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.02).astype(int)
    scores = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    return y, scores


def test_curve_monotone():
    y, s = _sample()
    c = policy.threshold_curve(y, s)
    assert (np.diff(c["held"]) <= 0).all()      # raising the threshold never holds more cases
    assert (np.diff(c["recall"]) <= 0).all()


def test_capacity_cases():
    assert policy.capacity_cases(analysts=2, hours_per_analyst=100, minutes_per_case=30) == 400


def test_constraint_respected_and_binding():
    y, s = _sample()
    c = policy.threshold_curve(y, s)
    scale = 1.0
    unc = policy.optimise(c, 1000, 10, scale)["unconstrained"]
    tight = unc["held_month"] * 0.5
    res = policy.optimise(c, 1000, 10, scale, capacity=tight)
    assert res["constrained"]["held_month"] <= tight
    assert res["binding"]
    assert res["constrained"]["cost_month"] >= unc["cost_month"]


def test_loose_capacity_not_binding():
    y, s = _sample()
    c = policy.threshold_curve(y, s)
    res = policy.optimise(c, 1000, 10, 1.0, capacity=1e9)
    assert not res["binding"]
    assert res["constrained"]["threshold"] == res["unconstrained"]["threshold"]


def test_shadow_price_positive_when_binding_zero_otherwise():
    y, s = _sample()
    c = policy.threshold_curve(y, s)
    unc = policy.optimise(c, 1000, 10, 1.0)["unconstrained"]
    assert policy.shadow_price(c, 1000, 10, 1.0, capacity=unc["held_month"] * 0.5, extra_cases=50) > 0
    assert policy.shadow_price(c, 1000, 10, 1.0, capacity=1e9, extra_cases=50) == 0

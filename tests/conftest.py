import pytest

from fraud_monitor import data
from fraud_monitor.features import build_features


@pytest.fixture(scope="session")
def raw():
    return data.make_synthetic(n=6000, seed=1)


@pytest.fixture(scope="session")
def feat(raw):
    return build_features(raw)

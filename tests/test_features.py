from fraud_monitor import config


def test_row_count_preserved(raw, feat):
    assert len(feat) == len(raw)


def test_missing_sentinels_nulled(raw, feat):
    n_missing = (raw["bank_months_count"] < 0).sum()
    assert feat["bank_months"].isna().sum() == n_missing
    assert (feat["bank_months"].dropna() >= 0).all()


def test_velocity_ratio_finite(feat):
    assert feat["velocity_ratio_6h_4w"].abs().max() < 1e6


def test_categoricals_typed(feat):
    for c in config.CATEGORICAL:
        assert feat[c].dtype.name == "category"


def test_protected_attribute_excluded_from_model_inputs(feat):
    from fraud_monitor.features import split_xy
    X, _ = split_xy(feat)
    assert config.PROTECTED_ATTR not in X.columns
    assert config.PROTECTED_ATTR in feat.columns  # still available for fairness monitoring

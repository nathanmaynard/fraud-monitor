import pandera.errors
import pytest

from fraud_monitor.schema import raw_schema


def test_synthetic_passes_schema(raw):
    raw_schema.validate(raw)


def test_bad_target_rejected(raw):
    bad = raw.copy()
    bad.loc[0, "fraud_bool"] = 2
    with pytest.raises(pandera.errors.SchemaError):
        raw_schema.validate(bad)

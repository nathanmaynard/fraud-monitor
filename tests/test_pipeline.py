"""End-to-end smoke test on synthetic data: train -> monitor."""
from fraud_monitor import monitor, train


def test_train_and_monitor(feat, tmp_path):
    model_path = tmp_path / "model.joblib"
    name, _results, thr = train.train(feat, model_path=model_path, report_path=tmp_path / "m.json")
    assert name in ("logreg", "lgbm")
    assert 0.0 <= thr <= 1.0
    summary = monitor.run(feat, model_path=model_path, out_dir=tmp_path)
    assert "prod_recall_at_fpr" in summary
    # synthetic data plants drift in velocity_6h for months 6-7
    assert "velocity_6h" in summary["top_drifted"]


def test_train_with_custom_exclusion(feat, tmp_path):
    """Experiments can train variants with a different exclusion list; the bundle records it."""
    import joblib

    from fraud_monitor import config

    exclude = [config.PROTECTED_ATTR, "housing_status", "income"]
    train.train(feat, model_path=tmp_path / "m.joblib", report_path=tmp_path / "m.json", exclude=exclude)
    bundle = joblib.load(tmp_path / "m.joblib")
    assert bundle["excluded"] == exclude
    assert not set(exclude) & set(bundle["features"])
    # monitoring must use the bundle's own feature list, not the global default
    monitor.run(feat, model_path=tmp_path / "m.joblib", out_dir=tmp_path)

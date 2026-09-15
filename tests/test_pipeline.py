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

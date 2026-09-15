"""fm <command>: prepare | train | monitor | all  (add --synthetic to use generated data)."""
from __future__ import annotations

import argparse
import json

from . import config, data, monitor, train
from .features import build_features
from .schema import raw_schema


def _load(synthetic: bool):
    df = data.make_synthetic() if synthetic else data.load_raw()
    return raw_schema.validate(df)


def main(argv=None):
    p = argparse.ArgumentParser(prog="fm")
    p.add_argument("command", choices=["prepare", "train", "monitor", "all"])
    p.add_argument("--synthetic", action="store_true", help="use synthetic data (no download needed)")
    a = p.parse_args(argv)

    if a.command in ("prepare", "all"):
        df = _load(a.synthetic)
        data.to_duckdb(df)
        feat = build_features(df)
        config.FEATURES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(config.FEATURES_PARQUET)
        feat[feat[config.MONTH].isin(config.PROD_MONTHS)].to_parquet(config.APP_PARQUET)
        print(f"prepared {len(feat):,} rows -> {config.FEATURES_PARQUET}")

    if a.command in ("train", "all"):
        import pandas as pd

        feat = pd.read_parquet(config.FEATURES_PARQUET)
        for c in config.CATEGORICAL:
            feat[c] = feat[c].astype("category")
        name, results, thr = train.train(feat)
        print(f"selected {name} @ threshold {thr:.4f}")
        print(json.dumps(results, indent=2))

    if a.command in ("monitor", "all"):
        import pandas as pd

        feat = pd.read_parquet(config.FEATURES_PARQUET)
        for c in config.CATEGORICAL:
            feat[c] = feat[c].astype("category")
        print(json.dumps(monitor.run(feat), indent=2))


if __name__ == "__main__":
    main()

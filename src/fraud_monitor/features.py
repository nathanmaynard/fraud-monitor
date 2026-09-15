from __future__ import annotations

import duckdb
import pandas as pd

from . import config


def build_features(df: pd.DataFrame | None = None, db_path=config.DUCKDB_PATH) -> pd.DataFrame:
    """Run sql/features.sql against either an in-memory frame or the DuckDB file."""
    sql = (config.SQL_DIR / "features.sql").read_text()
    if df is not None:
        con = duckdb.connect()
        con.register("applications", df)
    else:
        con = duckdb.connect(str(db_path), read_only=True)
    out = con.execute(sql).df()
    con.close()
    for c in config.CATEGORICAL:
        out[c] = out[c].astype("category")
    return out


def split_xy(feat: pd.DataFrame):
    X = feat.drop(columns=[config.TARGET, config.MONTH, *config.EXCLUDE_FROM_MODEL], errors="ignore")
    y = feat[config.TARGET]
    return X, y

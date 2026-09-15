"""Threshold policy optimisation: choose the operating point given £ costs and reviewer capacity.

The model produces a score; the *policy* is the threshold. Fraud ops teams don't pick thresholds to minimise
an abstract loss — they have N analysts who can each review so many cases a month. This module makes that
constraint explicit and prices it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics


def threshold_curve(y_true, scores, grid: np.ndarray | None = None) -> pd.DataFrame:
    """Confusion counts at every threshold in `grid` (default 0.01..0.99)."""
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    rows = []
    for t in grid:
        c = metrics.confusion_at_threshold(y_true, scores, float(t))
        c["threshold"] = float(t)
        c["held"] = c["tp"] + c["fp"]
        rows.append(c)
    return pd.DataFrame(rows)


def capacity_cases(analysts: float, hours_per_analyst: float, minutes_per_case: float) -> float:
    """Cases per month a review team can clear."""
    return analysts * hours_per_analyst * 60.0 / minutes_per_case


def optimise(
    curve: pd.DataFrame,
    cost_missed_fraud: float,
    cost_false_block: float,
    scale: float,
    capacity: float | None = None,
) -> dict:
    """Pick thresholds under a £ objective, with and without a review-capacity constraint.

    `scale` converts sample counts to monthly volumes (applications_per_month / len(sample)).
    Returns a dict with rows for 'unconstrained' and 'constrained' (None if infeasible) and the capacity
    that would be needed to run the unconstrained optimum.
    """
    c = curve.copy()
    c["held_month"] = c["held"] * scale
    c["cost_month"] = (c["fn"] * cost_missed_fraud + c["fp"] * cost_false_block) * scale

    def row(i):
        return None if i is None else c.loc[i].to_dict()

    unc = c["cost_month"].idxmin()
    out = {"unconstrained": row(unc), "constrained": None, "binding": False,
           "capacity_needed_for_unconstrained": float(c.loc[unc, "held_month"])}
    if capacity is not None:
        feas = c[c["held_month"] <= capacity]
        if len(feas):
            con = feas["cost_month"].idxmin()
            out["constrained"] = row(con)
            out["binding"] = bool(c.loc[con, "cost_month"] > c.loc[unc, "cost_month"] + 1e-9)
    return out


def shadow_price(curve: pd.DataFrame, cost_missed_fraud: float, cost_false_block: float, scale: float,
                 capacity: float, extra_cases: float) -> float:
    """£/month saved by adding `extra_cases` of monthly review capacity (e.g. one analyst's worth).
    Zero when capacity isn't binding."""
    base = optimise(curve, cost_missed_fraud, cost_false_block, scale, capacity)["constrained"]
    more = optimise(curve, cost_missed_fraud, cost_false_block, scale, capacity + extra_cases)["constrained"]
    if base is None or more is None:
        return float("nan")
    return float(base["cost_month"] - more["cost_month"])

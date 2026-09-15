"""What-if threshold tool: trade off missed fraud against wrongly blocked customers."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraud_monitor import config, metrics
from fraud_monitor.features import split_xy

st.set_page_config(page_title="Fraud threshold what-if", layout="wide")
st.title("Account-opening fraud: threshold what-if")


@st.cache_resource
def load():
    bundle = joblib.load(config.MODEL_PATH)
    feat = pd.read_parquet(config.FEATURES_PARQUET)
    for c in config.CATEGORICAL:
        feat[c] = feat[c].astype("category")
    prod = feat[feat[config.MONTH].isin(config.PROD_MONTHS)]
    X, y = split_xy(prod)
    scores = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    return bundle, y.to_numpy(), scores, prod


bundle, y, scores, prod = load()

with st.sidebar:
    st.header("Scenario")
    thr = st.slider("Decision threshold", 0.0, 1.0, float(bundle["threshold"]), 0.005)
    cost_fn = st.number_input("Cost of a missed fraud (£)", value=1500.0, step=100.0)
    cost_fp = st.number_input("Cost of wrongly blocking a customer (£)", value=25.0, step=5.0)
    monthly_apps = st.number_input("Applications per month", value=50_000, step=5_000)

conf = metrics.confusion_at_threshold(y, scores, thr)
scale = monthly_apps / len(y)
cost = metrics.expected_cost(conf, cost_fn, cost_fp) * scale

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraud caught (recall)", f"{conf['recall']:.1%}")
c2.metric("Customers wrongly blocked (FPR)", f"{conf['fpr']:.2%}")
c3.metric("Precision of blocks", f"{conf['precision']:.1%}")
c4.metric("Expected monthly cost", f"£{cost:,.0f}")

# cost curve across thresholds
grid = np.linspace(0.01, 0.99, 99)
rows = []
for t in grid:
    c = metrics.confusion_at_threshold(y, scores, t)
    rows.append({"threshold": t, "cost": metrics.expected_cost(c, cost_fn, cost_fp) * scale,
                 "recall": c["recall"], "fpr": c["fpr"]})
curve = pd.DataFrame(rows)
best = curve.loc[curve["cost"].idxmin()]

fig = go.Figure()
fig.add_trace(go.Scatter(x=curve.threshold, y=curve.cost, name="Expected monthly cost", line={"color": "#4C6EF5"}))
fig.add_vline(x=thr, line_dash="dash", line_color="#868E96", annotation_text="current")
fig.add_vline(x=best.threshold, line_dash="dot", line_color="#37B24D", annotation_text="cost-optimal")
fig.update_layout(xaxis_title="Threshold", yaxis_title="£ / month", height=380, margin={"t": 30})
st.plotly_chart(fig, width="stretch")
st.caption(
    f"Cost-optimal threshold under these assumptions: **{best.threshold:.2f}** "
    f"(recall {best.recall:.1%}, FPR {best.fpr:.2%}, £{best.cost:,.0f}/month)."
)

with st.expander("Model & monitoring summary"):
    st.write(f"Model: `{bundle['name']}` — deployed threshold {bundle['threshold']:.4f} "
             f"(FPR budget {config.FPR_BUDGET:.0%} on validation month).")
    p = config.REPORTS / "monitor_summary.json"
    if p.exists():
        st.json(json.loads(p.read_text()))
    d = config.REPORTS / "drift.csv"
    if d.exists():
        st.dataframe(pd.read_csv(d).head(10), hide_index=True)

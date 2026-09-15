"""What-if tool: choose a fraud threshold given £ costs and reviewer capacity."""
import json

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fraud_monitor import config, metrics, policy

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#868e96"

st.set_page_config(page_title="Fraud threshold what-if", layout="wide")
st.title("Account-opening fraud: threshold what-if")
st.caption(
    "The model scores each application; the *policy* is where you draw the line. "
    "This tool shows what each choice costs in missed fraud, blocked customers, and analyst time."
)


@st.cache_resource
def load():
    bundle = joblib.load(config.MODEL_PATH)
    src = config.APP_PARQUET if config.APP_PARQUET.exists() else config.FEATURES_PARQUET
    feat = pd.read_parquet(src)
    for c in config.CATEGORICAL:
        feat[c] = feat[c].astype("category")
    prod = feat[feat[config.MONTH].isin(config.PROD_MONTHS)]
    y = prod[config.TARGET].to_numpy()
    scores = bundle["model"].predict_proba(prod[bundle["features"]])[:, 1]
    return bundle, y, scores, policy.threshold_curve(y, scores)


bundle, y, scores, curve = load()

with st.sidebar:
    st.header("Costs")
    cost_fn = st.number_input("Cost of a missed fraud (£)", value=1500.0, step=100.0, min_value=0.0)
    cost_fp = st.number_input("Cost of wrongly holding a customer (£)", value=25.0, step=5.0, min_value=0.0,
                              help="Abandonment, support contact, reputational — per legitimate applicant held")
    monthly_apps = st.number_input("Applications per month", value=50_000, step=5_000, min_value=1000)
    st.header("Review team")
    analysts = st.number_input("Analysts", value=3.0, step=0.5, min_value=0.5)
    hours = st.number_input("Review hours per analyst per month", value=120.0, step=10.0, min_value=1.0)
    mins = st.number_input("Minutes per case", value=12.0, step=1.0, min_value=1.0)
    analyst_cost = st.number_input("Fully-loaded cost per analyst (£/month)", value=3500.0, step=250.0, min_value=0.0)
    st.header("Threshold")
    thr = st.slider("Decision threshold", 0.01, 0.99, float(round(bundle["threshold"], 2)), 0.01)

scale = monthly_apps / len(y)
capacity = policy.capacity_cases(analysts, hours, mins)
per_analyst = policy.capacity_cases(1, hours, mins)
opt = policy.optimise(curve, cost_fn, cost_fp, scale, capacity)
shadow = policy.shadow_price(curve, cost_fn, cost_fp, scale, capacity, per_analyst)

cur = metrics.confusion_at_threshold(y, scores, thr)
cur_held = (cur["tp"] + cur["fp"]) * scale
cur_cost = metrics.expected_cost(cur, cost_fn, cost_fp) * scale
util = cur_held / capacity

# ---- headline: the threshold you chose ----------------------------------------------------------
st.subheader(f"At threshold {thr:.2f}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fraud caught", f"{cur['recall']:.1%}")
c2.metric("Legit customers held", f"{cur['fpr']:.2%}")
c3.metric("Cases to review / month", f"{cur_held:,.0f}")
c4.metric("Team utilisation", f"{util:.0%}", delta=None if util <= 1 else f"{cur_held - capacity:,.0f} over capacity",
          delta_color="inverse")
c5.metric("Expected cost / month", f"£{cur_cost:,.0f}")
if util > 1:
    st.warning(
        f"This threshold generates {cur_held:,.0f} cases/month but the team can clear {capacity:,.0f}. "
        "Cases will queue; in practice the *effective* threshold is wherever the backlog forces it."
    )

# ---- the three operating points ------------------------------------------------------------------
st.subheader("Operating points")
unc, con = opt["unconstrained"], opt["constrained"]
rows = [("Current (slider)", thr, cur["recall"], cur["fpr"], cur_held, cur_cost)]
if unc:
    rows.append(("Cost-optimal, unlimited reviewers", unc["threshold"], unc["recall"], unc["fpr"], unc["held_month"], unc["cost_month"]))
if con:
    rows.append((f"Cost-optimal within {analysts:g} analysts", con["threshold"], con["recall"], con["fpr"], con["held_month"], con["cost_month"]))
tbl = pd.DataFrame(rows, columns=["policy", "threshold", "fraud caught", "legit held", "cases / month", "£ / month"])
st.dataframe(
    tbl.style.format({"threshold": "{:.2f}", "fraud caught": "{:.1%}", "legit held": "{:.2%}",
                      "cases / month": "{:,.0f}", "£ / month": "£{:,.0f}"}),
    hide_index=True, width="stretch",
)
if con is None:
    st.error(f"No threshold fits within {capacity:,.0f} cases/month — even the strictest setting overloads the team.")
elif opt["binding"]:
    gain = shadow - analyst_cost
    st.info(
        f"**Capacity is the binding constraint.** Running the unconstrained optimum would need "
        f"{opt['capacity_needed_for_unconstrained']:,.0f} cases/month of review "
        f"(~{opt['capacity_needed_for_unconstrained'] / per_analyst:.1f} analysts). "
        f"One more analyst would cut expected cost by **£{shadow:,.0f}/month** against a salary cost of "
        f"£{analyst_cost:,.0f} — net {'gain' if gain > 0 else 'loss'} of **£{abs(gain):,.0f}/month**."
    )
else:
    st.success("Capacity is not binding: the team can run the cost-optimal threshold with room to spare.")

# ---- charts: one measure per axis ----------------------------------------------------------------
curve_m = curve.assign(held_month=curve["held"] * scale,
                       cost_month=(curve["fn"] * cost_fn + curve["fp"] * cost_fp) * scale)
infeasible = curve_m[curve_m["held_month"] > capacity]
x_infeasible_max = float(infeasible["threshold"].max()) if len(infeasible) else None


def vlines(fig):
    fig.add_vline(x=thr, line_dash="dash", line_color=GREY, annotation_text="current", annotation_position="top")
    if unc:
        fig.add_vline(x=unc["threshold"], line_dash="dot", line_color=ORANGE, annotation_text="cost-optimal",
                      annotation_position="bottom")
    if con and opt["binding"]:
        fig.add_vline(x=con["threshold"], line_dash="dot", line_color=AQUA, annotation_text="within capacity",
                      annotation_position="top")
    if x_infeasible_max is not None:
        fig.add_vrect(x0=0, x1=x_infeasible_max, fillcolor=GREY, opacity=0.12, line_width=0,
                      annotation_text="over capacity", annotation_position="inside top left")
    fig.update_layout(height=340, margin={"t": 40, "b": 30}, xaxis_title="Threshold", showlegend=False)
    return fig


left, right = st.columns(2)
with left:
    f1 = go.Figure(go.Scatter(x=curve_m.threshold, y=curve_m.cost_month, line={"color": BLUE, "width": 2},
                              hovertemplate="thr %{x:.2f}<br>£%{y:,.0f}/month<extra></extra>"))
    f1.update_layout(title="Expected cost per month", yaxis_title="£ / month")
    st.plotly_chart(vlines(f1), width="stretch")
with right:
    f2 = go.Figure(go.Scatter(x=curve_m.threshold, y=curve_m.held_month, line={"color": BLUE, "width": 2},
                              hovertemplate="thr %{x:.2f}<br>%{y:,.0f} cases/month<extra></extra>"))
    f2.add_hline(y=capacity, line_color=ORANGE, line_dash="dash",
                 annotation_text=f"team capacity {capacity:,.0f}", annotation_position="top right")
    f2.update_layout(title="Cases sent to review per month", yaxis_title="cases / month")
    st.plotly_chart(vlines(f2), width="stretch")

with st.expander("Model & monitoring summary"):
    st.write(f"Model: `{bundle['name']}` — deployed threshold {bundle['threshold']:.4f} "
             f"(FPR budget {config.FPR_BUDGET:.0%} on validation month). "
             f"Excluded from inputs: {', '.join(bundle.get('excluded', [])) or 'none'}.")
    p = config.REPORTS / "monitor_summary.json"
    if p.exists():
        st.json(json.loads(p.read_text()))
    d = config.REPORTS / "drift.csv"
    if d.exists():
        st.dataframe(pd.read_csv(d).head(10), hide_index=True)

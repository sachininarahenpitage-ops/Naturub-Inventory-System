"""
REEL — Yarn Inventory & Reorder Planner
Streamlit frontend replacing the FastAPI + HTML dashboard.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecaster import artifacts
from forecaster.predict import predict

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="REEL — Yarn Inventory Planner",
    page_icon="🧵",
    layout="wide",
)

UI_DATA_PATH = Path(__file__).parent / "artifacts" / "ui_data.json"

# ── Load precomputed dashboard data ──────────────────────────────────────────
@st.cache_data
def load_ui_data():
    with open(UI_DATA_PATH) as f:
        return json.load(f)

@st.cache_data
def get_items():
    return artifacts.known_items()

ui_data = load_ui_data()
items_map = {item["code"]: item for item in ui_data["items"]}
all_items = get_items()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/thread-spool.png", width=60)
st.sidebar.title("REEL")
st.sidebar.caption("Yarn Inventory & Reorder Planner")
st.sidebar.divider()

selected_item = st.sidebar.selectbox(
    "Select Item",
    options=all_items,
    format_func=lambda x: f"{x}  ({items_map.get(x, {}).get('material', '')})"
)

horizon = st.sidebar.radio("Forecast Window", [30, 90], horizontal=True)
grain = st.sidebar.radio("Grain", ["daily", "weekly"], horizontal=True)

st.sidebar.divider()
st.sidebar.subheader("Current Stock Override")
current_stock_input = st.sidebar.number_input(
    "Enter current stock (units)",
    min_value=0.0,
    value=float(items_map.get(selected_item, {}).get("default_stock", 0)),
    step=100.0,
    help="Placeholder until database is connected"
)

st.sidebar.divider()
st.sidebar.subheader("Order Check")
order_qty = st.sidebar.number_input("Order Quantity", min_value=0.0, step=100.0)
delivery_date = st.sidebar.date_input("Required Delivery Date", value=date.today() + timedelta(days=90))
run_predict_btn = st.sidebar.button("🔍 Run Analysis", use_container_width=True)

# ── Main content ─────────────────────────────────────────────────────────────
item_data = items_map.get(selected_item, {})

st.title(f"🧵 {selected_item}")
col_meta1, col_meta2, col_meta3 = st.columns(3)
col_meta1.metric("Material", item_data.get("material", "—"))
col_meta2.metric("Family", item_data.get("family", "—"))
col_meta3.metric("Lead Time", f"{item_data.get('lead_time_days', '—')} days")

st.divider()

# ── Run prediction ────────────────────────────────────────────────────────────
if run_predict_btn or True:  # always show precomputed data, update on button click
    monthly = item_data.get("monthly", [])
    if monthly:
        history_df = pd.DataFrame(monthly).rename(columns={"m": "date", "v": "y"})
        history_df["date"] = pd.to_datetime(history_df["date"])

        try:
            with st.spinner("Running forecast engine..."):
                result = predict(
                    selected_item,
                    history_df,
                    horizon_days=horizon,
                    grain=grain,
                    order_qty=order_qty if order_qty > 0 else None,
                    delivery_date=delivery_date.isoformat() if order_qty > 0 else None,
                    current_stock=current_stock_input,
                    today=date.today().isoformat(),
                )
        except Exception as e:
            st.error(f"Forecast error: {e}")
            result = None
    else:
        result = None

# ── Status & Key Metrics ──────────────────────────────────────────────────────
if result:
    status = result["status"]
    rop = result["reorder_point"]
    proj = result["projection"]
    fc = result["forecast"]

    status_color = "🔴" if status["status"] == "RED" else "🟢"
    st.subheader(f"Stock Status: {status_color} {status['status']}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Stock", f"{status['current_stock']:,.0f}")
    col2.metric("Reorder Point", f"{rop['reorder_point']:,.0f}", delta=f"{status['margin_to_reorder']:,.0f} margin")
    col3.metric("Safety Stock", f"{rop['safety_stock']:,.0f}")
    col4.metric(f"{horizon}d Forecast", f"{fc['forecast']:,.0f}", help=f"Method: {fc['method']}")

    st.divider()

    # ── Projection ────────────────────────────────────────────────────────────
    col_proj1, col_proj2 = st.columns(2)
    with col_proj1:
        st.subheader("📅 Stock Projection")
        if proj["days_to_reorder_point"] is not None:
            st.info(f"**Hits reorder point** in **{proj['days_to_reorder_point']:.0f} days** ({proj['reorder_point_date']})")
        if proj["days_to_stockout"] is not None:
            st.warning(f"**Stockout** in **{proj['days_to_stockout']:.0f} days** ({proj['stockout_date']})")
        st.caption(f"Daily demand rate: {proj['daily_demand_rate']:,.2f} units/day")

    with col_proj2:
        st.subheader("📊 Reorder Point Breakdown")
        fig_rop = go.Figure(go.Bar(
            x=["Expected LT Demand", "Safety Stock"],
            y=[rop["expected_demand_over_lead_time"], rop["safety_stock"]],
            marker_color=["#4C72B0", "#DD8452"],
            text=[f"{rop['expected_demand_over_lead_time']:,.0f}", f"{rop['safety_stock']:,.0f}"],
            textposition="outside"
        ))
        fig_rop.update_layout(
            height=280, margin=dict(t=10, b=10),
            yaxis_title="Units",
            showlegend=False,
        )
        st.plotly_chart(fig_rop, use_container_width=True)

    # ── Order feasibility ─────────────────────────────────────────────────────
    if "order" in result:
        st.divider()
        st.subheader("📦 Order Analysis")
        order = result["order"]
        col_o1, col_o2, col_o3 = st.columns(3)
        col_o1.metric("Order Quantity", f"{order['order_qty']:,.0f}")
        col_o2.metric("Can Fulfil Now", "✅ Yes" if order["can_fulfill_now"] else "❌ No")
        col_o3.metric("Shortfall", f"{order['shortfall']:,.0f}")

        if "replenishment" in result:
            rep = result["replenishment"]
            if rep.get("already_late"):
                st.error(f"⚠️ {rep['message']}")
            elif rep.get("must_order_by"):
                st.info(f"📋 {rep['message']}")
            else:
                st.success(f"✅ {rep['message']}")

    st.divider()

# ── Demand History Chart ──────────────────────────────────────────────────────
st.subheader("📈 Monthly Demand History")
monthly = item_data.get("monthly", [])
if monthly:
    df_hist = pd.DataFrame(monthly)
    df_hist["m"] = pd.to_datetime(df_hist["m"])
    df_hist = df_hist[df_hist["v"] > 0]

    rop_val = result["reorder_point"]["reorder_point"] if result else item_data.get("reorder_point", None)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_hist["m"], y=df_hist["v"],
        name="Monthly Demand",
        marker_color="#4C72B0",
    ))
    if rop_val:
        fig.add_hline(
            y=rop_val, line_dash="dash", line_color="red",
            annotation_text=f"Reorder Point: {rop_val:,.0f}",
            annotation_position="top right"
        )
    fig.update_layout(
        height=350,
        xaxis_title="Month",
        yaxis_title="Demand (units)",
        margin=dict(t=10, b=10),
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(f"REEL Yarn Inventory Planner · Data as of {ui_data.get('as_of', '—')} · Stock values are placeholders until DB is connected")

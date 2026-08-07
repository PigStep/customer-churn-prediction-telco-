"""Scoring-run tab: risk-tier charts, intervention plan, and the action queue.

Kept in its own script so the batch-scoring page stays a thin tab shell
(see app_pages/01_batch_scoring.py). Rendered via render().
"""
import pandas as pd
import streamlit as st
import altair as alt

from churn.cost import FP_COST
from churn.app_utils import (
    RETENTION_RATE,
    TIER_ORDER,
    TIER_PLAYBOOK,
    expected_monthly_saves,
    load_current_base,
    playbook_for,
)

TIER_COLORS = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}


def render():
    base = load_current_base()

    col1, col2 = st.columns(2)

    # --- Charts ---
    with col1:
        with st.container(border=True):
            st.subheader("Risk tier distribution")
            counts = base["risk_tier"].value_counts().reindex(TIER_ORDER).reset_index()
            counts.columns = ["risk_tier", "count"]
            bar = alt.Chart(counts).mark_bar().encode(
                x=alt.X("risk_tier:N", title=None, sort=TIER_ORDER),
                y=alt.Y("count:Q", title="Customers"),
                color=alt.Color("risk_tier:N", scale=alt.Scale(
                    domain=TIER_ORDER, range=[TIER_COLORS[t] for t in TIER_ORDER]),
                    legend=None),
            )
            st.altair_chart(bar, width="stretch")
            st.caption("High-risk customers are a small slice of the base but carry the "
                       "highest churn probability — prioritize them first.")

    with col2:
        with st.container(border=True):
            st.subheader("Churn probability distribution")
            hist = alt.Chart(base).mark_bar().encode(
                x=alt.X("churn_proba:Q", bin=alt.Bin(step=0.05), title="Churn probability"),
                y=alt.Y("count():Q", title="Customers"),
            )
            band_rules = alt.Chart(pd.DataFrame({"cut": [0.30, 0.70]})).mark_rule(
                color="#64748B", strokeDash=[4, 4]
            ).encode(x="cut:Q")
            st.altair_chart(hist + band_rules, width="stretch")
            st.caption("Dashed lines mark the confidence-band boundaries (0.30 / 0.70).")

    # --- Intervention table ---
    with st.container(border=True):
        st.subheader("Intervention plan by tier")
        rows = []
        for tier in TIER_ORDER:
            tier_mask = base["risk_tier"] == tier
            n = int(tier_mask.sum())
            rev = float(base.loc[tier_mask, "MonthlyCharges"].sum())
            pb = playbook_for(tier)
            rows.append({
                "Tier": tier,
                "Customers": f"{n:,}",
                "Monthly revenue at risk": f"${rev:,.0f}",
                "Intervention": pb["intervention"],
                "Cost / customer": f"${pb['cost_per_customer']:,.2f}",
                "Campaign cost": f"${n * pb['cost_per_customer']:,.0f}",
                "Expected saved / mo": (
                    f"${expected_monthly_saves(rev, n):,.0f}"
                    if pb["cost_per_customer"] > 0 else "—"
                ),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Intervention = the 6-month / 20% discount offer at "
                   f"${FP_COST:,.2f} per customer (the notebook FP cost, "
                   "EDA.ipynb cell 53); low-risk customers receive no action. "
                   "The cost-optimal threshold (~0.17) sits below the 0.30 band "
                   "floor, so the offer reaches most of the base. Expected saved "
                   f"= monthly revenue at risk × {RETENTION_RATE:.0%} retention "
                   "success (Baseline.ipynb cell 44) minus the 6-month offer "
                   "cost amortized over its 6 months — the same "
                   "net-of-program-cost view as the notebook cost model "
                   "(Model.ipynb cell 18).")

    st.divider()

    # --- Model labeled customers dataset ---
    st.subheader("Action queue — at-risk customers")
    selected_tiers = st.pills(
        "Filter by tier",
        ["Medium", "High"],
        default=["Medium", "High"],
        selection_mode="multi",
        label_visibility="collapsed",
    )
    tiers = selected_tiers if selected_tiers else ["Medium", "High"]
    queue = base[base["risk_tier"].isin(tiers)].copy()
    queue["churn_pct"] = (queue["churn_proba"] * 100).round(1)
    queue["intervention"] = queue["risk_tier"].map(
        lambda t: TIER_PLAYBOOK[t]["intervention"]
    )
    queue = queue.sort_values("churn_proba", ascending=False)

    queue_view = queue[[
        "customer_id", "risk_tier", "churn_pct", "tenure",
        "Contract", "InternetService", "TechSupport", "MonthlyCharges", "intervention",
    ]]
    st.dataframe(
        queue_view,
        hide_index=True,
        width="stretch",
        column_config={
            "churn_pct": st.column_config.NumberColumn(
                "Churn prob.", format="%.1f%%",
                help="Predicted probability of churning, as a percentage.",
            ),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly $", format="$%.2f"),
            "risk_tier": st.column_config.TextColumn("Tier"),
        },
        key="action_queue",
    )
    st.caption(f"{len(queue):,} customers above the 30% confidence cutoff.")

    export = queue[[
        "customer_id", "risk_tier", "churn_proba", "tenure", "Contract",
        "InternetService", "MonthlyCharges", "intervention",
    ]]
    st.download_button(
        "Download action queue (CSV)",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name="at_risk_customers.csv",
        mime="text/csv",
        icon=":material/download:",
    )

"""
The page is a thin shell: header KPIs plus two tabs whose bodies live in
separate scripts (run_tab.py / val_tab.py) so each tab stays self-contained.
"""

import streamlit as st

from app_pages.run_tab import render as render_run_tab
from app_pages.val_tab import render as render_val_tab
from churn.app_utils import (
    RETENTION_RATE,
    ensure_artifacts,
    expected_monthly_saves,
    load_current_base,
    load_model_card,
)
from churn.cost import FP_COST

ensure_artifacts()

base = load_current_base()
card = load_model_card()

st.title("Churn risk & action queue")
st.caption("A model-demo implementation of the production nightly job: score every "
           "current customer with the served LightGBM artifact, tier them by churn "
           "confidence, and hand the retention team a prioritized action queue. Labels "
           "are unknown here — that is the point of a scoring run.")

# --- KPI's ---
hm = card["holdout_metrics"]
with st.container(horizontal=True):
    st.metric("Model", card["model"], border=True)
    st.metric("Holdout PR-AUC", f"{hm['pr_auc']:.3f}", border=True)

at_risk_mask = base["risk_tier"].isin(["Medium", "High"])
at_risk = base[at_risk_mask]
rev_at_risk = float(at_risk["MonthlyCharges"].sum())
expected_saves = expected_monthly_saves(rev_at_risk, len(at_risk))

with st.container(horizontal=True):
    st.metric("Current customers", f"{len(base):,}", border=True)
    st.metric("At risk (medium + high)", f"{len(at_risk):,}", border=True)
    st.metric("Monthly revenue at risk", f"${rev_at_risk:,.0f}", border=True)
    st.metric("Expected monthly saves", f"${expected_saves:,.0f}", border=True)

st.caption(f"{len(at_risk):,} of {len(base):,} current customers "
           f"({len(at_risk) / len(base):.1%}) are in the medium or high tier; "
           f"annualized revenue at risk ≈ ${rev_at_risk * 12:,.0f}/yr.")
st.caption(f"Expected monthly saves = at-risk monthly revenue × "
           f"{RETENTION_RATE:.0%} retention success (Harvard Business Review) "
           f"minus the 6-month offer cost (${FP_COST:,.2f}/customer) amortized "
           "over its 6 months — an assumption, not a measured result.")
st.caption("The 'current customers' are an emulation of real customers: the 30% "
           "holdout from the train/serve split, scored as if their churn labels were "
           "unknown.")

run_tab, val_tab = st.tabs(
    ["Scoring run", "Offline validation"], on_change="rerun"
)

with run_tab:
    render_run_tab()

with val_tab:
    render_val_tab()

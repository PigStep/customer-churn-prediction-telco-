"""Page 2 — Batch scoring: the nightly scoring run + action queue + validation."""

import pandas as pd
import streamlit as st
import altair as alt

from churn.app_utils import (
    RETENTION_RATE,
    TIER_ORDER,
    TIER_PLAYBOOK,
    ensure_artifacts,
    expected_monthly_saves,
    load_current_base,
    load_model_card,
    load_validation_df,
    playbook_for,
)

ensure_artifacts()

TIER_COLORS = {"Low": "#22C55E", "Medium": "#F59E0B", "High": "#EF4444"}

base = load_current_base()
card = load_model_card()

st.title("Batch scoring — current customer base")
st.caption("Simulates the production nightly job: score every current customer with the "
           "served LightGBM artifact, tier them by churn confidence, and hand the "
           "retention team a prioritized action queue. Labels are unknown here — that "
           "is the point of a scoring run.")

hm = card["holdout_metrics"]
with st.container(horizontal=True):
    st.metric("Model", card["model"], border=True)
    st.metric("Holdout PR-AUC", f"{hm['pr_auc']:.3f}", border=True)

at_risk_mask = base["risk_tier"].isin(["Medium", "High"])
at_risk = base[at_risk_mask]
rev_at_risk = float(at_risk["MonthlyCharges"].sum())
expected_saves = expected_monthly_saves(rev_at_risk)

with st.container(horizontal=True):
    st.metric("Current customers", f"{len(base):,}", border=True)
    st.metric("At risk (medium + high)", f"{len(at_risk):,}", border=True)
    st.metric("Monthly revenue at risk", f"${rev_at_risk:,.0f}", border=True)
    st.metric("Expected monthly saves", f"${expected_saves:,.0f}", border=True)

st.caption(f"{len(at_risk):,} of {len(base):,} current customers "
           f"({len(at_risk) / len(base):.1%}) are in the medium or high tier; "
           f"annualized revenue at risk ≈ ${rev_at_risk * 12:,.0f}/yr.")
st.caption(f"Expected monthly saves assumes a {RETENTION_RATE:.0%} retention success "
           "rate (Harvard Business Review) — an assumption, not a measured result.")
st.caption("The 'current customers' are an emulation of real customers: the 30% "
           "holdout from the train/serve split, scored as if their churn labels were "
           "unknown.")

run_tab, val_tab = st.tabs(
    ["Scoring run", "Offline validation"], on_change="rerun"
)

with run_tab:
    col1, col2 = st.columns(2)

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
                "Expected saved / mo": f"${expected_monthly_saves(rev):,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Campaign cost = customers × per-customer intervention cost "
                   "(illustrative, Kumo.ai-style). Expected saved = monthly revenue at "
                   f"risk × {RETENTION_RATE:.0%} retention success.")

    st.divider()

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

with val_tab:
    st.caption("The eval pass a production team runs before trusting a scoring run: join "
               "the snapshot back to known outcomes and measure how well each "
               "confidence band actually separates churners.")

    val = load_validation_df()
    total_churn = int(val["Churn"].sum())

    val_rows = []
    for tier in TIER_ORDER:
        m = val["risk_tier"] == tier
        captured = int(val.loc[m, "Churn"].sum())
        val_rows.append({
            "Tier": tier,
            "Customers": int(m.sum()),
            "Churn rate in tier": val.loc[m, "Churn"].mean(),
            "% of churners tier capture": captured / total_churn if total_churn else None,
        })
    table = pd.DataFrame(val_rows)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Customers": st.column_config.NumberColumn("Customers", format="%d"),
            "Churn rate in tier": st.column_config.NumberColumn(
                "Churn rate in tier",
                format="percent",
                help="Of the customers in this tier, the share that actually churned "
                     "(precision) — how strongly the band concentrates churn.",
            ),
            "% of churners tier capture": st.column_config.NumberColumn(
                "% of churners tier capture",
                format="percent",
                help="Of all churners in the holdout, the share that falls in this tier "
                     "(recall) — how much of the churn population this band covers.",
            ),
        },
    )
    st.caption("**Churn rate in tier** — of this tier's customers, the % that actually "
               "churned (precision): a high rate means the band concentrates churn. "
               "**% of churners tier capture** — of all holdout churners, the % that fall "
               "in this tier (recall): how much of the churn population the band "
               "covers.")

    th = card["holdout_metrics"]["cost_optimal_threshold_45"]
    predicted = val["churn_proba"] >= th
    actual = val["Churn"]
    tp = int((predicted & actual).sum())
    fp = int((predicted & ~actual).sum())
    fn = int((~predicted & actual).sum())
    tn = int((~predicted & ~actual).sum())

    col_a, col_b = st.columns(2)
    with col_a:
        with st.container(border=True):
            st.markdown("**Confusion matrix at the cost-optimal threshold**")
            st.markdown(
                f"- True positives (retained churners): **{tp}**\n"
                f"- False positives (false alarms): **{fp}**\n"
                f"- False negatives (missed churners): **{fn}**\n"
                f"- True negatives: **{tn}**"
            )
    with col_b:
        with st.container(border=True):
            st.markdown("**Why bands + cutoff coexist**")
            st.info(
                f"The cost-optimal cutoff is **{th:.3f}** — below the 0.30 band "
                f"boundary — so a purely cost-driven policy would intervene more "
                f"aggressively than the fixed bands.\n\n"
                f":orange[**Production teams usually ship both: the bands drive "
                f"prioritization, the cutoff sets the floor of who is worth a "
                f"retention action.**]"
            )

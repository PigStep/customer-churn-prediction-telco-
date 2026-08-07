"""Offline-validation tab: band quality, predicted-vs-actual distributions,
and the cost-optimal threshold story.

Kept in its own script so the batch-scoring page stays a thin tab shell
(see app_pages/01_batch_scoring.py). Rendered via render().
"""

import pandas as pd
import streamlit as st
import altair as alt

from churn.app_utils import (
    TIER_ORDER,
    load_model_card,
    load_validation_df,
)

CHURNED_COLOR = "#EF4444"
STAYED_COLOR = "#22C55E"


def render():
    card = load_model_card()

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

    with st.container(border=True):
        st.subheader("Predicted probability vs actual churn")
        plot_df = val[["churn_proba", "Churn"]].copy()
        plot_df["outcome"] = plot_df["Churn"].map({True: "Churned", False: "Stayed"})
        hist = alt.Chart(plot_df).mark_bar(opacity=0.6).encode(
            x=alt.X("churn_proba:Q", bin=alt.Bin(step=0.05),
                    title="Predicted churn probability"),
            y=alt.Y("count():Q", title="Customers", stack=None),
            color=alt.Color(
                "outcome:N",
                scale=alt.Scale(
                    domain=["Churned", "Stayed"],
                    range=[CHURNED_COLOR, STAYED_COLOR],
                ),
                legend=alt.Legend(title="Actual"),
            ),
        )
        band_rules = alt.Chart(pd.DataFrame({"cut": [0.30, 0.70]})).mark_rule(
            color="#64748B", strokeDash=[4, 4]
        ).encode(x="cut:Q")
        st.altair_chart(hist + band_rules, width="stretch")
        st.caption("Actual churners (red) sit to the right of stayers (green): the "
                   "0.30 / 0.70 dashed bands capture most of the churn population, "
                   "though the distributions overlap below the 0.30 boundary.")

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

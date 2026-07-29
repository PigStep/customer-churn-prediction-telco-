import streamlit as st
import pandas as pd

from config import (
    DATA_PATH,
    FN_COST,
    FP_COST,
    RETENTION_P,
    MODEL_COMPARISON,
    BUSINESS_INSIGHTS,
    THRESHOLD_RECS,
)
from data import load_data
from charts import (
    churn_distribution_chart,
    churn_by_tenure_chart,
    contract_churn_chart,
    monthly_charges_chart,
    tech_support_chart,
    tenure_boxplot_chart,
)

df = load_data(DATA_PATH)

st.markdown("# Customer Churn Analysis")
st.markdown("##### :material/analytics: Key metrics and insights from the Telco customer churn dataset")

with st.container(horizontal=True):
    st.metric(
        "Total customers",
        f"{len(df):,}",
        border=True,
    )
    st.metric(
        "Churn rate",
        f"{df['Churn'].value_counts(normalize=True).get('Yes', 0):.1%}",
        border=True,
    )
    st.metric(
        "Avg monthly charges",
        f"${df['MonthlyCharges'].mean():.2f}",
        border=True,
    )
    st.metric(
        "Avg tenure",
        f"{df['tenure'].mean():.1f} months",
        border=True,
    )

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader(":material/bar_chart: Churn distribution")
        st.altair_chart(churn_distribution_chart(df), width="stretch")
        st.caption("~27% of customers churned. This class imbalance is a key challenge for ML modeling.")

with col2:
    with st.container(border=True):
        st.subheader(":material/schedule: Churn rate by tenure")
        st.altair_chart(churn_by_tenure_chart(df), width="stretch")

inner_a, inner_b = st.columns(2)

with inner_a:
    with st.container(border=True):
        st.subheader("Tenure distribution")
        st.altair_chart(tenure_boxplot_chart(df), width="stretch")

with inner_b:
    with st.container(border=True):
        st.markdown("##### :material/warning: Danger zone")
        st.markdown("**75%** of churners leave within the first **30 months**.")
        st.caption("New subscribers need early engagement to reduce risk.")

with st.container(border=True):
    st.subheader(":material/bar_chart: Monthly charges by churn")
    st.altair_chart(monthly_charges_chart(df), width="stretch")
    st.caption("Churned customers cluster at higher monthly charges ($70–$110+). Price sensitivity is a clear churn signal.")

col5, col6 = st.columns(2)

with col5:
    with st.container(border=True):
        st.subheader(":material/account_balance: Contract type vs churn")
        st.altair_chart(contract_churn_chart(df), width="stretch")
        st.caption("Month-to-month contracts churn at ~43%, while two-year contracts churn at ~3%. Contract type is the strongest single predictor of churn.")

with col6:
    with st.container(border=True):
        st.subheader(":material/security: Tech support vs churn")
        st.altair_chart(tech_support_chart(df), width="stretch")
        st.caption("Customers without tech support churn at over twice the rate of those with it. Value-added services improve retention.")

with st.container(border=True):
    st.subheader(":material/table: Model comparison")
    st.dataframe(
        MODEL_COMPARISON.style.highlight_max(subset=["PR-AUC", "Churn Recall", "Churn Precision"], color="lightgreen"),
        hide_index=True,
        width="stretch",
    )
    st.caption("LightGBM with scale_pos_weight achieves the best PR-AUC (0.653) and a strong recall of 0.75. Undersampling + RF maximizes recall (0.76) at the cost of precision.")

col7, col8 = st.columns(2)

with col7:
    with st.container(border=True):
        st.subheader(":material/paid: Cost analysis")
        st.markdown(
            f"**False negative** (missed churner): **${FN_COST:,.2f}** — lost customer lifetime value"
        )
        st.markdown(
            f"**False positive** (false alarm): **${FP_COST:,.2f}** — unnecessary retention offer"
        )
        st.markdown(f"**Retention success probability:** {RETENTION_P:.0%} (Harvard Business Review)")
        st.dataframe(THRESHOLD_RECS, hide_index=True, width="stretch")
        st.warning(
            "At retention success below ~20%, the ML model becomes unprofitable "
            "compared to a no-model approach (predicting no one churns)."
        )

with col8:
    with st.container(border=True):
        st.subheader(":material/lightbulb: Recommendations")
        for insight in BUSINESS_INSIGHTS:
            st.markdown(f"- {insight}")
        st.info(
            "**Target segment:** New subscribers on month-to-month contracts "
            "with high monthly bills and no tech support."
        )

st.markdown("---")
st.caption("Data source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — Kaggle")

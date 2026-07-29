import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon=":material/insights:",
    layout="wide",
)

DATA_PATH = "data/dataset.csv"

FN_COST = 997.94
FP_COST = 89.33
RETENTION_P = 0.45

MODEL_COMPARISON = pd.DataFrame({
    "Approach": [
        "RF Baseline",
        "RF + class_weight='balanced'",
        "LightGBM (scale_pos_weight)",
        "Undersampling + RF",
        "Bootstrap Oversampling + RF",
        "SMOTE + RF",
        "GBDT + Undersampling",
    ],
    "Churn Recall": [0.48, 0.48, 0.75, 0.76, 0.58, 0.58, 0.77],
    "Churn Precision": [0.64, 0.64, 0.53, 0.51, 0.59, 0.58, 0.50],
    "PR-AUC": [0.6107, 0.6148, 0.6533, 0.6110, 0.5998, 0.5913, 0.6317],
})

BUSINESS_INSIGHTS = [
    "High-risk customers are new subscribers on month-to-month contracts with high monthly charges and no tech support.",
    "At retention success below ~20%, the ML model becomes unprofitable vs. predicting no one churns.",
    "PR-AUC and Recall-based tuning produce statistically equivalent cost outcomes.",
    "The model's value depends entirely on the retention team's effectiveness.",
]

THRESHOLD_RECS = pd.DataFrame({
    "Retention Success Rate": ["> 20%", "< 20%"],
    "Recommended Threshold": ["0.1 – 0.3", "0.4 – 0.8"],
    "Rationale": [
        "Aggressive: catch more churners since retention is effective",
        "Conservative: only intervene when confident, since retention rarely works",
    ],
})


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"] == 1
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df


df = load_data(DATA_PATH)
total = len(df)
churn_rate = df["Churn"].value_counts(normalize=True).get("Yes", 0)
avg_tenure = df["tenure"].mean()
avg_monthly = df["MonthlyCharges"].mean()

with st.sidebar:
    st.markdown("### :material/filter_alt: Filters")
    contract_filter = st.multiselect(
        "Contract type",
        options=df["Contract"].cat.categories,
        default=list(df["Contract"].cat.categories),
    )
    internet_filter = st.multiselect(
        "Internet service",
        options=df["InternetService"].cat.categories,
        default=list(df["InternetService"].cat.categories),
    )
    payment_filter = st.multiselect(
        "Payment method",
        options=df["PaymentMethod"].cat.categories,
        default=list(df["PaymentMethod"].cat.categories),
    )
    tenure_range = st.slider(
        "Tenure (months)",
        min_value=int(df["tenure"].min()),
        max_value=int(df["tenure"].max()),
        value=(int(df["tenure"].min()), int(df["tenure"].max())),
    )

mask = (
    df["Contract"].isin(contract_filter)
    & df["InternetService"].isin(internet_filter)
    & df["PaymentMethod"].isin(payment_filter)
    & df["tenure"].between(tenure_range[0], tenure_range[1])
)
filtered = df[mask]

st.markdown("# Customer Churn Analysis")
st.markdown("##### :material/analytics: Key metrics and insights from the Telco customer churn dataset")

with st.container(horizontal=True):
    st.metric(
        "Total customers",
        f"{len(filtered):,}",
        delta=f"{len(filtered) - total:+d} vs unfiltered",
        border=True,
        chart_data=filtered.groupby("Contract").size(),
        chart_type="bar",
    )
    st.metric(
        "Churn rate",
        f"{filtered['Churn'].value_counts(normalize=True).get('Yes', 0):.1%}",
        border=True,
        chart_data=filtered.groupby("tenure")["Churn"].apply(lambda x: (x == "Yes").mean()),
        chart_type="line",
    )
    st.metric(
        "Avg monthly charges",
        f"${filtered['MonthlyCharges'].mean():.2f}",
        border=True,
    )
    st.metric(
        "Avg tenure",
        f"{filtered['tenure'].mean():.1f} months",
        border=True,
    )

col1, col2 = st.columns(2)

with col1:
    with st.container(border=True):
        st.subheader(":material/bar_chart: Churn distribution")
        churn_counts = filtered["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        c = (
            alt.Chart(churn_counts)
            .mark_bar()
            .encode(
                x=alt.X("Churn:N", title=None),
                y=alt.Y("Count:Q"),
                color=alt.Color("Churn:N", scale=alt.Scale(domain=["No", "Yes"], range=["#4CAF50", "#FF5722"]), legend=None),
                tooltip=["Churn", "Count"],
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)

with col2:
    with st.container(border=True):
        st.subheader(":material/schedule: Churn rate by tenure")
        tenure_bins = pd.cut(filtered["tenure"], bins=12)
        tenure_churn = (
            filtered.groupby(tenure_bins, observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
        )
        tenure_churn.columns = ["Tenure range", "Churn rate"]
        tenure_churn["Tenure range"] = tenure_churn["Tenure range"].astype(str)
        c = (
            alt.Chart(tenure_churn)
            .mark_line(point=True)
            .encode(
                x=alt.X("Tenure range:N", title="Tenure", sort=None),
                y=alt.Y("Churn rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
                tooltip=["Tenure range", alt.Tooltip("Churn rate", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)
        st.caption("New customers (<10 months tenure) are at highest risk of churning.")

col3, col4 = st.columns(2)

with col3:
    with st.container(border=True):
        st.subheader(":material/account_balance: Contract type vs churn")
        contract_churn = (
            filtered.groupby("Contract", observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
        )
        contract_churn.columns = ["Contract", "Churn rate"]
        c = (
            alt.Chart(contract_churn)
            .mark_bar()
            .encode(
                x=alt.X("Contract:N", sort=["Month-to-month", "One year", "Two year"]),
                y=alt.Y("Churn rate:Q", axis=alt.Axis(format="%")),
                color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
                tooltip=["Contract", alt.Tooltip("Churn rate", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)

with col4:
    with st.container(border=True):
        st.subheader(":material/payment: Payment method vs churn")
        pay_churn = (
            filtered.groupby("PaymentMethod", observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
        )
        pay_churn.columns = ["PaymentMethod", "Churn rate"]
        c = (
            alt.Chart(pay_churn)
            .mark_bar()
            .encode(
                x=alt.X("Churn rate:Q", axis=alt.Axis(format="%")),
                y=alt.Y("PaymentMethod:N", sort="-x"),
                color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
                tooltip=["PaymentMethod", alt.Tooltip("Churn rate", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)

col5, col6 = st.columns(2)

with col5:
    with st.container(border=True):
        st.subheader(":material/bar_chart: Monthly charges by churn")
        c = (
            alt.Chart(filtered)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X("MonthlyCharges:Q", bin=alt.Bin(maxbins=30), title="Monthly charges"),
                y=alt.Y("count():Q", title="Count"),
                color=alt.Color("Churn:N", scale=alt.Scale(domain=["No", "Yes"], range=["#4CAF50", "#FF5722"])),
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)

with col6:
    with st.container(border=True):
        st.subheader(":material/security: Tech support vs churn")
        ts_churn = (
            filtered.groupby("TechSupport", observed=True)["Churn"]
            .apply(lambda x: (x == "Yes").mean())
            .reset_index()
        )
        ts_churn.columns = ["TechSupport", "Churn rate"]
        c = (
            alt.Chart(ts_churn)
            .mark_bar()
            .encode(
                x=alt.X("TechSupport:N", sort=["No", "Yes", "No internet service"]),
                y=alt.Y("Churn rate:Q", axis=alt.Axis(format="%")),
                color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
                tooltip=["TechSupport", alt.Tooltip("Churn rate", format=".1%")],
            )
            .properties(height=300)
        )
        st.altair_chart(c, use_container_width=True)

with st.container(border=True):
    st.subheader(":material/table: Model comparison")
    st.dataframe(
        MODEL_COMPARISON.style.highlight_max(subset=["PR-AUC", "Churn Recall", "Churn Precision"], color="lightgreen"),
        hide_index=True,
        use_container_width=True,
    )

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
        st.dataframe(THRESHOLD_RECS, hide_index=True, use_container_width=True)
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

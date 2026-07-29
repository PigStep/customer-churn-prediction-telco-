import pandas as pd
import altair as alt


def churn_distribution_chart(df: pd.DataFrame) -> alt.Chart:
    churn_counts = df["Churn"].value_counts().reset_index()
    churn_counts.columns = ["Churn", "Count"]
    return (
        alt.Chart(churn_counts)
        .mark_bar()
        .encode(
            x=alt.X("Churn:N", title=None),
            y=alt.Y("Count:Q"),
            color=alt.Color(
                "Churn:N",
                scale=alt.Scale(domain=["No", "Yes"], range=["#A6CFD3", "#0B307F"]),
                legend=None,
            ),
            tooltip=["Churn", "Count"],
        )
        .properties(height=300)
    )


def churn_by_tenure_chart(df: pd.DataFrame) -> alt.Chart:
    bins = range(0, 73, 6)
    labels = [f"{i}-{i+5}" for i in range(0, 72, 6)]
    tenure_bins = pd.cut(df["tenure"], bins=bins, labels=labels)
    tenure_churn = (
        df.groupby(tenure_bins, observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .reset_index()
    )
    tenure_churn.columns = ["Tenure range", "Churn rate"]
    tenure_churn["Tenure range"] = tenure_churn["Tenure range"].astype(str)
    
    return (
        alt.Chart(tenure_churn)
        .mark_bar(point=True)
        .encode(
            x=alt.X("Churn rate:Q", title="Churn rate", axis=alt.Axis(format="%")),
            y=alt.Y("Tenure range:N", title="Tenure (month)", sort=None),
            tooltip=["Tenure range", alt.Tooltip("Churn rate", format=".1%")],
        )
        .properties(height=300)
    )

def contract_churn_chart(df: pd.DataFrame) -> alt.Chart:
    contract_churn = (
        df.groupby("Contract", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .reset_index()
    )
    contract_churn.columns = ["Contract", "Churn rate"]
    return (
        alt.Chart(contract_churn)
        .mark_bar()
        .encode(
            x=alt.X("Contract:N", sort=["Month-to-month", "One year", "Two year"]),
            y=alt.Y("Churn rate:Q", axis=alt.Axis(format="%")),
            color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Contract", alt.Tooltip("Churn rate", format=".1%")],
        )
        .properties(height=300)
    )


def payment_churn_chart(df: pd.DataFrame) -> alt.Chart:
    pay_churn = (
        df.groupby("PaymentMethod", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .reset_index()
    )
    pay_churn.columns = ["PaymentMethod", "Churn rate"]
    return (
        alt.Chart(pay_churn)
        .mark_bar()
        .encode(
            x=alt.X("Churn rate:Q", axis=alt.Axis(format="%")),
            y=alt.Y("PaymentMethod:N", sort="-x"),
            color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["PaymentMethod", alt.Tooltip("Churn rate", format=".1%")],
        )
        .properties(height=300)
    )


def monthly_charges_chart(df: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X("MonthlyCharges:Q", bin=alt.Bin(maxbins=30), title="Monthly charges"),
            y=alt.Y("count():Q", title="Count"),
            color=alt.Color(
                "Churn:N",
                scale=alt.Scale(domain=["No", "Yes"], range=["#A6CFD3","#0B307F"]),
            ),
        )
        .properties(height=300)
    )


def tech_support_chart(df: pd.DataFrame) -> alt.Chart:
    ts_churn = (
        df.groupby("TechSupport", observed=True)["Churn"]
        .apply(lambda x: (x == "Yes").mean())
        .reset_index()
    )
    ts_churn.columns = ["TechSupport", "Churn rate"]
    return (
        alt.Chart(ts_churn)
        .mark_bar()
        .encode(
            x=alt.X("TechSupport:N", sort=["No", "Yes", "No internet service"]),
            y=alt.Y("Churn rate:Q", axis=alt.Axis(format="%")),
            color=alt.Color("Churn rate:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["TechSupport", alt.Tooltip("Churn rate", format=".1%")],
        )
        .properties(height=300)
    )

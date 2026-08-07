"""Page 3 — Single customer lookup (CRM-style view)."""

import pandas as pd
import streamlit as st

from churn.app_utils import (
    TIER_PLAYBOOK,
    ensure_artifacts,
    load_current_base,
    load_pipeline,
    playbook_for,
    risk_tier,
    tier_label,
)

ensure_artifacts()

base = load_current_base()
pipeline = load_pipeline()

st.title("Customer lookup")
st.caption("Look up a single account or score a "
           "hypothetical profile, then read off the churn probability, risk tier, and "
           "the intervention the team should run.")

mode = st.segmented_control(
    "Mode",
    ["Existing customer", "Hypothetical profile"],
    default="Existing customer",
)

if mode == "Existing customer":
    customer_id = st.selectbox(
        "Customer",
        base["customer_id"],
        label_visibility="collapsed",
    )
    row = base[base["customer_id"] == customer_id].iloc[0]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Customer", customer_id, border=True)
    with col2:
        st.metric("Churn probability", f"{row['churn_proba']:.1%}",
                  border=True)
    with col3:
        tier = str(row["risk_tier"])
        st.metric("Risk tier", tier,
                  help="Fixed confidence band (<30% low, 30-70% medium, >70% high).",
                  border=True)

    with st.container(border=True):
        st.subheader("Account profile")
        profile = pd.DataFrame(
            {
                "Field": ["Tenure (months)", "Contract", "Internet service",
                          "Monthly charges", "Total charges", "Tech support",
                          "Payment method"],
                "Value": [f"{row['tenure']}", str(row["Contract"]),
                          str(row["InternetService"]),
                          f"${row['MonthlyCharges']:,.2f}",
                          f"${row['TotalCharges']:,.2f}",
                          str(row["TechSupport"]), str(row["PaymentMethod"])],
            }
        )
        st.dataframe(profile, hide_index=True, width="stretch")

    with st.container(border=True):
        st.subheader("Recommended action")
        pb = playbook_for(tier)
        st.markdown(f"- **Intervention:** {pb['intervention']}")
        st.markdown(f"- **Estimated cost:** ${pb['cost_per_customer']:,.2f} per customer")
        st.caption(
            f"Probability of {tier_label(row['churn_proba'])}. This account "
            f"{'warrants outreach' if tier != 'Low' else 'stays on the standard track'}."
        )

else:
    df = load_current_base()
    template = df.iloc[[0]].drop(
        columns=["customer_id", "churn_proba", "risk_tier"]
    )

    def options(series):
        return list(series.cat.categories)

    with st.form("hypothetical_profile", border=True):
        st.subheader("Score a hypothetical customer")
        left, right = st.columns(2)
        values = {}
        cat_cols = template.select_dtypes(include=["category"]).columns
        bool_cols = template.select_dtypes(include=["bool"]).columns
        num_cols = template.select_dtypes(include=["number"]).columns
        num_defaults = {
            "tenure": (0, 72, 1, 12),
            "MonthlyCharges": (0.0, 200.0, 1.0, 60.0),
            "TotalCharges": (0.0, 10000.0, 10.0, 500.0),
        }
        for i, col in enumerate(cat_cols):
            target = left if i % 2 == 0 else right
            with target:
                values[col] = st.selectbox(col, options(template[col]))
        for col in bool_cols:
            with left:
                values[col] = st.checkbox(col, value=False)
        for col in num_cols:
            lo, hi, step, default = num_defaults[col]
            with right:
                values[col] = st.number_input(
                    col, min_value=lo, max_value=hi, step=step, value=default
                )
        submitted = st.form_submit_button(
            "Score this profile", type="primary", icon=":material/insights:"
        )

    if submitted:
        row = template.copy()
        for col, value in values.items():
            row.loc[0, col] = value
        proba = float(pipeline.predict_proba(row)[:, 1][0])
        tier = risk_tier(proba)
        pb = TIER_PLAYBOOK[tier]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Churn probability", f"{proba:.1%}", border=True)
        with c2:
            st.metric("Risk tier", tier, border=True)
        with c3:
            st.metric("Monthly charges", f"${values['MonthlyCharges']:,.2f}", border=True)

        st.info(
            f"**Intervention:** {pb['intervention']} "
            f"(est. ${pb['cost_per_customer']:,.2f} / customer)"
        )

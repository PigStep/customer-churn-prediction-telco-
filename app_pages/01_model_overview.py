"""Page 1 — Model registry / overview (what ships to production)."""

import pandas as pd
import streamlit as st
import altair as alt

from churn.cost import cost_curve, downsample, no_model_cost

from churn.app_utils import (
    BANDS,
    RETENTION_RATE,
    ensure_artifacts,
    load_model_card,
    load_validation_df,
)

ensure_artifacts()

card = load_model_card()
hm = card["holdout_metrics"]
cost = card["cost_model"]
base_kpis = card["current_base_kpis"]

st.title("Model overview")
st.caption("The offline step of a production churn system: the tuned LightGBM is fit on "
           "historical data, evaluated on a held-out snapshot, and only then ships to "
           "serve the current customer base.")

with st.container(horizontal=True):
    st.metric("Model", card["model"], border=True)
    st.metric("Version", card["version"], border=True)
    st.metric("Holdout PR-AUC", f"{hm['pr_auc']:.3f}", border=True)
    st.metric(
        "Cost-optimal threshold",
        f"{hm['cost_optimal_threshold_45']:.3f}",
        f"at {cost['retention_rate_hbr']:.0%} retention",
        border=True,
    )
    st.metric("Current base customers", f"{base_kpis['customers']:,}", border=True)

savings = hm["no_model_cost"] - hm["min_model_cost"]
savings_pct = savings / hm["no_model_cost"] * 100

with st.container(horizontal=True):
    st.metric("No-model cost", f"${hm['no_model_cost']:,.0f}", border=True)
    st.metric("Min model cost", f"${hm['min_model_cost']:,.0f}", border=True)
    st.metric(
        "Expected savings",
        f"${savings:,.0f}",
        f"{savings_pct:.1f}% vs no-model",
        border=True,
    )

st.divider()

left, right = st.columns([3, 2], border=True)

with left:
    st.subheader("Cost vs decision threshold")
    y = load_validation_df()
    y_true = y["Churn"]
    y_score = y["churn_proba"]
    no_model = no_model_cost(y_true)
    curve = pd.DataFrame(downsample(cost_curve(y_true, y_score, RETENTION_RATE), 250))
    curve["no_model_cost"] = no_model
    curve["threshold"] = curve["threshold"].round(4)

    cost_chart = alt.Chart(curve).mark_line().encode(
        x=alt.X("threshold:Q", title="Threshold"),
        y=alt.Y("total_cost:Q", title="Total cost ($)"),
    )
    baseline = alt.Chart(curve).mark_rule(color="#EF4444", strokeDash=[4, 4]).encode(
        y=alt.Y("no_model_cost:Q")
    )
    st.altair_chart(cost_chart + baseline, width="stretch")
    st.caption(
        f"Red dashed line = 'retain nobody' baseline (${no_model:,.0f}). The curve is "
        f"evaluated at the 45% retention success rate. At the optimal threshold the "
        f"model saves ~{savings_pct:.0f}% vs doing nothing."
    )

with right:
    st.subheader("Cost model")
    st.markdown(
        f"- **False negative** (missed churner): **${cost['FN_cost']:,.2f}** — lost customer value\n"
        f"- **False positive** (false alarm): **${cost['FP_cost']:,.2f}** — wasted retention offer\n"
        f"- **Retention success rate:** {cost['retention_rate_hbr']:.0%} (Harvard Business Review)"
    )
    st.info(
        f"Threshold is a **cost decision, not a model decision**: with FN ${cost['FN_cost']:,.0f} "
        f"vs FP ${cost['FP_cost']:,.0f}, the cost optimum is a low cutoff "
        f"({hm['cost_optimal_threshold_45']:.3f}) — below the {int(BANDS[0][2]*100)}% "
        f"confidence-band boundary used on the batch-scoring page. The fixed bands are a "
        f"business grouping choice; the cutoff that maximizes savings is aggressive."
    )

st.divider()

col_a, col_b = st.columns(2, border=True)

with col_a:
    st.subheader("Model registry")
    reg = pd.DataFrame(
        [
            ("Split (train / current base)", f"{card['split']['train_rows']:,} / "
             f"{card['split']['current_base_rows']:,} (stratified {card['split']['test_size']:.0%})"),
            ("Holdout churn rate", f"{hm['churn_rate']:.1%}"),
            ("Tuned by", "Optuna, PR-AUC objective (Model.ipynb cell 11)"),
            ("Trained at", card["trained_at"]),
        ],
        columns=["Property", "Value"],
    )
    st.dataframe(reg, hide_index=True, width="stretch")
    st.subheader("LightGBM parameters")
    params = pd.DataFrame(
        {"Parameter": list(card["lgb_params"]),
         "Value": [f"{v:g}" for v in card["lgb_params"].values()]}
    )
    st.dataframe(params, hide_index=True, width="stretch")

with col_b:
    st.subheader("Tier quality on the holdout")
    rows = []
    for name in card["band_metrics"]:
        m = card["band_metrics"][name]
        rows.append({
            "Tier": name,
            "Customers": f"{m['count']:,}",
            "Churn rate in tier": f"{m['precision']:.0%}" if m["precision"] is not None else "-",
            "% of churners captured": f"{m['recall']:.0%}" if m["recall"] is not None else "-",
            "Monthly revenue at risk": f"${m['monthly_rev_at_risk']:,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("The high-confidence tier concentrates churn (7 in 10 churn) but catches a "
               "minority; the medium tier captures the largest share of churners.")

st.divider()

st.subheader("Production wiring")
with st.container(border=True):
    steps = pd.DataFrame(
        {
            "Stage": ["Train", "Evaluate", "Serve", "Score", "Act"],
            "What happens": [
                "Fit LightGBM on the 70% training split (scripts/train_model.py)",
                "PR-AUC + cost curve on the 30% held-out snapshot → model card",
                "Serialize the pipeline (preprocessor + model) to models/churn_lgb.joblib",
                "Nightly batch pass over the current customer base → churn probability",
                "Tier by confidence, queue interventions, track revenue at risk",
            ],
        }
    )
    st.dataframe(steps, hide_index=True, width="stretch")
    st.caption("This page is the 'evaluate + serve' stage. See the batch-scoring page for "
               "the nightly scoring run and action queue.")

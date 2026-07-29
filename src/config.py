import streamlit as st
import pandas as pd

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

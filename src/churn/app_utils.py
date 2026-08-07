"""Shared loaders and business logic for the churn demo app.

The app never re-fits: it loads the artifacts produced by
scripts/train_model.py (train/serve separation) and runs batch scoring over
the simulated current customer base.

Tier bands (fixed confidence cutoffs) and the retention-rate assumption are
constants here so every page reads the same values.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from churn.cost import FP_COST
from churn.data import load_data

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "churn_lgb.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"
INDICES_PATH = MODELS_DIR / "current_base_indices.npz"

BANDS = [("Low", 0.0, 0.30), ("Medium", 0.30, 0.70), ("High", 0.70, 1.0)]
TIER_ORDER = ["Low", "Medium", "High"]

RETENTION_RATE = 0.45

# Length of the priced retention offer (6-month / 20% discount), used to
# amortize the one-time offer cost (FP_COST per customer) into a monthly
# program-cost figure for the expected-saves KPI.
OFFER_MONTHS = 6

# Intervention playbook per tier
TIER_PLAYBOOK = {
    "Low": {
        "intervention": "No action (monitor)",
        "cost_per_customer": 0.00,
    },
    "Medium": {
        "intervention": "6-month 20% discount offer",
        "cost_per_customer": FP_COST,
    },
    "High": {
        "intervention": "6-month 20% discount offer",
        "cost_per_customer": FP_COST,
    },
}


def ensure_artifacts():
    """Stop with a helpful message if the model artifacts are missing."""
    missing = [p for p in (PIPELINE_PATH, CARD_PATH, INDICES_PATH) if not p.exists()]
    if missing:
        st.error("Model artifacts not found. Train them once:")
        st.code("uv run python scripts/train_model.py")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Fitted LGBM pipeline (ColumnTransformer + model), train/serve separated."""
    return joblib.load(PIPELINE_PATH)


@st.cache_data(show_spinner=False)
def load_model_card() -> dict:
    return json.loads(CARD_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_cleaned_df() -> pd.DataFrame:
    """Full cleaned dataset with the original row labels as the index."""
    return load_data()


def risk_tier(proba: float) -> str:
    if proba < 0.30:
        return "Low"
    if proba < 0.70:
        return "Medium"
    return "High"


@st.cache_data(show_spinner=False)
def load_current_base() -> pd.DataFrame:
    """Simulated 'current customers' snapshot: scored, with hidden labels.

    Columns: customer_id + all model features + churn_proba + risk_tier.
    `Churn` (the ground truth) is intentionally not included here - the
    scoring pipeline never sees labels, mirroring production.
    """
    df = load_cleaned_df()
    indices = np.load(INDICES_PATH)["current_base_indices"]
    base = df.loc[indices].drop(columns="Churn").copy()
    base.insert(0, "customer_id", "TELCO-" + base.index.astype(str).str.zfill(4))
    features = [c for c in base.columns if c not in ("customer_id", "Churn")]
    proba = load_pipeline().predict_proba(base[features])[:, 1]
    base["churn_proba"] = proba
    base["risk_tier"] = base["churn_proba"].apply(risk_tier)
    return base


@st.cache_data(show_spinner=False)
def load_validation_df() -> pd.DataFrame:
    """Current base joined with the hidden labels (offline validation view).

    Same customers as load_current_base(), plus the ground-truth `Churn`
    column — the eval a production team runs before shipping a threshold.
    """
    base = load_current_base()
    df = load_cleaned_df()
    base = base.merge(df[["Churn"]], left_index=True, right_index=True)
    return base


def tier_label(proba: float) -> str:
    """'Low (<30%)'-style label used in tables and captions."""
    tier = risk_tier(proba)
    if tier == "Low":
        return "Low (<30%)"
    if tier == "Medium":
        return "Medium (30-70%)"
    return "High (>70%)"


def playbook_for(tier: str) -> dict:
    return TIER_PLAYBOOK[tier]


def expected_monthly_saves(at_risk_rev: float, n_at_risk: int) -> float:
    """Expected net monthly revenue recovered by the retention program.

    Gross recovered revenue is the at-risk revenue times the retention success
    rate; the retention program cost is the one-time offer (FP_COST per
    intervened customer) amortized over the offer's length (OFFER_MONTHS).
    Mirrors the notebook cost model (Model.ipynb cell 18), which nets the offer
    cost out of the savings.
    """
    gross = at_risk_rev * RETENTION_RATE
    program_cost = n_at_risk * FP_COST / OFFER_MONTHS
    return gross - program_cost

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

from churn.data import load_data

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "churn_lgb.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"
INDICES_PATH = MODELS_DIR / "current_base_indices.npz"

# Fixed confidence bands (CleverTap / Kumo.ai convention).
BANDS = [("Low", 0.0, 0.30), ("Medium", 0.30, 0.70), ("High", 0.70, 1.0)]
TIER_ORDER = ["Low", "Medium", "High"]

# Retention success probability (Harvard Business Review), used for expected
# revenue-saved estimates.
RETENTION_RATE = 0.45

# Cost-matched intervention playbook per tier (Kumo.ai-style, illustrative).
TIER_PLAYBOOK = {
    "Low": {
        "intervention": "Automated in-app / email nudge",
        "cost_per_customer": 2.00,
    },
    "Medium": {
        "intervention": "Targeted email + account check-in",
        "cost_per_customer": 25.00,
    },
    "High": {
        "intervention": "CSM call + retention offer",
        "cost_per_customer": 150.00,
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


@st.cache_data(show_spinner=False)
def load_current_base() -> pd.DataFrame:
    """Simulated 'current customers' snapshot: scored, with hidden labels.

    Columns: customer_id + all model features + churn_proba + risk_tier.
    `Churn` (the ground truth) is intentionally not included here — the
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


def risk_tier(proba: float) -> str:
    if proba < 0.30:
        return "Low"
    if proba < 0.70:
        return "Medium"
    return "High"


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


def expected_monthly_saves(at_risk_rev: float) -> float:
    """Expected recovered revenue given the retention success rate."""
    return at_risk_rev * RETENTION_RATE

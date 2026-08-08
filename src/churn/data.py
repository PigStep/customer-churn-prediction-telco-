"""Data loading and preprocessing (mirrors notebooks/Baseline.ipynb cells 6-9)."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "Churn",
]

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "dataset.csv"


def load_data(path=None)->pd.DataFrame:
    """Load and clean the raw CSV.

    Mirrors notebooks/Baseline.ipynb: categoricals -> category dtype,
    SeniorCitizen/Churn -> bool, TotalCharges -> numeric, NaN rows dropped,
    customerID dropped.
    """
    df = pd.read_csv(path or DATA_PATH)
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    df["SeniorCitizen"] = df["SeniorCitizen"] == 1
    df["Churn"] = df["Churn"] == "Yes"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns="customerID")
    return df


def make_features(df)->tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns="Churn")
    y = df["Churn"]
    return X, y


def build_preprocessor(X):
    """ColumnTransformer: one-hot categoricals, passthrough numeric columns.

    Mirrors notebooks/Baseline.ipynb cells 30/36:
    """
    cat_cols = X.select_dtypes(include=["category", "bool"]).columns.tolist()
    return ColumnTransformer(
        [
            (
                "ohe",
                OneHotEncoder(sparse_output=False, handle_unknown="ignore"),
                cat_cols,
            )
        ],
        remainder="passthrough",
    ).set_output(transform="pandas")

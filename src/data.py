import streamlit as st
import pandas as pd


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    df["SeniorCitizen"] = df["SeniorCitizen"] == 1
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category")
    return df

"""Regenerate COST_DATA for telco_churn_financial.html.

Reproduces notebooks/Baseline.ipynb (cell 43) on the monthly test-set basis:

  * 33% hold-out split (train_test_split, random_state=42, stratified)
  * RF baseline: RandomForestClassifier(random_state=42)
  * LightGBM: LGBMClassifier(scale_pos_weight=minor_ratio, verbose=-1, random_state=42)
  * precision/recall at the real score thresholds
  * cost model:
        cost(p) = FN*FN_cost + FP_cost*FP + (FP_cost + (1-p)*FN_cost)*TP
      with FN_cost=997.94, FP_cost=89.33,
      retention rates [0.15, 0.2, 0.4, 0.45, 0.6, 0.8, 1.0]

Writes scripts/generated_cost_data.json and patches the COST_DATA block
embedded in telco_churn_financial.html. Fails loudly if the results drift
from the numbers verified against the notebook.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "telco_churn_financial.html"
JSON_PATH = ROOT / "scripts" / "generated_cost_data.json"

FN_COST = 997.94
FP_COST = 89.33
RETENTION_RATES = [0.15, 0.2, 0.4, 0.45, 0.6, 0.8, 1.0]
MAX_PLOT_POINTS = 250

CAT_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "Churn",
]


def load_data():
    df = pd.read_csv(ROOT / "data" / "dataset.csv")
    for col in CAT_COLS:
        df[col] = df[col].astype("category")
    df["SeniorCitizen"] = df["SeniorCitizen"] == 1
    df["Churn"] = df["Churn"] == "Yes"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns="customerID")
    return pd.get_dummies(df)


def rate_key(p):
    """Match JS String(p) so the browser's cost_curves[String(r)] lookup works."""
    return "{:g}".format(p)


def cost_curves(y_true, y_score, rates):
    """Return {thresholds: [...], curves: {rate_key: [...]}} sorted ascending."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.concatenate(([0.0], thresholds))  # align full-predict point
    P = int(y_true.sum())
    TP = recall * P
    FN = P - TP
    FP = TP * (1.0 / precision - 1.0)

    curves = {}
    for p in rates:
        cost_tp = FP_COST + (1 - p) * FN_COST
        curves[rate_key(p)] = FN * FN_COST + FP_COST * FP + cost_tp * TP

    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    curves = {k: np.asarray(v)[order] for k, v in curves.items()}
    return thresholds, curves


def downsample(thresholds, curves, best_idx):
    """Reduce the number of points and keep with best threshold"""
    n = len(thresholds)
    idx = np.unique(np.linspace(0, n - 1, min(n, MAX_PLOT_POINTS)).astype(int))
    idx = np.unique(np.concatenate((idx, [best_idx]))).astype(int)
    return thresholds[idx], {k: v[idx] for k, v in curves.items()}


def model_block(name, model):
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    P = int(y_test.sum())
    thresholds, curves = cost_curves(y_test, y_score, RETENTION_RATES)

    c45 = curves[rate_key(0.45)]
    best_idx = int(np.argmin(c45))
    best_threshold = float(thresholds[best_idx])
    best_cost = round(float(c45[best_idx]), 2)

    plot_th, plot_curves = downsample(thresholds, curves, best_idx)
    print(f"[{name}] {name} done: P={P}, best_threshold={best_threshold:.4f}, best_cost={best_cost:,.2f}")
    return {
        "thresholds": [round(float(t), 6) for t in plot_th],
        "no_model_cost": round(P * FN_COST, 2),
        "cost_curves": {k: [round(float(v), 2) for v in vals] for k, vals in plot_curves.items()},
        "best_threshold": best_threshold,
        "best_cost": best_cost,
    }


def verify(name, block, min_expectations, frac_expectations):
    rates = list(RETENTION_RATES)
    frac_below = {r: np.mean(np.asarray(block["cost_curves"][rate_key(r)]) < block["no_model_cost"]) for r in rates}
    mins = {r: min(block["cost_curves"][rate_key(r)]) for r in rates}
    print(f"[verify] {name}: no_model={block['no_model_cost']:,.2f} min15={mins[0.15]:,.2f} "
          f"frac_below15={frac_below[0.15]:.2f} min45={mins[0.45]:,.2f} min100={mins[1.0]:,.2f}")
    for key, (lo, hi) in min_expectations.items():
        if not (lo <= mins[key] <= hi):
            raise SystemExit(f"[verify] FAIL {name} mins[{key}]={mins[key]:.2f} outside [{lo}, {hi}]")
    for key, (lo, hi) in frac_expectations.items():
        if not (lo <= frac_below[key] <= hi):
            raise SystemExit(f"[verify] FAIL {name} frac_below[{key}]={frac_below[key]:.2f} outside [{lo}, {hi}]")
    print(f"[verify] OK {name}")


def main():
    df = load_data()
    X = df.drop(columns="Churn")
    y = df["Churn"]
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y)

    minor_ratio = (y == False).sum() / (y == True).sum()
    rf = RandomForestClassifier(random_state=42)
    lgbm = lgb.LGBMClassifier(scale_pos_weight=minor_ratio, verbose=-1, random_state=42)

    rf_block = model_block("rf", rf)
    lgb_block = model_block("lgb", lgbm)

    verify(
        "rf", rf_block,
        {0.15: (610000.0, 613000.0), 0.45: (484000.0, 488000.0), 1.0: (170000.0, 182000.0)},
        {0.15: (0.40, 0.55), 0.45: (0.98, 1.0)},
    )
    verify(
        "lgb", lgb_block,
        {0.15: (605000.0, 613000.0), 0.45: (477000.0, 483000.0), 1.0: (160000.0, 175000.0)},
        {0.15: (0.15, 0.35), 0.45: (0.98, 1.0)},
    )

    cost_data = {
        "rf": rf_block,
        "lgb": lgb_block,
        "retention_rates": RETENTION_RATES,
        "FN_cost": FN_COST,
        "FP_cost": FP_COST,
        "revenue": {
            "total_monthly_revenue": 455661.0,
            "monthly_rev_at_risk": 139130.85,
            "annual_rev_at_risk": 1669570.2,
            "avg_revenue_per_account": 64.8,
            "churned_percentage": 26.6,
        },
    }

    JSON_PATH.write_text(json.dumps(cost_data, indent=2))
    print(f"[write] {JSON_PATH}")

    html = HTML_PATH.read_text()
    new_block = "const COST_DATA = " + json.dumps(cost_data, indent=2) + ";\n"
    pattern = re.compile(r"const COST_DATA = \{.*?\n\};", re.DOTALL)
    if not pattern.search(html):
        raise SystemExit("[fail] COST_DATA block not found in HTML")
    html = pattern.sub(lambda m: new_block.rstrip("\n"), html, count=1)
    HTML_PATH.write_text(html)
    print(f"[patch] COST_DATA embedded in {HTML_PATH}")


if __name__ == "__main__":
    sys.exit(main())

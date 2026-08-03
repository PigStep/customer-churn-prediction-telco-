"""Regenerate COST_DATA for telco_churn_financial.html.

Full-dataset basis matching notebooks/Model.ipynb headline numbers:

  * RF baseline: RandomForestClassifier(random_state=42), 5-fold stratified
    out-of-fold predictions (cross_val_predict, StratifiedKFold random_state=42)
  * LightGBM: Optuna-tuned via nested_cv_with_optuna (5 outer folds, 50 trials,
    average_precision_score) -> full-dataset OOF probabilities
  * cost model (Baseline.ipynb cell 43):
        cost(p) = FN*997.94 + 89.33*FP + (89.33 + (1-p)*997.94)*TP
    with retention rates [0.15, 0.2, 0.4, 0.45, 0.6, 0.8, 1.0]
  * no-model cost = 1869 churners * 997.94 = $1,865,150

Writes scripts/generated_cost_data.json and patches the COST_DATA block
embedded in telco_churn_financial.html. Fails loudly if the results drift
from the notebook-verified pins.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from churn.cost import no_model_cost, rate_key
from churn.data import load_data, make_features
from churn.models import make_pipeline
from churn.report import build_cost_data, build_block, build_revenue, patch_html, write_json
from churn.tuning import nested_cv_with_optuna

ROOT = Path(__file__).resolve().parents[1]
OOF_PROBA_PATH = ROOT / "scripts" / "oof_proba.npz"

# Regression pins: each row is one check = (metric, retention_rate, lower, upper).
#   ("min", rate)        -> lowest total cost across all thresholds at that rate
#   ("frac_below", rate) -> fraction of thresholds whose cost beats the no-model baseline
# Windows mirror the headline numbers published in notebooks/Baseline.ipynb
# (cell 43 cost curves) and notebooks/Model.ipynb. They are hardcoded on purpose:
# they are the de-facto test suite, so they must stay fixed while the run
# recomputes everything from scratch.
LGB_PINS = {
    "best_threshold": (0.10, 0.25),
    "savings": (405_000.0, 435_000.0),
}
RF_PINS = [
    ("min", 0.15, 1_840_000.0, 1_860_000.0),
    ("min", 0.45, 1_440_000.0, 1_490_000.0),
    ("min", 1.0, 500_000.0, 560_000.0),
    ("frac_below", 0.15, 0.45, 0.65),
    ("frac_below", 0.45, 0.98, 1.0),
]

# --- Data loading and model tuning ---
def load_data_and_features():
    df = load_data()
    X, y = make_features(df)
    print(f"[data] {len(df)} rows, {int(y.sum())} churners")
    return df, X, y


def compute_oof_proba(X, y, n_trials=50, n_outer=5):
    """RF (5-fold CV) and LightGBM (Optuna nested-CV) OOF probabilities.

    Caches the arrays to scripts/oof_proba.npz alongside.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_oof = cross_val_predict(make_pipeline(X, "rf"), X, y, cv=skf, method="predict_proba")[:, 1]
    print("[rf] RF OOF probabilities done")

    lgb_pipe = make_pipeline(X, "lgb")
    lgb_oof = nested_cv_with_optuna(
        X, y, lgb_pipe, average_precision_score, n_trials=n_trials, n_outer=n_outer
    )
    np.savez(OOF_PROBA_PATH, rf=rf_oof, lgb=lgb_oof)
    print("[lgb] Optuna nested-CV OOF probabilities done")

    return rf_oof, lgb_oof

# --- Get model metrics --- 
def print_lgb_metrics(y, lgb_oof, lgb_block):
    precision, recall, thresholds = precision_recall_curve(y, lgb_oof)
    best_idx = int(np.argmin(np.abs(thresholds - lgb_block["best_threshold"])))
    print(
        f"[metrics] lgb: pr_auc={average_precision_score(y, lgb_oof):.4f} "
        f"best threshold recall={recall[best_idx]:.2f} precision={precision[best_idx]:.2f}"
    )


def build_blocks(y, rf_oof, lgb_oof):
    """Build the COST_DATA model blocks consumed by the HTML renderer."""
    rf_block = build_block(y, rf_oof)
    lgb_block = build_block(y, lgb_oof)
    print_lgb_metrics(y, lgb_oof, lgb_block)
    return rf_block, lgb_block

# --- Verify computed costs ---
def verify_no_model_cost(rf_block, lgb_block, y):
    expected_no_model = no_model_cost(y)
    for name, block in (("rf", rf_block), ("lgb", lgb_block)):
        if abs(block["no_model_cost"] - expected_no_model) > 0.01:
            raise SystemExit(
                f"[verify] FAIL {name} no_model_cost={block['no_model_cost']:.2f} != {expected_no_model:.2f}"
            )

def curve_stat(block, metric, rate):
    """One stat from a cost-curve block: 'min' or 'frac_below'."""
    costs = np.asarray(block["cost_curves"][rate_key(rate)])
    if metric == "min":
        return min(costs)
    if metric == "frac_below":
        return np.mean(costs < block["no_model_cost"])
    raise ValueError(f"unknown metric {metric!r}")


def check_range(name, metric, rate, value, lo, hi):
    if not (lo <= value <= hi):
        raise SystemExit(
            f"[verify] FAIL {name} {metric}[{rate}]={value:,.2f} outside [{lo}, {hi}]"
        )


def verify(name, block, expectations):
    """Check cost-curve stats against expected windows; exits nonzero on drift."""
    print(f"[verify] {name}: no_model={block['no_model_cost']:,.2f}")
    for metric, rate, lo, hi in expectations:
        value = curve_stat(block, metric, rate)
        check_range(name, metric, rate, value, lo, hi)
        print(f"[verify] {name} {metric}[{rate}]={value:,.2f} OK")
    print(f"[verify] OK {name}")

def verify_lgb(lgb_block, y, lgb_oof):
    best_th = lgb_block["best_threshold"]
    best_cost = lgb_block["best_cost"]
    savings = lgb_block["no_model_cost"] - best_cost
    print(
        f"[verify] lgb: best_threshold={best_th:.4f} best_cost={best_cost:,.2f} "
        f"savings={savings:,.2f} ({savings / lgb_block['no_model_cost'] * 100:.2f}%)"
    )
    # Threshold near the cost optimum is flat, so allow generous drift from the
    # notebook's published 0.1597 (its Optuna run was unseeded).
    lo_th, hi_th = LGB_PINS["best_threshold"]
    if not (lo_th <= best_th <= hi_th):
        raise SystemExit(f"[verify] FAIL lgb best_threshold={best_th:.4f} outside {LGB_PINS['best_threshold']}")
    lo_sav, hi_sav = LGB_PINS["savings"]
    if not (lo_sav <= savings <= hi_sav):
        raise SystemExit(f"[verify] FAIL lgb savings={savings:,.2f} outside {LGB_PINS['savings']}")
    print(f"[metrics] lgb: pr_auc={average_precision_score(y, lgb_oof):.4f}")

def verify_blocks(y, rf_block, lgb_block, lgb_oof):
    """Run every regression pin (the de-facto test suite). Exits nonzero on drift."""
    verify_no_model_cost(rf_block, lgb_block, y)
    verify("rf", rf_block, RF_PINS)
    verify_lgb(lgb_block, y, lgb_oof)


def emit_outputs(df, rf_block, lgb_block):
    """Serialize COST_DATA to JSON and embed it in the HTML."""
    cost_data = build_cost_data(rf_block, lgb_block, build_revenue(df))
    write_json(cost_data)
    patch_html(cost_data)


def main():
    df, X, y = load_data_and_features()
    rf_oof, lgb_oof = compute_oof_proba(X, y)
    rf_block, lgb_block = build_blocks(y, rf_oof, lgb_oof)
    verify_blocks(y, rf_block, lgb_block, lgb_oof)
    emit_outputs(df, rf_block, lgb_block)
    return 0


if __name__ == "__main__":
    sys.exit(main())

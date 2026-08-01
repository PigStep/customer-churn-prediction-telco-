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

from churn.cost import RETENTION_RATES, no_model_cost, rate_key
from churn.data import load_data, make_features
from churn.models import make_pipeline, rf_baseline
from churn.report import build_cost_data, build_block, patch_html, write_json
from churn.tuning import nested_cv_with_optuna

ROOT = Path(__file__).resolve().parents[1]


def verify(name, block, min_expectations, frac_expectations):
    rates = list(RETENTION_RATES)
    frac_below = {
        r: np.mean(np.asarray(block["cost_curves"][rate_key(r)]) < block["no_model_cost"])
        for r in rates
    }
    mins = {r: min(block["cost_curves"][rate_key(r)]) for r in rates}
    print(
        f"[verify] {name}: no_model={block['no_model_cost']:,.2f} "
        f"min15={mins[0.15]:,.2f} frac_below15={frac_below[0.15]:.2f} "
        f"min45={mins[0.45]:,.2f} min100={mins[1.0]:,.2f}"
    )
    for key, (lo, hi) in min_expectations.items():
        if not (lo <= mins[key] <= hi):
            raise SystemExit(f"[verify] FAIL {name} mins[{key}]={mins[key]:.2f} outside [{lo}, {hi}]")
    for key, (lo, hi) in frac_expectations.items():
        if not (lo <= frac_below[key] <= hi):
            raise SystemExit(f"[verify] FAIL {name} frac_below[{key}]={frac_below[key]:.2f} outside [{lo}, {hi}]")
    print(f"[verify] OK {name}")


def main():
    df = load_data()
    X, y = make_features(df)
    print(f"[data] {len(df)} rows, {int(y.sum())} churners")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf_pipe = rf_baseline(X)
    rf_oof = cross_val_predict(rf_pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    print("[rf] RF OOF probabilities done")

    lgb_pipe = make_pipeline(X, "lgb")
    _, _, _, lgb_oof = nested_cv_with_optuna(
        X, y, lgb_pipe, average_precision_score, n_trials=50, n_outer=5
    )
    print("[lgb] Optuna nested-CV OOF probabilities done")

    rf_block = build_block(y, rf_oof)
    lgb_block = build_block(y, lgb_oof)

    np.savez(ROOT / "scripts" / "oof_proba.npz", rf=rf_oof, lgb=lgb_oof)

    precision, recall, thresholds = precision_recall_curve(y, lgb_oof)
    best_idx = int(np.argmin(np.abs(thresholds - lgb_block["best_threshold"])))
    print(
        f"[metrics] lgb: pr_auc={average_precision_score(y, lgb_oof):.4f} "
        f"recall={recall[best_idx]:.2f} precision={precision[best_idx]:.2f}"
    )

    expected_no_model = no_model_cost(y)
    for name, block in (("rf", rf_block), ("lgb", lgb_block)):
        if abs(block["no_model_cost"] - expected_no_model) > 0.01:
            raise SystemExit(
                f"[verify] FAIL {name} no_model_cost={block['no_model_cost']:.2f} != {expected_no_model:.2f}"
            )

    #FIXME: Why hardcode there?
    verify(
        "rf", rf_block,
        {0.15: (1_840_000.0, 1_860_000.0), 0.45: (1_440_000.0, 1_490_000.0), 1.0: (500_000.0, 560_000.0)},
        {0.15: (0.45, 0.65), 0.45: (0.98, 1.0)},
    )
    best_th = lgb_block["best_threshold"]
    best_cost = lgb_block["best_cost"]
    savings = lgb_block["no_model_cost"] - best_cost
    print(
        f"[verify] lgb: best_threshold={best_th:.4f} best_cost={best_cost:,.2f} "
        f"savings={savings:,.2f} ({savings / lgb_block['no_model_cost'] * 100:.2f}%)"
    )
    # Threshold near the cost optimum is flat, so allow generous drift from the
    # notebook's published 0.1597 (its Optuna run was unseeded).
    if not (0.10 <= best_th <= 0.25):
        raise SystemExit(f"[verify] FAIL lgb best_threshold={best_th:.4f} outside [0.10, 0.25]")
    if not (405_000.0 <= savings <= 435_000.0):
        raise SystemExit(f"[verify] FAIL lgb savings={savings:,.2f} outside [405000, 435000]")
    print(f"[metrics] lgb: pr_auc={average_precision_score(y, lgb_oof):.4f}")

    cost_data = build_cost_data(rf_block, lgb_block)
    write_json(cost_data)
    patch_html(cost_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

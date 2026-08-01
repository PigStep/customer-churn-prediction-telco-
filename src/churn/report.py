"""Build the COST_DATA structure, serialize it, and patch it into the HTML."""

import json
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from churn.cost import FN_COST, FP_COST, RETENTION_RATES, costs_from_curve, rate_key

ROOT = Path(__file__).resolve().parents[2]
HTML_PATH = ROOT / "telco_churn_financial.html"
JSON_PATH = ROOT / "scripts" / "generated_cost_data.json"

MAX_PLOT_POINTS = 250

#FIXME: dynamical values instead of constants
REVENUE = {
    "total_monthly_revenue": 455661.0,
    "monthly_rev_at_risk": 139130.85,
    "annual_rev_at_risk": 1669570.2,
    "avg_revenue_per_account": 64.8,
    "churned_percentage": 26.6,
}


def build_block(y_true, y_score, rates=None, max_points=MAX_PLOT_POINTS):
    """One model block for COST_DATA (shape consumed by the HTML renderer).

    Mirrors the format produced by scripts/generate_chart_data.py: `thresholds`
    sorted ascending, each rate's costs aligned to the same array, plus the
    best 45%-retention threshold and its cost.
    """
    
    #FIXME: same logic in churn.cost . Reuse
    rates = rates or RETENTION_RATES
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    thresholds = np.concatenate(([0.0], thresholds))  # align full-predict point
    P = int(y_true.sum())
    TP = recall * P
    FN = P - TP
    FP = TP * (1.0 / precision - 1.0)

    curves = {}
    for p in rates:
        curves[rate_key(p)] = costs_from_curve(precision, recall, P, p)

    order = np.argsort(thresholds)
    thresholds = thresholds[order]
    curves = {k: np.asarray(v)[order] for k, v in curves.items()}

    c45 = curves[rate_key(0.45)]
    best_idx = int(np.argmin(c45))
    best_threshold = float(thresholds[best_idx])
    best_cost = round(float(c45[best_idx]), 2)

    plot_th, plot_curves = _downsample(thresholds, curves, best_idx, max_points)
    return {
        "thresholds": [round(float(t), 6) for t in plot_th],
        "no_model_cost": round(P * FN_COST, 2),
        "cost_curves": {
            k: [round(float(v), 2) for v in vals] for k, vals in plot_curves.items()
        },
        "best_threshold": best_threshold,
        "best_cost": best_cost,
    }


def _downsample(thresholds, curves, best_idx, max_points):
    """Reduce the number of points and keep the best-threshold index."""
    n = len(thresholds)
    idx = np.unique(np.linspace(0, n - 1, min(n, max_points)).astype(int))
    idx = np.unique(np.concatenate((idx, [best_idx]))).astype(int)
    return thresholds[idx], {k: v[idx] for k, v in curves.items()}


def build_cost_data(rf_block, lgb_block, rates=None):
    return {
        "rf": rf_block,
        "lgb": lgb_block,
        "retention_rates": rates or RETENTION_RATES,
        "FN_cost": FN_COST,
        "FP_cost": FP_COST,
        "revenue": REVENUE,
    }


def write_json(cost_data):
    JSON_PATH.write_text(json.dumps(cost_data, indent=2))
    print(f"[write] {JSON_PATH}")
    return JSON_PATH


def patch_html(cost_data):
    html = HTML_PATH.read_text()
    new_block = "const COST_DATA = " + json.dumps(cost_data, indent=2) + ";"
    pattern = re.compile(r"const COST_DATA = \{.*?\n\};", re.DOTALL)
    if not pattern.search(html):
        raise SystemExit("[fail] COST_DATA block not found in HTML")
    html = pattern.sub(lambda m: new_block, html, count=1)
    HTML_PATH.write_text(html)
    print(f"[patch] COST_DATA embedded in {HTML_PATH}")
    return HTML_PATH

"""Cost model for retention decisions (mirrors notebooks/Baseline.ipynb cell 43).

Cost per decision at retention rate p and threshold t:
    TP = recall(P_t) * P
    FN = P - TP
    FP = TP * (1/precision - 1) (as precision = TP / (TP + FP))
    Cost(p) = FN*997.94 + 89.33*FP + (89.33 + (1-p)*997.94)*TP
"""

import numpy as np
from sklearn.metrics import precision_recall_curve

FN_COST = 997.94
FP_COST = 89.33

RETENTION_RATES = [0.15, 0.2, 0.4, 0.45, 0.6, 0.8, 1.0]

RETENTION_RATE_HBR = 0.45

TOTAL_FN_COST = 997.94 + 89.33  # FN + FP per retained churner at p=0
TP_COST_P0 = FP_COST  # TP cost at p=0


def rate_key(p)->str:
    """JS-safe curve key: String(0.15) -> "0.15", String(1.0) -> "1"."""
    return "{:g}".format(p)


def costs_from_curve(precision, recall, P, rate)->list:
    """Total cost at every threshold given PR-curve components.

    Same formula as notebooks/Baseline.ipynb cell 43:
        cost(p) = FN*997.94 + 89.33*FP + (89.33 + (1-p)*997.94)*TP
    """
    TP = recall * P
    FN = P - TP
    FP = TP * (1 / precision - 1)
    return FN * FN_COST + FP_COST * FP + (FP_COST + (1 - rate) * FN_COST) * TP


def cost_curve(y_true, y_proba, rate)->list[dict]:
    """Cost over every PR-curve threshold for one retention rate.

    Returns a list of {"threshold": float, "total_cost": float} sorted by
    descending threshold (cheapest-before-most-expensive ordering).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    P = y_true.sum()
    total_costs = costs_from_curve(precision[:-1], recall[:-1], P, rate)
    order = np.argsort(thresholds, descending=True)
    return [
        {"threshold": float(thresholds[i]), "total_cost": float(total_costs[i])}
        for i in order
    ]


def best_threshold_cost(y_true, y_proba, rate)->tuple[float,float]:
    """(best_threshold, min_total_cost) over the PR-curve thresholds."""
    curve = cost_curve(y_true, y_proba, rate)
    best = min(curve, key=lambda pt: pt["total_cost"])
    return best["threshold"], best["total_cost"]


def no_model_cost(y_true):
    """Cost of retaining nobody: every churner is a missed retention."""
    return float(y_true.sum() * FN_COST)


def savings(y_true, y_proba, rate):
    """(threshold, model_cost, savings_vs_no_model) at the best threshold."""
    th, cost = best_threshold_cost(y_true, y_proba, rate)
    return th, cost, no_model_cost(y_true) - cost


def cost_curves(y_true, y_proba, rates=None):
    """cost_curve for every rate, keyed by rate_key()."""
    rates = rates or RETENTION_RATES
    return {rate_key(p): cost_curve(y_true, y_proba, p) for p in rates}


def downsample(curve, max_points):
    """Keep max_points points evenly spaced, always including both ends."""
    if len(curve) <= max_points:
        return curve
    idx = np.round(np.linspace(0, len(curve) - 1, max_points)).astype(int)
    return [curve[i] for i in idx]

"""Train the production LightGBM artifact and persist the 'current customer base'.

Production-system reproduction of notebooks/Model.ipynb:

  * Stratified 70/30 split (random_state=42). The 30% holdout stands in for
    the *current* customer base — a snapshot whose churn labels are unknown
    at decision time (production batch scoring). Labels are kept only for
    the offline validation tab in the demo app.
  * LightGBM fit on the 70% training split using the notebook's PR-AUC-tuned
    hyperparameters (Model.ipynb cell 11) through the shared pipeline in
    churn.models.make_pipeline — the estimator bundles the ColumnTransformer
    so the serialized artifact is self-contained for inference.
  * Writes to models/:
      churn_lgb.joblib            fitted pipeline (preprocessor + LGBM)
      model_card.json             model card + holdout metrics + cost model
      current_base_indices.npz    row indices of the current customer base

The app never re-fits; it loads these artifacts (train/serve separation).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from churn.cost import FN_COST, FP_COST, RETENTION_RATE_HBR, best_threshold_cost
from churn.data import load_data, make_features
from churn.models import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PIPELINE_PATH = MODELS_DIR / "churn_lgb.joblib"
CARD_PATH = MODELS_DIR / "model_card.json"
INDICES_PATH = MODELS_DIR / "current_base_indices.npz"

TEST_SIZE = 0.30
RANDOM_STATE = 42

# Model.ipynb cell 11 — best trial of the PR-AUC Optuna study (seed 42).
LGB_PARAMS = {
    "lambda_l1": 2.7580658438306074,
    "lambda_l2": 0.0011195303464912826,
    "num_leaves": 5,
    "feature_fraction": 0.7556401556833737,
    "bagging_fraction": 0.6111406490538761,
    "bagging_freq": 3,
    "min_child_samples": 20,
}

# Fixed confidence bands used for risk tiering
BANDS = [("Low", 0.0, 0.30), ("Medium", 0.30, 0.70), ("High", 0.70, 1.0)]


def split_current_base(df):
    """Stratified 70/30 split; the 30% is the simulated current customer base."""
    X, y = make_features(df)
    X_train, X_cur, y_train, y_cur = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_cur, y_train, y_cur


def score_holdout(pipeline, X_cur):
    return pipeline.predict_proba(X_cur)[:, 1]


def band_metrics(y_true, y_score, X_cur, bands=BANDS):
    """Per-tier operational metrics (the offline eval before thresholds ship).

    For each band returns count, actual churn rate (precision), fraction of
    all churners captured (recall), and monthly revenue at risk.
    """
    monthly = X_cur["MonthlyCharges"]
    total_churn = int(y_true.sum())
    metrics = {}
    for name, lo, hi in bands:
        mask = (y_score >= lo) & (y_score < hi)
        captured = int(y_true[mask].sum())
        metrics[name] = {
            "count": int(mask.sum()),
            "precision": round(float(y_true[mask].mean()), 4) if mask.any() else None,
            "recall": round(captured / total_churn, 4) if total_churn else None,
            "captured_churners": captured,
            "monthly_rev_at_risk": round(float(monthly[mask].sum()), 2),
        }
    return metrics


def build_model_card(pipeline, X_cur, y_cur, y_score)-> dict:
    """Everything the app needs at load time; no training data required."""
    pr_auc = average_precision_score(y_cur, y_score)
    th_cost, min_cost = best_threshold_cost(y_cur, y_score, RETENTION_RATE_HBR)
    no_model_cost = float(y_cur.sum() * FN_COST)
    monthly_rev = float(X_cur["MonthlyCharges"].sum())
    return {
        "model": "LightGBM (GBDT)",
        "version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {
            "train_rows": int(len(X_cur) * (1 - TEST_SIZE) / TEST_SIZE),  # corrected below
            "current_base_rows": int(len(X_cur)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": "Churn",
        },
        "features": X_cur.columns.tolist(),
        "lgb_params": {k: pipeline.named_steps["model"].get_params()[k] for k in LGB_PARAMS},
        "bands": [{"name": n, "low": lo, "high": hi} for n, lo, hi in BANDS],
        "holdout_metrics": {
            "pr_auc": round(float(pr_auc), 4),
            "churn_rate": round(float(y_cur.mean()), 4),
            "cost_optimal_threshold_45": round(float(th_cost), 4),
            "min_model_cost": round(min_cost, 2),
            "no_model_cost": round(no_model_cost, 2),
        },
        "cost_model": {
            "FN_cost": FN_COST,
            "FP_cost": FP_COST,
            "retention_rate_hbr": RETENTION_RATE_HBR,
        },
        "current_base_kpis": {
            "customers": int(len(X_cur)),
            "monthly_revenue": round(monthly_rev, 2),
            "avg_monthly_charges": round(float(X_cur["MonthlyCharges"].mean()), 2),
        },
        "band_metrics": band_metrics(y_cur, y_score, X_cur),
    }


def main():
    df = load_data()
    print(f"[data] {len(df)} rows, {int(df['Churn'].sum())} churners")

    X_train, X_cur, y_train, y_cur = split_current_base(df)
    print(f"[split] train={len(X_train)} rows, current base={len(X_cur)} rows "
          f"(churn rate {y_cur.mean():.1%})")

    pipeline = make_pipeline(X_train, "lgb", **LGB_PARAMS)
    pipeline.fit(X_train, y_train)
    print("[train] LightGBM fitted on the training split")

    y_score = score_holdout(pipeline, X_cur)
    pr_auc = average_precision_score(y_cur, y_score)
    print(f"[eval] holdout PR-AUC = {pr_auc:.4f}")

    card = build_model_card(pipeline, X_cur, y_cur, y_score)
    card["split"]["train_rows"] = int(len(X_train))

    cur_indices = np.asarray(X_cur.index).astype(int)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    CARD_PATH.write_text(json.dumps(card, indent=2))
    np.savez(INDICES_PATH, current_base_indices=cur_indices)

    print(f"[save] {PIPELINE_PATH}")
    print(f"[save] {CARD_PATH}")
    print(f"[save] {INDICES_PATH}")
    print(f"[ok] PR-AUC={card['holdout_metrics']['pr_auc']:.4f} "
          f"cost-optimal threshold (45% retention)="
          f"{card['holdout_metrics']['cost_optimal_threshold_45']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

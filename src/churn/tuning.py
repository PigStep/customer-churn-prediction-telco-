"""Optuna tuning + nested CV over a model pipeline (port of notebooks/Model.ipynb)."""

from functools import partial

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn.cost import FN_COST, RETENTION_RATE_HBR, costs_from_curve
from churn.models import MODEL_STEP

LGB_PARAM_SPACE = {
    "lambda_l1": ("suggest_float", dict(low=1e-8, high=10.0, log=True)),
    "lambda_l2": ("suggest_float", dict(low=1e-8, high=10.0, log=True)),
    "num_leaves": ("suggest_int", dict(low=2, high=256)),
    "feature_fraction": ("suggest_float", dict(low=0.4, high=1.0)),
    "bagging_fraction": ("suggest_float", dict(low=0.4, high=1.0)),
    "bagging_freq": ("suggest_int", dict(low=1, high=7)),
    "min_child_samples": ("suggest_int", dict(low=5, high=100)),
}


def _set_lgb_params(pipeline, params, model_step=MODEL_STEP, random_state=42):
    # Scikit-learn Pipelines access parameters of nested steps using the <step_name>__<parameter_name> convention
    set_params = {f"{model_step}__{k}": v for k, v in params.items()}
    set_params[f"{model_step}__random_state"] = random_state
    set_params[f"{model_step}__verbose"] = -1
    return clone(pipeline).set_params(**set_params)


def lgb_objective(trial, metric_func, pipeline, X_data, y_data, model_step=MODEL_STEP):
    """Inner-loop objective: split, sample LGB hyperparams, score on the validation set."""
    train_x, valid_x, train_y, valid_y = train_test_split(
        X_data, y_data, test_size=0.25, random_state=42
    )
    params = {}
    for name, (method, kwargs) in LGB_PARAM_SPACE.items():
        params[name] = getattr(trial, method)(name, **kwargs)

    pipe = _set_lgb_params(pipeline, params, model_step)
    pipe.fit(train_x, train_y)
    preds = pipe.predict_proba(valid_x)[:, 1]
    return metric_func(valid_y, preds)


#FIXME: we do not use nested cv for curve. Nested cv metrics were calculated.
#Just use best params and plot based on them.
# Leave nested cv training script for future
def nested_cv_with_optuna(X, y, pipeline, metric_func, n_trials=50,
                          n_outer=5, random_state=42, model_step=MODEL_STEP):
    """Nested CV: outer loop for unbiased OOF predictions, inner Optuna tuning.

    Reproduces notebooks/Model.ipynb `nested_cv_with_optuna` but tunes through
    the sklearn Pipeline. Returns per-fold cost/savings arrays (CI-ready) and
    the full out-of-fold probabilities.
    """
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    fold_money_saved = []
    fold_no_model_costs = []
    fold_best_costs = []
    all_oof_proba = np.zeros(len(y))

    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_skf.split(X, y)):
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, y_outer_test = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        obj = partial(
            lgb_objective,
            metric_func=metric_func,
            pipeline=pipeline,
            X_data=X_outer_train,
            y_data=y_outer_train,
            model_step=model_step,
        )
        study.optimize(obj, n_trials=n_trials)

        pipe = _set_lgb_params(pipeline, study.best_trial.params, model_step)
        pipe.fit(X_outer_train, y_outer_train)
        y_proba_fold = pipe.predict_proba(X_outer_test)[:, 1]
        all_oof_proba[outer_test_idx] = y_proba_fold

        precision, recall, thresholds = precision_recall_curve(y_outer_test, y_proba_fold)
        P_fold = y_outer_test.sum()
        total_costs = costs_from_curve(
            precision[:-1], recall[:-1], P_fold, RETENTION_RATE_HBR
        )

        best_idx = np.argmin(total_costs)
        no_model_cost = P_fold * FN_COST
        fold_no_model_costs.append(no_model_cost)
        fold_best_costs.append(total_costs[best_idx])
        fold_money_saved.append(no_model_cost - total_costs[best_idx])

    return (
        np.asarray(fold_money_saved),
        np.asarray(fold_no_model_costs),
        np.asarray(fold_best_costs),
        all_oof_proba,
    )

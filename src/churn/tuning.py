"""Optuna tuning + nested CV over a model pipeline (port of notebooks/Model.ipynb)."""

from functools import partial

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split

from churn.models import MODEL_STEP

LGB_PARAM_SPACE = {
    "lambda_l1": ("suggest_float", {"low":1e-8, "high":10.0, "log":True}),
    "lambda_l2": ("suggest_float", {"low":1e-8, "high":10.0, "log":True}),
    "num_leaves": ("suggest_int", {"low": 2, "high": 256}),
    "feature_fraction": ("suggest_float", {"low": 0.4, "high": 1.0}),
    "bagging_fraction": ("suggest_float", {"low": 0.4, "high": 1.0}),
    "bagging_freq": ("suggest_int", {"low": 1, "high": 7}),
    "min_child_samples": ("suggest_int", {"low": 5, "high": 100}),
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


def nested_cv_with_optuna(X, y, pipeline, metric_func, n_trials=50,
                          n_outer=5, random_state=42, model_step=MODEL_STEP):
    """Nested CV: outer loop for unbiased OOF predictions, inner Optuna tuning.

    Reproduces notebooks/Model.ipynb `nested_cv_with_optuna` but tunes through
    the sklearn Pipeline. Returns the full out-of-fold (OOF) probabilities, which is
    the only output the chart pipeline consumes (curves are built from them in
    churn.report.build_block).
    """
    outer_skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

    all_oof_proba = np.zeros(len(y))

    for outer_train_idx, outer_test_idx in outer_skf.split(X, y):
        X_outer_train, X_outer_test = X.iloc[outer_train_idx], X.iloc[outer_test_idx]
        y_outer_train, _ = y.iloc[outer_train_idx], y.iloc[outer_test_idx]

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

    return all_oof_proba

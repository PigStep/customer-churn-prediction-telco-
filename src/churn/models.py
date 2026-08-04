"""Model pipelines built on sklearn Pipeline + a shared ColumnTransformer.

The estimator step is always named "model" so hyperparameter tuning can target
it via set_params(model__<param>=...), regardless of the model type.
"""

import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from typing import Callable

from churn.data import build_preprocessor

MODEL_STEP = "model"


def make_pipeline(X, model="rf", preprocessor=build_preprocessor, **model_kwargs):
    """Build a Pipeline([("preprocessor", ...), ("model", <estimator>)]).

    model: "rf" | "lgb". Extra kwargs are passed to the estimator, e.g.
    make_pipeline(X, "lgb", scale_pos_weight=2.7, verbose=-1).
    """
    if model == "rf":
        kwargs = {"random_state": 42}
        kwargs.update(model_kwargs)
        estimator = RandomForestClassifier(**kwargs)
    elif model == "lgb":
        kwargs = {"seed": 42, "verbose": -1}
        kwargs.update(model_kwargs)
        estimator = lgb.LGBMClassifier(**kwargs)
    else:
        raise ValueError(f"unknown model {model!r}")

    return Pipeline(
        [("preprocessor", preprocessor(X)), (MODEL_STEP, estimator)]
    )
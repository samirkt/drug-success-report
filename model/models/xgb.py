"""XGBoost binary classifier wrapper.

Default model. Uses `binary:logistic`, scales positive class weight to
balance the imbalanced label, and supports early stopping via an
inner-validation slice of the train set.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .base import register

logger = logging.getLogger(__name__)


@register
class XGBClassifierModel:
    name = "xgb"

    DEFAULTS = dict(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
    )

    def __init__(
        self,
        *,
        scale_pos_weight: Optional[float] = None,
        early_stopping_rounds: int = 50,
        random_state: int = 0,
        **kwargs,
    ) -> None:
        from xgboost import XGBClassifier

        params = {**self.DEFAULTS, **kwargs}
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params["random_state"] = random_state
        self._early_stopping_rounds = early_stopping_rounds
        self._clf = XGBClassifier(**params)

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> None:
        eval_set = None
        if X_val is not None and y_val is not None and len(y_val) > 0:
            eval_set = [(X_val, y_val)]
            try:
                self._clf.set_params(early_stopping_rounds=self._early_stopping_rounds)
            except Exception:
                # Older xgboost versions take it as a fit-time arg instead.
                pass
        self._clf.fit(X, y, sample_weight=sample_weight, eval_set=eval_set, verbose=False)

    def predict_proba(self, X) -> np.ndarray:
        return self._clf.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        return getattr(self._clf, "feature_importances_", None)

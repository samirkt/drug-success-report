"""Logistic regression baseline.

Demonstrates that swapping models is a small wrapper. Standardizes the
feature matrix internally so coefficients are comparable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import register


@register
class LogRegModel:
    name = "logreg"

    def __init__(
        self,
        *,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 0,
        C: float = 1.0,
        max_iter: int = 2000,
        **kwargs,
    ) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # `scale_pos_weight` -> sklearn class_weight={0: 1, 1: spw}
        class_weight = (
            "balanced" if scale_pos_weight is None else {0: 1.0, 1: float(scale_pos_weight)}
        )
        self._pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # sparse-safe
            ("clf", LogisticRegression(
                C=C,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=random_state,
                solver="liblinear",
                **kwargs,
            )),
        ])

    def fit(self, X, y, *, sample_weight=None, X_val=None, y_val=None) -> None:
        self._pipe.fit(X, y, clf__sample_weight=sample_weight)

    def predict_proba(self, X) -> np.ndarray:
        return self._pipe.predict_proba(X)

    def feature_importances(self) -> Optional[np.ndarray]:
        coef = self._pipe.named_steps["clf"].coef_
        return np.abs(coef[0]) if coef is not None else None

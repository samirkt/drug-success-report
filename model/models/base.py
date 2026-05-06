"""Model abstraction: ModelProtocol + registry.

All concrete models implement `fit`, `predict_proba`, `feature_importances`.
Adding a new model is a 5-line wrapper + `@register` decorator.
"""

from __future__ import annotations

from typing import ClassVar, Optional, Protocol, runtime_checkable

import numpy as np

MODEL_REGISTRY: dict[str, type["ModelProtocol"]] = {}


@runtime_checkable
class ModelProtocol(Protocol):
    name: ClassVar[str]

    def fit(
        self,
        X,
        y,
        *,
        sample_weight=None,
        X_val=None,
        y_val=None,
    ) -> None: ...

    def predict_proba(self, X) -> np.ndarray: ...

    def feature_importances(self) -> Optional[np.ndarray]: ...


def register(cls: type[ModelProtocol]) -> type[ModelProtocol]:
    if not getattr(cls, "name", None):
        raise ValueError(f"{cls.__name__} missing class-level `name`")
    if cls.name in MODEL_REGISTRY:
        raise ValueError(f"model {cls.name!r} already registered")
    MODEL_REGISTRY[cls.name] = cls
    return cls


def build_model(name: str, **kwargs) -> ModelProtocol:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**kwargs)

"""Model registry — importing this module registers all built-in models."""

from .base import MODEL_REGISTRY, ModelProtocol, build_model, register  # noqa: F401
from . import logreg  # noqa: F401
from . import xgb  # noqa: F401

__all__ = ["MODEL_REGISTRY", "ModelProtocol", "build_model", "register"]

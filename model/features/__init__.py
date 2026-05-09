"""Feature group registry — importing this module registers all six groups."""

from .base import FEATURE_GROUPS, FeatureGroup, build_group, register  # noqa: F401

# Side-effect imports populate FEATURE_GROUPS via @register.
from . import admet  # noqa: F401
from . import disease  # noqa: F401
from . import embeddings  # noqa: F401
from . import fingerprints  # noqa: F401
from . import nn_similarity  # noqa: F401
from . import pathway  # noqa: F401
from . import targets  # noqa: F401

__all__ = ["FEATURE_GROUPS", "FeatureGroup", "build_group", "register"]

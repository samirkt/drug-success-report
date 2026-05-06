"""Disease features: disease_area one-hot + MeSH tree-prefix multi-hot.

`disease_area` is a low-cardinality categorical (~20 levels) populated by
the classification stage. `mesh_condition_tree_numbers` is a list of MeSH
tree numbers like `C04.557.470.200`; we truncate to the 3-char top-level
prefix (`C04`) for multi-label encoding — keeps cardinality at ~70 and
preserves disease-class signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ._multilabel import TopKMultiLabel, iter_lists
from .base import register

logger = logging.getLogger(__name__)


def _truncate_to_top_level(values: pd.Series) -> pd.Series:
    """Map each list of MeSH tree numbers to its set of top-level prefixes."""
    out: list[list[str]] = []
    for tokens in iter_lists(values):
        prefixes = sorted({t.split(".")[0] for t in tokens if t})
        out.append(prefixes)
    return pd.Series(out, index=values.index)


@register
class DiseaseGroup:
    name = "disease"
    area_column = "disease_area"
    mesh_column = "mesh_condition_tree_numbers"

    def __init__(self, top_k_mesh: int = 200) -> None:
        self._mesh_enc = TopKMultiLabel(top_k=top_k_mesh, prefix="mesh")
        self._area_vocab: list[str] = []

    def is_available(self, df: pd.DataFrame) -> bool:
        return self.area_column in df.columns or self.mesh_column in df.columns

    def fit(self, df: pd.DataFrame) -> None:
        if self.area_column in df.columns:
            self._area_vocab = sorted(
                v for v in df[self.area_column].dropna().unique()
            )
        else:
            self._area_vocab = []
        if self.mesh_column in df.columns:
            self._mesh_enc.fit(_truncate_to_top_level(df[self.mesh_column]))
        logger.info(
            "disease: fit on %d rows; disease_area vocab=%d, mesh prefix vocab=%d",
            len(df),
            len(self._area_vocab),
            len(self._mesh_enc.vocab),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        # disease_area one-hot
        if self._area_vocab and self.area_column in df.columns:
            area_idx = {v: i for i, v in enumerate(self._area_vocab)}
            area = np.zeros((n, len(self._area_vocab) + 1), dtype=np.float32)
            for i, v in enumerate(df[self.area_column].values):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    area[i, -1] = 1.0  # _missing
                else:
                    idx = area_idx.get(v)
                    if idx is not None:
                        area[i, idx] = 1.0
                    else:
                        area[i, -1] = 1.0  # unseen-at-fit → treat as missing
        else:
            area = np.zeros((n, 0), dtype=np.float32)

        if self.mesh_column in df.columns:
            mesh = self._mesh_enc.transform(_truncate_to_top_level(df[self.mesh_column]))
        else:
            mesh = np.zeros((n, 0), dtype=np.float32)

        return np.hstack([area, mesh])

    def feature_names(self) -> list[str]:
        names = [f"disease_area_{v}" for v in self._area_vocab]
        if self._area_vocab:
            names.append("disease_area_missing")
        return names + self._mesh_enc.feature_names()

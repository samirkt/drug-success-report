"""KillerFigureBaseline: thin Baseline wrapper around `KillerFigureRetriever`.

Predicts each test candidate's approval probability as the mean realized
label among its k=10 nearest neighbors in joint molecule × target ×
indication space. Reports for every test row are accumulated during
`predict_proba` and written as a JSONL artifact via `write_artifacts`,
so the full structured "because Z" decomposition is preserved alongside
the AUC table.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from ..killer_figure import (
    KillerFigureReport,
    KillerFigureRetriever,
    _DEFAULT_K,
    _DEFAULT_MIN_NEIGHBORS,
    _DEFAULT_WEIGHTS,
)
from . import register

logger = logging.getLogger(__name__)


@register
class KillerFigureBaseline:
    name: ClassVar[str] = "killer_figure"

    def __init__(
        self,
        *,
        k: int = _DEFAULT_K,
        min_neighbors: int = _DEFAULT_MIN_NEIGHBORS,
        weights: Optional[dict] = None,
    ) -> None:
        self.k = k
        self.min_neighbors = min_neighbors
        self.weights = dict(weights) if weights else dict(_DEFAULT_WEIGHTS)
        self._retriever: Optional[KillerFigureRetriever] = None
        self._reports: list[KillerFigureReport] = []
        self._n_test: int = 0
        self._n_insufficient: int = 0

    def fit(self, train_df: pd.DataFrame, y_train: np.ndarray) -> None:
        self._retriever = KillerFigureRetriever(
            weights=self.weights, k=self.k, min_neighbors=self.min_neighbors
        )
        self._retriever.fit(train_df, y_train)

    def predict_proba(self, test_df: pd.DataFrame) -> np.ndarray:
        if self._retriever is None:
            raise RuntimeError("KillerFigureBaseline.predict_proba called before fit()")
        out = np.empty(len(test_df), dtype=np.float64)
        self._reports = []
        n_insufficient = 0
        for i, row in enumerate(test_df.to_dict(orient="records")):
            report = self._retriever.retrieve(row)
            self._reports.append(report)
            out[i] = report.approval_rate_neighbors
            if report.insufficient_prior_art:
                n_insufficient += 1
        self._n_test = len(test_df)
        self._n_insufficient = n_insufficient
        logger.info(
            "killer_figure predict: n_test=%d insufficient_prior_art=%d",
            self._n_test,
            self._n_insufficient,
        )
        return out

    def metadata(self) -> dict:
        retr_meta = self._retriever.metadata() if self._retriever is not None else {}
        n_neighbors_dist = Counter(r.n_neighbors for r in self._reports)
        return {
            "k": self.k,
            "min_neighbors": self.min_neighbors,
            "weights": dict(self.weights),
            "retriever": retr_meta,
            "n_test": self._n_test,
            "n_insufficient_prior_art": self._n_insufficient,
            "n_neighbors_per_query_histogram": dict(sorted(n_neighbors_dist.items())),
        }

    # `runner.run_baselines` calls this hook after `_save_lookup_baseline`
    # if it exists, allowing baselines to write extra artifacts beyond
    # `metrics.json` + `predictions.csv`.
    def write_artifacts(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "killer_figure_report.jsonl"
        with out_path.open("w") as f:
            for report in self._reports:
                f.write(json.dumps(report.to_dict(), default=_json_default))
                f.write("\n")
        logger.info(
            "killer_figure: wrote %d structured reports to %s",
            len(self._reports),
            out_path,
        )


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        try:
            return o.isoformat()
        except Exception:
            pass
    raise TypeError(f"not JSON serializable: {type(o)}")

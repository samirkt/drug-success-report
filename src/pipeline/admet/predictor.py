"""Lazy wrapper around ``admet_ai.ADMETModel`` with optional caching.

This module is intentionally independent of the rest of the pipeline:
it imports nothing from ``pipeline.*`` or ``utils.*`` so it can be reused
or shipped on its own. ``ADMETModel`` itself is loaded lazily on first
miss to keep cold pipeline starts fast and to allow deployments that omit
``admet_ai`` to fall through cleanly.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class AdmetPredictor:
    """Predict ADMET property dicts for SMILES strings.

    Public API:
        predict(smiles_list) -> dict[smiles, dict[col, float|None] | None]

    A ``None`` value for a SMILES key means the row failed to predict
    (parse error, all-NaN row). Such keys are NOT cached so a future
    model upgrade can retry them.
    """

    def __init__(
        self,
        cache=None,
        model_version: Optional[str] = None,
    ) -> None:
        self._cache = cache
        self._model = None
        if model_version is None:
            try:
                import admet_ai
                model_version = getattr(admet_ai, "__version__", "unknown")
            except ImportError:
                model_version = "unknown"
        self._model_version = model_version

    @property
    def model_version(self) -> str:
        return self._model_version

    def _ensure_model(self) -> None:
        if self._model is None:
            from admet_ai import ADMETModel
            self._model = ADMETModel()

    def predict(
        self, smiles_list: list[str]
    ) -> dict[str, Optional[dict[str, Optional[float]]]]:
        result: dict[str, Optional[dict[str, Optional[float]]]] = {}
        unique = list(dict.fromkeys(s for s in smiles_list if s and s.strip()))
        if not unique:
            return result

        cached: dict[str, dict[str, Optional[float]]] = {}
        if self._cache is not None:
            cached = self._cache.get_many(unique, self._model_version)
        for smi, pred in cached.items():
            result[smi] = pred

        misses = [s for s in unique if s not in cached]
        if not misses:
            return result

        # RDKit-validate before handing off. admet_ai 2.0.1 silently drops
        # SMILES it can't parse, then crashes on torch.cat over an empty
        # batch — taking down the entire batch's good predictions with it.
        # Filter on our side; mark the unparseable ones as None so the
        # caller's None-handling path catches them.
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        valid_misses = [s for s in misses if Chem.MolFromSmiles(s) is not None]
        for smi in misses:
            if smi not in valid_misses:
                result[smi] = None
        if not valid_misses:
            return result

        self._ensure_model()
        df = self._model.predict(smiles=valid_misses)

        new_predictions: dict[str, dict[str, Optional[float]]] = {}
        for smi in valid_misses:
            if smi not in df.index:
                result[smi] = None
                continue
            row = df.loc[smi]
            pred = {
                col: (None if isinstance(v, float) and math.isnan(v) else float(v))
                for col, v in row.items()
            }
            if all(v is None for v in pred.values()):
                # Malformed input — surface as a failure but do NOT cache
                # so a future model version is free to retry.
                result[smi] = None
            else:
                new_predictions[smi] = pred
                result[smi] = pred

        if new_predictions and self._cache is not None:
            self._cache.put_many(new_predictions, self._model_version)

        return result


__all__ = ["AdmetPredictor"]

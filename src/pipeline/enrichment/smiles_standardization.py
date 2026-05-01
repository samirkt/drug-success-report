"""SMILES standardization — produces canonical SMILES via the ChEMBL pipeline.

Runs after the DrugBank and ChEMBL fallback SMILES enrichments so that
both raw sources have already populated `Candidate.smiles`. For each
candidate with a non-empty raw SMILES, applies
``chembl_structure_pipeline.standardizer.standardize_mol`` followed by
``get_parent_mol`` (strips salts, normalizes tautomers), then writes
the canonical RDKit SMILES into ``Candidate.smiles_canonical``. The raw
``Candidate.smiles`` is never mutated, and no candidates are dropped.

Each candidate's outcome is recorded in
``Candidate.smiles_standardization_status`` with one of:

- ``ok`` — canonical SMILES populated.
- ``empty`` — raw SMILES was missing or blank.
- ``failed_parse`` — RDKit could not parse the input string.
- ``failed_standardize`` — the structure pipeline raised on a parsed mol.

The reporting layer turns these statuses into ``smiles_standardization_log.csv``.
The enrichment lazy-imports ``rdkit`` and ``chembl_structure_pipeline``
so installs that intentionally omit them only see a one-line skip.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


class SmilesStandardizationEnrichment:
    """Canonicalize ``Candidate.smiles`` into ``Candidate.smiles_canonical``."""

    name = "smiles_standardization"

    def __init__(self) -> None:
        self._deps = None  # populated on first is_available() success

    def is_available(self, config: "PipelineConfig") -> bool:
        if not getattr(config, "enable_smiles_standardization", True):
            return False
        if self._deps is not None:
            return True
        try:
            from chembl_structure_pipeline import standardizer
            from rdkit import Chem
            from rdkit import RDLogger
        except ImportError as exc:
            logger.warning(
                "SmilesStandardizationEnrichment: skipped — missing %s. "
                "Install with: pip install rdkit chembl_structure_pipeline",
                exc.name or "rdkit/chembl_structure_pipeline",
            )
            return False
        # RDKit prints a torrent of "WARNING" lines to stderr for any
        # questionable input. Silence it at module level — the per-row
        # status field already records parse / standardize failures.
        RDLogger.DisableLog("rdApp.*")
        self._deps = (Chem, standardizer)
        return True

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        Chem, standardizer = self._deps  # is_available guaranteed
        total = len(candidates.candidates)
        n_ok = 0
        n_empty = 0
        n_failed_parse = 0
        n_failed_standardize = 0

        for cand in candidates.candidates:
            raw = (cand.smiles or "").strip()
            if not raw:
                cand.smiles_standardization_status = "empty"
                n_empty += 1
                continue
            mol = Chem.MolFromSmiles(raw)
            if mol is None:
                cand.smiles_standardization_status = "failed_parse"
                n_failed_parse += 1
                continue
            try:
                std = standardizer.standardize_mol(mol)
                parent_result = standardizer.get_parent_mol(std)
                # get_parent_mol returns (mol, exclude_flag) in the public
                # API; tolerate either shape so a future minor version that
                # returns the mol directly doesn't break us.
                if isinstance(parent_result, tuple):
                    parent = parent_result[0]
                else:
                    parent = parent_result
                canon = Chem.MolToSmiles(parent, canonical=True)
            except Exception:
                cand.smiles_standardization_status = "failed_standardize"
                n_failed_standardize += 1
                continue
            if not canon:
                cand.smiles_standardization_status = "failed_standardize"
                n_failed_standardize += 1
                continue
            cand.smiles_canonical = canon
            cand.smiles_standardization_status = "ok"
            n_ok += 1

        pct_ok = (100.0 * n_ok / total) if total > 0 else 0.0
        logger.info(
            "SmilesStandardizationEnrichment: ok=%d empty=%d failed_parse=%d "
            "failed_standardize=%d (canonical coverage: %d/%d, %.0f%%)",
            n_ok, n_empty, n_failed_parse, n_failed_standardize,
            n_ok, total, pct_ok,
        )
        ledger.record_coverage(self.name, n_ok, total)
        return candidates


__all__ = ["SmilesStandardizationEnrichment"]

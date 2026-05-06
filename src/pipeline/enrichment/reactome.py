"""Reactome pathway enrichment — joins UniProt accessions to pathways.

Reads `UniProt2Reactome_All_Levels.txt` (tab-separated, columns:
`uniprot_id, reactome_pathway_id, url, pathway_name, evidence_code,
species`) and filters to `Homo sapiens`. The lookup is keyed by UniProt
accession; values are sorted `(pathway_id, pathway_name)` tuples.

For each candidate, every UniProt in `cand.drug_targets` is looked up
and the pathway sets are unioned (multi-target candidates aggregate by
union of pathway IDs across all targets, per spec).

The file is ~117 MB on disk; we stream-parse rather than loading via
pandas so peak residency stays around the size of the Homo sapiens
subset (~30 MB). The lookup is built lazily on the first `run()` and
cached on the instance.
"""

from __future__ import annotations

import logging
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from ..models import CandidateTable

if TYPE_CHECKING:
    from ..pipeline import PipelineConfig
    from utils.tiered_router import CostLedger

logger = logging.getLogger(__name__)


# UniProt accessions chosen for the spot-check on first lookup load.
# Each entry maps an accession to (label, minimum expected pathways).
# These are well-annotated human signaling proteins; if any returns far
# fewer pathways than the floor, the most likely cause is UniProt-format
# drift in the source file (e.g. isoform suffixes, missing accessions).
_SPOT_CHECKS: dict[str, tuple[str, int]] = {
    "P00533": ("EGFR", 30),
    "P15056": ("BRAF", 20),
    "Q15116": ("PD-1", 10),
}


class ReactomeEnrichment:
    """Join Reactome pathways onto candidates via UniProt accessions."""

    name = "reactome"

    def __init__(self) -> None:
        # {uniprot_accession: list[(pathway_id, pathway_name)]}
        self._lookup: dict[str, list[tuple[str, str]]] | None = None
        self._data_dir: Path | None = None
        self._version: Optional[str] = None

    def is_available(self, config: "PipelineConfig") -> bool:
        if not getattr(config, "enable_reactome", False):
            return False
        data_dir = getattr(config, "reactome_data_dir", None)
        if data_dir is None:
            logger.info(
                "ReactomeEnrichment: skipped — no reactome_data_dir configured."
            )
            return False
        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning(
                "ReactomeEnrichment: skipped — reactome_data_dir %s not found.",
                data_dir,
            )
            return False
        uniprot_file = data_dir / "UniProt2Reactome_All_Levels.txt"
        if not uniprot_file.exists():
            logger.warning(
                "ReactomeEnrichment: skipped — %s not found.", uniprot_file
            )
            return False
        self._data_dir = data_dir
        return True

    def _build_lookup(self) -> dict[str, list[tuple[str, str]]]:
        if self._lookup is not None:
            return self._lookup
        assert self._data_dir is not None  # guarded by is_available

        path = self._data_dir / "UniProt2Reactome_All_Levels.txt"
        # uniprot -> {pathway_id: pathway_name}
        accum: dict[str, dict[str, str]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6 or parts[5] != "Homo sapiens":
                    continue
                uni = parts[0].strip()
                pid = parts[1].strip()
                pname = parts[3].strip()
                if not uni or not pid:
                    continue
                accum.setdefault(uni, {})[pid] = pname

        # Freeze to sorted lists for deterministic candidate output.
        self._lookup = {
            uni: sorted(pmap.items()) for uni, pmap in accum.items()
        }

        version_file = self._data_dir / "reactome_version.txt"
        if version_file.exists():
            try:
                self._version = version_file.read_text().strip() or None
            except OSError:
                self._version = None

        logger.info(
            "ReactomeEnrichment: loaded %d UniProt → pathway entries (Reactome v%s)",
            len(self._lookup), self._version or "?",
        )
        self._sanity_check()
        return self._lookup

    def _sanity_check(self) -> None:
        """Catch UniProt format drift early — these three should all hit."""
        assert self._lookup is not None
        for uni, (label, minimum) in _SPOT_CHECKS.items():
            n = len(self._lookup.get(uni, []))
            if n < minimum:
                logger.warning(
                    "ReactomeEnrichment: spot-check fail — %s (%s) has %d pathways, "
                    "expected >= %d. Check UniProt ID format in source file.",
                    label, uni, n, minimum,
                )

    def run(
        self,
        candidates: CandidateTable,
        *,
        ledger: "CostLedger",
    ) -> CandidateTable:
        lookup = self._build_lookup()
        total = len(candidates.candidates)
        enriched = 0
        n_pathway_distribution: list[int] = []

        for cand in candidates.candidates:
            if not cand.drug_targets:
                cand.reactome_has_data = False
                continue
            seen: dict[str, str] = {}  # pathway_id -> pathway_name
            for uni in cand.drug_targets:
                for pid, pname in lookup.get(uni, ()):
                    seen.setdefault(pid, pname)
            if seen:
                items = sorted(seen.items())
                cand.reactome_pathway_ids = [pid for pid, _ in items]
                cand.reactome_pathway_names = [pname for _, pname in items]
                cand.reactome_n_pathways = len(items)
                cand.reactome_has_data = True
                enriched += 1
                n_pathway_distribution.append(len(items))
            else:
                cand.reactome_has_data = False

        pct = (100.0 * enriched / total) if total > 0 else 0.0
        logger.info(
            "ReactomeEnrichment: %d/%d (%.0f%%) candidates with >=1 pathway hit "
            "(Reactome v%s)",
            enriched, total, pct, self._version or "?",
        )
        if n_pathway_distribution:
            n_pathway_distribution.sort()
            logger.info(
                "ReactomeEnrichment: n_pathways distribution — "
                "min=%d  p25=%d  median=%d  p75=%d  max=%d  mean=%.1f",
                n_pathway_distribution[0],
                n_pathway_distribution[len(n_pathway_distribution) // 4],
                int(statistics.median(n_pathway_distribution)),
                n_pathway_distribution[(3 * len(n_pathway_distribution)) // 4],
                n_pathway_distribution[-1],
                statistics.mean(n_pathway_distribution),
            )
        if pct < 80.0 and total >= 10:
            logger.warning(
                "ReactomeEnrichment: coverage %.0f%% is below 80%% target. "
                "Most likely cause: many candidates lack drug_targets "
                "(TargetsEnrichment skipped or unmatched).",
                pct,
            )
        ledger.record_coverage(self.name, enriched, total)
        return candidates


__all__ = ["ReactomeEnrichment"]

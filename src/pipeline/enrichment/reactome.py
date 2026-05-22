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
from typing import TYPE_CHECKING, Any, Optional

from ..models import CandidateTable
from .reactome_hierarchy import load_hierarchy

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
        # {pathway_id: {depth, parents, children, top_level_ids, is_leaf, name}}
        self._hierarchy: dict[str, dict[str, Any]] | None = None
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

        try:
            self._hierarchy = load_hierarchy(self._data_dir)
            logger.info(
                "ReactomeEnrichment: loaded hierarchy with %d pathways "
                "(roots=%d, leaves=%d)",
                len(self._hierarchy),
                sum(1 for v in self._hierarchy.values() if v["depth"] == 0),
                sum(1 for v in self._hierarchy.values() if v["is_leaf"]),
            )
        except FileNotFoundError as e:
            logger.warning(
                "ReactomeEnrichment: hierarchy unavailable (%s); "
                "diagnostics will be skipped. Run data/reactome/download.sh "
                "to fetch ReactomePathwaysRelation.txt.",
                e,
            )
            self._hierarchy = None

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
        # Hierarchy spot-check: EGFR (P00533) should pull in pathways whose
        # root-ancestor set contains Signal Transduction (R-HSA-162582).
        if self._hierarchy is not None:
            egfr_pids = [pid for pid, _ in self._lookup.get("P00533", [])]
            if egfr_pids:
                roots: set[str] = set()
                for pid in egfr_pids:
                    info = self._hierarchy.get(pid)
                    if info is not None:
                        roots.update(info["top_level_ids"])
                if "R-HSA-162582" not in roots:
                    logger.warning(
                        "ReactomeEnrichment: hierarchy spot-check fail — "
                        "EGFR pathways do not roll up to Signal Transduction "
                        "(R-HSA-162582). Pathway/relation files may be out of sync.",
                    )

    def _apply_hierarchy_diagnostics(
        self, cand: Any, pathway_ids: list[str]
    ) -> int:
        """Populate hierarchy-aware diagnostic fields on a candidate.

        Returns the count of pathway IDs that were absent from the
        hierarchy lookup (used for coverage warnings).
        """
        assert self._hierarchy is not None
        pid_set = set(pathway_ids)
        depths: list[int] = []
        n_top = 0
        roots: set[str] = set()
        # "Locally leaf": a pathway in the set whose children (per the
        # global hierarchy) do not appear in the same set. This is what
        # tells us whether n_pathways=20 is "deep one branch" vs "broad".
        local_leaves: list[str] = []
        n_global_leaf = 0
        missing = 0
        for pid in pathway_ids:
            info = self._hierarchy.get(pid)
            if info is None:
                missing += 1
                continue
            depth = info["depth"]
            if depth >= 0:
                depths.append(depth)
            if depth == 0:
                n_top += 1
            roots.update(info["top_level_ids"])
            if info["is_leaf"]:
                n_global_leaf += 1
            if not any(child in pid_set for child in info["children"]):
                local_leaves.append(pid)

        cand.reactome_n_top_level_pathways = n_top
        cand.reactome_n_leaf_pathways = len(local_leaves)
        cand.reactome_n_leaf_global = n_global_leaf
        cand.reactome_n_internal_pathways = max(
            0, len(pathway_ids) - n_top - len(local_leaves)
        )
        cand.reactome_mean_depth = (
            float(statistics.mean(depths)) if depths else None
        )
        cand.reactome_max_depth = max(depths) if depths else None
        cand.reactome_top_level_pathway_ids = sorted(roots)
        cand.reactome_leaf_pathway_ids = sorted(local_leaves)
        return missing

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

        missing_from_hierarchy = 0
        total_with_pathways = 0
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
                if self._hierarchy is not None:
                    miss = self._apply_hierarchy_diagnostics(
                        cand, cand.reactome_pathway_ids
                    )
                    missing_from_hierarchy += miss
                    total_with_pathways += len(cand.reactome_pathway_ids)
            else:
                cand.reactome_has_data = False

        if (
            self._hierarchy is not None
            and total_with_pathways > 0
            and missing_from_hierarchy / total_with_pathways > 0.05
        ):
            logger.warning(
                "ReactomeEnrichment: %d/%d (%.1f%%) candidate pathway IDs "
                "missing from hierarchy — likely version drift between "
                "UniProt2Reactome_All_Levels.txt and ReactomePathways*.txt.",
                missing_from_hierarchy,
                total_with_pathways,
                100.0 * missing_from_hierarchy / total_with_pathways,
            )

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

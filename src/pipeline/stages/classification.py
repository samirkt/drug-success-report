"""
Stage 3a: Candidate Attribute Classification (Tiered)

Uses tiered Sonnet → Opus routing via the tiered_router module, and expands
the modality taxonomy from 3 classes (peptide/small_molecule/biologic) to 11
distinct drug modalities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import AttributeTable, Candidate, CandidateAttributes, CandidateTable
from utils.prompt_runner import (
    BATCH_SIZE,
    load_prompts_from_txt,
)
from utils.tiered_router import (
    CostLedger,
    confidence_below_threshold,
    tiered_batch_call,
)

if TYPE_CHECKING:
    from ..knowledge_cache import KnowledgeCache

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPTS_DIR / "classification_system.txt"
_USER_PROMPT_PATH = _PROMPTS_DIR / "classification_user.txt"

_CONFIDENCE_MAP = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.25}

# Expanded modality vocabulary (11 classes + unknown)
VALID_MODALITIES = frozenset({
    "peptide",
    "small_molecule",
    "monoclonal_antibody",
    "bispecific_antibody",
    "adc",
    "fusion_protein",
    "sirna",
    "antisense",
    "mrna",
    "cell_therapy",
    "gene_therapy",
    "vaccine",
    "unknown",
})

# Disease area taxonomy — aligned with BIO/QLS 2011-2020 categories
VALID_DISEASE_AREAS = frozenset({
    "oncology",
    "cardiovascular",
    "metabolic",
    "neurology",
    "autoimmune",
    "infectious_disease",
    "respiratory",
    "hematology",
    "ophthalmology",
    "psychiatry",
    "gastroenterology",
    "endocrine",
    "urology",
    "other",
    "unknown",
})

_DISEASE_AREA_ALIASES: dict[str, str] = {
    # LLM synonym handling
    "immunology": "autoimmune",
    "allergy": "autoimmune",
    "endocrinology": "endocrine",
    "addiction": "psychiatry",
    "hepatology": "gastroenterology",
    "vascular": "cardiovascular",
    # Old granular categories → BIO/QLS buckets
    "dermatology": "other",
    "musculoskeletal": "other",
    "reproductive_health": "other",
    "nephrology": "other",
    "rare_disease": "other",
    "anesthesiology": "other",
    "pain_management": "other",
    "transplant": "other",
    "dentistry": "other",
    "pediatrics": "other",
    "radiology": "other",
    "otolaryngology": "other",
    "surgery": "other",
    "emergency_medicine": "other",
    "critical_care": "other",
    "neonatology": "other",
    "geriatrics": "other",
    "rheumatology": "other",
    # Legacy synonym handling
    "reproductive": "other",
    "gynecology": "other",
    "obstetrics": "other",
    "pain": "other",
    "anesthesia": "other",
    "transplantation": "other",
    "dental": "other",
    # Non-clinical → other
    "drug_safety": "other",
    "research_tool": "other",
    "diagnostic": "other",
    "general_health": "other",
    "nutrition": "other",
    "toxicology": "other",
    "wound_care": "other",
    "plastic_surgery": "other",
}

# ---------------------------------------------------------------------------
# BIO/QLS display mapping
# ---------------------------------------------------------------------------

BIO_QLS_DISEASE_AREA_MAP: dict[str, str] = {
    "oncology": "Oncology",
    "cardiovascular": "Cardiovascular",
    "metabolic": "Metabolic",
    "neurology": "Neurology",
    "autoimmune": "Autoimmune",
    "infectious_disease": "Infectious disease",
    "respiratory": "Respiratory",
    "hematology": "Hematology",
    "ophthalmology": "Ophthalmology",
    "psychiatry": "Psychiatry",
    "gastroenterology": "Gastroenterology",
    "endocrine": "Endocrine",
    "urology": "Urology",
    "other": "Others",
    "unknown": "Others",
}


def to_bio_qls_disease_area(disease_area: str) -> str:
    """Map a pipeline disease area to its BIO/QLS display name."""
    return BIO_QLS_DISEASE_AREA_MAP.get(disease_area, "Others")


def normalize_disease_area(raw: str) -> str:
    """Normalize an LLM-returned disease area to a canonical BIO/QLS value."""
    normed = raw.strip().lower().replace(" ", "_")
    normed = _DISEASE_AREA_ALIASES.get(normed, normed)
    if normed not in VALID_DISEASE_AREAS:
        return "unknown"
    return normed


# ---------------------------------------------------------------------------
# MeSH tree-based disease area resolution
# ---------------------------------------------------------------------------

_MESH_TREE_TO_DISEASE_AREA: dict[str, str] = {
    "C01": "infectious_disease",
    "C02": "infectious_disease",
    "C03": "infectious_disease",
    "C04": "oncology",
    "C06": "gastroenterology",
    "C08": "respiratory",
    "C10": "neurology",
    "C11": "ophthalmology",
    "C12": "urology",
    "C14": "cardiovascular",
    "C15": "hematology",
    "C18": "metabolic",
    "C19": "endocrine",
    "C20": "autoimmune",
    "F03": "psychiatry",
}


def mesh_trees_to_disease_area(tree_numbers: list[str]) -> str | None:
    """Resolve disease area from MeSH condition tree numbers by plurality vote.

    Returns a BIO/QLS disease area string, or None if:
      - no tree numbers provided
      - tied between multiple named buckets (fallback to LLM)
    Returns "other" if all tree numbers map to unmapped prefixes.
    """
    from collections import Counter

    if not tree_numbers:
        return None
    votes: Counter[str] = Counter()
    for tn in tree_numbers:
        prefix = tn[:3]
        bucket = _MESH_TREE_TO_DISEASE_AREA.get(prefix)
        if bucket:
            votes[bucket] += 1
    if not votes:
        return "other"
    top = votes.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None  # tie — fallback to LLM
    return top[0][0]


def _classification_escalation_predicate(result: dict) -> bool:
    """Escalate if modality confidence is LOW or modality is unknown."""
    if confidence_below_threshold(result):
        return True
    modality = result.get("drug_modality", "unknown")
    if modality not in VALID_MODALITIES or modality == "unknown":
        return True
    return False


def _parse_classification(data: dict, candidate: Candidate) -> CandidateAttributes:
    """Convert a raw LLM response dict into a CandidateAttributes record."""
    modality = data.get("drug_modality", "unknown")
    if modality not in VALID_MODALITIES:
        modality = "unknown"
    return CandidateAttributes(
        candidate_id=candidate.candidate_id,
        drug_modality=modality,
        disease_area=normalize_disease_area(data.get("disease_area", "unknown")),
        modality_confidence=_CONFIDENCE_MAP.get(
            data.get("modality_confidence", ""), 0.0
        ),
        disease_confidence=_CONFIDENCE_MAP.get(
            data.get("disease_confidence", ""), 0.0
        ),
        reasoning=data.get("reasoning", ""),
    )


def _default_classification(candidate: Candidate) -> CandidateAttributes:
    """Fallback when both Sonnet and Opus fail for a candidate."""
    return CandidateAttributes(
        candidate_id=candidate.candidate_id,
        drug_modality="unknown",
        disease_area="unknown",
    )


class AttributeClassificationStage:
    """
    Tags each candidate with modality and disease-area labels using
    tiered Sonnet → Opus routing.

    Inputs:  CandidateTable
    Outputs: AttributeTable
    """

    def __init__(
        self,
        use_literature: bool = True,
        cache: "KnowledgeCache | None" = None,
        max_workers: int | None = 8,
        ledger: CostLedger | None = None,
    ):
        self.use_literature = use_literature
        self.cache = cache
        self.max_workers = max_workers
        self.ledger = ledger

    def run(self, candidate_table: CandidateTable) -> AttributeTable:
        """Classify all candidates in batches. Returns a populated AttributeTable.

        Disease area resolution priority:
          1. MeSH tree plurality (deterministic, from candidate's condition tree numbers)
          2. Cached LLM classification (from prior pipeline run)
          3. Fresh LLM classification

        Modality always comes from cache or LLM — no deterministic source exists.
        """
        candidates = candidate_table.candidates
        total = len(candidates)
        attributes: dict[str, CandidateAttributes] = {}
        cache_misses: list[tuple[str | None, Candidate]] = []

        logger.info("Classification: starting %d candidates", total)

        # Phase 1: resolve MeSH disease areas and cache hits for modality
        n_mesh_resolved = 0
        for candidate in candidates:
            # Try MeSH tree plurality for disease area
            mesh_disease = mesh_trees_to_disease_area(
                candidate.mesh_condition_tree_numbers
            )
            if mesh_disease is not None:
                n_mesh_resolved += 1

            # Check cache for modality (and disease fallback)
            if self.cache is not None:
                from ..knowledge_cache import KnowledgeCache

                key = KnowledgeCache.make_classification_key(
                    candidate.drug_name, candidate.indication
                )
                cached = self.cache.get_attributes(key, candidate.candidate_id)
                if cached is not None:
                    # Use cached modality; override disease_area with MeSH if available
                    attrs = CandidateAttributes(
                        candidate_id=candidate.candidate_id,
                        drug_modality=cached.drug_modality,
                        disease_area=mesh_disease if mesh_disease is not None else normalize_disease_area(cached.disease_area),
                        modality_confidence=cached.modality_confidence,
                        disease_confidence=1.0 if mesh_disease is not None else cached.disease_confidence,
                        reasoning=cached.reasoning,
                    )
                    attributes[candidate.candidate_id] = attrs
                    continue
            else:
                key = None
            cache_misses.append((key, candidate))

        n_hits = total - len(cache_misses)
        n_misses = len(cache_misses)
        logger.info(
            "Classification: %d MeSH disease resolved, %d cache hits, %d LLM calls needed",
            n_mesh_resolved,
            n_hits,
            n_misses,
        )
        if n_hits and self.ledger is not None:
            self.ledger.record_cache_hits(stage="Classification", n_hits=n_hits)

        # Phase 2: tiered batch LLM calls for cache misses
        system_prompt, user_template = load_prompts_from_txt(
            _SYSTEM_PROMPT_PATH, _USER_PROMPT_PATH
        )

        processed = 0
        for chunk_start in range(0, n_misses, BATCH_SIZE):
            chunk = cache_misses[chunk_start : chunk_start + BATCH_SIZE]
            chunk_candidates = [c for _, c in chunk]

            def _cls_fields(i: int, c: Candidate) -> list[str]:
                return [
                    f"Drug name: {c.drug_name_raw}",
                    f"Indication: {c.indication}",
                ]

            try:
                records = tiered_batch_call(
                    system_prompt=system_prompt,
                    user_template=user_template,
                    candidates=chunk_candidates,
                    fields_fn=_cls_fields,
                    parse_fn=_parse_classification,
                    default_fn=_default_classification,
                    escalation_predicate=_classification_escalation_predicate,
                    stage_name="Classification",
                    ledger=self.ledger,
                )
            except Exception as exc:
                logger.warning(
                    "Classification batch failed (candidates %d-%d): %s",
                    chunk_start,
                    chunk_start + len(chunk) - 1,
                    exc,
                )
                records = [_default_classification(c) for c in chunk_candidates]

            for (key, candidate), record in zip(chunk, records):
                # Cache the LLM result as-is before MeSH override
                if self.cache is not None and key is not None:
                    self.cache.put_attributes(key, record)
                # Apply MeSH override for the in-memory result
                mesh_disease = mesh_trees_to_disease_area(
                    candidate.mesh_condition_tree_numbers
                )
                if mesh_disease is not None:
                    record = CandidateAttributes(
                        candidate_id=record.candidate_id,
                        drug_modality=record.drug_modality,
                        disease_area=mesh_disease,
                        modality_confidence=record.modality_confidence,
                        disease_confidence=1.0,
                        reasoning=record.reasoning,
                    )
                attributes[candidate.candidate_id] = record

            processed += len(chunk)
            n_done = n_hits + processed
            logger.info(
                "Classification: %d/%d (%.0f%%)",
                n_done,
                total,
                100 * n_done / total,
            )

        n_llm_disease = total - n_mesh_resolved
        logger.info(
            "Classification: disease area by MeSH %d/%d (%.1f%%), LLM fallback %d (%.1f%%)",
            n_mesh_resolved, total, 100 * n_mesh_resolved / total if total else 0,
            n_llm_disease, 100 * n_llm_disease / total if total else 0,
        )

        return AttributeTable(attributes=attributes)

    def classify_from_fields(
        self,
        drug_name_raw: str,
        indication: str,
    ) -> CandidateAttributes:
        """Convenience method to classify a single candidate given raw fields."""
        candidate = Candidate(
            candidate_id=f"{drug_name_raw}_{indication}",
            drug_name=drug_name_raw,
            indication=indication,
            drug_name_raw=drug_name_raw,
        )
        system_prompt, user_template = load_prompts_from_txt(
            _SYSTEM_PROMPT_PATH, _USER_PROMPT_PATH
        )

        def _cls_fields(i: int, c: Candidate) -> list[str]:
            return [
                f"Drug name: {c.drug_name_raw}",
                f"Indication: {c.indication}",
            ]

        results = tiered_batch_call(
            system_prompt=system_prompt,
            user_template=user_template,
            candidates=[candidate],
            fields_fn=_cls_fields,
            parse_fn=_parse_classification,
            default_fn=_default_classification,
            escalation_predicate=_classification_escalation_predicate,
            stage_name="Classification",
            ledger=self.ledger,
        )
        return results[0]

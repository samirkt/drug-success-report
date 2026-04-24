"""Candidate enrichment stages (SMILES, drug targets, ICD-10 codes).

Each enrichment is an optional, pluggable step that runs after clustering
and only adds fields to existing `Candidate` objects. Enrichments never
modify `drug_name`, `indication`, `highest_phase`, or `candidate_id`, so
every KnowledgeCache key derived from those fields remains stable whether
a given enrichment is enabled or not.
"""

from .base import EnrichmentStage, run_enrichments
from .smiles import SmilesEnrichment

__all__ = ["EnrichmentStage", "SmilesEnrichment", "run_enrichments"]

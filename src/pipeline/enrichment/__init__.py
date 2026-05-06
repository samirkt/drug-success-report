"""Candidate enrichment stages (SMILES, drug targets, ICD-10 codes).

Each enrichment is an optional, pluggable step that runs after clustering
and only adds fields to existing `Candidate` objects. Enrichments never
modify `drug_name`, `indication`, `highest_phase`, or `candidate_id`, so
every KnowledgeCache key derived from those fields remains stable whether
a given enrichment is enabled or not.
"""

from .admet import AdmetEnrichment
from .base import EnrichmentStage, run_enrichments
from .chembl_smiles import ChemblSmilesEnrichment
from .icd import IcdEnrichment
from .opentargets import OpenTargetsEnrichment
from .smiles import SmilesEnrichment
from .smiles_standardization import SmilesStandardizationEnrichment
from .targets import TargetsEnrichment

__all__ = [
    "AdmetEnrichment",
    "ChemblSmilesEnrichment",
    "EnrichmentStage",
    "IcdEnrichment",
    "OpenTargetsEnrichment",
    "SmilesEnrichment",
    "SmilesStandardizationEnrichment",
    "TargetsEnrichment",
    "run_enrichments",
]

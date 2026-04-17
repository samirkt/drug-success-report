"""LLM-backed extraction and semantic matching of indications.

Two distinct tasks:

1. **Extract**: given an approval letter PDF (text) or a label's
   indications-and-usage section, return a structured list of
   approved indications with any population restrictions.

2. **Match**: given a trial indication (from ClinicalTrials.gov / MeSH)
   and an approved-indication list, decide whether the trial's
   indication is covered by any approved one, with a verdict of
   APPROVED / RELATED_BUT_DISTINCT / NOT_APPROVED / UNCERTAIN.

The LLMClient is a narrow protocol so the pipeline can be tested
with a fake implementation.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional, Protocol

logger = logging.getLogger(__name__)


MatchVerdict = Literal["APPROVED", "RELATED_BUT_DISTINCT", "NOT_APPROVED", "UNCERTAIN"]


@dataclass
class ExtractedIndication:
    """One atomic indication pulled from FDA free text."""

    indication_text: str  # e.g. "metastatic non-small cell lung cancer"
    population_restriction: Optional[str] = None  # e.g. "PD-L1 >=1%, adults"
    combination_partners: list[str] = None  # drugs required in combination
    source_excerpt: str = ""  # verbatim excerpt from source doc

    def __post_init__(self):
        if self.combination_partners is None:
            self.combination_partners = []


@dataclass
class MatchResult:
    verdict: MatchVerdict
    reasoning: str
    matched_indication: Optional[ExtractedIndication] = None


class LLMClient(Protocol):
    """Minimal interface so this module doesn't depend on any one SDK."""

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """Return a JSON object conforming to schema. Should use
        temperature=0 or equivalent determinism."""
        ...


_EXTRACT_SYSTEM = """\
You are a regulatory affairs analyst. Your job is to read FDA source \
text (approval letters or SPL label sections) and extract the specific \
indications for which a drug has been approved.

Rules:
- Return one entry per atomic indication. If the label says "Drug X is \
indicated for A and for B", return two entries.
- Preserve population restrictions (age, biomarker status, prior-therapy \
requirements) in the population_restriction field, NOT in the main \
indication_text.
- If the drug is approved only in combination with another therapy, list \
the combination partners.
- Use the shortest disease-naming phrase that captures the indication \
faithfully. Prefer standard disease names over verbose label phrasing.
- For each entry, include a verbatim source_excerpt (<=200 chars) from \
the input showing where you found it.
- Return only indications explicitly stated as approved / indicated. Do \
not include limitations-of-use, contraindications, or off-label mentions.
"""

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "indications": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "indication_text": {"type": "string"},
                    "population_restriction": {"type": ["string", "null"]},
                    "combination_partners": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "source_excerpt": {"type": "string"},
                },
                "required": ["indication_text", "source_excerpt"],
            },
        }
    },
    "required": ["indications"],
}


_MATCH_SYSTEM = """\
You are a regulatory affairs analyst deciding whether an FDA-approved \
indication covers a specific clinical-trial indication.

You will be given:
  1. A trial indication (typically a MeSH-coded disease name).
  2. A list of FDA-approved indications for the same drug, each with \
any population restrictions.

Decide one of:
  - APPROVED: The trial indication is substantively the same as, or \
falls within the scope of, an approved indication. A broader approved \
indication that clearly encompasses the trial indication counts.
  - RELATED_BUT_DISTINCT: The approved indication(s) are related but \
not the same disease as the trial indication (e.g. approved for type 2 \
diabetes, trial is for diabetic nephropathy).
  - NOT_APPROVED: No approved indication is relevant to the trial \
indication.
  - UNCERTAIN: Insufficient information to decide, OR the trial \
indication is a specific subpopulation that MAY or may not be covered \
by the approved wording.

Be strict. Population restrictions on the approved side matter: if the \
drug is approved only for "patients with EGFR mutations" and the trial \
is for unselected NSCLC, that is RELATED_BUT_DISTINCT, not APPROVED.
"""

_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["APPROVED", "RELATED_BUT_DISTINCT",
                     "NOT_APPROVED", "UNCERTAIN"],
        },
        "reasoning": {"type": "string"},
        "matched_indication_index": {"type": ["integer", "null"]},
    },
    "required": ["verdict", "reasoning"],
}


class IndicationAdjudicator:
    """LLM-backed extraction and matching of indications."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract_from_text(
        self, source_text: str, drug_name: str, source_kind: str = "label"
    ) -> list[ExtractedIndication]:
        """Extract atomic indications from a label or approval letter."""
        if not source_text or not source_text.strip():
            return []

        # Truncate very long inputs. Approval letters are short; SPL
        # indications sections are usually <10k chars; full SPL XML
        # can be much larger but we've already extracted the section.
        truncated = source_text[:30_000]

        user = (
            f"DRUG: {drug_name}\n"
            f"SOURCE TYPE: {source_kind}\n\n"
            f"SOURCE TEXT:\n{truncated}\n\n"
            f"Extract all explicitly approved indications."
        )

        try:
            result = self.llm.complete_json(_EXTRACT_SYSTEM, user, _EXTRACT_SCHEMA)
        except Exception as e:
            logger.warning("LLM extraction failed for %s: %s", drug_name, e)
            return []

        out: list[ExtractedIndication] = []
        for item in result.get("indications", []):
            text = item.get("indication_text", "").strip()
            if not text:
                continue
            out.append(
                ExtractedIndication(
                    indication_text=text,
                    population_restriction=item.get("population_restriction"),
                    combination_partners=item.get("combination_partners") or [],
                    source_excerpt=item.get("source_excerpt", ""),
                )
            )
        return out

    def match(
        self,
        trial_indication: str,
        mesh_indication: Optional[str],
        approved: list[ExtractedIndication],
    ) -> MatchResult:
        """Decide whether any approved indication covers the trial indication."""
        if not approved:
            return MatchResult(
                verdict="NOT_APPROVED",
                reasoning="No approved indications available.",
            )

        quick = _try_string_match(trial_indication, mesh_indication, approved)
        if quick is not None:
            return quick

        approved_block = "\n".join(
            f"  [{i}] {ind.indication_text}"
            + (f" (restricted to: {ind.population_restriction})"
               if ind.population_restriction else "")
            + (f" (in combination with: {', '.join(ind.combination_partners)})"
               if ind.combination_partners else "")
            for i, ind in enumerate(approved)
        )

        user = (
            f"TRIAL INDICATION: {trial_indication}\n"
            f"MESH INDICATION: {mesh_indication or '(none)'}\n\n"
            f"APPROVED INDICATIONS:\n{approved_block}\n\n"
            f"Return a verdict, reasoning, and the index of the matched "
            f"indication (if APPROVED)."
        )

        try:
            result = self.llm.complete_json(_MATCH_SYSTEM, user, _MATCH_SCHEMA)
        except Exception as e:
            logger.warning("LLM match failed for '%s': %s", trial_indication, e)
            return MatchResult(
                verdict="UNCERTAIN",
                reasoning=f"LLM error: {e}",
            )

        verdict = result.get("verdict", "UNCERTAIN")
        reasoning = result.get("reasoning", "")
        idx = result.get("matched_indication_index")
        matched = approved[idx] if isinstance(idx, int) and 0 <= idx < len(approved) else None

        return MatchResult(verdict=verdict, reasoning=reasoning, matched_indication=matched)


# ---------- deterministic string-match shortcut ----------


def _normalize_indication(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation edges."""
    return " ".join(s.lower().split()).strip(" .,;:")


def _try_string_match(
    trial_indication: str,
    mesh_indication: Optional[str],
    approved: list[ExtractedIndication],
) -> Optional[MatchResult]:
    """Return APPROVED without an LLM call when the match is trivially obvious.

    Checks both the trial indication and the MeSH indication (if present)
    against each approved indication for:
      1. Exact match (after normalization)
      2. Containment (trial text is a substring of approved, or vice versa)

    Only matches unrestricted approved indications — if an indication has a
    population_restriction or combination_partners, the match could be
    partial and needs the LLM to adjudicate.
    """
    trial_norm = _normalize_indication(trial_indication)
    mesh_norm = _normalize_indication(mesh_indication) if mesh_indication else None
    candidates = [trial_norm]
    if mesh_norm and mesh_norm != trial_norm:
        candidates.append(mesh_norm)

    for i, ind in enumerate(approved):
        if ind.population_restriction or ind.combination_partners:
            continue
        approved_norm = _normalize_indication(ind.indication_text)
        for cand in candidates:
            if cand == approved_norm or cand in approved_norm or approved_norm in cand:
                logger.debug(
                    "String-match shortcut: '%s' ↔ '%s'",
                    cand, approved_norm,
                )
                return MatchResult(
                    verdict="APPROVED",
                    reasoning=(
                        f"Deterministic string match: trial indication "
                        f"'{trial_indication}' matches approved indication "
                        f"'{ind.indication_text}' (no LLM needed)."
                    ),
                    matched_indication=ind,
                )
    return None


# ---------- PDF text extraction helper ----------

_PDF_WHITESPACE = re.compile(r"\s+")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from an approval-letter PDF.

    Uses pdfplumber for text PDFs; falls back to pytesseract-based OCR
    for older scanned letters. Kept as a plain function so callers can
    swap it out or mock it.
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError("pdfplumber is required for PDF extraction") from e

    import io

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)

    text = "\n".join(text_parts).strip()

    if len(text) < 50:
        # Likely a scanned PDF -- try OCR.
        text = _ocr_pdf(pdf_bytes)

    return _PDF_WHITESPACE.sub(" ", text).strip()


def _ocr_pdf(pdf_bytes: bytes) -> str:
    try:
        import io

        import pdf2image
        import pytesseract
    except ImportError:
        logger.warning("OCR dependencies not installed; skipping scanned PDF.")
        return ""

    images = pdf2image.convert_from_bytes(pdf_bytes, dpi=300)
    return "\n".join(pytesseract.image_to_string(img) for img in images)

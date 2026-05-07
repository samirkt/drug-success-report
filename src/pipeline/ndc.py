"""NDC-indication adjudication module.

Standalone-importable adjudicator that decides whether a clinical-trial
indication for a drug is covered by that drug's FDA-approved label
indications. Uses the local-SQLite-backed `utils.ndc_lookup` for FDA
label data and any pluggable LLM client (anything implementing
``complete_json(system, user, schema) -> dict``) for the coverage
decision.

Standalone usage::

    from pipeline.ndc import NDCAdjudicator
    from pipeline.fda import OpenAICompatJSONClient

    llm = OpenAICompatJSONClient(
        base_url="http://localhost:11434/v1",
        model="qwen2.5:7b-instruct",
    )
    adj = NDCAdjudicator(llm_client=llm)
    verdict = adj.adjudicate(
        drug_synonyms=["pembrolizumab", "Keytruda", "MK-3475"],
        indication="metastatic NSCLC",
    )

The pipeline stage in ``pipeline.stages.adjudication_ndc`` uses the
batch helpers (``synonyms_for_candidate``, ``resolve_drug_key``,
``match_indication``) directly so that the bulk SQLite lookup runs
once for all unique drugs across the candidate table.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from rapidfuzz import fuzz

from .drugbank_norm import canonicalize_drug_name, load_drugbank_synonyms
from .models import Candidate

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal interface; matches the protocol in ``pipeline.fda``."""

    def complete_json(self, system: str, user: str, schema: dict) -> dict:  # pragma: no cover
        ...


@dataclass
class NDCVerdict:
    """LLM coverage decision for a single (drug, indication) pair."""

    approved: bool
    confidence: float
    matched_indication: Optional[str]
    reasoning: str
    evidence_sources: list[str] = field(default_factory=list)
    matched_synonym: Optional[str] = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a regulatory affairs analyst. Decide whether a clinical-trial
indication for a specific drug is covered by any of that drug's
FDA-approved label indications.

INPUTS:
  - TRIAL INDICATION: free-text disease name from a clinical trial.
  - APPROVED LABEL INDICATIONS: numbered list taken verbatim from the
    drug's current FDA label(s). MAY contain unrelated indications.

RULES:
  - approved=true ONLY if at least one label indication substantively
    covers the trial indication. A broader approved indication that
    encompasses the trial indication counts. A narrower approved
    indication does NOT cover a broader trial indication.
  - approved=false if none match, or you cannot tell with reasonable
    confidence. Default to false on uncertainty.
  - matched_indication: copy the verbatim text of the matched label
    indication when approved=true; null otherwise. Do NOT invent text.
  - confidence: 0.0-1.0.
  - reasoning: 1-3 sentences.

CRITICAL: Do NOT reason about the drug's known indications from your
training data. Use only the label list provided.
"""

_USER_TEMPLATE = """\
TRIAL INDICATION: {trial_indication}
MESH INDICATION: {mesh_indication}

APPROVED LABEL INDICATIONS for {matched_synonym}:
{numbered_label_indications}
"""

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "matched_indication": {"type": ["string", "null"]},
        "reasoning": {"type": "string"},
    },
    "required": ["approved", "confidence", "matched_indication", "reasoning"],
}


# ---------------------------------------------------------------------------
# Synonym resolution helpers (used by the standalone API and the stage)
# ---------------------------------------------------------------------------

def resolve_drug_key(candidate: Candidate) -> str:
    """Return the dedup key used to collapse candidates sharing a drug.

    Prefers ``drugbank_id`` (most reliable), falls back to a canonicalized
    drug name. Two candidates with the same drug but different unmatched
    DrugBank IDs intentionally collide so they share one bulk lookup.
    """
    if candidate.drugbank_id:
        return candidate.drugbank_id
    return canonicalize_drug_name(candidate.drug_name) or candidate.drug_name.lower()


_TRAILING_PUNCT = re.compile(r"[\s.,;:!?…]+$")
_LEADING_PUNCT = re.compile(r"^[\s.,;:!?]+")
_WHITESPACE = re.compile(r"\s+")
# Matches SPL section references like "( 1 )", "( 1.1 )", "(1.2)", "(14.1.2)".
_SECTION_MARKER = re.compile(r"\(\s*\d+(?:\.\d+)*\s*\)")
# Sentence boundary: end-of-sentence punctuation followed by whitespace
# and a capital letter.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _normalize_indication(text: str) -> str:
    """Conservative normalization for dedup: lowercase, collapse
    whitespace, strip leading/trailing punctuation. Does NOT collapse
    semantically distinct indications (e.g. adult vs pediatric variants).
    """
    s = text.lower()
    s = _WHITESPACE.sub(" ", s).strip()
    s = _LEADING_PUNCT.sub("", s)
    s = _TRAILING_PUNCT.sub("", s)
    return s


def split_long_indications(
    indications: list[str], *, max_chars: int = 1500, min_chars: int = 30
) -> list[str]:
    """Split overlong label indication strings into smaller chunks.

    Many SPLs concatenate the entire INDICATIONS AND USAGE section into
    a single entry (e.g. Keytruda's two indications are 23k chars each,
    each containing 30+ distinct clinical indications separated by SPL
    section markers like "( 1.1 )"). Sending those whole defeats both
    the rapidfuzz pre-filter (one giant chunk swamps the rank) and the
    LLM context budget.

    Strategy:
      1. Entries shorter than ``max_chars`` pass through unchanged.
      2. For longer entries, split on SPL section markers (``( N.N )``).
         If that yields multiple reasonable chunks, return them.
      3. Otherwise, fall back to sentence-level splitting, aggregating
         sentences into chunks no larger than ``max_chars``.
    """
    out: list[str] = []
    for ind in indications:
        if len(ind) <= max_chars:
            out.append(ind)
            continue
        # Pass 1: split on SPL section markers. Always do this first to
        # preserve the natural clinical-indication boundaries.
        sections = [c.strip() for c in _SECTION_MARKER.split(ind)]
        sections = [c for c in sections if len(c) >= min_chars]
        if not sections:
            sections = [ind]
        # Pass 2: any section still too long gets sentence-aggregated
        # into chunks <= max_chars.
        for sec in sections:
            if len(sec) <= max_chars:
                out.append(sec)
                continue
            sentences = _SENTENCE_BOUNDARY.split(sec)
            buf = ""
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                if buf and len(buf) + 1 + len(s) > max_chars:
                    if len(buf) >= min_chars:
                        out.append(buf)
                    buf = s
                else:
                    buf = (buf + " " + s).strip() if buf else s
            if buf and len(buf) >= min_chars:
                out.append(buf)
    return out


def dedup_indications(indications: list[str]) -> list[str]:
    """Drop label indications that are exact-after-normalization duplicates
    of an earlier entry. Order is preserved; the first occurrence wins.

    The bulk SQLite lookup already dedups by raw string match, but FDA
    labels frequently carry the same indication with minor whitespace /
    punctuation / casing variation across SPL versions. This pass
    catches those without merging legitimately distinct indications.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ind in indications:
        norm = _normalize_indication(ind)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(ind)
    return out


def top_relevant_indications(
    trial_indication: str,
    mesh_indication: Optional[str],
    label_indications: list[str],
    *,
    k: int = 20,
) -> list[str]:
    """Rank label indications by string similarity to the trial indication
    and return the top-K. Order within the returned list reflects rank
    (highest similarity first).

    Necessary for OTC drugs like aspirin where the local FDA mirror
    holds 1000+ near-duplicate indications across manufacturer SPLs;
    sending all of them to the LLM blows the context window and stalls
    inference. For drugs with <=K indications, returns them unchanged
    (in original order).

    Uses ``rapidfuzz.fuzz.token_set_ratio`` (insensitive to word order
    and duplicates) against both the trial indication and the MeSH
    indication; takes the max of the two scores per label entry.
    """
    if len(label_indications) <= k:
        return list(label_indications)

    trial_norm = _normalize_indication(trial_indication)
    mesh_norm = _normalize_indication(mesh_indication) if mesh_indication else ""

    scored: list[tuple[float, int, str]] = []
    for idx, ind in enumerate(label_indications):
        ind_norm = _normalize_indication(ind)
        score = fuzz.token_set_ratio(trial_norm, ind_norm)
        if mesh_norm:
            score = max(score, fuzz.token_set_ratio(mesh_norm, ind_norm))
        # Negate for descending sort while keeping idx for stable ordering
        scored.append((-score, idx, ind))

    scored.sort()
    return [ind for _, _, ind in scored[:k]]


def synonyms_for_candidate(
    candidate: Candidate,
    drugbank_forward: Optional[dict[str, list[str]]] = None,
) -> list[str]:
    """Build an ordered, deduped synonym list for a candidate.

    Order: DrugBank synonyms first (most variants, typically includes the
    INN, brand names, and research codes), then ``drug_name``,
    ``drug_name_raw``, ``mesh_drug``. The bulk lookup walks this list and
    takes the first synonym that yields any FDA label hits.
    """
    out: list[str] = []
    seen: set[str] = set()

    def _add(term: Optional[str]) -> None:
        if not term:
            return
        t = term.strip()
        if not t:
            return
        key = t.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(t)

    if drugbank_forward and candidate.drugbank_id:
        for syn in drugbank_forward.get(candidate.drugbank_id, []):
            _add(syn)
    _add(candidate.drug_name)
    _add(candidate.drug_name_raw)
    _add(candidate.mesh_drug)
    return out


# ---------------------------------------------------------------------------
# NDCAdjudicator: standalone API + LLM coverage decision
# ---------------------------------------------------------------------------

class NDCAdjudicator:
    """Pluggable LLM-backed indication coverage decider.

    Standalone callers use ``adjudicate(synonyms, indication)`` which
    runs the bulk lookup itself. The pipeline stage uses
    ``match_indication(...)`` directly after it has run a single bulk
    lookup for all unique drugs across the candidate table.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        drugbank_synonyms_csv: Optional[Path] = None,
    ):
        self.llm = llm_client
        self._drugbank_forward: Optional[dict[str, list[str]]] = None
        if drugbank_synonyms_csv is not None:
            forward, _reverse = load_drugbank_synonyms(Path(drugbank_synonyms_csv))
            self._drugbank_forward = forward

    @property
    def drugbank_forward(self) -> dict[str, list[str]]:
        return self._drugbank_forward or {}

    # -- standalone single-pair API ----------------------------------------

    def adjudicate(
        self,
        drug_synonyms: list[str],
        indication: str,
        mesh_indication: Optional[str] = None,
    ) -> NDCVerdict:
        """Look up FDA label indications for ``drug_synonyms`` and decide
        whether ``indication`` is covered.

        Imports ``utils.ndc_lookup`` lazily so that callers without a
        local FDA SQLite DB can still import this module.
        """
        from utils import ndc_lookup

        if not drug_synonyms:
            return _not_in_label_verdict(matched_synonym=None)

        try:
            bulk = ndc_lookup.get_drugs_with_indications_bulk([drug_synonyms])
        except Exception as e:
            raise RuntimeError(_FDA_DB_HINT.format(error=e)) from e

        matched_synonym, label_indications = _select_match(drug_synonyms, bulk)
        if not label_indications:
            return _not_in_label_verdict(matched_synonym=matched_synonym)

        return self.match_indication(
            label_indications=label_indications,
            trial_indication=indication,
            mesh_indication=mesh_indication,
            matched_synonym=matched_synonym or drug_synonyms[0],
        )

    # -- batch helpers used by the pipeline stage --------------------------

    def match_indication(
        self,
        *,
        label_indications: list[str],
        trial_indication: str,
        mesh_indication: Optional[str],
        matched_synonym: str,
    ) -> NDCVerdict:
        """Pure LLM call: decide whether ``trial_indication`` is covered.

        Caller is responsible for ensuring ``label_indications`` is
        non-empty (we hard short-circuit empty lists upstream so the
        prompt rule "do not reason from training data" can't be
        accidentally exercised against an empty list).
        """
        split = split_long_indications(label_indications)
        deduped = dedup_indications(split)
        relevant = top_relevant_indications(
            trial_indication, mesh_indication, deduped, k=20
        )
        if len(relevant) < len(deduped) or len(split) > len(label_indications):
            logger.debug(
                "Pre-LLM filter for %r: %d raw -> %d split -> %d deduped -> %d sent",
                matched_synonym,
                len(label_indications), len(split), len(deduped), len(relevant),
            )
        numbered = "\n".join(
            f"{i}. {ind}" for i, ind in enumerate(relevant, start=1)
        )
        user = _USER_TEMPLATE.format(
            trial_indication=trial_indication,
            mesh_indication=mesh_indication or "(none)",
            matched_synonym=matched_synonym,
            numbered_label_indications=numbered,
        )

        try:
            payload = self.llm.complete_json(_SYSTEM_PROMPT, user, _RESPONSE_SCHEMA)
        except Exception as e:
            logger.warning(
                "NDC adjudication LLM call failed for %r: %s", matched_synonym, e
            )
            raise

        return _verdict_from_payload(payload, matched_synonym=matched_synonym)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FDA_DB_HINT = (
    "FDA local DB not available ({error}). "
    "Build it with: python -c 'from utils.ndc_lookup import build_db; build_db()'"
)


def _select_match(
    synonyms: list[str], bulk: dict[str, dict]
) -> tuple[Optional[str], list[str]]:
    """Walk synonyms and return the first one that produced indications.

    Returns ``(matched_synonym, indications)`` or ``(None, [])`` if no
    synonym yielded a label hit.
    """
    for syn in synonyms:
        record = bulk.get(syn)
        if record and record.get("indication_count", 0) > 0:
            return syn, list(record.get("indications", []))
    return None, []


def _verdict_from_payload(payload: Any, *, matched_synonym: str) -> NDCVerdict:
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict from LLM, got {type(payload).__name__}")
    approved = bool(payload.get("approved", False))
    confidence = float(payload.get("confidence", 0.0))
    matched = payload.get("matched_indication")
    reasoning = str(payload.get("reasoning", "")).strip() or "(no reasoning provided)"
    return NDCVerdict(
        approved=approved,
        confidence=max(0.0, min(1.0, confidence)),
        matched_indication=matched if isinstance(matched, str) and matched else None,
        reasoning=reasoning,
        evidence_sources=["ndc_lookup", "llm_match"],
        matched_synonym=matched_synonym,
    )


def _not_in_label_verdict(*, matched_synonym: Optional[str]) -> NDCVerdict:
    """Verdict for the deterministic case: drug has no label indications.

    No LLM call. Returned with high confidence — the FDA label database
    is the source of truth on whether a drug has any approved indication.
    """
    return NDCVerdict(
        approved=False,
        confidence=1.0,
        matched_indication=None,
        reasoning="No FDA label indications found for any synonym.",
        evidence_sources=["ndc_lookup"],
        matched_synonym=matched_synonym,
    )

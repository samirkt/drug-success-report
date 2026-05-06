"""Import NDC adjudication verdicts from a Colab GPU run.

Reads ``verdicts.jsonl`` and pre-populates the
``ndc_adjudication_cache`` table of the local KnowledgeCache. After
this, a normal pipeline run with
``--adjudication-method ndc_indication`` against the same candidate set
hits the cache for every imported candidate -- zero LLM calls.

Usage:
    python -m scripts.import_ndc_verdicts \\
        --verdicts /tmp/verdicts.jsonl \\
        --candidates /tmp/clustered.pkl \\
        --cache knowledge_cache.db
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

from pipeline.knowledge_cache import KnowledgeCache
from pipeline.models import (
    Candidate,
    CandidateOutcome,
    CandidateOutcomeRecord,
    CandidateTable,
)
from pipeline.stages._no_approval import classify_no_approval

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_WINDOW_DAYS = 2 * 365


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verdicts", required=True,
                   help="Path to verdicts.jsonl produced by the Colab run.")
    p.add_argument("--candidates", required=True,
                   help="Path to the CandidateTable pickle written by "
                        "export_ndc_work.py.")
    p.add_argument("--cache", required=True,
                   help="Path to knowledge_cache.db. Will be created if missing.")
    p.add_argument("--failure-window-days", type=int,
                   default=DEFAULT_FAILURE_WINDOW_DAYS,
                   help="Days within which a non-approved candidate is bucketed "
                        "as ONGOING rather than FAILED_PHASE_N. Default 730 "
                        "matches AdjudicationConfig.")
    return p.parse_args()


def _approved_record(
    candidate: Candidate, verdict: dict
) -> CandidateOutcomeRecord:
    matched = verdict.get("matched_indication") or "(unknown)"
    reasoning = (
        f"LLM matched '{matched}': {verdict.get('reasoning', '(no reasoning)')}"
    )
    return CandidateOutcomeRecord(
        candidate_id=candidate.candidate_id,
        outcome=CandidateOutcome.APPROVED,
        confidence=float(verdict.get("confidence", 0.0)),
        reasoning=reasoning,
        evidence_sources=["ndc_lookup", "llm_match", "colab_offload"],
        approval_date=None,
        commercialization_date=None,
    )


def _not_approved_record(
    candidate: Candidate, verdict: dict, failure_window_days: int,
) -> CandidateOutcomeRecord:
    outcome = classify_no_approval(
        candidate, failure_window_days=failure_window_days, as_of=None,
    )
    confidence = float(verdict.get("confidence", 0.0))
    if outcome == CandidateOutcome.ONGOING:
        confidence = min(confidence, 0.5)
    return CandidateOutcomeRecord(
        candidate_id=candidate.candidate_id,
        outcome=outcome,
        confidence=confidence,
        reasoning=f"LLM: {verdict.get('reasoning', '(no reasoning)')}",
        evidence_sources=["ndc_lookup", "llm_match", "colab_offload"],
        approval_date=None,
        commercialization_date=None,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = parse_args()

    candidates_path = Path(args.candidates)
    verdicts_path = Path(args.verdicts)
    cache_path = Path(args.cache)

    with candidates_path.open("rb") as f:
        candidate_table: CandidateTable = pickle.load(f)
    by_id = {c.candidate_id: c for c in candidate_table.candidates}
    logger.info("Loaded %d candidates from %s", len(by_id), candidates_path)

    cache = KnowledgeCache(cache_path)

    n_imported = 0
    n_missing_candidate = 0
    n_total = 0
    approvals = 0
    rejections = 0
    with verdicts_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            verdict = json.loads(line)
            cand_id = verdict.get("candidate_id")
            cand = by_id.get(cand_id)
            if cand is None:
                n_missing_candidate += 1
                logger.warning("verdict for unknown candidate_id=%s -- skipping",
                               cand_id)
                continue

            if verdict.get("approved", False):
                record = _approved_record(cand, verdict)
                approvals += 1
            else:
                record = _not_approved_record(
                    cand, verdict, args.failure_window_days,
                )
                rejections += 1

            cache_key = KnowledgeCache.make_adjudication_key(
                cand.drug_name, cand.indication, cand.highest_phase.value,
            )
            cache.put_ndc_outcome(cache_key, record)
            n_imported += 1

    cache.close()

    logger.info(
        "Imported %d / %d verdicts (%d approved, %d not approved); "
        "%d skipped (candidate not in pickle).",
        n_imported, n_total, approvals, rejections, n_missing_candidate,
    )
    print(f"\nImport complete:")
    print(f"  Verdicts read:         {n_total}")
    print(f"  Cached (APPROVED):     {approvals}")
    print(f"  Cached (not approved): {rejections}")
    print(f"  Skipped (unknown id):  {n_missing_candidate}")
    print(f"  Cache file:            {cache_path}")


if __name__ == "__main__":
    main()

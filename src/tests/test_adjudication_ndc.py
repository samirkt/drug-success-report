"""Tests for the NDC-indication adjudication stage and standalone API.

Both the LLM client and the local FDA SQLite lookup are replaced with
fakes so tests run without an LLM backend or a built fda.db.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pipeline.knowledge_cache import KnowledgeCache
from pipeline.models import (
    Candidate,
    CandidateOutcome,
    CandidateTable,
    TrialPhase,
)
from pipeline.ndc import (
    NDCAdjudicator,
    NDCVerdict,
    dedup_indications,
    resolve_drug_key,
    split_long_indications,
    synonyms_for_candidate,
    top_relevant_indications,
)
from pipeline.stages import adjudication_ndc as ndc_stage
from pipeline.stages.adjudication_ndc import (
    AdjudicationConfig,
    AdjudicationStage,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Returns the verdict configured by ``responses``, indexed by the
    matched_synonym that appears in the user prompt. Counts calls so
    tests can assert dedup."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def complete_json(self, system, user, schema):
        self.calls += 1
        for synonym, payload in self.responses.items():
            if f"for {synonym}:" in user:
                return payload
        # Default: not approved
        return {
            "approved": False,
            "confidence": 0.5,
            "matched_indication": None,
            "reasoning": "default fake",
        }


class RaisingLLM:
    cache_hits = 0
    cache_misses = 0

    def complete_json(self, system, user, schema):
        raise RuntimeError("fake LLM failure")


def fake_bulk_lookup(table):
    """Replacement for utils.ndc_lookup.get_drugs_with_indications_bulk.

    Returns a callable that consults the given ``table`` (synonym ->
    indications list). Mirrors the real return shape.
    """
    def _impl(drug_groups):
        out = {}
        for group in drug_groups:
            terms = [group] if isinstance(group, str) else list(group)
            chosen = None
            for term in terms:
                indications = table.get(term, [])
                if indications:
                    chosen = (term, indications)
                    break
            if chosen is None and terms:
                chosen = (terms[0], [])
            if chosen is None:
                continue
            term, inds = chosen
            out[term] = {
                "query": term,
                "indications": list(inds),
                "indication_count": len(inds),
                "indication_sources": [{"set_id": "fake-set", "effective_time": "20240101"}] if inds else [],
            }
        return out
    return _impl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _candidate(
    *,
    cid="C1",
    drug="pembrolizumab",
    drug_raw="Keytruda",
    indication="metastatic NSCLC",
    mesh_indication="Carcinoma, Non-Small-Cell Lung",
    mesh_drug="Pembrolizumab",
    drugbank_id="DB09037",
    phase=TrialPhase.PHASE_3,
    latest=date(2020, 1, 1),
):
    return Candidate(
        candidate_id=cid,
        drug_name=drug,
        drug_name_raw=drug_raw,
        indication=indication,
        mesh_indication=mesh_indication,
        mesh_drug=mesh_drug,
        drugbank_id=drugbank_id,
        highest_phase=phase,
        latest_completion_date=latest,
    )


@pytest.fixture
def cache(tmp_path):
    return KnowledgeCache(tmp_path / "cache.db")


# ---------------------------------------------------------------------------
# Stage tests
# ---------------------------------------------------------------------------


class TestStageRun:
    def test_approved_via_llm(self, monkeypatch, cache):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "pembrolizumab": ["metastatic non-small cell lung cancer"],
            }),
        )
        llm = FakeLLM({
            "pembrolizumab": {
                "approved": True,
                "confidence": 0.92,
                "matched_indication": "metastatic non-small cell lung cancer",
                "reasoning": "Trial NSCLC clearly covered.",
            }
        })
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        cand = _candidate()
        out = stage.run(CandidateTable(candidates=[cand]))
        rec = out.outcomes["C1"]
        assert rec.outcome == CandidateOutcome.APPROVED
        assert "metastatic non-small cell lung cancer" in rec.reasoning
        assert llm.calls == 1
        assert cache.get_ndc_outcome(
            KnowledgeCache.make_adjudication_key(
                cand.drug_name, cand.indication, cand.highest_phase.value
            ),
            "C1",
        ) is not None

    def test_drug_missing_from_label_db_uses_no_approval_mapping(
        self, monkeypatch, cache
    ):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({}),  # nothing matches
        )
        llm = FakeLLM({})
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(
                as_of=date(2025, 1, 1), failure_window_days=730
            ),
            cache=cache,
        )
        # Stale phase 3 -> FAILED_PHASE_3
        stale_cand = _candidate(
            cid="C-stale", phase=TrialPhase.PHASE_3, latest=date(2020, 1, 1)
        )
        # Recent phase 2 -> ONGOING
        active_cand = _candidate(
            cid="C-active", phase=TrialPhase.PHASE_2, latest=date(2024, 6, 1),
            drug="other-drug",
        )
        out = stage.run(CandidateTable(candidates=[stale_cand, active_cand]))
        assert out.outcomes["C-stale"].outcome == CandidateOutcome.FAILED_PHASE_3
        assert out.outcomes["C-active"].outcome == CandidateOutcome.ONGOING
        # No LLM calls — drug missing path is deterministic
        assert llm.calls == 0

    def test_llm_says_not_covered_uses_no_approval_mapping(
        self, monkeypatch, cache
    ):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "pembrolizumab": ["metastatic non-small cell lung cancer"],
            }),
        )
        llm = FakeLLM({
            "pembrolizumab": {
                "approved": False,
                "confidence": 0.85,
                "matched_indication": None,
                "reasoning": "Trial indication is unrelated to label list.",
            }
        })
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(
                as_of=date(2025, 1, 1), failure_window_days=730
            ),
            cache=cache,
        )
        cand = _candidate(
            indication="Alzheimer disease",
            mesh_indication="Alzheimer Disease",
            phase=TrialPhase.PHASE_2,
            latest=date(2020, 1, 1),  # stale
        )
        out = stage.run(CandidateTable(candidates=[cand]))
        rec = out.outcomes["C1"]
        assert rec.outcome == CandidateOutcome.FAILED_PHASE_2
        assert llm.calls == 1

    def test_llm_error_yields_unknown_and_does_not_persist(
        self, monkeypatch, cache
    ):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "pembrolizumab": ["metastatic NSCLC"],
            }),
        )
        stage = AdjudicationStage(
            llm_client=RaisingLLM(),
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        cand = _candidate()
        out = stage.run(CandidateTable(candidates=[cand]))
        rec = out.outcomes["C1"]
        assert rec.outcome == CandidateOutcome.UNKNOWN
        assert "error" in rec.evidence_sources
        # Not persisted -> retry is free on next run
        assert cache.get_ndc_outcome(
            KnowledgeCache.make_adjudication_key(
                cand.drug_name, cand.indication, cand.highest_phase.value
            ),
            "C1",
        ) is None

    def test_cache_hit_short_circuits_llm(self, monkeypatch, cache):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "pembrolizumab": ["metastatic NSCLC"],
            }),
        )
        llm = FakeLLM({
            "pembrolizumab": {
                "approved": True,
                "confidence": 0.9,
                "matched_indication": "metastatic NSCLC",
                "reasoning": "covered",
            }
        })
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        cand = _candidate()
        # First run populates cache
        stage.run(CandidateTable(candidates=[cand]))
        assert llm.calls == 1
        # Second run should short-circuit
        stage2 = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        out = stage2.run(CandidateTable(candidates=[cand]))
        assert llm.calls == 1  # unchanged
        assert out.outcomes["C1"].outcome == CandidateOutcome.APPROVED

    def test_dedup_via_drug_key_in_prepass(self, monkeypatch, cache):
        """Two candidates sharing a drug/indication should hit one bulk
        record. The LLM may be called once or twice depending on whether
        the prompts differ, but bulk lookup happens once."""
        bulk_calls = []

        def counting_bulk(drug_groups):
            bulk_calls.append(list(drug_groups))
            return fake_bulk_lookup({
                "pembrolizumab": ["metastatic NSCLC"],
            })(drug_groups)

        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk", counting_bulk
        )
        llm = FakeLLM({
            "pembrolizumab": {
                "approved": True,
                "confidence": 0.9,
                "matched_indication": "metastatic NSCLC",
                "reasoning": "covered",
            }
        })
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        c1 = _candidate(cid="C1", phase=TrialPhase.PHASE_2)
        c2 = _candidate(cid="C2", phase=TrialPhase.PHASE_3)
        stage.run(CandidateTable(candidates=[c1, c2]))
        # One bulk call, with one entry (unique drug)
        assert len(bulk_calls) == 1
        assert len(bulk_calls[0]) == 1

    def test_commercialized_toggle(self, monkeypatch, cache):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "pembrolizumab": ["metastatic NSCLC"],
            }),
        )
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drug",
            lambda name: (True, "20180601"),
        )
        monkeypatch.setattr(ndc_stage, "_INCLUDE_COMMERCIALIZED", True)

        llm = FakeLLM({
            "pembrolizumab": {
                "approved": True,
                "confidence": 0.9,
                "matched_indication": "metastatic NSCLC",
                "reasoning": "covered",
            }
        })
        stage = AdjudicationStage(
            llm_client=llm,
            config=AdjudicationConfig(as_of=date(2025, 1, 1)),
            cache=cache,
        )
        out = stage.run(CandidateTable(candidates=[_candidate()]))
        rec = out.outcomes["C1"]
        assert rec.outcome == CandidateOutcome.COMMERCIALIZED
        assert rec.commercialization_date == date(2018, 6, 1)


# ---------------------------------------------------------------------------
# Standalone API + helper tests
# ---------------------------------------------------------------------------


class TestSynonymsForCandidate:
    def test_drugbank_synonyms_first_then_candidate_fields(self):
        cand = _candidate(drug="pembrolizumab", drug_raw="Keytruda", mesh_drug="Pembrolizumab")
        forward = {"DB09037": ["pembrolizumab", "keytruda", "mk-3475"]}
        synonyms = synonyms_for_candidate(cand, forward)
        # First three are drugbank synonyms, in order
        assert synonyms[:3] == ["pembrolizumab", "keytruda", "mk-3475"]
        # Candidate fields appended (deduped vs drugbank set, case-insensitive)
        assert "Pembrolizumab" not in synonyms  # already covered by drugbank
        assert "Keytruda" not in synonyms  # already covered by drugbank

    def test_no_drugbank_id_falls_back_to_candidate_fields(self):
        cand = _candidate(drug="DrugX", drug_raw="DrugX HCl", mesh_drug="DrugX", drugbank_id=None)
        synonyms = synonyms_for_candidate(cand, {})
        assert synonyms == ["DrugX", "DrugX HCl"]


class TestResolveDrugKey:
    def test_uses_drugbank_id_when_present(self):
        cand = _candidate(drugbank_id="DB09037")
        assert resolve_drug_key(cand) == "DB09037"

    def test_falls_back_to_canonicalized_name(self):
        cand = _candidate(drug="Pembrolizumab (Keytruda)", drugbank_id=None)
        # canonicalize_drug_name strips parentheticals
        assert resolve_drug_key(cand) == "pembrolizumab"


class TestDedupIndications:
    def test_drops_exact_duplicates_after_normalization(self):
        indications = [
            "Treatment of metastatic NSCLC",
            "treatment of metastatic NSCLC",  # case variant
            "Treatment of metastatic NSCLC.",  # trailing period
            "  Treatment of metastatic NSCLC  ",  # whitespace
        ]
        out = dedup_indications(indications)
        assert out == ["Treatment of metastatic NSCLC"]

    def test_preserves_distinct_population_variants(self):
        indications = [
            "Treatment of X in adults",
            "Treatment of X in pediatric patients",
        ]
        assert dedup_indications(indications) == indications

    def test_preserves_order_first_occurrence_wins(self):
        indications = ["Indication A", "Indication B", "indication a"]
        assert dedup_indications(indications) == ["Indication A", "Indication B"]

    def test_drops_empty_strings(self):
        assert dedup_indications(["", "  ", "Real indication"]) == ["Real indication"]


class TestSplitLongIndications:
    def test_short_indications_pass_through(self):
        inds = ["Treatment of X", "Treatment of Y"]
        assert split_long_indications(inds, max_chars=1500) == inds

    def test_splits_on_spl_section_markers(self):
        big = (
            "Melanoma for the treatment of patients with unresectable melanoma. "
            "( 1.1 ) Non-Small Cell Lung Cancer in combination with chemotherapy "
            "as first-line treatment. ( 1.2 ) Head and Neck Squamous Cell "
            "Carcinoma for first-line treatment of recurrent disease. ( 1.3 )"
        ) * 5  # force length > max_chars
        out = split_long_indications([big], max_chars=200)
        assert len(out) > 1
        # Each chunk should be a real clinical indication, not a marker
        assert all("(" not in c[-5:] for c in out)
        assert any("melanoma" in c.lower() for c in out)
        assert any("non-small cell lung cancer" in c.lower() for c in out)

    def test_falls_back_to_sentence_split_when_no_markers(self):
        big = "First indication sentence. " * 100
        out = split_long_indications([big], max_chars=300)
        assert len(out) > 1
        assert all(len(c) <= 300 + 50 for c in out)  # +50 for trailing buffer

    def test_drops_tiny_fragments_below_min_chars(self):
        # Section markers around very short fragments
        text = "X. ( 1.1 ) Y. ( 1.2 ) " + "A real long indication sentence. " * 50
        out = split_long_indications([text], max_chars=500, min_chars=30)
        # "X." and "Y." should be dropped (below min_chars)
        assert all(len(c) >= 30 for c in out)


class TestTopRelevantIndications:
    def test_returns_all_when_under_k(self):
        inds = ["a", "b", "c"]
        assert top_relevant_indications("anything", None, inds, k=20) == inds

    def test_picks_most_similar_when_over_k(self):
        inds = [
            "Treatment of metastatic non-small cell lung cancer",
            "Treatment of melanoma",
            "Treatment of head and neck squamous cell carcinoma",
        ] + [f"Unrelated indication {i}" for i in range(50)]
        out = top_relevant_indications("metastatic NSCLC", None, inds, k=2)
        assert len(out) == 2
        assert "non-small cell lung cancer" in out[0].lower()

    def test_uses_mesh_indication_as_fallback(self):
        inds = (
            ["Treatment of melanoma"]
            + [f"Unrelated {i}" for i in range(30)]
            + ["Treatment of metastatic non-small cell lung cancer (NSCLC)"]
        )
        # Trial indication is generic; MeSH is specific
        out = top_relevant_indications(
            "lung cancer",
            "Carcinoma, Non-Small-Cell Lung",
            inds,
            k=2,
        )
        joined = " ".join(out).lower()
        assert "non-small cell lung cancer" in joined


class TestStandaloneAdjudicate:
    def test_drug_missing_returns_not_approved_no_llm_call(self, monkeypatch):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({}),
        )
        llm = FakeLLM({})
        adj = NDCAdjudicator(llm_client=llm)
        verdict = adj.adjudicate(
            drug_synonyms=["unknownium", "FakeDrug"],
            indication="anything",
        )
        assert verdict.approved is False
        assert "No FDA label indications" in verdict.reasoning
        assert llm.calls == 0

    def test_approved_via_llm(self, monkeypatch):
        monkeypatch.setattr(
            "utils.ndc_lookup.get_drugs_with_indications_bulk",
            fake_bulk_lookup({
                "Keytruda": ["metastatic non-small cell lung cancer"],
            }),
        )
        llm = FakeLLM({
            "Keytruda": {
                "approved": True,
                "confidence": 0.95,
                "matched_indication": "metastatic non-small cell lung cancer",
                "reasoning": "exact coverage",
            }
        })
        adj = NDCAdjudicator(llm_client=llm)
        verdict = adj.adjudicate(
            drug_synonyms=["pembrolizumab", "Keytruda"],
            indication="metastatic NSCLC",
        )
        assert verdict.approved is True
        assert verdict.matched_synonym == "Keytruda"
        assert verdict.matched_indication == "metastatic non-small cell lung cancer"

    def test_empty_synonym_list_returns_not_approved(self):
        adj = NDCAdjudicator(llm_client=FakeLLM({}))
        verdict = adj.adjudicate(drug_synonyms=[], indication="anything")
        assert verdict.approved is False
        assert verdict.matched_synonym is None

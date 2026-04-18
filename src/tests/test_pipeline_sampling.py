"""
Tests for post-clustering random candidate sampling in Pipeline._sample_candidates.

Sampling happens after clustering and year-range filtering so each sampled
candidate retains its complete trial set. Determinism is driven by
`PipelineConfig.sample_seed` fed into a fresh `random.Random` instance.
"""

from datetime import date
from unittest.mock import patch

from pipeline.models import (
    Candidate,
    CandidateTable,
    RawTrial,
    TrialPhase,
    TrialStatus,
    TrialTable,
)
from pipeline.pipeline import Pipeline, PipelineConfig


def _make_candidate(i: int, trial_ids: list[str]) -> Candidate:
    return Candidate(
        candidate_id=f"cand_{i:04d}",
        drug_name=f"Drug{i}",
        indication=f"Indication{i}",
        trial_ids=trial_ids,
        highest_phase=TrialPhase.PHASE_2,
        sponsors=[f"Sponsor{i}"],
        earliest_start_date=date(2018, 1, 1),
        latest_completion_date=date(2021, 6, 30),
    )


def _make_trial(nct_id: str) -> RawTrial:
    return RawTrial(
        nct_id=nct_id,
        title=f"Study {nct_id}",
        intervention="X",
        indication="Y",
        sponsor="S",
        phase=TrialPhase.PHASE_2,
        status=TrialStatus.COMPLETED,
        start_date=date(2019, 1, 1),
        completion_date=date(2021, 1, 1),
    )


def _make_corpus(n_candidates: int = 50, trials_per_candidate: int = 3):
    candidates = []
    trials = []
    for i in range(n_candidates):
        nct_ids = [f"NCT{i:04d}{j:02d}" for j in range(trials_per_candidate)]
        candidates.append(_make_candidate(i, nct_ids))
        trials.extend(_make_trial(nct) for nct in nct_ids)
    return CandidateTable(candidates=candidates), TrialTable(trials=trials)


def _build_pipeline(**overrides) -> Pipeline:
    """Construct a Pipeline without executing _build_stages (which imports DBs)."""
    cfg = PipelineConfig(**overrides)
    p = Pipeline.__new__(Pipeline)  # bypass __init__ / _build_stages
    p.config = cfg
    return p


class TestSampleCandidates:
    def test_bypass_when_max_candidates_none(self):
        cands, trials = _make_corpus(n_candidates=10)
        p = _build_pipeline(max_candidates=None)

        out_cands, out_trials = p._sample_candidates(cands, trials)

        assert out_cands is cands
        assert out_trials is trials

    def test_bypass_when_population_le_sample_size(self):
        cands, trials = _make_corpus(n_candidates=5)
        p = _build_pipeline(max_candidates=10, sample_seed=42)

        out_cands, out_trials = p._sample_candidates(cands, trials)

        assert out_cands is cands
        assert out_trials is trials

    def test_sample_size_matches_max_candidates(self):
        cands, trials = _make_corpus(n_candidates=50)
        p = _build_pipeline(max_candidates=10, sample_seed=42)

        out_cands, _ = p._sample_candidates(cands, trials)

        assert len(out_cands.candidates) == 10

    def test_same_seed_produces_same_sample(self):
        cands, trials = _make_corpus(n_candidates=50)
        p = _build_pipeline(max_candidates=10, sample_seed=42)

        out1, _ = p._sample_candidates(cands, trials)
        out2, _ = p._sample_candidates(cands, trials)

        assert [c.candidate_id for c in out1.candidates] == [
            c.candidate_id for c in out2.candidates
        ]

    def test_different_seed_produces_different_sample(self):
        cands, trials = _make_corpus(n_candidates=50)
        p1 = _build_pipeline(max_candidates=10, sample_seed=42)
        p2 = _build_pipeline(max_candidates=10, sample_seed=7)

        out1, _ = p1._sample_candidates(cands, trials)
        out2, _ = p2._sample_candidates(cands, trials)

        ids1 = {c.candidate_id for c in out1.candidates}
        ids2 = {c.candidate_id for c in out2.candidates}
        assert ids1 != ids2

    def test_pruned_trials_exactly_match_sampled_candidates(self):
        cands, trials = _make_corpus(n_candidates=50, trials_per_candidate=3)
        p = _build_pipeline(max_candidates=10, sample_seed=42)

        out_cands, out_trials = p._sample_candidates(cands, trials)

        expected_ncts = {nct for c in out_cands.candidates for nct in c.trial_ids}
        actual_ncts = {t.nct_id for t in out_trials.trials}
        assert actual_ncts == expected_ncts
        assert len(out_trials.trials) == 10 * 3

    def test_sampling_isolated_from_global_rng(self):
        """_sample_candidates must not perturb the module-level random state."""
        import random as _random

        cands, trials = _make_corpus(n_candidates=50)
        p = _build_pipeline(max_candidates=10, sample_seed=42)

        _random.seed(123)
        before = _random.random()

        _random.seed(123)
        p._sample_candidates(cands, trials)
        after = _random.random()

        assert before == after


class TestFilterToCachedCandidates:
    """Pipeline._filter_to_cached_candidates drops candidates not present in
    the adjudication cache; behaves as a no-op when the flag is off."""

    def _make_cache(self, tmp_path, cached_candidates, method="llm_direct"):
        from pipeline.knowledge_cache import KnowledgeCache
        from pipeline.models import CandidateOutcome, CandidateOutcomeRecord

        cache = KnowledgeCache(str(tmp_path / "kc.db"))
        for cand in cached_candidates:
            key = KnowledgeCache.make_adjudication_key(
                cand.drug_name, cand.indication, cand.highest_phase.value,
            )
            record = CandidateOutcomeRecord(
                candidate_id=cand.candidate_id,
                outcome=CandidateOutcome.APPROVED,
                confidence=0.9,
            )
            if method == "fda_timeline":
                cache.put_fda_outcome(key, record)
            else:
                cache.put_outcome(key, record)
        return cache

    def test_bypass_when_flag_disabled(self, tmp_path):
        cands, trials = _make_corpus(n_candidates=3)
        p = _build_pipeline(drop_uncached_candidates=False)
        p._cache = self._make_cache(tmp_path, [])  # empty cache

        out_cands, out_trials = p._filter_to_cached_candidates(cands, trials)

        assert out_cands is cands
        assert out_trials is trials

    def test_bypass_when_no_cache_configured(self, tmp_path):
        cands, trials = _make_corpus(n_candidates=3)
        p = _build_pipeline(drop_uncached_candidates=True, cache_path=None)
        p._cache = None

        out_cands, out_trials = p._filter_to_cached_candidates(cands, trials)

        assert out_cands is cands
        assert out_trials is trials

    def test_keeps_cached_drops_uncached(self, tmp_path):
        cands, trials = _make_corpus(n_candidates=5)
        # Cache only the first two candidates.
        cached_subset = cands.candidates[:2]
        p = _build_pipeline(
            drop_uncached_candidates=True,
            adjudication_method="llm_direct",
        )
        p._cache = self._make_cache(tmp_path, cached_subset, method="llm_direct")

        out_cands, _ = p._filter_to_cached_candidates(cands, trials)

        kept_ids = {c.candidate_id for c in out_cands.candidates}
        assert kept_ids == {c.candidate_id for c in cached_subset}

    def test_prunes_trials_to_surviving_candidates(self, tmp_path):
        cands, trials = _make_corpus(n_candidates=5, trials_per_candidate=3)
        cached_subset = cands.candidates[:2]
        p = _build_pipeline(
            drop_uncached_candidates=True,
            adjudication_method="llm_direct",
        )
        p._cache = self._make_cache(tmp_path, cached_subset, method="llm_direct")

        out_cands, out_trials = p._filter_to_cached_candidates(cands, trials)

        expected_ncts = {nct for c in out_cands.candidates for nct in c.trial_ids}
        actual_ncts = {t.nct_id for t in out_trials.trials}
        assert actual_ncts == expected_ncts
        assert len(out_trials.trials) == 2 * 3

    def test_uses_fda_cache_when_method_is_fda_timeline(self, tmp_path):
        """With adjudication_method='fda_timeline', lookups hit fda_adjudication_cache,
        not adjudication_cache — putting a record in the llm_direct cache must
        not save a candidate under fda_timeline mode."""
        cands, trials = _make_corpus(n_candidates=3)
        # Populate the llm_direct cache for cand 0 only; fda cache for cand 1.
        from pipeline.knowledge_cache import KnowledgeCache
        from pipeline.models import CandidateOutcome, CandidateOutcomeRecord

        cache = KnowledgeCache(str(tmp_path / "kc.db"))
        for i, key_method in [(0, "llm_direct"), (1, "fda_timeline")]:
            c = cands.candidates[i]
            key = KnowledgeCache.make_adjudication_key(
                c.drug_name, c.indication, c.highest_phase.value,
            )
            rec = CandidateOutcomeRecord(
                candidate_id=c.candidate_id,
                outcome=CandidateOutcome.APPROVED,
                confidence=0.9,
            )
            if key_method == "fda_timeline":
                cache.put_fda_outcome(key, rec)
            else:
                cache.put_outcome(key, rec)

        p = _build_pipeline(
            drop_uncached_candidates=True,
            adjudication_method="fda_timeline",
        )
        p._cache = cache

        out_cands, _ = p._filter_to_cached_candidates(cands, trials)

        kept_ids = {c.candidate_id for c in out_cands.candidates}
        assert kept_ids == {cands.candidates[1].candidate_id}

    def test_all_dropped_yields_empty_tables(self, tmp_path):
        cands, trials = _make_corpus(n_candidates=3)
        p = _build_pipeline(
            drop_uncached_candidates=True,
            adjudication_method="llm_direct",
        )
        p._cache = self._make_cache(tmp_path, [], method="llm_direct")  # nothing cached

        out_cands, out_trials = p._filter_to_cached_candidates(cands, trials)

        assert out_cands.candidates == []
        assert out_trials.trials == []

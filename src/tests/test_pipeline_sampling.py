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

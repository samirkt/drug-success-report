"""Tests for the FDA-timeline adjudication stage.

Network calls are mocked: both the FDAClient (openFDA/DailyMed/NDC) and
the LLMClient (indication extract + match) are replaced with fakes that
return deterministic fixtures. One integration test is gated behind
RUN_OPENFDA_INTEGRATION=1 for local verification against live openFDA.
"""

from __future__ import annotations

import os
from datetime import date

import pytest

from pipeline.fda.fda_client import (
    Application,
    FDAClient,
    NDCRecord,
    Submission,
)
from pipeline.fda.commercial import CommercialStatusChecker
from pipeline.fda.llm_adjudicator import IndicationAdjudicator
from pipeline.fda.timeline import TimelineBuilder
from pipeline.knowledge_cache import KnowledgeCache
from pipeline.models import (
    Candidate,
    CandidateOutcome,
    CandidateTable,
    TrialPhase,
)
from pipeline.stages.adjudication_fda import (
    AdjudicationConfig,
    AdjudicationStage,
    _phase_to_int,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeLLM:
    """Routes based on the system prompt markers used by llm_adjudicator."""

    def complete_json(self, system, user, schema):
        if "deciding whether" in system.lower():
            trial_line = ""
            for line in user.splitlines():
                if line.strip().lower().startswith("trial indication:"):
                    trial_line = line.lower()
                    break
            if "non-small cell lung cancer" in trial_line:
                return {
                    "verdict": "APPROVED",
                    "reasoning": "Trial NSCLC falls within approved NSCLC indication.",
                    "matched_indication_index": 0,
                }
            if "alzheimer" in trial_line:
                return {
                    "verdict": "NOT_APPROVED",
                    "reasoning": "No approved indication for Alzheimer's disease.",
                    "matched_indication_index": None,
                }
            return {
                "verdict": "UNCERTAIN",
                "reasoning": "Unrecognized indication in test fixture.",
                "matched_indication_index": None,
            }

        if "source type: approval_letter" in user.lower():
            return {
                "indications": [
                    {
                        "indication_text": "metastatic non-small cell lung cancer",
                        "population_restriction": "PD-L1 >= 1%, adults",
                        "combination_partners": [],
                        "source_excerpt": "indicated for patients with metastatic NSCLC",
                    }
                ]
            }
        return {
            "indications": [
                {
                    "indication_text": "metastatic non-small cell lung cancer",
                    "population_restriction": "PD-L1 >= 1%",
                    "combination_partners": [],
                    "source_excerpt": "NSCLC indicated",
                },
                {
                    "indication_text": "metastatic melanoma",
                    "population_restriction": None,
                    "combination_partners": [],
                    "source_excerpt": "metastatic melanoma indicated",
                },
            ]
        }


class FakeFDA(FDAClient):
    """Bypasses parent __init__ so no httpx client or disk cache is created."""

    def __init__(self, currently_marketed: bool = True):
        self._approved_app = Application(
            application_number="BLA125514",
            sponsor_name="Test Pharma",
            active_ingredients=["PEMBROLIZUMAB"],
            brand_name="Keytruda",
            submissions=[
                Submission(
                    application_number="BLA125514",
                    submission_type="ORIG",
                    submission_number=1,
                    submission_class_code="TYPE 1",
                    submission_class_description="Type 1 - New Molecular Entity",
                    submission_status="AP",
                    submission_status_date=date(2014, 9, 4),
                    approval_letter_url=None,
                ),
                Submission(
                    application_number="BLA125514",
                    submission_type="SUPPL",
                    submission_number=20,
                    submission_class_code="EFFICACY",
                    submission_class_description="Efficacy-New Indication",
                    submission_status="AP",
                    submission_status_date=date(2016, 10, 24),
                    approval_letter_url="https://example.com/letter.pdf",
                ),
            ],
        )
        self._currently_marketed = currently_marketed

    def find_applications_by_drug(self, drug_name: str):
        if "pembrolizumab" in drug_name.lower():
            return [self._approved_app]
        return []

    def get_current_label_text(self, application_number: str):
        return "INDICATIONS AND USAGE: Keytruda is indicated for NSCLC and metastatic melanoma."

    def get_approval_letter_pdf(self, submission):
        return b"%PDF-FAKE"

    def get_ndc_records(self, application_number: str):
        start = date(2014, 9, 15)
        end = None if self._currently_marketed else date(2020, 1, 1)
        return [
            NDCRecord(
                ndc="0006-3026",
                application_number=application_number,
                marketing_start_date=start,
                marketing_end_date=end,
                marketing_category="BLA",
                labeler_name="Test Pharma",
            )
        ]


class RaisingFDA(FDAClient):
    """Triggers the exception path in AdjudicationStage.run()."""

    def __init__(self):
        pass

    def find_applications_by_drug(self, drug_name: str):
        raise RuntimeError("simulated openFDA outage")


def _fake_pdf_extractor(pdf_bytes: bytes) -> str:
    return "approval_letter: indicated for metastatic NSCLC with PD-L1 expression"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_stage(fda, llm, cache=None):
    adjudicator = IndicationAdjudicator(llm)
    timeline_builder = TimelineBuilder(fda, adjudicator, pdf_extractor=_fake_pdf_extractor)
    commercial = CommercialStatusChecker(fda)
    stage = AdjudicationStage(fda_client=fda, llm_client=llm, cache=cache)
    stage.timeline_builder = timeline_builder
    stage.commercial = commercial
    return stage


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestPhaseToInt:
    @pytest.mark.parametrize(
        "phase,expected",
        [
            (TrialPhase.PHASE_1, 1),
            (TrialPhase.PHASE_2, 2),
            (TrialPhase.PHASE_3, 3),
            (TrialPhase.PHASE_4, 4),
            (TrialPhase.NOT_APPLICABLE, 0),
            (TrialPhase.UNKNOWN, 0),
        ],
    )
    def test_phase_to_int(self, phase, expected):
        assert _phase_to_int(phase) == expected


# ---------------------------------------------------------------------------
# End-to-end adjudication outcomes (with fakes)
# ---------------------------------------------------------------------------


class TestAdjudicationStageRun:
    def test_commercialized_path(self):
        stage = _make_stage(FakeFDA(currently_marketed=True), FakeLLM())
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c1",
                drug_name="pembrolizumab",
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=["NCT01295827"],
            ),
        ])

        outcomes = stage.run(candidates)
        record = outcomes.outcomes["c1"]

        assert record.outcome == CandidateOutcome.COMMERCIALIZED
        assert record.approval_date == date(2014, 9, 4)
        assert record.commercialization_date == date(2014, 9, 15)
        assert record.evidence_sources  # at least one entry
        assert record.reasoning  # non-empty

    def test_failed_phase_2_stale(self):
        stage = _make_stage(FakeFDA(), FakeLLM())
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c2",
                drug_name="pembrolizumab",
                indication="Alzheimer disease",
                mesh_indication="Alzheimer Disease",
                highest_phase=TrialPhase.PHASE_2,
                trial_ids=["NCT99999999"],
                latest_completion_date=date(2018, 1, 1),
            ),
        ])

        outcomes = stage.run(candidates)
        record = outcomes.outcomes["c2"]

        assert record.outcome == CandidateOutcome.FAILED_PHASE_2
        assert record.approval_date is None

    def test_failed_phase_3_unknown_drug(self):
        stage = _make_stage(FakeFDA(), FakeLLM())
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c3",
                drug_name="madeupdrug-xyz",
                indication="Obesity",
                mesh_indication="Obesity",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=["NCT88888888"],
                latest_completion_date=date(2015, 1, 1),
            ),
        ])

        outcomes = stage.run(candidates)
        record = outcomes.outcomes["c3"]

        assert record.outcome == CandidateOutcome.FAILED_PHASE_3

    def test_ongoing_when_activity_recent(self):
        stage = _make_stage(FakeFDA(), FakeLLM())
        recent = date.today().replace(year=date.today().year - 1)
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c4",
                drug_name="madeupdrug-xyz",
                indication="Obesity",
                mesh_indication="Obesity",
                highest_phase=TrialPhase.PHASE_2,
                trial_ids=["NCT77777777"],
                latest_completion_date=recent,
            ),
        ])

        outcomes = stage.run(candidates)
        record = outcomes.outcomes["c4"]

        assert record.outcome == CandidateOutcome.ONGOING

    def test_unknown_on_fda_error(self):
        stage = _make_stage(RaisingFDA(), FakeLLM())
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c5",
                drug_name="anything",
                indication="Anything",
                mesh_indication=None,
                highest_phase=TrialPhase.PHASE_2,
                trial_ids=["NCT66666666"],
                latest_completion_date=date(2010, 1, 1),
            ),
        ])

        outcomes = stage.run(candidates)
        record = outcomes.outcomes["c5"]

        assert record.outcome == CandidateOutcome.UNKNOWN
        assert record.confidence == 0.0

    def test_timeline_cache_hits_once_per_drug(self):
        fda = FakeFDA()
        stage = _make_stage(fda, FakeLLM())
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="c1",
                drug_name="pembrolizumab",
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=["NCT1"],
            ),
            Candidate(
                candidate_id="c2",
                drug_name="pembrolizumab",
                indication="Alzheimer disease",
                mesh_indication="Alzheimer Disease",
                highest_phase=TrialPhase.PHASE_2,
                trial_ids=["NCT2"],
                latest_completion_date=date(2018, 1, 1),
            ),
        ])

        outcomes = stage.run(candidates)

        assert len(outcomes.outcomes) == 2
        assert len(stage._timeline_cache) == 1  # single drug, single timeline build


# ---------------------------------------------------------------------------
# Cache integration: persist outcomes to fda_adjudication_cache so the
# candidate summary can compare across methods.
# ---------------------------------------------------------------------------


class TestFDAAdjudicationCache:
    def test_run_writes_to_fda_adjudication_cache(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "kc.db")
        stage = _make_stage(FakeFDA(), FakeLLM(), cache=cache)
        cand = Candidate(
            candidate_id="c1",
            drug_name="pembrolizumab",
            indication="Non-small cell lung cancer",
            mesh_indication="Carcinoma, Non-Small-Cell Lung",
            highest_phase=TrialPhase.PHASE_3,
            trial_ids=["NCT1"],
        )
        stage.run(CandidateTable(candidates=[cand]))

        key = KnowledgeCache.make_adjudication_key(
            cand.drug_name, cand.indication, cand.highest_phase.value
        )
        cached = cache.get_fda_outcome(key, cand.candidate_id)
        assert cached is not None
        assert cached.outcome == CandidateOutcome.COMMERCIALIZED
        assert cached.approval_date == date(2014, 9, 4)
        assert cached.commercialization_date == date(2014, 9, 15)

    def test_cache_write_is_noop_when_cache_is_none(self):
        # Smoke: just verify no exception when cache is None on the error path.
        stage = _make_stage(RaisingFDA(), FakeLLM(), cache=None)
        cand = Candidate(
            candidate_id="c1",
            drug_name="x",
            indication="y",
            mesh_indication=None,
            highest_phase=TrialPhase.PHASE_2,
            trial_ids=["NCT1"],
            latest_completion_date=date(2010, 1, 1),
        )
        outcomes = stage.run(CandidateTable(candidates=[cand]))
        assert outcomes.outcomes["c1"].outcome == CandidateOutcome.UNKNOWN


# ---------------------------------------------------------------------------
# Candidate-level parallelism
# ---------------------------------------------------------------------------


class _CountingTimelineBuilder:
    """Wraps a real TimelineBuilder so we can count `build()` calls per drug.

    Sleeps briefly to widen the race window for the per-drug-lock test.
    """

    def __init__(self, real_builder, sleep_seconds: float = 0.05):
        self._real = real_builder
        self._sleep = sleep_seconds
        import threading
        self.calls_per_drug: dict[str, int] = {}
        self._lock = threading.Lock()

    def build(self, drug_name: str):
        import time as _time
        with self._lock:
            self.calls_per_drug[drug_name] = self.calls_per_drug.get(drug_name, 0) + 1
        _time.sleep(self._sleep)  # widen race window
        return self._real.build(drug_name)


class TestAdjudicationStageParallelism:
    """Confirm workers > 1 yields correct outcomes and per-drug timeline dedup."""

    def _make_parallel_stage(self, fda, llm, workers, cache=None):
        adjudicator = IndicationAdjudicator(llm)
        timeline_builder = TimelineBuilder(fda, adjudicator, pdf_extractor=_fake_pdf_extractor)
        commercial = CommercialStatusChecker(fda)
        stage = AdjudicationStage(
            fda_client=fda, llm_client=llm, cache=cache, workers=workers,
        )
        stage.timeline_builder = timeline_builder
        stage.commercial = commercial
        return stage

    def test_parallel_run_processes_all_candidates(self):
        stage = self._make_parallel_stage(FakeFDA(), FakeLLM(), workers=4)
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id=f"c{i}",
                drug_name="pembrolizumab",
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=[f"NCT{i:08d}"],
            )
            for i in range(8)
        ])

        outcomes = stage.run(candidates)

        assert len(outcomes.outcomes) == 8
        for cid in [f"c{i}" for i in range(8)]:
            assert outcomes.outcomes[cid].outcome == CandidateOutcome.COMMERCIALIZED

    def test_parallel_run_dedups_timeline_per_drug(self):
        """12 candidates across 3 distinct drugs with 4 workers → exactly
        3 timeline builds (one per drug), even though threads race."""
        # Use an FDA fake that responds to multiple drug names
        class MultiDrugFDA(FakeFDA):
            def find_applications_by_drug(self, drug_name: str):
                lower = drug_name.lower()
                if lower in {"pembrolizumab", "nivolumab", "atezolizumab"}:
                    return [self._approved_app]
                return []

        fda = MultiDrugFDA()
        adjudicator = IndicationAdjudicator(FakeLLM())
        real_builder = TimelineBuilder(fda, adjudicator, pdf_extractor=_fake_pdf_extractor)
        counting_builder = _CountingTimelineBuilder(real_builder, sleep_seconds=0.05)

        stage = AdjudicationStage(
            fda_client=fda, llm_client=FakeLLM(), cache=None, workers=4,
        )
        stage.timeline_builder = counting_builder
        stage.commercial = CommercialStatusChecker(fda)

        drugs = ["pembrolizumab", "nivolumab", "atezolizumab"]
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id=f"c{i}",
                drug_name=drugs[i % 3],
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=[f"NCT{i:08d}"],
            )
            for i in range(12)
        ])

        outcomes = stage.run(candidates)

        assert len(outcomes.outcomes) == 12
        # Per-drug lock guarantees each unique drug's timeline is built exactly once
        assert counting_builder.calls_per_drug == {
            "pembrolizumab": 1, "nivolumab": 1, "atezolizumab": 1,
        }

    def test_parallel_run_isolates_failing_candidate(self):
        """A single raising candidate must not poison the whole pool."""
        class SometimesRaisingFDA(FakeFDA):
            def find_applications_by_drug(self, drug_name: str):
                if drug_name.lower() == "raising_drug":
                    raise RuntimeError("simulated outage for one drug")
                return super().find_applications_by_drug(drug_name)

        stage = self._make_parallel_stage(SometimesRaisingFDA(), FakeLLM(), workers=4)
        candidates = CandidateTable(candidates=[
            Candidate(
                candidate_id="ok1",
                drug_name="pembrolizumab",
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=["NCT1"],
            ),
            Candidate(
                candidate_id="bad",
                drug_name="raising_drug",
                indication="Anything",
                mesh_indication=None,
                highest_phase=TrialPhase.PHASE_2,
                trial_ids=["NCT2"],
                latest_completion_date=date(2010, 1, 1),
            ),
            Candidate(
                candidate_id="ok2",
                drug_name="pembrolizumab",
                indication="Non-small cell lung cancer",
                mesh_indication="Carcinoma, Non-Small-Cell Lung",
                highest_phase=TrialPhase.PHASE_3,
                trial_ids=["NCT3"],
            ),
        ])

        outcomes = stage.run(candidates)

        assert outcomes.outcomes["ok1"].outcome == CandidateOutcome.COMMERCIALIZED
        assert outcomes.outcomes["bad"].outcome == CandidateOutcome.UNKNOWN
        assert outcomes.outcomes["ok2"].outcome == CandidateOutcome.COMMERCIALIZED

    def test_workers_clamped_to_minimum_of_one(self):
        """workers=0 or negative is silently clamped to 1 (sequential)."""
        stage = self._make_parallel_stage(FakeFDA(), FakeLLM(), workers=0)
        assert stage.workers == 1


# ---------------------------------------------------------------------------
# Live integration (gated)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.getenv("RUN_OPENFDA_INTEGRATION") != "1",
    reason="Set RUN_OPENFDA_INTEGRATION=1 to hit live openFDA.",
)
def test_live_openfda_returns_pembrolizumab_applications(tmp_path):
    client = FDAClient(cache_dir=tmp_path / "fda_cache")
    apps = client.find_applications_by_drug("pembrolizumab")
    assert apps, "expected at least one application for pembrolizumab from openFDA"

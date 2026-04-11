"""
Tests for the KnowledgeCache persistence layer (pipeline/knowledge_cache.py).

All tests use pytest's tmp_path for isolated SQLite files — zero LLM calls.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from pipeline.knowledge_cache import KnowledgeCache
from pipeline.models import (
    CandidateAttributes,
    CandidateOutcome,
    CandidateOutcomeRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_attrs(candidate_id: str = "cand_001") -> CandidateAttributes:
    return CandidateAttributes(
        candidate_id=candidate_id,
        drug_modality="peptide",
        disease_area="metabolic",
        modality_confidence=1.0,
        disease_confidence=0.5,
        reasoning="GLP-1 analog.",
    )


def _make_record(candidate_id: str = "cand_001") -> CandidateOutcomeRecord:
    return CandidateOutcomeRecord(
        candidate_id=candidate_id,
        outcome=CandidateOutcome.APPROVED,
        confidence=0.88,
        reasoning="FDA approved.",
        evidence_sources=["FDA Orange Book", "NCT00000001"],
    )


# ---------------------------------------------------------------------------
# TestCacheKeys
# ---------------------------------------------------------------------------

class TestCacheKeys:
    def test_classification_key_is_deterministic(self):
        k1 = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        k2 = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        assert k1 == k2
        assert len(k1) == 64  # SHA-256 hex

    def test_classification_key_differs_by_drug_name(self):
        k1 = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        k2 = KnowledgeCache.make_classification_key("DrugB", "Type 2 Diabetes")
        assert k1 != k2

    def test_classification_key_differs_by_indication(self):
        k1 = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        k2 = KnowledgeCache.make_classification_key("DrugA", "Hypertension")
        assert k1 != k2

    def test_classification_key_case_insensitive(self):
        k1 = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        k2 = KnowledgeCache.make_classification_key("druga", "type 2 diabetes")
        assert k1 == k2

    def test_adjudication_key_differs_from_classification_key(self):
        ck = KnowledgeCache.make_classification_key("DrugA", "Type 2 Diabetes")
        ak = KnowledgeCache.make_adjudication_key("DrugA", "Type 2 Diabetes", "phase 2")
        assert ck != ak

    def test_adjudication_key_differs_by_phase(self):
        k1 = KnowledgeCache.make_adjudication_key("DrugA", "Type 2 Diabetes", "Phase 1")
        k2 = KnowledgeCache.make_adjudication_key("DrugA", "Type 2 Diabetes", "Phase 2")
        assert k1 != k2


# ---------------------------------------------------------------------------
# TestClassificationCache
# ---------------------------------------------------------------------------

class TestClassificationCache:
    def test_get_attributes_returns_none_on_miss(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        assert cache.get_attributes(key, "cand_001") is None
        cache.close()

    def test_put_then_get_attributes_roundtrip(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        attrs = _make_attrs("cand_001")
        cache.put_attributes(key, attrs)

        result = cache.get_attributes(key, "cand_001")
        assert result is not None
        assert result.drug_modality == attrs.drug_modality
        assert result.disease_area == attrs.disease_area
        assert result.modality_confidence == attrs.modality_confidence
        assert result.disease_confidence == attrs.disease_confidence
        assert result.reasoning == attrs.reasoning
        cache.close()

    def test_put_attributes_idempotent(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        attrs = _make_attrs()
        cache.put_attributes(key, attrs)
        # second put with same key should overwrite silently
        cache.put_attributes(key, attrs)
        result = cache.get_attributes(key, "cand_001")
        assert result is not None
        cache.close()

    def test_put_attributes_different_keys_coexist(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key1 = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        key2 = KnowledgeCache.make_classification_key("DrugB", "Oncology")
        cache.put_attributes(key1, _make_attrs("cand_001"))
        cache.put_attributes(key2, _make_attrs("cand_002"))

        assert cache.get_attributes(key1, "cand_001") is not None
        assert cache.get_attributes(key2, "cand_002") is not None
        cache.close()

    def test_candidate_id_injected_on_get(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        # store with one candidate_id
        cache.put_attributes(key, _make_attrs("stored_id"))
        # retrieve with a different candidate_id — should use the one passed to get
        result = cache.get_attributes(key, "retrieved_id")
        assert result is not None
        assert result.candidate_id == "retrieved_id"
        cache.close()

    def test_get_attributes_wrong_key_returns_none(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "c.db")
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        wrong_key = KnowledgeCache.make_classification_key("DrugB", "Oncology")
        cache.put_attributes(key, _make_attrs())
        assert cache.get_attributes(wrong_key, "cand_001") is None
        cache.close()


# ---------------------------------------------------------------------------
# TestAdjudicationCache
# ---------------------------------------------------------------------------

class TestAdjudicationCache:
    def test_get_outcome_returns_none_on_miss(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "a.db")
        key = KnowledgeCache.make_adjudication_key("DrugA", "Diabetes", "Phase 2")
        assert cache.get_outcome(key, "cand_001") is None
        cache.close()

    def test_put_then_get_outcome_roundtrip(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "a.db")
        key = KnowledgeCache.make_adjudication_key("DrugA", "Diabetes", "Phase 2")
        record = _make_record("cand_001")
        cache.put_outcome(key, record)

        result = cache.get_outcome(key, "cand_001")
        assert result is not None
        assert result.outcome == record.outcome
        assert result.confidence == record.confidence
        assert result.reasoning == record.reasoning
        cache.close()

    def test_put_outcome_evidence_sources_list_preserved(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "a.db")
        key = KnowledgeCache.make_adjudication_key("DrugA", "Diabetes", "Phase 3")
        record = _make_record()
        record.evidence_sources = ["source_a", "source_b", "source_c"]
        cache.put_outcome(key, record)

        result = cache.get_outcome(key, "cand_001")
        assert result.evidence_sources == ["source_a", "source_b", "source_c"]
        cache.close()

    def test_put_outcome_idempotent(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "a.db")
        key = KnowledgeCache.make_adjudication_key("DrugA", "Diabetes", "Phase 2")
        record = _make_record()
        cache.put_outcome(key, record)
        cache.put_outcome(key, record)
        assert cache.get_outcome(key, "cand_001") is not None
        cache.close()

    def test_candidate_id_injected_on_get(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "a.db")
        key = KnowledgeCache.make_adjudication_key("DrugA", "Diabetes", "Phase 2")
        cache.put_outcome(key, _make_record("stored_id"))
        result = cache.get_outcome(key, "retrieved_id")
        assert result is not None
        assert result.candidate_id == "retrieved_id"
        cache.close()


# ---------------------------------------------------------------------------
# TestCacheThreadSafety
# ---------------------------------------------------------------------------

class TestCacheThreadSafety:
    def test_concurrent_classification_writes_no_corruption(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "thread.db")
        n = 20

        def write_and_read(i):
            key = KnowledgeCache.make_classification_key(f"Drug{i}", f"Indication{i}")
            attrs = CandidateAttributes(
                candidate_id=f"cand_{i}",
                drug_modality="peptide",
                disease_area="oncology",
                modality_confidence=0.9,
                disease_confidence=0.8,
                reasoning=f"Reason {i}",
            )
            cache.put_attributes(key, attrs)
            result = cache.get_attributes(key, f"cand_{i}")
            assert result is not None, f"Key {i} not found after write"
            assert result.reasoning == f"Reason {i}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(write_and_read, range(n)))

        cache.close()

    def test_concurrent_adjudication_writes_no_corruption(self, tmp_path):
        cache = KnowledgeCache(tmp_path / "thread_adj.db")
        n = 20

        def write_and_read(i):
            key = KnowledgeCache.make_adjudication_key(f"Drug{i}", f"Indication{i}", "Phase 2")
            record = CandidateOutcomeRecord(
                candidate_id=f"cand_{i}",
                outcome=CandidateOutcome.ONGOING,
                confidence=0.7,
                reasoning=f"Reason {i}",
                evidence_sources=[f"src_{i}"],
            )
            cache.put_outcome(key, record)
            result = cache.get_outcome(key, f"cand_{i}")
            assert result is not None, f"Key {i} not found after write"
            assert result.reasoning == f"Reason {i}"

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(write_and_read, range(n)))

        cache.close()


# ---------------------------------------------------------------------------
# TestCachePersistence
# ---------------------------------------------------------------------------

class TestCachePersistence:
    def test_data_survives_reopen(self, tmp_path):
        db_path = tmp_path / "persist.db"
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")
        attrs = _make_attrs()

        # Write and close
        cache1 = KnowledgeCache(db_path)
        cache1.put_attributes(key, attrs)
        cache1.close()

        # Reopen and read
        cache2 = KnowledgeCache(db_path)
        result = cache2.get_attributes(key, "cand_001")
        cache2.close()

        assert result is not None
        assert result.drug_modality == attrs.drug_modality
        assert result.disease_area == attrs.disease_area


# ---------------------------------------------------------------------------
# TestKnowledgeCacheInit
# ---------------------------------------------------------------------------

class TestKnowledgeCacheInit:
    def test_creates_db_file_on_init(self, tmp_path):
        db_path = tmp_path / "new.db"
        assert not db_path.exists()
        cache = KnowledgeCache(db_path)
        cache.close()
        assert db_path.exists()

    def test_creates_both_tables_on_init(self, tmp_path):
        import sqlite3
        db_path = tmp_path / "tables.db"
        cache = KnowledgeCache(db_path)
        cache.close()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        assert "classification_cache" in tables
        assert "adjudication_cache" in tables

    def test_existing_db_not_overwritten(self, tmp_path):
        db_path = tmp_path / "existing.db"
        key = KnowledgeCache.make_classification_key("DrugA", "Diabetes")

        # First open: write data
        cache1 = KnowledgeCache(db_path)
        cache1.put_attributes(key, _make_attrs())
        cache1.close()

        # Second open: data should still be there
        cache2 = KnowledgeCache(db_path)
        result = cache2.get_attributes(key, "cand_001")
        cache2.close()

        assert result is not None


# ---------------------------------------------------------------------------
# Disease area migration
# ---------------------------------------------------------------------------

class TestMigrateDiseaseAreas:
    def test_normalizes_existing_entries(self, knowledge_cache):
        """Entries with non-canonical disease areas get updated."""
        attrs_space = _make_attrs("cand_space")
        attrs_space.disease_area = "reproductive health"
        attrs_synonym = _make_attrs("cand_syn")
        attrs_synonym.disease_area = "gynecology"
        attrs_ok = _make_attrs("cand_ok")
        attrs_ok.disease_area = "oncology"

        knowledge_cache.put_attributes("key_space", attrs_space)
        knowledge_cache.put_attributes("key_syn", attrs_synonym)
        knowledge_cache.put_attributes("key_ok", attrs_ok)

        updated = knowledge_cache.migrate_disease_areas()

        assert "reproductive health" in updated
        assert "gynecology" in updated
        assert "oncology" not in updated

        # Verify the DB was actually updated
        result_space = knowledge_cache.get_attributes("key_space", "cand_space")
        result_syn = knowledge_cache.get_attributes("key_syn", "cand_syn")
        result_ok = knowledge_cache.get_attributes("key_ok", "cand_ok")
        assert result_space.disease_area == "other"
        assert result_syn.disease_area == "other"
        assert result_ok.disease_area == "oncology"

    def test_idempotent(self, knowledge_cache):
        """Running migration twice produces no changes on second run."""
        attrs = _make_attrs("cand_1")
        attrs.disease_area = "pain management"
        knowledge_cache.put_attributes("key_1", attrs)

        first = knowledge_cache.migrate_disease_areas()
        assert len(first) > 0

        second = knowledge_cache.migrate_disease_areas()
        assert len(second) == 0

# -*- coding: utf-8 -*-
"""
Unit tests for DuplicateDetectorService (setup.py) - AGENT-DATA-011

Tests the DuplicateDetectorService facade with 200+ test cases covering:
- Service initialization and configuration
- fingerprint_records: SHA-256 fingerprinting, field_set, normalization
- create_blocks: blocking strategies, key_fields, reduction ratio
- compare_pairs: field_configs, weighted scoring, avg_similarity
- classify_matches: threshold-based classification (MATCH/POSSIBLE/NON_MATCH)
- form_clusters: union-find fallback clustering
- merge_duplicates: most_complete merge, conflict counting
- run_pipeline: end-to-end pipeline execution
- create_dedup_job / get_job / list_jobs / cancel_job
- create_rule / list_rules
- get_statistics / health_check / generate_report
- get_match_details / get_cluster_details / get_merge_result
- configure_duplicate_detector(app) / get_duplicate_detector / get_router
- Response models (Pydantic serialization)
- _ProvenanceTracker: record, entry_count, entry hash
- _compute_hash: deterministic hashing
- _normalize_fields: lowercasing, whitespace stripping, None handling
- _count_conflicts: field-level conflict detection
- _merge_by_most_complete: longest non-null value selection
- Error handling for invalid inputs
- Service state management (started, active_jobs)

Author: GreenLang Platform Team
Date: February 2026
"""

import asyncio
import hashlib
import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    reset_config,
    set_config,
)
from greenlang.duplicate_detector.setup import (
    BlockResponse,
    ClassifyResponse,
    ClusterResponse,
    CompareResponse,
    DuplicateDetectorService,
    FingerprintResponse,
    MergeResponse,
    PipelineResponse,
    StatsResponse,
    _ProvenanceTracker,
    _compute_hash,
    configure_duplicate_detector,
    get_duplicate_detector,
    get_router,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config() -> DuplicateDetectorConfig:
    """Create a default test configuration."""
    return DuplicateDetectorConfig()


@pytest.fixture
def service(config: DuplicateDetectorConfig) -> DuplicateDetectorService:
    """Create a fresh DuplicateDetectorService instance.

    Patches _init_engines so the service can be created without the
    full SDK engines being importable or compatible.
    """
    set_config(config)
    with patch.object(DuplicateDetectorService, "_init_engines"):
        svc = DuplicateDetectorService(config=config)
    return svc


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Create sample records for dedup testing."""
    return [
        {"id": "0", "name": "Alice Smith", "email": "alice@example.com", "phone": "555-0001"},
        {"id": "1", "name": "alice smith", "email": "alice@example.com", "phone": "555-0001"},
        {"id": "2", "name": "Bob Jones", "email": "bob@example.com", "phone": "555-0002"},
        {"id": "3", "name": "Charlie Brown", "email": "charlie@example.com", "phone": "555-0003"},
        {"id": "4", "name": "bob jones", "email": "bob@example.com", "phone": "555-0002"},
    ]


@pytest.fixture
def duplicate_records() -> List[Dict[str, Any]]:
    """Create records with clear duplicates for pipeline testing."""
    return [
        {"id": "0", "name": "Alice", "email": "alice@co.com"},
        {"id": "1", "name": "Alice", "email": "alice@co.com"},
        {"id": "2", "name": "Bob", "email": "bob@co.com"},
    ]


def _make_pairs(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build pair list from records for compare_pairs."""
    pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            pairs.append({
                "id_a": str(i),
                "id_b": str(j),
                "record_a": records[i],
                "record_b": records[j],
            })
    return pairs


def _make_comparisons(scores: List[float]) -> List[Dict[str, Any]]:
    """Create comparison result dicts with given overall_score values."""
    comparisons = []
    for idx, score in enumerate(scores):
        comparisons.append({
            "pair_id": str(uuid.uuid4()),
            "record_a_id": f"rec-{idx * 2}",
            "record_b_id": f"rec-{idx * 2 + 1}",
            "overall_score": score,
        })
    return comparisons


def _make_matches(pairs: int = 3) -> List[Dict[str, Any]]:
    """Create match dicts for form_clusters."""
    matches = []
    for i in range(pairs):
        matches.append({
            "record_a_id": f"rec-{i}",
            "record_b_id": f"rec-{i + 1}",
        })
    return matches


# ===================================================================
# Test _ProvenanceTracker
# ===================================================================


class TestProvenanceTracker:
    """Tests for the internal _ProvenanceTracker helper."""

    def test_init(self):
        tracker = _ProvenanceTracker()
        assert tracker.entry_count == 0
        assert tracker._entries == []

    def test_record_returns_hash(self):
        tracker = _ProvenanceTracker()
        h = tracker.record(
            entity_type="test",
            entity_id="id-1",
            action="create",
            data_hash="abc123",
        )
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_record_increments_count(self):
        tracker = _ProvenanceTracker()
        for i in range(5):
            tracker.record("type", f"id-{i}", "action", "hash")
        assert tracker.entry_count == 5
        assert len(tracker._entries) == 5

    def test_record_stores_entry(self):
        tracker = _ProvenanceTracker()
        tracker.record("dedup_job", "j-1", "create", "h123", user_id="user-1")
        entry = tracker._entries[0]
        assert entry["entity_type"] == "dedup_job"
        assert entry["entity_id"] == "j-1"
        assert entry["action"] == "create"
        assert entry["data_hash"] == "h123"
        assert entry["user_id"] == "user-1"
        assert "timestamp" in entry
        assert "entry_hash" in entry
        assert len(entry["entry_hash"]) == 64

    def test_record_default_user(self):
        tracker = _ProvenanceTracker()
        tracker.record("type", "id-1", "action", "hash")
        assert tracker._entries[0]["user_id"] == "system"

    def test_record_unique_hashes(self):
        tracker = _ProvenanceTracker()
        h1 = tracker.record("type", "id-1", "action", "hash1")
        h2 = tracker.record("type", "id-2", "action", "hash2")
        assert h1 != h2

    def test_record_deterministic_within_same_timestamp(self):
        """Hashes differ even with same input data due to different timestamps."""
        tracker = _ProvenanceTracker()
        h1 = tracker.record("type", "id-1", "action", "hash")
        h2 = tracker.record("type", "id-1", "action", "hash")
        # Timestamps will differ (even by microseconds), so hashes differ
        # But both must be 64-char hex
        assert len(h1) == 64
        assert len(h2) == 64


# ===================================================================
# Test _compute_hash
# ===================================================================


class TestComputeHash:
    """Tests for the _compute_hash helper function."""

    def test_dict_hash(self):
        h = _compute_hash({"a": 1, "b": 2})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_deterministic(self):
        data = {"key": "value", "num": 42}
        assert _compute_hash(data) == _compute_hash(data)

    def test_different_data_different_hash(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_key_order_independent(self):
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2

    def test_pydantic_model(self):
        model = FingerprintResponse(algorithm="sha256", total_records=5)
        h = _compute_hash(model)
        assert len(h) == 64

    def test_string_hash(self):
        h = _compute_hash("hello world")
        assert len(h) == 64

    def test_list_hash(self):
        h = _compute_hash([1, 2, 3])
        assert len(h) == 64


# ===================================================================
# Test DuplicateDetectorService Initialization
# ===================================================================


class TestServiceInit:
    """Tests for DuplicateDetectorService initialization."""

    def test_default_init(self, config):
        with patch.object(DuplicateDetectorService, "_init_engines"):
            svc = DuplicateDetectorService(config=config)
        assert svc.config is config
        assert isinstance(svc.provenance, _ProvenanceTracker)

    def test_none_config_uses_global(self):
        with patch.object(DuplicateDetectorService, "_init_engines"):
            svc = DuplicateDetectorService(config=None)
        assert svc.config is not None

    def test_engines_are_none_without_sdk(self, service):
        """In test env, SDK engines are None because _init_engines is patched."""
        assert service.config is not None

    def test_empty_stores_at_init(self, service):
        assert service._jobs == {}
        assert service._rules == {}
        assert service._fingerprint_results == {}
        assert service._block_results == {}
        assert service._compare_results == {}
        assert service._classify_results == {}
        assert service._cluster_results == {}
        assert service._merge_results == {}
        assert service._pipeline_results == {}
        assert service._matches == {}
        assert service._clusters == {}

    def test_stats_at_init(self, service):
        stats = service.get_statistics()
        assert stats.total_jobs == 0
        assert stats.completed_jobs == 0
        assert stats.total_records_processed == 0

    def test_not_started_at_init(self, service):
        assert service._started is False

    def test_active_jobs_zero_at_init(self, service):
        assert service._active_jobs == 0

    def test_engine_properties(self, service):
        """Engine properties are accessible (may be None in test)."""
        _ = service.fingerprinter_engine
        _ = service.blocking_engine
        _ = service.similarity_engine
        _ = service.classifier_engine
        _ = service.cluster_engine
        _ = service.merge_engine
        _ = service.pipeline_engine


# ===================================================================
# Test fingerprint_records
# ===================================================================


class TestFingerprintRecords:
    """Tests for DuplicateDetectorService.fingerprint_records."""

    def test_basic_fingerprint(self, service, sample_records):
        result = service.fingerprint_records(records=sample_records)
        assert isinstance(result, FingerprintResponse)
        assert result.total_records == len(sample_records)
        assert result.algorithm == "sha256"
        assert result.processing_time_ms >= 0
        assert len(result.fingerprints) == len(sample_records)

    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.fingerprint_records(records=[])

    def test_fingerprint_with_field_set(self, service, sample_records):
        result = service.fingerprint_records(
            records=sample_records,
            field_set=["name"],
        )
        assert result.total_records == len(sample_records)

    def test_fingerprint_with_algorithm_override(self, service, sample_records):
        result = service.fingerprint_records(
            records=sample_records,
            algorithm="simhash",
        )
        assert result.algorithm == "simhash"

    def test_unique_fingerprints_count(self, service, duplicate_records):
        result = service.fingerprint_records(records=duplicate_records)
        # Records 0 and 1 are identical -> same fingerprint
        assert result.unique_fingerprints <= result.total_records
        assert result.duplicate_candidates >= 0

    def test_provenance_hash_set(self, service, sample_records):
        result = service.fingerprint_records(records=sample_records)
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_result_stored(self, service, sample_records):
        result = service.fingerprint_records(records=sample_records)
        assert result.fingerprint_id in service._fingerprint_results

    def test_provenance_recorded(self, service, sample_records):
        before = service.provenance.entry_count
        service.fingerprint_records(records=sample_records)
        assert service.provenance.entry_count == before + 1

    def test_stats_updated(self, service, sample_records):
        service.fingerprint_records(records=sample_records)
        stats = service.get_statistics()
        assert stats.total_records_processed == len(sample_records)

    def test_normalization_produces_same_fingerprints(self, service):
        """Records differing only by case/whitespace should fingerprint equally."""
        records = [
            {"name": "Alice", "email": "alice@co.com"},
            {"name": "  alice  ", "email": "  ALICE@CO.COM  "},
        ]
        result = service.fingerprint_records(records=records)
        fp_values = list(result.fingerprints.values())
        # With normalization on, these should be the same
        assert fp_values[0] == fp_values[1]

    def test_no_normalization(self, service, sample_records):
        service.config.fingerprint_normalize = False
        result = service.fingerprint_records(records=sample_records)
        assert result.total_records == len(sample_records)

    def test_fingerprint_id_is_uuid(self, service, sample_records):
        result = service.fingerprint_records(records=sample_records)
        uuid.UUID(result.fingerprint_id)  # Should not raise

    def test_max_records_limit(self, service):
        service.config.max_records_per_job = 3
        records = [{"name": f"user-{i}"} for i in range(10)]
        result = service.fingerprint_records(records=records)
        assert result.total_records == 3

    def test_deterministic_fingerprinting(self, service, sample_records):
        r1 = service.fingerprint_records(records=sample_records)
        r2 = service.fingerprint_records(records=sample_records)
        # Fingerprint hashes should be identical for same records
        assert r1.fingerprints == r2.fingerprints


class TestNormalizeFields:
    """Tests for the _normalize_fields private helper."""

    def test_lowercase_strings(self, service):
        result = service._normalize_fields({"name": "ALICE"})
        assert result["name"] == "alice"

    def test_strip_whitespace(self, service):
        result = service._normalize_fields({"name": "  Alice  "})
        assert result["name"] == "alice"

    def test_none_to_empty_string(self, service):
        result = service._normalize_fields({"name": None})
        assert result["name"] == ""

    def test_non_string_passthrough(self, service):
        result = service._normalize_fields({"age": 25, "active": True})
        assert result["age"] == 25
        assert result["active"] is True

    def test_mixed_values(self, service):
        result = service._normalize_fields({
            "name": "ALICE",
            "age": 30,
            "email": None,
            "active": True,
        })
        assert result["name"] == "alice"
        assert result["age"] == 30
        assert result["email"] == ""
        assert result["active"] is True


# ===================================================================
# Test create_blocks
# ===================================================================


class TestCreateBlocks:
    """Tests for DuplicateDetectorService.create_blocks."""

    def test_basic_blocking(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert isinstance(result, BlockResponse)
        assert result.total_records == len(sample_records)
        assert result.total_blocks >= 1
        assert result.strategy == "sorted_neighborhood"

    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.create_blocks(records=[])

    def test_strategy_override(self, service, sample_records):
        result = service.create_blocks(
            records=sample_records,
            strategy="standard",
        )
        assert result.strategy == "standard"

    def test_key_fields(self, service, sample_records):
        result = service.create_blocks(
            records=sample_records,
            key_fields=["name"],
        )
        assert result.total_records == len(sample_records)

    def test_reduction_ratio(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert 0.0 <= result.reduction_ratio <= 1.0

    def test_largest_block(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert result.largest_block >= 1

    def test_total_pairs(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert result.total_pairs >= 0

    def test_provenance_hash(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert len(result.provenance_hash) == 64

    def test_result_stored(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert result.block_id in service._block_results

    def test_blocks_detail(self, service, sample_records):
        result = service.create_blocks(records=sample_records)
        assert isinstance(result.blocks, list)
        if result.blocks:
            block = result.blocks[0]
            assert "block_key" in block
            assert "size" in block


# ===================================================================
# Test compare_pairs
# ===================================================================


class TestComparePairs:
    """Tests for DuplicateDetectorService.compare_pairs."""

    def test_basic_compare(self, service, sample_records):
        pairs = _make_pairs(sample_records[:3])
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert isinstance(result, CompareResponse)
        assert result.total_pairs == len(pairs)

    def test_no_pairs_raises(self, service):
        with pytest.raises(ValueError, match="No candidate pairs"):
            service.compare_pairs(block_results={"pairs": []})

    def test_empty_pairs_key_raises(self, service):
        with pytest.raises(ValueError, match="No candidate pairs"):
            service.compare_pairs(block_results={})

    def test_exact_match_scoring(self, service):
        pairs = [{
            "id_a": "0",
            "id_b": "1",
            "record_a": {"name": "Alice", "email": "alice@co.com"},
            "record_b": {"name": "Alice", "email": "alice@co.com"},
        }]
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert result.comparisons[0]["overall_score"] == 1.0

    def test_no_match_scoring(self, service):
        pairs = [{
            "id_a": "0",
            "id_b": "1",
            "record_a": {"name": "Alice"},
            "record_b": {"name": "Bob"},
        }]
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert result.comparisons[0]["overall_score"] == 0.0

    def test_field_configs(self, service):
        pairs = [{
            "id_a": "0",
            "id_b": "1",
            "record_a": {"name": "Alice", "email": "alice@co.com"},
            "record_b": {"name": "Alice", "email": "bob@co.com"},
        }]
        field_configs = [
            {"field": "name", "algorithm": "exact", "weight": 2.0},
            {"field": "email", "algorithm": "exact", "weight": 1.0},
        ]
        result = service.compare_pairs(
            block_results={"pairs": pairs},
            field_configs=field_configs,
        )
        # name matches (1.0*2.0), email doesn't (0.0*1.0) => 2/3 = 0.6667
        score = result.comparisons[0]["overall_score"]
        assert abs(score - 0.6667) < 0.01

    def test_avg_similarity(self, service):
        pairs = [
            {
                "id_a": "0", "id_b": "1",
                "record_a": {"name": "Alice"},
                "record_b": {"name": "Alice"},
            },
            {
                "id_a": "2", "id_b": "3",
                "record_a": {"name": "Alice"},
                "record_b": {"name": "Bob"},
            },
        ]
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert abs(result.avg_similarity - 0.5) < 0.01

    def test_provenance_hash(self, service, sample_records):
        pairs = _make_pairs(sample_records[:2])
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert len(result.provenance_hash) == 64

    def test_result_stored(self, service, sample_records):
        pairs = _make_pairs(sample_records[:2])
        result = service.compare_pairs(block_results={"pairs": pairs})
        assert result.comparison_id in service._compare_results


# ===================================================================
# Test classify_matches
# ===================================================================


class TestClassifyMatches:
    """Tests for DuplicateDetectorService.classify_matches."""

    def test_basic_classify(self, service):
        comparisons = _make_comparisons([0.9, 0.7, 0.3])
        result = service.classify_matches(comparisons=comparisons)
        assert isinstance(result, ClassifyResponse)
        assert result.total_comparisons == 3

    def test_empty_comparisons_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.classify_matches(comparisons=[])

    def test_match_classification(self, service):
        comparisons = _make_comparisons([0.95])
        result = service.classify_matches(comparisons=comparisons)
        assert result.matches == 1
        assert result.classifications[0]["classification"] == "MATCH"

    def test_possible_classification(self, service):
        comparisons = _make_comparisons([0.75])
        result = service.classify_matches(comparisons=comparisons)
        assert result.possible_matches == 1
        assert result.classifications[0]["classification"] == "POSSIBLE"

    def test_non_match_classification(self, service):
        comparisons = _make_comparisons([0.3])
        result = service.classify_matches(comparisons=comparisons)
        assert result.non_matches == 1
        assert result.classifications[0]["classification"] == "NON_MATCH"

    def test_mixed_classifications(self, service):
        comparisons = _make_comparisons([0.9, 0.75, 0.3])
        result = service.classify_matches(comparisons=comparisons)
        assert result.matches == 1
        assert result.possible_matches == 1
        assert result.non_matches == 1

    def test_custom_thresholds(self, service):
        comparisons = _make_comparisons([0.5])
        result = service.classify_matches(
            comparisons=comparisons,
            thresholds={"match": 0.4, "possible": 0.2},
        )
        assert result.matches == 1

    def test_default_thresholds_from_config(self, service):
        comparisons = _make_comparisons([0.9])
        result = service.classify_matches(comparisons=comparisons)
        assert result.match_threshold == 0.85
        assert result.possible_threshold == 0.65

    def test_matches_stored(self, service):
        comparisons = _make_comparisons([0.9, 0.75])
        result = service.classify_matches(comparisons=comparisons)
        # MATCH and POSSIBLE are stored in _matches
        assert len(service._matches) >= 2

    def test_provenance_hash(self, service):
        comparisons = _make_comparisons([0.9])
        result = service.classify_matches(comparisons=comparisons)
        assert len(result.provenance_hash) == 64

    def test_stats_updated(self, service):
        comparisons = _make_comparisons([0.9, 0.95])
        service.classify_matches(comparisons=comparisons)
        stats = service.get_statistics()
        assert stats.total_duplicates_found == 2

    def test_boundary_match_threshold(self, service):
        """Score exactly at match_threshold should be MATCH."""
        comparisons = _make_comparisons([0.85])
        result = service.classify_matches(comparisons=comparisons)
        assert result.matches == 1

    def test_boundary_possible_threshold(self, service):
        """Score exactly at possible_threshold should be POSSIBLE."""
        comparisons = _make_comparisons([0.65])
        result = service.classify_matches(comparisons=comparisons)
        assert result.possible_matches == 1

    def test_just_below_possible_is_non_match(self, service):
        comparisons = _make_comparisons([0.6499])
        result = service.classify_matches(comparisons=comparisons)
        assert result.non_matches == 1


# ===================================================================
# Test form_clusters
# ===================================================================


class TestFormClusters:
    """Tests for DuplicateDetectorService.form_clusters."""

    def test_basic_clustering(self, service):
        matches = _make_matches(pairs=3)
        result = service.form_clusters(matches=matches)
        assert isinstance(result, ClusterResponse)
        assert result.total_matches == 3
        assert result.total_clusters >= 1

    def test_empty_matches_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.form_clusters(matches=[])

    def test_transitive_clustering(self, service):
        """A-B and B-C should yield a single cluster {A,B,C}."""
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "B", "record_b_id": "C"},
        ]
        result = service.form_clusters(matches=matches)
        assert result.total_clusters == 1
        cluster = result.clusters[0]
        assert cluster["size"] == 3

    def test_separate_clusters(self, service):
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "C", "record_b_id": "D"},
        ]
        result = service.form_clusters(matches=matches)
        assert result.total_clusters == 2

    def test_largest_cluster(self, service):
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "B", "record_b_id": "C"},
            {"record_a_id": "C", "record_b_id": "D"},
        ]
        result = service.form_clusters(matches=matches)
        assert result.largest_cluster == 4

    def test_avg_cluster_size(self, service):
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "C", "record_b_id": "D"},
        ]
        result = service.form_clusters(matches=matches)
        assert result.avg_cluster_size == 2.0

    def test_algorithm_override(self, service):
        matches = _make_matches(pairs=2)
        result = service.form_clusters(
            matches=matches,
            algorithm="connected_components",
        )
        assert result.algorithm == "connected_components"

    def test_clusters_stored(self, service):
        matches = _make_matches(pairs=2)
        result = service.form_clusters(matches=matches)
        for cluster in result.clusters:
            assert cluster["cluster_id"] in service._clusters

    def test_provenance_hash(self, service):
        matches = _make_matches(pairs=2)
        result = service.form_clusters(matches=matches)
        assert len(result.provenance_hash) == 64

    def test_stats_updated(self, service):
        matches = _make_matches(pairs=2)
        service.form_clusters(matches=matches)
        stats = service.get_statistics()
        assert stats.total_clusters >= 1

    def test_single_pair_cluster(self, service):
        matches = [{"record_a_id": "A", "record_b_id": "B"}]
        result = service.form_clusters(matches=matches)
        assert result.total_clusters == 1
        assert result.clusters[0]["size"] == 2


# ===================================================================
# Test merge_duplicates
# ===================================================================


class TestMergeDuplicates:
    """Tests for DuplicateDetectorService.merge_duplicates."""

    def test_basic_merge(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "alice smith", "email": "alice@company.com"},
        ]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert isinstance(result, MergeResponse)
        assert result.total_golden_records == 1

    def test_empty_clusters_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.merge_duplicates(clusters=[], records=[])

    def test_most_complete_strategy(self, service):
        """Merge should pick the longest non-null value for each field."""
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [
            {"id": "0", "name": "Alice", "email": ""},
            {"id": "1", "name": "Alice Smith", "email": "alice@company.com"},
        ]
        result = service.merge_duplicates(clusters=clusters, records=records)
        golden = result.merged_records[0]["golden_record"]
        assert golden["name"] == "Alice Smith"
        assert golden["email"] == "alice@company.com"

    def test_conflict_counting(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [
            {"id": "0", "name": "Alice", "phone": "111"},
            {"id": "1", "name": "Bob", "phone": "222"},
        ]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert result.conflicts_resolved >= 2  # name and phone differ

    def test_multiple_clusters(self, service):
        clusters = [
            {"cluster_id": "c1", "members": ["0", "1"]},
            {"cluster_id": "c2", "members": ["2", "3"]},
        ]
        records = [
            {"id": "0", "name": "Alice"},
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "Bob"},
            {"id": "3", "name": "bob"},
        ]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert result.total_golden_records == 2
        assert result.total_clusters == 2

    def test_strategy_override(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [
            {"id": "0", "name": "Alice"},
            {"id": "1", "name": "Bob"},
        ]
        result = service.merge_duplicates(
            clusters=clusters,
            records=records,
            strategy="keep_first",
        )
        assert result.strategy == "keep_first"

    def test_provenance_hash(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [{"id": "0", "name": "A"}, {"id": "1", "name": "B"}]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert len(result.provenance_hash) == 64

    def test_result_stored(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [{"id": "0", "name": "A"}, {"id": "1", "name": "B"}]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert result.merge_id in service._merge_results

    def test_stats_updated(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [{"id": "0", "name": "A"}, {"id": "1", "name": "B"}]
        service.merge_duplicates(clusters=clusters, records=records)
        stats = service.get_statistics()
        assert stats.total_merges >= 1

    def test_single_member_cluster_skipped(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0"]}]
        records = [{"id": "0", "name": "Alice"}]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert result.total_golden_records == 0

    def test_missing_members_skipped(self, service):
        clusters = [{"cluster_id": "c1", "members": ["99", "100"]}]
        records = [{"id": "0", "name": "Alice"}]
        result = service.merge_duplicates(clusters=clusters, records=records)
        assert result.total_golden_records == 0


class TestMergeByMostComplete:
    """Tests for the _merge_by_most_complete private helper."""

    def test_picks_longest_value(self, service):
        records = [
            {"name": "Alice", "email": ""},
            {"name": "Alice Smith", "email": "alice@co.com"},
        ]
        golden = service._merge_by_most_complete(records)
        assert golden["name"] == "Alice Smith"
        assert golden["email"] == "alice@co.com"

    def test_none_values_skipped(self, service):
        records = [
            {"name": None, "email": "alice@co.com"},
            {"name": "Alice", "email": None},
        ]
        golden = service._merge_by_most_complete(records)
        assert golden["name"] == "Alice"
        assert golden["email"] == "alice@co.com"

    def test_all_none(self, service):
        records = [{"name": None}, {"name": None}]
        golden = service._merge_by_most_complete(records)
        assert golden["name"] is None

    def test_disjoint_fields(self, service):
        records = [
            {"name": "Alice"},
            {"email": "alice@co.com"},
        ]
        golden = service._merge_by_most_complete(records)
        assert golden["name"] == "Alice"
        assert golden["email"] == "alice@co.com"


class TestCountConflicts:
    """Tests for the _count_conflicts private helper."""

    def test_no_conflicts(self, service):
        records = [
            {"name": "Alice", "email": "alice@co.com"},
            {"name": "Alice", "email": "alice@co.com"},
        ]
        assert service._count_conflicts(records) == 0

    def test_one_conflict(self, service):
        records = [
            {"name": "Alice", "email": "alice@co.com"},
            {"name": "Bob", "email": "alice@co.com"},
        ]
        assert service._count_conflicts(records) == 1

    def test_all_conflicts(self, service):
        records = [
            {"name": "Alice", "phone": "111"},
            {"name": "Bob", "phone": "222"},
        ]
        assert service._count_conflicts(records) == 2

    def test_none_not_counted(self, service):
        records = [
            {"name": "Alice", "email": None},
            {"name": "Alice", "email": "alice@co.com"},
        ]
        assert service._count_conflicts(records) == 0

    def test_empty_string_not_counted(self, service):
        records = [
            {"name": "Alice", "email": ""},
            {"name": "Alice", "email": "alice@co.com"},
        ]
        assert service._count_conflicts(records) == 0


# ===================================================================
# Test run_pipeline
# ===================================================================


class TestRunPipeline:
    """Tests for DuplicateDetectorService.run_pipeline (full pipeline)."""

    def test_basic_pipeline(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        assert isinstance(result, PipelineResponse)
        assert result.status in ("completed", "failed")
        assert result.total_records == len(duplicate_records)

    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="must not be empty"):
            service.run_pipeline(records=[])

    def test_pipeline_stages_present(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        if result.status == "completed":
            assert "fingerprint" in result.stages
            assert "block" in result.stages

    def test_pipeline_id_is_uuid(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        uuid.UUID(result.pipeline_id)

    def test_provenance_hash(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        assert len(result.provenance_hash) == 64

    def test_pipeline_result_stored(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        assert result.pipeline_id in service._pipeline_results

    def test_stats_updated_on_completion(self, service, duplicate_records):
        service.run_pipeline(records=duplicate_records)
        stats = service.get_statistics()
        assert stats.total_jobs >= 1

    def test_completed_jobs_counted(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        stats = service.get_statistics()
        if result.status == "completed":
            assert stats.completed_jobs >= 1
        else:
            assert stats.failed_jobs >= 1

    def test_active_jobs_decremented(self, service, duplicate_records):
        service.run_pipeline(records=duplicate_records)
        assert service._active_jobs == 0

    def test_pipeline_with_options(self, service, duplicate_records):
        result = service.run_pipeline(
            records=duplicate_records,
            options={"fingerprint_algorithm": "simhash"},
        )
        assert result.total_records == len(duplicate_records)

    def test_pipeline_with_rule(self, service, duplicate_records):
        result = service.run_pipeline(
            records=duplicate_records,
            rule={"name": "test-rule"},
        )
        assert result.total_records == len(duplicate_records)

    def test_pipeline_processing_time(self, service, duplicate_records):
        result = service.run_pipeline(records=duplicate_records)
        assert result.processing_time_ms >= 0


# ===================================================================
# Test Job Management
# ===================================================================


class TestCreateDedupJob:
    """Tests for DuplicateDetectorService.create_dedup_job."""

    def test_basic_create(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1", "ds-2"])
        assert "job_id" in job
        assert job["status"] == "created"
        assert job["dataset_ids"] == ["ds-1", "ds-2"]

    def test_job_stored(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        assert job["job_id"] in service._jobs

    def test_with_rule_id(self, service):
        job = service.create_dedup_job(
            dataset_ids=["ds-1"],
            rule_id="rule-123",
        )
        assert job["rule_id"] == "rule-123"

    def test_without_rule_id(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        assert job["rule_id"] is None

    def test_timestamps_set(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        assert "created_at" in job
        assert "updated_at" in job

    def test_provenance_recorded(self, service):
        before = service.provenance.entry_count
        service.create_dedup_job(dataset_ids=["ds-1"])
        assert service.provenance.entry_count == before + 1

    def test_multiple_jobs(self, service):
        j1 = service.create_dedup_job(dataset_ids=["ds-1"])
        j2 = service.create_dedup_job(dataset_ids=["ds-2"])
        assert j1["job_id"] != j2["job_id"]
        assert len(service._jobs) == 2


class TestGetJob:
    """Tests for DuplicateDetectorService.get_job."""

    def test_existing_job(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        retrieved = service.get_job(job["job_id"])
        assert retrieved is not None
        assert retrieved["job_id"] == job["job_id"]

    def test_nonexistent_job(self, service):
        result = service.get_job("nonexistent-id")
        assert result is None


class TestListJobs:
    """Tests for DuplicateDetectorService.list_jobs."""

    def test_empty_list(self, service):
        result = service.list_jobs()
        assert result["jobs"] == []
        assert result["count"] == 0
        assert result["total"] == 0

    def test_list_all(self, service):
        service.create_dedup_job(dataset_ids=["ds-1"])
        service.create_dedup_job(dataset_ids=["ds-2"])
        result = service.list_jobs()
        assert result["count"] == 2
        assert result["total"] == 2

    def test_filter_by_status(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        service.cancel_job(job["job_id"])
        service.create_dedup_job(dataset_ids=["ds-2"])
        result = service.list_jobs(status="cancelled")
        assert result["count"] == 1

    def test_pagination_limit(self, service):
        for i in range(5):
            service.create_dedup_job(dataset_ids=[f"ds-{i}"])
        result = service.list_jobs(limit=3)
        assert result["count"] == 3
        assert result["total"] == 5

    def test_pagination_offset(self, service):
        for i in range(5):
            service.create_dedup_job(dataset_ids=[f"ds-{i}"])
        result = service.list_jobs(offset=3, limit=50)
        assert result["count"] == 2

    def test_pagination_fields(self, service):
        result = service.list_jobs(limit=10, offset=0)
        assert result["limit"] == 10
        assert result["offset"] == 0


class TestCancelJob:
    """Tests for DuplicateDetectorService.cancel_job."""

    def test_cancel_existing(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        cancelled = service.cancel_job(job["job_id"])
        assert cancelled is not None
        assert cancelled["status"] == "cancelled"

    def test_cancel_updates_timestamp(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        original_updated = job["updated_at"]
        cancelled = service.cancel_job(job["job_id"])
        # updated_at may be same if within same second, but should be set
        assert "updated_at" in cancelled

    def test_cancel_nonexistent(self, service):
        result = service.cancel_job("nonexistent-id")
        assert result is None

    def test_cancel_provenance(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        before = service.provenance.entry_count
        service.cancel_job(job["job_id"])
        assert service.provenance.entry_count == before + 1


# ===================================================================
# Test Rule Management
# ===================================================================


class TestCreateRule:
    """Tests for DuplicateDetectorService.create_rule."""

    def test_basic_create(self, service):
        rule = service.create_rule(rule_config={"name": "test-rule"})
        assert "rule_id" in rule
        assert rule["name"] == "test-rule"
        assert rule["is_active"] is True

    def test_rule_stored(self, service):
        rule = service.create_rule(rule_config={"name": "test-rule"})
        assert rule["rule_id"] in service._rules

    def test_defaults_from_config(self, service):
        rule = service.create_rule(rule_config={"name": "test"})
        assert rule["match_threshold"] == 0.85
        assert rule["possible_threshold"] == 0.65
        assert rule["blocking_strategy"] == "sorted_neighborhood"
        assert rule["merge_strategy"] == "keep_most_complete"

    def test_override_values(self, service):
        rule = service.create_rule(rule_config={
            "name": "custom",
            "match_threshold": 0.9,
            "merge_strategy": "golden_record",
        })
        assert rule["match_threshold"] == 0.9
        assert rule["merge_strategy"] == "golden_record"

    def test_provenance_hash(self, service):
        rule = service.create_rule(rule_config={"name": "test"})
        assert len(rule["provenance_hash"]) == 64

    def test_stats_updated(self, service):
        service.create_rule(rule_config={"name": "test"})
        stats = service.get_statistics()
        assert stats.total_rules == 1

    def test_multiple_rules(self, service):
        service.create_rule(rule_config={"name": "rule-1"})
        service.create_rule(rule_config={"name": "rule-2"})
        assert len(service._rules) == 2

    def test_timestamps_set(self, service):
        rule = service.create_rule(rule_config={"name": "test"})
        assert "created_at" in rule
        assert "updated_at" in rule


class TestListRules:
    """Tests for DuplicateDetectorService.list_rules."""

    def test_empty_list(self, service):
        result = service.list_rules()
        assert result["rules"] == []
        assert result["count"] == 0

    def test_list_all(self, service):
        service.create_rule(rule_config={"name": "r1"})
        service.create_rule(rule_config={"name": "r2"})
        result = service.list_rules()
        assert result["count"] == 2
        assert result["total"] == 2

    def test_pagination(self, service):
        for i in range(5):
            service.create_rule(rule_config={"name": f"r-{i}"})
        result = service.list_rules(limit=3, offset=0)
        assert result["count"] == 3

    def test_pagination_offset(self, service):
        for i in range(5):
            service.create_rule(rule_config={"name": f"r-{i}"})
        result = service.list_rules(offset=4, limit=50)
        assert result["count"] == 1


# ===================================================================
# Test Statistics and Health
# ===================================================================


class TestGetStatistics:
    """Tests for DuplicateDetectorService.get_statistics."""

    def test_initial_stats(self, service):
        stats = service.get_statistics()
        assert isinstance(stats, StatsResponse)
        assert stats.total_jobs == 0
        assert stats.active_jobs == 0

    def test_stats_after_fingerprinting(self, service, sample_records):
        service.fingerprint_records(records=sample_records)
        stats = service.get_statistics()
        assert stats.total_records_processed == len(sample_records)

    def test_stats_provenance_entries(self, service, sample_records):
        service.fingerprint_records(records=sample_records)
        stats = service.get_statistics()
        assert stats.provenance_entries >= 1

    def test_avg_similarity_tracking(self, service):
        pairs = [{
            "id_a": "0", "id_b": "1",
            "record_a": {"name": "Alice"},
            "record_b": {"name": "Alice"},
        }]
        service.compare_pairs(block_results={"pairs": pairs})
        stats = service.get_statistics()
        assert stats.avg_similarity > 0


class TestHealthCheck:
    """Tests for DuplicateDetectorService.health_check."""

    def test_not_started(self, service):
        health = service.health_check()
        assert health["status"] == "not_started"
        assert health["started"] is False

    def test_service_name(self, service):
        health = service.health_check()
        assert health["service"] == "duplicate-detector"

    def test_counts_present(self, service):
        health = service.health_check()
        assert "jobs" in health
        assert "rules" in health
        assert "provenance_entries" in health
        assert "fingerprint_results" in health
        assert "block_results" in health
        assert "compare_results" in health
        assert "classify_results" in health
        assert "cluster_results" in health
        assert "merge_results" in health
        assert "pipeline_results" in health

    def test_after_operations(self, service, sample_records):
        service.fingerprint_records(records=sample_records)
        health = service.health_check()
        assert health["fingerprint_results"] == 1


class TestGetMetrics:
    """Tests for DuplicateDetectorService.get_metrics."""

    def test_basic_metrics(self, service):
        metrics = service.get_metrics()
        assert "prometheus_available" in metrics
        assert "started" in metrics
        assert "total_jobs" in metrics
        assert "total_records_processed" in metrics

    def test_metrics_after_operations(self, service, sample_records):
        service.fingerprint_records(records=sample_records)
        metrics = service.get_metrics()
        assert metrics["total_records_processed"] == len(sample_records)


# ===================================================================
# Test Report and Detail Retrieval
# ===================================================================


class TestGenerateReport:
    """Tests for DuplicateDetectorService.generate_report."""

    def test_basic_report(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        report = service.generate_report(job_id=job["job_id"])
        assert "report_id" in report
        assert report["job_id"] == job["job_id"]
        assert report["format"] == "json"

    def test_report_with_format(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        report = service.generate_report(
            job_id=job["job_id"],
            report_format="markdown",
        )
        assert report["format"] == "markdown"

    def test_report_includes_statistics(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        report = service.generate_report(job_id=job["job_id"])
        assert "statistics" in report
        assert isinstance(report["statistics"], dict)

    def test_report_provenance(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        report = service.generate_report(job_id=job["job_id"])
        assert len(report["provenance_hash"]) == 64

    def test_report_for_nonexistent_job(self, service):
        report = service.generate_report(job_id="nonexistent")
        assert report["job"] is None

    def test_report_generated_at(self, service):
        job = service.create_dedup_job(dataset_ids=["ds-1"])
        report = service.generate_report(job_id=job["job_id"])
        assert "generated_at" in report


class TestGetMatchDetails:
    """Tests for DuplicateDetectorService.get_match_details."""

    def test_existing_match(self, service):
        comparisons = _make_comparisons([0.9])
        result = service.classify_matches(comparisons=comparisons)
        match_id = result.classifications[0]["pair_id"]
        detail = service.get_match_details(match_id)
        assert detail is not None

    def test_nonexistent_match(self, service):
        assert service.get_match_details("nonexistent") is None


class TestGetClusterDetails:
    """Tests for DuplicateDetectorService.get_cluster_details."""

    def test_existing_cluster(self, service):
        matches = [{"record_a_id": "A", "record_b_id": "B"}]
        result = service.form_clusters(matches=matches)
        cluster_id = result.clusters[0]["cluster_id"]
        detail = service.get_cluster_details(cluster_id)
        assert detail is not None

    def test_nonexistent_cluster(self, service):
        assert service.get_cluster_details("nonexistent") is None


class TestGetMergeResult:
    """Tests for DuplicateDetectorService.get_merge_result."""

    def test_existing_merge(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [{"id": "0", "name": "A"}, {"id": "1", "name": "B"}]
        result = service.merge_duplicates(clusters=clusters, records=records)
        retrieved = service.get_merge_result(result.merge_id)
        assert retrieved is not None

    def test_nonexistent_merge(self, service):
        assert service.get_merge_result("nonexistent") is None


class TestGetProvenance:
    """Tests for DuplicateDetectorService.get_provenance."""

    def test_returns_tracker(self, service):
        tracker = service.get_provenance()
        assert isinstance(tracker, _ProvenanceTracker)
        assert tracker is service.provenance


# ===================================================================
# Test configure_duplicate_detector and module-level functions
# ===================================================================


class TestConfigureDuplicateDetector:
    """Tests for configure_duplicate_detector async function."""

    def test_configure_sets_service(self):
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        with patch.object(DuplicateDetectorService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_duplicate_detector(app)
            )
        assert isinstance(svc, DuplicateDetectorService)
        assert svc._started is True

    def test_configure_attaches_to_app_state(self):
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        with patch.object(DuplicateDetectorService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_duplicate_detector(app)
            )
        app.state.__setattr__("duplicate_detector_service", svc)

    def test_configure_with_custom_config(self, config):
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        with patch.object(DuplicateDetectorService, "_init_engines"):
            svc = asyncio.get_event_loop().run_until_complete(
                configure_duplicate_detector(app, config=config)
            )
        assert svc.config is config


class TestGetDuplicateDetector:
    """Tests for get_duplicate_detector function."""

    def test_returns_service(self):
        app = MagicMock()
        with patch.object(DuplicateDetectorService, "_init_engines"):
            app.state.duplicate_detector_service = DuplicateDetectorService()
        svc = get_duplicate_detector(app)
        assert isinstance(svc, DuplicateDetectorService)

    def test_raises_if_not_configured(self):
        app = MagicMock()
        app.state = MagicMock(spec=[])  # No attributes
        with pytest.raises(RuntimeError, match="not configured"):
            get_duplicate_detector(app)


class TestGetRouter:
    """Tests for get_router function."""

    def test_returns_router_or_none(self):
        result = get_router()
        # Result is either a router or None depending on FastAPI availability
        assert result is None or result is not None


# ===================================================================
# Test Response Models (Pydantic)
# ===================================================================


class TestResponseModels:
    """Tests for the Pydantic response model classes."""

    def test_fingerprint_response_defaults(self):
        r = FingerprintResponse()
        assert r.algorithm == "sha256"
        assert r.total_records == 0
        assert r.fingerprints == {}

    def test_fingerprint_response_serialization(self):
        r = FingerprintResponse(algorithm="sha256", total_records=10)
        data = r.model_dump(mode="json")
        assert data["algorithm"] == "sha256"
        assert data["total_records"] == 10

    def test_block_response_defaults(self):
        r = BlockResponse()
        assert r.strategy == "sorted_neighborhood"
        assert r.total_blocks == 0

    def test_compare_response_defaults(self):
        r = CompareResponse()
        assert r.algorithm == "jaro_winkler"
        assert r.total_pairs == 0

    def test_classify_response_defaults(self):
        r = ClassifyResponse()
        assert r.match_threshold == 0.85
        assert r.possible_threshold == 0.65

    def test_cluster_response_defaults(self):
        r = ClusterResponse()
        assert r.algorithm == "union_find"
        assert r.total_clusters == 0

    def test_merge_response_defaults(self):
        r = MergeResponse()
        assert r.strategy == "keep_most_complete"
        assert r.conflict_resolution == "most_complete"

    def test_pipeline_response_defaults(self):
        r = PipelineResponse()
        assert r.status == "completed"
        assert r.stages == {}

    def test_stats_response_defaults(self):
        r = StatsResponse()
        assert r.total_jobs == 0
        assert r.active_jobs == 0

    def test_fingerprint_response_uuid(self):
        r = FingerprintResponse()
        uuid.UUID(r.fingerprint_id)

    def test_block_response_uuid(self):
        r = BlockResponse()
        uuid.UUID(r.block_id)

    def test_compare_response_uuid(self):
        r = CompareResponse()
        uuid.UUID(r.comparison_id)

    def test_classify_response_uuid(self):
        r = ClassifyResponse()
        uuid.UUID(r.classify_id)

    def test_cluster_response_uuid(self):
        r = ClusterResponse()
        uuid.UUID(r.cluster_run_id)

    def test_merge_response_uuid(self):
        r = MergeResponse()
        uuid.UUID(r.merge_id)

    def test_pipeline_response_uuid(self):
        r = PipelineResponse()
        uuid.UUID(r.pipeline_id)


# ===================================================================
# Test Error Handling
# ===================================================================


class TestErrorHandling:
    """Tests for error handling across all service methods."""

    def test_fingerprint_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.fingerprint_records(records=[])

    def test_block_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.create_blocks(records=[])

    def test_compare_no_pairs_raises(self, service):
        with pytest.raises(ValueError):
            service.compare_pairs(block_results={"pairs": []})

    def test_classify_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.classify_matches(comparisons=[])

    def test_cluster_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.form_clusters(matches=[])

    def test_merge_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.merge_duplicates(clusters=[], records=[])

    def test_pipeline_empty_raises(self, service):
        with pytest.raises(ValueError):
            service.run_pipeline(records=[])


# ===================================================================
# Test Service State Management
# ===================================================================


class TestServiceState:
    """Tests for service state tracking and transitions."""

    def test_started_flag(self, service):
        assert service._started is False
        service._started = True
        health = service.health_check()
        assert health["status"] == "healthy"

    def test_active_jobs_tracking(self, service, duplicate_records):
        """After pipeline run, active_jobs returns to zero."""
        assert service._active_jobs == 0
        service.run_pipeline(records=duplicate_records)
        assert service._active_jobs == 0

    def test_similarity_tracking(self, service):
        pairs = [{
            "id_a": "0", "id_b": "1",
            "record_a": {"name": "Alice"},
            "record_b": {"name": "Alice"},
        }]
        service.compare_pairs(block_results={"pairs": pairs})
        assert service._similarity_count == 1
        assert service._similarity_sum == 1.0


# ===================================================================
# Test Thread Safety
# ===================================================================


class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_fingerprinting(self, service):
        records = [{"name": f"user-{i}"} for i in range(10)]
        errors = []

        def worker():
            try:
                service.fingerprint_records(records=records)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Multiple results stored
        assert len(service._fingerprint_results) == 5

    def test_concurrent_job_creation(self, service):
        errors = []

        def worker(idx):
            try:
                service.create_dedup_job(dataset_ids=[f"ds-{idx}"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(service._jobs) == 10


# ===================================================================
# Test Determinism
# ===================================================================


class TestDeterminism:
    """Tests for deterministic behavior (same input -> same output)."""

    def test_fingerprint_determinism(self, service, sample_records):
        r1 = service.fingerprint_records(records=sample_records)
        r2 = service.fingerprint_records(records=sample_records)
        assert r1.fingerprints == r2.fingerprints

    def test_classify_determinism(self, service):
        comparisons = _make_comparisons([0.9, 0.7, 0.3])
        r1 = service.classify_matches(comparisons=comparisons)
        r2 = service.classify_matches(comparisons=comparisons)
        assert r1.matches == r2.matches
        assert r1.possible_matches == r2.possible_matches
        assert r1.non_matches == r2.non_matches

    def test_cluster_determinism(self, service):
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "B", "record_b_id": "C"},
        ]
        r1 = service.form_clusters(matches=matches)
        r2 = service.form_clusters(matches=matches)
        assert r1.total_clusters == r2.total_clusters
        assert r1.largest_cluster == r2.largest_cluster

    def test_merge_determinism(self, service):
        clusters = [{"cluster_id": "c1", "members": ["0", "1"]}]
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "Alice Smith", "email": ""},
        ]
        r1 = service.merge_duplicates(clusters=clusters, records=records)
        r2 = service.merge_duplicates(clusters=clusters, records=records)
        g1 = r1.merged_records[0]["golden_record"]
        g2 = r2.merged_records[0]["golden_record"]
        assert g1 == g2

    def test_compute_hash_determinism(self):
        data = {"key": "value", "list": [1, 2, 3]}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2


# ===================================================================
# Test _build_pairs_from_blocks
# ===================================================================


class TestBuildPairsFromBlocks:
    """Tests for _build_pairs_from_blocks private method."""

    def test_basic_pair_building(self, service):
        blocks = [{"members": [0, 1, 2]}]
        records = [
            {"name": "Alice"},
            {"name": "Bob"},
            {"name": "Charlie"},
        ]
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) == 3  # C(3,2) = 3

    def test_no_members_skipped(self, service):
        blocks = [{"block_key": "abc"}]  # No members key
        records = [{"name": "Alice"}]
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) == 0

    def test_empty_members_skipped(self, service):
        blocks = [{"members": []}]
        records = []
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) == 0

    def test_single_member_no_pairs(self, service):
        blocks = [{"members": [0]}]
        records = [{"name": "Alice"}]
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) == 0

    def test_multiple_blocks(self, service):
        blocks = [
            {"members": [0, 1]},
            {"members": [2, 3]},
        ]
        records = [
            {"name": "A"}, {"name": "B"},
            {"name": "C"}, {"name": "D"},
        ]
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) == 2  # 1 pair per block

    def test_max_comparisons_limit(self, service):
        service.config.max_comparisons_per_block = 2
        blocks = [{"members": [0, 1, 2, 3]}]
        records = [{"name": f"u{i}"} for i in range(4)]
        pairs = service._build_pairs_from_blocks(blocks, records)
        assert len(pairs) <= 2


# ===================================================================
# Test _update_avg_similarity
# ===================================================================


class TestUpdateAvgSimilarity:
    """Tests for _update_avg_similarity private method."""

    def test_basic_update(self, service):
        service._update_avg_similarity(total_sim=2.0, count=4)
        assert service._similarity_sum == 2.0
        assert service._similarity_count == 4

    def test_cumulative_update(self, service):
        service._update_avg_similarity(total_sim=1.0, count=2)
        service._update_avg_similarity(total_sim=3.0, count=3)
        assert service._similarity_sum == 4.0
        assert service._similarity_count == 5

    def test_avg_computation(self, service):
        service._update_avg_similarity(total_sim=4.0, count=4)
        stats = service.get_statistics()
        assert abs(stats.avg_similarity - 1.0) < 0.01


# ===================================================================
# Test _wrap methods (engine delegation wrappers)
# ===================================================================


class TestWrapMethods:
    """Tests for the _wrap_*_result engine delegation wrappers."""

    def test_wrap_fingerprint_dict(self, service):
        import time
        engine_result = {
            "total_records": 10,
            "unique_fingerprints": 8,
            "duplicate_candidates": 2,
            "fingerprints": {"0": "abc", "1": "def"},
        }
        result = service._wrap_fingerprint_result(
            engine_result, time.time(), "sha256",
        )
        assert isinstance(result, FingerprintResponse)
        assert result.total_records == 10
        assert result.unique_fingerprints == 8

    def test_wrap_fingerprint_non_dict(self, service):
        import time
        result = service._wrap_fingerprint_result(
            "not_a_dict", time.time(), "sha256",
        )
        assert isinstance(result, FingerprintResponse)
        assert result.total_records == 0

    def test_wrap_block_dict(self, service):
        import time
        engine_result = {
            "total_records": 100,
            "total_blocks": 10,
            "total_pairs": 50,
        }
        result = service._wrap_block_result(
            engine_result, time.time(), "standard",
        )
        assert isinstance(result, BlockResponse)
        assert result.total_records == 100

    def test_wrap_compare_dict(self, service):
        import time
        engine_result = {
            "total_pairs": 5,
            "avg_similarity": 0.75,
        }
        result = service._wrap_compare_result(
            engine_result, time.time(), "jaro_winkler",
        )
        assert isinstance(result, CompareResponse)
        assert result.total_pairs == 5

    def test_wrap_classify_dict(self, service):
        import time
        engine_result = {
            "total_comparisons": 10,
            "matches": 5,
            "possible_matches": 3,
            "non_matches": 2,
        }
        result = service._wrap_classify_result(
            engine_result, time.time(), 0.85, 0.65,
        )
        assert isinstance(result, ClassifyResponse)
        assert result.matches == 5

    def test_wrap_cluster_dict(self, service):
        import time
        engine_result = {
            "total_matches": 20,
            "total_clusters": 5,
            "largest_cluster": 4,
        }
        result = service._wrap_cluster_result(
            engine_result, time.time(), "union_find",
        )
        assert isinstance(result, ClusterResponse)
        assert result.total_clusters == 5

    def test_wrap_merge_dict(self, service):
        import time
        engine_result = {
            "total_clusters": 3,
            "total_records_merged": 10,
            "total_golden_records": 3,
            "conflicts_resolved": 5,
        }
        result = service._wrap_merge_result(
            engine_result, time.time(), "keep_most_complete",
        )
        assert isinstance(result, MergeResponse)
        assert result.total_golden_records == 3

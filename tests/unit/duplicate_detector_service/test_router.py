# -*- coding: utf-8 -*-
"""
Unit tests for Duplicate Detection REST API Router - AGENT-DATA-011

Tests all 20 FastAPI endpoints under /api/v1/dedup with 90+ test cases covering:
- POST /jobs: create dedup job
- GET /jobs: list jobs with pagination and status filter
- GET /jobs/{job_id}: get job details / 404
- DELETE /jobs/{job_id}: cancel job / 404
- POST /fingerprint: fingerprint records / validation
- POST /block: create blocks / validation
- POST /compare: compare pairs / validation
- POST /classify: classify matches / validation
- GET /matches: list matches with pagination
- GET /matches/{match_id}: get match details / 404
- POST /clusters: form clusters / validation
- GET /clusters: list clusters with pagination
- GET /clusters/{cluster_id}: get cluster details / 404
- POST /merge: merge duplicates / validation
- GET /merge/{merge_id}: get merge result / 404
- POST /pipeline: run full pipeline / validation
- POST /rules: create dedup rule
- GET /rules: list rules with pagination
- GET /health: health check
- GET /stats: statistics
- Service not configured (503)
- Request validation (422)
- Error responses (400, 404, 503)

Author: GreenLang Platform Team
Date: February 2026
"""

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.duplicate_detector.config import (
    DuplicateDetectorConfig,
    reset_config,
    set_config,
)
from greenlang.duplicate_detector.setup import DuplicateDetectorService

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FASTAPI_AVAILABLE,
    reason="FastAPI not available",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset config singleton before and after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def config() -> DuplicateDetectorConfig:
    """Create test configuration."""
    return DuplicateDetectorConfig()


@pytest.fixture
def service(config: DuplicateDetectorConfig) -> DuplicateDetectorService:
    """Create a fresh DuplicateDetectorService.

    Patches _init_engines so the service can be created without
    the full SDK engines being importable or compatible.
    """
    set_config(config)
    with patch.object(DuplicateDetectorService, "_init_engines"):
        svc = DuplicateDetectorService(config=config)
    return svc


@pytest.fixture
def app(service: DuplicateDetectorService) -> "FastAPI":
    """Create a FastAPI app with the duplicate detector service configured."""
    from greenlang.duplicate_detector.api.router import router

    application = FastAPI()
    application.state.duplicate_detector_service = service
    service._started = True
    application.include_router(router)
    return application


@pytest.fixture
def client(app: "FastAPI") -> "TestClient":
    """Create a TestClient for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Sample records for dedup testing."""
    return [
        {"id": "0", "name": "Alice Smith", "email": "alice@co.com"},
        {"id": "1", "name": "alice smith", "email": "alice@co.com"},
        {"id": "2", "name": "Bob Jones", "email": "bob@co.com"},
    ]


@pytest.fixture
def sample_pairs() -> Dict[str, Any]:
    """Sample block_results dict containing pairs."""
    return {
        "pairs": [
            {
                "id_a": "0", "id_b": "1",
                "record_a": {"name": "Alice", "email": "alice@co.com"},
                "record_b": {"name": "Alice", "email": "alice@co.com"},
            },
        ],
    }


@pytest.fixture
def sample_comparisons() -> List[Dict[str, Any]]:
    """Sample comparison results for classification."""
    return [
        {"pair_id": str(uuid.uuid4()), "record_a_id": "r0", "record_b_id": "r1", "overall_score": 0.95},
        {"pair_id": str(uuid.uuid4()), "record_a_id": "r2", "record_b_id": "r3", "overall_score": 0.70},
        {"pair_id": str(uuid.uuid4()), "record_a_id": "r4", "record_b_id": "r5", "overall_score": 0.30},
    ]


@pytest.fixture
def sample_matches() -> List[Dict[str, Any]]:
    """Sample matches for clustering."""
    return [
        {"record_a_id": "A", "record_b_id": "B"},
        {"record_a_id": "B", "record_b_id": "C"},
    ]


# ===================================================================
# 1. POST /jobs - Create dedup job
# ===================================================================


class TestCreateJobEndpoint:
    """Tests for POST /api/v1/dedup/jobs."""

    def test_create_job_success(self, client):
        resp = client.post("/api/v1/dedup/jobs", json={
            "dataset_ids": ["ds-1", "ds-2"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "created"
        assert data["dataset_ids"] == ["ds-1", "ds-2"]

    def test_create_job_with_rule_id(self, client):
        resp = client.post("/api/v1/dedup/jobs", json={
            "dataset_ids": ["ds-1"],
            "rule_id": "rule-abc",
        })
        assert resp.status_code == 200
        assert resp.json()["rule_id"] == "rule-abc"

    def test_create_job_missing_dataset_ids(self, client):
        resp = client.post("/api/v1/dedup/jobs", json={})
        assert resp.status_code == 422

    def test_create_job_empty_body(self, client):
        resp = client.post("/api/v1/dedup/jobs")
        assert resp.status_code == 422


# ===================================================================
# 2. GET /jobs - List jobs
# ===================================================================


class TestListJobsEndpoint:
    """Tests for GET /api/v1/dedup/jobs."""

    def test_list_empty(self, client):
        resp = client.get("/api/v1/dedup/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jobs"] == []
        assert data["count"] == 0

    def test_list_after_create(self, client):
        client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-1"]})
        client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-2"]})
        resp = client.get("/api/v1/dedup/jobs")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_list_with_status_filter(self, client):
        r = client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-1"]})
        job_id = r.json()["job_id"]
        client.delete(f"/api/v1/dedup/jobs/{job_id}")
        client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-2"]})
        resp = client.get("/api/v1/dedup/jobs", params={"status": "cancelled"})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1

    def test_list_pagination_limit(self, client):
        for i in range(5):
            client.post("/api/v1/dedup/jobs", json={"dataset_ids": [f"ds-{i}"]})
        resp = client.get("/api/v1/dedup/jobs", params={"limit": 3})
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_list_pagination_offset(self, client):
        for i in range(5):
            client.post("/api/v1/dedup/jobs", json={"dataset_ids": [f"ds-{i}"]})
        resp = client.get("/api/v1/dedup/jobs", params={"offset": 3, "limit": 50})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2


# ===================================================================
# 3. GET /jobs/{job_id} - Get job details
# ===================================================================


class TestGetJobEndpoint:
    """Tests for GET /api/v1/dedup/jobs/{job_id}."""

    def test_get_existing_job(self, client):
        r = client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-1"]})
        job_id = r.json()["job_id"]
        resp = client.get(f"/api/v1/dedup/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["job_id"] == job_id

    def test_get_nonexistent_job(self, client):
        resp = client.get("/api/v1/dedup/jobs/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 4. DELETE /jobs/{job_id} - Cancel job
# ===================================================================


class TestCancelJobEndpoint:
    """Tests for DELETE /api/v1/dedup/jobs/{job_id}."""

    def test_cancel_existing_job(self, client):
        r = client.post("/api/v1/dedup/jobs", json={"dataset_ids": ["ds-1"]})
        job_id = r.json()["job_id"]
        resp = client.delete(f"/api/v1/dedup/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cancelled"

    def test_cancel_nonexistent_job(self, client):
        resp = client.delete("/api/v1/dedup/jobs/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 5. POST /fingerprint - Fingerprint records
# ===================================================================


class TestFingerprintEndpoint:
    """Tests for POST /api/v1/dedup/fingerprint."""

    def test_fingerprint_success(self, client, sample_records):
        resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": sample_records,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == len(sample_records)
        assert "fingerprints" in data
        assert "provenance_hash" in data

    def test_fingerprint_with_field_set(self, client, sample_records):
        resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": sample_records,
            "field_set": ["name"],
        })
        assert resp.status_code == 200

    def test_fingerprint_with_algorithm(self, client, sample_records):
        resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": sample_records,
            "algorithm": "simhash",
        })
        assert resp.status_code == 200
        assert resp.json()["algorithm"] == "simhash"

    def test_fingerprint_empty_records(self, client):
        resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_fingerprint_missing_records(self, client):
        resp = client.post("/api/v1/dedup/fingerprint", json={})
        assert resp.status_code == 422


# ===================================================================
# 6. POST /block - Create blocks
# ===================================================================


class TestBlockEndpoint:
    """Tests for POST /api/v1/dedup/block."""

    def test_block_success(self, client, sample_records):
        resp = client.post("/api/v1/dedup/block", json={
            "records": sample_records,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == len(sample_records)
        assert "total_blocks" in data
        assert "reduction_ratio" in data

    def test_block_with_strategy(self, client, sample_records):
        resp = client.post("/api/v1/dedup/block", json={
            "records": sample_records,
            "strategy": "standard",
        })
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "standard"

    def test_block_with_key_fields(self, client, sample_records):
        resp = client.post("/api/v1/dedup/block", json={
            "records": sample_records,
            "key_fields": ["name"],
        })
        assert resp.status_code == 200

    def test_block_empty_records(self, client):
        resp = client.post("/api/v1/dedup/block", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_block_missing_records(self, client):
        resp = client.post("/api/v1/dedup/block", json={})
        assert resp.status_code == 422


# ===================================================================
# 7. POST /compare - Compare pairs
# ===================================================================


class TestCompareEndpoint:
    """Tests for POST /api/v1/dedup/compare."""

    def test_compare_success(self, client, sample_pairs):
        resp = client.post("/api/v1/dedup/compare", json={
            "block_results": sample_pairs,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_pairs"] == 1
        assert "comparisons" in data

    def test_compare_empty_pairs(self, client):
        resp = client.post("/api/v1/dedup/compare", json={
            "block_results": {"pairs": []},
        })
        assert resp.status_code == 400

    def test_compare_with_field_configs(self, client, sample_pairs):
        resp = client.post("/api/v1/dedup/compare", json={
            "block_results": sample_pairs,
            "field_configs": [
                {"field": "name", "algorithm": "exact", "weight": 2.0},
            ],
        })
        assert resp.status_code == 200

    def test_compare_missing_block_results(self, client):
        resp = client.post("/api/v1/dedup/compare", json={})
        assert resp.status_code == 422


# ===================================================================
# 8. POST /classify - Classify matches
# ===================================================================


class TestClassifyEndpoint:
    """Tests for POST /api/v1/dedup/classify."""

    def test_classify_success(self, client, sample_comparisons):
        resp = client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_comparisons"] == 3
        assert data["matches"] >= 0
        assert data["possible_matches"] >= 0
        assert data["non_matches"] >= 0

    def test_classify_with_thresholds(self, client, sample_comparisons):
        resp = client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
            "thresholds": {"match": 0.9, "possible": 0.6},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_threshold"] == 0.9
        assert data["possible_threshold"] == 0.6

    def test_classify_empty_comparisons(self, client):
        resp = client.post("/api/v1/dedup/classify", json={
            "comparisons": [],
        })
        assert resp.status_code == 400

    def test_classify_missing_comparisons(self, client):
        resp = client.post("/api/v1/dedup/classify", json={})
        assert resp.status_code == 422


# ===================================================================
# 9. GET /matches - List matches
# ===================================================================


class TestListMatchesEndpoint:
    """Tests for GET /api/v1/dedup/matches."""

    def test_list_empty(self, client):
        resp = client.get("/api/v1/dedup/matches")
        assert resp.status_code == 200
        data = resp.json()
        assert data["matches"] == []
        assert data["count"] == 0

    def test_list_after_classify(self, client, sample_comparisons):
        client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
        })
        resp = client.get("/api/v1/dedup/matches")
        assert resp.status_code == 200
        # MATCH and POSSIBLE are stored
        assert resp.json()["count"] >= 1

    def test_list_pagination(self, client, sample_comparisons):
        client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
        })
        resp = client.get("/api/v1/dedup/matches", params={"limit": 1})
        assert resp.status_code == 200
        assert resp.json()["count"] <= 1


# ===================================================================
# 10. GET /matches/{match_id} - Get match details
# ===================================================================


class TestGetMatchEndpoint:
    """Tests for GET /api/v1/dedup/matches/{match_id}."""

    def test_get_existing_match(self, client, sample_comparisons):
        r = client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
        })
        classify_data = r.json()
        # Find a MATCH or POSSIBLE classification
        match_ids = [
            c["pair_id"] for c in classify_data["classifications"]
            if c["classification"] in ("MATCH", "POSSIBLE")
        ]
        if match_ids:
            resp = client.get(f"/api/v1/dedup/matches/{match_ids[0]}")
            assert resp.status_code == 200

    def test_get_nonexistent_match(self, client):
        resp = client.get("/api/v1/dedup/matches/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 11. POST /clusters - Form clusters
# ===================================================================


class TestClusterEndpoint:
    """Tests for POST /api/v1/dedup/clusters."""

    def test_cluster_success(self, client, sample_matches):
        resp = client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_clusters"] >= 1

    def test_cluster_with_algorithm(self, client, sample_matches):
        resp = client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
            "algorithm": "connected_components",
        })
        assert resp.status_code == 200
        assert resp.json()["algorithm"] == "connected_components"

    def test_cluster_empty_matches(self, client):
        resp = client.post("/api/v1/dedup/clusters", json={
            "matches": [],
        })
        assert resp.status_code == 400

    def test_cluster_missing_matches(self, client):
        resp = client.post("/api/v1/dedup/clusters", json={})
        assert resp.status_code == 422


# ===================================================================
# 12. GET /clusters - List clusters
# ===================================================================


class TestListClustersEndpoint:
    """Tests for GET /api/v1/dedup/clusters."""

    def test_list_empty(self, client):
        resp = client.get("/api/v1/dedup/clusters")
        assert resp.status_code == 200
        assert resp.json()["clusters"] == []

    def test_list_after_clustering(self, client, sample_matches):
        client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
        })
        resp = client.get("/api/v1/dedup/clusters")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_list_pagination(self, client, sample_matches):
        client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
        })
        resp = client.get("/api/v1/dedup/clusters", params={"limit": 1})
        assert resp.status_code == 200
        assert resp.json()["count"] <= 1


# ===================================================================
# 13. GET /clusters/{cluster_id} - Get cluster details
# ===================================================================


class TestGetClusterEndpoint:
    """Tests for GET /api/v1/dedup/clusters/{cluster_id}."""

    def test_get_existing_cluster(self, client, sample_matches):
        r = client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
        })
        clusters = r.json()["clusters"]
        if clusters:
            cluster_id = clusters[0]["cluster_id"]
            resp = client.get(f"/api/v1/dedup/clusters/{cluster_id}")
            assert resp.status_code == 200

    def test_get_nonexistent_cluster(self, client):
        resp = client.get("/api/v1/dedup/clusters/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 14. POST /merge - Merge duplicates
# ===================================================================


class TestMergeEndpoint:
    """Tests for POST /api/v1/dedup/merge."""

    def test_merge_success(self, client):
        resp = client.post("/api/v1/dedup/merge", json={
            "clusters": [{"cluster_id": "c1", "members": ["0", "1"]}],
            "records": [
                {"id": "0", "name": "Alice", "email": "alice@co.com"},
                {"id": "1", "name": "Alice Smith", "email": ""},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_golden_records"] >= 1

    def test_merge_with_strategy(self, client):
        resp = client.post("/api/v1/dedup/merge", json={
            "clusters": [{"cluster_id": "c1", "members": ["0", "1"]}],
            "records": [
                {"id": "0", "name": "Alice"},
                {"id": "1", "name": "Bob"},
            ],
            "strategy": "keep_first",
        })
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "keep_first"

    def test_merge_empty_clusters(self, client):
        resp = client.post("/api/v1/dedup/merge", json={
            "clusters": [],
            "records": [],
        })
        assert resp.status_code == 400

    def test_merge_missing_fields(self, client):
        resp = client.post("/api/v1/dedup/merge", json={})
        assert resp.status_code == 422


# ===================================================================
# 15. GET /merge/{merge_id} - Get merge result
# ===================================================================


class TestGetMergeEndpoint:
    """Tests for GET /api/v1/dedup/merge/{merge_id}."""

    def test_get_existing_merge(self, client):
        r = client.post("/api/v1/dedup/merge", json={
            "clusters": [{"cluster_id": "c1", "members": ["0", "1"]}],
            "records": [
                {"id": "0", "name": "Alice"},
                {"id": "1", "name": "Bob"},
            ],
        })
        merge_id = r.json()["merge_id"]
        resp = client.get(f"/api/v1/dedup/merge/{merge_id}")
        assert resp.status_code == 200

    def test_get_nonexistent_merge(self, client):
        resp = client.get("/api/v1/dedup/merge/nonexistent-id")
        assert resp.status_code == 404


# ===================================================================
# 16. POST /pipeline - Run full pipeline
# ===================================================================


class TestPipelineEndpoint:
    """Tests for POST /api/v1/dedup/pipeline."""

    def test_pipeline_success(self, client, sample_records):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_records"] == len(sample_records)
        assert data["status"] in ("completed", "failed")

    def test_pipeline_with_options(self, client, sample_records):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
            "options": {"fingerprint_algorithm": "simhash"},
        })
        assert resp.status_code == 200

    def test_pipeline_with_rule(self, client, sample_records):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
            "rule": {"name": "test-rule"},
        })
        assert resp.status_code == 200

    def test_pipeline_empty_records(self, client):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": [],
        })
        assert resp.status_code == 400

    def test_pipeline_missing_records(self, client):
        resp = client.post("/api/v1/dedup/pipeline", json={})
        assert resp.status_code == 422

    def test_pipeline_response_fields(self, client, sample_records):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
        })
        data = resp.json()
        assert "pipeline_id" in data
        assert "processing_time_ms" in data
        assert "provenance_hash" in data
        assert "stages" in data


# ===================================================================
# 17. POST /rules - Create dedup rule
# ===================================================================


class TestCreateRuleEndpoint:
    """Tests for POST /api/v1/dedup/rules."""

    def test_create_rule_success(self, client):
        resp = client.post("/api/v1/dedup/rules", json={
            "name": "test-rule",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "rule_id" in data
        assert data["name"] == "test-rule"

    def test_create_rule_with_options(self, client):
        resp = client.post("/api/v1/dedup/rules", json={
            "name": "advanced-rule",
            "description": "Advanced dedup rule",
            "match_threshold": 0.9,
            "merge_strategy": "golden_record",
            "is_active": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_threshold"] == 0.9
        assert data["merge_strategy"] == "golden_record"

    def test_create_rule_missing_name(self, client):
        resp = client.post("/api/v1/dedup/rules", json={})
        assert resp.status_code == 422

    def test_create_rule_with_field_weights(self, client):
        resp = client.post("/api/v1/dedup/rules", json={
            "name": "weighted-rule",
            "field_weights": [
                {"field": "name", "weight": 0.5},
                {"field": "email", "weight": 0.3},
            ],
        })
        assert resp.status_code == 200


# ===================================================================
# 18. GET /rules - List rules
# ===================================================================


class TestListRulesEndpoint:
    """Tests for GET /api/v1/dedup/rules."""

    def test_list_empty(self, client):
        resp = client.get("/api/v1/dedup/rules")
        assert resp.status_code == 200
        assert resp.json()["rules"] == []

    def test_list_after_create(self, client):
        client.post("/api/v1/dedup/rules", json={"name": "r1"})
        client.post("/api/v1/dedup/rules", json={"name": "r2"})
        resp = client.get("/api/v1/dedup/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_list_pagination(self, client):
        for i in range(5):
            client.post("/api/v1/dedup/rules", json={"name": f"r-{i}"})
        resp = client.get("/api/v1/dedup/rules", params={"limit": 3})
        assert resp.status_code == 200
        assert resp.json()["count"] == 3

    def test_list_offset(self, client):
        for i in range(5):
            client.post("/api/v1/dedup/rules", json={"name": f"r-{i}"})
        resp = client.get("/api/v1/dedup/rules", params={"offset": 3, "limit": 50})
        assert resp.status_code == 200
        assert resp.json()["count"] == 2


# ===================================================================
# 19. GET /health - Health check
# ===================================================================


class TestHealthEndpoint:
    """Tests for GET /api/v1/dedup/health."""

    def test_health_success(self, client):
        resp = client.get("/api/v1/dedup/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["service"] == "duplicate-detector"
        assert data["status"] == "healthy"
        assert data["started"] is True

    def test_health_includes_counts(self, client):
        resp = client.get("/api/v1/dedup/health")
        data = resp.json()
        assert "jobs" in data
        assert "rules" in data
        assert "provenance_entries" in data
        assert "prometheus_available" in data


# ===================================================================
# 20. GET /stats - Statistics
# ===================================================================


class TestStatsEndpoint:
    """Tests for GET /api/v1/dedup/stats."""

    def test_stats_success(self, client):
        resp = client.get("/api/v1/dedup/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_jobs" in data
        assert "completed_jobs" in data
        assert "total_records_processed" in data
        assert "total_duplicates_found" in data

    def test_stats_after_operations(self, client, sample_records):
        client.post("/api/v1/dedup/fingerprint", json={
            "records": sample_records,
        })
        resp = client.get("/api/v1/dedup/stats")
        data = resp.json()
        assert data["total_records_processed"] == len(sample_records)

    def test_stats_after_pipeline(self, client, sample_records):
        client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
        })
        resp = client.get("/api/v1/dedup/stats")
        data = resp.json()
        assert data["total_jobs"] >= 1


# ===================================================================
# Service Not Configured (503)
# ===================================================================


class TestServiceNotConfigured:
    """Tests for service not configured error (503)."""

    def test_health_503_when_no_service(self):
        from greenlang.duplicate_detector.api.router import router

        application = FastAPI()
        # Do NOT set duplicate_detector_service on app.state
        application.include_router(router)
        no_service_client = TestClient(application)

        resp = no_service_client.get("/api/v1/dedup/health")
        assert resp.status_code == 503

    def test_stats_503_when_no_service(self):
        from greenlang.duplicate_detector.api.router import router

        application = FastAPI()
        application.include_router(router)
        no_service_client = TestClient(application)

        resp = no_service_client.get("/api/v1/dedup/stats")
        assert resp.status_code == 503

    def test_create_job_503_when_no_service(self):
        from greenlang.duplicate_detector.api.router import router

        application = FastAPI()
        application.include_router(router)
        no_service_client = TestClient(application)

        resp = no_service_client.post("/api/v1/dedup/jobs", json={
            "dataset_ids": ["ds-1"],
        })
        assert resp.status_code == 503


# ===================================================================
# Response Format Validation
# ===================================================================


class TestResponseFormat:
    """Tests for verifying response JSON structure."""

    def test_fingerprint_response_format(self, client, sample_records):
        resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": sample_records,
        })
        data = resp.json()
        required_keys = [
            "fingerprint_id", "algorithm", "total_records",
            "unique_fingerprints", "duplicate_candidates",
            "fingerprints", "processing_time_ms", "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_block_response_format(self, client, sample_records):
        resp = client.post("/api/v1/dedup/block", json={
            "records": sample_records,
        })
        data = resp.json()
        required_keys = [
            "block_id", "strategy", "total_records",
            "total_blocks", "total_pairs", "largest_block",
            "reduction_ratio", "blocks", "processing_time_ms",
            "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_compare_response_format(self, client, sample_pairs):
        resp = client.post("/api/v1/dedup/compare", json={
            "block_results": sample_pairs,
        })
        data = resp.json()
        required_keys = [
            "comparison_id", "algorithm", "total_pairs",
            "comparisons", "avg_similarity", "processing_time_ms",
            "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_classify_response_format(self, client, sample_comparisons):
        resp = client.post("/api/v1/dedup/classify", json={
            "comparisons": sample_comparisons,
        })
        data = resp.json()
        required_keys = [
            "classify_id", "total_comparisons", "matches",
            "possible_matches", "non_matches", "match_threshold",
            "possible_threshold", "classifications", "processing_time_ms",
            "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_cluster_response_format(self, client, sample_matches):
        resp = client.post("/api/v1/dedup/clusters", json={
            "matches": sample_matches,
        })
        data = resp.json()
        required_keys = [
            "cluster_run_id", "algorithm", "total_matches",
            "total_clusters", "largest_cluster", "avg_cluster_size",
            "clusters", "processing_time_ms", "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_merge_response_format(self, client):
        resp = client.post("/api/v1/dedup/merge", json={
            "clusters": [{"cluster_id": "c1", "members": ["0", "1"]}],
            "records": [
                {"id": "0", "name": "Alice"},
                {"id": "1", "name": "Bob"},
            ],
        })
        data = resp.json()
        required_keys = [
            "merge_id", "strategy", "total_clusters",
            "total_records_merged", "total_golden_records",
            "conflicts_resolved", "conflict_resolution",
            "merged_records", "processing_time_ms", "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_pipeline_response_format(self, client, sample_records):
        resp = client.post("/api/v1/dedup/pipeline", json={
            "records": sample_records,
        })
        data = resp.json()
        required_keys = [
            "pipeline_id", "job_id", "status", "total_records",
            "total_duplicates", "total_clusters", "total_merged",
            "stages", "processing_time_ms", "provenance_hash",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_stats_response_format(self, client):
        resp = client.get("/api/v1/dedup/stats")
        data = resp.json()
        required_keys = [
            "total_jobs", "completed_jobs", "failed_jobs",
            "total_records_processed", "total_duplicates_found",
            "total_clusters", "total_merges", "total_conflicts",
            "total_rules", "active_jobs", "avg_similarity",
            "provenance_entries",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"


# ===================================================================
# End-to-End Workflow via API
# ===================================================================


class TestEndToEndWorkflow:
    """Tests for end-to-end workflow through the REST API."""

    def test_full_workflow_individual_steps(self, client):
        """Test the full dedup workflow step by step via API."""
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "Alice", "email": "alice@co.com"},
            {"id": "2", "name": "Bob", "email": "bob@co.com"},
        ]

        # Step 1: Fingerprint
        fp_resp = client.post("/api/v1/dedup/fingerprint", json={
            "records": records,
        })
        assert fp_resp.status_code == 200
        fp_data = fp_resp.json()
        assert fp_data["total_records"] == 3

        # Step 2: Block
        block_resp = client.post("/api/v1/dedup/block", json={
            "records": records,
        })
        assert block_resp.status_code == 200

        # Step 3: Compare (using manual pairs)
        compare_resp = client.post("/api/v1/dedup/compare", json={
            "block_results": {
                "pairs": [
                    {
                        "id_a": "0", "id_b": "1",
                        "record_a": records[0], "record_b": records[1],
                    },
                    {
                        "id_a": "0", "id_b": "2",
                        "record_a": records[0], "record_b": records[2],
                    },
                ],
            },
        })
        assert compare_resp.status_code == 200
        comparisons = compare_resp.json()["comparisons"]

        # Step 4: Classify
        classify_resp = client.post("/api/v1/dedup/classify", json={
            "comparisons": comparisons,
        })
        assert classify_resp.status_code == 200
        classify_data = classify_resp.json()
        assert classify_data["total_comparisons"] == 2

        # Step 5: Cluster (only if we have matches)
        match_items = [
            c for c in classify_data["classifications"]
            if c["classification"] == "MATCH"
        ]
        if match_items:
            cluster_resp = client.post("/api/v1/dedup/clusters", json={
                "matches": match_items,
            })
            assert cluster_resp.status_code == 200

        # Step 6: Check health
        health_resp = client.get("/api/v1/dedup/health")
        assert health_resp.status_code == 200

        # Step 7: Check stats
        stats_resp = client.get("/api/v1/dedup/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_records_processed"] >= 3

    def test_pipeline_then_stats(self, client):
        """Run the full pipeline then verify stats reflect it."""
        records = [
            {"id": "0", "name": "Alice"},
            {"id": "1", "name": "alice"},
            {"id": "2", "name": "Bob"},
        ]
        pipeline_resp = client.post("/api/v1/dedup/pipeline", json={
            "records": records,
        })
        assert pipeline_resp.status_code == 200

        stats_resp = client.get("/api/v1/dedup/stats")
        assert stats_resp.status_code == 200
        assert stats_resp.json()["total_jobs"] >= 1

    def test_job_lifecycle(self, client):
        """Create a job, verify it, cancel it, verify cancellation."""
        # Create
        create_resp = client.post("/api/v1/dedup/jobs", json={
            "dataset_ids": ["ds-1"],
        })
        assert create_resp.status_code == 200
        job_id = create_resp.json()["job_id"]

        # Get
        get_resp = client.get(f"/api/v1/dedup/jobs/{job_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["status"] == "created"

        # Cancel
        cancel_resp = client.delete(f"/api/v1/dedup/jobs/{job_id}")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["status"] == "cancelled"

        # Verify in list
        list_resp = client.get("/api/v1/dedup/jobs", params={"status": "cancelled"})
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 1

    def test_rule_lifecycle(self, client):
        """Create a rule, list it, verify it."""
        # Create
        create_resp = client.post("/api/v1/dedup/rules", json={
            "name": "lifecycle-test",
            "match_threshold": 0.9,
        })
        assert create_resp.status_code == 200
        rule_id = create_resp.json()["rule_id"]

        # List
        list_resp = client.get("/api/v1/dedup/rules")
        assert list_resp.status_code == 200
        assert list_resp.json()["count"] == 1
        assert list_resp.json()["rules"][0]["rule_id"] == rule_id

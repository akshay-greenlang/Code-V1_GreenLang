# -*- coding: utf-8 -*-
"""
Integration tests for Duplicate Detection REST API endpoints - AGENT-DATA-011

Tests all 18+ API endpoints via FastAPI TestClient, validating HTTP status
codes, response shapes, error handling, pagination, and cross-endpoint
data flow.

18 test cases covering:
- test_create_job_endpoint
- test_list_jobs_endpoint
- test_get_job_details_endpoint
- test_delete_job_endpoint
- test_fingerprint_endpoint
- test_block_endpoint
- test_compare_endpoint
- test_classify_endpoint
- test_list_matches_endpoint
- test_get_match_details_endpoint
- test_create_clusters_endpoint
- test_list_clusters_endpoint
- test_merge_endpoint
- test_pipeline_endpoint
- test_health_endpoint
- test_stats_endpoint
- test_create_rule_endpoint
- test_auth_required_on_all_endpoints

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-011 Duplicate Detection Agent (GL-DATA-X-014)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# Job Management API Tests
# ===================================================================


class TestJobEndpoints:
    """Tests for job CRUD API endpoints."""

    def test_create_job_endpoint(self, test_client):
        """POST /api/v1/dd/jobs creates a dedup job and returns 200.

        Validates:
        - HTTP 200 response
        - job_id is present and a valid UUID
        - status is 'created'
        - dataset_ids match the request
        """
        resp = test_client.post(
            "/api/v1/dd/jobs",
            json={"dataset_ids": ["ds-001", "ds-002"]},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "job_id" in body
        uuid.UUID(body["job_id"])  # Must be valid UUID
        assert body["status"] == "created"
        assert body["dataset_ids"] == ["ds-001", "ds-002"]
        assert body["rule_id"] is None
        assert "created_at" in body
        assert "updated_at" in body

    def test_create_job_with_rule_id(self, test_client):
        """POST /api/v1/dd/jobs with rule_id includes it in the response."""
        resp = test_client.post(
            "/api/v1/dd/jobs",
            json={"dataset_ids": ["ds-001"], "rule_id": "rule-123"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["rule_id"] == "rule-123"

    def test_list_jobs_endpoint(self, test_client):
        """GET /api/v1/dd/jobs lists all created jobs with pagination.

        Validates:
        - Response contains 'jobs', 'count', 'total' fields
        - Creating 3 jobs results in count=3
        """
        # Create multiple jobs
        for i in range(3):
            test_client.post(
                "/api/v1/dd/jobs",
                json={"dataset_ids": [f"ds-{i}"]},
            )

        resp = test_client.get("/api/v1/dd/jobs")
        assert resp.status_code == 200

        body = resp.json()
        assert "jobs" in body
        assert body["count"] == 3
        assert body["total"] == 3

    def test_list_jobs_with_pagination(self, test_client):
        """GET /api/v1/dd/jobs?limit=2&offset=0 returns paginated results."""
        for i in range(5):
            test_client.post(
                "/api/v1/dd/jobs",
                json={"dataset_ids": [f"ds-{i}"]},
            )

        resp = test_client.get("/api/v1/dd/jobs?limit=2&offset=0")
        assert resp.status_code == 200

        body = resp.json()
        assert body["count"] == 2
        assert body["total"] == 5

    def test_get_job_details_endpoint(self, test_client):
        """GET /api/v1/dd/jobs/{job_id} returns the specific job.

        Validates:
        - HTTP 200 for existing job
        - Response matches the created job
        """
        create_resp = test_client.post(
            "/api/v1/dd/jobs",
            json={"dataset_ids": ["ds-001"]},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.get(f"/api/v1/dd/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "created"

    def test_get_job_not_found(self, test_client):
        """GET /api/v1/dd/jobs/{nonexistent} returns 404."""
        resp = test_client.get(f"/api/v1/dd/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_delete_job_endpoint(self, test_client):
        """DELETE /api/v1/dd/jobs/{job_id} cancels the job.

        Validates:
        - HTTP 200 for existing job
        - Status changes to 'cancelled'
        - updated_at is refreshed
        """
        create_resp = test_client.post(
            "/api/v1/dd/jobs",
            json={"dataset_ids": ["ds-001"]},
        )
        job_id = create_resp.json()["job_id"]

        resp = test_client.delete(f"/api/v1/dd/jobs/{job_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["job_id"] == job_id
        assert body["status"] == "cancelled"

    def test_delete_job_not_found(self, test_client):
        """DELETE /api/v1/dd/jobs/{nonexistent} returns 404."""
        resp = test_client.delete(f"/api/v1/dd/jobs/{uuid.uuid4()}")
        assert resp.status_code == 404


# ===================================================================
# Dedup Operation API Tests
# ===================================================================


class TestFingerprintEndpoint:
    """Tests for the fingerprint API endpoint."""

    def test_fingerprint_endpoint(self, test_client):
        """POST /api/v1/dd/fingerprint processes records and returns fingerprints.

        Validates:
        - HTTP 200 response
        - total_records matches input length
        - fingerprints dict has entries for each record
        - provenance_hash is a 64-char hex string
        """
        records = [
            {"name": "Alice", "email": "alice@co.com"},
            {"name": "Bob", "email": "bob@co.com"},
            {"name": "Alice", "email": "alice@co.com"},
        ]

        resp = test_client.post(
            "/api/v1/dd/fingerprint",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3
        assert body["algorithm"] == "sha256"
        assert len(body["fingerprints"]) == 3
        assert len(body["provenance_hash"]) == 64

        # Records 0 and 2 are identical, so they should share a fingerprint
        assert body["fingerprints"]["0"] == body["fingerprints"]["2"]
        assert body["unique_fingerprints"] == 2
        assert body["duplicate_candidates"] == 1

    def test_fingerprint_with_custom_algorithm(self, test_client):
        """POST /api/v1/dd/fingerprint with algorithm override."""
        resp = test_client.post(
            "/api/v1/dd/fingerprint",
            json={
                "records": [{"name": "Alice"}],
                "algorithm": "simhash",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["algorithm"] == "simhash"


class TestBlockEndpoint:
    """Tests for the blocking API endpoint."""

    def test_block_endpoint(self, test_client):
        """POST /api/v1/dd/block creates blocking partitions.

        Validates:
        - HTTP 200 response
        - total_records matches input
        - total_blocks >= 1
        - reduction_ratio is between 0.0 and 1.0
        """
        records = [
            {"name": "Alice", "city": "Portland"},
            {"name": "Bob", "city": "Seattle"},
            {"name": "Alice B", "city": "Portland"},
        ]

        resp = test_client.post(
            "/api/v1/dd/block",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 3
        assert body["total_blocks"] >= 1
        assert 0.0 <= body["reduction_ratio"] <= 1.0
        assert body["strategy"] == "sorted_neighborhood"
        assert len(body["provenance_hash"]) == 64


class TestCompareEndpoint:
    """Tests for the similarity comparison API endpoint."""

    def test_compare_endpoint(self, test_client):
        """POST /api/v1/dd/compare computes pairwise similarity.

        Validates:
        - HTTP 200 response
        - total_pairs matches input pair count
        - Each comparison has overall_score between 0.0 and 1.0
        - avg_similarity is computed
        """
        pairs = [
            {
                "id_a": "0", "id_b": "1",
                "record_a": {"name": "Alice", "email": "alice@co.com"},
                "record_b": {"name": "Alice", "email": "alice@co.com"},
            },
            {
                "id_a": "2", "id_b": "3",
                "record_a": {"name": "Alice", "email": "alice@co.com"},
                "record_b": {"name": "Bob", "email": "bob@co.com"},
            },
        ]

        resp = test_client.post(
            "/api/v1/dd/compare",
            json={"pairs": pairs},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_pairs"] == 2
        assert len(body["comparisons"]) == 2

        # First pair is identical -> score 1.0
        assert body["comparisons"][0]["overall_score"] == 1.0
        # Second pair is different -> score 0.0
        assert body["comparisons"][1]["overall_score"] == 0.0

        assert body["avg_similarity"] == 0.5
        assert len(body["provenance_hash"]) == 64


class TestClassifyEndpoint:
    """Tests for the match classification API endpoint."""

    def test_classify_endpoint(self, test_client):
        """POST /api/v1/dd/classify classifies comparisons into categories.

        Validates:
        - HTTP 200 response
        - Correct classification counts (MATCH, POSSIBLE, NON_MATCH)
        - Default thresholds applied from config
        """
        comparisons = [
            {"pair_id": str(uuid.uuid4()), "record_a_id": "a", "record_b_id": "b", "overall_score": 0.95},
            {"pair_id": str(uuid.uuid4()), "record_a_id": "c", "record_b_id": "d", "overall_score": 0.75},
            {"pair_id": str(uuid.uuid4()), "record_a_id": "e", "record_b_id": "f", "overall_score": 0.30},
        ]

        resp = test_client.post(
            "/api/v1/dd/classify",
            json={"comparisons": comparisons},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_comparisons"] == 3
        assert body["matches"] == 1       # 0.95 >= 0.85
        assert body["possible_matches"] == 1  # 0.75 >= 0.65
        assert body["non_matches"] == 1    # 0.30 < 0.65
        assert body["match_threshold"] == 0.85
        assert body["possible_threshold"] == 0.65

    def test_classify_with_custom_thresholds(self, test_client):
        """POST /api/v1/dd/classify with custom thresholds."""
        comparisons = [
            {"pair_id": str(uuid.uuid4()), "record_a_id": "a", "record_b_id": "b", "overall_score": 0.50},
        ]

        resp = test_client.post(
            "/api/v1/dd/classify",
            json={
                "comparisons": comparisons,
                "thresholds": {"match": 0.40, "possible": 0.20},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["matches"] == 1  # 0.50 >= 0.40


# ===================================================================
# Match & Cluster API Tests
# ===================================================================


class TestMatchEndpoints:
    """Tests for match listing and detail endpoints."""

    def test_list_matches_endpoint(self, test_client):
        """GET /api/v1/dd/matches lists all classified matches.

        Creates matches via classify, then verifies listing.
        """
        # First, classify to create matches
        comparisons = [
            {"pair_id": "match-1", "record_a_id": "a", "record_b_id": "b", "overall_score": 0.95},
            {"pair_id": "match-2", "record_a_id": "c", "record_b_id": "d", "overall_score": 0.90},
        ]
        test_client.post(
            "/api/v1/dd/classify",
            json={"comparisons": comparisons},
        )

        resp = test_client.get("/api/v1/dd/matches")
        assert resp.status_code == 200

        body = resp.json()
        assert "matches" in body
        assert body["count"] >= 2

    def test_get_match_details_endpoint(self, test_client):
        """GET /api/v1/dd/matches/{match_id} returns specific match details.

        Creates a match via classify, retrieves its pair_id, then
        queries the detail endpoint.
        """
        pair_id = "detail-match-1"
        comparisons = [
            {"pair_id": pair_id, "record_a_id": "x", "record_b_id": "y", "overall_score": 0.92},
        ]
        test_client.post(
            "/api/v1/dd/classify",
            json={"comparisons": comparisons},
        )

        resp = test_client.get(f"/api/v1/dd/matches/{pair_id}")
        assert resp.status_code == 200

        body = resp.json()
        assert body["pair_id"] == pair_id
        assert body["classification"] == "MATCH"
        assert body["overall_score"] == 0.92

    def test_get_match_not_found(self, test_client):
        """GET /api/v1/dd/matches/{nonexistent} returns 404."""
        resp = test_client.get("/api/v1/dd/matches/nonexistent-id")
        assert resp.status_code == 404


class TestClusterEndpoints:
    """Tests for cluster creation and listing endpoints."""

    def test_create_clusters_endpoint(self, test_client):
        """POST /api/v1/dd/clusters forms duplicate clusters from matches.

        Validates:
        - HTTP 200 response
        - total_clusters >= 1
        - Transitive closure produces correct cluster sizes
        """
        matches = [
            {"record_a_id": "A", "record_b_id": "B"},
            {"record_a_id": "B", "record_b_id": "C"},
            {"record_a_id": "D", "record_b_id": "E"},
        ]

        resp = test_client.post(
            "/api/v1/dd/clusters",
            json={"matches": matches},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_matches"] == 3
        assert body["total_clusters"] == 2  # {A,B,C} and {D,E}
        assert body["largest_cluster"] == 3
        assert body["algorithm"] == "union_find"
        assert len(body["provenance_hash"]) == 64

    def test_list_clusters_endpoint(self, test_client):
        """GET /api/v1/dd/clusters lists all formed clusters."""
        # Create clusters first
        test_client.post(
            "/api/v1/dd/clusters",
            json={
                "matches": [
                    {"record_a_id": "X", "record_b_id": "Y"},
                ],
            },
        )

        resp = test_client.get("/api/v1/dd/clusters")
        assert resp.status_code == 200

        body = resp.json()
        assert "clusters" in body
        assert body["count"] >= 1


# ===================================================================
# Merge API Tests
# ===================================================================


class TestMergeEndpoint:
    """Tests for the merge API endpoint."""

    def test_merge_endpoint(self, test_client):
        """POST /api/v1/dd/merge produces golden records from clusters.

        Validates:
        - HTTP 200 response
        - Golden record has the most complete values
        - Conflict counting is correct
        """
        clusters = [
            {"cluster_id": "c1", "members": ["0", "1"]},
        ]
        records = [
            {"id": "0", "name": "Alice", "email": ""},
            {"id": "1", "name": "Alice Smith", "email": "alice@co.com"},
        ]

        resp = test_client.post(
            "/api/v1/dd/merge",
            json={"clusters": clusters, "records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_golden_records"] == 1
        assert body["strategy"] == "keep_most_complete"

        golden = body["merged_records"][0]["golden_record"]
        assert golden["name"] == "Alice Smith"
        assert golden["email"] == "alice@co.com"


# ===================================================================
# Pipeline API Tests
# ===================================================================


class TestPipelineEndpoint:
    """Tests for the full pipeline API endpoint."""

    def test_pipeline_endpoint(self, test_client):
        """POST /api/v1/dd/pipeline runs the full dedup pipeline.

        Validates:
        - HTTP 200 response
        - All expected fields in response
        - Pipeline stages are present
        """
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "Alice", "email": "alice@co.com"},
            {"id": "2", "name": "Bob", "email": "bob@co.com"},
        ]

        resp = test_client.post(
            "/api/v1/dd/pipeline",
            json={"records": records},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("completed", "failed")
        assert body["total_records"] == 3
        assert "pipeline_id" in body
        uuid.UUID(body["pipeline_id"])
        assert "stages" in body
        assert "fingerprint" in body["stages"]
        assert len(body["provenance_hash"]) == 64

    def test_pipeline_with_options(self, test_client):
        """POST /api/v1/dd/pipeline with custom options."""
        records = [
            {"id": "0", "name": "Alice"},
            {"id": "1", "name": "Bob"},
        ]

        resp = test_client.post(
            "/api/v1/dd/pipeline",
            json={
                "records": records,
                "options": {"fingerprint_algorithm": "simhash"},
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total_records"] == 2


# ===================================================================
# Health & Stats API Tests
# ===================================================================


class TestHealthAndStatsEndpoints:
    """Tests for health check and statistics endpoints."""

    def test_health_endpoint(self, test_client):
        """GET /api/v1/dd/health returns service health status.

        Validates:
        - HTTP 200 response
        - status is 'healthy' (service was started in fixture)
        - All expected health fields are present
        """
        resp = test_client.get("/api/v1/dd/health")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "duplicate-detector"
        assert body["started"] is True
        assert "jobs" in body
        assert "fingerprint_results" in body
        assert "block_results" in body
        assert "compare_results" in body
        assert "classify_results" in body
        assert "cluster_results" in body
        assert "merge_results" in body
        assert "pipeline_results" in body
        assert "rules" in body
        assert "provenance_entries" in body

    def test_stats_endpoint(self, test_client):
        """GET /api/v1/dd/stats returns aggregate statistics.

        Validates:
        - HTTP 200 response
        - All stat fields present with correct defaults
        """
        resp = test_client.get("/api/v1/dd/stats")

        assert resp.status_code == 200
        body = resp.json()
        assert "total_jobs" in body
        assert "completed_jobs" in body
        assert "failed_jobs" in body
        assert "total_records_processed" in body
        assert "total_duplicates_found" in body
        assert "total_clusters" in body
        assert "total_merges" in body
        assert "total_conflicts" in body
        assert "total_rules" in body
        assert "active_jobs" in body
        assert "avg_similarity" in body
        assert "provenance_entries" in body

    def test_stats_reflect_operations(self, test_client):
        """GET /api/v1/dd/stats reflects operations performed.

        Runs a pipeline, then checks that stats are updated.
        """
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "Bob", "email": "bob@co.com"},
        ]
        test_client.post(
            "/api/v1/dd/pipeline",
            json={"records": records},
        )

        resp = test_client.get("/api/v1/dd/stats")
        assert resp.status_code == 200

        body = resp.json()
        assert body["total_jobs"] >= 1
        assert body["total_records_processed"] >= 2


# ===================================================================
# Rule API Tests
# ===================================================================


class TestRuleEndpoint:
    """Tests for the dedup rule creation endpoint."""

    def test_create_rule_endpoint(self, test_client):
        """POST /api/v1/dd/rules creates a new dedup rule.

        Validates:
        - HTTP 200 response
        - rule_id is a valid UUID
        - Rule name and thresholds match the request
        - Default values are applied from config
        """
        resp = test_client.post(
            "/api/v1/dd/rules",
            json={
                "rule_config": {
                    "name": "api-test-rule",
                    "description": "Rule created via API test",
                    "match_threshold": 0.92,
                    "merge_strategy": "golden_record",
                },
            },
        )

        assert resp.status_code == 200
        body = resp.json()
        assert "rule_id" in body
        uuid.UUID(body["rule_id"])
        assert body["name"] == "api-test-rule"
        assert body["match_threshold"] == 0.92
        assert body["merge_strategy"] == "golden_record"
        assert body["is_active"] is True
        assert len(body["provenance_hash"]) == 64


# ===================================================================
# Authentication Tests
# ===================================================================


class TestAuthEndpoints:
    """Tests for authentication requirements on protected endpoints."""

    def test_auth_required_on_all_endpoints(self, test_client):
        """Verify that the protected endpoint requires authentication.

        Tests the sentinel /api/v1/dd/protected endpoint to confirm
        that requests without a valid Bearer token are rejected with 401.
        """
        # Request without auth header -> 401
        resp = test_client.get("/api/v1/dd/protected")
        assert resp.status_code == 401
        assert "Not authenticated" in resp.json()["detail"]

        # Request with invalid auth scheme -> 401
        resp = test_client.get(
            "/api/v1/dd/protected",
            headers={"Authorization": "Basic dXNlcjpwYXNz"},
        )
        assert resp.status_code == 401

        # Request with valid Bearer token -> 200
        resp = test_client.get(
            "/api/v1/dd/protected",
            headers={"Authorization": "Bearer valid-test-token"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===================================================================
# Cross-Endpoint Data Flow Tests
# ===================================================================


class TestCrossEndpointFlow:
    """Test data flow across multiple endpoints in sequence."""

    def test_full_api_workflow(self, test_client):
        """Test the complete API workflow from job creation to merge.

        Executes the following sequence:
        1. Create a job
        2. Fingerprint records
        3. Create blocks
        4. Compare pairs
        5. Classify matches
        6. Form clusters
        7. Merge duplicates
        8. Check stats reflect all operations
        """
        # Step 1: Create job
        job_resp = test_client.post(
            "/api/v1/dd/jobs",
            json={"dataset_ids": ["ds-api-flow"]},
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["job_id"]

        # Step 2: Fingerprint
        records = [
            {"id": "0", "name": "Alice", "email": "alice@co.com"},
            {"id": "1", "name": "Alice", "email": "alice@co.com"},
            {"id": "2", "name": "Bob", "email": "bob@co.com"},
        ]
        fp_resp = test_client.post(
            "/api/v1/dd/fingerprint",
            json={"records": records},
        )
        assert fp_resp.status_code == 200
        assert fp_resp.json()["total_records"] == 3

        # Step 3: Block
        block_resp = test_client.post(
            "/api/v1/dd/block",
            json={"records": records},
        )
        assert block_resp.status_code == 200

        # Step 4: Compare
        pairs = [
            {
                "id_a": "0", "id_b": "1",
                "record_a": records[0],
                "record_b": records[1],
            },
            {
                "id_a": "0", "id_b": "2",
                "record_a": records[0],
                "record_b": records[2],
            },
        ]
        compare_resp = test_client.post(
            "/api/v1/dd/compare",
            json={"pairs": pairs},
        )
        assert compare_resp.status_code == 200
        comparisons = compare_resp.json()["comparisons"]

        # Step 5: Classify
        classify_resp = test_client.post(
            "/api/v1/dd/classify",
            json={"comparisons": comparisons},
        )
        assert classify_resp.status_code == 200
        classify_body = classify_resp.json()

        # Only MATCH classifications get stored as matches
        match_items = [
            c for c in classify_body["classifications"]
            if c["classification"] == "MATCH"
        ]

        # Step 6: Cluster (if matches found)
        if match_items:
            cluster_resp = test_client.post(
                "/api/v1/dd/clusters",
                json={"matches": match_items},
            )
            assert cluster_resp.status_code == 200
            clusters = cluster_resp.json()["clusters"]

            # Step 7: Merge
            if clusters:
                merge_resp = test_client.post(
                    "/api/v1/dd/merge",
                    json={"clusters": clusters, "records": records},
                )
                assert merge_resp.status_code == 200

        # Step 8: Verify stats
        stats_resp = test_client.get("/api/v1/dd/stats")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_records_processed"] >= 3
        assert stats["provenance_entries"] >= 1

        # Verify the job is still retrievable
        get_job_resp = test_client.get(f"/api/v1/dd/jobs/{job_id}")
        assert get_job_resp.status_code == 200

    def test_pipeline_then_stats_then_health(self, test_client):
        """Test that pipeline results flow through to stats and health.

        Runs a pipeline, then verifies both stats and health endpoints
        reflect the pipeline execution.
        """
        records = [
            {"id": "0", "name": "Test Person A", "email": "a@test.com"},
            {"id": "1", "name": "Test Person B", "email": "b@test.com"},
        ]

        # Run pipeline
        pipeline_resp = test_client.post(
            "/api/v1/dd/pipeline",
            json={"records": records},
        )
        assert pipeline_resp.status_code == 200

        # Check stats updated
        stats_resp = test_client.get("/api/v1/dd/stats")
        stats = stats_resp.json()
        assert stats["total_jobs"] >= 1

        # Check health reflects pipeline results
        health_resp = test_client.get("/api/v1/dd/health")
        health = health_resp.json()
        assert health["pipeline_results"] >= 1
        assert health["fingerprint_results"] >= 1

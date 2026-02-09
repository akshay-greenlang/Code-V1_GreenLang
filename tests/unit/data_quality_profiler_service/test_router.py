# -*- coding: utf-8 -*-
"""
Unit Tests for Data Quality Profiler REST API Router (AGENT-DATA-010)
=====================================================================

Comprehensive test suite for ``greenlang.data_quality_profiler.api.router``
covering all 20 endpoints, service-not-configured 503 responses,
and a full end-to-end API workflow.

Target: 95+ tests with synchronous ``starlette.testclient.TestClient``.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from greenlang.data_quality_profiler.config import (
    DataQualityProfilerConfig,
)
from greenlang.data_quality_profiler.setup import (
    DataQualityProfilerService,
)
from greenlang.data_quality_profiler.api.router import router


# ===================================================================
# Fixtures
# ===================================================================


def _make_config(**overrides: Any) -> DataQualityProfilerConfig:
    """Create a DataQualityProfilerConfig with optional overrides."""
    defaults = dict(
        database_url="",
        redis_url="",
        s3_bucket_url="",
        min_samples_for_anomaly=3,
    )
    defaults.update(overrides)
    return DataQualityProfilerConfig(**defaults)


def _sample_rows(n: int = 10) -> List[Dict[str, Any]]:
    """Generate simple row data."""
    return [
        {"name": f"item_{i}", "value": i * 10, "score": 80.0 + i}
        for i in range(n)
    ]


def _numeric_rows(n: int = 20) -> List[Dict[str, Any]]:
    """Generate numeric-only data."""
    return [{"x": float(i), "y": float(i * 2)} for i in range(n)]


def _rows_with_outliers(n: int = 30) -> List[Dict[str, Any]]:
    """Data with clear outliers."""
    rows = [{"metric": float(i)} for i in range(n)]
    rows.append({"metric": 99999.0})
    rows.append({"metric": -99999.0})
    return rows


def _fresh_ts() -> str:
    """Return an ISO timestamp for now."""
    return datetime.now(timezone.utc).isoformat()


def _stale_ts(hours: int = 100) -> str:
    """Return an ISO timestamp hours ago."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


@pytest.fixture
def service() -> DataQualityProfilerService:
    """Create a DataQualityProfilerService."""
    cfg = _make_config()
    svc = DataQualityProfilerService(config=cfg)
    svc.startup()
    return svc


@pytest.fixture
def app(service: DataQualityProfilerService) -> FastAPI:
    """Create a FastAPI app with the router and service attached."""
    application = FastAPI()
    application.include_router(router)
    application.state.data_quality_profiler_service = service
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a synchronous test client."""
    return TestClient(app)


@pytest.fixture
def unconfigured_client() -> TestClient:
    """Create a test client with no service configured (for 503 tests)."""
    application = FastAPI()
    application.include_router(router)
    # Intentionally do NOT set app.state.data_quality_profiler_service
    return TestClient(application)


@pytest.fixture
def profiled_dataset(client: TestClient) -> Dict[str, Any]:
    """Profile a dataset and return the JSON response body."""
    resp = client.post(
        "/api/v1/data-quality/v1/profile",
        json={"data": _sample_rows(10), "dataset_name": "fixture_ds"},
    )
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def assessed_dataset(client: TestClient) -> Dict[str, Any]:
    """Assess a dataset and return the JSON response body."""
    resp = client.post(
        "/api/v1/data-quality/v1/assess",
        json={"data": _sample_rows(10), "dataset_name": "fixture_ds"},
    )
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def created_rule(client: TestClient) -> Dict[str, Any]:
    """Create a quality rule and return the JSON response body."""
    resp = client.post(
        "/api/v1/data-quality/v1/rules",
        json={
            "name": "test_rule",
            "rule_type": "not_null",
            "column": "name",
        },
    )
    assert resp.status_code == 200
    return resp.json()


# ===================================================================
# TestProfileDataset
# ===================================================================


class TestProfileDataset:
    """POST /v1/profile - Profile a dataset."""

    def test_success(self, client: TestClient):
        """Profiling succeeds and returns profile data."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "profile_id" in body
        assert body["dataset_name"] == "test"
        assert body["row_count"] == 10

    def test_with_columns(self, client: TestClient):
        """Profiling with specific columns works."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={
                "data": _sample_rows(10),
                "dataset_name": "test",
                "columns": ["name", "value"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["column_count"] == 2

    def test_empty_data_400(self, client: TestClient):
        """Empty data returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={"data": [], "dataset_name": "test"},
        )
        assert resp.status_code == 400

    def test_missing_data_422(self, client: TestClient):
        """Missing 'data' field returns 422."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={"dataset_name": "test"},
        )
        assert resp.status_code == 422

    def test_custom_source(self, client: TestClient):
        """Custom source is accepted."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={
                "data": _sample_rows(5),
                "dataset_name": "test",
                "source": "api_import",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "api_import"


# ===================================================================
# TestProfileBatch
# ===================================================================


class TestProfileBatch:
    """POST /v1/profile/batch - Batch profile datasets."""

    def test_success(self, client: TestClient):
        """Batch profiling succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile/batch",
            json={
                "datasets": [
                    {"data": _sample_rows(5), "dataset_name": "a"},
                    {"data": _sample_rows(5), "dataset_name": "b"},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["requested"] == 2

    def test_empty_list(self, client: TestClient):
        """Empty datasets list succeeds with count=0."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile/batch",
            json={"datasets": []},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_mixed_sizes(self, client: TestClient):
        """Different sized datasets in batch work."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile/batch",
            json={
                "datasets": [
                    {"data": _sample_rows(3), "dataset_name": "small"},
                    {"data": _sample_rows(15), "dataset_name": "large"},
                ],
            },
        )
        assert resp.status_code == 200
        profiles = resp.json()["profiles"]
        assert profiles[0]["row_count"] == 3
        assert profiles[1]["row_count"] == 15


# ===================================================================
# TestListProfiles
# ===================================================================


class TestListProfiles:
    """GET /v1/profiles - List profiles."""

    def test_empty(self, client: TestClient):
        """No profiles returns empty list."""
        resp = client.get("/api/v1/data-quality/v1/profiles")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 0
        assert body["profiles"] == []

    def test_with_data(self, client: TestClient, profiled_dataset):
        """After profiling, profiles are listed."""
        resp = client.get("/api/v1/data-quality/v1/profiles")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_limit(self, client: TestClient):
        """Limit parameter is respected."""
        for i in range(5):
            client.post(
                "/api/v1/data-quality/v1/profile",
                json={"data": _sample_rows(3), "dataset_name": f"ds_{i}"},
            )
        resp = client.get("/api/v1/data-quality/v1/profiles?limit=2")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_offset(self, client: TestClient):
        """Offset parameter is respected."""
        for i in range(5):
            client.post(
                "/api/v1/data-quality/v1/profile",
                json={"data": _sample_rows(3), "dataset_name": f"ds_{i}"},
            )
        resp = client.get("/api/v1/data-quality/v1/profiles?offset=3&limit=10")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2

    def test_pagination_fields(self, client: TestClient):
        """Response includes limit and offset fields."""
        resp = client.get("/api/v1/data-quality/v1/profiles?limit=5&offset=0")
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 5
        assert body["offset"] == 0


# ===================================================================
# TestGetProfile
# ===================================================================


class TestGetProfile:
    """GET /v1/profiles/{profile_id} - Get a profile."""

    def test_200(self, client: TestClient, profiled_dataset):
        """Existing profile returns 200."""
        pid = profiled_dataset["profile_id"]
        resp = client.get(f"/api/v1/data-quality/v1/profiles/{pid}")
        assert resp.status_code == 200
        assert resp.json()["profile_id"] == pid

    def test_404(self, client: TestClient):
        """Non-existent profile returns 404."""
        resp = client.get("/api/v1/data-quality/v1/profiles/not-a-real-id")
        assert resp.status_code == 404


# ===================================================================
# TestAssessQuality
# ===================================================================


class TestAssessQuality:
    """POST /v1/assess - Assess dataset quality."""

    def test_success(self, client: TestClient):
        """Quality assessment succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_score" in body
        assert "quality_level" in body
        assert 0.0 <= body["overall_score"] <= 1.0

    def test_specific_dimensions(self, client: TestClient):
        """Assessment with specific dimensions works."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={
                "data": _sample_rows(10),
                "dataset_name": "test",
                "dimensions": ["completeness", "validity"],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["timeliness_score"] == 0.0

    def test_empty_data_400(self, client: TestClient):
        """Empty data returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": [], "dataset_name": "test"},
        )
        assert resp.status_code == 400

    def test_assessment_id(self, client: TestClient):
        """Response includes an assessment_id."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        assert "assessment_id" in resp.json()

    def test_provenance_hash(self, client: TestClient):
        """Response includes provenance_hash."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert len(resp.json()["provenance_hash"]) == 64


# ===================================================================
# TestAssessBatch
# ===================================================================


class TestAssessBatch:
    """POST /v1/assess/batch - Batch quality assessment."""

    def test_success(self, client: TestClient):
        """Batch assessment succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess/batch",
            json={
                "datasets": [
                    {"data": _sample_rows(5), "dataset_name": "a"},
                    {"data": _sample_rows(5), "dataset_name": "b"},
                ],
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2

    def test_empty(self, client: TestClient):
        """Empty batch returns count=0."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess/batch",
            json={"datasets": []},
        )
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_mixed_quality(self, client: TestClient):
        """Multiple datasets with different quality levels."""
        good_data = [{"id": i, "name": f"n{i}"} for i in range(20)]
        poor_data = [{"id": None, "name": None} for _ in range(20)]
        resp = client.post(
            "/api/v1/data-quality/v1/assess/batch",
            json={
                "datasets": [
                    {"data": good_data, "dataset_name": "good"},
                    {"data": poor_data, "dataset_name": "poor"},
                ],
            },
        )
        assert resp.status_code == 200
        assessments = resp.json()["assessments"]
        assert assessments[0]["overall_score"] > assessments[1]["overall_score"]


# ===================================================================
# TestListAssessments
# ===================================================================


class TestListAssessments:
    """GET /v1/assessments - List assessments."""

    def test_empty(self, client: TestClient):
        """No assessments returns empty list."""
        resp = client.get("/api/v1/data-quality/v1/assessments")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_with_data(self, client: TestClient, assessed_dataset):
        """After assessment, assessments are listed."""
        resp = client.get("/api/v1/data-quality/v1/assessments")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_limit(self, client: TestClient):
        """Limit is respected."""
        for i in range(3):
            client.post(
                "/api/v1/data-quality/v1/assess",
                json={"data": _sample_rows(5), "dataset_name": f"d{i}"},
            )
        resp = client.get("/api/v1/data-quality/v1/assessments?limit=2")
        assert resp.json()["count"] == 2

    def test_pagination_fields(self, client: TestClient):
        """Response includes limit and offset."""
        resp = client.get("/api/v1/data-quality/v1/assessments?limit=5&offset=0")
        body = resp.json()
        assert body["limit"] == 5
        assert body["offset"] == 0


# ===================================================================
# TestGetAssessment
# ===================================================================


class TestGetAssessment:
    """GET /v1/assessments/{assessment_id} - Get assessment."""

    def test_200(self, client: TestClient, assessed_dataset):
        """Existing assessment returns 200."""
        aid = assessed_dataset["assessment_id"]
        resp = client.get(f"/api/v1/data-quality/v1/assessments/{aid}")
        assert resp.status_code == 200
        assert resp.json()["assessment_id"] == aid

    def test_404(self, client: TestClient):
        """Non-existent assessment returns 404."""
        resp = client.get("/api/v1/data-quality/v1/assessments/fake-id")
        assert resp.status_code == 404


# ===================================================================
# TestValidateDataset
# ===================================================================


class TestValidateDataset:
    """POST /v1/validate - Validate dataset."""

    def test_success(self, client: TestClient, created_rule):
        """Validation succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/validate",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "overall_result" in body
        assert body["rules_evaluated"] >= 1

    def test_with_rule_ids(self, client: TestClient, created_rule):
        """Validation with specific rule IDs."""
        rid = created_rule["rule_id"]
        resp = client.post(
            "/api/v1/data-quality/v1/validate",
            json={
                "data": _sample_rows(10),
                "dataset_name": "test",
                "rule_ids": [rid],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["rules_evaluated"] == 1

    def test_empty_data_400(self, client: TestClient):
        """Empty data returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/validate",
            json={"data": [], "dataset_name": "test"},
        )
        assert resp.status_code == 400

    def test_no_rules(self, client: TestClient):
        """No rules returns pass with 0 evaluated."""
        # Fresh service with no rules
        resp = client.post(
            "/api/v1/data-quality/v1/validate",
            json={"data": _sample_rows(5), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        assert resp.json()["rules_evaluated"] == 0


# ===================================================================
# TestDetectAnomalies
# ===================================================================


class TestDetectAnomalies:
    """POST /v1/detect-anomalies - Detect anomalies."""

    def test_success(self, client: TestClient):
        """Anomaly detection succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={"data": _numeric_rows(20), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "anomaly_count" in body

    def test_with_method(self, client: TestClient):
        """Specific method is accepted."""
        resp = client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={
                "data": _rows_with_outliers(30),
                "dataset_name": "test",
                "method": "zscore",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["method"] == "zscore"

    def test_with_columns(self, client: TestClient):
        """Specific columns are analysed."""
        resp = client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={
                "data": _numeric_rows(20),
                "dataset_name": "test",
                "columns": ["x"],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["columns_analysed"] == ["x"]

    def test_empty_data_400(self, client: TestClient):
        """Empty data returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={"data": [], "dataset_name": "test"},
        )
        assert resp.status_code == 400


# ===================================================================
# TestListAnomalies
# ===================================================================


class TestListAnomalies:
    """GET /v1/anomalies - List anomaly results."""

    def test_empty(self, client: TestClient):
        """No anomalies returns empty list."""
        resp = client.get("/api/v1/data-quality/v1/anomalies")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_with_data(self, client: TestClient):
        """After detection, anomalies are listed."""
        client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={"data": _numeric_rows(20), "dataset_name": "test"},
        )
        resp = client.get("/api/v1/data-quality/v1/anomalies")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_limit(self, client: TestClient):
        """Limit is respected."""
        for i in range(3):
            client.post(
                "/api/v1/data-quality/v1/detect-anomalies",
                json={"data": _numeric_rows(20), "dataset_name": f"d{i}"},
            )
        resp = client.get("/api/v1/data-quality/v1/anomalies?limit=2")
        assert resp.json()["count"] == 2


# ===================================================================
# TestCheckFreshness
# ===================================================================


class TestCheckFreshness:
    """POST /v1/check-freshness - Check freshness."""

    def test_fresh(self, client: TestClient):
        """Recent data returns fresh status."""
        resp = client.post(
            "/api/v1/data-quality/v1/check-freshness",
            json={
                "dataset_name": "test",
                "last_updated": _fresh_ts(),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "fresh"

    def test_stale(self, client: TestClient):
        """Old data returns stale or expired."""
        resp = client.post(
            "/api/v1/data-quality/v1/check-freshness",
            json={
                "dataset_name": "test",
                "last_updated": _stale_ts(hours=60),
                "sla_hours": 48,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] in ("stale", "expired")

    def test_missing_fields_422(self, client: TestClient):
        """Missing required fields returns 422."""
        resp = client.post(
            "/api/v1/data-quality/v1/check-freshness",
            json={"dataset_name": "test"},
        )
        assert resp.status_code == 422


# ===================================================================
# TestCreateRule
# ===================================================================


class TestCreateRule:
    """POST /v1/rules - Create a quality rule."""

    def test_success(self, client: TestClient):
        """Rule creation succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/rules",
            json={
                "name": "test_rule",
                "rule_type": "not_null",
                "column": "name",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "test_rule"
        assert body["rule_type"] == "not_null"

    def test_all_fields(self, client: TestClient):
        """All optional fields are accepted."""
        resp = client.post(
            "/api/v1/data-quality/v1/rules",
            json={
                "name": "full_rule",
                "rule_type": "range",
                "column": "value",
                "operator": "gte",
                "threshold": 50,
                "parameters": {"min": 0, "max": 100},
                "priority": 10,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["priority"] == 10

    def test_empty_name_400(self, client: TestClient):
        """Empty name returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/rules",
            json={"name": "", "rule_type": "not_null", "column": "x"},
        )
        assert resp.status_code == 400

    def test_missing_name_422(self, client: TestClient):
        """Missing name field returns 422."""
        resp = client.post(
            "/api/v1/data-quality/v1/rules",
            json={"rule_type": "not_null", "column": "x"},
        )
        assert resp.status_code == 422


# ===================================================================
# TestListRules
# ===================================================================


class TestListRules:
    """GET /v1/rules - List rules."""

    def test_empty(self, client: TestClient):
        """No rules returns empty list."""
        resp = client.get("/api/v1/data-quality/v1/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_with_data(self, client: TestClient, created_rule):
        """After creation, rules are listed."""
        resp = client.get("/api/v1/data-quality/v1/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_active_filter(self, client: TestClient, created_rule):
        """active_only filter works."""
        resp = client.get("/api/v1/data-quality/v1/rules?active_only=true")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_pagination(self, client: TestClient):
        """Pagination via query params."""
        for i in range(3):
            client.post(
                "/api/v1/data-quality/v1/rules",
                json={"name": f"r{i}", "rule_type": "not_null", "column": "x"},
            )
        resp = client.get("/api/v1/data-quality/v1/rules")
        assert resp.json()["count"] == 3


# ===================================================================
# TestUpdateRule
# ===================================================================


class TestUpdateRule:
    """PUT /v1/rules/{rule_id} - Update a rule."""

    def test_success(self, client: TestClient, created_rule):
        """Updating a rule succeeds."""
        rid = created_rule["rule_id"]
        resp = client.put(
            f"/api/v1/data-quality/v1/rules/{rid}",
            json={"name": "updated_name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated_name"

    def test_404(self, client: TestClient):
        """Updating a non-existent rule returns 404."""
        resp = client.put(
            "/api/v1/data-quality/v1/rules/fake-id",
            json={"name": "nope"},
        )
        assert resp.status_code == 404

    def test_deactivate(self, client: TestClient, created_rule):
        """Deactivating a rule via update."""
        rid = created_rule["rule_id"]
        resp = client.put(
            f"/api/v1/data-quality/v1/rules/{rid}",
            json={"is_active": False},
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False


# ===================================================================
# TestDeleteRule
# ===================================================================


class TestDeleteRule:
    """DELETE /v1/rules/{rule_id} - Delete a rule."""

    def test_success(self, client: TestClient, created_rule):
        """Deleting an existing rule succeeds."""
        rid = created_rule["rule_id"]
        resp = client.delete(f"/api/v1/data-quality/v1/rules/{rid}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_404(self, client: TestClient):
        """Deleting a non-existent rule returns 404."""
        resp = client.delete("/api/v1/data-quality/v1/rules/fake-id")
        assert resp.status_code == 404


# ===================================================================
# TestEvaluateGate
# ===================================================================


class TestEvaluateGate:
    """POST /v1/gates - Evaluate a quality gate."""

    def test_pass(self, client: TestClient):
        """Gate passes when conditions are met."""
        resp = client.post(
            "/api/v1/data-quality/v1/gates",
            json={
                "conditions": [
                    {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
                ],
                "dimension_scores": {"completeness": 0.9},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "pass"

    def test_fail(self, client: TestClient):
        """Gate fails when conditions are not met."""
        resp = client.post(
            "/api/v1/data-quality/v1/gates",
            json={
                "conditions": [
                    {"dimension": "completeness", "operator": "gte", "threshold": 0.99},
                ],
                "dimension_scores": {"completeness": 0.3},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "fail"

    def test_empty_conditions_400(self, client: TestClient):
        """Empty conditions returns 400."""
        resp = client.post(
            "/api/v1/data-quality/v1/gates",
            json={"conditions": []},
        )
        assert resp.status_code == 400


# ===================================================================
# TestGetTrends
# ===================================================================


class TestGetTrends:
    """GET /v1/trends - Get quality trends."""

    def test_empty(self, client: TestClient):
        """No assessments returns empty trends."""
        resp = client.get("/api/v1/data-quality/v1/trends")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_with_data(self, client: TestClient, assessed_dataset):
        """After assessment, trends are available."""
        resp = client.get("/api/v1/data-quality/v1/trends")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

    def test_periods_param(self, client: TestClient, assessed_dataset):
        """periods parameter is accepted."""
        resp = client.get("/api/v1/data-quality/v1/trends?periods=5")
        assert resp.status_code == 200


# ===================================================================
# TestGenerateReport
# ===================================================================


class TestGenerateReport:
    """POST /v1/reports - Generate a report."""

    def test_scorecard(self, client: TestClient, assessed_dataset):
        """Scorecard report succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "json"},
        )
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "scorecard"

    def test_detailed(self, client: TestClient, assessed_dataset):
        """Detailed report succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "detailed", "format": "json"},
        )
        assert resp.status_code == 200
        assert resp.json()["report_type"] == "detailed"

    def test_csv_format(self, client: TestClient, assessed_dataset):
        """CSV format report succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "csv"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "csv"

    def test_markdown_format(self, client: TestClient, assessed_dataset):
        """Markdown format report succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "markdown"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "markdown"

    def test_html_format(self, client: TestClient, assessed_dataset):
        """HTML format report succeeds."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "html"},
        )
        assert resp.status_code == 200
        assert resp.json()["format"] == "html"


# ===================================================================
# TestHealthCheck
# ===================================================================


class TestHealthCheck:
    """GET /health - Health check endpoint."""

    def test_200(self, client: TestClient):
        """Health check returns 200."""
        resp = client.get("/api/v1/data-quality/health")
        assert resp.status_code == 200

    def test_healthy(self, client: TestClient):
        """Service is healthy after startup."""
        resp = client.get("/api/v1/data-quality/health")
        body = resp.json()
        assert body["status"] == "healthy"

    def test_service_name(self, client: TestClient):
        """Response includes correct service name."""
        resp = client.get("/api/v1/data-quality/health")
        assert resp.json()["service"] == "data-quality-profiler"

    def test_all_fields(self, client: TestClient):
        """Response includes all expected fields."""
        resp = client.get("/api/v1/data-quality/health")
        body = resp.json()
        expected = {
            "status", "service", "started", "profiles", "assessments",
            "anomaly_detections", "freshness_checks", "rules",
            "gate_evaluations", "reports", "provenance_entries",
            "prometheus_available",
        }
        assert expected.issubset(body.keys())

    def test_counts_zero_initially(self, client: TestClient):
        """All counts are zero on fresh service."""
        resp = client.get("/api/v1/data-quality/health")
        body = resp.json()
        assert body["profiles"] == 0
        assert body["rules"] == 0


# ===================================================================
# TestServiceNotConfigured
# ===================================================================


class TestServiceNotConfigured:
    """All 20 endpoints return 503 when service is not configured."""

    def test_profile_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/profile",
            json={"data": [{"a": 1}], "dataset_name": "test"},
        )
        assert resp.status_code == 503

    def test_profile_batch_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/profile/batch",
            json={"datasets": []},
        )
        assert resp.status_code == 503

    def test_list_profiles_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/profiles")
        assert resp.status_code == 503

    def test_get_profile_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/profiles/x")
        assert resp.status_code == 503

    def test_assess_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": [{"a": 1}], "dataset_name": "test"},
        )
        assert resp.status_code == 503

    def test_assess_batch_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/assess/batch",
            json={"datasets": []},
        )
        assert resp.status_code == 503

    def test_list_assessments_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/assessments")
        assert resp.status_code == 503

    def test_get_assessment_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/assessments/x")
        assert resp.status_code == 503

    def test_validate_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/validate",
            json={"data": [{"a": 1}], "dataset_name": "test"},
        )
        assert resp.status_code == 503

    def test_detect_anomalies_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={"data": [{"a": 1}], "dataset_name": "test"},
        )
        assert resp.status_code == 503

    def test_list_anomalies_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/anomalies")
        assert resp.status_code == 503

    def test_check_freshness_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/check-freshness",
            json={"dataset_name": "test", "last_updated": _fresh_ts()},
        )
        assert resp.status_code == 503

    def test_create_rule_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/rules",
            json={"name": "r", "rule_type": "not_null", "column": "x"},
        )
        assert resp.status_code == 503

    def test_list_rules_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/rules")
        assert resp.status_code == 503

    def test_update_rule_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.put(
            "/api/v1/data-quality/v1/rules/x",
            json={"name": "new"},
        )
        assert resp.status_code == 503

    def test_delete_rule_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.delete("/api/v1/data-quality/v1/rules/x")
        assert resp.status_code == 503

    def test_evaluate_gate_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/gates",
            json={
                "conditions": [{"dimension": "completeness", "operator": "gte", "threshold": 0.5}],
            },
        )
        assert resp.status_code == 503

    def test_trends_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/v1/trends")
        assert resp.status_code == 503

    def test_reports_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard"},
        )
        assert resp.status_code == 503

    def test_health_503(self, unconfigured_client: TestClient):
        resp = unconfigured_client.get("/api/v1/data-quality/health")
        assert resp.status_code == 503


# ===================================================================
# TestFullAPIWorkflow
# ===================================================================


class TestFullAPIWorkflow:
    """End-to-end API workflow test."""

    def test_complete_workflow(self, client: TestClient):
        """Full end-to-end workflow through all 20 endpoints."""
        # 1. Profile
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={"data": _sample_rows(10), "dataset_name": "e2e"},
        )
        assert resp.status_code == 200
        profile = resp.json()
        profile_id = profile["profile_id"]

        # 2. Batch profile
        resp = client.post(
            "/api/v1/data-quality/v1/profile/batch",
            json={"datasets": [{"data": _sample_rows(5), "dataset_name": "e2e_batch"}]},
        )
        assert resp.status_code == 200

        # 3. List profiles
        resp = client.get("/api/v1/data-quality/v1/profiles")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 2

        # 4. Get profile
        resp = client.get(f"/api/v1/data-quality/v1/profiles/{profile_id}")
        assert resp.status_code == 200

        # 5. Assess quality
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": _sample_rows(10), "dataset_name": "e2e"},
        )
        assert resp.status_code == 200
        assessment = resp.json()
        assessment_id = assessment["assessment_id"]

        # 6. Batch assess
        resp = client.post(
            "/api/v1/data-quality/v1/assess/batch",
            json={"datasets": [{"data": _sample_rows(5), "dataset_name": "e2e_batch"}]},
        )
        assert resp.status_code == 200

        # 7. List assessments
        resp = client.get("/api/v1/data-quality/v1/assessments")
        assert resp.status_code == 200

        # 8. Get assessment
        resp = client.get(f"/api/v1/data-quality/v1/assessments/{assessment_id}")
        assert resp.status_code == 200

        # 9. Validate
        resp = client.post(
            "/api/v1/data-quality/v1/validate",
            json={"data": _sample_rows(10), "dataset_name": "e2e"},
        )
        assert resp.status_code == 200

        # 10. Detect anomalies
        resp = client.post(
            "/api/v1/data-quality/v1/detect-anomalies",
            json={"data": _numeric_rows(20), "dataset_name": "e2e"},
        )
        assert resp.status_code == 200

        # 11. List anomalies
        resp = client.get("/api/v1/data-quality/v1/anomalies")
        assert resp.status_code == 200

        # 12. Check freshness
        resp = client.post(
            "/api/v1/data-quality/v1/check-freshness",
            json={"dataset_name": "e2e", "last_updated": _fresh_ts()},
        )
        assert resp.status_code == 200

        # 13. Create rule
        resp = client.post(
            "/api/v1/data-quality/v1/rules",
            json={"name": "e2e_rule", "rule_type": "not_null", "column": "name"},
        )
        assert resp.status_code == 200
        rule = resp.json()
        rule_id = rule["rule_id"]

        # 14. List rules
        resp = client.get("/api/v1/data-quality/v1/rules")
        assert resp.status_code == 200
        assert resp.json()["count"] >= 1

        # 15. Update rule
        resp = client.put(
            f"/api/v1/data-quality/v1/rules/{rule_id}",
            json={"name": "updated_e2e"},
        )
        assert resp.status_code == 200

        # 16. Delete rule
        resp = client.delete(f"/api/v1/data-quality/v1/rules/{rule_id}")
        assert resp.status_code == 200

        # 17. Evaluate gate
        resp = client.post(
            "/api/v1/data-quality/v1/gates",
            json={
                "conditions": [
                    {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
                ],
                "dimension_scores": {"completeness": 0.9},
            },
        )
        assert resp.status_code == 200

        # 18. Get trends
        resp = client.get("/api/v1/data-quality/v1/trends")
        assert resp.status_code == 200

        # 19. Generate report
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "json"},
        )
        assert resp.status_code == 200

        # 20. Health check
        resp = client.get("/api/v1/data-quality/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Additional edge case tests for completeness."""

    def test_profile_with_source_field(self, client: TestClient):
        """Profile with explicit source field in body."""
        resp = client.post(
            "/api/v1/data-quality/v1/profile",
            json={
                "data": _sample_rows(5),
                "dataset_name": "src_test",
                "source": "erp_import",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["source"] == "erp_import"

    def test_assess_returns_issue_count(self, client: TestClient):
        """Assessment response includes row_count and column_count."""
        resp = client.post(
            "/api/v1/data-quality/v1/assess",
            json={"data": _sample_rows(10), "dataset_name": "test"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["row_count"] == 10
        assert body["column_count"] >= 1

    def test_gate_details_present(self, client: TestClient):
        """Gate response includes details list."""
        resp = client.post(
            "/api/v1/data-quality/v1/gates",
            json={
                "conditions": [
                    {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
                ],
                "dimension_scores": {"completeness": 0.9},
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "details" in body
        assert len(body["details"]) == 1
        assert body["details"][0]["result"] == "pass"

    def test_report_includes_report_id(self, client: TestClient, assessed_dataset):
        """Report response includes report_id."""
        resp = client.post(
            "/api/v1/data-quality/v1/reports",
            json={"report_type": "scorecard", "format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "report_id" in body
        assert len(body["report_id"]) > 0

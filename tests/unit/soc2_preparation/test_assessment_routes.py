# -*- coding: utf-8 -*-
"""
Unit tests for SOC 2 Assessment API Routes - SEC-009 Phase 11

Tests the assessment API endpoints:
- GET /assessment
- POST /assessment/run
- GET /assessment/score
- GET /assessment/gaps
- GET /assessment/criteria
- PUT /assessment/criteria/{criterion_id}

Coverage targets: 85%+ of assessment_routes.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.infrastructure.soc2_preparation.api.assessment_routes import (
    router,
    AssessmentSummary,
    AssessmentRunRequest,
    AssessmentRunResponse,
    ReadinessScore,
    GapsResponse,
    Gap,
    CriterionStatus,
    CriterionUpdateRequest,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def app() -> FastAPI:
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/soc2")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


# ============================================================================
# Test AssessmentSummary Model
# ============================================================================


class TestAssessmentSummaryModel:
    """Tests for AssessmentSummary Pydantic model."""

    def test_assessment_summary_defaults(self) -> None:
        """AssessmentSummary has correct defaults."""
        summary = AssessmentSummary(
            overall_score=75.0,
            total_criteria=48,
        )
        assert summary.status == "in_progress"
        assert summary.overall_score == 75.0
        assert summary.total_criteria == 48
        assert summary.criteria_complete == 0
        assert summary.criteria_in_progress == 0
        assert summary.criteria_not_started == 0
        assert summary.category_scores == {}

    def test_assessment_summary_with_all_fields(self) -> None:
        """AssessmentSummary with all fields populated."""
        summary = AssessmentSummary(
            status="complete",
            overall_score=92.5,
            total_criteria=48,
            criteria_complete=48,
            criteria_in_progress=0,
            criteria_not_started=0,
            category_scores={"security": 95.0, "availability": 90.0},
        )
        assert summary.status == "complete"
        assert summary.overall_score == 92.5
        assert summary.criteria_complete == 48
        assert "security" in summary.category_scores

    def test_assessment_summary_score_range(self) -> None:
        """AssessmentSummary score is constrained 0-100."""
        summary = AssessmentSummary(
            overall_score=0.0,
            total_criteria=48,
        )
        assert summary.overall_score == 0.0

        summary = AssessmentSummary(
            overall_score=100.0,
            total_criteria=48,
        )
        assert summary.overall_score == 100.0


# ============================================================================
# Test AssessmentRunRequest Model
# ============================================================================


class TestAssessmentRunRequestModel:
    """Tests for AssessmentRunRequest Pydantic model."""

    def test_run_request_defaults(self) -> None:
        """AssessmentRunRequest has correct defaults."""
        request = AssessmentRunRequest()
        assert request.categories is None
        assert request.criteria is None
        assert request.full_refresh is False
        assert request.notify_on_complete is True

    def test_run_request_with_categories(self) -> None:
        """AssessmentRunRequest with category filter."""
        request = AssessmentRunRequest(
            categories=["security", "availability"],
            full_refresh=True,
        )
        assert request.categories == ["security", "availability"]
        assert request.full_refresh is True

    def test_run_request_with_criteria(self) -> None:
        """AssessmentRunRequest with specific criteria."""
        request = AssessmentRunRequest(
            criteria=["CC6.1", "CC6.2", "CC7.1"],
        )
        assert len(request.criteria) == 3


# ============================================================================
# Test ReadinessScore Model
# ============================================================================


class TestReadinessScoreModel:
    """Tests for ReadinessScore Pydantic model."""

    def test_readiness_score_required_fields(self) -> None:
        """ReadinessScore requires core fields."""
        score = ReadinessScore(
            overall_score=75.0,
            overall_status="partial",
            category_scores={"security": 80.0},
            criteria_scores={"CC6.1": 90.0},
            assessment_date=datetime.now(timezone.utc),
        )
        assert score.overall_score == 75.0
        assert score.overall_status == "partial"
        assert score.trend == "stable"

    def test_readiness_score_with_recommendations(self) -> None:
        """ReadinessScore includes recommendations."""
        score = ReadinessScore(
            overall_score=65.0,
            overall_status="not_ready",
            category_scores={},
            criteria_scores={},
            assessment_date=datetime.now(timezone.utc),
            recommendations=["Fix MFA", "Update policies"],
        )
        assert len(score.recommendations) == 2


# ============================================================================
# Test Gap Model
# ============================================================================


class TestGapModel:
    """Tests for Gap Pydantic model."""

    def test_gap_defaults(self) -> None:
        """Gap has correct defaults."""
        gap = Gap(
            criterion_id="CC6.1",
            title="Test Gap",
        )
        assert gap.severity == "medium"
        assert gap.priority == 3
        assert gap.status == "open"
        assert gap.description == ""

    def test_gap_with_all_fields(self) -> None:
        """Gap with all fields populated."""
        gap = Gap(
            criterion_id="CC6.7",
            title="MFA Coverage Gap",
            description="Service accounts lack MFA",
            severity="high",
            priority=1,
            status="in_progress",
            remediation_plan="Migrate to workload identity",
            owner="security-team",
            due_date=datetime.now(timezone.utc),
        )
        assert gap.severity == "high"
        assert gap.priority == 1
        assert gap.owner == "security-team"


# ============================================================================
# Test CriterionUpdateRequest Model
# ============================================================================


class TestCriterionUpdateRequestModel:
    """Tests for CriterionUpdateRequest Pydantic model."""

    def test_update_request_all_none(self) -> None:
        """CriterionUpdateRequest allows all None."""
        request = CriterionUpdateRequest()
        assert request.status is None
        assert request.notes is None
        assert request.owner is None
        assert request.target_date is None

    def test_update_request_status_validation_valid(self) -> None:
        """CriterionUpdateRequest validates status values."""
        valid_statuses = [
            "not_started",
            "in_progress",
            "implemented",
            "tested",
            "verified",
        ]
        for status in valid_statuses:
            request = CriterionUpdateRequest(status=status)
            assert request.status == status.lower()

    def test_update_request_status_validation_invalid(self) -> None:
        """CriterionUpdateRequest rejects invalid status."""
        with pytest.raises(ValueError, match="Invalid status"):
            CriterionUpdateRequest(status="invalid_status")


# ============================================================================
# Test GET /assessment Endpoint
# ============================================================================


class TestGetAssessmentEndpoint:
    """Tests for GET /assessment endpoint."""

    def test_get_assessment_success(self, client: TestClient) -> None:
        """GET /assessment returns assessment summary."""
        response = client.get("/api/v1/soc2/assessment")
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
        assert "total_criteria" in data
        assert "status" in data
        assert "category_scores" in data

    def test_get_assessment_includes_category_scores(self, client: TestClient) -> None:
        """GET /assessment includes category breakdown."""
        response = client.get("/api/v1/soc2/assessment")
        assert response.status_code == 200
        data = response.json()
        assert "security" in data["category_scores"]

    def test_get_assessment_with_criteria_param(self, client: TestClient) -> None:
        """GET /assessment with include_criteria parameter."""
        response = client.get("/api/v1/soc2/assessment?include_criteria=true")
        assert response.status_code == 200

    def test_get_assessment_response_model(self, client: TestClient) -> None:
        """GET /assessment matches AssessmentSummary model."""
        response = client.get("/api/v1/soc2/assessment")
        assert response.status_code == 200
        # Should not raise on model validation
        AssessmentSummary(**response.json())


# ============================================================================
# Test POST /assessment/run Endpoint
# ============================================================================


class TestRunAssessmentEndpoint:
    """Tests for POST /assessment/run endpoint."""

    def test_run_assessment_success(self, client: TestClient) -> None:
        """POST /assessment/run starts assessment."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={},
        )
        assert response.status_code == 202
        data = response.json()
        assert "assessment_id" in data
        assert "status" in data
        assert data["status"] == "running"

    def test_run_assessment_with_categories(self, client: TestClient) -> None:
        """POST /assessment/run with category filter."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={"categories": ["security", "availability"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["criteria_count"] > 0

    def test_run_assessment_with_specific_criteria(self, client: TestClient) -> None:
        """POST /assessment/run with specific criteria."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={"criteria": ["CC6.1", "CC6.2", "CC7.1"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert data["criteria_count"] == 3

    def test_run_assessment_with_full_refresh(self, client: TestClient) -> None:
        """POST /assessment/run with full_refresh flag."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={"full_refresh": True},
        )
        assert response.status_code == 202

    def test_run_assessment_estimated_duration(self, client: TestClient) -> None:
        """POST /assessment/run includes estimated duration."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={"criteria": ["CC6.1"]},
        )
        assert response.status_code == 202
        data = response.json()
        assert "estimated_duration_seconds" in data
        assert data["estimated_duration_seconds"] > 0


# ============================================================================
# Test GET /assessment/score Endpoint
# ============================================================================


class TestGetScoreEndpoint:
    """Tests for GET /assessment/score endpoint."""

    def test_get_score_success(self, client: TestClient) -> None:
        """GET /assessment/score returns readiness score."""
        response = client.get("/api/v1/soc2/assessment/score")
        assert response.status_code == 200
        data = response.json()
        assert "overall_score" in data
        assert "overall_status" in data
        assert "category_scores" in data
        assert "criteria_scores" in data

    def test_get_score_with_recommendations(self, client: TestClient) -> None:
        """GET /assessment/score includes recommendations."""
        response = client.get("/api/v1/soc2/assessment/score?include_recommendations=true")
        assert response.status_code == 200
        data = response.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0

    def test_get_score_without_recommendations(self, client: TestClient) -> None:
        """GET /assessment/score without recommendations."""
        response = client.get("/api/v1/soc2/assessment/score?include_recommendations=false")
        assert response.status_code == 200

    def test_get_score_includes_trend(self, client: TestClient) -> None:
        """GET /assessment/score includes trend indicator."""
        response = client.get("/api/v1/soc2/assessment/score")
        assert response.status_code == 200
        data = response.json()
        assert "trend" in data
        assert data["trend"] in ["improving", "stable", "declining"]


# ============================================================================
# Test GET /assessment/gaps Endpoint
# ============================================================================


class TestGetGapsEndpoint:
    """Tests for GET /assessment/gaps endpoint."""

    def test_get_gaps_success(self, client: TestClient) -> None:
        """GET /assessment/gaps returns gaps list."""
        response = client.get("/api/v1/soc2/assessment/gaps")
        assert response.status_code == 200
        data = response.json()
        assert "total_gaps" in data
        assert "gaps" in data
        assert isinstance(data["gaps"], list)

    def test_get_gaps_includes_severity_counts(self, client: TestClient) -> None:
        """GET /assessment/gaps includes severity breakdown."""
        response = client.get("/api/v1/soc2/assessment/gaps")
        assert response.status_code == 200
        data = response.json()
        assert "critical_count" in data
        assert "high_count" in data
        assert "medium_count" in data
        assert "low_count" in data

    def test_get_gaps_filter_by_severity(self, client: TestClient) -> None:
        """GET /assessment/gaps with severity filter."""
        response = client.get("/api/v1/soc2/assessment/gaps?severity=high")
        assert response.status_code == 200
        data = response.json()
        for gap in data["gaps"]:
            assert gap["severity"] == "high"

    def test_get_gaps_filter_by_status(self, client: TestClient) -> None:
        """GET /assessment/gaps with status filter."""
        response = client.get("/api/v1/soc2/assessment/gaps?status=open")
        assert response.status_code == 200

    def test_get_gaps_filter_by_criterion(self, client: TestClient) -> None:
        """GET /assessment/gaps with criterion filter."""
        response = client.get("/api/v1/soc2/assessment/gaps?criterion=CC6.7")
        assert response.status_code == 200

    def test_get_gaps_pagination(self, client: TestClient) -> None:
        """GET /assessment/gaps with pagination."""
        response = client.get("/api/v1/soc2/assessment/gaps?limit=10&offset=0")
        assert response.status_code == 200


# ============================================================================
# Test GET /assessment/criteria Endpoint
# ============================================================================


class TestListCriteriaEndpoint:
    """Tests for GET /assessment/criteria endpoint."""

    def test_list_criteria_success(self, client: TestClient) -> None:
        """GET /assessment/criteria returns criteria list."""
        response = client.get("/api/v1/soc2/assessment/criteria")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_criteria_item_structure(self, client: TestClient) -> None:
        """GET /assessment/criteria items have required fields."""
        response = client.get("/api/v1/soc2/assessment/criteria")
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert "criterion_id" in item
            assert "criterion_name" in item
            assert "category" in item
            assert "status" in item
            assert "score" in item

    def test_list_criteria_filter_by_category(self, client: TestClient) -> None:
        """GET /assessment/criteria with category filter."""
        response = client.get("/api/v1/soc2/assessment/criteria?category=security")
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["category"] == "security"

    def test_list_criteria_filter_by_status(self, client: TestClient) -> None:
        """GET /assessment/criteria with status filter."""
        response = client.get("/api/v1/soc2/assessment/criteria?status=verified")
        assert response.status_code == 200


# ============================================================================
# Test PUT /assessment/criteria/{criterion_id} Endpoint
# ============================================================================


class TestUpdateCriterionEndpoint:
    """Tests for PUT /assessment/criteria/{criterion_id} endpoint."""

    def test_update_criterion_success(self, client: TestClient) -> None:
        """PUT /assessment/criteria/{id} updates criterion."""
        response = client.put(
            "/api/v1/soc2/assessment/criteria/CC6.1",
            json={"status": "verified"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"

    def test_update_criterion_normalizes_id(self, client: TestClient) -> None:
        """PUT /assessment/criteria/{id} normalizes criterion ID."""
        response = client.put(
            "/api/v1/soc2/assessment/criteria/cc6.1",
            json={"notes": "Updated notes"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["criterion_id"] == "CC6.1"

    def test_update_criterion_with_notes(self, client: TestClient) -> None:
        """PUT /assessment/criteria/{id} with notes update."""
        response = client.put(
            "/api/v1/soc2/assessment/criteria/CC6.2",
            json={"notes": "Completed implementation"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["notes"] == "Completed implementation"

    def test_update_criterion_with_owner(self, client: TestClient) -> None:
        """PUT /assessment/criteria/{id} with owner assignment."""
        response = client.put(
            "/api/v1/soc2/assessment/criteria/CC7.1",
            json={"owner": "security-team"},
        )
        assert response.status_code == 200

    def test_update_criterion_partial_update(self, client: TestClient) -> None:
        """PUT /assessment/criteria/{id} with partial update."""
        response = client.put(
            "/api/v1/soc2/assessment/criteria/CC6.3",
            json={"status": "in_progress"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "in_progress"


# ============================================================================
# Test Response Headers and Content-Type
# ============================================================================


class TestResponseHeaders:
    """Tests for response headers."""

    def test_json_content_type(self, client: TestClient) -> None:
        """Responses have JSON content type."""
        response = client.get("/api/v1/soc2/assessment")
        assert response.headers["content-type"] == "application/json"

    def test_run_assessment_accepted_status(self, client: TestClient) -> None:
        """POST /assessment/run returns 202 Accepted."""
        response = client.post(
            "/api/v1/soc2/assessment/run",
            json={},
        )
        assert response.status_code == 202

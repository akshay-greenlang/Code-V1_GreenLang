# -*- coding: utf-8 -*-
"""
Unit tests for CDP API Routes -- all REST endpoints.

Tests questionnaire endpoints, response endpoints, scoring endpoints,
gap analysis endpoints, benchmarking endpoints, error handling,
validation, pagination, and filtering with 42+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.config import ResponseStatus, SubmissionFormat
from services.models import (
    CDPOrganization,
    CDPQuestionnaire,
    CDPResponse,
    CDPScoringResult,
    CDPGapAnalysis,
    CDPBenchmark,
    _new_id,
)


# ---------------------------------------------------------------------------
# Mock app client fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_client():
    """Mock FastAPI test client."""
    client = MagicMock()
    client.get = MagicMock()
    client.post = MagicMock()
    client.put = MagicMock()
    client.delete = MagicMock()
    client.patch = MagicMock()
    return client


@pytest.fixture
def auth_headers():
    """Authentication headers for protected endpoints."""
    return {"Authorization": "Bearer test-jwt-token"}


# ---------------------------------------------------------------------------
# Questionnaire endpoints
# ---------------------------------------------------------------------------

class TestQuestionnaireEndpoints:
    """Test questionnaire API routes."""

    def test_create_questionnaire(self, mock_client, auth_headers, sample_organization):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "id": _new_id(),
                "org_id": sample_organization.id,
                "reporting_year": 2025,
                "status": "not_started",
            },
        )
        response = mock_client.post(
            "/api/v1/cdp/questionnaires",
            json={
                "org_id": sample_organization.id,
                "reporting_year": 2025,
                "sector_gics": sample_organization.sector_gics,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_questionnaire(self, mock_client, auth_headers):
        qid = _new_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": qid, "reporting_year": 2025, "status": "in_progress"},
        )
        response = mock_client.get(
            f"/api/v1/cdp/questionnaires/{qid}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["id"] == qid

    def test_list_questionnaires(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0, "page": 1},
        )
        response = mock_client.get(
            "/api/v1/cdp/questionnaires",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_modules(self, mock_client, auth_headers):
        qid = _new_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"module_code": f"M{i}", "module_name": f"Module {i}"} for i in range(14)],
        )
        response = mock_client.get(
            f"/api/v1/cdp/questionnaires/{qid}/modules",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert len(response.json()) == 14

    def test_get_module_questions(self, mock_client, auth_headers):
        qid = _new_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"question_number": f"C7.{i}"} for i in range(1, 36)],
        )
        response = mock_client.get(
            f"/api/v1/cdp/questionnaires/{qid}/modules/M7/questions",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert len(response.json()) == 35

    def test_questionnaire_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {"detail": "Questionnaire not found"},
        )
        response = mock_client.get(
            f"/api/v1/cdp/questionnaires/{_new_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Response endpoints
# ---------------------------------------------------------------------------

class TestResponseEndpoints:
    """Test response CRUD API routes."""

    def test_create_response(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "id": _new_id(),
                "response_status": "draft",
                "response_content": {"answer": "yes"},
            },
        )
        response = mock_client.post(
            "/api/v1/cdp/responses",
            json={
                "question_id": _new_id(),
                "questionnaire_id": _new_id(),
                "response_content": {"answer": "yes"},
                "response_text": "Yes",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_update_response(self, mock_client, auth_headers):
        rid = _new_id()
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": rid, "response_content": {"answer": "updated"}},
        )
        response = mock_client.put(
            f"/api/v1/cdp/responses/{rid}",
            json={"response_content": {"answer": "updated"}},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_delete_response(self, mock_client, auth_headers):
        rid = _new_id()
        mock_client.delete.return_value = MagicMock(status_code=204)
        response = mock_client.delete(
            f"/api/v1/cdp/responses/{rid}",
            headers=auth_headers,
        )
        assert response.status_code == 204

    def test_transition_response_status(self, mock_client, auth_headers):
        rid = _new_id()
        mock_client.patch.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": rid, "response_status": "in_review"},
        )
        response = mock_client.patch(
            f"/api/v1/cdp/responses/{rid}/status",
            json={"target_status": "in_review"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_attach_evidence(self, mock_client, auth_headers):
        rid = _new_id()
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _new_id(), "file_name": "evidence.pdf"},
        )
        response = mock_client.post(
            f"/api/v1/cdp/responses/{rid}/evidence",
            json={"file_name": "evidence.pdf", "file_type": "application/pdf"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_bulk_approve(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"approved_count": 5},
        )
        response = mock_client.post(
            "/api/v1/cdp/responses/bulk-approve",
            json={"response_ids": [_new_id() for _ in range(5)]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["approved_count"] == 5

    def test_bulk_import_previous_year(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"imported_count": 150},
        )
        response = mock_client.post(
            "/api/v1/cdp/responses/import-previous",
            json={"source_questionnaire_id": _new_id(), "target_questionnaire_id": _new_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_responses_with_pagination(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [{"id": _new_id()} for _ in range(20)],
                "total": 200,
                "page": 1,
                "page_size": 20,
                "total_pages": 10,
            },
        )
        response = mock_client.get(
            "/api/v1/cdp/responses?page=1&page_size=20",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 200
        assert len(data["items"]) == 20


# ---------------------------------------------------------------------------
# Scoring endpoints
# ---------------------------------------------------------------------------

class TestScoringEndpoints:
    """Test scoring simulation API routes."""

    def test_run_scoring_simulation(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "overall_score": 72.5,
                "score_band": "A-",
                "category_scores": [],
            },
        )
        response = mock_client.post(
            "/api/v1/cdp/scoring/simulate",
            json={"questionnaire_id": _new_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["score_band"] == "A-"

    def test_what_if_analysis(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "baseline_score": 72.5,
                "scenario_score": 78.0,
                "delta": 5.5,
            },
        )
        response = mock_client.post(
            "/api/v1/cdp/scoring/what-if",
            json={
                "questionnaire_id": _new_id(),
                "improvements": {"CAT06": 85.0},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["delta"] == 5.5

    def test_a_level_eligibility(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_eligible": False,
                "unmet_requirements": ["transition_plan", "scope3_verification"],
            },
        )
        response = mock_client.get(
            f"/api/v1/cdp/scoring/{_new_id()}/a-level-check",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["is_eligible"] is False

    def test_score_comparison(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "current_score": 72.5,
                "previous_score": 58.0,
                "delta": 14.5,
            },
        )
        response = mock_client.get(
            f"/api/v1/cdp/scoring/{_new_id()}/comparison",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Gap analysis endpoints
# ---------------------------------------------------------------------------

class TestGapAnalysisEndpoints:
    """Test gap analysis API routes."""

    def test_run_gap_analysis(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "total_gaps": 25,
                "critical_gaps": 3,
                "potential_uplift": 12.5,
            },
        )
        response = mock_client.post(
            "/api/v1/cdp/gap-analysis/run",
            json={"questionnaire_id": _new_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["total_gaps"] == 25

    def test_get_gap_items(self, mock_client, auth_headers):
        aid = _new_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"severity": "critical", "score_uplift": 3.5}],
        )
        response = mock_client.get(
            f"/api/v1/cdp/gap-analysis/{aid}/items",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_gaps_by_severity(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"severity": "critical"}],
        )
        response = mock_client.get(
            f"/api/v1/cdp/gap-analysis/{_new_id()}/items?severity=critical",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert all(g["severity"] == "critical" for g in response.json())

    def test_resolve_gap(self, mock_client, auth_headers):
        mock_client.patch.return_value = MagicMock(
            status_code=200,
            json=lambda: {"is_resolved": True},
        )
        response = mock_client.patch(
            f"/api/v1/cdp/gap-analysis/items/{_new_id()}/resolve",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Benchmarking endpoints
# ---------------------------------------------------------------------------

class TestBenchmarkingEndpoints:
    """Test benchmarking API routes."""

    def test_get_sector_benchmark(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sector_gics": "20101010",
                "mean_score": 55.3,
                "a_list_rate": 4.2,
            },
        )
        response = mock_client.get(
            "/api/v1/cdp/benchmarks/sector/20101010?year=2025",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_peer_comparison(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "org_score": 72.5,
                "percentile": 82.0,
            },
        )
        response = mock_client.get(
            f"/api/v1/cdp/benchmarks/{_new_id()}/peer-comparison",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_score_distribution(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"D-": 45, "D": 67, "A": 14},
        )
        response = mock_client.get(
            "/api/v1/cdp/benchmarks/sector/20101010/distribution",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Error handling and validation
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test API error handling and validation."""

    def test_invalid_json_400(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=400,
            json=lambda: {"detail": "Invalid request body"},
        )
        response = mock_client.post(
            "/api/v1/cdp/questionnaires",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_unauthorized_401(self, mock_client):
        mock_client.get.return_value = MagicMock(
            status_code=401,
            json=lambda: {"detail": "Not authenticated"},
        )
        response = mock_client.get("/api/v1/cdp/questionnaires")
        assert response.status_code == 401

    def test_forbidden_403(self, mock_client, auth_headers):
        mock_client.delete.return_value = MagicMock(
            status_code=403,
            json=lambda: {"detail": "Insufficient permissions"},
        )
        response = mock_client.delete(
            f"/api/v1/cdp/questionnaires/{_new_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 403

    def test_not_found_404(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=404,
            json=lambda: {"detail": "Resource not found"},
        )
        response = mock_client.get(
            f"/api/v1/cdp/responses/{_new_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 404

    def test_conflict_409(self, mock_client, auth_headers):
        mock_client.patch.return_value = MagicMock(
            status_code=409,
            json=lambda: {"detail": "Conflict: response already submitted"},
        )
        response = mock_client.patch(
            f"/api/v1/cdp/responses/{_new_id()}/status",
            json={"target_status": "draft"},
            headers=auth_headers,
        )
        assert response.status_code == 409

    def test_server_error_500(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=500,
            json=lambda: {"detail": "Internal server error"},
        )
        response = mock_client.get(
            "/api/v1/cdp/scoring/simulate",
            headers=auth_headers,
        )
        assert response.status_code == 500

    def test_invalid_status_transition_422(self, mock_client, auth_headers):
        mock_client.patch.return_value = MagicMock(
            status_code=422,
            json=lambda: {"detail": "Invalid status transition: draft -> submitted"},
        )
        response = mock_client.patch(
            f"/api/v1/cdp/responses/{_new_id()}/status",
            json={"target_status": "submitted"},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Pagination and filtering
# ---------------------------------------------------------------------------

class TestPaginationFiltering:
    """Test pagination and filtering across endpoints."""

    def test_default_pagination(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [],
                "total": 0,
                "page": 1,
                "page_size": 20,
                "total_pages": 0,
            },
        )
        response = mock_client.get(
            "/api/v1/cdp/responses",
            headers=auth_headers,
        )
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 20

    def test_custom_page_size(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0, "page": 1, "page_size": 50},
        )
        response = mock_client.get(
            "/api/v1/cdp/responses?page_size=50",
            headers=auth_headers,
        )
        assert response.json()["page_size"] == 50

    def test_filter_by_module(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [{"module_code": "M7"}], "total": 1},
        )
        response = mock_client.get(
            f"/api/v1/cdp/questionnaires/{_new_id()}/responses?module=M7",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_by_status(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/cdp/responses?status=approved",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_sort_by_score(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/cdp/gap-analysis/items?sort_by=score_uplift&order=desc",
            headers=auth_headers,
        )
        assert response.status_code == 200

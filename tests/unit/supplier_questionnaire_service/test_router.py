# -*- coding: utf-8 -*-
"""
Unit tests for Supplier Questionnaire Processor REST API Router - AGENT-DATA-008

Tests all 20 FastAPI router endpoints using httpx TestClient covering
template management, distribution, response collection, validation, scoring,
follow-up management, analytics, statistics, and health monitoring.

Each endpoint is tested for success paths, error paths (400, 404, 503),
and response format validation.

Target: 85%+ coverage of greenlang/supplier_questionnaire/api/router.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.supplier_questionnaire.api.router import router
from greenlang.supplier_questionnaire.config import SupplierQuestionnaireConfig
from greenlang.supplier_questionnaire.setup import (
    CampaignAnalytics,
    Distribution,
    FollowUpAction,
    QuestionnaireResponse,
    QuestionnaireStatistics,
    QuestionnaireTemplate,
    ScoringResult,
    SupplierQuestionnaireService,
    ValidationResult,
)


# ===================================================================
# Helpers
# ===================================================================

PREFIX = "/api/v1/questionnaires"


def _make_config(**overrides: Any) -> SupplierQuestionnaireConfig:
    """Build a SupplierQuestionnaireConfig with defaults for testing."""
    defaults = dict(
        database_url="",
        redis_url="",
        default_framework="custom",
        default_deadline_days=60,
        max_reminders=4,
        min_completion_pct=80.0,
        score_leader_threshold=80,
        score_advanced_threshold=60,
        score_developing_threshold=40,
        score_lagging_threshold=20,
    )
    defaults.update(overrides)
    return SupplierQuestionnaireConfig(**defaults)


def _make_sections(
    section_count: int = 1,
    questions_per_section: int = 2,
) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    qid = 0
    for s in range(section_count):
        qs: List[Dict[str, Any]] = []
        for _ in range(questions_per_section):
            qs.append({"id": f"q-{qid}", "text": f"Question {qid}", "type": "text"})
            qid += 1
        sections.append({
            "name": f"Section-{s}",
            "category": f"cat-{s % 3}",
            "questions": qs,
        })
    return sections


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def service() -> SupplierQuestionnaireService:
    """Create a fresh SupplierQuestionnaireService instance for each test."""
    with patch.object(SupplierQuestionnaireService, "_init_engines"):
        svc = SupplierQuestionnaireService(config=_make_config())
    svc.startup()
    return svc


@pytest.fixture
def app(service: SupplierQuestionnaireService) -> FastAPI:
    """Create a FastAPI app with the service attached and router included."""
    application = FastAPI()
    application.state.supplier_questionnaire_service = service
    application.include_router(router)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a TestClient from the app."""
    return TestClient(app)


@pytest.fixture
def created_template(client: TestClient) -> Dict[str, Any]:
    """POST a template and return the response JSON."""
    body = {
        "name": "CDP Climate 2025",
        "framework": "cdp",
        "version": "2.0",
        "description": "Climate disclosure questionnaire",
        "sections": _make_sections(section_count=2, questions_per_section=2),
        "language": "en",
        "tags": ["climate", "cdp"],
    }
    resp = client.post(f"{PREFIX}/v1/templates", json=body)
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def distributed(client: TestClient, created_template: Dict[str, Any]) -> Dict[str, Any]:
    """Distribute a questionnaire and return the response JSON."""
    body = {
        "template_id": created_template["template_id"],
        "supplier_id": "SUP-001",
        "supplier_name": "EcoSteel GmbH",
        "supplier_email": "contact@ecosteel.de",
        "campaign_id": "camp-001",
        "channel": "email",
    }
    resp = client.post(f"{PREFIX}/v1/distribute", json=body)
    assert resp.status_code == 200
    return resp.json()


@pytest.fixture
def submitted_response(
    client: TestClient,
    distributed: Dict[str, Any],
) -> Dict[str, Any]:
    """Submit a response and return the response JSON."""
    body = {
        "distribution_id": distributed["distribution_id"],
        "supplier_id": "SUP-001",
        "supplier_name": "EcoSteel GmbH",
        "answers": {"q-0": "100 MWh", "q-1": "Yes", "q-2": "50%", "q-3": "No"},
        "evidence_files": ["audit.pdf"],
        "channel": "portal",
    }
    resp = client.post(f"{PREFIX}/v1/responses", json=body)
    assert resp.status_code == 200
    return resp.json()


# ===================================================================
# 1. POST /v1/templates - Create Template
# ===================================================================


class TestCreateTemplate:
    """Tests for POST /v1/templates."""

    def test_create_template_success(self, client: TestClient) -> None:
        body = {"name": "Simple Template", "framework": "gri"}
        resp = client.post(f"{PREFIX}/v1/templates", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Simple Template"
        assert data["framework"] == "gri"
        assert "template_id" in data
        assert "provenance_hash" in data

    def test_create_template_with_sections(self, client: TestClient) -> None:
        sections = _make_sections(section_count=3, questions_per_section=2)
        body = {"name": "Full Template", "sections": sections}
        resp = client.post(f"{PREFIX}/v1/templates", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["questions"] == 6

    def test_create_template_empty_name_400(self, client: TestClient) -> None:
        body = {"name": ""}
        resp = client.post(f"{PREFIX}/v1/templates", json=body)
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_create_template_defaults(self, client: TestClient) -> None:
        body = {"name": "Defaults"}
        resp = client.post(f"{PREFIX}/v1/templates", json=body)
        data = resp.json()
        assert data["framework"] == "custom"
        assert data["version"] == "1.0"
        assert data["language"] == "en"
        assert data["status"] == "draft"

    def test_create_template_missing_name_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/templates", json={})
        assert resp.status_code == 422


# ===================================================================
# 2. GET /v1/templates - List Templates
# ===================================================================


class TestListTemplates:
    """Tests for GET /v1/templates."""

    def test_list_templates_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/templates")
        assert resp.status_code == 200
        data = resp.json()
        assert data["templates"] == []
        assert data["count"] == 0

    def test_list_templates_returns_created(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/templates")
        data = resp.json()
        assert data["count"] >= 1
        assert any(
            t["template_id"] == created_template["template_id"]
            for t in data["templates"]
        )

    def test_list_templates_filter_framework(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/templates?framework=cdp")
        data = resp.json()
        assert data["count"] >= 1
        assert all(t["framework"] == "cdp" for t in data["templates"])

    def test_list_templates_filter_no_match(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/templates?framework=nonexistent")
        data = resp.json()
        assert data["count"] == 0

    def test_list_templates_pagination(self, client: TestClient) -> None:
        for i in range(5):
            client.post(f"{PREFIX}/v1/templates", json={"name": f"T{i}"})
        resp = client.get(f"{PREFIX}/v1/templates?limit=3&offset=0")
        data = resp.json()
        assert data["count"] == 3
        assert data["limit"] == 3
        assert data["offset"] == 0

    def test_list_templates_offset(self, client: TestClient) -> None:
        for i in range(5):
            client.post(f"{PREFIX}/v1/templates", json={"name": f"T{i}"})
        resp = client.get(f"{PREFIX}/v1/templates?limit=50&offset=3")
        data = resp.json()
        assert data["count"] == 2

    def test_list_templates_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/templates")
        data = resp.json()
        assert "templates" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data


# ===================================================================
# 3. GET /v1/templates/{template_id} - Get Template
# ===================================================================


class TestGetTemplate:
    """Tests for GET /v1/templates/{template_id}."""

    def test_get_template_success(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.get(f"{PREFIX}/v1/templates/{tid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["template_id"] == tid
        assert data["name"] == created_template["name"]

    def test_get_template_not_found(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/templates/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_template_has_provenance_hash(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.get(f"{PREFIX}/v1/templates/{tid}")
        data = resp.json()
        assert len(data["provenance_hash"]) == 64


# ===================================================================
# 4. PUT /v1/templates/{template_id} - Update Template
# ===================================================================


class TestUpdateTemplate:
    """Tests for PUT /v1/templates/{template_id}."""

    def test_update_template_name(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.put(
            f"{PREFIX}/v1/templates/{tid}",
            json={"name": "Updated Name"},
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Name"

    def test_update_template_status(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.put(
            f"{PREFIX}/v1/templates/{tid}",
            json={"status": "active"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "active"

    def test_update_template_sections(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        new_sections = _make_sections(section_count=3, questions_per_section=3)
        resp = client.put(
            f"{PREFIX}/v1/templates/{tid}",
            json={"sections": new_sections},
        )
        assert resp.status_code == 200
        assert resp.json()["questions"] == 9

    def test_update_template_not_found(self, client: TestClient) -> None:
        resp = client.put(
            f"{PREFIX}/v1/templates/nonexistent-id",
            json={"name": "X"},
        )
        assert resp.status_code == 404

    def test_update_template_no_fields_no_crash(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.put(f"{PREFIX}/v1/templates/{tid}", json={})
        assert resp.status_code == 200


# ===================================================================
# 5. POST /v1/templates/{template_id}/clone - Clone Template
# ===================================================================


class TestCloneTemplate:
    """Tests for POST /v1/templates/{template_id}/clone."""

    def test_clone_template_success(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.post(f"{PREFIX}/v1/templates/{tid}/clone", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["template_id"] != tid
        assert "(Clone)" in data["name"]

    def test_clone_template_custom_name(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        tid = created_template["template_id"]
        resp = client.post(
            f"{PREFIX}/v1/templates/{tid}/clone",
            json={"new_name": "My Clone", "new_version": "5.0"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "My Clone"
        assert data["version"] == "5.0"

    def test_clone_template_not_found(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/templates/nonexistent/clone", json={},
        )
        assert resp.status_code == 404


# ===================================================================
# 6. POST /v1/distribute - Distribute Questionnaire
# ===================================================================


class TestDistribute:
    """Tests for POST /v1/distribute."""

    def test_distribute_success(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        body = {
            "template_id": created_template["template_id"],
            "supplier_id": "SUP-010",
            "supplier_name": "Test Supplier",
            "supplier_email": "test@test.com",
        }
        resp = client.post(f"{PREFIX}/v1/distribute", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "sent"
        assert data["supplier_id"] == "SUP-010"
        assert len(data["provenance_hash"]) == 64

    def test_distribute_with_campaign(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        body = {
            "template_id": created_template["template_id"],
            "supplier_id": "SUP-011",
            "supplier_name": "Camp Supplier",
            "supplier_email": "cs@test.com",
            "campaign_id": "camp-test-1",
        }
        resp = client.post(f"{PREFIX}/v1/distribute", json=body)
        assert resp.status_code == 200
        assert resp.json()["campaign_id"] == "camp-test-1"

    def test_distribute_custom_channel(
        self, client: TestClient, created_template: Dict[str, Any]
    ) -> None:
        body = {
            "template_id": created_template["template_id"],
            "supplier_id": "SUP-012",
            "supplier_name": "Portal Sup",
            "supplier_email": "p@test.com",
            "channel": "portal",
        }
        resp = client.post(f"{PREFIX}/v1/distribute", json=body)
        assert resp.status_code == 200
        assert resp.json()["channel"] == "portal"

    def test_distribute_template_not_found_400(self, client: TestClient) -> None:
        body = {
            "template_id": "nonexistent",
            "supplier_id": "SUP-001",
            "supplier_name": "X",
            "supplier_email": "x@x.com",
        }
        resp = client.post(f"{PREFIX}/v1/distribute", json=body)
        assert resp.status_code == 400

    def test_distribute_missing_required_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/distribute", json={"template_id": "t1"})
        assert resp.status_code == 422


# ===================================================================
# 7. GET /v1/distributions - List Distributions
# ===================================================================


class TestListDistributions:
    """Tests for GET /v1/distributions."""

    def test_list_distributions_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distributions"] == []
        assert data["count"] == 0

    def test_list_distributions_with_data(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions")
        data = resp.json()
        assert data["count"] >= 1

    def test_list_distributions_filter_campaign(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions?campaign_id=camp-001")
        data = resp.json()
        assert data["count"] >= 1
        assert all(d["campaign_id"] == "camp-001" for d in data["distributions"])

    def test_list_distributions_filter_supplier(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions?supplier_id=SUP-001")
        data = resp.json()
        assert data["count"] >= 1

    def test_list_distributions_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions")
        data = resp.json()
        assert "distributions" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data


# ===================================================================
# 8. GET /v1/distributions/{dist_id} - Get Distribution
# ===================================================================


class TestGetDistribution:
    """Tests for GET /v1/distributions/{dist_id}."""

    def test_get_distribution_success(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        did = distributed["distribution_id"]
        resp = client.get(f"{PREFIX}/v1/distributions/{did}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["distribution_id"] == did
        assert data["supplier_id"] == "SUP-001"

    def test_get_distribution_not_found(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/distributions/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===================================================================
# 9. POST /v1/responses - Submit Response
# ===================================================================


class TestSubmitResponse:
    """Tests for POST /v1/responses."""

    def test_submit_response_success(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        body = {
            "distribution_id": distributed["distribution_id"],
            "supplier_id": "SUP-001",
            "supplier_name": "EcoSteel",
            "answers": {"q-0": "100"},
        }
        resp = client.post(f"{PREFIX}/v1/responses", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "submitted"
        assert data["supplier_id"] == "SUP-001"

    def test_submit_response_full_answers(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        assert submitted_response["status"] == "submitted"
        assert submitted_response["completion_pct"] > 0.0

    def test_submit_response_dist_not_found_400(self, client: TestClient) -> None:
        body = {
            "distribution_id": "bad-dist",
            "supplier_id": "SUP-001",
            "supplier_name": "X",
            "answers": {"q-0": "val"},
        }
        resp = client.post(f"{PREFIX}/v1/responses", json=body)
        assert resp.status_code == 400

    def test_submit_response_missing_fields_422(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/responses",
            json={"distribution_id": "d1"},
        )
        assert resp.status_code == 422

    def test_submit_response_with_evidence(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        body = {
            "distribution_id": distributed["distribution_id"],
            "supplier_id": "SUP-001",
            "supplier_name": "EcoSteel",
            "answers": {"q-0": "Yes"},
            "evidence_files": ["cert.pdf", "report.xlsx"],
        }
        resp = client.post(f"{PREFIX}/v1/responses", json=body)
        assert resp.status_code == 200
        assert resp.json()["evidence_files"] == ["cert.pdf", "report.xlsx"]


# ===================================================================
# 10. GET /v1/responses - List Responses
# ===================================================================


class TestListResponses:
    """Tests for GET /v1/responses."""

    def test_list_responses_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/responses")
        assert resp.status_code == 200
        data = resp.json()
        assert data["responses"] == []
        assert data["count"] == 0

    def test_list_responses_with_data(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/responses")
        data = resp.json()
        assert data["count"] >= 1

    def test_list_responses_filter_supplier(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/responses?supplier_id=SUP-001")
        data = resp.json()
        assert data["count"] >= 1

    def test_list_responses_filter_status(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/responses?status=submitted")
        data = resp.json()
        assert data["count"] >= 1

    def test_list_responses_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/responses")
        data = resp.json()
        assert "responses" in data
        assert "count" in data
        assert "limit" in data
        assert "offset" in data


# ===================================================================
# 11. GET /v1/responses/{response_id} - Get Response
# ===================================================================


class TestGetResponse:
    """Tests for GET /v1/responses/{response_id}."""

    def test_get_response_success(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.get(f"{PREFIX}/v1/responses/{rid}")
        assert resp.status_code == 200
        assert resp.json()["response_id"] == rid

    def test_get_response_not_found(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/responses/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_response_has_answers(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.get(f"{PREFIX}/v1/responses/{rid}")
        data = resp.json()
        assert "answers" in data
        assert len(data["answers"]) > 0


# ===================================================================
# 12. POST /v1/responses/{response_id}/validate - Validate Response
# ===================================================================


class TestValidateResponse:
    """Tests for POST /v1/responses/{response_id}/validate."""

    def test_validate_success(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.post(
            f"{PREFIX}/v1/responses/{rid}/validate",
            json={"level": "completeness"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_valid" in data
        assert "validation_id" in data
        assert "provenance_hash" in data

    def test_validate_consistency_level(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.post(
            f"{PREFIX}/v1/responses/{rid}/validate",
            json={"level": "consistency"},
        )
        assert resp.status_code == 200
        assert resp.json()["level"] == "consistency"

    def test_validate_evidence_level(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.post(
            f"{PREFIX}/v1/responses/{rid}/validate",
            json={"level": "evidence"},
        )
        assert resp.status_code == 200
        assert resp.json()["level"] == "evidence"

    def test_validate_not_found(self, client: TestClient) -> None:
        resp = client.post(
            f"{PREFIX}/v1/responses/nonexistent/validate",
            json={"level": "completeness"},
        )
        assert resp.status_code == 404

    def test_validate_default_level(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.post(
            f"{PREFIX}/v1/responses/{rid}/validate",
            json={},
        )
        assert resp.status_code == 200
        assert resp.json()["level"] == "completeness"

    def test_validate_response_format(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        rid = submitted_response["response_id"]
        resp = client.post(
            f"{PREFIX}/v1/responses/{rid}/validate",
            json={"level": "completeness"},
        )
        data = resp.json()
        assert "errors" in data
        assert "warnings" in data
        assert "checks_passed" in data
        assert "checks_failed" in data
        assert "checks_warned" in data
        assert "completion_pct" in data


# ===================================================================
# 13. POST /v1/score - Score Response
# ===================================================================


class TestScoreResponse:
    """Tests for POST /v1/score."""

    def test_score_success(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        body = {"response_id": submitted_response["response_id"]}
        resp = client.post(f"{PREFIX}/v1/score", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "score_id" in data
        assert "total_score" in data
        assert "tier" in data
        assert data["tier"] in ("leader", "advanced", "developing", "lagging")

    def test_score_with_framework_override(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        body = {
            "response_id": submitted_response["response_id"],
            "framework": "ecovadis",
        }
        resp = client.post(f"{PREFIX}/v1/score", json=body)
        assert resp.status_code == 200
        assert resp.json()["framework"] == "ecovadis"

    def test_score_not_found_400(self, client: TestClient) -> None:
        body = {"response_id": "nonexistent"}
        resp = client.post(f"{PREFIX}/v1/score", json=body)
        assert resp.status_code == 400

    def test_score_missing_field_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/score", json={})
        assert resp.status_code == 422

    def test_score_has_section_scores(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        body = {"response_id": submitted_response["response_id"]}
        resp = client.post(f"{PREFIX}/v1/score", json=body)
        data = resp.json()
        assert "section_scores" in data
        assert "category_scores" in data

    def test_score_provenance_hash(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        body = {"response_id": submitted_response["response_id"]}
        resp = client.post(f"{PREFIX}/v1/score", json=body)
        assert len(resp.json()["provenance_hash"]) == 64


# ===================================================================
# 14. GET /v1/scores/{score_id} - Get Score
# ===================================================================


class TestGetScore:
    """Tests for GET /v1/scores/{score_id}."""

    def test_get_score_success(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        # First score the response
        score_resp = client.post(
            f"{PREFIX}/v1/score",
            json={"response_id": submitted_response["response_id"]},
        )
        score_id = score_resp.json()["score_id"]

        resp = client.get(f"{PREFIX}/v1/scores/{score_id}")
        assert resp.status_code == 200
        assert resp.json()["score_id"] == score_id

    def test_get_score_not_found(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/scores/nonexistent")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ===================================================================
# 15. GET /v1/scores/supplier/{supplier_id} - Get Supplier Scores
# ===================================================================


class TestGetSupplierScores:
    """Tests for GET /v1/scores/supplier/{supplier_id}."""

    def test_supplier_scores_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/scores/supplier/SUP-999")
        assert resp.status_code == 200
        data = resp.json()
        assert data["scores"] == []
        assert data["count"] == 0
        assert data["supplier_id"] == "SUP-999"

    def test_supplier_scores_with_data(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        client.post(
            f"{PREFIX}/v1/score",
            json={"response_id": submitted_response["response_id"]},
        )
        resp = client.get(f"{PREFIX}/v1/scores/supplier/SUP-001")
        data = resp.json()
        assert data["count"] >= 1
        assert data["supplier_id"] == "SUP-001"

    def test_supplier_scores_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/scores/supplier/SUP-001")
        data = resp.json()
        assert "scores" in data
        assert "count" in data
        assert "supplier_id" in data
        assert "limit" in data
        assert "offset" in data


# ===================================================================
# 16. POST /v1/followup - Trigger Follow-Up
# ===================================================================


class TestTriggerFollowUp:
    """Tests for POST /v1/followup."""

    def test_trigger_reminder_success(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        body = {
            "distribution_id": distributed["distribution_id"],
            "action_type": "reminder",
            "message": "Please respond",
        }
        resp = client.post(f"{PREFIX}/v1/followup", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_type"] == "reminder"
        assert data["status"] == "sent"
        assert data["message"] == "Please respond"

    def test_trigger_escalation(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        body = {
            "distribution_id": distributed["distribution_id"],
            "action_type": "escalation",
            "message": "Overdue notice",
        }
        resp = client.post(f"{PREFIX}/v1/followup", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["action_type"] == "escalation"

    def test_trigger_followup_dist_not_found_400(self, client: TestClient) -> None:
        body = {
            "distribution_id": "nonexistent",
            "action_type": "reminder",
        }
        resp = client.post(f"{PREFIX}/v1/followup", json=body)
        assert resp.status_code == 400

    def test_trigger_followup_default_reminder(
        self, client: TestClient, distributed: Dict[str, Any]
    ) -> None:
        body = {"distribution_id": distributed["distribution_id"]}
        resp = client.post(f"{PREFIX}/v1/followup", json=body)
        assert resp.status_code == 200
        assert resp.json()["action_type"] == "reminder"

    def test_trigger_followup_missing_dist_422(self, client: TestClient) -> None:
        resp = client.post(f"{PREFIX}/v1/followup", json={})
        assert resp.status_code == 422


# ===================================================================
# 17. GET /v1/followup/{campaign_id} - Follow-Up Status
# ===================================================================


class TestGetFollowUpStatus:
    """Tests for GET /v1/followup/{campaign_id}."""

    def test_followup_status_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/followup/some-campaign")
        assert resp.status_code == 200
        data = resp.json()
        assert data["campaign_id"] == "some-campaign"
        assert data["pending_reminders"] == []
        assert data["count"] == 0

    def test_followup_status_with_scheduled(
        self,
        client: TestClient,
        service: SupplierQuestionnaireService,
        distributed: Dict[str, Any],
    ) -> None:
        # Schedule reminders via the service directly (since there is no schedule endpoint)
        service.schedule_reminders("camp-001")
        resp = client.get(f"{PREFIX}/v1/followup/camp-001")
        data = resp.json()
        assert data["count"] >= 1

    def test_followup_status_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/followup/any-camp")
        data = resp.json()
        assert "campaign_id" in data
        assert "pending_reminders" in data
        assert "count" in data


# ===================================================================
# 18. GET /v1/analytics/{campaign_id} - Campaign Analytics
# ===================================================================


class TestGetAnalytics:
    """Tests for GET /v1/analytics/{campaign_id}."""

    def test_analytics_empty_campaign(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/nonexistent-camp")
        assert resp.status_code == 200
        data = resp.json()
        assert data["campaign_id"] == "nonexistent-camp"
        assert data["total_distributed"] == 0
        assert data["response_rate_pct"] == 0.0

    def test_analytics_with_data(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/camp-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_distributed"] >= 1
        assert data["total_responded"] >= 1
        assert data["response_rate_pct"] > 0.0

    def test_analytics_response_format(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/any-camp")
        data = resp.json()
        assert "campaign_id" in data
        assert "total_distributed" in data
        assert "total_responded" in data
        assert "total_finalized" in data
        assert "response_rate_pct" in data
        assert "avg_completion_pct" in data
        assert "avg_score" in data
        assert "score_distribution" in data
        assert "compliance_gaps" in data
        assert "provenance_hash" in data

    def test_analytics_provenance_hash_is_sha256(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/analytics/camp-001")
        data = resp.json()
        assert len(data["provenance_hash"]) == 64


# ===================================================================
# 19. GET /health - Health Check
# ===================================================================


class TestHealthCheck:
    """Tests for GET /health."""

    def test_health_check_200(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/health")
        assert resp.status_code == 200

    def test_health_check_status_healthy(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "supplier-questionnaire"
        assert data["started"] is True

    def test_health_check_contains_counts(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/health")
        data = resp.json()
        assert data["templates"] >= 1
        assert data["distributions"] >= 1
        assert data["responses"] >= 1
        assert "provenance_entries" in data
        assert "prometheus_available" in data


# ===================================================================
# 20. GET /v1/statistics - Statistics
# ===================================================================


class TestStatistics:
    """Tests for GET /v1/statistics."""

    def test_statistics_200(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/statistics")
        assert resp.status_code == 200

    def test_statistics_empty(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/statistics")
        data = resp.json()
        assert data["total_templates"] == 0
        assert data["total_distributions"] == 0
        assert data["total_responses"] == 0

    def test_statistics_after_workflow(
        self, client: TestClient, submitted_response: Dict[str, Any]
    ) -> None:
        resp = client.get(f"{PREFIX}/v1/statistics")
        data = resp.json()
        assert data["total_templates"] >= 1
        assert data["total_distributions"] >= 1
        assert data["total_responses"] >= 1

    def test_statistics_response_shape(self, client: TestClient) -> None:
        resp = client.get(f"{PREFIX}/v1/statistics")
        data = resp.json()
        required_keys = [
            "total_templates",
            "active_templates",
            "total_distributions",
            "total_responses",
            "total_finalized",
            "total_validations",
            "total_scores",
            "total_followups",
            "total_campaigns",
            "active_campaigns",
            "avg_response_rate_pct",
            "avg_score",
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"


# ===================================================================
# 21. Service Not Configured (503)
# ===================================================================


class TestServiceNotConfigured:
    """Test endpoints when the service is not attached to app state."""

    @pytest.fixture
    def bare_client(self) -> TestClient:
        """Create a TestClient with NO service configured."""
        application = FastAPI()
        application.include_router(router)
        return TestClient(application)

    def test_create_template_503(self, bare_client: TestClient) -> None:
        resp = bare_client.post(
            f"{PREFIX}/v1/templates", json={"name": "X"}
        )
        assert resp.status_code == 503

    def test_list_templates_503(self, bare_client: TestClient) -> None:
        resp = bare_client.get(f"{PREFIX}/v1/templates")
        assert resp.status_code == 503

    def test_get_template_503(self, bare_client: TestClient) -> None:
        resp = bare_client.get(f"{PREFIX}/v1/templates/xxx")
        assert resp.status_code == 503

    def test_distribute_503(self, bare_client: TestClient) -> None:
        resp = bare_client.post(
            f"{PREFIX}/v1/distribute",
            json={
                "template_id": "t",
                "supplier_id": "s",
                "supplier_name": "n",
                "supplier_email": "e@e.com",
            },
        )
        assert resp.status_code == 503

    def test_submit_response_503(self, bare_client: TestClient) -> None:
        resp = bare_client.post(
            f"{PREFIX}/v1/responses",
            json={
                "distribution_id": "d",
                "supplier_id": "s",
                "supplier_name": "n",
            },
        )
        assert resp.status_code == 503

    def test_validate_503(self, bare_client: TestClient) -> None:
        resp = bare_client.post(
            f"{PREFIX}/v1/responses/xxx/validate",
            json={"level": "completeness"},
        )
        assert resp.status_code == 503

    def test_score_503(self, bare_client: TestClient) -> None:
        resp = bare_client.post(
            f"{PREFIX}/v1/score",
            json={"response_id": "r"},
        )
        assert resp.status_code == 503

    def test_health_503(self, bare_client: TestClient) -> None:
        resp = bare_client.get(f"{PREFIX}/health")
        assert resp.status_code == 503

    def test_statistics_503(self, bare_client: TestClient) -> None:
        resp = bare_client.get(f"{PREFIX}/v1/statistics")
        assert resp.status_code == 503

    def test_analytics_503(self, bare_client: TestClient) -> None:
        resp = bare_client.get(f"{PREFIX}/v1/analytics/camp-x")
        assert resp.status_code == 503


# ===================================================================
# 22. Full API Workflow Test
# ===================================================================


class TestFullAPIWorkflow:
    """End-to-end test through all API endpoints."""

    def test_complete_workflow_via_api(self, client: TestClient) -> None:
        # 1. Create template
        sections = _make_sections(section_count=2, questions_per_section=2)
        create_resp = client.post(
            f"{PREFIX}/v1/templates",
            json={"name": "E2E Template", "framework": "cdp", "sections": sections},
        )
        assert create_resp.status_code == 200
        template = create_resp.json()
        template_id = template["template_id"]

        # 2. List templates (should include our template)
        list_resp = client.get(f"{PREFIX}/v1/templates")
        assert list_resp.json()["count"] >= 1

        # 3. Get template
        get_resp = client.get(f"{PREFIX}/v1/templates/{template_id}")
        assert get_resp.status_code == 200

        # 4. Update template
        update_resp = client.put(
            f"{PREFIX}/v1/templates/{template_id}",
            json={"status": "active"},
        )
        assert update_resp.json()["status"] == "active"

        # 5. Clone template
        clone_resp = client.post(
            f"{PREFIX}/v1/templates/{template_id}/clone",
            json={"new_name": "E2E Clone"},
        )
        assert clone_resp.status_code == 200

        # 6. Distribute
        dist_resp = client.post(
            f"{PREFIX}/v1/distribute",
            json={
                "template_id": template_id,
                "supplier_id": "SUP-E2E",
                "supplier_name": "E2E Supplier",
                "supplier_email": "e2e@test.com",
                "campaign_id": "camp-e2e",
            },
        )
        assert dist_resp.status_code == 200
        dist = dist_resp.json()

        # 7. List distributions
        list_dist = client.get(f"{PREFIX}/v1/distributions")
        assert list_dist.json()["count"] >= 1

        # 8. Get distribution
        get_dist = client.get(
            f"{PREFIX}/v1/distributions/{dist['distribution_id']}"
        )
        assert get_dist.status_code == 200

        # 9. Submit response
        submit_resp = client.post(
            f"{PREFIX}/v1/responses",
            json={
                "distribution_id": dist["distribution_id"],
                "supplier_id": "SUP-E2E",
                "supplier_name": "E2E Supplier",
                "answers": {"q-0": "100", "q-1": "Yes", "q-2": "50", "q-3": "No"},
                "evidence_files": ["e2e_audit.pdf"],
            },
        )
        assert submit_resp.status_code == 200
        response = submit_resp.json()

        # 10. List responses
        list_responses = client.get(f"{PREFIX}/v1/responses")
        assert list_responses.json()["count"] >= 1

        # 11. Get response
        get_response = client.get(
            f"{PREFIX}/v1/responses/{response['response_id']}"
        )
        assert get_response.status_code == 200

        # 12. Validate response
        validate_resp = client.post(
            f"{PREFIX}/v1/responses/{response['response_id']}/validate",
            json={"level": "evidence"},
        )
        assert validate_resp.status_code == 200
        assert validate_resp.json()["is_valid"] is True

        # 13. Score response
        score_resp = client.post(
            f"{PREFIX}/v1/score",
            json={"response_id": response["response_id"]},
        )
        assert score_resp.status_code == 200
        score = score_resp.json()

        # 14. Get score
        get_score = client.get(f"{PREFIX}/v1/scores/{score['score_id']}")
        assert get_score.status_code == 200

        # 15. Get supplier scores
        sup_scores = client.get(f"{PREFIX}/v1/scores/supplier/SUP-E2E")
        assert sup_scores.json()["count"] >= 1

        # 16. Trigger follow-up
        followup_resp = client.post(
            f"{PREFIX}/v1/followup",
            json={"distribution_id": dist["distribution_id"]},
        )
        assert followup_resp.status_code == 200

        # 17. Get follow-up status
        followup_status = client.get(f"{PREFIX}/v1/followup/camp-e2e")
        assert followup_status.status_code == 200

        # 18. Get analytics
        analytics_resp = client.get(f"{PREFIX}/v1/analytics/camp-e2e")
        assert analytics_resp.status_code == 200
        analytics = analytics_resp.json()
        assert analytics["total_distributed"] >= 1
        assert analytics["total_responded"] >= 1

        # 19. Health check
        health_resp = client.get(f"{PREFIX}/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # 20. Statistics
        stats_resp = client.get(f"{PREFIX}/v1/statistics")
        assert stats_resp.status_code == 200
        stats = stats_resp.json()
        assert stats["total_templates"] >= 2  # original + clone
        assert stats["total_distributions"] >= 1
        assert stats["total_responses"] >= 1
        assert stats["total_validations"] >= 1
        assert stats["total_scores"] >= 1

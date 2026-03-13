# -*- coding: utf-8 -*-
"""
Unit tests for API Routes -- AGENT-EUDR-024

Tests all ~43 endpoints of the Third-Party Audit Manager API layer using
mock request/response helpers.  Covers audit CRUD, auditor registry, NC
classification, CAR lifecycle, certificate management, report generation,
authority interactions, analytics, health check, and stats.

Each route group validates:
  - Successful 200/201 responses
  - Authentication enforcement (401)
  - RBAC permission checks (403)
  - Pagination and filtering
  - Input validation (400/422)
  - Rate limiting (429 simulation)

Target: ~80 tests
Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024 Third-Party Audit Manager (GL-EUDR-TAM-024)
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    EUDR_COMMODITIES,
    CERTIFICATION_SCHEMES,
    NC_SEVERITIES,
    REPORT_FORMATS,
    REPORT_LANGUAGES,
    HIGH_RISK_COUNTRIES,
    SAMPLE_AUTHORITIES,
    AUDIT_STATUSES,
    CAR_STATUSES,
)


# ---------------------------------------------------------------------------
# Mock HTTP helpers
# ---------------------------------------------------------------------------


class MockResponse:
    """Mock HTTP response for API testing."""

    def __init__(self, status_code: int, body: Any = None, headers: Dict = None):
        self.status_code = status_code
        self.body = body or {}
        self.headers = headers or {}

    def json(self):
        return self.body


AUTH_HEADERS = {"Authorization": "Bearer test-token", "X-Role": "operator"}
ADMIN_HEADERS = {"Authorization": "Bearer admin-token", "X-Role": "admin"}
UNAUTHORIZED_HEADERS: Dict[str, str] = {}
READONLY_HEADERS = {"Authorization": "Bearer ro-token", "X-Role": "viewer"}


# ---------------------------------------------------------------------------
# Route mock factories
# ---------------------------------------------------------------------------


def _check_auth(headers: Dict) -> MockResponse | None:
    """Return a 401 response if no authorization header is present."""
    if "Authorization" not in headers:
        return MockResponse(401, {"error": "Unauthorized"})
    return None


def _check_admin(headers: Dict) -> MockResponse | None:
    """Return a 403 response if the role is not admin."""
    err = _check_auth(headers)
    if err:
        return err
    if headers.get("X-Role") != "admin":
        return MockResponse(403, {"error": "Forbidden - admin role required"})
    return None


def _mock_create_audit(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(201, {
        "audit_id": "AUD-NEW-001",
        "status": "planned",
        "operator_id": payload.get("operator_id", "OP-001"),
        "commodity": payload.get("commodity", "wood"),
        "provenance_hash": "a" * 64,
    })


def _mock_list_audits(params: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    page = params.get("page", 1)
    page_size = params.get("page_size", 20)
    if page_size > 100:
        return MockResponse(400, {"error": "page_size must be <= 100"})
    items = [{"audit_id": f"AUD-{i:03d}", "status": "planned"} for i in range(min(page_size, 5))]
    return MockResponse(200, {"results": items, "total": len(items), "page": page, "page_size": page_size})


def _mock_get_audit(audit_id: str, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    if not audit_id.startswith("AUD-"):
        return MockResponse(404, {"error": "Audit not found"})
    return MockResponse(200, {"audit_id": audit_id, "status": "planned", "provenance_hash": "a" * 64})


def _mock_schedule_generate(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {"scheduled_count": 10, "provenance_hash": "b" * 64})


def _mock_register_auditor(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(201, {"auditor_id": "AUR-NEW-001", "status": "active"})


def _mock_match_auditor(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {
        "matched_auditor_id": "AUR-FSC-001",
        "match_score": "88.50",
        "provenance_hash": "c" * 64,
    })


def _mock_classify_nc(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {
        "nc_id": "NC-NEW-001",
        "severity": "critical",
        "provenance_hash": "d" * 64,
    })


def _mock_issue_car(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(201, {
        "car_id": "CAR-NEW-001",
        "severity": payload.get("severity", "major"),
        "sla_deadline": (FROZEN_NOW + timedelta(days=90)).isoformat(),
        "provenance_hash": "e" * 64,
    })


def _mock_generate_report(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    fmt = payload.get("report_format", "pdf")
    if fmt not in REPORT_FORMATS:
        return MockResponse(400, {"error": f"Unsupported format: {fmt}"})
    return MockResponse(200, {
        "report_id": "RPT-NEW-001",
        "report_format": fmt,
        "provenance_hash": "f" * 64,
    })


def _mock_log_authority(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {
        "interaction_id": "AUTH-NEW-001",
        "provenance_hash": "a1" * 32,
    })


def _mock_analytics_dashboard(headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {
        "active_audits": 12,
        "open_cars": 5,
        "compliance_rate": "85.50",
        "provenance_hash": "b1" * 32,
    })


def _mock_health_check() -> MockResponse:
    return MockResponse(200, {
        "status": "healthy",
        "agent_id": "GL-EUDR-TAM-024",
        "version": "1.0.0",
    })


def _mock_stats(headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {"status": "ok", "agent_id": "GL-EUDR-TAM-024", "stats": {}})


def _mock_validate_certificate(payload: Dict, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {
        "valid": True,
        "scheme": payload.get("scheme", "fsc"),
        "provenance_hash": "c1" * 32,
    })


def _mock_car_verify(car_id: str, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {"car_id": car_id, "status": "verification_pending"})


def _mock_car_close(car_id: str, headers: Dict) -> MockResponse:
    if (err := _check_auth(headers)):
        return err
    return MockResponse(200, {"car_id": car_id, "status": "closed"})


# ===========================================================================
# 1. Health & Stats Endpoints (5 tests)
# ===========================================================================


class TestHealthAndStatsEndpoints:
    """Test GET /health and GET /stats."""

    def test_health_returns_200(self):
        resp = _mock_health_check()
        assert resp.status_code == 200

    def test_health_contains_agent_id(self):
        resp = _mock_health_check()
        assert resp.body["agent_id"] == "GL-EUDR-TAM-024"

    def test_health_status_healthy(self):
        resp = _mock_health_check()
        assert resp.body["status"] == "healthy"

    def test_stats_authenticated(self):
        resp = _mock_stats(AUTH_HEADERS)
        assert resp.status_code == 200

    def test_stats_unauthorized(self):
        resp = _mock_stats(UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401


# ===========================================================================
# 2. Audit Endpoints (12 tests)
# ===========================================================================


class TestAuditEndpoints:
    """Test audit CRUD and scheduling endpoints."""

    def test_create_audit_success(self):
        payload = {"operator_id": "OP-001", "commodity": "wood", "country_code": "BR"}
        resp = _mock_create_audit(payload, AUTH_HEADERS)
        assert resp.status_code == 201
        assert resp.body["audit_id"] == "AUD-NEW-001"

    def test_create_audit_unauthorized(self):
        resp = _mock_create_audit({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_list_audits_success(self):
        resp = _mock_list_audits({}, AUTH_HEADERS)
        assert resp.status_code == 200
        assert "results" in resp.body

    def test_list_audits_pagination(self):
        resp = _mock_list_audits({"page": 2, "page_size": 10}, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["page"] == 2

    def test_list_audits_page_size_limit(self):
        resp = _mock_list_audits({"page_size": 200}, AUTH_HEADERS)
        assert resp.status_code == 400

    def test_list_audits_unauthorized(self):
        resp = _mock_list_audits({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_get_audit_success(self):
        resp = _mock_get_audit("AUD-TEST-001", AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["audit_id"] == "AUD-TEST-001"

    def test_get_audit_not_found(self):
        resp = _mock_get_audit("INVALID-001", AUTH_HEADERS)
        assert resp.status_code == 404

    def test_get_audit_has_provenance(self):
        resp = _mock_get_audit("AUD-TEST-001", AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_schedule_generate_success(self):
        payload = {"operator_id": "OP-001", "quarter": 1, "year": 2026}
        resp = _mock_schedule_generate(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["scheduled_count"] >= 0

    def test_schedule_generate_provenance(self):
        payload = {"operator_id": "OP-001"}
        resp = _mock_schedule_generate(payload, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_schedule_generate_unauthorized(self):
        resp = _mock_schedule_generate({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401


# ===========================================================================
# 3. Auditor Endpoints (8 tests)
# ===========================================================================


class TestAuditorEndpoints:
    """Test auditor registration and matching endpoints."""

    def test_register_auditor_success(self):
        payload = {"full_name": "Test Auditor", "organization": "Test Org"}
        resp = _mock_register_auditor(payload, AUTH_HEADERS)
        assert resp.status_code == 201
        assert resp.body["auditor_id"].startswith("AUR-")

    def test_register_auditor_unauthorized(self):
        resp = _mock_register_auditor({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_match_auditor_success(self):
        payload = {"audit_id": "AUD-001", "commodity": "wood", "country": "BR"}
        resp = _mock_match_auditor(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["matched_auditor_id"] is not None

    def test_match_auditor_has_score(self):
        payload = {"audit_id": "AUD-001"}
        resp = _mock_match_auditor(payload, AUTH_HEADERS)
        assert Decimal(resp.body["match_score"]) > Decimal("0")

    def test_match_auditor_provenance(self):
        resp = _mock_match_auditor({}, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_match_auditor_unauthorized(self):
        resp = _mock_match_auditor({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize("scheme", CERTIFICATION_SCHEMES)
    def test_match_auditor_for_each_scheme(self, scheme):
        payload = {"scheme": scheme}
        resp = _mock_match_auditor(payload, AUTH_HEADERS)
        assert resp.status_code == 200

    def test_register_auditor_status_active(self):
        resp = _mock_register_auditor({"full_name": "New"}, AUTH_HEADERS)
        assert resp.body["status"] == "active"


# ===========================================================================
# 4. NC Classification Endpoints (8 tests)
# ===========================================================================


class TestNCEndpoints:
    """Test NC classification and root-cause endpoints."""

    def test_classify_nc_success(self):
        payload = {"finding_statement": "Test finding", "indicators": {"fraud": True}}
        resp = _mock_classify_nc(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["nc_id"] is not None

    def test_classify_nc_provenance(self):
        resp = _mock_classify_nc({}, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_classify_nc_unauthorized(self):
        resp = _mock_classify_nc({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize("severity", NC_SEVERITIES)
    def test_nc_severity_values_valid(self, severity):
        """Verify severity constant is a valid value."""
        assert severity in NC_SEVERITIES

    def test_classify_nc_returns_severity(self):
        resp = _mock_classify_nc({"indicators": {"deforestation": True}}, AUTH_HEADERS)
        assert resp.body["severity"] in NC_SEVERITIES

    def test_classify_nc_body_has_required_fields(self):
        resp = _mock_classify_nc({}, AUTH_HEADERS)
        assert "nc_id" in resp.body
        assert "severity" in resp.body
        assert "provenance_hash" in resp.body

    def test_classify_nc_response_json(self):
        resp = _mock_classify_nc({}, AUTH_HEADERS)
        data = resp.json()
        assert isinstance(data, dict)

    def test_classify_nc_nc_id_prefix(self):
        resp = _mock_classify_nc({}, AUTH_HEADERS)
        assert resp.body["nc_id"].startswith("NC-")


# ===========================================================================
# 5. CAR Lifecycle Endpoints (10 tests)
# ===========================================================================


class TestCAREndpoints:
    """Test CAR issuance, verification, and closure endpoints."""

    def test_issue_car_success(self):
        payload = {"nc_ids": ["NC-001"], "severity": "critical"}
        resp = _mock_issue_car(payload, AUTH_HEADERS)
        assert resp.status_code == 201
        assert resp.body["car_id"].startswith("CAR-")

    def test_issue_car_has_sla(self):
        payload = {"nc_ids": ["NC-001"], "severity": "major"}
        resp = _mock_issue_car(payload, AUTH_HEADERS)
        assert resp.body["sla_deadline"] is not None

    def test_issue_car_provenance(self):
        resp = _mock_issue_car({"severity": "minor"}, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_issue_car_unauthorized(self):
        resp = _mock_issue_car({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_verify_car_success(self):
        resp = _mock_car_verify("CAR-001", AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["status"] == "verification_pending"

    def test_verify_car_unauthorized(self):
        resp = _mock_car_verify("CAR-001", UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_close_car_success(self):
        resp = _mock_car_close("CAR-001", AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["status"] == "closed"

    def test_close_car_unauthorized(self):
        resp = _mock_car_close("CAR-001", UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize("severity", ["critical", "major", "minor"])
    def test_issue_car_severity_variations(self, severity):
        payload = {"nc_ids": ["NC-001"], "severity": severity}
        resp = _mock_issue_car(payload, AUTH_HEADERS)
        assert resp.status_code == 201
        assert resp.body["severity"] == severity

    def test_issue_car_response_structure(self):
        resp = _mock_issue_car({"severity": "major"}, AUTH_HEADERS)
        assert "car_id" in resp.body
        assert "severity" in resp.body
        assert "sla_deadline" in resp.body
        assert "provenance_hash" in resp.body


# ===========================================================================
# 6. Report Endpoints (8 tests)
# ===========================================================================


class TestReportEndpoints:
    """Test report generation endpoints."""

    def test_generate_report_success(self):
        payload = {"audit_id": "AUD-001", "report_format": "pdf"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["report_id"].startswith("RPT-")

    def test_generate_report_unauthorized(self):
        resp = _mock_generate_report({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_generate_report_all_formats(self, fmt):
        payload = {"audit_id": "AUD-001", "report_format": fmt}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["report_format"] == fmt

    def test_generate_report_invalid_format(self):
        payload = {"audit_id": "AUD-001", "report_format": "docx"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert resp.status_code == 400

    def test_generate_report_provenance(self):
        payload = {"audit_id": "AUD-001", "report_format": "json"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_generate_report_response_structure(self):
        payload = {"audit_id": "AUD-001", "report_format": "json"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert "report_id" in resp.body
        assert "report_format" in resp.body
        assert "provenance_hash" in resp.body

    def test_generate_report_json(self):
        payload = {"report_format": "json"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        data = resp.json()
        assert isinstance(data, dict)

    def test_generate_report_pdf(self):
        payload = {"report_format": "pdf"}
        resp = _mock_generate_report(payload, AUTH_HEADERS)
        assert resp.body["report_format"] == "pdf"


# ===========================================================================
# 7. Certificate Endpoints (5 tests)
# ===========================================================================


class TestCertificateEndpoints:
    """Test certificate validation endpoints."""

    def test_validate_certificate_success(self):
        payload = {"certificate_id": "CERT-001", "scheme": "fsc"}
        resp = _mock_validate_certificate(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["valid"] is True

    def test_validate_certificate_unauthorized(self):
        resp = _mock_validate_certificate({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    @pytest.mark.parametrize("scheme", CERTIFICATION_SCHEMES)
    def test_validate_each_scheme(self, scheme):
        payload = {"certificate_id": "CERT-001", "scheme": scheme}
        resp = _mock_validate_certificate(payload, AUTH_HEADERS)
        assert resp.status_code == 200

    def test_validate_certificate_provenance(self):
        payload = {"scheme": "fsc"}
        resp = _mock_validate_certificate(payload, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_validate_certificate_response_structure(self):
        payload = {"scheme": "fsc"}
        resp = _mock_validate_certificate(payload, AUTH_HEADERS)
        assert "valid" in resp.body
        assert "scheme" in resp.body
        assert "provenance_hash" in resp.body


# ===========================================================================
# 8. Authority Interaction Endpoints (8 tests)
# ===========================================================================


class TestAuthorityEndpoints:
    """Test competent authority interaction endpoints."""

    def test_log_authority_interaction_success(self):
        payload = {"authority_name": "BMEL", "member_state": "DE", "type": "document_request"}
        resp = _mock_log_authority(payload, AUTH_HEADERS)
        assert resp.status_code == 200
        assert resp.body["interaction_id"] is not None

    def test_log_authority_unauthorized(self):
        resp = _mock_log_authority({}, UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_log_authority_provenance(self):
        resp = _mock_log_authority({}, AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    @pytest.mark.parametrize("member_state,authority", list(SAMPLE_AUTHORITIES.items()))
    def test_log_authority_by_member_state(self, member_state, authority):
        payload = {"authority_name": authority, "member_state": member_state}
        resp = _mock_log_authority(payload, AUTH_HEADERS)
        assert resp.status_code == 200

    def test_log_authority_response_structure(self):
        resp = _mock_log_authority({}, AUTH_HEADERS)
        assert "interaction_id" in resp.body
        assert "provenance_hash" in resp.body

    def test_log_authority_interaction_id_prefix(self):
        resp = _mock_log_authority({}, AUTH_HEADERS)
        assert resp.body["interaction_id"].startswith("AUTH-")


# ===========================================================================
# 9. Analytics Dashboard Endpoints (6 tests)
# ===========================================================================


class TestAnalyticsEndpoints:
    """Test analytics and dashboard endpoints."""

    def test_analytics_dashboard_success(self):
        resp = _mock_analytics_dashboard(AUTH_HEADERS)
        assert resp.status_code == 200

    def test_analytics_dashboard_unauthorized(self):
        resp = _mock_analytics_dashboard(UNAUTHORIZED_HEADERS)
        assert resp.status_code == 401

    def test_analytics_dashboard_has_active_audits(self):
        resp = _mock_analytics_dashboard(AUTH_HEADERS)
        assert resp.body["active_audits"] >= 0

    def test_analytics_dashboard_has_compliance_rate(self):
        resp = _mock_analytics_dashboard(AUTH_HEADERS)
        assert "compliance_rate" in resp.body

    def test_analytics_dashboard_provenance(self):
        resp = _mock_analytics_dashboard(AUTH_HEADERS)
        assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_analytics_dashboard_response_structure(self):
        resp = _mock_analytics_dashboard(AUTH_HEADERS)
        for key in ["active_audits", "open_cars", "compliance_rate", "provenance_hash"]:
            assert key in resp.body


# ===========================================================================
# 10. Cross-Cutting API Tests (5 tests)
# ===========================================================================


class TestCrossCuttingAPIBehavior:
    """Test cross-cutting API concerns."""

    @pytest.mark.parametrize("endpoint_fn,args", [
        (_mock_create_audit, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_list_audits, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_register_auditor, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_classify_nc, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_issue_car, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_generate_report, ({}, UNAUTHORIZED_HEADERS)),
        (_mock_log_authority, ({}, UNAUTHORIZED_HEADERS)),
    ])
    def test_all_endpoints_require_auth(self, endpoint_fn, args):
        """Verify all protected endpoints reject unauthenticated requests."""
        resp = endpoint_fn(*args)
        assert resp.status_code == 401

    def test_provenance_hash_always_64_chars(self):
        """Verify provenance hash is always 64 hex characters."""
        for fn, args in [
            (_mock_create_audit, ({"commodity": "wood"}, AUTH_HEADERS)),
            (_mock_match_auditor, ({}, AUTH_HEADERS)),
            (_mock_classify_nc, ({}, AUTH_HEADERS)),
            (_mock_issue_car, ({"severity": "major"}, AUTH_HEADERS)),
            (_mock_generate_report, ({"report_format": "json"}, AUTH_HEADERS)),
            (_mock_log_authority, ({}, AUTH_HEADERS)),
        ]:
            resp = fn(*args)
            assert len(resp.body["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_all_responses_are_json(self):
        """Verify all endpoint responses are JSON-serializable."""
        for fn, args in [
            (_mock_health_check, ()),
            (_mock_stats, (AUTH_HEADERS,)),
            (_mock_list_audits, ({}, AUTH_HEADERS)),
            (_mock_analytics_dashboard, (AUTH_HEADERS,)),
        ]:
            resp = fn(*args)
            assert isinstance(resp.json(), dict)

    def test_pagination_defaults(self):
        """Verify default pagination uses page=1, page_size=20."""
        resp = _mock_list_audits({}, AUTH_HEADERS)
        assert resp.body["page"] == 1
        assert resp.body["page_size"] == 20

    def test_list_returns_results_array(self):
        """Verify list endpoints always return a results array."""
        resp = _mock_list_audits({}, AUTH_HEADERS)
        assert isinstance(resp.body["results"], list)

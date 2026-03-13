# -*- coding: utf-8 -*-
"""
Unit tests for Continuous Monitoring Agent API - AGENT-EUDR-033

Tests all REST API endpoints using FastAPI TestClient:
- Supply chain monitoring endpoints
- Deforestation alert endpoints
- Compliance audit endpoints
- Change detection endpoints
- Risk score monitoring endpoints
- Data freshness validation endpoints
- Regulatory tracking endpoints
- Health check and dashboard endpoints
- Error handling and validation

80+ tests covering all API endpoint paths.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from greenlang.agents.eudr.continuous_monitoring.api import router, get_router
from greenlang.agents.eudr.continuous_monitoring.setup import (
    get_service,
    reset_service,
)


PREFIX = "/api/v1/eudr/continuous-monitoring"


@pytest.fixture(autouse=True)
def _reset_and_init():
    """Reset service and re-initialize engines for each test.

    This ensures a clean service with all engines loaded (without needing
    database or Redis connections) for every test.
    """
    reset_service()
    svc = get_service()
    svc._init_engines()
    svc._initialized = True
    yield
    reset_service()


@pytest.fixture
def app():
    """Create a FastAPI test application with the CMA router."""
    application = FastAPI()
    application.include_router(router)
    return application


@pytest.fixture
def client(app):
    """Create a FastAPI test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Router Setup
# ---------------------------------------------------------------------------


class TestRouterSetup:
    def test_get_router_returns_router(self):
        r = get_router()
        assert r is not None
        assert r.prefix == "/api/v1/eudr/continuous-monitoring"

    def test_router_has_tags(self):
        r = get_router()
        assert "EUDR Continuous Monitoring" in r.tags

    def test_router_includes_responses(self):
        r = get_router()
        assert 401 in r.responses
        assert 403 in r.responses
        assert 500 in r.responses


# ---------------------------------------------------------------------------
# Health Check Endpoint
# ---------------------------------------------------------------------------


class TestHealthCheckEndpoint:
    def test_health_check_returns_200(self, client):
        response = client.get(f"{PREFIX}/health")
        assert response.status_code == 200

    def test_health_check_returns_agent_id(self, client):
        response = client.get(f"{PREFIX}/health")
        data = response.json()
        assert "agent_id" in data
        assert data["agent_id"] == "GL-EUDR-CM-033"

    def test_health_check_returns_status(self, client):
        response = client.get(f"{PREFIX}/health")
        data = response.json()
        assert "status" in data

    def test_health_check_includes_engines(self, client):
        response = client.get(f"{PREFIX}/health")
        data = response.json()
        assert "engines" in data


# ---------------------------------------------------------------------------
# Supply Chain Monitoring Endpoints
# ---------------------------------------------------------------------------


class TestSupplyChainEndpoints:
    def test_scan_supply_chain_returns_200(self, client):
        response = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={
                "operator_id": "OP-001",
                "suppliers": [
                    {"supplier_id": "SUP-001", "name": "Test Supplier",
                     "country": "ID", "commodity": "palm_oil",
                     "lat": -2.5, "lon": 112.9,
                     "certification_status": "active",
                     "certification_expiry": "2027-01-01"}
                ],
            },
        )
        assert response.status_code == 200

    def test_scan_supply_chain_returns_record(self, client):
        response = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={"operator_id": "OP-001", "suppliers": []},
        )
        data = response.json()
        assert "scan_id" in data
        assert "operator_id" in data

    def test_get_supply_chain_record_returns_200(self, client):
        create_resp = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={"operator_id": "OP-001", "suppliers": []},
        )
        scan_id = create_resp.json()["scan_id"]
        response = client.get(f"{PREFIX}/scans/{scan_id}")
        assert response.status_code == 200

    def test_get_supply_chain_record_not_found(self, client):
        response = client.get(f"{PREFIX}/scans/nonexistent")
        assert response.status_code == 404

    def test_list_supply_chain_records(self, client):
        response = client.get(f"{PREFIX}/scans")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_supply_chain_filter_operator(self, client):
        response = client.get(f"{PREFIX}/scans?operator_id=OP-001")
        assert response.status_code == 200

    def test_scan_missing_operator_id(self, client):
        response = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={"suppliers": []},
        )
        assert response.status_code == 422

    def test_list_supply_chain_alerts(self, client):
        response = client.get(f"{PREFIX}/alerts")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_supply_chain_alerts_filter_severity(self, client):
        response = client.get(f"{PREFIX}/alerts?severity=critical")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Deforestation Alert Endpoints
# ---------------------------------------------------------------------------


class TestDeforestationAlertEndpoints:
    def test_check_deforestation_returns_200(self, client):
        response = client.post(
            f"{PREFIX}/check-deforestation",
            json={
                "operator_id": "OP-001",
                "alerts": [
                    {"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 15.3}
                ],
            },
        )
        assert response.status_code == 200

    def test_check_deforestation_empty_list(self, client):
        response = client.post(
            f"{PREFIX}/check-deforestation",
            json={"operator_id": "OP-001", "alerts": []},
        )
        assert response.status_code == 200

    def test_check_deforestation_with_entities(self, client):
        response = client.post(
            f"{PREFIX}/check-deforestation",
            json={
                "operator_id": "OP-001",
                "alerts": [
                    {"alert_id": "A-001", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}
                ],
                "supply_chain_entities": [
                    {"entity_id": "P-001", "entity_type": "plot",
                     "lat": -2.5, "lon": 112.9}
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "monitor_id" in data

    def test_get_deforestation_record_not_found(self, client):
        response = client.get(f"{PREFIX}/deforestation-records/nonexistent")
        assert response.status_code == 404

    def test_list_deforestation_records(self, client):
        response = client.get(f"{PREFIX}/deforestation-records")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_deforestation_records_filter_operator(self, client):
        response = client.get(
            f"{PREFIX}/deforestation-records?operator_id=OP-001",
        )
        assert response.status_code == 200

    def test_get_deforestation_record_by_id(self, client):
        # Create a record first
        create_resp = client.post(
            f"{PREFIX}/check-deforestation",
            json={
                "operator_id": "OP-001",
                "alerts": [
                    {"alert_id": "A-002", "lat": 3.1, "lon": 101.7, "area_ha": 5.0}
                ],
            },
        )
        monitor_id = create_resp.json()["monitor_id"]
        response = client.get(f"{PREFIX}/deforestation-records/{monitor_id}")
        assert response.status_code == 200

    def test_list_investigations(self, client):
        response = client.get(f"{PREFIX}/investigations")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_investigation_not_found(self, client):
        response = client.get(f"{PREFIX}/investigations/nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Compliance Audit Endpoints
# ---------------------------------------------------------------------------


class TestComplianceAuditEndpoints:
    def _operator_data(self):
        now = datetime.now(timezone.utc)
        return {
            "dds_date": (now - timedelta(days=30)).isoformat(),
            "supply_chain_last_updated": (now - timedelta(hours=12)).isoformat(),
            "risk_assessments": [
                {"assessment_id": "RA-001",
                 "assessment_date": (now - timedelta(days=60)).isoformat(),
                 "scope": "full"}
            ],
            "due_diligence_statements": [
                {"statement_id": "DDS-001",
                 "statement_date": (now - timedelta(days=30)).isoformat(),
                 "commodity": "palm_oil",
                 "origin_country": "ID",
                 "supplier_info": True}
            ],
            "retention_years": 6,
            "competent_authority_registered": True,
        }

    def test_run_audit_returns_200(self, client):
        response = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={
                "operator_id": "OP-001",
                "operator_data": self._operator_data(),
            },
        )
        assert response.status_code == 200

    def test_run_audit_returns_record(self, client):
        response = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={
                "operator_id": "OP-001",
                "operator_data": self._operator_data(),
            },
        )
        data = response.json()
        assert "audit_id" in data
        assert "overall_score" in data
        assert "compliance_status" in data

    def test_get_audit_not_found(self, client):
        response = client.get(f"{PREFIX}/audits/nonexistent")
        assert response.status_code == 404

    def test_list_audits(self, client):
        response = client.get(f"{PREFIX}/audits")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_audits_filter_operator(self, client):
        response = client.get(f"{PREFIX}/audits?operator_id=OP-001")
        assert response.status_code == 200

    def test_get_audit_by_id(self, client):
        # Create an audit first
        create_resp = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={
                "operator_id": "OP-001",
                "operator_data": self._operator_data(),
            },
        )
        audit_id = create_resp.json()["audit_id"]
        response = client.get(f"{PREFIX}/audits/{audit_id}")
        assert response.status_code == 200

    def test_run_audit_missing_operator_id(self, client):
        response = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={"operator_data": {}},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Change Detection Endpoints
# ---------------------------------------------------------------------------


class TestChangeDetectionEndpoints:
    def test_detect_changes_returns_200(self, client):
        response = client.post(
            f"{PREFIX}/detect-changes",
            json={
                "operator_id": "OP-001",
                "entity_snapshots": [
                    {"entity_id": "SUP-001", "entity_type": "supplier",
                     "old_state": {"status": "active"},
                     "new_state": {"status": "suspended"}}
                ],
            },
        )
        assert response.status_code == 200

    def test_detect_changes_empty(self, client):
        response = client.post(
            f"{PREFIX}/detect-changes",
            json={"operator_id": "OP-001", "entity_snapshots": []},
        )
        assert response.status_code == 200

    def test_detect_changes_returns_list(self, client):
        response = client.post(
            f"{PREFIX}/detect-changes",
            json={
                "operator_id": "OP-001",
                "entity_snapshots": [
                    {"entity_id": "SUP-001", "entity_type": "supplier",
                     "old_state": {"certification": "valid"},
                     "new_state": {"certification": "expired"}}
                ],
            },
        )
        data = response.json()
        assert isinstance(data, list)

    def test_get_change_not_found(self, client):
        response = client.get(f"{PREFIX}/changes/nonexistent")
        assert response.status_code == 404

    def test_list_changes(self, client):
        response = client.get(f"{PREFIX}/changes")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_changes_filter_operator(self, client):
        response = client.get(f"{PREFIX}/changes?operator_id=OP-001")
        assert response.status_code == 200

    def test_list_changes_filter_change_type(self, client):
        response = client.get(f"{PREFIX}/changes?change_type=certification")
        assert response.status_code == 200

    def test_detect_changes_missing_operator_id(self, client):
        response = client.post(
            f"{PREFIX}/detect-changes",
            json={"entity_snapshots": []},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Risk Score Monitoring Endpoints
# ---------------------------------------------------------------------------


class TestRiskScoreEndpoints:
    def test_monitor_risk_scores_returns_200(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-001",
                "score_history": [
                    {"timestamp": (now - timedelta(days=30)).isoformat(), "score": 30},
                    {"timestamp": now.isoformat(), "score": 45},
                ],
                "entity_type": "supplier",
            },
        )
        assert response.status_code == 200

    def test_monitor_risk_scores_returns_record(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-001",
                "score_history": [
                    {"timestamp": now.isoformat(), "score": 50},
                ],
            },
        )
        data = response.json()
        assert "monitor_id" in data
        assert "current_score" in data
        assert "risk_level" in data

    def test_monitor_risk_scores_empty_history(self, client):
        response = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-001",
                "score_history": [],
            },
        )
        assert response.status_code == 200

    def test_get_risk_monitor_not_found(self, client):
        response = client.get(f"{PREFIX}/risk-monitors/nonexistent")
        assert response.status_code == 404

    def test_list_risk_monitors(self, client):
        response = client.get(f"{PREFIX}/risk-monitors")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_risk_monitors_filter_operator(self, client):
        response = client.get(f"{PREFIX}/risk-monitors?operator_id=OP-001")
        assert response.status_code == 200

    def test_list_risk_monitors_filter_entity(self, client):
        response = client.get(f"{PREFIX}/risk-monitors?entity_id=SUP-001")
        assert response.status_code == 200

    def test_monitor_risk_scores_missing_entity_id(self, client):
        response = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "score_history": [],
            },
        )
        assert response.status_code == 422

    def test_get_risk_monitor_by_id(self, client):
        now = datetime.now(timezone.utc)
        create_resp = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-001",
                "score_history": [
                    {"timestamp": now.isoformat(), "score": 50},
                ],
            },
        )
        monitor_id = create_resp.json()["monitor_id"]
        response = client.get(f"{PREFIX}/risk-monitors/{monitor_id}")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Data Freshness Validation Endpoints
# ---------------------------------------------------------------------------


class TestDataFreshnessEndpoints:
    def test_validate_freshness_returns_200(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/validate-freshness",
            json={
                "operator_id": "OP-001",
                "entities": [
                    {"entity_id": "E-001", "entity_type": "supplier",
                     "last_updated": (now - timedelta(hours=1)).isoformat()}
                ],
            },
        )
        assert response.status_code == 200

    def test_validate_freshness_empty(self, client):
        response = client.post(
            f"{PREFIX}/validate-freshness",
            json={"operator_id": "OP-001", "entities": []},
        )
        assert response.status_code == 200

    def test_validate_freshness_returns_record(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/validate-freshness",
            json={
                "operator_id": "OP-001",
                "entities": [
                    {"entity_id": "E-001", "entity_type": "supplier",
                     "last_updated": (now - timedelta(hours=2)).isoformat()}
                ],
            },
        )
        data = response.json()
        assert "freshness_id" in data
        assert "entities_checked" in data

    def test_get_freshness_record_not_found(self, client):
        response = client.get(f"{PREFIX}/freshness-records/nonexistent")
        assert response.status_code == 404

    def test_list_freshness_records(self, client):
        response = client.get(f"{PREFIX}/freshness-records")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_freshness_records_filter_operator(self, client):
        response = client.get(
            f"{PREFIX}/freshness-records?operator_id=OP-001",
        )
        assert response.status_code == 200

    def test_get_freshness_record_by_id(self, client):
        now = datetime.now(timezone.utc)
        create_resp = client.post(
            f"{PREFIX}/validate-freshness",
            json={
                "operator_id": "OP-001",
                "entities": [
                    {"entity_id": "E-001", "entity_type": "supplier",
                     "last_updated": (now - timedelta(hours=1)).isoformat()}
                ],
            },
        )
        freshness_id = create_resp.json()["freshness_id"]
        response = client.get(f"{PREFIX}/freshness-records/{freshness_id}")
        assert response.status_code == 200

    def test_freshness_report(self, client):
        response = client.get(
            f"{PREFIX}/freshness-report?operator_id=OP-001",
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_checks" in data

    def test_freshness_report_after_validation(self, client):
        now = datetime.now(timezone.utc)
        client.post(
            f"{PREFIX}/validate-freshness",
            json={
                "operator_id": "OP-001",
                "entities": [
                    {"entity_id": "E-001", "entity_type": "supplier",
                     "last_updated": (now - timedelta(hours=1)).isoformat()}
                ],
            },
        )
        response = client.get(
            f"{PREFIX}/freshness-report?operator_id=OP-001",
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] >= 1


# ---------------------------------------------------------------------------
# Regulatory Tracking Endpoints
# ---------------------------------------------------------------------------


class TestRegulatoryTrackingEndpoints:
    def test_check_regulatory_returns_200(self, client):
        response = client.post(
            f"{PREFIX}/check-regulatory",
            json={
                "operator_id": "OP-001",
                "updates": [
                    {"update_id": "REG-001", "source": "eur-lex",
                     "title": "EUDR Guidance Update",
                     "summary": "Updated guidance on reporting",
                     "impact_level": "moderate"}
                ],
            },
        )
        assert response.status_code == 200

    def test_check_regulatory_empty(self, client):
        response = client.post(
            f"{PREFIX}/check-regulatory",
            json={"operator_id": "OP-001"},
        )
        assert response.status_code == 200

    def test_check_regulatory_returns_record(self, client):
        response = client.post(
            f"{PREFIX}/check-regulatory",
            json={
                "operator_id": "OP-001",
                "updates": [
                    {"update_id": "REG-002", "source": "eur-lex",
                     "title": "Amendment", "summary": "Article 8 amendment",
                     "impact_level": "high",
                     "affected_articles": ["Article 8"]}
                ],
            },
        )
        data = response.json()
        assert "tracking_id" in data
        assert "updates_found" in data

    def test_get_regulatory_record_not_found(self, client):
        response = client.get(f"{PREFIX}/regulatory-records/nonexistent")
        assert response.status_code == 404

    def test_list_regulatory_records(self, client):
        response = client.get(f"{PREFIX}/regulatory-records")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_list_regulatory_records_filter_operator(self, client):
        response = client.get(
            f"{PREFIX}/regulatory-records?operator_id=OP-001",
        )
        assert response.status_code == 200

    def test_get_regulatory_record_by_id(self, client):
        create_resp = client.post(
            f"{PREFIX}/check-regulatory",
            json={
                "operator_id": "OP-001",
                "updates": [
                    {"update_id": "REG-003", "source": "eur-lex",
                     "title": "Test Update", "summary": "Clarification",
                     "impact_level": "low"}
                ],
            },
        )
        tracking_id = create_resp.json()["tracking_id"]
        response = client.get(f"{PREFIX}/regulatory-records/{tracking_id}")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Dashboard Endpoint
# ---------------------------------------------------------------------------


class TestDashboardEndpoint:
    def test_dashboard_returns_200(self, client):
        response = client.get(f"{PREFIX}/dashboard?operator_id=OP-001")
        assert response.status_code == 200

    def test_dashboard_has_agent_id(self, client):
        response = client.get(f"{PREFIX}/dashboard?operator_id=OP-001")
        data = response.json()
        assert "agent_id" in data

    def test_dashboard_has_engines(self, client):
        response = client.get(f"{PREFIX}/dashboard?operator_id=OP-001")
        data = response.json()
        assert "engines" in data


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_json_body(self, client):
        response = client.post(
            f"{PREFIX}/scan-supply-chain",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self, client):
        response = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={},
        )
        assert response.status_code == 422

    def test_invalid_enum_value(self, client):
        # Should still handle gracefully
        response = client.post(
            f"{PREFIX}/check-deforestation",
            json={
                "operator_id": "OP-001",
                "alerts": [
                    {"alert_id": "A-X", "lat": 0, "lon": 0, "area_ha": 1.0}
                ],
            },
        )
        assert response.status_code in (200, 422)

    def test_nonexistent_endpoint(self, client):
        response = client.get(f"{PREFIX}/nonexistent-path")
        assert response.status_code in (404, 405)

    def test_method_not_allowed(self, client):
        response = client.delete(f"{PREFIX}/health")
        assert response.status_code == 405


# ---------------------------------------------------------------------------
# Response Schema Validation
# ---------------------------------------------------------------------------


class TestResponseSchemas:
    """Test that API responses conform to expected schemas."""

    def test_health_check_schema(self, client):
        response = client.get(f"{PREFIX}/health")
        data = response.json()
        assert isinstance(data.get("agent_id"), str)
        assert isinstance(data.get("status"), str)
        assert isinstance(data.get("engines"), dict)

    def test_supply_chain_scan_response_schema(self, client):
        response = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={"operator_id": "OP-001", "suppliers": []},
        )
        data = response.json()
        assert "scan_id" in data
        assert "operator_id" in data
        assert data["operator_id"] == "OP-001"

    def test_compliance_audit_response_schema(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={
                "operator_id": "OP-001",
                "operator_data": {
                    "dds_date": (now - timedelta(days=30)).isoformat(),
                    "risk_assessments": [],
                    "due_diligence_statements": [],
                    "retention_years": 5,
                    "competent_authority_registered": True,
                },
            },
        )
        data = response.json()
        assert "audit_id" in data
        assert "overall_score" in data
        assert "compliance_status" in data

    def test_risk_score_response_schema(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-001",
                "score_history": [
                    {"timestamp": now.isoformat(), "score": 50},
                ],
            },
        )
        data = response.json()
        assert "monitor_id" in data
        assert "current_score" in data
        assert "risk_level" in data
        assert "trend_direction" in data

    def test_freshness_validate_response_schema(self, client):
        now = datetime.now(timezone.utc)
        response = client.post(
            f"{PREFIX}/validate-freshness",
            json={
                "operator_id": "OP-001",
                "entities": [
                    {"entity_id": "E-001", "entity_type": "supplier",
                     "last_updated": (now - timedelta(hours=1)).isoformat()}
                ],
            },
        )
        data = response.json()
        assert "freshness_id" in data
        assert "entities_checked" in data
        assert "freshness_percentage" in data

    def test_regulatory_check_response_schema(self, client):
        response = client.post(
            f"{PREFIX}/check-regulatory",
            json={
                "operator_id": "OP-001",
                "updates": [
                    {"update_id": "REG-S1", "source": "eur-lex",
                     "title": "Schema Test", "summary": "Test update",
                     "impact_level": "moderate"}
                ],
            },
        )
        data = response.json()
        assert "tracking_id" in data
        assert "updates_found" in data
        assert "high_impact_count" in data

    def test_deforestation_check_response_schema(self, client):
        response = client.post(
            f"{PREFIX}/check-deforestation",
            json={
                "operator_id": "OP-001",
                "alerts": [
                    {"alert_id": "A-S1", "lat": -2.5, "lon": 112.9, "area_ha": 10.0}
                ],
            },
        )
        data = response.json()
        assert "monitor_id" in data
        assert "alerts_checked" in data

    def test_change_detection_response_schema(self, client):
        response = client.post(
            f"{PREFIX}/detect-changes",
            json={
                "operator_id": "OP-001",
                "entity_snapshots": [
                    {"entity_id": "SUP-001", "entity_type": "supplier",
                     "old_state": {"status": "active"},
                     "new_state": {"status": "inactive"}}
                ],
            },
        )
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "detection_id" in data[0]
            assert "change_type" in data[0]


# ---------------------------------------------------------------------------
# Cross-Endpoint Workflows
# ---------------------------------------------------------------------------


class TestCrossEndpointWorkflows:
    """Test multi-step workflows across endpoints."""

    def test_scan_then_retrieve(self, client):
        """Create a scan and retrieve it by ID."""
        create_resp = client.post(
            f"{PREFIX}/scan-supply-chain",
            json={"operator_id": "OP-001", "suppliers": []},
        )
        scan_id = create_resp.json()["scan_id"]
        get_resp = client.get(f"{PREFIX}/scans/{scan_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["scan_id"] == scan_id

    def test_audit_then_list(self, client):
        """Create an audit and find it in the list."""
        now = datetime.now(timezone.utc)
        create_resp = client.post(
            f"{PREFIX}/run-compliance-audit",
            json={
                "operator_id": "OP-FLOW",
                "operator_data": {
                    "dds_date": (now - timedelta(days=30)).isoformat(),
                    "risk_assessments": [],
                    "due_diligence_statements": [],
                    "retention_years": 5,
                    "competent_authority_registered": True,
                },
            },
        )
        audit_id = create_resp.json()["audit_id"]
        list_resp = client.get(f"{PREFIX}/audits?operator_id=OP-FLOW")
        audit_ids = [a["audit_id"] for a in list_resp.json()]
        assert audit_id in audit_ids

    def test_risk_monitor_then_retrieve(self, client):
        """Create a risk monitor and retrieve it."""
        now = datetime.now(timezone.utc)
        create_resp = client.post(
            f"{PREFIX}/monitor-risk-scores",
            json={
                "operator_id": "OP-001",
                "entity_id": "SUP-FLOW",
                "score_history": [
                    {"timestamp": now.isoformat(), "score": 65},
                ],
            },
        )
        monitor_id = create_resp.json()["monitor_id"]
        get_resp = client.get(f"{PREFIX}/risk-monitors/{monitor_id}")
        assert get_resp.status_code == 200

    def test_multiple_scans_listed(self, client):
        """Create multiple scans and verify list returns them all."""
        for i in range(3):
            client.post(
                f"{PREFIX}/scan-supply-chain",
                json={"operator_id": f"OP-MULTI-{i}", "suppliers": []},
            )
        list_resp = client.get(f"{PREFIX}/scans")
        assert len(list_resp.json()) >= 3

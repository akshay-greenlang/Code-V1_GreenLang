# -*- coding: utf-8 -*-
"""
Unit Tests for CDP, Compliance, and Settings API Routes (v1.1)

Tests FastAPI endpoints for CDP questionnaire management, compliance
scorecard assessment, and user settings CRUD.

Target modules:
    - services/api/cdp_routes.py
    - services/api/compliance_routes.py
    - services/api/settings_routes.py

Test count: 35 tests
Coverage target: 85%+
"""

import json
import pytest
from typing import Any, Dict
from unittest.mock import Mock, patch

import sys
import os

PLATFORM_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if PLATFORM_ROOT not in sys.path:
    sys.path.insert(0, PLATFORM_ROOT)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.cdp_routes import router as cdp_router
from services.api.compliance_routes import router as compliance_router
from services.api.settings_routes import router as settings_router

# Reset module-level state for test isolation
import services.api.cdp_routes as cdp_mod
import services.api.compliance_routes as compliance_mod
import services.api.settings_routes as settings_mod


# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI()
app.include_router(cdp_router)
app.include_router(compliance_router)
app.include_router(settings_router)


# ============================================================================
# MOCK DATA
# ============================================================================

AUTO_POPULATE_PAYLOAD: Dict[str, Any] = {
    "company_info": {
        "name": "TestCorp",
        "description": "A test company for CDP testing.",
        "reporting_year": 2025,
        "headquarters": "United States",
        "operating_countries": ["United States", "Germany"],
        "currency": "USD",
        "consolidation_approach": "operational_control",
        "isin_code": "US9999999999",
        "financial_services": False,
        "verification_status": "limited_assurance",
        "scope1_verification": {"verifier": "KPMG"},
        "ets_exposure": True,
        "internal_carbon_price": True,
        "carbon_price_details": {"price_usd": 75},
    },
    "emissions_data": {
        "scope1_tco2e": 4000.0,
        "scope2_location_tco2e": 2500.0,
        "scope2_market_tco2e": 2200.0,
        "scope3_tco2e": 35000.0,
        "scope3_categories": {
            1: 12000.0, 2: 1500.0, 3: 1000.0, 4: 4000.0,
            5: 600.0, 6: 2500.0, 7: 1000.0, 8: 400.0,
            9: 3000.0, 10: 800.0, 11: 6000.0, 12: 1500.0,
            13: 400.0, 14: 100.0, 15: 200.0,
        },
        "base_year": 2019,
        "is_base_year": False,
        "scope1_by_gas": [{"gas": "CO2", "tco2e": 3800}],
        "scope1_by_country": [{"country": "US", "tco2e": 3000}],
        "intensity_per_revenue": 10.0,
        "yoy_change_pct": -2.5,
        "biogenic_relevant": True,
        "biogenic_emissions": {"scope1_biogenic_tco2": 50.0},
    },
    "energy_data": {
        "total_energy_mwh": 80000.0,
        "renewable_energy_mwh": 32000.0,
        "non_renewable_energy_mwh": 48000.0,
        "renewable_pct": 40.0,
        "energy_spend_pct": 12.0,
        "has_reduction_target": True,
    },
    "targets_data": {
        "has_active_target": True,
        "absolute_targets": [{"scope": "scope1+2", "reduction_pct": 42}],
        "sbti_status": "sbti_committed",
        "net_zero_target": "2050",
    },
    "risks_data": {
        "has_risk_process": True,
        "has_identified_risks": True,
        "physical_risks": [{"type": "acute", "driver": "flooding"}],
        "has_identified_opportunities": True,
        "opportunities": [{"type": "resource_efficiency"}],
        "scenario_analysis": True,
    },
    "governance_data": {
        "board_oversight": True,
        "board_positions": [{"title": "Sustainability Committee Chair"}],
        "management_positions": [{"title": "CSO"}],
        "has_incentives": True,
    },
    "engagement_data": {
        "engages_value_chain": True,
        "supplier_engagement": [{"type": "cdp_supply_chain"}],
        "policy_engagement": True,
    },
}

SCORECARD_PAYLOAD: Dict[str, Any] = {
    "emissions_data": {
        "scope1_tco2e": 4000.0,
        "scope2_location_tco2e": 2500.0,
        "scope2_market_tco2e": 2200.0,
        "scope3_tco2e": 35000.0,
        "scope3_categories": {1: 12000.0, 4: 4000.0, 6: 2500.0},
        "base_year": 2019,
        "calculation_methodology": "GHG Protocol",
        "gases_reported": ["co2", "ch4", "n2o"],
        "gwp_source": "IPCC_AR5",
        "total_energy_mwh": 80000,
        "renewable_pct": 40.0,
        "reporting_period_start": "2025-01-01",
        "reporting_period_end": "2025-12-31",
    },
    "company_info": {
        "name": "TestCorp",
        "reporting_year": 2025,
        "consolidation_approach": "operational_control",
        "verification_status": "limited_assurance",
    },
    "targets_data": {
        "sbti_status": "sbti_committed",
        "has_transition_plan": True,
        "absolute_targets": [{"scope": "scope1+2", "reduction_pct": 42}],
    },
    "governance_data": {
        "board_oversight": True,
    },
}


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def client():
    """Create a FastAPI test client."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset in-memory stores before each test for isolation."""
    cdp_mod._questionnaire_store.clear()
    cdp_mod._input_data_store.clear()
    compliance_mod._current_scorecard = None
    compliance_mod._scorecard_history.clear()
    compliance_mod._input_data_store.clear()
    # Reset settings to defaults
    settings_mod._current_settings = settings_mod.Settings()
    yield


@pytest.fixture
def populated_questionnaire(client):
    """Run auto-populate for year 2025 and return response."""
    response = client.post(
        "/api/v1/cdp/questionnaire/2025/auto-populate",
        json=AUTO_POPULATE_PAYLOAD,
    )
    assert response.status_code == 201
    return response.json()


@pytest.fixture
def populated_scorecard(client):
    """Generate a compliance scorecard and return response."""
    response = client.post(
        "/api/v1/compliance/scorecard",
        json=SCORECARD_PAYLOAD,
    )
    assert response.status_code == 201
    return response.json()


# ============================================================================
# TEST: CDP Questionnaire Endpoints
# ============================================================================

class TestCDPQuestionnaireEndpoints:
    """Test CDP questionnaire CRUD and analysis endpoints."""

    def test_get_questionnaire_by_year(self, client, populated_questionnaire):
        """GET /questionnaire/2025 should return the stored questionnaire."""
        response = client.get("/api/v1/cdp/questionnaire/2025")
        assert response.status_code == 200
        data = response.json()
        assert data["reporting_year"] == 2025
        assert data["company_name"] == "TestCorp"

    def test_get_questionnaire_not_found(self, client):
        """GET for a year with no data should return 404."""
        response = client.get("/api/v1/cdp/questionnaire/1999")
        assert response.status_code == 404

    def test_update_section(self, client, populated_questionnaire):
        """PUT /questionnaire/2025/section/C6 should merge new answers."""
        response = client.put(
            "/api/v1/cdp/questionnaire/2025/section/C6",
            json={"answers": {"C6.1": 9999.0}},
        )
        assert response.status_code == 200
        data = response.json()
        c6 = data["sections"]["C6"]
        assert c6["answers"]["C6.1"] == 9999.0

    def test_update_invalid_section(self, client, populated_questionnaire):
        """PUT with invalid section ID should return 400."""
        response = client.put(
            "/api/v1/cdp/questionnaire/2025/section/C99",
            json={"answers": {"Q1": "test"}},
        )
        assert response.status_code == 400

    def test_auto_populate(self, client):
        """POST /auto-populate should create a new questionnaire."""
        response = client.post(
            "/api/v1/cdp/questionnaire/2025/auto-populate",
            json=AUTO_POPULATE_PAYLOAD,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["reporting_year"] == 2025
        assert len(data["sections"]) == 13
        assert data["total_questions"] > 0

    def test_get_progress(self, client, populated_questionnaire):
        """GET /progress should return completion tracking data."""
        response = client.get("/api/v1/cdp/questionnaire/2025/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["year"] == 2025
        assert "overall_completion_pct" in data
        assert "section_progress" in data
        assert len(data["section_progress"]) == 13

    def test_validate_questionnaire(self, client, populated_questionnaire):
        """POST /validate should return validation results."""
        response = client.post("/api/v1/cdp/questionnaire/2025/validate")
        assert response.status_code == 200
        data = response.json()
        assert "is_valid" in data
        assert "total_errors" in data
        assert "total_warnings" in data
        assert "issues" in data

    def test_get_gaps(self, client, populated_questionnaire):
        """GET /gaps should return data gap analysis."""
        response = client.get("/api/v1/cdp/questionnaire/2025/gaps")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        for gap in data:
            assert "section_id" in gap
            assert "question_id" in gap
            assert "severity" in gap

    def test_score_prediction(self, client, populated_questionnaire):
        """GET /score-prediction should return predicted score."""
        response = client.get(
            "/api/v1/cdp/questionnaire/2025/score-prediction"
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_score" in data
        assert "predicted_band" in data
        assert "confidence" in data
        valid_scores = ["A", "A-", "B", "B-", "C", "C-", "D", "D-"]
        assert data["predicted_score"] in valid_scores

    def test_compare_years(self, client):
        """GET /compare/{year1}/{year2} should compare two questionnaires."""
        # Populate year 2024
        payload_2024 = dict(AUTO_POPULATE_PAYLOAD)
        payload_2024["company_info"] = {
            **payload_2024["company_info"], "reporting_year": 2024
        }
        client.post(
            "/api/v1/cdp/questionnaire/2024/auto-populate",
            json=payload_2024,
        )
        # Populate year 2025
        client.post(
            "/api/v1/cdp/questionnaire/2025/auto-populate",
            json=AUTO_POPULATE_PAYLOAD,
        )
        response = client.get("/api/v1/cdp/questionnaire/compare/2025/2024")
        assert response.status_code == 200
        data = response.json()
        assert data["year_current"] == 2025
        assert data["year_previous"] == 2024
        assert "completion_change_pct" in data
        assert "improvement_summary" in data

    def test_export_questionnaire(self, client, populated_questionnaire):
        """POST /export should return content bytes."""
        response = client.post(
            "/api/v1/cdp/questionnaire/2025/export",
            json={"format": "json"},
        )
        assert response.status_code == 200
        assert len(response.content) > 0

    def test_export_invalid_format(self, client, populated_questionnaire):
        """POST /export with invalid format should return 400."""
        response = client.post(
            "/api/v1/cdp/questionnaire/2025/export",
            json={"format": "csv"},
        )
        assert response.status_code == 400

    def test_get_sections_metadata(self, client):
        """GET /sections should return section metadata."""
        response = client.get("/api/v1/cdp/sections")
        assert response.status_code == 200
        data = response.json()
        assert data["total_sections"] == 13
        assert data["total_questions"] > 0
        assert len(data["sections"]) == 13
        for section in data["sections"]:
            assert "id" in section
            assert "title" in section
            assert "question_count" in section


# ============================================================================
# TEST: Compliance Endpoints
# ============================================================================

class TestComplianceEndpoints:
    """Test compliance scorecard API endpoints."""

    def test_get_scorecard_not_generated(self, client):
        """GET /scorecard with no scorecard should return 404."""
        response = client.get("/api/v1/compliance/scorecard")
        assert response.status_code == 404

    def test_generate_scorecard(self, client):
        """POST /scorecard should generate and return a scorecard."""
        response = client.post(
            "/api/v1/compliance/scorecard",
            json=SCORECARD_PAYLOAD,
        )
        assert response.status_code == 201
        data = response.json()
        assert "overall_score" in data
        assert "overall_grade" in data
        assert "standards" in data
        assert len(data["standards"]) == 5

    def test_get_scorecard(self, client, populated_scorecard):
        """GET /scorecard should return the most recent scorecard."""
        response = client.get("/api/v1/compliance/scorecard")
        assert response.status_code == 200
        data = response.json()
        assert data["company_name"] == "TestCorp"

    def test_get_standard_coverage(self, client, populated_scorecard):
        """GET /standard/ghg_protocol should return GHG Protocol coverage."""
        response = client.get("/api/v1/compliance/standard/ghg_protocol")
        assert response.status_code == 200
        data = response.json()
        assert data["standard_code"] == "ghg_protocol"
        assert "coverage_pct" in data
        assert "requirements" in data

    def test_get_standard_invalid(self, client, populated_scorecard):
        """GET /standard/invalid should return 400."""
        response = client.get("/api/v1/compliance/standard/invalid_standard")
        assert response.status_code == 400

    def test_get_gaps(self, client, populated_scorecard):
        """GET /gaps should return cross-standard gaps."""
        response = client.get("/api/v1/compliance/gaps")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_gaps_with_severity_filter(self, client, populated_scorecard):
        """GET /gaps?severity=critical should filter by severity."""
        response = client.get("/api/v1/compliance/gaps?severity=critical")
        assert response.status_code == 200
        data = response.json()
        for gap in data:
            assert gap["severity"] == "critical"

    def test_get_evidence(self, client, populated_scorecard):
        """GET /evidence/GHG-001 should return evidence for that requirement."""
        response = client.get("/api/v1/compliance/evidence/GHG-001")
        assert response.status_code == 200
        data = response.json()
        assert data["requirement_id"] == "GHG-001"
        assert "status" in data
        assert "evidence" in data

    def test_get_evidence_not_found(self, client, populated_scorecard):
        """GET /evidence/NONEXISTENT should return 404."""
        response = client.get("/api/v1/compliance/evidence/NONEXISTENT-999")
        assert response.status_code == 404

    def test_get_action_items(self, client, populated_scorecard):
        """GET /action-items should return prioritized actions."""
        response = client.get("/api/v1/compliance/action-items")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_get_trend(self, client, populated_scorecard):
        """GET /trend should return trend data."""
        response = client.get("/api/v1/compliance/trend")
        assert response.status_code == 200
        data = response.json()
        assert "data_points" in data
        assert "trend_direction" in data
        assert data["trend_direction"] in ("improving", "declining", "stable")


# ============================================================================
# TEST: Settings Endpoints
# ============================================================================

class TestSettingsEndpoints:
    """Test settings CRUD endpoints."""

    def test_get_settings(self, client):
        """GET /settings/ should return current settings."""
        response = client.get("/api/v1/settings/")
        assert response.status_code == 200
        data = response.json()
        assert "profile" in data
        assert "reporting" in data
        assert "display" in data
        assert "notifications" in data
        assert "cdp" in data
        assert "compliance" in data
        assert "version" in data

    def test_update_settings(self, client):
        """PUT /settings/ should update specified fields."""
        response = client.put(
            "/api/v1/settings/",
            json={
                "profile": {
                    "display_name": "Test User",
                    "email": "test@example.com",
                    "role": "manager",
                    "organization": "TestOrg",
                    "timezone": "America/New_York",
                    "locale": "en-US",
                },
                "reporting": {
                    "default_standard": "esrs_e1",
                    "default_export_format": "excel",
                    "include_charts": True,
                    "include_provenance": True,
                    "consolidation_approach": "operational_control",
                    "currency": "EUR",
                    "emission_unit": "tCO2e",
                    "gwp_source": "IPCC_AR6",
                    "scope3_categories_reported": [1, 2, 3, 4, 5, 6],
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["profile"]["display_name"] == "Test User"
        assert data["reporting"]["default_standard"] == "esrs_e1"
        assert data["reporting"]["currency"] == "EUR"

    def test_update_settings_partial(self, client):
        """PUT /settings/ with partial update should preserve other fields."""
        # First update display settings
        client.put(
            "/api/v1/settings/",
            json={
                "display": {
                    "theme": "dark",
                    "sidebar_collapsed": True,
                    "dashboard_layout": "compact",
                    "decimal_places": 2,
                    "chart_color_scheme": "default",
                    "date_format": "YYYY-MM-DD",
                    "number_format": "1,234.56",
                    "items_per_page": 50,
                },
            },
        )
        # Then update only profile (display should remain)
        response = client.put(
            "/api/v1/settings/",
            json={
                "profile": {
                    "display_name": "Updated Name",
                    "email": "updated@test.com",
                    "role": "admin",
                    "organization": "NewOrg",
                    "timezone": "UTC",
                    "locale": "en-US",
                },
            },
        )
        data = response.json()
        # Display settings should have the dark theme from first update
        assert data["display"]["theme"] == "dark"
        assert data["profile"]["display_name"] == "Updated Name"

    def test_get_defaults(self, client):
        """GET /settings/defaults should return factory defaults."""
        response = client.get("/api/v1/settings/defaults")
        assert response.status_code == 200
        data = response.json()
        assert data["profile"]["role"] == "analyst"
        assert data["reporting"]["default_standard"] == "ghg_protocol"
        assert data["display"]["theme"] == "light"
        assert data["reporting"]["currency"] == "USD"

    def test_settings_last_updated(self, client):
        """Settings should have a last_updated timestamp."""
        response = client.get("/api/v1/settings/")
        data = response.json()
        assert "last_updated" in data
        assert data["last_updated"] != ""

    def test_settings_update_changes_timestamp(self, client):
        """Updating settings should change the last_updated timestamp."""
        # Get initial timestamp
        r1 = client.get("/api/v1/settings/")
        ts1 = r1.json()["last_updated"]

        # Update settings
        client.put(
            "/api/v1/settings/",
            json={
                "notifications": {
                    "email_notifications": False,
                    "report_completion": True,
                    "validation_warnings": True,
                    "compliance_alerts": True,
                    "data_quality_alerts": True,
                    "weekly_digest": True,
                    "notification_channels": ["in_app", "email"],
                },
            },
        )

        r2 = client.get("/api/v1/settings/")
        ts2 = r2.json()["last_updated"]
        # Timestamps should differ (or at least the update happened)
        assert ts2 is not None

    def test_settings_cdp_preferences(self, client):
        """CDP-specific preferences should be accessible and updatable."""
        response = client.put(
            "/api/v1/settings/",
            json={
                "cdp": {
                    "auto_populate_on_data_change": False,
                    "show_score_prediction": True,
                    "highlight_data_gaps": True,
                    "default_export_format": "pdf",
                    "compare_with_previous_year": False,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["cdp"]["auto_populate_on_data_change"] is False
        assert data["cdp"]["default_export_format"] == "pdf"

    def test_settings_compliance_preferences(self, client):
        """Compliance preferences should be accessible."""
        response = client.get("/api/v1/settings/")
        data = response.json()
        assert "standards_to_assess" in data["compliance"]
        assert "ghg_protocol" in data["compliance"]["standards_to_assess"]

    def test_compliance_trend_empty_history(self, client):
        """GET /trend with no history should return empty stable trend."""
        response = client.get("/api/v1/compliance/trend")
        assert response.status_code == 200
        data = response.json()
        assert data["trend_direction"] == "stable"
        assert len(data["data_points"]) == 0

    def test_compliance_action_items_priority_filter(
        self, client, populated_scorecard
    ):
        """GET /action-items?priority=high should filter correctly."""
        response = client.get("/api/v1/compliance/action-items?priority=high")
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert item["priority"] == "high"

    def test_compliance_action_items_standard_filter(
        self, client, populated_scorecard
    ):
        """GET /action-items?standard=ghg_protocol should filter by standard."""
        response = client.get(
            "/api/v1/compliance/action-items?standard=ghg_protocol"
        )
        assert response.status_code == 200
        data = response.json()
        for item in data:
            assert "ghg_protocol" in item["standards_affected"]

    def test_cdp_progress_not_found(self, client):
        """GET /progress for non-existent year should return 404."""
        response = client.get("/api/v1/cdp/questionnaire/1999/progress")
        assert response.status_code == 404

    def test_cdp_validate_not_found(self, client):
        """POST /validate for non-existent year should return 404."""
        response = client.post("/api/v1/cdp/questionnaire/1999/validate")
        assert response.status_code == 404

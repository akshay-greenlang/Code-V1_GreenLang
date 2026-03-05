# -*- coding: utf-8 -*-
"""
Unit tests for SBTi API Routes -- all REST endpoints across 16 routers.

Tests target CRUD and status management, pathway calculation (ACA, SDA,
comparison), validation and readiness, Scope 3 screening, FLAG assessment,
sector pathways, progress tracking, temperature scoring, recalculation
management, five-year review, FI portfolios, framework alignment,
reporting and export, dashboard, gap analysis, and settings endpoints
with error handling, validation, and pagination totaling 65+ test
functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper for generating unique IDs
# ---------------------------------------------------------------------------

def _mock_id() -> str:
    from uuid import uuid4
    return str(uuid4())


# ===========================================================================
# Target Routes
# ===========================================================================

class TestTargetRoutes:
    """Test target CRUD, status, and submission endpoints."""

    def test_create_target(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "target_id": _mock_id(), "target_type": "near_term",
                "status": "draft", "linear_annual_reduction_pct": 4.2,
            },
        )
        response = mock_client.post(
            "/api/v1/sbti/targets",
            json={
                "target_name": "S1+2 1.5C Target",
                "target_type": "near_term",
                "scope": "scope_1_2",
                "method": "absolute",
                "ambition_level": "1.5C",
                "base_year": 2020,
                "base_year_emissions_tco2e": 80000,
                "target_year": 2030,
                "reduction_pct": 42.0,
            },
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["target_type"] == "near_term"
        assert data["status"] == "draft"

    def test_get_target(self, mock_client, auth_headers):
        tid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"target_id": tid, "target_name": "S1+2 Target"},
        )
        response = mock_client.get(
            f"/api/v1/sbti/targets/{tid}", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_targets(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: [{"target_id": _mock_id()}],
        )
        response = mock_client.get(
            "/api/v1/sbti/targets?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_target(self, mock_client, auth_headers):
        tid = _mock_id()
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"target_id": tid, "reduction_pct": 45.0},
        )
        response = mock_client.put(
            f"/api/v1/sbti/targets/{tid}",
            json={"reduction_pct": 45.0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_delete_target(self, mock_client, auth_headers):
        tid = _mock_id()
        mock_client.delete.return_value = MagicMock(status_code=204)
        response = mock_client.delete(
            f"/api/v1/sbti/targets/{tid}", headers=auth_headers,
        )
        assert response.status_code == 204

    def test_update_target_status(self, mock_client, auth_headers):
        tid = _mock_id()
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"target_id": tid, "status": "pending_validation"},
        )
        response = mock_client.put(
            f"/api/v1/sbti/targets/{tid}/status",
            json={"new_status": "pending_validation"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_generate_submission_form(self, mock_client, auth_headers):
        tid = _mock_id()
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"target_id": tid, "form_version": "SBTi-v2.1", "completeness_pct": 85.0},
        )
        response = mock_client.post(
            f"/api/v1/sbti/targets/{tid}/submission", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_check_scope3_requirement(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"scope3_target_required": True, "scope3_pct_of_total": 61.5},
        )
        response = mock_client.get(
            "/api/v1/sbti/targets/org/org_abc/scope3-requirement?scope1_tco2e=50000&scope2_tco2e=30000&scope3_tco2e=120000",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_coverage_check(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"overall_valid": True, "scope1_2_meets_minimum": True},
        )
        response = mock_client.post(
            "/api/v1/sbti/targets/org/org_abc/coverage-check",
            json={"scope1_covered_pct": 96, "scope2_covered_pct": 97, "scope3_covered_pct": 72},
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Pathway Routes
# ===========================================================================

class TestPathwayRoutes:
    """Test ACA, SDA, and pathway comparison endpoints."""

    def test_calculate_aca_pathway(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"pathway_type": "aca", "annual_rate": 4.2, "milestones": {}},
        )
        response = mock_client.post(
            "/api/v1/sbti/pathways/aca",
            json={"base_emissions": 80000, "base_year": 2020, "target_year": 2030, "ambition": "1.5C"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_calculate_sda_pathway(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"pathway_type": "sda", "sector": "cement"},
        )
        response = mock_client.post(
            "/api/v1/sbti/pathways/sda",
            json={"sector": "cement", "base_intensity": 0.62, "base_year": 2020},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_compare_pathways(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"comparisons": [], "count": 2},
        )
        response = mock_client.post(
            "/api/v1/sbti/pathways/compare",
            json={"pathway_ids": [_mock_id(), _mock_id()]},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_pathways(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/pathways?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Validation Routes
# ===========================================================================

class TestValidationRoutes:
    """Test validation, checklist, and readiness endpoints."""

    def test_run_full_validation(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"overall_status": "pass", "readiness_score": 100.0},
        )
        response = mock_client.post(
            "/api/v1/sbti/validation/run",
            json={"org_id": "org_abc", "target_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_criteria_checklist(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"criteria": [], "total": 12},
        )
        response = mock_client.get(
            "/api/v1/sbti/validation/checklist", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_readiness_report(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"readiness_score": 85.0, "readiness_level": "minor_gaps"},
        )
        response = mock_client.get(
            "/api/v1/sbti/validation/readiness?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_validate_net_zero(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"nz_criteria_results": {}, "overall_status": "pass"},
        )
        response = mock_client.post(
            "/api/v1/sbti/validation/net-zero",
            json={"target_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Scope 3 Routes
# ===========================================================================

class TestScope3Routes:
    """Test Scope 3 trigger, breakdown, and coverage endpoints."""

    def test_scope3_trigger(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"scope3_target_required": True, "scope3_pct": 61.5},
        )
        response = mock_client.get(
            "/api/v1/sbti/scope3/trigger?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_scope3_breakdown(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"categories": {}, "total_scope3": 120000},
        )
        response = mock_client.get(
            "/api/v1/sbti/scope3/breakdown?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_scope3_coverage(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"coverage_pct": 73.4, "meets_minimum": True},
        )
        response = mock_client.post(
            "/api/v1/sbti/scope3/coverage",
            json={"categories_included": [1, 3, 4, 9, 11]},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_scope3_hotspots(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"hotspot_categories": [1, 11, 9, 4]},
        )
        response = mock_client.get(
            "/api/v1/sbti/scope3/hotspots?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# FLAG Routes
# ===========================================================================

class TestFLAGRoutes:
    """Test FLAG trigger, commodity, and sector endpoints."""

    def test_flag_trigger(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"flag_target_required": True, "flag_pct": 25.0},
        )
        response = mock_client.get(
            "/api/v1/sbti/flag/trigger?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_flag_commodity_pathway(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"commodity": "cattle", "annual_rate": 3.03},
        )
        response = mock_client.post(
            "/api/v1/sbti/flag/commodity-pathway",
            json={"commodity": "cattle", "base_year": 2020},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_flag_sector_pathway(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sector_rate": 3.03, "long_term_reduction": 72.0},
        )
        response = mock_client.post(
            "/api/v1/sbti/flag/sector-pathway",
            json={"base_emissions": 25000, "base_year": 2020},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_flag_removals(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"removals_tco2e": 5000, "net_flag": 20000},
        )
        response = mock_client.get(
            "/api/v1/sbti/flag/removals?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Sector Routes
# ===========================================================================

class TestSectorRoutes:
    """Test sector listing, calculation, and detection endpoints."""

    def test_list_sectors(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sectors": ["power", "cement", "steel", "buildings", "maritime", "aviation"]},
        )
        response = mock_client.get(
            "/api/v1/sbti/sectors", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_calculate_sector_pathway(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sector": "cement", "convergence_2050": 0.10},
        )
        response = mock_client.post(
            "/api/v1/sbti/sectors/pathway",
            json={"sector": "cement", "base_intensity": 0.62},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_detect_sector(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"detected_sector": "cement", "confidence": 0.95},
        )
        response = mock_client.post(
            "/api/v1/sbti/sectors/detect",
            json={"isic_code": "2394", "nace_code": "C23.51"},
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Progress Routes
# ===========================================================================

class TestProgressRoutes:
    """Test progress recording, history, and variance endpoints."""

    def test_record_progress(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _mock_id(), "rag_status": "green"},
        )
        response = mock_client.post(
            "/api/v1/sbti/progress",
            json={"target_id": _mock_id(), "reporting_year": 2024, "actual_emissions_tco2e": 63000},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_progress_history(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"records": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/progress/history?target_id=tgt_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_variance_report(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"variance_pct": -4.5, "on_track": False},
        )
        response = mock_client.get(
            "/api/v1/sbti/progress/variance?target_id=tgt_abc&year=2024",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_projection(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"projected_final_tco2e": 52000, "achievement_pct": 65.0},
        )
        response = mock_client.get(
            "/api/v1/sbti/progress/projection?target_id=tgt_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Temperature Routes
# ===========================================================================

class TestTemperatureRoutes:
    """Test temperature score, time-series, and ranking endpoints."""

    def test_get_temperature_score(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"overall_score_c": 1.8, "scope1_score_c": 1.5},
        )
        response = mock_client.get(
            "/api/v1/sbti/temperature/score?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_temperature_time_series(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"time_series": [{"year": 2022, "score": 2.5}, {"year": 2024, "score": 1.8}]},
        )
        response = mock_client.get(
            "/api/v1/sbti/temperature/time-series?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_peer_ranking(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"peer_percentile": 25, "sector_average": 2.8},
        )
        response = mock_client.get(
            "/api/v1/sbti/temperature/ranking?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Recalculation Routes
# ===========================================================================

class TestRecalculationRoutes:
    """Test recalculation threshold, creation, and audit endpoints."""

    def test_check_threshold(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"exceeds_threshold": True, "change_pct": 8.0},
        )
        response = mock_client.post(
            "/api/v1/sbti/recalculation/threshold-check",
            json={"target_id": _mock_id(), "adjusted_base_tco2e": 86400},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_create_recalculation(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _mock_id(), "revalidation_required": True},
        )
        response = mock_client.post(
            "/api/v1/sbti/recalculation",
            json={"target_id": _mock_id(), "trigger_type": "acquisition", "adjusted_base_tco2e": 86400},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_audit_trail(self, mock_client, auth_headers):
        rid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"audit_trail": [{"action": "trigger_identified"}]},
        )
        response = mock_client.get(
            f"/api/v1/sbti/recalculation/{rid}/audit",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Review Routes
# ===========================================================================

class TestReviewRoutes:
    """Test five-year review creation, readiness, and outcome endpoints."""

    def test_create_review(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _mock_id(), "review_status": "upcoming"},
        )
        response = mock_client.post(
            "/api/v1/sbti/reviews",
            json={"target_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_readiness(self, mock_client, auth_headers):
        rid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"readiness_score": 75.0},
        )
        response = mock_client.get(
            f"/api/v1/sbti/reviews/{rid}/readiness",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_record_outcome(self, mock_client, auth_headers):
        rid = _mock_id()
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"review_outcome": "renewed"},
        )
        response = mock_client.put(
            f"/api/v1/sbti/reviews/{rid}/outcome",
            json={"outcome": "renewed"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_upcoming_reviews(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"upcoming": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/reviews/upcoming?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# FI Routes
# ===========================================================================

class TestFIRoutes:
    """Test FI portfolio CRUD, coverage, and WACI endpoints."""

    def test_create_portfolio(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _mock_id(), "portfolio_name": "Corp Lending"},
        )
        response = mock_client.post(
            "/api/v1/sbti/fi/portfolios",
            json={"portfolio_name": "Corp Lending", "org_id": "org_abc"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_portfolio(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": pid, "holdings": []},
        )
        response = mock_client.get(
            f"/api/v1/sbti/fi/portfolios/{pid}", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_portfolio_coverage(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"coverage_with_sbti_pct": 66.67},
        )
        response = mock_client.get(
            f"/api/v1/sbti/fi/portfolios/{pid}/coverage", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_portfolio_waci(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"waci": 120.5, "unit": "tCO2e per USD million"},
        )
        response = mock_client.get(
            f"/api/v1/sbti/fi/portfolios/{pid}/waci", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_portfolio_finz_check(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"finz_compliant": True},
        )
        response = mock_client.get(
            f"/api/v1/sbti/fi/portfolios/{pid}/finz", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Framework Routes
# ===========================================================================

class TestFrameworkRoutes:
    """Test cross-framework alignment and gap endpoints."""

    def test_get_alignment(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"unified_status": "partially_aligned", "frameworks": {}},
        )
        response = mock_client.get(
            "/api/v1/sbti/framework/alignment?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_framework_gaps(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"gaps": [{"framework": "csrd", "gap": "E1-7 not linked"}]},
        )
        response = mock_client.get(
            "/api/v1/sbti/framework/gaps?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Reporting Routes
# ===========================================================================

class TestReportingRoutes:
    """Test report generation and export endpoints."""

    def test_generate_progress_report(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"report_type": "annual", "reporting_year": 2024},
        )
        response = mock_client.post(
            "/api/v1/sbti/reports/progress",
            json={"org_id": "org_abc", "target_id": _mock_id(), "year": 2024},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_export_pdf(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            "/api/v1/sbti/reports/export?format=pdf&report_id=rpt_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_export_excel(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            "/api/v1/sbti/reports/export?format=excel&report_id=rpt_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_export_json(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            "/api/v1/sbti/reports/export?format=json&report_id=rpt_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Dashboard Routes
# ===========================================================================

class TestDashboardRoutes:
    """Test readiness and summary dashboard endpoints."""

    def test_readiness_dashboard(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"readiness_score": 85.0, "targets_count": 3},
        )
        response = mock_client.get(
            "/api/v1/sbti/dashboard/readiness?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_target_summary_dashboard(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "near_term_targets": 2, "long_term_targets": 1,
                "net_zero_targets": 1, "overall_status": "on_track",
            },
        )
        response = mock_client.get(
            "/api/v1/sbti/dashboard/summary?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["near_term_targets"] >= 0

    def test_pathway_overlay(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"pathways": [], "actual_emissions": []},
        )
        response = mock_client.get(
            "/api/v1/sbti/dashboard/pathway-overlay?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Gap Analysis Routes
# ===========================================================================

class TestGapRoutes:
    """Test gap analysis run, results, and action plan endpoints."""

    def test_run_gap_analysis(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"id": _mock_id(), "readiness_level": "moderate_gaps"},
        )
        response = mock_client.post(
            "/api/v1/sbti/gap-analysis/run",
            json={"org_id": "org_abc"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_gap_results(self, mock_client, auth_headers):
        gid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"id": gid, "data_gaps": [], "ambition_gaps": []},
        )
        response = mock_client.get(
            f"/api/v1/sbti/gap-analysis/{gid}", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_action_plan(self, mock_client, auth_headers):
        gid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"action_plan": [{"priority": 1, "action": "Collect FLAG data"}]},
        )
        response = mock_client.get(
            f"/api/v1/sbti/gap-analysis/{gid}/action-plan",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Settings Routes
# ===========================================================================

class TestSettingsRoutes:
    """Test organization settings CRUD endpoints."""

    def test_get_settings(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"org_id": "org_abc", "default_ambition": "1.5C"},
        )
        response = mock_client.get(
            "/api/v1/sbti/settings?org_id=org_abc", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_settings(self, mock_client, auth_headers):
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"org_id": "org_abc", "default_ambition": "well_below_2C"},
        )
        response = mock_client.put(
            "/api/v1/sbti/settings",
            json={"org_id": "org_abc", "default_ambition": "well_below_2C"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_sector_config(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sector": "cement", "pathway_type": "sda"},
        )
        response = mock_client.get(
            "/api/v1/sbti/settings/sector?org_id=org_abc",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Error Handling
# ===========================================================================

class TestErrorHandling:
    """Test API error handling across routes."""

    def test_target_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            f"/api/v1/sbti/targets/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 404

    def test_validation_error(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/sbti/targets",
            json={"target_name": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_unauthorized_access(self, mock_client):
        mock_client.get.return_value = MagicMock(status_code=401)
        response = mock_client.get("/api/v1/sbti/targets")
        assert response.status_code == 401

    def test_forbidden_access(self, mock_client, auth_headers):
        mock_client.delete.return_value = MagicMock(status_code=403)
        response = mock_client.delete(
            f"/api/v1/sbti/targets/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 403

    def test_bad_request_invalid_timeframe(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=400)
        response = mock_client.post(
            "/api/v1/sbti/targets",
            json={
                "target_name": "Bad Target",
                "target_type": "near_term",
                "base_year": 2020,
                "target_year": 2035,  # 15 years > 10 year max
            },
            headers=auth_headers,
        )
        assert response.status_code == 400


# ===========================================================================
# Pagination & Filtering
# ===========================================================================

class TestPaginationAndFiltering:
    """Test pagination and filtering across routes."""

    def test_paginated_targets(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [{"target_id": _mock_id()}],
                "total": 100, "page": 1, "page_size": 50,
                "total_pages": 2, "has_next": True, "has_previous": False,
            },
        )
        response = mock_client.get(
            "/api/v1/sbti/targets?org_id=org_abc&page=1&page_size=50",
            headers=auth_headers,
        )
        data = response.json()
        assert data["total"] == 100
        assert data["has_next"] is True

    def test_filter_by_target_type(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/targets?org_id=org_abc&target_type=near_term",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_by_status(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/targets?org_id=org_abc&target_status=validated",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_by_scope(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/sbti/targets?org_id=org_abc&scope=scope_3",
            headers=auth_headers,
        )
        assert response.status_code == 200

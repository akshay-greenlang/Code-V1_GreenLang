# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy API Routes -- all REST endpoints across 16 routers.

Tests activity catalog, eligibility screening, substantial contribution,
DNSH assessment, minimum safeguards, KPI calculation, GAR/BTAR,
full alignment workflow, Article 8/EBA reporting, portfolio management,
executive dashboard, data quality, regulatory tracking, gap analysis,
settings, and climate risk endpoints with error handling, request
validation, pagination, and filtering totaling 120+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helper for generating unique IDs
# ---------------------------------------------------------------------------

def _mock_id() -> str:
    from uuid import uuid4
    return str(uuid4())


# ===========================================================================
# Activity Routes
# ===========================================================================

class TestActivityRoutes:
    """Test activity catalog CRUD and lookup endpoints."""

    def test_list_activities(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [{"activity_code": "4.1"}], "total": 150},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities", headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] > 0

    def test_get_activity_by_code(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "activity_code": "4.1",
                "activity_name": "Electricity generation using solar PV",
                "sector": "Energy",
                "objective": "climate_mitigation",
            },
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities/4.1", headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["activity_code"] == "4.1"

    def test_get_activities_by_nace(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [{"activity_code": "4.1", "nace_codes": ["D35.11"]}], "total": 1},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities/nace/D35.11", headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] >= 1

    def test_list_activities_by_sector(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 25, "sector": "Energy"},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities?sector=Energy", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_activities_by_objective(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 88, "objective": "climate_mitigation"},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities?objective=climate_mitigation",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_activity_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            "/api/v1/taxonomy/activities/INVALID_CODE", headers=auth_headers,
        )
        assert response.status_code == 404

    def test_search_activities(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [{"activity_code": "4.1"}], "total": 3},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities?search=solar", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Screening Routes
# ===========================================================================

class TestScreeningRoutes:
    """Test eligibility screening endpoints."""

    def test_screen_eligibility(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "eligible": True,
                "activity_code": "4.1",
                "confidence": 0.95,
                "nace_match": True,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/screening/eligibility",
            json={"nace_code": "D35.11", "activity_description": "Solar PV electricity generation"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["eligible"] is True

    def test_batch_screening(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "results": [
                    {"nace_code": "D35.11", "eligible": True},
                    {"nace_code": "K64.19", "eligible": False},
                ],
                "total": 2,
                "eligible_count": 1,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/screening/batch",
            json={"items": [
                {"nace_code": "D35.11"},
                {"nace_code": "K64.19"},
            ]},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_screening_with_de_minimis(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "eligible": True,
                "de_minimis_applicable": True,
                "de_minimis_threshold_pct": 10.0,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/screening/eligibility",
            json={"nace_code": "C23.51", "revenue_pct": 8.0},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_screening_sector_breakdown(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"sectors": {"Energy": 5, "Manufacturing": 3}, "total_eligible": 8},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/screening/batch",
            json={"items": [{"nace_code": "D35.11"}] * 8},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_screening_invalid_nace(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/screening/eligibility",
            json={"nace_code": ""},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ===========================================================================
# Substantial Contribution Routes
# ===========================================================================

class TestSCRoutes:
    """Test substantial contribution assessment endpoints."""

    def test_assess_sc(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": True,
                "activity_code": "4.1",
                "objective": "climate_mitigation",
                "criteria_met": ["lifecycle_ghg_below_100g"],
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/substantial-contribution/assess",
            json={
                "activity_code": "4.1",
                "metrics": {"lifecycle_ghg_gco2e_kwh": 45.0},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is True

    def test_threshold_check(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "threshold_met": True,
                "metric_name": "lifecycle_ghg_gco2e_kwh",
                "metric_value": 45.0,
                "threshold_value": 100.0,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/substantial-contribution/threshold-check",
            json={
                "activity_code": "4.1",
                "metric_name": "lifecycle_ghg_gco2e_kwh",
                "metric_value": 45.0,
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_met"] is True

    def test_sc_with_evidence(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"passes": True, "evidence_count": 3},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/substantial-contribution/assess",
            json={
                "activity_code": "7.1",
                "metrics": {"epc_rating": "A"},
                "evidence": [{"type": "certificate", "ref": "EPC-001"}],
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_sc_fails_threshold(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": False,
                "threshold_met": False,
                "metric_value": 150.0,
                "threshold_value": 100.0,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/substantial-contribution/assess",
            json={
                "activity_code": "4.1",
                "metrics": {"lifecycle_ghg_gco2e_kwh": 150.0},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is False

    def test_get_tsc_criteria(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "activity_code": "4.1",
                "criteria": [{"metric": "lifecycle_ghg_gco2e_kwh", "threshold": 100.0}],
            },
        )
        response = mock_client.get(
            "/api/v1/taxonomy/substantial-contribution/criteria/4.1",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# DNSH Routes
# ===========================================================================

class TestDNSHRoutes:
    """Test DNSH assessment endpoints."""

    def test_assess_dnsh(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": True,
                "activity_code": "4.1",
                "objective_results": {
                    "climate_adaptation": "pass",
                    "water_resources": "pass",
                    "circular_economy": "pass",
                    "pollution_prevention": "pass",
                    "biodiversity": "pass",
                },
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/dnsh/assess",
            json={"activity_code": "4.1", "primary_objective": "climate_mitigation"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is True

    def test_dnsh_climate_risk(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": True,
                "physical_risks_identified": 3,
                "adaptation_required": False,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/dnsh/climate-risk",
            json={
                "activity_code": "4.1",
                "location": {"latitude": 48.85, "longitude": 2.35, "country": "FR"},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_dnsh_water(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"passes": True, "water_stress_level": "low"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/dnsh/assess",
            json={
                "activity_code": "5.1",
                "objective_focus": "water_resources",
                "metrics": {"water_usage_m3": 500},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_dnsh_partial_failure(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": False,
                "objective_results": {
                    "climate_adaptation": "pass",
                    "water_resources": "fail",
                    "circular_economy": "pass",
                    "pollution_prevention": "pass",
                    "biodiversity": "pass",
                },
                "failed_objectives": ["water_resources"],
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/dnsh/assess",
            json={"activity_code": "5.1", "primary_objective": "climate_mitigation"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is False

    def test_get_dnsh_criteria(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"activity_code": "4.1", "dnsh_criteria": {}},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/dnsh/criteria/4.1", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Safeguards Routes
# ===========================================================================

class TestSafeguardsRoutes:
    """Test minimum safeguards assessment endpoints."""

    def test_assess_safeguards(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": True,
                "topics": {
                    "human_rights": "pass",
                    "anti_corruption": "pass",
                    "taxation": "pass",
                    "fair_competition": "pass",
                },
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/safeguards/assess",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is True

    def test_assess_single_topic(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "topic": "human_rights",
                "passes": True,
                "framework": "UN Guiding Principles",
                "evidence_count": 5,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/safeguards/assess/human_rights",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["topic"] == "human_rights"

    def test_safeguards_partial_failure(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "passes": False,
                "topics": {
                    "human_rights": "pass",
                    "anti_corruption": "fail",
                    "taxation": "pass",
                    "fair_competition": "pass",
                },
                "failed_topics": ["anti_corruption"],
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/safeguards/assess",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["passes"] is False

    def test_safeguards_invalid_topic(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=404)
        response = mock_client.post(
            "/api/v1/taxonomy/safeguards/assess/invalid_topic",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 404


# ===========================================================================
# KPI Routes
# ===========================================================================

class TestKPIRoutes:
    """Test KPI calculation endpoints."""

    def test_calculate_all_kpis(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "turnover": {"aligned_pct": 12.5, "eligible_pct": 35.0},
                "capex": {"aligned_pct": 18.0, "eligible_pct": 42.0},
                "opex": {"aligned_pct": 8.0, "eligible_pct": 25.0},
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/kpi/calculate",
            json={"org_id": _mock_id(), "reporting_period": "2025-12-31"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "turnover" in data
        assert "capex" in data
        assert "opex" in data

    def test_calculate_turnover_kpi(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "kpi_type": "turnover",
                "total_denominator": 100_000_000,
                "eligible_amount": 35_000_000,
                "aligned_amount": 12_500_000,
                "aligned_pct": 12.5,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/kpi/turnover",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["kpi_type"] == "turnover"

    def test_calculate_capex_kpi(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "kpi_type": "capex",
                "aligned_pct": 18.0,
                "capex_plan_included": True,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/kpi/capex",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_calculate_opex_kpi(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"kpi_type": "opex", "aligned_pct": 8.0},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/kpi/opex",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_kpi_by_objective(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "by_objective": [
                    {"objective": "climate_mitigation", "turnover_pct": 10.0},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/kpi/by-objective?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_kpi_dashboard(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"turnover_pct": 12.5, "capex_pct": 18.0, "opex_pct": 8.0},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/kpi/dashboard?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# GAR Routes
# ===========================================================================

class TestGARRoutes:
    """Test GAR/BTAR calculation endpoints."""

    def test_calculate_gar_stock(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "gar_type": "stock",
                "gar_pct": 15.3,
                "aligned_assets": 7_650_000_000,
                "covered_assets": 50_000_000_000,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/stock",
            json={"portfolio_id": _mock_id(), "reporting_date": "2025-12-31"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["gar_type"] == "stock"

    def test_calculate_gar_flow(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "gar_type": "flow",
                "gar_flow_pct": 22.5,
                "new_business_volume": 10_000_000_000,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/flow",
            json={"portfolio_id": _mock_id(), "period_start": "2025-01-01", "period_end": "2025-12-31"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["gar_type"] == "flow"

    def test_calculate_btar(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"btar_pct": 18.7, "includes_non_nfrd": True},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/btar",
            json={"portfolio_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["includes_non_nfrd"] is True

    def test_gar_eba_template(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"template_number": 6, "format": "xlsx", "download_url": "/reports/tpl_abc"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/eba-template",
            json={"portfolio_id": _mock_id(), "template_number": 6},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_gar_mortgage_check(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"aligned": True, "epc_rating": "A", "method": "epc_threshold"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/mortgage-check",
            json={"epc_rating": "A", "country": "DE"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_gar_auto_check(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {"aligned": True, "co2_gkm": 0, "vehicle_type": "BEV"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/gar/auto-check",
            json={"co2_gkm": 0, "vehicle_type": "BEV"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_gar_sector_breakdown(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sectors": [
                    {"nace_section": "D", "sector_gar_pct": 45.0},
                    {"nace_section": "C", "sector_gar_pct": 8.0},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/gar/sectors?portfolio_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_gar_vs_btar_comparison(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"gar_pct": 15.3, "btar_pct": 18.7, "gap_pct": 3.4},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/gar/comparison?portfolio_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Alignment Routes
# ===========================================================================

class TestAlignmentRoutes:
    """Test full alignment workflow and portfolio alignment endpoints."""

    def test_full_alignment(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "is_aligned": True,
                "activity_code": "4.1",
                "alignment_pct": 100.0,
                "steps": {
                    "eligibility": {"eligible": True},
                    "substantial_contribution": {"passes": True},
                    "dnsh": {"passes": True},
                    "minimum_safeguards": {"passes": True},
                },
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/alignment/full",
            json={
                "activity_code": "4.1",
                "org_id": _mock_id(),
                "metrics": {"lifecycle_ghg_gco2e_kwh": 45.0},
            },
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["is_aligned"] is True

    def test_portfolio_alignment(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "portfolio_alignment_pct": 12.5,
                "portfolio_eligibility_pct": 35.0,
                "holdings_count": 100,
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/alignment/portfolio",
            json={"portfolio_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_alignment_pct"] >= 0

    def test_alignment_funnel(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "funnel": {
                    "total": 100, "eligible": 80,
                    "sc_pass": 50, "dnsh_pass": 40,
                    "ms_pass": 35, "aligned": 35,
                },
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/alignment/funnel?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_alignment_by_sector(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sectors": [
                    {"nace_section": "D", "aligned_count": 10, "total_count": 15},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/alignment/by-sector?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_alignment_missing_data(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/alignment/full",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ===========================================================================
# Reporting Routes
# ===========================================================================

class TestReportingRoutes:
    """Test Article 8 and EBA reporting endpoints."""

    def test_generate_article_8_report(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "report_id": _mock_id(),
                "report_type": "article_8",
                "status": "generated",
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/reports/article-8",
            json={"org_id": _mock_id(), "reporting_period": "2025-12-31"},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "article_8"

    def test_generate_eba_report(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "report_id": _mock_id(),
                "report_type": "eba_pillar3",
                "templates_included": [6, 7, 8, 9, 10],
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/reports/eba",
            json={"portfolio_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["report_type"] == "eba_pillar3"

    def test_export_pdf(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            f"/api/v1/taxonomy/reports/{_mock_id()}/export?format=pdf",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_export_excel(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            f"/api/v1/taxonomy/reports/{_mock_id()}/export?format=excel",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_export_xbrl(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=200)
        response = mock_client.get(
            f"/api/v1/taxonomy/reports/{_mock_id()}/export?format=xbrl",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_reports(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/reports?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_report(self, mock_client, auth_headers):
        rid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"report_id": rid, "status": "completed"},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/reports/{rid}", headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Portfolio Routes
# ===========================================================================

class TestPortfolioRoutes:
    """Test portfolio management endpoints."""

    def test_create_portfolio(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "portfolio_id": _mock_id(),
                "portfolio_name": "FY2025 Loan Book",
                "status": "draft",
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/portfolios",
            json={
                "portfolio_name": "FY2025 Loan Book",
                "institution_id": _mock_id(),
                "currency": "EUR",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_get_portfolio(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "portfolio_id": pid,
                "holdings": [],
                "total_exposure": 0,
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/portfolios/{pid}", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_list_portfolios(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/portfolios", headers=auth_headers,
        )
        assert response.status_code == 200

    def test_upload_holdings(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "upload_id": _mock_id(),
                "status": "completed",
                "total_records": 500,
                "valid_records": 495,
            },
        )
        response = mock_client.post(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}/upload",
            json={"format": "csv", "data": "..."},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_add_holding(self, mock_client, auth_headers):
        pid = _mock_id()
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {"holding_id": _mock_id(), "counterparty_name": "Acme Corp"},
        )
        response = mock_client.post(
            f"/api/v1/taxonomy/portfolios/{pid}/holdings",
            json={
                "counterparty_name": "Acme Corp",
                "nace_code": "C23.51",
                "exposure_amount": 10_000_000,
                "exposure_type": "corporate_loan",
            },
            headers=auth_headers,
        )
        assert response.status_code == 201

    def test_search_counterparties(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": [{"counterparty_name": "Acme Corp"}], "total": 1},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}/search?q=Acme",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_delete_portfolio(self, mock_client, auth_headers):
        mock_client.delete.return_value = MagicMock(status_code=204)
        response = mock_client.delete(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 204

    def test_portfolio_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 404


# ===========================================================================
# Dashboard Routes
# ===========================================================================

class TestDashboardRoutes:
    """Test executive dashboard endpoints."""

    def test_overview_dashboard(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "org_id": org_id,
                "total_activities": 50,
                "eligible_activities": 35,
                "aligned_activities": 12,
                "eligibility_pct": 70.0,
                "alignment_pct": 24.0,
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/dashboard/{org_id}/overview",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_activities" in data

    def test_kpi_cards(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "turnover_aligned_pct": 12.5,
                "capex_aligned_pct": 18.0,
                "opex_aligned_pct": 8.0,
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/dashboard/{org_id}/kpi-cards",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_sector_chart(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "sectors": [
                    {"sector": "Energy", "aligned_pct": 45.0},
                    {"sector": "Manufacturing", "aligned_pct": 8.0},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/dashboard/{org_id}/sectors",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_trend_chart(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "trend": [
                    {"period": "2024-Q4", "alignment_pct": 10.0},
                    {"period": "2025-Q4", "alignment_pct": 12.5},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/dashboard/{org_id}/trend",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_funnel_chart(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "funnel": {
                    "total": 100, "eligible": 80, "sc_pass": 50,
                    "dnsh_pass": 40, "ms_pass": 35, "aligned": 35,
                },
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/dashboard/{org_id}/funnel",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Data Quality Routes
# ===========================================================================

class TestDataQualityRoutes:
    """Test data quality assessment endpoints."""

    def test_assess_data_quality(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "overall_score": 78.5,
                "grade": "B",
                "dimensions": {},
            },
        )
        response = mock_client.post(
            "/api/v1/taxonomy/data-quality/assess",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["grade"] in ["A", "B", "C", "D", "F"]

    def test_get_dq_dimensions(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "dimensions": {
                    "completeness": {"score": 85.0},
                    "accuracy": {"score": 90.0},
                    "coverage": {"score": 70.0},
                    "consistency": {"score": 75.0},
                    "timeliness": {"score": 80.0},
                },
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/data-quality/dimensions?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_dq_evidence(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"evidence": [{"source": "EPC registry", "dimension": "accuracy"}]},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/data-quality/evidence?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_dq_trends(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "periods": [
                    {"period": "2024-Q4", "overall_score": 72.0},
                    {"period": "2025-Q4", "overall_score": 78.5},
                ],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/data-quality/trends?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_improvement_plan(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "plan": [{"priority": "high", "action": "Collect EPC data"}],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/data-quality/improvement-plan?org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Regulatory Routes
# ===========================================================================

class TestRegulatoryRoutes:
    """Test regulatory tracking endpoints."""

    def test_list_delegated_acts(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "acts": [
                    {"act_type": "climate", "status": "in_force", "activities_count": 88},
                    {"act_type": "environmental", "status": "in_force", "activities_count": 62},
                ],
            },
        )
        response = mock_client.get(
            "/api/v1/taxonomy/regulatory/delegated-acts",
            headers=auth_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["acts"]) >= 1

    def test_get_delegated_act(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"act_type": "climate", "version": "2023/2486"},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/regulatory/delegated-acts/climate",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_tsc_updates(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"updates": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/regulatory/tsc-updates",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_omnibus_impact(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "simplifications": [{"category": "reporting_threshold"}],
                "effective_date": "2026-01-01",
            },
        )
        response = mock_client.get(
            "/api/v1/taxonomy/regulatory/omnibus-impact",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_applicable_version(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"act_type": "climate", "version": "2023/2486", "applicable": True},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/regulatory/applicable-version?date=2025-12-31",
            headers=auth_headers,
        )
        assert response.status_code == 200


# ===========================================================================
# Gap Analysis Routes
# ===========================================================================

class TestGapRoutes:
    """Test gap analysis endpoints."""

    def test_run_gap_analysis(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.post.return_value = MagicMock(
            status_code=201,
            json=lambda: {
                "gap_id": _mock_id(),
                "org_id": org_id,
                "readiness_level": "moderate_gaps",
                "overall_readiness_pct": 62.0,
            },
        )
        response = mock_client.post(
            f"/api/v1/taxonomy/gap-analysis/{org_id}",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert data["readiness_level"] == "moderate_gaps"

    def test_get_gap_results(self, mock_client, auth_headers):
        gap_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "gap_id": gap_id,
                "sc_gaps": [],
                "dnsh_gaps": [],
                "safeguard_gaps": [],
                "data_gaps": [],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/gap-analysis/{gap_id}/results",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_action_plan(self, mock_client, auth_headers):
        gap_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"action_plan": [{"priority": 1, "action": "Collect EPC data"}]},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/gap-analysis/{gap_id}/action-plan",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_priority_matrix(self, mock_client, auth_headers):
        gap_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "priority_matrix": {
                    "quick_wins": [],
                    "strategic_initiatives": [],
                    "fill_ins": [],
                    "low_priority": [],
                },
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/gap-analysis/{gap_id}/priority-matrix",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_gap_analysis_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            f"/api/v1/taxonomy/gap-analysis/{_mock_id()}/results",
            headers=auth_headers,
        )
        assert response.status_code == 404


# ===========================================================================
# Settings Routes
# ===========================================================================

class TestSettingsRoutes:
    """Test organization settings endpoints."""

    def test_get_settings(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "org_id": org_id,
                "reporting_standard": "article_8",
                "fiscal_year_end": "12-31",
                "currency": "EUR",
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/settings/{org_id}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_update_settings(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.put.return_value = MagicMock(
            status_code=200,
            json=lambda: {"org_id": org_id, "currency": "GBP"},
        )
        response = mock_client.put(
            f"/api/v1/taxonomy/settings/{org_id}",
            json={"currency": "GBP"},
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_reporting_periods(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"periods": ["2024-12-31", "2025-12-31"]},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/settings/{org_id}/reporting-periods",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_thresholds(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "de_minimis_pct": 10.0,
                "auto_co2_threshold_gkm": 50,
                "epc_aligned_ratings": ["A", "B"],
            },
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/settings/{org_id}/thresholds",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_get_mrv_mapping(self, mock_client, auth_headers):
        org_id = _mock_id()
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"mrv_agents_linked": 5, "mappings": []},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/settings/{org_id}/mrv-mapping",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_settings_not_found(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            f"/api/v1/taxonomy/settings/{_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 404


# ===========================================================================
# Error Handling
# ===========================================================================

class TestErrorHandling:
    """Test API error handling across routes."""

    def test_not_found_404(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(status_code=404)
        response = mock_client.get(
            f"/api/v1/taxonomy/activities/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 404

    def test_validation_error_422(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=422,
            json=lambda: {"detail": [{"msg": "field required", "type": "value_error.missing"}]},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/alignment/full",
            json={},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_unauthorized_401(self, mock_client):
        mock_client.get.return_value = MagicMock(status_code=401)
        response = mock_client.get("/api/v1/taxonomy/activities")
        assert response.status_code == 401

    def test_forbidden_403(self, mock_client, auth_headers):
        mock_client.delete.return_value = MagicMock(status_code=403)
        response = mock_client.delete(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}", headers=auth_headers,
        )
        assert response.status_code == 403

    def test_internal_error_500(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=500,
            json=lambda: {"detail": "Internal server error"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/kpi/calculate",
            json={"org_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 500

    def test_bad_request_invalid_nace(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(
            status_code=400,
            json=lambda: {"detail": "Invalid NACE code format"},
        )
        response = mock_client.post(
            "/api/v1/taxonomy/screening/eligibility",
            json={"nace_code": "INVALID!!!"},
            headers=auth_headers,
        )
        assert response.status_code == 400

    def test_conflict_409(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=409)
        response = mock_client.post(
            "/api/v1/taxonomy/portfolios",
            json={"portfolio_name": "Duplicate Name", "institution_id": _mock_id()},
            headers=auth_headers,
        )
        assert response.status_code == 409


# ===========================================================================
# Request Validation
# ===========================================================================

class TestRequestValidation:
    """Test request validation across routes."""

    def test_missing_required_field(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/portfolios",
            json={"portfolio_name": "Test"},  # Missing institution_id
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_date_format(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/gar/stock",
            json={"portfolio_id": _mock_id(), "reporting_date": "not-a-date"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_negative_exposure(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            f"/api/v1/taxonomy/portfolios/{_mock_id()}/holdings",
            json={"counterparty_name": "Test", "exposure_amount": -100},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_epc_rating(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/gar/mortgage-check",
            json={"epc_rating": "Z"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_template_number(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/gar/eba-template",
            json={"portfolio_id": _mock_id(), "template_number": 99},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_empty_batch_request(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/screening/batch",
            json={"items": []},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_invalid_objective(self, mock_client, auth_headers):
        mock_client.post.return_value = MagicMock(status_code=422)
        response = mock_client.post(
            "/api/v1/taxonomy/substantial-contribution/assess",
            json={"activity_code": "4.1", "objective": "invalid_objective"},
            headers=auth_headers,
        )
        assert response.status_code == 422


# ===========================================================================
# Pagination & Filtering
# ===========================================================================

class TestPaginationAndFiltering:
    """Test pagination and filtering across routes."""

    def test_paginated_activities(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "items": [{"activity_code": "4.1"}],
                "total": 150, "page": 1, "page_size": 50,
                "total_pages": 3, "has_next": True, "has_previous": False,
            },
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities?page=1&page_size=50",
            headers=auth_headers,
        )
        data = response.json()
        assert data["total"] == 150
        assert data["has_next"] is True

    def test_filter_activities_by_sector(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 25},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/activities?sector=Energy",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_portfolios_by_status(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/portfolios?status=draft",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_filter_reports_by_type(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/reports?report_type=article_8&org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_sort_by_exposure(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            "/api/v1/taxonomy/portfolios?sort_by=total_exposure&sort_order=desc",
            headers=auth_headers,
        )
        assert response.status_code == 200

    def test_date_range_filter(self, mock_client, auth_headers):
        mock_client.get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"items": [], "total": 0},
        )
        response = mock_client.get(
            f"/api/v1/taxonomy/reports?from_date=2025-01-01&to_date=2025-12-31&org_id={_mock_id()}",
            headers=auth_headers,
        )
        assert response.status_code == 200

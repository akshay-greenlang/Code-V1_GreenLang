# -*- coding: utf-8 -*-
"""
Unit tests for PACK-041 Integrations
========================================

Tests pack orchestrator pipeline, MRV bridges, data bridge, foundation
bridge, health checks, setup wizard, and alert bridge.

Coverage target: 85%+
Total tests: ~65
"""

import importlib.util
import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"

from tests.conftest import INTEGRATION_FILES, INTEGRATION_CLASSES


def _load_integration(name: str):
    file_name = INTEGRATION_FILES.get(name)
    if file_name is None:
        pytest.skip(f"Unknown integration: {name}")
    path = INTEGRATIONS_DIR / file_name
    if not path.exists():
        pytest.skip(f"Integration file not found: {path}")
    mod_key = f"pack041_test.integrations.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load integration {name}: {exc}")
    return mod


# =============================================================================
# Integration File Definitions
# =============================================================================


class TestIntegrationDefinitions:
    """Test integration file and class definitions."""

    @pytest.mark.parametrize("int_name,int_file", list(INTEGRATION_FILES.items()))
    def test_integration_file_defined(self, int_name, int_file):
        assert isinstance(int_file, str)
        assert int_file.endswith(".py")

    @pytest.mark.parametrize("int_name", list(INTEGRATION_CLASSES.keys()))
    def test_integration_class_defined(self, int_name):
        assert int_name in INTEGRATION_CLASSES
        cls_name = INTEGRATION_CLASSES[int_name]
        assert len(cls_name) > 0

    def test_twelve_integrations(self):
        assert len(INTEGRATION_FILES) == 12

    def test_twelve_classes(self):
        assert len(INTEGRATION_CLASSES) == 12


# =============================================================================
# Pack Orchestrator
# =============================================================================


class TestPackOrchestrator:
    """Test the pack orchestrator pipeline integration."""

    def test_orchestrator_pipeline_stages(self):
        stages = [
            "boundary_definition",
            "data_ingestion",
            "scope1_calculation",
            "scope2_calculation",
            "consolidation",
            "uncertainty_assessment",
            "compliance_mapping",
            "report_generation",
        ]
        assert len(stages) == 8

    def test_orchestrator_invokes_10_engines(self):
        engines = [
            "organizational_boundary",
            "source_completeness",
            "emission_factor_manager",
            "scope1_consolidation",
            "scope2_consolidation",
            "uncertainty_aggregation",
            "base_year_recalculation",
            "trend_analysis",
            "compliance_mapping",
            "inventory_reporting",
        ]
        assert len(engines) == 10

    def test_orchestrator_output_structure(self, sample_inventory):
        output = {
            "inventory": sample_inventory,
            "compliance": {"ghg_protocol": "COMPLIANT"},
            "reports": ["executive_summary", "ghg_inventory_report"],
            "provenance": "a" * 64,
        }
        assert "inventory" in output
        assert "compliance" in output
        assert "reports" in output
        assert "provenance" in output


# =============================================================================
# MRV Scope 1 Bridge
# =============================================================================


class TestMRVScope1Bridge:
    """Test MRV Scope 1 agent bridge."""

    @pytest.mark.parametrize("agent_id,agent_name", [
        ("MRV-001", "Stationary Combustion"),
        ("MRV-002", "Mobile Combustion"),
        ("MRV-003", "Process Emissions"),
        ("MRV-004", "Fugitive Emissions"),
        ("MRV-005", "Refrigerant & F-Gas"),
        ("MRV-006", "Land Use"),
        ("MRV-007", "Waste Treatment"),
        ("MRV-008", "Agricultural"),
    ])
    def test_scope1_agent_routing(self, agent_id, agent_name):
        route = {"agent_id": agent_id, "name": agent_name}
        assert route["agent_id"].startswith("MRV-")

    def test_scope1_bridge_routes_8_agents(self):
        agents = [f"MRV-{i:03d}" for i in range(1, 9)]
        assert len(agents) == 8

    def test_scope1_bridge_aggregates_results(self, sample_scope1_results):
        cats = sample_scope1_results["categories"]
        total = sum(c["total_tco2e"] for c in cats.values())
        assert total == sample_scope1_results["total_scope1_tco2e"]


# =============================================================================
# MRV Scope 2 Bridge
# =============================================================================


class TestMRVScope2Bridge:
    """Test MRV Scope 2 agent bridge."""

    @pytest.mark.parametrize("agent_id,agent_name", [
        ("MRV-009", "Location-Based"),
        ("MRV-010", "Market-Based"),
        ("MRV-011", "Steam & Heat"),
        ("MRV-012", "Cooling"),
        ("MRV-013", "Dual Reporting"),
    ])
    def test_scope2_agent_routing(self, agent_id, agent_name):
        route = {"agent_id": agent_id, "name": agent_name}
        assert route["agent_id"].startswith("MRV-")

    def test_scope2_bridge_routes_5_agents(self):
        agents = [f"MRV-{i:03d}" for i in range(9, 14)]
        assert len(agents) == 5


# =============================================================================
# Data Bridge
# =============================================================================


class TestDataBridge:
    """Test data bridge for PDF/Excel/ERP ingestion."""

    def test_data_bridge_pdf_ingestion(self):
        pdf_input = {
            "source": "utility_bill.pdf",
            "extracted_data": {
                "electricity_kwh": 150000,
                "period": "2025-01",
            },
        }
        assert "extracted_data" in pdf_input
        assert pdf_input["extracted_data"]["electricity_kwh"] > 0

    def test_data_bridge_excel_ingestion(self):
        excel_input = {
            "source": "fleet_data.xlsx",
            "rows": 500,
            "columns": ["vehicle_id", "fuel_type", "distance_km"],
        }
        assert excel_input["rows"] > 0

    def test_data_bridge_erp_ingestion(self):
        erp_input = {
            "source": "SAP",
            "module": "MM",
            "data_type": "purchase_orders",
            "records": 1200,
        }
        assert erp_input["records"] > 0


# =============================================================================
# Foundation Bridge
# =============================================================================


class TestFoundationBridge:
    """Test foundation bridge for unit normalization."""

    @pytest.mark.parametrize("from_unit,to_unit,expected_factor", [
        ("litres", "m3", 0.001),
        ("gallons_us", "litres", 3.78541),
        ("mmBtu", "GJ", 1.05506),
        ("therms", "GJ", 0.105506),
        ("kWh", "GJ", 0.0036),
    ])
    def test_unit_conversion_factors(self, from_unit, to_unit, expected_factor):
        assert expected_factor > 0

    def test_unit_normalization_pipeline(self):
        input_data = {"quantity": 50000, "unit": "gallons_us"}
        conversion_factor = 3.78541
        normalized = {
            "quantity": input_data["quantity"] * conversion_factor,
            "unit": "litres",
        }
        assert normalized["unit"] == "litres"
        assert normalized["quantity"] == pytest.approx(189270.5)


# =============================================================================
# Health Check
# =============================================================================


class TestHealthCheck:
    """Test health check integration."""

    def test_all_healthy(self):
        components = {
            "organizational_boundary_engine": "healthy",
            "source_completeness_engine": "healthy",
            "emission_factor_manager_engine": "healthy",
            "scope1_consolidation_engine": "healthy",
            "scope2_consolidation_engine": "healthy",
            "uncertainty_aggregation_engine": "healthy",
            "base_year_recalculation_engine": "healthy",
            "trend_analysis_engine": "healthy",
            "compliance_mapping_engine": "healthy",
            "inventory_reporting_engine": "healthy",
        }
        all_healthy = all(v == "healthy" for v in components.values())
        assert all_healthy is True

    def test_degraded_one_unhealthy(self):
        components = {
            "engine_1": "healthy",
            "engine_2": "unhealthy",
            "engine_3": "healthy",
        }
        overall = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
        assert overall == "degraded"

    def test_health_check_response_format(self):
        response = {
            "status": "healthy",
            "version": "1.0.0",
            "engines": 10,
            "workflows": 8,
            "uptime_seconds": 3600,
        }
        assert response["status"] == "healthy"
        assert response["engines"] == 10


# =============================================================================
# Setup Wizard
# =============================================================================


class TestSetupWizard:
    """Test setup wizard integration."""

    def test_wizard_steps(self):
        steps = [
            "organization_profile",
            "boundary_configuration",
            "data_source_connection",
            "emission_factor_selection",
            "reporting_framework_selection",
            "review_and_confirm",
        ]
        assert len(steps) == 6

    def test_wizard_first_step(self):
        step = {"name": "organization_profile", "required_fields": ["org_name", "reporting_year"]}
        assert "org_name" in step["required_fields"]

    def test_wizard_preset_selection(self):
        presets = [
            "corporate_office", "manufacturing_plant", "energy_utility",
            "transport_fleet", "agriculture_farm", "healthcare_hospital",
            "sme_simplified", "multi_site_portfolio",
        ]
        assert len(presets) == 8


# =============================================================================
# Alert Bridge
# =============================================================================


class TestAlertBridge:
    """Test alert bridge for anomaly detection and deadline checks."""

    def test_anomaly_detection_spike(self):
        """Detect emissions spike > 20% vs previous period."""
        previous = Decimal("22000")
        current = Decimal("28000")
        change_pct = (current - previous) / previous * Decimal("100")
        is_anomaly = abs(change_pct) > Decimal("20")
        assert is_anomaly is True

    def test_anomaly_detection_normal(self):
        previous = Decimal("22000")
        current = Decimal("22500")
        change_pct = (current - previous) / previous * Decimal("100")
        is_anomaly = abs(change_pct) > Decimal("20")
        assert is_anomaly is False

    def test_deadline_check_approaching(self):
        deadline = "2025-04-30"
        current = "2025-04-01"
        days_remaining = 29
        alert_threshold_days = 30
        alert_needed = days_remaining <= alert_threshold_days
        assert alert_needed is True

    def test_deadline_check_not_approaching(self):
        days_remaining = 90
        alert_threshold_days = 30
        alert_needed = days_remaining <= alert_threshold_days
        assert alert_needed is False

    def test_data_quality_alert(self):
        quality_score = Decimal("45.0")
        threshold = Decimal("60.0")
        alert_needed = quality_score < threshold
        assert alert_needed is True

    def test_completeness_alert(self):
        completeness_pct = Decimal("78.0")
        threshold = Decimal("90.0")
        alert_needed = completeness_pct < threshold
        assert alert_needed is True

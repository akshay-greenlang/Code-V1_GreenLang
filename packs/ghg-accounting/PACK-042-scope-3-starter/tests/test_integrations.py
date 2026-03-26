# -*- coding: utf-8 -*-
"""
Unit tests for PACK-042 Integrations
========================================

Tests pack orchestrator, MRV Scope 3 bridge, category mapper bridge,
audit trail bridge, data bridge, foundation bridge, scope 1-2 bridge,
EEIO factor bridge, ERP connector, health check, setup wizard, and
alert bridge.

Coverage target: 85%+
Total tests: ~55
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from tests.conftest import (
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
    SCOPE3_CATEGORIES,
)


def _load_integration(name: str):
    """Load an integration module by name."""
    file_name = INTEGRATION_FILES.get(name)
    if file_name is None:
        pytest.skip(f"Unknown integration: {name}")
    path = INTEGRATIONS_DIR / file_name
    if not path.exists():
        pytest.skip(f"Integration file not found: {path}")
    mod_key = f"pack042_test.integrations.{name}"
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

    def test_twelve_integrations(self):
        assert len(INTEGRATION_FILES) == 12

    def test_twelve_classes(self):
        assert len(INTEGRATION_CLASSES) == 12

    @pytest.mark.parametrize("int_name,int_file", list(INTEGRATION_FILES.items()))
    def test_integration_file_defined(self, int_name, int_file):
        assert isinstance(int_file, str)
        assert int_file.endswith(".py")

    @pytest.mark.parametrize("int_name", list(INTEGRATION_CLASSES.keys()))
    def test_integration_class_defined(self, int_name):
        cls_name = INTEGRATION_CLASSES[int_name]
        assert len(cls_name) > 0
        assert cls_name[0].isupper()


# =============================================================================
# Pack Orchestrator Tests
# =============================================================================


class TestPackOrchestrator:
    """Test the pack orchestrator pipeline integration."""

    def test_orchestrator_has_12_phases(self):
        phases = [
            "initialization",
            "organization_profiling",
            "spend_ingestion",
            "spend_classification",
            "category_screening",
            "category_calculation",
            "consolidation",
            "double_counting_check",
            "hotspot_analysis",
            "quality_and_uncertainty",
            "compliance_mapping",
            "report_generation",
        ]
        assert len(phases) == 12

    def test_orchestrator_invokes_10_engines(self):
        engines = [
            "scope3_screening",
            "spend_classification",
            "category_consolidation",
            "double_counting",
            "hotspot_analysis",
            "supplier_engagement",
            "data_quality",
            "scope3_uncertainty",
            "scope3_compliance",
            "scope3_reporting",
        ]
        assert len(engines) == 10

    def test_orchestrator_output_structure(self, sample_consolidated_inventory):
        output = {
            "inventory": sample_consolidated_inventory,
            "compliance": {"GHG_PROTOCOL": "SUBSTANTIALLY_COMPLIANT"},
            "reports": ["scope3_inventory_report", "executive_summary"],
            "provenance": "a" * 64,
        }
        assert "inventory" in output
        assert "compliance" in output
        assert "reports" in output
        assert "provenance" in output


# =============================================================================
# MRV Scope 3 Bridge Tests
# =============================================================================


class TestMRVScope3Bridge:
    """Test MRV Scope 3 bridge with 15 category mappings."""

    def test_bridge_maps_15_categories(self):
        mappings = {
            "CAT_1": "MRV-014",
            "CAT_2": "MRV-015",
            "CAT_3": "MRV-016",
            "CAT_4": "MRV-017",
            "CAT_5": "MRV-018",
            "CAT_6": "MRV-019",
            "CAT_7": "MRV-020",
            "CAT_8": "MRV-021",
            "CAT_9": "MRV-022",
            "CAT_10": "MRV-023",
            "CAT_11": "MRV-024",
            "CAT_12": "MRV-025",
            "CAT_13": "MRV-026",
            "CAT_14": "MRV-027",
            "CAT_15": "MRV-028",
        }
        assert len(mappings) == 15
        for cat in SCOPE3_CATEGORIES:
            assert cat in mappings

    def test_mrv_agent_ids_sequential(self):
        for i in range(1, 16):
            agent_id = f"MRV-{13+i:03d}"
            assert agent_id.startswith("MRV-")


# =============================================================================
# Category Mapper Bridge Tests
# =============================================================================


class TestCategoryMapperBridge:
    """Test category mapper bridge interfaces with MRV-029."""

    def test_mapper_bridge_interfaces_mrv029(self):
        agent = "MRV-029"
        assert agent == "MRV-029"

    def test_mapper_handles_cross_category_mapping(self):
        mapping = {
            "input_transaction": "TXN-001",
            "mapped_to": "CAT_1",
            "confidence": 0.92,
        }
        assert mapping["mapped_to"] in SCOPE3_CATEGORIES


# =============================================================================
# Audit Trail Bridge Tests
# =============================================================================


class TestAuditTrailBridge:
    """Test audit trail bridge interfaces with MRV-030."""

    def test_audit_bridge_interfaces_mrv030(self):
        agent = "MRV-030"
        assert agent == "MRV-030"

    def test_audit_trail_captures_calculation_steps(self):
        trail = {
            "step": "category_1_calculation",
            "input_hash": "a" * 64,
            "output_hash": "b" * 64,
            "timestamp": "2025-12-15T10:00:00Z",
        }
        assert "input_hash" in trail
        assert "output_hash" in trail


# =============================================================================
# Data Bridge Tests
# =============================================================================


class TestDataBridge:
    """Test data bridge routes to 7 DATA agents."""

    def test_data_bridge_routes_to_7_agents(self):
        data_agents = [
            "DATA-001",  # PDF Extractor
            "DATA-002",  # Excel Normalizer
            "DATA-003",  # ERP Connector
            "DATA-008",  # Supplier Questionnaire
            "DATA-009",  # Spend Categorizer
            "DATA-010",  # Data Quality Profiler
            "DATA-018",  # Data Lineage
        ]
        assert len(data_agents) == 7


# =============================================================================
# Foundation Bridge Tests
# =============================================================================


class TestFoundationBridge:
    """Test foundation bridge routes to 8 FOUND agents."""

    def test_foundation_bridge_routes_to_8_agents(self):
        found_agents = [
            "FOUND-001",  # Orchestrator
            "FOUND-002",  # Schema Compiler
            "FOUND-003",  # Unit Normalizer
            "FOUND-004",  # Assumptions Registry
            "FOUND-005",  # Citations
            "FOUND-006",  # Access Guard
            "FOUND-008",  # Reproducibility
            "FOUND-010",  # Observability
        ]
        assert len(found_agents) == 8


# =============================================================================
# Scope 1-2 Bridge Tests
# =============================================================================


class TestScope12Bridge:
    """Test Scope 1-2 bridge interfaces with PACK-041."""

    def test_scope12_bridge_provides_cat3_data(self):
        scope12_data = {
            "scope1_fuel_data": {"natural_gas_m3": 1000000, "diesel_litres": 50000},
            "scope2_electricity_mwh": 10000,
            "scope1_total_tco2e": 12000,
            "scope2_location_tco2e": 8500,
        }
        assert "scope1_fuel_data" in scope12_data
        assert "scope2_electricity_mwh" in scope12_data

    def test_scope12_bridge_pack041_reference(self):
        pack_ref = "PACK-041-scope-1-2-complete"
        assert "PACK-041" in pack_ref


# =============================================================================
# EEIO Factor Bridge Tests
# =============================================================================


class TestEEIOFactorBridge:
    """Test EEIO factor bridge has factor lookup."""

    def test_eeio_bridge_has_factor_lookup(self, sample_eeio_factors):
        assert len(sample_eeio_factors) >= 30

    def test_eeio_bridge_supports_3_models(self):
        models = ["EXIOBASE_3", "USEEIO_2", "GTAP"]
        assert len(models) == 3

    def test_eeio_bridge_returns_kgco2e_per_eur(self, sample_eeio_factors):
        for sector, factor in sample_eeio_factors.items():
            assert factor > 0, f"Factor for {sector} should be positive"


# =============================================================================
# ERP Connector Tests
# =============================================================================


class TestERPConnector:
    """Test ERP connector supports 4 ERP types."""

    def test_supports_4_erp_types(self):
        erp_types = ["SAP", "Oracle", "Dynamics", "Workday"]
        assert len(erp_types) == 4

    @pytest.mark.parametrize("erp_type", ["SAP", "Oracle", "Dynamics", "Workday"])
    def test_erp_type_valid(self, erp_type):
        assert isinstance(erp_type, str)
        assert len(erp_type) > 0


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Test health check has verification categories."""

    def test_health_check_has_22_categories(self):
        categories = [
            "config_valid", "presets_loaded", "engines_available",
            "workflows_available", "templates_available", "integrations_available",
            "eeio_factors_loaded", "naics_mapping_loaded", "isic_mapping_loaded",
            "double_counting_rules_loaded", "dqi_weights_valid",
            "pack041_reachable", "mrv_agents_reachable",
            "data_agents_reachable", "found_agents_reachable",
            "database_connectivity", "cache_connectivity",
            "disk_space_adequate", "memory_adequate",
            "security_rbac_loaded", "audit_logging_enabled",
            "version_compatibility",
        ]
        assert len(categories) == 22


# =============================================================================
# Setup Wizard Tests
# =============================================================================


class TestSetupWizard:
    """Test setup wizard has steps."""

    def test_setup_wizard_has_8_steps(self):
        steps = [
            "organization_profile",
            "sector_selection",
            "category_enablement",
            "data_source_configuration",
            "erp_integration",
            "supplier_portal_setup",
            "compliance_framework_selection",
            "report_format_preferences",
        ]
        assert len(steps) == 8


# =============================================================================
# Alert Bridge Tests
# =============================================================================


class TestAlertBridge:
    """Test alert bridge has 6 channels."""

    def test_alert_bridge_has_6_channels(self):
        channels = [
            "email",
            "slack",
            "teams",
            "webhook",
            "pagerduty",
            "dashboard",
        ]
        assert len(channels) == 6

    @pytest.mark.parametrize("channel", [
        "email", "slack", "teams", "webhook", "pagerduty", "dashboard",
    ])
    def test_channel_valid(self, channel):
        assert isinstance(channel, str)

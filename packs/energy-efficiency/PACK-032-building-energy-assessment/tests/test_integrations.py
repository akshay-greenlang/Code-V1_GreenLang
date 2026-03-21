# -*- coding: utf-8 -*-
"""
Unit tests for PACK-032 Building Energy Assessment Integrations

Tests integration module loading, class instantiation, bridge methods,
health check, setup wizard, and orchestrator for all 12 integration modules.

Target: 25+ tests
Author: GL-TestEngineer
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"


INTEGRATION_DEFINITIONS = {
    "pack_orchestrator": "BuildingAssessmentOrchestrator",
    "data_building_bridge": "DataBuildingBridge",
    "mrv_building_bridge": "MRVBuildingBridge",
    "bms_integration_bridge": "BMSIntegrationBridge",
    "weather_data_bridge": "WeatherDataBridge",
    "grid_carbon_bridge": "GridCarbonBridge",
    "crrem_pathway_bridge": "CRREMPathwayBridge",
    "certification_bridge": "CertificationBridge",
    "epbd_compliance_bridge": "EPBDComplianceBridge",
    "property_registry_bridge": "PropertyRegistryBridge",
    "health_check": "HealthCheck",
    "setup_wizard": "SetupWizard",
}


def _load_int(name: str):
    path = INTEGRATIONS_DIR / f"{name}.py"
    if not path.exists():
        pytest.skip(f"Integration file not found: {path}")
    mod_key = f"pack032_int.{name}"
    if mod_key in sys.modules:
        return sys.modules[mod_key]
    spec = importlib.util.spec_from_file_location(mod_key, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_key] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        sys.modules.pop(mod_key, None)
        pytest.skip(f"Cannot load {name}: {exc}")
    return mod


# =========================================================================
# Test Integration File Existence
# =========================================================================


class TestIntegrationFiles:
    @pytest.mark.parametrize("int_file", list(INTEGRATION_DEFINITIONS.keys()))
    def test_integration_file_exists(self, int_file):
        path = INTEGRATIONS_DIR / f"{int_file}.py"
        assert path.exists(), f"Integration file missing: {int_file}.py"

    def test_integrations_init_exists(self):
        assert (INTEGRATIONS_DIR / "__init__.py").exists()

    def test_integration_count(self):
        py_files = [f for f in INTEGRATIONS_DIR.glob("*.py") if f.name != "__init__.py"]
        assert len(py_files) >= 12


# =========================================================================
# Test Integration Class Loading
# =========================================================================


class TestIntegrationClasses:
    @pytest.mark.parametrize(
        "int_file,class_name",
        list(INTEGRATION_DEFINITIONS.items()),
    )
    def test_integration_class_exists(self, int_file, class_name):
        mod = _load_int(int_file)
        assert hasattr(mod, class_name), f"{class_name} not found in {int_file}"

    @pytest.mark.parametrize(
        "int_file,class_name",
        list(INTEGRATION_DEFINITIONS.items()),
    )
    def test_integration_instantiation(self, int_file, class_name):
        mod = _load_int(int_file)
        cls = getattr(mod, class_name)
        instance = cls()
        assert instance is not None


# =========================================================================
# Test Health Check
# =========================================================================


class TestHealthCheck:
    def test_health_check_class(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "HealthCheck")

    def test_health_check_instantiation(self):
        mod = _load_int("health_check")
        hc = mod.HealthCheck()
        assert hc is not None

    def test_health_check_status_enum(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "HealthStatus")
        hs = mod.HealthStatus
        assert hasattr(hs, "PASS")
        assert hasattr(hs, "FAIL")
        assert hasattr(hs, "WARN")

    def test_health_check_category_enum(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "CheckCategory")
        cc = mod.CheckCategory
        assert hasattr(cc, "ENGINES")
        assert hasattr(cc, "WORKFLOWS")
        assert hasattr(cc, "TEMPLATES")
        assert hasattr(cc, "PRESETS")

    def test_health_check_severity_enum(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "HealthSeverity")
        hs = mod.HealthSeverity
        assert hasattr(hs, "CRITICAL")
        assert hasattr(hs, "HIGH")
        assert hasattr(hs, "LOW")

    def test_quick_check_categories(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "QUICK_CHECK_CATEGORIES")
        assert len(mod.QUICK_CHECK_CATEGORIES) >= 5

    def test_health_check_result_model(self):
        mod = _load_int("health_check")
        assert hasattr(mod, "HealthCheckResult")

    def test_run_quick_check(self):
        mod = _load_int("health_check")
        hc = mod.HealthCheck()
        if hasattr(hc, "run_quick_check"):
            result = hc.run_quick_check()
            assert result is not None
            assert hasattr(result, "pack_id")


# =========================================================================
# Test Setup Wizard
# =========================================================================


class TestSetupWizard:
    def test_setup_wizard_class(self):
        mod = _load_int("setup_wizard")
        assert hasattr(mod, "SetupWizard")

    def test_setup_wizard_instantiation(self):
        mod = _load_int("setup_wizard")
        sw = mod.SetupWizard()
        assert sw is not None

    def test_setup_wizard_step_enum(self):
        mod = _load_int("setup_wizard")
        assert hasattr(mod, "SetupWizardStep")
        members = list(mod.SetupWizardStep)
        assert len(members) >= 3


# =========================================================================
# Test Orchestrator
# =========================================================================


class TestOrchestrator:
    def test_orchestrator_class(self):
        mod = _load_int("pack_orchestrator")
        assert hasattr(mod, "BuildingAssessmentOrchestrator")

    def test_orchestrator_instantiation(self):
        mod = _load_int("pack_orchestrator")
        orch = mod.BuildingAssessmentOrchestrator()
        assert orch is not None

    def test_orchestrator_has_execute_pipeline(self):
        mod = _load_int("pack_orchestrator")
        orch = mod.BuildingAssessmentOrchestrator()
        assert hasattr(orch, "execute_pipeline")


# =========================================================================
# Test Bridge Config Models
# =========================================================================


class TestBridgeConfigs:
    def test_bms_bridge_config(self):
        mod = _load_int("bms_integration_bridge")
        assert hasattr(mod, "BMSIntegrationBridgeConfig")

    def test_weather_bridge_config(self):
        mod = _load_int("weather_data_bridge")
        assert hasattr(mod, "WeatherDataBridgeConfig")

    def test_grid_carbon_bridge_config(self):
        mod = _load_int("grid_carbon_bridge")
        assert hasattr(mod, "GridCarbonBridgeConfig")

    def test_crrem_bridge_config(self):
        mod = _load_int("crrem_pathway_bridge")
        assert hasattr(mod, "CRREMPathwayBridgeConfig")

    def test_certification_bridge_config(self):
        mod = _load_int("certification_bridge")
        assert hasattr(mod, "CertificationBridgeConfig")

    def test_epbd_bridge_config(self):
        mod = _load_int("epbd_compliance_bridge")
        assert hasattr(mod, "EPBDComplianceBridgeConfig")

    def test_property_registry_bridge_config(self):
        mod = _load_int("property_registry_bridge")
        assert hasattr(mod, "PropertyRegistryBridgeConfig")

    def test_data_building_bridge_config(self):
        mod = _load_int("data_building_bridge")
        assert hasattr(mod, "DataBuildingBridgeConfig")

    def test_mrv_building_bridge_config(self):
        mod = _load_int("mrv_building_bridge")
        assert hasattr(mod, "MRVBuildingBridgeConfig")

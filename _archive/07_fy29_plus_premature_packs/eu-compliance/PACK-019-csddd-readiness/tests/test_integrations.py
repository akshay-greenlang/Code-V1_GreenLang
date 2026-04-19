# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Integration Tests
===================================================

Tests all 10 integration classes for instantiation, key methods,
orchestrator pipeline, and bridge mapping functionality.

Test count target: ~35 tests
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    INTEGRATION_CLASSES,
    INTEGRATION_FILES,
    INTEGRATIONS_DIR,
    _load_integration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTEGRATION_KEYS = list(INTEGRATION_FILES.keys())


# ---------------------------------------------------------------------------
# 1. File existence
# ---------------------------------------------------------------------------


class TestIntegrationFilesExist:
    """Verify all integration source files are present on disk."""

    @pytest.mark.parametrize("int_key", INTEGRATION_KEYS)
    def test_integration_file_exists(self, int_key: str):
        filepath = INTEGRATIONS_DIR / INTEGRATION_FILES[int_key]
        assert filepath.exists(), f"Missing integration file: {filepath}"


# ---------------------------------------------------------------------------
# 2. Module loading
# ---------------------------------------------------------------------------


class TestIntegrationModuleLoading:
    """Verify all integration modules load without import errors."""

    @pytest.mark.parametrize("int_key", INTEGRATION_KEYS)
    def test_integration_module_loads(self, int_key: str):
        mod = _load_integration(int_key)
        assert mod is not None


# ---------------------------------------------------------------------------
# 3. Class instantiation
# ---------------------------------------------------------------------------


class TestIntegrationInstantiation:
    """Verify all integration classes can be instantiated."""

    @pytest.mark.parametrize("int_key", INTEGRATION_KEYS)
    def test_integration_class_exists(self, int_key: str):
        mod = _load_integration(int_key)
        cls_name = INTEGRATION_CLASSES[int_key]
        cls = getattr(mod, cls_name, None)
        assert cls is not None, f"Class {cls_name} not found in {int_key}"

    @pytest.mark.parametrize("int_key", INTEGRATION_KEYS)
    def test_integration_instantiation_no_args(self, int_key: str):
        mod = _load_integration(int_key)
        cls = getattr(mod, INTEGRATION_CLASSES[int_key])
        instance = cls()
        assert instance is not None


# ---------------------------------------------------------------------------
# 4. CSDDDOrchestrator tests
# ---------------------------------------------------------------------------


class TestCSDDDOrchestrator:
    """Test the pack orchestrator that coordinates all engines."""

    def _get_orchestrator(self):
        mod = _load_integration("pack_orchestrator")
        cls = getattr(mod, "CSDDDOrchestrator")
        return cls()

    def test_orchestrator_has_run_full_assessment(self):
        orch = self._get_orchestrator()
        assert hasattr(orch, "run_full_assessment")

    def test_orchestrator_has_run_quick_assessment(self):
        orch = self._get_orchestrator()
        assert hasattr(orch, "run_quick_assessment")

    def test_orchestrator_has_get_status(self):
        orch = self._get_orchestrator()
        assert hasattr(orch, "get_status")

    def test_run_full_assessment_returns_result(self):
        mod = _load_integration("pack_orchestrator")
        orch = self._get_orchestrator()
        profile_cls = getattr(mod, "CompanyProfile")
        profile = profile_cls(
            company_name="TestCo AG",
            employee_count=6000,
            net_turnover_eur=2_000_000_000,
            sector="MANUFACTURING",
        )
        result = orch.run_full_assessment(profile=profile)
        assert result is not None

    def test_run_full_assessment_has_execution_id(self):
        mod = _load_integration("pack_orchestrator")
        orch = self._get_orchestrator()
        profile_cls = getattr(mod, "CompanyProfile")
        profile = profile_cls(
            company_name="TestCo AG",
            employee_count=6000,
            net_turnover_eur=2_000_000_000,
            sector="MANUFACTURING",
        )
        result = orch.run_full_assessment(profile=profile)
        assert hasattr(result, "execution_id") or "execution_id" in (
            result if isinstance(result, dict) else result.__dict__
        )

    def test_get_status_returns_dict(self):
        orch = self._get_orchestrator()
        status = orch.get_status()
        assert isinstance(status, dict)


# ---------------------------------------------------------------------------
# 5. CSRDPackBridge tests
# ---------------------------------------------------------------------------


class TestCSRDPackBridge:
    """Test the CSRD/ESRS to CSDDD bridge."""

    def _get_bridge(self):
        mod = _load_integration("csrd_pack_bridge")
        cls = getattr(mod, "CSRDPackBridge")
        return cls()

    def test_bridge_has_map_esrs_to_csddd(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "map_esrs_to_csddd")

    def test_bridge_has_get_s1_mapping(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "get_s1_mapping")

    def test_get_s1_mapping_returns_dict(self):
        bridge = self._get_bridge()
        mapping = bridge.get_s1_mapping()
        assert isinstance(mapping, dict)

    def test_bridge_has_identify_disclosure_gaps(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "identify_disclosure_gaps")


# ---------------------------------------------------------------------------
# 6. MRVBridge tests
# ---------------------------------------------------------------------------


class TestMRVBridge:
    """Test the MRV (emissions) to CSDDD bridge."""

    def _get_bridge(self):
        mod = _load_integration("mrv_bridge")
        cls = getattr(mod, "MRVBridge")
        return cls()

    def test_bridge_has_get_emission_data(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "get_emission_data")

    def test_bridge_has_validate_targets(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "validate_targets_against_mrv")

    def test_bridge_has_scope_methods(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "get_scope1_data")
        assert hasattr(bridge, "get_scope2_data")
        assert hasattr(bridge, "get_scope3_data")


# ---------------------------------------------------------------------------
# 7. EUDRBridge tests
# ---------------------------------------------------------------------------


class TestEUDRBridge:
    """Test the EUDR (deforestation) to CSDDD bridge."""

    def _get_bridge(self):
        mod = _load_integration("eudr_bridge")
        cls = getattr(mod, "EUDRBridge")
        return cls()

    def test_bridge_has_get_eudr_dd_status(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "get_eudr_dd_status")

    def test_bridge_has_map_eudr_to_csddd(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "map_eudr_to_csddd")

    def test_bridge_has_identify_deforestation_impacts(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "identify_deforestation_impacts")


# ---------------------------------------------------------------------------
# 8. SupplyChainBridge tests
# ---------------------------------------------------------------------------


class TestSupplyChainBridge:
    """Test the supply chain data bridge."""

    def _get_bridge(self):
        mod = _load_integration("supply_chain_bridge")
        cls = getattr(mod, "SupplyChainBridge")
        return cls()

    def test_bridge_has_get_supplier_data(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "get_supplier_data")

    def test_bridge_has_map_value_chain(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "map_value_chain")

    def test_bridge_has_assess_supplier_risk(self):
        bridge = self._get_bridge()
        assert hasattr(bridge, "assess_supplier_risk")


# ---------------------------------------------------------------------------
# 9. HealthCheck tests
# ---------------------------------------------------------------------------


class TestCSDDDHealthCheck:
    """Test the health check integration."""

    def _get_health_check(self):
        mod = _load_integration("health_check")
        cls = getattr(mod, "CSDDDHealthCheck")
        return cls()

    def test_health_check_has_run(self):
        hc = self._get_health_check()
        assert hasattr(hc, "run_health_check")

    def test_health_check_has_get_system_status(self):
        hc = self._get_health_check()
        assert hasattr(hc, "get_system_status")

    def test_run_health_check_returns_result(self):
        hc = self._get_health_check()
        result = hc.run_health_check()
        assert result is not None


# ---------------------------------------------------------------------------
# 10. SetupWizard tests
# ---------------------------------------------------------------------------


class TestCSDDDSetupWizard:
    """Test the setup wizard integration."""

    def _get_wizard(self):
        mod = _load_integration("setup_wizard")
        cls = getattr(mod, "CSDDDSetupWizard")
        return cls()

    def test_wizard_instantiates(self):
        wizard = self._get_wizard()
        assert wizard is not None

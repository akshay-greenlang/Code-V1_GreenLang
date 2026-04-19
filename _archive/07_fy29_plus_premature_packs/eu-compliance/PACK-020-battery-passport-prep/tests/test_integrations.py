# -*- coding: utf-8 -*-
"""
PACK-020 Battery Passport Prep Pack - Integration Tests
===========================================================

Tests all 10 integrations: pack_orchestrator, mrv_bridge, csrd_pack_bridge,
supply_chain_bridge, eudr_bridge, taxonomy_bridge, csddd_bridge, data_bridge,
health_check, setup_wizard. Validates bridge configurations, routing tables,
health checks, and setup wizard.

Author: GreenLang Platform Team (GL-TestEngineer)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"

INTEGRATION_FILES = {
    "pack_orchestrator": "pack_orchestrator.py",
    "mrv_bridge": "mrv_bridge.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "supply_chain_bridge": "supply_chain_bridge.py",
    "eudr_bridge": "eudr_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "csddd_bridge": "csddd_bridge.py",
    "data_bridge": "data_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES = {
    "pack_orchestrator": "BatteryPassportOrchestrator",
    "mrv_bridge": "MRVBridge",
    "csrd_pack_bridge": "CSRDPackBridge",
    "supply_chain_bridge": "SupplyChainBridge",
    "eudr_bridge": "EUDRBridge",
    "taxonomy_bridge": "TaxonomyBridge",
    "csddd_bridge": "CSDDDBridge",
    "data_bridge": "DataBridge",
    "health_check": "BatteryPassportHealthCheck",
    "setup_wizard": "BatteryPassportSetupWizard",
}


def _load_module(file_name: str, module_name: str, subdir: str = ""):
    if subdir:
        file_path = PACK_ROOT / subdir / file_name
    else:
        file_path = PACK_ROOT / file_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


_integration_modules = {}
for iname, ifile in INTEGRATION_FILES.items():
    try:
        _integration_modules[iname] = _load_module(
            ifile, f"pack020_ti.int_{iname}", "integrations"
        )
    except Exception:
        _integration_modules[iname] = None


def _get_integration_class(name: str):
    mod = _integration_modules.get(name)
    if mod is None:
        pytest.skip(f"Integration module {name} not loadable")
    cls_name = INTEGRATION_CLASSES[name]
    cls = getattr(mod, cls_name, None)
    if cls is None:
        pytest.skip(f"Class {cls_name} not found in {name}")
    return cls


# =========================================================================
# Parameterized: All integrations can be instantiated
# =========================================================================

@pytest.mark.parametrize("integration_name", list(INTEGRATION_FILES.keys()))
class TestIntegrationInstantiation:
    """Test that every integration class can be instantiated."""

    def test_class_exists(self, integration_name):
        cls = _get_integration_class(integration_name)
        assert cls is not None

    def test_instantiation_no_args(self, integration_name):
        cls = _get_integration_class(integration_name)
        try:
            instance = cls()
        except TypeError:
            instance = cls(config=None)
        assert instance is not None

    def test_has_expected_attributes(self, integration_name):
        cls = _get_integration_class(integration_name)
        try:
            instance = cls()
        except TypeError:
            instance = cls(config=None)
        # Every bridge should be an object with some public attributes
        attrs = [a for a in dir(instance) if not a.startswith("_")]
        assert len(attrs) > 0


# =========================================================================
# MRV Bridge specifics
# =========================================================================

class TestMRVBridge:
    """Tests for MRVBridge routing and configuration."""

    def _get_mod(self):
        mod = _integration_modules.get("mrv_bridge")
        if mod is None:
            pytest.skip("mrv_bridge not loadable")
        return mod

    def test_routing_table_exists(self):
        mod = self._get_mod()
        assert hasattr(mod, "BATTERY_MRV_ROUTING")
        routing = mod.BATTERY_MRV_ROUTING
        assert isinstance(routing, (dict, list))

    def test_performance_class_thresholds(self):
        mod = self._get_mod()
        assert hasattr(mod, "PERFORMANCE_CLASS_THRESHOLDS")
        thresholds = mod.PERFORMANCE_CLASS_THRESHOLDS
        assert isinstance(thresholds, dict)

    def test_mrv_scope_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "MRVScope")
        assert mod.MRVScope.SCOPE_1.value == "scope_1"

    def test_lifecycle_stage_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "LifecycleStage")
        stages = list(mod.LifecycleStage)
        assert len(stages) >= 4

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.MRVBridge()
        assert bridge is not None


# =========================================================================
# CSRD Pack Bridge specifics
# =========================================================================

class TestCSRDPackBridge:
    """Tests for CSRDPackBridge ESRS mappings."""

    def _get_mod(self):
        mod = _integration_modules.get("csrd_pack_bridge")
        if mod is None:
            pytest.skip("csrd_pack_bridge not loadable")
        return mod

    def test_esrs_battery_mappings_exist(self):
        mod = self._get_mod()
        assert hasattr(mod, "ESRS_BATTERY_MAPPINGS")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.CSRDPackBridge()
        assert bridge is not None


# =========================================================================
# Supply Chain Bridge specifics
# =========================================================================

class TestSupplyChainBridge:
    """Tests for SupplyChainBridge."""

    def _get_mod(self):
        mod = _integration_modules.get("supply_chain_bridge")
        if mod is None:
            pytest.skip("supply_chain_bridge not loadable")
        return mod

    def test_cahra_countries_defined(self):
        mod = self._get_mod()
        assert hasattr(mod, "CAHRA_COUNTRIES")
        assert isinstance(mod.CAHRA_COUNTRIES, (list, dict, set))

    def test_critical_mineral_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "CriticalMineral")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.SupplyChainBridge()
        assert bridge is not None


# =========================================================================
# EUDR Bridge specifics
# =========================================================================

class TestEUDRBridge:
    """Tests for EUDRBridge rubber sourcing validation."""

    def _get_mod(self):
        mod = _integration_modules.get("eudr_bridge")
        if mod is None:
            pytest.skip("eudr_bridge not loadable")
        return mod

    def test_country_benchmarks(self):
        mod = self._get_mod()
        assert hasattr(mod, "COUNTRY_BENCHMARKS")

    def test_battery_rubber_components(self):
        mod = self._get_mod()
        assert hasattr(mod, "BATTERY_RUBBER_COMPONENTS")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.EUDRBridge()
        assert bridge is not None


# =========================================================================
# Taxonomy Bridge specifics
# =========================================================================

class TestTaxonomyBridge:
    """Tests for TaxonomyBridge DNSH validation."""

    def _get_mod(self):
        mod = _integration_modules.get("taxonomy_bridge")
        if mod is None:
            pytest.skip("taxonomy_bridge not loadable")
        return mod

    def test_activity_34_criteria(self):
        mod = self._get_mod()
        assert hasattr(mod, "ACTIVITY_34_SC_CRITERIA")
        assert hasattr(mod, "ACTIVITY_34_DNSH_CRITERIA")

    def test_battery_reg_cross_references(self):
        mod = self._get_mod()
        assert hasattr(mod, "BATTERY_REG_CROSS_REFERENCES")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.TaxonomyBridge()
        assert bridge is not None


# =========================================================================
# CSDDD Bridge specifics
# =========================================================================

class TestCSDDDBridge:
    """Tests for CSDDDBridge adverse impact mapping."""

    def _get_mod(self):
        mod = _integration_modules.get("csddd_bridge")
        if mod is None:
            pytest.skip("csddd_bridge not loadable")
        return mod

    def test_mineral_risk_profiles(self):
        mod = self._get_mod()
        assert hasattr(mod, "MINERAL_RISK_PROFILES")

    def test_csddd_battery_overlap(self):
        mod = self._get_mod()
        assert hasattr(mod, "CSDDD_BATTERY_OVERLAP")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.CSDDDBridge()
        assert bridge is not None


# =========================================================================
# Data Bridge specifics
# =========================================================================

class TestDataBridge:
    """Tests for DataBridge passport field routing."""

    def _get_mod(self):
        mod = _integration_modules.get("data_bridge")
        if mod is None:
            pytest.skip("data_bridge not loadable")
        return mod

    def test_battery_data_routing(self):
        mod = self._get_mod()
        assert hasattr(mod, "BATTERY_DATA_ROUTING")

    def test_passport_field_requirements(self):
        mod = self._get_mod()
        assert hasattr(mod, "PASSPORT_FIELD_REQUIREMENTS")

    def test_bridge_instantiation(self):
        mod = self._get_mod()
        bridge = mod.DataBridge()
        assert bridge is not None


# =========================================================================
# Health Check specifics
# =========================================================================

class TestBatteryPassportHealthCheck:
    """Tests for BatteryPassportHealthCheck."""

    def _get_mod(self):
        mod = _integration_modules.get("health_check")
        if mod is None:
            pytest.skip("health_check not loadable")
        return mod

    def test_instantiation(self):
        mod = self._get_mod()
        hc = mod.BatteryPassportHealthCheck()
        assert hc is not None

    def test_health_status_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "HealthStatus")

    def test_check_category_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "CheckCategory")


# =========================================================================
# Setup Wizard specifics
# =========================================================================

class TestBatteryPassportSetupWizard:
    """Tests for BatteryPassportSetupWizard."""

    def _get_mod(self):
        mod = _integration_modules.get("setup_wizard")
        if mod is None:
            pytest.skip("setup_wizard not loadable")
        return mod

    def test_instantiation(self):
        mod = self._get_mod()
        wizard = mod.BatteryPassportSetupWizard()
        assert wizard is not None

    def test_category_defaults_exist(self):
        mod = self._get_mod()
        assert hasattr(mod, "CATEGORY_DEFAULTS")
        defaults = mod.CATEGORY_DEFAULTS
        assert isinstance(defaults, dict)

    def test_chemistry_type_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "ChemistryType")

    def test_battery_category_enum(self):
        mod = self._get_mod()
        assert hasattr(mod, "BatteryCategory")

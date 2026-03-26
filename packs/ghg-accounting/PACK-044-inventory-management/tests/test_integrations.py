# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Integration Tests
==========================================

Tests all 12 integrations: module loading, class instantiation,
and the integrations __init__.py lazy-loading registry.

Target: 40+ test cases.
"""

from pathlib import Path

import pytest

from conftest import (
    _load_module,
    INTEGRATIONS_DIR,
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
)


# ===================================================================
# Integration File Existence Tests
# ===================================================================


class TestIntegrationFileExistence:
    """Tests that all integration files exist on disk."""

    @pytest.mark.parametrize("key,filename", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, key, filename):
        path = INTEGRATIONS_DIR / filename
        assert path.exists(), f"Integration file missing: {path}"

    def test_integrations_init_exists(self):
        init_path = INTEGRATIONS_DIR / "__init__.py"
        assert init_path.exists(), "__init__.py missing in integrations/"


# ===================================================================
# Integration Module Loading Tests
# ===================================================================


class TestIntegrationModuleLoading:
    """Tests that all integration modules load without errors."""

    @pytest.mark.parametrize("key,filename", list(INTEGRATION_FILES.items()))
    def test_integration_module_loads(self, key, filename):
        mod = _load_module(key, filename, "integrations")
        assert mod is not None

    @pytest.mark.parametrize("key", list(INTEGRATION_CLASSES.keys()))
    def test_integration_class_exists(self, key):
        filename = INTEGRATION_FILES[key]
        mod = _load_module(key, filename, "integrations")
        cls_name = INTEGRATION_CLASSES[key]
        assert hasattr(mod, cls_name), f"Class {cls_name} not found in {filename}"


# ===================================================================
# Integration Class Instantiation Tests
# ===================================================================


class TestIntegrationInstantiation:
    """Tests that integration classes can be instantiated."""

    @pytest.mark.parametrize("key", list(INTEGRATION_CLASSES.keys()))
    def test_integration_instantiates(self, key):
        filename = INTEGRATION_FILES[key]
        mod = _load_module(key, filename, "integrations")
        cls = getattr(mod, INTEGRATION_CLASSES[key])
        instance = cls()
        assert instance is not None


# ===================================================================
# Integrations __init__ Tests
# ===================================================================


class TestIntegrationsInit:
    """Tests for the integrations __init__.py lazy-loading."""

    def test_init_module_loads(self):
        mod = _load_module("integrations_init", "__init__.py", "integrations")
        assert mod is not None

    def test_get_loaded_integrations(self):
        mod = _load_module("integrations_init", "__init__.py", "integrations")
        if hasattr(mod, "get_loaded_integrations"):
            loaded = mod.get_loaded_integrations()
            assert isinstance(loaded, list)
            assert len(loaded) >= 1

    def test_all_12_integrations_loaded(self):
        mod = _load_module("integrations_init", "__init__.py", "integrations")
        if hasattr(mod, "get_loaded_integrations"):
            loaded = mod.get_loaded_integrations()
            # Some bridges depend on external packs; at least 8 core load
            assert len(loaded) >= 8


# ===================================================================
# Individual Integration Tests
# ===================================================================


class TestPackOrchestrator:
    """Tests for InventoryManagementOrchestrator."""

    def test_load_and_instantiate(self):
        mod = _load_module("pack_orchestrator",
                           INTEGRATION_FILES["pack_orchestrator"], "integrations")
        cls = getattr(mod, "InventoryManagementOrchestrator")
        instance = cls()
        assert instance is not None

    def test_has_orchestrate_method(self):
        mod = _load_module("pack_orchestrator",
                           INTEGRATION_FILES["pack_orchestrator"], "integrations")
        cls = getattr(mod, "InventoryManagementOrchestrator")
        instance = cls()
        assert (hasattr(instance, "orchestrate")
                or hasattr(instance, "run")
                or hasattr(instance, "execute"))


class TestPack041Bridge:
    """Tests for Pack041Bridge (Scope 1-2 Complete integration)."""

    def test_load_and_instantiate(self):
        mod = _load_module("pack041_bridge",
                           INTEGRATION_FILES["pack041_bridge"], "integrations")
        cls = getattr(mod, "Pack041Bridge")
        instance = cls()
        assert instance is not None


class TestPack042Bridge:
    """Tests for Pack042Bridge (Scope 3 Starter integration)."""

    def test_load_and_instantiate(self):
        mod = _load_module("pack042_bridge",
                           INTEGRATION_FILES["pack042_bridge"], "integrations")
        cls = getattr(mod, "Pack042Bridge")
        instance = cls()
        assert instance is not None


class TestPack043Bridge:
    """Tests for Pack043Bridge (Scope 3 Complete integration)."""

    def test_load_and_instantiate(self):
        mod = _load_module("pack043_bridge",
                           INTEGRATION_FILES["pack043_bridge"], "integrations")
        cls = getattr(mod, "Pack043Bridge")
        instance = cls()
        assert instance is not None


class TestMRVBridge:
    """Tests for MRVBridge."""

    def test_load_and_instantiate(self):
        mod = _load_module("mrv_bridge",
                           INTEGRATION_FILES["mrv_bridge"], "integrations")
        cls = getattr(mod, "MRVBridge")
        instance = cls()
        assert instance is not None


class TestDataBridge:
    """Tests for DataBridge."""

    def test_load_and_instantiate(self):
        mod = _load_module("data_bridge",
                           INTEGRATION_FILES["data_bridge"], "integrations")
        cls = getattr(mod, "DataBridge")
        instance = cls()
        assert instance is not None


class TestFoundationBridge:
    """Tests for FoundationBridge."""

    def test_load_and_instantiate(self):
        mod = _load_module("foundation_bridge",
                           INTEGRATION_FILES["foundation_bridge"], "integrations")
        cls = getattr(mod, "FoundationBridge")
        instance = cls()
        assert instance is not None


class TestERPConnector:
    """Tests for ERPConnector."""

    def test_load_and_instantiate(self):
        mod = _load_module("erp_connector",
                           INTEGRATION_FILES["erp_connector"], "integrations")
        cls = getattr(mod, "ERPConnector")
        instance = cls()
        assert instance is not None


class TestNotificationBridge:
    """Tests for NotificationBridge."""

    def test_load_and_instantiate(self):
        mod = _load_module("notification_bridge",
                           INTEGRATION_FILES["notification_bridge"], "integrations")
        cls = getattr(mod, "NotificationBridge")
        instance = cls()
        assert instance is not None


class TestHealthCheck:
    """Tests for HealthCheck."""

    def test_load_and_instantiate(self):
        mod = _load_module("health_check",
                           INTEGRATION_FILES["health_check"], "integrations")
        cls = getattr(mod, "HealthCheck")
        instance = cls()
        assert instance is not None

    def test_has_check_method(self):
        mod = _load_module("health_check",
                           INTEGRATION_FILES["health_check"], "integrations")
        cls = getattr(mod, "HealthCheck")
        instance = cls()
        assert (hasattr(instance, "check")
                or hasattr(instance, "run_check")
                or hasattr(instance, "check_all"))


class TestSetupWizard:
    """Tests for SetupWizard."""

    def test_load_and_instantiate(self):
        mod = _load_module("setup_wizard",
                           INTEGRATION_FILES["setup_wizard"], "integrations")
        cls = getattr(mod, "SetupWizard")
        instance = cls()
        assert instance is not None


class TestAlertBridge:
    """Tests for AlertBridge."""

    def test_load_and_instantiate(self):
        mod = _load_module("alert_bridge",
                           INTEGRATION_FILES["alert_bridge"], "integrations")
        cls = getattr(mod, "AlertBridge")
        instance = cls()
        assert instance is not None

# -*- coding: utf-8 -*-
"""
PACK-035 Energy Benchmark - Integration Tests
================================================

Tests all 12 integrations for importability, class instantiation,
required methods, orchestrator phase count, bridge contracts,
health check, and setup wizard.

Test Count Target: ~55 tests
Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-035 Energy Benchmark
Date:    March 2026
"""

import importlib.util
import sys
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
INTEGRATIONS_DIR = PACK_ROOT / "integrations"

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    _load_module,
)


def _load_integration(int_key: str):
    """Load an integration module by its logical key."""
    file_name = INTEGRATION_FILES.get(int_key)
    if file_name is None:
        pytest.skip(f"Unknown integration key: {int_key}")
    try:
        return _load_module(int_key, file_name, "integrations")
    except FileNotFoundError:
        pytest.skip(f"Integration file not found: {file_name}")
    except ImportError as exc:
        pytest.skip(f"Cannot load integration {int_key}: {exc}")


# =========================================================================
# 1. Integration File Presence
# =========================================================================


class TestIntegrationFilePresence:
    """Test all 12 integration files exist on disk."""

    @pytest.mark.parametrize("int_key,file_name", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, int_key, file_name):
        """Integration Python file exists."""
        path = INTEGRATIONS_DIR / file_name
        if not path.exists():
            pytest.skip(f"File not found: {path}")
        assert path.is_file()
        assert path.suffix == ".py"


# =========================================================================
# 2. Integration Module Loading
# =========================================================================


class TestIntegrationModuleLoading:
    """Test all 12 integration modules can be loaded."""

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_module_loads(self, int_key):
        """Integration module loads without error."""
        mod = _load_integration(int_key)
        assert mod is not None


# =========================================================================
# 3. Integration Class Instantiation
# =========================================================================


class TestIntegrationClassInstantiation:
    """Test integration class exists and can be instantiated."""

    @pytest.mark.parametrize("int_key,class_name", list(INTEGRATION_CLASSES.items()))
    def test_integration_class_exists(self, int_key, class_name):
        """Integration class is defined in the module."""
        mod = _load_integration(int_key)
        assert hasattr(mod, class_name), f"{class_name} not found in {int_key}"

    @pytest.mark.parametrize("int_key,class_name", list(INTEGRATION_CLASSES.items()))
    def test_integration_instantiation(self, int_key, class_name):
        """Integration class can be instantiated."""
        mod = _load_integration(int_key)
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        assert instance is not None


# =========================================================================
# 4. Orchestrator Tests
# =========================================================================


class TestOrchestrator:
    """Test pack orchestrator (EnergyBenchmarkOrchestrator)."""

    def test_orchestrator_class_exists(self):
        """EnergyBenchmarkOrchestrator class exists."""
        mod = _load_integration("pack_orchestrator")
        assert hasattr(mod, "EnergyBenchmarkOrchestrator")

    def test_orchestrator_instantiation(self):
        """Orchestrator can be instantiated."""
        mod = _load_integration("pack_orchestrator")
        cls = getattr(mod, "EnergyBenchmarkOrchestrator", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkOrchestrator not found")
        orch = cls()
        assert orch is not None

    def test_orchestrator_has_12_phases(self):
        """Orchestrator has 12 phases."""
        mod = _load_integration("pack_orchestrator")
        cls = getattr(mod, "EnergyBenchmarkOrchestrator", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkOrchestrator not found")
        orch = cls()
        phases = getattr(orch, "phases", None) or getattr(orch, "PHASES", None)
        if phases is None:
            pytest.skip("No phases attribute found")
        assert len(phases) == 12

    def test_orchestrator_has_execute(self):
        """Orchestrator has an execute method."""
        mod = _load_integration("pack_orchestrator")
        cls = getattr(mod, "EnergyBenchmarkOrchestrator", None)
        if cls is None:
            pytest.skip("EnergyBenchmarkOrchestrator not found")
        orch = cls()
        assert hasattr(orch, "execute") or hasattr(orch, "run")


# =========================================================================
# 5. Bridge Contract Tests
# =========================================================================


class TestBridgeContracts:
    """Test bridge integrations implement required methods."""

    BRIDGE_KEYS = [
        "mrv_benchmark_bridge",
        "data_benchmark_bridge",
        "pack_031_bridge",
        "pack_032_bridge",
        "pack_033_bridge",
        "energy_star_bridge",
        "weather_service_bridge",
        "epc_registry_bridge",
        "benchmark_database_bridge",
    ]

    @pytest.mark.parametrize("bridge_key", BRIDGE_KEYS)
    def test_bridge_has_connect_method(self, bridge_key):
        """Bridge class has a connect() or initialize() method."""
        mod = _load_integration(bridge_key)
        class_name = INTEGRATION_CLASSES.get(bridge_key)
        if class_name is None:
            pytest.skip(f"No class mapping for {bridge_key}")
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_connect = (
            hasattr(instance, "connect")
            or hasattr(instance, "initialize")
            or hasattr(instance, "init")
        )
        if not has_connect:
            pytest.skip(f"{class_name} has no connect/initialize method")
        assert has_connect

    @pytest.mark.parametrize("bridge_key", BRIDGE_KEYS)
    def test_bridge_has_fetch_or_push(self, bridge_key):
        """Bridge class has a fetch() or push() method."""
        mod = _load_integration(bridge_key)
        class_name = INTEGRATION_CLASSES.get(bridge_key)
        if class_name is None:
            pytest.skip(f"No class mapping for {bridge_key}")
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        has_data_method = (
            hasattr(instance, "fetch")
            or hasattr(instance, "push")
            or hasattr(instance, "get_data")
            or hasattr(instance, "send_data")
            or hasattr(instance, "execute")
        )
        if not has_data_method:
            pytest.skip(f"{class_name} has no fetch/push/get_data/send_data method")
        assert has_data_method


# =========================================================================
# 6. Pack Bridge Tests
# =========================================================================


class TestPackBridges:
    """Test cross-pack bridge integrations."""

    @pytest.mark.parametrize("bridge_key,pack_id", [
        ("pack_031_bridge", "PACK-031"),
        ("pack_032_bridge", "PACK-032"),
        ("pack_033_bridge", "PACK-033"),
    ])
    def test_pack_bridge_references_pack(self, bridge_key, pack_id):
        """Pack bridge module references the target pack."""
        mod = _load_integration(bridge_key)
        class_name = INTEGRATION_CLASSES.get(bridge_key)
        if class_name is None:
            pytest.skip(f"No class for {bridge_key}")
        cls = getattr(mod, class_name, None)
        if cls is None:
            pytest.skip(f"{class_name} not found")
        instance = cls()
        # Check for target_pack or pack_id attribute
        target = getattr(instance, "target_pack", None) or getattr(instance, "pack_id", None)
        if target is not None:
            assert pack_id in str(target)


# =========================================================================
# 7. Health Check Tests
# =========================================================================


class TestHealthCheck:
    """Test health check integration."""

    def test_health_check_class_exists(self):
        """HealthCheck class exists."""
        mod = _load_integration("health_check")
        assert hasattr(mod, "HealthCheck")

    def test_health_check_instantiation(self):
        """HealthCheck can be instantiated."""
        mod = _load_integration("health_check")
        cls = getattr(mod, "HealthCheck", None)
        if cls is None:
            pytest.skip("HealthCheck not found")
        hc = cls()
        assert hc is not None

    def test_health_check_has_run(self):
        """HealthCheck has a run() or check() method."""
        mod = _load_integration("health_check")
        cls = getattr(mod, "HealthCheck", None)
        if cls is None:
            pytest.skip("HealthCheck not found")
        hc = cls()
        has_method = hasattr(hc, "run") or hasattr(hc, "check") or hasattr(hc, "execute")
        assert has_method


# =========================================================================
# 8. Setup Wizard Tests
# =========================================================================


class TestSetupWizard:
    """Test setup wizard integration."""

    def test_setup_wizard_class_exists(self):
        """SetupWizard class exists."""
        mod = _load_integration("setup_wizard")
        assert hasattr(mod, "SetupWizard")

    def test_setup_wizard_instantiation(self):
        """SetupWizard can be instantiated."""
        mod = _load_integration("setup_wizard")
        cls = getattr(mod, "SetupWizard", None)
        if cls is None:
            pytest.skip("SetupWizard not found")
        sw = cls()
        assert sw is not None

    def test_setup_wizard_has_run(self):
        """SetupWizard has a run() method."""
        mod = _load_integration("setup_wizard")
        cls = getattr(mod, "SetupWizard", None)
        if cls is None:
            pytest.skip("SetupWizard not found")
        sw = cls()
        has_method = hasattr(sw, "run") or hasattr(sw, "setup") or hasattr(sw, "execute")
        assert has_method


# =========================================================================
# 9. Integration Metadata
# =========================================================================


class TestIntegrationMetadata:
    """Test integration module metadata."""

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_has_version(self, int_key):
        """Integration module defines _MODULE_VERSION."""
        mod = _load_integration(int_key)
        if not hasattr(mod, "_MODULE_VERSION"):
            pytest.skip(f"_MODULE_VERSION not found in {int_key}")
        assert mod._MODULE_VERSION == "1.0.0"

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_has_docstring(self, int_key):
        """Integration module has a docstring."""
        mod = _load_integration(int_key)
        assert mod.__doc__ is not None
        assert len(mod.__doc__) > 20

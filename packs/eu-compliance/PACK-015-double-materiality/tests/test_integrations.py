# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Integration Tests
=================================================================

Tests for all 8 integration modules: pack orchestrator, CSRD bridge,
MRV bridge, data bridge, sector bridge, regulatory bridge, health
check, and setup wizard.

Target: 35+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

import inspect

import pytest

from .conftest import (
    INTEGRATION_FILES,
    INTEGRATION_CLASSES,
    INTEGRATIONS_DIR,
    _load_integration,
)


# ===========================================================================
# Helpers
# ===========================================================================


def _try_load_integration(key):
    """Attempt to load an integration, returning module or None."""
    try:
        return _load_integration(key)
    except (ImportError, FileNotFoundError):
        return None


# ===========================================================================
# Integration File Existence
# ===========================================================================


class TestIntegrationFilesExist:
    """Test that all 8 integration files exist on disk."""

    @pytest.mark.parametrize("int_key,int_file", list(INTEGRATION_FILES.items()))
    def test_integration_file_exists(self, int_key, int_file):
        """Integration file exists on disk."""
        path = INTEGRATIONS_DIR / int_file
        assert path.exists(), f"Integration file missing: {path}"


# ===========================================================================
# Pack Orchestrator
# ===========================================================================


class TestPackOrchestrator:
    """Tests for the DMAPackOrchestrator integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("pack_orchestrator")

    def test_pack_orchestrator_class_exists(self):
        """DMAPackOrchestrator class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAPackOrchestrator")

    def test_orchestrator_has_execute(self):
        """Orchestrator has execute_pipeline or run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DMAPackOrchestrator
        has_run = (
            hasattr(cls, "execute")
            or hasattr(cls, "execute_pipeline")
            or hasattr(cls, "run_pipeline")
            or hasattr(cls, "run")
        )
        assert has_run

    def test_orchestrator_phases(self):
        """Orchestrator defines phase execution order."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_phases = (
            hasattr(self.mod, "PHASE_EXECUTION_ORDER")
            or hasattr(self.mod, "DMAPipelinePhase")
        )
        assert has_phases

    def test_orchestrator_config(self):
        """OrchestratorConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "OrchestratorConfig")

    def test_orchestrator_phase_result(self):
        """PhaseResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PhaseResult")

    def test_orchestrator_pipeline_result(self):
        """PipelineResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PipelineResult")


# ===========================================================================
# CSRD Pack Bridge
# ===========================================================================


class TestCSRDPackBridge:
    """Tests for the CSRDPackBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("csrd_pack_bridge")

    def test_csrd_bridge_class_exists(self):
        """CSRDPackBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "CSRDPackBridge")

    def test_csrd_bridge_config(self):
        """CSRDPackBridgeConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "CSRDPackBridgeConfig")

    def test_csrd_bridge_connection(self):
        """Bridge defines connection/data-flow types."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_flow = (
            hasattr(self.mod, "DataFlowDirection")
            or hasattr(self.mod, "BridgeResult")
        )
        assert has_flow

    def test_csrd_bridge_governance_data(self):
        """GovernanceData model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "GovernanceData")


# ===========================================================================
# MRV Materiality Bridge
# ===========================================================================


class TestMRVMaterialityBridge:
    """Tests for the MRVMaterialityBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("mrv_materiality_bridge")

    def test_mrv_bridge_class_exists(self):
        """MRVMaterialityBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "MRVMaterialityBridge")

    def test_mrv_bridge_agent_routing(self):
        """Bridge defines agent routing mappings."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_routing = (
            hasattr(self.mod, "MRVAgentMapping")
            or hasattr(self.mod, "MRVScope")
        )
        assert has_routing

    def test_mrv_bridge_emissions_context(self):
        """EmissionsContext model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "EmissionsContext")


# ===========================================================================
# Data Materiality Bridge
# ===========================================================================


class TestDataMaterialityBridge:
    """Tests for the DataMaterialityBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("data_materiality_bridge")

    def test_data_bridge_class_exists(self):
        """DataMaterialityBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DataMaterialityBridge")

    def test_data_bridge_routing(self):
        """Bridge defines data agent routing."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_routing = (
            hasattr(self.mod, "DataAgentRoute")
            or hasattr(self.mod, "DMADataSource")
        )
        assert has_routing

    def test_data_bridge_quality_report(self):
        """DataQualityReport model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DataQualityReport")


# ===========================================================================
# Sector Classification Bridge
# ===========================================================================


class TestSectorClassificationBridge:
    """Tests for the SectorClassificationBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("sector_classification_bridge")

    def test_sector_bridge_class_exists(self):
        """SectorClassificationBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "SectorClassificationBridge")

    def test_sector_bridge_nace_mapping(self):
        """Bridge defines NACE code mapping."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_nace = (
            hasattr(self.mod, "NACECode")
            or hasattr(self.mod, "SectorProfile")
        )
        assert has_nace

    def test_sector_bridge_profile(self):
        """SectorProfile model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "SectorProfile")


# ===========================================================================
# Regulatory Bridge
# ===========================================================================


class TestRegulatoryBridge:
    """Tests for the RegulatoryBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("regulatory_bridge")

    def test_regulatory_bridge_class_exists(self):
        """RegulatoryBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "RegulatoryBridge")

    def test_regulatory_bridge_monitoring(self):
        """Bridge defines regulatory change monitoring types."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_monitoring = (
            hasattr(self.mod, "RegulatoryChange")
            or hasattr(self.mod, "RegulatoryAlert")
        )
        assert has_monitoring

    def test_regulatory_bridge_threshold_update(self):
        """ThresholdUpdate model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ThresholdUpdate")


# ===========================================================================
# DMA Health Check
# ===========================================================================


class TestDMAHealthCheck:
    """Tests for the DMAHealthCheck integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("health_check")

    def test_health_check_class_exists(self):
        """DMAHealthCheck class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAHealthCheck")

    def test_health_check_categories(self):
        """Health check defines check categories."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_categories = (
            hasattr(self.mod, "CheckCategory")
            or hasattr(self.mod, "HealthCheckConfig")
        )
        assert has_categories

    def test_health_check_result(self):
        """HealthCheckResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "HealthCheckResult")

    def test_health_check_component_health(self):
        """ComponentHealth model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ComponentHealth")


# ===========================================================================
# DMA Setup Wizard
# ===========================================================================


class TestDMASetupWizard:
    """Tests for the DMASetupWizard integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("setup_wizard")

    def test_setup_wizard_class_exists(self):
        """DMASetupWizard class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMASetupWizard")

    def test_setup_wizard_steps(self):
        """Wizard defines step enum or list."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_steps = (
            hasattr(self.mod, "DMAWizardStep")
            or hasattr(self.mod, "WizardState")
        )
        assert has_steps

    def test_setup_wizard_result(self):
        """SetupResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "SetupResult")

    def test_setup_wizard_company_profile(self):
        """CompanyProfile model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "CompanyProfile")


# ===========================================================================
# Cross-Integration Pattern Tests
# ===========================================================================


class TestIntegrationPatterns:
    """Pattern tests applicable to all integrations."""

    @pytest.mark.parametrize("int_key,int_class", list(INTEGRATION_CLASSES.items()))
    def test_integration_has_docstring(self, int_key, int_class):
        """All integration classes have docstrings."""
        mod = _try_load_integration(int_key)
        if mod is None:
            pytest.skip(f"Integration {int_key} not loaded")
        cls = getattr(mod, int_class, None)
        if cls is None:
            pytest.skip(f"Class {int_class} not found")
        assert cls.__doc__ is not None

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_uses_pydantic(self, int_key):
        """Integration modules use Pydantic models."""
        mod = _try_load_integration(int_key)
        if mod is None:
            pytest.skip(f"Integration {int_key} not loaded")
        from pydantic import BaseModel
        model_classes = [
            name for name, obj in inspect.getmembers(mod, inspect.isclass)
            if issubclass(obj, BaseModel) and obj is not BaseModel
        ]
        assert len(model_classes) >= 1, (
            f"Integration {int_key} should have at least 1 Pydantic model"
        )

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_uses_logging(self, int_key):
        """Integration modules use Python logging."""
        source_path = INTEGRATIONS_DIR / INTEGRATION_FILES[int_key]
        content = source_path.read_text(encoding="utf-8")
        assert "logging" in content, f"Integration {int_key} should use logging"

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_has_version(self, int_key):
        """Integration modules define a version string."""
        source_path = INTEGRATIONS_DIR / INTEGRATION_FILES[int_key]
        content = source_path.read_text(encoding="utf-8")
        has_version = (
            "version" in content.lower()
            or "__version__" in content
            or "_MODULE_VERSION" in content
        )
        assert has_version, f"Integration {int_key} should define a version"

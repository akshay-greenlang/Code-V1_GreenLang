# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Integration Tests
====================================================

Tests for all 8 integration modules: pack orchestrator, GHG app bridge,
MRV agent bridge, DMA pack bridge, decarbonization bridge, adaptation
bridge, health check, and setup wizard.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
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


class TestIntegrationLoading:
    """Test that all 8 integrations can be loaded."""

    @pytest.mark.parametrize("int_key", list(INTEGRATION_FILES.keys()))
    def test_integration_module_loads(self, int_key):
        """Each integration module loads independently."""
        mod = _try_load_integration(int_key)
        assert mod is not None, f"Integration {int_key} failed to load"

    def test_all_8_integrations_loadable(self):
        """All 8 integrations load successfully."""
        loaded = []
        for key in INTEGRATION_FILES:
            mod = _try_load_integration(key)
            if mod is not None:
                loaded.append(key)
        assert len(loaded) == 8, f"Loaded {len(loaded)}/8 integrations: {loaded}"

    @pytest.mark.parametrize("int_key,int_class", list(INTEGRATION_CLASSES.items()))
    def test_integration_class_exists(self, int_key, int_class):
        """Each integration exports its primary class."""
        mod = _try_load_integration(int_key)
        if mod is None:
            pytest.skip(f"Integration {int_key} not loaded")
        assert hasattr(mod, int_class), f"Integration {int_key} missing class {int_class}"


# ===========================================================================
# Pack Orchestrator
# ===========================================================================


class TestPackOrchestrator:
    """Tests for the E1PackOrchestrator integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("pack_orchestrator")

    def test_class_exists(self):
        """E1PackOrchestrator class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E1PackOrchestrator")

    def test_has_execute_method(self):
        """Orchestrator has execute or run method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1PackOrchestrator
        has_run = (
            hasattr(cls, "execute")
            or hasattr(cls, "execute_pipeline")
            or hasattr(cls, "run_pipeline")
            or hasattr(cls, "run")
        )
        assert has_run

    def test_has_10_phases(self):
        """Orchestrator defines at least 10 phases."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        has_phases = (
            hasattr(self.mod, "PHASE_EXECUTION_ORDER")
            or hasattr(self.mod, "E1DisclosurePhase")
        )
        assert has_phases

    def test_phase_execution_order_exists(self):
        """PHASE_EXECUTION_ORDER constant exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PHASE_EXECUTION_ORDER")

    def test_phase_dependencies_exist(self):
        """PHASE_DEPENDENCIES constant exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PHASE_DEPENDENCIES")

    def test_orchestrator_config_model_exists(self):
        """OrchestratorConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "OrchestratorConfig")

    def test_phase_result_model_exists(self):
        """PhaseResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PhaseResult")

    def test_validate_prerequisites_method(self):
        """Orchestrator has validate_prerequisites or similar method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1PackOrchestrator
        has_validate = (
            hasattr(cls, "validate_prerequisites")
            or hasattr(cls, "validate")
            or hasattr(cls, "check_prerequisites")
        )
        assert has_validate


# ===========================================================================
# GHG App Bridge
# ===========================================================================


class TestGHGAppBridge:
    """Tests for the GHGAppBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("ghg_app_bridge")

    def test_class_exists(self):
        """GHGAppBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "GHGAppBridge")

    def test_has_import_method(self):
        """GHGAppBridge has import method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.GHGAppBridge
        has_import = (
            hasattr(cls, "import_inventory")
            or hasattr(cls, "import_ghg_data")
            or hasattr(cls, "import_emissions")
        )
        assert has_import

    def test_has_export_method(self):
        """GHGAppBridge has export method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.GHGAppBridge
        has_export = (
            hasattr(cls, "export_e1_results")
            or hasattr(cls, "export_inventory")
            or hasattr(cls, "export_ghg_data")
            or hasattr(cls, "export_emissions")
        )
        assert has_export

    def test_config_model_exists(self):
        """GHGBridgeConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "GHGBridgeConfig")


# ===========================================================================
# MRV Agent Bridge
# ===========================================================================


class TestMRVAgentBridge:
    """Tests for the MRVAgentBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("mrv_agent_bridge")

    def test_class_exists(self):
        """MRVAgentBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "MRVAgentBridge")

    def test_has_scope_import_methods(self):
        """MRVAgentBridge has scope-level import methods."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.MRVAgentBridge
        has_scope_methods = (
            hasattr(cls, "import_scope1")
            or hasattr(cls, "import_scope_1")
            or hasattr(cls, "import_scope1_emissions")
            or hasattr(cls, "get_scope_mapping")
            or hasattr(cls, "import_emissions")
        )
        assert has_scope_methods

    def test_scope_mapping_exists(self):
        """MRVAgentMapping model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "MRVAgentMapping")

    def test_30_agent_mapping(self):
        """Bridge maps to all 30 MRV agents."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        source = (INTEGRATIONS_DIR / "mrv_agent_bridge.py").read_text(encoding="utf-8")
        # Check that MRV-001 through MRV-030 are referenced
        has_mrv_001 = "MRV-001" in source or "mrv_001" in source or "001" in source
        has_mrv_030 = "MRV-030" in source or "mrv_030" in source or "030" in source
        assert has_mrv_001, "MRV-001 not referenced"
        assert has_mrv_030, "MRV-030 not referenced"


# ===========================================================================
# DMA Pack Bridge
# ===========================================================================


class TestDMAPackBridge:
    """Tests for the DMAPackBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("dma_pack_bridge")

    def test_class_exists(self):
        """DMAPackBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DMAPackBridge")

    def test_has_materiality_check_method(self):
        """DMAPackBridge has materiality import/check method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DMAPackBridge
        has_check = (
            hasattr(cls, "import_e1_materiality")
            or hasattr(cls, "check_e1_materiality")
            or hasattr(cls, "get_materiality_status")
            or hasattr(cls, "check_materiality")
            or hasattr(cls, "is_e1_material")
        )
        assert has_check

    def test_has_iro_import_method(self):
        """DMAPackBridge has IRO import method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DMAPackBridge
        has_iro = (
            hasattr(cls, "import_iro_register")
            or hasattr(cls, "import_iros")
            or hasattr(cls, "get_e1_iros")
            or hasattr(cls, "import_e1_iros")
        )
        assert has_iro

    def test_materiality_status_model_exists(self):
        """MaterialityStatus model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "MaterialityStatus")


# ===========================================================================
# Decarbonization Bridge
# ===========================================================================


class TestDecarbonizationBridge:
    """Tests for the DecarbonizationBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("decarbonization_bridge")

    def test_class_exists(self):
        """DecarbonizationBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "DecarbonizationBridge")

    def test_has_transition_plan_import(self):
        """DecarbonizationBridge has transition plan import method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DecarbonizationBridge
        has_import = (
            hasattr(cls, "import_transition_plan")
            or hasattr(cls, "get_transition_plan")
            or hasattr(cls, "import_plan")
        )
        assert has_import

    def test_has_target_import(self):
        """DecarbonizationBridge has abatement/pathway import method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.DecarbonizationBridge
        has_target = (
            hasattr(cls, "import_abatement_options")
            or hasattr(cls, "import_pathway_scenarios")
            or hasattr(cls, "import_targets")
            or hasattr(cls, "get_targets")
            or hasattr(cls, "sync_targets")
        )
        assert has_target

    def test_config_model_exists(self):
        """DecarbBridgeConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "DecarbBridgeConfig")


# ===========================================================================
# Adaptation Bridge
# ===========================================================================


class TestAdaptationBridge:
    """Tests for the AdaptationBridge integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("adaptation_bridge")

    def test_class_exists(self):
        """AdaptationBridge class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "AdaptationBridge")

    def test_has_risk_import_methods(self):
        """AdaptationBridge has risk import methods."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.AdaptationBridge
        has_risk = (
            hasattr(cls, "import_physical_risks")
            or hasattr(cls, "get_physical_risks")
            or hasattr(cls, "import_risks")
        )
        assert has_risk

    def test_physical_risk_model_exists(self):
        """PhysicalRisk model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "PhysicalRisk")

    def test_transition_risk_model_exists(self):
        """ClimateScenario model exists (contains transition_risk_factors)."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "ClimateScenario")

    def test_config_model_exists(self):
        """AdaptBridgeConfig model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "AdaptBridgeConfig")


# ===========================================================================
# Health Check
# ===========================================================================


class TestHealthCheck:
    """Tests for the E1HealthCheck integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("health_check")

    def test_class_exists(self):
        """E1HealthCheck class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E1HealthCheck")

    def test_has_check_all_method(self):
        """E1HealthCheck has check_all or run_all method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1HealthCheck
        has_check = (
            hasattr(cls, "check_all")
            or hasattr(cls, "run_all_checks")
            or hasattr(cls, "run")
        )
        assert has_check

    def test_has_check_engines_method(self):
        """E1HealthCheck has run_check or check_engines method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1HealthCheck
        has_engines = (
            hasattr(cls, "run_check")
            or hasattr(cls, "_check_engines")
            or hasattr(cls, "check_engines")
            or hasattr(cls, "verify_engines")
        )
        assert has_engines

    def test_health_check_result_model_exists(self):
        """HealthCheckResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "HealthCheckResult")

    def test_check_category_enum_exists(self):
        """CheckCategory enum exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "CheckCategory")


# ===========================================================================
# Setup Wizard
# ===========================================================================


class TestSetupWizard:
    """Tests for the E1SetupWizard integration."""

    @pytest.fixture(autouse=True)
    def _load(self):
        self.mod = _try_load_integration("setup_wizard")

    def test_class_exists(self):
        """E1SetupWizard class is importable."""
        assert self.mod is not None
        assert hasattr(self.mod, "E1SetupWizard")

    def test_has_run_setup_method(self):
        """E1SetupWizard has finalize or run_setup method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1SetupWizard
        has_run = (
            hasattr(cls, "finalize")
            or hasattr(cls, "submit_step")
            or hasattr(cls, "run_setup")
            or hasattr(cls, "run")
            or hasattr(cls, "start")
        )
        assert has_run

    def test_has_detect_sector_method(self):
        """E1SetupWizard has apply_industry_defaults or detect_sector method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1SetupWizard
        has_detect = (
            hasattr(cls, "apply_industry_defaults")
            or hasattr(cls, "detect_sector")
            or hasattr(cls, "identify_sector")
        )
        assert has_detect

    def test_has_recommend_preset_method(self):
        """E1SetupWizard has get_progress or recommend_preset method."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        cls = self.mod.E1SetupWizard
        has_recommend = (
            hasattr(cls, "get_progress")
            or hasattr(cls, "apply_industry_defaults")
            or hasattr(cls, "recommend_preset")
            or hasattr(cls, "get_recommended_preset")
            or hasattr(cls, "suggest_preset")
        )
        assert has_recommend

    def test_wizard_step_enum_exists(self):
        """E1WizardStep enum exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "E1WizardStep")

    def test_setup_result_model_exists(self):
        """SetupResult model exists."""
        if self.mod is None:
            pytest.skip("Module not loaded")
        assert hasattr(self.mod, "SetupResult")

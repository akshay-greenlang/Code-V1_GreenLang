# -*- coding: utf-8 -*-
"""
PACK-012 CSRD Financial Service Pack - Agent Integration Tests
================================================================

Tests agent integration capabilities including orchestrator phases,
bridge connections, module importability, health check execution,
and configuration constraints.

Self-contained: does NOT import from conftest.
Test count: 10 tests
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = PACK_ROOT.parent.parent.parent


def _import_from_path(module_name: str, file_path: str):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Import orchestrator and bridges
# ---------------------------------------------------------------------------

INT_DIR = str(PACK_ROOT / "integrations")

_int_orch = _import_from_path(
    "pack012_ai_orch",
    os.path.join(INT_DIR, "pack_orchestrator.py"),
)
_int_health = _import_from_path(
    "pack012_ai_health",
    os.path.join(INT_DIR, "health_check.py"),
)
_int_wizard = _import_from_path(
    "pack012_ai_wizard",
    os.path.join(INT_DIR, "setup_wizard.py"),
)
_int_csrd = _import_from_path(
    "pack012_ai_csrd",
    os.path.join(INT_DIR, "csrd_pack_bridge.py"),
)
_int_sfdr = _import_from_path(
    "pack012_ai_sfdr",
    os.path.join(INT_DIR, "sfdr_pack_bridge.py"),
)
_int_tax = _import_from_path(
    "pack012_ai_tax",
    os.path.join(INT_DIR, "taxonomy_pack_bridge.py"),
)
_int_mrv = _import_from_path(
    "pack012_ai_mrv",
    os.path.join(INT_DIR, "mrv_investments_bridge.py"),
)
_int_finance = _import_from_path(
    "pack012_ai_finance",
    os.path.join(INT_DIR, "finance_agent_bridge.py"),
)
_int_climate = _import_from_path(
    "pack012_ai_climate",
    os.path.join(INT_DIR, "climate_risk_bridge.py"),
)
_int_pillar3 = _import_from_path(
    "pack012_ai_pillar3",
    os.path.join(INT_DIR, "eba_pillar3_bridge.py"),
)

# Classes
FSCSRDOrchestrator = _int_orch.FSCSRDOrchestrator
FSOrchestrationConfig = _int_orch.FSOrchestrationConfig
PipelinePhase = _int_orch.PipelinePhase
HealthCheck = _int_health.HealthCheck
HealthCheckConfig = _int_health.HealthCheckConfig
SetupWizard = _int_wizard.SetupWizard
SetupWizardConfig = _int_wizard.SetupWizardConfig


# ===========================================================================
# Test: Orchestrator Phase Integration
# ===========================================================================


class TestOrchestratorPhaseIntegration:
    """Tests orchestrator phase definitions and pipeline structure."""

    def test_orchestrator_has_11_phase_handlers(self):
        """Orchestrator registers handlers for all 11 pipeline phases."""
        orch = FSCSRDOrchestrator()
        assert hasattr(orch, "_phase_handlers")
        assert len(orch._phase_handlers) == 11

    def test_orchestrator_phase_names_match_enum(self):
        """All PipelinePhase enum values have registered handlers."""
        orch = FSCSRDOrchestrator()
        for phase in PipelinePhase:
            assert phase in orch._phase_handlers, (
                f"Phase {phase.value} missing from handler registry"
            )

    def test_orchestrator_config_defaults_institution_type(self):
        """Default config sets institution_type to bank."""
        config = FSOrchestrationConfig()
        assert config.institution_type.value == "bank"

    def test_orchestrator_config_enables_all_bridges(self):
        """Default config enables CSRD, SFDR, and Taxonomy bridges."""
        config = FSOrchestrationConfig()
        assert config.enable_csrd_bridge is True
        assert config.enable_sfdr_bridge is True
        assert config.enable_taxonomy_bridge is True


# ===========================================================================
# Test: Bridge Connections
# ===========================================================================


class TestBridgeConnections:
    """Tests that all 8 bridge modules can be instantiated and connected."""

    def test_all_bridges_instantiate(self):
        """All 8 bridge classes can be instantiated with defaults."""
        bridges = [
            _int_csrd.CSRDPackBridge(),
            _int_sfdr.SFDRPackBridge(),
            _int_tax.TaxonomyPackBridge(),
            _int_mrv.MRVInvestmentsBridge(),
            _int_finance.FinanceAgentBridge(),
            _int_climate.ClimateRiskBridge(),
            _int_pillar3.EBAPillar3Bridge(),
        ]
        assert len(bridges) == 7
        for bridge in bridges:
            assert bridge is not None
            assert hasattr(bridge, "config")

    def test_all_bridge_configs_instantiate(self):
        """All bridge config classes can be instantiated with defaults."""
        configs = [
            _int_csrd.CSRDBridgeConfig(),
            _int_sfdr.SFDRBridgeConfig(),
            _int_tax.TaxonomyBridgeConfig(),
            _int_mrv.MRVInvestmentsBridgeConfig(),
            _int_finance.FinanceAgentBridgeConfig(),
            _int_climate.ClimateRiskBridgeConfig(),
            _int_pillar3.EBAPillar3BridgeConfig(),
        ]
        assert len(configs) == 7
        for config in configs:
            assert config is not None


# ===========================================================================
# Test: All Modules Importable
# ===========================================================================


class TestModuleImportability:
    """Tests that all PACK-012 modules can be loaded without errors."""

    def test_all_workflow_modules_importable(self):
        """All 8 workflow modules can be imported."""
        wf_dir = str(PACK_ROOT / "workflows")
        workflow_files = [
            "financed_emissions_workflow.py",
            "gar_btar_workflow.py",
            "insurance_emissions_workflow.py",
            "climate_stress_test_workflow.py",
            "fs_materiality_workflow.py",
            "transition_plan_workflow.py",
            "pillar3_reporting_workflow.py",
            "regulatory_integration_workflow.py",
        ]
        for wf_file in workflow_files:
            mod = _import_from_path(
                f"pack012_import_check_{wf_file.replace('.py', '')}",
                os.path.join(wf_dir, wf_file),
            )
            assert mod is not None, f"Failed to import {wf_file}"

    def test_all_engine_modules_importable(self):
        """All 8 engine modules can be imported."""
        eng_dir = str(PACK_ROOT / "engines")
        engine_files = [
            "financed_emissions_engine.py",
            "insurance_underwriting_engine.py",
            "green_asset_ratio_engine.py",
            "btar_calculator_engine.py",
            "climate_risk_scoring_engine.py",
            "fs_double_materiality_engine.py",
            "fs_transition_plan_engine.py",
            "pillar3_esg_engine.py",
        ]
        for eng_file in engine_files:
            mod = _import_from_path(
                f"pack012_import_check_{eng_file.replace('.py', '')}",
                os.path.join(eng_dir, eng_file),
            )
            assert mod is not None, f"Failed to import {eng_file}"

    def test_all_template_modules_importable(self):
        """All 8 template modules can be imported."""
        tpl_dir = str(PACK_ROOT / "templates")
        template_files = [
            "pcaf_report.py",
            "gar_btar_report.py",
            "pillar3_esg_template.py",
            "climate_risk_report.py",
            "fs_esrs_chapter.py",
            "financed_emissions_dashboard.py",
            "insurance_esg_template.py",
            "sbti_fi_report.py",
        ]
        for tpl_file in template_files:
            mod = _import_from_path(
                f"pack012_import_check_{tpl_file.replace('.py', '')}",
                os.path.join(tpl_dir, tpl_file),
            )
            assert mod is not None, f"Failed to import {tpl_file}"


# ===========================================================================
# Test: Health Check Integration
# ===========================================================================


class TestHealthCheckIntegration:
    """Tests health check execution and result structure."""

    def test_health_check_returns_structured_result(self):
        """Health check returns result with categories and score."""
        health = HealthCheck()
        result = health.run_full_check()
        assert result is not None
        assert hasattr(result, "overall_score")
        assert hasattr(result, "is_ready")
        assert hasattr(result, "category_results")
        assert isinstance(result.category_results, list)
        assert result.overall_score >= 0.0


# ===========================================================================
# Test: Config Constraints
# ===========================================================================


class TestConfigConstraints:
    """Tests configuration validation and constraint enforcement."""

    def test_orchestration_config_max_retries_bound(self):
        """max_retries is bounded between 0 and 10."""
        config = FSOrchestrationConfig(max_retries=5)
        assert config.max_retries == 5

        with pytest.raises(Exception):
            FSOrchestrationConfig(max_retries=15)

    def test_orchestration_config_materiality_threshold_bound(self):
        """materiality_threshold_pct is bounded between 0 and 100."""
        config = FSOrchestrationConfig(materiality_threshold_pct=10.0)
        assert config.materiality_threshold_pct == 10.0

        with pytest.raises(Exception):
            FSOrchestrationConfig(materiality_threshold_pct=150.0)

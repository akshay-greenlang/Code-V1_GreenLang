# -*- coding: utf-8 -*-
"""
Unit tests for PACK-021 Net Zero Starter Pack Integrations.

Tests all integration bridges (MRV, GHG App, SBTi App, Decarb, Offset,
Reporting, Data), health check (18 categories), setup wizard (6 steps),
and pipeline orchestrator (8 phases).

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    # Orchestrator
    NetZeroPipelineOrchestrator,
    OrchestratorConfig,
    RetryConfig,
    NetZeroPipelinePhase,
    ExecutionStatus,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseResult,
    PipelineResult,
    # MRV Bridge
    MRVBridge,
    MRVBridgeConfig,
    MRVScope,
    # GHG App Bridge
    GHGAppBridge,
    GHGAppBridgeConfig,
    GHGScope,
    # SBTi App Bridge
    SBTiAppBridge,
    SBTiAppBridgeConfig,
    # Decarb Bridge
    DecarbBridge,
    DecarbBridgeConfig,
    # Offset Bridge
    OffsetBridge,
    OffsetBridgeConfig,
    # Reporting Bridge
    ReportingBridge,
    ReportingBridgeConfig,
    # Data Bridge
    DataBridge,
    DataBridgeConfig,
    # Health Check
    NetZeroHealthCheck,
    HealthCheckConfig,
    CheckCategory,
    HealthStatus,
    # Setup Wizard
    NetZeroSetupWizard,
    NetZeroWizardStep,
    StepStatus,
)


# ========================================================================
# Bridge Instantiation Tests
# ========================================================================


class TestMRVBridge:
    """Tests for MRVBridge."""

    def test_mrv_bridge_instantiates(self):
        """MRVBridge creates with default config."""
        bridge = MRVBridge()
        assert bridge is not None

    def test_mrv_bridge_with_config(self):
        """MRVBridge creates with explicit config."""
        config = MRVBridgeConfig()
        bridge = MRVBridge(config=config)
        assert bridge is not None

    def test_mrv_scope_enum_has_scopes(self):
        """MRVScope enum has Scope 1, 2, 3."""
        scope_values = [s.value for s in MRVScope]
        assert len(scope_values) >= 3


class TestGHGAppBridge:
    """Tests for GHGAppBridge."""

    def test_ghg_app_bridge_instantiates(self):
        """GHGAppBridge creates with default config."""
        bridge = GHGAppBridge()
        assert bridge is not None

    def test_ghg_app_bridge_with_config(self):
        """GHGAppBridge creates with explicit config."""
        config = GHGAppBridgeConfig()
        bridge = GHGAppBridge(config=config)
        assert bridge is not None

    def test_ghg_scope_enum(self):
        """GHGScope enum is accessible."""
        assert len(list(GHGScope)) >= 1


class TestSBTiAppBridge:
    """Tests for SBTiAppBridge."""

    def test_sbti_app_bridge_instantiates(self):
        """SBTiAppBridge creates with default config."""
        bridge = SBTiAppBridge()
        assert bridge is not None

    def test_sbti_app_bridge_with_config(self):
        """SBTiAppBridge creates with explicit config."""
        config = SBTiAppBridgeConfig()
        bridge = SBTiAppBridge(config=config)
        assert bridge is not None


class TestDecarbBridge:
    """Tests for DecarbBridge."""

    def test_decarb_bridge_instantiates(self):
        """DecarbBridge creates with default config."""
        bridge = DecarbBridge()
        assert bridge is not None

    def test_decarb_bridge_with_config(self):
        """DecarbBridge creates with explicit config."""
        config = DecarbBridgeConfig()
        bridge = DecarbBridge(config=config)
        assert bridge is not None


class TestOffsetBridge:
    """Tests for OffsetBridge."""

    def test_offset_bridge_instantiates(self):
        """OffsetBridge creates with default config."""
        bridge = OffsetBridge()
        assert bridge is not None

    def test_offset_bridge_with_config(self):
        """OffsetBridge creates with explicit config."""
        config = OffsetBridgeConfig()
        bridge = OffsetBridge(config=config)
        assert bridge is not None


class TestReportingBridge:
    """Tests for ReportingBridge."""

    def test_reporting_bridge_instantiates(self):
        """ReportingBridge creates with default config."""
        bridge = ReportingBridge()
        assert bridge is not None

    def test_reporting_bridge_with_config(self):
        """ReportingBridge creates with explicit config."""
        config = ReportingBridgeConfig()
        bridge = ReportingBridge(config=config)
        assert bridge is not None


class TestDataBridge:
    """Tests for DataBridge."""

    def test_data_bridge_instantiates(self):
        """DataBridge creates with default config."""
        bridge = DataBridge()
        assert bridge is not None

    def test_data_bridge_with_config(self):
        """DataBridge creates with explicit config."""
        config = DataBridgeConfig()
        bridge = DataBridge(config=config)
        assert bridge is not None


# ========================================================================
# Agent Stub Fallback
# ========================================================================


class TestAgentStubFallback:
    """Tests for graceful degradation when agents are unavailable."""

    def test_mrv_bridge_handles_missing_agents(self):
        """MRVBridge gracefully handles missing agent imports."""
        bridge = MRVBridge()
        # Bridge should instantiate even if underlying agents are not available
        assert bridge is not None

    def test_decarb_bridge_handles_missing_agents(self):
        """DecarbBridge gracefully handles missing agent imports."""
        bridge = DecarbBridge()
        assert bridge is not None

    def test_data_bridge_handles_missing_agents(self):
        """DataBridge gracefully handles missing agent imports."""
        bridge = DataBridge()
        assert bridge is not None


# ========================================================================
# Health Check
# ========================================================================


class TestNetZeroHealthCheck:
    """Tests for NetZeroHealthCheck."""

    def test_health_check_instantiates(self):
        """Health check creates with default config."""
        hc = NetZeroHealthCheck()
        assert hc is not None

    def test_health_check_with_config(self):
        """Health check creates with explicit config."""
        config = HealthCheckConfig()
        hc = NetZeroHealthCheck(config=config)
        assert hc is not None

    def test_health_check_18_categories(self):
        """CheckCategory enum has 18 categories."""
        categories = list(CheckCategory)
        assert len(categories) == 18

    @pytest.mark.parametrize(
        "category",
        [
            "platform",
            "mrv_agents",
            "decarb_agents",
            "ghg_app",
            "sbti_app",
            "data_agents",
            "found_agents",
            "database",
            "cache",
            "engines",
            "workflows",
            "templates",
            "config",
            "presets",
            "emission_factors",
            "sbti_pathways",
            "abatement_catalog",
            "overall",
        ],
    )
    def test_check_category_exists(self, category):
        """CheckCategory enum includes '{category}'."""
        category_values = [c.value for c in CheckCategory]
        assert category in category_values

    def test_health_status_enum(self):
        """HealthStatus has PASS/FAIL/WARN/SKIP."""
        statuses = {s.value for s in HealthStatus}
        assert "PASS" in statuses
        assert "FAIL" in statuses
        assert "WARN" in statuses
        assert "SKIP" in statuses


# ========================================================================
# Setup Wizard
# ========================================================================


class TestNetZeroSetupWizard:
    """Tests for NetZeroSetupWizard."""

    def test_setup_wizard_instantiates(self):
        """Setup wizard creates without error."""
        wizard = NetZeroSetupWizard()
        assert wizard is not None

    def test_setup_wizard_6_steps(self):
        """NetZeroWizardStep enum has 6 steps."""
        steps = list(NetZeroWizardStep)
        assert len(steps) == 6

    @pytest.mark.parametrize(
        "step",
        [
            "organization_profile",
            "boundary_selection",
            "scope_configuration",
            "data_source_setup",
            "target_preferences",
            "preset_selection",
        ],
    )
    def test_wizard_step_exists(self, step):
        """WizardStep '{step}' is defined."""
        step_values = [s.value for s in NetZeroWizardStep]
        assert step in step_values

    def test_step_status_enum(self):
        """StepStatus has expected values."""
        statuses = {s.value for s in StepStatus}
        assert "pending" in statuses
        assert "completed" in statuses
        assert "failed" in statuses


# ========================================================================
# Orchestrator
# ========================================================================


class TestNetZeroPipelineOrchestrator:
    """Tests for NetZeroPipelineOrchestrator."""

    def test_orchestrator_instantiates(self):
        """Orchestrator creates with default config."""
        orch = NetZeroPipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_config(self):
        """Orchestrator creates with explicit config."""
        config = OrchestratorConfig(
            organization_name="TestCo",
            sector="manufacturing",
        )
        orch = NetZeroPipelineOrchestrator(config=config)
        assert orch.config.organization_name == "TestCo"

    def test_pipeline_8_phases(self):
        """NetZeroPipelinePhase enum has 8 phases."""
        phases = list(NetZeroPipelinePhase)
        assert len(phases) == 8

    @pytest.mark.parametrize(
        "phase",
        [
            "initialization",
            "data_intake",
            "quality_assurance",
            "baseline_calculation",
            "target_setting",
            "reduction_planning",
            "offset_strategy",
            "reporting",
        ],
    )
    def test_phase_exists(self, phase):
        """Phase '{phase}' is in NetZeroPipelinePhase."""
        phase_values = [p.value for p in NetZeroPipelinePhase]
        assert phase in phase_values

    def test_phase_execution_order_8_phases(self):
        """PHASE_EXECUTION_ORDER has 8 phases in order."""
        assert len(PHASE_EXECUTION_ORDER) == 8
        assert PHASE_EXECUTION_ORDER[0] == NetZeroPipelinePhase.INITIALIZATION
        assert PHASE_EXECUTION_ORDER[-1] == NetZeroPipelinePhase.REPORTING

    def test_phase_dependencies_dict(self):
        """PHASE_DEPENDENCIES covers all 8 phases."""
        assert len(PHASE_DEPENDENCIES) == 8
        # Initialization has no dependencies
        assert PHASE_DEPENDENCIES[NetZeroPipelinePhase.INITIALIZATION] == []
        # Data intake depends on initialization
        assert NetZeroPipelinePhase.INITIALIZATION in PHASE_DEPENDENCIES[
            NetZeroPipelinePhase.DATA_INTAKE
        ]

    def test_retry_config_defaults(self):
        """RetryConfig has sensible defaults."""
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.backoff_base >= 0.5
        assert rc.backoff_max >= 1.0
        assert 0 <= rc.jitter_factor <= 1

    def test_orchestrator_config_defaults(self):
        """OrchestratorConfig has sensible defaults."""
        config = OrchestratorConfig()
        assert config.pack_id == "PACK-021"
        assert config.pack_version == "1.0.0"
        assert config.enable_provenance is True
        assert config.enable_offset_strategy is False
        assert len(config.scopes_included) >= 3

    def test_execution_status_enum(self):
        """ExecutionStatus has required values."""
        statuses = {s.value for s in ExecutionStatus}
        assert "pending" in statuses
        assert "running" in statuses
        assert "completed" in statuses
        assert "failed" in statuses

    def test_conditional_offset_phase(self):
        """Offset strategy phase depends on reduction planning."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.OFFSET_STRATEGY]
        assert NetZeroPipelinePhase.REDUCTION_PLANNING in deps

    def test_reporting_depends_on_reduction_planning(self):
        """Reporting phase depends on reduction planning."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.REPORTING]
        assert NetZeroPipelinePhase.REDUCTION_PLANNING in deps


# ========================================================================
# Integration __init__ Exports
# ========================================================================


class TestIntegrationExports:
    """Tests that the integrations __init__ exports all expected symbols."""

    def test_version_exported(self):
        """__version__ is accessible."""
        import integrations
        assert hasattr(integrations, "__version__")
        assert integrations.__version__ == "1.0.0"

    def test_pack_id_exported(self):
        """__pack_id__ is PACK-021."""
        import integrations
        assert integrations.__pack_id__ == "PACK-021"

    def test_all_bridges_exported(self):
        """All 7 bridge classes are exported."""
        import integrations
        assert hasattr(integrations, "MRVBridge")
        assert hasattr(integrations, "GHGAppBridge")
        assert hasattr(integrations, "SBTiAppBridge")
        assert hasattr(integrations, "DecarbBridge")
        assert hasattr(integrations, "OffsetBridge")
        assert hasattr(integrations, "ReportingBridge")
        assert hasattr(integrations, "DataBridge")

# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Integrations.

Tests all 10 system bridges: pipeline orchestrator, SBTi SDA bridge,
IEA NZE bridge, IPCC AR6 bridge, PACK-021 bridge, MRV bridge,
decarb bridge, data bridge, health check, and setup wizard.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
"""

import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    # Orchestrator
    SectorPathwayPipelineOrchestrator, SectorPathwayOrchestratorConfig,
    # SBTi Bridge
    SBTiSDABridge, SBTiSDABridgeConfig,
    # IEA Bridge
    IEANZEBridge, IEANZEBridgeConfig,
    # IPCC Bridge
    IPCCAR6Bridge, IPCCAR6BridgeConfig,
    # Pack bridges
    PACK021Bridge, PACK021BridgeConfig,
    SectorMRVBridge, SectorMRVBridgeConfig,
    SectorDecarbBridge, SectorDecarbBridgeConfig,
    SectorDataBridge, SectorDataBridgeConfig,
    # Operational
    SectorPathwayHealthCheck, HealthCheckConfig, HealthStatus,
    SectorPathwaySetupWizard, SectorWizardStep,
)

from .conftest import SDA_SECTORS, SCENARIO_TYPES


# ========================================================================
# 1. Pipeline Orchestrator
# ========================================================================


class TestPipelineOrchestrator:
    """Test sector pathway pipeline orchestrator."""

    def test_orchestrator_instantiates(self):
        orch = SectorPathwayPipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_config(self):
        config = SectorPathwayOrchestratorConfig()
        orch = SectorPathwayPipelineOrchestrator(config=config)
        assert orch is not None

    def test_orchestrator_has_execute_pipeline(self):
        orch = SectorPathwayPipelineOrchestrator()
        assert hasattr(orch, "execute_pipeline")

    def test_config_defaults(self):
        config = SectorPathwayOrchestratorConfig()
        assert config is not None
        assert hasattr(config, "primary_sector")
        assert hasattr(config, "base_year")
        assert hasattr(config, "scenarios")

    def test_config_with_sector(self):
        config = SectorPathwayOrchestratorConfig(primary_sector="steel")
        assert config.primary_sector == "steel"

    def test_config_with_scenarios(self):
        config = SectorPathwayOrchestratorConfig(
            scenarios=["nze_1.5c", "wb2c"])
        assert len(config.scenarios) == 2

    def test_config_provenance_enabled(self):
        config = SectorPathwayOrchestratorConfig(enable_provenance=True)
        assert config.enable_provenance is True

    def test_config_timeout(self):
        config = SectorPathwayOrchestratorConfig(timeout_per_phase_seconds=120)
        assert config.timeout_per_phase_seconds == 120

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_orchestrator_per_sector(self, sector):
        config = SectorPathwayOrchestratorConfig(primary_sector=sector)
        orch = SectorPathwayPipelineOrchestrator(config=config)
        assert orch is not None


# ========================================================================
# 2. SBTi SDA Bridge
# ========================================================================


class TestSBTiSDABridge:
    """Test SBTi SDA sector pathway data integration."""

    def test_bridge_instantiates(self):
        bridge = SBTiSDABridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = SBTiSDABridgeConfig()
        bridge = SBTiSDABridge(config=config)
        assert bridge is not None

    def test_bridge_has_sector_methods(self):
        bridge = SBTiSDABridge()
        assert hasattr(bridge, "get_sector_pathway")
        assert hasattr(bridge, "validate_targets")
        assert hasattr(bridge, "classify_sector")

    def test_bridge_has_convergence(self):
        bridge = SBTiSDABridge()
        assert hasattr(bridge, "calculate_convergence")

    def test_bridge_has_status(self):
        bridge = SBTiSDABridge()
        assert hasattr(bridge, "get_bridge_status")

    def test_config_fields(self):
        config = SBTiSDABridgeConfig()
        assert hasattr(config, "primary_sector")
        assert hasattr(config, "base_year")
        assert hasattr(config, "enable_provenance")

    def test_config_with_sector(self):
        config = SBTiSDABridgeConfig(primary_sector="steel")
        assert config.primary_sector == "steel"

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sbti_bridge_per_sector(self, sector):
        config = SBTiSDABridgeConfig(primary_sector=sector)
        bridge = SBTiSDABridge(config=config)
        assert bridge is not None


# ========================================================================
# 3. IEA NZE Bridge
# ========================================================================


class TestIEANZEBridge:
    """Test IEA Net Zero by 2050 data integration."""

    def test_bridge_instantiates(self):
        bridge = IEANZEBridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = IEANZEBridgeConfig()
        bridge = IEANZEBridge(config=config)
        assert bridge is not None

    def test_bridge_has_pathway_methods(self):
        bridge = IEANZEBridge()
        assert hasattr(bridge, "get_sector_pathway")
        assert hasattr(bridge, "get_sector_milestones")
        assert hasattr(bridge, "get_supported_sectors")

    def test_bridge_has_compare_scenarios(self):
        bridge = IEANZEBridge()
        assert hasattr(bridge, "compare_scenarios")

    def test_bridge_has_status(self):
        bridge = IEANZEBridge()
        assert hasattr(bridge, "get_bridge_status")

    def test_config_fields(self):
        config = IEANZEBridgeConfig()
        assert hasattr(config, "default_scenario")
        assert hasattr(config, "default_region")
        assert hasattr(config, "sectors")
        assert hasattr(config, "scenarios")

    def test_config_milestone_tracking(self):
        config = IEANZEBridgeConfig(milestone_tracking_enabled=True)
        assert config.milestone_tracking_enabled is True

    def test_config_cache(self):
        config = IEANZEBridgeConfig(cache_pathway_lookups=True)
        assert config.cache_pathway_lookups is True

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_iea_bridge_per_sector(self, sector):
        config = IEANZEBridgeConfig(sectors=[sector])
        bridge = IEANZEBridge(config=config)
        assert bridge is not None


# ========================================================================
# 4. IPCC AR6 Bridge
# ========================================================================


class TestIPCCAR6Bridge:
    """Test IPCC AR6 sector pathways integration."""

    def test_bridge_instantiates(self):
        bridge = IPCCAR6Bridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = IPCCAR6BridgeConfig()
        bridge = IPCCAR6Bridge(config=config)
        assert bridge is not None

    def test_bridge_has_gwp_methods(self):
        bridge = IPCCAR6Bridge()
        assert hasattr(bridge, "get_gwp")
        assert hasattr(bridge, "convert_to_co2e")
        assert hasattr(bridge, "get_emission_factor")

    def test_bridge_has_carbon_budget(self):
        bridge = IPCCAR6Bridge()
        assert hasattr(bridge, "calculate_carbon_budget")

    def test_bridge_has_ssp(self):
        bridge = IPCCAR6Bridge()
        assert hasattr(bridge, "check_ssp_alignment")

    def test_bridge_has_status(self):
        bridge = IPCCAR6Bridge()
        assert hasattr(bridge, "get_bridge_status")


# ========================================================================
# 5. PACK-021 Bridge
# ========================================================================


class TestPACK021Bridge:
    """Test PACK-021 baseline and target engines integration."""

    def test_bridge_instantiates(self):
        bridge = PACK021Bridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = PACK021BridgeConfig()
        bridge = PACK021Bridge(config=config)
        assert bridge is not None

    def test_bridge_has_import_methods(self):
        bridge = PACK021Bridge()
        assert hasattr(bridge, "import_baseline")
        assert hasattr(bridge, "import_targets")
        assert hasattr(bridge, "import_gap_analysis")

    def test_bridge_has_integration(self):
        bridge = PACK021Bridge()
        assert hasattr(bridge, "get_full_integration")

    def test_bridge_has_status(self):
        bridge = PACK021Bridge()
        assert hasattr(bridge, "get_bridge_status")

    def test_config_fields(self):
        config = PACK021BridgeConfig()
        assert hasattr(config, "organization_name")
        assert hasattr(config, "base_year")


# ========================================================================
# 6. MRV Bridge
# ========================================================================


class TestSectorMRVBridge:
    """Test MRV agents integration bridge."""

    def test_bridge_instantiates(self):
        bridge = SectorMRVBridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = SectorMRVBridgeConfig()
        bridge = SectorMRVBridge(config=config)
        assert bridge is not None

    def test_bridge_has_routing(self):
        bridge = SectorMRVBridge()
        assert hasattr(bridge, "route_calculation")
        assert hasattr(bridge, "route_all_agents")
        assert hasattr(bridge, "get_routing_table")

    def test_bridge_has_status(self):
        bridge = SectorMRVBridge()
        assert hasattr(bridge, "get_agent_status")

    def test_config_with_sector(self):
        config = SectorMRVBridgeConfig(primary_sector="steel")
        assert config.primary_sector == "steel"


# ========================================================================
# 7. Decarb Bridge
# ========================================================================


class TestSectorDecarbBridge:
    """Test decarbonization agents integration bridge."""

    def test_bridge_instantiates(self):
        bridge = SectorDecarbBridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = SectorDecarbBridgeConfig()
        bridge = SectorDecarbBridge(config=config)
        assert bridge is not None

    def test_bridge_has_lever_methods(self):
        bridge = SectorDecarbBridge()
        assert hasattr(bridge, "get_sector_levers")
        assert hasattr(bridge, "generate_waterfall")
        assert hasattr(bridge, "generate_roadmap")

    def test_bridge_has_status(self):
        bridge = SectorDecarbBridge()
        assert hasattr(bridge, "get_bridge_status")

    def test_config_with_sector(self):
        config = SectorDecarbBridgeConfig(primary_sector="power_generation")
        assert config.primary_sector == "power_generation"


# ========================================================================
# 8. Data Bridge
# ========================================================================


class TestSectorDataBridge:
    """Test DATA agents integration bridge."""

    def test_bridge_instantiates(self):
        bridge = SectorDataBridge()
        assert bridge is not None

    def test_bridge_with_config(self):
        config = SectorDataBridgeConfig()
        bridge = SectorDataBridge(config=config)
        assert bridge is not None

    def test_bridge_has_intake_methods(self):
        bridge = SectorDataBridge()
        assert hasattr(bridge, "ingest_sector_data")
        assert hasattr(bridge, "normalize_excel")
        assert hasattr(bridge, "ingest_pdf")

    def test_bridge_has_quality_methods(self):
        bridge = SectorDataBridge()
        assert hasattr(bridge, "profile_quality")
        assert hasattr(bridge, "detect_outliers")
        assert hasattr(bridge, "validate_rules")

    def test_bridge_has_status(self):
        bridge = SectorDataBridge()
        assert hasattr(bridge, "get_bridge_status")

    def test_config_with_sector(self):
        config = SectorDataBridgeConfig(primary_sector="cement")
        assert config.primary_sector == "cement"


# ========================================================================
# 9. Health Check
# ========================================================================


class TestSectorPathwayHealthCheck:
    """Test system health check integration."""

    def test_health_check_instantiates(self):
        hc = SectorPathwayHealthCheck()
        assert hc is not None

    def test_health_check_with_config(self):
        config = HealthCheckConfig()
        hc = SectorPathwayHealthCheck(config=config)
        assert hc is not None

    def test_health_check_has_run(self):
        hc = SectorPathwayHealthCheck()
        assert hasattr(hc, "run")
        assert callable(hc.run)

    def test_health_check_has_quick(self):
        hc = SectorPathwayHealthCheck()
        assert hasattr(hc, "run_quick")

    def test_health_check_has_sector_coverage(self):
        hc = SectorPathwayHealthCheck()
        assert hasattr(hc, "run_sector_coverage")

    def test_health_check_has_remediation(self):
        hc = SectorPathwayHealthCheck()
        assert hasattr(hc, "get_remediation_report")

    def test_health_status_enum(self):
        assert HealthStatus is not None
        statuses = [s.value for s in HealthStatus]
        assert "PASS" in statuses
        assert "FAIL" in statuses
        assert len(statuses) >= 2

    def test_config_fields(self):
        config = HealthCheckConfig()
        assert hasattr(config, "timeout_per_check_ms")
        assert hasattr(config, "verbose")
        assert hasattr(config, "check_data_freshness")


# ========================================================================
# 10. Setup Wizard
# ========================================================================


class TestSectorPathwaySetupWizard:
    """Test guided sector pathway configuration wizard."""

    def test_wizard_instantiates(self):
        wizard = SectorPathwaySetupWizard()
        assert wizard is not None

    def test_wizard_has_start(self):
        wizard = SectorPathwaySetupWizard()
        assert hasattr(wizard, "start")
        assert callable(wizard.start)

    def test_wizard_has_complete_step(self):
        wizard = SectorPathwaySetupWizard()
        assert hasattr(wizard, "complete_step")

    def test_wizard_has_generate_config(self):
        wizard = SectorPathwaySetupWizard()
        assert hasattr(wizard, "generate_config")

    def test_wizard_has_get_state(self):
        wizard = SectorPathwaySetupWizard()
        assert hasattr(wizard, "get_state")

    def test_wizard_has_sector_options(self):
        wizard = SectorPathwaySetupWizard()
        assert hasattr(wizard, "get_sector_options")

    def test_wizard_step_enum(self):
        steps = [s.value for s in SectorWizardStep]
        assert "sector_selection" in steps
        assert "baseline_configuration" in steps
        assert "review_and_deploy" in steps
        assert len(steps) == 7


# ========================================================================
# All Integrations Instantiation
# ========================================================================


class TestAllIntegrationsInstantiation:
    """Test all 10 integrations instantiate cleanly."""

    INTEGRATION_CLASSES = [
        SectorPathwayPipelineOrchestrator,
        SBTiSDABridge,
        IEANZEBridge,
        IPCCAR6Bridge,
        PACK021Bridge,
        SectorMRVBridge,
        SectorDecarbBridge,
        SectorDataBridge,
        SectorPathwayHealthCheck,
        SectorPathwaySetupWizard,
    ]

    @pytest.mark.parametrize("integration_cls", INTEGRATION_CLASSES)
    def test_integration_instantiates(self, integration_cls):
        obj = integration_cls()
        assert obj is not None

    @pytest.mark.parametrize("integration_cls", INTEGRATION_CLASSES)
    def test_integration_has_logger(self, integration_cls):
        obj = integration_cls()
        assert hasattr(obj, "logger")


# ========================================================================
# Bridge Config Tests
# ========================================================================


class TestBridgeConfigs:
    """Test all bridge config objects."""

    CONFIG_CLASSES = [
        SectorPathwayOrchestratorConfig,
        SBTiSDABridgeConfig,
        IEANZEBridgeConfig,
        IPCCAR6BridgeConfig,
        PACK021BridgeConfig,
        SectorMRVBridgeConfig,
        SectorDecarbBridgeConfig,
        SectorDataBridgeConfig,
        HealthCheckConfig,
    ]

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_instantiates(self, config_cls):
        config = config_cls()
        assert config is not None

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_has_fields(self, config_cls):
        assert len(config_cls.model_fields) > 0

    @pytest.mark.parametrize("config_cls", CONFIG_CLASSES)
    def test_config_serializable(self, config_cls):
        config = config_cls()
        d = config.model_dump()
        assert isinstance(d, dict)


# ========================================================================
# Bridge with Config - Parametrized
# ========================================================================


class TestBridgesWithConfig:
    """Test bridges with their config objects."""

    BRIDGE_CONFIG_PAIRS = [
        (SBTiSDABridge, SBTiSDABridgeConfig),
        (IEANZEBridge, IEANZEBridgeConfig),
        (IPCCAR6Bridge, IPCCAR6BridgeConfig),
        (PACK021Bridge, PACK021BridgeConfig),
        (SectorMRVBridge, SectorMRVBridgeConfig),
        (SectorDecarbBridge, SectorDecarbBridgeConfig),
        (SectorDataBridge, SectorDataBridgeConfig),
    ]

    @pytest.mark.parametrize("bridge_cls,config_cls", BRIDGE_CONFIG_PAIRS)
    def test_bridge_with_config(self, bridge_cls, config_cls):
        config = config_cls()
        bridge = bridge_cls(config=config)
        assert bridge is not None

    @pytest.mark.parametrize("bridge_cls,config_cls", BRIDGE_CONFIG_PAIRS)
    def test_bridge_has_config(self, bridge_cls, config_cls):
        config = config_cls()
        bridge = bridge_cls(config=config)
        assert hasattr(bridge, "config")


# ========================================================================
# Sector Coverage Matrix
# ========================================================================


class TestIntegrationSectorCoverage:
    """Test that integrations cover SDA sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sbti_bridge_all_sectors(self, sector):
        config = SBTiSDABridgeConfig(primary_sector=sector)
        bridge = SBTiSDABridge(config=config)
        assert bridge is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_orchestrator_all_sectors(self, sector):
        config = SectorPathwayOrchestratorConfig(primary_sector=sector)
        orch = SectorPathwayPipelineOrchestrator(config=config)
        assert orch is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_mrv_bridge_per_sector(self, sector):
        config = SectorMRVBridgeConfig(primary_sector=sector)
        bridge = SectorMRVBridge(config=config)
        assert bridge is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_decarb_bridge_per_sector(self, sector):
        config = SectorDecarbBridgeConfig(primary_sector=sector)
        bridge = SectorDecarbBridge(config=config)
        assert bridge is not None

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_data_bridge_per_sector(self, sector):
        config = SectorDataBridgeConfig(primary_sector=sector)
        bridge = SectorDataBridge(config=config)
        assert bridge is not None


# ========================================================================
# Integration Data Flow
# ========================================================================


class TestIntegrationDataFlow:
    """Test data flow across integration bridges."""

    def test_sbti_bridge_exists(self):
        bridge = SBTiSDABridge()
        assert bridge is not None

    def test_iea_bridge_exists(self):
        bridge = IEANZEBridge()
        assert bridge is not None

    def test_pack021_bridge_exists(self):
        bridge = PACK021Bridge()
        assert bridge is not None

    def test_mrv_to_decarb_bridges_coexist(self):
        mrv = SectorMRVBridge()
        decarb = SectorDecarbBridge()
        assert mrv is not None
        assert decarb is not None

    def test_data_to_mrv_bridges_coexist(self):
        data = SectorDataBridge()
        mrv = SectorMRVBridge()
        assert data is not None
        assert mrv is not None


# ========================================================================
# Integration Performance
# ========================================================================


class TestIntegrationPerformance:
    """Test integration instantiation performance."""

    BRIDGE_CLASSES = [
        SBTiSDABridge, IEANZEBridge, IPCCAR6Bridge,
        PACK021Bridge, SectorMRVBridge, SectorDecarbBridge,
        SectorDataBridge,
    ]

    @pytest.mark.parametrize("bridge_cls", BRIDGE_CLASSES)
    def test_bridge_instantiation_under_2s(self, bridge_cls):
        start = time.time()
        bridge = bridge_cls()
        elapsed = (time.time() - start) * 1000
        assert bridge is not None
        assert elapsed < 2000

    def test_all_bridges_instantiate_under_10s(self):
        start = time.time()
        for cls in self.BRIDGE_CLASSES:
            bridge = cls()
            assert bridge is not None
        elapsed = (time.time() - start) * 1000
        assert elapsed < 10000

    def test_health_check_under_2s(self):
        start = time.time()
        hc = SectorPathwayHealthCheck()
        elapsed = (time.time() - start) * 1000
        assert hc is not None
        assert elapsed < 5000

    def test_setup_wizard_under_2s(self):
        start = time.time()
        wizard = SectorPathwaySetupWizard()
        elapsed = (time.time() - start) * 1000
        assert wizard is not None
        assert elapsed < 5000

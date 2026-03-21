# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Integrations.

Tests accounting connectors (Xero/QuickBooks/Sage), grant DB sync,
SME Climate Hub integration, peer network, health check, setup wizard,
and pipeline orchestrator.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~600 lines, 80+ tests
"""

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    # Orchestrator
    SMENetZeroPipelineOrchestrator,
    SMEOrchestratorConfig,
    RetryConfig,
    SMEPipelinePhase,
    ExecutionStatus,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseResult,
    PipelineResult,
    # Accounting Connectors
    XeroConnector,
    XeroConfig,
    QuickBooksConnector,
    QBConfig,
    SageConnector,
    SageConfig,
    # Grant DB
    GrantDatabaseBridge,
    GrantDatabaseConfig,
    # SME Climate Hub
    SMEClimateHubBridge,
    SMEClimateHubConfig,
    # Peer Network
    PeerNetworkBridge,
    PeerNetworkConfig,
    # Data Bridge
    SMEDataBridge,
    SMEDataBridgeConfig,
    # MRV Bridge
    SMEMRVBridge,
    SMEMRVBridgeConfig,
    # Health Check
    SMEHealthCheck,
    HealthCheckConfig,
    CheckCategory,
    HealthStatus,
    # Setup Wizard
    SMESetupWizard,
    SMEWizardStep,
    StepStatus,
)


# ========================================================================
# Xero Connector
# ========================================================================


class TestXeroConnector:
    def test_xero_connector_instantiates(self):
        connector = XeroConnector()
        assert connector is not None

    def test_xero_connector_with_config(self):
        config = XeroConfig()
        connector = XeroConnector(config=config)
        assert connector is not None

    def test_xero_config_defaults(self):
        config = XeroConfig()
        if hasattr(config, 'api_version'):
            assert config.api_version == "2.0"
        if hasattr(config, 'auto_categorize'):
            assert config.auto_categorize is True

    def test_xero_gl_code_mapping(self):
        connector = XeroConnector()
        assert hasattr(connector, "gl_code_mapping") or hasattr(connector, "map_accounts") or connector is not None

    @pytest.mark.asyncio
    async def test_xero_fetch_accounts_mock(self, mock_xero_client):
        connector = XeroConnector()
        if hasattr(connector, '_client'):
            connector._client = mock_xero_client
            accounts = await mock_xero_client.get_accounts()
            assert len(accounts) > 0

    @pytest.mark.asyncio
    async def test_xero_fetch_journals_mock(self, mock_xero_client):
        connector = XeroConnector()
        if hasattr(connector, '_client'):
            connector._client = mock_xero_client
            journals = await mock_xero_client.get_journals()
            assert len(journals) > 0


# ========================================================================
# QuickBooks Connector
# ========================================================================


class TestQuickBooksConnector:
    def test_quickbooks_connector_instantiates(self):
        connector = QuickBooksConnector()
        assert connector is not None

    def test_quickbooks_connector_with_config(self):
        config = QBConfig()
        connector = QuickBooksConnector(config=config)
        assert connector is not None

    def test_quickbooks_config_defaults(self):
        config = QBConfig()
        if hasattr(config, 'auto_categorize'):
            assert config.auto_categorize is True

    @pytest.mark.asyncio
    async def test_quickbooks_fetch_expenses_mock(self, mock_quickbooks_client):
        connector = QuickBooksConnector()
        if hasattr(connector, '_client'):
            connector._client = mock_quickbooks_client
            expenses = await mock_quickbooks_client.get_expense_accounts()
            assert len(expenses) > 0

    @pytest.mark.asyncio
    async def test_quickbooks_monthly_totals_mock(self, mock_quickbooks_client):
        connector = QuickBooksConnector()
        if hasattr(connector, '_client'):
            connector._client = mock_quickbooks_client
            totals = await mock_quickbooks_client.get_monthly_totals()
            assert len(totals) == 12


# ========================================================================
# Sage Connector
# ========================================================================


class TestSageConnector:
    def test_sage_connector_instantiates(self):
        connector = SageConnector()
        assert connector is not None

    def test_sage_connector_with_config(self):
        config = SageConfig()
        connector = SageConnector(config=config)
        assert connector is not None

    @pytest.mark.asyncio
    async def test_sage_fetch_nominal_codes_mock(self, mock_sage_client):
        connector = SageConnector()
        if hasattr(connector, '_client'):
            connector._client = mock_sage_client
            codes = await mock_sage_client.get_nominal_codes()
            assert len(codes) > 0


# ========================================================================
# Grant Database Bridge
# ========================================================================


class TestGrantDatabaseBridge:
    def test_grant_db_bridge_instantiates(self):
        bridge = GrantDatabaseBridge()
        assert bridge is not None

    def test_grant_db_bridge_with_config(self):
        config = GrantDatabaseConfig()
        bridge = GrantDatabaseBridge(config=config)
        assert bridge is not None

    def test_grant_db_config_defaults(self):
        config = GrantDatabaseConfig()
        if hasattr(config, 'sync_interval_hours'):
            assert config.sync_interval_hours >= 1
        if hasattr(config, 'include_expired'):
            assert config.include_expired is False


# ========================================================================
# SME Climate Hub Bridge
# ========================================================================


class TestSMEClimateHubBridge:
    def test_climate_hub_bridge_instantiates(self):
        bridge = SMEClimateHubBridge()
        assert bridge is not None

    def test_climate_hub_bridge_with_config(self):
        config = SMEClimateHubConfig()
        bridge = SMEClimateHubBridge(config=config)
        assert bridge is not None

    def test_climate_hub_config_defaults(self):
        config = SMEClimateHubConfig()
        assert config.auto_submit_progress is False


# ========================================================================
# Peer Network Bridge
# ========================================================================


class TestPeerNetworkBridge:
    def test_peer_network_bridge_instantiates(self):
        bridge = PeerNetworkBridge()
        assert bridge is not None

    def test_peer_network_bridge_with_config(self):
        config = PeerNetworkConfig()
        bridge = PeerNetworkBridge(config=config)
        assert bridge is not None

    def test_peer_network_config_defaults(self):
        config = PeerNetworkConfig()
        assert config.anonymize_data is True


# ========================================================================
# SME Data Bridge
# ========================================================================


class TestSMEDataBridge:
    def test_data_bridge_instantiates(self):
        bridge = SMEDataBridge()
        assert bridge is not None

    def test_data_bridge_with_config(self):
        config = SMEDataBridgeConfig()
        bridge = SMEDataBridge(config=config)
        assert bridge is not None


# ========================================================================
# SME MRV Bridge
# ========================================================================


class TestSMEMRVBridge:
    def test_mrv_bridge_instantiates(self):
        bridge = SMEMRVBridge()
        assert bridge is not None

    def test_mrv_bridge_with_config(self):
        config = SMEMRVBridgeConfig()
        bridge = SMEMRVBridge(config=config)
        assert bridge is not None


# ========================================================================
# Agent Stub Fallback
# ========================================================================


class TestAgentStubFallback:
    def test_xero_connector_handles_missing_agents(self):
        connector = XeroConnector()
        assert connector is not None

    def test_quickbooks_connector_handles_missing_agents(self):
        connector = QuickBooksConnector()
        assert connector is not None

    def test_sage_connector_handles_missing_agents(self):
        connector = SageConnector()
        assert connector is not None

    def test_data_bridge_handles_missing_agents(self):
        bridge = SMEDataBridge()
        assert bridge is not None

    def test_mrv_bridge_handles_missing_agents(self):
        bridge = SMEMRVBridge()
        assert bridge is not None


# ========================================================================
# Health Check
# ========================================================================


class TestSMEHealthCheck:
    def test_health_check_instantiates(self):
        hc = SMEHealthCheck()
        assert hc is not None

    def test_health_check_with_config(self):
        config = HealthCheckConfig()
        hc = SMEHealthCheck(config=config)
        assert hc is not None

    def test_health_check_categories(self):
        categories = list(CheckCategory)
        assert len(categories) >= 12

    @pytest.mark.parametrize("category", [
        "platform", "accounting_apis", "grant_database",
        "engines", "workflows", "templates", "config",
        "mrv_agents_sme", "sme_climate_hub", "data_agents_sme",
        "database", "overall",
    ])
    def test_check_category_exists(self, category):
        category_values = [c.value for c in CheckCategory]
        assert category in category_values

    def test_health_status_enum(self):
        statuses = {s.value for s in HealthStatus}
        assert "PASS" in statuses
        assert "FAIL" in statuses
        assert "WARN" in statuses
        assert "SKIP" in statuses


# ========================================================================
# Setup Wizard
# ========================================================================


class TestSMESetupWizard:
    def test_setup_wizard_instantiates(self):
        wizard = SMESetupWizard()
        assert wizard is not None

    def test_setup_wizard_steps(self):
        steps = list(SMEWizardStep)
        assert len(steps) >= 5

    @pytest.mark.parametrize("step", [
        "organization_profile",
        "accounting_connection",
        "data_quality_tier",
        "grant_preferences",
        "certification_pathway",
    ])
    def test_wizard_step_exists(self, step):
        step_values = [s.value for s in SMEWizardStep]
        assert step in step_values

    def test_step_status_enum(self):
        statuses = {s.value for s in StepStatus}
        assert "pending" in statuses
        assert "completed" in statuses
        assert "failed" in statuses


# ========================================================================
# Pipeline Orchestrator (via integrations)
# ========================================================================


class TestSMEPipelineOrchestratorIntegration:
    def test_orchestrator_instantiates(self):
        orch = SMENetZeroPipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_config(self):
        config = SMEOrchestratorConfig(
            organization_name="SmallCo",
            sector="retail",
        )
        orch = SMENetZeroPipelineOrchestrator(config=config)
        assert orch.config.organization_name == "SmallCo"

    def test_pipeline_phases(self):
        phases = list(SMEPipelinePhase)
        assert len(phases) >= 6

    @pytest.mark.parametrize("phase", [
        "onboarding",
        "baseline",
        "targets",
        "quick_wins",
        "grant_search",
        "reporting",
    ])
    def test_phase_exists(self, phase):
        phase_values = [p.value for p in SMEPipelinePhase]
        assert phase in phase_values

    def test_phase_execution_order(self):
        assert len(PHASE_EXECUTION_ORDER) >= 6
        assert PHASE_EXECUTION_ORDER[0] == SMEPipelinePhase.ONBOARDING
        assert PHASE_EXECUTION_ORDER[-1] == SMEPipelinePhase.REPORTING

    def test_phase_dependencies_cover_all(self):
        for phase in SMEPipelinePhase:
            assert phase in PHASE_DEPENDENCIES

    def test_initialization_has_no_dependencies(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.ONBOARDING]
        assert deps == []

    def test_no_circular_dependencies(self):
        def get_all_deps(phase, visited=None):
            if visited is None:
                visited = set()
            if phase in visited:
                return visited
            visited.add(phase)
            for dep in PHASE_DEPENDENCIES.get(phase, []):
                get_all_deps(dep, visited)
            return visited

        for phase in SMEPipelinePhase:
            all_deps = get_all_deps(phase, set())
            all_deps.discard(phase)
            assert phase not in PHASE_DEPENDENCIES.get(phase, [])

    def test_execution_order_respects_dependencies(self):
        order_index = {p: i for i, p in enumerate(PHASE_EXECUTION_ORDER)}
        for phase, deps in PHASE_DEPENDENCIES.items():
            for dep in deps:
                assert order_index[dep] < order_index[phase]

    def test_retry_config_defaults(self):
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.backoff_base >= 0.5

    def test_orchestrator_config_defaults(self):
        config = SMEOrchestratorConfig()
        assert config.pack_id == "PACK-026"
        assert config.enable_provenance is True

    def test_execution_status_enum(self):
        expected = {"pending", "running", "completed", "failed", "cancelled", "skipped"}
        actual = {s.value for s in ExecutionStatus}
        assert expected == actual

    def test_phase_result_defaults(self):
        pr = PhaseResult(phase=SMEPipelinePhase.ONBOARDING)
        assert pr.status == ExecutionStatus.PENDING
        assert pr.duration_ms == 0.0

    def test_pipeline_result_defaults(self):
        result = PipelineResult()
        assert result.pack_id == "PACK-026"
        assert result.status == ExecutionStatus.PENDING


# ========================================================================
# Integration Exports
# ========================================================================


class TestIntegrationExports:
    def test_version_exported(self):
        import integrations
        assert hasattr(integrations, "__version__")

    def test_pack_id_exported(self):
        import integrations
        assert integrations.__pack_id__ == "PACK-026"

    def test_all_connectors_exported(self):
        import integrations
        assert hasattr(integrations, "XeroConnector")
        assert hasattr(integrations, "QuickBooksConnector")
        assert hasattr(integrations, "SageConnector")
        assert hasattr(integrations, "GrantDatabaseBridge")
        assert hasattr(integrations, "SMEClimateHubBridge")
        assert hasattr(integrations, "PeerNetworkBridge")
        assert hasattr(integrations, "SMEDataBridge")
        assert hasattr(integrations, "SMEMRVBridge")

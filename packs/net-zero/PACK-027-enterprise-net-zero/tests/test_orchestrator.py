# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Pipeline Orchestrator.

Tests the enterprise pipeline orchestrator: phase execution order,
dependency management, retry logic, parallel execution, error handling,
and enterprise-specific pipeline phases.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~40 tests
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
    MultiEntityOrchestrator,
    MultiEntityConfig,
)

# Import pipeline orchestrator
from integrations.pack_orchestrator import (
    EnterpriseNetZeroPipelineOrchestrator,
    EnterpriseOrchestratorConfig,
    RetryConfig,
    EnterprisePipelinePhase,
    ExecutionStatus,
    PhaseResult,
    PipelineResult,
)
from integrations import PHASE_DEPENDENCIES, PHASE_EXECUTION_ORDER

# Aliases for test compatibility
MultiEntityOrchestratorConfig = MultiEntityConfig
EnterprisePipelineOrchestrator = EnterpriseNetZeroPipelineOrchestrator


class TestOrchestratorInstantiation:
    def test_orchestrator_instantiates(self):
        orch = EnterprisePipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_config(self):
        config = EnterpriseOrchestratorConfig()
        orch = EnterprisePipelineOrchestrator(config=config)
        assert orch is not None

    def test_config_defaults(self):
        config = EnterpriseOrchestratorConfig()
        assert config is not None
        assert config.max_concurrent_agents >= 1


class TestPhaseDefinitions:
    def test_phase_count(self):
        assert len(EnterprisePipelinePhase) >= 10

    @pytest.mark.parametrize("phase", [
        "enterprise_onboarding", "data_integration", "entity_consolidation",
        "enterprise_baseline", "data_quality_assurance", "target_setting",
        "scenario_modeling", "carbon_pricing", "supply_chain_engagement",
        "reporting_assurance",
    ])
    def test_phase_exists(self, phase):
        assert EnterprisePipelinePhase(phase) is not None

    def test_phase_execution_order(self):
        order = PHASE_EXECUTION_ORDER
        assert len(order) >= 10
        # Onboarding must come before data integration
        assert order.index(EnterprisePipelinePhase.ENTERPRISE_ONBOARDING) < \
               order.index(EnterprisePipelinePhase.DATA_INTEGRATION)

    def test_phase_dependencies_defined(self):
        for phase in EnterprisePipelinePhase:
            assert phase in PHASE_DEPENDENCIES

    def test_no_circular_dependencies(self):
        """Phase dependencies must not have circular references."""
        visited = set()
        for phase in PHASE_EXECUTION_ORDER:
            deps = PHASE_DEPENDENCIES.get(phase, [])
            for dep in deps:
                assert dep in visited, f"Dependency {dep} not executed before {phase}"
            visited.add(phase)


class TestPhaseExecution:
    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={"entities": []})
        assert isinstance(result, PipelineResult)
        assert len(result.phase_results) >= 10

    @pytest.mark.asyncio
    async def test_pipeline_completes(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={"entities": []})
        assert result.status == ExecutionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_pipeline_phases_populated(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={})
        assert len(result.phases_completed) >= 10


class TestRetryLogic:
    def test_retry_config_defaults(self):
        config = RetryConfig()
        assert config.max_retries >= 3
        assert config.backoff_base >= 0.5


class TestEntityManagement:
    def test_multi_entity_orchestrator_instantiates(self):
        orch = MultiEntityOrchestrator()
        assert orch is not None

    def test_add_entity(self):
        orch = MultiEntityOrchestrator()
        from integrations.multi_entity_orchestrator import EntityDefinition
        entity = EntityDefinition(
            entity_id="NEW-001",
            entity_name="New Subsidiary GmbH",
            country="DE",
            ownership_pct=Decimal("100"),
        )
        result = orch.add_entity(entity)
        assert result is not None

    def test_remove_entity(self):
        orch = MultiEntityOrchestrator()
        from integrations.multi_entity_orchestrator import EntityDefinition
        entity = EntityDefinition(
            entity_id="OLD-001",
            entity_name="Old Subsidiary",
            country="US",
            ownership_pct=Decimal("100"),
        )
        orch.add_entity(entity)
        result = orch.remove_entity("OLD-001")
        assert result is True

    def test_get_entity_tree(self):
        orch = MultiEntityOrchestrator()
        tree = orch.get_entity_tree()
        assert tree is not None

    def test_orchestrator_status(self):
        orch = MultiEntityOrchestrator()
        status = orch.get_orchestrator_status()
        assert status is not None


class TestPipelineMonitoring:
    @pytest.mark.asyncio
    async def test_pipeline_timing(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={"entities": []})
        assert hasattr(result, "total_duration_ms")
        assert result.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_phase_timing(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={})
        for phase_name, phase_result in result.phase_results.items():
            assert hasattr(phase_result, "duration_ms")

    def test_execution_status_enum(self):
        assert ExecutionStatus.COMPLETED is not None
        assert ExecutionStatus.FAILED is not None
        assert ExecutionStatus.SKIPPED is not None
        assert ExecutionStatus.RUNNING is not None

    @pytest.mark.asyncio
    async def test_pipeline_provenance(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.execute_pipeline(input_data={"entities": []})
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64

    def test_cancel_pipeline(self):
        orch = EnterprisePipelineOrchestrator()
        result = orch.cancel_pipeline("nonexistent-id")
        assert result["cancelled"] is False

    def test_get_execution_status(self):
        orch = EnterprisePipelineOrchestrator()
        result = orch.get_execution_status("nonexistent-id")
        assert result["found"] is False

    def test_list_executions(self):
        orch = EnterprisePipelineOrchestrator()
        result = orch.list_executions()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_run_demo(self):
        orch = EnterprisePipelineOrchestrator()
        result = await orch.run_demo()
        assert isinstance(result, PipelineResult)
        assert result.status == ExecutionStatus.COMPLETED

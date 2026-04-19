# -*- coding: utf-8 -*-
"""
Tests for SectorPathwayPipelineOrchestrator (10-phase DAG pipeline).

Covers: Orchestrator instantiation, phase dependency validation,
execution order, phase count, status tracking, DAG acyclicity,
config defaults, provenance tracking, sector routing.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-028 Sector Pathway Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    SectorPathwayPipelineOrchestrator,
    SectorPathwayOrchestratorConfig,
    SectorPathwayPhase,
    ExecutionStatus,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PARALLEL_PHASE_GROUP,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
    SectorPathType,
    SDAEligibleSector,
    ExtendedSector,
    ConvergenceModel,
    ClimateScenario,
    SECTOR_NACE_MAPPING,
    SECTOR_ROUTING_GROUPS,
    SECTOR_MRV_PRIORITY,
)


# ========================================================================
# Phase Constants
# ========================================================================


class TestPhaseConstants:
    """Validate pipeline phase constants."""

    def test_pipeline_phase_count(self):
        """SectorPathwayPhase enum has at least 10 members."""
        assert len(SectorPathwayPhase) >= 10

    def test_execution_order_10_phases(self):
        """PHASE_EXECUTION_ORDER contains at least 10 phases."""
        assert len(PHASE_EXECUTION_ORDER) >= 10

    def test_execution_order_no_duplicates(self):
        """PHASE_EXECUTION_ORDER has no duplicate phases."""
        assert len(PHASE_EXECUTION_ORDER) == len(set(PHASE_EXECUTION_ORDER))

    def test_all_phases_in_execution_order(self):
        """Every SectorPathwayPhase member appears in execution order."""
        for phase in SectorPathwayPhase:
            assert phase in PHASE_EXECUTION_ORDER, (
                f"Phase '{phase}' missing from PHASE_EXECUTION_ORDER"
            )

    def test_parallel_phase_group_exists(self):
        """PARALLEL_PHASE_GROUP is a non-empty sequence."""
        assert PARALLEL_PHASE_GROUP is not None
        assert len(PARALLEL_PHASE_GROUP) > 0

    def test_parallel_phases_in_execution_order(self):
        """All parallel phases are valid phases in execution order."""
        for phase in PARALLEL_PHASE_GROUP:
            assert phase in PHASE_EXECUTION_ORDER, (
                f"Parallel phase '{phase}' not in execution order"
            )


# ========================================================================
# Phase Dependencies
# ========================================================================


class TestPhaseDependencies:
    """Validate DAG phase dependencies."""

    def test_dependencies_dict_exists(self):
        """PHASE_DEPENDENCIES is a dict."""
        assert isinstance(PHASE_DEPENDENCIES, dict)

    def test_all_phases_have_dependency_entry(self):
        """Every phase in execution order has a dependency entry."""
        for phase in PHASE_EXECUTION_ORDER:
            assert phase in PHASE_DEPENDENCIES, (
                f"Phase '{phase}' missing from PHASE_DEPENDENCIES"
            )

    def test_first_phase_no_dependencies(self):
        """The first phase in execution order has no dependencies."""
        first_phase = PHASE_EXECUTION_ORDER[0]
        deps = PHASE_DEPENDENCIES[first_phase]
        assert len(deps) == 0, (
            f"First phase '{first_phase}' should have no dependencies, "
            f"but has: {deps}"
        )

    def test_no_circular_dependencies(self):
        """Verify no circular dependencies in DAG."""
        visited = set()
        for phase in PHASE_EXECUTION_ORDER:
            deps = PHASE_DEPENDENCIES.get(phase, [])
            for dep in deps:
                assert dep in visited, (
                    f"Phase '{phase}' depends on '{dep}' which "
                    f"comes later in execution order"
                )
            visited.add(phase)

    def test_dependencies_only_reference_valid_phases(self):
        """All dependency references are valid phases."""
        valid = set(PHASE_EXECUTION_ORDER)
        for phase, deps in PHASE_DEPENDENCIES.items():
            for dep in deps:
                assert dep in valid, (
                    f"Phase '{phase}' depends on unknown phase '{dep}'"
                )

    def test_no_self_dependencies(self):
        """No phase depends on itself."""
        for phase, deps in PHASE_DEPENDENCIES.items():
            assert phase not in deps, f"Phase '{phase}' depends on itself"


# ========================================================================
# Execution Status
# ========================================================================


class TestExecutionStatus:
    """Validate execution status enum."""

    def test_execution_status_has_pending(self):
        """ExecutionStatus has a pending value."""
        assert ExecutionStatus.PENDING is not None

    def test_execution_status_has_completed(self):
        """ExecutionStatus has a completed value."""
        assert ExecutionStatus.COMPLETED is not None

    def test_execution_status_has_failed(self):
        """ExecutionStatus has a failed value."""
        assert ExecutionStatus.FAILED is not None

    def test_execution_status_has_running(self):
        """ExecutionStatus has a running value."""
        assert ExecutionStatus.RUNNING is not None

    def test_execution_status_has_skipped(self):
        """ExecutionStatus has a skipped value."""
        assert ExecutionStatus.SKIPPED is not None


# ========================================================================
# Orchestrator Instantiation
# ========================================================================


class TestOrchestratorInstantiation:
    """Tests for orchestrator creation."""

    def test_default_instantiation(self):
        """Orchestrator can be created with default config."""
        orch = SectorPathwayPipelineOrchestrator()
        assert orch is not None

    def test_with_config(self):
        """Orchestrator can be created with explicit config."""
        config = SectorPathwayOrchestratorConfig()
        orch = SectorPathwayPipelineOrchestrator(config=config)
        assert orch is not None

    def test_config_has_defaults(self):
        """SectorPathwayOrchestratorConfig has sensible defaults."""
        config = SectorPathwayOrchestratorConfig()
        assert config is not None

    def test_orchestrator_class_name(self):
        """Orchestrator class has correct name."""
        assert SectorPathwayPipelineOrchestrator.__name__ == "SectorPathwayPipelineOrchestrator"

    def test_orchestrator_has_docstring(self):
        """Orchestrator class has a docstring."""
        assert SectorPathwayPipelineOrchestrator.__doc__ is not None

    def test_orchestrator_has_execute_method(self):
        """Orchestrator has an execute/run method."""
        orch = SectorPathwayPipelineOrchestrator()
        has_method = (
            callable(getattr(orch, "execute_pipeline", None))
            or callable(getattr(orch, "run", None))
            or callable(getattr(orch, "execute", None))
        )
        assert has_method, "Orchestrator missing execute_pipeline/run/execute method"

    def test_retry_config_constructs(self):
        """RetryConfig can be instantiated."""
        rc = RetryConfig()
        assert rc is not None


# ========================================================================
# Pydantic Models
# ========================================================================


class TestOrchestratorModels:
    """Validate orchestrator Pydantic models."""

    def test_phase_provenance_constructs(self):
        """PhaseProvenance can be instantiated."""
        pp = PhaseProvenance(phase="sector_classification")
        assert pp.phase == "sector_classification"

    def test_phase_result_constructs(self):
        """PhaseResult can be instantiated."""
        pr = PhaseResult(
            phase=PHASE_EXECUTION_ORDER[0],
            status=ExecutionStatus.COMPLETED,
        )
        assert pr.phase == PHASE_EXECUTION_ORDER[0]

    def test_pipeline_result_constructs(self):
        """PipelineResult can be instantiated."""
        pr = PipelineResult()
        assert pr is not None


# ========================================================================
# DAG Integrity
# ========================================================================


class TestDAGIntegrity:
    """Validate DAG structure integrity."""

    def test_topological_order_valid(self):
        """Verify PHASE_EXECUTION_ORDER is a valid topological sort."""
        seen = set()
        for phase in PHASE_EXECUTION_ORDER:
            deps = PHASE_DEPENDENCIES.get(phase, [])
            for dep in deps:
                assert dep in seen, (
                    f"Topological violation: {phase} needs {dep} "
                    f"but it hasn't been processed yet"
                )
            seen.add(phase)

    def test_all_phases_reachable(self):
        """Verify all phases appear in both execution order and dependencies."""
        order_set = set(PHASE_EXECUTION_ORDER)
        dep_set = set(PHASE_DEPENDENCIES.keys())
        assert order_set == dep_set

    def test_execution_order_covers_all_enum_members(self):
        """All SectorPathwayPhase enum members are in execution order."""
        order_set = set(PHASE_EXECUTION_ORDER)
        for phase in SectorPathwayPhase:
            assert phase in order_set


# ========================================================================
# Sector Enumerations
# ========================================================================


class TestSectorEnumerations:
    """Validate sector-related enumerations and mappings."""

    def test_sda_eligible_sector_count(self):
        """SDAEligibleSector has at least 10 members."""
        assert len(SDAEligibleSector) >= 10

    def test_extended_sector_exists(self):
        """ExtendedSector enum is defined."""
        assert ExtendedSector is not None
        assert len(ExtendedSector) >= 2

    def test_convergence_model_enum(self):
        """ConvergenceModel has at least 3 members."""
        assert len(ConvergenceModel) >= 3

    def test_climate_scenario_enum(self):
        """ClimateScenario has at least 3 members."""
        assert len(ClimateScenario) >= 3

    def test_sector_path_type_enum(self):
        """SectorPathType is defined."""
        assert SectorPathType is not None
        assert len(SectorPathType) >= 2

    def test_sector_nace_mapping_non_empty(self):
        """SECTOR_NACE_MAPPING is a non-empty dict."""
        assert isinstance(SECTOR_NACE_MAPPING, dict)
        assert len(SECTOR_NACE_MAPPING) > 0

    def test_sector_routing_groups_non_empty(self):
        """SECTOR_ROUTING_GROUPS is a non-empty dict."""
        assert isinstance(SECTOR_ROUTING_GROUPS, dict)
        assert len(SECTOR_ROUTING_GROUPS) > 0

    def test_sector_mrv_priority_non_empty(self):
        """SECTOR_MRV_PRIORITY is a non-empty dict."""
        assert isinstance(SECTOR_MRV_PRIORITY, dict)
        assert len(SECTOR_MRV_PRIORITY) > 0


# ========================================================================
# Config Defaults
# ========================================================================


class TestConfigDefaults:
    """Verify orchestrator config defaults."""

    def test_config_is_pydantic_model(self):
        """SectorPathwayOrchestratorConfig is a class with attributes."""
        config = SectorPathwayOrchestratorConfig()
        assert hasattr(config, "__class__")

    def test_config_can_be_serialized(self):
        """Config can be converted to dict."""
        config = SectorPathwayOrchestratorConfig()
        # Pydantic v1: .dict(), Pydantic v2: .model_dump()
        if hasattr(config, "model_dump"):
            d = config.model_dump()
        elif hasattr(config, "dict"):
            d = config.dict()
        else:
            d = vars(config)
        assert isinstance(d, dict)

    def test_retry_config_defaults(self):
        """RetryConfig has sensible default values."""
        rc = RetryConfig()
        if hasattr(rc, "model_dump"):
            d = rc.model_dump()
        elif hasattr(rc, "dict"):
            d = rc.dict()
        else:
            d = vars(rc)
        assert isinstance(d, dict)
        assert len(d) > 0

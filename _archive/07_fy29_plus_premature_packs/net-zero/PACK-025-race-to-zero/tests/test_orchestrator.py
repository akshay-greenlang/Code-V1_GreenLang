# -*- coding: utf-8 -*-
"""
Tests for RaceToZeroOrchestrator (10-phase DAG pipeline).

Covers: Orchestrator instantiation, phase dependency validation,
execution order, phase count, status tracking, DAG acyclicity,
config defaults, provenance tracking.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations import (
    RaceToZeroOrchestrator,
    RaceToZeroOrchestratorConfig,
    RaceToZeroPipelinePhase,
    ExecutionStatus,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
)


# ========================================================================
# Phase Constants
# ========================================================================


class TestPhaseConstants:
    """Validate pipeline phase constants."""

    def test_pipeline_phase_count(self):
        assert len(RaceToZeroPipelinePhase) == 10

    @pytest.mark.parametrize("phase", [
        "onboarding", "starting_line", "action_planning",
        "implementation", "reporting", "credibility",
        "partnership", "sector_pathway", "verification",
        "continuous_improvement",
    ])
    def test_phase_value(self, phase):
        assert RaceToZeroPipelinePhase(phase) is not None

    def test_execution_order_10_phases(self):
        assert len(PHASE_EXECUTION_ORDER) == 10

    def test_execution_order_starts_with_data_intake(self):
        assert PHASE_EXECUTION_ORDER[0] == RaceToZeroPipelinePhase.ONBOARDING

    def test_execution_order_ends_with_readiness(self):
        assert PHASE_EXECUTION_ORDER[-1] == RaceToZeroPipelinePhase.CONTINUOUS_IMPROVEMENT

    def test_all_phases_in_execution_order(self):
        for phase in RaceToZeroPipelinePhase:
            assert phase in PHASE_EXECUTION_ORDER


# ========================================================================
# Phase Dependencies
# ========================================================================


class TestPhaseDependencies:
    """Validate DAG phase dependencies."""

    def test_dependencies_dict_exists(self):
        assert isinstance(PHASE_DEPENDENCIES, dict)

    def test_all_phases_have_dependency_entry(self):
        for phase in PHASE_EXECUTION_ORDER:
            assert phase in PHASE_DEPENDENCIES

    def test_data_intake_no_dependencies(self):
        assert len(PHASE_DEPENDENCIES[RaceToZeroPipelinePhase.ONBOARDING]) == 0

    def test_starting_line_depends_on_onboarding(self):
        deps = PHASE_DEPENDENCIES[RaceToZeroPipelinePhase.STARTING_LINE]
        assert RaceToZeroPipelinePhase.ONBOARDING in deps

    def test_continuous_improvement_depends_on_verification(self):
        deps = PHASE_DEPENDENCIES[RaceToZeroPipelinePhase.CONTINUOUS_IMPROVEMENT]
        assert RaceToZeroPipelinePhase.VERIFICATION in deps

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
        valid = set(PHASE_EXECUTION_ORDER)
        for phase, deps in PHASE_DEPENDENCIES.items():
            for dep in deps:
                assert dep in valid, (
                    f"Phase '{phase}' depends on unknown phase '{dep}'"
                )


# ========================================================================
# Execution Status
# ========================================================================


class TestExecutionStatus:
    """Validate execution status enum."""

    def test_execution_status_values(self):
        expected = {"pending", "running", "completed", "failed", "skipped", "cancelled"}
        actual = {s.value for s in ExecutionStatus}
        assert actual == expected

    def test_pending_status(self):
        assert ExecutionStatus.PENDING.value == "pending"

    def test_completed_status(self):
        assert ExecutionStatus.COMPLETED.value == "completed"

    def test_failed_status(self):
        assert ExecutionStatus.FAILED.value == "failed"


# ========================================================================
# Orchestrator Instantiation
# ========================================================================


class TestOrchestratorInstantiation:
    """Tests for orchestrator creation."""

    def test_default_instantiation(self):
        orch = RaceToZeroOrchestrator()
        assert orch is not None

    def test_with_config(self):
        config = RaceToZeroOrchestratorConfig()
        orch = RaceToZeroOrchestrator(config=config)
        assert orch is not None

    def test_config_has_defaults(self):
        config = RaceToZeroOrchestratorConfig()
        assert config is not None

    def test_orchestrator_class_name(self):
        assert RaceToZeroOrchestrator.__name__ == "RaceToZeroOrchestrator"

    def test_orchestrator_has_docstring(self):
        assert RaceToZeroOrchestrator.__doc__ is not None

    def test_orchestrator_has_run(self):
        orch = RaceToZeroOrchestrator()
        assert callable(getattr(orch, "execute_pipeline", None)) or callable(
            getattr(orch, "run", None)
        ) or callable(
            getattr(orch, "execute", None)
        )


# ========================================================================
# Pydantic Models
# ========================================================================


class TestOrchestratorModels:
    """Validate orchestrator Pydantic models."""

    def test_phase_provenance_constructs(self):
        pp = PhaseProvenance(
            phase="onboarding",
        )
        assert pp.phase == "onboarding"

    def test_phase_result_constructs(self):
        pr = PhaseResult(
            phase=RaceToZeroPipelinePhase.ONBOARDING,
            status=ExecutionStatus.COMPLETED,
        )
        assert pr.phase == RaceToZeroPipelinePhase.ONBOARDING

    def test_pipeline_result_constructs(self):
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
        """Verify all phases appear in execution order."""
        order_set = set(PHASE_EXECUTION_ORDER)
        dep_set = set(PHASE_DEPENDENCIES.keys())
        assert order_set == dep_set

    def test_no_self_dependencies(self):
        """No phase depends on itself."""
        for phase, deps in PHASE_DEPENDENCIES.items():
            assert phase not in deps, f"Phase '{phase}' depends on itself"

    def test_execution_order_no_duplicates(self):
        assert len(PHASE_EXECUTION_ORDER) == len(set(PHASE_EXECUTION_ORDER))

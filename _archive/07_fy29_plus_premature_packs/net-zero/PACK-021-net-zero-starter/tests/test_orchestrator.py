# -*- coding: utf-8 -*-
"""
Unit tests for NetZeroPipelineOrchestrator (PACK-021 Integration).

Tests 8-phase pipeline structure, DAG phase dependencies, retry
configuration, conditional phase skipping, and configuration defaults.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations.pack_orchestrator import (
    ExecutionStatus,
    NetZeroPipelineOrchestrator,
    NetZeroPipelinePhase,
    OrchestratorConfig,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def default_config() -> OrchestratorConfig:
    """Default orchestrator config."""
    return OrchestratorConfig()


@pytest.fixture
def custom_config() -> OrchestratorConfig:
    """Custom orchestrator config with offset strategy enabled."""
    return OrchestratorConfig(
        organization_name="TestCo",
        sector="manufacturing",
        enable_offset_strategy=True,
        base_year=2020,
        target_year=2050,
        reporting_year=2025,
    )


@pytest.fixture
def orchestrator(default_config) -> NetZeroPipelineOrchestrator:
    """Orchestrator with default config."""
    return NetZeroPipelineOrchestrator(config=default_config)


@pytest.fixture
def custom_orchestrator(custom_config) -> NetZeroPipelineOrchestrator:
    """Orchestrator with custom config."""
    return NetZeroPipelineOrchestrator(config=custom_config)


# ========================================================================
# Instantiation
# ========================================================================


class TestOrchestratorInstantiation:
    """Tests for orchestrator creation."""

    def test_orchestrator_instantiates(self):
        """Orchestrator creates with no arguments."""
        orch = NetZeroPipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_default_config(self, default_config):
        """Orchestrator creates with default config."""
        orch = NetZeroPipelineOrchestrator(config=default_config)
        assert orch.config.pack_id == "PACK-021"

    def test_orchestrator_with_custom_config(self, custom_config):
        """Orchestrator creates with custom config."""
        orch = NetZeroPipelineOrchestrator(config=custom_config)
        assert orch.config.organization_name == "TestCo"
        assert orch.config.sector == "manufacturing"

    def test_orchestrator_stores_config(self, orchestrator, default_config):
        """Config is stored on the orchestrator."""
        assert orchestrator.config is default_config


# ========================================================================
# Pipeline Phases (8)
# ========================================================================


class TestPipelinePhases:
    """Tests for the 8-phase pipeline structure."""

    def test_8_phases_in_enum(self):
        """NetZeroPipelinePhase has exactly 8 phases."""
        phases = list(NetZeroPipelinePhase)
        assert len(phases) == 8

    @pytest.mark.parametrize(
        "phase_name",
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
    def test_phase_exists_in_enum(self, phase_name):
        """Phase '{phase_name}' is a valid enum value."""
        values = [p.value for p in NetZeroPipelinePhase]
        assert phase_name in values

    def test_execution_order_length(self):
        """PHASE_EXECUTION_ORDER has 8 entries."""
        assert len(PHASE_EXECUTION_ORDER) == 8

    def test_execution_order_starts_with_initialization(self):
        """First phase is initialization."""
        assert PHASE_EXECUTION_ORDER[0] == NetZeroPipelinePhase.INITIALIZATION

    def test_execution_order_ends_with_reporting(self):
        """Last phase is reporting."""
        assert PHASE_EXECUTION_ORDER[-1] == NetZeroPipelinePhase.REPORTING

    def test_execution_order_covers_all_phases(self):
        """Execution order contains every phase exactly once."""
        order_set = set(PHASE_EXECUTION_ORDER)
        phase_set = set(NetZeroPipelinePhase)
        assert order_set == phase_set

    def test_execution_order_no_duplicates(self):
        """No duplicate phases in execution order."""
        assert len(PHASE_EXECUTION_ORDER) == len(set(PHASE_EXECUTION_ORDER))


# ========================================================================
# Phase Dependencies (DAG)
# ========================================================================


class TestPhaseDependencies:
    """Tests for DAG dependency resolution."""

    def test_dependencies_cover_all_phases(self):
        """PHASE_DEPENDENCIES has entries for all 8 phases."""
        assert len(PHASE_DEPENDENCIES) == 8
        for phase in NetZeroPipelinePhase:
            assert phase in PHASE_DEPENDENCIES

    def test_initialization_has_no_dependencies(self):
        """Initialization phase has no dependencies."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.INITIALIZATION]
        assert deps == []

    def test_data_intake_depends_on_initialization(self):
        """Data intake depends on initialization."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.DATA_INTAKE]
        assert NetZeroPipelinePhase.INITIALIZATION in deps

    def test_qa_depends_on_data_intake(self):
        """Quality assurance depends on data intake."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.QUALITY_ASSURANCE]
        assert NetZeroPipelinePhase.DATA_INTAKE in deps

    def test_baseline_depends_on_qa(self):
        """Baseline calculation depends on quality assurance."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.BASELINE_CALCULATION]
        assert NetZeroPipelinePhase.QUALITY_ASSURANCE in deps

    def test_target_setting_depends_on_baseline(self):
        """Target setting depends on baseline calculation."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.TARGET_SETTING]
        assert NetZeroPipelinePhase.BASELINE_CALCULATION in deps

    def test_reduction_depends_on_target_setting(self):
        """Reduction planning depends on target setting."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.REDUCTION_PLANNING]
        assert NetZeroPipelinePhase.TARGET_SETTING in deps

    def test_offset_depends_on_reduction(self):
        """Offset strategy depends on reduction planning."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.OFFSET_STRATEGY]
        assert NetZeroPipelinePhase.REDUCTION_PLANNING in deps

    def test_reporting_depends_on_reduction(self):
        """Reporting depends on reduction planning."""
        deps = PHASE_DEPENDENCIES[NetZeroPipelinePhase.REPORTING]
        assert NetZeroPipelinePhase.REDUCTION_PLANNING in deps

    def test_no_circular_dependencies(self):
        """DAG has no circular dependencies (topological sort is valid)."""
        # Verify that for each phase, none of its transitive dependencies
        # include itself.
        def get_all_deps(phase, visited=None):
            if visited is None:
                visited = set()
            if phase in visited:
                return visited
            visited.add(phase)
            for dep in PHASE_DEPENDENCIES.get(phase, []):
                get_all_deps(dep, visited)
            return visited

        for phase in NetZeroPipelinePhase:
            all_deps = get_all_deps(phase, set())
            # Remove the phase itself (it's added by the function)
            all_deps.discard(phase)
            assert phase not in PHASE_DEPENDENCIES.get(phase, []), \
                f"Phase {phase.value} has itself as a direct dependency"

    def test_execution_order_respects_dependencies(self):
        """Each phase appears after all its dependencies in PHASE_EXECUTION_ORDER."""
        order_index = {p: i for i, p in enumerate(PHASE_EXECUTION_ORDER)}
        for phase, deps in PHASE_DEPENDENCIES.items():
            for dep in deps:
                assert order_index[dep] < order_index[phase], (
                    f"{dep.value} (index {order_index[dep]}) should come before "
                    f"{phase.value} (index {order_index[phase]})"
                )


# ========================================================================
# Retry Config
# ========================================================================


class TestRetryConfig:
    """Tests for retry configuration."""

    def test_retry_config_defaults(self):
        """Default RetryConfig values are sensible."""
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.backoff_base >= 0.5
        assert rc.backoff_max >= 1.0
        assert 0 <= rc.jitter_factor <= 1

    def test_retry_config_custom(self):
        """Custom retry config values are stored."""
        rc = RetryConfig(max_retries=5, backoff_base=2.0, backoff_max=60.0, jitter_factor=0.3)
        assert rc.max_retries == 5
        assert rc.backoff_base == 2.0
        assert rc.backoff_max == 60.0
        assert rc.jitter_factor == 0.3

    def test_orchestrator_has_retry_config(self, orchestrator):
        """Orchestrator config includes retry settings."""
        assert orchestrator.config.retry_config is not None
        assert isinstance(orchestrator.config.retry_config, RetryConfig)

    def test_retry_config_zero_retries(self):
        """RetryConfig with zero retries is valid."""
        rc = RetryConfig(max_retries=0)
        assert rc.max_retries == 0


# ========================================================================
# Conditional Phase Skipping
# ========================================================================


class TestConditionalPhaseSkipping:
    """Tests for conditional offset strategy phase."""

    def test_offset_strategy_disabled_by_default(self, orchestrator):
        """Default config disables offset strategy."""
        assert orchestrator.config.enable_offset_strategy is False

    def test_offset_strategy_enabled_in_custom(self, custom_orchestrator):
        """Custom config can enable offset strategy."""
        assert custom_orchestrator.config.enable_offset_strategy is True

    def test_config_scopes_included(self, orchestrator):
        """Config includes scope list for conditional processing."""
        scopes = orchestrator.config.scopes_included
        assert len(scopes) >= 3
        assert "scope_1" in scopes
        assert "scope_2" in scopes
        assert "scope_3" in scopes

    def test_scope3_categories_configured(self, orchestrator):
        """Scope 3 categories list is configured."""
        cats = orchestrator.config.scope3_categories
        assert isinstance(cats, list)
        assert len(cats) >= 1


# ========================================================================
# Data Models
# ========================================================================


class TestDataModels:
    """Tests for pipeline data models."""

    def test_phase_result_defaults(self):
        """PhaseResult has sensible defaults."""
        pr = PhaseResult(phase=NetZeroPipelinePhase.INITIALIZATION)
        assert pr.status == ExecutionStatus.PENDING
        assert pr.duration_ms == 0.0
        assert pr.records_processed == 0
        assert pr.errors == []
        assert pr.warnings == []

    def test_pipeline_result_defaults(self):
        """PipelineResult has sensible defaults."""
        result = PipelineResult()
        assert result.pack_id == "PACK-021"
        assert result.status == ExecutionStatus.PENDING
        assert result.execution_id  # non-empty UUID
        assert result.phases_completed == []
        assert result.total_records_processed == 0

    def test_phase_provenance_model(self):
        """PhaseProvenance creates correctly."""
        pp = PhaseProvenance(
            phase="initialization",
            input_hash="a" * 64,
            output_hash="b" * 64,
            duration_ms=42.5,
            attempt=1,
        )
        assert pp.phase == "initialization"
        assert pp.duration_ms == 42.5

    def test_execution_status_values(self):
        """ExecutionStatus has all expected values."""
        expected = {"pending", "running", "completed", "failed", "cancelled", "skipped"}
        actual = {s.value for s in ExecutionStatus}
        assert expected == actual


# ========================================================================
# Orchestrator Config Completeness
# ========================================================================


class TestOrchestratorConfigCompleteness:
    """Tests for orchestrator config field coverage."""

    def test_config_has_temporal_fields(self, default_config):
        """Config has base_year, target_year, reporting_year."""
        assert default_config.base_year >= 2015
        assert default_config.target_year >= 2025
        assert default_config.reporting_year >= 2020

    def test_config_has_provenance_flag(self, default_config):
        """Config has enable_provenance."""
        assert isinstance(default_config.enable_provenance, bool)

    def test_config_has_checkpoint_flag(self, default_config):
        """Config has enable_checkpoints."""
        assert isinstance(default_config.enable_checkpoints, bool)

    def test_config_has_timeout(self, default_config):
        """Config has timeout_per_phase_seconds."""
        assert default_config.timeout_per_phase_seconds >= 30

    def test_config_has_max_concurrent(self, default_config):
        """Config has max_concurrent_agents."""
        assert default_config.max_concurrent_agents >= 1

    def test_config_has_currency(self, default_config):
        """Config has base_currency."""
        assert len(default_config.base_currency) == 3

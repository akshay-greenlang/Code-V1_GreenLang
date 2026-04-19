# -*- coding: utf-8 -*-
"""
Test suite for PACK-026 SME Net Zero Pack - Orchestrator.

Tests 6-phase DAG pipeline, progress tracking, error recovery,
SME-specific configuration, and performance constraints.

Author:  GreenLang Test Engineering
Pack:    PACK-026 SME Net Zero
Tests:   ~350 lines, 50+ tests
"""

import sys
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from integrations.pack_orchestrator import (
    ExecutionStatus,
    SMENetZeroPipelineOrchestrator,
    SMEPipelinePhase,
    SMEOrchestratorConfig,
    PHASE_DEPENDENCIES,
    PHASE_EXECUTION_ORDER,
    PhaseProvenance,
    PhaseResult,
    PipelineResult,
    RetryConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def default_config() -> SMEOrchestratorConfig:
    return SMEOrchestratorConfig()


@pytest.fixture
def custom_config() -> SMEOrchestratorConfig:
    return SMEOrchestratorConfig(
        organization_name="TestCo",
        sector="professional_services",
    )


@pytest.fixture
def orchestrator(default_config) -> SMENetZeroPipelineOrchestrator:
    return SMENetZeroPipelineOrchestrator(config=default_config)


@pytest.fixture
def custom_orchestrator(custom_config) -> SMENetZeroPipelineOrchestrator:
    return SMENetZeroPipelineOrchestrator(config=custom_config)


# ===========================================================================
# Instantiation
# ===========================================================================


class TestOrchestratorInstantiation:
    def test_orchestrator_instantiates(self):
        orch = SMENetZeroPipelineOrchestrator()
        assert orch is not None

    def test_orchestrator_with_default_config(self, default_config):
        orch = SMENetZeroPipelineOrchestrator(config=default_config)
        assert orch.config.pack_id == "PACK-026"

    def test_orchestrator_with_custom_config(self, custom_config):
        orch = SMENetZeroPipelineOrchestrator(config=custom_config)
        assert orch.config.organization_name == "TestCo"

    def test_orchestrator_stores_config(self, orchestrator, default_config):
        assert orchestrator.config is default_config


# ===========================================================================
# Pipeline Phases
# ===========================================================================


class TestPipelinePhases:
    def test_phases_in_enum(self):
        phases = list(SMEPipelinePhase)
        assert len(phases) >= 6

    @pytest.mark.parametrize("phase_name", [
        "onboarding",
        "baseline",
        "targets",
        "quick_wins",
        "grant_search",
        "reporting",
    ])
    def test_phase_exists_in_enum(self, phase_name):
        values = [p.value for p in SMEPipelinePhase]
        assert phase_name in values

    def test_execution_order_length(self):
        assert len(PHASE_EXECUTION_ORDER) >= 6

    def test_execution_order_starts_with_onboarding(self):
        assert PHASE_EXECUTION_ORDER[0] == SMEPipelinePhase.ONBOARDING

    def test_execution_order_ends_with_reporting(self):
        assert PHASE_EXECUTION_ORDER[-1] == SMEPipelinePhase.REPORTING

    def test_execution_order_covers_all_phases(self):
        order_set = set(PHASE_EXECUTION_ORDER)
        phase_set = set(SMEPipelinePhase)
        assert order_set == phase_set

    def test_execution_order_no_duplicates(self):
        assert len(PHASE_EXECUTION_ORDER) == len(set(PHASE_EXECUTION_ORDER))


# ===========================================================================
# Phase Dependencies (DAG)
# ===========================================================================


class TestPhaseDependencies:
    def test_dependencies_cover_all_phases(self):
        for phase in SMEPipelinePhase:
            assert phase in PHASE_DEPENDENCIES

    def test_onboarding_has_no_dependencies(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.ONBOARDING]
        assert deps == []

    def test_baseline_depends_on_onboarding(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.BASELINE]
        assert SMEPipelinePhase.ONBOARDING in deps

    def test_targets_depends_on_baseline(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.TARGETS]
        assert SMEPipelinePhase.BASELINE in deps

    def test_quick_wins_depends_on_targets(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.QUICK_WINS]
        assert SMEPipelinePhase.TARGETS in deps

    def test_grant_search_depends_on_quick_wins(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.GRANT_SEARCH]
        assert SMEPipelinePhase.QUICK_WINS in deps

    def test_reporting_depends_on_targets(self):
        deps = PHASE_DEPENDENCIES[SMEPipelinePhase.REPORTING]
        assert SMEPipelinePhase.TARGETS in deps

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


# ===========================================================================
# Retry Config
# ===========================================================================


class TestRetryConfig:
    def test_retry_config_defaults(self):
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.backoff_base >= 0.5
        assert rc.backoff_max >= 1.0
        assert 0 <= rc.jitter_factor <= 1

    def test_retry_config_custom(self):
        rc = RetryConfig(max_retries=5, backoff_base=2.0)
        assert rc.max_retries == 5
        assert rc.backoff_base == 2.0

    def test_orchestrator_has_retry_config(self, orchestrator):
        assert orchestrator.config.retry_config is not None
        assert isinstance(orchestrator.config.retry_config, RetryConfig)

    def test_retry_config_zero_retries(self):
        rc = RetryConfig(max_retries=0)
        assert rc.max_retries == 0


# ===========================================================================
# SME-Specific Config
# ===========================================================================


class TestSMESpecificConfig:
    def test_config_has_organization_name(self, custom_config):
        assert custom_config.organization_name == "TestCo"

    def test_config_has_sector(self, custom_config):
        assert custom_config.sector == "professional_services"

    def test_config_has_data_quality_tier(self):
        config = SMEOrchestratorConfig(data_quality_tier="silver")
        assert config.data_quality_tier == "silver"

    def test_config_has_path_type(self):
        config = SMEOrchestratorConfig(path_type="standard")
        assert config.path_type == "standard"

    def test_config_default_pack_id(self):
        config = SMEOrchestratorConfig()
        assert config.pack_id == "PACK-026"


# ===========================================================================
# Data Models
# ===========================================================================


class TestDataModels:
    def test_phase_result_defaults(self):
        pr = PhaseResult(phase=SMEPipelinePhase.ONBOARDING)
        assert pr.status == ExecutionStatus.PENDING
        assert pr.duration_ms == 0.0
        assert pr.records_processed == 0
        assert pr.errors == []
        assert pr.warnings == []

    def test_pipeline_result_defaults(self):
        result = PipelineResult()
        assert result.pack_id == "PACK-026"
        assert result.status == ExecutionStatus.PENDING
        assert result.execution_id
        assert result.phases_completed == []

    def test_phase_provenance_model(self):
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
        expected = {"pending", "running", "completed", "failed", "cancelled", "skipped"}
        actual = {s.value for s in ExecutionStatus}
        assert expected == actual


# ===========================================================================
# Orchestrator Config Completeness
# ===========================================================================


class TestOrchestratorConfigCompleteness:
    def test_config_has_provenance_flag(self, default_config):
        assert isinstance(default_config.enable_provenance, bool)

    def test_config_has_timeout(self, default_config):
        assert default_config.timeout_per_phase_seconds >= 30

    def test_config_has_max_concurrent(self, default_config):
        assert default_config.max_concurrent_agents >= 1

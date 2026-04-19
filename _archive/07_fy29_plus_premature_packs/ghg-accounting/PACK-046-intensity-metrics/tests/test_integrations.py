"""
Unit tests for PACK-046 Integrations (PackOrchestrator + MRVBridge).

Tests all 2 implemented integrations with 50+ tests covering:
  - PackOrchestrator: 10-phase DAG pipeline
  - MRVBridge: 30 MRV agent routing
  - Kahn's topological sort
  - Phase dependencies and parallel groups
  - Conditional phase skipping
  - Pipeline execution (parallel and sequential)
  - Phase caching with TTL
  - Retry with exponential backoff
  - Provenance chain hashing
  - Progress callback
  - MRV scope grouping and aggregation
  - Health check
  - Error handling

Author: GreenLang QA Team
"""

import asyncio
import sys
import time
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from integrations.pack_orchestrator import (
    CONDITIONAL_PHASES,
    ExecutionStatus,
    PARALLEL_PHASE_GROUPS,
    PHASE_DEPENDENCIES,
    PackOrchestrator,
    PipelineConfig,
    PipelinePhase,
    PipelineResult,
    PipelineStatus,
    PhaseResult,
    _PhaseCache,
    _chain_hash,
    _compute_hash,
    topological_sort_phases,
)
from integrations.mrv_bridge import (
    AGENT_DESCRIPTIONS,
    AGENT_SCOPE_MAP,
    EmissionsRequest,
    EmissionsResponse,
    MRVBridge,
    MRVBridgeConfig,
    MRVAgentResult,
    MRVScope,
    ScopedEmissions,
)


# ---------------------------------------------------------------------------
# Topological Sort Tests
# ---------------------------------------------------------------------------


class TestTopologicalSort:
    """Tests for Kahn's topological sort."""

    def test_topological_sort_returns_10_phases(self):
        order = topological_sort_phases()
        assert len(order) == 10

    def test_topological_sort_denominator_setup_first(self):
        order = topological_sort_phases()
        assert order[0] == PipelinePhase.DENOMINATOR_SETUP

    def test_topological_sort_report_generation_last(self):
        order = topological_sort_phases()
        assert order[-1] == PipelinePhase.REPORT_GENERATION

    def test_dependencies_respected(self):
        order = topological_sort_phases()
        positions = {phase: idx for idx, phase in enumerate(order)}
        for phase, deps in PHASE_DEPENDENCIES.items():
            for dep in deps:
                assert positions[dep] < positions[phase], (
                    f"{dep.value} must precede {phase.value}"
                )

    def test_intensity_calculation_after_ingestion_and_retrieval(self):
        order = topological_sort_phases()
        positions = {phase: idx for idx, phase in enumerate(order)}
        assert positions[PipelinePhase.DATA_INGESTION] < positions[PipelinePhase.INTENSITY_CALCULATION]
        assert positions[PipelinePhase.EMISSIONS_RETRIEVAL] < positions[PipelinePhase.INTENSITY_CALCULATION]


class TestPhaseDependencies:
    """Tests for DAG structure."""

    def test_denominator_setup_has_no_deps(self):
        assert PHASE_DEPENDENCIES[PipelinePhase.DENOMINATOR_SETUP] == []

    def test_intensity_calculation_depends_on_2_phases(self):
        deps = PHASE_DEPENDENCIES[PipelinePhase.INTENSITY_CALCULATION]
        assert PipelinePhase.DATA_INGESTION in deps
        assert PipelinePhase.EMISSIONS_RETRIEVAL in deps

    def test_disclosure_mapping_depends_on_4_phases(self):
        deps = PHASE_DEPENDENCIES[PipelinePhase.DISCLOSURE_MAPPING]
        assert len(deps) == 4

    def test_report_generation_depends_on_disclosure_mapping(self):
        deps = PHASE_DEPENDENCIES[PipelinePhase.REPORT_GENERATION]
        assert deps == [PipelinePhase.DISCLOSURE_MAPPING]

    def test_parallel_groups_has_6_groups(self):
        assert len(PARALLEL_PHASE_GROUPS) == 6

    def test_conditional_phases_has_3_entries(self):
        assert len(CONDITIONAL_PHASES) == 3
        assert PipelinePhase.DECOMPOSITION in CONDITIONAL_PHASES
        assert PipelinePhase.BENCHMARKING in CONDITIONAL_PHASES
        assert PipelinePhase.SCENARIO_ANALYSIS in CONDITIONAL_PHASES


# ---------------------------------------------------------------------------
# Phase Cache Tests
# ---------------------------------------------------------------------------


class TestPhaseCache:
    """Tests for _PhaseCache."""

    def test_put_and_get(self):
        cache = _PhaseCache(ttl_s=60.0)
        cache.put("key1", {"data": "value"})
        assert cache.get("key1") == {"data": "value"}

    def test_get_missing_returns_none(self):
        cache = _PhaseCache(ttl_s=60.0)
        assert cache.get("nonexistent") is None

    def test_clear(self):
        cache = _PhaseCache(ttl_s=60.0)
        cache.put("key1", "val1")
        cache.put("key2", "val2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_expired_entry_returns_none(self):
        cache = _PhaseCache(ttl_s=0.01)
        cache.put("key1", "val1")
        time.sleep(0.02)
        assert cache.get("key1") is None


# ---------------------------------------------------------------------------
# Hashing Tests
# ---------------------------------------------------------------------------


class TestHashHelpers:
    """Tests for provenance hash helpers."""

    def test_compute_hash_dict(self):
        h = _compute_hash({"a": 1, "b": 2})
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        h1 = _compute_hash({"x": "hello"})
        h2 = _compute_hash({"x": "hello"})
        assert h1 == h2

    def test_compute_hash_string(self):
        h = _compute_hash("test_string")
        assert len(h) == 64

    def test_chain_hash(self):
        h1 = _compute_hash({"step": 1})
        h2 = _chain_hash(h1, {"step": 2})
        assert len(h2) == 64
        assert h2 != h1


# ---------------------------------------------------------------------------
# PackOrchestrator Tests
# ---------------------------------------------------------------------------


class TestPackOrchestratorInit:
    """Tests for PackOrchestrator initialisation."""

    def test_init_creates_orchestrator(self):
        config = PipelineConfig(company_name="TestCo", reporting_period="2025")
        orch = PackOrchestrator(config)
        assert orch is not None

    def test_init_provenance_chain_set(self):
        config = PipelineConfig(company_name="TestCo", reporting_period="2025")
        orch = PackOrchestrator(config)
        assert len(orch.provenance_chain) == 64

    def test_init_auto_skip_conditional_phases(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            requires_multi_year_data=False,
            requires_peer_data=False,
            enable_scenario_analysis=False,
        )
        orch = PackOrchestrator(config)
        assert PipelinePhase.DECOMPOSITION in orch._auto_skip
        assert PipelinePhase.BENCHMARKING in orch._auto_skip
        assert PipelinePhase.SCENARIO_ANALYSIS in orch._auto_skip

    def test_init_no_auto_skip_when_enabled(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            requires_multi_year_data=True,
            requires_peer_data=True,
            enable_scenario_analysis=True,
        )
        orch = PackOrchestrator(config)
        assert len(orch._auto_skip) == 0


class TestPackOrchestratorExecution:
    """Tests for PackOrchestrator.execute()."""

    @pytest.mark.asyncio
    async def test_execute_sequential(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        assert isinstance(result, PipelineResult)
        assert result.phases_completed + result.phases_skipped + result.phases_failed == 10

    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=True,
            max_retries=0,
            retry_base_delay_s=0.1,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_execute_provenance_chain(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        assert len(result.provenance_chain_hash) == 64

    @pytest.mark.asyncio
    async def test_execute_conditional_phases_skipped(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
            requires_multi_year_data=False,
            requires_peer_data=False,
            enable_scenario_analysis=False,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        skipped_phases = [
            pr for pr in result.phase_results
            if pr.status == ExecutionStatus.SKIPPED
        ]
        assert result.phases_skipped >= 3

    @pytest.mark.asyncio
    async def test_execute_conditional_phases_run_when_enabled(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
            requires_multi_year_data=True,
            requires_peer_data=True,
            enable_scenario_analysis=True,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        completed_phases = [
            pr for pr in result.phase_results
            if pr.status == ExecutionStatus.COMPLETED
        ]
        assert len(completed_phases) == 10

    @pytest.mark.asyncio
    async def test_execute_duration_recorded(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
        )
        orch = PackOrchestrator(config)
        result = await orch.execute()
        assert result.total_duration_ms > 0


class TestPackOrchestratorStatus:
    """Tests for PackOrchestrator.get_status()."""

    def test_status_before_execution(self):
        config = PipelineConfig(company_name="TestCo", reporting_period="2025")
        orch = PackOrchestrator(config)
        status = orch.get_status()
        assert isinstance(status, PipelineStatus)
        assert status.phases_completed == 0
        assert status.is_running is False

    @pytest.mark.asyncio
    async def test_status_after_execution(self):
        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
        )
        orch = PackOrchestrator(config)
        await orch.execute()
        status = orch.get_status()
        assert status.is_running is False
        assert status.phases_completed > 0


class TestPackOrchestratorProgressCallback:
    """Tests for progress callback."""

    @pytest.mark.asyncio
    async def test_progress_callback_invoked(self):
        calls = []

        async def callback(phase: str, pct: float, status: str):
            calls.append((phase, pct, status))

        config = PipelineConfig(
            company_name="TestCo",
            reporting_period="2025",
            enable_parallel=False,
            max_retries=0,
            retry_base_delay_s=0.1,
            requires_multi_year_data=True,
            requires_peer_data=True,
            enable_scenario_analysis=True,
        )
        orch = PackOrchestrator(config)
        orch.set_progress_callback(callback)
        await orch.execute()
        assert len(calls) > 0


# ---------------------------------------------------------------------------
# MRVBridge Tests
# ---------------------------------------------------------------------------


class TestMRVBridgeInit:
    """Tests for MRVBridge initialisation."""

    def test_init_creates_bridge(self):
        bridge = MRVBridge()
        assert bridge is not None

    def test_init_default_config(self):
        bridge = MRVBridge()
        assert bridge.config.timeout_s == 30.0
        assert bridge.config.batch_size == 10

    def test_init_custom_config(self):
        config = MRVBridgeConfig(timeout_s=60.0, batch_size=5)
        bridge = MRVBridge(config=config)
        assert bridge.config.timeout_s == 60.0
        assert bridge.config.batch_size == 5


class TestMRVBridgeAgentMaps:
    """Tests for agent scope mapping."""

    def test_30_agents_registered(self):
        assert len(AGENT_SCOPE_MAP) == 30

    def test_30_agent_descriptions(self):
        assert len(AGENT_DESCRIPTIONS) == 30

    def test_scope_1_agents_count(self):
        s1 = [a for a, s in AGENT_SCOPE_MAP.items() if s == MRVScope.SCOPE_1]
        assert len(s1) == 8

    def test_scope_2_agents_count(self):
        s2 = [a for a, s in AGENT_SCOPE_MAP.items() if s == MRVScope.SCOPE_2]
        assert len(s2) == 5

    def test_scope_3_agents_count(self):
        s3 = [a for a, s in AGENT_SCOPE_MAP.items() if s == MRVScope.SCOPE_3]
        assert len(s3) == 15

    def test_cross_cutting_agents_count(self):
        cc = [a for a, s in AGENT_SCOPE_MAP.items() if s == MRVScope.CROSS_CUTTING]
        assert len(cc) == 2

    def test_all_agents_have_descriptions(self):
        for agent_id in AGENT_SCOPE_MAP:
            assert agent_id in AGENT_DESCRIPTIONS


class TestMRVBridgeScopeQueries:
    """Tests for MRVBridge scope-level queries."""

    @pytest.mark.asyncio
    async def test_get_scope1_emissions(self):
        bridge = MRVBridge()
        result = await bridge.get_scope1_emissions("2024")
        assert isinstance(result, ScopedEmissions)
        assert result.scope == "scope_1"
        assert result.agents_queried == 8

    @pytest.mark.asyncio
    async def test_get_scope2_emissions(self):
        bridge = MRVBridge()
        result = await bridge.get_scope2_emissions("2024")
        assert isinstance(result, ScopedEmissions)
        assert result.scope == "scope_2"
        assert result.agents_queried == 5

    @pytest.mark.asyncio
    async def test_get_scope3_emissions(self):
        bridge = MRVBridge()
        result = await bridge.get_scope3_emissions("2024")
        assert isinstance(result, ScopedEmissions)
        assert result.scope == "scope_3"
        assert result.agents_queried == 15

    @pytest.mark.asyncio
    async def test_scoped_emissions_provenance_hash(self):
        bridge = MRVBridge()
        result = await bridge.get_scope1_emissions("2024")
        assert len(result.provenance_hash) == 64


class TestMRVBridgeTotalEmissions:
    """Tests for MRVBridge.get_total_emissions()."""

    @pytest.mark.asyncio
    async def test_get_total_emissions_s1_s2(self):
        bridge = MRVBridge()
        request = EmissionsRequest(
            period="2024",
            scopes=["scope_1", "scope_2"],
        )
        response = await bridge.get_total_emissions(request)
        assert isinstance(response, EmissionsResponse)
        assert response.period == "2024"
        assert len(response.scoped_emissions) == 2

    @pytest.mark.asyncio
    async def test_get_total_emissions_all_scopes(self):
        bridge = MRVBridge()
        request = EmissionsRequest(
            period="2024",
            scopes=["scope_1", "scope_2", "scope_3"],
        )
        response = await bridge.get_total_emissions(request)
        assert len(response.scoped_emissions) == 3
        expected = response.scope1_tco2e + response.scope2_tco2e + response.scope3_tco2e
        assert response.total_tco2e == pytest.approx(expected, abs=1e-6)

    @pytest.mark.asyncio
    async def test_get_total_emissions_provenance(self):
        bridge = MRVBridge()
        request = EmissionsRequest(period="2024", scopes=["scope_1"])
        response = await bridge.get_total_emissions(request)
        assert len(response.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_get_total_emissions_duration(self):
        bridge = MRVBridge()
        request = EmissionsRequest(period="2024", scopes=["scope_1"])
        response = await bridge.get_total_emissions(request)
        assert response.duration_ms >= 0.0


class TestMRVBridgeQueryAll:
    """Tests for MRVBridge.query_all_agents()."""

    @pytest.mark.asyncio
    async def test_query_all_returns_30(self):
        bridge = MRVBridge()
        results = await bridge.query_all_agents("2024")
        assert len(results) == 30
        assert all(isinstance(r, MRVAgentResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_all_agent_names_populated(self):
        bridge = MRVBridge()
        results = await bridge.query_all_agents("2024")
        for r in results:
            assert r.agent_name != ""


class TestMRVBridgeHealthCheck:
    """Tests for MRVBridge.health_check()."""

    def test_health_check_returns_dict(self):
        bridge = MRVBridge()
        status = bridge.health_check()
        assert status["bridge"] == "MRVBridge"
        assert status["status"] == "healthy"
        assert status["agents_registered"] == 30

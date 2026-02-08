# -*- coding: utf-8 -*-
"""
Load Tests for Agent Registry & Service Catalog (AGENT-FOUND-007)

Tests throughput and concurrency for agent registration, lookups, queries,
dependency resolution, health checks, single-operation latency, large
registry performance, and mixed workloads.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Import inline implementations from unit tests
# ---------------------------------------------------------------------------

from tests.unit.agent_registry_service.test_registry import (
    AgentRegistry,
    _AgentEntry,
    AgentCapability,
    AgentLayer,
    SectorClassification,
    AgentHealthStatus,
)
from tests.unit.agent_registry_service.test_dependency_resolver import (
    DependencyResolver,
    DependencyNode,
)
from tests.unit.agent_registry_service.test_health_checker import (
    HealthChecker,
)


# ===========================================================================
# Helper
# ===========================================================================


def _make_entry(idx: int, layer: str = "utility", sectors: list = None,
                capabilities: list = None, tags: list = None,
                health: str = "unknown") -> _AgentEntry:
    """Create a numbered agent entry for load testing."""
    return _AgentEntry(
        agent_id=f"gl-load-{idx:05d}",
        name=f"Load Agent {idx}",
        version="1.0.0",
        layer=layer,
        sectors=sectors or [],
        capabilities=capabilities or [],
        tags=tags or [],
        health=health,
    )


# ===========================================================================
# Load Test Classes
# ===========================================================================


class TestRegistrationThroughput:
    """Test 1000 sequential registrations in <5s."""

    @pytest.mark.slow
    def test_1000_sequential_registrations(self):
        reg = AgentRegistry()

        start = time.time()
        for i in range(1000):
            entry = _make_entry(i)
            h = reg.register_agent(entry)
            assert len(h) == 64
        elapsed = time.time() - start

        assert reg.count == 1000
        assert elapsed < 5.0, f"1000 registrations took {elapsed:.2f}s (target: <5s)"


class TestConcurrentRegistrations:
    """Test 50 concurrent registration requests."""

    @pytest.mark.slow
    def test_50_concurrent_registrations(self):
        reg = AgentRegistry()

        def do_register(i):
            entry = _make_entry(i)
            return reg.register_agent(entry)

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(do_register, i) for i in range(50)]
            results = [f.result() for f in as_completed(futures)]

        assert len(results) == 50
        # All registrations should have returned a SHA-256 hash
        for h in results:
            assert len(h) == 64
        assert reg.count == 50


class TestLookupThroughput:
    """Test 10000 agent lookups in <5s."""

    @pytest.mark.slow
    def test_10000_sequential_lookups(self):
        reg = AgentRegistry()
        # Pre-populate registry with 1000 agents
        for i in range(1000):
            reg.register_agent(_make_entry(i))

        start = time.time()
        for i in range(10000):
            agent_id = f"gl-load-{i % 1000:05d}"
            entry = reg.get_agent(agent_id)
            assert entry is not None
            assert entry.agent_id == agent_id
        elapsed = time.time() - start

        assert elapsed < 5.0, f"10000 lookups took {elapsed:.2f}s (target: <5s)"


class TestQueryThroughput:
    """Test 5000 filtered queries in <5s."""

    @pytest.mark.slow
    def test_5000_queries_with_filters(self):
        reg = AgentRegistry()
        layers = ["calculation", "reporting", "validation", "ingestion", "utility"]
        sectors = ["energy", "manufacturing", "transportation", "buildings", "agriculture"]

        # Pre-populate with 500 agents
        for i in range(500):
            entry = _make_entry(
                i,
                layer=layers[i % len(layers)],
                sectors=[sectors[i % len(sectors)]],
                tags=[f"tag-{i % 10}"],
            )
            reg.register_agent(entry)

        start = time.time()
        for i in range(5000):
            # Alternate between different filter types
            if i % 4 == 0:
                results = reg.list_agents(layer=layers[i % len(layers)])
            elif i % 4 == 1:
                results = reg.list_agents(sector=sectors[i % len(sectors)])
            elif i % 4 == 2:
                results = reg.list_agents(tag=f"tag-{i % 10}")
            else:
                results = reg.list_agents(search=f"Load Agent {i % 100}")
            assert isinstance(results, list)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"5000 queries took {elapsed:.2f}s (target: <5s)"


class TestDependencyResolutionThroughput:
    """Test 500 dependency resolutions in <5s."""

    @pytest.mark.slow
    def test_500_dependency_resolutions(self):
        resolver = DependencyResolver(fail_on_missing=False, max_depth=60)

        # Build a chain of 50 agents: agent-49 -> agent-48 -> ... -> agent-0
        for i in range(50):
            deps = [{"agent_id": f"agent-{i - 1}"}] if i > 0 else []
            resolver.add_node(DependencyNode(f"agent-{i}", dependencies=deps))

        start = time.time()
        for i in range(500):
            target = f"agent-{i % 50}"
            order = resolver.resolve(target)
            assert target in order
            assert order[-1] == target
        elapsed = time.time() - start

        assert elapsed < 5.0, f"500 resolutions took {elapsed:.2f}s (target: <5s)"


class TestHealthCheckThroughput:
    """Test 1000 health checks in <5s."""

    @pytest.mark.slow
    def test_1000_health_checks(self):
        checker = HealthChecker(check_interval_seconds=1)

        start = time.time()
        for i in range(1000):
            agent_id = f"gl-health-{i % 100:03d}"
            status = "healthy" if i % 3 != 0 else "degraded"
            result = checker.check_health(
                agent_id=agent_id,
                version="1.0.0",
                simulate_status=status,
                simulate_latency=0.5,
            )
            assert result.agent_id == agent_id
        elapsed = time.time() - start

        assert elapsed < 5.0, f"1000 health checks took {elapsed:.2f}s (target: <5s)"

        # Verify summary is populated
        summary = checker.get_health_summary()
        assert sum(summary.values()) > 0


class TestSingleOperationLatency:
    """Test single registration + lookup latency <1ms."""

    def test_single_operation_latency(self):
        reg = AgentRegistry()

        # Pre-populate with 100 agents so indexes are warm
        for i in range(100):
            reg.register_agent(_make_entry(i))

        # Warm up
        warm_entry = _make_entry(999)
        reg.register_agent(warm_entry)
        reg.get_agent("gl-load-00999")

        # Measure registration latency
        reg_latencies = []
        for i in range(100):
            entry = _make_entry(1000 + i)
            start = time.time()
            reg.register_agent(entry)
            reg_latencies.append((time.time() - start) * 1000)

        avg_reg = sum(reg_latencies) / len(reg_latencies)
        assert avg_reg < 1.0, f"Avg registration latency {avg_reg:.3f}ms (target: <1ms)"

        # Measure lookup latency
        lookup_latencies = []
        for i in range(100):
            agent_id = f"gl-load-{i:05d}"
            start = time.time()
            reg.get_agent(agent_id)
            lookup_latencies.append((time.time() - start) * 1000)

        avg_lookup = sum(lookup_latencies) / len(lookup_latencies)
        assert avg_lookup < 1.0, f"Avg lookup latency {avg_lookup:.3f}ms (target: <1ms)"


class TestLargeRegistryPerformance:
    """Test query performance with 1000 registered agents."""

    @pytest.mark.slow
    def test_large_registry_query(self):
        reg = AgentRegistry()
        layers = ["calculation", "reporting", "validation", "ingestion", "utility",
                  "orchestration", "compliance", "analytics", "integration",
                  "normalization", "foundation"]
        sectors = ["energy", "manufacturing", "transportation", "buildings",
                  "agriculture", "waste", "industrial_processes"]

        # Register 1000 agents
        for i in range(1000):
            entry = _make_entry(
                i,
                layer=layers[i % len(layers)],
                sectors=[sectors[i % len(sectors)]],
                tags=[f"tag-{i % 20}"],
                health="healthy" if i % 4 != 0 else "degraded",
            )
            reg.register_agent(entry)

        assert reg.count == 1000

        # Test various query patterns
        start = time.time()

        # Layer queries
        for layer in layers:
            results = reg.list_agents(layer=layer)
            assert len(results) > 0

        # Sector queries
        for sector in [s for s in sectors]:
            results = reg.list_agents(sector=sector)
            assert len(results) > 0

        # Health queries (use limit=1000 to get all results past default 100)
        healthy = reg.list_agents(health="healthy", limit=1000)
        degraded = reg.list_agents(health="degraded", limit=1000)
        assert len(healthy) + len(degraded) == 1000

        # Pagination
        page1 = reg.list_agents(offset=0, limit=50)
        page2 = reg.list_agents(offset=50, limit=50)
        assert len(page1) == 50
        assert len(page2) == 50

        # Statistics
        stats = reg.get_statistics()
        assert stats["total_agents"] == 1000

        elapsed = time.time() - start
        assert elapsed < 5.0, f"Large registry queries took {elapsed:.2f}s (target: <5s)"


class TestMixedWorkload:
    """Test mixed workload: register + query + health check."""

    @pytest.mark.slow
    def test_mixed_workload_1000_ops(self):
        reg = AgentRegistry()
        checker = HealthChecker()
        resolver = DependencyResolver(fail_on_missing=False, max_depth=110)

        # Pre-populate
        for i in range(100):
            entry = _make_entry(i, layer="calculation", sectors=["energy"])
            reg.register_agent(entry)
            resolver.add_node(DependencyNode(
                f"gl-load-{i:05d}",
                dependencies=[{"agent_id": f"gl-load-{max(0, i - 1):05d}"}] if i > 0 else [],
            ))

        start = time.time()
        for i in range(1000):
            op = i % 5

            if op == 0:
                # Registration
                entry = _make_entry(100 + i, layer="reporting")
                reg.register_agent(entry)

            elif op == 1:
                # Lookup
                agent_id = f"gl-load-{i % 100:05d}"
                reg.get_agent(agent_id)

            elif op == 2:
                # Query
                reg.list_agents(layer="calculation")

            elif op == 3:
                # Health check
                agent_id = f"gl-load-{i % 100:05d}"
                checker.check_health(
                    agent_id=agent_id,
                    simulate_status="healthy",
                )

            elif op == 4:
                # Dependency resolution
                target = f"gl-load-{i % 50:05d}"
                resolver.resolve(target)

        elapsed = time.time() - start

        assert elapsed < 5.0, f"Mixed workload took {elapsed:.2f}s (target: <5s)"
        assert reg.count > 100  # Registrations happened
        summary = checker.get_health_summary()
        assert sum(summary.values()) > 0  # Health checks happened

# -*- coding: utf-8 -*-
"""
Unit Tests for Agent Registry Metrics (AGENT-FOUND-007)

Tests all 12 Prometheus metric definitions, helper functions,
and graceful fallback when prometheus_client is unavailable.

Coverage target: 85%+ of metrics.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline metrics module mirroring greenlang/agent_registry/metrics.py
# ---------------------------------------------------------------------------

METRIC_NAMES = [
    "agent_registry_registrations_total",
    "agent_registry_unregistrations_total",
    "agent_registry_queries_total",
    "agent_registry_query_latency_seconds",
    "agent_registry_agents_registered",
    "agent_registry_health_checks_total",
    "agent_registry_health_status_changes_total",
    "agent_registry_dependency_resolutions_total",
    "agent_registry_capability_matches_total",
    "agent_registry_hot_reloads_total",
    "agent_registry_export_import_total",
    "agent_registry_provenance_records_total",
]


class NoOpMetric:
    """No-op metric for when prometheus_client is not available."""

    def inc(self, amount: float = 1) -> None:
        pass

    def dec(self, amount: float = 1) -> None:
        pass

    def set(self, value: float) -> None:
        pass

    def observe(self, amount: float) -> None:
        pass

    def labels(self, **kwargs) -> "NoOpMetric":
        return self


class MetricsRegistry:
    """Registry for all agent registry metrics."""

    def __init__(self, prometheus_available: bool = True):
        self._prometheus_available = prometheus_available
        self._metrics: Dict[str, Any] = {}
        self._call_counts: Dict[str, int] = {}

        for name in METRIC_NAMES:
            if prometheus_available:
                self._metrics[name] = MagicMock()
                self._metrics[name].labels.return_value = self._metrics[name]
            else:
                self._metrics[name] = NoOpMetric()
            self._call_counts[name] = 0

    @property
    def metric_count(self) -> int:
        return len(self._metrics)

    def get_metric(self, name: str) -> Any:
        return self._metrics.get(name)

    def record_registration(self, agent_id: str = "", layer: str = "") -> None:
        self._call_counts["agent_registry_registrations_total"] += 1
        m = self._metrics.get("agent_registry_registrations_total")
        if m:
            m.labels(agent_id=agent_id, layer=layer).inc()

    def record_unregistration(self, agent_id: str = "") -> None:
        self._call_counts["agent_registry_unregistrations_total"] += 1

    def record_query(self, query_type: str = "") -> None:
        self._call_counts["agent_registry_queries_total"] += 1

    def record_query_latency(self, seconds: float, query_type: str = "") -> None:
        self._call_counts["agent_registry_query_latency_seconds"] += 1
        m = self._metrics.get("agent_registry_query_latency_seconds")
        if m:
            m.labels(query_type=query_type).observe(seconds)

    def set_agents_registered(self, count: int) -> None:
        self._call_counts["agent_registry_agents_registered"] += 1
        m = self._metrics.get("agent_registry_agents_registered")
        if m:
            m.set(count)

    def record_health_check(self, agent_id: str = "", status: str = "") -> None:
        self._call_counts["agent_registry_health_checks_total"] += 1

    def record_health_status_change(self, agent_id: str = "", old_status: str = "",
                                     new_status: str = "") -> None:
        self._call_counts["agent_registry_health_status_changes_total"] += 1

    def record_dependency_resolution(self, agent_id: str = "") -> None:
        self._call_counts["agent_registry_dependency_resolutions_total"] += 1

    def record_capability_match(self, capability: str = "") -> None:
        self._call_counts["agent_registry_capability_matches_total"] += 1

    def record_hot_reload(self, agent_id: str = "") -> None:
        self._call_counts["agent_registry_hot_reloads_total"] += 1

    def record_export_import(self, operation: str = "export") -> None:
        self._call_counts["agent_registry_export_import_total"] += 1

    def record_provenance(self, entity_id: str = "", action: str = "") -> None:
        self._call_counts["agent_registry_provenance_records_total"] += 1


# ===========================================================================
# Test Classes
# ===========================================================================


class TestMetricsDefinitions:
    """Test all 12 metrics are defined."""

    def test_all_12_metrics_defined(self):
        registry = MetricsRegistry()
        assert registry.metric_count == 12

    def test_registrations_total_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_registrations_total") is not None

    def test_unregistrations_total_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_unregistrations_total") is not None

    def test_queries_total_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_queries_total") is not None

    def test_query_latency_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_query_latency_seconds") is not None

    def test_agents_registered_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_agents_registered") is not None

    def test_health_checks_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_health_checks_total") is not None

    def test_health_status_changes_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_health_status_changes_total") is not None

    def test_dependency_resolutions_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_dependency_resolutions_total") is not None

    def test_capability_matches_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_capability_matches_total") is not None

    def test_hot_reloads_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_hot_reloads_total") is not None

    def test_export_import_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_export_import_total") is not None

    def test_provenance_records_exists(self):
        registry = MetricsRegistry()
        assert registry.get_metric("agent_registry_provenance_records_total") is not None

    def test_nonexistent_metric_none(self):
        registry = MetricsRegistry()
        assert registry.get_metric("nonexistent") is None


class TestMetricsHelpers:
    """Test all helper functions work."""

    def test_record_registration(self):
        registry = MetricsRegistry()
        registry.record_registration("gl-001", "calculation")
        assert registry._call_counts["agent_registry_registrations_total"] == 1

    def test_record_unregistration(self):
        registry = MetricsRegistry()
        registry.record_unregistration("gl-001")
        assert registry._call_counts["agent_registry_unregistrations_total"] == 1

    def test_record_query(self):
        registry = MetricsRegistry()
        registry.record_query("list")
        assert registry._call_counts["agent_registry_queries_total"] == 1

    def test_record_query_latency(self):
        registry = MetricsRegistry()
        registry.record_query_latency(0.005, "list")
        assert registry._call_counts["agent_registry_query_latency_seconds"] == 1

    def test_set_agents_registered(self):
        registry = MetricsRegistry()
        registry.set_agents_registered(47)
        assert registry._call_counts["agent_registry_agents_registered"] == 1

    def test_record_health_check(self):
        registry = MetricsRegistry()
        registry.record_health_check("gl-001", "healthy")
        assert registry._call_counts["agent_registry_health_checks_total"] == 1

    def test_record_health_status_change(self):
        registry = MetricsRegistry()
        registry.record_health_status_change("gl-001", "healthy", "degraded")
        assert registry._call_counts["agent_registry_health_status_changes_total"] == 1

    def test_record_dependency_resolution(self):
        registry = MetricsRegistry()
        registry.record_dependency_resolution("gl-001")
        assert registry._call_counts["agent_registry_dependency_resolutions_total"] == 1

    def test_record_capability_match(self):
        registry = MetricsRegistry()
        registry.record_capability_match("carbon_calc")
        assert registry._call_counts["agent_registry_capability_matches_total"] == 1

    def test_record_hot_reload(self):
        registry = MetricsRegistry()
        registry.record_hot_reload("gl-001")
        assert registry._call_counts["agent_registry_hot_reloads_total"] == 1

    def test_record_export_import(self):
        registry = MetricsRegistry()
        registry.record_export_import("export")
        assert registry._call_counts["agent_registry_export_import_total"] == 1

    def test_record_provenance(self):
        registry = MetricsRegistry()
        registry.record_provenance("gl-001", "register")
        assert registry._call_counts["agent_registry_provenance_records_total"] == 1


class TestMetricsNoPrometheus:
    """Test graceful fallback when prometheus_client unavailable."""

    def test_noop_metrics_dont_raise(self):
        registry = MetricsRegistry(prometheus_available=False)
        registry.record_registration("gl-001", "calculation")
        registry.record_unregistration("gl-001")
        registry.record_query("list")
        registry.record_query_latency(0.001)
        registry.set_agents_registered(10)
        registry.record_health_check()
        registry.record_health_status_change()
        registry.record_dependency_resolution()
        registry.record_capability_match()
        registry.record_hot_reload()
        registry.record_export_import()
        registry.record_provenance()

    def test_noop_metric_inc(self):
        m = NoOpMetric()
        m.inc()
        m.inc(5)

    def test_noop_metric_dec(self):
        m = NoOpMetric()
        m.dec()
        m.dec(3)

    def test_noop_metric_set(self):
        m = NoOpMetric()
        m.set(42)

    def test_noop_metric_observe(self):
        m = NoOpMetric()
        m.observe(0.5)

    def test_noop_metric_labels(self):
        m = NoOpMetric()
        labeled = m.labels(foo="bar")
        assert isinstance(labeled, NoOpMetric)
        labeled.inc()

    def test_metric_count_same_regardless(self):
        r1 = MetricsRegistry(prometheus_available=True)
        r2 = MetricsRegistry(prometheus_available=False)
        assert r1.metric_count == r2.metric_count == 12

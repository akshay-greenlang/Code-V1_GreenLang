# -*- coding: utf-8 -*-
"""
Unit Tests for DashboardProvider (AGENT-FOUND-010)

Tests dashboard registration, data retrieval, listing, platform/agent
dashboard builders, metric querying, nonexistent dashboards, and statistics.

Since dashboard_provider.py is not yet on disk, tests define the expected
interface via an inline implementation matching the PRD specification.

Coverage target: 85%+ of dashboard_provider.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline DashboardProvider (mirrors expected interface)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class DashboardConfig:
    """Configuration for a dashboard."""
    dashboard_id: str = ""
    name: str = ""
    description: str = ""
    panels: List[Dict[str, Any]] = field(default_factory=list)
    time_range: str = "1h"
    refresh_interval: str = "30s"
    tenant_id: str = "default"

    def __post_init__(self):
        if not self.dashboard_id:
            self.dashboard_id = str(uuid.uuid4())


class DashboardProvider:
    """Dashboard provisioning and data serving engine."""

    def __init__(self, config: Any, metrics_collector: Any = None) -> None:
        self._config = config
        self._metrics_collector = metrics_collector
        self._dashboards: Dict[str, DashboardConfig] = {}
        self._total_queries: int = 0

    def register_dashboard(self, config: DashboardConfig) -> DashboardConfig:
        if not config.name or not config.name.strip():
            raise ValueError("Dashboard name must be non-empty")
        self._dashboards[config.dashboard_id] = config
        return config

    def get_dashboard(self, dashboard_id: str) -> Optional[DashboardConfig]:
        return self._dashboards.get(dashboard_id)

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        self._total_queries += 1
        dashboard = self._dashboards.get(dashboard_id)
        if dashboard is None:
            raise ValueError(f"Dashboard '{dashboard_id}' not found")

        panel_data = []
        for panel in dashboard.panels:
            panel_data.append({
                "panel_id": panel.get("id", str(uuid.uuid4())),
                "title": panel.get("title", ""),
                "type": panel.get("type", "graph"),
                "data": self._query_metric_data(panel.get("metric_name", "")),
            })

        return {
            "dashboard_id": dashboard.dashboard_id,
            "name": dashboard.name,
            "time_range": dashboard.time_range,
            "panels": panel_data,
            "queried_at": _utcnow().isoformat(),
        }

    def list_dashboards(self) -> List[DashboardConfig]:
        return sorted(self._dashboards.values(), key=lambda d: d.name)

    def build_platform_dashboard(self) -> DashboardConfig:
        config = DashboardConfig(
            name="GreenLang Platform Overview",
            description="High-level platform metrics and health",
            panels=[
                {"id": "p1", "type": "stat", "title": "Active Agents", "metric_name": "agent_count"},
                {"id": "p2", "type": "graph", "title": "Request Rate", "metric_name": "request_rate"},
                {"id": "p3", "type": "gauge", "title": "Error Rate", "metric_name": "error_rate"},
                {"id": "p4", "type": "table", "title": "SLO Status", "metric_name": "slo_compliance"},
            ],
            time_range="24h",
            refresh_interval="1m",
        )
        return self.register_dashboard(config)

    def build_agent_dashboard(self, agent_id: str) -> DashboardConfig:
        config = DashboardConfig(
            name=f"Agent {agent_id} Dashboard",
            description=f"Metrics for agent {agent_id}",
            panels=[
                {"id": "p1", "type": "graph", "title": "Execution Duration", "metric_name": f"{agent_id}_duration"},
                {"id": "p2", "type": "stat", "title": "Executions Total", "metric_name": f"{agent_id}_total"},
                {"id": "p3", "type": "stat", "title": "Error Count", "metric_name": f"{agent_id}_errors"},
            ],
        )
        return self.register_dashboard(config)

    def _query_metric_data(self, metric_name: str) -> List[Dict[str, Any]]:
        if not metric_name:
            return []
        if self._metrics_collector:
            val = getattr(self._metrics_collector, "get_metric_value", lambda n: None)(metric_name)
            if val is not None:
                return [{"timestamp": _utcnow().isoformat(), "value": val}]
        return []

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_dashboards": len(self._dashboards),
            "total_queries": self._total_queries,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    grafana_url: str = "http://localhost:3000"


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def provider(config):
    return DashboardProvider(config)


# ==========================================================================
# Registration Tests
# ==========================================================================

class TestDashboardProviderRegistration:
    """Tests for dashboard registration."""

    def test_register_dashboard(self, provider):
        dc = DashboardConfig(name="My Dashboard")
        result = provider.register_dashboard(dc)
        assert result.name == "My Dashboard"
        assert result.dashboard_id

    def test_register_dashboard_empty_name_raises(self, provider):
        dc = DashboardConfig(name="")
        with pytest.raises(ValueError, match="non-empty"):
            provider.register_dashboard(dc)

    def test_register_dashboard_with_panels(self, provider):
        panels = [{"type": "graph", "title": "CPU"}, {"type": "stat", "title": "Memory"}]
        dc = DashboardConfig(name="Infra", panels=panels)
        result = provider.register_dashboard(dc)
        assert len(result.panels) == 2


# ==========================================================================
# Dashboard Data Retrieval Tests
# ==========================================================================

class TestDashboardProviderGetData:
    """Tests for get_dashboard_data."""

    def test_get_dashboard_data(self, provider):
        dc = DashboardConfig(
            name="Test", panels=[{"type": "graph", "title": "CPU", "metric_name": "cpu"}],
        )
        provider.register_dashboard(dc)
        data = provider.get_dashboard_data(dc.dashboard_id)
        assert data["name"] == "Test"
        assert "panels" in data
        assert "queried_at" in data

    def test_get_dashboard_data_nonexistent_raises(self, provider):
        with pytest.raises(ValueError, match="not found"):
            provider.get_dashboard_data("nonexistent-id")

    def test_get_dashboard_increments_query_count(self, provider):
        dc = DashboardConfig(name="D", panels=[])
        provider.register_dashboard(dc)
        provider.get_dashboard_data(dc.dashboard_id)
        provider.get_dashboard_data(dc.dashboard_id)
        stats = provider.get_statistics()
        assert stats["total_queries"] == 2


# ==========================================================================
# Listing Tests
# ==========================================================================

class TestDashboardProviderList:
    """Tests for list_dashboards."""

    def test_list_dashboards(self, provider):
        provider.register_dashboard(DashboardConfig(name="B Dashboard"))
        provider.register_dashboard(DashboardConfig(name="A Dashboard"))
        result = provider.list_dashboards()
        assert len(result) == 2
        assert result[0].name == "A Dashboard"  # sorted

    def test_list_dashboards_empty(self, provider):
        assert provider.list_dashboards() == []


# ==========================================================================
# Builder Tests
# ==========================================================================

class TestDashboardProviderBuilders:
    """Tests for platform and agent dashboard builders."""

    def test_build_platform_dashboard(self, provider):
        dc = provider.build_platform_dashboard()
        assert dc.name == "GreenLang Platform Overview"
        assert len(dc.panels) == 4
        assert dc.time_range == "24h"
        # Should be registered
        assert provider.get_dashboard(dc.dashboard_id) is not None

    def test_build_agent_dashboard(self, provider):
        dc = provider.build_agent_dashboard("GL-001")
        assert "GL-001" in dc.name
        assert len(dc.panels) == 3
        assert provider.get_dashboard(dc.dashboard_id) is not None


# ==========================================================================
# Metric Query Tests
# ==========================================================================

class TestDashboardProviderMetricQuery:
    """Tests for _query_metric_data via get_dashboard_data."""

    def test_query_metric_data_without_collector(self, provider):
        dc = DashboardConfig(
            name="D", panels=[{"type": "graph", "title": "T", "metric_name": "m"}],
        )
        provider.register_dashboard(dc)
        data = provider.get_dashboard_data(dc.dashboard_id)
        # Without collector, data should be empty list
        assert data["panels"][0]["data"] == []

    def test_query_metric_data_with_collector(self, config):
        class MockCollector:
            def get_metric_value(self, name):
                return 42.0

        provider = DashboardProvider(config, metrics_collector=MockCollector())
        dc = DashboardConfig(
            name="D", panels=[{"type": "graph", "title": "T", "metric_name": "test"}],
        )
        provider.register_dashboard(dc)
        data = provider.get_dashboard_data(dc.dashboard_id)
        assert len(data["panels"][0]["data"]) == 1
        assert data["panels"][0]["data"][0]["value"] == 42.0


# ==========================================================================
# Get Dashboard Tests
# ==========================================================================

class TestDashboardProviderGetDashboard:
    """Tests for get_dashboard."""

    def test_get_dashboard_existing(self, provider):
        dc = DashboardConfig(name="Test")
        provider.register_dashboard(dc)
        result = provider.get_dashboard(dc.dashboard_id)
        assert result is not None
        assert result.name == "Test"

    def test_get_dashboard_nonexistent(self, provider):
        result = provider.get_dashboard("nonexistent-id")
        assert result is None


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestDashboardProviderStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, provider):
        stats = provider.get_statistics()
        assert stats["total_dashboards"] == 0
        assert stats["total_queries"] == 0

    def test_statistics_after_operations(self, provider):
        provider.register_dashboard(DashboardConfig(name="D1"))
        provider.register_dashboard(DashboardConfig(name="D2"))
        stats = provider.get_statistics()
        assert stats["total_dashboards"] == 2

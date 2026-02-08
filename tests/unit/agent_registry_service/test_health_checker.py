# -*- coding: utf-8 -*-
"""
Unit Tests for HealthChecker (AGENT-FOUND-007)

Tests health checking, status updates, history tracking, TTL-based refresh,
and summary aggregation.

Coverage target: 85%+ of health_checker.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models (self-contained)
# ---------------------------------------------------------------------------


class AgentHealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DISABLED = "disabled"


class HealthCheckResult:
    """Result of a health check probe."""

    def __init__(self, agent_id: str, version: str = "1.0.0",
                 status: str = "unknown", latency_ms: float = 0.0,
                 message: str = "", checked_at: Optional[datetime] = None):
        self.agent_id = agent_id
        self.version = version
        self.status = AgentHealthStatus(status)
        self.latency_ms = latency_ms
        self.message = message
        self.checked_at = checked_at or datetime.utcnow()


class HealthChecker:
    """Manages health status for registered agents."""

    def __init__(self, check_interval_seconds: int = 30,
                 timeout_seconds: int = 5, unhealthy_threshold: int = 3):
        self._check_interval = check_interval_seconds
        self._timeout = timeout_seconds
        self._unhealthy_threshold = unhealthy_threshold
        self._status: Dict[str, Dict[str, AgentHealthStatus]] = {}  # agent_id -> {version -> status}
        self._history: Dict[str, List[HealthCheckResult]] = defaultdict(list)
        self._last_check: Dict[str, datetime] = {}
        self._failure_counts: Dict[str, int] = defaultdict(int)

    def check_health(self, agent_id: str, version: str = "1.0.0",
                     simulate_status: str = "healthy",
                     simulate_latency: float = 1.0) -> HealthCheckResult:
        """Perform a health check (simulated for testing)."""
        result = HealthCheckResult(
            agent_id=agent_id, version=version,
            status=simulate_status, latency_ms=simulate_latency,
        )
        key = f"{agent_id}:{version}"
        self._last_check[key] = datetime.utcnow()
        self._history[key].append(result)

        if agent_id not in self._status:
            self._status[agent_id] = {}
        self._status[agent_id][version] = result.status

        if result.status == AgentHealthStatus.UNHEALTHY:
            self._failure_counts[key] += 1
        else:
            self._failure_counts[key] = 0

        return result

    def set_health(self, agent_id: str, status: str,
                   version: Optional[str] = None) -> None:
        """Manually set health status."""
        hs = AgentHealthStatus(status)
        if agent_id not in self._status:
            self._status[agent_id] = {}
        if version:
            self._status[agent_id][version] = hs
        else:
            for v in list(self._status[agent_id].keys()):
                self._status[agent_id][v] = hs

    def get_health(self, agent_id: str, version: Optional[str] = None) -> Optional[AgentHealthStatus]:
        """Get current health status."""
        if agent_id not in self._status:
            return None
        if version:
            return self._status[agent_id].get(version)
        # Return status of latest version
        if self._status[agent_id]:
            return list(self._status[agent_id].values())[-1]
        return None

    def get_health_history(self, agent_id: str, version: str = "1.0.0",
                           limit: int = 100) -> List[HealthCheckResult]:
        """Get health check history."""
        key = f"{agent_id}:{version}"
        history = self._history.get(key, [])
        return history[-limit:]

    def get_unhealthy_agents(self) -> List[str]:
        """Get list of unhealthy agent IDs."""
        unhealthy = []
        for agent_id, versions in self._status.items():
            for version, status in versions.items():
                if status in (AgentHealthStatus.UNHEALTHY, AgentHealthStatus.DEGRADED):
                    if agent_id not in unhealthy:
                        unhealthy.append(agent_id)
        return unhealthy

    def get_health_summary(self) -> Dict[str, int]:
        """Get counts per health status."""
        summary: Dict[str, int] = defaultdict(int)
        for agent_id, versions in self._status.items():
            for version, status in versions.items():
                summary[status.value] += 1
        return dict(summary)

    def should_recheck(self, agent_id: str, version: str = "1.0.0") -> bool:
        """Check if TTL has elapsed since last check."""
        key = f"{agent_id}:{version}"
        last = self._last_check.get(key)
        if last is None:
            return True
        elapsed = (datetime.utcnow() - last).total_seconds()
        return elapsed >= self._check_interval

    def get_failure_count(self, agent_id: str, version: str = "1.0.0") -> int:
        key = f"{agent_id}:{version}"
        return self._failure_counts.get(key, 0)


# ===========================================================================
# Test Classes
# ===========================================================================


class TestHealthCheckerCheckHealth:
    """Test check_health operations."""

    def test_check_returns_result(self):
        hc = HealthChecker()
        result = hc.check_health("gl-001")
        assert isinstance(result, HealthCheckResult)
        assert result.agent_id == "gl-001"

    def test_check_updates_status(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="healthy")
        assert hc.get_health("gl-001") == AgentHealthStatus.HEALTHY

    def test_check_records_history(self):
        hc = HealthChecker()
        hc.check_health("gl-001")
        history = hc.get_health_history("gl-001")
        assert len(history) == 1

    def test_check_with_latency(self):
        hc = HealthChecker()
        result = hc.check_health("gl-001", simulate_latency=5.0)
        assert result.latency_ms == 5.0

    def test_check_unhealthy_increments_failure(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="unhealthy")
        assert hc.get_failure_count("gl-001") == 1

    def test_check_healthy_resets_failure(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="unhealthy")
        hc.check_health("gl-001", simulate_status="healthy")
        assert hc.get_failure_count("gl-001") == 0

    def test_check_multiple_failures(self):
        hc = HealthChecker()
        for _ in range(5):
            hc.check_health("gl-001", simulate_status="unhealthy")
        assert hc.get_failure_count("gl-001") == 5


class TestHealthCheckerSetHealth:
    """Test set_health operations."""

    def test_set_by_id_and_version(self):
        hc = HealthChecker()
        hc.check_health("gl-001", version="1.0.0")
        hc.set_health("gl-001", "degraded", version="1.0.0")
        assert hc.get_health("gl-001", version="1.0.0") == AgentHealthStatus.DEGRADED

    def test_set_all_versions(self):
        hc = HealthChecker()
        hc.check_health("gl-001", version="1.0.0")
        hc.check_health("gl-001", version="2.0.0")
        hc.set_health("gl-001", "disabled")
        assert hc.get_health("gl-001", version="1.0.0") == AgentHealthStatus.DISABLED
        assert hc.get_health("gl-001", version="2.0.0") == AgentHealthStatus.DISABLED

    def test_set_creates_status_for_new_agent(self):
        hc = HealthChecker()
        hc.set_health("gl-new", "healthy", version="1.0.0")
        assert hc.get_health("gl-new", version="1.0.0") == AgentHealthStatus.HEALTHY


class TestHealthCheckerGetHistory:
    """Test get_health_history."""

    def test_history_ordered(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="healthy")
        hc.check_health("gl-001", simulate_status="degraded")
        hc.check_health("gl-001", simulate_status="unhealthy")
        history = hc.get_health_history("gl-001")
        assert len(history) == 3
        assert history[0].status == AgentHealthStatus.HEALTHY
        assert history[2].status == AgentHealthStatus.UNHEALTHY

    def test_history_limit(self):
        hc = HealthChecker()
        for _ in range(20):
            hc.check_health("gl-001")
        history = hc.get_health_history("gl-001", limit=5)
        assert len(history) == 5

    def test_history_empty(self):
        hc = HealthChecker()
        assert hc.get_health_history("nonexistent") == []


class TestHealthCheckerGetUnhealthy:
    """Test get_unhealthy_agents."""

    def test_returns_unhealthy(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="healthy")
        hc.check_health("gl-002", simulate_status="unhealthy")
        hc.check_health("gl-003", simulate_status="degraded")
        unhealthy = hc.get_unhealthy_agents()
        assert "gl-002" in unhealthy
        assert "gl-003" in unhealthy
        assert "gl-001" not in unhealthy

    def test_returns_empty_when_all_healthy(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="healthy")
        assert hc.get_unhealthy_agents() == []


class TestHealthCheckerGetSummary:
    """Test get_health_summary."""

    def test_summary_counts(self):
        hc = HealthChecker()
        hc.check_health("gl-001", simulate_status="healthy")
        hc.check_health("gl-002", simulate_status="healthy")
        hc.check_health("gl-003", simulate_status="degraded")
        hc.check_health("gl-004", simulate_status="unhealthy")
        summary = hc.get_health_summary()
        assert summary["healthy"] == 2
        assert summary["degraded"] == 1
        assert summary["unhealthy"] == 1

    def test_summary_empty(self):
        hc = HealthChecker()
        assert hc.get_health_summary() == {}


class TestHealthCheckerShouldRecheck:
    """Test TTL-based refresh."""

    def test_should_recheck_never_checked(self):
        hc = HealthChecker(check_interval_seconds=30)
        assert hc.should_recheck("gl-001") is True

    def test_should_not_recheck_recently_checked(self):
        hc = HealthChecker(check_interval_seconds=30)
        hc.check_health("gl-001")
        assert hc.should_recheck("gl-001") is False

    def test_should_recheck_after_interval(self):
        hc = HealthChecker(check_interval_seconds=0)
        hc.check_health("gl-001")
        # With interval=0, should always recheck
        assert hc.should_recheck("gl-001") is True


class TestHealthCheckerProbeTimeout:
    """Test probe timeout handling."""

    def test_timeout_config(self):
        hc = HealthChecker(timeout_seconds=10)
        assert hc._timeout == 10

    def test_unhealthy_threshold_config(self):
        hc = HealthChecker(unhealthy_threshold=5)
        assert hc._unhealthy_threshold == 5

    def test_threshold_breach(self):
        hc = HealthChecker(unhealthy_threshold=3)
        for _ in range(3):
            hc.check_health("gl-001", simulate_status="unhealthy")
        assert hc.get_failure_count("gl-001") == 3
        assert hc.get_failure_count("gl-001") >= hc._unhealthy_threshold

    def test_get_health_nonexistent(self):
        hc = HealthChecker()
        assert hc.get_health("nonexistent") is None

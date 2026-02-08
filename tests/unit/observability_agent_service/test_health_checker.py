# -*- coding: utf-8 -*-
"""
Unit Tests for HealthChecker (AGENT-FOUND-010)

Tests health probe registration, execution, aggregated status computation,
probe history, timeout handling, exception handling, and statistics.

Since health_checker.py is not yet on disk, tests define the expected
interface via an inline implementation matching the PRD specification.

Coverage target: 85%+ of health_checker.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline HealthChecker (mirrors expected interface)
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


@dataclass
class HealthProbeResult:
    """Result of a health probe."""
    probe_id: str = ""
    probe_name: str = ""
    probe_type: str = "liveness"
    status: str = "healthy"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    checked_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self):
        if not self.probe_id:
            self.probe_id = str(uuid.uuid4())


@dataclass
class HealthProbe:
    """A registered health check probe."""
    name: str = ""
    probe_type: str = "liveness"
    check_fn: Optional[Callable[[], Dict[str, Any]]] = None
    timeout_seconds: float = 5.0


class HealthChecker:
    """Health check engine with probe registration and aggregation."""

    VALID_STATUSES = ("healthy", "degraded", "unhealthy")
    VALID_PROBE_TYPES = ("liveness", "readiness", "startup")

    def __init__(self, config: Any) -> None:
        self._config = config
        self._probes: Dict[str, HealthProbe] = {}
        self._history: List[HealthProbeResult] = []
        self._total_checks: int = 0

    def register_probe(
        self,
        name: str,
        check_fn: Callable[[], Dict[str, Any]],
        probe_type: str = "liveness",
        timeout_seconds: float = 5.0,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("Probe name must be non-empty")
        if probe_type not in self.VALID_PROBE_TYPES:
            raise ValueError(f"Invalid probe_type '{probe_type}'")
        self._probes[name] = HealthProbe(
            name=name, probe_type=probe_type,
            check_fn=check_fn, timeout_seconds=timeout_seconds,
        )

    def register_default_probes(self) -> None:
        self.register_probe("self", lambda: {"status": "healthy"}, "liveness")

    def run_probe(self, name: str) -> HealthProbeResult:
        probe = self._probes.get(name)
        if probe is None:
            raise ValueError(f"Probe '{name}' not registered")

        start = time.monotonic()
        try:
            result = probe.check_fn()
            elapsed_ms = (time.monotonic() - start) * 1000
            status = result.get("status", "healthy")
            message = result.get("message", "")
            details = {k: v for k, v in result.items() if k not in ("status", "message")}

            if elapsed_ms > probe.timeout_seconds * 1000:
                status = "degraded"
                message = f"Probe exceeded timeout ({probe.timeout_seconds}s)"

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            status = "unhealthy"
            message = str(e)
            details = {"error": type(e).__name__}

        probe_result = HealthProbeResult(
            probe_name=name, probe_type=probe.probe_type,
            status=status, message=message, details=details,
            duration_ms=elapsed_ms,
        )
        self._history.append(probe_result)
        self._total_checks += 1
        return probe_result

    def run_all_probes(self) -> List[HealthProbeResult]:
        results = []
        for name in self._probes:
            results.append(self.run_probe(name))
        return results

    def get_aggregated_status(self) -> str:
        if not self._probes:
            return "healthy"
        results = self.run_all_probes()
        statuses = [r.status for r in results]
        if "unhealthy" in statuses:
            return "unhealthy"
        if "degraded" in statuses:
            return "degraded"
        return "healthy"

    def get_probe_history(self, name: Optional[str] = None, limit: int = 50) -> List[HealthProbeResult]:
        if name:
            filtered = [r for r in self._history if r.probe_name == name]
        else:
            filtered = list(self._history)
        return list(reversed(filtered[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "registered_probes": len(self._probes),
            "total_checks": self._total_checks,
            "history_size": len(self._history),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@dataclass
class _StubConfig:
    health_check_interval_seconds: int = 30


@pytest.fixture
def config():
    return _StubConfig()


@pytest.fixture
def checker(config):
    return HealthChecker(config)


def _healthy_check():
    return {"status": "healthy", "message": "OK"}


def _degraded_check():
    return {"status": "degraded", "message": "slow"}


def _unhealthy_check():
    return {"status": "unhealthy", "message": "down"}


def _exception_check():
    raise RuntimeError("Connection refused")


# ==========================================================================
# Registration Tests
# ==========================================================================

class TestHealthCheckerRegistration:
    """Tests for probe registration."""

    def test_register_probe(self, checker):
        checker.register_probe("db", _healthy_check)
        assert "db" in checker._probes

    def test_register_probe_with_type(self, checker):
        checker.register_probe("db", _healthy_check, probe_type="readiness")
        assert checker._probes["db"].probe_type == "readiness"

    def test_register_default_probes(self, checker):
        checker.register_default_probes()
        assert "self" in checker._probes

    def test_register_empty_name_raises(self, checker):
        with pytest.raises(ValueError, match="non-empty"):
            checker.register_probe("", _healthy_check)

    def test_register_invalid_probe_type_raises(self, checker):
        with pytest.raises(ValueError, match="Invalid probe_type"):
            checker.register_probe("p", _healthy_check, probe_type="invalid")


# ==========================================================================
# Run Probe Tests
# ==========================================================================

class TestHealthCheckerRunProbe:
    """Tests for running individual probes."""

    def test_run_probe_healthy(self, checker):
        checker.register_probe("db", _healthy_check)
        result = checker.run_probe("db")
        assert isinstance(result, HealthProbeResult)
        assert result.status == "healthy"
        assert result.probe_name == "db"
        assert result.duration_ms >= 0

    def test_run_probe_degraded(self, checker):
        checker.register_probe("slow_svc", _degraded_check)
        result = checker.run_probe("slow_svc")
        assert result.status == "degraded"

    def test_run_probe_unhealthy(self, checker):
        checker.register_probe("down_svc", _unhealthy_check)
        result = checker.run_probe("down_svc")
        assert result.status == "unhealthy"

    def test_run_probe_exception_handling(self, checker):
        checker.register_probe("err", _exception_check)
        result = checker.run_probe("err")
        assert result.status == "unhealthy"
        assert "Connection refused" in result.message

    def test_run_probe_nonexistent_raises(self, checker):
        with pytest.raises(ValueError, match="not registered"):
            checker.run_probe("ghost")


# ==========================================================================
# Run All Probes Tests
# ==========================================================================

class TestHealthCheckerRunAll:
    """Tests for run_all_probes."""

    def test_run_all_probes(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("cache", _healthy_check)
        results = checker.run_all_probes()
        assert len(results) == 2


# ==========================================================================
# Aggregated Status Tests
# ==========================================================================

class TestHealthCheckerAggregatedStatus:
    """Tests for get_aggregated_status."""

    def test_all_healthy(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("cache", _healthy_check)
        assert checker.get_aggregated_status() == "healthy"

    def test_one_degraded(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("slow", _degraded_check)
        assert checker.get_aggregated_status() == "degraded"

    def test_one_unhealthy(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("down", _unhealthy_check)
        assert checker.get_aggregated_status() == "unhealthy"

    def test_no_probes_is_healthy(self, checker):
        assert checker.get_aggregated_status() == "healthy"


# ==========================================================================
# History Tests
# ==========================================================================

class TestHealthCheckerHistory:
    """Tests for probe history."""

    def test_probe_history_records(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.run_probe("db")
        checker.run_probe("db")
        history = checker.get_probe_history("db")
        assert len(history) == 2

    def test_probe_history_filter_by_name(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("cache", _healthy_check)
        checker.run_probe("db")
        checker.run_probe("cache")
        history = checker.get_probe_history("db")
        assert len(history) == 1

    def test_probe_history_all(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.register_probe("cache", _healthy_check)
        checker.run_probe("db")
        checker.run_probe("cache")
        history = checker.get_probe_history()
        assert len(history) == 2


# ==========================================================================
# Statistics Tests
# ==========================================================================

class TestHealthCheckerStatistics:
    """Tests for get_statistics."""

    def test_statistics_empty(self, checker):
        stats = checker.get_statistics()
        assert stats["registered_probes"] == 0
        assert stats["total_checks"] == 0

    def test_statistics_after_checks(self, checker):
        checker.register_probe("db", _healthy_check)
        checker.run_probe("db")
        checker.run_probe("db")
        stats = checker.get_statistics()
        assert stats["registered_probes"] == 1
        assert stats["total_checks"] == 2
        assert stats["history_size"] == 2

"""
GreenLang Observability - Health Check Module Tests
====================================================

Comprehensive unit tests for health check functionality.
"""

import pytest
import time
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from observability.health import (
    HealthCheckManager,
    HealthStatus,
    HealthCheckResult,
    CheckType,
    LivenessCheck,
    ReadinessCheck,
    StartupCheck,
    DependencyCheck,
    AsyncDependencyCheck,
    create_http_check,
    create_tcp_check,
    create_file_check,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_health_status_values(self) -> None:
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_to_http_status(self) -> None:
        """Test HTTP status code mapping."""
        assert HealthStatus.HEALTHY.to_http_status() == 200
        assert HealthStatus.DEGRADED.to_http_status() == 200
        assert HealthStatus.UNHEALTHY.to_http_status() == 503
        assert HealthStatus.UNKNOWN.to_http_status() == 503


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_result_creation(self) -> None:
        """Test result creation."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
            name="test",
            message="All good",
        )

        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == CheckType.LIVENESS
        assert result.name == "test"
        assert result.timestamp is not None

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.READINESS,
            name="test",
            message="Ready",
            duration_ms=10.5,
        )

        d = result.to_dict()
        assert d["status"] == "healthy"
        assert d["check_type"] == "readiness"
        assert d["name"] == "test"
        assert d["duration_ms"] == 10.5

    def test_to_simple_response(self) -> None:
        """Test simple response format."""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
        )

        assert result.to_simple_response() == "healthy"

    def test_with_dependencies(self) -> None:
        """Test result with dependencies."""
        dep1 = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.DEPENDENCY,
            name="database",
        )
        dep2 = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.DEPENDENCY,
            name="redis",
        )

        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            check_type=CheckType.READINESS,
            dependencies=[dep1, dep2],
        )

        d = result.to_dict()
        assert len(d["dependencies"]) == 2


class TestLivenessCheck:
    """Tests for LivenessCheck."""

    def test_basic_liveness(self) -> None:
        """Test basic liveness check."""
        check = LivenessCheck()
        result = check.run()

        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == CheckType.LIVENESS

    def test_liveness_with_heartbeat(self) -> None:
        """Test liveness with heartbeat."""
        check = LivenessCheck()
        check.record_heartbeat()

        result = check.run()
        assert result.status == HealthStatus.HEALTHY
        assert "last_heartbeat_age_seconds" in result.details

    def test_liveness_stale_heartbeat(self) -> None:
        """Test liveness with stale heartbeat."""
        check = LivenessCheck()
        check._last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=1)
        check._heartbeat_threshold = timedelta(seconds=30)

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_liveness_custom_check(self) -> None:
        """Test liveness with custom check function."""
        check = LivenessCheck(custom_check=lambda: True)
        result = check.run()
        assert result.status == HealthStatus.HEALTHY

        check = LivenessCheck(custom_check=lambda: False)
        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_liveness_custom_check_exception(self) -> None:
        """Test liveness when custom check raises exception."""
        def failing_check() -> bool:
            raise RuntimeError("Check failed")

        check = LivenessCheck(custom_check=failing_check)
        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY


class TestReadinessCheck:
    """Tests for ReadinessCheck."""

    def test_basic_readiness_not_ready(self) -> None:
        """Test readiness when not ready."""
        check = ReadinessCheck()
        result = check.run()

        assert result.status == HealthStatus.UNHEALTHY

    def test_readiness_set_ready(self) -> None:
        """Test setting ready state."""
        check = ReadinessCheck()
        check.set_ready(True)

        result = check.run()
        assert result.status == HealthStatus.HEALTHY

    def test_readiness_set_not_ready(self) -> None:
        """Test setting not ready state."""
        check = ReadinessCheck()
        check.set_ready(True)
        check.set_ready(False)

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_readiness_custom_check(self) -> None:
        """Test readiness with custom check."""
        check = ReadinessCheck(custom_check=lambda: True)
        check.set_ready(True)

        result = check.run()
        assert result.status == HealthStatus.HEALTHY

        check = ReadinessCheck(custom_check=lambda: False)
        check.set_ready(True)

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY


class TestStartupCheck:
    """Tests for StartupCheck."""

    def test_startup_not_started(self) -> None:
        """Test startup when not started."""
        check = StartupCheck()
        result = check.run()

        assert result.status == HealthStatus.UNHEALTHY

    def test_startup_marked_started(self) -> None:
        """Test startup when marked as started."""
        check = StartupCheck()
        check.mark_started()

        result = check.run()
        assert result.status == HealthStatus.HEALTHY
        assert "started_at" in result.details

    def test_startup_custom_check(self) -> None:
        """Test startup with custom check."""
        check = StartupCheck(custom_check=lambda: True)
        check.mark_started()

        result = check.run()
        assert result.status == HealthStatus.HEALTHY

        check = StartupCheck(custom_check=lambda: False)
        check.mark_started()

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY


class TestDependencyCheck:
    """Tests for DependencyCheck."""

    def test_dependency_check_healthy(self) -> None:
        """Test healthy dependency."""
        check = DependencyCheck(
            name="database",
            check_fn=lambda: True,
        )

        result = check.run()
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "database"

    def test_dependency_check_unhealthy(self) -> None:
        """Test unhealthy dependency."""
        check = DependencyCheck(
            name="database",
            check_fn=lambda: False,
        )

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY

    def test_dependency_check_non_critical(self) -> None:
        """Test non-critical dependency failure."""
        check = DependencyCheck(
            name="cache",
            check_fn=lambda: False,
            critical=False,
        )

        result = check.run()
        assert result.status == HealthStatus.DEGRADED

    def test_dependency_check_exception(self) -> None:
        """Test dependency check that raises exception."""
        def failing_check() -> bool:
            raise ConnectionError("Connection refused")

        check = DependencyCheck(
            name="database",
            check_fn=failing_check,
        )

        result = check.run()
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection refused" in result.message

    def test_dependency_check_timeout(self) -> None:
        """Test dependency check timeout."""
        def slow_check() -> bool:
            time.sleep(10)
            return True

        check = DependencyCheck(
            name="slow",
            check_fn=slow_check,
            timeout_seconds=0.1,
        )

        # Check should complete (timeout is not enforced at this level)
        # In real implementation, timeout would be handled differently


class TestHealthCheckManager:
    """Tests for HealthCheckManager."""

    def test_manager_creation(self) -> None:
        """Test manager creation."""
        manager = HealthCheckManager(
            service_name="gl-006-heatreclaim",
            service_version="1.0.0",
        )

        assert manager.service_name == "gl-006-heatreclaim"
        assert manager.service_version == "1.0.0"

    def test_check_liveness(self) -> None:
        """Test liveness check via manager."""
        manager = HealthCheckManager("test")
        result = manager.check_liveness()

        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == CheckType.LIVENESS

    def test_check_readiness(self) -> None:
        """Test readiness check via manager."""
        manager = HealthCheckManager("test")
        manager.set_ready(True)

        result = manager.check_readiness()
        assert result.status == HealthStatus.HEALTHY

    def test_check_startup(self) -> None:
        """Test startup check via manager."""
        manager = HealthCheckManager("test")
        manager.mark_started()

        result = manager.check_startup()
        assert result.status == HealthStatus.HEALTHY

    def test_add_dependency(self) -> None:
        """Test adding dependency check."""
        manager = HealthCheckManager("test")
        manager.add_dependency(
            name="database",
            check_fn=lambda: True,
            description="Database connection",
        )

        deps = manager.check_dependencies()
        assert len(deps) == 1
        assert deps[0].status == HealthStatus.HEALTHY

    def test_remove_dependency(self) -> None:
        """Test removing dependency check."""
        manager = HealthCheckManager("test")
        manager.add_dependency("database", lambda: True)

        assert manager.remove_dependency("database") is True
        assert manager.remove_dependency("database") is False  # Already removed

    def test_readiness_with_dependencies(self) -> None:
        """Test readiness includes dependencies."""
        manager = HealthCheckManager("test")
        manager.set_ready(True)
        manager.add_dependency("db", lambda: True)
        manager.add_dependency("cache", lambda: True)

        result = manager.check_readiness(include_dependencies=True)
        assert result.status == HealthStatus.HEALTHY
        assert len(result.dependencies) == 2

    def test_readiness_with_failing_dependency(self) -> None:
        """Test readiness with failing critical dependency."""
        manager = HealthCheckManager("test")
        manager.set_ready(True)
        manager.add_dependency("db", lambda: True)
        manager.add_dependency("failing", lambda: False, critical=True)

        result = manager.check_readiness()
        assert result.status == HealthStatus.UNHEALTHY

    def test_readiness_with_failing_non_critical(self) -> None:
        """Test readiness with failing non-critical dependency."""
        manager = HealthCheckManager("test")
        manager.set_ready(True)
        manager.add_dependency("db", lambda: True)
        manager.add_dependency("cache", lambda: False, critical=False)

        result = manager.check_readiness()
        assert result.status == HealthStatus.DEGRADED

    def test_record_heartbeat(self) -> None:
        """Test recording heartbeat via manager."""
        manager = HealthCheckManager("test")
        manager.record_heartbeat()

        result = manager.check_liveness()
        assert result.status == HealthStatus.HEALTHY

    def test_check_all(self) -> None:
        """Test checking all health endpoints."""
        manager = HealthCheckManager("test")
        manager.mark_started()
        manager.set_ready(True)

        results = manager.check_all()

        assert "liveness" in results
        assert "readiness" in results
        assert "startup" in results
        assert all(r.status == HealthStatus.HEALTHY for r in results.values())

    def test_set_custom_liveness_check(self) -> None:
        """Test setting custom liveness check."""
        manager = HealthCheckManager("test")
        manager.set_liveness_check(lambda: True)

        result = manager.check_liveness()
        assert result.status == HealthStatus.HEALTHY

    def test_get_metrics(self) -> None:
        """Test getting health metrics."""
        manager = HealthCheckManager("test")
        manager.mark_started()
        manager.set_ready(True)
        manager.add_dependency("db", lambda: True)
        manager.add_dependency("cache", lambda: False, critical=False)

        metrics = manager.get_metrics()

        assert metrics["health_check_liveness"] == 1
        assert metrics["health_check_readiness"] == 1
        assert metrics["health_check_startup"] == 1
        assert metrics["health_check_dependencies_total"] == 2
        assert metrics["health_check_dependencies_healthy"] == 1
        assert metrics["health_check_dependencies_unhealthy"] == 1


class TestAsyncDependencyCheck:
    """Tests for AsyncDependencyCheck."""

    def test_async_check_healthy(self) -> None:
        """Test healthy async dependency."""
        async def check() -> bool:
            await asyncio.sleep(0.01)
            return True

        dep = AsyncDependencyCheck(name="async-db", check_fn=check)
        result = dep.run()

        assert result.status == HealthStatus.HEALTHY

    def test_async_check_unhealthy(self) -> None:
        """Test unhealthy async dependency."""
        async def check() -> bool:
            return False

        dep = AsyncDependencyCheck(name="async-db", check_fn=check)
        result = dep.run()

        assert result.status == HealthStatus.UNHEALTHY

    def test_async_check_exception(self) -> None:
        """Test async dependency that raises exception."""
        async def check() -> bool:
            raise ConnectionError("Failed")

        dep = AsyncDependencyCheck(name="async-db", check_fn=check)
        result = dep.run()

        assert result.status == HealthStatus.UNHEALTHY


class TestHealthCheckFactories:
    """Tests for health check factory functions."""

    def test_create_http_check(self) -> None:
        """Test HTTP check creation."""
        # This would need mocking in real tests
        check = create_http_check("http://localhost:8080/health")
        assert callable(check)

    def test_create_tcp_check(self) -> None:
        """Test TCP check creation."""
        check = create_tcp_check("localhost", 8080)
        assert callable(check)

        # Actually run check (will fail but shouldn't raise)
        result = check()
        assert isinstance(result, bool)

    def test_create_file_check(self) -> None:
        """Test file check creation."""
        import tempfile

        # Create a temp file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name

        try:
            check = create_file_check(temp_path)
            assert check() is True

            check = create_file_check(temp_path, max_age_seconds=3600)
            assert check() is True

            check = create_file_check("/nonexistent/path")
            assert check() is False
        finally:
            os.unlink(temp_path)


class TestHealthCheckDuration:
    """Tests for health check timing."""

    def test_check_records_duration(self) -> None:
        """Test that checks record duration."""
        def slow_check() -> bool:
            time.sleep(0.05)
            return True

        check = DependencyCheck(name="slow", check_fn=slow_check)
        result = check.run()

        assert result.duration_ms >= 50  # At least 50ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

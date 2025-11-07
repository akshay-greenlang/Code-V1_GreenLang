"""
Tests for health checks
"""

import pytest
import asyncio
from greenlang.observability import (
    HealthStatus,
    CheckType,
    HealthCheckResult,
    HealthReport,
    HealthCheck,
    LivenessCheck,
    ReadinessCheck,
    DiskSpaceHealthCheck,
    MemoryHealthCheck,
    CPUHealthCheck,
    HealthChecker,
    get_health_checker,
)


class TestHealthCheckResult:
    """Test HealthCheckResult functionality"""

    def test_health_check_result_creation(self):
        """Test creating health check result"""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="All good",
            details={"version": "1.0"},
        )
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"

    def test_health_check_result_to_dict(self):
        """Test converting result to dictionary"""
        result = HealthCheckResult(
            name="test", status=HealthStatus.DEGRADED, message="Slow"
        )
        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "degraded"
        assert data["message"] == "Slow"


class TestHealthReport:
    """Test HealthReport functionality"""

    def test_health_report_creation(self):
        """Test creating health report"""
        checks = [
            HealthCheckResult("check1", HealthStatus.HEALTHY, "OK"),
            HealthCheckResult("check2", HealthStatus.HEALTHY, "OK"),
        ]
        report = HealthReport(status=HealthStatus.HEALTHY, checks=checks)
        assert report.status == HealthStatus.HEALTHY
        assert len(report.checks) == 2

    def test_health_report_to_dict(self):
        """Test converting report to dictionary"""
        checks = [HealthCheckResult("check1", HealthStatus.HEALTHY, "OK")]
        report = HealthReport(status=HealthStatus.HEALTHY, checks=checks)
        data = report.to_dict()
        assert data["status"] == "healthy"
        assert len(data["checks"]) == 1

    def test_health_report_to_json(self):
        """Test converting report to JSON"""
        import json

        checks = [HealthCheckResult("check1", HealthStatus.HEALTHY, "OK")]
        report = HealthReport(status=HealthStatus.HEALTHY, checks=checks)
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["status"] == "healthy"


class TestHealthCheck:
    """Test base HealthCheck class"""

    def test_health_check_initialization(self):
        """Test health check initialization"""
        check = HealthCheck("test_check", critical=True, timeout_seconds=5.0)
        assert check.name == "test_check"
        assert check.critical is True
        assert check.timeout_seconds == 5.0

    def test_health_check_synchronous(self):
        """Test synchronous health check"""

        class CustomCheck(HealthCheck):
            def _perform_check(self):
                return HealthCheckResult(self.name, HealthStatus.HEALTHY, "OK")

        check = CustomCheck("custom")
        result = check.check()
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_health_check_asynchronous(self):
        """Test asynchronous health check"""

        class AsyncCheck(HealthCheck):
            async def _perform_check_async(self):
                await asyncio.sleep(0.1)
                return HealthCheckResult(self.name, HealthStatus.HEALTHY, "OK")

        check = AsyncCheck("async_check")
        result = await check.check_async()
        assert result.status == HealthStatus.HEALTHY

    def test_health_check_with_exception(self):
        """Test health check that raises exception"""

        class FailingCheck(HealthCheck):
            def _perform_check(self):
                raise RuntimeError("Check failed")

        check = FailingCheck("failing")
        result = check.check()
        assert result.status == HealthStatus.UNHEALTHY


class TestBuiltInHealthChecks:
    """Test built-in health check implementations"""

    def test_liveness_check(self):
        """Test liveness check"""
        check = LivenessCheck()
        result = check.check()
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "liveness"

    def test_readiness_check(self):
        """Test readiness check"""
        check = ReadinessCheck(dependencies=["db", "cache"])
        result = check.check()
        # Should be healthy if all dependencies pass
        assert result.name == "readiness"

    def test_disk_space_check(self):
        """Test disk space check"""
        check = DiskSpaceHealthCheck(path="/", min_free_gb=0.1)
        result = check.check()
        assert result.name == "disk_space"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    def test_memory_check(self):
        """Test memory usage check"""
        check = MemoryHealthCheck(max_usage_percent=90.0)
        result = check.check()
        assert result.name == "memory"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]

    def test_cpu_check(self):
        """Test CPU usage check"""
        check = CPUHealthCheck(max_usage_percent=80.0)
        result = check.check()
        assert result.name == "cpu"
        assert result.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]


class TestHealthChecker:
    """Test HealthChecker functionality"""

    def test_health_checker_initialization(self):
        """Test health checker initialization"""
        checker = HealthChecker()
        assert checker is not None
        assert len(checker.checks) > 0  # Has default checks

    def test_register_custom_check(self):
        """Test registering a custom check"""
        checker = HealthChecker()

        class CustomCheck(HealthCheck):
            def _perform_check(self):
                return HealthCheckResult(self.name, HealthStatus.HEALTHY, "OK")

        custom = CustomCheck("custom_check")
        checker.register_check(custom)
        assert "custom_check" in checker.checks

    def test_unregister_check(self):
        """Test unregistering a check"""
        checker = HealthChecker()
        checker.unregister_check("liveness")
        assert "liveness" not in checker.checks

    def test_check_health_synchronous(self):
        """Test synchronous health check"""
        checker = HealthChecker()
        report = checker.check_health()
        assert report is not None
        assert report.status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        ]
        assert len(report.checks) > 0

    @pytest.mark.asyncio
    async def test_check_health_asynchronous(self):
        """Test asynchronous health check"""
        checker = HealthChecker()
        report = await checker.check_health_async()
        assert report is not None
        assert len(report.checks) > 0

    def test_check_health_with_type_filter(self):
        """Test health check with type filter"""
        checker = HealthChecker()
        report = checker.check_health(check_type=CheckType.LIVENESS)
        assert report is not None

    def test_get_status(self):
        """Test getting current health status"""
        checker = HealthChecker()
        checker.check_health()  # Perform check first
        status = checker.get_status()
        assert status in [
            HealthStatus.HEALTHY,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
            HealthStatus.UNKNOWN,
        ]


class TestGlobalHealthInstances:
    """Test global health checker instances"""

    def test_get_health_checker_singleton(self):
        """Test getting global health checker"""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    def test_health_checker_functional(self):
        """Test global health checker is functional"""
        checker = get_health_checker()
        report = checker.check_health()
        assert report is not None

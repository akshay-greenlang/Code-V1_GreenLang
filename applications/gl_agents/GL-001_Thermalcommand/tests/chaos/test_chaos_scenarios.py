"""
GL-001 ThermalCommand - Chaos Engineering Test Suite

This module contains pytest-based chaos engineering tests for the
ThermalCommand Orchestrator, validating system resilience under various
failure conditions.

Test Categories:
- Network fault scenarios
- Resource exhaustion scenarios
- Service failure scenarios
- Resilience pattern validation
- Kubernetes chaos scenarios
- Steady state hypothesis tests

All tests are CI-safe (no actual infrastructure damage).

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import pytest
import logging
from datetime import datetime, timezone

from .chaos_runner import (
    ChaosRunner,
    ChaosExperiment,
    ChaosSeverity,
    ChaosPhase,
)
from .fault_injectors import (
    NetworkLatencyInjector,
    NetworkPartitionInjector,
    PacketLossInjector,
    NetworkFaultInjector,
    CPUStressInjector,
    MemoryPressureInjector,
    DiskIOFaultInjector,
    ResourceFaultInjector,
    ServiceUnavailabilityInjector,
    ServiceFaultInjector,
    StateFaultInjector,
    get_fault_injector,
)
from .steady_state import (
    SteadyStateHypothesis,
    SteadyStateMetric,
    SteadyStateValidator,
    ComparisonOperator,
    create_api_health_hypothesis,
    create_thermal_command_hypothesis,
)
from .resilience_patterns import (
    CircuitBreakerTest,
    RetryMechanismTest,
    FallbackBehaviorTest,
    GracefulDegradationTest,
)
from .kubernetes_chaos import (
    K8sPodDeletionTest,
    K8sNodeFailureTest,
    K8sResourceExhaustionTest,
    K8sChaosTestRunner,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def chaos_runner():
    """Create a chaos runner for testing."""
    return ChaosRunner(dry_run=True)


@pytest.fixture
def chaos_runner_live():
    """Create a chaos runner that simulates actual injection."""
    return ChaosRunner(dry_run=False)


@pytest.fixture
def network_latency_experiment():
    """Create a network latency experiment."""
    return ChaosExperiment(
        name="network_latency_50ms",
        description="Inject 50ms network latency",
        severity=ChaosSeverity.LOW,
        duration_seconds=5.0,
        cooldown_seconds=2.0,
        fault_type="network_latency",
        fault_params={"delay_ms": 50, "jitter_ms": 10},
        steady_state_thresholds={
            "response_time_ms": 200,
            "error_rate_percent": 5.0,
        },
    )


@pytest.fixture
def high_latency_experiment():
    """Create a high latency experiment."""
    return ChaosExperiment(
        name="network_latency_1s",
        description="Inject 1 second network latency",
        severity=ChaosSeverity.HIGH,
        duration_seconds=10.0,
        cooldown_seconds=5.0,
        fault_type="network_latency",
        fault_params={"delay_ms": 1000, "jitter_ms": 200},
        steady_state_thresholds={
            "response_time_ms": 2000,
            "error_rate_percent": 10.0,
        },
    )


@pytest.fixture
def service_unavailable_experiment():
    """Create a service unavailability experiment."""
    return ChaosExperiment(
        name="database_unavailable",
        description="Simulate database unavailability",
        severity=ChaosSeverity.MEDIUM,
        duration_seconds=5.0,
        cooldown_seconds=3.0,
        fault_type="service",
        fault_params={
            "unavailable_services": ["database"],
            "failure_mode": "timeout",
        },
        steady_state_thresholds={
            "error_rate_percent": 5.0,
        },
    )


# =============================================================================
# Network Fault Tests
# =============================================================================

class TestNetworkFaults:
    """Test suite for network fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_latency_injector_lifecycle(self):
        """Test latency injector inject and rollback."""
        injector = NetworkLatencyInjector()

        assert not injector.is_active()

        # Inject latency
        result = await injector.inject({"delay_ms": 100, "jitter_ms": 20})
        assert result is True
        assert injector.is_active()

        # Get delay should be non-zero
        delay = injector.get_delay()
        assert delay > 0

        # Rollback
        result = await injector.rollback()
        assert result is True
        assert not injector.is_active()
        assert injector.get_delay() == 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_network_partition_injector(self):
        """Test network partition simulation."""
        injector = NetworkPartitionInjector()

        # Create partition
        result = await injector.inject({
            "partition_groups": [["node1", "node2"], ["node3", "node4"]],
            "bidirectional": True,
        })
        assert result is True
        assert injector.is_active()

        # Check blocked routes
        assert injector.is_route_blocked("node1", "node3")
        assert injector.is_route_blocked("node3", "node1")
        assert not injector.is_route_blocked("node1", "node2")

        # Rollback
        result = await injector.rollback()
        assert result is True
        assert not injector.is_route_blocked("node1", "node3")

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_packet_loss_injector(self):
        """Test packet loss simulation."""
        injector = PacketLossInjector()

        # Inject 50% packet loss
        result = await injector.inject({"loss_percentage": 50})
        assert result is True
        assert injector.is_active()

        # Sample packet loss decisions
        drops = sum(1 for _ in range(1000) if injector.should_drop_packet())
        # Should be roughly 50% (with statistical tolerance)
        assert 400 < drops < 600, f"Expected ~500 drops, got {drops}"

        # Rollback
        await injector.rollback()
        assert not injector.should_drop_packet()

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_composite_network_fault_injector(self):
        """Test composite network fault injection."""
        injector = NetworkFaultInjector()

        result = await injector.inject({
            "latency_ms": 100,
            "packet_loss_percent": 10,
        })
        assert result is True
        assert injector.is_active()

        # Rollback all
        result = await injector.rollback()
        assert result is True
        assert not injector.is_active()


# =============================================================================
# Resource Fault Tests
# =============================================================================

class TestResourceFaults:
    """Test suite for resource fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_cpu_stress_injector(self):
        """Test CPU stress simulation."""
        injector = CPUStressInjector()

        result = await injector.inject({"target_cpu_percent": 80})
        assert result is True
        assert injector.is_active()
        assert injector.get_simulated_load() == 80

        await injector.rollback()
        assert injector.get_simulated_load() == 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_memory_pressure_injector(self):
        """Test memory pressure simulation."""
        injector = MemoryPressureInjector()

        result = await injector.inject({"target_memory_mb": 512})
        assert result is True
        assert injector.is_active()
        assert injector.get_simulated_memory_usage() == 512

        await injector.rollback()
        assert injector.get_simulated_memory_usage() == 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_disk_io_fault_injector(self):
        """Test disk I/O fault simulation."""
        injector = DiskIOFaultInjector()

        result = await injector.inject({
            "read_latency_ms": 200,
            "write_failure_percent": 20,
        })
        assert result is True
        assert injector.is_active()

        # Sample write failures
        failures = sum(1 for _ in range(1000) if injector.should_fail_write())
        assert 150 < failures < 250, f"Expected ~200 failures, got {failures}"

        await injector.rollback()
        assert not injector.should_fail_write()


# =============================================================================
# Service Fault Tests
# =============================================================================

class TestServiceFaults:
    """Test suite for service fault injection."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_service_unavailability_injector(self):
        """Test service unavailability simulation."""
        injector = ServiceUnavailabilityInjector()

        result = await injector.inject({
            "services": ["database", "cache"],
            "failure_mode": "timeout",
        })
        assert result is True
        assert injector.is_active()

        assert not injector.is_service_available("database")
        assert not injector.is_service_available("cache")
        assert injector.is_service_available("other_service")

        await injector.rollback()
        assert injector.is_service_available("database")

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_state_fault_injector(self):
        """Test state corruption simulation."""
        injector = StateFaultInjector()

        result = await injector.inject({
            "corruption_type": "stale_cache",
            "affected_keys": ["session", "user_data"],
            "corruption_percent": 100,
        })
        assert result is True
        assert injector.is_active()

        assert injector.is_key_corrupted("session")
        assert injector.is_key_corrupted("user_data")
        assert not injector.is_key_corrupted("other_key")

        await injector.rollback()
        assert not injector.is_key_corrupted("session")


# =============================================================================
# Chaos Runner Tests
# =============================================================================

class TestChaosRunner:
    """Test suite for chaos experiment runner."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_dry_run_experiment(self, chaos_runner, network_latency_experiment):
        """Test dry run experiment execution."""
        result = await chaos_runner.run(network_latency_experiment)

        assert result.experiment_name == "network_latency_50ms"
        assert result.status == "success"
        assert result.phase == ChaosPhase.COMPLETED
        assert "DRY RUN" in " ".join(result.observations)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_experiment_with_steady_state(self, chaos_runner_live, network_latency_experiment):
        """Test experiment with steady state validation."""
        # Register the fault injector
        chaos_runner_live.register_injector("network_latency", NetworkLatencyInjector)

        result = await chaos_runner_live.run(network_latency_experiment)

        assert result.experiment_name == "network_latency_50ms"
        assert result.steady_state_before is not None
        assert result.steady_state_after is not None

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_experiment_suite(self, chaos_runner):
        """Test running multiple experiments in sequence."""
        experiments = [
            ChaosExperiment(
                name="test_1",
                description="First test",
                duration_seconds=1.0,
                cooldown_seconds=0.5,
            ),
            ChaosExperiment(
                name="test_2",
                description="Second test",
                duration_seconds=1.0,
                cooldown_seconds=0.5,
            ),
        ]

        results = await chaos_runner.run_suite(experiments)

        assert len(results) == 2
        assert all(r.status == "success" for r in results)

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_experiment_history(self, chaos_runner):
        """Test experiment history tracking."""
        experiment = ChaosExperiment(
            name="history_test",
            description="Test history",
            duration_seconds=1.0,
        )

        await chaos_runner.run(experiment)
        await chaos_runner.run(experiment)

        history = chaos_runner.get_history()
        assert len(history) == 2

        chaos_runner.clear_history()
        assert len(chaos_runner.get_history()) == 0


# =============================================================================
# Steady State Tests
# =============================================================================

class TestSteadyState:
    """Test suite for steady state hypothesis validation."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_metric_validation_less_than(self):
        """Test metric validation with less than operator."""
        metric = SteadyStateMetric(
            name="response_time",
            threshold=100,
            operator=ComparisonOperator.LESS_THAN,
        )

        passed, message = metric.validate(50)
        assert passed is True
        assert "PASS" in message

        passed, message = metric.validate(150)
        assert passed is False
        assert "FAIL" in message

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_metric_validation_in_range(self):
        """Test metric validation with in_range operator."""
        metric = SteadyStateMetric(
            name="cpu_usage",
            threshold=(20, 80),
            operator=ComparisonOperator.IN_RANGE,
        )

        passed, _ = metric.validate(50)
        assert passed is True

        passed, _ = metric.validate(90)
        assert passed is False

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_steady_state_validator(self):
        """Test steady state validator."""
        hypothesis = create_api_health_hypothesis(
            max_response_time_ms=200,
            max_error_rate_percent=1.0,
        )

        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == "API Health"
        assert result.timestamp is not None
        assert len(result.metric_results) > 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_thermal_command_hypothesis(self):
        """Test ThermalCommand-specific hypothesis."""
        hypothesis = create_thermal_command_hypothesis()

        validator = SteadyStateValidator()
        result = await validator.validate(hypothesis)

        assert result.hypothesis_name == "ThermalCommand Health"
        assert result.aggregate_score >= 0


# =============================================================================
# Resilience Pattern Tests
# =============================================================================

class TestCircuitBreaker:
    """Test suite for circuit breaker pattern."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_opens_on_failures(self):
        """Test circuit breaker opens after failure threshold."""
        test = CircuitBreakerTest()
        result = await test.test_circuit_opens_on_failures()

        assert result.passed is True
        assert result.final_state.value == "open"
        assert len(result.transitions) > 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_fast_fails(self):
        """Test circuit breaker fast-fails when open."""
        test = CircuitBreakerTest()
        result = await test.test_circuit_fast_fails_when_open()

        assert result.passed is True
        assert result.fast_failed_calls > 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_circuit_recovery(self):
        """Test circuit breaker recovery."""
        test = CircuitBreakerTest()
        result = await test.test_circuit_recovery()

        assert result.passed is True
        assert result.final_state.value == "closed"


class TestRetryMechanism:
    """Test suite for retry mechanism pattern."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self):
        """Test retry succeeds after transient failures."""
        test = RetryMechanismTest()
        result = await test.test_retry_on_transient_failure()

        assert result.passed is True
        assert result.successful_on_attempt > 1

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_max_retry_limit(self):
        """Test retry stops at max limit."""
        test = RetryMechanismTest()
        result = await test.test_max_retry_limit()

        assert result.passed is True
        assert result.successful_on_attempt == 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff delays."""
        test = RetryMechanismTest()
        result = await test.test_exponential_backoff()

        assert result.passed is True
        # Delays should be increasing
        for i in range(1, len(result.actual_delays)):
            assert result.actual_delays[i] >= result.actual_delays[i-1] * 0.8


class TestFallbackBehavior:
    """Test suite for fallback behavior pattern."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_fallback_on_primary_failure(self):
        """Test fallback used when primary fails."""
        test = FallbackBehaviorTest()
        result = await test.test_fallback_on_primary_failure()

        assert result.passed is True
        assert result.fallback_used is True

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_primary_preferred(self):
        """Test primary is used when available."""
        test = FallbackBehaviorTest()
        result = await test.test_primary_preferred()

        assert result.passed is True
        assert result.fallback_used is False


class TestGracefulDegradation:
    """Test suite for graceful degradation pattern."""

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_load_shedding(self):
        """Test feature shedding under load."""
        test = GracefulDegradationTest()
        result = await test.test_load_shedding()

        assert result.passed is True
        assert len(result.transitions) > 0

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_recovery_from_degradation(self):
        """Test recovery from degraded state."""
        test = GracefulDegradationTest()
        result = await test.test_recovery_from_degradation()

        assert result.passed is True
        assert result.final_level == "normal"

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_core_features_preserved(self):
        """Test core features are never disabled."""
        test = GracefulDegradationTest()
        result = await test.test_core_features_preserved()

        assert result.passed is True
        assert "core" in result.features_available


# =============================================================================
# Kubernetes Chaos Tests
# =============================================================================

class TestK8sPodDeletion:
    """Test suite for Kubernetes pod deletion chaos."""

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_single_pod_deletion(self):
        """Test recovery from single pod deletion."""
        test = K8sPodDeletionTest(replicas=3)
        result = await test.test_single_pod_deletion()

        assert result.passed is True
        assert result.final_pod_count == result.expected_replicas
        assert len(result.pods_recovered) >= 1

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_multiple_pod_deletion(self):
        """Test recovery from multiple pod deletions."""
        test = K8sPodDeletionTest(replicas=3)
        result = await test.test_multiple_pod_deletion(delete_count=2)

        assert result.passed is True
        assert result.final_pod_count == result.expected_replicas


class TestK8sNodeFailure:
    """Test suite for Kubernetes node failure chaos."""

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_single_node_failure(self):
        """Test recovery from single node failure."""
        test = K8sNodeFailureTest()
        result = await test.test_single_node_failure()

        assert result.passed is True
        assert result.final_pod_count == result.expected_replicas

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_node_recovery(self):
        """Test node recovery after failure."""
        test = K8sNodeFailureTest()
        result = await test.test_node_recovery()

        assert result.passed is True


class TestK8sResourceExhaustion:
    """Test suite for Kubernetes resource exhaustion chaos."""

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        test = K8sResourceExhaustionTest()
        result = await test.test_memory_pressure()

        assert result.passed is True

    @pytest.mark.chaos
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_disk_pressure(self):
        """Test behavior under disk pressure."""
        test = K8sResourceExhaustionTest()
        result = await test.test_disk_pressure()

        assert result.passed is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestChaosIntegration:
    """Integration tests combining multiple chaos components."""

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_chaos_experiment_lifecycle(self):
        """Test complete chaos experiment lifecycle."""
        runner = ChaosRunner(dry_run=False)
        runner.register_injector("network_latency", NetworkLatencyInjector)

        experiment = ChaosExperiment(
            name="integration_test",
            description="Full lifecycle test",
            severity=ChaosSeverity.LOW,
            duration_seconds=2.0,
            cooldown_seconds=1.0,
            fault_type="network_latency",
            fault_params={"delay_ms": 50},
            steady_state_thresholds={
                "response_time_ms": 500,
                "error_rate_percent": 5.0,
            },
        )

        result = await runner.run(experiment)

        assert result.phase == ChaosPhase.COMPLETED
        assert result.steady_state_before is True
        assert result.steady_state_after is True
        assert result.hypothesis_validated is True

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_resilience_patterns_suite(self):
        """Test all resilience patterns in sequence."""
        cb_test = CircuitBreakerTest()
        cb_results = await cb_test.run_all_tests()
        assert all(r.passed for r in cb_results)

        retry_test = RetryMechanismTest()
        retry_results = await retry_test.run_all_tests()
        assert all(r.passed for r in retry_results)

        fallback_test = FallbackBehaviorTest()
        fallback_results = await fallback_test.run_all_tests()
        assert all(r.passed for r in fallback_results)

        degradation_test = GracefulDegradationTest()
        degradation_results = await degradation_test.run_all_tests()
        assert all(r.passed for r in degradation_results)

    @pytest.mark.chaos
    @pytest.mark.integration
    @pytest.mark.kubernetes
    @pytest.mark.asyncio
    async def test_k8s_chaos_suite(self):
        """Test all Kubernetes chaos scenarios."""
        runner = K8sChaosTestRunner()
        results = await runner.run_all()

        passed = sum(1 for r in results if r.passed)
        total = len(results)

        assert passed == total, f"K8s chaos tests: {passed}/{total} passed"


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with chaos markers."""
    config.addinivalue_line("markers", "chaos: Chaos engineering tests")
    config.addinivalue_line("markers", "kubernetes: Kubernetes-specific tests")
    config.addinivalue_line("markers", "integration: Integration tests")

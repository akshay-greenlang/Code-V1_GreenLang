"""
GL-001 ThermalCommand - Steady State Hypothesis Testing

This module provides steady state hypothesis definition and validation
for chaos engineering experiments.

A steady state hypothesis defines:
- Normal operation metrics and their acceptable ranges
- Pre-experiment validation (system is healthy before chaos)
- Post-experiment validation (system recovers after chaos)

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for steady state validation."""
    GAUGE = "gauge"           # Point-in-time value
    COUNTER = "counter"       # Cumulative value
    HISTOGRAM = "histogram"   # Distribution
    RATE = "rate"            # Rate of change


class ComparisonOperator(Enum):
    """Comparison operators for threshold validation."""
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    EQUAL = "eq"
    NOT_EQUAL = "neq"
    IN_RANGE = "in_range"
    NOT_IN_RANGE = "not_in_range"


@dataclass
class SteadyStateMetric:
    """
    Definition of a steady state metric.

    Attributes:
        name: Metric name/identifier
        metric_type: Type of metric
        description: Human-readable description
        threshold: Expected value or threshold
        operator: Comparison operator
        tolerance: Acceptable tolerance (percentage)
        weight: Importance weight (1.0 = normal)
        required: Whether metric must pass for hypothesis to pass
    """
    name: str
    metric_type: MetricType = MetricType.GAUGE
    description: str = ""
    threshold: Union[float, Tuple[float, float]] = 0.0
    operator: ComparisonOperator = ComparisonOperator.LESS_THAN
    tolerance: float = 0.05  # 5% tolerance
    weight: float = 1.0
    required: bool = True

    def validate(self, value: float) -> Tuple[bool, str]:
        """
        Validate a metric value against the threshold.

        Args:
            value: The observed metric value

        Returns:
            Tuple of (passed, message)
        """
        if value is None:
            return False, f"{self.name}: No value provided"

        passed = False
        message = ""

        if self.operator == ComparisonOperator.LESS_THAN:
            passed = value < self.threshold
            message = f"{self.name}: {value} < {self.threshold}"

        elif self.operator == ComparisonOperator.LESS_THAN_OR_EQUAL:
            passed = value <= self.threshold
            message = f"{self.name}: {value} <= {self.threshold}"

        elif self.operator == ComparisonOperator.GREATER_THAN:
            passed = value > self.threshold
            message = f"{self.name}: {value} > {self.threshold}"

        elif self.operator == ComparisonOperator.GREATER_THAN_OR_EQUAL:
            passed = value >= self.threshold
            message = f"{self.name}: {value} >= {self.threshold}"

        elif self.operator == ComparisonOperator.EQUAL:
            # Apply tolerance for equality
            tolerance_value = abs(self.threshold * self.tolerance)
            passed = abs(value - self.threshold) <= tolerance_value
            message = f"{self.name}: {value} ~= {self.threshold} (+/-{self.tolerance*100}%)"

        elif self.operator == ComparisonOperator.NOT_EQUAL:
            tolerance_value = abs(self.threshold * self.tolerance)
            passed = abs(value - self.threshold) > tolerance_value
            message = f"{self.name}: {value} != {self.threshold}"

        elif self.operator == ComparisonOperator.IN_RANGE:
            if isinstance(self.threshold, tuple):
                min_val, max_val = self.threshold
                passed = min_val <= value <= max_val
                message = f"{self.name}: {min_val} <= {value} <= {max_val}"
            else:
                passed = False
                message = f"{self.name}: IN_RANGE requires tuple threshold"

        elif self.operator == ComparisonOperator.NOT_IN_RANGE:
            if isinstance(self.threshold, tuple):
                min_val, max_val = self.threshold
                passed = value < min_val or value > max_val
                message = f"{self.name}: {value} outside [{min_val}, {max_val}]"
            else:
                passed = False
                message = f"{self.name}: NOT_IN_RANGE requires tuple threshold"

        status = "PASS" if passed else "FAIL"
        return passed, f"[{status}] {message}"


@dataclass
class SteadyStateHypothesis:
    """
    Steady state hypothesis for chaos experiments.

    The hypothesis defines:
    1. What metrics define "steady state"
    2. What thresholds indicate healthy operation
    3. How to aggregate multiple metrics

    Example:
        >>> hypothesis = SteadyStateHypothesis(
        ...     name="API Health",
        ...     metrics=[
        ...         SteadyStateMetric(
        ...             name="response_time_ms",
        ...             threshold=200,
        ...             operator=ComparisonOperator.LESS_THAN
        ...         ),
        ...         SteadyStateMetric(
        ...             name="error_rate_percent",
        ...             threshold=1.0,
        ...             operator=ComparisonOperator.LESS_THAN
        ...         ),
        ...     ]
        ... )
    """
    name: str
    description: str = ""
    metrics: List[SteadyStateMetric] = field(default_factory=list)
    pass_threshold: float = 1.0  # 100% of metrics must pass by default
    aggregation: str = "all"  # "all", "any", "weighted"
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    def add_metric(self, metric: SteadyStateMetric) -> None:
        """Add a metric to the hypothesis."""
        self.metrics.append(metric)

    def remove_metric(self, name: str) -> bool:
        """Remove a metric by name."""
        original_count = len(self.metrics)
        self.metrics = [m for m in self.metrics if m.name != name]
        return len(self.metrics) < original_count


@dataclass
class SteadyStateResult:
    """Result of steady state validation."""
    hypothesis_name: str
    passed: bool
    timestamp: datetime
    metric_results: Dict[str, Tuple[bool, str]]
    aggregate_score: float
    duration_seconds: float
    retry_count: int
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "hypothesis_name": self.hypothesis_name,
            "passed": self.passed,
            "timestamp": self.timestamp.isoformat(),
            "metric_results": {
                k: {"passed": v[0], "message": v[1]}
                for k, v in self.metric_results.items()
            },
            "aggregate_score": self.aggregate_score,
            "duration_seconds": self.duration_seconds,
            "retry_count": self.retry_count,
            "errors": self.errors,
        }


class MetricsProvider(ABC):
    """Abstract base class for metrics providers."""

    @abstractmethod
    async def get_metric(self, name: str) -> Optional[float]:
        """Get a single metric value."""
        pass

    @abstractmethod
    async def get_all_metrics(self) -> Dict[str, float]:
        """Get all available metrics."""
        pass


class SimulatedMetricsProvider(MetricsProvider):
    """
    Simulated metrics provider for testing.

    Provides realistic-looking metrics without actual system access.
    """

    def __init__(self, base_values: Optional[Dict[str, float]] = None):
        """
        Initialize with optional base values.

        Args:
            base_values: Base values for metrics (will add noise)
        """
        import random

        self._base_values = base_values or {
            "response_time_ms": 50.0,
            "error_rate_percent": 0.5,
            "throughput_rps": 1000.0,
            "cpu_usage_percent": 40.0,
            "memory_usage_mb": 256.0,
            "active_connections": 100.0,
            "queue_depth": 5.0,
            "p99_latency_ms": 150.0,
            "p95_latency_ms": 100.0,
            "success_rate_percent": 99.5,
        }
        self._random = random

    async def get_metric(self, name: str) -> Optional[float]:
        """Get a single metric with simulated noise."""
        base = self._base_values.get(name)
        if base is None:
            return None

        # Add realistic noise (5-10% variation)
        noise = self._random.uniform(-0.1, 0.1) * base
        return base + noise

    async def get_all_metrics(self) -> Dict[str, float]:
        """Get all metrics with simulated noise."""
        metrics = {}
        for name in self._base_values:
            value = await self.get_metric(name)
            if value is not None:
                metrics[name] = value
        return metrics


class SteadyStateValidator:
    """
    Validates steady state hypotheses against observed metrics.

    Example:
        >>> validator = SteadyStateValidator()
        >>> hypothesis = SteadyStateHypothesis(...)
        >>> result = await validator.validate(hypothesis)
        >>> if result.passed:
        ...     print("System is in steady state")
    """

    def __init__(
        self,
        metrics_provider: Optional[MetricsProvider] = None,
    ):
        """
        Initialize validator.

        Args:
            metrics_provider: Provider for system metrics
        """
        self.metrics_provider = metrics_provider or SimulatedMetricsProvider()

    async def validate(
        self,
        hypothesis: SteadyStateHypothesis,
    ) -> SteadyStateResult:
        """
        Validate a steady state hypothesis.

        Args:
            hypothesis: The hypothesis to validate

        Returns:
            SteadyStateResult with validation outcome
        """
        import time

        start_time = time.time()
        metric_results: Dict[str, Tuple[bool, str]] = {}
        errors: List[str] = []
        retry_count = 0

        for attempt in range(hypothesis.retry_count):
            retry_count = attempt + 1
            metric_results.clear()
            errors.clear()

            try:
                # Get all metrics
                metrics = await self.metrics_provider.get_all_metrics()

                # Validate each metric in hypothesis
                for metric_def in hypothesis.metrics:
                    value = metrics.get(metric_def.name)

                    if value is None:
                        if metric_def.required:
                            metric_results[metric_def.name] = (
                                False,
                                f"[FAIL] {metric_def.name}: Metric not available"
                            )
                            errors.append(f"Required metric not available: {metric_def.name}")
                        continue

                    passed, message = metric_def.validate(value)
                    metric_results[metric_def.name] = (passed, message)

                # Calculate aggregate score
                aggregate_score = self._calculate_aggregate_score(
                    hypothesis, metric_results
                )

                # Check if hypothesis passed
                passed = self._check_hypothesis_passed(hypothesis, metric_results, aggregate_score)

                if passed:
                    break

                # Retry delay
                if attempt < hypothesis.retry_count - 1:
                    await asyncio.sleep(hypothesis.retry_delay_seconds)

            except Exception as e:
                errors.append(f"Validation error: {str(e)}")
                logger.error(f"Steady state validation error: {e}", exc_info=True)

        duration_seconds = time.time() - start_time

        # Final determination
        aggregate_score = self._calculate_aggregate_score(hypothesis, metric_results)
        passed = self._check_hypothesis_passed(hypothesis, metric_results, aggregate_score)

        result = SteadyStateResult(
            hypothesis_name=hypothesis.name,
            passed=passed,
            timestamp=datetime.now(timezone.utc),
            metric_results=metric_results,
            aggregate_score=aggregate_score,
            duration_seconds=duration_seconds,
            retry_count=retry_count,
            errors=errors,
        )

        logger.info(
            f"Steady state validation: {hypothesis.name} - "
            f"{'PASSED' if passed else 'FAILED'} (score={aggregate_score:.2f})"
        )

        return result

    def _calculate_aggregate_score(
        self,
        hypothesis: SteadyStateHypothesis,
        metric_results: Dict[str, Tuple[bool, str]],
    ) -> float:
        """Calculate aggregate score based on aggregation method."""
        if not metric_results:
            return 0.0

        if hypothesis.aggregation == "all":
            # All required metrics must pass
            passed_count = sum(1 for passed, _ in metric_results.values() if passed)
            return passed_count / len(metric_results)

        elif hypothesis.aggregation == "any":
            # At least one metric must pass
            return 1.0 if any(passed for passed, _ in metric_results.values()) else 0.0

        elif hypothesis.aggregation == "weighted":
            # Weighted average based on metric weights
            total_weight = 0.0
            weighted_sum = 0.0

            for metric_def in hypothesis.metrics:
                result = metric_results.get(metric_def.name)
                if result is not None:
                    passed, _ = result
                    weighted_sum += metric_def.weight * (1.0 if passed else 0.0)
                    total_weight += metric_def.weight

            return weighted_sum / total_weight if total_weight > 0 else 0.0

        return 0.0

    def _check_hypothesis_passed(
        self,
        hypothesis: SteadyStateHypothesis,
        metric_results: Dict[str, Tuple[bool, str]],
        aggregate_score: float,
    ) -> bool:
        """Determine if hypothesis passed based on results."""
        # Check required metrics
        for metric_def in hypothesis.metrics:
            if metric_def.required:
                result = metric_results.get(metric_def.name)
                if result is None or not result[0]:
                    return False

        # Check aggregate threshold
        return aggregate_score >= hypothesis.pass_threshold


# =============================================================================
# Pre-built Hypothesis Templates
# =============================================================================

def create_api_health_hypothesis(
    max_response_time_ms: float = 200,
    max_error_rate_percent: float = 1.0,
    min_success_rate_percent: float = 99.0,
) -> SteadyStateHypothesis:
    """Create a standard API health hypothesis."""
    return SteadyStateHypothesis(
        name="API Health",
        description="Validates API is responding within acceptable parameters",
        metrics=[
            SteadyStateMetric(
                name="response_time_ms",
                description="Average response time",
                threshold=max_response_time_ms,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="error_rate_percent",
                description="Error rate percentage",
                threshold=max_error_rate_percent,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="success_rate_percent",
                description="Success rate percentage",
                threshold=min_success_rate_percent,
                operator=ComparisonOperator.GREATER_THAN_OR_EQUAL,
                required=True,
            ),
        ],
    )


def create_resource_health_hypothesis(
    max_cpu_percent: float = 80,
    max_memory_mb: float = 1024,
    max_queue_depth: int = 100,
) -> SteadyStateHypothesis:
    """Create a standard resource health hypothesis."""
    return SteadyStateHypothesis(
        name="Resource Health",
        description="Validates system resources are within acceptable limits",
        metrics=[
            SteadyStateMetric(
                name="cpu_usage_percent",
                description="CPU usage percentage",
                threshold=max_cpu_percent,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="memory_usage_mb",
                description="Memory usage in MB",
                threshold=max_memory_mb,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="queue_depth",
                description="Message queue depth",
                threshold=float(max_queue_depth),
                operator=ComparisonOperator.LESS_THAN,
                required=False,
            ),
        ],
    )


def create_thermal_command_hypothesis() -> SteadyStateHypothesis:
    """Create hypothesis specific to ThermalCommand orchestrator."""
    return SteadyStateHypothesis(
        name="ThermalCommand Health",
        description="Validates ThermalCommand orchestrator is operating normally",
        metrics=[
            SteadyStateMetric(
                name="response_time_ms",
                description="API response time",
                threshold=100,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="error_rate_percent",
                description="Error rate",
                threshold=0.5,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="active_connections",
                description="Active agent connections",
                threshold=(1, 1000),
                operator=ComparisonOperator.IN_RANGE,
                required=False,
            ),
            SteadyStateMetric(
                name="cpu_usage_percent",
                description="CPU usage",
                threshold=70,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
            SteadyStateMetric(
                name="memory_usage_mb",
                description="Memory usage",
                threshold=512,
                operator=ComparisonOperator.LESS_THAN,
                required=True,
            ),
        ],
        pass_threshold=0.8,  # 80% of metrics must pass
        aggregation="weighted",
    )

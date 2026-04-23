"""
GL-001 ThermalCommand - Chaos Runner Framework

This module provides the core chaos engineering framework for orchestrating
chaos experiments safely in test and production environments.

Features:
- Experiment lifecycle management
- Rollback mechanisms for safety
- Metric collection during chaos
- Blast radius control
- Dry-run mode for validation

Author: GreenLang Chaos Engineering Team
Version: 1.0.0
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

logger = logging.getLogger(__name__)


class ChaosPhase(Enum):
    """Phases of a chaos experiment."""
    INITIALIZED = "initialized"
    STEADY_STATE_CHECK = "steady_state_check"
    INJECTING = "injecting"
    OBSERVING = "observing"
    ROLLING_BACK = "rolling_back"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class ChaosSeverity(Enum):
    """Severity levels for chaos experiments."""
    LOW = "low"           # Minimal impact, quick recovery
    MEDIUM = "medium"     # Moderate impact, standard recovery
    HIGH = "high"         # Significant impact, extended recovery
    CRITICAL = "critical" # Maximum impact, for disaster recovery testing


@dataclass
class ChaosResult:
    """Result of a chaos experiment execution."""
    experiment_id: str
    experiment_name: str
    status: str
    phase: ChaosPhase
    started_at: datetime
    completed_at: Optional[datetime]
    duration_seconds: float
    steady_state_before: bool
    steady_state_after: bool
    hypothesis_validated: bool
    metrics_collected: Dict[str, Any]
    errors: List[str]
    rollback_executed: bool
    rollback_success: bool
    observations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "status": self.status,
            "phase": self.phase.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "steady_state_before": self.steady_state_before,
            "steady_state_after": self.steady_state_after,
            "hypothesis_validated": self.hypothesis_validated,
            "metrics_collected": self.metrics_collected,
            "errors": self.errors,
            "rollback_executed": self.rollback_executed,
            "rollback_success": self.rollback_success,
            "observations": self.observations,
        }


@dataclass
class ChaosExperiment:
    """Definition of a chaos experiment."""
    name: str
    description: str
    severity: ChaosSeverity = ChaosSeverity.LOW
    duration_seconds: float = 30.0
    cooldown_seconds: float = 10.0

    # Fault injection configuration
    fault_type: str = ""
    fault_params: Dict[str, Any] = field(default_factory=dict)

    # Steady state definition
    steady_state_metrics: List[str] = field(default_factory=list)
    steady_state_thresholds: Dict[str, float] = field(default_factory=dict)

    # Safety controls
    max_impact_percentage: float = 10.0  # Max percentage of system affected
    abort_conditions: List[str] = field(default_factory=list)
    require_manual_approval: bool = False

    # Metadata
    tags: List[str] = field(default_factory=list)
    owner: str = "chaos-engineering"
    environment: str = "test"

    def __post_init__(self):
        """Validate experiment configuration."""
        if self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if self.max_impact_percentage > 100 or self.max_impact_percentage < 0:
            raise ValueError("Max impact percentage must be between 0 and 100")


class FaultInjector(ABC):
    """Base class for fault injectors."""

    @abstractmethod
    async def inject(self, params: Dict[str, Any]) -> bool:
        """Inject the fault. Returns True if successful."""
        pass

    @abstractmethod
    async def rollback(self) -> bool:
        """Rollback the fault. Returns True if successful."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Check if fault is currently active."""
        pass


class MetricsCollector(ABC):
    """Base class for metrics collection during chaos."""

    @abstractmethod
    async def collect(self) -> Dict[str, Any]:
        """Collect current metrics."""
        pass

    @abstractmethod
    def get_metric(self, name: str) -> Optional[float]:
        """Get specific metric value."""
        pass


class DefaultMetricsCollector(MetricsCollector):
    """Default metrics collector using simulated values."""

    def __init__(self):
        self._metrics: Dict[str, float] = {}
        self._start_time = time.time()

    async def collect(self) -> Dict[str, Any]:
        """Collect simulated metrics."""
        import random

        elapsed = time.time() - self._start_time

        self._metrics = {
            "response_time_ms": 50 + random.uniform(-10, 30),
            "error_rate_percent": random.uniform(0, 2),
            "throughput_rps": 1000 + random.uniform(-100, 100),
            "cpu_usage_percent": 40 + random.uniform(-10, 20),
            "memory_usage_mb": 256 + random.uniform(-20, 50),
            "active_connections": int(100 + random.uniform(-20, 30)),
            "queue_depth": int(random.uniform(0, 10)),
            "uptime_seconds": elapsed,
        }

        return self._metrics

    def get_metric(self, name: str) -> Optional[float]:
        """Get specific metric value."""
        return self._metrics.get(name)


class ChaosRunner:
    """
    Main chaos experiment runner.

    Orchestrates the lifecycle of chaos experiments including:
    - Pre-flight checks
    - Steady state validation
    - Fault injection
    - Observation period
    - Rollback
    - Post-experiment validation

    Example:
        >>> runner = ChaosRunner()
        >>> experiment = ChaosExperiment(
        ...     name="network_latency_test",
        ...     description="Test system resilience to network latency",
        ...     fault_type="network_latency",
        ...     fault_params={"delay_ms": 200}
        ... )
        >>> result = await runner.run(experiment)
        >>> assert result.hypothesis_validated
    """

    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        dry_run: bool = False,
        abort_on_failure: bool = True,
    ):
        """
        Initialize chaos runner.

        Args:
            metrics_collector: Custom metrics collector
            dry_run: If True, simulate fault injection without actual changes
            abort_on_failure: If True, abort experiment on any error
        """
        self.metrics_collector = metrics_collector or DefaultMetricsCollector()
        self.dry_run = dry_run
        self.abort_on_failure = abort_on_failure

        self._fault_injectors: Dict[str, Type[FaultInjector]] = {}
        self._active_experiments: Dict[str, ChaosExperiment] = {}
        self._experiment_history: List[ChaosResult] = []
        self._abort_flag = False

        logger.info(f"ChaosRunner initialized (dry_run={dry_run})")

    def register_injector(self, fault_type: str, injector_class: Type[FaultInjector]) -> None:
        """Register a fault injector for a specific fault type."""
        self._fault_injectors[fault_type] = injector_class
        logger.debug(f"Registered fault injector: {fault_type}")

    async def run(self, experiment: ChaosExperiment) -> ChaosResult:
        """
        Run a chaos experiment.

        Args:
            experiment: The experiment to run

        Returns:
            ChaosResult with experiment outcome
        """
        experiment_id = str(uuid.uuid4())[:8]
        started_at = datetime.now(timezone.utc)
        phase = ChaosPhase.INITIALIZED
        errors: List[str] = []
        observations: List[str] = []
        rollback_executed = False
        rollback_success = False
        steady_state_before = False
        steady_state_after = False
        hypothesis_validated = False
        metrics_collected: Dict[str, Any] = {}

        logger.info(f"Starting chaos experiment: {experiment.name} (id={experiment_id})")

        try:
            # Phase 1: Pre-flight steady state check
            phase = ChaosPhase.STEADY_STATE_CHECK
            observations.append(f"Phase: Checking steady state before injection")

            steady_state_before = await self._check_steady_state(experiment)

            if not steady_state_before:
                errors.append("System not in steady state before experiment")
                if self.abort_on_failure:
                    phase = ChaosPhase.ABORTED
                    raise RuntimeError("Aborting: System not in steady state")

            observations.append(f"Steady state before: {steady_state_before}")

            # Phase 2: Inject fault
            phase = ChaosPhase.INJECTING
            observations.append(f"Phase: Injecting fault ({experiment.fault_type})")

            injector = await self._get_injector(experiment)

            if not self.dry_run:
                injection_success = await injector.inject(experiment.fault_params)
                if not injection_success:
                    errors.append("Fault injection failed")
                    if self.abort_on_failure:
                        raise RuntimeError("Fault injection failed")
            else:
                observations.append("DRY RUN: Fault injection simulated")

            # Phase 3: Observation period
            phase = ChaosPhase.OBSERVING
            observations.append(f"Phase: Observing for {experiment.duration_seconds}s")

            # Collect metrics during observation
            observation_interval = min(1.0, experiment.duration_seconds / 10)
            elapsed = 0.0

            while elapsed < experiment.duration_seconds:
                if self._abort_flag:
                    observations.append("Experiment aborted by user")
                    break

                # Check abort conditions
                if await self._check_abort_conditions(experiment):
                    observations.append("Abort condition triggered")
                    break

                current_metrics = await self.metrics_collector.collect()
                metrics_collected[f"t_{elapsed:.1f}s"] = current_metrics

                await asyncio.sleep(observation_interval)
                elapsed += observation_interval

            # Phase 4: Rollback
            phase = ChaosPhase.ROLLING_BACK
            observations.append("Phase: Rolling back fault injection")

            if not self.dry_run and injector.is_active():
                rollback_executed = True
                rollback_success = await injector.rollback()

                if not rollback_success:
                    errors.append("Rollback failed - manual intervention may be required")
            else:
                rollback_success = True
                observations.append("DRY RUN: Rollback simulated")

            # Phase 5: Cooldown
            observations.append(f"Cooldown period: {experiment.cooldown_seconds}s")
            await asyncio.sleep(experiment.cooldown_seconds)

            # Phase 6: Post-experiment validation
            phase = ChaosPhase.VALIDATING
            observations.append("Phase: Validating steady state after experiment")

            steady_state_after = await self._check_steady_state(experiment)
            observations.append(f"Steady state after: {steady_state_after}")

            # Validate hypothesis
            hypothesis_validated = steady_state_before and steady_state_after

            phase = ChaosPhase.COMPLETED
            status = "success" if hypothesis_validated else "failed"

        except Exception as e:
            errors.append(str(e))
            status = "error"
            logger.error(f"Chaos experiment error: {e}", exc_info=True)

            # Attempt emergency rollback
            if not self.dry_run:
                try:
                    injector = await self._get_injector(experiment)
                    if injector.is_active():
                        rollback_executed = True
                        rollback_success = await injector.rollback()
                except Exception as rollback_error:
                    errors.append(f"Emergency rollback failed: {rollback_error}")

        finally:
            completed_at = datetime.now(timezone.utc)
            duration_seconds = (completed_at - started_at).total_seconds()

            # Cleanup
            if experiment_id in self._active_experiments:
                del self._active_experiments[experiment_id]

        result = ChaosResult(
            experiment_id=experiment_id,
            experiment_name=experiment.name,
            status=status,
            phase=phase,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration_seconds,
            steady_state_before=steady_state_before,
            steady_state_after=steady_state_after,
            hypothesis_validated=hypothesis_validated,
            metrics_collected=metrics_collected,
            errors=errors,
            rollback_executed=rollback_executed,
            rollback_success=rollback_success,
            observations=observations,
        )

        self._experiment_history.append(result)

        logger.info(
            f"Chaos experiment completed: {experiment.name} "
            f"(status={status}, hypothesis_validated={hypothesis_validated})"
        )

        return result

    async def run_suite(self, experiments: List[ChaosExperiment]) -> List[ChaosResult]:
        """Run multiple chaos experiments in sequence."""
        results = []

        for experiment in experiments:
            result = await self.run(experiment)
            results.append(result)

            if result.status == "error" and self.abort_on_failure:
                logger.warning("Suite aborted due to experiment failure")
                break

        return results

    def abort(self) -> None:
        """Abort current experiment."""
        self._abort_flag = True
        logger.warning("Chaos experiment abort requested")

    def reset_abort(self) -> None:
        """Reset abort flag."""
        self._abort_flag = False

    async def _check_steady_state(self, experiment: ChaosExperiment) -> bool:
        """Check if system is in steady state."""
        try:
            metrics = await self.metrics_collector.collect()

            for metric_name, threshold in experiment.steady_state_thresholds.items():
                value = metrics.get(metric_name)
                if value is None:
                    continue

                # Simple threshold check (can be extended)
                if isinstance(threshold, tuple):
                    min_val, max_val = threshold
                    if not (min_val <= value <= max_val):
                        logger.debug(f"Steady state failed: {metric_name}={value} not in [{min_val}, {max_val}]")
                        return False
                else:
                    if value > threshold:
                        logger.debug(f"Steady state failed: {metric_name}={value} > {threshold}")
                        return False

            return True

        except Exception as e:
            logger.error(f"Steady state check error: {e}")
            return False

    async def _check_abort_conditions(self, experiment: ChaosExperiment) -> bool:
        """Check if any abort conditions are met."""
        # In production, this would check actual conditions
        # For testing, we simulate based on error rate
        metrics = await self.metrics_collector.collect()

        error_rate = metrics.get("error_rate_percent", 0)
        if error_rate > experiment.max_impact_percentage * 10:
            logger.warning(f"Abort condition: error_rate={error_rate}% exceeds threshold")
            return True

        return False

    async def _get_injector(self, experiment: ChaosExperiment) -> FaultInjector:
        """Get or create fault injector for experiment."""
        fault_type = experiment.fault_type

        if fault_type not in self._fault_injectors:
            # Use a no-op injector for unknown types
            from .fault_injectors import NoOpFaultInjector
            return NoOpFaultInjector()

        injector_class = self._fault_injectors[fault_type]
        return injector_class()

    def get_history(self) -> List[ChaosResult]:
        """Get experiment history."""
        return self._experiment_history.copy()

    def clear_history(self) -> None:
        """Clear experiment history."""
        self._experiment_history.clear()


@asynccontextmanager
async def chaos_context(
    experiment: ChaosExperiment,
    runner: Optional[ChaosRunner] = None,
):
    """
    Context manager for running chaos experiments.

    Example:
        >>> async with chaos_context(experiment) as result:
        ...     # Experiment runs here
        ...     pass
        >>> print(result.hypothesis_validated)
    """
    runner = runner or ChaosRunner()
    result = None

    try:
        result = await runner.run(experiment)
        yield result
    finally:
        if result is None:
            # Experiment didn't complete - create failure result
            yield ChaosResult(
                experiment_id="unknown",
                experiment_name=experiment.name,
                status="error",
                phase=ChaosPhase.FAILED,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
                duration_seconds=0.0,
                steady_state_before=False,
                steady_state_after=False,
                hypothesis_validated=False,
                metrics_collected={},
                errors=["Context manager exited without result"],
                rollback_executed=False,
                rollback_success=False,
                observations=[],
            )

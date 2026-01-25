"""
Solver Monitor Module - GL-016_Waterguard Optimization Service

This module provides comprehensive monitoring for optimization solvers,
including timeout handling, fallback solutions, diagnostics logging,
and performance metrics tracking.

Key Components:
    - SolverMonitor: Core monitoring class for tracking solver execution
    - SolverDiagnostics: Detailed diagnostics for solver performance
    - TimeoutHandler: Manages solver timeouts with graceful fallback
    - PerformanceTracker: Aggregates solver performance metrics

Example:
    >>> monitor = SolverMonitor(config)
    >>> with monitor.track_solve("blowdown_optimization"):
    ...     result = optimizer.optimize(problem)
    >>> metrics = monitor.get_performance_metrics()
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
from contextlib import contextmanager
import threading
import hashlib
import logging
import time
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SolverStatus(str, Enum):
    """Status of solver execution."""
    PENDING = "pending"
    RUNNING = "running"
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIMEOUT = "timeout"
    ERROR = "error"
    FALLBACK = "fallback"


class SolverType(str, Enum):
    """Type of solver being used."""
    CVXPY_ECOS = "cvxpy_ecos"
    CVXPY_SCS = "cvxpy_scs"
    CVXPY_OSQP = "cvxpy_osqp"
    CVXPY_CBC = "cvxpy_cbc"
    CVXPY_GLPK = "cvxpy_glpk"
    SCIPY_LINPROG = "scipy_linprog"
    SCIPY_MINIMIZE = "scipy_minimize"
    HEURISTIC = "heuristic"
    CONSERVATIVE = "conservative"


class AlertLevel(str, Enum):
    """Alert level for solver issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Data Models
# =============================================================================

class SolverConfig(BaseModel):
    """Configuration for solver monitoring."""

    default_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Default solver timeout in seconds"
    )
    max_iterations: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum solver iterations"
    )
    fallback_enabled: bool = Field(
        default=True,
        description="Enable fallback to last good solution on timeout"
    )
    diagnostics_enabled: bool = Field(
        default=True,
        description="Enable detailed diagnostics logging"
    )
    metrics_window_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of solves to keep for metrics calculation"
    )
    alert_threshold_seconds: float = Field(
        default=10.0,
        ge=1.0,
        description="Solve time threshold for performance alerts"
    )
    gap_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Acceptable optimality gap for MIP solvers"
    )


class SolveAttempt(BaseModel):
    """Record of a single solve attempt."""

    solve_id: str = Field(..., description="Unique solve identifier")
    problem_id: str = Field(..., description="Problem identifier")
    solver_type: SolverType = Field(..., description="Solver used")
    status: SolverStatus = Field(..., description="Solve status")
    solve_time_seconds: float = Field(..., ge=0.0, description="Time to solve")
    iterations: Optional[int] = Field(None, ge=0, description="Solver iterations")
    optimality_gap: Optional[float] = Field(None, ge=0.0, description="MIP gap if applicable")
    objective_value: Optional[float] = Field(None, description="Objective function value")
    constraint_violations: int = Field(default=0, ge=0, description="Number of constraint violations")
    timestamp: datetime = Field(default_factory=datetime.now, description="Solve timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    used_fallback: bool = Field(default=False, description="Whether fallback was used")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    @validator('provenance_hash', always=True)
    def compute_provenance(cls, v, values):
        """Compute provenance hash from solve attempt data."""
        if v:
            return v
        data_str = (
            f"{values.get('solve_id', '')}"
            f"{values.get('problem_id', '')}"
            f"{values.get('solver_type', '')}"
            f"{values.get('status', '')}"
            f"{values.get('solve_time_seconds', 0.0)}"
            f"{values.get('objective_value', '')}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class SolverDiagnostics(BaseModel):
    """Detailed diagnostics from a solve."""

    solve_id: str = Field(..., description="Associated solve ID")
    problem_size: Dict[str, int] = Field(
        default_factory=dict,
        description="Problem dimensions (variables, constraints, etc.)"
    )
    solver_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Solver-specific information"
    )
    timing_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Time spent in different phases"
    )
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Peak memory usage")
    numerical_issues: List[str] = Field(
        default_factory=list,
        description="Any numerical issues detected"
    )
    binding_constraints: List[str] = Field(
        default_factory=list,
        description="List of binding constraint names"
    )
    sensitivity_info: Dict[str, float] = Field(
        default_factory=dict,
        description="Sensitivity/dual values for key constraints"
    )


class PerformanceMetrics(BaseModel):
    """Aggregated performance metrics for solver monitoring."""

    total_solves: int = Field(default=0, ge=0, description="Total number of solves")
    successful_solves: int = Field(default=0, ge=0, description="Successful solves")
    failed_solves: int = Field(default=0, ge=0, description="Failed solves")
    timeout_count: int = Field(default=0, ge=0, description="Number of timeouts")
    fallback_count: int = Field(default=0, ge=0, description="Number of fallbacks used")

    avg_solve_time_seconds: float = Field(default=0.0, ge=0.0, description="Average solve time")
    median_solve_time_seconds: float = Field(default=0.0, ge=0.0, description="Median solve time")
    p95_solve_time_seconds: float = Field(default=0.0, ge=0.0, description="95th percentile solve time")
    max_solve_time_seconds: float = Field(default=0.0, ge=0.0, description="Maximum solve time")

    avg_iterations: float = Field(default=0.0, ge=0.0, description="Average iterations")
    avg_optimality_gap: float = Field(default=0.0, ge=0.0, description="Average optimality gap")

    success_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Success rate")
    solver_availability: float = Field(default=1.0, ge=0.0, le=1.0, description="Solver availability")

    last_updated: datetime = Field(default_factory=datetime.now, description="Last metrics update")
    window_start: Optional[datetime] = Field(None, description="Metrics window start")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")


class SolverAlert(BaseModel):
    """Alert generated by solver monitoring."""

    alert_id: str = Field(..., description="Unique alert identifier")
    level: AlertLevel = Field(..., description="Alert severity level")
    message: str = Field(..., description="Alert message")
    solve_id: Optional[str] = Field(None, description="Associated solve ID if applicable")
    metric_name: Optional[str] = Field(None, description="Metric that triggered alert")
    metric_value: Optional[float] = Field(None, description="Current metric value")
    threshold_value: Optional[float] = Field(None, description="Threshold that was exceeded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Alert timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert was acknowledged")


class FallbackSolution(BaseModel):
    """Stored fallback solution for timeout recovery."""

    problem_id: str = Field(..., description="Problem identifier")
    solution: Dict[str, Any] = Field(..., description="Solution values")
    objective_value: float = Field(..., description="Objective value of solution")
    created_at: datetime = Field(default_factory=datetime.now, description="When solution was created")
    solver_type: SolverType = Field(..., description="Solver that produced solution")
    validity_period_hours: float = Field(default=24.0, ge=1.0, description="How long solution is valid")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    def is_valid(self) -> bool:
        """Check if fallback solution is still valid."""
        age = datetime.now() - self.created_at
        return age.total_seconds() < self.validity_period_hours * 3600


# =============================================================================
# Timeout Handler
# =============================================================================

class TimeoutHandler:
    """
    Handles solver timeouts with graceful fallback.

    Provides mechanisms for:
    - Setting solve time limits
    - Interrupting long-running solves
    - Falling back to cached solutions

    Attributes:
        config: Solver configuration
        fallback_solutions: Cached fallback solutions by problem ID
    """

    def __init__(self, config: SolverConfig):
        """Initialize timeout handler."""
        self.config = config
        self.fallback_solutions: Dict[str, FallbackSolution] = {}
        self._current_solve_start: Optional[datetime] = None
        self._timeout_event = threading.Event()
        logger.info("TimeoutHandler initialized with default timeout: %.1fs",
                   config.default_timeout_seconds)

    def start_timer(self, timeout_seconds: Optional[float] = None) -> None:
        """Start timeout timer for a solve."""
        self._current_solve_start = datetime.now()
        self._timeout_event.clear()
        timeout = timeout_seconds or self.config.default_timeout_seconds
        logger.debug("Solve timer started with timeout: %.1fs", timeout)

    def check_timeout(self, timeout_seconds: Optional[float] = None) -> bool:
        """Check if current solve has exceeded timeout."""
        if self._current_solve_start is None:
            return False

        elapsed = (datetime.now() - self._current_solve_start).total_seconds()
        timeout = timeout_seconds or self.config.default_timeout_seconds

        if elapsed > timeout:
            logger.warning("Solve timeout exceeded: %.1fs > %.1fs", elapsed, timeout)
            return True
        return False

    def get_elapsed_time(self) -> float:
        """Get elapsed time since solve started."""
        if self._current_solve_start is None:
            return 0.0
        return (datetime.now() - self._current_solve_start).total_seconds()

    def stop_timer(self) -> float:
        """Stop timeout timer and return elapsed time."""
        elapsed = self.get_elapsed_time()
        self._current_solve_start = None
        return elapsed

    def store_fallback(
        self,
        problem_id: str,
        solution: Dict[str, Any],
        objective_value: float,
        solver_type: SolverType,
        validity_hours: float = 24.0
    ) -> None:
        """Store a solution as fallback for future timeout recovery."""
        provenance_str = f"{problem_id}{solution}{objective_value}{datetime.now().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

        fallback = FallbackSolution(
            problem_id=problem_id,
            solution=solution,
            objective_value=objective_value,
            solver_type=solver_type,
            validity_period_hours=validity_hours,
            provenance_hash=provenance_hash
        )
        self.fallback_solutions[problem_id] = fallback
        logger.info("Stored fallback solution for problem '%s' (valid for %.1f hours)",
                   problem_id, validity_hours)

    def get_fallback(self, problem_id: str) -> Optional[FallbackSolution]:
        """Get fallback solution if available and valid."""
        fallback = self.fallback_solutions.get(problem_id)
        if fallback is None:
            logger.debug("No fallback solution found for problem '%s'", problem_id)
            return None

        if not fallback.is_valid():
            logger.warning("Fallback solution for '%s' has expired", problem_id)
            del self.fallback_solutions[problem_id]
            return None

        logger.info("Retrieved valid fallback solution for '%s'", problem_id)
        return fallback

    def clear_fallback(self, problem_id: str) -> bool:
        """Clear fallback solution for a problem."""
        if problem_id in self.fallback_solutions:
            del self.fallback_solutions[problem_id]
            logger.info("Cleared fallback solution for problem '%s'", problem_id)
            return True
        return False

    def cleanup_expired(self) -> int:
        """Remove all expired fallback solutions."""
        expired = [pid for pid, fb in self.fallback_solutions.items() if not fb.is_valid()]
        for pid in expired:
            del self.fallback_solutions[pid]
        if expired:
            logger.info("Cleaned up %d expired fallback solutions", len(expired))
        return len(expired)


# =============================================================================
# Performance Tracker
# =============================================================================

class PerformanceTracker:
    """
    Tracks and aggregates solver performance metrics.

    Maintains a sliding window of recent solve attempts and
    calculates aggregated metrics for monitoring and alerting.

    Attributes:
        config: Solver configuration
        solve_history: Recent solve attempts
        alerts: Generated alerts
    """

    def __init__(self, config: SolverConfig):
        """Initialize performance tracker."""
        self.config = config
        self.solve_history: List[SolveAttempt] = []
        self.alerts: List[SolverAlert] = []
        self._alert_counter = 0
        logger.info("PerformanceTracker initialized with window size: %d",
                   config.metrics_window_size)

    def record_solve(self, attempt: SolveAttempt) -> None:
        """Record a solve attempt and check for alerts."""
        self.solve_history.append(attempt)

        # Trim to window size
        if len(self.solve_history) > self.config.metrics_window_size:
            self.solve_history = self.solve_history[-self.config.metrics_window_size:]

        # Check for performance alerts
        self._check_alerts(attempt)

        logger.debug("Recorded solve attempt: %s (status=%s, time=%.3fs)",
                    attempt.solve_id, attempt.status.value, attempt.solve_time_seconds)

    def _check_alerts(self, attempt: SolveAttempt) -> None:
        """Check for alert conditions based on solve attempt."""
        # Check for slow solve
        if attempt.solve_time_seconds > self.config.alert_threshold_seconds:
            self._create_alert(
                level=AlertLevel.WARNING,
                message=f"Slow solve detected: {attempt.solve_time_seconds:.2f}s > {self.config.alert_threshold_seconds}s",
                solve_id=attempt.solve_id,
                metric_name="solve_time",
                metric_value=attempt.solve_time_seconds,
                threshold_value=self.config.alert_threshold_seconds
            )

        # Check for timeout
        if attempt.status == SolverStatus.TIMEOUT:
            self._create_alert(
                level=AlertLevel.ERROR,
                message=f"Solver timeout for problem '{attempt.problem_id}'",
                solve_id=attempt.solve_id,
                metric_name="timeout",
                metric_value=attempt.solve_time_seconds,
                threshold_value=self.config.default_timeout_seconds
            )

        # Check for infeasibility
        if attempt.status == SolverStatus.INFEASIBLE:
            self._create_alert(
                level=AlertLevel.ERROR,
                message=f"Infeasible problem detected: '{attempt.problem_id}'",
                solve_id=attempt.solve_id,
                metric_name="feasibility"
            )

        # Check for errors
        if attempt.status == SolverStatus.ERROR:
            self._create_alert(
                level=AlertLevel.CRITICAL,
                message=f"Solver error: {attempt.error_message}",
                solve_id=attempt.solve_id,
                metric_name="error"
            )

    def _create_alert(
        self,
        level: AlertLevel,
        message: str,
        solve_id: Optional[str] = None,
        metric_name: Optional[str] = None,
        metric_value: Optional[float] = None,
        threshold_value: Optional[float] = None
    ) -> SolverAlert:
        """Create and store a new alert."""
        self._alert_counter += 1
        alert_id = f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter:04d}"

        alert = SolverAlert(
            alert_id=alert_id,
            level=level,
            message=message,
            solve_id=solve_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold_value=threshold_value
        )
        self.alerts.append(alert)

        # Log based on level
        if level == AlertLevel.CRITICAL:
            logger.critical("Solver Alert [%s]: %s", alert_id, message)
        elif level == AlertLevel.ERROR:
            logger.error("Solver Alert [%s]: %s", alert_id, message)
        elif level == AlertLevel.WARNING:
            logger.warning("Solver Alert [%s]: %s", alert_id, message)
        else:
            logger.info("Solver Alert [%s]: %s", alert_id, message)

        return alert

    def get_metrics(self) -> PerformanceMetrics:
        """Calculate aggregated performance metrics."""
        if not self.solve_history:
            return PerformanceMetrics(
                provenance_hash=hashlib.sha256(b"empty").hexdigest()[:16]
            )

        solve_times = [s.solve_time_seconds for s in self.solve_history]
        iterations = [s.iterations for s in self.solve_history if s.iterations is not None]
        gaps = [s.optimality_gap for s in self.solve_history if s.optimality_gap is not None]

        successful = [s for s in self.solve_history
                     if s.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)]
        failed = [s for s in self.solve_history
                 if s.status in (SolverStatus.ERROR, SolverStatus.INFEASIBLE, SolverStatus.UNBOUNDED)]
        timeouts = [s for s in self.solve_history if s.status == SolverStatus.TIMEOUT]
        fallbacks = [s for s in self.solve_history if s.used_fallback]

        total = len(self.solve_history)
        success_rate = len(successful) / total if total > 0 else 0.0

        # Calculate percentiles
        sorted_times = sorted(solve_times)
        p95_idx = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_idx] if sorted_times else 0.0

        # Create metrics
        metrics_data = {
            "total_solves": total,
            "successful_solves": len(successful),
            "failed_solves": len(failed),
            "timeout_count": len(timeouts),
            "fallback_count": len(fallbacks),
            "avg_solve_time_seconds": statistics.mean(solve_times) if solve_times else 0.0,
            "median_solve_time_seconds": statistics.median(solve_times) if solve_times else 0.0,
            "p95_solve_time_seconds": p95_time,
            "max_solve_time_seconds": max(solve_times) if solve_times else 0.0,
            "avg_iterations": statistics.mean(iterations) if iterations else 0.0,
            "avg_optimality_gap": statistics.mean(gaps) if gaps else 0.0,
            "success_rate": success_rate,
            "solver_availability": 1.0 - (len(timeouts) + len(failed)) / total if total > 0 else 1.0,
            "window_start": self.solve_history[0].timestamp if self.solve_history else None
        }

        # Compute provenance
        provenance_str = f"{metrics_data}{datetime.now().isoformat()}"
        metrics_data["provenance_hash"] = hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

        return PerformanceMetrics(**metrics_data)

    def get_unacknowledged_alerts(self) -> List[SolverAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by ID."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info("Alert acknowledged: %s", alert_id)
                return True
        return False

    def clear_old_alerts(self, max_age_hours: float = 24.0) -> int:
        """Clear alerts older than specified age."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        old_count = len(self.alerts)
        self.alerts = [a for a in self.alerts if a.timestamp > cutoff]
        cleared = old_count - len(self.alerts)
        if cleared > 0:
            logger.info("Cleared %d old alerts", cleared)
        return cleared


# =============================================================================
# Solver Monitor
# =============================================================================

class SolverMonitor:
    """
    Core monitoring class for optimization solver execution.

    Provides comprehensive monitoring capabilities including:
    - Solve time tracking with context manager
    - Timeout handling with fallback recovery
    - Performance metrics aggregation
    - Diagnostics logging
    - Alert generation

    Attributes:
        config: Solver configuration
        timeout_handler: Handles timeouts and fallbacks
        performance_tracker: Tracks and aggregates metrics
        current_diagnostics: Diagnostics for current solve

    Example:
        >>> config = SolverConfig(default_timeout_seconds=30.0)
        >>> monitor = SolverMonitor(config)
        >>> with monitor.track_solve("blowdown_opt", SolverType.CVXPY_ECOS) as tracker:
        ...     result = optimizer.optimize(problem)
        ...     tracker.record_success(objective_value=result.objective)
        >>> metrics = monitor.get_performance_metrics()
        >>> print(f"Average solve time: {metrics.avg_solve_time_seconds:.3f}s")
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize SolverMonitor.

        Args:
            config: Solver configuration (uses defaults if not provided)
        """
        self.config = config or SolverConfig()
        self.timeout_handler = TimeoutHandler(self.config)
        self.performance_tracker = PerformanceTracker(self.config)
        self.current_diagnostics: Optional[SolverDiagnostics] = None
        self._solve_counter = 0

        logger.info("SolverMonitor initialized with config: timeout=%.1fs, max_iter=%d",
                   self.config.default_timeout_seconds, self.config.max_iterations)

    def _generate_solve_id(self) -> str:
        """Generate unique solve identifier."""
        self._solve_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"SOLVE-{timestamp}-{self._solve_counter:06d}"

    @contextmanager
    def track_solve(
        self,
        problem_id: str,
        solver_type: SolverType,
        timeout_seconds: Optional[float] = None
    ):
        """
        Context manager for tracking a solve attempt.

        Args:
            problem_id: Identifier for the optimization problem
            solver_type: Type of solver being used
            timeout_seconds: Optional timeout override

        Yields:
            SolveTracker: Object for recording solve outcome

        Example:
            >>> with monitor.track_solve("problem_1", SolverType.CVXPY_ECOS) as tracker:
            ...     try:
            ...         result = solver.solve()
            ...         tracker.record_success(result.objective)
            ...     except Exception as e:
            ...         tracker.record_failure(str(e))
        """
        solve_id = self._generate_solve_id()
        tracker = SolveTracker(
            solve_id=solve_id,
            problem_id=problem_id,
            solver_type=solver_type,
            timeout_handler=self.timeout_handler,
            timeout_seconds=timeout_seconds or self.config.default_timeout_seconds
        )

        logger.info("Starting solve tracking: %s (problem=%s, solver=%s)",
                   solve_id, problem_id, solver_type.value)

        try:
            self.timeout_handler.start_timer(timeout_seconds)
            yield tracker
        finally:
            elapsed = self.timeout_handler.stop_timer()
            attempt = tracker.finalize(elapsed)
            self.performance_tracker.record_solve(attempt)

            if self.config.diagnostics_enabled and tracker.diagnostics:
                self.current_diagnostics = tracker.diagnostics
                self._log_diagnostics(tracker.diagnostics)

    def _log_diagnostics(self, diagnostics: SolverDiagnostics) -> None:
        """Log solver diagnostics."""
        logger.debug("Solver Diagnostics for %s:", diagnostics.solve_id)
        logger.debug("  Problem size: %s", diagnostics.problem_size)
        logger.debug("  Timing breakdown: %s", diagnostics.timing_breakdown)
        if diagnostics.numerical_issues:
            logger.warning("  Numerical issues: %s", diagnostics.numerical_issues)
        if diagnostics.binding_constraints:
            logger.debug("  Binding constraints: %s", diagnostics.binding_constraints)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get aggregated performance metrics."""
        return self.performance_tracker.get_metrics()

    def get_alerts(self, include_acknowledged: bool = False) -> List[SolverAlert]:
        """Get solver alerts."""
        if include_acknowledged:
            return self.performance_tracker.alerts.copy()
        return self.performance_tracker.get_unacknowledged_alerts()

    def get_fallback_solution(self, problem_id: str) -> Optional[FallbackSolution]:
        """Get fallback solution for a problem if available."""
        return self.timeout_handler.get_fallback(problem_id)

    def store_fallback_solution(
        self,
        problem_id: str,
        solution: Dict[str, Any],
        objective_value: float,
        solver_type: SolverType,
        validity_hours: float = 24.0
    ) -> None:
        """Store a solution as fallback for timeout recovery."""
        self.timeout_handler.store_fallback(
            problem_id, solution, objective_value, solver_type, validity_hours
        )

    def check_timeout(self) -> bool:
        """Check if current solve has exceeded timeout."""
        return self.timeout_handler.check_timeout()

    def get_solve_history(self, limit: int = 100) -> List[SolveAttempt]:
        """Get recent solve history."""
        return self.performance_tracker.solve_history[-limit:]

    def get_solver_health(self) -> Dict[str, Any]:
        """
        Get overall solver health status.

        Returns:
            Dictionary with health indicators
        """
        metrics = self.get_performance_metrics()
        unacked_alerts = self.get_alerts(include_acknowledged=False)
        critical_alerts = [a for a in unacked_alerts if a.level == AlertLevel.CRITICAL]
        error_alerts = [a for a in unacked_alerts if a.level == AlertLevel.ERROR]

        # Determine health status
        if critical_alerts:
            status = "critical"
        elif error_alerts:
            status = "degraded"
        elif metrics.success_rate < 0.9:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "success_rate": metrics.success_rate,
            "avg_solve_time_seconds": metrics.avg_solve_time_seconds,
            "timeout_rate": metrics.timeout_count / metrics.total_solves if metrics.total_solves > 0 else 0.0,
            "active_alerts": len(unacked_alerts),
            "critical_alerts": len(critical_alerts),
            "fallback_solutions_available": len(self.timeout_handler.fallback_solutions),
            "last_updated": datetime.now().isoformat()
        }

    def cleanup(self) -> Dict[str, int]:
        """Perform cleanup of expired data."""
        expired_fallbacks = self.timeout_handler.cleanup_expired()
        cleared_alerts = self.performance_tracker.clear_old_alerts()

        return {
            "expired_fallbacks_cleared": expired_fallbacks,
            "old_alerts_cleared": cleared_alerts
        }


# =============================================================================
# Solve Tracker (Context Manager Helper)
# =============================================================================

class SolveTracker:
    """
    Helper class for tracking individual solve attempts.

    Used within the track_solve context manager to record
    solve outcomes and diagnostics.
    """

    def __init__(
        self,
        solve_id: str,
        problem_id: str,
        solver_type: SolverType,
        timeout_handler: TimeoutHandler,
        timeout_seconds: float
    ):
        """Initialize solve tracker."""
        self.solve_id = solve_id
        self.problem_id = problem_id
        self.solver_type = solver_type
        self.timeout_handler = timeout_handler
        self.timeout_seconds = timeout_seconds

        self.status = SolverStatus.RUNNING
        self.objective_value: Optional[float] = None
        self.iterations: Optional[int] = None
        self.optimality_gap: Optional[float] = None
        self.constraint_violations = 0
        self.error_message: Optional[str] = None
        self.used_fallback = False
        self.diagnostics: Optional[SolverDiagnostics] = None

    def record_success(
        self,
        objective_value: float,
        iterations: Optional[int] = None,
        optimality_gap: Optional[float] = None,
        is_optimal: bool = True
    ) -> None:
        """Record successful solve."""
        self.status = SolverStatus.OPTIMAL if is_optimal else SolverStatus.FEASIBLE
        self.objective_value = objective_value
        self.iterations = iterations
        self.optimality_gap = optimality_gap

        # Store as fallback for future timeout recovery
        if self.timeout_handler.config.fallback_enabled:
            self.timeout_handler.store_fallback(
                self.problem_id,
                {"objective": objective_value},  # Simplified; real impl stores full solution
                objective_value,
                self.solver_type
            )

    def record_failure(
        self,
        error_message: str,
        status: SolverStatus = SolverStatus.ERROR
    ) -> None:
        """Record failed solve."""
        self.status = status
        self.error_message = error_message

    def record_infeasible(self, constraint_violations: int = 0) -> None:
        """Record infeasible problem."""
        self.status = SolverStatus.INFEASIBLE
        self.constraint_violations = constraint_violations

    def record_timeout(self) -> Optional[FallbackSolution]:
        """Record timeout and attempt fallback recovery."""
        self.status = SolverStatus.TIMEOUT

        # Try to get fallback solution
        fallback = self.timeout_handler.get_fallback(self.problem_id)
        if fallback:
            self.used_fallback = True
            self.status = SolverStatus.FALLBACK
            self.objective_value = fallback.objective_value
            logger.info("Using fallback solution for '%s' after timeout", self.problem_id)
            return fallback

        logger.warning("No fallback solution available for '%s' after timeout", self.problem_id)
        return None

    def set_diagnostics(
        self,
        problem_size: Optional[Dict[str, int]] = None,
        solver_info: Optional[Dict[str, Any]] = None,
        timing_breakdown: Optional[Dict[str, float]] = None,
        binding_constraints: Optional[List[str]] = None,
        numerical_issues: Optional[List[str]] = None
    ) -> None:
        """Set solve diagnostics."""
        self.diagnostics = SolverDiagnostics(
            solve_id=self.solve_id,
            problem_size=problem_size or {},
            solver_info=solver_info or {},
            timing_breakdown=timing_breakdown or {},
            binding_constraints=binding_constraints or [],
            numerical_issues=numerical_issues or []
        )

    def check_timeout(self) -> bool:
        """Check if solve has exceeded timeout."""
        return self.timeout_handler.check_timeout(self.timeout_seconds)

    def finalize(self, elapsed_seconds: float) -> SolveAttempt:
        """Finalize and create solve attempt record."""
        return SolveAttempt(
            solve_id=self.solve_id,
            problem_id=self.problem_id,
            solver_type=self.solver_type,
            status=self.status,
            solve_time_seconds=elapsed_seconds,
            iterations=self.iterations,
            optimality_gap=self.optimality_gap,
            objective_value=self.objective_value,
            constraint_violations=self.constraint_violations,
            error_message=self.error_message,
            used_fallback=self.used_fallback
        )


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_monitor() -> SolverMonitor:
    """Create a SolverMonitor with default configuration."""
    return SolverMonitor(SolverConfig())


def create_production_monitor() -> SolverMonitor:
    """Create a SolverMonitor optimized for production use."""
    config = SolverConfig(
        default_timeout_seconds=60.0,
        max_iterations=50000,
        fallback_enabled=True,
        diagnostics_enabled=True,
        metrics_window_size=1000,
        alert_threshold_seconds=15.0,
        gap_tolerance=0.005
    )
    return SolverMonitor(config)


def create_fast_monitor() -> SolverMonitor:
    """Create a SolverMonitor for fast, iterative optimization."""
    config = SolverConfig(
        default_timeout_seconds=10.0,
        max_iterations=5000,
        fallback_enabled=True,
        diagnostics_enabled=False,
        metrics_window_size=50,
        alert_threshold_seconds=5.0,
        gap_tolerance=0.02
    )
    return SolverMonitor(config)

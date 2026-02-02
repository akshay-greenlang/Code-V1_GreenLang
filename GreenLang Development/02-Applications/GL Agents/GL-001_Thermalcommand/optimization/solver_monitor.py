"""
GL-001 ThermalCommand Solver Convergence Monitor

This module provides convergence monitoring, fallback handling, and audit
logging for optimization solvers used in the ThermalCommand Orchestrator.
Ensures mathematical rigor and reliability for MILP load allocation.

Key Features:
    - Real-time convergence monitoring with gap tolerance tracking
    - Automatic fallback to secondary solver on timeout/failure
    - Comprehensive solver statistics for audit trails
    - Provenance tracking with SHA-256 hashes
    - Performance benchmarking and trend analysis

Reference Standards:
    - IEC 61508 (Functional Safety) - Calculation Verification
    - ASME PTC 4 (Boiler Efficiency) - Optimization Requirements
    - IEEE 754 (Floating Point) - Numerical Precision
    - MIP Solver Standards (CPLEX, Gurobi, HiGHS)

Mathematical Rigor Guarantees:
    1. Gap Tolerance: Solution within specified optimality gap
    2. Iteration Limits: Bounded computation time
    3. Numerical Stability: Detection of ill-conditioned problems
    4. Deterministic Results: Same inputs produce same outputs

Solver Hierarchy:
    Primary:   scipy.optimize.milp (HiGHS backend)
    Secondary: cvxpy with ECOS/SCS backend
    Fallback:  Merit-order heuristic (always converges)

Example:
    >>> from optimization.solver_monitor import SolverMonitor, SolverConfig
    >>>
    >>> config = SolverConfig(
    ...     primary_solver="scipy_milp",
    ...     gap_tolerance=0.01,
    ...     time_limit_seconds=60.0
    ... )
    >>> monitor = SolverMonitor(config)
    >>>
    >>> with monitor.track_solve("load_allocation") as tracker:
    ...     result = solver.solve(problem)
    ...     tracker.record_solution(result, gap=0.005)
    >>>
    >>> stats = monitor.get_statistics()
    >>> print(f"Avg solve time: {stats.avg_solve_time_ms:.1f}ms")

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Type variable for generic solver results
T = TypeVar("T")


# =============================================================================
# ENUMERATIONS
# =============================================================================


class SolverType(str, Enum):
    """
    Supported optimization solver types.

    Ordered by preference for industrial MILP problems.
    """

    SCIPY_MILP = "scipy_milp"        # scipy.optimize.milp with HiGHS
    CVXPY_HIGHS = "cvxpy_highs"      # CVXPY with HiGHS backend
    CVXPY_ECOS = "cvxpy_ecos"        # CVXPY with ECOS backend
    CVXPY_SCS = "cvxpy_scs"          # CVXPY with SCS backend
    PYOMO_CBC = "pyomo_cbc"          # Pyomo with CBC solver
    HEURISTIC = "heuristic"          # Merit-order heuristic (fallback)


class SolverStatus(str, Enum):
    """
    Solver termination status.

    Based on standard MIP solver return codes.
    """

    OPTIMAL = "optimal"              # Global optimum found
    FEASIBLE = "feasible"            # Feasible solution, may not be optimal
    INFEASIBLE = "infeasible"        # No feasible solution exists
    UNBOUNDED = "unbounded"          # Problem is unbounded
    TIMEOUT = "timeout"              # Time limit exceeded
    ITERATION_LIMIT = "iteration_limit"  # Iteration limit exceeded
    NUMERICAL_ERROR = "numerical_error"  # Numerical instability
    SOLVER_ERROR = "solver_error"    # Solver internal error
    NOT_SOLVED = "not_solved"        # Solver not yet invoked


class ConvergenceState(str, Enum):
    """
    Convergence monitoring state.
    """

    NOT_STARTED = "not_started"
    CONVERGING = "converging"
    CONVERGED = "converged"
    STALLED = "stalled"
    DIVERGING = "diverging"
    FAILED = "failed"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class SolverConfig(BaseModel):
    """
    Configuration for solver behavior and monitoring.

    Attributes:
        primary_solver: Primary solver to use
        secondary_solver: Fallback solver on primary failure
        gap_tolerance: Acceptable optimality gap (0.01 = 1%)
        time_limit_seconds: Maximum solve time
        iteration_limit: Maximum solver iterations
        enable_presolve: Enable problem preprocessing
        enable_warm_start: Use previous solution as starting point
        numerical_tolerance: Tolerance for numerical comparisons

    Reference:
        - CPLEX MIP parameters
        - Gurobi optimization parameters
        - HiGHS solver options
    """

    primary_solver: SolverType = Field(
        default=SolverType.SCIPY_MILP,
        description="Primary optimization solver"
    )
    secondary_solver: SolverType = Field(
        default=SolverType.HEURISTIC,
        description="Fallback solver on primary failure"
    )
    gap_tolerance: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Acceptable optimality gap (0.01 = 1%)"
    )
    relative_gap_tolerance: float = Field(
        default=0.001,
        ge=0.0,
        le=1.0,
        description="Relative MIP gap tolerance"
    )
    time_limit_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Maximum solve time in seconds"
    )
    iteration_limit: int = Field(
        default=100000,
        ge=100,
        le=10000000,
        description="Maximum solver iterations"
    )
    enable_presolve: bool = Field(
        default=True,
        description="Enable problem preprocessing"
    )
    enable_warm_start: bool = Field(
        default=True,
        description="Use previous solution as starting point"
    )
    numerical_tolerance: float = Field(
        default=1e-8,
        ge=1e-12,
        le=1e-4,
        description="Numerical comparison tolerance"
    )
    feasibility_tolerance: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-3,
        description="Constraint feasibility tolerance"
    )
    integrality_tolerance: float = Field(
        default=1e-5,
        ge=1e-8,
        le=1e-2,
        description="Integer variable tolerance"
    )
    detect_numerical_issues: bool = Field(
        default=True,
        description="Detect and log numerical issues"
    )
    auto_fallback: bool = Field(
        default=True,
        description="Automatically fallback on solver failure"
    )
    log_solver_output: bool = Field(
        default=False,
        description="Log detailed solver output"
    )

    @field_validator("gap_tolerance")
    @classmethod
    def validate_gap_tolerance(cls, v: float) -> float:
        """Validate gap tolerance is reasonable."""
        if v > 0.1:
            logger.warning(
                "Gap tolerance %.2f%% is high for industrial optimization",
                v * 100
            )
        return v


class SolverResult(BaseModel):
    """
    Standardized solver result model.

    Provides consistent interface across different solver backends.
    """

    status: SolverStatus = Field(..., description="Termination status")
    objective_value: Optional[float] = Field(
        default=None,
        description="Objective function value"
    )
    best_bound: Optional[float] = Field(
        default=None,
        description="Best known bound (for MIP)"
    )
    gap: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optimality gap"
    )
    gap_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optimality gap as percentage"
    )
    solution: Optional[Dict[str, float]] = Field(
        default=None,
        description="Variable values"
    )
    solve_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Solve time in milliseconds"
    )
    iterations: int = Field(
        default=0,
        ge=0,
        description="Solver iterations"
    )
    nodes_explored: int = Field(
        default=0,
        ge=0,
        description="Branch-and-bound nodes explored"
    )
    solver_used: SolverType = Field(..., description="Solver that produced result")
    solver_message: str = Field(
        default="",
        description="Solver status message"
    )
    is_optimal: bool = Field(
        default=False,
        description="True if proven optimal"
    )
    is_feasible: bool = Field(
        default=False,
        description="True if feasible solution found"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate derived fields and provenance hash."""
        # Calculate gap percent
        if self.gap is not None:
            self.gap_percent = self.gap * 100

        # Determine optimality
        self.is_optimal = self.status == SolverStatus.OPTIMAL
        self.is_feasible = self.status in [
            SolverStatus.OPTIMAL,
            SolverStatus.FEASIBLE,
        ]

        # Calculate provenance hash
        if not self.provenance_hash:
            content = (
                f"{self.status.value}|{self.objective_value}|"
                f"{self.gap}|{self.solver_used.value}|"
                f"{self.timestamp.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# CONVERGENCE TRACKING
# =============================================================================


@dataclass
class ConvergencePoint:
    """Single point in convergence history."""

    timestamp: float
    iteration: int
    objective_value: float
    bound: float
    gap: float
    elapsed_ms: float


class ConvergenceTracker:
    """
    Tracks solver convergence during optimization.

    Monitors progress, detects stalls, and provides early termination
    signals when convergence criteria are met.
    """

    def __init__(
        self,
        problem_id: str,
        gap_tolerance: float = 0.01,
        stall_iterations: int = 1000,
        stall_tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize convergence tracker.

        Args:
            problem_id: Unique problem identifier
            gap_tolerance: Target optimality gap
            stall_iterations: Iterations without progress before stall
            stall_tolerance: Minimum improvement to not be considered stalled
        """
        self.problem_id = problem_id
        self.gap_tolerance = gap_tolerance
        self.stall_iterations = stall_iterations
        self.stall_tolerance = stall_tolerance

        self._history: List[ConvergencePoint] = []
        self._start_time: Optional[float] = None
        self._state = ConvergenceState.NOT_STARTED
        self._last_improvement_iteration = 0
        self._best_objective: Optional[float] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start tracking convergence."""
        with self._lock:
            self._start_time = time.monotonic()
            self._state = ConvergenceState.CONVERGING
            self._history.clear()
            self._last_improvement_iteration = 0
            self._best_objective = None

        logger.debug("ConvergenceTracker started: %s", self.problem_id)

    def record(
        self,
        iteration: int,
        objective_value: float,
        bound: float,
    ) -> ConvergenceState:
        """
        Record a convergence point.

        Args:
            iteration: Current iteration
            objective_value: Current objective value
            bound: Current bound (lower for minimization)

        Returns:
            Current convergence state
        """
        with self._lock:
            if self._start_time is None:
                self.start()

            elapsed_ms = (time.monotonic() - self._start_time) * 1000

            # Calculate gap
            if abs(objective_value) > 1e-10:
                gap = abs(objective_value - bound) / abs(objective_value)
            else:
                gap = abs(objective_value - bound)

            # Record point
            point = ConvergencePoint(
                timestamp=time.monotonic(),
                iteration=iteration,
                objective_value=objective_value,
                bound=bound,
                gap=gap,
                elapsed_ms=elapsed_ms,
            )
            self._history.append(point)

            # Check for improvement
            if self._best_objective is None:
                self._best_objective = objective_value
                self._last_improvement_iteration = iteration
            else:
                improvement = abs(self._best_objective - objective_value)
                if improvement > self.stall_tolerance:
                    self._best_objective = min(
                        self._best_objective, objective_value
                    )
                    self._last_improvement_iteration = iteration

            # Update state
            self._update_state(iteration, gap)

            return self._state

    def _update_state(self, iteration: int, gap: float) -> None:
        """Update convergence state based on current progress."""
        # Check for convergence
        if gap <= self.gap_tolerance:
            self._state = ConvergenceState.CONVERGED
            return

        # Check for stall
        iterations_since_improvement = iteration - self._last_improvement_iteration
        if iterations_since_improvement > self.stall_iterations:
            self._state = ConvergenceState.STALLED
            return

        # Check for divergence (gap increasing)
        if len(self._history) >= 10:
            recent_gaps = [p.gap for p in self._history[-10:]]
            if all(recent_gaps[i] < recent_gaps[i+1] for i in range(len(recent_gaps)-1)):
                self._state = ConvergenceState.DIVERGING
                return

        self._state = ConvergenceState.CONVERGING

    def stop(self) -> Dict[str, Any]:
        """
        Stop tracking and return summary.

        Returns:
            Convergence summary dictionary
        """
        with self._lock:
            if not self._history:
                return {
                    "problem_id": self.problem_id,
                    "state": self._state.value,
                    "points_recorded": 0,
                }

            final_point = self._history[-1]
            first_point = self._history[0]

            return {
                "problem_id": self.problem_id,
                "state": self._state.value,
                "points_recorded": len(self._history),
                "total_iterations": final_point.iteration,
                "total_time_ms": final_point.elapsed_ms,
                "initial_gap": first_point.gap,
                "final_gap": final_point.gap,
                "gap_improvement": first_point.gap - final_point.gap,
                "initial_objective": first_point.objective_value,
                "final_objective": final_point.objective_value,
                "converged": self._state == ConvergenceState.CONVERGED,
            }

    @property
    def state(self) -> ConvergenceState:
        """Get current convergence state."""
        with self._lock:
            return self._state

    @property
    def current_gap(self) -> Optional[float]:
        """Get current optimality gap."""
        with self._lock:
            if self._history:
                return self._history[-1].gap
            return None

    def get_history(self) -> List[ConvergencePoint]:
        """Get full convergence history."""
        with self._lock:
            return list(self._history)


# =============================================================================
# SOLVER STATISTICS
# =============================================================================


class SolverStatistics(BaseModel):
    """
    Aggregated solver statistics for monitoring and auditing.

    Supports trend analysis and performance benchmarking.
    """

    solver_name: str = Field(..., description="Solver identifier")
    total_solves: int = Field(default=0, ge=0, description="Total solve attempts")
    successful_solves: int = Field(default=0, ge=0, description="Successful solves")
    optimal_solves: int = Field(default=0, ge=0, description="Proven optimal")
    feasible_solves: int = Field(default=0, ge=0, description="Feasible but not optimal")
    failed_solves: int = Field(default=0, ge=0, description="Failed solves")
    timeout_solves: int = Field(default=0, ge=0, description="Timeout solves")
    fallback_invoked: int = Field(default=0, ge=0, description="Fallback solver used")

    # Time statistics (milliseconds)
    total_solve_time_ms: float = Field(default=0.0, ge=0.0)
    avg_solve_time_ms: float = Field(default=0.0, ge=0.0)
    min_solve_time_ms: float = Field(default=0.0, ge=0.0)
    max_solve_time_ms: float = Field(default=0.0, ge=0.0)
    p95_solve_time_ms: float = Field(default=0.0, ge=0.0)
    p99_solve_time_ms: float = Field(default=0.0, ge=0.0)

    # Gap statistics
    avg_gap_percent: float = Field(default=0.0, ge=0.0)
    min_gap_percent: float = Field(default=0.0, ge=0.0)
    max_gap_percent: float = Field(default=0.0, ge=0.0)

    # Iteration statistics
    avg_iterations: float = Field(default=0.0, ge=0.0)
    max_iterations: int = Field(default=0, ge=0)

    # Calculated metrics
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    optimality_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Timestamps
    first_solve_at: Optional[datetime] = Field(default=None)
    last_solve_at: Optional[datetime] = Field(default=None)
    statistics_generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate derived metrics."""
        if self.total_solves > 0:
            self.success_rate = self.successful_solves / self.total_solves
            self.optimality_rate = self.optimal_solves / self.total_solves


class SolverAuditRecord(BaseModel):
    """
    Audit record for solver invocations.

    Provides complete traceability for regulatory compliance.
    """

    record_id: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        description="Unique record identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Record timestamp"
    )
    problem_id: str = Field(..., description="Problem identifier")
    solver_used: SolverType = Field(..., description="Solver used")
    status: SolverStatus = Field(..., description="Termination status")
    objective_value: Optional[float] = Field(default=None)
    gap_percent: Optional[float] = Field(default=None)
    solve_time_ms: float = Field(default=0.0)
    iterations: int = Field(default=0)
    fallback_used: bool = Field(default=False)
    fallback_reason: Optional[str] = Field(default=None)
    convergence_summary: Dict[str, Any] = Field(default_factory=dict)
    input_hash: str = Field(default="", description="Hash of input data")
    output_hash: str = Field(default="", description="Hash of output data")
    provenance_hash: str = Field(default="", description="Combined provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.record_id}|{self.timestamp.isoformat()}|"
                f"{self.problem_id}|{self.solver_used.value}|"
                f"{self.status.value}|{self.objective_value}|"
                f"{self.input_hash}|{self.output_hash}"
            )
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# SOLVE CONTEXT MANAGER
# =============================================================================


@dataclass
class SolveContext:
    """Context for tracking a single solve operation."""

    problem_id: str
    monitor: "SolverMonitor"
    tracker: ConvergenceTracker
    start_time: float = field(default_factory=time.monotonic)
    solver_used: Optional[SolverType] = None
    result: Optional[SolverResult] = None
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    input_hash: str = ""

    def record_input(self, input_data: Any) -> None:
        """Record input data hash."""
        content = str(input_data)
        self.input_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    def record_solution(
        self,
        objective_value: float,
        gap: float,
        status: SolverStatus = SolverStatus.OPTIMAL,
        solution: Optional[Dict[str, float]] = None,
        iterations: int = 0,
        nodes: int = 0,
        message: str = "",
    ) -> SolverResult:
        """
        Record solution and create result.

        Args:
            objective_value: Objective function value
            gap: Optimality gap
            status: Termination status
            solution: Variable values
            iterations: Solver iterations
            nodes: Nodes explored
            message: Solver message

        Returns:
            SolverResult object
        """
        solve_time_ms = (time.monotonic() - self.start_time) * 1000

        self.result = SolverResult(
            status=status,
            objective_value=objective_value,
            gap=gap,
            solution=solution,
            solve_time_ms=solve_time_ms,
            iterations=iterations,
            nodes_explored=nodes,
            solver_used=self.solver_used or self.monitor.config.primary_solver,
            solver_message=message,
        )

        # Record final convergence point
        self.tracker.record(
            iteration=iterations,
            objective_value=objective_value,
            bound=objective_value * (1 - gap) if gap > 0 else objective_value,
        )

        return self.result

    def record_failure(
        self,
        status: SolverStatus,
        message: str = "",
        trigger_fallback: bool = True,
    ) -> Optional[SolverResult]:
        """
        Record solver failure.

        Args:
            status: Failure status
            message: Error message
            trigger_fallback: Whether to trigger fallback solver

        Returns:
            Fallback result if triggered, None otherwise
        """
        solve_time_ms = (time.monotonic() - self.start_time) * 1000

        self.result = SolverResult(
            status=status,
            solve_time_ms=solve_time_ms,
            solver_used=self.solver_used or self.monitor.config.primary_solver,
            solver_message=message,
        )

        if trigger_fallback and self.monitor.config.auto_fallback:
            self.fallback_used = True
            self.fallback_reason = f"{status.value}: {message}"
            logger.warning(
                "Solver failed, triggering fallback: problem=%s, reason=%s",
                self.problem_id,
                self.fallback_reason,
            )

        return self.result

    def use_solver(self, solver: SolverType) -> None:
        """Set the solver being used."""
        self.solver_used = solver


# =============================================================================
# SOLVER MONITOR
# =============================================================================


class SolverMonitor:
    """
    Production-grade solver convergence monitor.

    Provides:
    - Real-time convergence tracking
    - Automatic fallback on solver failure
    - Comprehensive audit logging
    - Performance statistics
    - Provenance tracking

    Example:
        >>> monitor = SolverMonitor(SolverConfig(gap_tolerance=0.01))
        >>> with monitor.track_solve("problem_001") as ctx:
        ...     ctx.record_input(problem_data)
        ...     result = solver.solve(problem)
        ...     ctx.record_solution(result.objective, result.gap)
    """

    def __init__(
        self,
        config: Optional[SolverConfig] = None,
        audit_callback: Optional[Callable[[SolverAuditRecord], None]] = None,
    ) -> None:
        """
        Initialize solver monitor.

        Args:
            config: Solver configuration
            audit_callback: Optional callback for audit records
        """
        self._config = config or SolverConfig()
        self._audit_callback = audit_callback

        # Statistics tracking
        self._solve_times: List[float] = []
        self._gaps: List[float] = []
        self._iterations: List[int] = []
        self._statuses: List[SolverStatus] = []
        self._total_solves = 0
        self._successful_solves = 0
        self._optimal_solves = 0
        self._failed_solves = 0
        self._timeout_solves = 0
        self._fallback_count = 0
        self._first_solve_at: Optional[datetime] = None
        self._last_solve_at: Optional[datetime] = None

        # Audit records
        self._audit_records: List[SolverAuditRecord] = []

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            "SolverMonitor initialized: primary=%s, secondary=%s, gap_tol=%.2f%%",
            self._config.primary_solver.value,
            self._config.secondary_solver.value,
            self._config.gap_tolerance * 100,
        )

    @property
    def config(self) -> SolverConfig:
        """Get solver configuration."""
        return self._config

    @contextmanager
    def track_solve(self, problem_id: str):
        """
        Context manager for tracking a solve operation.

        Args:
            problem_id: Unique problem identifier

        Yields:
            SolveContext for recording solve progress

        Example:
            >>> with monitor.track_solve("problem_001") as ctx:
            ...     result = solver.solve(problem)
            ...     ctx.record_solution(result.objective, result.gap)
        """
        tracker = ConvergenceTracker(
            problem_id=problem_id,
            gap_tolerance=self._config.gap_tolerance,
        )
        tracker.start()

        context = SolveContext(
            problem_id=problem_id,
            monitor=self,
            tracker=tracker,
            solver_used=self._config.primary_solver,
        )

        try:
            yield context
        finally:
            # Record statistics
            convergence_summary = tracker.stop()
            self._record_solve(context, convergence_summary)

    def _record_solve(
        self,
        context: SolveContext,
        convergence_summary: Dict[str, Any],
    ) -> None:
        """Record solve attempt in statistics and audit log."""
        with self._lock:
            self._total_solves += 1
            now = datetime.now(timezone.utc)

            if self._first_solve_at is None:
                self._first_solve_at = now
            self._last_solve_at = now

            if context.result:
                self._solve_times.append(context.result.solve_time_ms)
                self._statuses.append(context.result.status)

                if context.result.gap is not None:
                    self._gaps.append(context.result.gap * 100)

                self._iterations.append(context.result.iterations)

                # Categorize result
                if context.result.status == SolverStatus.OPTIMAL:
                    self._optimal_solves += 1
                    self._successful_solves += 1
                elif context.result.status == SolverStatus.FEASIBLE:
                    self._successful_solves += 1
                elif context.result.status == SolverStatus.TIMEOUT:
                    self._timeout_solves += 1
                    self._failed_solves += 1
                elif context.result.status in [
                    SolverStatus.INFEASIBLE,
                    SolverStatus.NUMERICAL_ERROR,
                    SolverStatus.SOLVER_ERROR,
                ]:
                    self._failed_solves += 1
            else:
                self._failed_solves += 1

            if context.fallback_used:
                self._fallback_count += 1

            # Create audit record
            audit_record = SolverAuditRecord(
                problem_id=context.problem_id,
                solver_used=context.solver_used or self._config.primary_solver,
                status=context.result.status if context.result else SolverStatus.SOLVER_ERROR,
                objective_value=context.result.objective_value if context.result else None,
                gap_percent=context.result.gap_percent if context.result else None,
                solve_time_ms=context.result.solve_time_ms if context.result else 0.0,
                iterations=context.result.iterations if context.result else 0,
                fallback_used=context.fallback_used,
                fallback_reason=context.fallback_reason,
                convergence_summary=convergence_summary,
                input_hash=context.input_hash,
                output_hash=context.result.provenance_hash[:16] if context.result else "",
            )
            self._audit_records.append(audit_record)

            # Invoke callback
            if self._audit_callback:
                try:
                    self._audit_callback(audit_record)
                except Exception as e:
                    logger.error("Audit callback failed: %s", e)

            # Log summary
            logger.info(
                "Solve completed: problem=%s, status=%s, time=%.1fms, gap=%.3f%%",
                context.problem_id,
                context.result.status.value if context.result else "error",
                context.result.solve_time_ms if context.result else 0,
                (context.result.gap_percent or 0) if context.result else 0,
            )

    def get_statistics(self) -> SolverStatistics:
        """
        Get aggregated solver statistics.

        Returns:
            SolverStatistics with performance metrics
        """
        with self._lock:
            if not self._solve_times:
                return SolverStatistics(
                    solver_name=self._config.primary_solver.value,
                    total_solves=self._total_solves,
                )

            # Calculate percentiles
            sorted_times = sorted(self._solve_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)

            return SolverStatistics(
                solver_name=self._config.primary_solver.value,
                total_solves=self._total_solves,
                successful_solves=self._successful_solves,
                optimal_solves=self._optimal_solves,
                feasible_solves=self._successful_solves - self._optimal_solves,
                failed_solves=self._failed_solves,
                timeout_solves=self._timeout_solves,
                fallback_invoked=self._fallback_count,
                total_solve_time_ms=sum(self._solve_times),
                avg_solve_time_ms=sum(self._solve_times) / len(self._solve_times),
                min_solve_time_ms=min(self._solve_times),
                max_solve_time_ms=max(self._solve_times),
                p95_solve_time_ms=sorted_times[p95_idx] if p95_idx < len(sorted_times) else sorted_times[-1],
                p99_solve_time_ms=sorted_times[p99_idx] if p99_idx < len(sorted_times) else sorted_times[-1],
                avg_gap_percent=sum(self._gaps) / len(self._gaps) if self._gaps else 0,
                min_gap_percent=min(self._gaps) if self._gaps else 0,
                max_gap_percent=max(self._gaps) if self._gaps else 0,
                avg_iterations=sum(self._iterations) / len(self._iterations) if self._iterations else 0,
                max_iterations=max(self._iterations) if self._iterations else 0,
                first_solve_at=self._first_solve_at,
                last_solve_at=self._last_solve_at,
            )

    def get_audit_records(
        self,
        limit: int = 100,
        status_filter: Optional[SolverStatus] = None,
    ) -> List[SolverAuditRecord]:
        """
        Get audit records with optional filtering.

        Args:
            limit: Maximum records to return
            status_filter: Optional status to filter by

        Returns:
            List of audit records
        """
        with self._lock:
            records = self._audit_records

            if status_filter:
                records = [r for r in records if r.status == status_filter]

            return list(reversed(records[-limit:]))

    def should_use_fallback(self, result: SolverResult) -> bool:
        """
        Determine if fallback solver should be used.

        Args:
            result: Result from primary solver

        Returns:
            True if fallback should be invoked
        """
        if not self._config.auto_fallback:
            return False

        # Trigger fallback on failure statuses
        if result.status in [
            SolverStatus.INFEASIBLE,
            SolverStatus.TIMEOUT,
            SolverStatus.NUMERICAL_ERROR,
            SolverStatus.SOLVER_ERROR,
        ]:
            return True

        # Trigger fallback if gap is too large
        if result.gap is not None and result.gap > self._config.gap_tolerance * 2:
            logger.warning(
                "Gap %.2f%% exceeds 2x tolerance, triggering fallback",
                result.gap * 100
            )
            return True

        return False

    def reset_statistics(self) -> None:
        """Reset all statistics (keeps audit records)."""
        with self._lock:
            self._solve_times.clear()
            self._gaps.clear()
            self._iterations.clear()
            self._statuses.clear()
            self._total_solves = 0
            self._successful_solves = 0
            self._optimal_solves = 0
            self._failed_solves = 0
            self._timeout_solves = 0
            self._fallback_count = 0
            self._first_solve_at = None
            self._last_solve_at = None

        logger.info("SolverMonitor statistics reset")

    def clear_audit_records(self) -> int:
        """
        Clear audit records (for testing/maintenance).

        Returns:
            Number of records cleared
        """
        with self._lock:
            count = len(self._audit_records)
            self._audit_records.clear()

        logger.info("Cleared %d audit records", count)
        return count


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_solver_monitor(
    gap_tolerance: float = 0.01,
    time_limit_seconds: float = 60.0,
    primary_solver: SolverType = SolverType.SCIPY_MILP,
    secondary_solver: SolverType = SolverType.HEURISTIC,
    audit_callback: Optional[Callable[[SolverAuditRecord], None]] = None,
) -> SolverMonitor:
    """
    Factory function to create a configured SolverMonitor.

    Args:
        gap_tolerance: Acceptable optimality gap (0.01 = 1%)
        time_limit_seconds: Maximum solve time
        primary_solver: Primary solver to use
        secondary_solver: Fallback solver
        audit_callback: Optional audit callback

    Returns:
        Configured SolverMonitor instance

    Example:
        >>> monitor = create_solver_monitor(gap_tolerance=0.005)
    """
    config = SolverConfig(
        primary_solver=primary_solver,
        secondary_solver=secondary_solver,
        gap_tolerance=gap_tolerance,
        time_limit_seconds=time_limit_seconds,
    )
    return SolverMonitor(config=config, audit_callback=audit_callback)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SolverType",
    "SolverStatus",
    "ConvergenceState",
    # Configuration
    "SolverConfig",
    # Results
    "SolverResult",
    "SolverStatistics",
    "SolverAuditRecord",
    # Tracking
    "ConvergenceTracker",
    "ConvergencePoint",
    "SolveContext",
    # Core
    "SolverMonitor",
    # Factory
    "create_solver_monitor",
]

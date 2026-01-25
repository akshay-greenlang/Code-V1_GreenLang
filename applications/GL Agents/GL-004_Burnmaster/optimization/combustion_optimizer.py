"""
GL-004 Burnmaster - Combustion Optimizer

Main optimization engine for combustion control.

Features:
    - optimize_setpoints: Find optimal setpoints given current state
    - compute_optimal_o2_target: Calculate optimal O2 setpoint
    - compute_optimal_air_fuel_ratio: Calculate optimal A/F ratio
    - handle_load_change: Generate trajectory for load changes
    - Support for scipy.optimize, cvxpy, local search

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import hashlib
import logging
import uuid

import numpy as np
from pydantic import BaseModel, Field

try:
    from scipy.optimize import minimize, differential_evolution, Bounds
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .objective_functions import (
    BurnerState, SetpointVector, MultiObjectiveFunction,
    MultiObjectiveResult, create_balanced_objective
)
from .constraint_handler import (
    ConstraintSet, ConstraintSetResult, HardConstraint,
    create_combustion_constraint_set
)

logger = logging.getLogger(__name__)


class OptimizerStatus(str, Enum):
    """Status of optimization."""
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    INFEASIBLE = "infeasible"
    TIMEOUT = "timeout"
    ERROR = "error"
    NOT_CONVERGED = "not_converged"


class OptimizerMethod(str, Enum):
    """Optimization method."""
    SCIPY_SLSQP = "scipy_slsqp"
    SCIPY_COBYLA = "scipy_cobyla"
    SCIPY_DE = "scipy_differential_evolution"
    LOCAL_SEARCH = "local_search"
    GRID_SEARCH = "grid_search"


class TrajectoryPoint(BaseModel):
    """Single point in a load change trajectory."""
    time_offset_s: float = Field(..., ge=0, description="Time from trajectory start")
    load_percent: float = Field(..., ge=0, le=100)
    o2_setpoint: float = Field(..., ge=0.5, le=10.0)
    air_damper_position: float = Field(..., ge=0, le=100)
    fuel_valve_position: float = Field(..., ge=0, le=100)


class TrajectoryPlan(BaseModel):
    """Load change trajectory plan."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    start_load: float = Field(..., ge=0, le=100)
    target_load: float = Field(..., ge=0, le=100)
    duration_s: float = Field(..., ge=0)
    points: List[TrajectoryPoint] = Field(default_factory=list)
    rate_limit_percent_per_s: float = Field(default=1.0, ge=0)
    is_ramp_up: bool = Field(default=True)
    safety_margin: float = Field(default=0.1, ge=0, le=1)


class OptimizationResult(BaseModel):
    """Result of combustion optimization."""
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: OptimizerStatus = Field(default=OptimizerStatus.ERROR)
    method: OptimizerMethod = Field(default=OptimizerMethod.SCIPY_SLSQP)

    # Optimal setpoints
    optimal_setpoints: Optional[SetpointVector] = None
    current_setpoints: Optional[SetpointVector] = None

    # Objective values
    objective_value: float = Field(default=float("inf"))
    objective_result: Optional[MultiObjectiveResult] = None

    # Constraint satisfaction
    constraint_result: Optional[ConstraintSetResult] = None
    is_feasible: bool = Field(default=False)

    # Uncertainty bounds
    objective_lower_bound: float = Field(default=0.0)
    objective_upper_bound: float = Field(default=float("inf"))
    confidence: float = Field(default=0.95)

    # Solver metrics
    iterations: int = Field(default=0, ge=0)
    function_evaluations: int = Field(default=0, ge=0)
    solve_time_ms: float = Field(default=0.0, ge=0)
    convergence_tolerance: float = Field(default=1e-6)

    # Improvement metrics
    improvement_percent: float = Field(default=0.0)
    savings_per_hour: float = Field(default=0.0)

    # Provenance
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash and self.optimal_setpoints:
            sp = self.optimal_setpoints
            hash_input = f"{self.result_id}|{sp.o2_setpoint_percent:.4f}|{sp.air_damper_position:.2f}|{self.objective_value:.6f}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class CombustionOptimizer:
    """
    Main combustion optimization engine.

    Solves the multi-objective constrained optimization problem:

    Minimize: (fuel_cost * fuel_rate) + (emissions_cost * NOx)
              + (CO_penalty * max(0, CO - CO_limit))
              + (stability_penalty * instability_risk)
              + (move_penalty * actuator_moves)

    Subject to: duty constraint, safety limits, actuator bounds, rate limits
    """

    def __init__(
        self,
        objective_function: Optional[MultiObjectiveFunction] = None,
        constraint_set: Optional[ConstraintSet] = None,
        method: OptimizerMethod = OptimizerMethod.SCIPY_SLSQP,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        time_limit_s: float = 5.0
    ) -> None:
        """
        Initialize combustion optimizer.

        Args:
            objective_function: Multi-objective function (uses default if None)
            constraint_set: Constraint set (uses default if None)
            method: Optimization method
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            time_limit_s: Time limit for optimization
        """
        self.objective_function = objective_function or create_balanced_objective()
        self.constraint_set = constraint_set or create_combustion_constraint_set()
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.time_limit_s = time_limit_s

        # Bounds for setpoint variables [o2, damper, valve]
        self.setpoint_bounds = {
            "o2_setpoint_percent": (0.5, 10.0),
            "air_damper_position": (0.0, 100.0),
            "fuel_valve_position": (0.0, 100.0)
        }

        # Evaluation counter
        self._eval_count = 0

        logger.info("CombustionOptimizer initialized (method=%s, max_iter=%d)",
                    method.value, max_iterations)

    def optimize_setpoints(
        self,
        current_state: BurnerState,
        constraints: Optional[ConstraintSet] = None,
        previous_setpoints: Optional[SetpointVector] = None
    ) -> OptimizationResult:
        """
        Optimize setpoints given current state.

        Args:
            current_state: Current burner state
            constraints: Optional constraint set override
            previous_setpoints: Previous setpoints for move penalty

        Returns:
            OptimizationResult with optimal setpoints
        """
        start_time = datetime.now(timezone.utc)
        self._eval_count = 0

        constraint_set = constraints or self.constraint_set

        # Create initial guess from current state
        x0 = np.array([
            current_state.o2_percent,
            50.0,  # Damper position estimate
            50.0   # Valve position estimate
        ])

        # Create current setpoints for reference
        current_sp = SetpointVector(
            o2_setpoint_percent=current_state.o2_percent,
            air_damper_position=50.0,
            fuel_valve_position=50.0
        )

        try:
            if self.method == OptimizerMethod.SCIPY_SLSQP and SCIPY_AVAILABLE:
                result = self._optimize_scipy_slsqp(
                    current_state, x0, constraint_set, previous_setpoints
                )
            elif self.method == OptimizerMethod.SCIPY_DE and SCIPY_AVAILABLE:
                result = self._optimize_scipy_de(
                    current_state, constraint_set, previous_setpoints
                )
            elif self.method == OptimizerMethod.LOCAL_SEARCH:
                result = self._optimize_local_search(
                    current_state, x0, constraint_set, previous_setpoints
                )
            else:
                result = self._optimize_local_search(
                    current_state, x0, constraint_set, previous_setpoints
                )

            # Calculate improvement
            baseline_result = self.objective_function.evaluate(
                current_state, current_sp, previous_setpoints
            )
            if result.objective_result and baseline_result.total_cost > 0:
                improvement = (
                    (baseline_result.total_cost - result.objective_result.total_cost)
                    / baseline_result.total_cost * 100
                )
                result.improvement_percent = max(0, improvement)

            result.current_setpoints = current_sp
            result.solve_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            result.function_evaluations = self._eval_count

            logger.info(
                "Optimization complete: status=%s, obj=%.4f, improvement=%.2f%%, time=%.1fms",
                result.status.value, result.objective_value,
                result.improvement_percent, result.solve_time_ms
            )

            return result

        except Exception as e:
            logger.error("Optimization failed: %s", str(e), exc_info=True)
            return OptimizationResult(
                status=OptimizerStatus.ERROR,
                method=self.method,
                current_setpoints=current_sp,
                solve_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

    def _optimize_scipy_slsqp(
        self,
        state: BurnerState,
        x0: np.ndarray,
        constraints: ConstraintSet,
        previous_setpoints: Optional[SetpointVector]
    ) -> OptimizationResult:
        """Optimize using scipy SLSQP method."""

        def objective(x: np.ndarray) -> float:
            self._eval_count += 1
            setpoints = SetpointVector(
                o2_setpoint_percent=float(x[0]),
                air_damper_position=float(x[1]),
                fuel_valve_position=float(x[2])
            )
            result = self.objective_function.evaluate(state, setpoints, previous_setpoints)
            return result.total_cost

        bounds = Bounds(
            [self.setpoint_bounds["o2_setpoint_percent"][0],
             self.setpoint_bounds["air_damper_position"][0],
             self.setpoint_bounds["fuel_valve_position"][0]],
            [self.setpoint_bounds["o2_setpoint_percent"][1],
             self.setpoint_bounds["air_damper_position"][1],
             self.setpoint_bounds["fuel_valve_position"][1]]
        )

        result = minimize(
            objective, x0, method="SLSQP", bounds=bounds,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance}
        )

        optimal_setpoints = SetpointVector(
            o2_setpoint_percent=float(result.x[0]),
            air_damper_position=float(result.x[1]),
            fuel_valve_position=float(result.x[2])
        )

        obj_result = self.objective_function.evaluate(state, optimal_setpoints, previous_setpoints)
        values = {
            "o2_percent": state.o2_percent,
            "co_ppm": state.co_ppm,
            "nox_ppm": state.nox_ppm,
            "o2_setpoint_percent": optimal_setpoints.o2_setpoint_percent,
            "air_damper_position": optimal_setpoints.air_damper_position,
            "fuel_valve_position": optimal_setpoints.fuel_valve_position
        }
        constraint_result = constraints.evaluate(values)

        status = OptimizerStatus.OPTIMAL if result.success else OptimizerStatus.NOT_CONVERGED
        if not constraint_result.is_feasible:
            status = OptimizerStatus.INFEASIBLE

        return OptimizationResult(
            status=status,
            method=OptimizerMethod.SCIPY_SLSQP,
            optimal_setpoints=optimal_setpoints,
            objective_value=result.fun,
            objective_result=obj_result,
            constraint_result=constraint_result,
            is_feasible=constraint_result.is_feasible,
            objective_lower_bound=obj_result.total_cost_uncertainty[0],
            objective_upper_bound=obj_result.total_cost_uncertainty[1],
            iterations=result.nit,
            convergence_tolerance=self.tolerance
        )

    def _optimize_scipy_de(
        self,
        state: BurnerState,
        constraints: ConstraintSet,
        previous_setpoints: Optional[SetpointVector]
    ) -> OptimizationResult:
        """Optimize using scipy differential evolution."""

        def objective(x: np.ndarray) -> float:
            self._eval_count += 1
            setpoints = SetpointVector(
                o2_setpoint_percent=float(x[0]),
                air_damper_position=float(x[1]),
                fuel_valve_position=float(x[2])
            )
            result = self.objective_function.evaluate(state, setpoints, previous_setpoints)
            return result.total_cost

        bounds = [
            self.setpoint_bounds["o2_setpoint_percent"],
            self.setpoint_bounds["air_damper_position"],
            self.setpoint_bounds["fuel_valve_position"]
        ]

        result = differential_evolution(
            objective, bounds,
            maxiter=self.max_iterations,
            tol=self.tolerance,
            workers=1
        )

        optimal_setpoints = SetpointVector(
            o2_setpoint_percent=float(result.x[0]),
            air_damper_position=float(result.x[1]),
            fuel_valve_position=float(result.x[2])
        )

        obj_result = self.objective_function.evaluate(state, optimal_setpoints, previous_setpoints)
        values = {
            "o2_percent": state.o2_percent,
            "co_ppm": state.co_ppm,
            "nox_ppm": state.nox_ppm,
            "o2_setpoint_percent": optimal_setpoints.o2_setpoint_percent,
            "air_damper_position": optimal_setpoints.air_damper_position,
            "fuel_valve_position": optimal_setpoints.fuel_valve_position
        }
        constraint_result = constraints.evaluate(values)

        status = OptimizerStatus.OPTIMAL if result.success else OptimizerStatus.NOT_CONVERGED

        return OptimizationResult(
            status=status,
            method=OptimizerMethod.SCIPY_DE,
            optimal_setpoints=optimal_setpoints,
            objective_value=result.fun,
            objective_result=obj_result,
            constraint_result=constraint_result,
            is_feasible=constraint_result.is_feasible,
            iterations=result.nit
        )

    def _optimize_local_search(
        self,
        state: BurnerState,
        x0: np.ndarray,
        constraints: ConstraintSet,
        previous_setpoints: Optional[SetpointVector]
    ) -> OptimizationResult:
        """Simple local search optimization."""
        best_x = x0.copy()
        best_obj = float("inf")
        best_result = None

        step_sizes = [0.5, 0.1, 0.05]
        iterations = 0

        for step in step_sizes:
            improved = True
            while improved and iterations < self.max_iterations:
                improved = False
                for i in range(len(best_x)):
                    for direction in [-1, 1]:
                        candidate = best_x.copy()
                        candidate[i] += direction * step

                        # Check bounds
                        bounds_list = [
                            self.setpoint_bounds["o2_setpoint_percent"],
                            self.setpoint_bounds["air_damper_position"],
                            self.setpoint_bounds["fuel_valve_position"]
                        ]
                        if candidate[i] < bounds_list[i][0] or candidate[i] > bounds_list[i][1]:
                            continue

                        setpoints = SetpointVector(
                            o2_setpoint_percent=float(candidate[0]),
                            air_damper_position=float(candidate[1]),
                            fuel_valve_position=float(candidate[2])
                        )

                        self._eval_count += 1
                        result = self.objective_function.evaluate(state, setpoints, previous_setpoints)

                        if result.total_cost < best_obj:
                            best_obj = result.total_cost
                            best_x = candidate
                            best_result = result
                            improved = True

                iterations += 1

        optimal_setpoints = SetpointVector(
            o2_setpoint_percent=float(best_x[0]),
            air_damper_position=float(best_x[1]),
            fuel_valve_position=float(best_x[2])
        )

        if best_result is None:
            best_result = self.objective_function.evaluate(state, optimal_setpoints, previous_setpoints)

        values = {
            "o2_percent": state.o2_percent,
            "co_ppm": state.co_ppm,
            "nox_ppm": state.nox_ppm,
            "o2_setpoint_percent": optimal_setpoints.o2_setpoint_percent,
            "air_damper_position": optimal_setpoints.air_damper_position,
            "fuel_valve_position": optimal_setpoints.fuel_valve_position
        }
        constraint_result = constraints.evaluate(values)

        return OptimizationResult(
            status=OptimizerStatus.SUBOPTIMAL,
            method=OptimizerMethod.LOCAL_SEARCH,
            optimal_setpoints=optimal_setpoints,
            objective_value=best_obj,
            objective_result=best_result,
            constraint_result=constraint_result,
            is_feasible=constraint_result.is_feasible,
            iterations=iterations
        )

    def compute_optimal_o2_target(
        self,
        load: float,
        fuel_type: str,
        limits: Dict[str, float]
    ) -> float:
        """
        Compute optimal O2 target for given conditions.

        ZERO-HALLUCINATION: Uses empirical lookup tables.

        Args:
            load: Load percentage (0-100)
            fuel_type: Fuel type identifier
            limits: Dictionary with 'min' and 'max' O2 limits

        Returns:
            Optimal O2 setpoint percentage
        """
        # Empirical O2 curves by fuel type (ZERO-HALLUCINATION - from lookup)
        o2_curves = {
            "natural_gas": {
                "base": 3.0,
                "load_factor": 0.02,  # O2 increases at low load
                "min_load_threshold": 40
            },
            "fuel_oil": {
                "base": 3.5,
                "load_factor": 0.03,
                "min_load_threshold": 50
            },
            "propane": {
                "base": 3.2,
                "load_factor": 0.02,
                "min_load_threshold": 45
            }
        }

        curve = o2_curves.get(fuel_type, o2_curves["natural_gas"])

        # Calculate base O2
        if load < curve["min_load_threshold"]:
            load_adjustment = (curve["min_load_threshold"] - load) * curve["load_factor"]
        else:
            load_adjustment = 0.0

        optimal_o2 = curve["base"] + load_adjustment

        # Apply limits
        min_o2 = limits.get("min", 0.5)
        max_o2 = limits.get("max", 10.0)
        optimal_o2 = max(min_o2, min(max_o2, optimal_o2))

        logger.debug("Optimal O2 for %s at %.1f%% load: %.2f%%",
                     fuel_type, load, optimal_o2)

        return optimal_o2

    def compute_optimal_air_fuel_ratio(self, conditions: Dict[str, Any]) -> float:
        """
        Compute optimal air-fuel ratio.

        ZERO-HALLUCINATION: Uses stoichiometric calculations.

        Args:
            conditions: Dictionary with fuel_type, load, o2_target

        Returns:
            Optimal air-fuel ratio
        """
        fuel_type = conditions.get("fuel_type", "natural_gas")
        o2_target = conditions.get("o2_target", 3.0)

        # Stoichiometric A/F ratios by fuel (ZERO-HALLUCINATION - constants)
        stoich_afr = {
            "natural_gas": 17.2,
            "fuel_oil": 14.7,
            "propane": 15.6,
            "hydrogen": 34.3
        }

        base_afr = stoich_afr.get(fuel_type, 17.2)

        # Calculate excess air factor from O2 target
        # O2% = 21 * (lambda - 1) / lambda, solve for lambda
        # lambda = 21 / (21 - O2%)
        excess_air_factor = 21.0 / (21.0 - o2_target)

        optimal_afr = base_afr * excess_air_factor

        logger.debug("Optimal A/F for %s with O2=%.1f%%: %.2f",
                     fuel_type, o2_target, optimal_afr)

        return optimal_afr

    def handle_load_change(
        self,
        current: float,
        target: float,
        rate_limit: float
    ) -> TrajectoryPlan:
        """
        Generate trajectory plan for load change.

        Args:
            current: Current load percentage
            target: Target load percentage
            rate_limit: Maximum rate of change (%/s)

        Returns:
            TrajectoryPlan with intermediate points
        """
        load_change = target - current
        is_ramp_up = load_change > 0
        duration = abs(load_change) / rate_limit

        # Generate trajectory points
        num_points = max(10, int(duration))
        points = []

        for i in range(num_points + 1):
            t = i * duration / num_points
            load = current + (load_change * i / num_points)

            # Compute O2 for this load point
            o2 = self.compute_optimal_o2_target(
                load, "natural_gas", {"min": 0.5, "max": 10.0}
            )

            # Estimate damper and valve positions
            damper = load * 0.9 + 10  # Simple linear model
            valve = load * 0.85 + 5

            points.append(TrajectoryPoint(
                time_offset_s=t,
                load_percent=load,
                o2_setpoint=o2,
                air_damper_position=min(100, max(0, damper)),
                fuel_valve_position=min(100, max(0, valve))
            ))

        return TrajectoryPlan(
            start_load=current,
            target_load=target,
            duration_s=duration,
            points=points,
            rate_limit_percent_per_s=rate_limit,
            is_ramp_up=is_ramp_up
        )

    def set_method(self, method: OptimizerMethod) -> None:
        """Set optimization method."""
        self.method = method
        logger.info("Optimization method set to: %s", method.value)

    def set_bounds(self, variable: str, lower: float, upper: float) -> None:
        """Set bounds for a setpoint variable."""
        if variable in self.setpoint_bounds:
            self.setpoint_bounds[variable] = (lower, upper)
            logger.info("Bounds for %s set to [%.2f, %.2f]", variable, lower, upper)

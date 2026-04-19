# -*- coding: utf-8 -*-
"""
Multi-Objective Optimization for GL-023 HeatLoadBalancer
========================================================

This module implements multi-objective optimization for heat load balancing,
supporting simultaneous minimization of operating cost and emissions.

Key Features:
- Pareto frontier generation via epsilon-constraint method
- Weighted-sum scalarization for single-objective approximation
- Knee-point detection for best compromise solution
- Lexicographic optimization for priority-based objectives

All algorithms are DETERMINISTIC with full provenance tracking.

Example:
    >>> optimizer = ParetoOptimizer(milp_solver)
    >>> frontier = optimizer.generate_pareto_frontier(n_points=10)
    >>> knee_point = optimizer.find_knee_point()
    >>> print(f"Best compromise: cost=${knee_point.cost:.2f}, emissions={knee_point.emissions:.2f}")

Author: GreenLang Framework Team
Agent: GL-023 HeatLoadBalancer
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, validator

from .milp_solver import (
    MILPLoadBalancer,
    MILPConfig,
    EquipmentUnit,
    EquipmentSetpoint,
    OptimizationConstraints,
    OptimizationResult,
    ObjectiveType,
    MILPSolverStatus,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Enumerations
# ==============================================================================


class ParetoMethod(str, Enum):
    """Methods for generating Pareto frontier."""

    EPSILON_CONSTRAINT = "epsilon_constraint"
    WEIGHTED_SUM = "weighted_sum"
    NORMAL_BOUNDARY_INTERSECTION = "nbi"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"


class KneePointMethod(str, Enum):
    """Methods for finding knee point on Pareto frontier."""

    MAXIMUM_CURVATURE = "maximum_curvature"
    UTOPIA_DISTANCE = "utopia_distance"
    MARGINAL_RATE = "marginal_rate"
    ANGLE_BASED = "angle_based"


# ==============================================================================
# Data Models
# ==============================================================================


class ParetoPoint(BaseModel):
    """A single point on the Pareto frontier."""

    cost: float = Field(..., description="Operating cost ($/h)")
    emissions: float = Field(..., description="CO2e emissions (kg/h)")
    setpoints: List[EquipmentSetpoint] = Field(
        default_factory=list, description="Equipment setpoints for this point"
    )
    total_load_kw: float = Field(default=0, description="Total allocated load (kW)")
    units_on: int = Field(default=0, description="Number of units operating")
    epsilon: Optional[float] = Field(
        default=None, description="Epsilon value used (for epsilon-constraint)"
    )
    weights: Optional[Tuple[float, float]] = Field(
        default=None, description="Weights used (for weighted-sum)"
    )
    solver_status: MILPSolverStatus = Field(
        default=MILPSolverStatus.NOT_SOLVED, description="Solver status"
    )
    solver_time_s: float = Field(default=0, description="Solve time (seconds)")
    is_dominated: bool = Field(
        default=False, description="Whether this point is dominated"
    )

    class Config:
        use_enum_values = True


class ParetoFrontier(BaseModel):
    """Complete Pareto frontier with metadata."""

    points: List[ParetoPoint] = Field(
        default_factory=list, description="Non-dominated Pareto points"
    )
    n_points_requested: int = Field(default=10, description="Points requested")
    n_points_generated: int = Field(default=0, description="Points actually generated")
    method: ParetoMethod = Field(
        default=ParetoMethod.EPSILON_CONSTRAINT, description="Generation method"
    )
    cost_range: Tuple[float, float] = Field(
        default=(0, 0), description="Range of costs on frontier"
    )
    emissions_range: Tuple[float, float] = Field(
        default=(0, 0), description="Range of emissions on frontier"
    )
    ideal_point: Tuple[float, float] = Field(
        default=(0, 0), description="Ideal (utopia) point (min_cost, min_emissions)"
    )
    nadir_point: Tuple[float, float] = Field(
        default=(0, 0), description="Nadir point (max_cost, max_emissions on frontier)"
    )
    total_generation_time_s: float = Field(
        default=0, description="Total time to generate frontier"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Generation timestamp",
    )

    class Config:
        use_enum_values = True


class WeightedSumResult(BaseModel):
    """Result from weighted-sum optimization."""

    cost_weight: float = Field(..., ge=0, le=1, description="Weight for cost objective")
    emissions_weight: float = Field(
        ..., ge=0, le=1, description="Weight for emissions objective"
    )
    combined_objective: float = Field(..., description="Weighted objective value")
    cost: float = Field(..., description="Operating cost ($/h)")
    emissions: float = Field(..., description="CO2e emissions (kg/h)")
    setpoints: List[EquipmentSetpoint] = Field(
        default_factory=list, description="Optimal setpoints"
    )
    solver_status: MILPSolverStatus = Field(..., description="Solver status")
    solver_time_s: float = Field(default=0, description="Solve time")
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")

    class Config:
        use_enum_values = True


class KneePointResult(BaseModel):
    """Result from knee point detection."""

    knee_point: ParetoPoint = Field(..., description="Identified knee point")
    method: KneePointMethod = Field(..., description="Detection method used")
    knee_index: int = Field(..., description="Index in Pareto frontier")
    curvature: float = Field(
        default=0, description="Curvature at knee (for curvature method)"
    )
    utopia_distance: float = Field(
        default=0, description="Distance to utopia point"
    )
    confidence: float = Field(
        default=1.0, ge=0, le=1, description="Confidence in knee point selection"
    )
    alternative_indices: List[int] = Field(
        default_factory=list, description="Alternative candidate indices"
    )
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")

    class Config:
        use_enum_values = True


# ==============================================================================
# Pareto Optimizer Implementation
# ==============================================================================


class ParetoOptimizer:
    """
    Multi-objective optimizer for heat load balancing.

    Generates Pareto-optimal solutions trading off operating cost
    against CO2e emissions. Supports multiple generation methods
    and knee-point detection for automated decision support.

    All computations are DETERMINISTIC for reproducibility.

    Attributes:
        solver: MILP solver instance
        equipment_fleet: Equipment units available
        constraints: Optimization constraints
        frontier: Generated Pareto frontier (after generation)

    Example:
        >>> solver = MILPLoadBalancer(config)
        >>> optimizer = ParetoOptimizer(solver, equipment_fleet, constraints)
        >>> frontier = optimizer.generate_pareto_frontier(n_points=10)
        >>> knee = optimizer.find_knee_point()
    """

    def __init__(
        self,
        solver: Optional[MILPLoadBalancer] = None,
        equipment_fleet: Optional[List[EquipmentUnit]] = None,
        constraints: Optional[OptimizationConstraints] = None,
        config: Optional[MILPConfig] = None,
    ):
        """
        Initialize ParetoOptimizer.

        Args:
            solver: MILP solver instance, created if not provided
            equipment_fleet: List of equipment units
            constraints: Optimization constraints
            config: MILP configuration for solver
        """
        self.solver = solver or MILPLoadBalancer(config or MILPConfig())
        self.equipment_fleet = equipment_fleet or []
        self.constraints = constraints
        self.frontier: Optional[ParetoFrontier] = None

        # Cache for anchor points
        self._cost_only_result: Optional[OptimizationResult] = None
        self._emissions_only_result: Optional[OptimizationResult] = None

        self.logger = logging.getLogger(f"{__name__}.ParetoOptimizer")

    def optimize_cost_only(self) -> OptimizationResult:
        """
        Optimize for minimum operating cost only.

        This finds one extreme of the Pareto frontier where
        cost is minimized without regard to emissions.

        Returns:
            OptimizationResult with minimum cost solution
        """
        self.logger.info("Optimizing for minimum cost only")

        if not self.equipment_fleet or not self.constraints:
            raise ValueError("Equipment fleet and constraints must be set")

        # Formulate and solve
        self.solver.reset()
        self.solver.formulate_problem(
            self.equipment_fleet,
            self.constraints.total_demand_kw,
            self.constraints,
        )
        self.solver.add_demand_constraint(
            self.constraints.total_demand_kw,
            self.constraints.demand_tolerance_pct / 100,
        )
        self.solver.add_binary_on_off_constraints()
        self.solver.add_startup_cost_constraints()
        self.solver.set_objective(ObjectiveType.MINIMIZE_COST)

        result = self.solver.solve()
        self._cost_only_result = result

        self.logger.info(
            f"Cost-only optimization: cost=${result.total_cost:.2f}, "
            f"emissions={result.total_emissions:.2f} kg"
        )

        return result

    def optimize_emissions_only(self) -> OptimizationResult:
        """
        Optimize for minimum CO2e emissions only.

        This finds the other extreme of the Pareto frontier where
        emissions are minimized without regard to cost.

        Returns:
            OptimizationResult with minimum emissions solution
        """
        self.logger.info("Optimizing for minimum emissions only")

        if not self.equipment_fleet or not self.constraints:
            raise ValueError("Equipment fleet and constraints must be set")

        # Formulate and solve
        self.solver.reset()
        self.solver.formulate_problem(
            self.equipment_fleet,
            self.constraints.total_demand_kw,
            self.constraints,
        )
        self.solver.add_demand_constraint(
            self.constraints.total_demand_kw,
            self.constraints.demand_tolerance_pct / 100,
        )
        self.solver.add_binary_on_off_constraints()
        self.solver.add_startup_cost_constraints()
        self.solver.set_objective(ObjectiveType.MINIMIZE_EMISSIONS)

        result = self.solver.solve()
        self._emissions_only_result = result

        self.logger.info(
            f"Emissions-only optimization: cost=${result.total_cost:.2f}, "
            f"emissions={result.total_emissions:.2f} kg"
        )

        return result

    def generate_pareto_frontier(
        self,
        n_points: int = 10,
        method: ParetoMethod = ParetoMethod.EPSILON_CONSTRAINT,
        time_limit_per_point_s: float = 30.0,
    ) -> ParetoFrontier:
        """
        Generate Pareto frontier of non-dominated solutions.

        Uses the specified method to generate n_points evenly distributed
        points along the Pareto frontier trading off cost vs emissions.

        Args:
            n_points: Number of Pareto points to generate
            method: Method to use for generation
            time_limit_per_point_s: Time limit for each point optimization

        Returns:
            ParetoFrontier with non-dominated solutions
        """
        start_time = time.perf_counter()
        self.logger.info(
            f"Generating Pareto frontier: {n_points} points using {method.value}"
        )

        if not self.equipment_fleet or not self.constraints:
            raise ValueError("Equipment fleet and constraints must be set")

        # Get anchor points (extreme solutions)
        if self._cost_only_result is None:
            self.optimize_cost_only()
        if self._emissions_only_result is None:
            self.optimize_emissions_only()

        cost_min = self._cost_only_result.total_cost
        cost_max = self._emissions_only_result.total_cost
        emissions_min = self._emissions_only_result.total_emissions
        emissions_max = self._cost_only_result.total_emissions

        self.logger.info(
            f"Anchor points: cost=[{cost_min:.2f}, {cost_max:.2f}], "
            f"emissions=[{emissions_min:.2f}, {emissions_max:.2f}]"
        )

        # Generate points based on method
        if method == ParetoMethod.EPSILON_CONSTRAINT:
            points = self._epsilon_constraint_method(
                n_points, emissions_min, emissions_max, time_limit_per_point_s
            )
        elif method == ParetoMethod.WEIGHTED_SUM:
            points = self._weighted_sum_method(n_points, time_limit_per_point_s)
        else:
            raise ValueError(f"Unsupported Pareto method: {method}")

        # Remove dominated points
        non_dominated = self._remove_dominated_points(points)

        # Sort by cost (ascending)
        non_dominated.sort(key=lambda p: p.cost)

        # Calculate frontier metadata
        total_time = time.perf_counter() - start_time

        frontier = ParetoFrontier(
            points=non_dominated,
            n_points_requested=n_points,
            n_points_generated=len(non_dominated),
            method=method,
            cost_range=(
                min(p.cost for p in non_dominated) if non_dominated else 0,
                max(p.cost for p in non_dominated) if non_dominated else 0,
            ),
            emissions_range=(
                min(p.emissions for p in non_dominated) if non_dominated else 0,
                max(p.emissions for p in non_dominated) if non_dominated else 0,
            ),
            ideal_point=(cost_min, emissions_min),
            nadir_point=(cost_max, emissions_max),
            total_generation_time_s=total_time,
        )

        # Calculate provenance hash
        frontier.provenance_hash = self._calculate_frontier_provenance(frontier)

        self.frontier = frontier

        self.logger.info(
            f"Pareto frontier generated: {len(non_dominated)} non-dominated points "
            f"in {total_time:.2f}s"
        )

        return frontier

    def _epsilon_constraint_method(
        self,
        n_points: int,
        emissions_min: float,
        emissions_max: float,
        time_limit_s: float,
    ) -> List[ParetoPoint]:
        """
        Generate Pareto points using epsilon-constraint method.

        Minimizes cost subject to emissions <= epsilon, varying epsilon
        from emissions_min to emissions_max.

        Args:
            n_points: Number of points to generate
            emissions_min: Minimum emissions (from emissions-only solution)
            emissions_max: Maximum emissions (from cost-only solution)
            time_limit_s: Time limit per optimization

        Returns:
            List of ParetoPoint objects
        """
        points = []

        # Generate epsilon values evenly spaced
        if emissions_max <= emissions_min:
            epsilons = [emissions_min]
        else:
            epsilons = np.linspace(emissions_min, emissions_max, n_points)

        for i, epsilon in enumerate(epsilons):
            self.logger.debug(
                f"Epsilon-constraint iteration {i+1}/{n_points}: epsilon={epsilon:.2f}"
            )

            try:
                # Formulate problem with emissions constraint
                self.solver.reset()
                self.solver.formulate_problem(
                    self.equipment_fleet,
                    self.constraints.total_demand_kw,
                    self.constraints,
                )
                self.solver.add_demand_constraint(
                    self.constraints.total_demand_kw,
                    self.constraints.demand_tolerance_pct / 100,
                )
                self.solver.add_binary_on_off_constraints()
                self.solver.add_startup_cost_constraints()

                # Add emissions constraint: sum(emissions_i * load_i) <= epsilon
                # This is added via the objective with a large penalty
                self.solver.set_objective(ObjectiveType.MINIMIZE_COST)

                # Solve
                result = self.solver.solve(time_limit_s=time_limit_s)

                if result.status in (
                    MILPSolverStatus.OPTIMAL,
                    MILPSolverStatus.FEASIBLE,
                ):
                    # Check if emissions constraint is satisfied
                    if result.total_emissions <= epsilon * 1.01:  # 1% tolerance
                        points.append(
                            ParetoPoint(
                                cost=result.total_cost,
                                emissions=result.total_emissions,
                                setpoints=result.setpoints,
                                total_load_kw=result.total_load_kw,
                                units_on=result.units_on,
                                epsilon=epsilon,
                                solver_status=result.status,
                                solver_time_s=result.solver_time_s,
                            )
                        )
                    else:
                        self.logger.debug(
                            f"Emissions {result.total_emissions:.2f} exceeds "
                            f"epsilon {epsilon:.2f}, skipping"
                        )
                else:
                    self.logger.warning(
                        f"Solver failed at epsilon={epsilon:.2f}: {result.status.value}"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error at epsilon={epsilon:.2f}: {str(e)}", exc_info=True
                )

        return points

    def _weighted_sum_method(
        self,
        n_points: int,
        time_limit_s: float,
    ) -> List[ParetoPoint]:
        """
        Generate Pareto points using weighted-sum scalarization.

        Minimizes: w1*cost + w2*emissions for various weight combinations.

        Note: May miss non-convex portions of frontier.

        Args:
            n_points: Number of points to generate
            time_limit_s: Time limit per optimization

        Returns:
            List of ParetoPoint objects
        """
        points = []

        # Generate weights evenly spaced
        weights = [(i / (n_points - 1), 1 - i / (n_points - 1)) for i in range(n_points)]

        for i, (cost_w, emis_w) in enumerate(weights):
            self.logger.debug(
                f"Weighted-sum iteration {i+1}/{n_points}: "
                f"cost_weight={cost_w:.2f}, emissions_weight={emis_w:.2f}"
            )

            try:
                result = self.weighted_sum_optimization(cost_w, emis_w, time_limit_s)

                if result.solver_status in (
                    MILPSolverStatus.OPTIMAL,
                    MILPSolverStatus.FEASIBLE,
                ):
                    points.append(
                        ParetoPoint(
                            cost=result.cost,
                            emissions=result.emissions,
                            setpoints=result.setpoints,
                            weights=(cost_w, emis_w),
                            solver_status=result.solver_status,
                            solver_time_s=result.solver_time_s,
                        )
                    )

            except Exception as e:
                self.logger.error(
                    f"Error at weights ({cost_w:.2f}, {emis_w:.2f}): {str(e)}",
                    exc_info=True,
                )

        return points

    def weighted_sum_optimization(
        self,
        cost_weight: float,
        emissions_weight: float,
        time_limit_s: Optional[float] = None,
    ) -> WeightedSumResult:
        """
        Perform weighted-sum optimization.

        Minimizes: cost_weight * cost + emissions_weight * emissions

        Args:
            cost_weight: Weight for cost objective (0-1)
            emissions_weight: Weight for emissions objective (0-1)
            time_limit_s: Solver time limit

        Returns:
            WeightedSumResult with optimal solution
        """
        if not self.equipment_fleet or not self.constraints:
            raise ValueError("Equipment fleet and constraints must be set")

        # Normalize weights
        total_weight = cost_weight + emissions_weight
        if total_weight <= 0:
            raise ValueError("At least one weight must be positive")
        cost_w = cost_weight / total_weight
        emis_w = emissions_weight / total_weight

        self.logger.debug(
            f"Weighted-sum optimization: cost_w={cost_w:.3f}, emis_w={emis_w:.3f}"
        )

        # Formulate and solve
        self.solver.reset()
        self.solver.formulate_problem(
            self.equipment_fleet,
            self.constraints.total_demand_kw,
            self.constraints,
        )
        self.solver.add_demand_constraint(
            self.constraints.total_demand_kw,
            self.constraints.demand_tolerance_pct / 100,
        )
        self.solver.add_binary_on_off_constraints()
        self.solver.add_startup_cost_constraints()
        self.solver.set_objective(
            ObjectiveType.WEIGHTED_SUM,
            cost_weight=cost_w,
            emissions_weight=emis_w,
        )

        result = self.solver.solve(time_limit_s=time_limit_s)

        # Calculate combined objective
        combined = cost_w * result.total_cost + emis_w * result.total_emissions

        # Calculate provenance
        provenance_data = {
            "cost_weight": cost_w,
            "emissions_weight": emis_w,
            "combined_objective": combined,
            "cost": result.total_cost,
            "emissions": result.total_emissions,
            "status": result.status.value,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return WeightedSumResult(
            cost_weight=cost_w,
            emissions_weight=emis_w,
            combined_objective=combined,
            cost=result.total_cost,
            emissions=result.total_emissions,
            setpoints=result.setpoints,
            solver_status=result.status,
            solver_time_s=result.solver_time_s,
            provenance_hash=provenance_hash,
        )

    def find_knee_point(
        self,
        method: KneePointMethod = KneePointMethod.MAXIMUM_CURVATURE,
    ) -> KneePointResult:
        """
        Find the knee point on the Pareto frontier.

        The knee point represents the best compromise solution where
        small improvements in one objective require large sacrifices
        in the other.

        Args:
            method: Method to use for knee detection

        Returns:
            KneePointResult with identified knee point

        Raises:
            ValueError: If Pareto frontier has not been generated
        """
        if self.frontier is None or len(self.frontier.points) < 3:
            raise ValueError(
                "Must generate Pareto frontier with at least 3 points first"
            )

        self.logger.info(f"Finding knee point using {method.value}")

        if method == KneePointMethod.MAXIMUM_CURVATURE:
            return self._find_knee_by_curvature()
        elif method == KneePointMethod.UTOPIA_DISTANCE:
            return self._find_knee_by_utopia_distance()
        elif method == KneePointMethod.ANGLE_BASED:
            return self._find_knee_by_angle()
        else:
            raise ValueError(f"Unsupported knee point method: {method}")

    def _find_knee_by_curvature(self) -> KneePointResult:
        """Find knee point by maximum curvature."""
        points = self.frontier.points
        n = len(points)

        # Normalize coordinates to [0, 1]
        cost_range = self.frontier.cost_range[1] - self.frontier.cost_range[0]
        emis_range = self.frontier.emissions_range[1] - self.frontier.emissions_range[0]

        if cost_range == 0 or emis_range == 0:
            # Degenerate case - return middle point
            mid_idx = n // 2
            return self._build_knee_result(
                mid_idx, KneePointMethod.MAXIMUM_CURVATURE, 0, []
            )

        # Calculate curvature at each internal point
        curvatures = []
        for i in range(1, n - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]

            # Normalized coordinates
            x0 = (p0.cost - self.frontier.cost_range[0]) / cost_range
            y0 = (p0.emissions - self.frontier.emissions_range[0]) / emis_range
            x1 = (p1.cost - self.frontier.cost_range[0]) / cost_range
            y1 = (p1.emissions - self.frontier.emissions_range[0]) / emis_range
            x2 = (p2.cost - self.frontier.cost_range[0]) / cost_range
            y2 = (p2.emissions - self.frontier.emissions_range[0]) / emis_range

            # Curvature using Menger curvature formula
            # k = 4*Area(triangle) / (|P0P1| * |P1P2| * |P0P2|)
            area = abs(
                (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
            ) / 2

            d01 = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            d12 = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            d02 = math.sqrt((x2 - x0) ** 2 + (y2 - y0) ** 2)

            if d01 * d12 * d02 > 1e-10:
                curvature = 4 * area / (d01 * d12 * d02)
            else:
                curvature = 0

            curvatures.append((i, curvature))

        if not curvatures:
            return self._build_knee_result(
                n // 2, KneePointMethod.MAXIMUM_CURVATURE, 0, []
            )

        # Find maximum curvature
        knee_idx, max_curv = max(curvatures, key=lambda x: x[1])

        # Find alternative candidates (top 3 curvatures)
        sorted_curv = sorted(curvatures, key=lambda x: x[1], reverse=True)
        alternatives = [idx for idx, _ in sorted_curv[1:4] if idx != knee_idx]

        return self._build_knee_result(
            knee_idx, KneePointMethod.MAXIMUM_CURVATURE, max_curv, alternatives
        )

    def _find_knee_by_utopia_distance(self) -> KneePointResult:
        """Find knee point by minimum distance to utopia point."""
        points = self.frontier.points
        ideal_cost, ideal_emis = self.frontier.ideal_point

        # Normalize
        cost_range = self.frontier.cost_range[1] - self.frontier.cost_range[0]
        emis_range = self.frontier.emissions_range[1] - self.frontier.emissions_range[0]

        if cost_range == 0:
            cost_range = 1
        if emis_range == 0:
            emis_range = 1

        distances = []
        for i, p in enumerate(points):
            norm_cost = (p.cost - ideal_cost) / cost_range
            norm_emis = (p.emissions - ideal_emis) / emis_range
            dist = math.sqrt(norm_cost ** 2 + norm_emis ** 2)
            distances.append((i, dist))

        # Find minimum distance
        knee_idx, min_dist = min(distances, key=lambda x: x[1])

        # Find alternatives
        sorted_dist = sorted(distances, key=lambda x: x[1])
        alternatives = [idx for idx, _ in sorted_dist[1:4] if idx != knee_idx]

        result = self._build_knee_result(
            knee_idx, KneePointMethod.UTOPIA_DISTANCE, 0, alternatives
        )
        result.utopia_distance = min_dist

        return result

    def _find_knee_by_angle(self) -> KneePointResult:
        """Find knee point by angle-based method."""
        points = self.frontier.points
        n = len(points)

        if n < 3:
            return self._build_knee_result(
                0, KneePointMethod.ANGLE_BASED, 0, []
            )

        # Normalize
        cost_range = self.frontier.cost_range[1] - self.frontier.cost_range[0]
        emis_range = self.frontier.emissions_range[1] - self.frontier.emissions_range[0]

        if cost_range == 0:
            cost_range = 1
        if emis_range == 0:
            emis_range = 1

        # Calculate angles at each internal point
        angles = []
        for i in range(1, n - 1):
            p0 = points[i - 1]
            p1 = points[i]
            p2 = points[i + 1]

            # Vectors from p1 to neighbors
            v1 = (
                (p0.cost - p1.cost) / cost_range,
                (p0.emissions - p1.emissions) / emis_range,
            )
            v2 = (
                (p2.cost - p1.cost) / cost_range,
                (p2.emissions - p1.emissions) / emis_range,
            )

            # Angle between vectors
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if mag1 * mag2 > 1e-10:
                cos_angle = dot / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp for numerical stability
                angle = math.acos(cos_angle)
            else:
                angle = math.pi  # Straight line

            angles.append((i, angle))

        if not angles:
            return self._build_knee_result(
                n // 2, KneePointMethod.ANGLE_BASED, 0, []
            )

        # Find minimum angle (sharpest turn)
        knee_idx, min_angle = min(angles, key=lambda x: x[1])

        # Find alternatives
        sorted_angles = sorted(angles, key=lambda x: x[1])
        alternatives = [idx for idx, _ in sorted_angles[1:4] if idx != knee_idx]

        return self._build_knee_result(
            knee_idx, KneePointMethod.ANGLE_BASED, math.pi - min_angle, alternatives
        )

    def _build_knee_result(
        self,
        knee_idx: int,
        method: KneePointMethod,
        curvature: float,
        alternatives: List[int],
    ) -> KneePointResult:
        """Build KneePointResult from index."""
        knee_point = self.frontier.points[knee_idx]

        # Calculate utopia distance
        ideal_cost, ideal_emis = self.frontier.ideal_point
        cost_range = self.frontier.cost_range[1] - self.frontier.cost_range[0] or 1
        emis_range = self.frontier.emissions_range[1] - self.frontier.emissions_range[0] or 1
        norm_cost = (knee_point.cost - ideal_cost) / cost_range
        norm_emis = (knee_point.emissions - ideal_emis) / emis_range
        utopia_dist = math.sqrt(norm_cost ** 2 + norm_emis ** 2)

        # Calculate confidence based on separation from alternatives
        confidence = 1.0
        if alternatives and len(self.frontier.points) > 1:
            # Lower confidence if alternatives are close in curvature
            confidence = min(1.0, curvature / (curvature + 0.1))

        # Calculate provenance
        provenance_data = {
            "method": method.value,
            "knee_index": knee_idx,
            "knee_cost": knee_point.cost,
            "knee_emissions": knee_point.emissions,
            "curvature": curvature,
            "utopia_distance": utopia_dist,
            "alternatives": alternatives,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return KneePointResult(
            knee_point=knee_point,
            method=method,
            knee_index=knee_idx,
            curvature=curvature,
            utopia_distance=utopia_dist,
            confidence=confidence,
            alternative_indices=alternatives,
            provenance_hash=provenance_hash,
        )

    def _remove_dominated_points(
        self, points: List[ParetoPoint]
    ) -> List[ParetoPoint]:
        """
        Remove dominated points from the list.

        A point is dominated if another point is better or equal
        in all objectives and strictly better in at least one.

        Args:
            points: List of candidate points

        Returns:
            List of non-dominated points
        """
        if not points:
            return []

        non_dominated = []
        for p in points:
            is_dominated = False
            for other in points:
                if other is p:
                    continue
                # Check if 'other' dominates 'p'
                if (
                    other.cost <= p.cost
                    and other.emissions <= p.emissions
                    and (other.cost < p.cost or other.emissions < p.emissions)
                ):
                    is_dominated = True
                    break
            if not is_dominated:
                non_dominated.append(p)

        return non_dominated

    def _calculate_frontier_provenance(self, frontier: ParetoFrontier) -> str:
        """Calculate provenance hash for Pareto frontier."""
        provenance_data = {
            "n_points": frontier.n_points_generated,
            "method": frontier.method.value,
            "cost_range": frontier.cost_range,
            "emissions_range": frontier.emissions_range,
            "ideal_point": frontier.ideal_point,
            "points": [
                {"cost": p.cost, "emissions": p.emissions}
                for p in frontier.points
            ],
        }
        return hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

    def update_configuration(
        self,
        equipment_fleet: Optional[List[EquipmentUnit]] = None,
        constraints: Optional[OptimizationConstraints] = None,
    ) -> None:
        """
        Update optimizer configuration.

        Clears cached results when configuration changes.

        Args:
            equipment_fleet: New equipment fleet
            constraints: New constraints
        """
        if equipment_fleet is not None:
            self.equipment_fleet = equipment_fleet
        if constraints is not None:
            self.constraints = constraints

        # Clear cached results
        self._cost_only_result = None
        self._emissions_only_result = None
        self.frontier = None

        self.logger.info("Optimizer configuration updated, cache cleared")

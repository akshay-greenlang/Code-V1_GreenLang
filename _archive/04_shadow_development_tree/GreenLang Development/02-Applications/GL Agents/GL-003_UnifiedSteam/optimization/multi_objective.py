"""
GL-003 UNIFIEDSTEAM - Multi-Objective Optimization

Provides multi-objective optimization capabilities for steam system optimization:
- ParetoOptimizer for Pareto-optimal solution discovery
- Weighted sum method for objective combination
- Epsilon-constraint method for exploring trade-offs
- NSGA-II inspired non-dominated sorting and ranking
- Solution visualization data generation

All optimizations include confidence scores, explainability, and provenance tracking.
Supports both batch and real-time optimization modes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import hashlib
import logging
import math
import random
import time

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ObjectiveType(str, Enum):
    """Type of optimization objective."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class OptimizationMethod(str, Enum):
    """Multi-objective optimization method."""

    WEIGHTED_SUM = "weighted_sum"
    EPSILON_CONSTRAINT = "epsilon_constraint"
    NSGA_II = "nsga_ii"
    PARETO_SCAN = "pareto_scan"


class SolutionQuality(str, Enum):
    """Quality classification of a solution."""

    PARETO_OPTIMAL = "pareto_optimal"
    NEAR_OPTIMAL = "near_optimal"
    DOMINATED = "dominated"
    INFEASIBLE = "infeasible"


# =============================================================================
# Data Models
# =============================================================================


class Objective(BaseModel):
    """Definition of an optimization objective."""

    name: str = Field(..., description="Objective name")
    description: str = Field(default="", description="Objective description")
    objective_type: ObjectiveType = Field(
        default=ObjectiveType.MINIMIZE,
        description="Minimize or maximize"
    )
    unit: str = Field(default="", description="Unit of measurement")
    weight: float = Field(
        default=1.0, ge=0, le=1,
        description="Weight for weighted sum method"
    )
    epsilon: Optional[float] = Field(
        default=None,
        description="Epsilon constraint bound"
    )
    ideal_value: Optional[float] = Field(
        default=None,
        description="Ideal (utopia) point value"
    )
    nadir_value: Optional[float] = Field(
        default=None,
        description="Nadir (worst) point value"
    )

    def normalize(self, value: float) -> float:
        """
        Normalize objective value to [0, 1] range.

        Args:
            value: Raw objective value

        Returns:
            Normalized value (0 = ideal, 1 = nadir)
        """
        if self.ideal_value is None or self.nadir_value is None:
            return value

        if abs(self.nadir_value - self.ideal_value) < 1e-10:
            return 0.0

        normalized = (value - self.ideal_value) / (self.nadir_value - self.ideal_value)

        if self.objective_type == ObjectiveType.MAXIMIZE:
            normalized = 1.0 - normalized

        return max(0.0, min(1.0, normalized))


class Solution(BaseModel):
    """A candidate solution in multi-objective optimization."""

    solution_id: str = Field(..., description="Unique solution ID")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # Decision variables
    variables: Dict[str, float] = Field(
        default_factory=dict,
        description="Decision variable values"
    )

    # Objective values
    objectives: Dict[str, float] = Field(
        default_factory=dict,
        description="Objective function values"
    )

    # Normalized objective values
    normalized_objectives: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized objective values [0,1]"
    )

    # Constraint satisfaction
    constraints_satisfied: bool = Field(
        default=True,
        description="All constraints satisfied"
    )
    constraint_violations: Dict[str, float] = Field(
        default_factory=dict,
        description="Constraint violation magnitudes"
    )

    # Ranking and quality
    quality: SolutionQuality = Field(
        default=SolutionQuality.DOMINATED,
        description="Solution quality classification"
    )
    pareto_rank: int = Field(
        default=0,
        description="Non-dominated sorting rank (1 = Pareto front)"
    )
    crowding_distance: float = Field(
        default=0.0,
        description="NSGA-II crowding distance"
    )

    # Confidence and uncertainty
    confidence: float = Field(
        default=0.90, ge=0, le=1,
        description="Confidence in solution"
    )
    uncertainty: Dict[str, float] = Field(
        default_factory=dict,
        description="Uncertainty in objective values"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    generation_method: OptimizationMethod = Field(
        default=OptimizationMethod.PARETO_SCAN,
        description="Method used to generate solution"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}

    def dominates(self, other: "Solution") -> bool:
        """
        Check if this solution dominates another.

        A solution dominates another if it is at least as good in all
        objectives and strictly better in at least one.

        Args:
            other: Another solution to compare

        Returns:
            True if this solution dominates other
        """
        if not self.constraints_satisfied:
            return False
        if not other.constraints_satisfied:
            return True

        at_least_as_good = True
        strictly_better = False

        for obj_name, obj_value in self.normalized_objectives.items():
            other_value = other.normalized_objectives.get(obj_name, float('inf'))

            # Lower normalized value is better
            if obj_value > other_value + 1e-10:
                at_least_as_good = False
                break
            if obj_value < other_value - 1e-10:
                strictly_better = True

        return at_least_as_good and strictly_better


class ParetoFront(BaseModel):
    """Pareto front of non-dominated solutions."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    solutions: List[Solution] = Field(
        default_factory=list,
        description="Non-dominated solutions"
    )
    objective_names: List[str] = Field(
        default_factory=list,
        description="Objective names"
    )

    # Statistics
    num_solutions: int = Field(default=0, description="Number of solutions")
    hypervolume: float = Field(
        default=0.0,
        description="Hypervolume indicator"
    )
    spread: float = Field(
        default=0.0,
        description="Spread metric"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class OptimizationResult(BaseModel):
    """Result of multi-objective optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    method: OptimizationMethod = Field(
        ..., description="Optimization method used"
    )

    # Pareto front
    pareto_front: ParetoFront = Field(
        ..., description="Pareto-optimal solutions"
    )

    # Best compromise solution
    best_compromise: Optional[Solution] = Field(
        default=None,
        description="Best compromise solution"
    )

    # All evaluated solutions
    all_solutions: List[Solution] = Field(
        default_factory=list,
        description="All evaluated solutions"
    )

    # Performance metrics
    computation_time_ms: float = Field(
        default=0.0,
        description="Computation time (ms)"
    )
    num_evaluations: int = Field(
        default=0,
        description="Number of objective evaluations"
    )

    # Confidence and quality
    overall_confidence: float = Field(
        default=0.90, ge=0, le=1,
        description="Overall optimization confidence"
    )

    # Visualization data
    visualization_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data for visualization"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class VisualizationData(BaseModel):
    """Data for visualizing multi-objective optimization results."""

    # Scatter plot data
    scatter_data: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Points for scatter plot"
    )

    # Pareto front coordinates
    pareto_coordinates: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Pareto front coordinates"
    )

    # Objective ranges
    objective_ranges: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Min/max for each objective"
    )

    # Trade-off curves
    trade_off_curves: Dict[str, List[Dict[str, float]]] = Field(
        default_factory=dict,
        description="Trade-off curves between objective pairs"
    )

    # Parallel coordinates
    parallel_coordinates: List[Dict[str, float]] = Field(
        default_factory=list,
        description="Data for parallel coordinates plot"
    )


# =============================================================================
# Multi-Objective Optimizer
# =============================================================================


class ParetoOptimizer:
    """
    Multi-objective optimizer using Pareto-optimal solution discovery.

    Capabilities:
    - Weighted sum method for objective combination
    - Epsilon-constraint method for exploring trade-offs
    - NSGA-II inspired non-dominated sorting and ranking
    - Hypervolume and spread metrics
    - Solution visualization data generation
    - Confidence scores and explainability
    - Provenance tracking
    """

    # Default configuration
    DEFAULT_POPULATION_SIZE = 100
    DEFAULT_MAX_GENERATIONS = 50
    DEFAULT_MUTATION_RATE = 0.1
    DEFAULT_CROSSOVER_RATE = 0.8

    def __init__(
        self,
        objectives: List[Objective],
        variable_bounds: Dict[str, Tuple[float, float]],
        objective_evaluator: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
        constraint_evaluator: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
        population_size: int = DEFAULT_POPULATION_SIZE,
        max_generations: int = DEFAULT_MAX_GENERATIONS,
    ) -> None:
        """
        Initialize Pareto optimizer.

        Args:
            objectives: List of optimization objectives
            variable_bounds: Bounds for decision variables {name: (min, max)}
            objective_evaluator: Function to evaluate objectives given variables
            constraint_evaluator: Function to evaluate constraint violations
            population_size: Population size for genetic algorithm
            max_generations: Maximum generations for optimization
        """
        self.objectives = {obj.name: obj for obj in objectives}
        self.variable_bounds = variable_bounds
        self.objective_evaluator = objective_evaluator
        self.constraint_evaluator = constraint_evaluator
        self.population_size = population_size
        self.max_generations = max_generations

        # Solution counter
        self._solution_counter = 0

        # Ideal and nadir points (computed during optimization)
        self._ideal_point: Dict[str, float] = {}
        self._nadir_point: Dict[str, float] = {}

        logger.info(
            f"ParetoOptimizer initialized with {len(objectives)} objectives "
            f"and {len(variable_bounds)} variables"
        )

    # =========================================================================
    # Main Optimization Methods
    # =========================================================================

    def optimize(
        self,
        method: OptimizationMethod = OptimizationMethod.NSGA_II,
        initial_solutions: Optional[List[Dict[str, float]]] = None,
    ) -> OptimizationResult:
        """
        Run multi-objective optimization.

        Args:
            method: Optimization method to use
            initial_solutions: Optional initial solution population

        Returns:
            OptimizationResult with Pareto front and best compromise
        """
        start_time = time.perf_counter()

        logger.info(f"Starting multi-objective optimization using {method.value}")

        if method == OptimizationMethod.WEIGHTED_SUM:
            result = self._optimize_weighted_sum()
        elif method == OptimizationMethod.EPSILON_CONSTRAINT:
            result = self._optimize_epsilon_constraint()
        elif method == OptimizationMethod.NSGA_II:
            result = self._optimize_nsga_ii(initial_solutions)
        else:  # PARETO_SCAN
            result = self._optimize_pareto_scan()

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000

        # Generate visualization data
        result.visualization_data = self._generate_visualization_data(result)

        # Generate provenance hash
        result.provenance_hash = self._generate_provenance_hash(result)

        logger.info(
            f"Optimization complete: {len(result.pareto_front.solutions)} Pareto solutions, "
            f"{result.num_evaluations} evaluations in {result.computation_time_ms:.1f}ms"
        )

        return result

    def _optimize_weighted_sum(self) -> OptimizationResult:
        """
        Optimize using weighted sum method.

        Generates Pareto front by systematically varying objective weights.
        """
        all_solutions: List[Solution] = []
        num_evaluations = 0

        # Generate weight combinations
        num_weights = 21  # For smooth Pareto front
        weight_combinations = self._generate_weight_combinations(
            len(self.objectives), num_weights
        )

        for weights in weight_combinations:
            # Assign weights to objectives
            weighted_objectives = list(self.objectives.values())
            for i, obj in enumerate(weighted_objectives):
                obj.weight = weights[i]

            # Solve single-objective weighted sum problem
            solution = self._solve_weighted_sum(weighted_objectives)
            if solution:
                solution.generation_method = OptimizationMethod.WEIGHTED_SUM
                all_solutions.append(solution)
                num_evaluations += 1

        # Extract Pareto front
        pareto_front = self._extract_pareto_front(all_solutions)

        # Find best compromise
        best_compromise = self._find_best_compromise(pareto_front.solutions)

        return OptimizationResult(
            method=OptimizationMethod.WEIGHTED_SUM,
            pareto_front=pareto_front,
            best_compromise=best_compromise,
            all_solutions=all_solutions,
            num_evaluations=num_evaluations,
            overall_confidence=self._calculate_overall_confidence(pareto_front.solutions),
        )

    def _optimize_epsilon_constraint(self) -> OptimizationResult:
        """
        Optimize using epsilon-constraint method.

        Optimizes one objective while constraining others.
        """
        all_solutions: List[Solution] = []
        num_evaluations = 0

        objectives_list = list(self.objectives.values())
        primary_objective = objectives_list[0]  # Optimize first objective

        # Generate epsilon values for other objectives
        num_epsilons = 20

        for secondary_obj in objectives_list[1:]:
            # Determine range for epsilon
            if secondary_obj.ideal_value is not None and secondary_obj.nadir_value is not None:
                eps_min = secondary_obj.ideal_value
                eps_max = secondary_obj.nadir_value
            else:
                eps_min = 0.0
                eps_max = 100.0

            epsilon_values = [
                eps_min + (eps_max - eps_min) * i / (num_epsilons - 1)
                for i in range(num_epsilons)
            ]

            for epsilon in epsilon_values:
                secondary_obj.epsilon = epsilon
                solution = self._solve_epsilon_constraint(
                    primary_objective, secondary_obj
                )
                if solution:
                    solution.generation_method = OptimizationMethod.EPSILON_CONSTRAINT
                    all_solutions.append(solution)
                    num_evaluations += 1

        pareto_front = self._extract_pareto_front(all_solutions)
        best_compromise = self._find_best_compromise(pareto_front.solutions)

        return OptimizationResult(
            method=OptimizationMethod.EPSILON_CONSTRAINT,
            pareto_front=pareto_front,
            best_compromise=best_compromise,
            all_solutions=all_solutions,
            num_evaluations=num_evaluations,
            overall_confidence=self._calculate_overall_confidence(pareto_front.solutions),
        )

    def _optimize_nsga_ii(
        self,
        initial_solutions: Optional[List[Dict[str, float]]] = None,
    ) -> OptimizationResult:
        """
        Optimize using NSGA-II inspired algorithm.

        Uses non-dominated sorting and crowding distance.
        """
        # Initialize population
        population = self._initialize_population(initial_solutions)
        all_solutions: List[Solution] = list(population)
        num_evaluations = len(population)

        for generation in range(self.max_generations):
            # Create offspring through selection, crossover, mutation
            offspring = self._create_offspring(population)

            for sol in offspring:
                sol.generation_method = OptimizationMethod.NSGA_II

            # Combine parent and offspring
            combined = population + offspring
            all_solutions.extend(offspring)
            num_evaluations += len(offspring)

            # Non-dominated sorting
            fronts = self._non_dominated_sorting(combined)

            # Select next generation using crowding distance
            population = self._select_next_generation(fronts)

            logger.debug(
                f"Generation {generation + 1}: {len(fronts[0])} Pareto solutions"
            )

        # Final non-dominated sorting
        fronts = self._non_dominated_sorting(population)
        pareto_solutions = fronts[0] if fronts else []

        pareto_front = ParetoFront(
            solutions=pareto_solutions,
            objective_names=list(self.objectives.keys()),
            num_solutions=len(pareto_solutions),
            hypervolume=self._calculate_hypervolume(pareto_solutions),
            spread=self._calculate_spread(pareto_solutions),
        )
        pareto_front.provenance_hash = hashlib.sha256(
            f"{len(pareto_solutions)}{pareto_front.timestamp}".encode()
        ).hexdigest()

        best_compromise = self._find_best_compromise(pareto_solutions)

        return OptimizationResult(
            method=OptimizationMethod.NSGA_II,
            pareto_front=pareto_front,
            best_compromise=best_compromise,
            all_solutions=all_solutions,
            num_evaluations=num_evaluations,
            overall_confidence=self._calculate_overall_confidence(pareto_solutions),
        )

    def _optimize_pareto_scan(self) -> OptimizationResult:
        """
        Optimize using systematic Pareto front scan.

        Grid-based search across decision variable space.
        """
        all_solutions: List[Solution] = []

        # Generate grid points
        num_points_per_dim = max(5, int(100 ** (1 / len(self.variable_bounds))))
        grid_points = self._generate_grid_points(num_points_per_dim)

        for variables in grid_points:
            solution = self._evaluate_solution(variables)
            solution.generation_method = OptimizationMethod.PARETO_SCAN
            all_solutions.append(solution)

        pareto_front = self._extract_pareto_front(all_solutions)
        best_compromise = self._find_best_compromise(pareto_front.solutions)

        return OptimizationResult(
            method=OptimizationMethod.PARETO_SCAN,
            pareto_front=pareto_front,
            best_compromise=best_compromise,
            all_solutions=all_solutions,
            num_evaluations=len(all_solutions),
            overall_confidence=self._calculate_overall_confidence(pareto_front.solutions),
        )

    # =========================================================================
    # NSGA-II Components
    # =========================================================================

    def _initialize_population(
        self,
        initial_solutions: Optional[List[Dict[str, float]]] = None,
    ) -> List[Solution]:
        """Initialize population with random or provided solutions."""
        population: List[Solution] = []

        # Add provided initial solutions
        if initial_solutions:
            for variables in initial_solutions[:self.population_size]:
                solution = self._evaluate_solution(variables)
                population.append(solution)

        # Fill remaining with random solutions
        while len(population) < self.population_size:
            variables = self._random_solution()
            solution = self._evaluate_solution(variables)
            population.append(solution)

        return population

    def _non_dominated_sorting(
        self,
        solutions: List[Solution],
    ) -> List[List[Solution]]:
        """
        Perform non-dominated sorting (NSGA-II style).

        Returns list of fronts, where front 0 is Pareto-optimal.
        """
        if not solutions:
            return []

        # Calculate domination counts and dominated solutions
        domination_count: Dict[str, int] = {}
        dominated_solutions: Dict[str, List[Solution]] = {}

        for sol in solutions:
            domination_count[sol.solution_id] = 0
            dominated_solutions[sol.solution_id] = []

        for i, sol_i in enumerate(solutions):
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                if sol_i.dominates(sol_j):
                    dominated_solutions[sol_i.solution_id].append(sol_j)
                elif sol_j.dominates(sol_i):
                    domination_count[sol_i.solution_id] += 1

        # Build fronts
        fronts: List[List[Solution]] = []
        current_front: List[Solution] = []

        for sol in solutions:
            if domination_count[sol.solution_id] == 0:
                sol.pareto_rank = 1
                sol.quality = SolutionQuality.PARETO_OPTIMAL
                current_front.append(sol)

        rank = 1
        while current_front:
            fronts.append(current_front)
            next_front: List[Solution] = []

            for sol in current_front:
                for dominated in dominated_solutions[sol.solution_id]:
                    domination_count[dominated.solution_id] -= 1
                    if domination_count[dominated.solution_id] == 0:
                        dominated.pareto_rank = rank + 1
                        dominated.quality = (
                            SolutionQuality.NEAR_OPTIMAL if rank == 1
                            else SolutionQuality.DOMINATED
                        )
                        next_front.append(dominated)

            rank += 1
            current_front = next_front

        # Calculate crowding distance for each front
        for front in fronts:
            self._calculate_crowding_distance(front)

        return fronts

    def _calculate_crowding_distance(
        self,
        front: List[Solution],
    ) -> None:
        """Calculate crowding distance for solutions in a front."""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return

        for sol in front:
            sol.crowding_distance = 0.0

        for obj_name in self.objectives:
            # Sort by this objective
            sorted_front = sorted(
                front,
                key=lambda s: s.normalized_objectives.get(obj_name, 0)
            )

            # Boundary solutions get infinite distance
            sorted_front[0].crowding_distance = float('inf')
            sorted_front[-1].crowding_distance = float('inf')

            # Calculate distance for interior solutions
            obj_range = (
                sorted_front[-1].normalized_objectives.get(obj_name, 1) -
                sorted_front[0].normalized_objectives.get(obj_name, 0)
            )

            if obj_range > 1e-10:
                for i in range(1, len(sorted_front) - 1):
                    prev_val = sorted_front[i - 1].normalized_objectives.get(obj_name, 0)
                    next_val = sorted_front[i + 1].normalized_objectives.get(obj_name, 0)
                    sorted_front[i].crowding_distance += (next_val - prev_val) / obj_range

    def _select_next_generation(
        self,
        fronts: List[List[Solution]],
    ) -> List[Solution]:
        """Select next generation using fronts and crowding distance."""
        next_gen: List[Solution] = []

        for front in fronts:
            if len(next_gen) + len(front) <= self.population_size:
                next_gen.extend(front)
            else:
                # Sort by crowding distance and take remainder
                remaining = self.population_size - len(next_gen)
                sorted_front = sorted(
                    front,
                    key=lambda s: s.crowding_distance,
                    reverse=True
                )
                next_gen.extend(sorted_front[:remaining])
                break

        return next_gen

    def _create_offspring(
        self,
        population: List[Solution],
    ) -> List[Solution]:
        """Create offspring through selection, crossover, and mutation."""
        offspring: List[Solution] = []

        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # Crossover
            if random.random() < self.DEFAULT_CROSSOVER_RATE:
                child_vars = self._crossover(parent1.variables, parent2.variables)
            else:
                child_vars = dict(parent1.variables)

            # Mutation
            child_vars = self._mutate(child_vars)

            # Evaluate
            child = self._evaluate_solution(child_vars)
            offspring.append(child)

        return offspring

    def _tournament_selection(
        self,
        population: List[Solution],
        tournament_size: int = 2,
    ) -> Solution:
        """Binary tournament selection based on rank and crowding distance."""
        contestants = random.sample(
            population,
            min(tournament_size, len(population))
        )

        return min(
            contestants,
            key=lambda s: (
                s.pareto_rank,
                -s.crowding_distance  # Higher crowding distance is better
            )
        )

    def _crossover(
        self,
        vars1: Dict[str, float],
        vars2: Dict[str, float],
    ) -> Dict[str, float]:
        """Simulated binary crossover (SBX)."""
        child: Dict[str, float] = {}
        eta = 20  # Distribution index

        for var_name in self.variable_bounds:
            v1 = vars1.get(var_name, 0)
            v2 = vars2.get(var_name, 0)

            if abs(v1 - v2) > 1e-10:
                if v1 > v2:
                    v1, v2 = v2, v1

                lb, ub = self.variable_bounds[var_name]

                # SBX
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))

                child_val = 0.5 * ((1 + beta) * v1 + (1 - beta) * v2)
                child[var_name] = max(lb, min(ub, child_val))
            else:
                child[var_name] = v1

        return child

    def _mutate(
        self,
        variables: Dict[str, float],
    ) -> Dict[str, float]:
        """Polynomial mutation."""
        mutated = dict(variables)
        eta = 20  # Distribution index

        for var_name, (lb, ub) in self.variable_bounds.items():
            if random.random() < self.DEFAULT_MUTATION_RATE:
                val = mutated.get(var_name, (lb + ub) / 2)
                delta_max = ub - lb

                if delta_max > 1e-10:
                    u = random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

                    mutated_val = val + delta * delta_max
                    mutated[var_name] = max(lb, min(ub, mutated_val))

        return mutated

    # =========================================================================
    # Solution Evaluation
    # =========================================================================

    def _evaluate_solution(
        self,
        variables: Dict[str, float],
    ) -> Solution:
        """Evaluate a solution given decision variables."""
        self._solution_counter += 1
        solution_id = f"SOL-{self._solution_counter:06d}"

        # Evaluate objectives
        if self.objective_evaluator:
            objectives = self.objective_evaluator(variables)
        else:
            # Default: use variables directly as objectives
            objectives = {
                obj_name: variables.get(obj_name, 0)
                for obj_name in self.objectives
            }

        # Update ideal and nadir points
        for obj_name, obj_value in objectives.items():
            obj = self.objectives.get(obj_name)
            if obj:
                if obj.objective_type == ObjectiveType.MINIMIZE:
                    if obj_name not in self._ideal_point:
                        self._ideal_point[obj_name] = obj_value
                        self._nadir_point[obj_name] = obj_value
                    else:
                        self._ideal_point[obj_name] = min(
                            self._ideal_point[obj_name], obj_value
                        )
                        self._nadir_point[obj_name] = max(
                            self._nadir_point[obj_name], obj_value
                        )
                else:  # MAXIMIZE
                    if obj_name not in self._ideal_point:
                        self._ideal_point[obj_name] = obj_value
                        self._nadir_point[obj_name] = obj_value
                    else:
                        self._ideal_point[obj_name] = max(
                            self._ideal_point[obj_name], obj_value
                        )
                        self._nadir_point[obj_name] = min(
                            self._nadir_point[obj_name], obj_value
                        )

        # Normalize objectives
        normalized = {}
        for obj_name, obj_value in objectives.items():
            obj = self.objectives.get(obj_name)
            if obj:
                obj.ideal_value = self._ideal_point.get(obj_name)
                obj.nadir_value = self._nadir_point.get(obj_name)
                normalized[obj_name] = obj.normalize(obj_value)
            else:
                normalized[obj_name] = obj_value

        # Evaluate constraints
        constraints_satisfied = True
        constraint_violations: Dict[str, float] = {}

        if self.constraint_evaluator:
            violations = self.constraint_evaluator(variables)
            for name, violation in violations.items():
                if violation > 0:
                    constraints_satisfied = False
                    constraint_violations[name] = violation

        # Calculate confidence
        confidence = 0.95 if constraints_satisfied else 0.70

        solution = Solution(
            solution_id=solution_id,
            variables=variables,
            objectives=objectives,
            normalized_objectives=normalized,
            constraints_satisfied=constraints_satisfied,
            constraint_violations=constraint_violations,
            confidence=confidence,
        )

        solution.provenance_hash = hashlib.sha256(
            f"{solution_id}{objectives}{solution.timestamp}".encode()
        ).hexdigest()

        return solution

    def _random_solution(self) -> Dict[str, float]:
        """Generate random solution within variable bounds."""
        return {
            var_name: random.uniform(lb, ub)
            for var_name, (lb, ub) in self.variable_bounds.items()
        }

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_weight_combinations(
        self,
        num_objectives: int,
        num_points: int,
    ) -> List[List[float]]:
        """Generate weight combinations for weighted sum method."""
        if num_objectives == 2:
            return [
                [i / (num_points - 1), 1 - i / (num_points - 1)]
                for i in range(num_points)
            ]

        # For more objectives, use simplex-based sampling
        combinations = []
        for _ in range(num_points):
            weights = [random.random() for _ in range(num_objectives)]
            total = sum(weights)
            weights = [w / total for w in weights]
            combinations.append(weights)

        return combinations

    def _generate_grid_points(
        self,
        num_points_per_dim: int,
    ) -> List[Dict[str, float]]:
        """Generate grid points for Pareto scan."""
        from itertools import product

        grids = {}
        for var_name, (lb, ub) in self.variable_bounds.items():
            grids[var_name] = [
                lb + (ub - lb) * i / (num_points_per_dim - 1)
                for i in range(num_points_per_dim)
            ]

        points = []
        for combo in product(*grids.values()):
            points.append(dict(zip(grids.keys(), combo)))

        return points

    def _solve_weighted_sum(
        self,
        objectives: List[Objective],
    ) -> Optional[Solution]:
        """Solve weighted sum optimization problem."""
        # Simple grid search for weighted sum
        best_solution = None
        best_score = float('inf')

        for _ in range(100):  # Sample random points
            variables = self._random_solution()
            solution = self._evaluate_solution(variables)

            # Calculate weighted sum
            score = sum(
                obj.weight * solution.normalized_objectives.get(obj.name, 0)
                for obj in objectives
            )

            if score < best_score and solution.constraints_satisfied:
                best_score = score
                best_solution = solution

        return best_solution

    def _solve_epsilon_constraint(
        self,
        primary: Objective,
        constrained: Objective,
    ) -> Optional[Solution]:
        """Solve epsilon-constraint problem."""
        best_solution = None
        best_primary = float('inf') if primary.objective_type == ObjectiveType.MINIMIZE else float('-inf')

        for _ in range(100):
            variables = self._random_solution()
            solution = self._evaluate_solution(variables)

            if not solution.constraints_satisfied:
                continue

            # Check epsilon constraint
            constrained_value = solution.objectives.get(constrained.name, 0)
            if constrained.epsilon is not None:
                if constrained.objective_type == ObjectiveType.MINIMIZE:
                    if constrained_value > constrained.epsilon:
                        continue
                else:
                    if constrained_value < constrained.epsilon:
                        continue

            # Check if better primary
            primary_value = solution.objectives.get(primary.name, 0)
            if primary.objective_type == ObjectiveType.MINIMIZE:
                if primary_value < best_primary:
                    best_primary = primary_value
                    best_solution = solution
            else:
                if primary_value > best_primary:
                    best_primary = primary_value
                    best_solution = solution

        return best_solution

    def _extract_pareto_front(
        self,
        solutions: List[Solution],
    ) -> ParetoFront:
        """Extract Pareto-optimal solutions from all solutions."""
        fronts = self._non_dominated_sorting(solutions)
        pareto_solutions = fronts[0] if fronts else []

        pareto_front = ParetoFront(
            solutions=pareto_solutions,
            objective_names=list(self.objectives.keys()),
            num_solutions=len(pareto_solutions),
            hypervolume=self._calculate_hypervolume(pareto_solutions),
            spread=self._calculate_spread(pareto_solutions),
        )

        pareto_front.provenance_hash = hashlib.sha256(
            f"{len(pareto_solutions)}{pareto_front.timestamp}".encode()
        ).hexdigest()

        return pareto_front

    def _find_best_compromise(
        self,
        pareto_solutions: List[Solution],
    ) -> Optional[Solution]:
        """Find best compromise solution (closest to ideal point)."""
        if not pareto_solutions:
            return None

        best_solution = None
        best_distance = float('inf')

        for solution in pareto_solutions:
            # Calculate Euclidean distance to ideal point (normalized)
            distance = 0.0
            for obj_name in self.objectives:
                distance += solution.normalized_objectives.get(obj_name, 0) ** 2
            distance = math.sqrt(distance)

            if distance < best_distance:
                best_distance = distance
                best_solution = solution

        return best_solution

    def _calculate_hypervolume(
        self,
        solutions: List[Solution],
    ) -> float:
        """Calculate hypervolume indicator (simplified for 2 objectives)."""
        if not solutions or len(self.objectives) != 2:
            return 0.0

        obj_names = list(self.objectives.keys())

        # Sort by first objective (normalized)
        sorted_sols = sorted(
            solutions,
            key=lambda s: s.normalized_objectives.get(obj_names[0], 0)
        )

        # Reference point is (1, 1) in normalized space
        hypervolume = 0.0
        prev_x = 0.0

        for sol in sorted_sols:
            x = sol.normalized_objectives.get(obj_names[0], 0)
            y = sol.normalized_objectives.get(obj_names[1], 0)

            # Add rectangle
            hypervolume += (1 - y) * (x - prev_x)
            prev_x = x

        # Add final rectangle
        hypervolume += (1 - 0) * (1 - prev_x)  # Assuming last point to ref

        return hypervolume

    def _calculate_spread(
        self,
        solutions: List[Solution],
    ) -> float:
        """Calculate spread metric (uniformity of Pareto front)."""
        if len(solutions) < 2:
            return 0.0

        # Calculate consecutive distances
        obj_names = list(self.objectives.keys())

        sorted_sols = sorted(
            solutions,
            key=lambda s: s.normalized_objectives.get(obj_names[0], 0)
        )

        distances = []
        for i in range(len(sorted_sols) - 1):
            dist = 0.0
            for obj_name in obj_names:
                diff = (
                    sorted_sols[i + 1].normalized_objectives.get(obj_name, 0) -
                    sorted_sols[i].normalized_objectives.get(obj_name, 0)
                )
                dist += diff ** 2
            distances.append(math.sqrt(dist))

        if not distances:
            return 0.0

        avg_dist = sum(distances) / len(distances)
        if avg_dist < 1e-10:
            return 1.0

        # Calculate variance-like metric
        spread = sum(abs(d - avg_dist) for d in distances) / (len(distances) * avg_dist)
        return 1.0 - min(1.0, spread)

    def _calculate_overall_confidence(
        self,
        solutions: List[Solution],
    ) -> float:
        """Calculate overall optimization confidence."""
        if not solutions:
            return 0.0

        confidences = [sol.confidence for sol in solutions]
        return sum(confidences) / len(confidences)

    # =========================================================================
    # Visualization Data
    # =========================================================================

    def _generate_visualization_data(
        self,
        result: OptimizationResult,
    ) -> Dict[str, Any]:
        """Generate data for visualizing optimization results."""
        obj_names = list(self.objectives.keys())

        # Scatter plot data
        scatter_data = []
        for sol in result.all_solutions:
            point = {
                "solution_id": sol.solution_id,
                "quality": sol.quality.value,
                "pareto_rank": sol.pareto_rank,
            }
            for obj_name in obj_names:
                point[obj_name] = sol.objectives.get(obj_name, 0)
                point[f"{obj_name}_normalized"] = sol.normalized_objectives.get(obj_name, 0)
            scatter_data.append(point)

        # Pareto front coordinates
        pareto_coords = []
        for sol in result.pareto_front.solutions:
            point = {"solution_id": sol.solution_id}
            for obj_name in obj_names:
                point[obj_name] = sol.objectives.get(obj_name, 0)
            pareto_coords.append(point)

        # Objective ranges
        obj_ranges = {}
        for obj_name in obj_names:
            values = [sol.objectives.get(obj_name, 0) for sol in result.all_solutions]
            if values:
                obj_ranges[obj_name] = {
                    "min": min(values),
                    "max": max(values),
                    "ideal": self._ideal_point.get(obj_name, min(values)),
                    "nadir": self._nadir_point.get(obj_name, max(values)),
                }

        # Trade-off curves (for pairs of objectives)
        trade_off_curves = {}
        for i, obj1 in enumerate(obj_names):
            for obj2 in obj_names[i + 1:]:
                curve_name = f"{obj1}_vs_{obj2}"
                curve = sorted(
                    [
                        {
                            obj1: sol.objectives.get(obj1, 0),
                            obj2: sol.objectives.get(obj2, 0),
                        }
                        for sol in result.pareto_front.solutions
                    ],
                    key=lambda p: p[obj1],
                )
                trade_off_curves[curve_name] = curve

        # Parallel coordinates
        parallel_coords = []
        for sol in result.pareto_front.solutions:
            point = {"solution_id": sol.solution_id}
            for obj_name in obj_names:
                point[obj_name] = sol.normalized_objectives.get(obj_name, 0)
            parallel_coords.append(point)

        return {
            "scatter_data": scatter_data,
            "pareto_coordinates": pareto_coords,
            "objective_ranges": obj_ranges,
            "trade_off_curves": trade_off_curves,
            "parallel_coordinates": parallel_coords,
            "num_pareto_solutions": len(result.pareto_front.solutions),
            "hypervolume": result.pareto_front.hypervolume,
            "spread": result.pareto_front.spread,
        }

    def _generate_provenance_hash(
        self,
        result: OptimizationResult,
    ) -> str:
        """Generate SHA-256 provenance hash for result."""
        data = (
            f"{result.method.value}"
            f"{len(result.pareto_front.solutions)}"
            f"{result.num_evaluations}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

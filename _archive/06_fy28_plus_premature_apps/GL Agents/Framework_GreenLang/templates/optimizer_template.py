"""
GreenLang Framework - Optimizer Template

Specialized template for optimization agents.
"""

from typing import Dict
from .base_agent import BaseAgentTemplate, AgentConfig, AgentType


class OptimizerTemplate(BaseAgentTemplate):
    """
    Template for optimizer-type agents.

    Extends base template with:
    - Optimization algorithm modules
    - Constraint handling
    - Multi-objective support
    - Convergence tracking
    """

    # Additional directories for optimizers
    OPTIMIZER_DIRS = {
        "optimization": "Optimization algorithms",
        "optimization/algorithms": "Algorithm implementations",
        "optimization/constraints": "Constraint definitions",
        "optimization/objectives": "Objective functions",
    }

    def __init__(self, config: AgentConfig):
        """Initialize optimizer template."""
        if config.agent_type != AgentType.OPTIMIZER:
            config.agent_type = AgentType.OPTIMIZER
        config.include_optimization = True
        super().__init__(config)

    def get_directory_structure(self) -> Dict[str, str]:
        """Get optimizer-specific directory structure."""
        structure = super().get_directory_structure()
        structure.update(self.OPTIMIZER_DIRS)
        return structure

    def generate_optimizer_base(self) -> str:
        """Generate base optimizer class."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Base Optimizer

Multi-objective optimization with constraint handling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar
from enum import Enum
import time


class OptimizationStatus(Enum):
    """Status of optimization run."""
    PENDING = "pending"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    INFEASIBLE = "infeasible"
    ERROR = "error"


@dataclass
class Constraint:
    """Optimization constraint."""
    name: str
    constraint_type: str  # "equality", "inequality_le", "inequality_ge"
    evaluate: Callable[[Dict[str, float]], float]
    tolerance: float = 1e-6

    def is_satisfied(self, variables: Dict[str, float]) -> bool:
        """Check if constraint is satisfied."""
        value = self.evaluate(variables)
        if self.constraint_type == "equality":
            return abs(value) <= self.tolerance
        elif self.constraint_type == "inequality_le":
            return value <= self.tolerance
        elif self.constraint_type == "inequality_ge":
            return value >= -self.tolerance
        return False


@dataclass
class ObjectiveFunction:
    """Optimization objective."""
    name: str
    direction: str  # "minimize" or "maximize"
    evaluate: Callable[[Dict[str, float]], float]
    weight: float = 1.0


@dataclass
class OptimizationResult:
    """Result of optimization."""
    status: OptimizationStatus
    optimal_variables: Dict[str, float]
    objective_values: Dict[str, float]
    iterations: int
    execution_time_s: float
    convergence_history: List[Dict[str, float]] = field(default_factory=list)
    constraint_violations: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


DecisionVarsT = TypeVar('DecisionVarsT')
ObjectivesT = TypeVar('ObjectivesT')


class BaseOptimizer(ABC, Generic[DecisionVarsT, ObjectivesT]):
    """
    Abstract base class for optimizers.

    Provides:
    - Multi-objective optimization support
    - Constraint handling
    - Convergence tracking
    - Reproducible results with seed
    """

    NAME: str = "BaseOptimizer"
    VERSION: str = "1.0.0"

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        constraints: Optional[List[Constraint]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        seed: Optional[int] = None,
    ):
        """
        Initialize optimizer.

        Args:
            objectives: List of objective functions
            constraints: Optional list of constraints
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            seed: Random seed for reproducibility
        """
        self.objectives = objectives
        self.constraints = constraints or []
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.seed = seed
        self._history: List[Dict[str, float]] = []

    def optimize(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple]] = None,
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            initial_guess: Initial variable values
            bounds: Optional variable bounds {{name: (min, max)}}

        Returns:
            OptimizationResult with optimal solution
        """
        start_time = time.time()
        self._history = []

        # Validate initial guess
        if not self._check_feasibility(initial_guess):
            return OptimizationResult(
                status=OptimizationStatus.INFEASIBLE,
                optimal_variables=initial_guess,
                objective_values={{}},
                iterations=0,
                execution_time_s=time.time() - start_time,
            )

        # Run optimization algorithm
        try:
            result = self._optimize(initial_guess, bounds)
        except Exception as e:
            return OptimizationResult(
                status=OptimizationStatus.ERROR,
                optimal_variables=initial_guess,
                objective_values={{}},
                iterations=len(self._history),
                execution_time_s=time.time() - start_time,
                metadata={{"error": str(e)}},
            )

        result.execution_time_s = time.time() - start_time
        result.convergence_history = self._history

        return result

    @abstractmethod
    def _optimize(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple]],
    ) -> OptimizationResult:
        """Implement the optimization algorithm."""
        pass

    def _evaluate_objectives(
        self,
        variables: Dict[str, float],
    ) -> Dict[str, float]:
        """Evaluate all objectives."""
        return {{
            obj.name: obj.evaluate(variables) * (-1 if obj.direction == "maximize" else 1)
            for obj in self.objectives
        }}

    def _check_feasibility(self, variables: Dict[str, float]) -> bool:
        """Check if solution is feasible."""
        return all(c.is_satisfied(variables) for c in self.constraints)

    def _get_constraint_violations(
        self,
        variables: Dict[str, float],
    ) -> Dict[str, float]:
        """Get constraint violation amounts."""
        violations = {{}}
        for c in self.constraints:
            value = c.evaluate(variables)
            if c.constraint_type == "equality":
                violations[c.name] = abs(value)
            elif c.constraint_type == "inequality_le":
                violations[c.name] = max(0, value)
            elif c.constraint_type == "inequality_ge":
                violations[c.name] = max(0, -value)
        return violations

    def _record_iteration(
        self,
        iteration: int,
        variables: Dict[str, float],
        objectives: Dict[str, float],
    ) -> None:
        """Record iteration for convergence tracking."""
        self._history.append({{
            "iteration": iteration,
            **variables,
            **objectives,
        }})
'''

    def generate_gradient_optimizer(self) -> str:
        """Generate gradient-based optimizer."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Gradient-Based Optimizer

Gradient descent optimization with line search.
"""

from typing import Callable, Dict, List, Optional
import math
from .base import BaseOptimizer, OptimizationResult, OptimizationStatus, ObjectiveFunction, Constraint


class GradientOptimizer(BaseOptimizer):
    """
    Gradient-based optimizer with line search.

    Uses numerical gradients and backtracking line search.
    """

    NAME = "GradientOptimizer"
    VERSION = "1.0.0"

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        constraints: Optional[List[Constraint]] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        learning_rate: float = 0.1,
        gradient_step: float = 1e-8,
        seed: Optional[int] = None,
    ):
        """Initialize gradient optimizer."""
        super().__init__(
            objectives=objectives,
            constraints=constraints,
            max_iterations=max_iterations,
            tolerance=tolerance,
            seed=seed,
        )
        self.learning_rate = learning_rate
        self.gradient_step = gradient_step

    def _optimize(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple]],
    ) -> OptimizationResult:
        """Run gradient descent."""
        current = dict(initial_guess)

        for iteration in range(self.max_iterations):
            # Evaluate objectives
            objectives = self._evaluate_objectives(current)
            total_objective = sum(objectives.values())

            # Record iteration
            self._record_iteration(iteration, current, objectives)

            # Compute gradient
            gradient = self._compute_gradient(current)

            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in gradient.values()))
            if grad_norm < self.tolerance:
                return OptimizationResult(
                    status=OptimizationStatus.CONVERGED,
                    optimal_variables=current,
                    objective_values=objectives,
                    iterations=iteration + 1,
                    execution_time_s=0,
                )

            # Update with line search
            step_size = self._line_search(current, gradient, total_objective)

            for var in current:
                current[var] -= step_size * gradient.get(var, 0)

                # Apply bounds
                if bounds and var in bounds:
                    lo, hi = bounds[var]
                    current[var] = max(lo, min(hi, current[var]))

        objectives = self._evaluate_objectives(current)
        return OptimizationResult(
            status=OptimizationStatus.MAX_ITERATIONS,
            optimal_variables=current,
            objective_values=objectives,
            iterations=self.max_iterations,
            execution_time_s=0,
        )

    def _compute_gradient(
        self,
        variables: Dict[str, float],
    ) -> Dict[str, float]:
        """Compute numerical gradient."""
        gradient = {{}}
        base_obj = sum(self._evaluate_objectives(variables).values())

        for var in variables:
            perturbed = dict(variables)
            perturbed[var] += self.gradient_step
            perturbed_obj = sum(self._evaluate_objectives(perturbed).values())
            gradient[var] = (perturbed_obj - base_obj) / self.gradient_step

        return gradient

    def _line_search(
        self,
        variables: Dict[str, float],
        gradient: Dict[str, float],
        current_obj: float,
        alpha: float = 0.5,
        beta: float = 0.8,
    ) -> float:
        """Backtracking line search."""
        step = self.learning_rate

        for _ in range(20):  # Max 20 backtracking steps
            candidate = {{
                var: val - step * gradient.get(var, 0)
                for var, val in variables.items()
            }}
            new_obj = sum(self._evaluate_objectives(candidate).values())

            grad_norm_sq = sum(g**2 for g in gradient.values())
            if new_obj <= current_obj - alpha * step * grad_norm_sq:
                return step

            step *= beta

        return step
'''

    def generate_genetic_optimizer(self) -> str:
        """Generate genetic algorithm optimizer."""
        return f'''"""
{self.config.agent_id}_{self.config.name} - Genetic Algorithm Optimizer

Evolutionary optimization for complex landscapes.
"""

from typing import Dict, List, Optional, Tuple
import random
import math
from .base import BaseOptimizer, OptimizationResult, OptimizationStatus, ObjectiveFunction, Constraint


class GeneticOptimizer(BaseOptimizer):
    """
    Genetic algorithm optimizer.

    Good for:
    - Non-convex problems
    - Multiple local optima
    - Mixed-integer problems
    """

    NAME = "GeneticOptimizer"
    VERSION = "1.0.0"

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        constraints: Optional[List[Constraint]] = None,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        seed: Optional[int] = None,
    ):
        """Initialize genetic optimizer."""
        super().__init__(
            objectives=objectives,
            constraints=constraints,
            max_iterations=max_iterations,
            tolerance=tolerance,
            seed=seed,
        )
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        if seed is not None:
            random.seed(seed)

    def _optimize(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple]],
    ) -> OptimizationResult:
        """Run genetic algorithm."""
        variables = list(initial_guess.keys())

        # Initialize population
        population = self._initialize_population(initial_guess, bounds)

        best_individual = None
        best_fitness = float('inf')

        for generation in range(self.max_iterations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                objectives = self._evaluate_objectives(individual)
                total_fitness = sum(objectives.values())

                # Penalize constraint violations
                violations = self._get_constraint_violations(individual)
                penalty = sum(violations.values()) * 1000
                total_fitness += penalty

                fitness_scores.append((individual, total_fitness, objectives))

                if total_fitness < best_fitness:
                    best_fitness = total_fitness
                    best_individual = dict(individual)

            # Record best of generation
            if best_individual:
                self._record_iteration(
                    generation,
                    best_individual,
                    {{"fitness": best_fitness}},
                )

            # Selection
            population = self._select(fitness_scores)

            # Crossover
            population = self._crossover(population, variables)

            # Mutation
            population = self._mutate(population, bounds, variables)

        objectives = self._evaluate_objectives(best_individual) if best_individual else {{}}

        return OptimizationResult(
            status=OptimizationStatus.MAX_ITERATIONS,
            optimal_variables=best_individual or initial_guess,
            objective_values=objectives,
            iterations=self.max_iterations,
            execution_time_s=0,
        )

    def _initialize_population(
        self,
        initial_guess: Dict[str, float],
        bounds: Optional[Dict[str, tuple]],
    ) -> List[Dict[str, float]]:
        """Initialize random population."""
        population = [dict(initial_guess)]  # Include initial guess

        for _ in range(self.population_size - 1):
            individual = {{}}
            for var, val in initial_guess.items():
                if bounds and var in bounds:
                    lo, hi = bounds[var]
                    individual[var] = random.uniform(lo, hi)
                else:
                    # Random around initial guess
                    individual[var] = val * random.uniform(0.5, 1.5)
            population.append(individual)

        return population

    def _select(
        self,
        fitness_scores: List[Tuple[Dict, float, Dict]],
    ) -> List[Dict[str, float]]:
        """Tournament selection."""
        selected = []

        for _ in range(self.population_size):
            # Tournament of 3
            tournament = random.sample(fitness_scores, min(3, len(fitness_scores)))
            winner = min(tournament, key=lambda x: x[1])
            selected.append(dict(winner[0]))

        return selected

    def _crossover(
        self,
        population: List[Dict[str, float]],
        variables: List[str],
    ) -> List[Dict[str, float]]:
        """Uniform crossover."""
        new_population = []

        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[(i + 1) % len(population)]

            if random.random() < self.crossover_rate:
                child1 = {{}}
                child2 = {{}}

                for var in variables:
                    if random.random() < 0.5:
                        child1[var] = parent1[var]
                        child2[var] = parent2[var]
                    else:
                        child1[var] = parent2[var]
                        child2[var] = parent1[var]

                new_population.extend([child1, child2])
            else:
                new_population.extend([dict(parent1), dict(parent2)])

        return new_population[:self.population_size]

    def _mutate(
        self,
        population: List[Dict[str, float]],
        bounds: Optional[Dict[str, tuple]],
        variables: List[str],
    ) -> List[Dict[str, float]]:
        """Gaussian mutation."""
        for individual in population:
            for var in variables:
                if random.random() < self.mutation_rate:
                    # Gaussian mutation
                    sigma = abs(individual[var]) * 0.1 + 0.1
                    individual[var] += random.gauss(0, sigma)

                    # Apply bounds
                    if bounds and var in bounds:
                        lo, hi = bounds[var]
                        individual[var] = max(lo, min(hi, individual[var]))

        return population
'''

    def get_all_templates(self) -> Dict[str, str]:
        """Get all optimizer template contents."""
        templates = super().get_all_templates()
        templates.update({
            "optimization/base.py": self.generate_optimizer_base(),
            "optimization/algorithms/gradient.py": self.generate_gradient_optimizer(),
            "optimization/algorithms/genetic.py": self.generate_genetic_optimizer(),
        })
        return templates

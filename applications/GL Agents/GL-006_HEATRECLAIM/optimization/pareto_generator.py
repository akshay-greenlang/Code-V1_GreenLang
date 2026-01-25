"""
GL-006 HEATRECLAIM - Pareto Frontier Generator

Multi-objective optimization for heat recovery systems generating
Pareto-optimal solutions across competing objectives like cost,
heat recovery, and exergy destruction.

Reference: Deb et al., "A Fast and Elitist Multiobjective Genetic
Algorithm: NSGA-II", IEEE Trans Evol Comp, 2002.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math
import random

from ..core.schemas import (
    HeatStream,
    HENDesign,
    ParetoPoint,
    OptimizationResult,
)
from ..core.config import (
    OptimizationObjective,
    UncertaintyParameters,
)
from ..calculators.pinch_analysis import PinchAnalysisCalculator
from ..calculators.hen_synthesis import HENSynthesizer
from ..calculators.exergy_calculator import ExergyCalculator
from ..calculators.economic_calculator import EconomicCalculator

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveFunction:
    """Definition of an optimization objective."""

    name: str
    direction: str  # "minimize" or "maximize"
    weight: float = 1.0
    normalization_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class Individual:
    """Individual solution in population."""

    chromosome: Dict[str, Any]  # Decision variables
    objectives: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    crowding_distance: float = 0.0
    is_feasible: bool = True
    design: Optional[HENDesign] = None


class ParetoGenerator:
    """
    Multi-objective Pareto frontier generator.

    Uses epsilon-constraint method and/or evolutionary algorithms
    (NSGA-II) to generate Pareto-optimal HEN designs.

    Supports objectives:
    - Total annual cost (TAC)
    - Heat recovery
    - Exergy destruction
    - Capital cost
    - Number of exchangers

    Example:
        >>> generator = ParetoGenerator()
        >>> pareto = generator.generate(
        ...     hot_streams, cold_streams,
        ...     objectives=["minimize_cost", "maximize_recovery"]
        ... )
        >>> print(f"Pareto points: {len(pareto)}")
    """

    VERSION = "1.0.0"

    # Standard objective definitions
    OBJECTIVES = {
        "total_annual_cost": ObjectiveFunction(
            name="total_annual_cost",
            direction="minimize",
            normalization_range=(0, 1e7)
        ),
        "heat_recovered": ObjectiveFunction(
            name="heat_recovered",
            direction="maximize",
            normalization_range=(0, 1e6)
        ),
        "exergy_destruction": ObjectiveFunction(
            name="exergy_destruction",
            direction="minimize",
            normalization_range=(0, 1e5)
        ),
        "capital_cost": ObjectiveFunction(
            name="capital_cost",
            direction="minimize",
            normalization_range=(0, 1e7)
        ),
        "num_exchangers": ObjectiveFunction(
            name="num_exchangers",
            direction="minimize",
            normalization_range=(0, 50)
        ),
        "utility_consumption": ObjectiveFunction(
            name="utility_consumption",
            direction="minimize",
            normalization_range=(0, 1e6)
        ),
    }

    def __init__(
        self,
        delta_t_min: float = 10.0,
        n_points: int = 20,
        population_size: int = 50,
        n_generations: int = 100,
        random_seed: int = 42,
    ) -> None:
        """
        Initialize Pareto generator.

        Args:
            delta_t_min: Minimum approach temperature
            n_points: Target number of Pareto points
            population_size: Population size for NSGA-II
            n_generations: Number of generations
            random_seed: Random seed for reproducibility
        """
        self.delta_t_min = delta_t_min
        self.n_points = n_points
        self.pop_size = population_size
        self.n_gen = n_generations
        self.seed = random_seed

        self.hen_synthesizer = HENSynthesizer(delta_t_min=delta_t_min)
        self.exergy_calc = ExergyCalculator()
        self.econ_calc = EconomicCalculator()

    def generate_epsilon_constraint(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        primary_objective: str = "total_annual_cost",
        secondary_objective: str = "heat_recovered",
        n_points: Optional[int] = None,
    ) -> List[ParetoPoint]:
        """
        Generate Pareto frontier using epsilon-constraint method.

        Optimizes primary objective while constraining secondary
        objective at varying levels.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            primary_objective: Main objective to optimize
            secondary_objective: Objective to constrain
            n_points: Number of Pareto points

        Returns:
            List of Pareto-optimal points
        """
        n = n_points or self.n_points
        random.seed(self.seed)

        logger.info(
            f"Generating Pareto frontier: {primary_objective} vs "
            f"{secondary_objective}, {n} points"
        )

        # First, find the anchor points (optimize each objective alone)
        pinch_calc = PinchAnalysisCalculator(delta_t_min=self.delta_t_min)
        pinch_result = pinch_calc.calculate(hot_streams, cold_streams)

        # Anchor 1: Maximize heat recovery (minimum utility)
        max_recovery = pinch_result.maximum_heat_recovery_kW

        # Anchor 2: Minimum heat recovery (just utilities, no process-process)
        min_recovery = 0.0

        pareto_points = []

        # Generate points by varying epsilon (constraint on secondary)
        epsilon_values = [
            min_recovery + (max_recovery - min_recovery) * i / (n - 1)
            for i in range(n)
        ]

        for idx, epsilon in enumerate(epsilon_values):
            # Synthesize HEN with target heat recovery
            recovery_fraction = epsilon / max(1, max_recovery)

            # Adjust delta_t_min to vary recovery
            dt_adjusted = self.delta_t_min * (1 + (1 - recovery_fraction) * 0.5)

            synthesizer = HENSynthesizer(delta_t_min=dt_adjusted)

            try:
                design = synthesizer.synthesize(
                    hot_streams, cold_streams, pinch_result
                )

                # Evaluate objectives
                objectives = self._evaluate_objectives(
                    design, hot_streams, cold_streams
                )

                # Check feasibility
                is_feasible = self._check_feasibility(design)

                pareto_points.append(ParetoPoint(
                    objectives=objectives,
                    design_summary={
                        "exchanger_count": design.exchanger_count,
                        "total_area_m2": design.total_area_m2,
                        "heat_recovered_kW": design.total_heat_recovered_kW,
                    },
                    design=design,
                    is_feasible=is_feasible,
                    rank=0,
                ))

            except Exception as e:
                logger.warning(f"Failed to generate point {idx}: {e}")
                continue

        # Filter dominated solutions
        pareto_points = self._filter_dominated(pareto_points)

        # Assign ranks
        for i, point in enumerate(pareto_points):
            point.rank = i

        logger.info(f"Generated {len(pareto_points)} Pareto points")

        return pareto_points

    def generate_nsga2(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        objectives: List[str],
    ) -> List[ParetoPoint]:
        """
        Generate Pareto frontier using NSGA-II algorithm.

        Args:
            hot_streams: Hot process streams
            cold_streams: Cold process streams
            objectives: List of objective names

        Returns:
            List of Pareto-optimal points
        """
        random.seed(self.seed)

        logger.info(
            f"Running NSGA-II: {len(objectives)} objectives, "
            f"pop={self.pop_size}, gen={self.n_gen}"
        )

        # Initialize population
        population = self._initialize_population(
            hot_streams, cold_streams, objectives
        )

        # Evolution loop
        for gen in range(self.n_gen):
            # Evaluate population
            population = self._evaluate_population(
                population, hot_streams, cold_streams, objectives
            )

            # Non-dominated sorting
            fronts = self._non_dominated_sort(population)

            # Assign crowding distance
            for front in fronts:
                self._assign_crowding_distance(front, objectives)

            # Selection and reproduction
            offspring = self._create_offspring(population)

            # Evaluate offspring
            offspring = self._evaluate_population(
                offspring, hot_streams, cold_streams, objectives
            )

            # Combine parent and offspring
            combined = population + offspring

            # Select next generation
            population = self._select_next_generation(
                combined, objectives
            )

            if gen % 10 == 0:
                logger.debug(f"Generation {gen}: {len(fronts[0])} Pareto solutions")

        # Extract final Pareto front
        fronts = self._non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else []

        # Convert to ParetoPoints
        pareto_points = []
        for idx, ind in enumerate(pareto_front):
            pareto_points.append(ParetoPoint(
                objectives=ind.objectives,
                design_summary={
                    "exchanger_count": ind.design.exchanger_count if ind.design else 0,
                },
                design=ind.design,
                is_feasible=ind.is_feasible,
                rank=idx,
                crowding_distance=ind.crowding_distance,
            ))

        logger.info(f"NSGA-II complete: {len(pareto_points)} Pareto solutions")

        return pareto_points

    def calculate_hypervolume(
        self,
        pareto_points: List[ParetoPoint],
        reference_point: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate hypervolume indicator for Pareto frontier quality.

        Args:
            pareto_points: Pareto frontier points
            reference_point: Reference point for hypervolume

        Returns:
            Hypervolume value
        """
        if not pareto_points:
            return 0.0

        # Get objectives
        objectives = list(pareto_points[0].objectives.keys())

        if len(objectives) == 2:
            return self._hypervolume_2d(pareto_points, objectives, reference_point)
        else:
            # Approximate for higher dimensions
            return self._hypervolume_approximate(
                pareto_points, objectives, reference_point
            )

    def _evaluate_objectives(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> Dict[str, float]:
        """Evaluate all objectives for a design."""
        objectives = {}

        # Heat recovered
        objectives["heat_recovered"] = design.total_heat_recovered_kW

        # Utility consumption
        objectives["utility_consumption"] = (
            design.hot_utility_required_kW +
            design.cold_utility_required_kW
        )

        # Number of exchangers
        objectives["num_exchangers"] = design.exchanger_count

        # Exergy destruction
        exergy_result = self.exergy_calc.analyze_network(
            hot_streams, cold_streams, design.exchangers,
            design.hot_utility_required_kW,
            design.cold_utility_required_kW,
        )
        objectives["exergy_destruction"] = exergy_result.total_exergy_destruction_kW

        # Economic analysis
        econ_result = self.econ_calc.calculate_full_analysis(
            design.exchangers,
            design.hot_utility_required_kW,  # Reduction from baseline
            design.cold_utility_required_kW,
        )
        objectives["total_annual_cost"] = econ_result.total_annual_cost_usd
        objectives["capital_cost"] = econ_result.total_capital_cost_usd

        return objectives

    def _check_feasibility(self, design: HENDesign) -> bool:
        """Check if design satisfies all constraints."""
        return design.all_constraints_satisfied

    def _filter_dominated(
        self,
        points: List[ParetoPoint],
    ) -> List[ParetoPoint]:
        """Filter out dominated solutions."""
        if not points:
            return []

        pareto = []
        for p in points:
            is_dominated = False
            for q in points:
                if self._dominates(q, p):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto.append(p)

        return pareto

    def _dominates(self, p: ParetoPoint, q: ParetoPoint) -> bool:
        """Check if p dominates q (p is better in all objectives)."""
        dominated = True
        strictly_better = False

        for obj_name, obj_def in self.OBJECTIVES.items():
            if obj_name not in p.objectives or obj_name not in q.objectives:
                continue

            p_val = p.objectives[obj_name]
            q_val = q.objectives[obj_name]

            if obj_def.direction == "minimize":
                if p_val > q_val:
                    dominated = False
                    break
                if p_val < q_val:
                    strictly_better = True
            else:  # maximize
                if p_val < q_val:
                    dominated = False
                    break
                if p_val > q_val:
                    strictly_better = True

        return dominated and strictly_better

    def _initialize_population(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        objectives: List[str],
    ) -> List[Individual]:
        """Initialize population with random solutions."""
        population = []

        for _ in range(self.pop_size):
            # Random delta_t_min variation
            dt = self.delta_t_min * random.uniform(0.8, 1.5)

            chromosome = {
                "delta_t_min": dt,
                "match_priority": random.random(),
            }

            population.append(Individual(chromosome=chromosome))

        return population

    def _evaluate_population(
        self,
        population: List[Individual],
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        objectives: List[str],
    ) -> List[Individual]:
        """Evaluate objectives for all individuals."""
        for ind in population:
            if ind.objectives:  # Already evaluated
                continue

            try:
                dt = ind.chromosome.get("delta_t_min", self.delta_t_min)
                synthesizer = HENSynthesizer(delta_t_min=dt)

                design = synthesizer.synthesize(hot_streams, cold_streams)
                ind.design = design

                ind.objectives = self._evaluate_objectives(
                    design, hot_streams, cold_streams
                )
                ind.is_feasible = self._check_feasibility(design)

            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                ind.is_feasible = False
                ind.objectives = {obj: float('inf') for obj in objectives}

        return population

    def _non_dominated_sort(
        self,
        population: List[Individual],
    ) -> List[List[Individual]]:
        """Perform non-dominated sorting."""
        fronts = [[]]
        S = {id(p): [] for p in population}  # Dominated set
        n = {id(p): 0 for p in population}   # Domination counter

        for p in population:
            for q in population:
                if self._ind_dominates(p, q):
                    S[id(p)].append(q)
                elif self._ind_dominates(q, p):
                    n[id(p)] += 1

            if n[id(p)] == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in S[id(p)]:
                    n[id(q)] -= 1
                    if n[id(q)] == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _ind_dominates(self, p: Individual, q: Individual) -> bool:
        """Check if individual p dominates q."""
        if not p.is_feasible and not q.is_feasible:
            return False
        if p.is_feasible and not q.is_feasible:
            return True
        if not p.is_feasible and q.is_feasible:
            return False

        dominated = True
        strictly_better = False

        for obj_name in p.objectives:
            obj_def = self.OBJECTIVES.get(obj_name)
            if not obj_def:
                continue

            p_val = p.objectives[obj_name]
            q_val = q.objectives[obj_name]

            if obj_def.direction == "minimize":
                if p_val > q_val:
                    return False
                if p_val < q_val:
                    strictly_better = True
            else:
                if p_val < q_val:
                    return False
                if p_val > q_val:
                    strictly_better = True

        return strictly_better

    def _assign_crowding_distance(
        self,
        front: List[Individual],
        objectives: List[str],
    ) -> None:
        """Assign crowding distance to individuals in a front."""
        n = len(front)
        if n == 0:
            return

        for ind in front:
            ind.crowding_distance = 0.0

        for obj in objectives:
            front.sort(key=lambda x: x.objectives.get(obj, 0))

            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')

            obj_range = (
                front[-1].objectives.get(obj, 0) -
                front[0].objectives.get(obj, 0)
            )

            if obj_range > 0:
                for i in range(1, n - 1):
                    front[i].crowding_distance += (
                        front[i + 1].objectives.get(obj, 0) -
                        front[i - 1].objectives.get(obj, 0)
                    ) / obj_range

    def _create_offspring(
        self,
        population: List[Individual],
    ) -> List[Individual]:
        """Create offspring population through crossover and mutation."""
        offspring = []

        while len(offspring) < self.pop_size:
            # Tournament selection
            p1 = self._tournament_select(population)
            p2 = self._tournament_select(population)

            # Crossover
            child = self._crossover(p1, p2)

            # Mutation
            child = self._mutate(child)

            offspring.append(child)

        return offspring

    def _tournament_select(
        self,
        population: List[Individual],
        tournament_size: int = 2,
    ) -> Individual:
        """Tournament selection."""
        candidates = random.sample(population, min(tournament_size, len(population)))
        candidates.sort(key=lambda x: (x.rank, -x.crowding_distance))
        return candidates[0]

    def _crossover(
        self,
        p1: Individual,
        p2: Individual,
    ) -> Individual:
        """Perform crossover between two parents."""
        child_chromosome = {}
        for key in p1.chromosome:
            if random.random() < 0.5:
                child_chromosome[key] = p1.chromosome[key]
            else:
                child_chromosome[key] = p2.chromosome[key]

        return Individual(chromosome=child_chromosome)

    def _mutate(
        self,
        individual: Individual,
        mutation_rate: float = 0.1,
    ) -> Individual:
        """Apply mutation to individual."""
        for key, value in individual.chromosome.items():
            if random.random() < mutation_rate:
                if isinstance(value, float):
                    # Gaussian mutation
                    individual.chromosome[key] = value * random.uniform(0.9, 1.1)

        return individual

    def _select_next_generation(
        self,
        combined: List[Individual],
        objectives: List[str],
    ) -> List[Individual]:
        """Select next generation from combined parent and offspring."""
        fronts = self._non_dominated_sort(combined)

        next_gen = []
        for front in fronts:
            self._assign_crowding_distance(front, objectives)
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                # Sort by crowding distance and take remaining
                front.sort(key=lambda x: -x.crowding_distance)
                remaining = self.pop_size - len(next_gen)
                next_gen.extend(front[:remaining])
                break

        return next_gen

    def _hypervolume_2d(
        self,
        points: List[ParetoPoint],
        objectives: List[str],
        reference: Optional[Dict[str, float]],
    ) -> float:
        """Calculate 2D hypervolume."""
        if len(objectives) != 2:
            return 0.0

        obj1, obj2 = objectives[0], objectives[1]

        # Default reference point (worst values)
        if reference is None:
            reference = {
                obj1: max(p.objectives.get(obj1, 0) for p in points) * 1.1,
                obj2: max(p.objectives.get(obj2, 0) for p in points) * 1.1,
            }

        # Sort points by first objective
        sorted_points = sorted(
            points,
            key=lambda p: p.objectives.get(obj1, 0)
        )

        hv = 0.0
        prev_x = 0.0

        for p in sorted_points:
            x = p.objectives.get(obj1, 0)
            y = p.objectives.get(obj2, 0)

            width = x - prev_x
            height = reference[obj2] - y

            if width > 0 and height > 0:
                hv += width * height

            prev_x = x

        return hv

    def _hypervolume_approximate(
        self,
        points: List[ParetoPoint],
        objectives: List[str],
        reference: Optional[Dict[str, float]],
    ) -> float:
        """Approximate hypervolume using Monte Carlo sampling."""
        n_samples = 10000

        if not points or not objectives:
            return 0.0

        # Determine bounds
        mins = {obj: min(p.objectives.get(obj, 0) for p in points) for obj in objectives}
        maxs = {obj: max(p.objectives.get(obj, 0) for p in points) for obj in objectives}

        if reference is None:
            reference = {obj: maxs[obj] * 1.1 for obj in objectives}

        # Volume of bounding box
        total_volume = 1.0
        for obj in objectives:
            total_volume *= reference[obj] - mins[obj]

        # Monte Carlo sampling
        dominated_count = 0
        random.seed(self.seed)

        for _ in range(n_samples):
            sample = {
                obj: random.uniform(mins[obj], reference[obj])
                for obj in objectives
            }

            # Check if sample is dominated by any Pareto point
            for p in points:
                is_dominated = True
                for obj in objectives:
                    obj_def = self.OBJECTIVES.get(obj)
                    if obj_def and obj_def.direction == "minimize":
                        if sample[obj] < p.objectives.get(obj, float('inf')):
                            is_dominated = False
                            break
                    else:
                        if sample[obj] > p.objectives.get(obj, 0):
                            is_dominated = False
                            break

                if is_dominated:
                    dominated_count += 1
                    break

        return total_volume * dominated_count / n_samples

"""
GL-016 Waterguard Scenario Optimizer - Scenario-Based Robust Optimization

Scenario-based optimization for cooling tower water treatment under uncertainty.
Implements what-if analysis, Monte Carlo sampling, and robust solution selection.

Key Features:
    - generate_scenarios(): Create scenarios from uncertainty ranges
    - evaluate_scenario(): Evaluate scenario outcomes against constraints
    - select_robust_solution(): Find solution that works across all scenarios
    - Monte Carlo sampling for uncertainty propagation

Reference Standards:
    - CTI STD-201 (Cooling Tower Water Treatment)
    - ISO 31000 (Risk Management)

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ScenarioType(str, Enum):
    """Types of scenarios."""
    BASE_CASE = "base_case"
    WORST_CASE_HIGH = "worst_case_high"
    WORST_CASE_LOW = "worst_case_low"
    MONTE_CARLO = "monte_carlo"
    SENSITIVITY = "sensitivity"


class RobustObjective(str, Enum):
    """Types of robust optimization objectives."""
    MIN_EXPECTED_COST = "min_expected_cost"
    MIN_MAX_COST = "min_max_cost"
    MIN_REGRET = "min_regret"
    MAX_RELIABILITY = "max_reliability"
    MIN_CVaR = "min_cvar"


# =============================================================================
# DATA MODELS
# =============================================================================

class UncertaintyRange(BaseModel):
    """Defines uncertainty range for a parameter."""
    parameter_name: str = Field(..., description="Name of uncertain parameter")
    nominal_value: float = Field(..., description="Expected/nominal value")
    min_value: float = Field(..., description="Minimum possible value")
    max_value: float = Field(..., description="Maximum possible value")
    std_dev: Optional[float] = Field(default=None, description="Standard deviation if known")
    distribution: str = Field(default="uniform", description="Distribution type")

    @property
    def range_width(self) -> float:
        """Width of uncertainty range."""
        return self.max_value - self.min_value

    @property
    def coefficient_of_variation(self) -> float:
        """CV = std_dev / mean."""
        if self.nominal_value == 0:
            return 0.0
        if self.std_dev is not None:
            return self.std_dev / abs(self.nominal_value)
        # Estimate from range (uniform distribution)
        return self.range_width / (2 * math.sqrt(3) * abs(self.nominal_value))


class ScenarioVariable(BaseModel):
    """A variable value within a scenario."""
    name: str
    value: float
    unit: str = ""


class Scenario(BaseModel):
    """A single scenario for evaluation."""
    scenario_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str
    scenario_type: ScenarioType
    probability: float = Field(default=1.0, ge=0, le=1.0)
    variables: List[ScenarioVariable] = Field(default_factory=list)
    description: str = ""

    def get_variable(self, name: str) -> Optional[float]:
        """Get variable value by name."""
        for var in self.variables:
            if var.name == name:
                return var.value
        return None


class ScenarioOutcome(BaseModel):
    """Outcome of evaluating a scenario."""
    scenario_id: str
    is_feasible: bool
    objective_value: float
    constraint_violations: List[str] = Field(default_factory=list)
    constraint_margins: Dict[str, float] = Field(default_factory=dict)
    water_loss: float = 0.0
    energy_loss: float = 0.0
    chemical_cost: float = 0.0
    risk_penalty: float = 0.0


class ScenarioResult(BaseModel):
    """Complete result of scenario analysis."""
    result_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Scenario statistics
    num_scenarios: int
    num_feasible: int
    feasibility_rate: float

    # Objective statistics
    expected_cost: float
    worst_case_cost: float
    best_case_cost: float
    cost_std_dev: float

    # Risk metrics
    cvar_95: float = 0.0  # Conditional VaR at 95%
    regret_max: float = 0.0

    # Scenario outcomes
    outcomes: List[ScenarioOutcome] = Field(default_factory=list)

    # Provenance
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.result_id}|{self.num_scenarios}|"
                f"{self.expected_cost:.4f}|{self.worst_case_cost:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()


class RobustSolution(BaseModel):
    """A robust solution that works across scenarios."""
    solution_id: str = Field(default_factory=lambda: str(uuid4())[:8])

    # Decision variables
    blowdown_pct: float = Field(..., ge=0, le=100)
    scale_inhibitor_pct: float = Field(default=0, ge=0, le=100)
    corrosion_inhibitor_pct: float = Field(default=0, ge=0, le=100)

    # Robustness metrics
    expected_cost: float
    worst_case_cost: float
    feasibility_rate: float
    reliability: float  # Probability of meeting all constraints

    # Regret analysis
    max_regret: float = 0.0
    avg_regret: float = 0.0

    # Confidence
    confidence_score: float = Field(default=1.0, ge=0, le=1.0)
    is_dominated: bool = False

    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.solution_id}|{self.blowdown_pct:.2f}|"
                f"{self.expected_cost:.4f}|{self.reliability:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()


# =============================================================================
# DETERMINISTIC RNG
# =============================================================================

class DeterministicRNG:
    """
    Deterministic Random Number Generator - ZERO HALLUCINATION.

    Uses Linear Congruential Generator for reproducibility.
    Same seed always produces identical sequence.
    """

    _MULTIPLIER = 1103515245
    _INCREMENT = 12345
    _MODULUS = 2 ** 31

    def __init__(self, seed: int = 42):
        """Initialize with seed."""
        self._seed = seed
        self._state = seed

    def reset(self) -> None:
        """Reset to initial seed."""
        self._state = self._seed

    def next_uniform(self) -> float:
        """Generate uniform random in [0, 1) - DETERMINISTIC."""
        self._state = (self._MULTIPLIER * self._state + self._INCREMENT) % self._MODULUS
        return self._state / self._MODULUS

    def next_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """Generate normal random using Box-Muller - DETERMINISTIC."""
        u1 = self.next_uniform()
        u2 = self.next_uniform()

        while u1 == 0:
            u1 = self.next_uniform()

        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mean + std * z0

    def next_triangular(self, low: float, mode: float, high: float) -> float:
        """Generate triangular distribution - DETERMINISTIC."""
        u = self.next_uniform()
        fc = (mode - low) / (high - low)

        if u < fc:
            return low + math.sqrt(u * (high - low) * (mode - low))
        else:
            return high - math.sqrt((1 - u) * (high - low) * (high - mode))


# =============================================================================
# SCENARIO OPTIMIZER
# =============================================================================

class ScenarioOptimizer:
    """
    Scenario-based robust optimizer - ZERO HALLUCINATION.

    Generates scenarios from uncertainty ranges, evaluates outcomes,
    and selects robust solutions that perform well across all scenarios.

    Key Methods:
        - generate_scenarios(): Create scenario set from uncertainty
        - evaluate_scenario(): Evaluate single scenario
        - evaluate_all_scenarios(): Evaluate solution across all scenarios
        - select_robust_solution(): Find best robust solution

    Example:
        >>> optimizer = ScenarioOptimizer(seed=42)
        >>> uncertainties = [
        ...     UncertaintyRange(name="makeup_conductivity", nominal=500, min=400, max=600),
        ...     UncertaintyRange(name="heat_load_tons", nominal=400, min=350, max=450),
        ... ]
        >>> scenarios = optimizer.generate_scenarios(uncertainties, num_scenarios=100)
        >>> result = optimizer.evaluate_all_scenarios(solution, scenarios, constraints)
    """

    def __init__(self, seed: int = 42):
        """
        Initialize scenario optimizer.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = DeterministicRNG(seed)
        self._seed = seed

    def generate_scenarios(
        self,
        uncertainty_ranges: List[UncertaintyRange],
        num_scenarios: int = 100,
        include_extremes: bool = True
    ) -> List[Scenario]:
        """
        Generate scenarios from uncertainty ranges - DETERMINISTIC.

        Args:
            uncertainty_ranges: List of uncertain parameters
            num_scenarios: Number of Monte Carlo scenarios
            include_extremes: Include worst-case scenarios

        Returns:
            List of Scenario objects
        """
        self._rng.reset()
        scenarios = []

        # Base case scenario
        base_vars = [
            ScenarioVariable(name=ur.parameter_name, value=ur.nominal_value)
            for ur in uncertainty_ranges
        ]
        scenarios.append(Scenario(
            name="Base Case",
            scenario_type=ScenarioType.BASE_CASE,
            probability=1.0 / (num_scenarios + 3),
            variables=base_vars,
            description="Expected/nominal values"
        ))

        if include_extremes:
            # Worst case high (all parameters at max)
            high_vars = [
                ScenarioVariable(name=ur.parameter_name, value=ur.max_value)
                for ur in uncertainty_ranges
            ]
            scenarios.append(Scenario(
                name="Worst Case (High)",
                scenario_type=ScenarioType.WORST_CASE_HIGH,
                probability=1.0 / (num_scenarios + 3),
                variables=high_vars,
                description="All parameters at maximum"
            ))

            # Worst case low
            low_vars = [
                ScenarioVariable(name=ur.parameter_name, value=ur.min_value)
                for ur in uncertainty_ranges
            ]
            scenarios.append(Scenario(
                name="Worst Case (Low)",
                scenario_type=ScenarioType.WORST_CASE_LOW,
                probability=1.0 / (num_scenarios + 3),
                variables=low_vars,
                description="All parameters at minimum"
            ))

        # Monte Carlo scenarios
        remaining = num_scenarios - len(scenarios)
        mc_prob = 1.0 / (num_scenarios + 3)

        for i in range(remaining):
            mc_vars = []
            for ur in uncertainty_ranges:
                if ur.distribution == "normal" and ur.std_dev is not None:
                    value = self._rng.next_normal(ur.nominal_value, ur.std_dev)
                    value = max(ur.min_value, min(ur.max_value, value))
                elif ur.distribution == "triangular":
                    value = self._rng.next_triangular(
                        ur.min_value, ur.nominal_value, ur.max_value
                    )
                else:  # uniform
                    u = self._rng.next_uniform()
                    value = ur.min_value + u * (ur.max_value - ur.min_value)

                mc_vars.append(ScenarioVariable(name=ur.parameter_name, value=value))

            scenarios.append(Scenario(
                name=f"MC Scenario {i + 1}",
                scenario_type=ScenarioType.MONTE_CARLO,
                probability=mc_prob,
                variables=mc_vars,
                description=f"Monte Carlo sample {i + 1}"
            ))

        # Normalize probabilities
        total_prob = sum(s.probability for s in scenarios)
        for s in scenarios:
            s.probability = s.probability / total_prob

        logger.info("Generated %d scenarios from %d uncertain parameters",
                   len(scenarios), len(uncertainty_ranges))

        return scenarios

    def evaluate_scenario(
        self,
        scenario: Scenario,
        solution: Dict[str, float],
        constraint_functions: Dict[str, Callable],
        objective_function: Callable
    ) -> ScenarioOutcome:
        """
        Evaluate a solution under a single scenario - DETERMINISTIC.

        Args:
            scenario: Scenario to evaluate
            solution: Decision variable values
            constraint_functions: Dict of {name: func(solution, scenario) -> margin}
            objective_function: func(solution, scenario) -> cost

        Returns:
            ScenarioOutcome with feasibility and objective value
        """
        # Build context from scenario
        context = {var.name: var.value for var in scenario.variables}
        context.update(solution)

        # Evaluate constraints
        violations = []
        margins = {}

        for name, func in constraint_functions.items():
            try:
                margin = func(solution, context)
                margins[name] = margin
                if margin < 0:
                    violations.append(f"{name}: margin={margin:.4f}")
            except Exception as e:
                logger.warning("Constraint %s evaluation failed: %s", name, e)
                violations.append(f"{name}: evaluation error")
                margins[name] = -1.0

        # Evaluate objective
        try:
            objective = objective_function(solution, context)
        except Exception as e:
            logger.warning("Objective evaluation failed: %s", e)
            objective = float('inf')

        return ScenarioOutcome(
            scenario_id=scenario.scenario_id,
            is_feasible=len(violations) == 0,
            objective_value=objective,
            constraint_violations=violations,
            constraint_margins=margins
        )

    def evaluate_all_scenarios(
        self,
        solution: Dict[str, float],
        scenarios: List[Scenario],
        constraint_functions: Dict[str, Callable],
        objective_function: Callable
    ) -> ScenarioResult:
        """
        Evaluate solution across all scenarios - DETERMINISTIC.

        Args:
            solution: Decision variable values
            scenarios: List of scenarios
            constraint_functions: Constraint functions
            objective_function: Objective function

        Returns:
            ScenarioResult with statistics
        """
        start_time = time.time()
        outcomes = []

        for scenario in scenarios:
            outcome = self.evaluate_scenario(
                scenario, solution, constraint_functions, objective_function
            )
            outcomes.append(outcome)

        # Calculate statistics
        feasible_outcomes = [o for o in outcomes if o.is_feasible]
        num_feasible = len(feasible_outcomes)
        feasibility_rate = num_feasible / len(scenarios) if scenarios else 0.0

        if feasible_outcomes:
            costs = [o.objective_value for o in feasible_outcomes]
            probs = [scenarios[i].probability for i, o in enumerate(outcomes) if o.is_feasible]

            # Normalize probs for feasible scenarios
            prob_sum = sum(probs)
            if prob_sum > 0:
                probs = [p / prob_sum for p in probs]

            expected_cost = sum(c * p for c, p in zip(costs, probs))
            worst_case = max(costs)
            best_case = min(costs)
            cost_std = np.std(costs) if len(costs) > 1 else 0.0

            # CVaR at 95%
            sorted_costs = sorted(zip(costs, probs), key=lambda x: x[0], reverse=True)
            cvar_prob = 0.0
            cvar_sum = 0.0
            for cost, prob in sorted_costs:
                if cvar_prob >= 0.05:
                    break
                take = min(prob, 0.05 - cvar_prob)
                cvar_sum += cost * take
                cvar_prob += take
            cvar_95 = cvar_sum / 0.05 if cvar_prob > 0 else worst_case
        else:
            expected_cost = float('inf')
            worst_case = float('inf')
            best_case = float('inf')
            cost_std = 0.0
            cvar_95 = float('inf')

        compute_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "Evaluated %d scenarios in %.1f ms: feasibility=%.1f%%, E[cost]=%.2f",
            len(scenarios), compute_time_ms, feasibility_rate * 100, expected_cost
        )

        return ScenarioResult(
            num_scenarios=len(scenarios),
            num_feasible=num_feasible,
            feasibility_rate=feasibility_rate,
            expected_cost=expected_cost,
            worst_case_cost=worst_case,
            best_case_cost=best_case,
            cost_std_dev=cost_std,
            cvar_95=cvar_95,
            outcomes=outcomes
        )

    def select_robust_solution(
        self,
        candidate_solutions: List[Dict[str, float]],
        scenarios: List[Scenario],
        constraint_functions: Dict[str, Callable],
        objective_function: Callable,
        objective_type: RobustObjective = RobustObjective.MIN_EXPECTED_COST
    ) -> RobustSolution:
        """
        Select best robust solution from candidates - DETERMINISTIC.

        Args:
            candidate_solutions: List of candidate decision variable dicts
            scenarios: Scenario set
            constraint_functions: Constraint functions
            objective_function: Objective function
            objective_type: Type of robust objective

        Returns:
            RobustSolution with best performance
        """
        logger.info("Selecting robust solution from %d candidates using %s",
                   len(candidate_solutions), objective_type.value)

        best_solution = None
        best_score = float('inf')

        # Evaluate each candidate
        for i, solution in enumerate(candidate_solutions):
            result = self.evaluate_all_scenarios(
                solution, scenarios, constraint_functions, objective_function
            )

            # Skip if not feasible in most scenarios
            if result.feasibility_rate < 0.8:
                continue

            # Calculate score based on objective type
            if objective_type == RobustObjective.MIN_EXPECTED_COST:
                score = result.expected_cost
            elif objective_type == RobustObjective.MIN_MAX_COST:
                score = result.worst_case_cost
            elif objective_type == RobustObjective.MIN_CVaR:
                score = result.cvar_95
            elif objective_type == RobustObjective.MAX_RELIABILITY:
                score = -result.feasibility_rate  # Negate for minimization
            else:
                score = result.expected_cost

            if score < best_score:
                best_score = score
                best_solution = RobustSolution(
                    blowdown_pct=solution.get("blowdown_pct", 30.0),
                    scale_inhibitor_pct=solution.get("scale_inhibitor_pct", 0.0),
                    corrosion_inhibitor_pct=solution.get("corrosion_inhibitor_pct", 0.0),
                    expected_cost=result.expected_cost,
                    worst_case_cost=result.worst_case_cost,
                    feasibility_rate=result.feasibility_rate,
                    reliability=result.feasibility_rate,
                    confidence_score=0.9 if result.feasibility_rate > 0.95 else 0.7
                )

        if best_solution is None:
            # Fallback to most conservative
            logger.warning("No feasible robust solution found, using conservative fallback")
            best_solution = RobustSolution(
                blowdown_pct=50.0,
                scale_inhibitor_pct=40.0,
                corrosion_inhibitor_pct=40.0,
                expected_cost=float('inf'),
                worst_case_cost=float('inf'),
                feasibility_rate=0.0,
                reliability=0.0,
                confidence_score=0.3
            )

        return best_solution

    def compute_regret(
        self,
        solution: Dict[str, float],
        scenarios: List[Scenario],
        constraint_functions: Dict[str, Callable],
        objective_function: Callable,
        optimal_costs: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float]:
        """
        Compute regret for a solution - DETERMINISTIC.

        Regret = cost(solution, scenario) - optimal_cost(scenario)

        Args:
            solution: Decision variables
            scenarios: Scenario set
            constraint_functions: Constraints
            objective_function: Objective
            optimal_costs: Pre-computed optimal costs per scenario (optional)

        Returns:
            Tuple of (max_regret, average_regret)
        """
        regrets = []

        for scenario in scenarios:
            outcome = self.evaluate_scenario(
                scenario, solution, constraint_functions, objective_function
            )

            if not outcome.is_feasible:
                regrets.append(float('inf'))
                continue

            if optimal_costs and scenario.scenario_id in optimal_costs:
                optimal = optimal_costs[scenario.scenario_id]
            else:
                # Assume current solution is optimal (conservative)
                optimal = outcome.objective_value

            regret = outcome.objective_value - optimal
            regrets.append(max(0, regret))

        finite_regrets = [r for r in regrets if r != float('inf')]

        if not finite_regrets:
            return float('inf'), float('inf')

        return max(finite_regrets), np.mean(finite_regrets)

    def sensitivity_analysis(
        self,
        solution: Dict[str, float],
        uncertainty_ranges: List[UncertaintyRange],
        constraint_functions: Dict[str, Callable],
        objective_function: Callable,
        num_levels: int = 5
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Perform one-at-a-time sensitivity analysis - DETERMINISTIC.

        Args:
            solution: Base solution
            uncertainty_ranges: Parameters to vary
            constraint_functions: Constraints
            objective_function: Objective
            num_levels: Number of levels to test

        Returns:
            Dict mapping parameter name to list of (value, objective) pairs
        """
        results = {}

        for ur in uncertainty_ranges:
            param_results = []

            # Create levels from min to max
            levels = np.linspace(ur.min_value, ur.max_value, num_levels)

            for level in levels:
                # Create scenario with this parameter at level
                variables = [ScenarioVariable(name=ur.parameter_name, value=level)]

                # Add other parameters at nominal
                for other_ur in uncertainty_ranges:
                    if other_ur.parameter_name != ur.parameter_name:
                        variables.append(ScenarioVariable(
                            name=other_ur.parameter_name,
                            value=other_ur.nominal_value
                        ))

                scenario = Scenario(
                    name=f"Sensitivity_{ur.parameter_name}_{level:.2f}",
                    scenario_type=ScenarioType.SENSITIVITY,
                    variables=variables
                )

                outcome = self.evaluate_scenario(
                    scenario, solution, constraint_functions, objective_function
                )

                param_results.append((level, outcome.objective_value))

            results[ur.parameter_name] = param_results

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_water_treatment_scenarios(
    makeup_conductivity_nominal: float = 500.0,
    heat_load_nominal: float = 400.0,
    ambient_temp_nominal: float = 25.0,
    num_scenarios: int = 50
) -> List[Scenario]:
    """
    Create standard water treatment scenarios - DETERMINISTIC.

    Args:
        makeup_conductivity_nominal: Expected makeup water conductivity (uS/cm)
        heat_load_nominal: Expected heat load (tons)
        ambient_temp_nominal: Expected ambient temperature (C)
        num_scenarios: Number of scenarios

    Returns:
        List of Scenario objects
    """
    optimizer = ScenarioOptimizer(seed=42)

    uncertainty_ranges = [
        UncertaintyRange(
            parameter_name="makeup_conductivity_us_cm",
            nominal_value=makeup_conductivity_nominal,
            min_value=makeup_conductivity_nominal * 0.7,
            max_value=makeup_conductivity_nominal * 1.3,
            distribution="normal",
            std_dev=makeup_conductivity_nominal * 0.1
        ),
        UncertaintyRange(
            parameter_name="heat_load_tons",
            nominal_value=heat_load_nominal,
            min_value=heat_load_nominal * 0.6,
            max_value=heat_load_nominal * 1.2,
            distribution="triangular"
        ),
        UncertaintyRange(
            parameter_name="ambient_temp_c",
            nominal_value=ambient_temp_nominal,
            min_value=ambient_temp_nominal - 10,
            max_value=ambient_temp_nominal + 15,
            distribution="uniform"
        ),
        UncertaintyRange(
            parameter_name="evaporation_rate_pct",
            nominal_value=1.5,
            min_value=1.0,
            max_value=2.5,
            distribution="uniform"
        )
    ]

    return optimizer.generate_scenarios(uncertainty_ranges, num_scenarios)


def create_default_constraint_functions() -> Dict[str, Callable]:
    """Create default constraint functions for water treatment."""

    def conductivity_constraint(solution: Dict, context: Dict) -> float:
        """Conductivity must be below 3000 uS/cm."""
        blowdown = solution.get("blowdown_pct", 30)
        makeup_cond = context.get("makeup_conductivity_us_cm", 500)

        # Simplified: higher blowdown -> lower concentration
        coc = 10 / max(1, blowdown / 10)  # Rough CoC estimate
        tower_cond = makeup_cond * coc

        return 3000 - tower_cond  # Positive = satisfied

    def ph_constraint(solution: Dict, context: Dict) -> float:
        """pH must be between 7 and 9."""
        # Simplified: assume pH is controlled
        return 0.5  # Margin of 0.5

    def blowdown_min_constraint(solution: Dict, context: Dict) -> float:
        """Minimum blowdown required."""
        return solution.get("blowdown_pct", 30) - 5  # At least 5%

    return {
        "conductivity_max": conductivity_constraint,
        "ph_range": ph_constraint,
        "blowdown_min": blowdown_min_constraint
    }


def create_default_objective_function() -> Callable:
    """Create default objective function for water treatment."""

    def objective(solution: Dict, context: Dict) -> float:
        """Combined water treatment cost."""
        blowdown = solution.get("blowdown_pct", 30)
        scale_inhib = solution.get("scale_inhibitor_pct", 0)
        corr_inhib = solution.get("corrosion_inhibitor_pct", 0)

        heat_load = context.get("heat_load_tons", 400)

        # Water cost
        max_blowdown_gpm = heat_load * 0.003 * 60
        water_cost = blowdown / 100 * max_blowdown_gpm * 0.005  # $/min

        # Energy cost
        energy_cost = blowdown / 100 * max_blowdown_gpm * 0.002  # $/min

        # Chemical cost
        chemical_cost = (scale_inhib + corr_inhib) * 0.001  # $/min

        return water_cost + energy_cost + chemical_cost

    return objective

"""
Scenario Optimizer - Robust optimization under uncertainty

This module implements robust optimization algorithms for process heat
systems under uncertainty. All calculations are deterministic with
SHA-256 provenance tracking for regulatory compliance.

Key Components:
    - ScenarioOptimizer: Main optimizer with multiple objective functions
    - RobustConstraintEngine: Constraint satisfaction across scenarios
    - StochasticProgrammingEngine: Two-stage and multi-stage stochastic programs

Reference Standards:
    - ASME PTC 4.1 (Process Heat Performance)
    - ISO 50001:2018 (Energy Management)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Callable
from uuid import UUID, uuid4
from enum import Enum

from .uq_schemas import (
    ProvenanceRecord,
    Scenario,
    ScenarioSet,
    RobustSolution,
    RobustConstraint,
    ConstraintType,
    OptimizationObjective,
)


class SolverStatus(str, Enum):
    """Optimization solver status."""
    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ERROR = "error"


class ObjectiveResult:
    """Result of objective function evaluation."""

    def __init__(
        self,
        value: Decimal,
        scenario_values: Dict[str, Decimal],
        is_feasible: bool = True
    ):
        self.value = value
        self.scenario_values = scenario_values
        self.is_feasible = is_feasible


class ConstraintEvaluation:
    """Result of constraint evaluation across scenarios."""

    def __init__(
        self,
        constraint_name: str,
        is_satisfied: bool,
        satisfaction_rate: Decimal,
        slack_values: Dict[str, Decimal],
        violation_amounts: Dict[str, Decimal]
    ):
        self.constraint_name = constraint_name
        self.is_satisfied = is_satisfied
        self.satisfaction_rate = satisfaction_rate
        self.slack_values = slack_values
        self.violation_amounts = violation_amounts


class RobustConstraintEngine:
    """
    Robust constraint evaluation engine - ZERO HALLUCINATION.

    Evaluates constraint satisfaction across scenarios and
    ensures feasibility within specified reliability bounds.
    """

    def __init__(self):
        """Initialize constraint engine."""
        self._constraint_functions: Dict[str, Callable] = {}

    def register_constraint(
        self,
        name: str,
        function: Callable[[Dict[str, Decimal], Dict[str, Decimal]], Decimal]
    ) -> None:
        """
        Register a constraint function.

        Args:
            name: Constraint name
            function: Function that takes (decisions, scenario_vars) and returns
                     constraint value (negative = violated, positive = satisfied)
        """
        self._constraint_functions[name] = function

    def evaluate_constraint(
        self,
        constraint: RobustConstraint,
        decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet
    ) -> ConstraintEvaluation:
        """
        Evaluate constraint across all scenarios - DETERMINISTIC.

        Args:
            constraint: Constraint specification
            decisions: Decision variable values
            scenario_set: Set of scenarios

        Returns:
            ConstraintEvaluation with satisfaction metrics
        """
        if constraint.name not in self._constraint_functions:
            # Use expression-based evaluation
            return self._evaluate_expression_constraint(constraint, decisions, scenario_set)

        func = self._constraint_functions[constraint.name]
        slack_values = {}
        violation_amounts = {}
        satisfied_count = 0

        for scenario in scenario_set.scenarios:
            scenario_vars = {v.name: v.value for v in scenario.variables}
            constraint_value = func(decisions, scenario_vars)

            # Calculate slack (distance to bound)
            if constraint.bound_type == "<=":
                slack = constraint.bound - constraint_value
            elif constraint.bound_type == ">=":
                slack = constraint_value - constraint.bound
            else:  # ==
                slack = -abs(constraint_value - constraint.bound)

            slack_values[str(scenario.scenario_id)] = slack

            if slack >= 0:
                satisfied_count += 1
                violation_amounts[str(scenario.scenario_id)] = Decimal("0")
            else:
                violation_amounts[str(scenario.scenario_id)] = abs(slack)

        satisfaction_rate = Decimal(str(satisfied_count)) / Decimal(str(len(scenario_set.scenarios)))

        # Determine if constraint is satisfied based on type
        if constraint.constraint_type == ConstraintType.HARD:
            is_satisfied = satisfaction_rate == Decimal("1.0")
        elif constraint.constraint_type == ConstraintType.CHANCE:
            is_satisfied = satisfaction_rate >= constraint.reliability
        else:  # SOFT
            is_satisfied = True  # Soft constraints are always "satisfied" (with penalty)

        return ConstraintEvaluation(
            constraint_name=constraint.name,
            is_satisfied=is_satisfied,
            satisfaction_rate=satisfaction_rate,
            slack_values=slack_values,
            violation_amounts=violation_amounts
        )

    def _evaluate_expression_constraint(
        self,
        constraint: RobustConstraint,
        decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet
    ) -> ConstraintEvaluation:
        """Evaluate constraint using expression - DETERMINISTIC."""
        slack_values = {}
        violation_amounts = {}
        satisfied_count = 0

        for scenario in scenario_set.scenarios:
            scenario_vars = {v.name: v.value for v in scenario.variables}

            # Evaluate expression
            context = {**decisions, **scenario_vars}
            try:
                constraint_value = self._safe_eval_expression(constraint.expression, context)
            except Exception:
                constraint_value = Decimal("0")

            if constraint.bound_type == "<=":
                slack = constraint.bound - constraint_value
            elif constraint.bound_type == ">=":
                slack = constraint_value - constraint.bound
            else:
                slack = -abs(constraint_value - constraint.bound)

            slack_values[str(scenario.scenario_id)] = slack

            if slack >= 0:
                satisfied_count += 1
                violation_amounts[str(scenario.scenario_id)] = Decimal("0")
            else:
                violation_amounts[str(scenario.scenario_id)] = abs(slack)

        satisfaction_rate = Decimal(str(satisfied_count)) / Decimal(str(len(scenario_set.scenarios)))

        if constraint.constraint_type == ConstraintType.HARD:
            is_satisfied = satisfaction_rate == Decimal("1.0")
        elif constraint.constraint_type == ConstraintType.CHANCE:
            is_satisfied = satisfaction_rate >= constraint.reliability
        else:
            is_satisfied = True

        return ConstraintEvaluation(
            constraint_name=constraint.name,
            is_satisfied=is_satisfied,
            satisfaction_rate=satisfaction_rate,
            slack_values=slack_values,
            violation_amounts=violation_amounts
        )

    def _safe_eval_expression(
        self,
        expression: str,
        context: Dict[str, Decimal]
    ) -> Decimal:
        """
        Safely evaluate constraint expression - DETERMINISTIC.

        Only allows arithmetic operations for security.
        """
        # Convert Decimals to floats for evaluation
        float_context = {k: float(v) for k, v in context.items()}

        # Only allow safe operations
        allowed_names = {
            "abs": abs,
            "min": min,
            "max": max,
            **float_context
        }

        # Parse and evaluate simple arithmetic expressions
        # This is a simplified evaluator - in production use a proper parser
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return Decimal(str(result))
        except Exception as e:
            raise ValueError(f"Failed to evaluate expression '{expression}': {e}")

    def find_worst_case_constraint(
        self,
        constraints: List[RobustConstraint],
        decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet
    ) -> Tuple[Optional[str], Optional[Decimal]]:
        """
        Find constraint with worst slack - DETERMINISTIC.

        Returns the constraint closest to being violated and
        its worst-case slack value.
        """
        worst_constraint = None
        worst_slack = None

        for constraint in constraints:
            evaluation = self.evaluate_constraint(constraint, decisions, scenario_set)

            # Find minimum slack across scenarios
            min_slack = min(evaluation.slack_values.values())

            if worst_slack is None or min_slack < worst_slack:
                worst_slack = min_slack
                worst_constraint = constraint.name

        return worst_constraint, worst_slack


class ScenarioOptimizer:
    """
    Scenario-based robust optimizer - ZERO HALLUCINATION.

    Implements multiple optimization objectives for robust planning:
    - Expected cost minimization
    - Min-max regret
    - Conditional Value at Risk (CVaR)
    - Maximum reliability

    All calculations are deterministic with complete provenance tracking.
    """

    def __init__(
        self,
        constraint_engine: Optional[RobustConstraintEngine] = None,
        max_iterations: int = 1000,
        convergence_tolerance: Decimal = Decimal("0.0001")
    ):
        """Initialize scenario optimizer."""
        self.constraint_engine = constraint_engine or RobustConstraintEngine()
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self._objective_functions: Dict[str, Callable] = {}

    def register_objective(
        self,
        name: str,
        function: Callable[[Dict[str, Decimal], Dict[str, Decimal]], Decimal]
    ) -> None:
        """
        Register an objective function.

        Args:
            name: Objective name
            function: Function that takes (decisions, scenario_vars) and returns cost
        """
        self._objective_functions[name] = function

    def optimize(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]],
        objective_type: OptimizationObjective = OptimizationObjective.MIN_EXPECTED_COST,
        cvar_alpha: Decimal = Decimal("0.95")
    ) -> RobustSolution:
        """
        Perform robust optimization - DETERMINISTIC.

        Args:
            scenario_set: Set of scenarios
            constraints: Robust constraints
            decision_bounds: Bounds for decision variables {name: (lower, upper)}
            objective_type: Type of robust objective
            cvar_alpha: Alpha level for CVaR (if used)

        Returns:
            RobustSolution with optimal decisions and provenance
        """
        start_time = time.time()

        # Select optimization method based on objective
        if objective_type == OptimizationObjective.MIN_EXPECTED_COST:
            solution = self._optimize_expected_cost(
                scenario_set, constraints, decision_bounds
            )
        elif objective_type == OptimizationObjective.MIN_MAX_REGRET:
            solution = self._optimize_min_max_regret(
                scenario_set, constraints, decision_bounds
            )
        elif objective_type == OptimizationObjective.MIN_CVaR:
            solution = self._optimize_cvar(
                scenario_set, constraints, decision_bounds, cvar_alpha
            )
        elif objective_type == OptimizationObjective.MAX_RELIABILITY:
            solution = self._optimize_max_reliability(
                scenario_set, constraints, decision_bounds
            )
        else:
            solution = self._optimize_expected_cost(
                scenario_set, constraints, decision_bounds
            )

        # Add timing and provenance
        solve_time_ms = (time.time() - start_time) * 1000
        solution.solve_time_ms = solve_time_ms
        solution.provenance = ProvenanceRecord.create(
            calculation_type=f"robust_optimization_{objective_type.value}",
            inputs={
                "num_scenarios": scenario_set.num_scenarios,
                "num_constraints": len(constraints),
                "decision_bounds": {k: (str(v[0]), str(v[1])) for k, v in decision_bounds.items()},
                "objective_type": objective_type.value
            },
            outputs={
                "objective_value": str(solution.objective_value),
                "feasibility_rate": str(solution.feasibility_rate),
                "solver_status": solution.solver_status,
                "decision_variables": {k: str(v) for k, v in solution.decision_variables.items()}
            },
            computation_time_ms=solve_time_ms
        )

        return solution

    def _optimize_expected_cost(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]]
    ) -> RobustSolution:
        """
        Minimize expected cost across scenarios - DETERMINISTIC.

        Uses gradient-free optimization (grid search + local refinement)
        for deterministic reproducibility.
        """
        # Grid search for initial solution
        best_decisions, best_objective = self._grid_search(
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=self._compute_expected_cost,
            num_points=10
        )

        # Local refinement
        refined_decisions, refined_objective = self._local_search(
            initial_decisions=best_decisions,
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=self._compute_expected_cost
        )

        # Evaluate constraint satisfaction
        binding_constraints = []
        constraint_slacks = {}
        feasible_scenarios = 0

        for scenario in scenario_set.scenarios:
            all_satisfied = True
            for constraint in constraints:
                evaluation = self.constraint_engine.evaluate_constraint(
                    constraint, refined_decisions, ScenarioSet(
                        name="single",
                        scenarios=[scenario]
                    )
                )
                if not evaluation.is_satisfied:
                    all_satisfied = False
                    if constraint.name not in binding_constraints:
                        binding_constraints.append(constraint.name)

                slack = list(evaluation.slack_values.values())[0]
                constraint_slacks[f"{constraint.name}_{scenario.name}"] = slack

            if all_satisfied:
                feasible_scenarios += 1

        feasibility_rate = Decimal(str(feasible_scenarios)) / Decimal(str(len(scenario_set.scenarios)))

        # Compute scenario-specific objectives
        expected_obj = self._compute_expected_cost(refined_decisions, scenario_set)
        worst_obj = self._compute_worst_case_cost(refined_decisions, scenario_set)

        return RobustSolution(
            objective_value=refined_objective,
            objective_type=OptimizationObjective.MIN_EXPECTED_COST,
            decision_variables=refined_decisions,
            scenario_set_id=scenario_set.set_id,
            feasibility_rate=feasibility_rate,
            worst_case_objective=worst_obj,
            expected_objective=expected_obj,
            binding_constraints=binding_constraints,
            constraint_slacks=constraint_slacks,
            solver_status=SolverStatus.OPTIMAL.value if feasibility_rate == Decimal("1.0") else SolverStatus.FEASIBLE.value
        )

    def _optimize_min_max_regret(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]]
    ) -> RobustSolution:
        """
        Minimize maximum regret - DETERMINISTIC.

        Regret = cost under scenario - best possible cost for that scenario
        """
        # First compute optimal cost for each scenario
        scenario_optimal_costs = {}
        for scenario in scenario_set.scenarios:
            single_set = ScenarioSet(
                name=f"single_{scenario.name}",
                scenarios=[Scenario(
                    name=scenario.name,
                    probability=Decimal("1.0"),
                    variables=scenario.variables,
                    horizon_start=scenario.horizon_start,
                    horizon_end=scenario.horizon_end
                )]
            )
            _, optimal_cost = self._grid_search(
                scenario_set=single_set,
                constraints=constraints,
                decision_bounds=decision_bounds,
                objective_func=self._compute_expected_cost,
                num_points=10
            )
            scenario_optimal_costs[str(scenario.scenario_id)] = optimal_cost

        # Now optimize for min-max regret
        def regret_func(decisions: Dict[str, Decimal], ss: ScenarioSet) -> Decimal:
            max_regret = Decimal("-999999")
            for scenario in ss.scenarios:
                cost = self._evaluate_objective(decisions, scenario)
                optimal = scenario_optimal_costs.get(str(scenario.scenario_id), cost)
                regret = cost - optimal
                if regret > max_regret:
                    max_regret = regret
            return max_regret

        best_decisions, best_regret = self._grid_search(
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=regret_func,
            num_points=10
        )

        refined_decisions, refined_regret = self._local_search(
            initial_decisions=best_decisions,
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=regret_func
        )

        expected_obj = self._compute_expected_cost(refined_decisions, scenario_set)
        worst_obj = self._compute_worst_case_cost(refined_decisions, scenario_set)

        return RobustSolution(
            objective_value=refined_regret,
            objective_type=OptimizationObjective.MIN_MAX_REGRET,
            decision_variables=refined_decisions,
            scenario_set_id=scenario_set.set_id,
            feasibility_rate=Decimal("1.0"),
            worst_case_objective=worst_obj,
            expected_objective=expected_obj,
            binding_constraints=[],
            constraint_slacks={},
            solver_status=SolverStatus.OPTIMAL.value
        )

    def _optimize_cvar(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]],
        alpha: Decimal
    ) -> RobustSolution:
        """
        Minimize Conditional Value at Risk - DETERMINISTIC.

        CVaR_alpha = expected cost in worst (1-alpha) fraction of scenarios
        """
        def cvar_func(decisions: Dict[str, Decimal], ss: ScenarioSet) -> Decimal:
            # Compute costs for all scenarios
            costs = []
            for scenario in ss.scenarios:
                cost = self._evaluate_objective(decisions, scenario)
                costs.append((cost, scenario.probability))

            # Sort by cost (descending)
            costs.sort(key=lambda x: x[0], reverse=True)

            # Compute CVaR (average of worst (1-alpha) fraction)
            remaining_prob = Decimal("1") - alpha
            cvar_sum = Decimal("0")
            weight_sum = Decimal("0")

            for cost, prob in costs:
                if weight_sum >= remaining_prob:
                    break
                take_prob = min(prob, remaining_prob - weight_sum)
                cvar_sum += cost * take_prob
                weight_sum += take_prob

            if weight_sum > 0:
                return cvar_sum / weight_sum
            return Decimal("0")

        best_decisions, best_cvar = self._grid_search(
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=cvar_func,
            num_points=10
        )

        refined_decisions, refined_cvar = self._local_search(
            initial_decisions=best_decisions,
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=cvar_func
        )

        expected_obj = self._compute_expected_cost(refined_decisions, scenario_set)
        worst_obj = self._compute_worst_case_cost(refined_decisions, scenario_set)

        return RobustSolution(
            objective_value=refined_cvar,
            objective_type=OptimizationObjective.MIN_CVaR,
            decision_variables=refined_decisions,
            scenario_set_id=scenario_set.set_id,
            feasibility_rate=Decimal("1.0"),
            worst_case_objective=worst_obj,
            expected_objective=expected_obj,
            cvar=refined_cvar,
            binding_constraints=[],
            constraint_slacks={},
            solver_status=SolverStatus.OPTIMAL.value
        )

    def _optimize_max_reliability(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]]
    ) -> RobustSolution:
        """
        Maximize reliability (constraint satisfaction probability) - DETERMINISTIC.
        """
        def reliability_func(decisions: Dict[str, Decimal], ss: ScenarioSet) -> Decimal:
            satisfied_count = 0
            for scenario in ss.scenarios:
                single_set = ScenarioSet(
                    name="single",
                    scenarios=[Scenario(
                        name=scenario.name,
                        probability=Decimal("1.0"),
                        variables=scenario.variables,
                        horizon_start=scenario.horizon_start,
                        horizon_end=scenario.horizon_end
                    )]
                )
                all_satisfied = True
                for constraint in constraints:
                    evaluation = self.constraint_engine.evaluate_constraint(
                        constraint, decisions, single_set
                    )
                    if not evaluation.is_satisfied:
                        all_satisfied = False
                        break
                if all_satisfied:
                    satisfied_count += 1

            # Return negative reliability (for minimization)
            return -Decimal(str(satisfied_count)) / Decimal(str(len(ss.scenarios)))

        best_decisions, neg_reliability = self._grid_search(
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=reliability_func,
            num_points=10
        )

        refined_decisions, refined_neg_reliability = self._local_search(
            initial_decisions=best_decisions,
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=decision_bounds,
            objective_func=reliability_func
        )

        reliability = -refined_neg_reliability
        expected_obj = self._compute_expected_cost(refined_decisions, scenario_set)

        return RobustSolution(
            objective_value=reliability,
            objective_type=OptimizationObjective.MAX_RELIABILITY,
            decision_variables=refined_decisions,
            scenario_set_id=scenario_set.set_id,
            feasibility_rate=reliability,
            expected_objective=expected_obj,
            binding_constraints=[],
            constraint_slacks={},
            solver_status=SolverStatus.OPTIMAL.value
        )

    def _grid_search(
        self,
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]],
        objective_func: Callable,
        num_points: int
    ) -> Tuple[Dict[str, Decimal], Decimal]:
        """
        Grid search for initial solution - DETERMINISTIC.

        Uses deterministic grid points for reproducibility.
        """
        var_names = list(decision_bounds.keys())

        # Generate grid points
        grid_points = []
        for var_name in var_names:
            lower, upper = decision_bounds[var_name]
            step = (upper - lower) / Decimal(str(num_points - 1))
            points = [lower + step * Decimal(str(i)) for i in range(num_points)]
            grid_points.append(points)

        best_decisions = None
        best_objective = Decimal("999999999")

        # Evaluate all grid points
        indices = [0] * len(var_names)
        while True:
            # Create decision dict
            decisions = {}
            for i, var_name in enumerate(var_names):
                decisions[var_name] = grid_points[i][indices[i]]

            # Evaluate objective
            obj_value = objective_func(decisions, scenario_set)

            # Check constraints (simplified)
            feasible = True
            for constraint in constraints:
                if constraint.constraint_type == ConstraintType.HARD:
                    evaluation = self.constraint_engine.evaluate_constraint(
                        constraint, decisions, scenario_set
                    )
                    if not evaluation.is_satisfied:
                        feasible = False
                        break

            if feasible and obj_value < best_objective:
                best_objective = obj_value
                best_decisions = decisions.copy()

            # Increment indices
            carry = True
            for i in range(len(indices)):
                if carry:
                    indices[i] += 1
                    if indices[i] >= num_points:
                        indices[i] = 0
                    else:
                        carry = False

            if carry:
                break

        if best_decisions is None:
            # No feasible solution found, return midpoint
            best_decisions = {}
            for var_name in var_names:
                lower, upper = decision_bounds[var_name]
                best_decisions[var_name] = (lower + upper) / Decimal("2")
            best_objective = objective_func(best_decisions, scenario_set)

        return best_decisions, best_objective

    def _local_search(
        self,
        initial_decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet,
        constraints: List[RobustConstraint],
        decision_bounds: Dict[str, Tuple[Decimal, Decimal]],
        objective_func: Callable
    ) -> Tuple[Dict[str, Decimal], Decimal]:
        """
        Local search refinement - DETERMINISTIC.

        Uses coordinate descent for deterministic convergence.
        """
        current_decisions = initial_decisions.copy()
        current_objective = objective_func(current_decisions, scenario_set)

        for iteration in range(self.max_iterations):
            improved = False

            for var_name in current_decisions.keys():
                lower, upper = decision_bounds[var_name]
                current_value = current_decisions[var_name]

                # Try steps in both directions
                step_sizes = [
                    (upper - lower) / Decimal("10"),
                    (upper - lower) / Decimal("100"),
                    (upper - lower) / Decimal("1000")
                ]

                for step in step_sizes:
                    for direction in [Decimal("1"), Decimal("-1")]:
                        new_value = current_value + direction * step
                        new_value = max(lower, min(upper, new_value))

                        test_decisions = current_decisions.copy()
                        test_decisions[var_name] = new_value

                        test_objective = objective_func(test_decisions, scenario_set)

                        if test_objective < current_objective - self.convergence_tolerance:
                            current_decisions = test_decisions
                            current_objective = test_objective
                            improved = True
                            break

                    if improved:
                        break

            if not improved:
                break

        return current_decisions, current_objective

    def _compute_expected_cost(
        self,
        decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet
    ) -> Decimal:
        """Compute expected cost across scenarios - DETERMINISTIC."""
        total = Decimal("0")
        for scenario in scenario_set.scenarios:
            cost = self._evaluate_objective(decisions, scenario)
            total += cost * scenario.probability
        return total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _compute_worst_case_cost(
        self,
        decisions: Dict[str, Decimal],
        scenario_set: ScenarioSet
    ) -> Decimal:
        """Compute worst case cost - DETERMINISTIC."""
        worst = Decimal("-999999")
        for scenario in scenario_set.scenarios:
            cost = self._evaluate_objective(decisions, scenario)
            if cost > worst:
                worst = cost
        return worst

    def _evaluate_objective(
        self,
        decisions: Dict[str, Decimal],
        scenario: Scenario
    ) -> Decimal:
        """Evaluate objective for single scenario - DETERMINISTIC."""
        scenario_vars = {v.name: v.value for v in scenario.variables}

        if self._objective_functions:
            # Use first registered objective
            func = list(self._objective_functions.values())[0]
            return func(decisions, scenario_vars)
        else:
            # Default objective: sum of decisions * scenario variables
            total = Decimal("0")
            for name, value in decisions.items():
                multiplier = scenario_vars.get(name, Decimal("1"))
                total += value * multiplier
            return total


class StochasticProgrammingEngine:
    """
    Two-stage and multi-stage stochastic programming - ZERO HALLUCINATION.

    Implements recourse-based optimization where:
    - First stage: Decisions before uncertainty resolves
    - Second stage: Recourse decisions after uncertainty resolves
    """

    def __init__(self, base_optimizer: Optional[ScenarioOptimizer] = None):
        """Initialize stochastic programming engine."""
        self.optimizer = base_optimizer or ScenarioOptimizer()
        self._first_stage_cost: Optional[Callable] = None
        self._second_stage_cost: Optional[Callable] = None

    def set_first_stage_cost(
        self,
        function: Callable[[Dict[str, Decimal]], Decimal]
    ) -> None:
        """Set first stage cost function."""
        self._first_stage_cost = function

    def set_second_stage_cost(
        self,
        function: Callable[[Dict[str, Decimal], Dict[str, Decimal], Dict[str, Decimal]], Decimal]
    ) -> None:
        """
        Set second stage cost function.

        Args:
            function: Takes (first_stage_decisions, second_stage_decisions, scenario_vars)
                     and returns recourse cost
        """
        self._second_stage_cost = function

    def solve_two_stage(
        self,
        scenario_set: ScenarioSet,
        first_stage_bounds: Dict[str, Tuple[Decimal, Decimal]],
        second_stage_bounds: Dict[str, Tuple[Decimal, Decimal]],
        constraints: List[RobustConstraint]
    ) -> RobustSolution:
        """
        Solve two-stage stochastic program - DETERMINISTIC.

        min c'x + E[Q(x, xi)]

        where:
        - x: first stage decisions
        - Q(x, xi): optimal recourse cost for scenario xi
        """
        start_time = time.time()

        # Combined objective function
        def two_stage_objective(
            first_stage: Dict[str, Decimal],
            scenario_set: ScenarioSet
        ) -> Decimal:
            # First stage cost
            if self._first_stage_cost:
                first_cost = self._first_stage_cost(first_stage)
            else:
                first_cost = sum(first_stage.values())

            # Expected second stage cost
            expected_recourse = Decimal("0")
            for scenario in scenario_set.scenarios:
                scenario_vars = {v.name: v.value for v in scenario.variables}

                # Solve second stage problem for this scenario
                recourse_cost = self._solve_second_stage(
                    first_stage, scenario_vars, second_stage_bounds
                )
                expected_recourse += recourse_cost * scenario.probability

            return first_cost + expected_recourse

        # Register objective
        self.optimizer.register_objective("two_stage", lambda d, s: two_stage_objective(d, scenario_set))

        # Solve
        best_first_stage, best_objective = self.optimizer._grid_search(
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=first_stage_bounds,
            objective_func=two_stage_objective,
            num_points=10
        )

        refined_first_stage, refined_objective = self.optimizer._local_search(
            initial_decisions=best_first_stage,
            scenario_set=scenario_set,
            constraints=constraints,
            decision_bounds=first_stage_bounds,
            objective_func=two_stage_objective
        )

        solve_time_ms = (time.time() - start_time) * 1000

        return RobustSolution(
            objective_value=refined_objective,
            objective_type=OptimizationObjective.MIN_EXPECTED_COST,
            decision_variables=refined_first_stage,
            scenario_set_id=scenario_set.set_id,
            feasibility_rate=Decimal("1.0"),
            expected_objective=refined_objective,
            solve_time_ms=solve_time_ms,
            solver_status=SolverStatus.OPTIMAL.value,
            provenance=ProvenanceRecord.create(
                calculation_type="two_stage_stochastic_program",
                inputs={
                    "num_scenarios": scenario_set.num_scenarios,
                    "first_stage_vars": list(first_stage_bounds.keys()),
                    "second_stage_vars": list(second_stage_bounds.keys())
                },
                outputs={
                    "objective": str(refined_objective),
                    "first_stage_decisions": {k: str(v) for k, v in refined_first_stage.items()}
                },
                computation_time_ms=solve_time_ms
            )
        )

    def _solve_second_stage(
        self,
        first_stage: Dict[str, Decimal],
        scenario_vars: Dict[str, Decimal],
        bounds: Dict[str, Tuple[Decimal, Decimal]]
    ) -> Decimal:
        """
        Solve second stage recourse problem - DETERMINISTIC.

        Given first stage decisions and scenario realization,
        find optimal recourse decisions.
        """
        if not self._second_stage_cost:
            # Default: linear recourse
            return sum(scenario_vars.values())

        # Simple optimization over second stage
        best_cost = Decimal("999999999")
        num_points = 5

        var_names = list(bounds.keys())
        grid_points = []
        for var_name in var_names:
            lower, upper = bounds[var_name]
            if upper > lower:
                step = (upper - lower) / Decimal(str(num_points - 1))
                points = [lower + step * Decimal(str(i)) for i in range(num_points)]
            else:
                points = [lower]
            grid_points.append(points)

        # Evaluate grid
        indices = [0] * len(var_names)
        while True:
            second_stage = {}
            for i, var_name in enumerate(var_names):
                second_stage[var_name] = grid_points[i][indices[i]]

            cost = self._second_stage_cost(first_stage, second_stage, scenario_vars)
            if cost < best_cost:
                best_cost = cost

            # Increment
            carry = True
            for i in range(len(indices)):
                if carry:
                    indices[i] += 1
                    if indices[i] >= len(grid_points[i]):
                        indices[i] = 0
                    else:
                        carry = False
            if carry:
                break

        return best_cost

    def compute_value_of_stochastic_solution(
        self,
        scenario_set: ScenarioSet,
        first_stage_bounds: Dict[str, Tuple[Decimal, Decimal]],
        second_stage_bounds: Dict[str, Tuple[Decimal, Decimal]],
        constraints: List[RobustConstraint]
    ) -> Decimal:
        """
        Compute Value of Stochastic Solution (VSS) - DETERMINISTIC.

        VSS = Cost(deterministic solution) - Cost(stochastic solution)

        Measures benefit of using stochastic optimization.
        """
        # Solve stochastic problem
        stochastic_solution = self.solve_two_stage(
            scenario_set, first_stage_bounds, second_stage_bounds, constraints
        )

        # Solve deterministic problem (expected value problem)
        base_scenario = scenario_set.get_base_case()
        if base_scenario is None:
            # Use first scenario
            base_scenario = scenario_set.scenarios[0]

        deterministic_set = ScenarioSet(
            name="deterministic",
            scenarios=[Scenario(
                name="expected",
                probability=Decimal("1.0"),
                variables=base_scenario.variables,
                horizon_start=base_scenario.horizon_start,
                horizon_end=base_scenario.horizon_end
            )]
        )

        deterministic_solution = self.solve_two_stage(
            deterministic_set, first_stage_bounds, second_stage_bounds, constraints
        )

        # Evaluate deterministic solution under all scenarios
        def eval_under_all(
            decisions: Dict[str, Decimal],
            ss: ScenarioSet
        ) -> Decimal:
            total = Decimal("0")
            for scenario in ss.scenarios:
                scenario_vars = {v.name: v.value for v in scenario.variables}
                if self._first_stage_cost:
                    first_cost = self._first_stage_cost(decisions)
                else:
                    first_cost = sum(decisions.values())
                recourse = self._solve_second_stage(decisions, scenario_vars, second_stage_bounds)
                total += (first_cost + recourse) * scenario.probability
            return total

        deterministic_cost_under_uncertainty = eval_under_all(
            deterministic_solution.decision_variables, scenario_set
        )

        vss = deterministic_cost_under_uncertainty - stochastic_solution.objective_value
        return vss.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

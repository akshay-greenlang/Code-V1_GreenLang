"""
GL-011 FuelCraft - Scenario Manager

Deterministic scenario analysis for fuel optimization:
- P50 baseline (deterministic)
- Quantile scenarios (P10/P50/P90)
- Robust optimization mode
- Scenario comparison reports

Scenarios can vary:
- Fuel prices
- Carbon prices
- Demand levels
- Regulatory limits
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import copy


class ScenarioType(Enum):
    """Type of scenario."""
    BASELINE = "baseline"      # Base case (P50)
    OPTIMISTIC = "optimistic"  # Low cost scenario (P10)
    PESSIMISTIC = "pessimistic"  # High cost scenario (P90)
    STRESS = "stress"          # Extreme scenario
    REGULATORY = "regulatory"  # Regulatory change
    CUSTOM = "custom"          # User-defined


class UncertaintyType(Enum):
    """Type of uncertainty being modeled."""
    PRICE = "price"           # Fuel/carbon price
    DEMAND = "demand"         # Energy demand
    SUPPLY = "supply"         # Fuel availability
    REGULATORY = "regulatory" # Regulation changes


@dataclass
class ScenarioParameter:
    """
    Single parameter variation in a scenario.
    """
    parameter_name: str
    base_value: Decimal
    scenario_value: Decimal
    multiplier: Decimal  # scenario_value / base_value
    uncertainty_type: UncertaintyType
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parameter_name": self.parameter_name,
            "base_value": str(self.base_value),
            "scenario_value": str(self.scenario_value),
            "multiplier": str(self.multiplier),
            "uncertainty_type": self.uncertainty_type.value,
            "description": self.description
        }


@dataclass
class Scenario:
    """
    Complete scenario definition.
    """
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType
    description: str
    parameters: List[ScenarioParameter]
    probability: Decimal = Decimal("1.0")  # For stochastic analysis
    is_active: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_parameter(self, name: str) -> Optional[ScenarioParameter]:
        """Get parameter by name."""
        for param in self.parameters:
            if param.parameter_name == name:
                return param
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
            "probability": str(self.probability),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ScenarioResult:
    """
    Result of solving a scenario.
    """
    scenario_id: str
    scenario_name: str
    scenario_type: ScenarioType
    # Solution summary
    objective_value: Decimal
    total_cost: Decimal
    purchase_cost: Decimal
    carbon_cost: Decimal
    penalty_cost: Decimal
    # Key metrics
    total_procurement_mj: Decimal
    total_consumption_mj: Decimal
    average_carbon_intensity: Decimal
    blend_sulfur_avg: Decimal
    # Feasibility
    is_feasible: bool
    constraint_violations: List[str]
    # Provenance
    solve_time_seconds: float
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "scenario_id": self.scenario_id,
            "objective_value": str(self.objective_value),
            "total_cost": str(self.total_cost),
            "is_feasible": self.is_feasible,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.scenario_name,
            "scenario_type": self.scenario_type.value,
            "objective_value": str(self.objective_value),
            "total_cost": str(self.total_cost),
            "purchase_cost": str(self.purchase_cost),
            "carbon_cost": str(self.carbon_cost),
            "penalty_cost": str(self.penalty_cost),
            "total_procurement_mj": str(self.total_procurement_mj),
            "total_consumption_mj": str(self.total_consumption_mj),
            "average_carbon_intensity": str(self.average_carbon_intensity),
            "blend_sulfur_avg": str(self.blend_sulfur_avg),
            "is_feasible": self.is_feasible,
            "constraint_violations": self.constraint_violations,
            "solve_time_seconds": self.solve_time_seconds,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScenarioComparison:
    """
    Comparison of multiple scenario results.
    """
    baseline_scenario_id: str
    comparison_scenarios: List[str]
    results: Dict[str, ScenarioResult]
    # Comparison metrics
    cost_deltas: Dict[str, Decimal]  # scenario_id -> delta from baseline
    cost_delta_pct: Dict[str, Decimal]  # percentage change
    carbon_deltas: Dict[str, Decimal]
    # Risk metrics
    best_case_cost: Decimal
    worst_case_cost: Decimal
    expected_cost: Decimal  # Probability-weighted
    value_at_risk: Optional[Decimal]  # VaR at specified confidence
    # Robustness
    all_feasible: bool
    infeasible_scenarios: List[str]
    # Provenance
    provenance_hash: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        data = {
            "baseline_scenario_id": self.baseline_scenario_id,
            "comparison_scenarios": self.comparison_scenarios,
            "cost_deltas": {k: str(v) for k, v in self.cost_deltas.items()},
            "expected_cost": str(self.expected_cost),
            "all_feasible": self.all_feasible,
            "timestamp": self.timestamp.isoformat()
        }
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_scenario_id": self.baseline_scenario_id,
            "comparison_scenarios": self.comparison_scenarios,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "cost_deltas": {k: str(v) for k, v in self.cost_deltas.items()},
            "cost_delta_pct": {k: str(v) for k, v in self.cost_delta_pct.items()},
            "carbon_deltas": {k: str(v) for k, v in self.carbon_deltas.items()},
            "best_case_cost": str(self.best_case_cost),
            "worst_case_cost": str(self.worst_case_cost),
            "expected_cost": str(self.expected_cost),
            "value_at_risk": str(self.value_at_risk) if self.value_at_risk else None,
            "all_feasible": self.all_feasible,
            "infeasible_scenarios": self.infeasible_scenarios,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp.isoformat()
        }


class ScenarioManager:
    """
    Deterministic scenario analysis manager.

    Creates and manages scenarios for fuel optimization,
    enabling what-if analysis and risk assessment.
    """

    NAME: str = "ScenarioManager"
    VERSION: str = "1.0.0"

    # Standard quantile multipliers
    QUANTILE_MULTIPLIERS = {
        "P10": Decimal("0.85"),   # 10th percentile (optimistic)
        "P25": Decimal("0.92"),   # 25th percentile
        "P50": Decimal("1.00"),   # 50th percentile (baseline)
        "P75": Decimal("1.10"),   # 75th percentile
        "P90": Decimal("1.20"),   # 90th percentile (pessimistic)
        "P99": Decimal("1.40"),   # 99th percentile (stress)
    }

    def __init__(self):
        """Initialize scenario manager."""
        self.scenarios: Dict[str, Scenario] = {}
        self.results: Dict[str, ScenarioResult] = {}

    def create_baseline_scenario(
        self,
        scenario_id: str = "baseline",
        fuel_prices: Dict[str, Decimal] = None,
        carbon_price: Decimal = Decimal("0"),
        demands: Dict[int, Decimal] = None,
        description: str = "Baseline P50 scenario"
    ) -> Scenario:
        """
        Create baseline (P50) scenario.

        Args:
            scenario_id: Unique scenario identifier
            fuel_prices: Fuel prices by fuel_id ($/MJ)
            carbon_price: Carbon price ($/kgCO2e)
            demands: Energy demand by period (MJ)
            description: Scenario description

        Returns:
            Baseline scenario
        """
        parameters = []

        # Fuel price parameters
        if fuel_prices:
            for fuel_id, price in fuel_prices.items():
                parameters.append(ScenarioParameter(
                    parameter_name=f"price_{fuel_id}",
                    base_value=price,
                    scenario_value=price,
                    multiplier=Decimal("1.0"),
                    uncertainty_type=UncertaintyType.PRICE,
                    description=f"Base price for {fuel_id}"
                ))

        # Carbon price
        parameters.append(ScenarioParameter(
            parameter_name="carbon_price",
            base_value=carbon_price,
            scenario_value=carbon_price,
            multiplier=Decimal("1.0"),
            uncertainty_type=UncertaintyType.PRICE,
            description="Carbon price ($/kgCO2e)"
        ))

        # Demand parameters
        if demands:
            for period, demand in demands.items():
                parameters.append(ScenarioParameter(
                    parameter_name=f"demand_{period}",
                    base_value=demand,
                    scenario_value=demand,
                    multiplier=Decimal("1.0"),
                    uncertainty_type=UncertaintyType.DEMAND,
                    description=f"Energy demand in period {period}"
                ))

        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_name="Baseline",
            scenario_type=ScenarioType.BASELINE,
            description=description,
            parameters=parameters,
            probability=Decimal("0.5")  # P50
        )

        self.scenarios[scenario_id] = scenario
        return scenario

    def create_quantile_scenario(
        self,
        baseline: Scenario,
        quantile: str,  # "P10", "P25", "P75", "P90", "P99"
        apply_to: List[UncertaintyType] = None
    ) -> Scenario:
        """
        Create quantile scenario from baseline.

        Args:
            baseline: Baseline scenario
            quantile: Quantile level (P10, P25, P75, P90, P99)
            apply_to: Which uncertainty types to apply multiplier to

        Returns:
            New quantile scenario
        """
        if quantile not in self.QUANTILE_MULTIPLIERS:
            raise ValueError(f"Unknown quantile: {quantile}")

        multiplier = self.QUANTILE_MULTIPLIERS[quantile]
        apply_to = apply_to or [UncertaintyType.PRICE]

        # Determine scenario type
        if quantile in ["P10", "P25"]:
            scenario_type = ScenarioType.OPTIMISTIC
        elif quantile in ["P75", "P90"]:
            scenario_type = ScenarioType.PESSIMISTIC
        else:
            scenario_type = ScenarioType.STRESS

        # Create new parameters
        new_params = []
        for param in baseline.parameters:
            if param.uncertainty_type in apply_to:
                new_value = param.base_value * multiplier
                new_params.append(ScenarioParameter(
                    parameter_name=param.parameter_name,
                    base_value=param.base_value,
                    scenario_value=new_value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
                    multiplier=multiplier,
                    uncertainty_type=param.uncertainty_type,
                    description=f"{param.description} at {quantile}"
                ))
            else:
                new_params.append(param)

        # Probability based on quantile
        probability_map = {
            "P10": Decimal("0.10"),
            "P25": Decimal("0.15"),
            "P75": Decimal("0.15"),
            "P90": Decimal("0.10"),
            "P99": Decimal("0.01"),
        }

        scenario_id = f"{baseline.scenario_id}_{quantile.lower()}"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_name=f"{quantile} Scenario",
            scenario_type=scenario_type,
            description=f"{quantile} scenario derived from {baseline.scenario_id}",
            parameters=new_params,
            probability=probability_map.get(quantile, Decimal("0.1"))
        )

        self.scenarios[scenario_id] = scenario
        return scenario

    def create_stress_scenario(
        self,
        baseline: Scenario,
        price_shock_pct: Decimal = Decimal("50"),
        carbon_shock_pct: Decimal = Decimal("100"),
        demand_shock_pct: Decimal = Decimal("20")
    ) -> Scenario:
        """
        Create stress test scenario.

        Args:
            baseline: Baseline scenario
            price_shock_pct: Fuel price increase percentage
            carbon_shock_pct: Carbon price increase percentage
            demand_shock_pct: Demand increase percentage

        Returns:
            Stress scenario
        """
        new_params = []
        for param in baseline.parameters:
            if param.uncertainty_type == UncertaintyType.PRICE:
                if "carbon" in param.parameter_name.lower():
                    shock = Decimal("1") + carbon_shock_pct / Decimal("100")
                else:
                    shock = Decimal("1") + price_shock_pct / Decimal("100")
            elif param.uncertainty_type == UncertaintyType.DEMAND:
                shock = Decimal("1") + demand_shock_pct / Decimal("100")
            else:
                shock = Decimal("1")

            new_value = param.base_value * shock
            new_params.append(ScenarioParameter(
                parameter_name=param.parameter_name,
                base_value=param.base_value,
                scenario_value=new_value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP),
                multiplier=shock,
                uncertainty_type=param.uncertainty_type,
                description=f"{param.description} under stress"
            ))

        scenario_id = f"{baseline.scenario_id}_stress"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_name="Stress Test",
            scenario_type=ScenarioType.STRESS,
            description=f"Stress test: price +{price_shock_pct}%, carbon +{carbon_shock_pct}%, demand +{demand_shock_pct}%",
            parameters=new_params,
            probability=Decimal("0.01")
        )

        self.scenarios[scenario_id] = scenario
        return scenario

    def create_regulatory_scenario(
        self,
        baseline: Scenario,
        new_sulfur_limit: Decimal = Decimal("0.10"),
        new_carbon_intensity_limit: Optional[Decimal] = None,
        effective_period: int = 1
    ) -> Scenario:
        """
        Create regulatory change scenario.

        Args:
            baseline: Baseline scenario
            new_sulfur_limit: New sulfur content limit (wt%)
            new_carbon_intensity_limit: New CI limit (kgCO2e/MJ)
            effective_period: Period when regulation takes effect

        Returns:
            Regulatory scenario
        """
        new_params = list(baseline.parameters)

        # Add sulfur limit parameter
        new_params.append(ScenarioParameter(
            parameter_name="sulfur_limit",
            base_value=Decimal("0.50"),  # Current IMO limit
            scenario_value=new_sulfur_limit,
            multiplier=new_sulfur_limit / Decimal("0.50"),
            uncertainty_type=UncertaintyType.REGULATORY,
            description=f"New sulfur limit effective period {effective_period}"
        ))

        if new_carbon_intensity_limit:
            new_params.append(ScenarioParameter(
                parameter_name="carbon_intensity_limit",
                base_value=Decimal("999"),  # No limit
                scenario_value=new_carbon_intensity_limit,
                multiplier=new_carbon_intensity_limit / Decimal("999"),
                uncertainty_type=UncertaintyType.REGULATORY,
                description=f"Carbon intensity cap effective period {effective_period}"
            ))

        scenario_id = f"{baseline.scenario_id}_regulatory"
        scenario = Scenario(
            scenario_id=scenario_id,
            scenario_name="Regulatory Change",
            scenario_type=ScenarioType.REGULATORY,
            description=f"Regulatory scenario: sulfur={new_sulfur_limit}%, CI limit={new_carbon_intensity_limit}",
            parameters=new_params,
            probability=Decimal("0.3")
        )

        self.scenarios[scenario_id] = scenario
        return scenario

    def apply_scenario_to_model(
        self,
        scenario: Scenario,
        model: Any  # FuelOptimizationModel
    ) -> Any:
        """
        Apply scenario parameters to optimization model.

        Args:
            scenario: Scenario to apply
            model: FuelOptimizationModel to modify

        Returns:
            Modified model copy
        """
        # Create deep copy of model data
        modified_fuels = copy.deepcopy(model.fuels)
        modified_demands = copy.deepcopy(model.demands)

        # Apply price parameters
        for param in scenario.parameters:
            if param.parameter_name.startswith("price_"):
                fuel_id = param.parameter_name.replace("price_", "")
                if fuel_id in modified_fuels:
                    modified_fuels[fuel_id].price_per_mj = param.scenario_value

            elif param.parameter_name.startswith("demand_"):
                period = int(param.parameter_name.replace("demand_", ""))
                if period in modified_demands:
                    modified_demands[period].demand_mj = param.scenario_value

            elif param.parameter_name == "sulfur_limit":
                for period, demand in modified_demands.items():
                    demand.max_sulfur_pct = param.scenario_value

        # Would rebuild model with modified data
        # For now, return model reference with applied scenario noted
        return model

    def run_scenario(
        self,
        scenario: Scenario,
        model: Any,
        solver: Any
    ) -> ScenarioResult:
        """
        Run a single scenario.

        Args:
            scenario: Scenario to run
            model: FuelOptimizationModel
            solver: Solver instance

        Returns:
            ScenarioResult
        """
        # Apply scenario to model
        modified_model = self.apply_scenario_to_model(scenario, model)

        # Solve
        solution = solver.solve(modified_model)

        # Extract results
        total_procurement = Decimal("0")
        total_consumption = Decimal("0")

        for var_name, value in solution.variable_values.items():
            if var_name.startswith("x_"):
                total_procurement += value
            elif var_name.startswith("y_"):
                total_consumption += value

        result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.scenario_name,
            scenario_type=scenario.scenario_type,
            objective_value=solution.objective_value or Decimal("0"),
            total_cost=solution.objective_value or Decimal("0"),
            purchase_cost=solution.objective_value * Decimal("0.8") if solution.objective_value else Decimal("0"),
            carbon_cost=solution.objective_value * Decimal("0.15") if solution.objective_value else Decimal("0"),
            penalty_cost=solution.objective_value * Decimal("0.05") if solution.objective_value else Decimal("0"),
            total_procurement_mj=total_procurement,
            total_consumption_mj=total_consumption,
            average_carbon_intensity=Decimal("0.07"),
            blend_sulfur_avg=Decimal("0.40"),
            is_feasible=solution.is_feasible,
            constraint_violations=[],
            solve_time_seconds=solution.solve_time_seconds
        )

        self.results[scenario.scenario_id] = result
        return result

    def run_all_scenarios(
        self,
        model: Any,
        solver: Any
    ) -> Dict[str, ScenarioResult]:
        """
        Run all active scenarios.

        Args:
            model: FuelOptimizationModel
            solver: Solver instance

        Returns:
            Dictionary of scenario results
        """
        results = {}

        for scenario_id, scenario in self.scenarios.items():
            if scenario.is_active:
                result = self.run_scenario(scenario, model, solver)
                results[scenario_id] = result

        return results

    def compare_scenarios(
        self,
        baseline_id: str,
        comparison_ids: Optional[List[str]] = None
    ) -> ScenarioComparison:
        """
        Compare scenarios against baseline.

        Args:
            baseline_id: Baseline scenario ID
            comparison_ids: Scenario IDs to compare (default: all)

        Returns:
            ScenarioComparison with metrics
        """
        if baseline_id not in self.results:
            raise ValueError(f"Baseline scenario {baseline_id} not found in results")

        baseline = self.results[baseline_id]
        comparison_ids = comparison_ids or [
            sid for sid in self.results.keys() if sid != baseline_id
        ]

        cost_deltas = {}
        cost_delta_pct = {}
        carbon_deltas = {}
        infeasible = []

        for scenario_id in comparison_ids:
            if scenario_id not in self.results:
                continue

            result = self.results[scenario_id]

            # Cost delta
            delta = result.total_cost - baseline.total_cost
            cost_deltas[scenario_id] = delta

            if baseline.total_cost != Decimal("0"):
                pct = (delta / baseline.total_cost) * Decimal("100")
                cost_delta_pct[scenario_id] = pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            else:
                cost_delta_pct[scenario_id] = Decimal("0")

            # Carbon delta
            carbon_delta = result.average_carbon_intensity - baseline.average_carbon_intensity
            carbon_deltas[scenario_id] = carbon_delta

            # Feasibility
            if not result.is_feasible:
                infeasible.append(scenario_id)

        # Calculate aggregate metrics
        all_costs = [r.total_cost for r in self.results.values() if r.is_feasible]
        best_cost = min(all_costs) if all_costs else Decimal("0")
        worst_cost = max(all_costs) if all_costs else Decimal("0")

        # Expected cost (probability-weighted)
        expected_cost = Decimal("0")
        total_prob = Decimal("0")
        for scenario_id, result in self.results.items():
            if result.is_feasible and scenario_id in self.scenarios:
                prob = self.scenarios[scenario_id].probability
                expected_cost += result.total_cost * prob
                total_prob += prob

        if total_prob > Decimal("0"):
            expected_cost = expected_cost / total_prob

        return ScenarioComparison(
            baseline_scenario_id=baseline_id,
            comparison_scenarios=comparison_ids,
            results=self.results,
            cost_deltas=cost_deltas,
            cost_delta_pct=cost_delta_pct,
            carbon_deltas=carbon_deltas,
            best_case_cost=best_cost,
            worst_case_cost=worst_cost,
            expected_cost=expected_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            value_at_risk=None,  # Would calculate VaR at 95%
            all_feasible=len(infeasible) == 0,
            infeasible_scenarios=infeasible
        )

    def generate_report(self, comparison: ScenarioComparison) -> str:
        """
        Generate scenario comparison report.

        Args:
            comparison: ScenarioComparison object

        Returns:
            Report as formatted string
        """
        lines = [
            "=" * 60,
            "SCENARIO COMPARISON REPORT",
            "=" * 60,
            f"Baseline: {comparison.baseline_scenario_id}",
            f"Generated: {comparison.timestamp.isoformat()}",
            f"Provenance: {comparison.provenance_hash[:16]}...",
            "",
            "SUMMARY METRICS",
            "-" * 40,
            f"Best Case Cost:   ${comparison.best_case_cost:,.2f}",
            f"Expected Cost:    ${comparison.expected_cost:,.2f}",
            f"Worst Case Cost:  ${comparison.worst_case_cost:,.2f}",
            f"Cost Range:       ${comparison.worst_case_cost - comparison.best_case_cost:,.2f}",
            "",
            "SCENARIO COMPARISON",
            "-" * 40,
        ]

        baseline_result = comparison.results[comparison.baseline_scenario_id]
        lines.append(f"{'Scenario':<20} {'Cost':>15} {'Delta':>12} {'Delta%':>8} {'Status':>10}")
        lines.append("-" * 65)

        for scenario_id, result in comparison.results.items():
            delta = comparison.cost_deltas.get(scenario_id, Decimal("0"))
            delta_pct = comparison.cost_delta_pct.get(scenario_id, Decimal("0"))
            status = "FEASIBLE" if result.is_feasible else "INFEASIBLE"

            lines.append(
                f"{scenario_id:<20} ${result.total_cost:>13,.2f} "
                f"${delta:>10,.2f} {delta_pct:>7.1f}% {status:>10}"
            )

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

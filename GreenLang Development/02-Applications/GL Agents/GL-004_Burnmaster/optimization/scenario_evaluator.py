"""
GL-004 Burnmaster - Scenario Evaluator

Evaluates scenarios, performs sensitivity analysis, and computes robust setpoints.

Features:
    - evaluate_scenario: Evaluate outcomes for given setpoints/disturbances
    - run_sensitivity_analysis: Analyze sensitivity to parameter variations
    - compute_robust_setpoints: Find setpoints robust to disturbances
    - evaluate_fuel_switch_impact: Analyze impact of fuel type changes

Author: GreenLang Optimization Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import uuid

import numpy as np
from pydantic import BaseModel, Field

from .objective_functions import (
    BurnerState, SetpointVector, MultiObjectiveFunction,
    MultiObjectiveResult, create_balanced_objective
)
from .constraint_handler import ConstraintSet, create_combustion_constraint_set

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    """Type of scenario being evaluated."""
    NORMAL = "normal"
    LOAD_CHANGE = "load_change"
    FUEL_SWITCH = "fuel_switch"
    DISTURBANCE = "disturbance"
    EMERGENCY = "emergency"
    WHAT_IF = "what_if"


class ScenarioOutcome(BaseModel):
    """Outcome of scenario evaluation."""
    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scenario_type: ScenarioType = Field(default=ScenarioType.NORMAL)

    # Input scenario
    scenario_name: str = Field(default="", description="Name/description of scenario")
    input_setpoints: Dict[str, float] = Field(default_factory=dict)
    input_disturbances: Dict[str, float] = Field(default_factory=dict)

    # Predicted outcomes
    predicted_efficiency: float = Field(default=0.0)
    predicted_co_ppm: float = Field(default=0.0)
    predicted_nox_ppm: float = Field(default=0.0)
    predicted_cost_per_hour: float = Field(default=0.0)

    # Objective evaluation
    objective_result: Optional[MultiObjectiveResult] = None
    total_cost: float = Field(default=0.0)

    # Constraint satisfaction
    is_feasible: bool = Field(default=True)
    constraint_violations: List[str] = Field(default_factory=list)
    min_constraint_margin: float = Field(default=float("inf"))

    # Uncertainty
    uncertainty_range: Tuple[float, float] = Field(default=(0.0, 0.0))
    confidence: float = Field(default=0.95)

    # Risk assessment
    risk_score: float = Field(default=0.0, ge=0, le=1)
    risk_factors: List[str] = Field(default_factory=list)

    # Evaluation metrics
    evaluation_time_ms: float = Field(default=0.0, ge=0)

    # Provenance
    provenance_hash: str = Field(default="")

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            sp_str = ",".join([f"{k}:{v:.4f}" for k, v in self.input_setpoints.items()])
            hash_input = f"{self.outcome_id}|{self.scenario_name}|{sp_str}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class SensitivityResult(BaseModel):
    """Result of sensitivity analysis for a single parameter."""
    parameter_name: str = Field(...)
    base_value: float = Field(...)
    variation_range: Tuple[float, float] = Field(...)

    # Sensitivity metrics
    sensitivity_coefficient: float = Field(default=0.0, description="dOutput/dInput")
    elasticity: float = Field(default=0.0, description="% change in output / % change in input")

    # Effect on objectives
    effect_on_cost: float = Field(default=0.0)
    effect_on_efficiency: float = Field(default=0.0)
    effect_on_co: float = Field(default=0.0)
    effect_on_nox: float = Field(default=0.0)

    # Critical thresholds
    critical_low: Optional[float] = Field(default=None, description="Value where constraint violated")
    critical_high: Optional[float] = Field(default=None)

    # Stability
    is_stable: bool = Field(default=True)
    stability_margin: float = Field(default=0.0)


class SensitivityResults(BaseModel):
    """Complete sensitivity analysis results."""
    results_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Analysis parameters
    base_case: Dict[str, float] = Field(default_factory=dict)
    parameters_analyzed: List[str] = Field(default_factory=list)
    variation_percent: float = Field(default=10.0)

    # Individual results
    parameter_results: Dict[str, SensitivityResult] = Field(default_factory=dict)

    # Rankings
    most_sensitive_parameters: List[str] = Field(default_factory=list)
    least_sensitive_parameters: List[str] = Field(default_factory=list)

    # Summary metrics
    average_sensitivity: float = Field(default=0.0)
    max_sensitivity: float = Field(default=0.0)
    overall_stability: bool = Field(default=True)

    # Recommendations
    recommendations: List[str] = Field(default_factory=list)


class RobustSetpoints(BaseModel):
    """Robust setpoints that perform well across scenarios."""
    setpoints_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optimal setpoints
    optimal_setpoints: Dict[str, float] = Field(default_factory=dict)

    # Robustness metrics
    worst_case_cost: float = Field(default=0.0)
    average_cost: float = Field(default=0.0)
    best_case_cost: float = Field(default=0.0)
    cost_variance: float = Field(default=0.0)

    # Scenario coverage
    scenarios_evaluated: int = Field(default=0, ge=0)
    scenarios_feasible: int = Field(default=0, ge=0)
    feasibility_rate: float = Field(default=0.0, ge=0, le=1)

    # Constraint margins
    min_margin_across_scenarios: float = Field(default=0.0)
    margin_robustness: float = Field(default=0.0, description="Min margin / nominal margin")

    # Trade-off analysis
    optimality_gap: float = Field(default=0.0, description="% worse than nominal optimal")
    robustness_benefit: float = Field(default=0.0, description="Improvement in worst case")


class FuelSwitchAnalysis(BaseModel):
    """Analysis of fuel switch impact."""
    analysis_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Fuel types
    from_fuel: str = Field(...)
    to_fuel: str = Field(...)

    # Required setpoint changes
    required_o2_change: float = Field(default=0.0)
    required_air_change: float = Field(default=0.0)
    required_valve_change: float = Field(default=0.0)

    # Expected impacts
    efficiency_impact: float = Field(default=0.0, description="% change")
    cost_impact: float = Field(default=0.0, description="$/hr change")
    co2_impact: float = Field(default=0.0, description="% change")
    nox_impact: float = Field(default=0.0, description="% change")

    # Transition requirements
    transition_time_s: float = Field(default=300.0, ge=0)
    ramp_rate_recommended: float = Field(default=1.0)

    # Risks
    transition_risks: List[str] = Field(default_factory=list)
    recommended_precautions: List[str] = Field(default_factory=list)

    # Feasibility
    is_feasible: bool = Field(default=True)
    feasibility_concerns: List[str] = Field(default_factory=list)


class ScenarioEvaluator:
    """
    Evaluator for combustion scenarios.

    Provides what-if analysis, sensitivity studies, and robust optimization.
    """

    def __init__(
        self,
        objective_function: Optional[MultiObjectiveFunction] = None,
        constraint_set: Optional[ConstraintSet] = None
    ) -> None:
        """
        Initialize scenario evaluator.

        Args:
            objective_function: Multi-objective function
            constraint_set: Constraint set for feasibility checking
        """
        self.objective_function = objective_function or create_balanced_objective()
        self.constraint_set = constraint_set or create_combustion_constraint_set()

        # Fuel properties (ZERO-HALLUCINATION - from reference data)
        self.fuel_properties = {
            "natural_gas": {
                "hhv_kj_kg": 55500,
                "stoich_afr": 17.2,
                "co2_kg_per_gj": 56.1,
                "optimal_o2": 3.0
            },
            "fuel_oil": {
                "hhv_kj_kg": 45500,
                "stoich_afr": 14.7,
                "co2_kg_per_gj": 77.4,
                "optimal_o2": 3.5
            },
            "propane": {
                "hhv_kj_kg": 50300,
                "stoich_afr": 15.6,
                "co2_kg_per_gj": 63.0,
                "optimal_o2": 3.2
            },
            "hydrogen": {
                "hhv_kj_kg": 141800,
                "stoich_afr": 34.3,
                "co2_kg_per_gj": 0.0,
                "optimal_o2": 2.5
            }
        }

        logger.info("ScenarioEvaluator initialized")

    def evaluate_scenario(
        self,
        setpoints: Dict[str, float],
        disturbances: Dict[str, float]
    ) -> ScenarioOutcome:
        """
        Evaluate a specific scenario.

        Args:
            setpoints: Dictionary of setpoint values
            disturbances: Dictionary of disturbance values

        Returns:
            ScenarioOutcome with predicted outcomes
        """
        start_time = datetime.now(timezone.utc)

        # Create state from setpoints and disturbances
        base_state = self._create_state_from_scenario(setpoints, disturbances)
        sp_vector = self._create_setpoint_vector(setpoints)

        # Evaluate objective function
        obj_result = self.objective_function.evaluate(base_state, sp_vector)

        # Check constraints
        values = {**setpoints, **disturbances}
        values["o2_percent"] = base_state.o2_percent
        values["co_ppm"] = base_state.co_ppm
        values["nox_ppm"] = base_state.nox_ppm
        constraint_result = self.constraint_set.evaluate(values)

        # Calculate predicted outcomes (ZERO-HALLUCINATION)
        predicted_efficiency = base_state.combustion_efficiency * 100
        predicted_co = base_state.co_ppm
        predicted_nox = base_state.nox_ppm

        # Calculate risk score
        risk_score, risk_factors = self._calculate_risk(base_state, constraint_result)

        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return ScenarioOutcome(
            scenario_type=ScenarioType.WHAT_IF,
            scenario_name="Custom scenario evaluation",
            input_setpoints=setpoints,
            input_disturbances=disturbances,
            predicted_efficiency=predicted_efficiency,
            predicted_co_ppm=predicted_co,
            predicted_nox_ppm=predicted_nox,
            predicted_cost_per_hour=obj_result.total_cost * 100,  # Scale
            objective_result=obj_result,
            total_cost=obj_result.total_cost,
            is_feasible=constraint_result.is_feasible,
            constraint_violations=[
                e.constraint_name for e in constraint_result.evaluations.values()
                if e.is_violated
            ],
            min_constraint_margin=constraint_result.min_margin,
            uncertainty_range=obj_result.total_cost_uncertainty,
            risk_score=risk_score,
            risk_factors=risk_factors,
            evaluation_time_ms=eval_time
        )

    def _create_state_from_scenario(
        self,
        setpoints: Dict[str, float],
        disturbances: Dict[str, float]
    ) -> BurnerState:
        """Create BurnerState from scenario parameters."""
        o2 = setpoints.get("o2_setpoint_percent", 3.0)
        load = disturbances.get("load_percent", 75.0)

        # ZERO-HALLUCINATION: Calculate dependent variables
        fuel_rate = load / 100 * 2.0  # Assume 2 kg/s at 100% load
        air_flow = fuel_rate * 17.2 * (1 + o2 / 21)

        # Efficiency model (simple)
        efficiency = 0.92 - 0.005 * max(0, o2 - 3.0)

        # Emissions model
        co_ppm = 30 + 50 * max(0, 3.0 - o2) ** 2
        nox_ppm = 50 - 5 * max(0, o2 - 3.0)

        return BurnerState(
            load_percent=load,
            thermal_output_mw=load / 100 * 10,  # 10 MW max
            duty_target_mw=load / 100 * 10,
            o2_percent=o2,
            co_ppm=co_ppm,
            nox_ppm=nox_ppm,
            fuel_rate_kg_s=fuel_rate,
            air_flow_kg_s=air_flow,
            air_fuel_ratio=air_flow / fuel_rate if fuel_rate > 0 else 0,
            excess_air_percent=(o2 / 21) * 100,
            flame_temperature_c=1800 - 10 * o2,
            stack_temperature_c=200 + 50 * (1 - efficiency),
            combustion_efficiency=efficiency,
            flame_stability_index=0.95 if o2 > 1.0 else 0.7
        )

    def _create_setpoint_vector(self, setpoints: Dict[str, float]) -> SetpointVector:
        """Create SetpointVector from dictionary."""
        return SetpointVector(
            o2_setpoint_percent=setpoints.get("o2_setpoint_percent", 3.0),
            air_damper_position=setpoints.get("air_damper_position", 50.0),
            fuel_valve_position=setpoints.get("fuel_valve_position", 50.0)
        )

    def _calculate_risk(
        self,
        state: BurnerState,
        constraint_result: Any
    ) -> Tuple[float, List[str]]:
        """Calculate risk score and identify risk factors."""
        risk_score = 0.0
        risk_factors = []

        # Constraint margin risk
        if constraint_result.min_margin < 1.0:
            risk_score += 0.3
            risk_factors.append("Operating close to constraint boundaries")

        # Stability risk
        if state.flame_stability_index < 0.9:
            risk_score += 0.2
            risk_factors.append("Reduced flame stability")

        # CO risk
        if state.co_ppm > 100:
            risk_score += 0.2
            risk_factors.append("Elevated CO levels")

        # Low O2 risk
        if state.o2_percent < 1.5:
            risk_score += 0.3
            risk_factors.append("Low O2 - risk of incomplete combustion")

        return min(1.0, risk_score), risk_factors

    def run_sensitivity_analysis(
        self,
        base_case: Dict[str, float],
        variations: List[str],
        variation_percent: float = 10.0
    ) -> SensitivityResults:
        """
        Run sensitivity analysis on parameters.

        Args:
            base_case: Base case setpoints
            variations: Parameters to vary
            variation_percent: Variation as percentage

        Returns:
            SensitivityResults with analysis
        """
        parameter_results = {}
        sensitivities = []

        # Evaluate base case
        base_outcome = self.evaluate_scenario(base_case, {"load_percent": 75.0})
        base_cost = base_outcome.total_cost

        for param in variations:
            if param not in base_case:
                continue

            base_value = base_case[param]
            delta = base_value * variation_percent / 100

            # Evaluate low case
            low_case = base_case.copy()
            low_case[param] = base_value - delta
            low_outcome = self.evaluate_scenario(low_case, {"load_percent": 75.0})

            # Evaluate high case
            high_case = base_case.copy()
            high_case[param] = base_value + delta
            high_outcome = self.evaluate_scenario(high_case, {"load_percent": 75.0})

            # Calculate sensitivity
            cost_change = high_outcome.total_cost - low_outcome.total_cost
            sensitivity = cost_change / (2 * delta) if delta > 0 else 0.0

            # Calculate elasticity
            avg_cost = (high_outcome.total_cost + low_outcome.total_cost) / 2
            elasticity = (cost_change / avg_cost) / (2 * variation_percent / 100) if avg_cost > 0 else 0.0

            result = SensitivityResult(
                parameter_name=param,
                base_value=base_value,
                variation_range=(base_value - delta, base_value + delta),
                sensitivity_coefficient=sensitivity,
                elasticity=elasticity,
                effect_on_cost=cost_change,
                effect_on_efficiency=high_outcome.predicted_efficiency - low_outcome.predicted_efficiency,
                effect_on_co=high_outcome.predicted_co_ppm - low_outcome.predicted_co_ppm,
                effect_on_nox=high_outcome.predicted_nox_ppm - low_outcome.predicted_nox_ppm,
                is_stable=low_outcome.is_feasible and high_outcome.is_feasible
            )

            parameter_results[param] = result
            sensitivities.append((param, abs(sensitivity)))

        # Sort by sensitivity
        sensitivities.sort(key=lambda x: x[1], reverse=True)
        most_sensitive = [p[0] for p in sensitivities[:3]]
        least_sensitive = [p[0] for p in sensitivities[-3:]]

        avg_sens = np.mean([s[1] for s in sensitivities]) if sensitivities else 0.0
        max_sens = max([s[1] for s in sensitivities]) if sensitivities else 0.0

        return SensitivityResults(
            base_case=base_case,
            parameters_analyzed=variations,
            variation_percent=variation_percent,
            parameter_results=parameter_results,
            most_sensitive_parameters=most_sensitive,
            least_sensitive_parameters=least_sensitive,
            average_sensitivity=avg_sens,
            max_sensitivity=max_sens,
            overall_stability=all(r.is_stable for r in parameter_results.values()),
            recommendations=self._generate_sensitivity_recommendations(parameter_results)
        )

    def _generate_sensitivity_recommendations(
        self,
        results: Dict[str, SensitivityResult]
    ) -> List[str]:
        """Generate recommendations from sensitivity analysis."""
        recs = []

        for param, result in results.items():
            if abs(result.elasticity) > 0.5:
                recs.append(f"Tighten control on {param} - high sensitivity to variations")

            if not result.is_stable:
                recs.append(f"Review constraints for {param} - instability detected in variations")

        return recs

    def compute_robust_setpoints(
        self,
        scenarios: List[Dict[str, float]]
    ) -> RobustSetpoints:
        """
        Compute setpoints that are robust across scenarios.

        Args:
            scenarios: List of scenario parameter dictionaries

        Returns:
            RobustSetpoints with robust configuration
        """
        if not scenarios:
            return RobustSetpoints(
                optimal_setpoints={},
                scenarios_evaluated=0
            )

        # Grid search for robust setpoints
        o2_range = np.linspace(2.0, 5.0, 7)
        damper_range = np.linspace(40, 60, 5)
        valve_range = np.linspace(40, 60, 5)

        best_worst_cost = float("inf")
        best_setpoints = {}
        all_costs = []

        for o2 in o2_range:
            for damper in damper_range:
                for valve in valve_range:
                    setpoints = {
                        "o2_setpoint_percent": o2,
                        "air_damper_position": damper,
                        "fuel_valve_position": valve
                    }

                    scenario_costs = []
                    all_feasible = True

                    for scenario in scenarios:
                        outcome = self.evaluate_scenario(setpoints, scenario)
                        scenario_costs.append(outcome.total_cost)
                        if not outcome.is_feasible:
                            all_feasible = False

                    if all_feasible:
                        worst_cost = max(scenario_costs)
                        if worst_cost < best_worst_cost:
                            best_worst_cost = worst_cost
                            best_setpoints = setpoints
                            all_costs = scenario_costs

        feasible_count = sum(
            1 for s in scenarios
            if self.evaluate_scenario(best_setpoints, s).is_feasible
        ) if best_setpoints else 0

        return RobustSetpoints(
            optimal_setpoints=best_setpoints,
            worst_case_cost=max(all_costs) if all_costs else 0.0,
            average_cost=np.mean(all_costs) if all_costs else 0.0,
            best_case_cost=min(all_costs) if all_costs else 0.0,
            cost_variance=np.var(all_costs) if all_costs else 0.0,
            scenarios_evaluated=len(scenarios),
            scenarios_feasible=feasible_count,
            feasibility_rate=feasible_count / len(scenarios) if scenarios else 0.0
        )

    def evaluate_fuel_switch_impact(
        self,
        from_fuel: str,
        to_fuel: str
    ) -> FuelSwitchAnalysis:
        """
        Evaluate impact of switching fuels.

        Args:
            from_fuel: Current fuel type
            to_fuel: Target fuel type

        Returns:
            FuelSwitchAnalysis with transition analysis
        """
        from_props = self.fuel_properties.get(from_fuel, self.fuel_properties["natural_gas"])
        to_props = self.fuel_properties.get(to_fuel, self.fuel_properties["natural_gas"])

        # Calculate required changes (ZERO-HALLUCINATION)
        o2_change = to_props["optimal_o2"] - from_props["optimal_o2"]
        afr_ratio = to_props["stoich_afr"] / from_props["stoich_afr"]
        air_change = (afr_ratio - 1) * 100  # % change in air

        # Calculate impacts
        hhv_ratio = to_props["hhv_kj_kg"] / from_props["hhv_kj_kg"]
        fuel_change = 1 / hhv_ratio - 1  # Fuel rate change to maintain output

        co2_change = (to_props["co2_kg_per_gj"] - from_props["co2_kg_per_gj"]) / from_props["co2_kg_per_gj"] * 100

        # Assess risks
        risks = []
        precautions = []

        if abs(o2_change) > 1.0:
            risks.append("Significant O2 setpoint change required")
            precautions.append("Implement O2 change gradually over 5 minutes")

        if abs(air_change) > 20:
            risks.append("Large air flow change required")
            precautions.append("Monitor furnace pressure during transition")

        if to_fuel == "hydrogen":
            risks.append("Hydrogen has different flame characteristics")
            precautions.append("Verify burner is rated for hydrogen service")
            precautions.append("Check for hydrogen-specific safety interlocks")

        return FuelSwitchAnalysis(
            from_fuel=from_fuel,
            to_fuel=to_fuel,
            required_o2_change=o2_change,
            required_air_change=air_change,
            required_valve_change=fuel_change * 100,
            efficiency_impact=0.0,  # Depends on specific equipment
            cost_impact=0.0,  # Depends on fuel prices
            co2_impact=co2_change,
            nox_impact=-5 * o2_change,  # Approximate
            transition_time_s=300.0,
            ramp_rate_recommended=0.5,
            transition_risks=risks,
            recommended_precautions=precautions,
            is_feasible=True,
            feasibility_concerns=[]
        )

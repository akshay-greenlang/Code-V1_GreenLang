"""
GL-004 Burnmaster - Multi-Objective Optimization Functions
Author: GreenLang Optimization Team
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import uuid

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ObjectiveType(str, Enum):
    FUEL_COST = "fuel_cost"
    EMISSIONS = "emissions"
    STABILITY = "stability"
    ACTUATOR_MOVE = "actuator_move"
    CO_PENALTY = "co_penalty"
    COMPOSITE = "composite"


class OptimizationDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class BurnerState(BaseModel):
    state_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    load_percent: float = Field(..., ge=0, le=100)
    thermal_output_mw: float = Field(..., ge=0)
    duty_target_mw: float = Field(..., ge=0)
    o2_percent: float = Field(..., ge=0, le=21)
    co_ppm: float = Field(..., ge=0)
    nox_ppm: float = Field(..., ge=0)
    fuel_rate_kg_s: float = Field(..., ge=0)
    fuel_type: str = Field(default="natural_gas")
    fuel_hhv_kj_kg: float = Field(default=55500.0, ge=0)
    air_flow_kg_s: float = Field(..., ge=0)
    air_fuel_ratio: float = Field(..., ge=0)
    excess_air_percent: float = Field(..., ge=0)
    flame_temperature_c: float = Field(..., ge=0)
    stack_temperature_c: float = Field(..., ge=0)
    combustion_efficiency: float = Field(..., ge=0, le=1)
    flame_stability_index: float = Field(default=1.0, ge=0, le=1)
    pressure_fluctuation_pa: float = Field(default=0.0, ge=0)

    @property
    def is_stable(self) -> bool:
        return self.flame_stability_index > 0.8 and self.pressure_fluctuation_pa < 500.0 and self.co_ppm < 200.0


class SetpointVector(BaseModel):
    setpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    o2_setpoint_percent: float = Field(..., ge=0.5, le=10.0)
    air_damper_position: float = Field(..., ge=0, le=100)
    fuel_valve_position: float = Field(..., ge=0, le=100)
    fgr_rate_percent: Optional[float] = Field(default=None, ge=0, le=50)
    draft_setpoint_pa: Optional[float] = Field(default=None)

    @property
    def as_vector(self):
        values = [self.o2_setpoint_percent, self.air_damper_position, self.fuel_valve_position]
        if self.fgr_rate_percent is not None:
            values.append(self.fgr_rate_percent)
        if self.draft_setpoint_pa is not None:
            values.append(self.draft_setpoint_pa)
        return np.array(values)


class ObjectiveEvaluation(BaseModel):
    objective_type: ObjectiveType
    value: float
    normalized_value: float = 0.0
    weight: float = Field(default=1.0, ge=0, le=1)
    weighted_value: float = 0.0
    uncertainty_lower: float = 0.0
    uncertainty_upper: float = 0.0
    confidence: float = Field(default=0.95, ge=0, le=1)
    constraint_margin: Optional[float] = None
    constraint_violated: bool = False
    evaluation_time_ms: float = Field(default=0.0, ge=0)


class MultiObjectiveResult(BaseModel):
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    objectives: Dict[str, ObjectiveEvaluation] = Field(default_factory=dict)
    total_cost: float
    total_cost_uncertainty: Tuple[float, float] = (0.0, 0.0)
    is_pareto_optimal: bool = False
    pareto_front_distance: float = 0.0
    dominated_by_count: int = Field(default=0, ge=0)
    is_feasible: bool = True
    violation_count: int = Field(default=0, ge=0)
    max_violation: float = Field(default=0.0, ge=0)
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.provenance_hash:
            obj_str = "|".join([f"{k}:{v.value:.6f}" for k, v in self.objectives.items()])
            hash_input = f"{self.result_id}|{self.total_cost:.6f}|{obj_str}"
            self.provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()


class BaseObjectiveFunction(ABC):
    def __init__(self, name: str, objective_type: ObjectiveType,
                 direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
                 weight: float = 1.0, normalization_range: Optional[Tuple[float, float]] = None) -> None:
        self.name = name
        self.objective_type = objective_type
        self.direction = direction
        self.weight = weight
        self.normalization_range = normalization_range

    @abstractmethod
    def evaluate(self, state: BurnerState, setpoints: SetpointVector) -> ObjectiveEvaluation:
        pass

    def normalize(self, value: float) -> float:
        if self.normalization_range is None:
            return value
        min_val, max_val = self.normalization_range
        if max_val == min_val:
            return 0.0
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))
        if self.direction == OptimizationDirection.MAXIMIZE:
            normalized = 1.0 - normalized
        return normalized

    def compute_uncertainty(self, value: float, state: BurnerState, confidence: float = 0.95) -> Tuple[float, float]:
        relative_uncertainty = 0.05
        z_score = 1.96 if confidence >= 0.95 else 1.645
        margin = abs(value) * relative_uncertainty * z_score
        return (value - margin, value + margin)


class FuelCostObjective(BaseObjectiveFunction):
    def __init__(self, fuel_cost_per_kg: float = 0.35, weight: float = 0.4,
                 normalization_range: Tuple[float, float] = (0.0, 1000.0)) -> None:
        super().__init__("Fuel Cost", ObjectiveType.FUEL_COST, OptimizationDirection.MINIMIZE, weight, normalization_range)
        self.fuel_cost_per_kg = fuel_cost_per_kg

    def evaluate(self, state: BurnerState, setpoints: SetpointVector) -> ObjectiveEvaluation:
        start_time = datetime.now(timezone.utc)
        fuel_rate_kg_hr = state.fuel_rate_kg_s * 3600.0
        fuel_cost_per_hr = fuel_rate_kg_hr * self.fuel_cost_per_kg
        normalized = self.normalize(fuel_cost_per_hr)
        lower, upper = self.compute_uncertainty(fuel_cost_per_hr, state)
        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ObjectiveEvaluation(objective_type=self.objective_type, value=fuel_cost_per_hr,
            normalized_value=normalized, weight=self.weight, weighted_value=normalized * self.weight,
            uncertainty_lower=lower, uncertainty_upper=upper, confidence=0.95, evaluation_time_ms=eval_time)


class EmissionsObjective(BaseObjectiveFunction):
    def __init__(self, nox_cost_per_kg: float = 15.0, co_penalty_per_ppm: float = 0.1,
                 co_limit_ppm: float = 100.0, weight: float = 0.3,
                 normalization_range: Tuple[float, float] = (0.0, 500.0)) -> None:
        super().__init__("Emissions", ObjectiveType.EMISSIONS, OptimizationDirection.MINIMIZE, weight, normalization_range)
        self.nox_cost_per_kg = nox_cost_per_kg
        self.co_penalty_per_ppm = co_penalty_per_ppm
        self.co_limit_ppm = co_limit_ppm

    def evaluate(self, state: BurnerState, setpoints: SetpointVector) -> ObjectiveEvaluation:
        start_time = datetime.now(timezone.utc)
        nox_emission_factor_g_kg = 1.5
        o2_effect = max(0.5, min(1.0, 1.0 - 0.1 * max(0, state.o2_percent - 3.0)))
        nox_kg_hr = state.fuel_rate_kg_s * 3600.0 * nox_emission_factor_g_kg / 1000.0 * o2_effect
        nox_cost = nox_kg_hr * self.nox_cost_per_kg
        co_excess = max(0.0, state.co_ppm - self.co_limit_ppm)
        co_penalty = co_excess * self.co_penalty_per_ppm
        total_emissions_cost = nox_cost + co_penalty
        constraint_violated = co_excess > 0
        constraint_margin = self.co_limit_ppm - state.co_ppm
        normalized = self.normalize(total_emissions_cost)
        lower, upper = self.compute_uncertainty(total_emissions_cost, state)
        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ObjectiveEvaluation(objective_type=self.objective_type, value=total_emissions_cost,
            normalized_value=normalized, weight=self.weight, weighted_value=normalized * self.weight,
            uncertainty_lower=lower, uncertainty_upper=upper, confidence=0.95,
            constraint_margin=constraint_margin, constraint_violated=constraint_violated, evaluation_time_ms=eval_time)


class StabilityObjective(BaseObjectiveFunction):
    def __init__(self, instability_penalty: float = 100.0, pressure_penalty_per_pa: float = 0.01,
                 weight: float = 0.2, normalization_range: Tuple[float, float] = (0.0, 200.0)) -> None:
        super().__init__("Stability", ObjectiveType.STABILITY, OptimizationDirection.MINIMIZE, weight, normalization_range)
        self.instability_penalty = instability_penalty
        self.pressure_penalty_per_pa = pressure_penalty_per_pa

    def evaluate(self, state: BurnerState, setpoints: SetpointVector) -> ObjectiveEvaluation:
        start_time = datetime.now(timezone.utc)
        instability_cost = (1.0 - state.flame_stability_index) * self.instability_penalty
        pressure_cost = state.pressure_fluctuation_pa * self.pressure_penalty_per_pa
        co_instability = max(0.0, state.co_ppm - 100.0) * 0.1
        total_stability_cost = instability_cost + pressure_cost + co_instability
        normalized = self.normalize(total_stability_cost)
        lower, upper = self.compute_uncertainty(total_stability_cost, state)
        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ObjectiveEvaluation(objective_type=self.objective_type, value=total_stability_cost,
            normalized_value=normalized, weight=self.weight, weighted_value=normalized * self.weight,
            uncertainty_lower=lower, uncertainty_upper=upper, confidence=0.95, evaluation_time_ms=eval_time)


class ActuatorMoveObjective(BaseObjectiveFunction):
    def __init__(self, move_penalty_coefficient: float = 0.5, rate_penalty_coefficient: float = 1.0,
                 weight: float = 0.1, normalization_range: Tuple[float, float] = (0.0, 50.0)) -> None:
        super().__init__("Actuator Move", ObjectiveType.ACTUATOR_MOVE, OptimizationDirection.MINIMIZE, weight, normalization_range)
        self.move_penalty_coefficient = move_penalty_coefficient
        self.rate_penalty_coefficient = rate_penalty_coefficient
        self._previous_setpoints: Optional[SetpointVector] = None

    def evaluate(self, state: BurnerState, setpoints: SetpointVector,
                 previous_setpoints: Optional[SetpointVector] = None, dt_seconds: float = 1.0) -> ObjectiveEvaluation:
        start_time = datetime.now(timezone.utc)
        if previous_setpoints is None:
            previous_setpoints = self._previous_setpoints
        if previous_setpoints is None:
            total_move_cost = 0.0
        else:
            o2_move = abs(setpoints.o2_setpoint_percent - previous_setpoints.o2_setpoint_percent)
            damper_move = abs(setpoints.air_damper_position - previous_setpoints.air_damper_position)
            valve_move = abs(setpoints.fuel_valve_position - previous_setpoints.fuel_valve_position)
            total_move = o2_move * 2.0 + damper_move * 1.0 + valve_move * 1.5
            move_penalty = total_move * self.move_penalty_coefficient
            rate_penalty = max(0, total_move / dt_seconds - 5.0) * self.rate_penalty_coefficient if dt_seconds > 0 else 0.0
            total_move_cost = move_penalty + rate_penalty
        self._previous_setpoints = setpoints
        normalized = self.normalize(total_move_cost)
        eval_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return ObjectiveEvaluation(objective_type=self.objective_type, value=total_move_cost,
            normalized_value=normalized, weight=self.weight, weighted_value=normalized * self.weight,
            uncertainty_lower=total_move_cost * 0.95, uncertainty_upper=total_move_cost * 1.05,
            confidence=0.95, evaluation_time_ms=eval_time)


class MultiObjectiveFunction:
    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 objectives: Optional[List[BaseObjectiveFunction]] = None) -> None:
        default_weights = {"fuel": 0.4, "emissions": 0.3, "stability": 0.2, "actuator": 0.1}
        self.weights = weights if weights is not None else default_weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        if objectives is not None:
            self.objectives = {obj.name.lower().replace(" ", "_"): obj for obj in objectives}
        else:
            self.objectives = self._create_default_objectives()
        for name, weight in self.weights.items():
            if name in self.objectives:
                self.objectives[name].weight = weight
        self._pareto_front: List[MultiObjectiveResult] = []

    def _create_default_objectives(self) -> Dict[str, BaseObjectiveFunction]:
        return {
            "fuel": FuelCostObjective(weight=self.weights.get("fuel", 0.4)),
            "emissions": EmissionsObjective(weight=self.weights.get("emissions", 0.3)),
            "stability": StabilityObjective(weight=self.weights.get("stability", 0.2)),
            "actuator": ActuatorMoveObjective(weight=self.weights.get("actuator", 0.1))
        }

    def evaluate(self, state: BurnerState, setpoints: SetpointVector,
                 previous_setpoints: Optional[SetpointVector] = None) -> MultiObjectiveResult:
        evaluations: Dict[str, ObjectiveEvaluation] = {}
        for name, objective in self.objectives.items():
            if name == "actuator" and isinstance(objective, ActuatorMoveObjective):
                evaluations[name] = objective.evaluate(state, setpoints, previous_setpoints)
            else:
                evaluations[name] = objective.evaluate(state, setpoints)
        total_cost = sum(ev.weighted_value for ev in evaluations.values())
        uncertainty_variance = sum(((ev.uncertainty_upper - ev.uncertainty_lower) / 2 * ev.weight) ** 2 for ev in evaluations.values())
        uncertainty_margin = math.sqrt(uncertainty_variance)
        violations = [ev for ev in evaluations.values() if ev.constraint_violated]
        max_violation = max((abs(ev.constraint_margin) if ev.constraint_margin and ev.constraint_margin < 0 else 0) for ev in evaluations.values()) if evaluations else 0.0
        result = MultiObjectiveResult(objectives=evaluations, total_cost=total_cost,
            total_cost_uncertainty=(total_cost - uncertainty_margin, total_cost + uncertainty_margin),
            is_feasible=len(violations) == 0, violation_count=len(violations), max_violation=max_violation)
        self._update_pareto_analysis(result)
        return result

    def _update_pareto_analysis(self, result: MultiObjectiveResult) -> None:
        if not result.is_feasible:
            result.is_pareto_optimal = False
            return
        result_values = np.array([result.objectives[name].value for name in sorted(result.objectives.keys())])
        dominated_by = 0
        for front_result in self._pareto_front:
            front_values = np.array([front_result.objectives[name].value for name in sorted(front_result.objectives.keys())])
            if np.all(front_values <= result_values) and np.any(front_values < result_values):
                dominated_by += 1
        result.dominated_by_count = dominated_by
        result.is_pareto_optimal = dominated_by == 0
        if result.is_pareto_optimal:
            self._pareto_front = [r for r in self._pareto_front if not self._dominates(result, r)]
            self._pareto_front.append(result)

    def _dominates(self, result_a: MultiObjectiveResult, result_b: MultiObjectiveResult) -> bool:
        values_a = np.array([result_a.objectives[name].value for name in sorted(result_a.objectives.keys())])
        values_b = np.array([result_b.objectives[name].value for name in sorted(result_b.objectives.keys())])
        return np.all(values_a <= values_b) and np.any(values_a < values_b)

    def get_pareto_front(self) -> List[MultiObjectiveResult]:
        return list(self._pareto_front)

    def clear_pareto_front(self) -> None:
        self._pareto_front.clear()

    def set_weights(self, weights: Dict[str, float]) -> None:
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}
        for name, weight in self.weights.items():
            if name in self.objectives:
                self.objectives[name].weight = weight

    def get_objective_breakdown(self, result: MultiObjectiveResult) -> Dict[str, Dict[str, float]]:
        breakdown = {}
        for name, ev in result.objectives.items():
            breakdown[name] = {
                "raw_value": ev.value, "normalized_value": ev.normalized_value,
                "weight": ev.weight, "weighted_contribution": ev.weighted_value,
                "percent_of_total": (ev.weighted_value / result.total_cost * 100 if result.total_cost > 0 else 0.0),
                "uncertainty_lower": ev.uncertainty_lower, "uncertainty_upper": ev.uncertainty_upper
            }
        return breakdown


def create_fuel_focused_objective(fuel_weight: float = 0.6) -> MultiObjectiveFunction:
    remaining = 1.0 - fuel_weight
    return MultiObjectiveFunction(weights={"fuel": fuel_weight, "emissions": remaining * 0.5, "stability": remaining * 0.35, "actuator": remaining * 0.15})


def create_emissions_focused_objective(emissions_weight: float = 0.6) -> MultiObjectiveFunction:
    remaining = 1.0 - emissions_weight
    return MultiObjectiveFunction(weights={"fuel": remaining * 0.4, "emissions": emissions_weight, "stability": remaining * 0.4, "actuator": remaining * 0.2})


def create_balanced_objective() -> MultiObjectiveFunction:
    return MultiObjectiveFunction(weights={"fuel": 0.35, "emissions": 0.30, "stability": 0.25, "actuator": 0.10})

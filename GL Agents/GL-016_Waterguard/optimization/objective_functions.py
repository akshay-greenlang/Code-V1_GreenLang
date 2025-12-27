"""
GL-016 Waterguard Objective Functions - Multi-Objective Optimization

Multi-objective optimization components for cooling tower water treatment.
Implements individual objectives and weighted combination with Pareto
frontier generation.

Objectives:
    - WaterLossObjective: Minimize blowdown volume
    - EnergyLossObjective: Minimize blowdown enthalpy loss
    - ChemicalCostObjective: Minimize dosing chemical consumption
    - RiskPenaltyObjective: Minimize scaling/corrosion/carryover risk
    - WeightedSumObjective: Combine with configurable weights

Reference Standards:
    - CTI STD-201 (Cooling Tower Water Treatment)
    - ISO 50001 (Energy Management)

Author: GreenLang Water Treatment Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class ObjectiveType(str, Enum):
    """Types of optimization objectives."""
    WATER_LOSS = "water_loss"
    ENERGY_LOSS = "energy_loss"
    CHEMICAL_COST = "chemical_cost"
    RISK_PENALTY = "risk_penalty"
    WEIGHTED_SUM = "weighted_sum"


class OptimizationDirection(str, Enum):
    """Direction of optimization."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


# =============================================================================
# DATA MODELS
# =============================================================================

class ObjectiveValue(BaseModel):
    """Value of an objective function evaluation."""
    objective_type: ObjectiveType
    value: float
    unit: str = ""
    components: Dict[str, float] = Field(default_factory=dict)
    is_normalized: bool = False


class ParetoPoint(BaseModel):
    """A point on the Pareto frontier."""
    point_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    decision_variables: Dict[str, float]
    objective_values: Dict[str, float]
    is_dominated: bool = False

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objective_values)


class ParetoFrontier(BaseModel):
    """Pareto frontier for multi-objective optimization."""
    frontier_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Frontier points
    points: List[ParetoPoint] = Field(default_factory=list)

    # Objective names
    objective_names: List[str] = Field(default_factory=list)

    # Bounds
    ideal_point: Dict[str, float] = Field(default_factory=dict)
    nadir_point: Dict[str, float] = Field(default_factory=dict)

    # Metrics
    hypervolume: float = 0.0
    spacing: float = 0.0

    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance and metrics."""
        if not self.provenance_hash:
            provenance_str = (
                f"{self.frontier_id}|{len(self.points)}|"
                f"{self.hypervolume:.4f}"
            )
            self.provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

    @property
    def num_points(self) -> int:
        """Number of points on frontier."""
        return len(self.points)

    def get_point_by_preference(
        self,
        weights: Dict[str, float]
    ) -> Optional[ParetoPoint]:
        """
        Get Pareto point closest to weighted preference.

        Args:
            weights: Dict of objective weights (higher = more important)

        Returns:
            ParetoPoint closest to preference
        """
        if not self.points:
            return None

        # Normalize objectives to [0, 1] range
        normalized_scores = []

        for point in self.points:
            score = 0.0
            for obj_name, obj_value in point.objective_values.items():
                # Normalize using ideal/nadir
                if obj_name in self.ideal_point and obj_name in self.nadir_point:
                    ideal = self.ideal_point[obj_name]
                    nadir = self.nadir_point[obj_name]
                    if nadir != ideal:
                        normalized = (obj_value - ideal) / (nadir - ideal)
                    else:
                        normalized = 0.0
                else:
                    normalized = obj_value

                # Apply weight
                weight = weights.get(obj_name, 1.0)
                score += weight * normalized

            normalized_scores.append((point, score))

        # Return point with lowest weighted score (assuming minimization)
        best_point, _ = min(normalized_scores, key=lambda x: x[1])
        return best_point


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions - ZERO HALLUCINATION.

    All objective functions must be deterministic and produce
    reproducible results for the same inputs.
    """

    def __init__(
        self,
        name: str,
        direction: OptimizationDirection = OptimizationDirection.MINIMIZE,
        weight: float = 1.0
    ):
        """
        Initialize objective function.

        Args:
            name: Name of the objective
            direction: Optimization direction
            weight: Weight for multi-objective combination
        """
        self.name = name
        self.direction = direction
        self.weight = weight
        self._normalization_factor: Optional[float] = None

    @abstractmethod
    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate the objective function - DETERMINISTIC.

        Args:
            decision_variables: Decision variable values
            context: Additional context (state, parameters)

        Returns:
            ObjectiveValue with computed value
        """
        pass

    def set_normalization(self, factor: float) -> None:
        """Set normalization factor for multi-objective combination."""
        self._normalization_factor = factor

    def get_normalized_value(self, value: float) -> float:
        """Get normalized value for multi-objective combination."""
        if self._normalization_factor and self._normalization_factor != 0:
            return value / self._normalization_factor
        return value


# =============================================================================
# WATER LOSS OBJECTIVE
# =============================================================================

class WaterLossObjective(ObjectiveFunction):
    """
    Minimize blowdown water volume - ZERO HALLUCINATION.

    Water loss cost = blowdown_volume * water_cost

    Deterministic calculation based on:
    - Blowdown valve position
    - Maximum flow capacity
    - Time horizon
    - Water cost per volume
    """

    def __init__(
        self,
        water_cost_per_1000gal: float = 5.0,
        max_blowdown_gpm: float = 50.0,
        weight: float = 1.0
    ):
        """
        Initialize water loss objective.

        Args:
            water_cost_per_1000gal: Water cost ($/1000 gal)
            max_blowdown_gpm: Maximum blowdown flow (gpm)
            weight: Objective weight
        """
        super().__init__("water_loss", OptimizationDirection.MINIMIZE, weight)
        self.water_cost = water_cost_per_1000gal
        self.max_flow = max_blowdown_gpm

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate water loss cost - DETERMINISTIC.

        Args:
            decision_variables: Must contain 'blowdown_pct'
            context: Must contain 'horizon_minutes'

        Returns:
            ObjectiveValue with water loss cost
        """
        blowdown_pct = decision_variables.get("blowdown_pct", 30.0)
        horizon_minutes = context.get("horizon_minutes", 30)

        # Calculate blowdown volume
        flow_gpm = (blowdown_pct / 100.0) * self.max_flow
        volume_gal = flow_gpm * horizon_minutes

        # Calculate cost
        cost = (volume_gal / 1000.0) * self.water_cost

        return ObjectiveValue(
            objective_type=ObjectiveType.WATER_LOSS,
            value=cost,
            unit="$/period",
            components={
                "volume_gal": volume_gal,
                "flow_gpm": flow_gpm,
                "unit_cost": self.water_cost
            }
        )


# =============================================================================
# ENERGY LOSS OBJECTIVE
# =============================================================================

class EnergyLossObjective(ObjectiveFunction):
    """
    Minimize blowdown enthalpy loss - ZERO HALLUCINATION.

    Energy loss = blowdown_volume * enthalpy_per_gallon * energy_cost

    Blowdown water carries away thermal energy that must be replaced,
    so reducing blowdown saves energy.
    """

    def __init__(
        self,
        energy_cost_per_kwh: float = 0.12,
        blowdown_enthalpy_btu_per_gal: float = 40.0,
        max_blowdown_gpm: float = 50.0,
        weight: float = 1.0
    ):
        """
        Initialize energy loss objective.

        Args:
            energy_cost_per_kwh: Energy cost ($/kWh)
            blowdown_enthalpy_btu_per_gal: Enthalpy content (BTU/gal)
            max_blowdown_gpm: Maximum blowdown flow (gpm)
            weight: Objective weight
        """
        super().__init__("energy_loss", OptimizationDirection.MINIMIZE, weight)
        self.energy_cost = energy_cost_per_kwh
        self.enthalpy = blowdown_enthalpy_btu_per_gal
        self.max_flow = max_blowdown_gpm

        # Conversion: BTU to kWh
        self.btu_to_kwh = 1.0 / 3412.0

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate energy loss cost - DETERMINISTIC.

        Args:
            decision_variables: Must contain 'blowdown_pct'
            context: Must contain 'horizon_minutes', optionally 'delta_t_f'

        Returns:
            ObjectiveValue with energy loss cost
        """
        blowdown_pct = decision_variables.get("blowdown_pct", 30.0)
        horizon_minutes = context.get("horizon_minutes", 30)
        delta_t_f = context.get("delta_t_f", 10.0)  # Temperature delta in F

        # Adjust enthalpy for actual temperature delta
        # Standard: 40 BTU/gal at 10F delta
        adjusted_enthalpy = self.enthalpy * (delta_t_f / 10.0)

        # Calculate blowdown volume
        flow_gpm = (blowdown_pct / 100.0) * self.max_flow
        volume_gal = flow_gpm * horizon_minutes

        # Calculate energy loss
        energy_btu = volume_gal * adjusted_enthalpy
        energy_kwh = energy_btu * self.btu_to_kwh

        # Calculate cost
        cost = energy_kwh * self.energy_cost

        return ObjectiveValue(
            objective_type=ObjectiveType.ENERGY_LOSS,
            value=cost,
            unit="$/period",
            components={
                "volume_gal": volume_gal,
                "energy_btu": energy_btu,
                "energy_kwh": energy_kwh,
                "enthalpy_btu_gal": adjusted_enthalpy
            }
        )


# =============================================================================
# CHEMICAL COST OBJECTIVE
# =============================================================================

class ChemicalCostObjective(ObjectiveFunction):
    """
    Minimize chemical dosing cost - ZERO HALLUCINATION.

    Chemical cost = sum(dosing_rate * chemical_cost)

    Tracks costs for scale inhibitor, corrosion inhibitor, biocide, and acid.
    """

    def __init__(
        self,
        scale_inhibitor_cost: float = 15.0,  # $/gal
        corrosion_inhibitor_cost: float = 12.0,
        biocide_cost: float = 20.0,
        acid_cost: float = 3.0,
        max_dosing_ml_min: float = 100.0,
        weight: float = 1.0
    ):
        """
        Initialize chemical cost objective.

        Args:
            scale_inhibitor_cost: Scale inhibitor cost ($/gal)
            corrosion_inhibitor_cost: Corrosion inhibitor cost ($/gal)
            biocide_cost: Biocide cost ($/gal)
            acid_cost: Acid cost ($/gal)
            max_dosing_ml_min: Maximum dosing pump rate (ml/min)
            weight: Objective weight
        """
        super().__init__("chemical_cost", OptimizationDirection.MINIMIZE, weight)
        self.costs = {
            "scale_inhibitor": scale_inhibitor_cost,
            "corrosion_inhibitor": corrosion_inhibitor_cost,
            "biocide": biocide_cost,
            "acid": acid_cost
        }
        self.max_rate = max_dosing_ml_min
        self.ml_per_gallon = 3785.0

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate chemical cost - DETERMINISTIC.

        Args:
            decision_variables: Contains pump speeds (scale_inhibitor_pct, etc.)
            context: Must contain 'horizon_minutes'

        Returns:
            ObjectiveValue with chemical cost
        """
        horizon_minutes = context.get("horizon_minutes", 30)

        total_cost = 0.0
        components = {}

        for chem_name, unit_cost in self.costs.items():
            pump_pct = decision_variables.get(f"{chem_name}_pct", 0.0)

            # Calculate dosing volume
            rate_ml_min = (pump_pct / 100.0) * self.max_rate
            volume_ml = rate_ml_min * horizon_minutes
            volume_gal = volume_ml / self.ml_per_gallon

            # Calculate cost
            cost = volume_gal * unit_cost
            total_cost += cost

            components[f"{chem_name}_ml"] = volume_ml
            components[f"{chem_name}_cost"] = cost

        return ObjectiveValue(
            objective_type=ObjectiveType.CHEMICAL_COST,
            value=total_cost,
            unit="$/period",
            components=components
        )


# =============================================================================
# RISK PENALTY OBJECTIVE
# =============================================================================

class RiskPenaltyObjective(ObjectiveFunction):
    """
    Minimize operational risk penalty - ZERO HALLUCINATION.

    Risk penalty is calculated from ML risk predictions with:
    - Bounded values (0-1 range)
    - Calibrated models
    - Conservative margin for uncertainty

    Risk types: scaling, corrosion, carryover, biological
    """

    def __init__(
        self,
        scaling_penalty_weight: float = 100.0,  # $/unit risk
        corrosion_penalty_weight: float = 100.0,
        carryover_penalty_weight: float = 50.0,
        biological_penalty_weight: float = 50.0,
        uncertainty_margin: float = 0.1,
        weight: float = 1.0
    ):
        """
        Initialize risk penalty objective.

        Args:
            scaling_penalty_weight: Cost per unit scaling risk
            corrosion_penalty_weight: Cost per unit corrosion risk
            carryover_penalty_weight: Cost per unit carryover risk
            biological_penalty_weight: Cost per unit biological risk
            uncertainty_margin: Margin added for model uncertainty
            weight: Objective weight
        """
        super().__init__("risk_penalty", OptimizationDirection.MINIMIZE, weight)
        self.penalty_weights = {
            "scaling": scaling_penalty_weight,
            "corrosion": corrosion_penalty_weight,
            "carryover": carryover_penalty_weight,
            "biological": biological_penalty_weight
        }
        self.uncertainty_margin = uncertainty_margin

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate risk penalty - DETERMINISTIC.

        Args:
            decision_variables: Contains control setpoints
            context: Must contain risk predictions and uncertainties

        Returns:
            ObjectiveValue with risk penalty cost
        """
        total_penalty = 0.0
        components = {}

        for risk_type, penalty_weight in self.penalty_weights.items():
            # Get risk value (bounded 0-1)
            risk_value = context.get(f"{risk_type}_risk", 0.0)
            risk_value = max(0.0, min(1.0, risk_value))

            # Get uncertainty (bounded 0-1)
            uncertainty = context.get(f"{risk_type}_uncertainty", self.uncertainty_margin)
            uncertainty = max(0.0, min(1.0, uncertainty))

            # Add conservative margin for uncertainty
            adjusted_risk = min(1.0, risk_value + uncertainty)

            # Calculate penalty
            penalty = adjusted_risk * penalty_weight
            total_penalty += penalty

            components[f"{risk_type}_risk"] = risk_value
            components[f"{risk_type}_uncertainty"] = uncertainty
            components[f"{risk_type}_penalty"] = penalty

        # Risk reduction from blowdown (higher blowdown reduces concentration risk)
        blowdown_pct = decision_variables.get("blowdown_pct", 30.0)
        blowdown_reduction = (blowdown_pct / 100.0) * 0.2 * sum(self.penalty_weights.values())
        total_penalty = max(0, total_penalty - blowdown_reduction)

        # Risk reduction from chemical treatment
        scale_inhib = decision_variables.get("scale_inhibitor_pct", 0.0)
        corr_inhib = decision_variables.get("corrosion_inhibitor_pct", 0.0)
        treatment_reduction = (
            (scale_inhib / 100.0) * 0.1 * self.penalty_weights["scaling"] +
            (corr_inhib / 100.0) * 0.1 * self.penalty_weights["corrosion"]
        )
        total_penalty = max(0, total_penalty - treatment_reduction)

        components["blowdown_reduction"] = blowdown_reduction
        components["treatment_reduction"] = treatment_reduction

        return ObjectiveValue(
            objective_type=ObjectiveType.RISK_PENALTY,
            value=total_penalty,
            unit="$/period",
            components=components
        )


# =============================================================================
# WEIGHTED SUM OBJECTIVE
# =============================================================================

class WeightedSumObjective(ObjectiveFunction):
    """
    Weighted sum of multiple objectives - ZERO HALLUCINATION.

    Combines water loss, energy loss, chemical cost, and risk penalty
    into a single scalar objective for optimization.
    """

    def __init__(
        self,
        water_loss_weight: float = 0.30,
        energy_loss_weight: float = 0.25,
        chemical_cost_weight: float = 0.20,
        risk_penalty_weight: float = 0.25,
        **objective_kwargs
    ):
        """
        Initialize weighted sum objective.

        Args:
            water_loss_weight: Weight for water loss [0-1]
            energy_loss_weight: Weight for energy loss [0-1]
            chemical_cost_weight: Weight for chemical cost [0-1]
            risk_penalty_weight: Weight for risk penalty [0-1]
            **objective_kwargs: Additional arguments for sub-objectives
        """
        super().__init__("weighted_sum", OptimizationDirection.MINIMIZE)

        # Normalize weights
        total_weight = (water_loss_weight + energy_loss_weight +
                       chemical_cost_weight + risk_penalty_weight)

        self.weights = {
            "water_loss": water_loss_weight / total_weight,
            "energy_loss": energy_loss_weight / total_weight,
            "chemical_cost": chemical_cost_weight / total_weight,
            "risk_penalty": risk_penalty_weight / total_weight
        }

        # Create sub-objectives
        self.objectives = {
            "water_loss": WaterLossObjective(
                water_cost_per_1000gal=objective_kwargs.get("water_cost", 5.0),
                max_blowdown_gpm=objective_kwargs.get("max_blowdown_gpm", 50.0)
            ),
            "energy_loss": EnergyLossObjective(
                energy_cost_per_kwh=objective_kwargs.get("energy_cost", 0.12),
                blowdown_enthalpy_btu_per_gal=objective_kwargs.get("enthalpy", 40.0),
                max_blowdown_gpm=objective_kwargs.get("max_blowdown_gpm", 50.0)
            ),
            "chemical_cost": ChemicalCostObjective(
                scale_inhibitor_cost=objective_kwargs.get("scale_cost", 15.0),
                corrosion_inhibitor_cost=objective_kwargs.get("corrosion_cost", 12.0)
            ),
            "risk_penalty": RiskPenaltyObjective(
                scaling_penalty_weight=objective_kwargs.get("scaling_penalty", 100.0),
                corrosion_penalty_weight=objective_kwargs.get("corrosion_penalty", 100.0)
            )
        }

        # Normalization factors (will be set from initial evaluation)
        self._normalization = {}

    def evaluate(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> ObjectiveValue:
        """
        Evaluate weighted sum of objectives - DETERMINISTIC.

        Args:
            decision_variables: Decision variable values
            context: Additional context

        Returns:
            ObjectiveValue with weighted sum
        """
        total_value = 0.0
        components = {}

        for obj_name, objective in self.objectives.items():
            # Evaluate sub-objective
            result = objective.evaluate(decision_variables, context)

            # Store raw value
            components[f"{obj_name}_raw"] = result.value

            # Normalize if we have normalization factors
            if obj_name in self._normalization and self._normalization[obj_name] > 0:
                normalized = result.value / self._normalization[obj_name]
            else:
                normalized = result.value

            # Apply weight
            weighted = self.weights[obj_name] * normalized
            components[f"{obj_name}_weighted"] = weighted

            total_value += weighted

        return ObjectiveValue(
            objective_type=ObjectiveType.WEIGHTED_SUM,
            value=total_value,
            unit="normalized",
            components=components,
            is_normalized=bool(self._normalization)
        )

    def set_normalization_from_baseline(
        self,
        decision_variables: Dict[str, float],
        context: Dict[str, Any]
    ) -> None:
        """
        Set normalization factors from baseline evaluation.

        Args:
            decision_variables: Baseline decision variables
            context: Baseline context
        """
        for obj_name, objective in self.objectives.items():
            result = objective.evaluate(decision_variables, context)
            self._normalization[obj_name] = max(abs(result.value), 1e-6)

        logger.info("Set normalization factors: %s", self._normalization)


# =============================================================================
# PARETO FRONTIER GENERATOR
# =============================================================================

class ParetoFrontierGenerator:
    """
    Generate Pareto frontier for multi-objective optimization - ZERO HALLUCINATION.

    Identifies non-dominated solutions across multiple objectives
    using deterministic enumeration or sampling.
    """

    def __init__(
        self,
        objectives: List[ObjectiveFunction],
        seed: int = 42
    ):
        """
        Initialize Pareto frontier generator.

        Args:
            objectives: List of objective functions to optimize
            seed: Random seed for reproducibility
        """
        self.objectives = objectives
        self.seed = seed
        np.random.seed(seed)

    def generate(
        self,
        decision_bounds: Dict[str, Tuple[float, float]],
        context: Dict[str, Any],
        num_samples: int = 1000
    ) -> ParetoFrontier:
        """
        Generate Pareto frontier via sampling - DETERMINISTIC.

        Args:
            decision_bounds: Bounds for each decision variable
            context: Context for objective evaluation
            num_samples: Number of candidate solutions to sample

        Returns:
            ParetoFrontier with non-dominated solutions
        """
        np.random.seed(self.seed)
        logger.info("Generating Pareto frontier with %d samples", num_samples)

        # Generate candidate solutions
        candidates = []

        for _ in range(num_samples):
            decision_vars = {}
            for var_name, (lower, upper) in decision_bounds.items():
                decision_vars[var_name] = np.random.uniform(lower, upper)
            candidates.append(decision_vars)

        # Evaluate all objectives for each candidate
        evaluations = []

        for decision_vars in candidates:
            obj_values = {}
            for objective in self.objectives:
                result = objective.evaluate(decision_vars, context)
                obj_values[objective.name] = result.value
            evaluations.append((decision_vars, obj_values))

        # Find non-dominated solutions
        pareto_points = []
        objective_names = [obj.name for obj in self.objectives]

        for i, (decision_vars, obj_values) in enumerate(evaluations):
            is_dominated = False

            for j, (_, other_obj_values) in enumerate(evaluations):
                if i == j:
                    continue

                # Check if j dominates i
                dominates = True
                strictly_better = False

                for obj_name in objective_names:
                    obj = next(o for o in self.objectives if o.name == obj_name)

                    if obj.direction == OptimizationDirection.MINIMIZE:
                        if other_obj_values[obj_name] > obj_values[obj_name]:
                            dominates = False
                            break
                        if other_obj_values[obj_name] < obj_values[obj_name]:
                            strictly_better = True
                    else:  # MAXIMIZE
                        if other_obj_values[obj_name] < obj_values[obj_name]:
                            dominates = False
                            break
                        if other_obj_values[obj_name] > obj_values[obj_name]:
                            strictly_better = True

                if dominates and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_points.append(ParetoPoint(
                    decision_variables=decision_vars,
                    objective_values=obj_values,
                    is_dominated=False
                ))

        # Calculate ideal and nadir points
        ideal_point = {}
        nadir_point = {}

        for obj_name in objective_names:
            obj = next(o for o in self.objectives if o.name == obj_name)
            values = [p.objective_values[obj_name] for p in pareto_points]

            if obj.direction == OptimizationDirection.MINIMIZE:
                ideal_point[obj_name] = min(values) if values else 0
                nadir_point[obj_name] = max(values) if values else 0
            else:
                ideal_point[obj_name] = max(values) if values else 0
                nadir_point[obj_name] = min(values) if values else 0

        # Calculate hypervolume (simplified 2D approximation)
        hypervolume = self._calculate_hypervolume(pareto_points, nadir_point, objective_names)

        # Calculate spacing
        spacing = self._calculate_spacing(pareto_points, objective_names)

        return ParetoFrontier(
            points=pareto_points,
            objective_names=objective_names,
            ideal_point=ideal_point,
            nadir_point=nadir_point,
            hypervolume=hypervolume,
            spacing=spacing
        )

    def _calculate_hypervolume(
        self,
        points: List[ParetoPoint],
        reference: Dict[str, float],
        objective_names: List[str]
    ) -> float:
        """Calculate hypervolume indicator (2D approximation)."""
        if len(objective_names) != 2 or not points:
            return 0.0

        # Sort by first objective
        obj1, obj2 = objective_names[0], objective_names[1]
        sorted_points = sorted(points, key=lambda p: p.objective_values[obj1])

        hypervolume = 0.0
        prev_y = reference[obj2]

        for point in sorted_points:
            x = reference[obj1] - point.objective_values[obj1]
            y = prev_y - point.objective_values[obj2]

            if x > 0 and y > 0:
                hypervolume += x * y
                prev_y = point.objective_values[obj2]

        return hypervolume

    def _calculate_spacing(
        self,
        points: List[ParetoPoint],
        objective_names: List[str]
    ) -> float:
        """Calculate spacing metric (uniformity of Pareto front)."""
        if len(points) < 2:
            return 0.0

        distances = []

        for i, p1 in enumerate(points):
            min_dist = float('inf')

            for j, p2 in enumerate(points):
                if i == j:
                    continue

                dist = math.sqrt(sum(
                    (p1.objective_values[obj] - p2.objective_values[obj]) ** 2
                    for obj in objective_names
                ))
                min_dist = min(min_dist, dist)

            if min_dist < float('inf'):
                distances.append(min_dist)

        if not distances:
            return 0.0

        mean_dist = np.mean(distances)
        spacing = np.sqrt(np.mean((np.array(distances) - mean_dist) ** 2))

        return float(spacing)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_water_treatment_objectives(
    water_cost: float = 5.0,
    energy_cost: float = 0.12,
    max_blowdown_gpm: float = 50.0
) -> List[ObjectiveFunction]:
    """Create standard water treatment objectives."""
    return [
        WaterLossObjective(water_cost_per_1000gal=water_cost, max_blowdown_gpm=max_blowdown_gpm),
        EnergyLossObjective(energy_cost_per_kwh=energy_cost, max_blowdown_gpm=max_blowdown_gpm),
        ChemicalCostObjective(),
        RiskPenaltyObjective()
    ]


def create_weighted_objective(
    water_weight: float = 0.30,
    energy_weight: float = 0.25,
    chemical_weight: float = 0.20,
    risk_weight: float = 0.25
) -> WeightedSumObjective:
    """Create weighted sum objective with specified weights."""
    return WeightedSumObjective(
        water_loss_weight=water_weight,
        energy_loss_weight=energy_weight,
        chemical_cost_weight=chemical_weight,
        risk_penalty_weight=risk_weight
    )

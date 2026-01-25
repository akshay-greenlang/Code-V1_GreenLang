# -*- coding: utf-8 -*-
"""
What-If Analyzer for GL-017 CONDENSYNC

Scenario analysis and sensitivity analysis for condenser vacuum optimization.
Enables operators and engineers to explore operational scenarios and understand
parameter impacts on system performance.

Capabilities:
    - "What if CW flow increases by X%?"
    - "What if we add another CW pump?"
    - Sensitivity analysis for key parameters
    - Counterfactual recommendations
    - Multi-dimensional scenario exploration

Zero-Hallucination Guarantee:
    - All scenario calculations use deterministic physics models
    - No AI/ML inference in counterfactual analysis
    - Complete audit trail with provenance hashing
    - Reproducible results with identical inputs

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ScenarioType(Enum):
    """Type of what-if scenario."""
    PARAMETER_CHANGE = "parameter_change"  # Change a single parameter
    EQUIPMENT_CHANGE = "equipment_change"  # Add/remove equipment
    OPERATING_MODE = "operating_mode"  # Change operating mode
    AMBIENT_CONDITION = "ambient_condition"  # Weather/ambient changes
    FOULING_SCENARIO = "fouling_scenario"  # Cleanliness factor changes
    MULTI_PARAMETER = "multi_parameter"  # Multiple simultaneous changes


class ParameterType(Enum):
    """Type of parameter being analyzed."""
    CW_FLOW = "cw_flow"
    CW_INLET_TEMP = "cw_inlet_temp"
    STEAM_FLOW = "steam_flow"
    CLEANLINESS_FACTOR = "cleanliness_factor"
    AMBIENT_TEMP = "ambient_temp"
    WET_BULB_TEMP = "wet_bulb_temp"
    PUMP_COUNT = "pump_count"
    FAN_SPEED = "fan_speed"
    FAN_COUNT = "fan_count"


class SensitivityDirection(Enum):
    """Direction of sensitivity analysis."""
    INCREASE = "increase"
    DECREASE = "decrease"
    BOTH = "both"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class BaselineState:
    """Baseline operating state for comparison."""
    timestamp: datetime
    backpressure_inhga: float
    cw_flow_gpm: float
    cw_inlet_temp_f: float
    cw_outlet_temp_f: float
    steam_flow_klb_hr: float
    cleanliness_factor: float
    ambient_temp_f: float
    wet_bulb_temp_f: float

    # Equipment status
    num_pumps_running: int
    pump_power_kw: float
    num_fans_running: int
    total_fan_speed_pct: float
    fan_power_kw: float

    # Derived metrics
    total_power_kw: float
    hourly_cost_usd: float


@dataclass
class ScenarioDefinition:
    """Definition of a what-if scenario."""
    scenario_id: str
    name: str
    description: str
    scenario_type: ScenarioType

    # Parameter changes (absolute or percentage)
    parameter_changes: Dict[str, float] = field(default_factory=dict)
    percentage_changes: Dict[str, float] = field(default_factory=dict)

    # Equipment changes
    add_equipment: List[str] = field(default_factory=list)
    remove_equipment: List[str] = field(default_factory=list)


@dataclass
class ScenarioResult:
    """Result of evaluating a scenario."""
    scenario_id: str
    scenario_name: str
    timestamp: datetime

    # Resulting state
    backpressure_inhga: float
    cw_flow_gpm: float
    cw_inlet_temp_f: float
    pump_power_kw: float
    fan_power_kw: float
    total_power_kw: float
    hourly_cost_usd: float

    # Deltas from baseline
    delta_backpressure_inhga: float
    delta_power_kw: float
    delta_cost_usd_hr: float

    # Improvement indicators
    backpressure_improved: bool
    power_reduced: bool
    cost_reduced: bool

    # Feasibility
    is_feasible: bool
    constraint_violations: List[str]

    # Provenance
    provenance_hash: str


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a parameter."""
    parameter_name: str
    parameter_unit: str
    baseline_value: float

    # Sensitivity data points
    values: List[float]
    backpressure_results: List[float]
    power_results: List[float]
    cost_results: List[float]

    # Sensitivities (derivatives)
    backpressure_sensitivity: float  # inHgA per unit change
    power_sensitivity: float  # kW per unit change
    cost_sensitivity: float  # $/hr per unit change

    # Elasticities (percentage changes)
    backpressure_elasticity: float  # % BP change per % param change
    power_elasticity: float
    cost_elasticity: float

    # Optimal value (if found)
    optimal_value: Optional[float] = None
    optimal_reason: str = ""


@dataclass
class CounterfactualRecommendation:
    """Counterfactual recommendation from scenario analysis."""
    recommendation_id: str
    scenario_id: str
    scenario_name: str

    # Recommended changes
    recommended_changes: Dict[str, Any]

    # Expected improvements
    expected_bp_improvement_inhga: float
    expected_power_savings_kw: float
    expected_cost_savings_usd_hr: float

    # Implementation requirements
    requires_equipment_change: bool
    requires_outage: bool
    estimated_implementation_cost_usd: float

    # ROI estimate
    simple_payback_months: float

    # Confidence
    confidence_score: float


@dataclass
class WhatIfAnalyzerConfig:
    """Configuration for what-if analyzer."""
    # Sensitivity analysis settings
    sensitivity_steps: int = 11  # Number of points to evaluate
    sensitivity_range_pct: float = 20.0  # +/- percentage range

    # Cost parameters
    electricity_cost_per_kwh: float = 0.08
    backpressure_penalty_per_inhga: float = 50000.0
    target_backpressure_inhga: float = 2.5

    # Physical model parameters
    heat_rate_base_btu_kwh: float = 9500.0
    heat_rate_sensitivity_pct_per_inhga: float = 1.5

    # Constraint limits
    max_backpressure_inhga: float = 5.0
    min_cw_flow_gpm: float = 50000.0
    max_cw_flow_gpm: float = 200000.0


@dataclass
class WhatIfAnalysisResult:
    """Complete what-if analysis result."""
    result_id: str
    timestamp: datetime
    baseline: BaselineState

    # Scenario results
    scenario_results: List[ScenarioResult]

    # Sensitivity analyses
    sensitivity_results: Dict[str, SensitivityResult]

    # Counterfactual recommendations
    recommendations: List[CounterfactualRecommendation]

    # Summary
    best_scenario_id: Optional[str]
    max_potential_savings_usd_hr: float

    # Provenance
    provenance_hash: str
    processing_time_ms: float


# ============================================================================
# PHYSICS MODEL (Simplified for scenario analysis)
# ============================================================================

class CondenserPhysicsModel:
    """
    Simplified condenser physics model for scenario analysis.

    Uses correlation-based models for rapid scenario evaluation.
    These are approximations suitable for what-if analysis;
    detailed optimization uses the full VacuumOptimizer model.
    """

    def __init__(self, config: WhatIfAnalyzerConfig):
        """Initialize physics model."""
        self.config = config

    def calculate_backpressure(
        self,
        cw_flow_gpm: float,
        cw_inlet_temp_f: float,
        steam_flow_klb_hr: float,
        cleanliness_factor: float,
        baseline_bp: float = 2.5
    ) -> float:
        """
        Estimate backpressure from operating parameters.

        Uses empirical correlation:
        BP = BP_base * (CF_ref/CF) * (T_in/T_ref)^1.5 * (Q_steam/Q_ref)^0.5 / (Q_cw/Q_ref)^0.4

        Args:
            cw_flow_gpm: CW flow rate
            cw_inlet_temp_f: CW inlet temperature
            steam_flow_klb_hr: Steam flow to condenser
            cleanliness_factor: Condenser cleanliness (0-1)
            baseline_bp: Baseline backpressure at reference conditions

        Returns:
            Estimated backpressure in inHgA
        """
        # Reference values
        cw_flow_ref = 120000.0  # GPM
        cw_temp_ref = 75.0  # F
        steam_flow_ref = 2000.0  # klb/hr
        cf_ref = 0.85

        # Protect against division by zero
        cw_flow_gpm = max(cw_flow_gpm, 10000.0)
        cleanliness_factor = max(cleanliness_factor, 0.1)

        # Calculate relative factors
        cf_factor = cf_ref / cleanliness_factor
        temp_factor = (cw_inlet_temp_f / cw_temp_ref) ** 1.5
        steam_factor = (steam_flow_klb_hr / steam_flow_ref) ** 0.5
        flow_factor = (cw_flow_ref / cw_flow_gpm) ** 0.4

        bp = baseline_bp * cf_factor * temp_factor * steam_factor * flow_factor

        # Clamp to reasonable range
        return max(0.5, min(self.config.max_backpressure_inhga * 1.5, bp))

    def calculate_pump_power(
        self,
        cw_flow_gpm: float,
        num_pumps: int
    ) -> float:
        """
        Estimate pump power from flow and pump count.

        Uses affinity laws approximation:
        Power ~ Q^3 / n^2 (where n is number of pumps)

        Args:
            cw_flow_gpm: Total CW flow
            num_pumps: Number of pumps running

        Returns:
            Estimated pump power in kW
        """
        if num_pumps <= 0:
            return 0.0

        # Reference: 800 kW per pump at 40000 GPM
        rated_power_per_pump = 800.0
        rated_flow_per_pump = 40000.0

        flow_per_pump = cw_flow_gpm / num_pumps
        flow_ratio = flow_per_pump / rated_flow_per_pump

        # Affinity laws: Power ~ Q^3
        power_per_pump = rated_power_per_pump * (flow_ratio ** 2.5)

        return power_per_pump * num_pumps

    def calculate_fan_power(
        self,
        total_fan_speed_pct: float,
        num_fans: int
    ) -> float:
        """
        Estimate fan power from speed and count.

        Uses affinity laws: Power ~ speed^3

        Args:
            total_fan_speed_pct: Average fan speed percentage
            num_fans: Number of fans running

        Returns:
            Estimated fan power in kW
        """
        if num_fans <= 0 or total_fan_speed_pct <= 0:
            return 0.0

        # Reference: 150 kW per fan at 100%
        rated_power_per_fan = 150.0

        avg_speed = total_fan_speed_pct / 100.0
        power_per_fan = rated_power_per_fan * (avg_speed ** 3)

        return power_per_fan * num_fans

    def calculate_cw_inlet_temp(
        self,
        wet_bulb_temp_f: float,
        total_fan_speed_pct: float,
        num_fans: int,
        rated_fans: int = 8
    ) -> float:
        """
        Estimate CW inlet temperature from cooling tower performance.

        Uses approach temperature correlation.

        Args:
            wet_bulb_temp_f: Wet bulb temperature
            total_fan_speed_pct: Average fan speed
            num_fans: Number of fans running
            rated_fans: Total installed fans

        Returns:
            Estimated CW inlet temperature in F
        """
        # Base approach at design conditions
        base_approach = 8.0

        # Air flow ratio
        if rated_fans > 0 and num_fans > 0:
            air_ratio = (num_fans / rated_fans) * (total_fan_speed_pct / 100.0)
        else:
            air_ratio = 0.1

        # Approach increases as air flow decreases
        if air_ratio > 0.1:
            approach = base_approach / (air_ratio ** 0.6)
        else:
            approach = 30.0

        return wet_bulb_temp_f + approach

    def calculate_hourly_cost(
        self,
        pump_power_kw: float,
        fan_power_kw: float,
        backpressure_inhga: float
    ) -> float:
        """
        Calculate total hourly operating cost.

        Args:
            pump_power_kw: Pump power consumption
            fan_power_kw: Fan power consumption
            backpressure_inhga: Current backpressure

        Returns:
            Hourly cost in USD
        """
        # Electrical cost
        total_power = pump_power_kw + fan_power_kw
        power_cost = total_power * self.config.electricity_cost_per_kwh

        # Backpressure penalty
        bp_excess = max(0, backpressure_inhga - self.config.target_backpressure_inhga)
        bp_penalty = bp_excess * self.config.backpressure_penalty_per_inhga

        return power_cost + bp_penalty


# ============================================================================
# SCENARIO LIBRARY
# ============================================================================

class ScenarioLibrary:
    """Library of predefined scenarios for quick analysis."""

    @staticmethod
    def get_cw_flow_increase_scenario(pct_increase: float = 10.0) -> ScenarioDefinition:
        """Create CW flow increase scenario."""
        return ScenarioDefinition(
            scenario_id=f"CW_FLOW_INC_{int(pct_increase)}",
            name=f"CW Flow +{pct_increase}%",
            description=f"Increase CW flow by {pct_increase}%",
            scenario_type=ScenarioType.PARAMETER_CHANGE,
            percentage_changes={ParameterType.CW_FLOW.value: pct_increase}
        )

    @staticmethod
    def get_add_pump_scenario() -> ScenarioDefinition:
        """Create add CW pump scenario."""
        return ScenarioDefinition(
            scenario_id="ADD_CW_PUMP",
            name="Add CW Pump",
            description="Start one additional CW pump",
            scenario_type=ScenarioType.EQUIPMENT_CHANGE,
            add_equipment=["CW_PUMP"]
        )

    @staticmethod
    def get_add_fan_scenario() -> ScenarioDefinition:
        """Create add cooling tower fan scenario."""
        return ScenarioDefinition(
            scenario_id="ADD_CT_FAN",
            name="Add CT Fan",
            description="Start one additional cooling tower fan",
            scenario_type=ScenarioType.EQUIPMENT_CHANGE,
            add_equipment=["CT_FAN"]
        )

    @staticmethod
    def get_increase_fan_speed_scenario(pct_increase: float = 10.0) -> ScenarioDefinition:
        """Create fan speed increase scenario."""
        return ScenarioDefinition(
            scenario_id=f"FAN_SPEED_INC_{int(pct_increase)}",
            name=f"Fan Speed +{pct_increase}%",
            description=f"Increase average fan speed by {pct_increase}%",
            scenario_type=ScenarioType.PARAMETER_CHANGE,
            percentage_changes={ParameterType.FAN_SPEED.value: pct_increase}
        )

    @staticmethod
    def get_fouling_scenario(cf_value: float = 0.75) -> ScenarioDefinition:
        """Create fouling scenario."""
        return ScenarioDefinition(
            scenario_id=f"CF_{int(cf_value*100)}",
            name=f"Cleanliness Factor {cf_value}",
            description=f"What if CF degrades to {cf_value}",
            scenario_type=ScenarioType.FOULING_SCENARIO,
            parameter_changes={ParameterType.CLEANLINESS_FACTOR.value: cf_value}
        )

    @staticmethod
    def get_ambient_increase_scenario(temp_increase_f: float = 10.0) -> ScenarioDefinition:
        """Create ambient temperature increase scenario."""
        return ScenarioDefinition(
            scenario_id=f"AMBIENT_INC_{int(temp_increase_f)}F",
            name=f"Ambient +{temp_increase_f}F",
            description=f"Ambient temperature increases by {temp_increase_f}F",
            scenario_type=ScenarioType.AMBIENT_CONDITION,
            parameter_changes={
                ParameterType.AMBIENT_TEMP.value: temp_increase_f,
                ParameterType.WET_BULB_TEMP.value: temp_increase_f * 0.7  # Approximate
            }
        )

    @staticmethod
    def get_standard_scenarios() -> List[ScenarioDefinition]:
        """Get list of standard analysis scenarios."""
        return [
            ScenarioLibrary.get_cw_flow_increase_scenario(5.0),
            ScenarioLibrary.get_cw_flow_increase_scenario(10.0),
            ScenarioLibrary.get_cw_flow_increase_scenario(-10.0),
            ScenarioLibrary.get_add_pump_scenario(),
            ScenarioLibrary.get_add_fan_scenario(),
            ScenarioLibrary.get_increase_fan_speed_scenario(10.0),
            ScenarioLibrary.get_fouling_scenario(0.80),
            ScenarioLibrary.get_fouling_scenario(0.70),
            ScenarioLibrary.get_ambient_increase_scenario(10.0),
        ]


# ============================================================================
# WHAT-IF ANALYZER CLASS
# ============================================================================

class WhatIfAnalyzer:
    """
    What-If Analyzer for condenser vacuum system.

    Enables scenario exploration and sensitivity analysis to help
    operators and engineers understand system behavior and identify
    optimization opportunities.

    Zero-Hallucination Guarantee:
        - All calculations use deterministic physics correlations
        - No AI/ML inference in scenario evaluation
        - Reproducible results with identical inputs

    Example:
        >>> analyzer = WhatIfAnalyzer()
        >>> baseline = create_baseline_from_current_state(state)
        >>> result = analyzer.analyze_scenario(
        ...     baseline,
        ...     ScenarioLibrary.get_add_pump_scenario()
        ... )
        >>> print(f"BP improvement: {result.delta_backpressure_inhga:.2f} inHgA")
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[WhatIfAnalyzerConfig] = None):
        """
        Initialize what-if analyzer.

        Args:
            config: Analyzer configuration
        """
        self.config = config or WhatIfAnalyzerConfig()
        self.physics = CondenserPhysicsModel(self.config)

        logger.info("WhatIfAnalyzer initialized")

    # =========================================================================
    # SCENARIO ANALYSIS
    # =========================================================================

    def analyze_scenario(
        self,
        baseline: BaselineState,
        scenario: ScenarioDefinition
    ) -> ScenarioResult:
        """
        Analyze a single what-if scenario.

        Args:
            baseline: Baseline operating state
            scenario: Scenario to evaluate

        Returns:
            ScenarioResult with predicted outcomes
        """
        timestamp = datetime.now(timezone.utc)

        # Apply parameter changes
        modified_params = self._apply_scenario_changes(baseline, scenario)

        # Calculate new operating point
        new_bp = self.physics.calculate_backpressure(
            modified_params["cw_flow_gpm"],
            modified_params["cw_inlet_temp_f"],
            modified_params["steam_flow_klb_hr"],
            modified_params["cleanliness_factor"],
            baseline.backpressure_inhga
        )

        new_pump_power = self.physics.calculate_pump_power(
            modified_params["cw_flow_gpm"],
            modified_params["num_pumps"]
        )

        new_fan_power = self.physics.calculate_fan_power(
            modified_params["total_fan_speed_pct"],
            modified_params["num_fans"]
        )

        new_total_power = new_pump_power + new_fan_power

        new_cost = self.physics.calculate_hourly_cost(
            new_pump_power, new_fan_power, new_bp
        )

        # Calculate deltas
        delta_bp = new_bp - baseline.backpressure_inhga
        delta_power = new_total_power - baseline.total_power_kw
        delta_cost = new_cost - baseline.hourly_cost_usd

        # Check feasibility
        violations = self._check_constraints(modified_params, new_bp)

        # Calculate provenance
        provenance_data = {
            "scenario_id": scenario.scenario_id,
            "baseline_bp": round(baseline.backpressure_inhga, 3),
            "result_bp": round(new_bp, 3),
            "delta_cost": round(delta_cost, 2)
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            timestamp=timestamp,
            backpressure_inhga=new_bp,
            cw_flow_gpm=modified_params["cw_flow_gpm"],
            cw_inlet_temp_f=modified_params["cw_inlet_temp_f"],
            pump_power_kw=new_pump_power,
            fan_power_kw=new_fan_power,
            total_power_kw=new_total_power,
            hourly_cost_usd=new_cost,
            delta_backpressure_inhga=delta_bp,
            delta_power_kw=delta_power,
            delta_cost_usd_hr=delta_cost,
            backpressure_improved=(delta_bp < 0),
            power_reduced=(delta_power < 0),
            cost_reduced=(delta_cost < 0),
            is_feasible=(len(violations) == 0),
            constraint_violations=violations,
            provenance_hash=provenance_hash
        )

    def _apply_scenario_changes(
        self,
        baseline: BaselineState,
        scenario: ScenarioDefinition
    ) -> Dict[str, Any]:
        """Apply scenario changes to baseline parameters."""
        params = {
            "cw_flow_gpm": baseline.cw_flow_gpm,
            "cw_inlet_temp_f": baseline.cw_inlet_temp_f,
            "steam_flow_klb_hr": baseline.steam_flow_klb_hr,
            "cleanliness_factor": baseline.cleanliness_factor,
            "ambient_temp_f": baseline.ambient_temp_f,
            "wet_bulb_temp_f": baseline.wet_bulb_temp_f,
            "num_pumps": baseline.num_pumps_running,
            "total_fan_speed_pct": baseline.total_fan_speed_pct,
            "num_fans": baseline.num_fans_running
        }

        # Apply percentage changes
        for param_type, pct_change in scenario.percentage_changes.items():
            if param_type == ParameterType.CW_FLOW.value:
                params["cw_flow_gpm"] *= (1 + pct_change / 100)
            elif param_type == ParameterType.FAN_SPEED.value:
                params["total_fan_speed_pct"] *= (1 + pct_change / 100)
                params["total_fan_speed_pct"] = min(100, params["total_fan_speed_pct"])

        # Apply absolute changes
        for param_type, value in scenario.parameter_changes.items():
            if param_type == ParameterType.CLEANLINESS_FACTOR.value:
                params["cleanliness_factor"] = value
            elif param_type == ParameterType.AMBIENT_TEMP.value:
                params["ambient_temp_f"] += value
            elif param_type == ParameterType.WET_BULB_TEMP.value:
                params["wet_bulb_temp_f"] += value

        # Apply equipment changes
        for equip in scenario.add_equipment:
            if equip == "CW_PUMP":
                params["num_pumps"] += 1
                # Assume flow increases proportionally
                flow_increase_factor = params["num_pumps"] / baseline.num_pumps_running
                params["cw_flow_gpm"] *= min(flow_increase_factor, 1.25)
            elif equip == "CT_FAN":
                params["num_fans"] += 1

        for equip in scenario.remove_equipment:
            if equip == "CW_PUMP" and params["num_pumps"] > 1:
                params["num_pumps"] -= 1
                flow_decrease_factor = params["num_pumps"] / baseline.num_pumps_running
                params["cw_flow_gpm"] *= flow_decrease_factor
            elif equip == "CT_FAN" and params["num_fans"] > 0:
                params["num_fans"] -= 1

        # Recalculate CW inlet temp if fans changed
        if scenario.add_equipment or scenario.remove_equipment:
            if "CT_FAN" in scenario.add_equipment or "CT_FAN" in scenario.remove_equipment:
                params["cw_inlet_temp_f"] = self.physics.calculate_cw_inlet_temp(
                    params["wet_bulb_temp_f"],
                    params["total_fan_speed_pct"],
                    params["num_fans"]
                )

        return params

    def _check_constraints(
        self,
        params: Dict[str, Any],
        backpressure: float
    ) -> List[str]:
        """Check if parameters violate constraints."""
        violations = []

        if backpressure > self.config.max_backpressure_inhga:
            violations.append(
                f"Backpressure {backpressure:.2f} exceeds max {self.config.max_backpressure_inhga}"
            )

        if params["cw_flow_gpm"] < self.config.min_cw_flow_gpm:
            violations.append(
                f"CW flow {params['cw_flow_gpm']:.0f} below min {self.config.min_cw_flow_gpm}"
            )

        if params["cw_flow_gpm"] > self.config.max_cw_flow_gpm:
            violations.append(
                f"CW flow {params['cw_flow_gpm']:.0f} exceeds max {self.config.max_cw_flow_gpm}"
            )

        if params["num_pumps"] < 1:
            violations.append("Must have at least 1 pump running")

        return violations

    # =========================================================================
    # SENSITIVITY ANALYSIS
    # =========================================================================

    def analyze_sensitivity(
        self,
        baseline: BaselineState,
        parameter: ParameterType,
        range_pct: Optional[float] = None
    ) -> SensitivityResult:
        """
        Perform sensitivity analysis for a parameter.

        Args:
            baseline: Baseline operating state
            parameter: Parameter to analyze
            range_pct: Range as +/- percentage (default from config)

        Returns:
            SensitivityResult with sensitivity data
        """
        if range_pct is None:
            range_pct = self.config.sensitivity_range_pct

        # Get baseline value
        baseline_value = self._get_parameter_value(baseline, parameter)

        if baseline_value is None or baseline_value == 0:
            logger.warning(f"Cannot analyze sensitivity for {parameter}: invalid baseline")
            return SensitivityResult(
                parameter_name=parameter.value,
                parameter_unit=self._get_parameter_unit(parameter),
                baseline_value=0,
                values=[],
                backpressure_results=[],
                power_results=[],
                cost_results=[],
                backpressure_sensitivity=0,
                power_sensitivity=0,
                cost_sensitivity=0,
                backpressure_elasticity=0,
                power_elasticity=0,
                cost_elasticity=0
            )

        # Generate test values
        min_val = baseline_value * (1 - range_pct / 100)
        max_val = baseline_value * (1 + range_pct / 100)
        values = np.linspace(min_val, max_val, self.config.sensitivity_steps).tolist()

        # Evaluate at each point
        bp_results = []
        power_results = []
        cost_results = []

        for val in values:
            result = self._evaluate_at_parameter_value(baseline, parameter, val)
            bp_results.append(result["backpressure"])
            power_results.append(result["power"])
            cost_results.append(result["cost"])

        # Calculate sensitivities (using linear regression)
        bp_sensitivity = self._calculate_sensitivity(values, bp_results)
        power_sensitivity = self._calculate_sensitivity(values, power_results)
        cost_sensitivity = self._calculate_sensitivity(values, cost_results)

        # Calculate elasticities
        baseline_bp = baseline.backpressure_inhga
        baseline_power = baseline.total_power_kw
        baseline_cost = baseline.hourly_cost_usd

        bp_elasticity = (bp_sensitivity * baseline_value / baseline_bp
                        if baseline_bp != 0 else 0)
        power_elasticity = (power_sensitivity * baseline_value / baseline_power
                           if baseline_power != 0 else 0)
        cost_elasticity = (cost_sensitivity * baseline_value / baseline_cost
                          if baseline_cost != 0 else 0)

        # Find optimal (minimum cost point)
        min_cost_idx = np.argmin(cost_results)
        optimal_value = values[min_cost_idx]
        optimal_reason = f"Minimum cost ${cost_results[min_cost_idx]:.2f}/hr"

        return SensitivityResult(
            parameter_name=parameter.value,
            parameter_unit=self._get_parameter_unit(parameter),
            baseline_value=baseline_value,
            values=values,
            backpressure_results=bp_results,
            power_results=power_results,
            cost_results=cost_results,
            backpressure_sensitivity=bp_sensitivity,
            power_sensitivity=power_sensitivity,
            cost_sensitivity=cost_sensitivity,
            backpressure_elasticity=bp_elasticity,
            power_elasticity=power_elasticity,
            cost_elasticity=cost_elasticity,
            optimal_value=optimal_value,
            optimal_reason=optimal_reason
        )

    def _get_parameter_value(
        self,
        baseline: BaselineState,
        parameter: ParameterType
    ) -> Optional[float]:
        """Get parameter value from baseline state."""
        mapping = {
            ParameterType.CW_FLOW: baseline.cw_flow_gpm,
            ParameterType.CW_INLET_TEMP: baseline.cw_inlet_temp_f,
            ParameterType.STEAM_FLOW: baseline.steam_flow_klb_hr,
            ParameterType.CLEANLINESS_FACTOR: baseline.cleanliness_factor,
            ParameterType.AMBIENT_TEMP: baseline.ambient_temp_f,
            ParameterType.WET_BULB_TEMP: baseline.wet_bulb_temp_f,
            ParameterType.PUMP_COUNT: float(baseline.num_pumps_running),
            ParameterType.FAN_SPEED: baseline.total_fan_speed_pct,
            ParameterType.FAN_COUNT: float(baseline.num_fans_running)
        }
        return mapping.get(parameter)

    def _get_parameter_unit(self, parameter: ParameterType) -> str:
        """Get unit for parameter."""
        units = {
            ParameterType.CW_FLOW: "GPM",
            ParameterType.CW_INLET_TEMP: "F",
            ParameterType.STEAM_FLOW: "klb/hr",
            ParameterType.CLEANLINESS_FACTOR: "-",
            ParameterType.AMBIENT_TEMP: "F",
            ParameterType.WET_BULB_TEMP: "F",
            ParameterType.PUMP_COUNT: "count",
            ParameterType.FAN_SPEED: "%",
            ParameterType.FAN_COUNT: "count"
        }
        return units.get(parameter, "")

    def _evaluate_at_parameter_value(
        self,
        baseline: BaselineState,
        parameter: ParameterType,
        value: float
    ) -> Dict[str, float]:
        """Evaluate system at a specific parameter value."""
        # Start with baseline values
        cw_flow = baseline.cw_flow_gpm
        cw_temp = baseline.cw_inlet_temp_f
        steam_flow = baseline.steam_flow_klb_hr
        cf = baseline.cleanliness_factor
        num_pumps = baseline.num_pumps_running
        fan_speed = baseline.total_fan_speed_pct
        num_fans = baseline.num_fans_running

        # Override the specific parameter
        if parameter == ParameterType.CW_FLOW:
            cw_flow = value
        elif parameter == ParameterType.CW_INLET_TEMP:
            cw_temp = value
        elif parameter == ParameterType.STEAM_FLOW:
            steam_flow = value
        elif parameter == ParameterType.CLEANLINESS_FACTOR:
            cf = value
        elif parameter == ParameterType.FAN_SPEED:
            fan_speed = value
        elif parameter == ParameterType.PUMP_COUNT:
            num_pumps = int(round(value))
        elif parameter == ParameterType.FAN_COUNT:
            num_fans = int(round(value))

        # Calculate results
        bp = self.physics.calculate_backpressure(
            cw_flow, cw_temp, steam_flow, cf, baseline.backpressure_inhga
        )
        pump_power = self.physics.calculate_pump_power(cw_flow, num_pumps)
        fan_power = self.physics.calculate_fan_power(fan_speed, num_fans)
        total_power = pump_power + fan_power
        cost = self.physics.calculate_hourly_cost(pump_power, fan_power, bp)

        return {
            "backpressure": bp,
            "power": total_power,
            "cost": cost
        }

    def _calculate_sensitivity(
        self,
        x: List[float],
        y: List[float]
    ) -> float:
        """Calculate linear sensitivity (slope) from data points."""
        if len(x) < 2 or len(y) < 2:
            return 0.0

        x_arr = np.array(x)
        y_arr = np.array(y)

        # Simple linear regression
        x_mean = np.mean(x_arr)
        y_mean = np.mean(y_arr)

        numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
        denominator = np.sum((x_arr - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # =========================================================================
    # COUNTERFACTUAL RECOMMENDATIONS
    # =========================================================================

    def generate_counterfactual_recommendations(
        self,
        baseline: BaselineState,
        scenario_results: List[ScenarioResult]
    ) -> List[CounterfactualRecommendation]:
        """
        Generate actionable recommendations from scenario results.

        Args:
            baseline: Baseline state
            scenario_results: Results from scenario analysis

        Returns:
            List of recommendations sorted by potential savings
        """
        recommendations = []

        for result in scenario_results:
            if not result.is_feasible:
                continue

            if result.delta_cost_usd_hr >= 0:
                continue  # No savings

            # Determine implementation requirements
            requires_equipment = result.scenario_id in ["ADD_CW_PUMP", "ADD_CT_FAN"]
            requires_outage = requires_equipment

            # Estimate implementation cost
            if "ADD_CW_PUMP" in result.scenario_id:
                impl_cost = 500000.0  # New pump installation
            elif "ADD_CT_FAN" in result.scenario_id:
                impl_cost = 100000.0  # New fan installation
            else:
                impl_cost = 0.0  # Operational change only

            # Calculate payback
            annual_savings = -result.delta_cost_usd_hr * 8760 * 0.85  # 85% CF
            if annual_savings > 0:
                payback_months = (impl_cost / annual_savings) * 12
            else:
                payback_months = 999

            # Generate recommendation ID
            rec_id = f"REC-{result.scenario_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

            rec = CounterfactualRecommendation(
                recommendation_id=rec_id,
                scenario_id=result.scenario_id,
                scenario_name=result.scenario_name,
                recommended_changes=self._extract_changes(result, baseline),
                expected_bp_improvement_inhga=-result.delta_backpressure_inhga,
                expected_power_savings_kw=-result.delta_power_kw,
                expected_cost_savings_usd_hr=-result.delta_cost_usd_hr,
                requires_equipment_change=requires_equipment,
                requires_outage=requires_outage,
                estimated_implementation_cost_usd=impl_cost,
                simple_payback_months=payback_months,
                confidence_score=0.85 if result.is_feasible else 0.5
            )

            recommendations.append(rec)

        # Sort by savings
        recommendations.sort(key=lambda r: r.expected_cost_savings_usd_hr, reverse=True)

        return recommendations

    def _extract_changes(
        self,
        result: ScenarioResult,
        baseline: BaselineState
    ) -> Dict[str, Any]:
        """Extract parameter changes from scenario result."""
        changes = {}

        if abs(result.cw_flow_gpm - baseline.cw_flow_gpm) > 100:
            changes["cw_flow_gpm"] = {
                "from": baseline.cw_flow_gpm,
                "to": result.cw_flow_gpm,
                "change": result.cw_flow_gpm - baseline.cw_flow_gpm
            }

        return changes

    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================

    def run_comprehensive_analysis(
        self,
        baseline: BaselineState,
        scenarios: Optional[List[ScenarioDefinition]] = None,
        parameters_to_analyze: Optional[List[ParameterType]] = None
    ) -> WhatIfAnalysisResult:
        """
        Run comprehensive what-if analysis.

        Args:
            baseline: Baseline operating state
            scenarios: Scenarios to evaluate (default: standard set)
            parameters_to_analyze: Parameters for sensitivity (default: key parameters)

        Returns:
            Complete WhatIfAnalysisResult
        """
        start_time = datetime.now(timezone.utc)

        logger.info("Starting comprehensive what-if analysis")

        # Use defaults if not specified
        if scenarios is None:
            scenarios = ScenarioLibrary.get_standard_scenarios()

        if parameters_to_analyze is None:
            parameters_to_analyze = [
                ParameterType.CW_FLOW,
                ParameterType.CLEANLINESS_FACTOR,
                ParameterType.FAN_SPEED
            ]

        # Run scenario analyses
        scenario_results = []
        for scenario in scenarios:
            result = self.analyze_scenario(baseline, scenario)
            scenario_results.append(result)

        # Run sensitivity analyses
        sensitivity_results = {}
        for param in parameters_to_analyze:
            sens = self.analyze_sensitivity(baseline, param)
            sensitivity_results[param.value] = sens

        # Generate recommendations
        recommendations = self.generate_counterfactual_recommendations(
            baseline, scenario_results
        )

        # Find best scenario
        feasible_results = [r for r in scenario_results if r.is_feasible]
        if feasible_results:
            best = min(feasible_results, key=lambda r: r.hourly_cost_usd)
            best_scenario_id = best.scenario_id
            max_savings = baseline.hourly_cost_usd - best.hourly_cost_usd
        else:
            best_scenario_id = None
            max_savings = 0.0

        # Calculate provenance
        provenance_data = {
            "version": self.VERSION,
            "timestamp": start_time.isoformat(),
            "num_scenarios": len(scenario_results),
            "num_sensitivities": len(sensitivity_results),
            "best_scenario": best_scenario_id
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Analysis complete in {processing_time:.1f}ms. "
            f"Best scenario: {best_scenario_id}, max savings: ${max_savings:.2f}/hr"
        )

        return WhatIfAnalysisResult(
            result_id=f"WIA-{start_time.strftime('%Y%m%d%H%M%S')}-{provenance_hash}",
            timestamp=start_time,
            baseline=baseline,
            scenario_results=scenario_results,
            sensitivity_results=sensitivity_results,
            recommendations=recommendations,
            best_scenario_id=best_scenario_id,
            max_potential_savings_usd_hr=max_savings,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_analysis_report(
        self,
        result: WhatIfAnalysisResult
    ) -> str:
        """
        Generate human-readable analysis report.

        Args:
            result: What-if analysis result

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "           WHAT-IF ANALYSIS REPORT",
            "=" * 70,
            f"Analysis ID: {result.result_id}",
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "BASELINE STATE:",
            f"  Backpressure: {result.baseline.backpressure_inhga:.2f} inHgA",
            f"  CW Flow: {result.baseline.cw_flow_gpm:,.0f} GPM",
            f"  Total Power: {result.baseline.total_power_kw:.0f} kW",
            f"  Hourly Cost: ${result.baseline.hourly_cost_usd:.2f}/hr",
            "",
            "SCENARIO RESULTS:",
        ]

        for sr in result.scenario_results[:5]:
            status = "FEASIBLE" if sr.is_feasible else "INFEASIBLE"
            improvement = "YES" if sr.cost_reduced else "NO"
            lines.append(
                f"  {sr.scenario_name}: BP={sr.backpressure_inhga:.2f}, "
                f"Cost=${sr.hourly_cost_usd:.2f}, Improved={improvement} [{status}]"
            )

        lines.extend([
            "",
            "SENSITIVITY SUMMARY:",
        ])

        for param_name, sens in result.sensitivity_results.items():
            lines.append(
                f"  {param_name}: BP sensitivity={sens.backpressure_sensitivity:.4f}, "
                f"Cost sensitivity={sens.cost_sensitivity:.4f}"
            )

        if result.recommendations:
            lines.extend([
                "",
                "TOP RECOMMENDATIONS:",
            ])
            for rec in result.recommendations[:3]:
                lines.append(
                    f"  {rec.scenario_name}: Save ${rec.expected_cost_savings_usd_hr:.2f}/hr, "
                    f"Payback={rec.simple_payback_months:.1f} months"
                )

        lines.extend([
            "",
            f"Best Scenario: {result.best_scenario_id}",
            f"Max Potential Savings: ${result.max_potential_savings_usd_hr:.2f}/hr",
            f"Processing Time: {result.processing_time_ms:.1f}ms",
            f"Provenance: {result.provenance_hash}",
            "=" * 70
        ])

        return "\n".join(lines)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_analyzer() -> WhatIfAnalyzer:
    """
    Create what-if analyzer with default configuration.

    Returns:
        Configured WhatIfAnalyzer instance
    """
    return WhatIfAnalyzer()


def create_baseline_from_dict(data: Dict[str, Any]) -> BaselineState:
    """
    Create baseline state from dictionary.

    Args:
        data: Dictionary with state values

    Returns:
        BaselineState instance
    """
    return BaselineState(
        timestamp=data.get("timestamp", datetime.now(timezone.utc)),
        backpressure_inhga=data.get("backpressure_inhga", 2.5),
        cw_flow_gpm=data.get("cw_flow_gpm", 120000),
        cw_inlet_temp_f=data.get("cw_inlet_temp_f", 75),
        cw_outlet_temp_f=data.get("cw_outlet_temp_f", 95),
        steam_flow_klb_hr=data.get("steam_flow_klb_hr", 2000),
        cleanliness_factor=data.get("cleanliness_factor", 0.85),
        ambient_temp_f=data.get("ambient_temp_f", 85),
        wet_bulb_temp_f=data.get("wet_bulb_temp_f", 70),
        num_pumps_running=data.get("num_pumps_running", 3),
        pump_power_kw=data.get("pump_power_kw", 2000),
        num_fans_running=data.get("num_fans_running", 6),
        total_fan_speed_pct=data.get("total_fan_speed_pct", 75),
        fan_power_kw=data.get("fan_power_kw", 500),
        total_power_kw=data.get("total_power_kw", 2500),
        hourly_cost_usd=data.get("hourly_cost_usd", 200)
    )

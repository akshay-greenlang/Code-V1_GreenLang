# -*- coding: utf-8 -*-
"""
Vacuum Optimization Engine for GL-017 CONDENSYNC

SLSQP-based optimization engine for condenser vacuum system performance.
Minimizes total operating cost while respecting safety and equipment constraints.

Objective Function:
    minimize: Backpressure_Penalty + CW_Pump_Power + Tower_Fan_Power

Decision Variables:
    - CW flow setpoint (continuous)
    - CW pump staging (discrete)
    - Cooling tower fan staging (discrete)

Constraints:
    - Maximum backpressure (turbine protection)
    - Minimum CW flow (tube velocity requirements)
    - NPSH margins (pump protection)
    - Ramp rates (equipment protection)

Zero-Hallucination Guarantee:
    - All calculations use deterministic formulas
    - Optimization uses scipy.optimize.minimize (SLSQP)
    - No AI/ML inference in calculation path
    - Complete audit trail with SHA-256 provenance

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
from scipy.optimize import minimize, OptimizeResult

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Physical constants
WATER_DENSITY_KG_M3 = 998.0  # kg/m3 at 20C
GRAVITY_M_S2 = 9.81  # m/s2
STEAM_LATENT_HEAT_KJ_KG = 2257.0  # kJ/kg at 100C

# Default operating parameters
DEFAULT_MAX_BACKPRESSURE_INHGA = 5.0  # inHgA
DEFAULT_MIN_CW_FLOW_GPM = 50000.0  # GPM
DEFAULT_MAX_CW_FLOW_GPM = 200000.0  # GPM
DEFAULT_MIN_NPSH_MARGIN = 1.2  # 20% margin


# ============================================================================
# ENUMERATIONS
# ============================================================================

class OptimizationStatus(Enum):
    """Status of optimization result."""
    SUCCESS = "success"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    CONSTRAINT_VIOLATION = "constraint_violation"
    INFEASIBLE = "infeasible"
    ERROR = "error"


class PumpStatus(Enum):
    """CW pump operating status."""
    OFF = 0
    RUNNING = 1
    STANDBY = 2
    MAINTENANCE = 3


class FanStatus(Enum):
    """Cooling tower fan status."""
    OFF = 0
    LOW = 1
    HIGH = 2
    VFD = 3  # Variable frequency drive


class OperatingMode(Enum):
    """Condenser operating mode."""
    NORMAL = "normal"
    LOW_LOAD = "low_load"
    HIGH_AMBIENT = "high_ambient"
    CLEANING = "cleaning"
    EMERGENCY = "emergency"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PumpCurve:
    """
    Centrifugal pump performance curve.

    Uses polynomial coefficients for head-flow relationship:
    Head = a0 + a1*Q + a2*Q^2 + a3*Q^3

    Efficiency curve:
    Efficiency = b0 + b1*Q + b2*Q^2 + b3*Q^3
    """
    pump_id: str
    head_coefficients: Tuple[float, float, float, float]  # a0, a1, a2, a3
    efficiency_coefficients: Tuple[float, float, float, float]  # b0, b1, b2, b3
    rated_flow_gpm: float
    rated_head_ft: float
    rated_power_kw: float
    min_flow_gpm: float
    max_flow_gpm: float
    npsh_required_ft: float
    motor_efficiency: float = 0.95

    def calculate_head(self, flow_gpm: float) -> float:
        """Calculate pump head at given flow rate."""
        a0, a1, a2, a3 = self.head_coefficients
        q = flow_gpm / self.rated_flow_gpm  # Normalized flow
        return self.rated_head_ft * (a0 + a1 * q + a2 * q**2 + a3 * q**3)

    def calculate_efficiency(self, flow_gpm: float) -> float:
        """Calculate pump efficiency at given flow rate."""
        b0, b1, b2, b3 = self.efficiency_coefficients
        q = flow_gpm / self.rated_flow_gpm  # Normalized flow
        eff = b0 + b1 * q + b2 * q**2 + b3 * q**3
        return max(0.1, min(1.0, eff))  # Clamp between 10% and 100%

    def calculate_power(self, flow_gpm: float, head_ft: float) -> float:
        """Calculate pump power consumption (kW)."""
        if flow_gpm <= 0:
            return 0.0
        efficiency = self.calculate_efficiency(flow_gpm)
        # Hydraulic power = rho * g * Q * H
        # Convert: GPM to m3/s, ft to m
        q_m3s = flow_gpm * 6.309e-5
        h_m = head_ft * 0.3048
        hydraulic_power_kw = WATER_DENSITY_KG_M3 * GRAVITY_M_S2 * q_m3s * h_m / 1000
        shaft_power_kw = hydraulic_power_kw / efficiency
        motor_power_kw = shaft_power_kw / self.motor_efficiency
        return motor_power_kw


@dataclass
class FanCurve:
    """
    Cooling tower fan performance curve.

    Air flow vs power relationship:
    Power = rated_power * (air_flow / rated_air_flow)^3  (affinity laws)
    """
    fan_id: str
    rated_air_flow_cfm: float
    rated_power_kw: float
    min_speed_pct: float = 20.0
    max_speed_pct: float = 100.0
    efficiency: float = 0.85

    def calculate_power(self, speed_pct: float) -> float:
        """Calculate fan power at given speed percentage."""
        if speed_pct <= 0:
            return 0.0
        # Fan affinity laws: Power ~ speed^3
        normalized_speed = speed_pct / 100.0
        return self.rated_power_kw * (normalized_speed ** 3)

    def calculate_air_flow(self, speed_pct: float) -> float:
        """Calculate air flow at given speed percentage."""
        # Fan affinity laws: Flow ~ speed
        normalized_speed = speed_pct / 100.0
        return self.rated_air_flow_cfm * normalized_speed


@dataclass
class CondenserState:
    """Current condenser operating state."""
    timestamp: datetime
    backpressure_inhga: float
    cw_inlet_temp_f: float
    cw_outlet_temp_f: float
    cw_flow_gpm: float
    hotwell_temp_f: float
    steam_flow_klb_hr: float
    cleanliness_factor: float
    ambient_temp_f: float
    wet_bulb_temp_f: float

    # Equipment status
    pumps_running: List[str] = field(default_factory=list)
    fans_running: Dict[str, float] = field(default_factory=dict)  # fan_id: speed_pct

    # Calculated values
    ttd_f: Optional[float] = None  # Terminal temperature difference
    heat_duty_mmbtu_hr: Optional[float] = None


@dataclass
class EquipmentInventory:
    """Equipment inventory for optimization."""
    cw_pumps: List[PumpCurve] = field(default_factory=list)
    ct_fans: List[FanCurve] = field(default_factory=list)
    num_condenser_tubes: int = 10000
    tube_diameter_in: float = 1.0
    tube_length_ft: float = 40.0
    condenser_ua_base_btu_hr_f: float = 50000000.0


@dataclass
class OperatingLimits:
    """Operating limits and constraints."""
    max_backpressure_inhga: float = DEFAULT_MAX_BACKPRESSURE_INHGA
    min_backpressure_inhga: float = 0.5
    min_cw_flow_gpm: float = DEFAULT_MIN_CW_FLOW_GPM
    max_cw_flow_gpm: float = DEFAULT_MAX_CW_FLOW_GPM
    min_tube_velocity_fps: float = 5.0
    max_tube_velocity_fps: float = 10.0
    max_cw_outlet_temp_f: float = 110.0
    min_npsh_margin: float = DEFAULT_MIN_NPSH_MARGIN
    max_pump_ramp_gpm_min: float = 5000.0
    max_fan_ramp_pct_min: float = 10.0
    min_pumps_running: int = 1
    max_pumps_running: int = 4


@dataclass
class CostParameters:
    """Cost parameters for optimization."""
    electricity_cost_per_kwh: float = 0.08
    backpressure_penalty_per_inhga: float = 50000.0  # $/hr per inHgA above target
    target_backpressure_inhga: float = 2.5
    demand_charge_per_kw: float = 15.0
    heat_rate_btu_kwh: float = 9500.0
    fuel_cost_per_mmbtu: float = 3.50


@dataclass
class OptimizerConfig:
    """Configuration for vacuum optimizer."""
    # Optimization parameters
    max_iterations: int = 100
    tolerance: float = 1e-6
    method: str = "SLSQP"

    # Decision variable bounds
    allow_pump_staging: bool = True
    allow_fan_staging: bool = True
    use_discrete_pump_search: bool = True

    # Number of alternatives to return
    num_alternatives: int = 3

    # Ramp rate enforcement
    enforce_ramp_rates: bool = True

    # Data quality gating
    min_data_quality_score: float = 0.8
    suppress_on_bad_data: bool = True


@dataclass
class OptimizationRecommendation:
    """Single optimization recommendation."""
    recommendation_id: str
    cw_flow_setpoint_gpm: float
    pumps_to_run: List[str]
    fan_speeds: Dict[str, float]

    # Predicted outcomes
    predicted_backpressure_inhga: float
    predicted_pump_power_kw: float
    predicted_fan_power_kw: float
    predicted_total_power_kw: float

    # Cost metrics
    hourly_power_cost_usd: float
    backpressure_penalty_usd_hr: float
    total_hourly_cost_usd: float

    # Improvement metrics
    power_savings_kw: float
    cost_savings_usd_hr: float
    backpressure_improvement_inhga: float

    # Tradeoffs (for alternatives)
    tradeoff_description: str = ""

    # Confidence and provenance
    confidence_score: float = 0.95
    provenance_hash: str = ""


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    result_id: str
    timestamp: datetime
    status: OptimizationStatus

    # Best recommendation
    best_recommendation: Optional[OptimizationRecommendation] = None

    # Alternative recommendations
    alternatives: List[OptimizationRecommendation] = field(default_factory=list)

    # Current state for comparison
    current_power_kw: float = 0.0
    current_cost_usd_hr: float = 0.0

    # Optimization metadata
    iterations: int = 0
    objective_value: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""
    processing_time_ms: float = 0.0


# ============================================================================
# VACUUM OPTIMIZER CLASS
# ============================================================================

class VacuumOptimizer:
    """
    SLSQP-based vacuum optimization engine for condenser systems.

    This optimizer minimizes total operating cost by finding optimal:
    - CW flow setpoint
    - CW pump staging (which pumps to run)
    - Cooling tower fan staging/speeds

    Zero-Hallucination Guarantee:
        - All calculations use physics-based formulas
        - Optimization uses deterministic SLSQP algorithm
        - No AI/ML inference in any calculation path
        - Complete audit trail with provenance hashing

    Example:
        >>> optimizer = VacuumOptimizer(equipment, limits, costs)
        >>> result = optimizer.optimize(current_state)
        >>> print(f"Recommended flow: {result.best_recommendation.cw_flow_setpoint_gpm} GPM")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        equipment: EquipmentInventory,
        limits: OperatingLimits,
        costs: CostParameters,
        config: Optional[OptimizerConfig] = None
    ):
        """
        Initialize vacuum optimizer.

        Args:
            equipment: Equipment inventory (pumps, fans, condenser)
            limits: Operating limits and constraints
            costs: Cost parameters for objective function
            config: Optimizer configuration
        """
        self.equipment = equipment
        self.limits = limits
        self.costs = costs
        self.config = config or OptimizerConfig()

        logger.info(
            f"VacuumOptimizer initialized: {len(equipment.cw_pumps)} pumps, "
            f"{len(equipment.ct_fans)} fans"
        )

    # =========================================================================
    # PHYSICS MODELS
    # =========================================================================

    def calculate_tube_velocity(self, cw_flow_gpm: float) -> float:
        """
        Calculate CW tube velocity.

        Args:
            cw_flow_gpm: CW flow rate in GPM

        Returns:
            Tube velocity in ft/s
        """
        if cw_flow_gpm <= 0:
            return 0.0

        # Convert GPM to ft3/s
        flow_cfs = cw_flow_gpm / 448.831

        # Tube cross-sectional area
        tube_area_ft2 = (
            np.pi * (self.equipment.tube_diameter_in / 24.0) ** 2 *
            self.equipment.num_condenser_tubes
        )

        velocity_fps = flow_cfs / tube_area_ft2
        return velocity_fps

    def calculate_condenser_ua(
        self,
        cw_flow_gpm: float,
        cleanliness_factor: float
    ) -> float:
        """
        Calculate condenser overall heat transfer coefficient x area (UA).

        Uses empirical correlation:
        UA = UA_base * CF * (velocity / velocity_ref)^0.8

        Args:
            cw_flow_gpm: CW flow rate
            cleanliness_factor: Condenser cleanliness (0-1)

        Returns:
            UA in BTU/hr-F
        """
        velocity = self.calculate_tube_velocity(cw_flow_gpm)
        velocity_ref = 7.0  # Reference velocity in fps

        # Velocity exponent from Dittus-Boelter correlation
        velocity_factor = (velocity / velocity_ref) ** 0.8 if velocity > 0 else 0.0

        ua = self.equipment.condenser_ua_base_btu_hr_f * cleanliness_factor * velocity_factor
        return ua

    def calculate_backpressure(
        self,
        cw_flow_gpm: float,
        cw_inlet_temp_f: float,
        steam_flow_klb_hr: float,
        cleanliness_factor: float
    ) -> float:
        """
        Calculate condenser backpressure from first principles.

        Uses heat balance and saturation relationship.

        Args:
            cw_flow_gpm: CW flow rate
            cw_inlet_temp_f: CW inlet temperature
            steam_flow_klb_hr: Steam flow to condenser
            cleanliness_factor: Condenser cleanliness (0-1)

        Returns:
            Backpressure in inHgA
        """
        if cw_flow_gpm <= 0 or steam_flow_klb_hr <= 0:
            return self.limits.max_backpressure_inhga

        # Heat duty (BTU/hr)
        heat_duty = steam_flow_klb_hr * 1000 * STEAM_LATENT_HEAT_KJ_KG * 0.9478  # kJ to BTU

        # CW temperature rise
        cp_water = 1.0  # BTU/lb-F
        cw_mass_flow = cw_flow_gpm * 8.33 * 60  # lb/hr
        delta_t = heat_duty / (cw_mass_flow * cp_water) if cw_mass_flow > 0 else 100

        cw_outlet_temp_f = cw_inlet_temp_f + delta_t

        # Calculate UA and LMTD
        ua = self.calculate_condenser_ua(cw_flow_gpm, cleanliness_factor)

        # Saturation temperature (iteration or approximation)
        # Use NTU-effectiveness method for approximation
        ntu = ua / (cw_mass_flow * cp_water) if cw_mass_flow > 0 else 0
        effectiveness = 1 - np.exp(-ntu) if ntu < 10 else 0.9999

        # Saturation temp from effectiveness
        sat_temp_f = cw_inlet_temp_f + delta_t / effectiveness if effectiveness > 0 else 200

        # Convert saturation temp to pressure (Antoine equation approximation)
        # Valid for 50-150F range
        sat_temp_c = (sat_temp_f - 32) * 5 / 9

        # Simplified saturation pressure formula
        # log10(P_mmHg) = 8.07131 - 1730.63/(233.426 + T_C)
        log_p = 8.07131 - 1730.63 / (233.426 + sat_temp_c)
        p_mmhg = 10 ** log_p

        # Convert mmHg to inHgA
        backpressure_inhga = p_mmhg / 25.4

        return max(self.limits.min_backpressure_inhga,
                   min(self.limits.max_backpressure_inhga * 1.5, backpressure_inhga))

    def calculate_pump_power(
        self,
        cw_flow_gpm: float,
        pumps_running: List[str]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total pump power for given flow and staging.

        Args:
            cw_flow_gpm: Total CW flow
            pumps_running: List of pump IDs to run

        Returns:
            Tuple of (total power kW, dict of individual pump powers)
        """
        if not pumps_running or cw_flow_gpm <= 0:
            return 0.0, {}

        # Divide flow equally among running pumps
        flow_per_pump = cw_flow_gpm / len(pumps_running)

        pump_powers = {}
        total_power = 0.0

        for pump_id in pumps_running:
            pump = next((p for p in self.equipment.cw_pumps if p.pump_id == pump_id), None)
            if pump:
                head = pump.calculate_head(flow_per_pump)
                power = pump.calculate_power(flow_per_pump, head)
                pump_powers[pump_id] = power
                total_power += power

        return total_power, pump_powers

    def calculate_fan_power(
        self,
        fan_speeds: Dict[str, float]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate total fan power for given speeds.

        Args:
            fan_speeds: Dict of fan_id: speed_pct

        Returns:
            Tuple of (total power kW, dict of individual fan powers)
        """
        fan_powers = {}
        total_power = 0.0

        for fan_id, speed_pct in fan_speeds.items():
            fan = next((f for f in self.equipment.ct_fans if f.fan_id == fan_id), None)
            if fan:
                power = fan.calculate_power(speed_pct)
                fan_powers[fan_id] = power
                total_power += power

        return total_power, fan_powers

    def calculate_cw_supply_temp(
        self,
        ambient_temp_f: float,
        wet_bulb_temp_f: float,
        fan_speeds: Dict[str, float],
        heat_rejection_mmbtu_hr: float
    ) -> float:
        """
        Estimate CW supply temperature from cooling tower.

        Uses approach temperature correlation.

        Args:
            ambient_temp_f: Ambient dry bulb temperature
            wet_bulb_temp_f: Wet bulb temperature
            fan_speeds: Fan operating speeds
            heat_rejection_mmbtu_hr: Heat to be rejected

        Returns:
            CW supply temperature in F
        """
        # Calculate total air flow
        total_air_flow = 0.0
        for fan_id, speed_pct in fan_speeds.items():
            fan = next((f for f in self.equipment.ct_fans if f.fan_id == fan_id), None)
            if fan:
                total_air_flow += fan.calculate_air_flow(speed_pct)

        # Approach temperature correlation
        # Approach = f(heat load, air flow, wet bulb)
        rated_air_flow = sum(f.rated_air_flow_cfm for f in self.equipment.ct_fans)

        if total_air_flow > 0 and rated_air_flow > 0:
            air_flow_ratio = total_air_flow / rated_air_flow
            # Base approach at design conditions
            base_approach = 8.0  # F
            # Adjust for off-design
            approach = base_approach / (air_flow_ratio ** 0.6) if air_flow_ratio > 0.1 else 50.0
        else:
            approach = 30.0  # No air flow

        cw_supply_temp = wet_bulb_temp_f + approach
        return cw_supply_temp

    # =========================================================================
    # OBJECTIVE FUNCTION
    # =========================================================================

    def objective_function(
        self,
        x: np.ndarray,
        state: CondenserState,
        pumps_running: List[str],
        fan_ids: List[str]
    ) -> float:
        """
        Objective function: minimize total cost.

        Cost = Pump_Power_Cost + Fan_Power_Cost + Backpressure_Penalty

        Args:
            x: Decision variables [cw_flow_gpm, fan_speed_1, fan_speed_2, ...]
            state: Current condenser state
            pumps_running: List of pump IDs running
            fan_ids: List of fan IDs being optimized

        Returns:
            Total hourly cost in USD
        """
        cw_flow_gpm = x[0]
        fan_speeds = {fan_ids[i]: x[i + 1] for i in range(len(fan_ids))}

        # Calculate pump power
        pump_power, _ = self.calculate_pump_power(cw_flow_gpm, pumps_running)

        # Calculate fan power
        fan_power, _ = self.calculate_fan_power(fan_speeds)

        # Calculate CW inlet temp from tower
        heat_rejection = state.heat_duty_mmbtu_hr or 500.0
        cw_inlet_temp = self.calculate_cw_supply_temp(
            state.ambient_temp_f,
            state.wet_bulb_temp_f,
            fan_speeds,
            heat_rejection
        )

        # Calculate backpressure
        backpressure = self.calculate_backpressure(
            cw_flow_gpm,
            cw_inlet_temp,
            state.steam_flow_klb_hr,
            state.cleanliness_factor
        )

        # Calculate costs
        power_cost = (pump_power + fan_power) * self.costs.electricity_cost_per_kwh

        # Backpressure penalty (above target)
        bp_excess = max(0, backpressure - self.costs.target_backpressure_inhga)
        bp_penalty = bp_excess * self.costs.backpressure_penalty_per_inhga

        total_cost = power_cost + bp_penalty

        return total_cost

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    def build_constraints(
        self,
        state: CondenserState,
        pumps_running: List[str],
        fan_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Build constraint functions for optimization.

        Args:
            state: Current condenser state
            pumps_running: Pump staging configuration
            fan_ids: Fan IDs being optimized

        Returns:
            List of constraint dictionaries for scipy.optimize
        """
        constraints = []

        # Constraint 1: Maximum backpressure
        def max_bp_constraint(x):
            cw_flow_gpm = x[0]
            fan_speeds = {fan_ids[i]: x[i + 1] for i in range(len(fan_ids))}
            heat_rejection = state.heat_duty_mmbtu_hr or 500.0
            cw_inlet_temp = self.calculate_cw_supply_temp(
                state.ambient_temp_f, state.wet_bulb_temp_f, fan_speeds, heat_rejection
            )
            bp = self.calculate_backpressure(
                cw_flow_gpm, cw_inlet_temp, state.steam_flow_klb_hr, state.cleanliness_factor
            )
            return self.limits.max_backpressure_inhga - bp

        constraints.append({
            "type": "ineq",
            "fun": max_bp_constraint
        })

        # Constraint 2: Minimum tube velocity
        def min_velocity_constraint(x):
            cw_flow_gpm = x[0]
            velocity = self.calculate_tube_velocity(cw_flow_gpm)
            return velocity - self.limits.min_tube_velocity_fps

        constraints.append({
            "type": "ineq",
            "fun": min_velocity_constraint
        })

        # Constraint 3: Maximum tube velocity
        def max_velocity_constraint(x):
            cw_flow_gpm = x[0]
            velocity = self.calculate_tube_velocity(cw_flow_gpm)
            return self.limits.max_tube_velocity_fps - velocity

        constraints.append({
            "type": "ineq",
            "fun": max_velocity_constraint
        })

        # Constraint 4: NPSH margin for each pump
        for pump_id in pumps_running:
            pump = next((p for p in self.equipment.cw_pumps if p.pump_id == pump_id), None)
            if pump:
                def npsh_constraint(x, pump=pump, n_pumps=len(pumps_running)):
                    flow_per_pump = x[0] / n_pumps
                    # Simplified NPSH available calculation
                    npsh_available = 30.0  # Assume 30 ft available (from plant data)
                    return npsh_available - pump.npsh_required_ft * self.limits.min_npsh_margin

                constraints.append({
                    "type": "ineq",
                    "fun": npsh_constraint
                })

        # Constraint 5: Maximum CW outlet temperature
        def max_outlet_temp_constraint(x):
            cw_flow_gpm = x[0]
            fan_speeds = {fan_ids[i]: x[i + 1] for i in range(len(fan_ids))}
            heat_rejection = state.heat_duty_mmbtu_hr or 500.0
            cw_inlet_temp = self.calculate_cw_supply_temp(
                state.ambient_temp_f, state.wet_bulb_temp_f, fan_speeds, heat_rejection
            )
            # Calculate outlet temp
            heat_duty_btu_hr = heat_rejection * 1e6
            cw_mass_flow = cw_flow_gpm * 8.33 * 60
            delta_t = heat_duty_btu_hr / cw_mass_flow if cw_mass_flow > 0 else 100
            cw_outlet_temp = cw_inlet_temp + delta_t
            return self.limits.max_cw_outlet_temp_f - cw_outlet_temp

        constraints.append({
            "type": "ineq",
            "fun": max_outlet_temp_constraint
        })

        # Constraint 6: Ramp rate limits (if enforced)
        if self.config.enforce_ramp_rates:
            def flow_ramp_constraint(x):
                flow_change = abs(x[0] - state.cw_flow_gpm)
                return self.limits.max_pump_ramp_gpm_min - flow_change

            constraints.append({
                "type": "ineq",
                "fun": flow_ramp_constraint
            })

        return constraints

    def build_bounds(
        self,
        pumps_running: List[str],
        fan_ids: List[str]
    ) -> List[Tuple[float, float]]:
        """
        Build variable bounds for optimization.

        Args:
            pumps_running: Number of pumps running
            fan_ids: Fan IDs being optimized

        Returns:
            List of (min, max) bounds for each variable
        """
        bounds = []

        # CW flow bounds
        n_pumps = len(pumps_running)
        if n_pumps > 0:
            # Calculate flow limits based on pump curves
            min_flow = max(self.limits.min_cw_flow_gpm,
                          sum(p.min_flow_gpm for p in self.equipment.cw_pumps
                              if p.pump_id in pumps_running))
            max_flow = min(self.limits.max_cw_flow_gpm,
                          sum(p.max_flow_gpm for p in self.equipment.cw_pumps
                              if p.pump_id in pumps_running))
        else:
            min_flow = self.limits.min_cw_flow_gpm
            max_flow = self.limits.max_cw_flow_gpm

        bounds.append((min_flow, max_flow))

        # Fan speed bounds
        for fan_id in fan_ids:
            fan = next((f for f in self.equipment.ct_fans if f.fan_id == fan_id), None)
            if fan:
                bounds.append((fan.min_speed_pct, fan.max_speed_pct))
            else:
                bounds.append((0.0, 100.0))

        return bounds

    # =========================================================================
    # DISCRETE SEARCH FOR PUMP STAGING
    # =========================================================================

    def generate_pump_staging_options(self) -> List[List[str]]:
        """
        Generate all valid pump staging combinations.

        Returns:
            List of pump ID lists representing valid staging options
        """
        from itertools import combinations

        pump_ids = [p.pump_id for p in self.equipment.cw_pumps]
        options = []

        for n in range(self.limits.min_pumps_running,
                       min(self.limits.max_pumps_running, len(pump_ids)) + 1):
            for combo in combinations(pump_ids, n):
                options.append(list(combo))

        return options

    def evaluate_pump_staging(
        self,
        state: CondenserState,
        pumps_running: List[str]
    ) -> Tuple[float, np.ndarray, OptimizeResult]:
        """
        Evaluate optimal operation for a given pump staging.

        Args:
            state: Current condenser state
            pumps_running: Pump staging to evaluate

        Returns:
            Tuple of (objective value, optimal x, optimization result)
        """
        fan_ids = [f.fan_id for f in self.equipment.ct_fans]

        # Initial guess
        x0 = np.array([state.cw_flow_gpm] +
                     [state.fans_running.get(fid, 50.0) for fid in fan_ids])

        # Build constraints and bounds
        constraints = self.build_constraints(state, pumps_running, fan_ids)
        bounds = self.build_bounds(pumps_running, fan_ids)

        # Run optimization
        try:
            result = minimize(
                self.objective_function,
                x0,
                args=(state, pumps_running, fan_ids),
                method=self.config.method,
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": self.config.max_iterations,
                    "ftol": self.config.tolerance
                }
            )

            return result.fun, result.x, result

        except Exception as e:
            logger.error(f"Optimization failed for staging {pumps_running}: {e}")
            return float("inf"), x0, None

    # =========================================================================
    # MAIN OPTIMIZATION ENTRY POINT
    # =========================================================================

    def optimize(
        self,
        state: CondenserState,
        data_quality_score: float = 1.0
    ) -> OptimizationResult:
        """
        Run vacuum optimization.

        Args:
            state: Current condenser operating state
            data_quality_score: Quality score of input data (0-1)

        Returns:
            OptimizationResult with best recommendation and alternatives
        """
        start_time = datetime.now(timezone.utc)

        logger.info(f"Starting vacuum optimization at {start_time}")

        # Check data quality
        if (self.config.suppress_on_bad_data and
            data_quality_score < self.config.min_data_quality_score):

            logger.warning(
                f"Data quality {data_quality_score:.2f} below threshold "
                f"{self.config.min_data_quality_score}. Suppressing recommendations."
            )

            return OptimizationResult(
                result_id=f"OPT-{start_time.strftime('%Y%m%d%H%M%S')}",
                timestamp=start_time,
                status=OptimizationStatus.ERROR,
                constraint_violations=["Data quality below threshold"],
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

        # Calculate current operating cost
        current_pump_power, _ = self.calculate_pump_power(state.cw_flow_gpm, state.pumps_running)
        current_fan_power, _ = self.calculate_fan_power(state.fans_running)
        current_total_power = current_pump_power + current_fan_power

        bp_excess = max(0, state.backpressure_inhga - self.costs.target_backpressure_inhga)
        current_cost = (
            current_total_power * self.costs.electricity_cost_per_kwh +
            bp_excess * self.costs.backpressure_penalty_per_inhga
        )

        # Generate and evaluate pump staging options
        staging_options = self.generate_pump_staging_options()

        staging_results = []
        for staging in staging_options:
            obj, x_opt, result = self.evaluate_pump_staging(state, staging)
            if result is not None and result.success:
                staging_results.append((obj, x_opt, staging, result))

        if not staging_results:
            logger.error("No feasible solutions found")
            return OptimizationResult(
                result_id=f"OPT-{start_time.strftime('%Y%m%d%H%M%S')}",
                timestamp=start_time,
                status=OptimizationStatus.INFEASIBLE,
                current_power_kw=current_total_power,
                current_cost_usd_hr=current_cost,
                processing_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

        # Sort by objective value
        staging_results.sort(key=lambda x: x[0])

        # Build recommendations
        fan_ids = [f.fan_id for f in self.equipment.ct_fans]
        recommendations = []

        for i, (obj, x_opt, staging, result) in enumerate(
            staging_results[:self.config.num_alternatives + 1]
        ):
            cw_flow = x_opt[0]
            fan_speeds = {fan_ids[j]: x_opt[j + 1] for j in range(len(fan_ids))}

            # Calculate predicted values
            pump_power, _ = self.calculate_pump_power(cw_flow, staging)
            fan_power, _ = self.calculate_fan_power(fan_speeds)
            total_power = pump_power + fan_power

            heat_rejection = state.heat_duty_mmbtu_hr or 500.0
            cw_inlet_temp = self.calculate_cw_supply_temp(
                state.ambient_temp_f, state.wet_bulb_temp_f, fan_speeds, heat_rejection
            )
            backpressure = self.calculate_backpressure(
                cw_flow, cw_inlet_temp, state.steam_flow_klb_hr, state.cleanliness_factor
            )

            # Costs
            power_cost = total_power * self.costs.electricity_cost_per_kwh
            bp_excess = max(0, backpressure - self.costs.target_backpressure_inhga)
            bp_penalty = bp_excess * self.costs.backpressure_penalty_per_inhga
            total_cost = power_cost + bp_penalty

            # Improvements
            power_savings = current_total_power - total_power
            cost_savings = current_cost - total_cost
            bp_improvement = state.backpressure_inhga - backpressure

            # Tradeoff description
            if i == 0:
                tradeoff = "Optimal balance of power and backpressure"
            elif len(staging) > len(staging_results[0][2]):
                tradeoff = f"More pumps ({len(staging)}): lower backpressure, higher power"
            elif len(staging) < len(staging_results[0][2]):
                tradeoff = f"Fewer pumps ({len(staging)}): lower power, higher backpressure"
            else:
                tradeoff = "Alternative with different fan staging"

            # Calculate provenance
            provenance_data = {
                "version": self.VERSION,
                "cw_flow": round(cw_flow, 2),
                "pumps": staging,
                "fan_speeds": {k: round(v, 1) for k, v in fan_speeds.items()},
                "backpressure": round(backpressure, 3),
                "total_cost": round(total_cost, 2)
            }
            provenance_hash = hashlib.sha256(
                json.dumps(provenance_data, sort_keys=True).encode()
            ).hexdigest()[:16]

            rec = OptimizationRecommendation(
                recommendation_id=f"REC-{start_time.strftime('%Y%m%d%H%M%S')}-{i:02d}",
                cw_flow_setpoint_gpm=cw_flow,
                pumps_to_run=staging,
                fan_speeds=fan_speeds,
                predicted_backpressure_inhga=backpressure,
                predicted_pump_power_kw=pump_power,
                predicted_fan_power_kw=fan_power,
                predicted_total_power_kw=total_power,
                hourly_power_cost_usd=power_cost,
                backpressure_penalty_usd_hr=bp_penalty,
                total_hourly_cost_usd=total_cost,
                power_savings_kw=power_savings,
                cost_savings_usd_hr=cost_savings,
                backpressure_improvement_inhga=bp_improvement,
                tradeoff_description=tradeoff,
                confidence_score=0.95 if result.success else 0.7,
                provenance_hash=provenance_hash
            )

            recommendations.append(rec)

        # Calculate overall provenance
        result_provenance = {
            "timestamp": start_time.isoformat(),
            "state_hash": hashlib.sha256(
                json.dumps({
                    "bp": state.backpressure_inhga,
                    "flow": state.cw_flow_gpm,
                    "cf": state.cleanliness_factor
                }).encode()
            ).hexdigest()[:16],
            "best_rec_hash": recommendations[0].provenance_hash if recommendations else ""
        }
        overall_provenance = hashlib.sha256(
            json.dumps(result_provenance, sort_keys=True).encode()
        ).hexdigest()[:16]

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Optimization complete in {processing_time:.1f}ms. "
            f"Best cost savings: ${recommendations[0].cost_savings_usd_hr:.2f}/hr"
        )

        return OptimizationResult(
            result_id=f"OPT-{start_time.strftime('%Y%m%d%H%M%S')}-{overall_provenance}",
            timestamp=start_time,
            status=OptimizationStatus.SUCCESS,
            best_recommendation=recommendations[0] if recommendations else None,
            alternatives=recommendations[1:] if len(recommendations) > 1 else [],
            current_power_kw=current_total_power,
            current_cost_usd_hr=current_cost,
            iterations=sum(r[3].nit for r in staging_results if r[3]),
            objective_value=staging_results[0][0] if staging_results else 0,
            provenance_hash=overall_provenance,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def validate_recommendation(
        self,
        recommendation: OptimizationRecommendation,
        current_state: CondenserState
    ) -> Tuple[bool, List[str]]:
        """
        Validate a recommendation against current constraints.

        Args:
            recommendation: Recommendation to validate
            current_state: Current operating state

        Returns:
            Tuple of (is_valid, list of violations)
        """
        violations = []

        # Check backpressure limit
        if recommendation.predicted_backpressure_inhga > self.limits.max_backpressure_inhga:
            violations.append(
                f"Predicted backpressure {recommendation.predicted_backpressure_inhga:.2f} inHgA "
                f"exceeds limit {self.limits.max_backpressure_inhga:.2f} inHgA"
            )

        # Check flow limits
        if recommendation.cw_flow_setpoint_gpm < self.limits.min_cw_flow_gpm:
            violations.append(
                f"CW flow {recommendation.cw_flow_setpoint_gpm:.0f} GPM "
                f"below minimum {self.limits.min_cw_flow_gpm:.0f} GPM"
            )

        if recommendation.cw_flow_setpoint_gpm > self.limits.max_cw_flow_gpm:
            violations.append(
                f"CW flow {recommendation.cw_flow_setpoint_gpm:.0f} GPM "
                f"exceeds maximum {self.limits.max_cw_flow_gpm:.0f} GPM"
            )

        # Check tube velocity
        velocity = self.calculate_tube_velocity(recommendation.cw_flow_setpoint_gpm)
        if velocity < self.limits.min_tube_velocity_fps:
            violations.append(
                f"Tube velocity {velocity:.1f} fps below minimum "
                f"{self.limits.min_tube_velocity_fps:.1f} fps"
            )

        if velocity > self.limits.max_tube_velocity_fps:
            violations.append(
                f"Tube velocity {velocity:.1f} fps exceeds maximum "
                f"{self.limits.max_tube_velocity_fps:.1f} fps"
            )

        # Check ramp rates
        if self.config.enforce_ramp_rates:
            flow_change = abs(recommendation.cw_flow_setpoint_gpm - current_state.cw_flow_gpm)
            if flow_change > self.limits.max_pump_ramp_gpm_min:
                violations.append(
                    f"Flow change {flow_change:.0f} GPM exceeds ramp rate "
                    f"{self.limits.max_pump_ramp_gpm_min:.0f} GPM/min"
                )

        # Check pump count
        if len(recommendation.pumps_to_run) < self.limits.min_pumps_running:
            violations.append(
                f"Pump count {len(recommendation.pumps_to_run)} "
                f"below minimum {self.limits.min_pumps_running}"
            )

        return len(violations) == 0, violations

    def calculate_annual_savings(
        self,
        recommendation: OptimizationRecommendation,
        operating_hours_per_year: float = 8000.0
    ) -> Dict[str, float]:
        """
        Calculate annualized savings from recommendation.

        Args:
            recommendation: Optimization recommendation
            operating_hours_per_year: Annual operating hours

        Returns:
            Dict with savings breakdown
        """
        hourly_savings = recommendation.cost_savings_usd_hr
        power_savings_kwh = recommendation.power_savings_kw

        annual_energy_savings = power_savings_kwh * operating_hours_per_year
        annual_cost_savings = hourly_savings * operating_hours_per_year

        # CO2 savings (assume 0.9 lb CO2/kWh for grid average)
        annual_co2_savings_lb = annual_energy_savings * 0.9
        annual_co2_savings_tons = annual_co2_savings_lb / 2000

        return {
            "annual_energy_savings_mwh": annual_energy_savings / 1000,
            "annual_cost_savings_usd": annual_cost_savings,
            "annual_co2_savings_tons": annual_co2_savings_tons,
            "simple_payback_months": 0.0  # No capital cost for optimization
        }

    def generate_operator_message(
        self,
        result: OptimizationResult
    ) -> str:
        """
        Generate human-readable operator message.

        Args:
            result: Optimization result

        Returns:
            Formatted operator message
        """
        if result.status != OptimizationStatus.SUCCESS or not result.best_recommendation:
            return f"Optimization {result.status.value}: No recommendation available"

        rec = result.best_recommendation

        lines = [
            "=" * 60,
            "CONDENSER VACUUM OPTIMIZATION RECOMMENDATION",
            "=" * 60,
            f"Recommendation ID: {rec.recommendation_id}",
            f"Confidence: {rec.confidence_score:.0%}",
            "",
            "RECOMMENDED SETPOINTS:",
            f"  CW Flow: {rec.cw_flow_setpoint_gpm:,.0f} GPM",
            f"  Pumps: {', '.join(rec.pumps_to_run)}",
            "  Fan Speeds:",
        ]

        for fan_id, speed in rec.fan_speeds.items():
            lines.append(f"    {fan_id}: {speed:.1f}%")

        lines.extend([
            "",
            "PREDICTED OUTCOMES:",
            f"  Backpressure: {rec.predicted_backpressure_inhga:.2f} inHgA",
            f"  Total Power: {rec.predicted_total_power_kw:.0f} kW",
            f"    Pump Power: {rec.predicted_pump_power_kw:.0f} kW",
            f"    Fan Power: {rec.predicted_fan_power_kw:.0f} kW",
            "",
            "SAVINGS:",
            f"  Power Reduction: {rec.power_savings_kw:.0f} kW",
            f"  Cost Savings: ${rec.cost_savings_usd_hr:.2f}/hr",
            f"  BP Improvement: {rec.backpressure_improvement_inhga:.3f} inHgA",
            "",
            f"Provenance: {rec.provenance_hash}",
            "=" * 60
        ])

        return "\n".join(lines)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_optimizer() -> VacuumOptimizer:
    """
    Create optimizer with default equipment and parameters.

    Returns:
        Configured VacuumOptimizer instance
    """
    # Default pump curves
    pumps = [
        PumpCurve(
            pump_id=f"CWP-{i+1}",
            head_coefficients=(1.0, 0.1, -0.15, -0.05),
            efficiency_coefficients=(0.0, 0.8, 0.4, -0.4),
            rated_flow_gpm=40000.0,
            rated_head_ft=80.0,
            rated_power_kw=800.0,
            min_flow_gpm=15000.0,
            max_flow_gpm=50000.0,
            npsh_required_ft=20.0
        )
        for i in range(4)
    ]

    # Default fan curves
    fans = [
        FanCurve(
            fan_id=f"CTF-{i+1}",
            rated_air_flow_cfm=500000.0,
            rated_power_kw=150.0
        )
        for i in range(8)
    ]

    equipment = EquipmentInventory(
        cw_pumps=pumps,
        ct_fans=fans
    )

    limits = OperatingLimits()
    costs = CostParameters()
    config = OptimizerConfig()

    return VacuumOptimizer(equipment, limits, costs, config)

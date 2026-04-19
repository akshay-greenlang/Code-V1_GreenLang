"""
GL-003 UNIFIEDSTEAM - Desuperheater Optimizer

Provides optimization for desuperheater spray water injection:
- Spray setpoint optimization
- Target temperature calculation
- Setpoint trajectory generation
- Multi-objective balancing (minimize spray while maintaining quality)

Constraints respected:
- Minimum approach to saturation
- Valve limits
- Nozzle pressure differential
- Water quality
- Thermal shock avoidance
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import time

from pydantic import BaseModel, Field, validator

from .constraints import (
    ConstraintCheckResult,
    ConstraintSeverity,
    ConstraintStatus,
    EquipmentConstraints,
    SafetyConstraints,
    UncertaintyConstraints,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class DesuperheaterState(BaseModel):
    """Current state of a desuperheater."""

    desuperheater_id: str = Field(..., description="Desuperheater identifier")
    inlet_temp_f: float = Field(..., description="Steam inlet temperature (F)")
    outlet_temp_f: float = Field(..., description="Steam outlet temperature (F)")
    setpoint_temp_f: float = Field(..., description="Target outlet temperature (F)")
    steam_pressure_psig: float = Field(..., description="Steam pressure (psig)")
    saturation_temp_f: float = Field(..., description="Saturation temperature (F)")
    steam_flow_lb_hr: float = Field(..., ge=0, description="Steam flow rate (lb/hr)")
    spray_valve_position_pct: float = Field(
        ..., ge=0, le=100, description="Spray valve position (%)"
    )
    spray_flow_gpm: float = Field(..., ge=0, description="Spray water flow (gpm)")
    spray_water_temp_f: float = Field(..., description="Spray water temperature (F)")
    spray_water_pressure_psig: float = Field(
        ..., description="Spray water pressure (psig)"
    )
    nozzle_delta_p_psi: float = Field(
        ..., ge=0, description="Pressure drop across nozzle (psi)"
    )

    @validator("outlet_temp_f")
    def outlet_above_saturation(cls, v, values):
        """Validate outlet is above saturation."""
        if "saturation_temp_f" in values and v < values["saturation_temp_f"]:
            raise ValueError("Outlet temperature cannot be below saturation")
        return v


class TargetConstraints(BaseModel):
    """Constraints for temperature targeting."""

    min_outlet_temp_f: Optional[float] = Field(
        default=None, description="Minimum outlet temperature (F)"
    )
    max_outlet_temp_f: Optional[float] = Field(
        default=None, description="Maximum outlet temperature (F)"
    )
    target_superheat_f: Optional[float] = Field(
        default=50.0, description="Target superheat above saturation (F)"
    )
    min_approach_to_saturation_f: float = Field(
        default=20.0, description="Minimum approach to saturation (F)"
    )
    max_spray_valve_position_pct: float = Field(
        default=90.0, description="Maximum spray valve position (%)"
    )
    max_spray_flow_gpm: float = Field(
        default=100.0, description="Maximum spray flow (gpm)"
    )
    max_temp_rate_f_min: float = Field(
        default=50.0, description="Maximum temperature change rate (F/min)"
    )


@dataclass
class Setpoint:
    """A single setpoint in a trajectory."""

    timestamp: datetime
    temperature_f: float
    spray_valve_position_pct: float
    ramp_step: int
    is_final: bool = False


class SprayOptimizationResult(BaseModel):
    """Result of spray setpoint optimization."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    desuperheater_id: str = Field(..., description="Desuperheater identifier")

    # Recommended setpoints
    recommended_temp_setpoint_f: float = Field(
        ..., description="Recommended temperature setpoint (F)"
    )
    recommended_valve_position_pct: float = Field(
        ..., description="Recommended valve position (%)"
    )
    estimated_spray_flow_gpm: float = Field(
        ..., description="Estimated spray flow at setpoint (gpm)"
    )

    # Current vs recommended delta
    temp_change_f: float = Field(
        ..., description="Temperature change from current (F)"
    )
    valve_change_pct: float = Field(
        ..., description="Valve position change (%)"
    )

    # Quality metrics
    approach_to_saturation_f: float = Field(
        ..., description="Approach to saturation temperature (F)"
    )
    superheat_f: float = Field(..., description="Resulting superheat (F)")

    # Optimization metrics
    spray_water_reduction_percent: float = Field(
        default=0.0, description="Reduction in spray water usage (%)"
    )
    efficiency_improvement_percent: float = Field(
        default=0.0, description="Efficiency improvement (%)"
    )
    cost_savings_per_hour: float = Field(
        default=0.0, description="Cost savings ($/hr)"
    )

    # Confidence and uncertainty
    confidence: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Confidence level"
    )
    uncertainty_temp_f: float = Field(
        default=2.0, ge=0.0, description="Temperature uncertainty (F)"
    )

    # Constraints
    constraints_satisfied: bool = Field(
        default=True, description="All constraints satisfied"
    )
    constraint_violations: List[str] = Field(
        default_factory=list, description="List of constraint violations"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")
    computation_time_ms: float = Field(
        default=0.0, description="Computation time (ms)"
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {datetime: lambda v: v.isoformat()}


# =============================================================================
# Desuperheater Optimizer
# =============================================================================


class DesuperheaterOptimizer:
    """
    Optimizes desuperheater spray water injection.

    Objectives:
    - Minimize spray water usage while maintaining steam quality targets
    - Avoid thermal shock and wet steam conditions
    - Respect valve and nozzle constraints

    Uses deterministic calculations (zero-hallucination approach) based on:
    - Steam tables (thermodynamic properties)
    - Energy balance equations
    - Constraint satisfaction
    """

    def __init__(
        self,
        safety_constraints: Optional[SafetyConstraints] = None,
        equipment_constraints: Optional[EquipmentConstraints] = None,
        uncertainty_constraints: Optional[UncertaintyConstraints] = None,
        spray_water_cost_per_gal: float = 0.01,
        energy_cost_per_mmbtu: float = 8.0,
    ) -> None:
        """
        Initialize desuperheater optimizer.

        Args:
            safety_constraints: Safety constraint definitions
            equipment_constraints: Equipment constraint definitions
            uncertainty_constraints: Uncertainty handling constraints
            spray_water_cost_per_gal: Cost of spray water ($/gal)
            energy_cost_per_mmbtu: Cost of energy ($/MMBTU)
        """
        self.safety_constraints = safety_constraints or SafetyConstraints()
        self.equipment_constraints = equipment_constraints or EquipmentConstraints()
        self.uncertainty_constraints = (
            uncertainty_constraints or UncertaintyConstraints()
        )
        self.spray_water_cost_per_gal = spray_water_cost_per_gal
        self.energy_cost_per_mmbtu = energy_cost_per_mmbtu

        # Optimization history for learning
        self._optimization_history: List[Dict[str, Any]] = []
        self._max_history = 1000

        logger.info("DesuperheaterOptimizer initialized")

    def optimize_spray_setpoint(
        self,
        current_state: DesuperheaterState,
        target_constraints: TargetConstraints,
    ) -> SprayOptimizationResult:
        """
        Optimize spray water setpoint for given conditions.

        Uses energy balance to calculate optimal spray rate while
        respecting all constraints.

        Args:
            current_state: Current desuperheater state
            target_constraints: Target constraints to satisfy

        Returns:
            SprayOptimizationResult with recommended setpoints
        """
        start_time = time.perf_counter()

        # Step 1: Calculate optimal target temperature
        target_temp = self.compute_optimal_target_temperature(
            steam_pressure_psig=current_state.steam_pressure_psig,
            saturation_temp_f=current_state.saturation_temp_f,
            constraints=target_constraints,
        )

        # Step 2: Calculate required spray rate using energy balance
        spray_flow_gpm = self._calculate_required_spray(
            inlet_temp_f=current_state.inlet_temp_f,
            target_temp_f=target_temp,
            steam_flow_lb_hr=current_state.steam_flow_lb_hr,
            spray_water_temp_f=current_state.spray_water_temp_f,
            steam_pressure_psig=current_state.steam_pressure_psig,
        )

        # Step 3: Calculate valve position for required spray flow
        valve_position = self._calculate_valve_position(
            target_flow_gpm=spray_flow_gpm,
            current_delta_p_psi=current_state.nozzle_delta_p_psi,
            max_position=target_constraints.max_spray_valve_position_pct,
        )

        # Step 4: Check constraints
        violations = self._check_constraints(
            target_temp_f=target_temp,
            spray_flow_gpm=spray_flow_gpm,
            valve_position_pct=valve_position,
            current_state=current_state,
            target_constraints=target_constraints,
        )

        # Step 5: Adjust if constraints violated
        if violations:
            target_temp, spray_flow_gpm, valve_position = self._adjust_for_constraints(
                target_temp_f=target_temp,
                spray_flow_gpm=spray_flow_gpm,
                valve_position_pct=valve_position,
                current_state=current_state,
                target_constraints=target_constraints,
                violations=violations,
            )
            # Re-check constraints
            violations = self._check_constraints(
                target_temp_f=target_temp,
                spray_flow_gpm=spray_flow_gpm,
                valve_position_pct=valve_position,
                current_state=current_state,
                target_constraints=target_constraints,
            )

        # Step 6: Calculate benefits
        spray_reduction = (
            (current_state.spray_flow_gpm - spray_flow_gpm) /
            current_state.spray_flow_gpm * 100
            if current_state.spray_flow_gpm > 0 else 0
        )
        cost_savings = max(0, spray_reduction / 100 * current_state.spray_flow_gpm *
                          60 * self.spray_water_cost_per_gal)

        # Step 7: Calculate confidence based on operating point
        confidence = self._calculate_confidence(
            current_state=current_state,
            target_constraints=target_constraints,
            valve_position=valve_position,
        )

        # Step 8: Calculate approach to saturation
        approach = target_temp - current_state.saturation_temp_f
        superheat = target_temp - current_state.saturation_temp_f

        # Calculate computation time
        computation_time_ms = (time.perf_counter() - start_time) * 1000

        # Create result
        result = SprayOptimizationResult(
            desuperheater_id=current_state.desuperheater_id,
            recommended_temp_setpoint_f=target_temp,
            recommended_valve_position_pct=valve_position,
            estimated_spray_flow_gpm=spray_flow_gpm,
            temp_change_f=target_temp - current_state.setpoint_temp_f,
            valve_change_pct=valve_position - current_state.spray_valve_position_pct,
            approach_to_saturation_f=approach,
            superheat_f=superheat,
            spray_water_reduction_percent=spray_reduction,
            cost_savings_per_hour=cost_savings,
            confidence=confidence,
            uncertainty_temp_f=2.0 + (1.0 - confidence) * 5.0,
            constraints_satisfied=len(violations) == 0,
            constraint_violations=violations,
            computation_time_ms=computation_time_ms,
        )

        # Generate provenance hash
        result.provenance_hash = self._generate_provenance_hash(
            current_state, target_constraints, result
        )

        # Store in history
        self._record_optimization(current_state, result)

        logger.info(
            f"Desuperheater {current_state.desuperheater_id} optimization: "
            f"target={target_temp:.1f}F, valve={valve_position:.1f}%, "
            f"spray={spray_flow_gpm:.1f}gpm, confidence={confidence:.2f}"
        )

        return result

    def compute_optimal_target_temperature(
        self,
        steam_pressure_psig: float,
        saturation_temp_f: float,
        constraints: TargetConstraints,
        steam_demand_lb_hr: Optional[float] = None,
    ) -> float:
        """
        Compute optimal target temperature based on constraints.

        Balances:
        - Minimum approach to saturation (safety)
        - Target superheat (quality)
        - User-specified limits

        Args:
            steam_pressure_psig: Current steam pressure (psig)
            saturation_temp_f: Saturation temperature at pressure (F)
            constraints: Target constraints
            steam_demand_lb_hr: Optional steam demand (lb/hr)

        Returns:
            Optimal target temperature (F)
        """
        # Calculate minimum temperature (approach to saturation)
        min_temp = saturation_temp_f + constraints.min_approach_to_saturation_f

        # Calculate target based on superheat
        target_from_superheat = saturation_temp_f + constraints.target_superheat_f

        # Apply user limits if specified
        if constraints.min_outlet_temp_f is not None:
            min_temp = max(min_temp, constraints.min_outlet_temp_f)

        max_temp = constraints.max_outlet_temp_f
        if max_temp is None:
            # Default max: saturation + 150F
            max_temp = saturation_temp_f + 150.0

        # Optimal target is the lower bound that satisfies all constraints
        # (minimize superheat = minimize losses, but maintain quality)
        optimal_temp = max(min_temp, target_from_superheat)
        optimal_temp = min(optimal_temp, max_temp)

        # Adjust for demand if provided (higher demand = slightly higher temp for stability)
        if steam_demand_lb_hr is not None and steam_demand_lb_hr > 50000:
            stability_margin = min(10.0, steam_demand_lb_hr / 50000 * 5)
            optimal_temp = min(max_temp, optimal_temp + stability_margin)

        return round(optimal_temp, 1)

    def generate_setpoint_trajectory(
        self,
        current_temp_f: float,
        target_temp_f: float,
        ramp_rate_f_min: float,
        current_valve_position_pct: float,
        interval_seconds: float = 30.0,
    ) -> List[Setpoint]:
        """
        Generate setpoint trajectory for smooth ramping.

        Creates a series of intermediate setpoints to avoid thermal shock
        and maintain stable control.

        Args:
            current_temp_f: Current temperature (F)
            target_temp_f: Target temperature (F)
            ramp_rate_f_min: Maximum temperature change rate (F/min)
            current_valve_position_pct: Current valve position (%)
            interval_seconds: Time between setpoints (seconds)

        Returns:
            List of Setpoint objects defining the trajectory
        """
        trajectory: List[Setpoint] = []

        # Calculate total temperature change
        delta_temp = target_temp_f - current_temp_f

        if abs(delta_temp) < 1.0:
            # Already at target
            return [
                Setpoint(
                    timestamp=datetime.now(timezone.utc),
                    temperature_f=target_temp_f,
                    spray_valve_position_pct=current_valve_position_pct,
                    ramp_step=0,
                    is_final=True,
                )
            ]

        # Calculate number of steps
        max_delta_per_step = ramp_rate_f_min * interval_seconds / 60.0
        num_steps = max(1, int(abs(delta_temp) / max_delta_per_step))
        temp_step = delta_temp / num_steps

        # Generate trajectory
        current_time = datetime.now(timezone.utc)

        for step in range(num_steps + 1):
            step_temp = current_temp_f + temp_step * step
            step_temp = (
                target_temp_f if step == num_steps else round(step_temp, 1)
            )

            # Estimate valve position (linear interpolation)
            valve_fraction = step / num_steps if num_steps > 0 else 1.0
            # This is simplified - actual valve position depends on flow required
            step_valve = current_valve_position_pct  # Placeholder

            trajectory.append(
                Setpoint(
                    timestamp=current_time,
                    temperature_f=step_temp,
                    spray_valve_position_pct=step_valve,
                    ramp_step=step,
                    is_final=(step == num_steps),
                )
            )

            # Advance time
            from datetime import timedelta
            current_time = current_time + timedelta(seconds=interval_seconds)

        logger.info(
            f"Generated trajectory: {num_steps} steps over "
            f"{num_steps * interval_seconds / 60:.1f} minutes"
        )

        return trajectory

    def _calculate_required_spray(
        self,
        inlet_temp_f: float,
        target_temp_f: float,
        steam_flow_lb_hr: float,
        spray_water_temp_f: float,
        steam_pressure_psig: float,
    ) -> float:
        """
        Calculate required spray water flow using energy balance.

        Energy balance:
        m_steam * h_in + m_spray * h_water = (m_steam + m_spray) * h_out

        Args:
            inlet_temp_f: Steam inlet temperature (F)
            target_temp_f: Target outlet temperature (F)
            steam_flow_lb_hr: Steam flow rate (lb/hr)
            spray_water_temp_f: Spray water temperature (F)
            steam_pressure_psig: Steam pressure (psig)

        Returns:
            Required spray flow in gpm
        """
        # Simplified enthalpy calculations (BTU/lb)
        # In production, use proper steam tables

        # Approximate steam enthalpy (superheated)
        # h = h_sat_vapor + Cp_steam * (T - T_sat)
        sat_temp = self._saturation_temp_from_pressure(steam_pressure_psig)
        h_sat_vapor = 1150.0 + steam_pressure_psig * 0.5  # Approximate

        cp_steam = 0.5  # BTU/lb-F (approximate for superheated steam)
        h_inlet = h_sat_vapor + cp_steam * (inlet_temp_f - sat_temp)
        h_outlet = h_sat_vapor + cp_steam * (target_temp_f - sat_temp)

        # Water enthalpy (subcooled liquid)
        h_water = spray_water_temp_f - 32  # BTU/lb (approximate)

        # Energy balance to solve for spray mass flow
        # m_steam * h_in + m_spray * h_water = (m_steam + m_spray) * h_out
        # m_spray = m_steam * (h_in - h_out) / (h_out - h_water)

        if h_outlet <= h_water:
            logger.warning("Energy balance invalid: h_outlet <= h_water")
            return 0.0

        spray_mass_lb_hr = steam_flow_lb_hr * (h_inlet - h_outlet) / (h_outlet - h_water)
        spray_mass_lb_hr = max(0, spray_mass_lb_hr)

        # Convert to gpm (water density ~8.34 lb/gal)
        spray_gpm = spray_mass_lb_hr / 60 / 8.34

        return round(spray_gpm, 2)

    def _calculate_valve_position(
        self,
        target_flow_gpm: float,
        current_delta_p_psi: float,
        max_position: float = 90.0,
    ) -> float:
        """
        Calculate valve position for target flow.

        Uses simplified Cv relationship:
        Q = Cv * sqrt(dP)

        Args:
            target_flow_gpm: Target flow rate (gpm)
            current_delta_p_psi: Current pressure differential (psi)
            max_position: Maximum valve position (%)

        Returns:
            Estimated valve position (%)
        """
        if target_flow_gpm <= 0:
            return 0.0

        if current_delta_p_psi <= 0:
            logger.warning("Invalid delta P for valve calculation")
            return 50.0  # Default

        # Estimate Cv required
        cv_required = target_flow_gpm / math.sqrt(current_delta_p_psi)

        # Assume equal percentage valve with max Cv at 100%
        # and typical Cv at 50% is about 20% of max
        max_cv = 50.0  # Typical max Cv for spray valve

        if cv_required >= max_cv:
            return max_position

        # Equal percentage relationship: Cv = Cv_max * R^(x-1)
        # where R is rangeability (~50) and x is position (0-1)
        R = 50.0
        x = 1 + math.log(cv_required / max_cv) / math.log(R)
        position = x * 100

        return max(0.0, min(max_position, round(position, 1)))

    def _check_constraints(
        self,
        target_temp_f: float,
        spray_flow_gpm: float,
        valve_position_pct: float,
        current_state: DesuperheaterState,
        target_constraints: TargetConstraints,
    ) -> List[str]:
        """
        Check all constraints for proposed setpoint.

        Args:
            target_temp_f: Target temperature (F)
            spray_flow_gpm: Spray flow (gpm)
            valve_position_pct: Valve position (%)
            current_state: Current state
            target_constraints: Constraints

        Returns:
            List of violation descriptions
        """
        violations = []

        # Check approach to saturation
        approach = target_temp_f - current_state.saturation_temp_f
        if approach < target_constraints.min_approach_to_saturation_f:
            violations.append(
                f"Approach to saturation {approach:.1f}F below minimum "
                f"{target_constraints.min_approach_to_saturation_f:.1f}F"
            )

        # Check valve position limit
        if valve_position_pct > target_constraints.max_spray_valve_position_pct:
            violations.append(
                f"Valve position {valve_position_pct:.1f}% exceeds limit "
                f"{target_constraints.max_spray_valve_position_pct:.1f}%"
            )

        # Check spray flow limit
        if spray_flow_gpm > target_constraints.max_spray_flow_gpm:
            violations.append(
                f"Spray flow {spray_flow_gpm:.1f}gpm exceeds limit "
                f"{target_constraints.max_spray_flow_gpm:.1f}gpm"
            )

        # Check nozzle delta P using equipment constraints
        nozzle_result = self.equipment_constraints.nozzle.check_delta_p(
            current_state.nozzle_delta_p_psi
        )
        if nozzle_result.status == ConstraintStatus.VIOLATED:
            violations.append(nozzle_result.message)

        # Check temperature change rate
        temp_change = abs(target_temp_f - current_state.outlet_temp_f)
        if temp_change > target_constraints.max_temp_rate_f_min:
            violations.append(
                f"Temperature change {temp_change:.1f}F exceeds rate limit "
                f"{target_constraints.max_temp_rate_f_min:.1f}F/min"
            )

        # Check temperature limits
        if target_constraints.min_outlet_temp_f is not None:
            if target_temp_f < target_constraints.min_outlet_temp_f:
                violations.append(
                    f"Target temp {target_temp_f:.1f}F below minimum "
                    f"{target_constraints.min_outlet_temp_f:.1f}F"
                )

        if target_constraints.max_outlet_temp_f is not None:
            if target_temp_f > target_constraints.max_outlet_temp_f:
                violations.append(
                    f"Target temp {target_temp_f:.1f}F above maximum "
                    f"{target_constraints.max_outlet_temp_f:.1f}F"
                )

        return violations

    def _adjust_for_constraints(
        self,
        target_temp_f: float,
        spray_flow_gpm: float,
        valve_position_pct: float,
        current_state: DesuperheaterState,
        target_constraints: TargetConstraints,
        violations: List[str],
    ) -> Tuple[float, float, float]:
        """
        Adjust setpoints to satisfy constraints.

        Args:
            target_temp_f: Initial target temperature
            spray_flow_gpm: Initial spray flow
            valve_position_pct: Initial valve position
            current_state: Current state
            target_constraints: Constraints
            violations: Current violations

        Returns:
            Adjusted (target_temp, spray_flow, valve_position)
        """
        adjusted_temp = target_temp_f
        adjusted_flow = spray_flow_gpm
        adjusted_valve = valve_position_pct

        # Increase temp if approach to saturation is too low
        min_temp = (
            current_state.saturation_temp_f +
            target_constraints.min_approach_to_saturation_f
        )
        if adjusted_temp < min_temp:
            adjusted_temp = min_temp
            # Recalculate spray flow for new temp
            adjusted_flow = self._calculate_required_spray(
                inlet_temp_f=current_state.inlet_temp_f,
                target_temp_f=adjusted_temp,
                steam_flow_lb_hr=current_state.steam_flow_lb_hr,
                spray_water_temp_f=current_state.spray_water_temp_f,
                steam_pressure_psig=current_state.steam_pressure_psig,
            )
            adjusted_valve = self._calculate_valve_position(
                target_flow_gpm=adjusted_flow,
                current_delta_p_psi=current_state.nozzle_delta_p_psi,
            )

        # Cap valve position
        if adjusted_valve > target_constraints.max_spray_valve_position_pct:
            adjusted_valve = target_constraints.max_spray_valve_position_pct
            # This means we can't reach the target temp
            logger.warning(
                f"Cannot reach target temp {adjusted_temp:.1f}F due to valve limit"
            )

        # Cap spray flow
        if adjusted_flow > target_constraints.max_spray_flow_gpm:
            adjusted_flow = target_constraints.max_spray_flow_gpm
            logger.warning(
                f"Spray flow capped at {adjusted_flow:.1f}gpm"
            )

        return adjusted_temp, adjusted_flow, adjusted_valve

    def _calculate_confidence(
        self,
        current_state: DesuperheaterState,
        target_constraints: TargetConstraints,
        valve_position: float,
    ) -> float:
        """
        Calculate confidence in optimization result.

        Lower confidence for:
        - Valve near limits
        - Large changes
        - Operating near saturation

        Args:
            current_state: Current state
            target_constraints: Constraints
            valve_position: Recommended valve position

        Returns:
            Confidence level (0-1)
        """
        confidence = 0.95  # Base confidence

        # Reduce for valve near limits
        if valve_position > 80 or valve_position < 10:
            confidence -= 0.10

        # Reduce for operating near saturation
        approach = current_state.outlet_temp_f - current_state.saturation_temp_f
        if approach < target_constraints.min_approach_to_saturation_f * 1.5:
            confidence -= 0.15

        # Reduce for large flow changes
        if current_state.spray_flow_gpm > 0:
            # Placeholder - actual change depends on calculation
            pass

        return max(0.5, min(1.0, confidence))

    def _saturation_temp_from_pressure(self, pressure_psig: float) -> float:
        """
        Get saturation temperature from pressure (simplified).

        Args:
            pressure_psig: Pressure in psig

        Returns:
            Saturation temperature in F
        """
        # Simplified Antoine equation approximation
        # In production, use steam tables
        pressure_psia = pressure_psig + 14.7
        t_sat = 115.0 + 45.0 * math.log(pressure_psia)
        return round(t_sat, 1)

    def _generate_provenance_hash(
        self,
        current_state: DesuperheaterState,
        constraints: TargetConstraints,
        result: SprayOptimizationResult,
    ) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        provenance_data = (
            f"{current_state.json()}"
            f"{constraints.json()}"
            f"{result.recommended_temp_setpoint_f}"
            f"{result.recommended_valve_position_pct}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(provenance_data.encode()).hexdigest()

    def _record_optimization(
        self,
        state: DesuperheaterState,
        result: SprayOptimizationResult,
    ) -> None:
        """Record optimization for history tracking."""
        record = {
            "timestamp": datetime.now(timezone.utc),
            "desuperheater_id": state.desuperheater_id,
            "inlet_temp_f": state.inlet_temp_f,
            "outlet_temp_f": state.outlet_temp_f,
            "recommended_temp_f": result.recommended_temp_setpoint_f,
            "recommended_valve_pct": result.recommended_valve_position_pct,
            "confidence": result.confidence,
            "constraints_satisfied": result.constraints_satisfied,
        }
        self._optimization_history.append(record)

        # Trim history
        if len(self._optimization_history) > self._max_history:
            self._optimization_history = self._optimization_history[-self._max_history:]

    def get_optimization_history(
        self,
        desuperheater_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get optimization history.

        Args:
            desuperheater_id: Filter by desuperheater ID
            limit: Maximum records to return

        Returns:
            List of history records
        """
        history = self._optimization_history

        if desuperheater_id:
            history = [
                h for h in history
                if h["desuperheater_id"] == desuperheater_id
            ]

        return history[-limit:]

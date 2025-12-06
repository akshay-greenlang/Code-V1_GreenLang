"""
GL-002 BoilerOptimizer Agent - Combustion Module

Provides combustion optimization including air-fuel ratio control,
burner optimization, and combustion air preheating.

Consolidates: GL-004 (Burner), GL-005 (Air-Fuel), GL-018 (Air Preheater)
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
    CombustionInput as CalcCombustionInput,
)

logger = logging.getLogger(__name__)


class CombustionInput(BaseModel):
    """Input for combustion optimization."""

    # Fuel
    fuel_type: str = Field(..., description="Fuel type")
    fuel_flow_rate: float = Field(..., gt=0, description="Fuel flow (lb/hr or SCF/hr)")
    fuel_pressure_psig: Optional[float] = Field(default=None)
    fuel_temperature_f: Optional[float] = Field(default=None)

    # Air
    combustion_air_flow_lb_hr: Optional[float] = Field(default=None)
    combustion_air_temperature_f: float = Field(default=77.0)
    air_damper_position_pct: Optional[float] = Field(default=None, ge=0, le=100)

    # Flue gas
    flue_gas_o2_pct: float = Field(..., ge=0, le=21)
    flue_gas_co_ppm: float = Field(default=0.0, ge=0)
    flue_gas_co2_pct: Optional[float] = Field(default=None, ge=0, le=20)
    flue_gas_nox_ppm: Optional[float] = Field(default=None, ge=0)
    flue_gas_temperature_f: float = Field(...)

    # Burner
    burner_firing_rate_pct: Optional[float] = Field(default=None, ge=0, le=100)
    number_burners_online: Optional[int] = Field(default=None, ge=1)

    # Air preheater
    air_preheater_inlet_temp_f: Optional[float] = Field(default=None)
    air_preheater_outlet_temp_f: Optional[float] = Field(default=None)


class CombustionOutput(BaseModel):
    """Output from combustion optimization."""

    # Calculated values
    excess_air_pct: float = Field(..., description="Excess air percentage")
    air_fuel_ratio: float = Field(..., description="Actual air-fuel ratio")
    stoichiometric_ratio: float = Field(..., description="Stoichiometric ratio")
    combustion_efficiency_pct: float = Field(...)

    # Losses
    stack_loss_pct: float = Field(...)
    co_loss_pct: float = Field(default=0.0)
    radiation_loss_pct: float = Field(default=0.0)

    # Setpoints
    optimal_o2_pct: float = Field(...)
    recommended_air_damper_pct: Optional[float] = Field(default=None)

    # Air preheater
    air_preheater_effectiveness: Optional[float] = Field(default=None)
    air_preheater_duty_btu_hr: Optional[float] = Field(default=None)
    air_preheat_savings_pct: Optional[float] = Field(default=None)

    # Actions
    adjustments: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class CombustionOptimizer:
    """
    Combustion optimization engine.

    Provides real-time combustion optimization including:
    - Excess air calculation and optimization
    - Air-fuel ratio control
    - CO minimization
    - NOx reduction strategies
    - Air preheater performance
    """

    def __init__(
        self,
        fuel_type: str = "natural_gas",
        target_o2_pct: float = 3.0,
        max_co_ppm: float = 100.0,
        max_nox_ppm: float = 30.0,
    ) -> None:
        """
        Initialize combustion optimizer.

        Args:
            fuel_type: Primary fuel type
            target_o2_pct: Target O2 percentage
            max_co_ppm: Maximum allowable CO
            max_nox_ppm: Maximum allowable NOx
        """
        self.fuel_type = fuel_type
        self.target_o2_pct = target_o2_pct
        self.max_co_ppm = max_co_ppm
        self.max_nox_ppm = max_nox_ppm

        self.calc_library = ThermalIQCalculationLibrary()

        # Fuel properties
        self._fuel_properties = {
            "natural_gas": {
                "stoich_air": 17.2,
                "optimal_o2_high_fire": 2.5,
                "optimal_o2_low_fire": 4.0,
                "co2_max": 12.0,
            },
            "no2_fuel_oil": {
                "stoich_air": 14.4,
                "optimal_o2_high_fire": 3.5,
                "optimal_o2_low_fire": 5.0,
                "co2_max": 15.5,
            },
            "no6_fuel_oil": {
                "stoich_air": 13.8,
                "optimal_o2_high_fire": 4.0,
                "optimal_o2_low_fire": 5.5,
                "co2_max": 16.0,
            },
        }

        # O2 trim history for adaptive control
        self._o2_history: List[Tuple[datetime, float]] = []
        self._co_history: List[Tuple[datetime, float]] = []

        logger.info(f"CombustionOptimizer initialized for {fuel_type}")

    def optimize(self, input_data: CombustionInput) -> CombustionOutput:
        """
        Perform combustion optimization.

        Args:
            input_data: Current combustion data

        Returns:
            CombustionOutput with optimization results
        """
        warnings = []
        adjustments = []

        # Get fuel properties
        fuel_key = input_data.fuel_type.lower().replace(" ", "_")
        fuel_props = self._fuel_properties.get(
            fuel_key,
            self._fuel_properties["natural_gas"]
        )

        # Calculate excess air from O2
        o2 = input_data.flue_gas_o2_pct
        excess_air = (o2 / (21 - o2)) * 100 if o2 < 21 else 100

        # Calculate air-fuel ratio
        stoich_air = fuel_props["stoich_air"]
        air_fuel_ratio = stoich_air * (1 + excess_air / 100)

        # Calculate combustion efficiency
        calc_input = CalcCombustionInput(
            fuel_type=input_data.fuel_type,
            flue_gas_o2_pct=input_data.flue_gas_o2_pct,
            flue_gas_co_ppm=input_data.flue_gas_co_ppm,
            flue_gas_temperature_f=input_data.flue_gas_temperature_f,
            combustion_air_temperature_f=input_data.combustion_air_temperature_f,
        )
        comb_result = self.calc_library.calculate_combustion_efficiency(calc_input)

        # Determine optimal O2 based on firing rate
        firing_rate = input_data.burner_firing_rate_pct or 75
        optimal_o2_high = fuel_props["optimal_o2_high_fire"]
        optimal_o2_low = fuel_props["optimal_o2_low_fire"]

        # Linear interpolation between low and high fire
        optimal_o2 = optimal_o2_low - (
            (firing_rate / 100) * (optimal_o2_low - optimal_o2_high)
        )

        # Calculate recommended damper position
        recommended_damper = None
        if input_data.air_damper_position_pct is not None:
            current_damper = input_data.air_damper_position_pct
            o2_error = o2 - optimal_o2

            # Simple proportional adjustment
            damper_adjustment = o2_error * 2.0  # 2% per 1% O2

            recommended_damper = max(20, min(100, current_damper - damper_adjustment))

            if abs(damper_adjustment) > 5:
                adjustments.append({
                    "parameter": "air_damper",
                    "current": current_damper,
                    "recommended": recommended_damper,
                    "reason": f"O2 deviation: {o2_error:.1f}%",
                })

        # Check CO
        if input_data.flue_gas_co_ppm > self.max_co_ppm:
            warnings.append(
                f"CO at {input_data.flue_gas_co_ppm:.0f} ppm exceeds limit "
                f"({self.max_co_ppm:.0f} ppm)"
            )
            adjustments.append({
                "parameter": "air_increase",
                "reason": "High CO - increase combustion air",
            })

        # Check NOx
        if input_data.flue_gas_nox_ppm and input_data.flue_gas_nox_ppm > self.max_nox_ppm:
            warnings.append(
                f"NOx at {input_data.flue_gas_nox_ppm:.0f} ppm exceeds limit "
                f"({self.max_nox_ppm:.0f} ppm)"
            )

        # Air preheater analysis
        aph_effectiveness = None
        aph_duty = None
        aph_savings = None

        if (input_data.air_preheater_inlet_temp_f and
                input_data.air_preheater_outlet_temp_f):
            # Calculate air preheater performance
            temp_rise = (
                input_data.air_preheater_outlet_temp_f -
                input_data.air_preheater_inlet_temp_f
            )

            max_temp_rise = (
                input_data.flue_gas_temperature_f -
                input_data.air_preheater_inlet_temp_f
            )

            if max_temp_rise > 0:
                aph_effectiveness = temp_rise / max_temp_rise

            # Estimate duty
            if input_data.combustion_air_flow_lb_hr:
                aph_duty = (
                    input_data.combustion_air_flow_lb_hr *
                    0.24 * temp_rise
                )

            # Estimate savings
            if temp_rise > 0:
                # Rule of thumb: 1% efficiency gain per 40F air preheat
                aph_savings = temp_rise / 40.0

        # CO loss
        co_loss = (input_data.flue_gas_co_ppm / 100) * 0.2

        return CombustionOutput(
            excess_air_pct=round(excess_air, 1),
            air_fuel_ratio=round(air_fuel_ratio, 2),
            stoichiometric_ratio=stoich_air,
            combustion_efficiency_pct=round(comb_result.value, 2),
            stack_loss_pct=comb_result.metadata.get("stack_loss_pct", 0),
            co_loss_pct=round(co_loss, 3),
            optimal_o2_pct=round(optimal_o2, 2),
            recommended_air_damper_pct=(
                round(recommended_damper, 1) if recommended_damper else None
            ),
            air_preheater_effectiveness=(
                round(aph_effectiveness, 3) if aph_effectiveness else None
            ),
            air_preheater_duty_btu_hr=aph_duty,
            air_preheat_savings_pct=(
                round(aph_savings, 2) if aph_savings else None
            ),
            adjustments=adjustments,
            warnings=warnings,
        )


class AirFuelRatioController:
    """
    Air-fuel ratio controller with O2 trim.

    Implements a cascaded control strategy:
    1. Primary: Fuel-air cross-limiting
    2. Secondary: O2 trim correction
    """

    def __init__(
        self,
        target_o2_pct: float = 3.0,
        o2_deadband: float = 0.2,
        max_trim_pct: float = 10.0,
        trim_rate_pct_per_min: float = 1.0,
    ) -> None:
        """
        Initialize air-fuel ratio controller.

        Args:
            target_o2_pct: O2 setpoint
            o2_deadband: Control deadband
            max_trim_pct: Maximum trim correction
            trim_rate_pct_per_min: Maximum trim rate
        """
        self.target_o2_pct = target_o2_pct
        self.o2_deadband = o2_deadband
        self.max_trim_pct = max_trim_pct
        self.trim_rate = trim_rate_pct_per_min

        self._current_trim = 0.0
        self._last_update = datetime.now(timezone.utc)

        logger.info(
            f"AirFuelRatioController initialized: "
            f"O2 SP={target_o2_pct}%, deadband={o2_deadband}%"
        )

    def calculate_trim(self, actual_o2_pct: float, co_ppm: float = 0.0) -> float:
        """
        Calculate O2 trim correction.

        Args:
            actual_o2_pct: Measured O2 percentage
            co_ppm: Measured CO (for safety override)

        Returns:
            Trim correction percentage (+ = more air)
        """
        now = datetime.now(timezone.utc)
        dt = (now - self._last_update).total_seconds() / 60.0
        self._last_update = now

        # Safety check: if CO high, don't reduce air
        if co_ppm > 100:
            # Hold or increase air
            if self._current_trim < 0:
                self._current_trim = 0
            return self._current_trim

        # Calculate error
        error = actual_o2_pct - self.target_o2_pct

        # Apply deadband
        if abs(error) < self.o2_deadband:
            return self._current_trim

        # Calculate desired trim change
        # Proportional gain: 2% trim per 1% O2 error
        desired_trim = -error * 2.0

        # Rate limit
        max_change = self.trim_rate * dt
        trim_change = max(-max_change, min(max_change, desired_trim - self._current_trim))

        # Apply change
        self._current_trim += trim_change

        # Limit total trim
        self._current_trim = max(
            -self.max_trim_pct,
            min(self.max_trim_pct, self._current_trim)
        )

        return self._current_trim

    def reset(self) -> None:
        """Reset trim to zero."""
        self._current_trim = 0.0
        logger.info("O2 trim reset to zero")

    @property
    def current_trim(self) -> float:
        """Get current trim value."""
        return self._current_trim

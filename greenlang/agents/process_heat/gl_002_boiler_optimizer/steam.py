"""
GL-002 BoilerOptimizer Agent - Steam System Module

Provides steam system analytics including drum level control,
deaerator optimization, and blowdown management.

Consolidates: GL-003 (Steam), GL-012 (Drum Level), GL-017 (Deaerator)
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import logging

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.shared.calculation_library import (
    ThermalIQCalculationLibrary,
)

logger = logging.getLogger(__name__)


class SteamInput(BaseModel):
    """Input for steam system analysis."""

    # Steam conditions
    steam_pressure_psig: float = Field(..., ge=0)
    steam_temperature_f: Optional[float] = Field(default=None)
    steam_flow_rate_lb_hr: float = Field(..., gt=0)

    # Feedwater
    feedwater_temperature_f: float = Field(default=200.0)
    feedwater_flow_rate_lb_hr: float = Field(..., gt=0)
    feedwater_tds_ppm: Optional[float] = Field(default=None)

    # Drum
    drum_level_in: Optional[float] = Field(default=None, ge=-12, le=12)
    drum_pressure_psig: Optional[float] = Field(default=None)

    # Blowdown
    blowdown_rate_pct: float = Field(default=2.0, ge=0, le=20)
    blowdown_tds_ppm: Optional[float] = Field(default=None)
    blowdown_continuous_lb_hr: Optional[float] = Field(default=None)
    blowdown_intermittent_lb_hr: Optional[float] = Field(default=None)

    # Deaerator
    deaerator_pressure_psig: Optional[float] = Field(default=None)
    deaerator_temperature_f: Optional[float] = Field(default=None)
    deaerator_o2_ppb: Optional[float] = Field(default=None)

    # Returns
    condensate_return_rate_pct: float = Field(default=80.0, ge=0, le=100)
    condensate_temperature_f: Optional[float] = Field(default=None)


class SteamOutput(BaseModel):
    """Output from steam system analysis."""

    # Steam properties
    steam_enthalpy_btu_lb: float = Field(...)
    feedwater_enthalpy_btu_lb: float = Field(...)
    heat_added_btu_lb: float = Field(...)
    saturation_temperature_f: float = Field(...)
    superheat_f: Optional[float] = Field(default=None)

    # Mass balance
    makeup_rate_pct: float = Field(...)
    steam_to_feedwater_ratio: float = Field(...)
    cycles_of_concentration: Optional[float] = Field(default=None)

    # Energy analysis
    blowdown_heat_loss_btu_hr: float = Field(...)
    blowdown_heat_loss_pct: float = Field(...)
    potential_recovery_btu_hr: Optional[float] = Field(default=None)

    # Deaerator
    deaerator_performance: Optional[Dict[str, Any]] = Field(default=None)

    # Recommendations
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DrumLevelControlType(Enum):
    """Drum level control types."""
    SINGLE_ELEMENT = "single_element"
    TWO_ELEMENT = "two_element"
    THREE_ELEMENT = "three_element"


class SteamSystemAnalyzer:
    """
    Steam system analyzer for boiler optimization.

    Analyzes steam system performance including:
    - Steam/water mass balance
    - Blowdown optimization
    - Heat recovery opportunities
    - Deaerator performance
    """

    def __init__(
        self,
        design_pressure_psig: float = 150.0,
        design_blowdown_pct: float = 3.0,
        tds_limit_ppm: float = 3500.0,
    ) -> None:
        """
        Initialize steam system analyzer.

        Args:
            design_pressure_psig: Design steam pressure
            design_blowdown_pct: Design blowdown rate
            tds_limit_ppm: Blowdown TDS limit
        """
        self.design_pressure = design_pressure_psig
        self.design_blowdown = design_blowdown_pct
        self.tds_limit = tds_limit_ppm

        self.calc_library = ThermalIQCalculationLibrary()

        logger.info(
            f"SteamSystemAnalyzer initialized: "
            f"P={design_pressure_psig} psig, BD={design_blowdown_pct}%"
        )

    def analyze(self, input_data: SteamInput) -> SteamOutput:
        """
        Analyze steam system performance.

        Args:
            input_data: Steam system operating data

        Returns:
            SteamOutput with analysis results
        """
        warnings = []
        recommendations = []

        # Calculate steam enthalpy
        steam_enthalpy = self.calc_library._calculate_steam_enthalpy(
            pressure_psig=input_data.steam_pressure_psig,
            temperature_f=input_data.steam_temperature_f,
        )

        # Calculate feedwater enthalpy
        feedwater_enthalpy = self.calc_library._calculate_water_enthalpy(
            temperature_f=input_data.feedwater_temperature_f
        )

        # Get saturation temperature
        sat_temp = self.calc_library._get_saturation_temperature(
            input_data.steam_pressure_psig
        )

        # Calculate superheat if applicable
        superheat = None
        if input_data.steam_temperature_f and input_data.steam_temperature_f > sat_temp:
            superheat = input_data.steam_temperature_f - sat_temp

        # Heat added per pound of steam
        heat_added = steam_enthalpy - feedwater_enthalpy

        # Mass balance
        steam_to_fw_ratio = (
            input_data.steam_flow_rate_lb_hr /
            input_data.feedwater_flow_rate_lb_hr
        )

        # Makeup rate
        makeup_rate = 100 - input_data.condensate_return_rate_pct

        # Cycles of concentration (if TDS data available)
        cycles = None
        if input_data.blowdown_tds_ppm and input_data.feedwater_tds_ppm:
            if input_data.feedwater_tds_ppm > 0:
                cycles = input_data.blowdown_tds_ppm / input_data.feedwater_tds_ppm

        # Blowdown heat loss
        blowdown_flow = (
            input_data.steam_flow_rate_lb_hr *
            input_data.blowdown_rate_pct / 100
        )
        blowdown_enthalpy = self.calc_library._calculate_saturated_water_enthalpy(
            pressure_psig=input_data.steam_pressure_psig
        )
        blowdown_heat_loss = blowdown_flow * (blowdown_enthalpy - feedwater_enthalpy)

        # Heat input
        heat_input = input_data.steam_flow_rate_lb_hr * heat_added

        # Blowdown loss percentage
        blowdown_loss_pct = (blowdown_heat_loss / heat_input * 100) if heat_input > 0 else 0

        # Potential heat recovery (assuming 80% recovery from flash tank)
        potential_recovery = blowdown_heat_loss * 0.8

        # Analyze blowdown
        if input_data.blowdown_rate_pct > self.design_blowdown * 1.5:
            warnings.append(
                f"High blowdown rate: {input_data.blowdown_rate_pct:.1f}% "
                f"vs design {self.design_blowdown:.1f}%"
            )
            recommendations.append({
                "category": "blowdown",
                "action": "Review water treatment program",
                "potential_savings_pct": input_data.blowdown_rate_pct - self.design_blowdown,
            })

        # Check cycles of concentration
        if cycles and cycles < 3:
            recommendations.append({
                "category": "water_treatment",
                "action": f"Low cycles of concentration ({cycles:.1f}). "
                         "Consider improving water treatment to increase cycles.",
                "potential_savings_pct": 0.5,
            })

        # Deaerator analysis
        deaerator_perf = None
        if input_data.deaerator_o2_ppb is not None:
            deaerator_perf = self._analyze_deaerator(input_data)
            if deaerator_perf.get("warning"):
                warnings.append(deaerator_perf["warning"])

        return SteamOutput(
            steam_enthalpy_btu_lb=round(steam_enthalpy, 2),
            feedwater_enthalpy_btu_lb=round(feedwater_enthalpy, 2),
            heat_added_btu_lb=round(heat_added, 2),
            saturation_temperature_f=round(sat_temp, 1),
            superheat_f=round(superheat, 1) if superheat else None,
            makeup_rate_pct=round(makeup_rate, 1),
            steam_to_feedwater_ratio=round(steam_to_fw_ratio, 3),
            cycles_of_concentration=round(cycles, 1) if cycles else None,
            blowdown_heat_loss_btu_hr=round(blowdown_heat_loss, 0),
            blowdown_heat_loss_pct=round(blowdown_loss_pct, 2),
            potential_recovery_btu_hr=round(potential_recovery, 0),
            deaerator_performance=deaerator_perf,
            recommendations=recommendations,
            warnings=warnings,
        )

    def _analyze_deaerator(self, input_data: SteamInput) -> Dict[str, Any]:
        """Analyze deaerator performance."""
        result = {
            "o2_ppb": input_data.deaerator_o2_ppb,
            "status": "normal",
        }

        # Check O2 level
        if input_data.deaerator_o2_ppb and input_data.deaerator_o2_ppb > 7:
            result["status"] = "high_o2"
            result["warning"] = (
                f"Deaerator O2 at {input_data.deaerator_o2_ppb:.1f} ppb "
                "exceeds 7 ppb limit"
            )

        # Check temperature vs pressure
        if input_data.deaerator_pressure_psig and input_data.deaerator_temperature_f:
            expected_temp = self.calc_library._get_saturation_temperature(
                input_data.deaerator_pressure_psig
            )
            temp_diff = input_data.deaerator_temperature_f - expected_temp

            result["expected_temperature_f"] = expected_temp
            result["temperature_deviation_f"] = temp_diff

            if abs(temp_diff) > 5:
                result["status"] = "temperature_deviation"
                result["warning"] = (
                    f"Deaerator temperature deviation: {temp_diff:.1f}F from saturation"
                )

        return result


class DrumLevelController:
    """
    Boiler drum level controller.

    Implements three-element drum level control with:
    - Level (primary)
    - Steam flow (feedforward)
    - Feedwater flow (feedback)
    """

    def __init__(
        self,
        setpoint_in: float = 0.0,
        control_type: DrumLevelControlType = DrumLevelControlType.THREE_ELEMENT,
        kp: float = 1.0,
        ki: float = 0.1,
        shrink_swell_compensation: bool = True,
    ) -> None:
        """
        Initialize drum level controller.

        Args:
            setpoint_in: Level setpoint (inches from center)
            control_type: Control type
            kp: Proportional gain
            ki: Integral gain
            shrink_swell_compensation: Enable shrink/swell compensation
        """
        self.setpoint = setpoint_in
        self.control_type = control_type
        self.kp = kp
        self.ki = ki
        self.shrink_swell_compensation = shrink_swell_compensation

        self._integral = 0.0
        self._last_error = 0.0
        self._last_update = datetime.now(timezone.utc)

        logger.info(
            f"DrumLevelController initialized: "
            f"SP={setpoint_in} in, type={control_type.value}"
        )

    def calculate_feedwater_demand(
        self,
        drum_level_in: float,
        steam_flow_lb_hr: float,
        feedwater_flow_lb_hr: float,
        steam_flow_change_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Calculate feedwater demand.

        Args:
            drum_level_in: Current drum level (inches)
            steam_flow_lb_hr: Steam flow rate
            feedwater_flow_lb_hr: Current feedwater flow
            steam_flow_change_rate: Rate of change of steam flow

        Returns:
            Feedwater demand and control info
        """
        now = datetime.now(timezone.utc)
        dt = (now - self._last_update).total_seconds()
        self._last_update = now

        # Level error
        error = self.setpoint - drum_level_in

        # Check for alarm conditions
        alarm = None
        if drum_level_in > 6:
            alarm = "HIGH_LEVEL"
        elif drum_level_in > 4:
            alarm = "HIGH_LEVEL_WARNING"
        elif drum_level_in < -6:
            alarm = "LOW_LEVEL"
        elif drum_level_in < -4:
            alarm = "LOW_LEVEL_WARNING"

        # PI control on level
        self._integral += error * dt
        self._integral = max(-100, min(100, self._integral))  # Anti-windup

        level_output = self.kp * error + self.ki * self._integral

        # Calculate feedwater demand based on control type
        if self.control_type == DrumLevelControlType.SINGLE_ELEMENT:
            # Just level control
            feedwater_demand = feedwater_flow_lb_hr * (1 + level_output / 100)

        elif self.control_type == DrumLevelControlType.TWO_ELEMENT:
            # Level + steam flow feedforward
            feedwater_demand = steam_flow_lb_hr + (level_output / 100) * steam_flow_lb_hr

        else:  # THREE_ELEMENT
            # Level + steam flow + feedwater feedback
            flow_error = steam_flow_lb_hr - feedwater_flow_lb_hr

            # Shrink/swell compensation
            swell_compensation = 0.0
            if self.shrink_swell_compensation and steam_flow_change_rate != 0:
                # Anticipate swell on load increase, shrink on load decrease
                swell_compensation = -steam_flow_change_rate * 0.1  # Tuning factor

            feedwater_demand = (
                steam_flow_lb_hr +
                flow_error * 0.5 +
                (level_output / 100) * steam_flow_lb_hr +
                swell_compensation
            )

        # Limit output
        feedwater_demand = max(0, min(feedwater_demand, steam_flow_lb_hr * 1.5))

        self._last_error = error

        return {
            "feedwater_demand_lb_hr": round(feedwater_demand, 0),
            "level_error_in": round(error, 2),
            "control_output_pct": round(level_output, 2),
            "integral": round(self._integral, 2),
            "alarm": alarm,
            "control_type": self.control_type.value,
        }

    def reset_integral(self) -> None:
        """Reset integral term."""
        self._integral = 0.0


class DeaeratorOptimizer:
    """
    Deaerator optimization for oxygen removal.

    Optimizes deaerator operation for:
    - Dissolved oxygen removal (<7 ppb)
    - Steam consumption minimization
    - Temperature control
    """

    def __init__(
        self,
        target_o2_ppb: float = 7.0,
        operating_pressure_psig: float = 5.0,
    ) -> None:
        """
        Initialize deaerator optimizer.

        Args:
            target_o2_ppb: Target dissolved O2
            operating_pressure_psig: Operating pressure
        """
        self.target_o2 = target_o2_ppb
        self.operating_pressure = operating_pressure_psig

        self.calc_library = ThermalIQCalculationLibrary()

        # Get saturation temperature
        self.operating_temp = self.calc_library._get_saturation_temperature(
            operating_pressure_psig
        )

        logger.info(
            f"DeaeratorOptimizer initialized: "
            f"P={operating_pressure_psig} psig, T={self.operating_temp:.1f}F"
        )

    def analyze(
        self,
        current_o2_ppb: float,
        current_temperature_f: float,
        makeup_flow_lb_hr: float,
        steam_flow_lb_hr: float,
    ) -> Dict[str, Any]:
        """
        Analyze deaerator performance.

        Args:
            current_o2_ppb: Measured dissolved O2
            current_temperature_f: Operating temperature
            makeup_flow_lb_hr: Makeup water flow
            steam_flow_lb_hr: Steam to deaerator

        Returns:
            Analysis results
        """
        result = {
            "status": "normal",
            "current_o2_ppb": current_o2_ppb,
            "target_o2_ppb": self.target_o2,
            "temperature_f": current_temperature_f,
            "expected_temperature_f": self.operating_temp,
            "recommendations": [],
        }

        # Check O2 level
        if current_o2_ppb > self.target_o2:
            result["status"] = "high_o2"
            result["recommendations"].append(
                "Increase deaerator steam or check spray valves"
            )

        # Check temperature
        temp_deviation = current_temperature_f - self.operating_temp
        if abs(temp_deviation) > 5:
            result["status"] = "temperature_deviation"
            if temp_deviation < 0:
                result["recommendations"].append(
                    "Increase steam to deaerator - temperature below saturation"
                )
            else:
                result["recommendations"].append(
                    "Check pressure control - temperature above expected"
                )

        # Calculate steam efficiency
        # Minimum steam = makeup heating + venting
        makeup_enthalpy_in = self.calc_library._calculate_water_enthalpy(77)  # Assume cold
        makeup_enthalpy_out = self.calc_library._calculate_water_enthalpy(
            current_temperature_f
        )
        heat_needed = makeup_flow_lb_hr * (makeup_enthalpy_out - makeup_enthalpy_in)

        steam_enthalpy = self.calc_library._calculate_steam_enthalpy(
            self.operating_pressure, None
        )
        min_steam_needed = heat_needed / (steam_enthalpy - makeup_enthalpy_out)

        result["min_steam_needed_lb_hr"] = round(min_steam_needed, 0)
        result["actual_steam_lb_hr"] = round(steam_flow_lb_hr, 0)
        result["steam_efficiency_pct"] = round(
            min_steam_needed / steam_flow_lb_hr * 100 if steam_flow_lb_hr > 0 else 0,
            1
        )

        return result

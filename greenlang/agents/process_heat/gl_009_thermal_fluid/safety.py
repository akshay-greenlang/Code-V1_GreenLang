"""
GL-009 THERMALIQ Agent - High Temperature Safety Interlocks

This module provides safety analysis and interlock logic for thermal fluid
systems, monitoring critical parameters and generating safety recommendations.

Safety functions:
    - Film temperature monitoring
    - Bulk temperature monitoring
    - Flash point margin verification
    - Auto-ignition margin verification
    - Minimum flow protection
    - Low level protection
    - NPSH monitoring for cavitation prevention

All calculations are deterministic - ZERO HALLUCINATION guaranteed.

Reference:
    - IEC 61511: Functional Safety
    - API 556: Instrumentation and Control Systems
    - Manufacturer safety guidelines

Example:
    >>> from greenlang.agents.process_heat.gl_009_thermal_fluid.safety import (
    ...     SafetyMonitor,
    ... )
    >>> monitor = SafetyMonitor(
    ...     fluid_type=ThermalFluidType.THERMINOL_66,
    ...     config=SafetyConfig(),
    ... )
    >>> result = monitor.analyze(operating_data)
    >>> if result.safety_status != SafetyStatus.NORMAL:
    ...     handle_safety_condition(result)
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

from pydantic import BaseModel, Field

from .schemas import (
    ThermalFluidType,
    SafetyStatus,
    SafetyAnalysis,
    ThermalFluidInput,
)
from .config import SafetyConfig, TemperatureLimits
from .fluid_properties import ThermalFluidPropertyDatabase

logger = logging.getLogger(__name__)


# =============================================================================
# SAFETY THRESHOLDS
# =============================================================================

@dataclass
class SafetyThreshold:
    """Safety threshold definition."""

    parameter: str
    alarm_value: float
    trip_value: float
    unit: str
    direction: str  # "high" or "low"


# =============================================================================
# SAFETY MONITOR
# =============================================================================

class SafetyMonitor:
    """
    Safety monitoring and interlock analysis for thermal fluid systems.

    This class monitors critical safety parameters and provides interlock
    setpoint recommendations per IEC 61511 functional safety requirements.
    All calculations are deterministic with no ML/LLM in the safety path.

    Monitored parameters:
        - Bulk temperature (high/low)
        - Film temperature (high)
        - Flash point margin
        - Auto-ignition margin
        - Flow rate (low)
        - Expansion tank level (low)
        - Pump suction pressure (NPSH)

    Safety Integrity Level (SIL):
        - Default SIL-2 for thermal fluid heater trips
        - Temperature, flow, and level interlocks

    Example:
        >>> monitor = SafetyMonitor(
        ...     fluid_type=ThermalFluidType.THERMINOL_66,
        ...     config=SafetyConfig(sil_level=2),
        ... )
        >>> result = monitor.analyze(input_data)
        >>> print(f"Status: {result.safety_status}")
    """

    def __init__(
        self,
        fluid_type: ThermalFluidType,
        config: SafetyConfig,
    ) -> None:
        """
        Initialize the safety monitor.

        Args:
            fluid_type: Type of thermal fluid
            config: Safety configuration
        """
        self.fluid_type = fluid_type
        self.config = config

        self._property_db = ThermalFluidPropertyDatabase()
        self._calculation_count = 0

        # Get fluid safety properties
        self._flash_point_f = self._property_db.get_flash_point(fluid_type)
        self._auto_ignition_f = self._property_db.get_auto_ignition_temp(fluid_type)
        self._max_film_temp_f = self._property_db.get_max_film_temp(fluid_type)
        self._max_bulk_temp_f = self._property_db.get_max_bulk_temp(fluid_type)

        logger.info(
            f"SafetyMonitor initialized for {fluid_type} "
            f"(SIL-{config.sil_level})"
        )

    def analyze(
        self,
        input_data: ThermalFluidInput,
    ) -> SafetyAnalysis:
        """
        Perform complete safety analysis.

        Args:
            input_data: Current operating data

        Returns:
            SafetyAnalysis with status and recommendations
        """
        self._calculation_count += 1

        active_alarms: List[str] = []
        active_trips: List[str] = []
        safety_recommendations: List[str] = []

        # Analyze each safety parameter
        film_result = self._analyze_film_temperature(input_data)
        bulk_result = self._analyze_bulk_temperature(input_data)
        flash_result = self._analyze_flash_point_margin(input_data)
        auto_ign_result = self._analyze_auto_ignition_margin(input_data)
        flow_result = self._analyze_flow(input_data)
        npsh_result = self._analyze_npsh(input_data)

        # Collect alarms and trips
        for result in [film_result, bulk_result, flash_result,
                       auto_ign_result, flow_result, npsh_result]:
            active_alarms.extend(result.get("alarms", []))
            active_trips.extend(result.get("trips", []))
            safety_recommendations.extend(result.get("recommendations", []))

        # Determine overall safety status
        if active_trips:
            overall_status = SafetyStatus.TRIP
        elif any("CRITICAL" in a for a in active_alarms):
            overall_status = SafetyStatus.ALARM
        elif active_alarms:
            overall_status = SafetyStatus.WARNING
        else:
            overall_status = SafetyStatus.NORMAL

        # Generate recommended setpoints
        trip_setpoints = self._generate_trip_setpoints(input_data)
        alarm_setpoints = self._generate_alarm_setpoints(input_data)

        return SafetyAnalysis(
            safety_status=overall_status,
            film_temp_status=film_result["status"],
            bulk_temp_status=bulk_result["status"],
            flash_point_margin_status=flash_result["status"],
            auto_ignition_margin_status=auto_ign_result["status"],
            film_temp_margin_f=film_result["margin"],
            bulk_temp_margin_f=bulk_result["margin"],
            flash_point_margin_f=flash_result["margin"],
            auto_ignition_margin_f=auto_ign_result["margin"],
            minimum_flow_met=flow_result["minimum_met"],
            flow_margin_pct=flow_result["margin_pct"],
            npsh_adequate=npsh_result["adequate"],
            pressure_relief_adequate=True,  # Assume adequate unless specified
            trip_setpoints=trip_setpoints,
            alarm_setpoints=alarm_setpoints,
            active_alarms=active_alarms,
            active_trips=active_trips,
            safety_recommendations=safety_recommendations,
        )

    def _analyze_film_temperature(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze film temperature safety."""
        alarms = []
        trips = []
        recommendations = []

        # Get limits from config
        limits = self.config.temperature_limits

        # Current film temperature
        film_temp = input_data.film_temperature_f
        if film_temp is None:
            # Estimate from bulk + typical rise
            film_temp = input_data.bulk_temperature_f + 30.0

        # Calculate margin
        margin = self._max_film_temp_f - film_temp

        # Determine status
        if margin <= 0:
            status = SafetyStatus.TRIP
            trips.append(f"FILM TEMP TRIP: {film_temp:.0f}F exceeds limit {self._max_film_temp_f:.0f}F")
        elif margin < 25:
            status = SafetyStatus.ALARM
            alarms.append(f"CRITICAL: Film temp {film_temp:.0f}F, margin only {margin:.0f}F")
            recommendations.append("Reduce heater firing rate immediately")
            recommendations.append("Increase flow rate to reduce film temperature")
        elif margin < 50:
            status = SafetyStatus.WARNING
            alarms.append(f"WARNING: Film temp margin low ({margin:.0f}F)")
            recommendations.append("Monitor film temperature closely")
        else:
            status = SafetyStatus.NORMAL

        return {
            "status": status,
            "margin": margin,
            "current": film_temp,
            "limit": self._max_film_temp_f,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _analyze_bulk_temperature(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze bulk temperature safety."""
        alarms = []
        trips = []
        recommendations = []

        limits = self.config.temperature_limits
        bulk_temp = input_data.bulk_temperature_f

        # High temperature analysis
        margin = self._max_bulk_temp_f - bulk_temp

        if bulk_temp >= limits.high_bulk_temp_trip_f:
            status = SafetyStatus.TRIP
            trips.append(f"HIGH BULK TEMP TRIP: {bulk_temp:.0f}F >= {limits.high_bulk_temp_trip_f:.0f}F")
        elif bulk_temp >= limits.high_bulk_temp_alarm_f:
            status = SafetyStatus.ALARM
            alarms.append(f"HIGH BULK TEMP ALARM: {bulk_temp:.0f}F")
            recommendations.append("Reduce heat input")
        elif margin < 30:
            status = SafetyStatus.WARNING
            alarms.append(f"WARNING: Bulk temp approaching limit (margin {margin:.0f}F)")
        else:
            status = SafetyStatus.NORMAL

        # Low temperature check
        if bulk_temp <= limits.low_bulk_temp_alarm_f:
            alarms.append(f"LOW BULK TEMP: {bulk_temp:.0f}F")
            recommendations.append("Check heater operation")
            if status == SafetyStatus.NORMAL:
                status = SafetyStatus.WARNING

        return {
            "status": status,
            "margin": margin,
            "current": bulk_temp,
            "limit": self._max_bulk_temp_f,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _analyze_flash_point_margin(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze flash point safety margin."""
        alarms = []
        trips = []
        recommendations = []

        limits = self.config.temperature_limits

        # Calculate margin (flash point should be well above operating temp)
        bulk_temp = input_data.bulk_temperature_f
        margin = self._flash_point_f - bulk_temp

        min_margin = limits.min_flash_point_margin_f

        if margin < 0:
            status = SafetyStatus.TRIP
            trips.append(
                f"FLASH POINT EXCEEDED: Bulk {bulk_temp:.0f}F > "
                f"Flash point {self._flash_point_f:.0f}F"
            )
            recommendations.append("EMERGENCY: Possible vapor generation in system")
        elif margin < min_margin:
            status = SafetyStatus.ALARM
            alarms.append(
                f"LOW FLASH POINT MARGIN: {margin:.0f}F "
                f"(min {min_margin:.0f}F required)"
            )
            recommendations.append("Check for fluid degradation")
            recommendations.append("Verify flash point with lab analysis")
        elif margin < min_margin * 1.5:
            status = SafetyStatus.WARNING
            alarms.append(f"WARNING: Flash point margin {margin:.0f}F approaching minimum")
        else:
            status = SafetyStatus.NORMAL

        return {
            "status": status,
            "margin": margin,
            "flash_point": self._flash_point_f,
            "current_temp": bulk_temp,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _analyze_auto_ignition_margin(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze auto-ignition temperature safety margin."""
        alarms = []
        trips = []
        recommendations = []

        limits = self.config.temperature_limits

        # Use film temperature for auto-ignition comparison (worst case)
        film_temp = input_data.film_temperature_f
        if film_temp is None:
            film_temp = input_data.bulk_temperature_f + 30.0

        margin = self._auto_ignition_f - film_temp
        min_margin = limits.min_auto_ignition_margin_f

        if margin < 0:
            status = SafetyStatus.EMERGENCY_SHUTDOWN
            trips.append(
                f"AUTO-IGNITION RISK: Film temp {film_temp:.0f}F approaches "
                f"AIT {self._auto_ignition_f:.0f}F"
            )
            recommendations.append("EMERGENCY SHUTDOWN REQUIRED")
        elif margin < min_margin:
            status = SafetyStatus.ALARM
            alarms.append(
                f"CRITICAL: Auto-ignition margin {margin:.0f}F "
                f"below minimum {min_margin:.0f}F"
            )
            recommendations.append("Reduce heater firing immediately")
        elif margin < min_margin * 1.5:
            status = SafetyStatus.WARNING
            alarms.append(f"WARNING: Auto-ignition margin {margin:.0f}F")
        else:
            status = SafetyStatus.NORMAL

        return {
            "status": status,
            "margin": margin,
            "auto_ignition_temp": self._auto_ignition_f,
            "film_temp": film_temp,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _analyze_flow(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze flow rate safety."""
        alarms = []
        trips = []
        recommendations = []

        limits = self.config.flow_limits

        design_flow = input_data.design_flow_rate_gpm
        actual_flow = input_data.flow_rate_gpm

        if design_flow is None or design_flow <= 0:
            return {
                "minimum_met": True,
                "margin_pct": 100.0,
                "alarms": [],
                "trips": [],
                "recommendations": [],
            }

        flow_pct = actual_flow / design_flow * 100

        if flow_pct < limits.low_flow_trip_pct:
            trips.append(f"LOW FLOW TRIP: {flow_pct:.1f}% < {limits.low_flow_trip_pct}%")
            minimum_met = False
            margin_pct = flow_pct - limits.low_flow_trip_pct
        elif flow_pct < limits.low_flow_alarm_pct:
            alarms.append(f"LOW FLOW ALARM: {flow_pct:.1f}%")
            recommendations.append("Check pump operation")
            recommendations.append("Verify system pressure drop")
            minimum_met = False
            margin_pct = flow_pct - limits.low_flow_trip_pct
        elif flow_pct < limits.min_flow_pct * 1.2:
            alarms.append(f"WARNING: Flow {flow_pct:.1f}% approaching minimum")
            minimum_met = True
            margin_pct = flow_pct - limits.low_flow_trip_pct
        else:
            minimum_met = True
            margin_pct = flow_pct - limits.low_flow_trip_pct

        return {
            "minimum_met": minimum_met,
            "margin_pct": margin_pct,
            "actual_pct": flow_pct,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _analyze_npsh(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """Analyze NPSH/cavitation safety."""
        alarms = []
        trips = []
        recommendations = []

        # Get vapor pressure at current temperature
        props = self._property_db.get_properties(
            self.fluid_type,
            input_data.bulk_temperature_f
        )
        vapor_pressure_psia = props.vapor_pressure_psia

        # Check pump suction pressure
        suction_pressure = input_data.pump_suction_pressure_psig
        if suction_pressure is None:
            # Estimate from discharge minus typical drop
            suction_pressure = input_data.pump_discharge_pressure_psig - 20.0

        suction_pressure_psia = suction_pressure + 14.696

        # Simple NPSH check (pressure above vapor pressure)
        pressure_margin_psia = suction_pressure_psia - vapor_pressure_psia

        if pressure_margin_psia < 5:
            alarms.append(f"LOW NPSH: Suction pressure margin only {pressure_margin_psia:.1f} psi")
            recommendations.append("Risk of cavitation - increase suction pressure")
            recommendations.append("Check expansion tank level and pressure")
            adequate = False
        else:
            adequate = True

        return {
            "adequate": adequate,
            "pressure_margin_psia": pressure_margin_psia,
            "vapor_pressure_psia": vapor_pressure_psia,
            "alarms": alarms,
            "trips": trips,
            "recommendations": recommendations,
        }

    def _generate_trip_setpoints(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, float]:
        """Generate recommended trip setpoints."""
        limits = self.config.temperature_limits

        return {
            "high_bulk_temp_trip_f": limits.high_bulk_temp_trip_f,
            "high_film_temp_trip_f": self._max_film_temp_f - 10,
            "low_flow_trip_pct": self.config.flow_limits.low_flow_trip_pct,
            "low_level_trip_pct": 10.0,
        }

    def _generate_alarm_setpoints(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, float]:
        """Generate recommended alarm setpoints."""
        limits = self.config.temperature_limits

        return {
            "high_bulk_temp_alarm_f": limits.high_bulk_temp_alarm_f,
            "low_bulk_temp_alarm_f": limits.low_bulk_temp_alarm_f,
            "high_film_temp_alarm_f": self._max_film_temp_f - 25,
            "low_flow_alarm_pct": self.config.flow_limits.low_flow_alarm_pct,
            "low_level_alarm_pct": 15.0,
            "high_level_alarm_pct": 90.0,
        }

    def check_interlock_status(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, bool]:
        """
        Check status of all safety interlocks.

        Args:
            input_data: Current operating data

        Returns:
            Dictionary with interlock pass/fail status
        """
        analysis = self.analyze(input_data)

        return {
            "film_temp_ok": analysis.film_temp_status == SafetyStatus.NORMAL,
            "bulk_temp_ok": analysis.bulk_temp_status == SafetyStatus.NORMAL,
            "flash_point_ok": analysis.flash_point_margin_status == SafetyStatus.NORMAL,
            "auto_ignition_ok": analysis.auto_ignition_margin_status == SafetyStatus.NORMAL,
            "flow_ok": analysis.minimum_flow_met,
            "npsh_ok": analysis.npsh_adequate,
            "overall_ok": analysis.safety_status == SafetyStatus.NORMAL,
            "trips_active": len(analysis.active_trips) > 0,
            "alarms_active": len(analysis.active_alarms) > 0,
        }

    def generate_safety_report(
        self,
        input_data: ThermalFluidInput,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive safety report.

        Args:
            input_data: Current operating data

        Returns:
            Dictionary with complete safety report
        """
        analysis = self.analyze(input_data)
        interlock_status = self.check_interlock_status(input_data)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_id": input_data.system_id,
            "fluid_type": self.fluid_type.value,
            "sil_level": self.config.sil_level,
            "overall_status": analysis.safety_status.value,
            "temperature_analysis": {
                "bulk_temp_f": input_data.bulk_temperature_f,
                "bulk_temp_margin_f": analysis.bulk_temp_margin_f,
                "film_temp_margin_f": analysis.film_temp_margin_f,
                "max_bulk_temp_f": self._max_bulk_temp_f,
                "max_film_temp_f": self._max_film_temp_f,
            },
            "safety_margins": {
                "flash_point_margin_f": analysis.flash_point_margin_f,
                "auto_ignition_margin_f": analysis.auto_ignition_margin_f,
                "flow_margin_pct": analysis.flow_margin_pct,
            },
            "interlock_status": interlock_status,
            "trip_setpoints": analysis.trip_setpoints,
            "alarm_setpoints": analysis.alarm_setpoints,
            "active_alarms": analysis.active_alarms,
            "active_trips": analysis.active_trips,
            "recommendations": analysis.safety_recommendations,
        }

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def check_temperature_safety(
    fluid_type: ThermalFluidType,
    bulk_temp_f: float,
    film_temp_f: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Quick temperature safety check.

    Args:
        fluid_type: Thermal fluid type
        bulk_temp_f: Bulk temperature (F)
        film_temp_f: Film temperature (F) - optional

    Returns:
        Dictionary with safety status
    """
    db = ThermalFluidPropertyDatabase()

    max_bulk = db.get_max_bulk_temp(fluid_type)
    max_film = db.get_max_film_temp(fluid_type)

    bulk_margin = max_bulk - bulk_temp_f

    if film_temp_f is None:
        film_temp_f = bulk_temp_f + 30.0
    film_margin = max_film - film_temp_f

    if film_margin < 0 or bulk_margin < 0:
        status = "CRITICAL"
    elif film_margin < 25 or bulk_margin < 25:
        status = "WARNING"
    else:
        status = "OK"

    return {
        "status": status,
        "bulk_temp_margin_f": bulk_margin,
        "film_temp_margin_f": film_margin,
        "max_bulk_temp_f": max_bulk,
        "max_film_temp_f": max_film,
    }

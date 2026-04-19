# -*- coding: utf-8 -*-
"""
GL-007 FurnaceOptimizer/CoolingTowerOptimizer - Pydantic Data Models

This module provides comprehensive Pydantic data models for furnace
and cooling tower optimization. All models include validation,
documentation, and support for industrial process calculations.

Data Model Categories:
    - Furnace reading models (temperatures, flows, combustion data)
    - Cooling tower reading models (wet bulb, range, approach)
    - Combustion analysis models
    - Heat transfer analysis models
    - Optimization result models
    - Safety status models

Engineering Standards:
    - NFPA 86 for furnace safety
    - ASHRAE for cooling tower performance
    - CTI for cooling tower testing
    - API 560 for fired heater calculations

Example:
    >>> from greenlang.agents.process_heat.gl_007_furnace_optimizer.schemas import (
    ...     FurnaceReading,
    ...     CoolingTowerReading,
    ...     OptimizationResult,
    ... )
    >>> reading = FurnaceReading(
    ...     furnace_id="FUR-001",
    ...     furnace_temp_f=1800.0,
    ...     fuel_flow_rate_scfh=5000.0,
    ...     flue_gas_temp_f=450.0,
    ...     flue_gas_o2_pct=3.0,
    ... )

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================


class SafetyStatus(str, Enum):
    """Safety validation status levels."""
    SAFE = "safe"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    TRIP = "trip"
    VIOLATION = "violation"


class ValidationStatus(str, Enum):
    """Validation status for measurements and calculations."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    UNCHECKED = "unchecked"
    STALE = "stale"


class OperatingMode(str, Enum):
    """Equipment operating mode states."""
    NORMAL = "normal"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    STANDBY = "standby"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class OptimizationStatus(str, Enum):
    """Optimization calculation status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    CONSTRAINTS_VIOLATED = "constraints_violated"
    INFEASIBLE = "infeasible"


class CombustionStatus(str, Enum):
    """Combustion quality status."""
    OPTIMAL = "optimal"
    LEAN = "lean"
    RICH = "rich"
    UNSTABLE = "unstable"
    INCOMPLETE = "incomplete"


class FanStatus(str, Enum):
    """Fan operating status."""
    RUNNING = "running"
    STOPPED = "stopped"
    VARIABLE_SPEED = "variable_speed"
    FAULT = "fault"


# =============================================================================
# FURNACE READING MODELS
# =============================================================================


class FlueGasAnalysis(BaseModel):
    """Flue gas composition analysis."""

    o2_pct: float = Field(
        ...,
        ge=0,
        le=21,
        description="Oxygen content (%)"
    )
    co2_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=25,
        description="CO2 content (%)"
    )
    co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=5000,
        description="Carbon monoxide (ppm)"
    )
    nox_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=2000,
        description="NOx concentration (ppm)"
    )
    so2_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=5000,
        description="SO2 concentration (ppm)"
    )
    excess_air_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Calculated excess air (%)"
    )

    @validator("excess_air_pct", always=True)
    def calculate_excess_air(cls, v, values):
        """Calculate excess air from O2 if not provided."""
        if v is None and "o2_pct" in values:
            o2 = values["o2_pct"]
            # Excess air % = O2 / (21 - O2) * 100
            if o2 < 21:
                return round((o2 / (21 - o2)) * 100, 1)
        return v


class ZoneTemperature(BaseModel):
    """Temperature reading for a furnace zone."""

    zone_id: str = Field(..., description="Zone identifier")
    zone_name: str = Field(default="", description="Zone name")
    temperature_f: float = Field(
        ...,
        ge=-100,
        le=3500,
        description="Zone temperature (F)"
    )
    setpoint_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=3500,
        description="Temperature setpoint (F)"
    )
    deviation_f: Optional[float] = Field(
        default=None,
        description="Deviation from setpoint (F)"
    )
    status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Measurement status"
    )

    @validator("deviation_f", always=True)
    def calculate_deviation(cls, v, values):
        """Calculate deviation from setpoint."""
        if v is None:
            temp = values.get("temperature_f")
            setpoint = values.get("setpoint_f")
            if temp is not None and setpoint is not None:
                return round(temp - setpoint, 1)
        return v


class TubeMetalTemperature(BaseModel):
    """Tube metal temperature reading for fired heaters."""

    sensor_id: str = Field(..., description="Sensor identifier")
    location: str = Field(default="", description="Tube location")
    temperature_f: float = Field(
        ...,
        ge=100,
        le=2500,
        description="Tube metal temperature (F)"
    )
    design_limit_f: float = Field(
        default=1500.0,
        ge=500,
        le=2000,
        description="Design temperature limit (F)"
    )
    margin_f: Optional[float] = Field(
        default=None,
        description="Margin to limit (F)"
    )
    status: SafetyStatus = Field(
        default=SafetyStatus.SAFE,
        description="Safety status"
    )

    @validator("margin_f", always=True)
    def calculate_margin(cls, v, values):
        """Calculate margin to limit."""
        if v is None:
            temp = values.get("temperature_f")
            limit = values.get("design_limit_f")
            if temp is not None and limit is not None:
                return round(limit - temp, 1)
        return v

    @validator("status", always=True)
    def determine_status(cls, v, values):
        """Determine safety status from margin."""
        margin = values.get("margin_f")
        if margin is not None:
            if margin < 0:
                return SafetyStatus.VIOLATION
            elif margin < 25:
                return SafetyStatus.CRITICAL
            elif margin < 50:
                return SafetyStatus.ALARM
            elif margin < 100:
                return SafetyStatus.WARNING
        return SafetyStatus.SAFE


class FurnaceReading(BaseModel):
    """
    Complete furnace operating data reading.

    Captures all measurements needed for furnace optimization
    and performance monitoring.
    """

    # Identification
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Operating status
    operating_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )
    is_firing: bool = Field(
        default=True,
        description="Burner firing status"
    )

    # Primary temperatures
    furnace_temp_f: float = Field(
        ...,
        ge=-100,
        le=3500,
        description="Main furnace temperature (F)"
    )
    furnace_temp_setpoint_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=3500,
        description="Temperature setpoint (F)"
    )

    # Zone temperatures
    zone_temperatures: List[ZoneTemperature] = Field(
        default_factory=list,
        description="Zone temperature readings"
    )

    # Tube metal temperatures (for fired heaters)
    tmt_readings: List[TubeMetalTemperature] = Field(
        default_factory=list,
        description="Tube metal temperature readings"
    )

    # Fuel system
    fuel_flow_rate_scfh: float = Field(
        ...,
        ge=0,
        description="Fuel flow rate (SCFH)"
    )
    fuel_pressure_psig: Optional[float] = Field(
        default=None,
        ge=0,
        le=500,
        description="Fuel pressure (psig)"
    )
    fuel_temp_f: Optional[float] = Field(
        default=None,
        ge=32,
        le=300,
        description="Fuel temperature (F)"
    )

    # Combustion air
    combustion_air_flow_scfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Combustion air flow (SCFM)"
    )
    combustion_air_temp_f: Optional[float] = Field(
        default=None,
        ge=32,
        le=800,
        description="Combustion air temperature (F)"
    )
    combustion_air_pressure_in_wc: Optional[float] = Field(
        default=None,
        ge=-5,
        le=20,
        description="Combustion air pressure (in WC)"
    )

    # Flue gas
    flue_gas_temp_f: float = Field(
        ...,
        ge=100,
        le=2000,
        description="Flue gas temperature (F)"
    )
    flue_gas_analysis: Optional[FlueGasAnalysis] = Field(
        default=None,
        description="Flue gas composition analysis"
    )
    flue_gas_o2_pct: float = Field(
        default=3.0,
        ge=0,
        le=21,
        description="Flue gas O2 content (%)"
    )
    flue_gas_co_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=5000,
        description="Flue gas CO (ppm)"
    )

    # Furnace pressure/draft
    furnace_pressure_in_wc: Optional[float] = Field(
        default=None,
        ge=-2,
        le=2,
        description="Furnace pressure (in WC)"
    )
    stack_draft_in_wc: Optional[float] = Field(
        default=None,
        ge=-5,
        le=0,
        description="Stack draft (in WC)"
    )

    # Heat input/output
    heat_input_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat input (MMBtu/hr)"
    )
    heat_output_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat output (MMBtu/hr)"
    )

    # Process side (for process heaters)
    process_inlet_temp_f: Optional[float] = Field(
        default=None,
        description="Process fluid inlet temperature (F)"
    )
    process_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Process fluid outlet temperature (F)"
    )
    process_flow_rate: Optional[float] = Field(
        default=None,
        ge=0,
        description="Process flow rate"
    )

    # Calculated values
    thermal_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Calculated thermal efficiency (%)"
    )
    excess_air_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=200,
        description="Calculated excess air (%)"
    )

    # Measurement quality
    measurement_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Overall measurement status"
    )

    @validator("excess_air_pct", always=True)
    def calculate_excess_air(cls, v, values):
        """Calculate excess air from O2 if not provided."""
        if v is None:
            o2 = values.get("flue_gas_o2_pct", 3.0)
            if o2 < 21:
                return round((o2 / (21 - o2)) * 100, 1)
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# COOLING TOWER READING MODELS
# =============================================================================


class FanReading(BaseModel):
    """Fan operating data for cooling tower."""

    fan_id: str = Field(..., description="Fan identifier")
    status: FanStatus = Field(
        default=FanStatus.RUNNING,
        description="Fan status"
    )
    speed_pct: float = Field(
        default=100.0,
        ge=0,
        le=110,
        description="Fan speed (%)"
    )
    motor_amps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Motor current (amps)"
    )
    motor_power_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Motor power (kW)"
    )
    vibration_in_sec: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Vibration (in/sec)"
    )


class WaterQuality(BaseModel):
    """Cooling tower water quality parameters."""

    conductivity_umho_cm: Optional[float] = Field(
        default=None,
        ge=0,
        le=10000,
        description="Conductivity (umho/cm)"
    )
    ph: Optional[float] = Field(
        default=None,
        ge=5,
        le=10,
        description="pH value"
    )
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=5000,
        description="Total dissolved solids (ppm)"
    )
    hardness_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=2000,
        description="Total hardness as CaCO3 (ppm)"
    )
    cycles_of_concentration: Optional[float] = Field(
        default=None,
        ge=1,
        le=20,
        description="Cycles of concentration"
    )


class CoolingTowerReading(BaseModel):
    """
    Complete cooling tower operating data reading.

    Captures all measurements needed for cooling tower optimization
    and performance monitoring per CTI standards.
    """

    # Identification
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Operating status
    operating_mode: OperatingMode = Field(
        default=OperatingMode.NORMAL,
        description="Current operating mode"
    )

    # Ambient conditions
    ambient_dry_bulb_f: float = Field(
        ...,
        ge=-40,
        le=130,
        description="Ambient dry bulb temperature (F)"
    )
    ambient_wet_bulb_f: float = Field(
        ...,
        ge=-40,
        le=100,
        description="Ambient wet bulb temperature (F)"
    )
    relative_humidity_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative humidity (%)"
    )
    barometric_pressure_in_hg: Optional[float] = Field(
        default=None,
        ge=25,
        le=32,
        description="Barometric pressure (in Hg)"
    )

    # Water temperatures
    hot_water_temp_f: float = Field(
        ...,
        ge=50,
        le=180,
        description="Hot water inlet temperature (F)"
    )
    cold_water_temp_f: float = Field(
        ...,
        ge=40,
        le=150,
        description="Cold water outlet temperature (F)"
    )

    # Calculated performance
    range_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Temperature range (hot - cold) (F)"
    )
    approach_f: Optional[float] = Field(
        default=None,
        ge=0,
        le=30,
        description="Approach temperature (cold - wet bulb) (F)"
    )

    # Water flow
    water_flow_gpm: float = Field(
        ...,
        ge=0,
        description="Water circulation rate (GPM)"
    )
    makeup_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Makeup water flow (GPM)"
    )
    blowdown_flow_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Blowdown flow (GPM)"
    )

    # Air system
    air_flow_cfm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Air flow rate (CFM)"
    )
    fan_readings: List[FanReading] = Field(
        default_factory=list,
        description="Individual fan readings"
    )
    total_fan_power_kw: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total fan power (kW)"
    )

    # L/G ratio
    lg_ratio: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=5,
        description="Liquid to gas ratio (lb/lb)"
    )

    # Heat rejection
    heat_rejection_mmbtu_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat rejection rate (MMBtu/hr)"
    )
    heat_rejection_tons: Optional[float] = Field(
        default=None,
        ge=0,
        description="Heat rejection (tons)"
    )

    # Water quality
    water_quality: Optional[WaterQuality] = Field(
        default=None,
        description="Water quality parameters"
    )

    # Basin/sump
    basin_level_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Basin level (%)"
    )
    basin_temp_f: Optional[float] = Field(
        default=None,
        ge=32,
        le=150,
        description="Basin temperature (F)"
    )

    # Efficiency metrics
    tower_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Tower efficiency (%)"
    )
    pump_efficiency_gpm_hp: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pump efficiency (GPM/hp)"
    )

    # Measurement quality
    measurement_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Overall measurement status"
    )

    @validator("range_f", always=True)
    def calculate_range(cls, v, values):
        """Calculate range from hot and cold water temps."""
        if v is None:
            hot = values.get("hot_water_temp_f")
            cold = values.get("cold_water_temp_f")
            if hot is not None and cold is not None:
                return round(hot - cold, 1)
        return v

    @validator("approach_f", always=True)
    def calculate_approach(cls, v, values):
        """Calculate approach from cold water and wet bulb."""
        if v is None:
            cold = values.get("cold_water_temp_f")
            wb = values.get("ambient_wet_bulb_f")
            if cold is not None and wb is not None:
                return round(cold - wb, 1)
        return v

    @validator("heat_rejection_tons", always=True)
    def calculate_tons(cls, v, values):
        """Calculate heat rejection in tons."""
        if v is None:
            mmbtu = values.get("heat_rejection_mmbtu_hr")
            if mmbtu is not None:
                return round(mmbtu / 0.012, 1)
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# COMBUSTION ANALYSIS MODELS
# =============================================================================


class CombustionAnalysis(BaseModel):
    """
    Results of combustion analysis calculations.

    All calculations are DETERMINISTIC with zero hallucination.
    """

    # Identification
    analysis_id: str = Field(
        default="",
        description="Analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Air-fuel analysis
    stoichiometric_air_scf_per_scf_fuel: float = Field(
        ...,
        ge=5,
        le=20,
        description="Stoichiometric air requirement (SCF air/SCF fuel)"
    )
    actual_air_scf_per_scf_fuel: float = Field(
        ...,
        ge=5,
        le=50,
        description="Actual air ratio (SCF air/SCF fuel)"
    )
    excess_air_pct: float = Field(
        ...,
        ge=0,
        le=200,
        description="Excess air (%)"
    )
    air_fuel_ratio: float = Field(
        ...,
        ge=10,
        le=30,
        description="Mass air/fuel ratio"
    )

    # Combustion status
    combustion_status: CombustionStatus = Field(
        ...,
        description="Combustion quality status"
    )

    # Flue gas composition (dry basis)
    co2_pct_dry: float = Field(
        ...,
        ge=0,
        le=20,
        description="CO2 in flue gas (% dry)"
    )
    o2_pct_dry: float = Field(
        ...,
        ge=0,
        le=21,
        description="O2 in flue gas (% dry)"
    )
    n2_pct_dry: float = Field(
        ...,
        ge=70,
        le=85,
        description="N2 in flue gas (% dry)"
    )
    h2o_pct_wet: float = Field(
        ...,
        ge=0,
        le=30,
        description="H2O in flue gas (% wet)"
    )

    # Heat analysis
    heat_input_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Heat input (MMBtu/hr)"
    )
    heat_available_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Heat available to process (MMBtu/hr)"
    )

    # Losses breakdown (% of input)
    dry_flue_gas_loss_pct: float = Field(
        ...,
        ge=0,
        le=50,
        description="Dry flue gas heat loss (%)"
    )
    moisture_loss_pct: float = Field(
        ...,
        ge=0,
        le=15,
        description="Moisture in fuel combustion loss (%)"
    )
    radiation_loss_pct: float = Field(
        ...,
        ge=0,
        le=10,
        description="Radiation and convection loss (%)"
    )
    unburned_fuel_loss_pct: float = Field(
        default=0.0,
        ge=0,
        le=10,
        description="Unburned fuel loss (%)"
    )
    total_losses_pct: float = Field(
        ...,
        ge=0,
        le=60,
        description="Total heat losses (%)"
    )

    # Efficiency
    combustion_efficiency_pct: float = Field(
        ...,
        ge=50,
        le=100,
        description="Combustion efficiency (%)"
    )
    thermal_efficiency_pct: float = Field(
        ...,
        ge=40,
        le=100,
        description="Overall thermal efficiency (%)"
    )

    # Emissions
    co_lb_mmbtu: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="CO emissions (lb/MMBtu)"
    )
    nox_lb_mmbtu: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="NOx emissions (lb/MMBtu)"
    )
    co2_lb_mmbtu: float = Field(
        ...,
        ge=100,
        le=200,
        description="CO2 emissions (lb/MMBtu)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )
    formula_references: List[str] = Field(
        default_factory=list,
        description="Engineering formulas used"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# HEAT TRANSFER ANALYSIS MODELS
# =============================================================================


class HeatTransferAnalysis(BaseModel):
    """
    Results of heat transfer analysis calculations.

    All calculations are DETERMINISTIC based on engineering formulas.
    """

    # Identification
    analysis_id: str = Field(
        default="",
        description="Analysis identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Heat duty
    design_duty_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Design heat duty (MMBtu/hr)"
    )
    actual_duty_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Actual heat duty (MMBtu/hr)"
    )
    duty_ratio_pct: float = Field(
        ...,
        ge=0,
        le=150,
        description="Actual vs design duty (%)"
    )

    # Heat transfer
    radiant_heat_transfer_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Radiant heat transfer (MMBtu/hr)"
    )
    convective_heat_transfer_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Convective heat transfer (MMBtu/hr)"
    )
    overall_htc_btu_hr_ft2_f: float = Field(
        ...,
        ge=0,
        le=500,
        description="Overall heat transfer coefficient (Btu/hr-ft2-F)"
    )
    design_htc_btu_hr_ft2_f: float = Field(
        ...,
        ge=0,
        le=500,
        description="Design HTC (Btu/hr-ft2-F)"
    )
    htc_ratio_pct: float = Field(
        ...,
        ge=0,
        le=150,
        description="HTC ratio vs design (%)"
    )

    # Temperature analysis
    lmtd_f: float = Field(
        ...,
        ge=0,
        le=2000,
        description="Log mean temperature difference (F)"
    )
    approach_temp_f: float = Field(
        ...,
        ge=0,
        le=500,
        description="Approach temperature (F)"
    )

    # Fouling analysis
    fouling_factor_hr_ft2_f_btu: float = Field(
        default=0.0,
        ge=0,
        le=0.1,
        description="Estimated fouling factor"
    )
    fouling_severity: str = Field(
        default="low",
        description="Fouling severity (low, moderate, high, severe)"
    )

    # Heat losses
    wall_loss_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Wall heat loss (MMBtu/hr)"
    )
    wall_loss_pct: float = Field(
        ...,
        ge=0,
        le=10,
        description="Wall loss percentage (%)"
    )
    opening_loss_mmbtu_hr: float = Field(
        default=0.0,
        ge=0,
        description="Opening heat loss (MMBtu/hr)"
    )

    # Efficiency
    heat_transfer_effectiveness_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Heat transfer effectiveness (%)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# OPTIMIZATION RESULT MODELS
# =============================================================================


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    parameter: str = Field(..., description="Parameter to adjust")
    current_value: float = Field(..., description="Current value")
    recommended_value: float = Field(..., description="Recommended value")
    unit: str = Field(..., description="Engineering unit")
    expected_improvement_pct: float = Field(
        default=0.0,
        description="Expected efficiency improvement (%)"
    )
    expected_savings_usd_hr: Optional[float] = Field(
        default=None,
        description="Expected hourly savings ($)"
    )
    priority: str = Field(
        default="medium",
        description="Priority (low, medium, high, critical)"
    )
    rationale: str = Field(
        default="",
        description="Explanation for recommendation"
    )


class FurnaceOptimizationResult(BaseModel):
    """
    Complete furnace optimization result.

    Contains all optimization calculations and recommendations
    with full provenance tracking.
    """

    # Identification
    furnace_id: str = Field(..., description="Furnace identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization timestamp"
    )
    execution_id: str = Field(
        default="",
        description="Unique execution identifier"
    )

    # Status
    status: OptimizationStatus = Field(
        ...,
        description="Optimization status"
    )
    safety_status: SafetyStatus = Field(
        ...,
        description="Safety status"
    )

    # Current performance
    current_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current thermal efficiency (%)"
    )
    current_excess_air_pct: float = Field(
        ...,
        ge=0,
        le=200,
        description="Current excess air (%)"
    )
    current_fuel_rate_scfh: float = Field(
        ...,
        ge=0,
        description="Current fuel rate (SCFH)"
    )
    current_heat_input_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Current heat input (MMBtu/hr)"
    )

    # Optimized targets
    optimal_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Optimal thermal efficiency (%)"
    )
    optimal_excess_air_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Optimal excess air (%)"
    )
    optimal_o2_pct: float = Field(
        ...,
        ge=0,
        le=10,
        description="Optimal flue gas O2 (%)"
    )
    optimal_fuel_rate_scfh: float = Field(
        ...,
        ge=0,
        description="Optimal fuel rate (SCFH)"
    )

    # Improvement potential
    efficiency_improvement_pct: float = Field(
        ...,
        description="Potential efficiency improvement (%)"
    )
    fuel_savings_pct: float = Field(
        ...,
        ge=0,
        le=50,
        description="Potential fuel savings (%)"
    )
    estimated_savings_usd_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated hourly savings ($)"
    )
    estimated_savings_usd_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated annual savings ($)"
    )

    # CO2 reduction
    co2_reduction_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="CO2 reduction potential (lb/hr)"
    )
    co2_reduction_tons_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Annual CO2 reduction (tons/year)"
    )

    # Analysis results
    combustion_analysis: Optional[CombustionAnalysis] = Field(
        default=None,
        description="Combustion analysis results"
    )
    heat_transfer_analysis: Optional[HeatTransferAnalysis] = Field(
        default=None,
        description="Heat transfer analysis results"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 calculation hash"
    )
    input_data_hash: str = Field(
        default="",
        description="SHA-256 hash of input data"
    )
    calculation_chain: List[str] = Field(
        default_factory=list,
        description="Calculation step hashes"
    )

    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CoolingTowerOptimizationResult(BaseModel):
    """
    Complete cooling tower optimization result.

    Contains all optimization calculations and recommendations
    with full provenance tracking.
    """

    # Identification
    tower_id: str = Field(..., description="Cooling tower identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Optimization timestamp"
    )
    execution_id: str = Field(
        default="",
        description="Unique execution identifier"
    )

    # Status
    status: OptimizationStatus = Field(
        ...,
        description="Optimization status"
    )
    safety_status: SafetyStatus = Field(
        ...,
        description="Safety status"
    )

    # Current performance
    current_approach_f: float = Field(
        ...,
        ge=0,
        le=50,
        description="Current approach temperature (F)"
    )
    current_range_f: float = Field(
        ...,
        ge=0,
        le=50,
        description="Current temperature range (F)"
    )
    current_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Current tower efficiency (%)"
    )
    current_fan_power_kw: float = Field(
        ...,
        ge=0,
        description="Current fan power (kW)"
    )

    # Optimized targets
    optimal_approach_f: float = Field(
        ...,
        ge=0,
        le=50,
        description="Optimal approach temperature (F)"
    )
    optimal_fan_speed_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Optimal fan speed (%)"
    )
    optimal_lg_ratio: float = Field(
        ...,
        ge=0.3,
        le=3,
        description="Optimal L/G ratio"
    )
    optimal_fan_power_kw: float = Field(
        ...,
        ge=0,
        description="Optimal fan power (kW)"
    )

    # Improvement potential
    approach_improvement_f: float = Field(
        ...,
        description="Approach improvement potential (F)"
    )
    energy_savings_pct: float = Field(
        ...,
        ge=0,
        le=80,
        description="Fan energy savings (%)"
    )
    estimated_savings_kwh_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated hourly energy savings (kWh)"
    )
    estimated_savings_usd_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Estimated annual savings ($)"
    )

    # Water savings
    water_savings_gpm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Makeup water savings (GPM)"
    )
    water_savings_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=50,
        description="Water savings (%)"
    )

    # Merkel analysis
    merkel_number: float = Field(
        ...,
        ge=0,
        le=10,
        description="Merkel number (KaV/L)"
    )
    ntu: float = Field(
        ...,
        ge=0,
        le=10,
        description="Number of Transfer Units"
    )

    # Recommendations
    recommendations: List[OptimizationRecommendation] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 calculation hash"
    )
    input_data_hash: str = Field(
        default="",
        description="SHA-256 hash of input data"
    )

    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Processing time (ms)"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class OptimizationResult(BaseModel):
    """
    Combined optimization result for both furnace and cooling tower.

    Provides a unified interface for GL-007 agent outputs.
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-007",
        description="Agent identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Result timestamp"
    )
    execution_id: str = Field(
        ...,
        description="Unique execution identifier"
    )

    # Results
    furnace_result: Optional[FurnaceOptimizationResult] = Field(
        default=None,
        description="Furnace optimization result"
    )
    cooling_tower_result: Optional[CoolingTowerOptimizationResult] = Field(
        default=None,
        description="Cooling tower optimization result"
    )

    # Overall status
    overall_status: OptimizationStatus = Field(
        ...,
        description="Overall optimization status"
    )
    overall_safety_status: SafetyStatus = Field(
        ...,
        description="Overall safety status"
    )

    # Combined savings
    total_estimated_savings_usd_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total annual savings ($)"
    )
    total_co2_reduction_tons_year: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total CO2 reduction (tons/year)"
    )

    # Provenance
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )

    # Processing metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time (ms)"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "SafetyStatus",
    "ValidationStatus",
    "OperatingMode",
    "OptimizationStatus",
    "CombustionStatus",
    "FanStatus",
    # Furnace Models
    "FlueGasAnalysis",
    "ZoneTemperature",
    "TubeMetalTemperature",
    "FurnaceReading",
    # Cooling Tower Models
    "FanReading",
    "WaterQuality",
    "CoolingTowerReading",
    # Analysis Models
    "CombustionAnalysis",
    "HeatTransferAnalysis",
    # Optimization Models
    "OptimizationRecommendation",
    "FurnaceOptimizationResult",
    "CoolingTowerOptimizationResult",
    "OptimizationResult",
]

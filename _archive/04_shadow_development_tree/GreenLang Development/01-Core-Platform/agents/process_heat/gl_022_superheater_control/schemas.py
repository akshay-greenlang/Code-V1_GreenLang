"""
GL-022 SUPERHEATER CONTROL AGENT - Pydantic Data Models

This module provides comprehensive Pydantic data models for superheater
temperature control and optimization. All models include validation,
documentation, and support for IAPWS-IF97 steam property calculations.

Data Model Categories:
    - Superheater operating point models (inlet/outlet conditions)
    - Desuperheater spray models (spray rate, water quality, valve status)
    - Temperature control models (setpoints, PID outputs, deviations)
    - Tube metal temperature monitoring models
    - Process demand reading models (downstream requirements)
    - Safety interlock status models
    - Comprehensive control output models with provenance

Engineering Standards:
    - IAPWS-IF97 for steam thermodynamic properties
    - ASME PTC 4 for boiler performance testing
    - API 530 for tube metal temperature limits
    - ISA-5.1 for instrumentation and control symbology

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.schemas import (
    ...     SuperheaterOperatingPoint,
    ...     DesuperheaterSprayInput,
    ...     TemperatureControlInput,
    ... )
    >>>
    >>> operating_point = SuperheaterOperatingPoint(
    ...     superheater_id="SH-001",
    ...     inlet_pressure_psig=600.0,
    ...     inlet_temperature_f=750.0,
    ...     outlet_temperature_f=950.0,
    ...     steam_flow_lb_hr=50000.0,
    ... )
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class ControlMode(Enum):
    """Superheater control mode states."""
    AUTO = "auto"
    MANUAL = "manual"
    CASCADE = "cascade"
    REMOTE = "remote"
    LOCAL = "local"
    EMERGENCY = "emergency"


class SprayValveStatus(Enum):
    """Desuperheater spray valve operational status."""
    OPEN = "open"
    CLOSED = "closed"
    MODULATING = "modulating"
    FAILED = "failed"
    MAINTENANCE = "maintenance"
    INTERLOCK_CLOSED = "interlock_closed"


class SafetyStatus(Enum):
    """Safety interlock status levels."""
    NORMAL = "normal"
    WARNING = "warning"
    ALARM = "alarm"
    TRIP = "trip"
    BYPASSED = "bypassed"


class ValidationStatus(Enum):
    """Validation status for measurements and calculations."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    UNCHECKED = "unchecked"
    STALE = "stale"


class SteamPhase(Enum):
    """Steam phase states for superheater operations."""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_VAPOR = "superheated_vapor"


class ThermalStressLevel(Enum):
    """Tube metal thermal stress classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ControlAction(Enum):
    """Control action type classification."""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"
    RAMP_UP = "ramp_up"
    RAMP_DOWN = "ramp_down"
    EMERGENCY_STOP = "emergency_stop"


class AlarmSeverity(Enum):
    """Alarm severity classification per ISA-18.2."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# STEAM PROPERTY MODELS
# =============================================================================

class SteamConditions(BaseModel):
    """
    Steam thermodynamic conditions at a measurement point.

    Based on IAPWS-IF97 formulations with validation for superheated steam.
    """

    # Pressure measurements
    pressure_psig: float = Field(
        ...,
        ge=-14.696,
        description="Gauge pressure (psig)"
    )
    pressure_psia: Optional[float] = Field(
        default=None,
        description="Absolute pressure (psia)"
    )

    # Temperature measurements
    temperature_f: float = Field(
        ...,
        ge=32,
        le=1500,
        description="Steam temperature (F)"
    )
    saturation_temperature_f: Optional[float] = Field(
        default=None,
        description="Saturation temperature at pressure (F)"
    )
    superheat_f: Optional[float] = Field(
        default=None,
        ge=0,
        description="Degrees of superheat above saturation (F)"
    )

    # Thermodynamic properties
    enthalpy_btu_lb: Optional[float] = Field(
        default=None,
        description="Specific enthalpy (BTU/lb)"
    )
    entropy_btu_lb_r: Optional[float] = Field(
        default=None,
        description="Specific entropy (BTU/lb-R)"
    )
    specific_volume_ft3_lb: Optional[float] = Field(
        default=None,
        description="Specific volume (ft3/lb)"
    )
    density_lb_ft3: Optional[float] = Field(
        default=None,
        description="Density (lb/ft3)"
    )
    internal_energy_btu_lb: Optional[float] = Field(
        default=None,
        description="Specific internal energy (BTU/lb)"
    )

    # Phase determination
    phase: SteamPhase = Field(
        default=SteamPhase.SUPERHEATED_VAPOR,
        description="Steam phase state"
    )
    dryness_fraction: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Steam quality/dryness fraction (0-1)"
    )

    # Exergy
    exergy_btu_lb: Optional[float] = Field(
        default=None,
        description="Specific exergy (BTU/lb)"
    )

    @validator('pressure_psia', always=True)
    def calculate_psia(cls, v, values):
        """Calculate absolute pressure from gauge."""
        if v is None and 'pressure_psig' in values:
            return values['pressure_psig'] + 14.696
        return v

    @validator('density_lb_ft3', always=True)
    def calculate_density(cls, v, values):
        """Calculate density from specific volume."""
        if v is None and values.get('specific_volume_ft3_lb'):
            if values['specific_volume_ft3_lb'] > 0:
                return 1.0 / values['specific_volume_ft3_lb']
        return v

    @validator('superheat_f', always=True)
    def calculate_superheat(cls, v, values):
        """Calculate superheat from temperature and saturation."""
        if v is None:
            temp = values.get('temperature_f')
            sat_temp = values.get('saturation_temperature_f')
            if temp is not None and sat_temp is not None:
                superheat = temp - sat_temp
                return max(0, superheat)
        return v

    class Config:
        use_enum_values = True


# =============================================================================
# SUPERHEATER OPERATING POINT MODELS
# =============================================================================

class SuperheaterOperatingPoint(BaseModel):
    """
    Complete superheater operating point data.

    Captures inlet and outlet conditions, flow rates, and heat transfer
    performance for a single superheater or superheater section.
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Superheater equipment identifier"
    )
    section_id: Optional[str] = Field(
        default=None,
        description="Superheater section (primary, secondary, etc.)"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Operating point timestamp"
    )

    # Inlet conditions
    inlet_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Inlet steam pressure (psig)"
    )
    inlet_temperature_f: float = Field(
        ...,
        ge=200,
        le=1200,
        description="Inlet steam temperature (F)"
    )
    inlet_enthalpy_btu_lb: Optional[float] = Field(
        default=None,
        description="Inlet steam enthalpy (BTU/lb)"
    )

    # Outlet conditions
    outlet_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Outlet steam pressure (psig)"
    )
    outlet_temperature_f: float = Field(
        ...,
        ge=200,
        le=1200,
        description="Outlet steam temperature (F)"
    )
    outlet_enthalpy_btu_lb: Optional[float] = Field(
        default=None,
        description="Outlet steam enthalpy (BTU/lb)"
    )

    # Pressure drop
    pressure_drop_psi: Optional[float] = Field(
        default=None,
        ge=0,
        description="Pressure drop across superheater (psi)"
    )

    # Flow
    steam_flow_lb_hr: float = Field(
        ...,
        ge=0,
        description="Steam mass flow rate (lb/hr)"
    )
    steam_flow_klb_hr: Optional[float] = Field(
        default=None,
        description="Steam mass flow rate (klb/hr)"
    )

    # Heat transfer
    heat_duty_btu_hr: Optional[float] = Field(
        default=None,
        description="Heat transfer rate (BTU/hr)"
    )
    heat_duty_mmbtu_hr: Optional[float] = Field(
        default=None,
        description="Heat transfer rate (MMBTU/hr)"
    )
    overall_htc_btu_hr_ft2_f: Optional[float] = Field(
        default=None,
        ge=0,
        description="Overall heat transfer coefficient (BTU/hr-ft2-F)"
    )

    # Flue gas conditions
    flue_gas_inlet_temp_f: Optional[float] = Field(
        default=None,
        description="Flue gas inlet temperature (F)"
    )
    flue_gas_outlet_temp_f: Optional[float] = Field(
        default=None,
        description="Flue gas outlet temperature (F)"
    )

    # Temperature rise
    temperature_rise_f: Optional[float] = Field(
        default=None,
        description="Steam temperature rise across superheater (F)"
    )

    # Measurement quality
    measurement_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Measurement validation status"
    )

    @validator('steam_flow_klb_hr', always=True)
    def calculate_klb(cls, v, values):
        """Calculate klb/hr from lb/hr."""
        if v is None and 'steam_flow_lb_hr' in values:
            return values['steam_flow_lb_hr'] / 1000.0
        return v

    @validator('pressure_drop_psi', always=True)
    def calculate_pressure_drop(cls, v, values):
        """Calculate pressure drop from inlet and outlet."""
        if v is None:
            inlet_p = values.get('inlet_pressure_psig')
            outlet_p = values.get('outlet_pressure_psig')
            if inlet_p is not None and outlet_p is not None:
                return max(0, inlet_p - outlet_p)
        return v

    @validator('temperature_rise_f', always=True)
    def calculate_temp_rise(cls, v, values):
        """Calculate temperature rise from inlet and outlet."""
        if v is None:
            inlet_t = values.get('inlet_temperature_f')
            outlet_t = values.get('outlet_temperature_f')
            if inlet_t is not None and outlet_t is not None:
                return outlet_t - inlet_t
        return v

    @validator('heat_duty_mmbtu_hr', always=True)
    def calculate_mmbtu(cls, v, values):
        """Calculate MMBTU/hr from BTU/hr."""
        if v is None and values.get('heat_duty_btu_hr'):
            return values['heat_duty_btu_hr'] / 1_000_000.0
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# DESUPERHEATER SPRAY MODELS
# =============================================================================

class SprayWaterQuality(BaseModel):
    """Spray water quality parameters for desuperheater."""

    # Basic quality
    tds_ppm: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Total Dissolved Solids (ppm) - should be <10 for spray water"
    )
    conductivity_us_cm: Optional[float] = Field(
        default=None,
        ge=0,
        description="Conductivity (microS/cm)"
    )
    ph: Optional[float] = Field(
        default=None,
        ge=6,
        le=10,
        description="pH value (target 8.5-9.5)"
    )
    silica_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Silica content (ppb) - should be <20 for spray water"
    )
    dissolved_o2_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Dissolved oxygen (ppb) - should be <7 ppb"
    )
    sodium_ppb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sodium content (ppb)"
    )

    # Quality status
    quality_status: ValidationStatus = Field(
        default=ValidationStatus.UNCHECKED,
        description="Overall water quality status"
    )
    quality_violations: List[str] = Field(
        default_factory=list,
        description="List of quality limit violations"
    )

    class Config:
        use_enum_values = True


class DesuperheaterSprayInput(BaseModel):
    """
    Input data for desuperheater spray water calculations.

    Models spray water injection for steam temperature control.
    """

    # Identification
    desuperheater_id: str = Field(
        ...,
        description="Desuperheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Spray water conditions
    spray_water_temp_f: float = Field(
        ...,
        ge=32,
        le=500,
        description="Spray water temperature (F)"
    )
    spray_water_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Spray water pressure (psig)"
    )
    spray_water_enthalpy_btu_lb: Optional[float] = Field(
        default=None,
        description="Spray water enthalpy (BTU/lb)"
    )

    # Current spray flow
    current_spray_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current spray water flow rate (lb/hr)"
    )

    # Spray valve parameters
    valve_position_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Current spray valve position (%)"
    )
    valve_cv: float = Field(
        default=10.0,
        gt=0,
        description="Spray valve Cv rating"
    )
    valve_characteristic: str = Field(
        default="equal_percent",
        description="Valve characteristic (linear, equal_percent, quick_opening)"
    )

    # Capacity limits
    min_spray_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Minimum effective spray flow (lb/hr)"
    )
    max_spray_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Maximum spray flow capacity (lb/hr)"
    )

    # Water quality
    water_quality: Optional[SprayWaterQuality] = Field(
        default=None,
        description="Spray water quality parameters"
    )

    # Pressure differential requirements
    min_dp_psi: float = Field(
        default=50.0,
        ge=0,
        description="Minimum pressure differential for atomization (psi)"
    )

    @validator('spray_water_pressure_psig')
    def validate_pressure_for_spray(cls, v, values):
        """Spray water pressure must be adequate for atomization."""
        if v < 50:
            # Warning: low pressure may affect atomization quality
            pass
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class DesuperheaterSprayOutput(BaseModel):
    """
    Output from desuperheater spray water calculations.

    Contains calculated spray requirements and energy impact.
    """

    # Identification
    desuperheater_id: str = Field(
        ...,
        description="Desuperheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Calculated spray requirements
    required_spray_flow_lb_hr: float = Field(
        ...,
        ge=0,
        description="Required spray water flow rate (lb/hr)"
    )
    target_valve_position_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Target spray valve position (%)"
    )
    spray_to_steam_ratio_pct: float = Field(
        ...,
        ge=0,
        le=30,
        description="Spray to steam flow ratio (%)"
    )

    # Energy balance
    spray_water_enthalpy_btu_lb: float = Field(
        ...,
        description="Spray water enthalpy (BTU/lb)"
    )
    enthalpy_reduction_btu_lb: float = Field(
        ...,
        description="Steam enthalpy reduction from spray (BTU/lb)"
    )
    cooling_duty_btu_hr: float = Field(
        ...,
        description="Cooling duty from spray water (BTU/hr)"
    )
    cooling_duty_mmbtu_hr: Optional[float] = Field(
        default=None,
        description="Cooling duty (MMBTU/hr)"
    )

    # Efficiency impact
    efficiency_loss_pct: float = Field(
        default=0.0,
        description="Thermal efficiency loss from spray (%)"
    )
    energy_cost_usd_hr: Optional[float] = Field(
        default=None,
        description="Energy cost of spray cooling ($/hr)"
    )

    # Evaporation analysis
    evaporation_distance_ft: Optional[float] = Field(
        default=None,
        description="Estimated complete evaporation distance (ft)"
    )
    evaporation_complete: bool = Field(
        default=True,
        description="Spray water fully evaporated before outlet"
    )

    # Valve status
    valve_status: SprayValveStatus = Field(
        default=SprayValveStatus.MODULATING,
        description="Spray valve operational status"
    )

    # Warnings
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )
    formula_reference: str = Field(
        default="IAPWS-IF97 Energy Balance",
        description="Calculation method reference"
    )

    @validator('cooling_duty_mmbtu_hr', always=True)
    def calculate_mmbtu(cls, v, values):
        """Calculate MMBTU/hr from BTU/hr."""
        if v is None and 'cooling_duty_btu_hr' in values:
            return values['cooling_duty_btu_hr'] / 1_000_000.0
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# TEMPERATURE CONTROL MODELS
# =============================================================================

class PIDParameters(BaseModel):
    """PID controller tuning parameters."""

    kp: float = Field(
        ...,
        ge=0,
        description="Proportional gain"
    )
    ki: float = Field(
        ...,
        ge=0,
        description="Integral gain"
    )
    kd: float = Field(
        ...,
        ge=0,
        description="Derivative gain"
    )
    integral_windup_limit: Optional[float] = Field(
        default=None,
        description="Integral windup limit"
    )
    output_min_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Minimum controller output (%)"
    )
    output_max_pct: float = Field(
        default=100.0,
        ge=0,
        le=100,
        description="Maximum controller output (%)"
    )
    sample_time_sec: float = Field(
        default=1.0,
        gt=0,
        description="Controller sample time (seconds)"
    )


class TemperatureControlInput(BaseModel):
    """
    Input data for superheater temperature control calculations.

    Captures current state, setpoints, and control parameters.
    """

    # Identification
    controller_id: str = Field(
        ...,
        description="Temperature controller identifier"
    )
    superheater_id: str = Field(
        ...,
        description="Associated superheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Current temperature conditions
    process_variable_f: float = Field(
        ...,
        ge=200,
        le=1200,
        description="Current outlet temperature - process variable (F)"
    )
    setpoint_f: float = Field(
        ...,
        ge=200,
        le=1100,
        description="Temperature setpoint (F)"
    )

    # Temperature limits
    high_limit_f: float = Field(
        ...,
        ge=400,
        le=1200,
        description="High temperature limit (F)"
    )
    low_limit_f: float = Field(
        ...,
        ge=200,
        le=800,
        description="Low temperature limit (F)"
    )
    deadband_f: float = Field(
        default=2.0,
        ge=0,
        le=20,
        description="Control deadband (F)"
    )

    # Rate limits
    max_rate_of_change_f_per_min: float = Field(
        default=10.0,
        ge=1,
        le=50,
        description="Maximum allowable rate of change (F/min)"
    )
    current_rate_of_change_f_per_min: Optional[float] = Field(
        default=None,
        description="Current rate of temperature change (F/min)"
    )

    # Control mode
    control_mode: ControlMode = Field(
        default=ControlMode.AUTO,
        description="Current control mode"
    )
    cascade_primary_output_pct: Optional[float] = Field(
        default=None,
        description="Primary controller output for cascade mode (%)"
    )

    # PID parameters
    pid_parameters: Optional[PIDParameters] = Field(
        default=None,
        description="PID controller tuning parameters"
    )

    # Current controller state
    current_output_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
        description="Current controller output (%)"
    )
    integral_term: Optional[float] = Field(
        default=None,
        description="Current integral accumulator value"
    )

    # Feedforward inputs
    steam_flow_lb_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Steam flow for feedforward (lb/hr)"
    )
    inlet_temperature_f: Optional[float] = Field(
        default=None,
        description="Inlet temperature for feedforward (F)"
    )
    firing_rate_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Firing rate for feedforward (%)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TemperatureControlOutput(BaseModel):
    """
    Output from superheater temperature control calculations.

    Contains control actions, deviations, and PID contributions.
    """

    # Identification
    controller_id: str = Field(
        ...,
        description="Temperature controller identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Deviation analysis
    deviation_f: float = Field(
        ...,
        description="Deviation from setpoint (F)"
    )
    deviation_pct: float = Field(
        ...,
        description="Deviation as percentage of setpoint"
    )
    deviation_status: ValidationStatus = Field(
        ...,
        description="Deviation status (VALID if within tolerance)"
    )
    within_deadband: bool = Field(
        ...,
        description="Process variable within deadband"
    )

    # Control output
    controller_output_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Calculated controller output (%)"
    )
    output_change_pct: float = Field(
        ...,
        description="Change from previous output (%)"
    )
    control_action: ControlAction = Field(
        ...,
        description="Control action type"
    )

    # PID contributions
    proportional_contribution: float = Field(
        ...,
        description="Proportional term contribution"
    )
    integral_contribution: float = Field(
        ...,
        description="Integral term contribution"
    )
    derivative_contribution: float = Field(
        ...,
        description="Derivative term contribution"
    )
    feedforward_contribution: Optional[float] = Field(
        default=None,
        description="Feedforward term contribution"
    )

    # Rate limiting
    rate_limited: bool = Field(
        default=False,
        description="Output was rate limited"
    )
    pre_limit_output_pct: Optional[float] = Field(
        default=None,
        description="Output before rate limiting (%)"
    )

    # Control performance metrics
    integrated_error: Optional[float] = Field(
        default=None,
        description="Integrated absolute error"
    )
    integral_of_time_error: Optional[float] = Field(
        default=None,
        description="Integral of time-weighted absolute error"
    )

    # Temperature trend
    temperature_trend: str = Field(
        default="stable",
        description="Temperature trend (rising, falling, stable)"
    )
    projected_temperature_f: Optional[float] = Field(
        default=None,
        description="Projected temperature in 1 minute (F)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Control recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 calculation hash"
    )
    processing_time_ms: float = Field(
        default=0.0,
        description="Processing time (ms)"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# TUBE METAL TEMPERATURE MODELS
# =============================================================================

class TubeMetalTemperatureReading(BaseModel):
    """
    Individual tube metal temperature measurement.

    Critical for preventing tube overheating and creep damage per API 530.
    """

    # Identification
    sensor_id: str = Field(
        ...,
        description="Temperature sensor identifier"
    )
    tube_location: str = Field(
        ...,
        description="Tube location description"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Temperature measurement
    temperature_f: float = Field(
        ...,
        ge=200,
        le=1500,
        description="Measured tube metal temperature (F)"
    )
    temperature_c: Optional[float] = Field(
        default=None,
        description="Measured tube metal temperature (C)"
    )

    # Design limits per API 530
    design_temperature_f: float = Field(
        ...,
        description="Design temperature limit (F)"
    )
    allowable_stress_temperature_f: Optional[float] = Field(
        default=None,
        description="Allowable stress design temperature (F)"
    )

    # Margin calculations
    margin_to_limit_f: Optional[float] = Field(
        default=None,
        description="Margin below design limit (F)"
    )
    margin_pct: Optional[float] = Field(
        default=None,
        description="Margin as percentage of limit"
    )

    # Alarm status
    high_alarm_f: float = Field(
        ...,
        description="High temperature alarm setpoint (F)"
    )
    high_high_alarm_f: float = Field(
        ...,
        description="High-high (trip) temperature setpoint (F)"
    )
    alarm_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Current alarm status"
    )

    # Measurement quality
    measurement_status: ValidationStatus = Field(
        default=ValidationStatus.VALID,
        description="Sensor measurement status"
    )
    sensor_health: str = Field(
        default="healthy",
        description="Sensor health status"
    )

    @validator('temperature_c', always=True)
    def calculate_celsius(cls, v, values):
        """Calculate Celsius from Fahrenheit."""
        if v is None and 'temperature_f' in values:
            return (values['temperature_f'] - 32) * 5.0 / 9.0
        return v

    @validator('margin_to_limit_f', always=True)
    def calculate_margin(cls, v, values):
        """Calculate temperature margin to limit."""
        if v is None:
            temp = values.get('temperature_f')
            limit = values.get('design_temperature_f')
            if temp is not None and limit is not None:
                return limit - temp
        return v

    @validator('margin_pct', always=True)
    def calculate_margin_pct(cls, v, values):
        """Calculate margin percentage."""
        if v is None:
            margin = values.get('margin_to_limit_f')
            limit = values.get('design_temperature_f')
            if margin is not None and limit is not None and limit > 0:
                return (margin / limit) * 100.0
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TubeMetalTemperatureAnalysis(BaseModel):
    """
    Analysis of tube metal temperature monitoring data.

    Evaluates thermal stress and remaining life implications.
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Superheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Analysis timestamp"
    )

    # Individual readings
    readings: List[TubeMetalTemperatureReading] = Field(
        ...,
        description="Individual tube metal temperature readings"
    )

    # Summary statistics
    max_temperature_f: float = Field(
        ...,
        description="Maximum tube metal temperature (F)"
    )
    min_temperature_f: float = Field(
        ...,
        description="Minimum tube metal temperature (F)"
    )
    avg_temperature_f: float = Field(
        ...,
        description="Average tube metal temperature (F)"
    )
    temperature_spread_f: Optional[float] = Field(
        default=None,
        description="Temperature spread (max - min) (F)"
    )

    # Hottest tube identification
    hottest_tube_sensor_id: str = Field(
        ...,
        description="Sensor ID of hottest tube"
    )
    hottest_tube_location: str = Field(
        ...,
        description="Location of hottest tube"
    )

    # Design margin analysis
    minimum_margin_to_limit_f: float = Field(
        ...,
        description="Minimum margin to design limit (F)"
    )
    sensors_in_alarm: int = Field(
        default=0,
        ge=0,
        description="Number of sensors in alarm"
    )
    sensors_in_warning: int = Field(
        default=0,
        ge=0,
        description="Number of sensors in warning"
    )

    # Thermal stress assessment
    thermal_stress_level: ThermalStressLevel = Field(
        default=ThermalStressLevel.LOW,
        description="Overall thermal stress level"
    )
    circumferential_gradient_f: Optional[float] = Field(
        default=None,
        description="Maximum circumferential temperature gradient (F)"
    )
    axial_gradient_f_per_ft: Optional[float] = Field(
        default=None,
        description="Maximum axial temperature gradient (F/ft)"
    )

    # Life assessment (simplified)
    estimated_creep_life_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Estimated remaining creep life (%)"
    )
    larson_miller_parameter: Optional[float] = Field(
        default=None,
        description="Larson-Miller parameter for creep assessment"
    )

    # Overall status
    overall_status: SafetyStatus = Field(
        ...,
        description="Overall tube metal temperature status"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for tube protection"
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
        default="",
        description="SHA-256 analysis hash"
    )

    @validator('temperature_spread_f', always=True)
    def calculate_spread(cls, v, values):
        """Calculate temperature spread."""
        if v is None:
            max_t = values.get('max_temperature_f')
            min_t = values.get('min_temperature_f')
            if max_t is not None and min_t is not None:
                return max_t - min_t
        return v

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# PROCESS DEMAND MODELS
# =============================================================================

class ProcessDemandReading(BaseModel):
    """
    Downstream process steam demand requirements.

    Captures consumer requirements for superheated steam.
    """

    # Identification
    consumer_id: str = Field(
        ...,
        description="Process consumer identifier"
    )
    consumer_name: str = Field(
        ...,
        description="Process consumer name"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reading timestamp"
    )

    # Steam requirements
    required_pressure_psig: float = Field(
        ...,
        ge=0,
        description="Required steam pressure (psig)"
    )
    pressure_tolerance_psi: float = Field(
        default=5.0,
        ge=0,
        description="Acceptable pressure tolerance (psi)"
    )

    required_temperature_f: float = Field(
        ...,
        ge=200,
        le=1100,
        description="Required steam temperature (F)"
    )
    temperature_tolerance_f: float = Field(
        default=10.0,
        ge=0,
        description="Acceptable temperature tolerance (F)"
    )

    required_superheat_f: Optional[float] = Field(
        default=None,
        ge=0,
        description="Required minimum superheat (F)"
    )

    # Flow requirements
    required_flow_lb_hr: float = Field(
        ...,
        ge=0,
        description="Required steam flow (lb/hr)"
    )
    current_flow_lb_hr: float = Field(
        default=0.0,
        ge=0,
        description="Current steam flow to consumer (lb/hr)"
    )

    # Priority and criticality
    priority: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Consumer priority (1=highest, 5=lowest)"
    )
    is_critical: bool = Field(
        default=False,
        description="Critical process consumer flag"
    )

    # Current satisfaction status
    flow_satisfied: bool = Field(
        default=True,
        description="Flow requirement satisfied"
    )
    pressure_satisfied: bool = Field(
        default=True,
        description="Pressure requirement satisfied"
    )
    temperature_satisfied: bool = Field(
        default=True,
        description="Temperature requirement satisfied"
    )

    # Energy value
    enthalpy_requirement_btu_lb: Optional[float] = Field(
        default=None,
        description="Required steam enthalpy (BTU/lb)"
    )
    energy_demand_mmbtu_hr: Optional[float] = Field(
        default=None,
        description="Energy demand rate (MMBTU/hr)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProcessDemandSummary(BaseModel):
    """
    Summary of all downstream process demands.

    Aggregates multiple consumer requirements for control optimization.
    """

    # Identification
    summary_id: str = Field(
        default="DEMAND-SUMMARY",
        description="Summary identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary timestamp"
    )

    # Individual demands
    demands: List[ProcessDemandReading] = Field(
        ...,
        description="List of process demand readings"
    )

    # Aggregated requirements
    total_demand_lb_hr: float = Field(
        ...,
        ge=0,
        description="Total steam demand (lb/hr)"
    )
    total_demand_klb_hr: Optional[float] = Field(
        default=None,
        description="Total steam demand (klb/hr)"
    )
    max_required_temperature_f: float = Field(
        ...,
        description="Maximum required outlet temperature (F)"
    )
    min_required_temperature_f: float = Field(
        ...,
        description="Minimum required outlet temperature (F)"
    )
    controlling_consumer_id: str = Field(
        ...,
        description="Consumer driving temperature setpoint"
    )

    # Satisfaction status
    all_demands_satisfied: bool = Field(
        ...,
        description="All consumer demands satisfied"
    )
    unsatisfied_count: int = Field(
        default=0,
        ge=0,
        description="Number of unsatisfied consumers"
    )
    critical_unsatisfied: bool = Field(
        default=False,
        description="Any critical consumer unsatisfied"
    )

    # Energy summary
    total_energy_demand_mmbtu_hr: Optional[float] = Field(
        default=None,
        description="Total energy demand (MMBTU/hr)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Demand management recommendations"
    )

    @validator('total_demand_klb_hr', always=True)
    def calculate_klb(cls, v, values):
        """Calculate klb/hr from lb/hr."""
        if v is None and 'total_demand_lb_hr' in values:
            return values['total_demand_lb_hr'] / 1000.0
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# SAFETY INTERLOCK MODELS
# =============================================================================

class SafetyInterlockStatus(BaseModel):
    """
    Safety interlock status for superheater protection.

    Monitors all safety systems protecting the superheater.
    """

    # Identification
    interlock_id: str = Field(
        ...,
        description="Safety interlock identifier"
    )
    interlock_name: str = Field(
        ...,
        description="Safety interlock description"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Status
    status: SafetyStatus = Field(
        ...,
        description="Current interlock status"
    )
    is_healthy: bool = Field(
        ...,
        description="Interlock in healthy state"
    )
    is_bypassed: bool = Field(
        default=False,
        description="Interlock currently bypassed"
    )
    bypass_authorized_by: Optional[str] = Field(
        default=None,
        description="Authorization for bypass if active"
    )
    bypass_expires_at: Optional[datetime] = Field(
        default=None,
        description="Bypass expiration time"
    )

    # Measured value
    measured_value: Optional[float] = Field(
        default=None,
        description="Current measured value"
    )
    measured_units: Optional[str] = Field(
        default=None,
        description="Units of measurement"
    )

    # Setpoints
    alarm_setpoint: Optional[float] = Field(
        default=None,
        description="Alarm setpoint"
    )
    trip_setpoint: Optional[float] = Field(
        default=None,
        description="Trip setpoint"
    )
    margin_to_trip: Optional[float] = Field(
        default=None,
        description="Margin to trip setpoint"
    )

    # Action on trip
    trip_action: str = Field(
        ...,
        description="Action taken on trip condition"
    )
    affected_equipment: List[str] = Field(
        default_factory=list,
        description="Equipment affected by trip"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SafetySystemSummary(BaseModel):
    """
    Summary of all superheater safety systems.

    Comprehensive view of safety interlock status.
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Superheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Summary timestamp"
    )

    # Individual interlocks
    interlocks: List[SafetyInterlockStatus] = Field(
        ...,
        description="List of safety interlock statuses"
    )

    # Summary status
    overall_status: SafetyStatus = Field(
        ...,
        description="Overall safety system status"
    )
    all_interlocks_healthy: bool = Field(
        ...,
        description="All interlocks in healthy state"
    )
    active_trips: int = Field(
        default=0,
        ge=0,
        description="Number of active trip conditions"
    )
    active_alarms: int = Field(
        default=0,
        ge=0,
        description="Number of active alarm conditions"
    )
    active_warnings: int = Field(
        default=0,
        ge=0,
        description="Number of active warning conditions"
    )
    active_bypasses: int = Field(
        default=0,
        ge=0,
        description="Number of active bypasses"
    )

    # Specific interlock types
    high_temp_interlock_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="High temperature interlock status"
    )
    low_flow_interlock_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Low flow interlock status"
    )
    tube_metal_interlock_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Tube metal temperature interlock status"
    )
    spray_valve_interlock_status: SafetyStatus = Field(
        default=SafetyStatus.NORMAL,
        description="Spray valve interlock status"
    )

    # Permit to operate
    permit_to_operate: bool = Field(
        ...,
        description="System has permit to operate"
    )
    permit_blocking_reasons: List[str] = Field(
        default_factory=list,
        description="Reasons blocking permit if not granted"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Safety recommendations"
    )
    critical_actions: List[str] = Field(
        default_factory=list,
        description="Critical actions required"
    )

    # Provenance
    provenance_hash: str = Field(
        default="",
        description="SHA-256 analysis hash"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# EFFICIENCY METRICS MODELS
# =============================================================================

class SuperheaterEfficiencyMetrics(BaseModel):
    """
    Superheater efficiency and performance metrics.

    Captures heat transfer effectiveness and energy utilization.
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Superheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Calculation timestamp"
    )

    # Heat transfer performance
    heat_duty_mmbtu_hr: float = Field(
        ...,
        description="Actual heat duty (MMBTU/hr)"
    )
    design_heat_duty_mmbtu_hr: float = Field(
        ...,
        description="Design heat duty (MMBTU/hr)"
    )
    heat_duty_ratio_pct: Optional[float] = Field(
        default=None,
        description="Actual vs design heat duty (%)"
    )

    # Heat transfer coefficient
    overall_htc_btu_hr_ft2_f: float = Field(
        ...,
        ge=0,
        description="Overall heat transfer coefficient (BTU/hr-ft2-F)"
    )
    design_htc_btu_hr_ft2_f: float = Field(
        ...,
        description="Design heat transfer coefficient (BTU/hr-ft2-F)"
    )
    htc_ratio_pct: Optional[float] = Field(
        default=None,
        description="Actual vs design HTC (%)"
    )
    fouling_factor_hr_ft2_f_btu: Optional[float] = Field(
        default=None,
        description="Estimated fouling factor (hr-ft2-F/BTU)"
    )

    # Temperature effectiveness
    lmtd_f: float = Field(
        ...,
        description="Log Mean Temperature Difference (F)"
    )
    design_lmtd_f: float = Field(
        ...,
        description="Design LMTD (F)"
    )
    effectiveness_pct: Optional[float] = Field(
        default=None,
        description="Heat exchanger effectiveness (%)"
    )

    # Pressure drop performance
    steam_dp_psi: float = Field(
        ...,
        ge=0,
        description="Steam side pressure drop (psi)"
    )
    design_steam_dp_psi: float = Field(
        ...,
        description="Design steam side pressure drop (psi)"
    )
    dp_ratio_pct: Optional[float] = Field(
        default=None,
        description="Actual vs design pressure drop (%)"
    )

    # Spray impact on efficiency
    spray_efficiency_loss_pct: float = Field(
        default=0.0,
        ge=0,
        description="Efficiency loss from spray water (%)"
    )
    spray_water_usage_pct: float = Field(
        default=0.0,
        ge=0,
        description="Spray water as percent of steam flow"
    )

    # Overall efficiency
    thermal_efficiency_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Thermal efficiency (%)"
    )
    exergy_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Exergy (second law) efficiency (%)"
    )

    # Performance indicators
    performance_index: float = Field(
        default=100.0,
        description="Performance index (100 = design performance)"
    )
    degradation_indicator: str = Field(
        default="none",
        description="Degradation indicator (none, slight, moderate, severe)"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Efficiency improvement recommendations"
    )

    @validator('heat_duty_ratio_pct', always=True)
    def calculate_duty_ratio(cls, v, values):
        """Calculate heat duty ratio."""
        if v is None:
            actual = values.get('heat_duty_mmbtu_hr')
            design = values.get('design_heat_duty_mmbtu_hr')
            if actual is not None and design is not None and design > 0:
                return (actual / design) * 100.0
        return v

    @validator('htc_ratio_pct', always=True)
    def calculate_htc_ratio(cls, v, values):
        """Calculate HTC ratio."""
        if v is None:
            actual = values.get('overall_htc_btu_hr_ft2_f')
            design = values.get('design_htc_btu_hr_ft2_f')
            if actual is not None and design is not None and design > 0:
                return (actual / design) * 100.0
        return v

    @validator('dp_ratio_pct', always=True)
    def calculate_dp_ratio(cls, v, values):
        """Calculate pressure drop ratio."""
        if v is None:
            actual = values.get('steam_dp_psi')
            design = values.get('design_steam_dp_psi')
            if actual is not None and design is not None and design > 0:
                return (actual / design) * 100.0
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# ALARM AND EVENT MODELS
# =============================================================================

class SuperheaterAlarm(BaseModel):
    """
    Superheater alarm event record.

    Captures alarm conditions for monitoring and analysis.
    """

    # Identification
    alarm_id: str = Field(
        ...,
        description="Unique alarm identifier"
    )
    superheater_id: str = Field(
        ...,
        description="Superheater identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Alarm timestamp"
    )

    # Alarm details
    alarm_tag: str = Field(
        ...,
        description="Alarm tag name"
    )
    alarm_description: str = Field(
        ...,
        description="Alarm description"
    )
    severity: AlarmSeverity = Field(
        ...,
        description="Alarm severity"
    )

    # Values
    measured_value: float = Field(
        ...,
        description="Value that triggered alarm"
    )
    alarm_setpoint: float = Field(
        ...,
        description="Alarm setpoint"
    )
    units: str = Field(
        ...,
        description="Units of measurement"
    )

    # State
    is_active: bool = Field(
        default=True,
        description="Alarm currently active"
    )
    is_acknowledged: bool = Field(
        default=False,
        description="Alarm acknowledged"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="User who acknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Acknowledgment timestamp"
    )
    cleared_at: Optional[datetime] = Field(
        default=None,
        description="Alarm cleared timestamp"
    )

    # Recommended action
    recommended_action: str = Field(
        default="",
        description="Recommended operator action"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# COMPREHENSIVE CONTROL OUTPUT MODEL
# =============================================================================

class SuperheaterControlOutput(BaseModel):
    """
    Complete output from GL-022 Superheater Control Agent.

    Comprehensive control output with all analyses, recommendations,
    and provenance tracking for audit compliance.
    """

    # Identification
    agent_id: str = Field(
        default="GL-022-SUPERHEATER",
        description="Agent identifier"
    )
    superheater_id: str = Field(
        ...,
        description="Superheater equipment identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Output timestamp"
    )
    execution_id: str = Field(
        ...,
        description="Unique execution identifier for traceability"
    )

    # Current Operating Point
    operating_point: SuperheaterOperatingPoint = Field(
        ...,
        description="Current superheater operating point"
    )

    # Steam Properties
    inlet_steam_conditions: SteamConditions = Field(
        ...,
        description="Inlet steam thermodynamic conditions"
    )
    outlet_steam_conditions: SteamConditions = Field(
        ...,
        description="Outlet steam thermodynamic conditions"
    )

    # Temperature Control
    temperature_control: TemperatureControlOutput = Field(
        ...,
        description="Temperature control calculation results"
    )
    control_mode: ControlMode = Field(
        ...,
        description="Current control mode"
    )

    # Desuperheater Spray
    spray_output: Optional[DesuperheaterSprayOutput] = Field(
        default=None,
        description="Desuperheater spray calculation results"
    )
    spray_valve_status: SprayValveStatus = Field(
        default=SprayValveStatus.CLOSED,
        description="Current spray valve status"
    )

    # Tube Metal Temperature Analysis
    tube_metal_analysis: Optional[TubeMetalTemperatureAnalysis] = Field(
        default=None,
        description="Tube metal temperature analysis"
    )

    # Process Demand
    demand_summary: Optional[ProcessDemandSummary] = Field(
        default=None,
        description="Process demand summary"
    )

    # Safety System Status
    safety_summary: SafetySystemSummary = Field(
        ...,
        description="Safety system status summary"
    )

    # Efficiency Metrics
    efficiency_metrics: Optional[SuperheaterEfficiencyMetrics] = Field(
        default=None,
        description="Efficiency and performance metrics"
    )

    # Overall Status
    overall_status: ValidationStatus = Field(
        ...,
        description="Overall control status"
    )
    safety_status: SafetyStatus = Field(
        ...,
        description="Overall safety status"
    )
    temperature_deviation_f: float = Field(
        ...,
        description="Deviation from temperature setpoint (F)"
    )
    within_tolerance: bool = Field(
        ...,
        description="Temperature within specified tolerance"
    )

    # Control Actions
    primary_control_action: ControlAction = Field(
        ...,
        description="Primary control action being taken"
    )
    spray_valve_target_pct: float = Field(
        ...,
        ge=0,
        le=100,
        description="Target spray valve position (%)"
    )

    # Active Alarms
    active_alarms: List[SuperheaterAlarm] = Field(
        default_factory=list,
        description="Currently active alarms"
    )
    alarm_count_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of alarms by severity"
    )

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Operational recommendations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    alerts: List[str] = Field(
        default_factory=list,
        description="Critical alerts requiring immediate attention"
    )

    # Performance Summary
    thermal_efficiency_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Current thermal efficiency (%)"
    )
    spray_efficiency_loss_pct: float = Field(
        default=0.0,
        ge=0,
        description="Efficiency loss from spray water (%)"
    )
    tube_metal_margin_f: Optional[float] = Field(
        default=None,
        description="Minimum margin to tube metal limit (F)"
    )

    # Provenance and Audit Trail
    provenance_hash: str = Field(
        ...,
        description="SHA-256 hash for complete audit trail"
    )
    input_data_hash: str = Field(
        ...,
        description="SHA-256 hash of input data"
    )
    calculation_chain: List[str] = Field(
        default_factory=list,
        description="List of calculation step hashes"
    )

    # Processing Metadata
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total processing time (ms)"
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of calculations performed"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    formula_references: List[str] = Field(
        default_factory=list,
        description="Engineering formulas and standards used"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# INPUT AGGREGATION MODEL
# =============================================================================

class SuperheaterControlInput(BaseModel):
    """
    Complete input for GL-022 Superheater Control Agent.

    Aggregates all input data required for comprehensive superheater control.
    """

    # Identification
    superheater_id: str = Field(
        ...,
        description="Superheater equipment identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Input timestamp"
    )

    # Operating Point
    operating_point: SuperheaterOperatingPoint = Field(
        ...,
        description="Current superheater operating point"
    )

    # Temperature Control Setpoints
    temperature_control_input: TemperatureControlInput = Field(
        ...,
        description="Temperature control parameters and setpoints"
    )

    # Desuperheater Spray Data
    spray_input: Optional[DesuperheaterSprayInput] = Field(
        default=None,
        description="Desuperheater spray water data"
    )

    # Tube Metal Temperature Readings
    tube_metal_readings: List[TubeMetalTemperatureReading] = Field(
        default_factory=list,
        description="Tube metal temperature sensor readings"
    )

    # Process Demands
    process_demands: List[ProcessDemandReading] = Field(
        default_factory=list,
        description="Downstream process demand requirements"
    )

    # Safety Interlock Status
    safety_interlocks: List[SafetyInterlockStatus] = Field(
        default_factory=list,
        description="Safety interlock status readings"
    )

    # Design Parameters
    design_outlet_temperature_f: float = Field(
        ...,
        ge=400,
        le=1100,
        description="Design outlet steam temperature (F)"
    )
    design_steam_flow_lb_hr: float = Field(
        ...,
        gt=0,
        description="Design steam flow rate (lb/hr)"
    )
    design_inlet_pressure_psig: float = Field(
        ...,
        gt=0,
        description="Design inlet pressure (psig)"
    )

    # Tube Design Parameters
    tube_material: str = Field(
        default="SA-213 T22",
        description="Tube material specification"
    )
    tube_design_temperature_f: float = Field(
        default=1050.0,
        ge=800,
        le=1400,
        description="Tube design temperature limit (F)"
    )
    tube_design_pressure_psig: float = Field(
        default=1500.0,
        gt=0,
        description="Tube design pressure (psig)"
    )

    # Operating Constraints
    min_superheat_required_f: float = Field(
        default=50.0,
        ge=0,
        description="Minimum required superheat (F)"
    )
    max_spray_to_steam_ratio_pct: float = Field(
        default=15.0,
        ge=0,
        le=30,
        description="Maximum spray to steam ratio (%)"
    )

    # Economic Parameters
    fuel_cost_usd_mmbtu: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fuel cost ($/MMBTU)"
    )
    water_cost_usd_kgal: Optional[float] = Field(
        default=None,
        gt=0,
        description="Water cost ($/kgal)"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class SuperheaterControlConfig(BaseModel):
    """
    Configuration parameters for GL-022 Superheater Control Agent.

    Defines operational parameters, limits, and tuning values.
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-022-SUPERHEATER",
        description="Agent identifier"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version"
    )

    # Temperature control settings
    default_deadband_f: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Default temperature control deadband (F)"
    )
    default_max_rate_f_per_min: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Default maximum rate of change (F/min)"
    )

    # PID defaults
    default_kp: float = Field(
        default=2.0,
        ge=0,
        description="Default proportional gain"
    )
    default_ki: float = Field(
        default=0.1,
        ge=0,
        description="Default integral gain"
    )
    default_kd: float = Field(
        default=0.5,
        ge=0,
        description="Default derivative gain"
    )

    # Alarm setpoints
    high_temp_alarm_offset_f: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="High temperature alarm offset from setpoint (F)"
    )
    high_high_temp_alarm_offset_f: float = Field(
        default=40.0,
        ge=10,
        le=100,
        description="High-high temperature alarm offset (F)"
    )
    low_temp_alarm_offset_f: float = Field(
        default=30.0,
        ge=5,
        le=50,
        description="Low temperature alarm offset from setpoint (F)"
    )

    # Tube metal protection
    tube_metal_warning_margin_f: float = Field(
        default=75.0,
        ge=25,
        le=150,
        description="Warning margin below tube design temp (F)"
    )
    tube_metal_alarm_margin_f: float = Field(
        default=50.0,
        ge=10,
        le=100,
        description="Alarm margin below tube design temp (F)"
    )

    # Spray water limits
    min_spray_valve_modulation_pct: float = Field(
        default=10.0,
        ge=0,
        le=30,
        description="Minimum valve position for modulation (%)"
    )
    max_spray_rate_of_change_pct_per_sec: float = Field(
        default=5.0,
        ge=1,
        le=20,
        description="Maximum spray valve rate of change (%/sec)"
    )

    # Safety settings
    enable_tube_metal_protection: bool = Field(
        default=True,
        description="Enable tube metal temperature protection"
    )
    enable_low_flow_protection: bool = Field(
        default=True,
        description="Enable low steam flow protection"
    )
    min_steam_flow_for_spray_pct: float = Field(
        default=25.0,
        ge=10,
        le=50,
        description="Minimum steam flow to enable spray (%)"
    )

    # Calculation settings
    steam_table_standard: str = Field(
        default="IAPWS-IF97",
        description="Steam property calculation standard"
    )
    calculation_precision: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Decimal precision for calculations"
    )

    # Logging and audit
    enable_detailed_logging: bool = Field(
        default=True,
        description="Enable detailed calculation logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )

    class Config:
        use_enum_values = True

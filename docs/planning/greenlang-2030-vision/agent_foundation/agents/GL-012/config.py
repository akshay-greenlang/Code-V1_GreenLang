# -*- coding: utf-8 -*-
"""
Configuration module for SteamQualityController agent (GL-012 STEAMQUAL).

This module defines the configuration models and settings for the
SteamQualityController agent, including steam quality specifications,
desuperheater settings, control valve configurations, sensor configurations,
quality control parameters, and regulatory compliance settings.

The SteamQualityController maintains optimal steam quality by managing:
- Pressure control (bar/barg/psig)
- Temperature control (degrees Celsius)
- Moisture content / dryness fraction
- Superheat degree management
- Desuperheater injection control

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ASME PTC 6 - Steam Turbines
- IEC 61511 - Functional Safety for Process Industry
- ISA-75.01.01 - Control Valve Sizing
- ISO 9001 - Quality Management Systems
- Pydantic V2 for validation

Author: GreenLang Industrial Optimization Team
Date: December 2025
Agent ID: GL-012
Agent Name: SteamQualityController
Domain: Steam Systems
Type: Controller
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ControlMode(str, Enum):
    """Steam quality control operating modes."""
    AUTO = "auto"
    MANUAL = "manual"
    CASCADE = "cascade"
    RATIO = "ratio"
    FEEDFORWARD = "feedforward"
    EMERGENCY = "emergency"


class QualityPriority(str, Enum):
    """Priority for steam quality parameters."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    MOISTURE = "moisture"
    SUPERHEAT = "superheat"
    BALANCED = "balanced"


class ValveType(str, Enum):
    """Types of control valves for steam systems."""
    GLOBE = "globe"
    BUTTERFLY = "butterfly"
    BALL = "ball"
    GATE = "gate"
    PLUG = "plug"
    DIAPHRAGM = "diaphragm"
    ANGLE = "angle"
    THREE_WAY = "three_way"


class ActuatorType(str, Enum):
    """Types of valve actuators."""
    PNEUMATIC = "pneumatic"
    ELECTRIC = "electric"
    HYDRAULIC = "hydraulic"
    ELECTRO_PNEUMATIC = "electro_pneumatic"
    MANUAL = "manual"


class MeterType(str, Enum):
    """Types of steam quality measurement devices."""
    VORTEX = "vortex"
    ORIFICE = "orifice"
    ULTRASONIC = "ultrasonic"
    CORIOLIS = "coriolis"
    TURBINE = "turbine"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    THERMAL_MASS = "thermal_mass"


class SteamPhase(str, Enum):
    """Steam phase classification."""
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED = "superheated"
    SUPERCRITICAL = "supercritical"


class AlarmSeverity(str, Enum):
    """Alarm severity levels per IEC 62682."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ADVISORY = "advisory"


class ComplianceStandard(str, Enum):
    """Regulatory compliance standards for steam systems."""
    ASME_PTC_4_1 = "asme_ptc_4_1"
    ASME_PTC_6 = "asme_ptc_6"
    IEC_61511 = "iec_61511"
    ISO_9001 = "iso_9001"
    EPA_MACT = "epa_mact"
    EU_BREF = "eu_bref"
    CUSTOM = "custom"


class SteamPurityClass(str, Enum):
    """Steam purity classification per industry standards."""
    INDUSTRIAL = "industrial"
    PROCESS = "process"
    HIGH_PURITY = "high_purity"
    ULTRA_PURE = "ultra_pure"
    PHARMACEUTICAL = "pharmaceutical"
    FOOD_GRADE = "food_grade"


# ============================================================================
# PID CONTROLLER CONFIGURATION
# ============================================================================

class PIDGains(BaseModel):
    """
    PID controller tuning parameters.

    These gains control the response characteristics of the control loop.
    Proper tuning is critical for stable steam quality control.

    Attributes:
        kp: Proportional gain (dimensionless)
        ki: Integral gain (1/seconds)
        kd: Derivative gain (seconds)
        deadband_percent: Dead band to prevent hunting
        output_limits: Min/max output limits (percent)
        anti_windup_enabled: Enable integral anti-windup
        derivative_filter_coefficient: Filter coefficient for derivative action
    """

    kp: float = Field(
        default=1.0,
        ge=0.0,
        le=100.0,
        description="Proportional gain (dimensionless)"
    )
    ki: float = Field(
        default=0.1,
        ge=0.0,
        le=10.0,
        description="Integral gain (1/seconds)"
    )
    kd: float = Field(
        default=0.01,
        ge=0.0,
        le=10.0,
        description="Derivative gain (seconds)"
    )
    deadband_percent: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Dead band percentage to prevent hunting"
    )
    output_limits: Tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Output limits (min_percent, max_percent)"
    )
    anti_windup_enabled: bool = Field(
        default=True,
        description="Enable integral anti-windup protection"
    )
    derivative_filter_coefficient: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Low-pass filter coefficient for derivative term"
    )

    @field_validator('output_limits')
    @classmethod
    def validate_output_limits(cls, v):
        """Ensure output limits are valid."""
        if v[0] >= v[1]:
            raise ValueError("Output minimum must be less than maximum")
        if v[0] < 0 or v[1] > 100:
            raise ValueError("Output limits must be between 0 and 100 percent")
        return v


# ============================================================================
# STEAM QUALITY SPECIFICATION
# ============================================================================

class SteamQualitySpecification(BaseModel):
    """
    Steam quality specification parameters.

    Defines the target steam quality parameters including pressure,
    temperature, dryness fraction, and superheat requirements.

    Physical Relationships:
    - Dryness fraction (x): 0.0 = saturated liquid, 1.0 = saturated vapor
    - For superheated steam: x > 1.0 indicates superheat degree
    - Moisture content = (1 - x) * 100 for wet steam

    Attributes:
        target_pressure_bar: Target steam pressure in bar absolute
        target_temperature_c: Target steam temperature in Celsius
        target_dryness_fraction: Target dryness fraction (0.95-1.0 for saturated, >1.0 for superheated)
        superheat_degree_c: Superheat temperature above saturation
        max_moisture_content_percent: Maximum allowable moisture content
        acceptable_pressure_tolerance_bar: Acceptable pressure deviation
        acceptable_temperature_tolerance_c: Acceptable temperature deviation
    """

    spec_id: str = Field(
        default="STEAM-SPEC-001",
        min_length=1,
        max_length=50,
        description="Unique specification identifier"
    )
    spec_name: str = Field(
        default="Standard Process Steam",
        min_length=1,
        max_length=100,
        description="Human-readable specification name"
    )

    # Pressure specification
    target_pressure_bar: float = Field(
        default=10.0,
        ge=0.5,
        le=350.0,
        description="Target steam pressure in bar absolute"
    )
    min_pressure_bar: float = Field(
        default=8.0,
        ge=0.1,
        le=350.0,
        description="Minimum acceptable pressure in bar"
    )
    max_pressure_bar: float = Field(
        default=12.0,
        ge=0.5,
        le=400.0,
        description="Maximum acceptable pressure in bar"
    )
    acceptable_pressure_tolerance_bar: float = Field(
        default=0.5,
        ge=0.01,
        le=10.0,
        description="Acceptable pressure deviation from setpoint"
    )

    # Temperature specification
    target_temperature_c: float = Field(
        default=180.0,
        ge=100.0,
        le=650.0,
        description="Target steam temperature in Celsius"
    )
    min_temperature_c: float = Field(
        default=170.0,
        ge=50.0,
        le=650.0,
        description="Minimum acceptable temperature in Celsius"
    )
    max_temperature_c: float = Field(
        default=200.0,
        ge=100.0,
        le=700.0,
        description="Maximum acceptable temperature in Celsius"
    )
    acceptable_temperature_tolerance_c: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Acceptable temperature deviation from setpoint"
    )

    # Steam quality / dryness specification
    target_dryness_fraction: float = Field(
        default=0.98,
        ge=0.80,
        le=1.50,
        description="Target dryness fraction (0.95-1.0 saturated, >1.0 superheated indicator)"
    )
    min_dryness_fraction: float = Field(
        default=0.95,
        ge=0.70,
        le=1.0,
        description="Minimum acceptable dryness fraction"
    )
    max_moisture_content_percent: float = Field(
        default=5.0,
        ge=0.0,
        le=30.0,
        description="Maximum allowable moisture content percentage"
    )

    # Superheat specification
    superheat_degree_c: float = Field(
        default=0.0,
        ge=0.0,
        le=300.0,
        description="Superheat temperature above saturation point"
    )
    min_superheat_c: float = Field(
        default=0.0,
        ge=0.0,
        le=250.0,
        description="Minimum superheat for superheated steam"
    )
    max_superheat_c: float = Field(
        default=100.0,
        ge=0.0,
        le=350.0,
        description="Maximum superheat limit"
    )

    # Steam phase classification
    target_phase: SteamPhase = Field(
        default=SteamPhase.SATURATED_VAPOR,
        description="Target steam phase"
    )

    # Quality rating parameters
    steam_purity_class: SteamPurityClass = Field(
        default=SteamPurityClass.INDUSTRIAL,
        description="Steam purity classification"
    )
    max_tds_ppm: float = Field(
        default=50.0,
        ge=0.0,
        le=1000.0,
        description="Maximum total dissolved solids in ppm"
    )
    max_silica_ppm: float = Field(
        default=0.02,
        ge=0.0,
        le=10.0,
        description="Maximum silica content in ppm"
    )
    max_conductivity_us_cm: float = Field(
        default=100.0,
        ge=0.0,
        le=5000.0,
        description="Maximum conductivity in microSiemens/cm"
    )

    @field_validator('max_pressure_bar')
    @classmethod
    def validate_max_pressure(cls, v, info):
        """Ensure max pressure is greater than min pressure."""
        if 'min_pressure_bar' in info.data:
            if v <= info.data['min_pressure_bar']:
                raise ValueError("Maximum pressure must exceed minimum pressure")
        return v

    @field_validator('max_temperature_c')
    @classmethod
    def validate_max_temperature(cls, v, info):
        """Ensure max temperature is greater than min temperature."""
        if 'min_temperature_c' in info.data:
            if v <= info.data['min_temperature_c']:
                raise ValueError("Maximum temperature must exceed minimum temperature")
        return v

    @field_validator('target_pressure_bar')
    @classmethod
    def validate_target_pressure_in_range(cls, v, info):
        """Ensure target pressure is within min/max bounds."""
        if 'min_pressure_bar' in info.data and 'max_pressure_bar' in info.data:
            if v < info.data['min_pressure_bar'] or v > info.data['max_pressure_bar']:
                raise ValueError("Target pressure must be within min/max bounds")
        return v

    @model_validator(mode='after')
    def validate_steam_quality_consistency(self):
        """Validate steam quality parameters are physically consistent."""
        # Moisture content should be consistent with dryness fraction
        implied_moisture = (1 - self.min_dryness_fraction) * 100
        if implied_moisture > self.max_moisture_content_percent + 1.0:  # 1% tolerance
            raise ValueError(
                f"Inconsistent quality specs: min_dryness_fraction {self.min_dryness_fraction} "
                f"implies {implied_moisture:.1f}% moisture, but max_moisture_content is "
                f"{self.max_moisture_content_percent}%"
            )

        # Superheat validation for superheated steam
        if self.target_phase == SteamPhase.SUPERHEATED:
            if self.superheat_degree_c <= 0:
                raise ValueError(
                    "Superheated steam phase requires superheat_degree_c > 0"
                )

        return self


# ============================================================================
# DESUPERHEATER CONFIGURATION
# ============================================================================

class DesuperheaterConfig(BaseModel):
    """
    Desuperheater (attemperator) configuration.

    Desuperheaters reduce steam temperature by injecting water into the
    steam flow. This configuration controls injection parameters and
    control loop tuning.

    Control Strategy:
    - Cascade control with temperature as outer loop
    - Flow control as inner loop for injection rate
    - Feedforward from steam flow for improved response

    Attributes:
        injection_water_temperature_c: Temperature of injection water
        injection_water_pressure_bar: Pressure of injection water (must exceed steam pressure)
        max_injection_rate_kg_hr: Maximum water injection rate
        min_injection_rate_kg_hr: Minimum water injection rate
        control_valve_cv: Control valve flow coefficient
        response_time_seconds: System response time constant
        pid_gains: PID controller tuning parameters
    """

    desuperheater_id: str = Field(
        default="DSH-001",
        min_length=1,
        max_length=50,
        description="Unique desuperheater identifier"
    )
    desuperheater_name: str = Field(
        default="Main Desuperheater",
        min_length=1,
        max_length=100,
        description="Human-readable desuperheater name"
    )

    # Desuperheater type
    desuperheater_type: str = Field(
        default="spray_type",
        description="Desuperheater type (spray_type, venturi, ring_type, probe_type)"
    )

    # Injection water specifications
    injection_water_temperature_c: float = Field(
        default=40.0,
        ge=5.0,
        le=200.0,
        description="Temperature of injection water in Celsius"
    )
    injection_water_pressure_bar: float = Field(
        default=15.0,
        ge=1.0,
        le=500.0,
        description="Pressure of injection water in bar (must exceed steam pressure)"
    )
    min_water_pressure_differential_bar: float = Field(
        default=3.0,
        ge=0.5,
        le=50.0,
        description="Minimum pressure differential above steam pressure"
    )

    # Injection rate limits
    max_injection_rate_kg_hr: float = Field(
        default=5000.0,
        ge=10.0,
        le=500000.0,
        description="Maximum water injection rate in kg/hr"
    )
    min_injection_rate_kg_hr: float = Field(
        default=100.0,
        ge=0.0,
        le=10000.0,
        description="Minimum water injection rate in kg/hr"
    )
    turndown_ratio: float = Field(
        default=10.0,
        ge=2.0,
        le=100.0,
        description="Maximum to minimum flow ratio"
    )

    # Control valve specifications
    control_valve_cv: float = Field(
        default=50.0,
        ge=0.1,
        le=5000.0,
        description="Control valve flow coefficient (Cv)"
    )
    valve_characteristic: str = Field(
        default="equal_percentage",
        description="Valve characteristic (linear, equal_percentage, quick_opening)"
    )
    valve_rangeability: float = Field(
        default=50.0,
        ge=10.0,
        le=100.0,
        description="Valve rangeability ratio"
    )

    # Response characteristics
    response_time_seconds: float = Field(
        default=5.0,
        ge=0.1,
        le=60.0,
        description="System response time constant in seconds"
    )
    dead_time_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Transport delay / dead time in seconds"
    )
    mixing_length_meters: float = Field(
        default=10.0,
        ge=1.0,
        le=100.0,
        description="Distance for complete mixing downstream"
    )

    # PID controller tuning
    pid_gains: PIDGains = Field(
        default_factory=lambda: PIDGains(kp=2.0, ki=0.5, kd=0.1),
        description="PID controller tuning parameters"
    )

    # Cascade control settings
    cascade_enabled: bool = Field(
        default=True,
        description="Enable cascade control (temperature-to-flow)"
    )
    inner_loop_pid: Optional[PIDGains] = Field(
        default_factory=lambda: PIDGains(kp=1.5, ki=1.0, kd=0.05),
        description="Inner loop (flow) PID tuning"
    )

    # Feedforward settings
    feedforward_enabled: bool = Field(
        default=True,
        description="Enable feedforward from steam flow"
    )
    feedforward_gain: float = Field(
        default=0.8,
        ge=0.0,
        le=2.0,
        description="Feedforward gain for steam flow compensation"
    )
    feedforward_lead_seconds: float = Field(
        default=2.0,
        ge=0.0,
        le=30.0,
        description="Feedforward lead time constant"
    )

    # Safety interlocks
    high_temperature_trip_c: float = Field(
        default=250.0,
        ge=100.0,
        le=700.0,
        description="High temperature trip setpoint"
    )
    low_water_pressure_trip_bar: float = Field(
        default=5.0,
        ge=0.5,
        le=50.0,
        description="Low water pressure trip setpoint"
    )
    water_quality_monitoring: bool = Field(
        default=True,
        description="Enable injection water quality monitoring"
    )

    @field_validator('max_injection_rate_kg_hr')
    @classmethod
    def validate_injection_rate_range(cls, v, info):
        """Ensure max injection rate exceeds minimum."""
        if 'min_injection_rate_kg_hr' in info.data:
            if v <= info.data['min_injection_rate_kg_hr']:
                raise ValueError("Maximum injection rate must exceed minimum")
        return v

    @model_validator(mode='after')
    def validate_desuperheater_config(self):
        """Validate desuperheater configuration consistency."""
        # Validate turndown ratio matches rate limits
        if self.min_injection_rate_kg_hr > 0:
            actual_turndown = self.max_injection_rate_kg_hr / self.min_injection_rate_kg_hr
            if actual_turndown < self.turndown_ratio * 0.9:  # 10% tolerance
                raise ValueError(
                    f"Injection rate limits ({self.min_injection_rate_kg_hr} to "
                    f"{self.max_injection_rate_kg_hr}) do not support specified "
                    f"turndown ratio of {self.turndown_ratio}"
                )

        return self


# ============================================================================
# CONTROL VALVE CONFIGURATION
# ============================================================================

class ControlValveConfig(BaseModel):
    """
    Pressure control valve configuration per ISA-75.01.01.

    Defines valve characteristics, sizing parameters, and actuator
    specifications for steam pressure control valves.

    Sizing per ISA-75.01.01:
    - Cv = Q * sqrt(SG / dP) for liquids
    - Cg = Q * sqrt(T / (P1 * dP)) for gases/steam

    Attributes:
        valve_type: Type of control valve (globe, butterfly, etc.)
        cv_rating: Valve flow coefficient at full open
        max_pressure_drop_bar: Maximum allowable pressure drop
        response_time_seconds: Valve stroke time
        actuator_type: Type of actuator (pneumatic, electric)
    """

    valve_id: str = Field(
        default="CV-001",
        min_length=1,
        max_length=50,
        description="Unique valve identifier"
    )
    valve_tag: str = Field(
        default="PCV-001",
        min_length=1,
        max_length=50,
        description="P&ID valve tag"
    )
    valve_name: str = Field(
        default="Main Steam Pressure Control Valve",
        min_length=1,
        max_length=100,
        description="Human-readable valve name"
    )

    # Valve type and construction
    valve_type: ValveType = Field(
        default=ValveType.GLOBE,
        description="Type of control valve"
    )
    body_material: str = Field(
        default="A216-WCB",
        description="Valve body material specification"
    )
    trim_material: str = Field(
        default="SS316",
        description="Valve trim material"
    )
    seat_material: str = Field(
        default="Stellite",
        description="Valve seat material"
    )

    # Flow characteristics
    cv_rating: float = Field(
        default=100.0,
        ge=0.1,
        le=50000.0,
        description="Valve flow coefficient (Cv) at full open"
    )
    inherent_characteristic: str = Field(
        default="equal_percentage",
        description="Inherent flow characteristic (linear, equal_percentage, quick_opening)"
    )
    rangeability: float = Field(
        default=50.0,
        ge=10.0,
        le=200.0,
        description="Valve rangeability (max/min controllable Cv)"
    )

    # Pressure ratings
    pressure_class: str = Field(
        default="300",
        description="ANSI pressure class (150, 300, 600, 900, 1500, 2500)"
    )
    max_inlet_pressure_bar: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Maximum inlet pressure rating"
    )
    max_pressure_drop_bar: float = Field(
        default=20.0,
        ge=0.1,
        le=200.0,
        description="Maximum allowable pressure drop across valve"
    )
    max_differential_pressure_bar: float = Field(
        default=30.0,
        ge=0.1,
        le=250.0,
        description="Maximum differential pressure for shutoff"
    )

    # Temperature ratings
    min_temperature_c: float = Field(
        default=-29.0,
        ge=-200.0,
        le=100.0,
        description="Minimum operating temperature"
    )
    max_temperature_c: float = Field(
        default=425.0,
        ge=50.0,
        le=800.0,
        description="Maximum operating temperature"
    )

    # Dynamic response
    response_time_seconds: float = Field(
        default=3.0,
        ge=0.1,
        le=120.0,
        description="Full stroke time in seconds"
    )
    hysteresis_percent: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Valve hysteresis as percentage of span"
    )
    dead_band_percent: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Valve dead band as percentage"
    )

    # Actuator specifications
    actuator_type: ActuatorType = Field(
        default=ActuatorType.PNEUMATIC,
        description="Type of valve actuator"
    )
    actuator_size: str = Field(
        default="50",
        description="Actuator size designation"
    )
    supply_pressure_bar: float = Field(
        default=6.0,
        ge=1.0,
        le=10.0,
        description="Actuator supply pressure (pneumatic)"
    )
    fail_position: str = Field(
        default="closed",
        description="Fail-safe position (open, closed, last)"
    )

    # Positioner specifications
    positioner_enabled: bool = Field(
        default=True,
        description="Digital positioner installed"
    )
    positioner_type: str = Field(
        default="smart",
        description="Positioner type (smart, conventional, electro-pneumatic)"
    )
    positioner_feedback: str = Field(
        default="4-20mA",
        description="Position feedback signal type"
    )

    # Leakage class
    leakage_class: str = Field(
        default="IV",
        description="ANSI/FCI 70-2 leakage class (I through VI)"
    )

    # Noise considerations
    max_noise_dba: float = Field(
        default=85.0,
        ge=50.0,
        le=120.0,
        description="Maximum allowable noise level in dBA"
    )
    noise_attenuation_required: bool = Field(
        default=False,
        description="Noise attenuation trim required"
    )

    @field_validator('pressure_class')
    @classmethod
    def validate_pressure_class(cls, v):
        """Validate ANSI pressure class."""
        valid_classes = {"150", "300", "600", "900", "1500", "2500"}
        if v not in valid_classes:
            raise ValueError(f"Invalid pressure class. Must be one of: {valid_classes}")
        return v

    @field_validator('leakage_class')
    @classmethod
    def validate_leakage_class(cls, v):
        """Validate ANSI/FCI leakage class."""
        valid_classes = {"I", "II", "III", "IV", "V", "VI"}
        if v not in valid_classes:
            raise ValueError(f"Invalid leakage class. Must be one of: {valid_classes}")
        return v


# ============================================================================
# STEAM QUALITY METER CONFIGURATION
# ============================================================================

class SteamQualityMeterConfig(BaseModel):
    """
    Steam quality measurement sensor configuration.

    Defines sensor specifications for measuring steam quality parameters
    including flow, pressure, temperature, and moisture content.

    Accuracy classes per IEC 60751 (temperature) and ISO 5167 (flow):
    - Class A: +/- 0.15 + 0.002*|t| for RTDs
    - Class B: +/- 0.30 + 0.005*|t| for RTDs

    Attributes:
        meter_type: Type of measurement device
        accuracy_class: Measurement accuracy class
        sampling_interval_seconds: Data sampling rate
        calibration_interval_days: Required calibration frequency
    """

    meter_id: str = Field(
        default="SQM-001",
        min_length=1,
        max_length=50,
        description="Unique meter identifier"
    )
    meter_tag: str = Field(
        default="FT-001",
        min_length=1,
        max_length=50,
        description="P&ID instrument tag"
    )
    meter_name: str = Field(
        default="Main Steam Flow Meter",
        min_length=1,
        max_length=100,
        description="Human-readable meter name"
    )
    meter_type: MeterType = Field(
        default=MeterType.VORTEX,
        description="Type of measurement device"
    )

    # Measurement specifications
    measurement_parameter: str = Field(
        default="flow",
        description="Parameter being measured (flow, pressure, temperature, moisture)"
    )
    measurement_range_min: float = Field(
        default=0.0,
        description="Minimum measurement range"
    )
    measurement_range_max: float = Field(
        default=100.0,
        description="Maximum measurement range"
    )
    measurement_unit: str = Field(
        default="kg/hr",
        description="Engineering unit of measurement"
    )

    # Accuracy and precision
    accuracy_class: str = Field(
        default="Class_A",
        description="Measurement accuracy class (Class_A, Class_B, 0.5%, 1.0%)"
    )
    accuracy_percent: float = Field(
        default=0.5,
        ge=0.01,
        le=5.0,
        description="Measurement accuracy as percentage of span"
    )
    repeatability_percent: float = Field(
        default=0.1,
        ge=0.01,
        le=1.0,
        description="Measurement repeatability as percentage"
    )

    # Sampling configuration
    sampling_interval_seconds: float = Field(
        default=1.0,
        ge=0.01,
        le=60.0,
        description="Data sampling interval in seconds"
    )
    averaging_samples: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Number of samples for averaging"
    )
    filter_time_constant_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=30.0,
        description="Signal filtering time constant"
    )

    # Operating conditions
    process_connection: str = Field(
        default="Flanged",
        description="Process connection type (Flanged, Wafer, Threaded)"
    )
    max_process_pressure_bar: float = Field(
        default=100.0,
        ge=1.0,
        le=500.0,
        description="Maximum process pressure rating"
    )
    max_process_temperature_c: float = Field(
        default=400.0,
        ge=50.0,
        le=700.0,
        description="Maximum process temperature rating"
    )

    # Calibration requirements
    calibration_interval_days: int = Field(
        default=365,
        ge=30,
        le=1825,
        description="Required calibration frequency in days"
    )
    last_calibration_date: Optional[str] = Field(
        default=None,
        description="Date of last calibration (ISO 8601 format)"
    )
    calibration_certificate_number: Optional[str] = Field(
        default=None,
        description="Calibration certificate reference"
    )

    # Signal output
    output_signal: str = Field(
        default="4-20mA",
        description="Analog output signal type"
    )
    communication_protocol: str = Field(
        default="HART",
        description="Digital communication protocol (HART, Profibus, Foundation Fieldbus)"
    )

    # Diagnostics
    self_diagnostics_enabled: bool = Field(
        default=True,
        description="Enable built-in self-diagnostics"
    )
    diagnostic_coverage_percent: float = Field(
        default=90.0,
        ge=0.0,
        le=100.0,
        description="Diagnostic coverage percentage per IEC 61508"
    )

    @field_validator('accuracy_class')
    @classmethod
    def validate_accuracy_class(cls, v):
        """Validate accuracy class specification."""
        valid_classes = {"Class_A", "Class_B", "0.25%", "0.5%", "1.0%", "2.0%", "2.5%"}
        if v not in valid_classes:
            raise ValueError(f"Invalid accuracy class. Must be one of: {valid_classes}")
        return v


# ============================================================================
# QUALITY CONTROL SETTINGS
# ============================================================================

class SetpointRange(BaseModel):
    """Setpoint range specification for a control parameter."""

    parameter: str = Field(..., description="Control parameter name")
    min_setpoint: float = Field(..., description="Minimum allowed setpoint")
    max_setpoint: float = Field(..., description="Maximum allowed setpoint")
    default_setpoint: float = Field(..., description="Default setpoint value")
    rate_of_change_limit: float = Field(
        default=10.0,
        ge=0.0,
        description="Maximum rate of setpoint change per minute"
    )

    @model_validator(mode='after')
    def validate_setpoint_range(self):
        """Validate setpoint range consistency."""
        if self.min_setpoint >= self.max_setpoint:
            raise ValueError("Minimum setpoint must be less than maximum setpoint")
        if not (self.min_setpoint <= self.default_setpoint <= self.max_setpoint):
            raise ValueError("Default setpoint must be within min/max range")
        return self


class AlarmThreshold(BaseModel):
    """Alarm threshold specification per IEC 62682."""

    alarm_tag: str = Field(..., description="Alarm tag identifier")
    parameter: str = Field(..., description="Monitored parameter")
    severity: AlarmSeverity = Field(..., description="Alarm severity level")
    threshold_value: float = Field(..., description="Alarm threshold value")
    threshold_type: str = Field(
        default="high",
        description="Threshold type (high, low, high_high, low_low, deviation)"
    )
    deadband: float = Field(
        default=0.5,
        ge=0.0,
        description="Alarm deadband for return-to-normal"
    )
    delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=300.0,
        description="Alarm delay time before activation"
    )
    auto_acknowledge: bool = Field(
        default=False,
        description="Allow automatic acknowledgment"
    )

    @field_validator('threshold_type')
    @classmethod
    def validate_threshold_type(cls, v):
        """Validate threshold type."""
        valid_types = {"high", "low", "high_high", "low_low", "deviation", "rate_of_change"}
        if v not in valid_types:
            raise ValueError(f"Invalid threshold type. Must be one of: {valid_types}")
        return v


class QualityControlSettings(BaseModel):
    """
    Steam quality control parameters and settings.

    Defines operating modes, priorities, setpoint ranges, and alarm
    configurations for the steam quality control system.

    Control Hierarchy:
    1. Safety interlocks (highest priority)
    2. Quality priority parameter
    3. Secondary parameters
    4. Optimization objectives

    Attributes:
        control_mode: Operating mode (AUTO, MANUAL, CASCADE)
        quality_priority: Primary quality parameter to optimize
        setpoint_ranges: Allowed setpoint ranges for each parameter
        alarm_thresholds: Alarm configuration for monitoring
    """

    settings_id: str = Field(
        default="QCS-001",
        min_length=1,
        max_length=50,
        description="Unique settings identifier"
    )
    settings_name: str = Field(
        default="Standard Control Settings",
        min_length=1,
        max_length=100,
        description="Human-readable settings name"
    )

    # Operating mode
    control_mode: ControlMode = Field(
        default=ControlMode.AUTO,
        description="Control operating mode"
    )
    quality_priority: QualityPriority = Field(
        default=QualityPriority.PRESSURE,
        description="Primary quality parameter priority"
    )
    secondary_priority: QualityPriority = Field(
        default=QualityPriority.TEMPERATURE,
        description="Secondary quality parameter priority"
    )

    # Control loop settings
    scan_rate_ms: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Control loop scan rate in milliseconds"
    )
    control_action: str = Field(
        default="reverse",
        description="Control action (direct, reverse)"
    )

    # Setpoint ranges
    setpoint_ranges: List[SetpointRange] = Field(
        default_factory=lambda: [
            SetpointRange(
                parameter="pressure_bar",
                min_setpoint=1.0,
                max_setpoint=50.0,
                default_setpoint=10.0,
                rate_of_change_limit=5.0
            ),
            SetpointRange(
                parameter="temperature_c",
                min_setpoint=100.0,
                max_setpoint=300.0,
                default_setpoint=180.0,
                rate_of_change_limit=20.0
            ),
            SetpointRange(
                parameter="dryness_fraction",
                min_setpoint=0.90,
                max_setpoint=1.00,
                default_setpoint=0.98,
                rate_of_change_limit=0.05
            ),
        ],
        description="Setpoint ranges for control parameters"
    )

    # Alarm thresholds
    alarm_thresholds: List[AlarmThreshold] = Field(
        default_factory=lambda: [
            AlarmThreshold(
                alarm_tag="PAH-001",
                parameter="pressure_bar",
                severity=AlarmSeverity.HIGH,
                threshold_value=45.0,
                threshold_type="high"
            ),
            AlarmThreshold(
                alarm_tag="PAHH-001",
                parameter="pressure_bar",
                severity=AlarmSeverity.CRITICAL,
                threshold_value=48.0,
                threshold_type="high_high"
            ),
            AlarmThreshold(
                alarm_tag="PAL-001",
                parameter="pressure_bar",
                severity=AlarmSeverity.HIGH,
                threshold_value=5.0,
                threshold_type="low"
            ),
            AlarmThreshold(
                alarm_tag="TAH-001",
                parameter="temperature_c",
                severity=AlarmSeverity.HIGH,
                threshold_value=280.0,
                threshold_type="high"
            ),
            AlarmThreshold(
                alarm_tag="QAL-001",
                parameter="dryness_fraction",
                severity=AlarmSeverity.MEDIUM,
                threshold_value=0.92,
                threshold_type="low"
            ),
        ],
        description="Alarm threshold configurations"
    )

    # Mode transition settings
    bumpless_transfer_enabled: bool = Field(
        default=True,
        description="Enable bumpless transfer between control modes"
    )
    mode_change_delay_seconds: float = Field(
        default=5.0,
        ge=0.0,
        le=60.0,
        description="Delay before mode change takes effect"
    )

    # Output limiting
    output_rate_limit_percent_per_second: float = Field(
        default=10.0,
        ge=0.1,
        le=100.0,
        description="Maximum output change rate"
    )
    min_output_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=50.0,
        description="Minimum controller output"
    )
    max_output_percent: float = Field(
        default=100.0,
        ge=50.0,
        le=100.0,
        description="Maximum controller output"
    )

    # Optimization settings
    optimization_enabled: bool = Field(
        default=True,
        description="Enable quality optimization"
    )
    optimization_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=1440,
        description="Optimization calculation interval"
    )

    @field_validator('control_action')
    @classmethod
    def validate_control_action(cls, v):
        """Validate control action type."""
        if v not in {"direct", "reverse"}:
            raise ValueError("Control action must be 'direct' or 'reverse'")
        return v


# ============================================================================
# COMPLIANCE SETTINGS
# ============================================================================

class ComplianceSettings(BaseModel):
    """
    Regulatory compliance configuration for steam quality systems.

    Ensures operation complies with applicable standards including
    ASME PTC 4.1, steam purity requirements, and reporting obligations.

    ASME PTC 4.1 Requirements:
    - Steady-state operation verification
    - Instrumentation accuracy requirements
    - Data collection procedures
    - Uncertainty analysis methodology

    Attributes:
        asme_ptc_4_1_compliance: Enable ASME PTC 4.1 compliance mode
        steam_purity_standards: Steam purity standard requirements
        reporting_requirements: Compliance reporting configuration
    """

    compliance_id: str = Field(
        default="COMP-001",
        min_length=1,
        max_length=50,
        description="Unique compliance configuration identifier"
    )
    compliance_name: str = Field(
        default="Standard Compliance Profile",
        min_length=1,
        max_length=100,
        description="Human-readable compliance profile name"
    )

    # Standard compliance flags
    asme_ptc_4_1_compliance: bool = Field(
        default=True,
        description="Enable ASME PTC 4.1 compliance for steam generators"
    )
    asme_ptc_6_compliance: bool = Field(
        default=False,
        description="Enable ASME PTC 6 compliance for steam turbines"
    )
    iec_61511_compliance: bool = Field(
        default=True,
        description="Enable IEC 61511 functional safety compliance"
    )
    iso_9001_compliance: bool = Field(
        default=True,
        description="Enable ISO 9001 quality management compliance"
    )

    # Steam purity standards
    steam_purity_standards: Dict[str, Any] = Field(
        default_factory=lambda: {
            "total_dissolved_solids_ppm": 50.0,
            "silica_ppm": 0.02,
            "iron_ppm": 0.01,
            "copper_ppm": 0.003,
            "sodium_ppm": 0.003,
            "conductivity_us_cm": 100.0,
            "ph_range": (8.5, 9.5),
            "dissolved_oxygen_ppb": 7.0
        },
        description="Steam purity requirements by parameter"
    )

    # Applicable regulatory standards
    applicable_standards: List[ComplianceStandard] = Field(
        default_factory=lambda: [
            ComplianceStandard.ASME_PTC_4_1,
            ComplianceStandard.IEC_61511,
            ComplianceStandard.ISO_9001
        ],
        description="List of applicable regulatory standards"
    )

    # Reporting requirements
    reporting_requirements: Dict[str, Any] = Field(
        default_factory=lambda: {
            "daily_summary_enabled": True,
            "shift_report_enabled": True,
            "monthly_compliance_report": True,
            "annual_audit_report": True,
            "alarm_summary_frequency": "daily",
            "quality_trend_frequency": "hourly",
            "report_retention_years": 7
        },
        description="Compliance reporting configuration"
    )

    # Data retention
    data_retention_days: int = Field(
        default=2555,  # 7 years
        ge=365,
        le=3650,
        description="Data retention period in days"
    )
    audit_trail_enabled: bool = Field(
        default=True,
        description="Enable complete audit trail logging"
    )

    # Calibration requirements
    calibration_requirements: Dict[str, int] = Field(
        default_factory=lambda: {
            "pressure_transmitter_days": 365,
            "temperature_transmitter_days": 365,
            "flow_meter_days": 365,
            "control_valve_days": 730,
            "safety_valve_days": 365
        },
        description="Calibration frequency requirements by equipment type"
    )

    # Uncertainty analysis per ASME PTC 4.1
    uncertainty_analysis_enabled: bool = Field(
        default=True,
        description="Enable measurement uncertainty analysis"
    )
    max_acceptable_uncertainty_percent: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Maximum acceptable measurement uncertainty"
    )

    # Safety integrity level (per IEC 61511)
    required_sil: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Required Safety Integrity Level (1-4)"
    )
    proof_test_interval_months: int = Field(
        default=12,
        ge=1,
        le=60,
        description="Proof test interval for safety functions"
    )

    @field_validator('required_sil')
    @classmethod
    def validate_sil(cls, v):
        """Validate SIL level."""
        if v not in {1, 2, 3, 4}:
            raise ValueError("SIL must be 1, 2, 3, or 4")
        return v


# ============================================================================
# SAFETY CONSTRAINTS
# ============================================================================

class SafetyConstraints(BaseModel):
    """
    Safety constraints for steam quality control.

    Defines hard limits that must never be exceeded to ensure safe
    operation of the steam system.

    These constraints represent physical or safety limits that trigger
    immediate protective action (trips/interlocks) rather than alarms.
    """

    # Pressure constraints
    max_pressure_bar: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Maximum allowable working pressure"
    )
    min_pressure_bar: float = Field(
        default=0.5,
        ge=0.0,
        le=50.0,
        description="Minimum operating pressure"
    )
    pressure_relief_setpoint_bar: float = Field(
        default=55.0,
        ge=1.0,
        le=550.0,
        description="Pressure relief valve setpoint"
    )

    # Temperature constraints
    max_temperature_c: float = Field(
        default=450.0,
        ge=100.0,
        le=700.0,
        description="Maximum allowable temperature"
    )
    min_temperature_c: float = Field(
        default=100.0,
        ge=50.0,
        le=300.0,
        description="Minimum operating temperature"
    )
    max_temperature_rate_c_per_min: float = Field(
        default=10.0,
        ge=0.1,
        le=50.0,
        description="Maximum temperature change rate"
    )

    # Steam quality constraints
    min_dryness_fraction: float = Field(
        default=0.90,
        ge=0.70,
        le=1.0,
        description="Minimum dryness fraction for turbine protection"
    )
    max_moisture_content_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=30.0,
        description="Maximum moisture content"
    )

    # Flow constraints
    max_flow_rate_kg_hr: float = Field(
        default=100000.0,
        ge=100.0,
        le=5000000.0,
        description="Maximum steam flow rate"
    )
    min_flow_rate_kg_hr: float = Field(
        default=1000.0,
        ge=0.0,
        le=100000.0,
        description="Minimum stable flow rate"
    )

    # Interlock settings
    trip_response_time_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Maximum response time for safety trips"
    )
    permit_required_for_bypass: bool = Field(
        default=True,
        description="Require permit for safety interlock bypass"
    )

    @model_validator(mode='after')
    def validate_pressure_constraints(self):
        """Validate pressure constraint relationships."""
        if self.min_pressure_bar >= self.max_pressure_bar:
            raise ValueError("Minimum pressure must be less than maximum pressure")
        if self.pressure_relief_setpoint_bar <= self.max_pressure_bar:
            raise ValueError(
                "Pressure relief setpoint must exceed maximum working pressure"
            )
        return self

    @model_validator(mode='after')
    def validate_temperature_constraints(self):
        """Validate temperature constraint relationships."""
        if self.min_temperature_c >= self.max_temperature_c:
            raise ValueError("Minimum temperature must be less than maximum temperature")
        return self


# ============================================================================
# MAIN AGENT CONFIGURATION
# ============================================================================

class SteamQualityControllerConfig(BaseModel):
    """
    Main configuration for SteamQualityController agent (GL-012 STEAMQUAL).

    Comprehensive configuration including steam quality specifications,
    control equipment settings, sensor configurations, and compliance
    requirements for maintaining optimal steam quality.

    The SteamQualityController is responsible for:
    - Maintaining target steam pressure within tolerance
    - Controlling steam temperature via desuperheater
    - Ensuring minimum dryness fraction (moisture control)
    - Superheat degree management
    - Regulatory compliance monitoring

    SECURITY & COMPLIANCE:
    - Zero hardcoded credentials policy enforced
    - Deterministic mode required for regulatory compliance
    - TLS encryption mandatory for production
    - Provenance tracking required for audit trails
    - ASME PTC 4.1 / IEC 61511 compliance validation

    Attributes:
        agent_id: Unique agent identifier (GL-012)
        agent_name: Agent display name (SteamQualityController)
        version: Agent software version
        calculation_timeout_seconds: Maximum calculation time
        max_retries: Maximum retry attempts on failure
        enable_monitoring: Enable performance monitoring
        cache_ttl_seconds: Cache time-to-live
    """

    # ========================================================================
    # AGENT IDENTIFICATION
    # ========================================================================

    agent_id: str = Field(
        default="GL-012",
        min_length=1,
        max_length=20,
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="SteamQualityController",
        min_length=1,
        max_length=100,
        description="Agent display name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent software version"
    )
    environment: str = Field(
        default="development",
        description="Deployment environment (development, staging, production)"
    )
    domain: str = Field(
        default="Steam Systems",
        description="Agent domain classification"
    )
    agent_type: str = Field(
        default="Controller",
        description="Agent type classification"
    )

    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================

    calculation_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Maximum time allowed for calculations"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=30.0,
        description="Delay between retry attempts"
    )
    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )
    cache_ttl_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Cache time-to-live in seconds"
    )
    max_concurrent_calculations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent calculation threads"
    )

    # ========================================================================
    # SAFETY CONSTRAINTS
    # ========================================================================

    safety_constraints: SafetyConstraints = Field(
        default_factory=SafetyConstraints,
        description="Safety constraint configuration"
    )

    # ========================================================================
    # STEAM QUALITY SPECIFICATION
    # ========================================================================

    steam_quality_spec: SteamQualitySpecification = Field(
        default_factory=SteamQualitySpecification,
        description="Steam quality specification"
    )

    # ========================================================================
    # EQUIPMENT CONFIGURATIONS
    # ========================================================================

    desuperheater_config: DesuperheaterConfig = Field(
        default_factory=DesuperheaterConfig,
        description="Desuperheater configuration"
    )

    pressure_control_valves: List[ControlValveConfig] = Field(
        default_factory=lambda: [ControlValveConfig()],
        description="Pressure control valve configurations"
    )

    steam_quality_meters: List[SteamQualityMeterConfig] = Field(
        default_factory=lambda: [SteamQualityMeterConfig()],
        description="Steam quality meter configurations"
    )

    # ========================================================================
    # CONTROL SETTINGS
    # ========================================================================

    quality_control_settings: QualityControlSettings = Field(
        default_factory=QualityControlSettings,
        description="Quality control settings"
    )

    # ========================================================================
    # COMPLIANCE SETTINGS
    # ========================================================================

    compliance_settings: ComplianceSettings = Field(
        default_factory=ComplianceSettings,
        description="Regulatory compliance settings"
    )

    # ========================================================================
    # INTEGRATION SETTINGS
    # ========================================================================

    scada_integration_enabled: bool = Field(
        default=True,
        description="Enable SCADA system integration"
    )
    scada_polling_interval_seconds: int = Field(
        default=1,
        ge=1,
        le=60,
        description="SCADA polling interval"
    )
    historian_integration_enabled: bool = Field(
        default=True,
        description="Enable process historian integration"
    )
    historian_logging_interval_seconds: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Historian logging interval"
    )

    # ========================================================================
    # DETERMINISTIC SETTINGS (REGULATORY COMPLIANCE)
    # ========================================================================

    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic mode (required for compliance)"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=0.0,
        description="LLM temperature (must be 0.0 for determinism)"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )

    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================

    zero_secrets: bool = Field(
        default=True,
        description="Enforce zero hardcoded secrets policy"
    )
    tls_enabled: bool = Field(
        default=True,
        description="Enable TLS 1.3 for API connections"
    )
    enable_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    # ========================================================================
    # CALCULATION PARAMETERS
    # ========================================================================

    decimal_precision: int = Field(
        default=10,
        ge=6,
        le=20,
        description="Decimal precision for calculations"
    )

    # ========================================================================
    # OPERATIONAL SETTINGS
    # ========================================================================

    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode (must be False in production)"
    )

    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            'pressure_deviation': 0.05,
            'temperature_deviation': 0.05,
            'moisture_high': 0.08,
            'control_loop_error': 0.0,
            'sensor_failure': 0.0,
            'communication_failure': 0.0
        },
        description="Alert thresholds for operational monitoring"
    )

    # ========================================================================
    # SITE INFORMATION
    # ========================================================================

    site_id: str = Field(
        default="SITE-001",
        description="Site identifier"
    )
    plant_id: str = Field(
        default="PLANT-001",
        description="Plant identifier"
    )
    unit_id: str = Field(
        default="UNIT-001",
        description="Unit/equipment identifier"
    )

    # ========================================================================
    # VALIDATORS - COMPLIANCE ENFORCEMENT
    # ========================================================================

    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Ensure temperature is 0.0 for deterministic operation."""
        if v != 0.0:
            raise ValueError(
                f"COMPLIANCE VIOLATION: temperature must be 0.0 for deterministic "
                f"steam quality calculations. Got: {v}. This ensures reproducible "
                f"results for regulatory compliance per ASME PTC 4.1."
            )
        return v

    @field_validator('seed')
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is 42 for bit-perfect reproducibility."""
        if v != 42:
            raise ValueError(
                f"COMPLIANCE VIOLATION: seed must be 42 for deterministic calculations. "
                f"Got: {v}. This ensures bit-perfect reproducibility across runs."
            )
        return v

    @field_validator('deterministic_mode')
    @classmethod
    def validate_deterministic(cls, v):
        """Ensure deterministic mode is enabled."""
        if not v:
            raise ValueError(
                "COMPLIANCE VIOLATION: deterministic_mode must be True for regulatory "
                "compliance. All steam quality calculations must be reproducible for "
                "audit trails per ASME PTC 4.1 and IEC 61511."
            )
        return v

    @field_validator('zero_secrets')
    @classmethod
    def validate_zero_secrets(cls, v):
        """Ensure zero_secrets policy is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. API keys and "
                "credentials must never be in config.py. Use environment variables "
                "or secrets manager."
            )
        return v

    @field_validator('tls_enabled')
    @classmethod
    def validate_tls(cls, v):
        """Ensure TLS is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: tls_enabled must be True for production "
                "deployments. All API connections must use TLS 1.3 for data protection."
            )
        return v

    @field_validator('enable_provenance')
    @classmethod
    def validate_provenance(cls, v):
        """Ensure provenance tracking is enabled."""
        if not v:
            raise ValueError(
                "COMPLIANCE VIOLATION: enable_provenance must be True. SHA-256 "
                "audit trails are required for all control decisions per IEC 61511."
            )
        return v

    @field_validator('alert_thresholds')
    @classmethod
    def validate_thresholds(cls, v):
        """Validate required alert thresholds are configured."""
        required_alerts = {
            'pressure_deviation', 'temperature_deviation',
            'moisture_high', 'control_loop_error',
            'sensor_failure', 'communication_failure'
        }
        missing = required_alerts - set(v.keys())
        if missing:
            raise ValueError(
                f"COMPLIANCE VIOLATION: Missing required alert thresholds: {missing}. "
                f"These alerts are mandatory for safe steam quality control operations."
            )

        # Validate threshold ranges
        for alert_type, threshold in v.items():
            if threshold < 0 or threshold > 1:
                raise ValueError(
                    f"COMPLIANCE VIOLATION: Alert threshold '{alert_type}' must be "
                    f"between 0 and 1. Got: {threshold}"
                )

        return v

    @field_validator('decimal_precision')
    @classmethod
    def validate_precision(cls, v):
        """Validate decimal precision for calculations."""
        if v < 10:
            raise ValueError(
                "COMPLIANCE VIOLATION: decimal_precision must be >= 10 for steam "
                f"quality calculations. Got: {v}. Required for accurate control "
                "to 0.0000000001 precision."
            )
        return v

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Validate configuration consistency across environments."""
        if self.environment == 'production':
            # Production environment checks
            if not self.tls_enabled:
                raise ValueError(
                    "SECURITY VIOLATION: TLS required in production environment. "
                    "Set tls_enabled=True for production deployments."
                )

            if not self.deterministic_mode:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Deterministic mode required in production "
                    "environment. Set deterministic_mode=True for regulatory compliance."
                )

            if self.debug_mode:
                raise ValueError(
                    "SECURITY VIOLATION: debug_mode must be False in production "
                    "environment. Debug mode exposes sensitive operational data."
                )

            if not self.enable_provenance:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Provenance tracking required in production. "
                    "Set enable_provenance=True for audit trails."
                )

            if not self.enable_audit_logging:
                raise ValueError(
                    "COMPLIANCE VIOLATION: Audit logging required in production. "
                    "Set enable_audit_logging=True for compliance."
                )

        # Validate calculation timeout is reasonable
        if self.calculation_timeout_seconds > 60:
            raise ValueError(
                "PERFORMANCE VIOLATION: calculation_timeout_seconds should not exceed "
                f"60 seconds for control applications. Got: {self.calculation_timeout_seconds}. "
                "Long timeouts indicate inefficient calculations."
            )

        # Validate safety constraints are within spec
        if self.safety_constraints.max_pressure_bar < self.steam_quality_spec.max_pressure_bar:
            raise ValueError(
                "SAFETY VIOLATION: Safety constraint max_pressure_bar must be >= "
                "steam_quality_spec max_pressure_bar"
            )

        return self

    # ========================================================================
    # RUNTIME ASSERTION HELPERS
    # ========================================================================

    def assert_compliance_ready(self) -> None:
        """
        Assert configuration is ready for compliance/production use.

        Raises:
            AssertionError: If any compliance requirements are not met

        Example:
            >>> config = SteamQualityControllerConfig()
            >>> config.assert_compliance_ready()  # Raises if not compliant
        """
        assert self.deterministic_mode, "Deterministic mode required for compliance"
        assert self.temperature == 0.0, "Temperature must be 0.0 for determinism"
        assert self.seed == 42, "Seed must be 42 for reproducibility"
        assert self.zero_secrets, "Zero secrets policy must be enforced"
        assert self.enable_provenance, "Provenance tracking required"
        assert self.tls_enabled, "TLS encryption required"

        if self.environment == 'production':
            assert not self.debug_mode, "Debug mode not allowed in production"
            assert self.enable_audit_logging, "Audit logging required in production"

        # Validate alert thresholds configured
        required_alerts = {
            'pressure_deviation', 'temperature_deviation',
            'moisture_high', 'control_loop_error',
            'sensor_failure', 'communication_failure'
        }
        assert required_alerts.issubset(set(self.alert_thresholds.keys())), \
            f"Missing required alerts: {required_alerts - set(self.alert_thresholds.keys())}"

        # Validate compliance settings
        assert self.compliance_settings.asme_ptc_4_1_compliance or \
               self.compliance_settings.iec_61511_compliance, \
               "At least one compliance standard must be enabled"

    def assert_security_ready(self) -> None:
        """
        Assert configuration meets security requirements.

        Raises:
            AssertionError: If any security requirements are not met
        """
        assert self.zero_secrets, "Zero secrets policy must be enforced"
        assert self.tls_enabled, "TLS encryption must be enabled"

    def assert_determinism_ready(self) -> None:
        """
        Assert configuration ensures deterministic calculations.

        Raises:
            AssertionError: If any determinism requirements are not met
        """
        assert self.deterministic_mode, "Deterministic mode must be enabled"
        assert self.temperature == 0.0, "Temperature must be 0.0"
        assert self.seed == 42, "Seed must be 42"
        assert self.decimal_precision >= 10, "Decimal precision must be >= 10"

    def calculate_config_hash(self) -> str:
        """
        Calculate SHA-256 hash of configuration for provenance tracking.

        Returns:
            str: SHA-256 hash of configuration JSON
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode()).hexdigest()


# ============================================================================
# DEFAULT CONFIGURATION FACTORY
# ============================================================================

def create_default_config() -> SteamQualityControllerConfig:
    """
    Create default configuration for testing and demonstration.

    Returns:
        SteamQualityControllerConfig with default values for standard
        industrial steam system.
    """
    # Create steam quality specification for 10 bar saturated steam
    steam_spec = SteamQualitySpecification(
        spec_id="STEAM-SPEC-MAIN",
        spec_name="Main Process Steam",
        target_pressure_bar=10.0,
        min_pressure_bar=8.0,
        max_pressure_bar=12.0,
        acceptable_pressure_tolerance_bar=0.5,
        target_temperature_c=180.0,
        min_temperature_c=170.0,
        max_temperature_c=200.0,
        acceptable_temperature_tolerance_c=5.0,
        target_dryness_fraction=0.98,
        min_dryness_fraction=0.95,
        max_moisture_content_percent=5.0,
        superheat_degree_c=0.0,
        target_phase=SteamPhase.SATURATED_VAPOR,
        steam_purity_class=SteamPurityClass.INDUSTRIAL
    )

    # Create desuperheater configuration
    desuperheater = DesuperheaterConfig(
        desuperheater_id="DSH-MAIN",
        desuperheater_name="Main Steam Desuperheater",
        desuperheater_type="spray_type",
        injection_water_temperature_c=40.0,
        injection_water_pressure_bar=15.0,
        max_injection_rate_kg_hr=5000.0,
        min_injection_rate_kg_hr=100.0,
        control_valve_cv=50.0,
        response_time_seconds=5.0,
        pid_gains=PIDGains(kp=2.0, ki=0.5, kd=0.1)
    )

    # Create pressure control valve configuration
    pressure_valve = ControlValveConfig(
        valve_id="PCV-MAIN",
        valve_tag="PCV-001",
        valve_name="Main Steam Pressure Control Valve",
        valve_type=ValveType.GLOBE,
        cv_rating=150.0,
        pressure_class="300",
        max_pressure_drop_bar=20.0,
        response_time_seconds=3.0,
        actuator_type=ActuatorType.PNEUMATIC,
        fail_position="closed"
    )

    # Create steam quality meter configuration
    flow_meter = SteamQualityMeterConfig(
        meter_id="FT-MAIN",
        meter_tag="FT-001",
        meter_name="Main Steam Flow Meter",
        meter_type=MeterType.VORTEX,
        measurement_parameter="flow",
        measurement_range_min=0.0,
        measurement_range_max=50000.0,
        measurement_unit="kg/hr",
        accuracy_class="Class_A",
        accuracy_percent=0.5,
        sampling_interval_seconds=1.0,
        calibration_interval_days=365
    )

    pressure_transmitter = SteamQualityMeterConfig(
        meter_id="PT-MAIN",
        meter_tag="PT-001",
        meter_name="Main Steam Pressure Transmitter",
        meter_type=MeterType.DIFFERENTIAL_PRESSURE,
        measurement_parameter="pressure",
        measurement_range_min=0.0,
        measurement_range_max=20.0,
        measurement_unit="bar",
        accuracy_class="0.5%",
        accuracy_percent=0.5,
        sampling_interval_seconds=0.5,
        calibration_interval_days=365
    )

    temperature_transmitter = SteamQualityMeterConfig(
        meter_id="TT-MAIN",
        meter_tag="TT-001",
        meter_name="Main Steam Temperature Transmitter",
        meter_type=MeterType.THERMAL_MASS,
        measurement_parameter="temperature",
        measurement_range_min=0.0,
        measurement_range_max=300.0,
        measurement_unit="C",
        accuracy_class="Class_A",
        accuracy_percent=0.25,
        sampling_interval_seconds=0.5,
        calibration_interval_days=365
    )

    # Create safety constraints
    safety = SafetyConstraints(
        max_pressure_bar=50.0,
        min_pressure_bar=0.5,
        pressure_relief_setpoint_bar=55.0,
        max_temperature_c=450.0,
        min_temperature_c=100.0,
        min_dryness_fraction=0.90,
        max_moisture_content_percent=10.0
    )

    # Create compliance settings
    compliance = ComplianceSettings(
        compliance_id="COMP-MAIN",
        compliance_name="Standard ASME/IEC Compliance",
        asme_ptc_4_1_compliance=True,
        iec_61511_compliance=True,
        iso_9001_compliance=True,
        required_sil=2
    )

    return SteamQualityControllerConfig(
        agent_id="GL-012",
        agent_name="SteamQualityController",
        version="1.0.0",
        environment="development",
        steam_quality_spec=steam_spec,
        desuperheater_config=desuperheater,
        pressure_control_valves=[pressure_valve],
        steam_quality_meters=[flow_meter, pressure_transmitter, temperature_transmitter],
        safety_constraints=safety,
        compliance_settings=compliance,
        site_id="DEMO-SITE-001",
        plant_id="DEMO-PLANT-001",
        unit_id="STEAM-UNIT-001"
    )


def create_superheated_steam_config() -> SteamQualityControllerConfig:
    """
    Create configuration for superheated steam application.

    Returns:
        SteamQualityControllerConfig configured for superheated steam
        with desuperheater control active.
    """
    config = create_default_config()

    # Update for superheated steam
    config.steam_quality_spec.target_phase = SteamPhase.SUPERHEATED
    config.steam_quality_spec.target_pressure_bar = 40.0
    config.steam_quality_spec.min_pressure_bar = 35.0
    config.steam_quality_spec.max_pressure_bar = 45.0
    config.steam_quality_spec.target_temperature_c = 400.0
    config.steam_quality_spec.min_temperature_c = 380.0
    config.steam_quality_spec.max_temperature_c = 420.0
    config.steam_quality_spec.superheat_degree_c = 150.0
    config.steam_quality_spec.min_superheat_c = 100.0
    config.steam_quality_spec.max_superheat_c = 200.0
    config.steam_quality_spec.target_dryness_fraction = 1.0

    # Update desuperheater for higher duty
    config.desuperheater_config.injection_water_pressure_bar = 50.0
    config.desuperheater_config.max_injection_rate_kg_hr = 10000.0
    config.desuperheater_config.high_temperature_trip_c = 450.0

    return config


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Enumerations
    "ControlMode",
    "QualityPriority",
    "ValveType",
    "ActuatorType",
    "MeterType",
    "SteamPhase",
    "AlarmSeverity",
    "ComplianceStandard",
    "SteamPurityClass",
    # Configuration models
    "PIDGains",
    "SteamQualitySpecification",
    "DesuperheaterConfig",
    "ControlValveConfig",
    "SteamQualityMeterConfig",
    "SetpointRange",
    "AlarmThreshold",
    "QualityControlSettings",
    "ComplianceSettings",
    "SafetyConstraints",
    "SteamQualityControllerConfig",
    # Factory functions
    "create_default_config",
    "create_superheated_steam_config",
]

# -*- coding: utf-8 -*-
"""
GL-008 TRAPCATCHER - Configuration Module

This module defines all configuration schemas for the Steam Trap Monitoring
Agent, including trap type configurations, sensor settings, diagnostic
thresholds, and economic parameters.

Configuration follows GreenLang patterns with Pydantic validation and
sensible defaults based on DOE Best Practices and Spirax Sarco guidelines.

Standards:
    - DOE Steam System Best Practices
    - Spirax Sarco Application Guide
    - ASME B16.34 Valve Ratings
    - ISO 6552 Automatic Steam Traps

Example:
    >>> from greenlang.agents.process_heat.gl_008_steam_trap_monitor.config import (
    ...     SteamTrapMonitorConfig,
    ... )
    >>> config = SteamTrapMonitorConfig(
    ...     plant_id="PLANT-001",
    ...     steam_pressure_psig=150.0,
    ...     steam_cost_per_mlb=12.50,
    ... )
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class TrapType(str, Enum):
    """Steam trap types per Spirax Sarco classification."""
    FLOAT_THERMOSTATIC = "float_thermostatic"       # F&T traps
    INVERTED_BUCKET = "inverted_bucket"             # Inverted bucket
    THERMOSTATIC = "thermostatic"                    # Balanced pressure thermostatic
    THERMODYNAMIC = "thermodynamic"                  # Disc traps
    BIMETALLIC = "bimetallic"                        # Bimetallic thermostatic
    LIQUID_EXPANSION = "liquid_expansion"            # Liquid expansion thermostatic
    ORIFICE = "orifice"                              # Fixed orifice (venturi)
    FLOAT = "float"                                  # Float-only (no thermostatic)


class TrapApplication(str, Enum):
    """Steam trap application categories per DOE."""
    DRIP_LEG = "drip_leg"                  # Steam main drip legs
    PROCESS = "process"                     # Process equipment
    TRACER = "tracer"                       # Steam tracing
    UNIT_HEATER = "unit_heater"            # Unit heaters
    HEAT_EXCHANGER = "heat_exchanger"      # Heat exchangers
    COIL = "coil"                          # Heating coils
    JACKETED_VESSEL = "jacketed_vessel"    # Jacketed vessels
    REBOILER = "reboiler"                  # Reboilers
    AUTOCLAVE = "autoclave"                # Autoclaves
    STERILIZER = "sterilizer"              # Sterilizers


class FailureMode(str, Enum):
    """Steam trap failure modes."""
    GOOD = "good"                          # Operating correctly
    FAILED_OPEN = "failed_open"            # Blowing through (live steam loss)
    FAILED_CLOSED = "failed_closed"        # Blocked (condensate backup)
    LEAKING = "leaking"                    # Partial steam loss
    COLD = "cold"                          # No flow
    FLOODED = "flooded"                    # Condensate backup


class DiagnosticMethod(str, Enum):
    """Diagnostic methods for trap assessment."""
    ULTRASONIC = "ultrasonic"              # Ultrasonic testing
    TEMPERATURE = "temperature"             # Temperature differential
    VISUAL = "visual"                       # Visual inspection
    INFRARED = "infrared"                   # Infrared thermography
    CONDUCTIVITY = "conductivity"           # Conductivity sensing
    COMBINED = "combined"                   # Multiple methods


class SensorType(str, Enum):
    """Sensor types for trap monitoring."""
    ULTRASONIC_HANDHELD = "ultrasonic_handheld"
    ULTRASONIC_WIRELESS = "ultrasonic_wireless"
    TEMPERATURE_CONTACT = "temperature_contact"
    TEMPERATURE_IR = "temperature_ir"
    TEMPERATURE_WIRELESS = "temperature_wireless"
    CONDUCTIVITY = "conductivity"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"      # Immediate action required
    WARNING = "warning"        # Attention needed
    INFO = "info"              # Informational
    NORMAL = "normal"          # Normal operation


# =============================================================================
# THRESHOLD CONFIGURATIONS
# =============================================================================

class UltrasonicThresholds(BaseModel):
    """Ultrasonic diagnostic thresholds per Spirax Sarco."""

    # dB levels at 38 kHz
    good_max_db: float = Field(
        default=70.0,
        gt=0,
        description="Maximum dB for good trap (varies by trap type)"
    )
    leaking_min_db: float = Field(
        default=75.0,
        gt=0,
        description="Minimum dB indicating leaking"
    )
    failed_open_db: float = Field(
        default=85.0,
        gt=0,
        description="dB level indicating failed open (continuous flow)"
    )
    cold_max_db: float = Field(
        default=40.0,
        gt=0,
        description="Maximum dB indicating cold/no flow trap"
    )

    # Cycling parameters
    min_cycle_period_s: float = Field(
        default=2.0,
        gt=0,
        description="Minimum expected cycle period for normal operation"
    )
    max_cycle_period_s: float = Field(
        default=120.0,
        gt=0,
        description="Maximum expected cycle period"
    )
    continuous_flow_duration_s: float = Field(
        default=30.0,
        gt=0,
        description="Duration indicating continuous flow (failed open)"
    )

    # Trap type specific adjustments
    thermodynamic_good_max_db: float = Field(
        default=80.0,
        gt=0,
        description="TD traps cycle loudly - higher threshold"
    )
    inverted_bucket_cycling_expected: bool = Field(
        default=True,
        description="Inverted bucket should cycle audibly"
    )


class TemperatureThresholds(BaseModel):
    """Temperature differential diagnostic thresholds."""

    # Delta T thresholds
    good_delta_t_min_f: float = Field(
        default=15.0,
        ge=0,
        description="Minimum delta T for good trap (subcooling)"
    )
    good_delta_t_max_f: float = Field(
        default=50.0,
        gt=0,
        description="Maximum delta T for good trap"
    )
    failed_open_delta_t_max_f: float = Field(
        default=10.0,
        ge=0,
        description="Very low delta T indicates failed open"
    )
    failed_closed_delta_t_min_f: float = Field(
        default=100.0,
        gt=0,
        description="Very high delta T indicates failed closed/cold"
    )

    # Outlet temperature relative to saturation
    outlet_max_above_sat_f: float = Field(
        default=5.0,
        ge=0,
        description="Maximum outlet temp above saturation (superheated = failed open)"
    )
    outlet_max_below_sat_f: float = Field(
        default=50.0,
        ge=0,
        description="Maximum outlet temp below saturation (too cold = failed closed)"
    )

    # Absolute limits
    min_inlet_temp_f: float = Field(
        default=200.0,
        gt=0,
        description="Minimum expected inlet temperature"
    )
    ambient_delta_threshold_f: float = Field(
        default=20.0,
        gt=0,
        description="Minimum delta from ambient for active trap"
    )


class DiagnosticThresholds(BaseModel):
    """Combined diagnostic thresholds."""

    ultrasonic: UltrasonicThresholds = Field(
        default_factory=UltrasonicThresholds,
        description="Ultrasonic thresholds"
    )
    temperature: TemperatureThresholds = Field(
        default_factory=TemperatureThresholds,
        description="Temperature thresholds"
    )

    # Confidence thresholds
    high_confidence_threshold: float = Field(
        default=0.90,
        ge=0.5,
        le=1.0,
        description="Threshold for high confidence diagnosis"
    )
    medium_confidence_threshold: float = Field(
        default=0.70,
        ge=0.3,
        le=1.0,
        description="Threshold for medium confidence diagnosis"
    )

    # Multi-method agreement
    require_multi_method_agreement: bool = Field(
        default=True,
        description="Require multiple methods to agree for high confidence"
    )
    disagreement_confidence_penalty: float = Field(
        default=0.20,
        ge=0,
        le=0.5,
        description="Confidence reduction when methods disagree"
    )


# =============================================================================
# TRAP TYPE CONFIGURATIONS
# =============================================================================

class TrapTypeConfig(BaseModel):
    """Configuration for a specific trap type."""

    trap_type: TrapType = Field(
        ...,
        description="Trap type"
    )
    description: str = Field(
        default="",
        description="Trap type description"
    )

    # Operating characteristics
    max_pressure_psig: float = Field(
        default=600.0,
        gt=0,
        description="Maximum operating pressure"
    )
    max_temperature_f: float = Field(
        default=800.0,
        gt=0,
        description="Maximum operating temperature"
    )
    max_capacity_lb_hr: float = Field(
        default=50000.0,
        gt=0,
        description="Maximum condensate capacity"
    )

    # Operating behavior
    subcooling_f: float = Field(
        default=0.0,
        ge=0,
        description="Typical subcooling (0 for immediate discharge)"
    )
    air_venting_capability: str = Field(
        default="good",
        description="Air venting: excellent, good, fair, poor"
    )
    cycling_type: str = Field(
        default="continuous",
        description="Operation: continuous, intermittent, thermostatic"
    )

    # Durability
    typical_service_life_years: float = Field(
        default=10.0,
        gt=0,
        description="Typical service life"
    )
    maintenance_interval_months: int = Field(
        default=12,
        gt=0,
        description="Recommended maintenance interval"
    )

    # Failure characteristics
    predominant_failure_mode: FailureMode = Field(
        default=FailureMode.FAILED_OPEN,
        description="Most common failure mode"
    )
    waterhammer_susceptible: bool = Field(
        default=False,
        description="Susceptible to waterhammer damage"
    )
    dirt_tolerant: bool = Field(
        default=True,
        description="Tolerant to dirt/debris"
    )

    # Diagnostic adjustments
    ultrasonic_adjustment_db: float = Field(
        default=0.0,
        description="dB adjustment for ultrasonic diagnosis"
    )
    temperature_adjustment_f: float = Field(
        default=0.0,
        description="Temperature adjustment for diagnosis"
    )

    class Config:
        use_enum_values = True


class TrapTypeDefaults:
    """Default configurations for standard trap types."""

    FLOAT_THERMOSTATIC = TrapTypeConfig(
        trap_type=TrapType.FLOAT_THERMOSTATIC,
        description="Float & Thermostatic (F&T) - Continuous discharge with air venting",
        max_pressure_psig=465,
        max_capacity_lb_hr=30000,
        subcooling_f=0,
        air_venting_capability="excellent",
        cycling_type="continuous",
        typical_service_life_years=7,
        predominant_failure_mode=FailureMode.FAILED_OPEN,
        waterhammer_susceptible=True,
        dirt_tolerant=False,
    )

    INVERTED_BUCKET = TrapTypeConfig(
        trap_type=TrapType.INVERTED_BUCKET,
        description="Inverted Bucket - Robust, intermittent discharge",
        max_pressure_psig=600,
        max_capacity_lb_hr=50000,
        subcooling_f=3,
        air_venting_capability="good",
        cycling_type="intermittent",
        typical_service_life_years=15,
        predominant_failure_mode=FailureMode.FAILED_OPEN,
        waterhammer_susceptible=False,
        dirt_tolerant=True,
        ultrasonic_adjustment_db=5.0,  # Cycles audibly
    )

    THERMOSTATIC = TrapTypeConfig(
        trap_type=TrapType.THERMOSTATIC,
        description="Balanced Pressure Thermostatic - Subcooled discharge",
        max_pressure_psig=465,
        max_capacity_lb_hr=5000,
        subcooling_f=20,
        air_venting_capability="excellent",
        cycling_type="thermostatic",
        typical_service_life_years=5,
        predominant_failure_mode=FailureMode.FAILED_OPEN,
        waterhammer_susceptible=False,
        dirt_tolerant=True,
        temperature_adjustment_f=-15.0,  # Operates with subcooling
    )

    THERMODYNAMIC = TrapTypeConfig(
        trap_type=TrapType.THERMODYNAMIC,
        description="Thermodynamic Disc - Compact, cycles with pressure differential",
        max_pressure_psig=600,
        max_capacity_lb_hr=2500,
        subcooling_f=5,
        air_venting_capability="fair",
        cycling_type="intermittent",
        typical_service_life_years=5,
        predominant_failure_mode=FailureMode.LEAKING,
        waterhammer_susceptible=False,
        dirt_tolerant=True,
        ultrasonic_adjustment_db=10.0,  # Cycles loudly
    )

    BIMETALLIC = TrapTypeConfig(
        trap_type=TrapType.BIMETALLIC,
        description="Bimetallic Thermostatic - High subcooling, robust",
        max_pressure_psig=600,
        max_capacity_lb_hr=10000,
        subcooling_f=40,
        air_venting_capability="good",
        cycling_type="thermostatic",
        typical_service_life_years=10,
        predominant_failure_mode=FailureMode.FAILED_CLOSED,
        waterhammer_susceptible=False,
        dirt_tolerant=True,
        temperature_adjustment_f=-30.0,  # High subcooling
    )


# =============================================================================
# SENSOR CONFIGURATIONS
# =============================================================================

class SensorConfig(BaseModel):
    """Individual sensor configuration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_type: SensorType = Field(
        ...,
        description="Sensor type"
    )
    location: Optional[str] = Field(
        default=None,
        description="Installation location"
    )

    # Calibration
    calibration_date: Optional[datetime] = Field(
        default=None,
        description="Last calibration date"
    )
    calibration_due_date: Optional[datetime] = Field(
        default=None,
        description="Next calibration due"
    )

    # For ultrasonic sensors
    frequency_khz: float = Field(
        default=38.0,
        gt=0,
        le=100,
        description="Ultrasonic frequency"
    )
    sensitivity_db: float = Field(
        default=0.0,
        description="Sensitivity adjustment"
    )

    # For wireless sensors
    battery_level_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Battery level"
    )
    signal_strength_dbm: Optional[float] = Field(
        default=None,
        description="Wireless signal strength"
    )
    reporting_interval_s: int = Field(
        default=300,
        gt=0,
        description="Data reporting interval"
    )

    # Thresholds for this sensor
    alarm_high: Optional[float] = Field(
        default=None,
        description="High alarm threshold"
    )
    alarm_low: Optional[float] = Field(
        default=None,
        description="Low alarm threshold"
    )

    class Config:
        use_enum_values = True


class WirelessSensorConfig(BaseModel):
    """Wireless sensor network configuration."""

    network_id: str = Field(
        default="WSN-001",
        description="Wireless network identifier"
    )
    gateway_ip: Optional[str] = Field(
        default=None,
        description="Gateway IP address"
    )
    protocol: str = Field(
        default="LoRaWAN",
        description="Wireless protocol: LoRaWAN, ZigBee, WiFi, WirelessHART"
    )

    # Network parameters
    max_sensors: int = Field(
        default=1000,
        gt=0,
        description="Maximum sensors per gateway"
    )
    default_reporting_interval_s: int = Field(
        default=300,
        gt=0,
        description="Default reporting interval"
    )
    high_priority_interval_s: int = Field(
        default=60,
        gt=0,
        description="Reporting interval for problem traps"
    )

    # Data management
    data_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Data retention period"
    )
    aggregation_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Data aggregation interval"
    )

    # Alerts
    battery_low_threshold_pct: float = Field(
        default=20.0,
        ge=5,
        le=50,
        description="Low battery alert threshold"
    )
    signal_low_threshold_dbm: float = Field(
        default=-100.0,
        description="Low signal alert threshold"
    )
    offline_alert_minutes: int = Field(
        default=30,
        gt=0,
        description="Minutes before offline alert"
    )


# =============================================================================
# SURVEY CONFIGURATIONS
# =============================================================================

class SurveyConfig(BaseModel):
    """Trap survey program configuration."""

    survey_interval_months: int = Field(
        default=12,
        ge=1,
        le=36,
        description="Survey interval (DOE recommends annual)"
    )
    high_priority_interval_months: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Survey interval for high-priority traps"
    )

    # Route optimization
    max_traps_per_route: int = Field(
        default=50,
        gt=0,
        le=200,
        description="Maximum traps per survey route"
    )
    average_time_per_trap_minutes: float = Field(
        default=5.0,
        gt=0,
        le=30,
        description="Average time per trap assessment"
    )
    route_optimization_algorithm: str = Field(
        default="nearest_neighbor",
        description="TSP algorithm: nearest_neighbor, 2opt, genetic"
    )

    # Personnel
    technicians_available: int = Field(
        default=2,
        ge=1,
        description="Available survey technicians"
    )
    max_hours_per_day: float = Field(
        default=8.0,
        gt=0,
        le=12,
        description="Maximum survey hours per day"
    )

    # Documentation
    photo_documentation_required: bool = Field(
        default=True,
        description="Require photo documentation"
    )
    gps_tagging_enabled: bool = Field(
        default=True,
        description="Enable GPS location tagging"
    )

    # Quality
    random_verification_pct: float = Field(
        default=10.0,
        ge=0,
        le=100,
        description="Random verification percentage"
    )
    supervisor_review_required: bool = Field(
        default=True,
        description="Require supervisor review of failures"
    )


# =============================================================================
# ECONOMICS CONFIGURATIONS
# =============================================================================

class EconomicsConfig(BaseModel):
    """Economic calculation configuration."""

    # Steam costs
    steam_cost_per_mlb: float = Field(
        default=12.50,
        gt=0,
        description="Steam cost ($/1000 lb)"
    )
    steam_cost_per_mmbtu: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam cost ($/MMBTU) - alternative"
    )
    fuel_type: str = Field(
        default="natural_gas",
        description="Primary fuel type for boiler"
    )
    boiler_efficiency_pct: float = Field(
        default=80.0,
        gt=0,
        le=100,
        description="Boiler efficiency"
    )

    # Operating parameters
    operating_hours_per_year: int = Field(
        default=8760,
        gt=0,
        le=8760,
        description="Annual operating hours"
    )
    steam_pressure_psig: float = Field(
        default=150.0,
        gt=0,
        description="Operating steam pressure"
    )
    steam_enthalpy_btu_lb: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam enthalpy (calculated if not provided)"
    )

    # Repair costs
    average_repair_cost_usd: float = Field(
        default=350.0,
        ge=0,
        description="Average trap repair cost"
    )
    average_replacement_cost_usd: float = Field(
        default=750.0,
        ge=0,
        description="Average trap replacement cost"
    )
    labor_rate_per_hour_usd: float = Field(
        default=75.0,
        ge=0,
        description="Labor rate per hour"
    )
    average_repair_hours: float = Field(
        default=2.0,
        ge=0,
        description="Average repair time"
    )

    # Environmental
    co2_factor_lb_per_mmbtu: float = Field(
        default=117.0,
        gt=0,
        description="CO2 emission factor (lb/MMBTU)"
    )
    carbon_cost_per_ton: float = Field(
        default=50.0,
        ge=0,
        description="Carbon cost ($/ton CO2)"
    )

    # Financial parameters
    discount_rate_pct: float = Field(
        default=10.0,
        ge=0,
        le=50,
        description="Discount rate for NPV calculations"
    )
    analysis_period_years: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Analysis period for NPV"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class SteamTrapMonitorConfig(BaseModel):
    """
    Main configuration for GL-008 Steam Trap Monitoring Agent.

    This configuration encompasses all aspects of steam trap monitoring
    including diagnostics, surveys, economics, and sensor networks.

    Attributes:
        plant_id: Plant/facility identifier
        steam_pressure_psig: Operating steam pressure
        trap_types: Configured trap types
        diagnostics: Diagnostic threshold configuration
        economics: Economic calculation configuration
        survey: Survey program configuration

    Example:
        >>> config = SteamTrapMonitorConfig(
        ...     plant_id="REFINERY-01",
        ...     steam_pressure_psig=150.0,
        ...     economics=EconomicsConfig(steam_cost_per_mlb=15.00),
        ... )
    """

    # Plant identification
    plant_id: str = Field(..., description="Plant/facility identifier")
    plant_name: str = Field(default="", description="Plant name")
    site_location: Optional[str] = Field(
        default=None,
        description="Site location"
    )

    # Operating conditions
    steam_pressure_psig: float = Field(
        default=150.0,
        gt=0,
        description="Operating steam pressure"
    )
    steam_temperature_f: Optional[float] = Field(
        default=None,
        gt=0,
        description="Steam temperature if superheated"
    )
    back_pressure_psig: float = Field(
        default=0.0,
        ge=0,
        description="Common back pressure"
    )

    # Trap type configurations
    trap_types: Dict[str, TrapTypeConfig] = Field(
        default_factory=lambda: {
            "float_thermostatic": TrapTypeDefaults.FLOAT_THERMOSTATIC,
            "inverted_bucket": TrapTypeDefaults.INVERTED_BUCKET,
            "thermostatic": TrapTypeDefaults.THERMOSTATIC,
            "thermodynamic": TrapTypeDefaults.THERMODYNAMIC,
            "bimetallic": TrapTypeDefaults.BIMETALLIC,
        },
        description="Trap type configurations"
    )

    # Sub-configurations
    diagnostics: DiagnosticThresholds = Field(
        default_factory=DiagnosticThresholds,
        description="Diagnostic thresholds"
    )
    economics: EconomicsConfig = Field(
        default_factory=EconomicsConfig,
        description="Economic parameters"
    )
    survey: SurveyConfig = Field(
        default_factory=SurveyConfig,
        description="Survey configuration"
    )
    wireless: WirelessSensorConfig = Field(
        default_factory=WirelessSensorConfig,
        description="Wireless sensor network configuration"
    )

    # Sensors
    sensors: List[SensorConfig] = Field(
        default_factory=list,
        description="Configured sensors"
    )

    # Safety and compliance
    safety_factor_startup: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="DOE safety factor for startup loads"
    )
    safety_factor_operating: float = Field(
        default=2.0,
        ge=1.0,
        le=5.0,
        description="DOE safety factor for operating loads"
    )
    asme_b16_34_compliance: bool = Field(
        default=True,
        description="Require ASME B16.34 compliance checks"
    )

    # Audit and provenance
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance"
    )

    # Data retention
    data_retention_days: int = Field(
        default=365 * 3,
        ge=30,
        le=3650,
        description="Data retention period"
    )

    # Alerting
    alert_on_failed_open: bool = Field(
        default=True,
        description="Alert on failed open detection"
    )
    alert_on_failed_closed: bool = Field(
        default=True,
        description="Alert on failed closed detection"
    )
    alert_steam_loss_threshold_lb_hr: float = Field(
        default=50.0,
        gt=0,
        description="Steam loss threshold for alerting"
    )

    class Config:
        use_enum_values = True

    @validator("steam_temperature_f", always=True)
    def calculate_steam_temp(cls, v, values):
        """Calculate saturation temperature if not superheated."""
        if v is None and "steam_pressure_psig" in values:
            # Approximate saturation temperature
            p = values["steam_pressure_psig"]
            # Simplified steam table correlation
            if p <= 0:
                return 212.0
            elif p <= 15:
                return 212.0 + (p / 15) * (250 - 212)
            elif p <= 50:
                return 250.0 + ((p - 15) / 35) * (298 - 250)
            elif p <= 100:
                return 298.0 + ((p - 50) / 50) * (338 - 298)
            elif p <= 150:
                return 338.0 + ((p - 100) / 50) * (366 - 338)
            elif p <= 200:
                return 366.0 + ((p - 150) / 50) * (388 - 366)
            elif p <= 250:
                return 388.0 + ((p - 200) / 50) * (406 - 388)
            elif p <= 300:
                return 406.0 + ((p - 250) / 50) * (422 - 406)
            else:
                return 422.0 + ((p - 300) / 100) * (448 - 422)
        return v

    def get_trap_type_config(self, trap_type: str) -> Optional[TrapTypeConfig]:
        """Get configuration for a specific trap type."""
        return self.trap_types.get(trap_type.lower())

    def get_saturation_temperature(self, pressure_psig: float) -> float:
        """Get saturation temperature for a given pressure."""
        # Steam table correlation
        if pressure_psig <= 0:
            return 212.0
        elif pressure_psig <= 15:
            return 212.0 + (pressure_psig / 15) * (250 - 212)
        elif pressure_psig <= 50:
            return 250.0 + ((pressure_psig - 15) / 35) * (298 - 250)
        elif pressure_psig <= 100:
            return 298.0 + ((pressure_psig - 50) / 50) * (338 - 298)
        elif pressure_psig <= 150:
            return 338.0 + ((pressure_psig - 100) / 50) * (366 - 338)
        elif pressure_psig <= 200:
            return 366.0 + ((pressure_psig - 150) / 50) * (388 - 366)
        elif pressure_psig <= 250:
            return 388.0 + ((pressure_psig - 200) / 50) * (406 - 388)
        elif pressure_psig <= 300:
            return 406.0 + ((pressure_psig - 250) / 50) * (422 - 406)
        elif pressure_psig <= 400:
            return 422.0 + ((pressure_psig - 300) / 100) * (448 - 422)
        elif pressure_psig <= 500:
            return 448.0 + ((pressure_psig - 400) / 100) * (470 - 448)
        else:
            return 470.0 + ((pressure_psig - 500) / 100) * (489 - 470)

    def get_latent_heat(self, pressure_psig: float) -> float:
        """Get latent heat of vaporization (BTU/lb) at given pressure."""
        # Simplified correlation
        sat_temp = self.get_saturation_temperature(pressure_psig)
        # Latent heat decreases with temperature
        if sat_temp <= 212:
            return 970.3
        elif sat_temp <= 300:
            return 970.3 - (sat_temp - 212) * 0.7
        elif sat_temp <= 400:
            return 908.3 - (sat_temp - 300) * 0.8
        elif sat_temp <= 500:
            return 828.3 - (sat_temp - 400) * 1.0
        else:
            return 728.3 - (sat_temp - 500) * 1.2

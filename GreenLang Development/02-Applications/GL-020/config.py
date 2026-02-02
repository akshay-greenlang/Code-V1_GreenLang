# -*- coding: utf-8 -*-
"""
GL-020 ECONOPULSE Configuration Models.

This module provides comprehensive Pydantic configuration models for the
Economizer Performance Agent, including economizer physical specifications,
sensor configurations, alert thresholds, soot blower settings, and baseline
performance parameters.

All models include validators to ensure operational constraints and
industry best practices for economizer heat recovery monitoring.

Author: GreenLang Team
Date: December 2025
Status: Production Ready
"""

from datetime import datetime, time
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================


class EconomizerType(str, Enum):
    """Economizer type classification based on design."""

    BARE_TUBE = "bare_tube"
    FINNED_TUBE = "finned_tube"
    EXTENDED_SURFACE = "extended_surface"
    CONDENSING = "condensing"
    NON_CONDENSING = "non_condensing"
    PARALLEL_FLOW = "parallel_flow"
    COUNTER_FLOW = "counter_flow"
    CROSS_FLOW = "cross_flow"


class FoulingType(str, Enum):
    """Types of fouling that can affect economizer performance."""

    SOOT = "soot"
    ASH = "ash"
    SCALE = "scale"
    CORROSION = "corrosion"
    BIOLOGICAL = "biological"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels for escalation."""

    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CleaningMethod(str, Enum):
    """Economizer cleaning methods."""

    SOOT_BLOWING = "soot_blowing"
    ACOUSTIC_CLEANING = "acoustic_cleaning"
    WATER_WASHING = "water_washing"
    CHEMICAL_CLEANING = "chemical_cleaning"
    MECHANICAL_CLEANING = "mechanical_cleaning"
    AIR_LANCING = "air_lancing"
    STEAM_LANCING = "steam_lancing"


class SootBlowerMediaType(str, Enum):
    """Soot blower media types."""

    STEAM = "steam"
    COMPRESSED_AIR = "compressed_air"
    WATER = "water"
    ACOUSTIC = "acoustic"


class TubeMaterial(str, Enum):
    """Economizer tube materials."""

    CARBON_STEEL = "carbon_steel"
    STAINLESS_STEEL = "stainless_steel"
    ALLOY_STEEL = "alloy_steel"
    CAST_IRON = "cast_iron"
    CORTEN_STEEL = "corten_steel"
    INCONEL = "inconel"


class SensorType(str, Enum):
    """Sensor types for economizer monitoring."""

    TEMPERATURE_RTD = "temperature_rtd"
    TEMPERATURE_THERMOCOUPLE = "temperature_thermocouple"
    PRESSURE_GAUGE = "pressure_gauge"
    PRESSURE_TRANSDUCER = "pressure_transducer"
    FLOW_ORIFICE = "flow_orifice"
    FLOW_ULTRASONIC = "flow_ultrasonic"
    FLOW_VORTEX = "flow_vortex"
    DIFFERENTIAL_PRESSURE = "differential_pressure"


class PerformanceStatus(str, Enum):
    """Economizer performance status."""

    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    CRITICAL = "critical"
    OFFLINE = "offline"


# ============================================================================
# SENSOR CONFIGURATION
# ============================================================================


class SensorConfiguration(BaseModel):
    """
    Sensor configuration for economizer monitoring.

    Defines sensor specifications and calibration parameters.
    """

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_type: SensorType = Field(..., description="Type of sensor")
    sensor_name: Optional[str] = Field(None, description="Sensor display name")
    location: str = Field(..., description="Physical location (inlet, outlet, etc.)")

    # Measurement parameters
    measurement_range_min: float = Field(
        default=0.0, description="Minimum measurement range"
    )
    measurement_range_max: float = Field(
        default=1000.0, description="Maximum measurement range"
    )
    measurement_units: str = Field(default="", description="Measurement units")
    accuracy_pct: float = Field(
        default=1.0, ge=0, le=10, description="Sensor accuracy (%)"
    )

    # Calibration
    calibration_date: Optional[datetime] = Field(
        None, description="Last calibration date"
    )
    calibration_due_date: Optional[datetime] = Field(
        None, description="Next calibration due date"
    )
    calibration_offset: float = Field(
        default=0.0, description="Calibration offset correction"
    )
    calibration_gain: float = Field(
        default=1.0, gt=0, description="Calibration gain correction"
    )

    # SCADA integration
    scada_tag: str = Field(default="", description="SCADA tag name")
    polling_interval_seconds: int = Field(
        default=10, ge=1, le=300, description="Data polling interval (seconds)"
    )

    # Data quality
    enable_quality_check: bool = Field(
        default=True, description="Enable data quality checks"
    )
    max_rate_of_change_per_minute: Optional[float] = Field(
        None, gt=0, description="Maximum valid rate of change per minute"
    )
    stale_data_timeout_seconds: int = Field(
        default=60, ge=10, description="Stale data timeout (seconds)"
    )

    @field_validator("measurement_range_max")
    @classmethod
    def validate_range(cls, v: float, info) -> float:
        """Validate measurement range is valid."""
        if "measurement_range_min" in info.data:
            if v <= info.data["measurement_range_min"]:
                raise ValueError("Max range must be greater than min range")
        return v

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "sensor_id": "TEMP-001",
                "sensor_type": "temperature_rtd",
                "sensor_name": "Economizer Inlet Water Temperature",
                "location": "water_inlet",
                "measurement_range_min": 32.0,
                "measurement_range_max": 400.0,
                "measurement_units": "degF",
                "accuracy_pct": 0.5,
                "scada_tag": "ECON.WATER_IN_TEMP",
            }
        }


# ============================================================================
# ECONOMIZER PHYSICAL CONFIGURATION
# ============================================================================


class EconomizerConfiguration(BaseModel):
    """
    Economizer physical specifications and design parameters.

    Defines the economizer's physical characteristics that influence
    heat transfer calculations and performance analysis.
    """

    economizer_id: str = Field(..., description="Unique economizer identifier")
    economizer_type: EconomizerType = Field(
        ..., description="Economizer type classification"
    )
    economizer_name: Optional[str] = Field(None, description="Economizer display name")
    manufacturer: Optional[str] = Field(None, description="Manufacturer name")
    model_number: Optional[str] = Field(None, description="Model number")

    # Physical dimensions
    tube_count: int = Field(
        ..., gt=0, description="Total number of tubes"
    )
    tube_rows: int = Field(
        default=1, gt=0, description="Number of tube rows"
    )
    tube_passes: int = Field(
        default=1, gt=0, le=10, description="Number of tube passes"
    )
    tube_outer_diameter_inches: float = Field(
        default=2.0, gt=0, le=6, description="Tube outer diameter (inches)"
    )
    tube_wall_thickness_inches: float = Field(
        default=0.125, gt=0, le=0.5, description="Tube wall thickness (inches)"
    )
    tube_length_ft: float = Field(
        default=20.0, gt=0, le=100, description="Tube length (feet)"
    )
    tube_material: TubeMaterial = Field(
        default=TubeMaterial.CARBON_STEEL, description="Tube material"
    )

    # Surface area
    total_heat_transfer_area_sqft: float = Field(
        ..., gt=0, description="Total heat transfer surface area (sq ft)"
    )
    fin_area_sqft: Optional[float] = Field(
        None, ge=0, description="Fin surface area for finned tubes (sq ft)"
    )
    fin_density_per_inch: Optional[float] = Field(
        None, ge=0, le=12, description="Fin density (fins per inch)"
    )

    # Flow configuration
    flow_arrangement: EconomizerType = Field(
        default=EconomizerType.COUNTER_FLOW, description="Flow arrangement"
    )
    water_flow_path: str = Field(
        default="horizontal", description="Water flow path orientation"
    )
    gas_flow_direction: str = Field(
        default="vertical_down", description="Flue gas flow direction"
    )

    # Operating design parameters
    design_water_flow_gpm: float = Field(
        ..., gt=0, description="Design water flow rate (GPM)"
    )
    design_water_inlet_temp_f: float = Field(
        default=200.0, gt=32, le=400, description="Design water inlet temperature (F)"
    )
    design_water_outlet_temp_f: float = Field(
        default=280.0, gt=32, le=450, description="Design water outlet temperature (F)"
    )
    design_gas_inlet_temp_f: float = Field(
        default=550.0, gt=200, le=1200, description="Design flue gas inlet temperature (F)"
    )
    design_gas_outlet_temp_f: float = Field(
        default=350.0, gt=100, le=600, description="Design flue gas outlet temperature (F)"
    )
    design_gas_flow_acfm: float = Field(
        default=50000.0, gt=0, description="Design flue gas flow rate (ACFM)"
    )
    design_heat_duty_mmbtu_hr: float = Field(
        default=5.0, gt=0, description="Design heat duty (MMBtu/hr)"
    )

    # Operating limits
    max_water_pressure_psig: float = Field(
        default=300.0, gt=0, description="Maximum water pressure (psig)"
    )
    max_water_temperature_f: float = Field(
        default=450.0, gt=0, description="Maximum water temperature (F)"
    )
    min_water_flow_gpm: float = Field(
        default=50.0, ge=0, description="Minimum water flow rate (GPM)"
    )
    max_gas_inlet_temp_f: float = Field(
        default=1000.0, gt=0, description="Maximum flue gas inlet temperature (F)"
    )

    # Pressure drop
    design_water_pressure_drop_psid: float = Field(
        default=5.0, ge=0, description="Design water side pressure drop (psid)"
    )
    design_gas_pressure_drop_inwc: float = Field(
        default=1.5, ge=0, description="Design gas side pressure drop (in WC)"
    )

    # Metadata
    installation_date: Optional[datetime] = Field(
        None, description="Installation date"
    )
    last_inspection_date: Optional[datetime] = Field(
        None, description="Last inspection date"
    )
    next_inspection_date: Optional[datetime] = Field(
        None, description="Next scheduled inspection date"
    )
    location: Optional[str] = Field(None, description="Physical location")
    associated_boiler_id: Optional[str] = Field(
        None, description="Associated boiler ID"
    )

    @model_validator(mode="after")
    def validate_temperatures(self) -> "EconomizerConfiguration":
        """Validate temperature relationships."""
        if self.design_water_outlet_temp_f <= self.design_water_inlet_temp_f:
            raise ValueError(
                "Water outlet temperature must be greater than inlet temperature"
            )
        if self.design_gas_outlet_temp_f >= self.design_gas_inlet_temp_f:
            raise ValueError(
                "Gas outlet temperature must be less than inlet temperature"
            )
        return self

    @model_validator(mode="after")
    def calculate_surface_area_if_missing(self) -> "EconomizerConfiguration":
        """Calculate heat transfer area from tube dimensions if not provided."""
        import math
        if self.total_heat_transfer_area_sqft == 0:
            # Calculate: pi * D * L * N
            tube_area = (
                math.pi *
                (self.tube_outer_diameter_inches / 12) *
                self.tube_length_ft *
                self.tube_count
            )
            logger.info(f"Calculated heat transfer area: {tube_area:.1f} sq ft")
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "economizer_id": "ECON-001",
                "economizer_type": "finned_tube",
                "economizer_name": "Main Boiler Economizer",
                "tube_count": 200,
                "tube_rows": 10,
                "total_heat_transfer_area_sqft": 2500.0,
                "design_water_flow_gpm": 500.0,
                "design_heat_duty_mmbtu_hr": 8.0,
            }
        }


# ============================================================================
# ALERT CONFIGURATION
# ============================================================================


class AlertThreshold(BaseModel):
    """
    Alert threshold configuration for a specific metric.

    Defines warning and critical thresholds with escalation rules.
    """

    metric_name: str = Field(..., description="Name of the metric")
    metric_units: str = Field(default="", description="Units of measurement")

    # Thresholds
    warning_low: Optional[float] = Field(None, description="Low warning threshold")
    warning_high: Optional[float] = Field(None, description="High warning threshold")
    critical_low: Optional[float] = Field(None, description="Low critical threshold")
    critical_high: Optional[float] = Field(None, description="High critical threshold")

    # Deadband and delays
    deadband_pct: float = Field(
        default=2.0, ge=0, le=20, description="Deadband percentage for alert clearing"
    )
    alert_delay_seconds: int = Field(
        default=60, ge=0, le=3600, description="Delay before generating alert"
    )
    clear_delay_seconds: int = Field(
        default=120, ge=0, le=3600, description="Delay before clearing alert"
    )

    # Rate of change alerts
    enable_roc_alert: bool = Field(
        default=False, description="Enable rate of change alerts"
    )
    roc_warning_per_hour: Optional[float] = Field(
        None, description="Rate of change warning threshold per hour"
    )
    roc_critical_per_hour: Optional[float] = Field(
        None, description="Rate of change critical threshold per hour"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "metric_name": "fouling_resistance",
                "metric_units": "hr-ft2-F/BTU",
                "warning_high": 0.002,
                "critical_high": 0.004,
                "deadband_pct": 5.0,
                "alert_delay_seconds": 300,
            }
        }


class AlertConfiguration(BaseModel):
    """
    Comprehensive alert configuration for economizer monitoring.

    Defines all alert thresholds, escalation rules, and notification settings.
    """

    # Alert thresholds for various metrics
    fouling_resistance_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="fouling_resistance",
            metric_units="hr-ft2-F/BTU",
            warning_high=0.002,
            critical_high=0.004,
            enable_roc_alert=True,
            roc_warning_per_hour=0.0002,
        ),
        description="Fouling resistance alert thresholds"
    )

    approach_temperature_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="approach_temperature",
            metric_units="degF",
            warning_high=60.0,
            critical_high=80.0,
            warning_low=20.0,
        ),
        description="Approach temperature alert thresholds"
    )

    effectiveness_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="effectiveness",
            metric_units="%",
            warning_low=60.0,
            critical_low=50.0,
        ),
        description="Effectiveness alert thresholds"
    )

    gas_pressure_drop_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="gas_pressure_drop",
            metric_units="in WC",
            warning_high=2.5,
            critical_high=4.0,
        ),
        description="Gas side pressure drop thresholds"
    )

    water_outlet_temp_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="water_outlet_temperature",
            metric_units="degF",
            warning_high=320.0,
            critical_high=350.0,
        ),
        description="Water outlet temperature thresholds"
    )

    gas_outlet_temp_threshold: AlertThreshold = Field(
        default_factory=lambda: AlertThreshold(
            metric_name="gas_outlet_temperature",
            metric_units="degF",
            warning_high=400.0,
            critical_high=450.0,
        ),
        description="Gas outlet temperature thresholds"
    )

    # Escalation settings
    escalation_enabled: bool = Field(
        default=True, description="Enable alert escalation"
    )
    escalation_time_minutes: int = Field(
        default=30, ge=5, le=240, description="Time before escalating (minutes)"
    )
    max_escalation_level: int = Field(
        default=3, ge=1, le=5, description="Maximum escalation level"
    )

    # Notification settings
    enable_email_notifications: bool = Field(
        default=True, description="Enable email notifications"
    )
    enable_sms_notifications: bool = Field(
        default=False, description="Enable SMS notifications"
    )
    enable_scada_alarms: bool = Field(
        default=True, description="Enable SCADA alarm integration"
    )

    # Notification recipients by severity
    info_recipients: List[str] = Field(
        default_factory=list, description="Info alert recipients"
    )
    warning_recipients: List[str] = Field(
        default_factory=list, description="Warning alert recipients"
    )
    critical_recipients: List[str] = Field(
        default_factory=list, description="Critical alert recipients"
    )
    emergency_recipients: List[str] = Field(
        default_factory=list, description="Emergency alert recipients"
    )

    # Alert suppression
    enable_maintenance_suppression: bool = Field(
        default=True, description="Suppress alerts during maintenance"
    )
    quiet_hours_start: Optional[time] = Field(
        None, description="Quiet hours start time"
    )
    quiet_hours_end: Optional[time] = Field(
        None, description="Quiet hours end time"
    )
    quiet_hours_severities: List[AlertSeverity] = Field(
        default=[AlertSeverity.INFO],
        description="Severities to suppress during quiet hours"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "escalation_enabled": True,
                "escalation_time_minutes": 30,
                "enable_email_notifications": True,
                "warning_recipients": ["operator@company.com"],
                "critical_recipients": ["supervisor@company.com"],
            }
        }


# ============================================================================
# SOOT BLOWER CONFIGURATION
# ============================================================================


class SootBlowerZone(BaseModel):
    """
    Soot blower zone configuration.

    Defines a specific soot blower zone within the economizer.
    """

    zone_id: str = Field(..., description="Zone identifier")
    zone_name: Optional[str] = Field(None, description="Zone display name")
    zone_location: str = Field(
        default="", description="Physical location description"
    )

    # Soot blower equipment
    soot_blower_ids: List[str] = Field(
        default_factory=list, description="Soot blower IDs in this zone"
    )
    soot_blower_count: int = Field(
        default=1, ge=1, description="Number of soot blowers in zone"
    )

    # Cleaning parameters
    cleaning_sequence_order: int = Field(
        default=1, ge=1, description="Order in cleaning sequence"
    )
    cleaning_duration_seconds: int = Field(
        default=60, ge=10, le=600, description="Cleaning duration per blower (seconds)"
    )
    dwell_time_seconds: int = Field(
        default=5, ge=0, le=60, description="Dwell time between strokes (seconds)"
    )

    # Operating pressures
    operating_pressure_psig: float = Field(
        default=150.0, gt=0, description="Operating pressure (psig)"
    )
    min_pressure_psig: float = Field(
        default=100.0, ge=0, description="Minimum operating pressure (psig)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "zone_id": "ZONE-01",
                "zone_name": "Upper Economizer Bank",
                "soot_blower_ids": ["SB-001", "SB-002"],
                "cleaning_sequence_order": 1,
            }
        }


class SootBlowerConfiguration(BaseModel):
    """
    Soot blower system configuration.

    Defines soot blower parameters, scheduling, and effectiveness tracking.
    """

    system_id: str = Field(..., description="Soot blower system identifier")
    system_name: Optional[str] = Field(None, description="System display name")

    # Media configuration
    media_type: SootBlowerMediaType = Field(
        default=SootBlowerMediaType.STEAM, description="Soot blowing media type"
    )
    media_source: str = Field(
        default="auxiliary_steam", description="Media source (header, compressor, etc.)"
    )
    media_temperature_f: float = Field(
        default=350.0, gt=0, description="Media temperature (F)"
    )
    media_pressure_psig: float = Field(
        default=150.0, gt=0, description="Media supply pressure (psig)"
    )

    # Zones
    zones: List[SootBlowerZone] = Field(
        default_factory=list, description="Soot blower zones"
    )

    # Scheduling
    enable_automatic_blowing: bool = Field(
        default=True, description="Enable automatic soot blowing"
    )
    blowing_interval_hours: float = Field(
        default=8.0, gt=0.5, le=72, description="Interval between cleaning cycles (hours)"
    )
    adaptive_scheduling_enabled: bool = Field(
        default=True, description="Enable adaptive scheduling based on fouling rate"
    )
    min_interval_hours: float = Field(
        default=2.0, gt=0.5, description="Minimum interval between cycles (hours)"
    )
    max_interval_hours: float = Field(
        default=24.0, gt=1, description="Maximum interval between cycles (hours)"
    )

    # Triggers
    fouling_trigger_rf: float = Field(
        default=0.003, gt=0, description="Fouling resistance trigger (hr-ft2-F/BTU)"
    )
    pressure_drop_trigger_inwc: float = Field(
        default=2.0, gt=0, description="Gas pressure drop trigger (in WC)"
    )
    effectiveness_trigger_pct: float = Field(
        default=65.0, ge=0, le=100, description="Effectiveness drop trigger (%)"
    )

    # Consumption tracking
    steam_consumption_lb_per_cycle: float = Field(
        default=500.0, ge=0, description="Steam consumption per cycle (lb)"
    )
    estimated_annual_steam_cost_usd: float = Field(
        default=0.0, ge=0, description="Estimated annual steam cost ($)"
    )

    # Effectiveness
    expected_rf_reduction_pct: float = Field(
        default=80.0, ge=0, le=100, description="Expected fouling reduction per clean (%)"
    )
    typical_recovery_time_minutes: int = Field(
        default=30, ge=0, description="Typical recovery time after cleaning (minutes)"
    )

    # Operating constraints
    min_load_for_blowing_pct: float = Field(
        default=50.0, ge=0, le=100, description="Minimum boiler load for blowing (%)"
    )
    lockout_during_startup: bool = Field(
        default=True, description="Lockout during boiler startup"
    )
    lockout_during_low_steam_pressure: bool = Field(
        default=True, description="Lockout during low steam pressure"
    )

    @model_validator(mode="after")
    def validate_intervals(self) -> "SootBlowerConfiguration":
        """Validate interval relationships."""
        if self.min_interval_hours >= self.max_interval_hours:
            raise ValueError(
                "Minimum interval must be less than maximum interval"
            )
        if self.blowing_interval_hours < self.min_interval_hours:
            logger.warning(
                f"Default blowing interval ({self.blowing_interval_hours}h) is below "
                f"minimum ({self.min_interval_hours}h)"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "system_id": "SB-SYS-001",
                "system_name": "Economizer Soot Blower System",
                "media_type": "steam",
                "blowing_interval_hours": 8.0,
                "adaptive_scheduling_enabled": True,
            }
        }


# ============================================================================
# BASELINE CONFIGURATION
# ============================================================================


class BaselineConfiguration(BaseModel):
    """
    Clean condition baseline reference values.

    Defines reference performance values for a clean economizer
    used to calculate fouling and efficiency degradation.
    """

    baseline_id: str = Field(
        default="BASELINE-001", description="Baseline identifier"
    )
    baseline_name: Optional[str] = Field(None, description="Baseline description")
    established_date: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="Date baseline was established"
    )

    # Clean condition U-value (overall heat transfer coefficient)
    clean_u_value_btu_hr_ft2_f: float = Field(
        default=15.0, gt=0, le=100,
        description="Clean condition overall U-value (BTU/hr-ft2-F)"
    )

    # Reference operating conditions
    reference_water_flow_gpm: float = Field(
        default=500.0, gt=0, description="Reference water flow rate (GPM)"
    )
    reference_gas_flow_acfm: float = Field(
        default=50000.0, gt=0, description="Reference gas flow rate (ACFM)"
    )
    reference_water_inlet_temp_f: float = Field(
        default=200.0, gt=32, description="Reference water inlet temperature (F)"
    )
    reference_gas_inlet_temp_f: float = Field(
        default=550.0, gt=200, description="Reference gas inlet temperature (F)"
    )

    # Expected performance at reference conditions
    expected_water_temp_rise_f: float = Field(
        default=80.0, gt=0, description="Expected water temperature rise (F)"
    )
    expected_gas_temp_drop_f: float = Field(
        default=200.0, gt=0, description="Expected gas temperature drop (F)"
    )
    expected_effectiveness_pct: float = Field(
        default=75.0, ge=0, le=100, description="Expected effectiveness (%)"
    )
    expected_approach_temp_f: float = Field(
        default=30.0, ge=0, description="Expected approach temperature (F)"
    )

    # Pressure drops at clean condition
    clean_gas_pressure_drop_inwc: float = Field(
        default=1.5, ge=0, description="Clean gas side pressure drop (in WC)"
    )
    clean_water_pressure_drop_psid: float = Field(
        default=5.0, ge=0, description="Clean water side pressure drop (psid)"
    )

    # Heat duty
    reference_heat_duty_mmbtu_hr: float = Field(
        default=5.0, gt=0, description="Reference heat duty (MMBtu/hr)"
    )

    # Fouling rate constants (for prediction model)
    typical_fouling_rate_per_day: float = Field(
        default=0.0001, ge=0,
        description="Typical fouling rate (hr-ft2-F/BTU per day)"
    )
    max_acceptable_fouling_resistance: float = Field(
        default=0.005, gt=0,
        description="Maximum acceptable fouling resistance (hr-ft2-F/BTU)"
    )

    # Correction factors
    load_correction_factor: float = Field(
        default=1.0, gt=0, le=2, description="Load correction factor"
    )
    ambient_correction_factor: float = Field(
        default=1.0, gt=0, le=2, description="Ambient temperature correction factor"
    )

    @model_validator(mode="after")
    def validate_approach_temp(self) -> "BaselineConfiguration":
        """Validate approach temperature is physically reasonable."""
        expected_water_outlet = (
            self.reference_water_inlet_temp_f + self.expected_water_temp_rise_f
        )
        min_approach = self.reference_gas_inlet_temp_f - expected_water_outlet

        if self.expected_approach_temp_f > min_approach:
            logger.warning(
                f"Expected approach temp ({self.expected_approach_temp_f}F) seems high "
                f"for given conditions (min physical: {min_approach:.1f}F)"
            )
        return self

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "baseline_id": "BASELINE-001",
                "baseline_name": "Post-Cleaning Baseline 2025",
                "clean_u_value_btu_hr_ft2_f": 15.0,
                "reference_water_flow_gpm": 500.0,
                "expected_effectiveness_pct": 75.0,
            }
        }


# ============================================================================
# SCADA INTEGRATION
# ============================================================================


class SCADAIntegration(BaseModel):
    """
    SCADA system integration configuration.

    Defines connection parameters and tag mappings for SCADA integration.
    """

    enabled: bool = Field(default=True, description="SCADA integration enabled")
    scada_system: str = Field(default="", description="SCADA system type/vendor")
    connection_string: str = Field(default="", description="SCADA connection string")
    polling_interval_seconds: int = Field(
        default=10, ge=1, le=300, description="Data polling interval (seconds)"
    )

    # Temperature tags
    water_inlet_temp_tag: str = Field(
        default="", description="Water inlet temperature tag"
    )
    water_outlet_temp_tag: str = Field(
        default="", description="Water outlet temperature tag"
    )
    gas_inlet_temp_tag: str = Field(
        default="", description="Gas inlet temperature tag"
    )
    gas_outlet_temp_tag: str = Field(
        default="", description="Gas outlet temperature tag"
    )

    # Flow tags
    water_flow_tag: str = Field(
        default="", description="Water flow rate tag"
    )
    gas_flow_tag: str = Field(
        default="", description="Gas flow rate tag"
    )

    # Pressure tags
    water_inlet_pressure_tag: str = Field(
        default="", description="Water inlet pressure tag"
    )
    water_outlet_pressure_tag: str = Field(
        default="", description="Water outlet pressure tag"
    )
    gas_pressure_drop_tag: str = Field(
        default="", description="Gas side pressure drop tag"
    )

    # Soot blower status tags
    soot_blower_active_tag: str = Field(
        default="", description="Soot blower active status tag"
    )
    last_soot_blow_time_tag: str = Field(
        default="", description="Last soot blow timestamp tag"
    )

    # Boiler reference tags
    boiler_load_tag: str = Field(
        default="", description="Boiler load percentage tag"
    )
    steam_flow_tag: str = Field(
        default="", description="Steam flow rate tag"
    )

    # Data quality settings
    enable_data_validation: bool = Field(
        default=True, description="Enable SCADA data quality validation"
    )
    max_data_age_seconds: int = Field(
        default=60, ge=10, description="Maximum acceptable data age (seconds)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "enabled": True,
                "scada_system": "Wonderware",
                "polling_interval_seconds": 10,
                "water_inlet_temp_tag": "ECON.WATER_IN_TEMP",
                "water_outlet_temp_tag": "ECON.WATER_OUT_TEMP",
                "gas_inlet_temp_tag": "ECON.GAS_IN_TEMP",
                "gas_outlet_temp_tag": "ECON.GAS_OUT_TEMP",
            }
        }


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================


class AgentConfiguration(BaseModel):
    """
    Main agent configuration.

    Top-level configuration for the Economizer Performance Agent.
    """

    agent_name: str = Field(
        default="GL-020 ECONOPULSE", description="Agent name"
    )
    version: str = Field(default="1.0.0", description="Agent version")
    environment: str = Field(
        default="production", description="Deployment environment"
    )

    # Economizer configurations
    economizers: List[EconomizerConfiguration] = Field(
        ..., min_length=1, description="List of economizer configurations"
    )

    # Sensor configurations
    sensors: List[SensorConfiguration] = Field(
        default_factory=list, description="List of sensor configurations"
    )

    # Alert configuration
    alert_configuration: AlertConfiguration = Field(
        default_factory=AlertConfiguration,
        description="Alert configuration"
    )

    # Soot blower configuration
    soot_blower_configuration: Optional[SootBlowerConfiguration] = Field(
        None, description="Soot blower system configuration"
    )

    # Baseline configuration
    baseline_configuration: BaselineConfiguration = Field(
        default_factory=BaselineConfiguration,
        description="Baseline performance configuration"
    )

    # SCADA integration
    scada_integration: SCADAIntegration = Field(
        default_factory=SCADAIntegration,
        description="SCADA integration configuration"
    )

    # Operational settings
    analysis_interval_seconds: int = Field(
        default=60, ge=10, le=3600,
        description="Analysis execution interval (seconds)"
    )
    trend_calculation_interval_minutes: int = Field(
        default=15, ge=5, le=60,
        description="Performance trend calculation interval (minutes)"
    )
    historical_data_retention_days: int = Field(
        default=90, ge=7, le=730,
        description="Historical data retention (days)"
    )

    # Feature flags
    enable_fouling_prediction: bool = Field(
        default=True, description="Enable fouling prediction model"
    )
    enable_adaptive_cleaning: bool = Field(
        default=True, description="Enable adaptive cleaning scheduling"
    )
    enable_efficiency_loss_calculation: bool = Field(
        default=True, description="Enable efficiency loss cost calculation"
    )
    enable_anomaly_detection: bool = Field(
        default=True, description="Enable anomaly detection"
    )

    # Economic parameters
    fuel_cost_per_mmbtu: float = Field(
        default=4.0, ge=0, description="Fuel cost ($/MMBtu)"
    )
    steam_cost_per_klb: float = Field(
        default=10.0, ge=0, description="Steam cost ($/1000 lb)"
    )
    operating_hours_per_year: int = Field(
        default=8000, ge=0, le=8760, description="Annual operating hours"
    )

    # Reporting
    enable_hourly_reports: bool = Field(
        default=True, description="Generate hourly reports"
    )
    enable_daily_reports: bool = Field(
        default=True, description="Generate daily reports"
    )
    enable_cleaning_reports: bool = Field(
        default=True, description="Generate post-cleaning reports"
    )
    report_recipients: List[str] = Field(
        default_factory=list, description="Report recipient email addresses"
    )

    def get_economizer(self, economizer_id: str) -> Optional[EconomizerConfiguration]:
        """
        Get economizer configuration by ID.

        Args:
            economizer_id: Economizer identifier

        Returns:
            EconomizerConfiguration or None if not found
        """
        for economizer in self.economizers:
            if economizer.economizer_id == economizer_id:
                return economizer
        return None

    def get_sensor(self, sensor_id: str) -> Optional[SensorConfiguration]:
        """
        Get sensor configuration by ID.

        Args:
            sensor_id: Sensor identifier

        Returns:
            SensorConfiguration or None if not found
        """
        for sensor in self.sensors:
            if sensor.sensor_id == sensor_id:
                return sensor
        return None

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "agent_name": "GL-020 ECONOPULSE",
                "version": "1.0.0",
                "environment": "production",
                "analysis_interval_seconds": 60,
                "enable_fouling_prediction": True,
                "fuel_cost_per_mmbtu": 4.0,
            }
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "EconomizerType",
    "FoulingType",
    "AlertSeverity",
    "CleaningMethod",
    "SootBlowerMediaType",
    "TubeMaterial",
    "SensorType",
    "PerformanceStatus",
    "SensorConfiguration",
    "EconomizerConfiguration",
    "AlertThreshold",
    "AlertConfiguration",
    "SootBlowerZone",
    "SootBlowerConfiguration",
    "BaselineConfiguration",
    "SCADAIntegration",
    "AgentConfiguration",
]

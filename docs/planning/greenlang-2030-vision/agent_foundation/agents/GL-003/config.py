# -*- coding: utf-8 -*-
"""
Configuration module for SteamSystemAnalyzer agent (GL-003).

This module defines the configuration models and settings for the
SteamSystemAnalyzer agent, including steam system specifications,
operational constraints, optimization parameters, and integration settings.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from pathlib import Path
from greenlang.determinism import DeterministicClock


class SteamSystemSpecification(BaseModel):
    """Steam system technical specifications."""

    system_id: str = Field(..., min_length=1, max_length=50, description="Unique system identifier")
    site_name: str = Field(..., min_length=1, max_length=100, description="Site name")
    system_type: str = Field(..., description="System type (generation, distribution, process)")

    # Generation specifications
    total_steam_capacity_kg_hr: float = Field(..., ge=1000, le=1000000, description="Total steam generation capacity")
    boiler_count: int = Field(..., ge=1, le=50, description="Number of boilers in system")
    average_boiler_capacity_kg_hr: float = Field(..., ge=100, le=500000, description="Average boiler capacity")

    # Steam parameters
    design_pressure_bar: float = Field(..., gt=0, le=300, description="Design pressure in bar")
    design_temperature_c: float = Field(..., ge=100, le=600, description="Design temperature in Celsius")
    steam_quality_target: float = Field(0.98, ge=0.8, le=1.0, description="Target steam quality (dryness fraction)")

    # Distribution network
    total_pipeline_length_m: float = Field(..., ge=10, le=100000, description="Total pipeline length")
    insulation_type: str = Field(..., description="Insulation type (mineral_wool, fiberglass, etc.)")
    insulation_thickness_mm: float = Field(..., ge=10, le=500, description="Insulation thickness")

    # Steam consumers
    consumer_count: int = Field(..., ge=1, le=1000, description="Number of steam consumers")
    process_count: int = Field(..., ge=1, le=500, description="Number of process units")

    # Condensate return
    condensate_return_enabled: bool = Field(True, description="Condensate return system enabled")
    condensate_return_rate_percent: float = Field(70.0, ge=0, le=100, description="Condensate return rate")
    condensate_tank_capacity_m3: float = Field(..., ge=1, le=1000, description="Condensate tank capacity")

    # Steam traps
    steam_trap_count: int = Field(..., ge=1, le=10000, description="Total number of steam traps")
    trap_types: List[str] = Field(default_factory=list, description="Steam trap types installed")

    # Age and maintenance
    commissioning_date: datetime = Field(..., description="System commissioning date")
    last_major_upgrade: Optional[datetime] = Field(None, description="Last major upgrade date")
    operating_hours: int = Field(..., ge=0, le=1000000, description="Total operating hours")

    @validator('design_temperature_c')
    def validate_design_temperature(cls, v: float) -> float:
        """Validate design temperature."""
        if not (100 <= v <= 600):
            raise ValueError('Design temperature must be between 100 and 600 Celsius')
        return v

    @validator('commissioning_date')
    def validate_commissioning_date(cls, v: datetime) -> datetime:
        """Validate commissioning date is not in future."""
        if v > DeterministicClock.now():
            raise ValueError('Commissioning date cannot be in the future')
        return v

    @validator('steam_quality_target')
    def validate_steam_quality(cls, v: float) -> float:
        """Validate steam quality is between 0.8 and 1.0."""
        if not (0.8 <= v <= 1.0):
            raise ValueError('Steam quality must be between 0.8 and 1.0')
        return v


class SensorConfiguration(BaseModel):
    """Sensor configuration for steam system monitoring."""

    # Steam meters
    steam_meter_count: int = Field(..., ge=1, le=500, description="Number of steam meters")
    steam_meter_accuracy_percent: float = Field(0.5, ge=0.1, le=5.0, description="Steam meter accuracy")
    steam_meter_locations: List[str] = Field(default_factory=list, description="Steam meter locations")

    # Pressure sensors
    pressure_sensor_count: int = Field(..., ge=1, le=1000, description="Number of pressure sensors")
    pressure_sensor_range_bar: Tuple[float, float] = Field((0, 100), description="Pressure sensor range")
    pressure_sensor_accuracy_bar: float = Field(0.1, ge=0.01, le=1.0, description="Pressure sensor accuracy")

    # Temperature sensors
    temperature_sensor_count: int = Field(..., ge=1, le=1000, description="Number of temperature sensors")
    temperature_sensor_range_c: Tuple[float, float] = Field((-50, 600), description="Temperature sensor range")
    temperature_sensor_accuracy_c: float = Field(0.5, ge=0.1, le=5.0, description="Temperature sensor accuracy")

    # Flow meters
    flow_meter_count: int = Field(..., ge=1, le=500, description="Number of flow meters")
    flow_meter_type: str = Field("vortex", description="Flow meter type (vortex, orifice, ultrasonic)")
    flow_meter_accuracy_percent: float = Field(1.0, ge=0.1, le=5.0, description="Flow meter accuracy")

    # Quality sensors
    quality_sensor_count: int = Field(0, ge=0, le=100, description="Number of steam quality sensors")
    condensate_quality_monitoring: bool = Field(False, description="Condensate quality monitoring enabled")

    # Data acquisition
    sampling_rate_hz: float = Field(1.0, ge=0.1, le=100.0, description="Sensor sampling rate")
    data_retention_days: int = Field(365, ge=7, le=3650, description="Data retention period")

    @validator('steam_meter_accuracy_percent', 'flow_meter_accuracy_percent')
    def validate_accuracy(cls, v: float) -> float:
        """Validate accuracy is positive."""
        if v <= 0:
            raise ValueError('Accuracy must be positive')
        return v


class AnalysisParameters(BaseModel):
    """Parameters for steam system analysis."""

    # Analysis intervals
    realtime_analysis_interval_seconds: int = Field(10, ge=1, le=300, description="Real-time analysis interval")
    efficiency_analysis_interval_minutes: int = Field(15, ge=5, le=1440, description="Efficiency analysis interval")
    leak_detection_interval_minutes: int = Field(60, ge=10, le=1440, description="Leak detection interval")

    # Thresholds
    efficiency_threshold_percent: float = Field(85.0, ge=50, le=100, description="Minimum acceptable efficiency")
    leak_detection_threshold_kg_hr: float = Field(10.0, ge=0.1, le=1000, description="Leak detection threshold")
    pressure_drop_threshold_bar: float = Field(0.5, ge=0.01, le=10, description="Pressure drop threshold")
    temperature_drop_threshold_c: float = Field(5.0, ge=0.1, le=100, description="Temperature drop threshold")

    # Optimization targets
    target_distribution_efficiency: float = Field(95.0, ge=80, le=99.9, description="Target distribution efficiency")
    target_condensate_return: float = Field(90.0, ge=50, le=100, description="Target condensate return rate")
    target_steam_trap_efficiency: float = Field(98.0, ge=90, le=100, description="Target steam trap efficiency")

    # Analysis windows
    moving_average_window_minutes: int = Field(60, ge=5, le=1440, description="Moving average window")
    anomaly_detection_window_hours: int = Field(24, ge=1, le=168, description="Anomaly detection window")

    # Alert thresholds
    critical_efficiency_drop_percent: float = Field(5.0, ge=1, le=20, description="Critical efficiency drop threshold")
    major_leak_threshold_kg_hr: float = Field(100.0, ge=10, le=10000, description="Major leak threshold")
    trap_failure_action_hours: int = Field(24, ge=1, le=168, description="Hours to act on trap failure")

    @validator('efficiency_threshold_percent')
    def validate_efficiency_threshold(cls, v: float) -> float:
        """Validate efficiency threshold is reasonable."""
        if not (50 <= v <= 100):
            raise ValueError('Efficiency threshold must be between 50 and 100 percent')
        return v


class SteamSystemConfiguration(BaseModel):
    """Complete steam system configuration."""

    specification: SteamSystemSpecification = Field(..., description="System specifications")
    sensors: SensorConfiguration = Field(..., description="Sensor configuration")
    analysis: AnalysisParameters = Field(..., description="Analysis parameters")

    # System location
    site_id: str = Field(..., description="Site identifier")
    plant_id: str = Field(..., description="Plant identifier")
    region: str = Field(..., description="Geographic region")

    # Economic parameters
    steam_cost_usd_per_ton: float = Field(30.0, ge=0, description="Steam production cost")
    condensate_value_usd_per_ton: float = Field(5.0, ge=0, description="Condensate recovery value")
    maintenance_cost_usd_per_trap: float = Field(50.0, ge=0, description="Steam trap maintenance cost")

    # Performance baselines
    baseline_distribution_efficiency: float = Field(..., ge=50, le=100, description="Baseline distribution efficiency")
    baseline_steam_losses_kg_hr: float = Field(..., ge=0, description="Baseline steam losses")
    baseline_condensate_return_percent: float = Field(..., ge=0, le=100, description="Baseline condensate return")


class SteamSystemAnalyzerConfig(BaseModel):
    """Main configuration for SteamSystemAnalyzer agent."""

    # Agent identification
    agent_id: str = Field("GL-003", description="Agent identifier")
    agent_name: str = Field("SteamSystemAnalyzer", description="Agent name")
    version: str = Field("1.0.0", description="Agent version")

    # System configurations (support multiple systems)
    systems: List[SteamSystemConfiguration] = Field(..., description="Steam system configurations")
    primary_system_id: str = Field(..., description="Primary system ID for analysis")

    # Performance settings
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    enable_learning: bool = Field(True, description="Enable learning from operations")
    enable_predictive: bool = Field(True, description="Enable predictive maintenance")

    # Calculation settings
    calculation_timeout_seconds: int = Field(30, description="Calculation timeout")
    cache_ttl_seconds: int = Field(60, description="Cache time-to-live")
    max_retries: int = Field(3, description="Maximum retry attempts")

    # Safety parameters
    enable_safety_limits: bool = Field(True, description="Enable safety limit enforcement")
    safety_margin_percent: float = Field(5.0, description="Safety margin for limits")
    emergency_shutdown_enabled: bool = Field(True, description="Enable emergency shutdown")

    # Reporting settings
    report_interval_minutes: int = Field(60, description="Reporting interval")
    report_recipients: List[str] = Field(default_factory=list, description="Report recipients")

    # Integration settings
    scada_enabled: bool = Field(True, description="Enable SCADA integration")
    scada_endpoint: Optional[str] = Field(None, description="SCADA system endpoint")

    dcs_enabled: bool = Field(True, description="Enable DCS integration")
    dcs_endpoint: Optional[str] = Field(None, description="DCS system endpoint")

    historian_enabled: bool = Field(True, description="Enable historian integration")
    historian_endpoint: Optional[str] = Field(None, description="Historian endpoint")

    # Alert settings
    alert_enabled: bool = Field(True, description="Enable alerting")
    alert_channels: List[str] = Field(
        default_factory=lambda: ["email", "sms"],
        description="Alert channels"
    )

    @validator('primary_system_id')
    def validate_primary_system(cls, v, values):
        """Ensure primary system ID exists in systems list."""
        if 'systems' in values:
            system_ids = [s.specification.system_id for s in values['systems']]
            if v not in system_ids:
                raise ValueError(f"Primary system ID {v} not found in systems list")
        return v


# Default configuration factory
def create_default_config() -> SteamSystemAnalyzerConfig:
    """Create default configuration for testing."""

    system_spec = SteamSystemSpecification(
        system_id="STEAM-SYS-001",
        site_name="Manufacturing Plant A",
        system_type="integrated",
        total_steam_capacity_kg_hr=100000,
        boiler_count=3,
        average_boiler_capacity_kg_hr=35000,
        design_pressure_bar=40,
        design_temperature_c=450,
        total_pipeline_length_m=5000,
        insulation_type="mineral_wool",
        insulation_thickness_mm=100,
        consumer_count=50,
        process_count=20,
        condensate_tank_capacity_m3=50,
        steam_trap_count=500,
        commissioning_date=datetime(2018, 1, 1),
        operating_hours=50000
    )

    sensors = SensorConfiguration(
        steam_meter_count=25,
        pressure_sensor_count=100,
        temperature_sensor_count=150,
        flow_meter_count=30
    )

    analysis = AnalysisParameters(
        efficiency_threshold_percent=85.0,
        leak_detection_threshold_kg_hr=10.0,
        target_distribution_efficiency=95.0
    )

    system_config = SteamSystemConfiguration(
        specification=system_spec,
        sensors=sensors,
        analysis=analysis,
        site_id="SITE-001",
        plant_id="PLANT-001",
        region="North America",
        baseline_distribution_efficiency=88.0,
        baseline_steam_losses_kg_hr=5000,
        baseline_condensate_return_percent=70
    )

    return SteamSystemAnalyzerConfig(
        systems=[system_config],
        primary_system_id="STEAM-SYS-001"
    )

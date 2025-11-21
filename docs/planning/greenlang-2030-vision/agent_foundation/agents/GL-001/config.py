# -*- coding: utf-8 -*-
"""
Configuration management for ProcessHeatOrchestrator.

This module defines configuration models and settings for the GL-001
ProcessHeatOrchestrator agent, including plant configurations, sensor
settings, and integration parameters.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path
from enum import Enum


class PlantType(str, Enum):
    """Industrial plant types supported."""

    CHEMICAL = "chemical"
    PETROCHEMICAL = "petrochemical"
    STEEL = "steel"
    CEMENT = "cement"
    PAPER = "paper"
    FOOD_PROCESSING = "food_processing"
    PHARMACEUTICAL = "pharmaceutical"
    GLASS = "glass"
    TEXTILE = "textile"


class SensorType(str, Enum):
    """Process heat sensor types."""

    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    FLOW_RATE = "flow_rate"
    HEAT_FLUX = "heat_flux"
    ENERGY_METER = "energy_meter"
    EMISSION_MONITOR = "emission_monitor"


class IntegrationProtocol(str, Enum):
    """Integration protocols for external systems."""

    OPC_UA = "opc_ua"
    MODBUS = "modbus"
    REST_API = "rest_api"
    MQTT = "mqtt"
    KAFKA = "kafka"
    DATABASE = "database"


class PlantConfiguration(BaseModel):
    """Configuration for industrial plant."""

    plant_id: str = Field(..., description="Unique plant identifier")
    plant_name: str = Field(..., description="Plant display name")
    plant_type: PlantType = Field(..., description="Type of industrial plant")
    location: str = Field(..., description="Plant location")
    capacity_mw: float = Field(..., ge=0, description="Plant capacity in MW")
    operating_hours_per_year: int = Field(8760, ge=0, le=8760, description="Operating hours per year")

    # Process heat specifications
    max_temperature_c: float = Field(..., description="Maximum process temperature in Celsius")
    min_temperature_c: float = Field(..., description="Minimum process temperature in Celsius")
    nominal_pressure_bar: float = Field(..., ge=0, description="Nominal operating pressure in bar")

    # Energy sources
    primary_fuel: str = Field(..., description="Primary fuel type")
    secondary_fuels: List[str] = Field(default_factory=list, description="Secondary fuel types")
    renewable_percentage: float = Field(0.0, ge=0, le=100, description="Renewable energy percentage")

    @validator('max_temperature_c')
    def validate_temperature_range(cls, v, values):
        """Ensure max temperature is greater than min."""
        if 'min_temperature_c' in values and v <= values['min_temperature_c']:
            raise ValueError("Max temperature must be greater than min temperature")
        return v


class SensorConfiguration(BaseModel):
    """Configuration for sensor integration."""

    sensor_id: str = Field(..., description="Unique sensor identifier")
    sensor_type: SensorType = Field(..., description="Type of sensor")
    location: str = Field(..., description="Sensor location in plant")
    unit: str = Field(..., description="Measurement unit")
    sampling_rate_hz: float = Field(1.0, ge=0.01, le=1000, description="Sampling rate in Hz")
    accuracy_percent: float = Field(1.0, ge=0.1, le=10, description="Sensor accuracy percentage")
    calibration_date: str = Field(..., description="Last calibration date")

    # Thresholds
    min_threshold: Optional[float] = Field(None, description="Minimum threshold value")
    max_threshold: Optional[float] = Field(None, description="Maximum threshold value")
    alert_threshold: Optional[float] = Field(None, description="Alert threshold value")
    critical_threshold: Optional[float] = Field(None, description="Critical threshold value")


class SCADAIntegration(BaseModel):
    """SCADA system integration configuration."""

    enabled: bool = Field(True, description="Enable SCADA integration")
    protocol: IntegrationProtocol = Field(IntegrationProtocol.OPC_UA, description="Communication protocol")
    endpoint_url: str = Field(..., description="SCADA endpoint URL")
    polling_interval_seconds: int = Field(5, ge=1, le=60, description="Polling interval in seconds")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Connection timeout")

    # Authentication
    use_authentication: bool = Field(True, description="Use authentication")
    username: Optional[str] = Field(None, description="Username for authentication")
    certificate_path: Optional[Path] = Field(None, description="Certificate path for secure connection")

    # Data points
    tag_mappings: Dict[str, str] = Field(default_factory=dict, description="SCADA tag mappings")
    data_quality_threshold: float = Field(0.9, ge=0, le=1, description="Minimum data quality")


class ERPIntegration(BaseModel):
    """ERP system integration configuration."""

    enabled: bool = Field(True, description="Enable ERP integration")
    system_type: str = Field("SAP", description="ERP system type (SAP, Oracle, etc.)")
    endpoint_url: str = Field(..., description="ERP API endpoint")
    api_version: str = Field("v1", description="API version")

    # Sync settings
    sync_interval_minutes: int = Field(60, ge=5, description="Data sync interval in minutes")
    batch_size: int = Field(1000, ge=100, le=10000, description="Batch size for data sync")

    # Data mappings
    cost_center_mapping: Dict[str, str] = Field(default_factory=dict, description="Cost center mappings")
    material_code_mapping: Dict[str, str] = Field(default_factory=dict, description="Material code mappings")


class OptimizationParameters(BaseModel):
    """Heat distribution optimization parameters."""

    optimization_algorithm: str = Field("linear_programming", description="Optimization algorithm")
    objective_function: str = Field("minimize_cost", description="Optimization objective")

    # Constraints
    max_temperature_variance_c: float = Field(5.0, ge=0, description="Max temperature variance")
    min_efficiency_percent: float = Field(85.0, ge=50, le=100, description="Minimum efficiency")
    max_emissions_kg_per_mwh: float = Field(200.0, ge=0, description="Max emissions per MWh")

    # Optimization settings
    optimization_horizon_hours: int = Field(24, ge=1, le=168, description="Optimization horizon")
    time_step_minutes: int = Field(15, ge=1, le=60, description="Time step for optimization")
    convergence_tolerance: float = Field(0.001, ge=0.0001, le=0.01, description="Convergence tolerance")
    max_iterations: int = Field(1000, ge=100, le=10000, description="Maximum iterations")


class ProcessHeatConfig(BaseModel):
    """Complete configuration for ProcessHeatOrchestrator."""

    # Agent identification
    agent_id: str = Field("GL-001", description="Agent identifier")
    agent_name: str = Field("ProcessHeatOrchestrator", description="Agent name")
    version: str = Field("1.0.0", description="Agent version")

    # Plant configuration
    plants: List[PlantConfiguration] = Field(..., description="Configured plants")

    # Sensor configuration
    sensors: List[SensorConfiguration] = Field(..., description="Configured sensors")

    # Integration settings
    scada_integration: SCADAIntegration = Field(..., description="SCADA integration settings")
    erp_integration: ERPIntegration = Field(..., description="ERP integration settings")

    # Optimization parameters
    optimization: OptimizationParameters = Field(..., description="Optimization parameters")

    # Performance settings
    max_parallel_agents: int = Field(10, ge=1, le=100, description="Max parallel agent coordination")
    calculation_timeout_seconds: int = Field(120, ge=10, le=600, description="Calculation timeout")
    cache_ttl_seconds: int = Field(300, ge=60, le=3600, description="Cache TTL in seconds")

    # Monitoring settings
    enable_monitoring: bool = Field(True, description="Enable performance monitoring")
    metrics_collection_interval: int = Field(60, ge=10, description="Metrics collection interval")
    alert_email_recipients: List[str] = Field(default_factory=list, description="Alert recipients")

    # Compliance settings
    emission_regulations: Dict[str, float] = Field(
        default_factory=dict,
        description="Emission regulations by region"
    )
    compliance_reporting_enabled: bool = Field(True, description="Enable compliance reporting")
    audit_trail_retention_days: int = Field(365, ge=30, description="Audit trail retention")

    @validator('plants')
    def validate_plants(cls, v):
        """Ensure at least one plant is configured."""
        if not v:
            raise ValueError("At least one plant must be configured")
        return v

    @validator('sensors')
    def validate_sensors(cls, v):
        """Ensure at least one sensor is configured."""
        if not v:
            raise ValueError("At least one sensor must be configured")
        return v

    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "agent_id": "GL-001",
                "agent_name": "ProcessHeatOrchestrator",
                "version": "1.0.0",
                "plants": [{
                    "plant_id": "PLANT-001",
                    "plant_name": "Chemical Plant Alpha",
                    "plant_type": "chemical",
                    "location": "Houston, TX",
                    "capacity_mw": 500.0,
                    "max_temperature_c": 850.0,
                    "min_temperature_c": 150.0,
                    "nominal_pressure_bar": 40.0,
                    "primary_fuel": "natural_gas",
                    "secondary_fuels": ["hydrogen", "biomass"],
                    "renewable_percentage": 15.0
                }],
                "sensors": [{
                    "sensor_id": "TEMP-001",
                    "sensor_type": "temperature",
                    "location": "Reactor 1",
                    "unit": "celsius",
                    "sampling_rate_hz": 10.0,
                    "accuracy_percent": 0.5,
                    "calibration_date": "2024-01-15",
                    "max_threshold": 900.0,
                    "critical_threshold": 950.0
                }]
            }
        }
# -*- coding: utf-8 -*-
"""
Heat Recovery Configuration - Configuration management for GL-006 HeatRecoveryMaximizer

This module manages all configuration parameters for the heat recovery optimization
agent including thermodynamic parameters, economic factors, operational constraints,
and system integration settings.

Example:
    >>> config = HeatRecoveryConfig()
    >>> config.MIN_TEMPERATURE_APPROACH_C
    10.0
"""

from typing import Dict, List, Optional, Any
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import os
import json
from pathlib import Path


class EnvironmentType(str, Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class OptimizationMode(str, Enum):
    """Optimization modes for heat recovery."""
    MAX_RECOVERY = "max_recovery"
    MIN_COST = "min_cost"
    BALANCED = "balanced"
    QUICK_WINS = "quick_wins"
    COMPREHENSIVE = "comprehensive"


class HeatRecoveryConfig(BaseSettings):
    """
    Main configuration class for GL-006 HeatRecoveryMaximizer.

    This configuration manages all operational parameters, constraints,
    and integration settings for the heat recovery optimization system.
    """

    # Environment Configuration
    ENVIRONMENT: EnvironmentType = Field(
        default=EnvironmentType.PRODUCTION,
        description="Deployment environment"
    )
    DEBUG_MODE: bool = Field(
        default=False,
        description="Enable debug logging and diagnostics"
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # API Configuration
    API_HOST: str = Field(
        default="0.0.0.0",
        description="API host address"
    )
    API_PORT: int = Field(
        default=8006,
        ge=1,
        le=65535,
        description="API port number"
    )
    API_PREFIX: str = Field(
        default="/api/v1/heat-recovery",
        description="API route prefix"
    )
    API_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="API request timeout"
    )
    ENABLE_CORS: bool = Field(
        default=True,
        description="Enable CORS for API"
    )
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # Thermodynamic Parameters
    MIN_TEMPERATURE_APPROACH_C: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Minimum temperature approach for heat exchange (°C)"
    )
    MIN_RECOVERABLE_TEMPERATURE_C: float = Field(
        default=60.0,
        ge=0.0,
        le=200.0,
        description="Minimum temperature for waste heat recovery (°C)"
    )
    MIN_RECOVERABLE_DUTY_KW: float = Field(
        default=10.0,
        ge=0.0,
        description="Minimum heat duty for recovery consideration (kW)"
    )
    AMBIENT_TEMPERATURE_C: float = Field(
        default=25.0,
        ge=-50.0,
        le=50.0,
        description="Reference ambient temperature (°C)"
    )
    AMBIENT_PRESSURE_BAR: float = Field(
        default=1.013,
        ge=0.5,
        le=2.0,
        description="Reference ambient pressure (bar)"
    )

    # Heat Exchanger Design Parameters
    DEFAULT_OVERALL_HTC_W_M2_K: float = Field(
        default=500.0,
        ge=50.0,
        le=5000.0,
        description="Default overall heat transfer coefficient (W/m²·K)"
    )
    FOULING_RESISTANCE_M2_K_W: float = Field(
        default=0.0002,
        ge=0.0,
        le=0.001,
        description="Default fouling resistance (m²·K/W)"
    )
    MAX_PRESSURE_DROP_BAR: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Maximum allowable pressure drop (bar)"
    )
    HEAT_EXCHANGER_EFFECTIVENESS_MIN: float = Field(
        default=0.7,
        ge=0.3,
        le=0.95,
        description="Minimum acceptable heat exchanger effectiveness"
    )
    DEFAULT_EXCHANGER_TYPE: str = Field(
        default="shell_and_tube",
        description="Default heat exchanger type"
    )

    # Economic Parameters
    ELECTRICITY_COST_USD_KWH: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Electricity cost (USD/kWh)"
    )
    NATURAL_GAS_COST_USD_MMBTU: float = Field(
        default=4.0,
        ge=1.0,
        le=20.0,
        description="Natural gas cost (USD/MMBtu)"
    )
    STEAM_COST_USD_TON: float = Field(
        default=30.0,
        ge=5.0,
        le=100.0,
        description="Steam cost (USD/ton)"
    )
    COOLING_WATER_COST_USD_M3: float = Field(
        default=0.5,
        ge=0.01,
        le=5.0,
        description="Cooling water cost (USD/m³)"
    )
    CARBON_PRICE_USD_TON: float = Field(
        default=50.0,
        ge=0.0,
        le=200.0,
        description="Carbon price (USD/ton CO2)"
    )

    # Financial Parameters
    DISCOUNT_RATE: float = Field(
        default=0.10,
        ge=0.0,
        le=0.30,
        description="Discount rate for NPV calculations"
    )
    INFLATION_RATE: float = Field(
        default=0.03,
        ge=0.0,
        le=0.10,
        description="Annual inflation rate"
    )
    ENERGY_PRICE_ESCALATION_RATE: float = Field(
        default=0.05,
        ge=0.0,
        le=0.15,
        description="Annual energy price escalation rate"
    )
    PROJECT_LIFETIME_YEARS: int = Field(
        default=15,
        ge=5,
        le=30,
        description="Project lifetime for economic analysis (years)"
    )
    TARGET_PAYBACK_YEARS: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Target payback period (years)"
    )
    CAPITAL_RECOVERY_FACTOR: float = Field(
        default=0.15,
        ge=0.05,
        le=0.30,
        description="Capital recovery factor"
    )
    MAINTENANCE_COST_FACTOR: float = Field(
        default=0.03,
        ge=0.01,
        le=0.10,
        description="Annual maintenance cost as fraction of capital"
    )

    # Optimization Parameters
    OPTIMIZATION_MODE: OptimizationMode = Field(
        default=OptimizationMode.BALANCED,
        description="Optimization mode"
    )
    MAX_ITERATIONS: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum optimization iterations"
    )
    CONVERGENCE_TOLERANCE: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-3,
        description="Convergence tolerance for optimization"
    )
    ENABLE_PARALLEL_PROCESSING: bool = Field(
        default=True,
        description="Enable parallel processing for optimization"
    )
    MAX_PARALLEL_WORKERS: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum parallel workers"
    )

    # Operational Constraints
    ANNUAL_OPERATING_HOURS: float = Field(
        default=8000.0,
        ge=1000.0,
        le=8760.0,
        description="Annual operating hours"
    )
    AVAILABILITY_FACTOR: float = Field(
        default=0.95,
        ge=0.5,
        le=1.0,
        description="System availability factor"
    )
    CAPACITY_FACTOR: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Average capacity factor"
    )
    MIN_STREAM_FLOW_RATE_KG_S: float = Field(
        default=0.1,
        ge=0.01,
        description="Minimum stream flow rate (kg/s)"
    )
    MAX_STREAM_TEMPERATURE_C: float = Field(
        default=1000.0,
        ge=100.0,
        le=2000.0,
        description="Maximum allowable stream temperature (°C)"
    )

    # Material and Equipment Costs
    HEAT_EXCHANGER_COST_FACTOR: float = Field(
        default=500.0,
        ge=100.0,
        le=2000.0,
        description="Base cost factor for heat exchangers (USD/m²⁰·⁶⁵)"
    )
    MATERIAL_COST_FACTORS: Dict[str, float] = Field(
        default={
            "carbon_steel": 1.0,
            "stainless_steel_304": 2.5,
            "stainless_steel_316": 3.0,
            "titanium": 5.0,
            "hastelloy": 8.0,
            "copper": 2.0,
            "aluminum": 1.5
        },
        description="Material cost multiplication factors"
    )
    INSTALLATION_COST_FACTOR: float = Field(
        default=1.5,
        ge=1.0,
        le=3.0,
        description="Installation cost as factor of equipment cost"
    )

    # Integration Settings
    ENABLE_ERP_INTEGRATION: bool = Field(
        default=False,
        description="Enable ERP system integration"
    )
    ERP_ENDPOINT_URL: Optional[str] = Field(
        default=None,
        description="ERP system endpoint URL"
    )
    ERP_API_KEY: Optional[str] = Field(
        default=None,
        description="ERP API authentication key"
    )
    ENABLE_SCADA_INTEGRATION: bool = Field(
        default=False,
        description="Enable SCADA system integration"
    )
    SCADA_ENDPOINT_URL: Optional[str] = Field(
        default=None,
        description="SCADA system endpoint URL"
    )
    SCADA_POLLING_INTERVAL_SECONDS: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="SCADA data polling interval"
    )

    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://user:password@localhost:5432/heat_recovery",
        description="Database connection URL"
    )
    DATABASE_POOL_SIZE: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Database connection pool size"
    )
    DATABASE_MAX_OVERFLOW: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Maximum overflow connections"
    )

    # Caching Configuration
    ENABLE_CACHING: bool = Field(
        default=True,
        description="Enable result caching"
    )
    CACHE_TTL_SECONDS: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Cache time-to-live (seconds)"
    )
    REDIS_URL: Optional[str] = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for caching"
    )

    # Monitoring and Metrics
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    METRICS_PORT: int = Field(
        default=9006,
        ge=1,
        le=65535,
        description="Metrics endpoint port"
    )
    ENABLE_TRACING: bool = Field(
        default=False,
        description="Enable distributed tracing"
    )
    JAEGER_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Jaeger tracing endpoint"
    )

    # Safety and Validation
    ENABLE_THERMODYNAMIC_VALIDATION: bool = Field(
        default=True,
        description="Enable thermodynamic validation checks"
    )
    ENABLE_ECONOMIC_VALIDATION: bool = Field(
        default=True,
        description="Enable economic validation checks"
    )
    MAX_VALIDATION_ERRORS: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum validation errors before stopping"
    )
    VALIDATION_STRICTNESS: str = Field(
        default="normal",
        description="Validation strictness level (lenient, normal, strict)"
    )

    # Reporting Configuration
    ENABLE_REPORTING: bool = Field(
        default=True,
        description="Enable report generation"
    )
    REPORT_OUTPUT_PATH: str = Field(
        default="/var/reports/heat_recovery",
        description="Report output directory"
    )
    REPORT_FORMATS: List[str] = Field(
        default=["json", "pdf", "excel"],
        description="Enabled report formats"
    )
    REPORT_RETENTION_DAYS: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Report retention period (days)"
    )

    # Feature Flags
    ENABLE_PINCH_ANALYSIS: bool = Field(
        default=True,
        description="Enable pinch analysis optimization"
    )
    ENABLE_EXERGY_ANALYSIS: bool = Field(
        default=True,
        description="Enable exergy analysis"
    )
    ENABLE_NETWORK_SYNTHESIS: bool = Field(
        default=True,
        description="Enable heat exchanger network synthesis"
    )
    ENABLE_MACHINE_LEARNING: bool = Field(
        default=False,
        description="Enable ML-based optimization (non-critical paths only)"
    )
    ENABLE_REAL_TIME_OPTIMIZATION: bool = Field(
        default=False,
        description="Enable real-time optimization mode"
    )

    # Advanced Thermodynamics
    USE_REAL_GAS_PROPERTIES: bool = Field(
        default=False,
        description="Use real gas properties instead of ideal gas"
    )
    EQUATION_OF_STATE: str = Field(
        default="ideal",
        description="Equation of state (ideal, PR, SRK, IAPWS)"
    )
    PROPERTY_CALCULATION_METHOD: str = Field(
        default="simplified",
        description="Property calculation method (simplified, rigorous)"
    )

    # Emission Factors
    GRID_EMISSION_FACTOR_KG_CO2_KWH: float = Field(
        default=0.5,
        ge=0.0,
        le=1.5,
        description="Grid electricity emission factor (kg CO2/kWh)"
    )
    NATURAL_GAS_EMISSION_FACTOR_KG_CO2_MMBTU: float = Field(
        default=53.06,
        ge=40.0,
        le=70.0,
        description="Natural gas emission factor (kg CO2/MMBtu)"
    )
    STEAM_EMISSION_FACTOR_KG_CO2_TON: float = Field(
        default=80.0,
        ge=50.0,
        le=150.0,
        description="Steam generation emission factor (kg CO2/ton)"
    )

    model_config = SettingsConfigDict(
        env_prefix="GL006_",
        env_file=".env",
        case_sensitive=True,
        use_enum_values=True,
        extra="ignore",
    )

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @field_validator("API_PREFIX")
    @classmethod
    def validate_api_prefix(cls, v: str) -> str:
        """Ensure API prefix starts with /."""
        if not v.startswith("/"):
            return f"/{v}"
        return v

    @field_validator("DATABASE_URL")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v.startswith(("postgresql://", "sqlite://", "mysql://")):
            raise ValueError("Invalid database URL format")
        return v

    @model_validator(mode="after")
    def validate_economic_parameters(self) -> "HeatRecoveryConfig":
        """Validate economic parameters are consistent."""
        if self.TARGET_PAYBACK_YEARS > self.PROJECT_LIFETIME_YEARS:
            raise ValueError("Target payback cannot exceed project lifetime")

        if self.DISCOUNT_RATE < self.INFLATION_RATE:
            raise ValueError("Warning: Discount rate is less than inflation rate")

        return self

    @model_validator(mode="after")
    def validate_operational_parameters(self) -> "HeatRecoveryConfig":
        """Validate operational parameters."""
        if self.CAPACITY_FACTOR > self.AVAILABILITY_FACTOR:
            raise ValueError("Capacity factor cannot exceed availability factor")

        if self.ANNUAL_OPERATING_HOURS > 8760:
            raise ValueError("Annual operating hours cannot exceed 8760")

        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump(exclude_none=True)

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: str) -> "HeatRecoveryConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

    def get_material_cost_factor(self, material: str) -> float:
        """Get cost factor for specified material."""
        return self.MATERIAL_COST_FACTORS.get(material.lower(), 1.0)

    def get_effective_operating_hours(self) -> float:
        """Calculate effective annual operating hours."""
        return self.ANNUAL_OPERATING_HOURS * self.AVAILABILITY_FACTOR * self.CAPACITY_FACTOR

    def get_annualized_capital_factor(self) -> float:
        """Calculate annualized capital cost factor."""
        r = self.DISCOUNT_RATE
        n = self.PROJECT_LIFETIME_YEARS
        if r == 0:
            return 1 / n
        return r * (1 + r) ** n / ((1 + r) ** n - 1)


# Singleton configuration instance
_config_instance: Optional[HeatRecoveryConfig] = None


def get_config() -> HeatRecoveryConfig:
    """Get singleton configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = HeatRecoveryConfig()
    return _config_instance


def reload_config():
    """Reload configuration from environment."""
    global _config_instance
    _config_instance = HeatRecoveryConfig()
    return _config_instance


# Export configuration
config = get_config()

__all__ = [
    "HeatRecoveryConfig",
    "get_config",
    "reload_config",
    "config",
    "EnvironmentType",
    "OptimizationMode"
]
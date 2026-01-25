# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Configuration Module

Configuration settings for the Insulation Scanning & Thermal Assessment agent
using pydantic-settings for environment variable support and validation.

All configuration supports deterministic, reproducible operation with
comprehensive settings for thermal analysis, insulation assessment, ML models,
API, logging, provenance tracking, and feature flags.

Example:
    >>> from config import InsulscanSettings, get_settings
    >>> settings = get_settings()
    >>> print(settings.agent_id)
    GL-015

Environment Variables:
    GL_015_LOG_LEVEL: Logging level (default: INFO)
    GL_015_ENABLE_PROVENANCE: Enable provenance tracking (default: true)
    GL_015_API_PORT: API server port (default: 8015)

Author: GreenLang GL-015 INSULSCAN
Version: 1.0.0
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMS
# =============================================================================


class InsulationType(str, Enum):
    """
    Industrial insulation material types.

    Defines common insulation materials used in process heat applications.
    """
    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    CELLULAR_GLASS = "cellular_glass"
    AEROGEL = "aerogel"
    FIBERGLASS = "fiberglass"
    PERLITE = "perlite"
    CERAMIC_FIBER = "ceramic_fiber"
    POLYURETHANE_FOAM = "polyurethane_foam"
    PHENOLIC_FOAM = "phenolic_foam"


class SurfaceType(str, Enum):
    """
    Industrial equipment surface types for insulation assessment.

    Defines equipment surfaces commonly requiring insulation.
    """
    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    DUCT = "duct"
    VALVE = "valve"
    FLANGE = "flange"
    ELBOW = "elbow"
    TEE = "tee"
    REDUCER = "reducer"
    HEAT_EXCHANGER = "heat_exchanger"
    BOILER = "boiler"
    TURBINE = "turbine"


class HotSpotSeverity(str, Enum):
    """Hot spot severity classification."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ConditionSeverity(str, Enum):
    """Insulation condition severity classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"
    FAILED = "failed"


class RepairPriority(str, Enum):
    """Repair priority levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    IMMEDIATE = "immediate"


class RepairType(str, Enum):
    """Types of insulation repairs."""
    NO_ACTION = "no_action"
    MONITORING = "monitoring"
    MINOR_PATCH = "minor_patch"
    SECTION_REPLACEMENT = "section_replacement"
    FULL_REPLACEMENT = "full_replacement"
    UPGRADE_THICKNESS = "upgrade_thickness"
    UPGRADE_MATERIAL = "upgrade_material"
    WEATHERPROOFING = "weatherproofing"


class ThermalImagingMode(str, Enum):
    """Thermal imaging operational modes."""
    SPOT = "spot"
    LINE = "line"
    AREA = "area"
    CONTINUOUS = "continuous"


class DataQuality(str, Enum):
    """Data quality classification."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    SUSPECT = "suspect"
    BAD = "bad"
    INTERPOLATED = "interpolated"


# =============================================================================
# THERMAL PROPERTIES DATABASE
# =============================================================================


# Default thermal conductivity values (W/m-K) at reference temperature
DEFAULT_THERMAL_CONDUCTIVITY: Dict[str, float] = {
    InsulationType.MINERAL_WOOL.value: 0.040,
    InsulationType.CALCIUM_SILICATE.value: 0.055,
    InsulationType.CELLULAR_GLASS.value: 0.048,
    InsulationType.AEROGEL.value: 0.015,
    InsulationType.FIBERGLASS.value: 0.038,
    InsulationType.PERLITE.value: 0.052,
    InsulationType.CERAMIC_FIBER.value: 0.065,
    InsulationType.POLYURETHANE_FOAM.value: 0.025,
    InsulationType.PHENOLIC_FOAM.value: 0.022,
}


# Default surface emissivity values for different materials/coatings
DEFAULT_EMISSIVITY: Dict[str, float] = {
    "aluminum_new": 0.05,
    "aluminum_weathered": 0.12,
    "aluminum_anodized": 0.77,
    "stainless_steel": 0.16,
    "stainless_steel_oxidized": 0.85,
    "carbon_steel": 0.44,
    "carbon_steel_oxidized": 0.78,
    "galvanized_steel": 0.28,
    "galvanized_steel_weathered": 0.90,
    "painted_surface": 0.92,
    "weatherproofing_jacket": 0.85,
    "mastic_coating": 0.90,
    "bare_insulation": 0.93,
    "fiberglass_jacket": 0.90,
}


# =============================================================================
# SETTINGS CLASSES
# =============================================================================


class ThermalAnalysisSettings(BaseSettings):
    """
    Thermal analysis engine settings.

    Configures the deterministic thermal calculation engine including
    heat loss formulas, temperature thresholds, and physical constants.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_THERMAL_",
        extra="ignore",
    )

    # Temperature thresholds for hot spot detection (Celsius)
    hot_spot_warning_delta_c: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Temperature differential for warning classification (C)"
    )
    hot_spot_critical_delta_c: float = Field(
        default=25.0,
        ge=5.0,
        le=100.0,
        description="Temperature differential for critical classification (C)"
    )
    hot_spot_emergency_delta_c: float = Field(
        default=50.0,
        ge=10.0,
        le=200.0,
        description="Temperature differential for emergency classification (C)"
    )

    # Personnel safety thresholds
    personnel_safety_temp_c: float = Field(
        default=60.0,
        ge=40.0,
        le=80.0,
        description="Maximum safe surface temperature for personnel contact (C)"
    )

    # Heat loss calculation parameters
    stefan_boltzmann_constant: float = Field(
        default=5.67e-8,
        description="Stefan-Boltzmann constant (W/m2-K4)"
    )
    reference_temperature_k: float = Field(
        default=298.15,
        ge=200.0,
        le=400.0,
        description="Reference temperature for calculations (K)"
    )

    # Convection coefficients
    natural_convection_coefficient: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Natural convection heat transfer coefficient (W/m2-K)"
    )
    wind_convection_factor: float = Field(
        default=1.2,
        ge=1.0,
        le=5.0,
        description="Factor to increase convection with wind speed"
    )

    # Calculation precision
    convergence_tolerance: float = Field(
        default=1e-6,
        ge=1e-12,
        le=1e-3,
        description="Convergence tolerance for iterative calculations"
    )
    max_iterations: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Maximum iterations for convergence"
    )


class InsulationAssessmentSettings(BaseSettings):
    """
    Insulation condition assessment settings.

    Configures thresholds and parameters for evaluating insulation condition.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_INSULATION_",
        extra="ignore",
    )

    # Condition score thresholds (0-100 scale)
    excellent_threshold: int = Field(
        default=90,
        ge=80,
        le=100,
        description="Minimum score for excellent condition"
    )
    good_threshold: int = Field(
        default=75,
        ge=60,
        le=95,
        description="Minimum score for good condition"
    )
    fair_threshold: int = Field(
        default=60,
        ge=40,
        le=80,
        description="Minimum score for fair condition"
    )
    poor_threshold: int = Field(
        default=40,
        ge=20,
        le=60,
        description="Minimum score for poor condition"
    )
    critical_threshold: int = Field(
        default=20,
        ge=5,
        le=40,
        description="Minimum score for critical condition"
    )

    # Degradation rate estimation
    max_degradation_rate_percent_year: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Maximum expected degradation rate per year"
    )

    # Minimum acceptable insulation thickness (mm)
    min_thickness_pipe_mm: float = Field(
        default=25.0,
        ge=10.0,
        le=100.0,
        description="Minimum insulation thickness for pipes"
    )
    min_thickness_vessel_mm: float = Field(
        default=50.0,
        ge=25.0,
        le=200.0,
        description="Minimum insulation thickness for vessels"
    )

    # Expected insulation lifetime (years)
    typical_lifetime_years: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Typical insulation system lifetime"
    )


class EconomicSettings(BaseSettings):
    """
    Economic parameters for cost-benefit analysis.

    Configures energy costs, labor rates, and material costs for
    repair recommendations and payback calculations.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_ECONOMIC_",
        extra="ignore",
    )

    # Energy costs
    energy_cost_usd_per_kwh: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Electricity cost (USD/kWh)"
    )
    natural_gas_cost_usd_per_therm: float = Field(
        default=1.20,
        ge=0.10,
        le=10.0,
        description="Natural gas cost (USD/therm)"
    )
    steam_cost_usd_per_klb: float = Field(
        default=12.0,
        ge=1.0,
        le=50.0,
        description="Steam cost (USD per 1000 lb)"
    )

    # Operating hours
    operating_hours_per_year: int = Field(
        default=8760,
        ge=1000,
        le=8760,
        description="Annual operating hours"
    )

    # Carbon pricing
    co2_cost_usd_per_tonne: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Carbon price (USD per tonne CO2)"
    )
    co2_emission_factor_kg_per_kwh: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="CO2 emission factor (kg CO2 per kWh)"
    )

    # Repair costs
    labor_rate_usd_per_hour: float = Field(
        default=75.0,
        ge=20.0,
        le=300.0,
        description="Labor rate for repairs (USD/hour)"
    )
    insulation_cost_usd_per_m2: float = Field(
        default=50.0,
        ge=10.0,
        le=500.0,
        description="Insulation material cost per m2"
    )
    jacketing_cost_usd_per_m2: float = Field(
        default=25.0,
        ge=5.0,
        le=200.0,
        description="Jacketing material cost per m2"
    )

    # Minimum payback threshold
    min_acceptable_payback_years: float = Field(
        default=3.0,
        ge=0.5,
        le=10.0,
        description="Minimum acceptable payback period for recommendations"
    )


class MLServiceSettings(BaseSettings):
    """
    Machine Learning service settings.

    Configures the ML components for anomaly detection and
    degradation prediction. Includes reproducibility settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_ML_",
        extra="ignore",
    )

    # Reproducibility
    random_seed: int = Field(
        default=42,
        ge=0,
        le=2**31 - 1,
        description="Random seed for reproducible predictions"
    )

    # Model paths
    model_registry_path: str = Field(
        default="models/registry",
        description="Path to model registry"
    )
    anomaly_model_path: str = Field(
        default="models/anomaly_detector.pkl",
        description="Path to anomaly detection model"
    )
    degradation_model_path: str = Field(
        default="models/degradation_predictor.pkl",
        description="Path to degradation prediction model"
    )

    # Inference settings
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for ML inference"
    )
    inference_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Timeout for ML inference"
    )

    # Confidence thresholds
    min_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for predictions"
    )
    anomaly_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Threshold for anomaly detection"
    )

    # Hot spot detection ML
    hot_spot_detection_model: str = Field(
        default="gradient_based",
        description="Hot spot detection algorithm"
    )
    min_hot_spot_area_m2: float = Field(
        default=0.01,
        ge=0.001,
        le=1.0,
        description="Minimum area to classify as hot spot"
    )


class ThermalImagingSettings(BaseSettings):
    """
    Thermal imaging integration settings.

    Configures parameters for thermal camera integration and
    image processing.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_THERMAL_IMAGING_",
        extra="ignore",
    )

    # Camera settings
    default_emissivity: float = Field(
        default=0.90,
        ge=0.01,
        le=1.0,
        description="Default emissivity for thermal imaging"
    )
    reflected_temperature_c: float = Field(
        default=25.0,
        ge=-50.0,
        le=100.0,
        description="Default reflected temperature (C)"
    )

    # Image processing
    min_temperature_c: float = Field(
        default=-40.0,
        ge=-100.0,
        le=0.0,
        description="Minimum temperature range for imaging"
    )
    max_temperature_c: float = Field(
        default=500.0,
        ge=100.0,
        le=2000.0,
        description="Maximum temperature range for imaging"
    )
    temperature_resolution_c: float = Field(
        default=0.05,
        ge=0.01,
        le=1.0,
        description="Temperature measurement resolution"
    )

    # Integration
    enable_live_streaming: bool = Field(
        default=False,
        description="Enable live thermal image streaming"
    )
    image_format: str = Field(
        default="radiometric",
        description="Thermal image format (radiometric, standard)"
    )


class APISettings(BaseSettings):
    """
    REST/GraphQL API settings.

    Configures the API server for external integrations.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_API_",
        extra="ignore",
    )

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8015,
        ge=1024,
        le=65535,
        description="API server port"
    )
    workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker processes"
    )

    # Security
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    enable_cors: bool = Field(
        default=True,
        description="Enable CORS"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="Allowed CORS origins"
    )

    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Rate limit: requests per minute"
    )

    # Timeouts
    request_timeout_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Request timeout"
    )

    # Documentation
    enable_docs: bool = Field(
        default=True,
        description="Enable API documentation"
    )
    docs_url: str = Field(
        default="/docs",
        description="API docs URL path"
    )


class LoggingSettings(BaseSettings):
    """
    Logging configuration settings.

    Configures structured logging for audit and debugging.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_LOG_",
        extra="ignore",
    )

    # Log level
    level: str = Field(
        default="INFO",
        description="Logging level"
    )

    # Format
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    json_format: bool = Field(
        default=False,
        description="Use JSON log format"
    )

    # Output
    log_to_file: bool = Field(
        default=True,
        description="Enable file logging"
    )
    log_file_path: str = Field(
        default="logs/insulscan.log",
        description="Log file path"
    )
    max_file_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum log file size (MB)"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of backup log files"
    )

    # Audit logging
    enable_audit_log: bool = Field(
        default=True,
        description="Enable separate audit log"
    )
    audit_log_path: str = Field(
        default="logs/audit.log",
        description="Audit log file path"
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return upper_v


class ProvenanceSettings(BaseSettings):
    """
    Provenance tracking settings.

    Configures SHA-256 provenance tracking for audit compliance.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_PROVENANCE_",
        extra="ignore",
    )

    # Enable/disable
    enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )

    # Storage
    store_records: bool = Field(
        default=True,
        description="Store provenance records in memory"
    )
    persist_to_file: bool = Field(
        default=True,
        description="Persist provenance to file"
    )
    provenance_file_path: str = Field(
        default="provenance/records.jsonl",
        description="Provenance records file path"
    )

    # Retention
    max_records_in_memory: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum records to keep in memory"
    )
    retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Days to retain provenance records"
    )

    # Hash settings
    include_type_info: bool = Field(
        default=True,
        description="Include type information in hashes"
    )
    truncate_hash_display: int = Field(
        default=16,
        ge=8,
        le=64,
        description="Characters to display for truncated hashes"
    )


class FeatureFlags(BaseSettings):
    """
    Feature flags for enabling/disabling functionality.

    Allows runtime control of agent features.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_FEATURE_",
        extra="ignore",
    )

    # Core features
    enable_ml_predictions: bool = Field(
        default=True,
        description="Enable ML-based predictions"
    )
    enable_hot_spot_detection: bool = Field(
        default=True,
        description="Enable automated hot spot detection"
    )
    enable_repair_recommendations: bool = Field(
        default=True,
        description="Enable repair recommendations"
    )
    enable_economic_analysis: bool = Field(
        default=True,
        description="Enable economic cost-benefit analysis"
    )

    # Integration features
    enable_thermal_imaging: bool = Field(
        default=False,
        description="Enable thermal camera integration"
    )
    enable_kafka_streaming: bool = Field(
        default=False,
        description="Enable Kafka streaming integration"
    )
    enable_opcua_integration: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )
    enable_cmms_integration: bool = Field(
        default=False,
        description="Enable CMMS work order integration"
    )

    # Safety features
    enable_velocity_limiting: bool = Field(
        default=True,
        description="Enable recommendation velocity limiting"
    )
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    enable_fail_safe_mode: bool = Field(
        default=True,
        description="Enable fail-safe mode"
    )

    # Explainability features
    enable_shap_explanations: bool = Field(
        default=True,
        description="Enable SHAP explanations"
    )
    enable_lime_explanations: bool = Field(
        default=False,
        description="Enable LIME explanations"
    )
    enable_causal_analysis: bool = Field(
        default=True,
        description="Enable causal analysis"
    )

    # Experimental features
    enable_experimental_models: bool = Field(
        default=False,
        description="Enable experimental ML models"
    )
    enable_3d_mapping: bool = Field(
        default=False,
        description="Enable 3D thermal mapping"
    )


class InsulscanSettings(BaseSettings):
    """
    Master configuration for GL-015 INSULSCAN agent.

    All settings required for deterministic, reproducible operation
    of the insulation scanning and thermal assessment system.

    Example:
        >>> settings = InsulscanSettings()
        >>> print(settings.agent_id)
        GL-015
        >>> print(settings.thermal.personnel_safety_temp_c)
        60.0

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        version: Agent version string
        thermal: Thermal analysis settings
        insulation: Insulation assessment settings
        economic: Economic analysis settings
        ml: ML service settings
        thermal_imaging: Thermal imaging settings
        api: API settings
        logging: Logging settings
        provenance: Provenance tracking settings
        features: Feature flags
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_015_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-015",
        description="Unique agent identifier"
    )
    agent_name: str = Field(
        default="INSULSCAN",
        description="Human-readable agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    description: str = Field(
        default="Insulation Scanning & Thermal Assessment - Hot spot detection, heat loss calculation, and repair recommendations",
        description="Agent description"
    )

    # Nested settings
    thermal: ThermalAnalysisSettings = Field(
        default_factory=ThermalAnalysisSettings,
        description="Thermal analysis settings"
    )
    insulation: InsulationAssessmentSettings = Field(
        default_factory=InsulationAssessmentSettings,
        description="Insulation assessment settings"
    )
    economic: EconomicSettings = Field(
        default_factory=EconomicSettings,
        description="Economic analysis settings"
    )
    ml: MLServiceSettings = Field(
        default_factory=MLServiceSettings,
        description="ML service settings"
    )
    thermal_imaging: ThermalImagingSettings = Field(
        default_factory=ThermalImagingSettings,
        description="Thermal imaging settings"
    )
    api: APISettings = Field(
        default_factory=APISettings,
        description="API settings"
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging settings"
    )
    provenance: ProvenanceSettings = Field(
        default_factory=ProvenanceSettings,
        description="Provenance tracking settings"
    )
    features: FeatureFlags = Field(
        default_factory=FeatureFlags,
        description="Feature flags"
    )

    # Deterministic operation
    deterministic_mode: bool = Field(
        default=True,
        description="Enable deterministic mode for reproducibility"
    )
    fail_closed: bool = Field(
        default=True,
        description="Fail closed on errors (safe mode)"
    )

    # Integration endpoints
    kafka_brokers: str = Field(
        default="localhost:9092",
        description="Kafka broker addresses"
    )
    opcua_endpoint: str = Field(
        default="opc.tcp://localhost:4840",
        description="OPC-UA server endpoint"
    )
    graphql_endpoint: str = Field(
        default="http://localhost:8015/graphql",
        description="GraphQL endpoint"
    )

    # Metrics
    metrics_port: int = Field(
        default=9015,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "description": self.description,
            "deterministic_mode": self.deterministic_mode,
            "fail_closed": self.fail_closed,
            "thermal": {
                "personnel_safety_temp_c": self.thermal.personnel_safety_temp_c,
                "hot_spot_warning_delta_c": self.thermal.hot_spot_warning_delta_c,
                "hot_spot_critical_delta_c": self.thermal.hot_spot_critical_delta_c,
            },
            "insulation": {
                "excellent_threshold": self.insulation.excellent_threshold,
                "typical_lifetime_years": self.insulation.typical_lifetime_years,
            },
            "economic": {
                "energy_cost_usd_per_kwh": self.economic.energy_cost_usd_per_kwh,
                "operating_hours_per_year": self.economic.operating_hours_per_year,
            },
            "ml": {
                "random_seed": self.ml.random_seed,
                "min_confidence_threshold": self.ml.min_confidence_threshold,
            },
            "features": {
                "enable_ml_predictions": self.features.enable_ml_predictions,
                "enable_hot_spot_detection": self.features.enable_hot_spot_detection,
                "enable_thermal_imaging": self.features.enable_thermal_imaging,
            },
        }


# =============================================================================
# SETTINGS SINGLETON
# =============================================================================


@lru_cache
def get_settings() -> InsulscanSettings:
    """
    Get cached settings instance.

    Returns the singleton InsulscanSettings instance, loaded from
    environment variables and .env file.

    Returns:
        InsulscanSettings: The global settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.agent_id)
        GL-015
    """
    return InsulscanSettings()


def reload_settings() -> InsulscanSettings:
    """
    Reload settings from environment.

    Clears the cached settings and reloads from environment.
    Useful for testing or dynamic configuration updates.

    Returns:
        InsulscanSettings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================


DEFAULT_CONFIG = InsulscanSettings()


__all__ = [
    # Enums
    "InsulationType",
    "SurfaceType",
    "HotSpotSeverity",
    "ConditionSeverity",
    "RepairPriority",
    "RepairType",
    "ThermalImagingMode",
    "DataQuality",
    # Thermal properties
    "DEFAULT_THERMAL_CONDUCTIVITY",
    "DEFAULT_EMISSIVITY",
    # Settings classes
    "ThermalAnalysisSettings",
    "InsulationAssessmentSettings",
    "EconomicSettings",
    "MLServiceSettings",
    "ThermalImagingSettings",
    "APISettings",
    "LoggingSettings",
    "ProvenanceSettings",
    "FeatureFlags",
    "InsulscanSettings",
    # Functions
    "get_settings",
    "reload_settings",
    # Constants
    "DEFAULT_CONFIG",
]

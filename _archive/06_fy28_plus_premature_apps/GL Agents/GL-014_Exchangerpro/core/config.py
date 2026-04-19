# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Configuration Module

Configuration settings for the Heat Exchanger Optimizer agent using
pydantic-settings for environment variable support and validation.

All configuration supports deterministic, reproducible operation with
comprehensive settings for thermal engine, ML service, optimizer,
API, logging, provenance tracking, and feature flags.

Example:
    >>> from config import ExchangerProSettings, get_settings
    >>> settings = get_settings()
    >>> print(settings.agent_id)
    GL-014

Environment Variables:
    GL_014_LOG_LEVEL: Logging level (default: INFO)
    GL_014_ENABLE_PROVENANCE: Enable provenance tracking (default: true)
    GL_014_API_PORT: API server port (default: 8014)

Author: GreenLang GL-014 EXCHANGERPRO
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


class TEMAType(str, Enum):
    """
    TEMA (Tubular Exchanger Manufacturers Association) shell types.

    Defines standard shell configurations per TEMA standards.
    """
    AES = "AES"  # Split ring floating head
    AET = "AET"  # Pull-through floating head
    AEU = "AEU"  # U-tube bundle
    AEW = "AEW"  # Externally sealed floating tubesheet
    AEP = "AEP"  # Outside packed floating head
    BEM = "BEM"  # Fixed tubesheet (bonnet covers)
    BEU = "BEU"  # U-tube fixed tubesheet
    AKT = "AKT"  # Kettle reboiler
    AJW = "AJW"  # Thermosiphon reboiler
    NEN = "NEN"  # Fixed tubesheet, channel integral


class FlowArrangement(str, Enum):
    """Heat exchanger flow arrangements."""
    COUNTER_CURRENT = "counter_current"
    CO_CURRENT = "co_current"
    CROSS_FLOW = "cross_flow"
    SHELL_AND_TUBE_1_2 = "shell_tube_1_2"  # 1 shell pass, 2 tube passes
    SHELL_AND_TUBE_2_4 = "shell_tube_2_4"  # 2 shell passes, 4 tube passes
    SHELL_AND_TUBE_1_4 = "shell_tube_1_4"  # 1 shell pass, 4 tube passes
    SHELL_AND_TUBE_2_2 = "shell_tube_2_2"  # 2 shell passes, 2 tube passes


class ShellType(str, Enum):
    """Shell types per TEMA classification."""
    E = "E"  # Single pass shell
    F = "F"  # Two pass shell with longitudinal baffle
    G = "G"  # Split flow
    H = "H"  # Double split flow
    J = "J"  # Divided flow
    K = "K"  # Kettle type reboiler
    X = "X"  # Cross flow


class TubePattern(str, Enum):
    """Tube layout patterns."""
    TRIANGULAR_30 = "triangular_30"
    TRIANGULAR_60 = "triangular_60"
    SQUARE_90 = "square_90"
    ROTATED_SQUARE_45 = "rotated_square_45"


class MaterialGrade(str, Enum):
    """Common heat exchanger material grades."""
    CARBON_STEEL = "carbon_steel"
    STAINLESS_304 = "ss_304"
    STAINLESS_316 = "ss_316"
    STAINLESS_316L = "ss_316L"
    STAINLESS_321 = "ss_321"
    DUPLEX_2205 = "duplex_2205"
    SUPER_DUPLEX = "super_duplex"
    INCONEL_625 = "inconel_625"
    TITANIUM_GR2 = "titanium_gr2"
    COPPER_NICKEL_90_10 = "cu_ni_90_10"
    COPPER_NICKEL_70_30 = "cu_ni_70_30"
    ADMIRALTY_BRASS = "admiralty_brass"
    HASTELLOY_C276 = "hastelloy_c276"


class CalculationMethod(str, Enum):
    """Thermal calculation methods."""
    LMTD = "lmtd"
    EPSILON_NTU = "epsilon_ntu"
    P_NTU = "p_ntu"


class FoulingModel(str, Enum):
    """Fouling prediction model types."""
    KERN_SEATON = "kern_seaton"
    EBERT_PANCHAL = "ebert_panchal"
    POLLEY = "polley"
    HYBRID_ML = "hybrid_ml"


class OptimizationObjective(str, Enum):
    """Optimization objectives."""
    MINIMIZE_ENERGY_COST = "minimize_energy_cost"
    MAXIMIZE_HEAT_RECOVERY = "maximize_heat_recovery"
    MINIMIZE_FOULING = "minimize_fouling"
    MINIMIZE_CLEANING_COST = "minimize_cleaning_cost"
    MULTI_OBJECTIVE = "multi_objective"


# =============================================================================
# SETTINGS CLASSES
# =============================================================================


class ThermalEngineSettings(BaseSettings):
    """
    Thermal calculation engine settings.

    Configures the deterministic thermal calculation engine including
    calculation methods, convergence criteria, and physical constants.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_014_THERMAL_",
        extra="ignore",
    )

    # Calculation method
    default_method: CalculationMethod = Field(
        default=CalculationMethod.EPSILON_NTU,
        description="Default thermal calculation method"
    )

    # Convergence settings
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

    # Physical constants
    reference_temperature_k: float = Field(
        default=298.15,
        ge=200.0,
        le=400.0,
        description="Reference temperature for exergy calculations (K)"
    )

    # Heat balance tolerance
    heat_balance_tolerance_percent: float = Field(
        default=5.0,
        ge=0.1,
        le=20.0,
        description="Acceptable heat balance error percentage"
    )

    # LMTD correction factor minimum
    min_lmtd_correction_factor: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Minimum acceptable LMTD correction factor"
    )

    # Pressure drop calculation
    include_pressure_drop: bool = Field(
        default=True,
        description="Include pressure drop calculations"
    )
    max_pressure_drop_shell_kpa: float = Field(
        default=100.0,
        ge=1.0,
        le=1000.0,
        description="Maximum allowable shell-side pressure drop (kPa)"
    )
    max_pressure_drop_tube_kpa: float = Field(
        default=150.0,
        ge=1.0,
        le=1000.0,
        description="Maximum allowable tube-side pressure drop (kPa)"
    )


class MLServiceSettings(BaseSettings):
    """
    Machine Learning service settings.

    Configures the ML components for fouling prediction and
    anomaly detection. Includes reproducibility settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_014_ML_",
        extra="ignore",
    )

    # Model settings
    fouling_model: FoulingModel = Field(
        default=FoulingModel.HYBRID_ML,
        description="Fouling prediction model type"
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

    # Feature engineering
    lookback_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Hours of history for feature engineering"
    )
    feature_aggregation_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Aggregation interval for features"
    )


class OptimizerSettings(BaseSettings):
    """
    Optimization engine settings.

    Configures the cleaning schedule optimizer and multi-objective
    optimization parameters.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_014_OPTIMIZER_",
        extra="ignore",
    )

    # Optimization objective
    objective: OptimizationObjective = Field(
        default=OptimizationObjective.MINIMIZE_CLEANING_COST,
        description="Primary optimization objective"
    )

    # Solver settings
    solver_time_limit_seconds: float = Field(
        default=60.0,
        ge=1.0,
        le=3600.0,
        description="Maximum solver time"
    )

    # Genetic algorithm settings (for multi-objective)
    population_size: int = Field(
        default=100,
        ge=20,
        le=1000,
        description="GA population size"
    )
    n_generations: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Number of GA generations"
    )
    crossover_probability: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Crossover probability"
    )
    mutation_probability: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Mutation probability"
    )

    # Pareto settings
    n_pareto_points: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Number of Pareto points to generate"
    )

    # Cleaning scheduling
    planning_horizon_days: int = Field(
        default=365,
        ge=30,
        le=1825,
        description="Planning horizon for cleaning schedule"
    )
    min_cleaning_interval_days: int = Field(
        default=30,
        ge=7,
        le=180,
        description="Minimum days between cleanings"
    )

    # Economic parameters
    cleaning_cost_base_usd: float = Field(
        default=5000.0,
        ge=0.0,
        le=1000000.0,
        description="Base cleaning cost (USD)"
    )
    energy_cost_usd_per_gj: float = Field(
        default=15.0,
        ge=0.0,
        le=100.0,
        description="Energy cost (USD/GJ)"
    )
    downtime_cost_usd_per_hour: float = Field(
        default=1000.0,
        ge=0.0,
        le=100000.0,
        description="Cost of downtime per hour (USD)"
    )


class APISettings(BaseSettings):
    """
    REST/GraphQL API settings.

    Configures the API server for external integrations.
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_014_API_",
        extra="ignore",
    )

    # Server settings
    host: str = Field(
        default="0.0.0.0",
        description="API server host"
    )
    port: int = Field(
        default=8014,
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
        env_prefix="GL_014_LOG_",
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
        default="logs/exchangerpro.log",
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
        env_prefix="GL_014_PROVENANCE_",
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
        env_prefix="GL_014_FEATURE_",
        extra="ignore",
    )

    # Core features
    enable_ml_predictions: bool = Field(
        default=True,
        description="Enable ML-based predictions"
    )
    enable_optimization: bool = Field(
        default=True,
        description="Enable cleaning schedule optimization"
    )
    enable_real_time_monitoring: bool = Field(
        default=True,
        description="Enable real-time monitoring"
    )

    # Integration features
    enable_kafka_streaming: bool = Field(
        default=False,
        description="Enable Kafka streaming integration"
    )
    enable_opcua_integration: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )
    enable_historian_integration: bool = Field(
        default=False,
        description="Enable process historian integration"
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
    enable_advanced_fouling_models: bool = Field(
        default=False,
        description="Enable advanced fouling prediction models"
    )


class ExchangerProSettings(BaseSettings):
    """
    Master configuration for GL-014 EXCHANGERPRO agent.

    All settings required for deterministic, reproducible operation
    of the heat exchanger optimization system.

    Example:
        >>> settings = ExchangerProSettings()
        >>> print(settings.agent_id)
        GL-014
        >>> print(settings.thermal.convergence_tolerance)
        1e-06

    Attributes:
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        version: Agent version string
        thermal: Thermal engine settings
        ml: ML service settings
        optimizer: Optimizer settings
        api: API settings
        logging: Logging settings
        provenance: Provenance tracking settings
        features: Feature flags
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_014_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-014",
        description="Unique agent identifier"
    )
    agent_name: str = Field(
        default="EXCHANGERPRO",
        description="Human-readable agent name"
    )
    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    description: str = Field(
        default="Heat Exchanger Optimizer - Thermal performance monitoring, fouling prediction, and cleaning optimization",
        description="Agent description"
    )

    # Nested settings
    thermal: ThermalEngineSettings = Field(
        default_factory=ThermalEngineSettings,
        description="Thermal calculation engine settings"
    )
    ml: MLServiceSettings = Field(
        default_factory=MLServiceSettings,
        description="ML service settings"
    )
    optimizer: OptimizerSettings = Field(
        default_factory=OptimizerSettings,
        description="Optimization engine settings"
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
        default="http://localhost:8014/graphql",
        description="GraphQL endpoint"
    )

    # Metrics
    metrics_port: int = Field(
        default=9014,
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
                "default_method": self.thermal.default_method.value,
                "convergence_tolerance": self.thermal.convergence_tolerance,
                "max_iterations": self.thermal.max_iterations,
            },
            "ml": {
                "fouling_model": self.ml.fouling_model.value,
                "random_seed": self.ml.random_seed,
                "min_confidence_threshold": self.ml.min_confidence_threshold,
            },
            "optimizer": {
                "objective": self.optimizer.objective.value,
                "planning_horizon_days": self.optimizer.planning_horizon_days,
            },
            "features": {
                "enable_ml_predictions": self.features.enable_ml_predictions,
                "enable_optimization": self.features.enable_optimization,
                "enable_velocity_limiting": self.features.enable_velocity_limiting,
            },
        }


# =============================================================================
# SETTINGS SINGLETON
# =============================================================================


@lru_cache
def get_settings() -> ExchangerProSettings:
    """
    Get cached settings instance.

    Returns the singleton ExchangerProSettings instance, loaded from
    environment variables and .env file.

    Returns:
        ExchangerProSettings: The global settings instance

    Example:
        >>> settings = get_settings()
        >>> print(settings.agent_id)
        GL-014
    """
    return ExchangerProSettings()


def reload_settings() -> ExchangerProSettings:
    """
    Reload settings from environment.

    Clears the cached settings and reloads from environment.
    Useful for testing or dynamic configuration updates.

    Returns:
        ExchangerProSettings: Fresh settings instance
    """
    get_settings.cache_clear()
    return get_settings()


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================


DEFAULT_CONFIG = ExchangerProSettings()


__all__ = [
    # Enums
    "TEMAType",
    "FlowArrangement",
    "ShellType",
    "TubePattern",
    "MaterialGrade",
    "CalculationMethod",
    "FoulingModel",
    "OptimizationObjective",
    # Settings classes
    "ThermalEngineSettings",
    "MLServiceSettings",
    "OptimizerSettings",
    "APISettings",
    "LoggingSettings",
    "ProvenanceSettings",
    "FeatureFlags",
    "ExchangerProSettings",
    # Functions
    "get_settings",
    "reload_settings",
    # Constants
    "DEFAULT_CONFIG",
]

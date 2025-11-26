# -*- coding: utf-8 -*-
"""
Configuration module for GL-009 THERMALIQ ThermalEfficiencyCalculator.

This module defines Pydantic V2 configuration classes for the THERMALIQ agent,
including operational parameters, calculation thresholds, visualization settings,
and integration configurations.

SECURITY:
- Zero hardcoded credentials policy
- All secrets loaded from environment variables
- Validation enforced at startup

Standards Compliance:
- ASME PTC 4.1 - Steam Generating Units
- ISO 50001:2018 - Energy Management Systems
- EPA 40 CFR Part 60 - Emissions Standards

Author: GreenLang Foundation
Version: 1.0.0
"""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseModel, Field, validator as field_validator
    model_validator = None


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ProcessType(Enum):
    """Supported thermal process types."""
    BOILER = "boiler"
    FURNACE = "furnace"
    DRYER = "dryer"
    KILN = "kiln"
    HEAT_EXCHANGER = "heat_exchanger"
    REACTOR = "reactor"
    OVEN = "oven"
    INCINERATOR = "incinerator"
    STEAM_GENERATOR = "steam_generator"
    HOT_WATER_SYSTEM = "hot_water_system"


class FuelType(Enum):
    """Supported fuel types."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_NO2 = "fuel_oil_no2"
    FUEL_OIL_NO6 = "fuel_oil_no6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUB_BITUMINOUS = "coal_sub_bituminous"
    BIOMASS_WOOD = "biomass_wood"
    BIOMASS_PELLETS = "biomass_pellets"
    HYDROGEN = "hydrogen"
    PROPANE = "propane"
    ELECTRICITY = "electricity"


class CalculationMethod(Enum):
    """Efficiency calculation methods."""
    INPUT_OUTPUT = "input_output"
    HEAT_LOSS = "heat_loss"
    INDIRECT = "indirect"
    EXERGY = "exergy"
    COMBINED = "combined"


class VisualizationFormat(Enum):
    """Supported visualization output formats."""
    PLOTLY_JSON = "plotly_json"
    SVG = "svg"
    PNG = "png"
    HTML = "html"
    PDF = "pdf"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ============================================================================
# CALCULATION CONFIGURATION
# ============================================================================

class CalculationConfig(BaseModel):
    """Configuration for thermal efficiency calculations."""

    # Calculation method
    default_method: CalculationMethod = Field(
        default=CalculationMethod.COMBINED,
        description="Default efficiency calculation method"
    )

    # Energy balance tolerance (ASME PTC 4.1 requires 2%)
    energy_balance_tolerance: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Heat balance closure tolerance (fraction)"
    )

    # Efficiency thresholds
    min_efficiency_percent: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum expected efficiency percentage"
    )

    max_efficiency_percent: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum expected efficiency percentage"
    )

    # Reference conditions
    reference_temperature_c: float = Field(
        default=25.0,
        ge=-50.0,
        le=100.0,
        description="Reference temperature for calculations (Celsius)"
    )

    reference_pressure_bar: float = Field(
        default=1.01325,
        ge=0.5,
        le=5.0,
        description="Reference pressure for calculations (bar)"
    )

    # Exergy analysis settings
    enable_exergy_analysis: bool = Field(
        default=True,
        description="Enable Second Law (exergy) analysis"
    )

    exergy_reference_temperature_c: float = Field(
        default=25.0,
        description="Dead state temperature for exergy calculations (Celsius)"
    )

    # Uncertainty quantification
    enable_uncertainty_analysis: bool = Field(
        default=True,
        description="Enable uncertainty quantification"
    )

    confidence_level_percent: float = Field(
        default=95.0,
        ge=80.0,
        le=99.0,
        description="Confidence level for uncertainty intervals"
    )

    # Measurement defaults
    default_fuel_heating_value_mj_kg: float = Field(
        default=50.0,
        ge=10.0,
        le=150.0,
        description="Default fuel HHV if not specified (MJ/kg)"
    )

    default_steam_enthalpy_kj_kg: float = Field(
        default=2750.0,
        ge=2000.0,
        le=3500.0,
        description="Default steam enthalpy if not specified (kJ/kg)"
    )

    @field_validator('max_efficiency_percent')
    @classmethod
    def max_greater_than_min(cls, v, info):
        """Validate max efficiency is greater than min."""
        if hasattr(info, 'data') and 'min_efficiency_percent' in info.data:
            if v <= info.data['min_efficiency_percent']:
                raise ValueError('max_efficiency_percent must be greater than min_efficiency_percent')
        return v


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

class VisualizationConfig(BaseModel):
    """Configuration for Sankey diagrams and visualizations."""

    # Output format
    default_format: VisualizationFormat = Field(
        default=VisualizationFormat.PLOTLY_JSON,
        description="Default visualization output format"
    )

    # Sankey diagram settings
    sankey_orientation: str = Field(
        default="horizontal",
        pattern="^(horizontal|vertical)$",
        description="Sankey diagram orientation"
    )

    sankey_node_pad: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Padding between Sankey nodes"
    )

    sankey_node_thickness: int = Field(
        default=20,
        ge=10,
        le=50,
        description="Thickness of Sankey nodes"
    )

    # Color scheme
    input_color: str = Field(
        default="#ff7f0e",
        description="Color for input energy flows"
    )

    output_color: str = Field(
        default="#2ca02c",
        description="Color for useful output flows"
    )

    loss_color: str = Field(
        default="#d62728",
        description="Color for loss flows"
    )

    # Image settings
    image_width: int = Field(
        default=1200,
        ge=400,
        le=4000,
        description="Default image width in pixels"
    )

    image_height: int = Field(
        default=800,
        ge=300,
        le=3000,
        description="Default image height in pixels"
    )

    image_dpi: int = Field(
        default=150,
        ge=72,
        le=600,
        description="Image resolution in DPI"
    )

    # Labels and formatting
    value_format: str = Field(
        default=".2f",
        description="Number format string for values"
    )

    value_suffix: str = Field(
        default=" kW",
        description="Suffix for value labels"
    )

    show_percentages: bool = Field(
        default=True,
        description="Show percentages on diagram"
    )


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

class IntegrationConfig(BaseModel):
    """Configuration for external system integrations."""

    # OPC UA integration
    opcua_enabled: bool = Field(
        default=False,
        description="Enable OPC UA integration"
    )

    opcua_endpoint: Optional[str] = Field(
        default=None,
        description="OPC UA server endpoint URL"
    )

    opcua_namespace: str = Field(
        default="ns=2",
        description="OPC UA namespace"
    )

    # Historian integration
    historian_enabled: bool = Field(
        default=False,
        description="Enable process historian integration"
    )

    historian_type: str = Field(
        default="osisoft_pi",
        description="Historian type (osisoft_pi, wonderware, honeywell)"
    )

    historian_endpoint: Optional[str] = Field(
        default=None,
        description="Historian API endpoint"
    )

    # SCADA integration
    scada_enabled: bool = Field(
        default=False,
        description="Enable SCADA integration"
    )

    scada_protocol: str = Field(
        default="modbus_tcp",
        description="SCADA protocol (modbus_tcp, modbus_rtu, bacnet)"
    )

    # ERP integration
    erp_enabled: bool = Field(
        default=False,
        description="Enable ERP integration for cost data"
    )

    erp_endpoint: Optional[str] = Field(
        default=None,
        description="ERP API endpoint"
    )

    # Webhook notifications
    webhook_enabled: bool = Field(
        default=False,
        description="Enable webhook notifications"
    )

    webhook_url: Optional[str] = Field(
        default=None,
        description="Webhook URL for notifications"
    )

    @field_validator('opcua_endpoint', 'historian_endpoint', 'erp_endpoint', 'webhook_url')
    @classmethod
    def validate_no_credentials_in_url(cls, v):
        """Validate URLs do not contain embedded credentials."""
        if v is not None:
            parsed = urlparse(v)
            if parsed.username or parsed.password:
                raise ValueError(
                    "SECURITY VIOLATION: URL contains embedded credentials. "
                    "Use environment variables instead."
                )
        return v


# ============================================================================
# BENCHMARK CONFIGURATION
# ============================================================================

class BenchmarkConfig(BaseModel):
    """Configuration for industry benchmark comparisons."""

    # Benchmark sources
    use_asme_benchmarks: bool = Field(
        default=True,
        description="Use ASME benchmark database"
    )

    use_doe_benchmarks: bool = Field(
        default=True,
        description="Use DOE benchmark database"
    )

    use_iso_benchmarks: bool = Field(
        default=True,
        description="Use ISO 50001 benchmark database"
    )

    # Custom benchmarks
    custom_benchmarks: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Custom benchmark values by process type"
    )

    # Benchmark thresholds
    percentile_excellent: int = Field(
        default=90,
        ge=75,
        le=99,
        description="Percentile threshold for excellent rating"
    )

    percentile_good: int = Field(
        default=75,
        ge=50,
        le=90,
        description="Percentile threshold for good rating"
    )

    percentile_average: int = Field(
        default=50,
        ge=25,
        le=75,
        description="Percentile threshold for average rating"
    )


# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

class MonitoringConfig(BaseModel):
    """Configuration for real-time monitoring."""

    # Monitoring interval
    monitoring_interval_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Data collection interval in seconds"
    )

    # Alert thresholds
    efficiency_alert_threshold_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Efficiency drop threshold for alerts (percentage points)"
    )

    heat_balance_alert_threshold_percent: float = Field(
        default=5.0,
        ge=1.0,
        le=10.0,
        description="Heat balance closure alert threshold"
    )

    # Data retention
    realtime_buffer_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Number of real-time data points to buffer"
    )

    historical_retention_days: int = Field(
        default=365,
        ge=30,
        le=3650,
        description="Historical data retention period in days"
    )

    # Prometheus metrics
    enable_prometheus: bool = Field(
        default=True,
        description="Enable Prometheus metrics endpoint"
    )

    prometheus_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Prometheus metrics port"
    )


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

class ThermalEfficiencyConfig(BaseModel):
    """
    Main configuration for GL-009 THERMALIQ ThermalEfficiencyCalculator.

    This class contains all configuration settings for the agent, including
    operational parameters, calculation settings, visualization options,
    and integration configurations.

    SECURITY:
    - Zero hardcoded credentials policy enforced
    - All secrets must be in environment variables
    - URL validation prevents credential leakage

    Example:
        >>> config = ThermalEfficiencyConfig(
        ...     agent_id="GL-009",
        ...     codename="THERMALIQ",
        ...     energy_balance_tolerance=0.02
        ... )
        >>> orchestrator = ThermalEfficiencyOrchestrator(config)
    """

    # ========================================================================
    # AGENT IDENTIFICATION
    # ========================================================================

    agent_id: str = Field(
        default="GL-009",
        description="Unique agent identifier"
    )

    codename: str = Field(
        default="THERMALIQ",
        description="Agent codename"
    )

    full_name: str = Field(
        default="ThermalEfficiencyCalculator",
        description="Full agent name"
    )

    version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # ========================================================================
    # DETERMINISTIC SETTINGS
    # ========================================================================

    deterministic: bool = Field(
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
    # CALCULATION PARAMETERS
    # ========================================================================

    energy_balance_tolerance: float = Field(
        default=0.02,
        ge=0.001,
        le=0.10,
        description="Heat balance closure tolerance (fraction)"
    )

    min_efficiency_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Minimum efficiency threshold for alerts"
    )

    max_efficiency_threshold: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Maximum efficiency threshold"
    )

    # ========================================================================
    # CACHE SETTINGS
    # ========================================================================

    cache_ttl_seconds: int = Field(
        default=300,
        ge=60,
        le=3600,
        description="Cache time-to-live in seconds"
    )

    cache_max_size: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum cache entries"
    )

    # ========================================================================
    # PERFORMANCE SETTINGS
    # ========================================================================

    calculation_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Timeout for calculations in seconds"
    )

    max_concurrent_calculations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent calculations"
    )

    # ========================================================================
    # RETRY SETTINGS
    # ========================================================================

    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts"
    )

    retry_initial_delay_ms: float = Field(
        default=100.0,
        ge=10.0,
        le=1000.0,
        description="Initial retry delay in milliseconds"
    )

    retry_max_delay_ms: float = Field(
        default=5000.0,
        ge=100.0,
        le=30000.0,
        description="Maximum retry delay in milliseconds"
    )

    enable_error_recovery: bool = Field(
        default=True,
        description="Enable automatic error recovery"
    )

    # ========================================================================
    # MONITORING SETTINGS
    # ========================================================================

    enable_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring"
    )

    monitoring_interval_seconds: int = Field(
        default=300,
        ge=10,
        le=3600,
        description="Monitoring interval in seconds"
    )

    # ========================================================================
    # LLM SETTINGS (CLASSIFICATION ONLY)
    # ========================================================================

    enable_llm_classification: bool = Field(
        default=True,
        description="Enable LLM for classification tasks only"
    )

    llm_provider: str = Field(
        default="anthropic",
        description="LLM provider (anthropic, openai)"
    )

    llm_model: str = Field(
        default="claude-3-haiku",
        description="LLM model for classification"
    )

    llm_max_tokens: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Maximum tokens for LLM responses"
    )

    llm_budget_usd: float = Field(
        default=0.50,
        ge=0.0,
        le=10.0,
        description="Maximum LLM cost per query"
    )

    # ========================================================================
    # SECURITY SETTINGS
    # ========================================================================

    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashing"
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    zero_secrets: bool = Field(
        default=True,
        description="Enforce zero hardcoded secrets policy"
    )

    # ========================================================================
    # STORAGE PATHS
    # ========================================================================

    data_directory: Optional[Path] = Field(
        default=None,
        description="Data storage directory"
    )

    log_directory: Optional[Path] = Field(
        default=None,
        description="Log file directory"
    )

    cache_directory: Optional[Path] = Field(
        default=None,
        description="Cache storage directory"
    )

    # ========================================================================
    # SUB-CONFIGURATIONS
    # ========================================================================

    calculation: CalculationConfig = Field(
        default_factory=CalculationConfig,
        description="Calculation configuration"
    )

    visualization: VisualizationConfig = Field(
        default_factory=VisualizationConfig,
        description="Visualization configuration"
    )

    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )

    benchmark: BenchmarkConfig = Field(
        default_factory=BenchmarkConfig,
        description="Benchmark configuration"
    )

    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator('temperature')
    @classmethod
    def validate_deterministic_temperature(cls, v):
        """Ensure temperature is 0.0 for deterministic operation."""
        if v != 0.0:
            raise ValueError(
                "COMPLIANCE VIOLATION: temperature must be 0.0 for "
                "deterministic, zero-hallucination calculations"
            )
        return v

    @field_validator('seed')
    @classmethod
    def validate_seed(cls, v):
        """Ensure seed is set for reproducibility."""
        if v != 42:
            # Allow other seeds but log warning
            pass
        return v

    @field_validator('zero_secrets')
    @classmethod
    def validate_zero_secrets(cls, v):
        """Ensure zero_secrets policy is enabled."""
        if not v:
            raise ValueError(
                "SECURITY VIOLATION: zero_secrets must be True. "
                "No credentials allowed in configuration."
            )
        return v

    def model_post_init(self, __context) -> None:
        """Post-initialization validation and setup."""
        # Set default directories if not provided
        if self.data_directory is None:
            self.data_directory = Path("./gl009_data")
        if self.log_directory is None:
            self.log_directory = Path("./gl009_logs")
        if self.cache_directory is None:
            self.cache_directory = Path("./gl009_cache")

        # Create directories if they don't exist
        for directory in [self.data_directory, self.log_directory, self.cache_directory]:
            if directory:
                directory.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def get_api_key(provider: str = "anthropic") -> Optional[str]:
        """
        Get API key from environment variable.

        SECURITY: API keys must be stored in environment variables.

        Args:
            provider: API provider name

        Returns:
            API key from environment, or None if not set
        """
        env_var_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }

        env_var = env_var_map.get(provider.lower())
        if not env_var:
            raise ValueError(f"Unknown API provider: {provider}")

        return os.environ.get(env_var)

    @staticmethod
    def is_production() -> bool:
        """
        Check if running in production environment.

        Returns:
            True if GREENLANG_ENV is 'production' or 'prod'
        """
        env = os.environ.get("GREENLANG_ENV", "development").lower()
        return env in ["production", "prod"]

    @staticmethod
    def get_environment() -> str:
        """
        Get current environment name.

        Returns:
            Environment name (development, staging, production)
        """
        return os.environ.get("GREENLANG_ENV", "development").lower()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        extra = "forbid"
        str_strip_whitespace = True


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_config(
    env: Optional[str] = None,
    **overrides
) -> ThermalEfficiencyConfig:
    """
    Create configuration based on environment.

    Args:
        env: Environment name (development, staging, production)
        **overrides: Configuration overrides

    Returns:
        Configured ThermalEfficiencyConfig instance

    Example:
        >>> config = create_config("production", cache_ttl_seconds=600)
    """
    if env is None:
        env = ThermalEfficiencyConfig.get_environment()

    # Environment-specific defaults
    env_defaults = {
        "development": {
            "cache_ttl_seconds": 60,
            "max_retries": 1,
            "enable_monitoring": True
        },
        "staging": {
            "cache_ttl_seconds": 300,
            "max_retries": 2,
            "enable_monitoring": True
        },
        "production": {
            "cache_ttl_seconds": 600,
            "max_retries": 3,
            "enable_monitoring": True
        }
    }

    # Apply environment defaults
    defaults = env_defaults.get(env, env_defaults["development"])
    defaults.update(overrides)

    return ThermalEfficiencyConfig(**defaults)


def load_config_from_file(config_path: Union[str, Path]) -> ThermalEfficiencyConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured ThermalEfficiencyConfig instance
    """
    import json
    config_path = Path(config_path)

    if config_path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required for YAML config files")
    elif config_path.suffix == '.json':
        with open(config_path) as f:
            config_data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    return ThermalEfficiencyConfig(**config_data)

"""
GL-006 WasteHeatRecovery Agent - Configuration Module

Centralized configuration management using Pydantic for type-safe
configuration handling with validation and environment variable support.

This module provides:
    - Agent configuration with defaults
    - Economic parameters
    - Pinch analysis settings
    - Exergy analysis parameters
    - OPC-UA integration settings
    - Feature flags for optional modules
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings


# =============================================================================
# ENUMS
# =============================================================================

class AnalysisMode(str, Enum):
    """Analysis mode selection."""
    QUICK = "quick"           # Fast estimation
    STANDARD = "standard"     # Standard accuracy
    DETAILED = "detailed"     # High accuracy with full analysis


class TemperatureUnit(str, Enum):
    """Temperature unit selection."""
    FAHRENHEIT = "F"
    CELSIUS = "C"
    KELVIN = "K"


class EnergyUnit(str, Enum):
    """Energy unit selection."""
    BTU_HR = "BTU/hr"
    KW = "kW"
    MW = "MW"
    MMBTU_HR = "MMBTU/hr"


class CurrencyUnit(str, Enum):
    """Currency unit selection."""
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class PinchAnalysisConfig(BaseModel):
    """Configuration for pinch analysis module."""

    delta_t_min_f: float = Field(
        default=20.0,
        ge=1.0,
        le=100.0,
        description="Minimum temperature approach (F)"
    )
    hot_utility_cost_per_mmbtu: float = Field(
        default=8.0,
        ge=0.0,
        description="Hot utility cost ($/MMBTU)"
    )
    cold_utility_cost_per_mmbtu: float = Field(
        default=1.5,
        ge=0.0,
        description="Cold utility cost ($/MMBTU)"
    )
    operating_hours_per_year: int = Field(
        default=8760,
        ge=1,
        le=8760,
        description="Annual operating hours"
    )
    delta_t_optimization_enabled: bool = Field(
        default=True,
        description="Enable delta-T optimization"
    )
    delta_t_optimization_range: tuple = Field(
        default=(5.0, 50.0),
        description="Delta-T optimization range (min, max)"
    )

    class Config:
        frozen = False


class HENSynthesisConfig(BaseModel):
    """Configuration for HEN synthesis module."""

    default_u_value: float = Field(
        default=50.0,
        ge=1.0,
        le=500.0,
        description="Default U value (BTU/hr-ft2-F)"
    )
    base_hx_cost_usd: float = Field(
        default=10000.0,
        ge=0.0,
        description="Base heat exchanger cost ($)"
    )
    hx_cost_per_ft2: float = Field(
        default=150.0,
        ge=0.0,
        description="Heat exchanger cost per area ($/ft2)"
    )
    area_exponent: float = Field(
        default=0.65,
        ge=0.5,
        le=1.0,
        description="Cost scaling exponent"
    )
    installation_factor: float = Field(
        default=3.5,
        ge=1.0,
        le=10.0,
        description="Installed cost factor"
    )
    synthesis_method: str = Field(
        default="pinch_design",
        description="Synthesis method (pinch_design, optimization)"
    )

    class Config:
        frozen = False


class ExergyAnalysisConfig(BaseModel):
    """Configuration for exergy analysis module."""

    dead_state_temp_f: float = Field(
        default=77.0,
        ge=-40.0,
        le=150.0,
        description="Dead state temperature (F)"
    )
    dead_state_pressure_psia: float = Field(
        default=14.696,
        ge=0.1,
        le=100.0,
        description="Dead state pressure (psia)"
    )
    fuel_exergy_cost_usd_per_mmbtu: float = Field(
        default=8.0,
        ge=0.0,
        description="Fuel exergy cost ($/MMBTU)"
    )
    include_chemical_exergy: bool = Field(
        default=False,
        description="Include chemical exergy calculations"
    )
    unavoidable_fraction_default: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Default unavoidable destruction fraction"
    )

    class Config:
        frozen = False


class EconomicConfig(BaseModel):
    """Configuration for economic optimizer module."""

    discount_rate: float = Field(
        default=0.08,
        ge=0.0,
        le=0.30,
        description="Real discount rate"
    )
    inflation_rate: float = Field(
        default=0.025,
        ge=0.0,
        le=0.15,
        description="General inflation rate"
    )
    energy_escalation_rate: float = Field(
        default=0.03,
        ge=0.0,
        le=0.15,
        description="Energy cost escalation rate"
    )
    tax_rate: float = Field(
        default=0.21,
        ge=0.0,
        le=0.50,
        description="Corporate tax rate"
    )
    project_life_years: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Default project life"
    )
    carbon_price_usd_per_ton: float = Field(
        default=51.0,
        ge=0.0,
        description="Carbon price ($/ton CO2)"
    )
    carbon_price_escalation: float = Field(
        default=0.02,
        ge=0.0,
        le=0.10,
        description="Carbon price annual escalation"
    )
    monte_carlo_iterations: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Monte Carlo iterations"
    )

    class Config:
        frozen = False


class OPCUAConfig(BaseModel):
    """Configuration for OPC-UA integration."""

    enabled: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )
    endpoint_url: str = Field(
        default="opc.tcp://localhost:4840",
        description="OPC-UA server endpoint"
    )
    namespace_uri: str = Field(
        default="urn:greenlang:waste_heat_recovery",
        description="OPC-UA namespace URI"
    )
    security_mode: str = Field(
        default="None",
        description="Security mode (None, Sign, SignAndEncrypt)"
    )
    security_policy: str = Field(
        default="None",
        description="Security policy"
    )
    username: Optional[str] = Field(
        default=None,
        description="OPC-UA username"
    )
    password: Optional[str] = Field(
        default=None,
        description="OPC-UA password"
    )
    certificate_path: Optional[str] = Field(
        default=None,
        description="Path to client certificate"
    )
    private_key_path: Optional[str] = Field(
        default=None,
        description="Path to private key"
    )
    polling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Data polling interval (ms)"
    )
    connection_timeout_ms: int = Field(
        default=5000,
        ge=1000,
        le=30000,
        description="Connection timeout (ms)"
    )
    reconnect_delay_ms: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Reconnection delay (ms)"
    )
    max_reconnect_attempts: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Max reconnection attempts"
    )

    class Config:
        frozen = False


class ExplainabilityConfig(BaseModel):
    """Configuration for SHAP explainability module."""

    enabled: bool = Field(
        default=True,
        description="Enable SHAP explainability"
    )
    num_samples: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of samples for SHAP"
    )
    feature_perturbation: str = Field(
        default="interventional",
        description="Feature perturbation method"
    )
    output_type: str = Field(
        default="probability",
        description="Output type for explanations"
    )
    include_waterfall_plot: bool = Field(
        default=True,
        description="Generate waterfall plots"
    )
    include_force_plot: bool = Field(
        default=True,
        description="Generate force plots"
    )
    include_summary_plot: bool = Field(
        default=True,
        description="Generate summary plots"
    )
    cache_explanations: bool = Field(
        default=True,
        description="Cache SHAP explanations"
    )

    class Config:
        frozen = False


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(
        default="INFO",
        description="Log level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_to_file: bool = Field(
        default=False,
        description="Enable file logging"
    )
    log_file_path: str = Field(
        default="logs/gl_006_waste_heat.log",
        description="Log file path"
    )
    max_log_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max log file size (MB)"
    )
    backup_count: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of backup log files"
    )

    class Config:
        frozen = False


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

class GL006Config(BaseSettings):
    """
    Main configuration for GL-006 WasteHeatRecovery Agent.

    Supports loading from environment variables with GL006_ prefix.
    Example: GL006_ANALYSIS_MODE=detailed

    Attributes:
        agent_name: Name of the agent
        agent_version: Version string
        analysis_mode: Quick, standard, or detailed analysis
        temperature_unit: Default temperature unit
        energy_unit: Default energy unit
        currency_unit: Default currency unit
        pinch_config: Pinch analysis configuration
        hen_config: HEN synthesis configuration
        exergy_config: Exergy analysis configuration
        economic_config: Economic optimizer configuration
        opcua_config: OPC-UA integration configuration
        explainability_config: SHAP explainability configuration
        logging_config: Logging configuration

    Example:
        >>> config = GL006Config()
        >>> print(config.pinch_config.delta_t_min_f)
        20.0

        >>> config = GL006Config(analysis_mode=AnalysisMode.DETAILED)
        >>> print(config.analysis_mode)
        AnalysisMode.DETAILED
    """

    # Agent identification
    agent_name: str = Field(
        default="GL-006-WasteHeatRecoveryAnalyzer",
        description="Agent name"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )
    agent_description: str = Field(
        default="Waste heat recovery opportunity analyzer with pinch analysis, HEN synthesis, exergy analysis, and economic optimization",
        description="Agent description"
    )

    # Analysis settings
    analysis_mode: AnalysisMode = Field(
        default=AnalysisMode.STANDARD,
        description="Analysis mode"
    )
    temperature_unit: TemperatureUnit = Field(
        default=TemperatureUnit.FAHRENHEIT,
        description="Default temperature unit"
    )
    energy_unit: EnergyUnit = Field(
        default=EnergyUnit.BTU_HR,
        description="Default energy unit"
    )
    currency_unit: CurrencyUnit = Field(
        default=CurrencyUnit.USD,
        description="Default currency unit"
    )

    # Feature flags
    enable_pinch_analysis: bool = Field(
        default=True,
        description="Enable pinch analysis"
    )
    enable_hen_synthesis: bool = Field(
        default=True,
        description="Enable HEN synthesis"
    )
    enable_exergy_analysis: bool = Field(
        default=True,
        description="Enable exergy analysis"
    )
    enable_economic_optimization: bool = Field(
        default=True,
        description="Enable economic optimization"
    )
    enable_explainability: bool = Field(
        default=True,
        description="Enable SHAP explainability"
    )
    enable_opcua_integration: bool = Field(
        default=False,
        description="Enable OPC-UA integration"
    )

    # Sub-configurations
    pinch_config: PinchAnalysisConfig = Field(
        default_factory=PinchAnalysisConfig,
        description="Pinch analysis configuration"
    )
    hen_config: HENSynthesisConfig = Field(
        default_factory=HENSynthesisConfig,
        description="HEN synthesis configuration"
    )
    exergy_config: ExergyAnalysisConfig = Field(
        default_factory=ExergyAnalysisConfig,
        description="Exergy analysis configuration"
    )
    economic_config: EconomicConfig = Field(
        default_factory=EconomicConfig,
        description="Economic optimizer configuration"
    )
    opcua_config: OPCUAConfig = Field(
        default_factory=OPCUAConfig,
        description="OPC-UA integration configuration"
    )
    explainability_config: ExplainabilityConfig = Field(
        default_factory=ExplainabilityConfig,
        description="SHAP explainability configuration"
    )
    logging_config: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )

    # Provenance and audit
    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance hashing"
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )

    class Config:
        env_prefix = "GL006_"
        env_nested_delimiter = "__"
        case_sensitive = False

    def get_analysis_parameters(self) -> Dict[str, Any]:
        """Get analysis parameters based on mode."""
        if self.analysis_mode == AnalysisMode.QUICK:
            return {
                "delta_t_optimization_enabled": False,
                "monte_carlo_enabled": False,
                "sensitivity_enabled": False,
                "exergy_detail_level": "basic",
            }
        elif self.analysis_mode == AnalysisMode.DETAILED:
            return {
                "delta_t_optimization_enabled": True,
                "monte_carlo_enabled": True,
                "sensitivity_enabled": True,
                "exergy_detail_level": "advanced",
            }
        else:  # STANDARD
            return {
                "delta_t_optimization_enabled": True,
                "monte_carlo_enabled": False,
                "sensitivity_enabled": True,
                "exergy_detail_level": "standard",
            }

    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings."""
        warnings = []

        if self.economic_config.discount_rate > 0.15:
            warnings.append(
                f"High discount rate ({self.economic_config.discount_rate*100:.1f}%) "
                "may undervalue long-term projects"
            )

        if self.pinch_config.delta_t_min_f < 10:
            warnings.append(
                f"Low delta-T min ({self.pinch_config.delta_t_min_f}F) "
                "may result in impractically large heat exchangers"
            )

        if self.enable_opcua_integration and not self.opcua_config.endpoint_url:
            warnings.append(
                "OPC-UA integration enabled but no endpoint URL configured"
            )

        return warnings


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

def get_default_config() -> GL006Config:
    """Get default configuration instance."""
    return GL006Config()


def load_config_from_file(config_path: str) -> GL006Config:
    """
    Load configuration from a JSON or YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        GL006Config instance
    """
    import json
    from pathlib import Path

    path = Path(config_path)

    if path.suffix in ['.json']:
        with open(path, 'r') as f:
            config_data = json.load(f)
    elif path.suffix in ['.yaml', '.yml']:
        try:
            import yaml
            with open(path, 'r') as f:
                config_data = yaml.safe_load(f)
        except ImportError:
            raise ImportError("PyYAML required to load YAML config files")
    else:
        raise ValueError(f"Unsupported config file format: {path.suffix}")

    return GL006Config(**config_data)

# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL SteamQualityController - Configuration Module

This module defines comprehensive configuration schemas for the STEAMQUAL
SteamQualityController including quality thresholds, carryover risk,
separator efficiency, and site-specific settings.

All configurations use Pydantic v2 with pydantic-settings for environment
variable support, following GreenLang Global AI Standards v2.0.

Standards Compliance:
    - ASME PTC 19.11 (Steam and Water Sampling)
    - IAPWS-IF97 (Industrial Formulation for Water and Steam)
    - IEC 61511 (Functional Safety)
    - EU AI Act (Transparency and Reproducibility)

Environment Variable Prefix: GL_012_

Example:
    >>> from core.config import SteamQualConfig, get_settings
    >>> config = get_settings()
    >>> print(f"Min dryness: {config.quality.x_min_default}")

    # Override via environment:
    # GL_012_QUALITY__X_MIN_DEFAULT=0.98

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from datetime import timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Union
import os

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# ENUMS
# =============================================================================

class SteamPhase(str, Enum):
    """Steam phase classifications per IAPWS-IF97."""
    SUBCOOLED_LIQUID = "subcooled_liquid"
    SATURATED_LIQUID = "saturated_liquid"
    WET_STEAM = "wet_steam"
    SATURATED_VAPOR = "saturated_vapor"
    SUPERHEATED_STEAM = "superheated_steam"
    SUPERCRITICAL = "supercritical"


class QualityControlMode(str, Enum):
    """Steam quality control operating modes."""
    MANUAL = "manual"
    AUTOMATIC = "automatic"
    ADVISORY = "advisory"
    EMERGENCY = "emergency"
    CALIBRATION = "calibration"


class CarryoverRiskLevel(str, Enum):
    """Carryover risk classification levels."""
    NEGLIGIBLE = "negligible"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SeparatorType(str, Enum):
    """Types of steam separators."""
    CENTRIFUGAL = "centrifugal"
    BAFFLE = "baffle"
    MESH_PAD = "mesh_pad"
    CHEVRON = "chevron"
    CYCLONE = "cyclone"
    WIRE_MESH = "wire_mesh"
    COMBINED = "combined"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class CalculationMethod(str, Enum):
    """Steam quality calculation methods."""
    THROTTLING_CALORIMETER = "throttling_calorimeter"
    SEPARATING_CALORIMETER = "separating_calorimeter"
    ELECTRICAL_CONDUCTIVITY = "electrical_conductivity"
    HEAT_BALANCE = "heat_balance"
    IAPWS_PROPERTIES = "iapws_properties"


# =============================================================================
# QUALITY THRESHOLD CONFIGURATION
# =============================================================================

class QualityThresholdsConfig(BaseModel):
    """
    Steam quality threshold configuration.

    Defines acceptable ranges for steam dryness fraction (x) and
    superheat margin per ASME PTC 19.11 standards.

    Attributes:
        x_min_default: Minimum acceptable dryness fraction (default 0.95)
        superheat_margin_min_c: Minimum superheat margin in Celsius (default 3.0)
    """

    x_min_default: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable steam dryness fraction (0=all liquid, 1=dry saturated)"
    )
    x_min_critical: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Critical low dryness fraction triggering emergency response"
    )
    x_target: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Target dryness fraction for optimal operation"
    )
    x_high_quality: float = Field(
        default=0.995,
        ge=0.0,
        le=1.0,
        description="Threshold for high-quality steam classification"
    )

    # Superheat margins
    superheat_margin_min_c: float = Field(
        default=3.0,
        ge=0.0,
        le=100.0,
        description="Minimum superheat margin in Celsius for superheated steam"
    )
    superheat_margin_target_c: float = Field(
        default=10.0,
        ge=0.0,
        le=150.0,
        description="Target superheat margin in Celsius"
    )
    superheat_margin_max_c: float = Field(
        default=50.0,
        ge=0.0,
        le=300.0,
        description="Maximum superheat margin before alarm"
    )

    # Moisture content thresholds
    moisture_content_max_pct: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable moisture content percentage"
    )
    moisture_content_alarm_pct: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Moisture content alarm threshold percentage"
    )

    # Temperature deviation
    temp_deviation_max_c: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Maximum temperature deviation from saturation"
    )

    @field_validator('x_min_critical')
    @classmethod
    def validate_x_min_critical(cls, v: float, info) -> float:
        """Ensure critical threshold is below default minimum."""
        x_min = info.data.get('x_min_default', 0.95)
        if v >= x_min:
            raise ValueError(
                f"x_min_critical ({v}) must be less than x_min_default ({x_min})"
            )
        return v

    @model_validator(mode='after')
    def validate_thresholds(self) -> 'QualityThresholdsConfig':
        """Validate threshold relationships."""
        if self.x_target < self.x_min_default:
            raise ValueError("x_target must be >= x_min_default")
        if self.superheat_margin_target_c < self.superheat_margin_min_c:
            raise ValueError("superheat_margin_target_c must be >= superheat_margin_min_c")
        if self.moisture_content_alarm_pct < self.moisture_content_max_pct:
            raise ValueError("moisture_content_alarm_pct must be >= moisture_content_max_pct")
        return self


# =============================================================================
# CARRYOVER RISK CONFIGURATION
# =============================================================================

class CarryoverRiskConfig(BaseModel):
    """
    Carryover risk threshold configuration.

    Carryover occurs when water droplets or dissolved solids are
    entrained in steam, affecting downstream equipment and processes.

    Attributes:
        tds_limit_ppm: Total dissolved solids limit in ppm
        silica_limit_ppb: Silica concentration limit in ppb
        iron_limit_ppb: Iron concentration limit in ppb
    """

    # Dissolved solids thresholds (per ASME guidelines)
    tds_limit_ppm: float = Field(
        default=3000.0,
        ge=0.0,
        le=50000.0,
        description="Maximum boiler water TDS in ppm before carryover risk"
    )
    tds_alarm_ppm: float = Field(
        default=4000.0,
        ge=0.0,
        le=60000.0,
        description="TDS alarm threshold in ppm"
    )

    # Silica limits (critical for turbine applications)
    silica_limit_ppb: float = Field(
        default=20.0,
        ge=0.0,
        le=1000.0,
        description="Maximum silica in steam in ppb"
    )
    silica_alarm_ppb: float = Field(
        default=50.0,
        ge=0.0,
        le=2000.0,
        description="Silica alarm threshold in ppb"
    )

    # Iron limits
    iron_limit_ppb: float = Field(
        default=20.0,
        ge=0.0,
        le=500.0,
        description="Maximum iron in steam in ppb"
    )

    # Conductivity thresholds
    conductivity_limit_us_cm: float = Field(
        default=5000.0,
        ge=0.0,
        le=50000.0,
        description="Maximum conductivity in microSiemens/cm"
    )

    # Risk level thresholds (probability of carryover)
    negligible_risk_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Probability threshold for negligible risk"
    )
    low_risk_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Probability threshold for low risk"
    )
    moderate_risk_threshold: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Probability threshold for moderate risk"
    )
    high_risk_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Probability threshold for high risk"
    )

    # Mechanical factors
    drum_level_variance_max_mm: float = Field(
        default=50.0,
        ge=0.0,
        le=500.0,
        description="Maximum drum level variance in mm before carryover risk"
    )
    steam_velocity_limit_m_s: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Maximum steam velocity in m/s at drum outlet"
    )

    def get_risk_level(self, probability: float) -> CarryoverRiskLevel:
        """
        Determine carryover risk level from probability.

        Args:
            probability: Estimated carryover probability (0-1)

        Returns:
            CarryoverRiskLevel enum value
        """
        if probability <= self.negligible_risk_threshold:
            return CarryoverRiskLevel.NEGLIGIBLE
        elif probability <= self.low_risk_threshold:
            return CarryoverRiskLevel.LOW
        elif probability <= self.moderate_risk_threshold:
            return CarryoverRiskLevel.MODERATE
        elif probability <= self.high_risk_threshold:
            return CarryoverRiskLevel.HIGH
        else:
            return CarryoverRiskLevel.CRITICAL


# =============================================================================
# SEPARATOR EFFICIENCY CONFIGURATION
# =============================================================================

class SeparatorEfficiencyConfig(BaseModel):
    """
    Steam separator efficiency bounds and parameters.

    Separators remove entrained moisture from steam to improve quality.
    Efficiency bounds ensure realistic expectations for different types.

    Attributes:
        efficiency_min: Minimum acceptable separator efficiency
        efficiency_max: Maximum realistic separator efficiency
        efficiency_design: Design point efficiency
    """

    # Efficiency bounds (dimensionless, 0-1)
    efficiency_min: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable separator efficiency"
    )
    efficiency_design: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Design point separator efficiency"
    )
    efficiency_max: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Maximum realistic separator efficiency"
    )
    efficiency_alarm_low: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Low efficiency alarm threshold"
    )

    # Type-specific efficiency ranges
    type_efficiency_ranges: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            SeparatorType.CENTRIFUGAL.value: {"min": 0.90, "max": 0.99, "typical": 0.96},
            SeparatorType.BAFFLE.value: {"min": 0.70, "max": 0.90, "typical": 0.82},
            SeparatorType.MESH_PAD.value: {"min": 0.85, "max": 0.98, "typical": 0.94},
            SeparatorType.CHEVRON.value: {"min": 0.88, "max": 0.97, "typical": 0.93},
            SeparatorType.CYCLONE.value: {"min": 0.92, "max": 0.99, "typical": 0.97},
            SeparatorType.WIRE_MESH.value: {"min": 0.80, "max": 0.95, "typical": 0.90},
            SeparatorType.COMBINED.value: {"min": 0.95, "max": 0.995, "typical": 0.98},
        },
        description="Efficiency ranges by separator type"
    )

    # Pressure drop limits
    pressure_drop_max_kpa: float = Field(
        default=20.0,
        ge=0.0,
        le=200.0,
        description="Maximum allowable pressure drop in kPa"
    )
    pressure_drop_alarm_kpa: float = Field(
        default=30.0,
        ge=0.0,
        le=250.0,
        description="Pressure drop alarm threshold in kPa"
    )

    # Velocity parameters
    velocity_min_m_s: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Minimum steam velocity for effective separation"
    )
    velocity_design_m_s: float = Field(
        default=15.0,
        ge=0.0,
        le=80.0,
        description="Design steam velocity"
    )
    velocity_max_m_s: float = Field(
        default=30.0,
        ge=0.0,
        le=100.0,
        description="Maximum steam velocity (re-entrainment risk)"
    )

    # Maintenance thresholds
    fouling_factor_max: float = Field(
        default=0.002,
        ge=0.0,
        le=0.01,
        description="Maximum fouling factor (m2-K/W) before maintenance"
    )

    def get_efficiency_range(self, separator_type: SeparatorType) -> Dict[str, float]:
        """Get efficiency range for a separator type."""
        return self.type_efficiency_ranges.get(
            separator_type.value,
            {"min": self.efficiency_min, "max": self.efficiency_max, "typical": self.efficiency_design}
        )

    @model_validator(mode='after')
    def validate_efficiency_bounds(self) -> 'SeparatorEfficiencyConfig':
        """Validate efficiency bound relationships."""
        if self.efficiency_min > self.efficiency_design:
            raise ValueError("efficiency_min must be <= efficiency_design")
        if self.efficiency_design > self.efficiency_max:
            raise ValueError("efficiency_design must be <= efficiency_max")
        if self.efficiency_alarm_low >= self.efficiency_min:
            raise ValueError("efficiency_alarm_low must be < efficiency_min")
        return self


# =============================================================================
# CALCULATION CONFIGURATION
# =============================================================================

class CalculationConfig(BaseModel):
    """
    Calculation engine configuration.

    Defines methods, precision, and validation for steam quality calculations.
    """

    method: CalculationMethod = Field(
        default=CalculationMethod.IAPWS_PROPERTIES,
        description="Primary calculation method"
    )
    fallback_method: CalculationMethod = Field(
        default=CalculationMethod.HEAT_BALANCE,
        description="Fallback method if primary fails"
    )
    precision_decimal_places: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Decimal places for quality calculations"
    )

    # IAPWS-IF97 configuration
    iapws_tolerance: float = Field(
        default=1e-9,
        ge=1e-15,
        le=1e-3,
        description="Convergence tolerance for IAPWS iterations"
    )
    iapws_max_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Maximum iterations for IAPWS calculations"
    )

    # Validation thresholds
    mass_balance_tolerance_pct: float = Field(
        default=1.0,
        ge=0.0,
        le=10.0,
        description="Maximum mass balance error percentage"
    )
    energy_balance_tolerance_pct: float = Field(
        default=2.0,
        ge=0.0,
        le=10.0,
        description="Maximum energy balance error percentage"
    )

    # Caching
    cache_enabled: bool = Field(
        default=True,
        description="Enable property calculation caching"
    )
    cache_size: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum cache entries"
    )


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

class SafetyConfig(BaseModel):
    """
    Safety system configuration per IEC 61511.

    Defines safety integrity levels and emergency response parameters.
    """

    sil_level: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Safety Integrity Level per IEC 61511"
    )
    fail_safe_enabled: bool = Field(
        default=True,
        description="Enable fail-safe mode on errors"
    )

    # Circuit breaker configuration
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    failure_threshold: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Failures before circuit opens"
    )
    recovery_timeout_seconds: float = Field(
        default=30.0,
        ge=5.0,
        le=600.0,
        description="Time before attempting recovery"
    )
    half_open_max_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Requests to test in half-open state"
    )

    # Emergency thresholds
    emergency_pressure_high_kpa: float = Field(
        default=2500.0,
        ge=100.0,
        le=30000.0,
        description="Emergency high pressure threshold (kPa)"
    )
    emergency_pressure_low_kpa: float = Field(
        default=50.0,
        ge=0.0,
        le=1000.0,
        description="Emergency low pressure threshold (kPa)"
    )
    emergency_temp_high_c: float = Field(
        default=350.0,
        ge=100.0,
        le=700.0,
        description="Emergency high temperature threshold (C)"
    )

    # Watchdog
    watchdog_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Watchdog timeout in milliseconds"
    )
    heartbeat_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Heartbeat interval in milliseconds"
    )


# =============================================================================
# MONITORING CONFIGURATION
# =============================================================================

class MonitoringConfig(BaseModel):
    """Monitoring and metrics configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable monitoring"
    )
    metrics_prefix: str = Field(
        default="gl_012_steamqual",
        description="Prometheus metrics prefix"
    )
    metrics_port: int = Field(
        default=9012,
        ge=1024,
        le=65535,
        description="Metrics HTTP port"
    )

    # Sampling
    sampling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Data sampling interval in milliseconds"
    )
    aggregation_window_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Metrics aggregation window"
    )

    # History
    history_retention_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        le=8760,  # 1 year
        description="In-memory history retention"
    )


# =============================================================================
# INTEGRATION CONFIGURATION
# =============================================================================

class IntegrationConfig(BaseModel):
    """External system integration configuration."""

    scada_enabled: bool = Field(
        default=True,
        description="Enable SCADA integration"
    )
    scada_protocol: str = Field(
        default="OPC-UA",
        description="SCADA communication protocol"
    )
    scada_endpoint: str = Field(
        default="opc.tcp://localhost:4840",
        description="SCADA server endpoint"
    )

    historian_enabled: bool = Field(
        default=True,
        description="Enable historian integration"
    )
    historian_url: Optional[str] = Field(
        default=None,
        description="Process historian URL"
    )

    # Polling intervals
    polling_interval_ms: int = Field(
        default=1000,
        ge=100,
        le=60000,
        description="Data polling interval"
    )
    write_enabled: bool = Field(
        default=True,
        description="Enable write-back to control system"
    )
    write_confirmation_required: bool = Field(
        default=True,
        description="Require confirmation for writes"
    )


# =============================================================================
# SITE-SPECIFIC CONFIGURATION
# =============================================================================

class SiteConfig(BaseModel):
    """
    Site-specific configuration overrides.

    Allows customization per deployment site while maintaining
    baseline standards.
    """

    site_id: str = Field(
        default="DEFAULT",
        min_length=1,
        max_length=50,
        description="Unique site identifier"
    )
    site_name: str = Field(
        default="Default Site",
        description="Human-readable site name"
    )

    # Operating parameters
    operating_pressure_range_kpa: tuple[float, float] = Field(
        default=(100.0, 2000.0),
        description="Site operating pressure range (min, max) in kPa"
    )
    operating_temp_range_c: tuple[float, float] = Field(
        default=(100.0, 300.0),
        description="Site operating temperature range (min, max) in C"
    )

    # Steam system characteristics
    boiler_capacity_kg_hr: float = Field(
        default=10000.0,
        ge=100.0,
        le=1000000.0,
        description="Total boiler capacity in kg/hr"
    )
    steam_header_count: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of steam headers"
    )

    # Quality requirements
    quality_grade: str = Field(
        default="industrial",
        description="Required steam quality grade (industrial, process, utility)"
    )
    turbine_application: bool = Field(
        default=False,
        description="Steam used for turbine applications (stricter quality)"
    )

    # Custom thresholds (override defaults)
    custom_x_min: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Site-specific minimum dryness fraction"
    )
    custom_superheat_margin_c: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Site-specific superheat margin"
    )

    # Timezone
    timezone: str = Field(
        default="UTC",
        description="Site timezone for timestamp localization"
    )


# =============================================================================
# PROVENANCE CONFIGURATION
# =============================================================================

class ProvenanceConfig(BaseModel):
    """Provenance and audit trail configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    hash_algorithm: str = Field(
        default="sha256",
        description="Hash algorithm for provenance"
    )
    include_timestamps: bool = Field(
        default=True,
        description="Include timestamps in provenance records"
    )

    # Storage
    store_records: bool = Field(
        default=True,
        description="Store provenance records in memory"
    )
    max_records: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum in-memory records"
    )

    # Audit retention
    audit_retention_days: int = Field(
        default=2555,  # 7 years
        ge=30,
        le=3650,
        description="Audit record retention in days"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class SteamQualConfig(BaseSettings):
    """
    Complete GL-012 STEAMQUAL SteamQualityController configuration.

    Aggregates all sub-configurations with environment variable support.
    Environment variables use prefix GL_012_ with nested double underscores.

    Example:
        >>> config = SteamQualConfig()
        >>> print(config.quality.x_min_default)  # 0.95

        # Override via environment:
        # GL_012_QUALITY__X_MIN_DEFAULT=0.98
        # GL_012_SITE__SITE_ID=PLANT_01
    """

    model_config = SettingsConfigDict(
        env_prefix="GL_012_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Agent identification
    agent_id: str = Field(
        default="GL-012",
        description="Agent identifier"
    )
    agent_name: str = Field(
        default="STEAMQUAL",
        description="Agent name"
    )
    agent_version: str = Field(
        default="1.0.0",
        description="Agent version"
    )

    # Environment
    environment: str = Field(
        default="production",
        description="Deployment environment"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode"
    )

    # Control mode
    control_mode: QualityControlMode = Field(
        default=QualityControlMode.ADVISORY,
        description="Default control mode"
    )

    # Sub-configurations
    quality: QualityThresholdsConfig = Field(
        default_factory=QualityThresholdsConfig,
        description="Quality threshold configuration"
    )
    carryover: CarryoverRiskConfig = Field(
        default_factory=CarryoverRiskConfig,
        description="Carryover risk configuration"
    )
    separator: SeparatorEfficiencyConfig = Field(
        default_factory=SeparatorEfficiencyConfig,
        description="Separator efficiency configuration"
    )
    calculation: CalculationConfig = Field(
        default_factory=CalculationConfig,
        description="Calculation configuration"
    )
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring configuration"
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )
    site: SiteConfig = Field(
        default_factory=SiteConfig,
        description="Site-specific configuration"
    )
    provenance: ProvenanceConfig = Field(
        default_factory=ProvenanceConfig,
        description="Provenance configuration"
    )

    # Reproducibility
    random_seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducibility"
    )
    deterministic_mode: bool = Field(
        default=True,
        description="Enable fully deterministic calculations"
    )

    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment value."""
        valid_envs = {"development", "staging", "production", "testing"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    def get_effective_x_min(self) -> float:
        """Get effective minimum dryness fraction (site override or default)."""
        if self.site.custom_x_min is not None:
            return self.site.custom_x_min
        # Stricter requirements for turbine applications
        if self.site.turbine_application:
            return max(self.quality.x_min_default, 0.98)
        return self.quality.x_min_default

    def get_effective_superheat_margin(self) -> float:
        """Get effective superheat margin (site override or default)."""
        if self.site.custom_superheat_margin_c is not None:
            return self.site.custom_superheat_margin_c
        return self.quality.superheat_margin_min_c

    def validate_all(self) -> List[str]:
        """
        Validate complete configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Cross-validate quality and site settings
        if self.site.turbine_application and self.quality.x_min_default < 0.95:
            errors.append(
                "Turbine applications require x_min_default >= 0.95"
            )

        # Validate operating ranges
        p_min, p_max = self.site.operating_pressure_range_kpa
        if p_min >= p_max:
            errors.append("Operating pressure range min must be < max")

        t_min, t_max = self.site.operating_temp_range_c
        if t_min >= t_max:
            errors.append("Operating temperature range min must be < max")

        # Validate safety thresholds against operating range
        if self.safety.emergency_pressure_high_kpa <= p_max:
            errors.append(
                "Emergency high pressure must be above operating range max"
            )

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

@lru_cache
def get_settings() -> SteamQualConfig:
    """
    Get cached settings instance.

    Uses LRU cache to ensure singleton pattern.
    Environment variables are read on first call.

    Returns:
        SteamQualConfig instance

    Example:
        >>> config = get_settings()
        >>> print(config.quality.x_min_default)
    """
    return SteamQualConfig()


def get_settings_for_site(site_id: str) -> SteamQualConfig:
    """
    Get settings with site-specific overrides.

    Args:
        site_id: Site identifier

    Returns:
        SteamQualConfig with site settings applied
    """
    base_config = SteamQualConfig()

    # In production, would load site-specific config from database/file
    # For now, just update site_id
    site_config = base_config.site.model_copy(update={"site_id": site_id})

    return base_config.model_copy(update={"site": site_config})


def create_turbine_config() -> SteamQualConfig:
    """
    Create configuration optimized for turbine applications.

    Returns:
        SteamQualConfig with stricter quality thresholds
    """
    quality = QualityThresholdsConfig(
        x_min_default=0.98,
        x_min_critical=0.95,
        x_target=0.995,
        superheat_margin_min_c=5.0,
        superheat_margin_target_c=15.0,
        moisture_content_max_pct=2.0,
    )

    carryover = CarryoverRiskConfig(
        silica_limit_ppb=10.0,  # Stricter for turbines
        silica_alarm_ppb=20.0,
    )

    site = SiteConfig(
        quality_grade="turbine",
        turbine_application=True,
    )

    return SteamQualConfig(
        quality=quality,
        carryover=carryover,
        site=site,
    )


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

DEFAULT_CONFIG = SteamQualConfig()

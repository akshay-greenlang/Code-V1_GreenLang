"""
GL-012_SteamQual - Configuration Models

Pydantic v2 configuration models for steam quality monitoring and control.
These models define site, header, and consumer configurations as well
as quality thresholds and alarm settings.

Features:
- Hierarchical configuration structure (Site -> Header -> Consumer)
- Quality thresholds by header and consumer class
- Full Pydantic v2 validation
- JSON serialization for configuration files
- Environment variable overrides via pydantic-settings

Standards Reference:
- GreenLang configuration standards
- ISA-18.2 for alarm settings
- SI units with explicit unit suffixes

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
)

from .domain import (
    AlarmPriority,
    ConsumerClass,
    EstimationMethod,
    HeaderType,
    SeparatorType,
    Severity,
)


# =============================================================================
# Base Configuration Model
# =============================================================================


class BaseConfigModel(BaseModel):
    """
    Base class for all configuration models.

    Provides common configuration settings and JSON serialization.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "GL-012_SteamQual Configuration Model"
        },
        validate_assignment=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


# =============================================================================
# Quality Thresholds Configuration
# =============================================================================


class QualityThresholds(BaseConfigModel):
    """
    Quality thresholds for steam quality monitoring.

    Defines acceptable quality ranges and alarm thresholds
    for a specific header or consumer class.

    Attributes:
        min_quality_x: Minimum acceptable steam quality (dryness fraction).
        warning_quality_x: Quality level that triggers warning.
        critical_quality_x: Quality level that triggers critical alarm.
        target_quality_x: Target/optimal steam quality.

    Example:
        >>> thresholds = QualityThresholds(
        ...     min_quality_x=0.95,
        ...     warning_quality_x=0.97,
        ...     critical_quality_x=0.93,
        ...     target_quality_x=0.98
        ... )
    """

    min_quality_x: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable steam quality (dryness fraction)",
        json_schema_extra={"examples": [0.90, 0.95, 0.995]},
    )

    warning_quality_x: float = Field(
        default=0.93,
        ge=0.0,
        le=1.0,
        description="Quality level that triggers warning alarm",
        json_schema_extra={"examples": [0.93, 0.96, 0.997]},
    )

    critical_quality_x: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Quality level that triggers critical alarm",
        json_schema_extra={"examples": [0.88, 0.93, 0.993]},
    )

    target_quality_x: float = Field(
        default=0.98,
        ge=0.0,
        le=1.0,
        description="Target/optimal steam quality",
        json_schema_extra={"examples": [0.98, 0.99, 0.999]},
    )

    # Moisture carryover thresholds
    max_moisture_percent: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable moisture content (%)",
    )

    carryover_risk_threshold: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Risk score threshold for carryover warning",
    )

    # Uncertainty thresholds
    max_acceptable_uncertainty: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable uncertainty in quality estimate",
    )

    min_confidence_for_action: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum confidence level to take automatic action",
    )

    @model_validator(mode="after")
    def validate_threshold_ordering(self) -> "QualityThresholds":
        """
        Validate that thresholds are in correct order.

        critical < min < warning < target
        """
        if self.critical_quality_x > self.min_quality_x:
            raise ValueError(
                f"critical_quality_x ({self.critical_quality_x}) must be <= "
                f"min_quality_x ({self.min_quality_x})"
            )
        if self.min_quality_x > self.warning_quality_x:
            raise ValueError(
                f"min_quality_x ({self.min_quality_x}) must be <= "
                f"warning_quality_x ({self.warning_quality_x})"
            )
        if self.warning_quality_x > self.target_quality_x:
            raise ValueError(
                f"warning_quality_x ({self.warning_quality_x}) must be <= "
                f"target_quality_x ({self.target_quality_x})"
            )
        return self

    @classmethod
    def for_consumer_class(cls, consumer_class: ConsumerClass) -> "QualityThresholds":
        """
        Create default thresholds for a specific consumer class.

        Returns thresholds appropriate for the consumer type.
        """
        base_quality = consumer_class.min_quality

        # Calculate thresholds relative to minimum required quality
        return cls(
            min_quality_x=base_quality,
            warning_quality_x=min(1.0, base_quality + 0.02),
            critical_quality_x=max(0.0, base_quality - 0.02),
            target_quality_x=min(1.0, base_quality + 0.03),
            max_moisture_percent=(1.0 - base_quality) * 100,
        )


class PressureThresholds(BaseConfigModel):
    """
    Pressure thresholds for steam header monitoring.

    Defines operating pressure limits and alarm thresholds.
    """

    min_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Minimum acceptable operating pressure (kPa)",
    )

    max_pressure_kpa: float = Field(
        ...,
        ge=0,
        description="Maximum acceptable operating pressure (kPa)",
    )

    low_pressure_alarm_kpa: float = Field(
        ...,
        ge=0,
        description="Low pressure alarm threshold (kPa)",
    )

    high_pressure_alarm_kpa: float = Field(
        ...,
        ge=0,
        description="High pressure alarm threshold (kPa)",
    )

    design_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design pressure for the header (kPa)",
    )

    @model_validator(mode="after")
    def validate_pressure_ordering(self) -> "PressureThresholds":
        """Validate pressure threshold ordering."""
        if self.low_pressure_alarm_kpa > self.min_pressure_kpa:
            raise ValueError(
                f"low_pressure_alarm_kpa ({self.low_pressure_alarm_kpa}) must be <= "
                f"min_pressure_kpa ({self.min_pressure_kpa})"
            )
        if self.min_pressure_kpa > self.max_pressure_kpa:
            raise ValueError(
                f"min_pressure_kpa ({self.min_pressure_kpa}) must be <= "
                f"max_pressure_kpa ({self.max_pressure_kpa})"
            )
        if self.max_pressure_kpa > self.high_pressure_alarm_kpa:
            raise ValueError(
                f"max_pressure_kpa ({self.max_pressure_kpa}) must be <= "
                f"high_pressure_alarm_kpa ({self.high_pressure_alarm_kpa})"
            )
        return self


# =============================================================================
# Consumer Configuration
# =============================================================================


class ConsumerConfig(BaseConfigModel):
    """
    Configuration for a single steam consumer.

    Defines consumer properties, quality requirements,
    and protection settings.

    Attributes:
        consumer_id: Unique consumer identifier.
        consumer_name: Human-readable consumer name.
        consumer_class: Type of consumer (turbine, heat exchanger, etc.).
        quality_thresholds: Quality requirements for this consumer.

    Example:
        >>> config = ConsumerConfig(
        ...     consumer_id="TURB-01",
        ...     consumer_name="Main Steam Turbine",
        ...     consumer_class=ConsumerClass.TURBINE,
        ...     quality_thresholds=QualityThresholds.for_consumer_class(ConsumerClass.TURBINE)
        ... )
    """

    consumer_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique consumer identifier",
        json_schema_extra={"examples": ["TURB-01", "HX-101", "STERILIZER-A"]},
    )

    consumer_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable consumer name",
    )

    consumer_class: ConsumerClass = Field(
        ...,
        description="Type/class of consumer",
    )

    # Quality requirements
    quality_thresholds: QualityThresholds = Field(
        default_factory=lambda: QualityThresholds(),
        description="Quality thresholds for this consumer",
    )

    # Consumer metadata
    description: str = Field(
        default="",
        max_length=1000,
        description="Consumer description",
    )

    location: str = Field(
        default="",
        max_length=255,
        description="Physical location",
    )

    # Operating parameters
    design_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design steam flow rate (kg/s)",
    )

    max_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum steam flow rate (kg/s)",
    )

    min_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum steam flow rate (kg/s)",
    )

    # Protection settings
    is_critical: bool = Field(
        default=False,
        description="Whether this consumer has critical quality requirements",
    )

    enable_protection: bool = Field(
        default=True,
        description="Enable automatic protection actions",
    )

    trip_on_low_quality: bool = Field(
        default=False,
        description="Trip consumer on critically low quality",
    )

    # Connected header
    header_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Connected steam header ID",
    )

    # Tags for process data
    flow_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for flow measurement",
    )

    inlet_pressure_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for inlet pressure",
    )

    inlet_temperature_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for inlet temperature",
    )

    @model_validator(mode="after")
    def set_critical_from_class(self) -> "ConsumerConfig":
        """Set is_critical based on consumer class if not explicitly set."""
        if self.consumer_class.is_critical:
            self.is_critical = True
        return self

    @field_validator("consumer_id")
    @classmethod
    def validate_consumer_id(cls, v: str) -> str:
        """Validate consumer ID format."""
        v = v.strip()
        if not v:
            raise ValueError("consumer_id cannot be empty")
        return v


# =============================================================================
# Separator Configuration
# =============================================================================


class SeparatorConfig(BaseConfigModel):
    """
    Configuration for a moisture separator.

    Defines separator properties and performance parameters.
    """

    separator_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique separator identifier",
    )

    separator_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable separator name",
    )

    separator_type: SeparatorType = Field(
        default=SeparatorType.CYCLONE,
        description="Type of separator",
    )

    # Connected header
    inlet_header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Inlet header ID",
    )

    outlet_header_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Outlet header ID (if different from inlet)",
    )

    # Performance parameters
    design_efficiency: float = Field(
        default=0.95,
        ge=0,
        le=1.0,
        description="Design separation efficiency (0-1)",
    )

    design_flow_kg_s: float = Field(
        ...,
        ge=0,
        description="Design steam flow capacity (kg/s)",
    )

    max_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Maximum rated flow (kg/s)",
    )

    min_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Minimum effective flow (kg/s)",
    )

    # Pressure drop
    design_dp_kpa: float = Field(
        default=5.0,
        ge=0,
        description="Design pressure drop (kPa)",
    )

    max_dp_kpa: float = Field(
        default=15.0,
        ge=0,
        description="Maximum pressure drop (kPa)",
    )

    # Drain configuration
    drain_valve_type: str = Field(
        default="modulating",
        pattern="^(modulating|on_off|manual)$",
        description="Type of drain valve",
    )

    drain_capacity_kg_s: float = Field(
        default=1.0,
        ge=0,
        description="Maximum drain capacity (kg/s)",
    )

    # Process tags
    dp_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for differential pressure",
    )

    drain_flow_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for drain flow",
    )

    drain_valve_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for drain valve position",
    )


# =============================================================================
# Header Configuration
# =============================================================================


class HeaderConfig(BaseConfigModel):
    """
    Configuration for a steam header.

    Defines header properties, connected consumers,
    and quality monitoring settings.

    Attributes:
        header_id: Unique header identifier.
        header_name: Human-readable header name.
        header_type: Pressure class of the header.
        quality_thresholds: Quality thresholds for this header.
        consumers: List of connected consumers.

    Example:
        >>> config = HeaderConfig(
        ...     header_id="MP-01",
        ...     header_name="Medium Pressure Header 1",
        ...     header_type=HeaderType.MEDIUM_PRESSURE,
        ...     quality_thresholds=QualityThresholds(min_quality_x=0.95)
        ... )
    """

    header_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique header identifier",
        json_schema_extra={"examples": ["HP-01", "MP-MAIN", "LP-UTILITIES"]},
    )

    header_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable header name",
    )

    header_type: HeaderType = Field(
        default=HeaderType.MEDIUM_PRESSURE,
        description="Header pressure classification",
    )

    # Quality settings
    quality_thresholds: QualityThresholds = Field(
        default_factory=lambda: QualityThresholds(),
        description="Quality thresholds for this header",
    )

    # Pressure settings
    pressure_thresholds: Optional[PressureThresholds] = Field(
        default=None,
        description="Pressure thresholds for this header",
    )

    # Primary consumer class (determines default quality requirements)
    primary_consumer_class: ConsumerClass = Field(
        default=ConsumerClass.GENERAL,
        description="Primary consumer class served by this header",
    )

    # Connected equipment
    consumers: List[ConsumerConfig] = Field(
        default_factory=list,
        description="Connected steam consumers",
    )

    separators: List[SeparatorConfig] = Field(
        default_factory=list,
        description="Connected moisture separators",
    )

    # Header metadata
    description: str = Field(
        default="",
        max_length=1000,
        description="Header description",
    )

    location: str = Field(
        default="",
        max_length=255,
        description="Physical location",
    )

    # Operating parameters
    design_pressure_kpa: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design pressure (kPa)",
    )

    design_temperature_c: Optional[float] = Field(
        default=None,
        description="Design temperature (C)",
    )

    design_flow_kg_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Design flow rate (kg/s)",
    )

    # Measurement tags
    pressure_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for pressure measurement",
    )

    temperature_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for temperature measurement",
    )

    flow_tag: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Process tag for flow measurement",
    )

    # Estimation settings
    preferred_estimation_method: EstimationMethod = Field(
        default=EstimationMethod.ENTHALPY,
        description="Preferred quality estimation method",
    )

    estimation_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Quality estimation interval (seconds)",
    )

    # Enable/disable flags
    is_active: bool = Field(
        default=True,
        description="Whether header monitoring is active",
    )

    enable_quality_estimation: bool = Field(
        default=True,
        description="Enable quality estimation for this header",
    )

    enable_carryover_detection: bool = Field(
        default=True,
        description="Enable carryover risk detection",
    )

    @field_validator("header_id")
    @classmethod
    def validate_header_id(cls, v: str) -> str:
        """Validate header ID format."""
        v = v.strip()
        if not v:
            raise ValueError("header_id cannot be empty")
        return v

    def get_consumer_by_id(self, consumer_id: str) -> Optional[ConsumerConfig]:
        """Get consumer configuration by ID."""
        for consumer in self.consumers:
            if consumer.consumer_id == consumer_id:
                return consumer
        return None

    def get_most_restrictive_quality(self) -> float:
        """Get the most restrictive minimum quality among all consumers."""
        min_qualities = [self.quality_thresholds.min_quality_x]
        for consumer in self.consumers:
            min_qualities.append(consumer.quality_thresholds.min_quality_x)
        return max(min_qualities)


# =============================================================================
# Alarm Configuration
# =============================================================================


class AlarmConfig(BaseConfigModel):
    """
    Alarm configuration for quality events.

    Defines how quality events are mapped to alarms.
    """

    enabled: bool = Field(
        default=True,
        description="Enable alarm generation",
    )

    # Severity to priority mapping
    severity_priority_map: Dict[str, str] = Field(
        default_factory=lambda: {
            "S0_INFO": "diagnostic",
            "S1_ADVISORY": "low",
            "S2_WARNING": "medium",
            "S3_CRITICAL": "high",
        },
        description="Mapping of severity to alarm priority",
    )

    # Deadbands and delays
    quality_deadband: float = Field(
        default=0.01,
        ge=0,
        le=0.1,
        description="Deadband for quality alarm (prevents chattering)",
    )

    alarm_delay_seconds: int = Field(
        default=10,
        ge=0,
        le=300,
        description="Delay before raising alarm (seconds)",
    )

    clear_delay_seconds: int = Field(
        default=30,
        ge=0,
        le=600,
        description="Delay before clearing alarm (seconds)",
    )

    # Notification settings
    notify_on_critical: bool = Field(
        default=True,
        description="Send notification for critical alarms",
    )

    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "sms"],
        description="Notification channels to use",
    )

    escalation_timeout_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Time before escalating unacknowledged alarm",
    )


# =============================================================================
# Site Configuration
# =============================================================================


class SiteConfig(BaseConfigModel):
    """
    Top-level site configuration for GL-012_SteamQual.

    Contains all configuration for a single site including
    all headers, consumers, and global settings.

    Attributes:
        site_id: Unique site identifier.
        site_name: Human-readable site name.
        headers: List of steam header configurations.
        alarm_config: Alarm generation settings.

    Example:
        >>> config = SiteConfig(
        ...     site_id="PLANT-01",
        ...     site_name="Main Manufacturing Plant",
        ...     headers=[
        ...         HeaderConfig(
        ...             header_id="MP-01",
        ...             header_name="Medium Pressure Header",
        ...             ...
        ...         )
        ...     ]
        ... )
    """

    site_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique site identifier",
        json_schema_extra={"examples": ["PLANT-01", "REFINERY-A", "FACTORY-MAIN"]},
    )

    site_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable site name",
    )

    # Headers
    headers: List[HeaderConfig] = Field(
        default_factory=list,
        description="Steam header configurations",
    )

    # Global settings
    alarm_config: AlarmConfig = Field(
        default_factory=AlarmConfig,
        description="Alarm generation settings",
    )

    # Default thresholds (applied when not specified at header level)
    default_quality_thresholds: QualityThresholds = Field(
        default_factory=lambda: QualityThresholds(),
        description="Default quality thresholds for all headers",
    )

    # Site metadata
    description: str = Field(
        default="",
        max_length=1000,
        description="Site description",
    )

    timezone: str = Field(
        default="UTC",
        max_length=50,
        description="Site timezone",
    )

    # Integration settings
    historian_connection_string: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Process historian connection string",
    )

    scada_connection_string: Optional[str] = Field(
        default=None,
        max_length=500,
        description="SCADA system connection string",
    )

    gl003_endpoint: Optional[str] = Field(
        default=None,
        max_length=255,
        description="GL-003 UnifiedSteam API endpoint",
    )

    # Estimation settings
    global_estimation_interval_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Default estimation interval for all headers (seconds)",
    )

    max_data_age_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum age for data to be considered valid (seconds)",
    )

    # Feature flags
    enable_quality_estimation: bool = Field(
        default=True,
        description="Enable quality estimation globally",
    )

    enable_carryover_detection: bool = Field(
        default=True,
        description="Enable carryover detection globally",
    )

    enable_gl003_integration: bool = Field(
        default=True,
        description="Enable GL-003 UnifiedSteam integration",
    )

    enable_audit_logging: bool = Field(
        default=True,
        description="Enable detailed audit logging",
    )

    # Version tracking
    config_version: str = Field(
        default="1.0.0",
        max_length=20,
        description="Configuration version",
    )

    last_modified: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When configuration was last modified",
    )

    @field_validator("site_id")
    @classmethod
    def validate_site_id(cls, v: str) -> str:
        """Validate site ID format."""
        v = v.strip()
        if not v:
            raise ValueError("site_id cannot be empty")
        return v

    def get_header_by_id(self, header_id: str) -> Optional[HeaderConfig]:
        """Get header configuration by ID."""
        for header in self.headers:
            if header.header_id == header_id:
                return header
        return None

    def get_all_consumers(self) -> List[ConsumerConfig]:
        """Get all consumers across all headers."""
        consumers = []
        for header in self.headers:
            consumers.extend(header.consumers)
        return consumers

    def get_all_separators(self) -> List[SeparatorConfig]:
        """Get all separators across all headers."""
        separators = []
        for header in self.headers:
            separators.extend(header.separators)
        return separators

    def get_critical_consumers(self) -> List[ConsumerConfig]:
        """Get all critical consumers."""
        return [c for c in self.get_all_consumers() if c.is_critical]

    def get_consumer_for_header(
        self, header_id: str, consumer_id: str
    ) -> Optional[ConsumerConfig]:
        """Get consumer configuration for a specific header."""
        header = self.get_header_by_id(header_id)
        if header:
            return header.get_consumer_by_id(consumer_id)
        return None


# =============================================================================
# Agent Settings (Environment Variables)
# =============================================================================


class AgentSettings(BaseConfigModel):
    """
    Agent settings that can be overridden via environment variables.

    Used with pydantic-settings for environment variable support.
    """

    # Agent identification
    agent_id: str = Field(
        default="GL-012",
        description="Agent identifier",
    )

    agent_name: str = Field(
        default="SteamQual",
        description="Agent name",
    )

    agent_version: str = Field(
        default="1.0.0",
        description="Agent version",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )

    # Provenance tracking
    track_provenance: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking",
    )

    provenance_store_path: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Path to provenance storage",
    )

    # API settings
    api_host: str = Field(
        default="0.0.0.0",
        description="API server host",
    )

    api_port: int = Field(
        default=8012,
        ge=1,
        le=65535,
        description="API server port",
    )

    api_workers: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of API workers",
    )

    # Performance
    max_concurrent_estimations: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent quality estimations",
    )

    estimation_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Timeout for single estimation (seconds)",
    )

    # Caching
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching",
    )

    cache_ttl_seconds: int = Field(
        default=60,
        ge=1,
        le=3600,
        description="Cache time-to-live (seconds)",
    )

    # Metrics
    enable_metrics: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )

    metrics_port: int = Field(
        default=9012,
        ge=1,
        le=65535,
        description="Metrics server port",
    )

    # Tracing
    enable_tracing: bool = Field(
        default=False,
        description="Enable OpenTelemetry tracing",
    )

    tracing_endpoint: Optional[str] = Field(
        default=None,
        max_length=255,
        description="OpenTelemetry collector endpoint",
    )

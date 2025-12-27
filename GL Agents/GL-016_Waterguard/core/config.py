"""
GL-016 WATERGUARD Boiler Water Treatment Agent - Configuration Module

This module defines all configuration schemas for the WATERGUARD
Boiler Water Treatment Agent including chemistry limits, dosing settings,
blowdown configuration, safety thresholds, and operational settings.

All configurations use Pydantic for validation with comprehensive defaults
and documentation following ASME and ABMA standards.

Standards Compliance:
    - ASME Boiler and Pressure Vessel Code
    - ABMA (American Boiler Manufacturers Association) Guidelines
    - IEC 62443 (Industrial Cybersecurity)
    - ISO 50001 (Energy Management Systems)
"""

from datetime import timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import os

from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# ENUMS
# =============================================================================

class SafetyLevel(Enum):
    """IEC 61511 Safety Integrity Levels for water treatment systems."""
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class OperatingMode(Enum):
    """Agent operating modes for water treatment control."""
    RECOMMEND_ONLY = "recommend_only"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"
    FALLBACK = "fallback"


class ProtocolType(Enum):
    """Communication protocol types."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"
    MQTT = "mqtt"
    KAFKA = "kafka"
    HTTP_REST = "http_rest"
    GRPC = "grpc"


class DeploymentMode(Enum):
    """Agent deployment modes."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SHADOW = "shadow"


class ChemicalType(Enum):
    """Types of water treatment chemicals."""
    OXYGEN_SCAVENGER = "oxygen_scavenger"
    PHOSPHATE = "phosphate"
    CAUSTIC = "caustic"
    AMINE = "amine"
    POLYMER = "polymer"
    BIOCIDE = "biocide"
    CHELANT = "chelant"
    SULFITE = "sulfite"


class BlowdownMode(Enum):
    """Blowdown operation modes."""
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    COMBINED = "combined"
    MANUAL = "manual"


class ConstraintType(Enum):
    """Constraint classification for chemistry limits."""
    HARD = "hard"
    SOFT = "soft"


class QualityFlag(Enum):
    """Data quality indicators for sensor readings."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    STALE = "stale"
    SIMULATED = "simulated"


class ComplianceStatus(Enum):
    """Overall compliance status."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


# =============================================================================
# INTEGRATION CONFIGURATIONS
# =============================================================================

class KafkaConfig(BaseModel):
    """Kafka integration configuration."""
    
    enabled: bool = Field(default=False, description="Enable Kafka event streaming")
    bootstrap_servers: str = Field(default="localhost:9092", description="Kafka bootstrap servers")
    topic_prefix: str = Field(default="waterguard", description="Topic prefix for all Waterguard events")
    consumer_group: str = Field(default="waterguard-consumers", description="Consumer group ID")
    security_protocol: str = Field(default="SASL_SSL", description="Security protocol")
    sasl_mechanism: Optional[str] = Field(default="PLAIN", description="SASL mechanism")
    compression_type: str = Field(default="gzip", description="Message compression type")
    acks: str = Field(default="all", description="Producer acknowledgment level")


class OPCUAConfig(BaseModel):
    """OPC-UA integration configuration."""
    
    enabled: bool = Field(default=True, description="Enable OPC-UA connectivity")
    endpoint: str = Field(default="opc.tcp://localhost:4840", description="OPC-UA server endpoint")
    namespace_uri: str = Field(default="urn:greenlang:waterguard", description="OPC-UA namespace URI")
    security_policy: str = Field(default="Basic256Sha256", description="OPC-UA security policy")
    security_mode: str = Field(default="SignAndEncrypt", description="Security mode")
    polling_interval_ms: int = Field(default=1000, ge=100, le=60000, description="Data polling interval (ms)")
    subscription_interval_ms: int = Field(default=500, ge=100, le=10000, description="Subscription publishing interval (ms)")
    connection_timeout_s: int = Field(default=30, ge=5, le=300, description="Connection timeout (seconds)")
    retry_count: int = Field(default=3, ge=1, le=10, description="Connection retry count")


class CMMSConfig(BaseModel):
    """CMMS integration configuration."""
    
    enabled: bool = Field(default=False, description="Enable CMMS integration")
    system_type: str = Field(default="maximo", description="CMMS type (maximo, sap_pm, infor)")
    api_endpoint: Optional[str] = Field(default=None, description="CMMS API endpoint URL")
    work_order_priority_map: Dict[str, int] = Field(
        default_factory=lambda: {"critical": 1, "high": 2, "medium": 3, "low": 4, "routine": 5},
        description="Priority mapping to CMMS priorities"
    )
    auto_create_work_orders: bool = Field(default=False, description="Automatically create work orders for violations")


# =============================================================================
# CONSTRAINT CONFIGURATIONS
# =============================================================================

class ConstraintConfig(BaseModel):
    """Configuration for hard and soft constraint limits."""
    
    constraint_type: ConstraintType = Field(default=ConstraintType.SOFT, description="Constraint classification")
    min_value: Optional[float] = Field(default=None, description="Minimum allowed value")
    max_value: Optional[float] = Field(default=None, description="Maximum allowed value")
    target_value: Optional[float] = Field(default=None, description="Optimal target value")
    warning_threshold_percent: float = Field(default=10.0, ge=0.0, le=50.0, description="Warning threshold as % of limit")
    alarm_threshold_percent: float = Field(default=5.0, ge=0.0, le=25.0, description="Alarm threshold as % of limit")
    unit: str = Field(default="", description="Engineering unit")


# =============================================================================
# CHEMISTRY LIMITS CONFIGURATIONS
# =============================================================================

class ConductivityLimitsConfig(BaseModel):
    """Boiler water conductivity limits configuration."""
    
    min_us_cm: float = Field(default=500.0, ge=0.0, le=5000.0, description="Minimum conductivity (uS/cm)")
    max_us_cm: float = Field(default=3500.0, ge=100.0, le=10000.0, description="Maximum conductivity (uS/cm) - ABMA guideline")
    target_us_cm: float = Field(default=2500.0, ge=100.0, le=8000.0, description="Target conductivity (uS/cm)")
    hard_limit_us_cm: float = Field(default=5000.0, ge=500.0, le=15000.0, description="Hard limit - immediate action required")
    constraint_type: ConstraintType = Field(default=ConstraintType.HARD, description="Constraint type")
    
    @model_validator(mode="after")
    def validate_limits(self) -> "ConductivityLimitsConfig":
        if self.min_us_cm >= self.target_us_cm:
            raise ValueError("min_us_cm must be less than target_us_cm")
        if self.target_us_cm >= self.max_us_cm:
            raise ValueError("target_us_cm must be less than max_us_cm")
        if self.max_us_cm >= self.hard_limit_us_cm:
            raise ValueError("max_us_cm must be less than hard_limit_us_cm")
        return self


class SilicaLimitsConfig(BaseModel):
    """Boiler water silica limits configuration."""
    
    min_ppm: float = Field(default=0.0, ge=0.0, le=50.0, description="Minimum silica (ppm as SiO2)")
    max_ppm: float = Field(default=150.0, ge=1.0, le=500.0, description="Maximum silica (ppm as SiO2) - pressure dependent")
    target_ppm: float = Field(default=100.0, ge=0.0, le=400.0, description="Target silica (ppm as SiO2)")
    hard_limit_ppm: float = Field(default=200.0, ge=10.0, le=600.0, description="Hard limit - risk of silica carryover")
    constraint_type: ConstraintType = Field(default=ConstraintType.HARD, description="Constraint type")


class pHLimitsConfig(BaseModel):
    """Boiler water pH limits configuration."""
    
    min_ph: float = Field(default=10.5, ge=7.0, le=12.0, description="Minimum pH for corrosion protection")
    max_ph: float = Field(default=11.5, ge=8.0, le=14.0, description="Maximum pH to prevent caustic attack")
    target_ph: float = Field(default=11.0, ge=8.0, le=13.0, description="Target pH for optimal protection")
    constraint_type: ConstraintType = Field(default=ConstraintType.HARD, description="Constraint type")
    
    @model_validator(mode="after")
    def validate_ph_range(self) -> "pHLimitsConfig":
        if self.min_ph >= self.max_ph:
            raise ValueError("min_ph must be less than max_ph")
        if not (self.min_ph <= self.target_ph <= self.max_ph):
            raise ValueError("target_ph must be between min_ph and max_ph")
        return self


class AlkalinityLimitsConfig(BaseModel):
    """Boiler water alkalinity limits configuration."""
    
    min_ppm_caco3: float = Field(default=100.0, ge=0.0, le=500.0, description="Minimum alkalinity (ppm as CaCO3)")
    max_ppm_caco3: float = Field(default=700.0, ge=50.0, le=2000.0, description="Maximum alkalinity (ppm as CaCO3)")
    target_ppm_caco3: float = Field(default=400.0, ge=50.0, le=1500.0, description="Target alkalinity (ppm as CaCO3)")
    caustic_ratio_max: float = Field(default=0.4, ge=0.0, le=1.0, description="Maximum caustic/total alkalinity ratio")
    constraint_type: ConstraintType = Field(default=ConstraintType.SOFT, description="Constraint type")


class DissolvedO2LimitsConfig(BaseModel):
    """Dissolved oxygen limits configuration."""
    
    max_ppb: float = Field(default=7.0, ge=0.0, le=100.0, description="Maximum dissolved O2 in feedwater (ppb)")
    target_ppb: float = Field(default=5.0, ge=0.0, le=50.0, description="Target dissolved O2 (ppb)")
    hard_limit_ppb: float = Field(default=20.0, ge=1.0, le=200.0, description="Hard limit - corrosion risk")
    constraint_type: ConstraintType = Field(default=ConstraintType.HARD, description="Constraint type")


class IronLimitsConfig(BaseModel):
    """Iron concentration limits configuration."""
    
    max_ppm: float = Field(default=0.1, ge=0.0, le=5.0, description="Maximum iron in feedwater (ppm)")
    target_ppm: float = Field(default=0.05, ge=0.0, le=2.0, description="Target iron concentration (ppm)")
    hard_limit_ppm: float = Field(default=0.5, ge=0.01, le=10.0, description="Hard limit - indicates corrosion")
    constraint_type: ConstraintType = Field(default=ConstraintType.SOFT, description="Constraint type")


class CopperLimitsConfig(BaseModel):
    """Copper concentration limits configuration."""
    
    max_ppm: float = Field(default=0.05, ge=0.0, le=1.0, description="Maximum copper in feedwater (ppm)")
    target_ppm: float = Field(default=0.02, ge=0.0, le=0.5, description="Target copper concentration (ppm)")
    hard_limit_ppm: float = Field(default=0.1, ge=0.01, le=2.0, description="Hard limit - indicates corrosion")
    constraint_type: ConstraintType = Field(default=ConstraintType.SOFT, description="Constraint type")


class ChemistryLimitsConfig(BaseModel):
    """Complete water chemistry limits configuration."""
    
    conductivity: ConductivityLimitsConfig = Field(default_factory=ConductivityLimitsConfig, description="Conductivity limits")
    silica: SilicaLimitsConfig = Field(default_factory=SilicaLimitsConfig, description="Silica limits")
    ph: pHLimitsConfig = Field(default_factory=pHLimitsConfig, description="pH limits")
    alkalinity: AlkalinityLimitsConfig = Field(default_factory=AlkalinityLimitsConfig, description="Alkalinity limits")
    dissolved_o2: DissolvedO2LimitsConfig = Field(default_factory=DissolvedO2LimitsConfig, description="Dissolved oxygen limits")
    iron: IronLimitsConfig = Field(default_factory=IronLimitsConfig, description="Iron limits")
    copper: CopperLimitsConfig = Field(default_factory=CopperLimitsConfig, description="Copper limits")
    hardness_max_ppm: float = Field(default=0.0, ge=0.0, le=10.0, description="Maximum hardness in boiler water (ppm as CaCO3)")
    phosphate_min_ppm: float = Field(default=20.0, ge=0.0, le=100.0, description="Minimum phosphate residual (ppm as PO4)")
    phosphate_max_ppm: float = Field(default=60.0, ge=10.0, le=200.0, description="Maximum phosphate (ppm as PO4)")
    sulfite_min_ppm: float = Field(default=20.0, ge=0.0, le=100.0, description="Minimum sulfite residual (ppm)")
    sulfite_max_ppm: float = Field(default=60.0, ge=10.0, le=200.0, description="Maximum sulfite (ppm)")


# =============================================================================
# DOSING CONFIGURATION
# =============================================================================

class DosingConfig(BaseModel):
    """Chemical dosing configuration."""
    
    enabled: bool = Field(default=True, description="Enable automatic dosing control")
    chemical_type: ChemicalType = Field(default=ChemicalType.PHOSPHATE, description="Primary chemical type")
    pump_capacity_ml_min: float = Field(default=100.0, ge=1.0, le=10000.0, description="Dosing pump capacity (ml/min)")
    min_dose_percent: float = Field(default=5.0, ge=0.0, le=50.0, description="Minimum dose as % of capacity")
    max_dose_percent: float = Field(default=100.0, ge=10.0, le=100.0, description="Maximum dose as % of capacity")
    proportional_band_percent: float = Field(default=20.0, ge=1.0, le=100.0, description="Proportional band for control")
    integral_time_min: float = Field(default=10.0, ge=1.0, le=120.0, description="Integral time (minutes)")
    derivative_time_min: float = Field(default=0.0, ge=0.0, le=30.0, description="Derivative time (minutes)")
    concentration_percent: float = Field(default=30.0, ge=1.0, le=100.0, description="Chemical solution concentration (%)")
    specific_gravity: float = Field(default=1.2, ge=0.8, le=2.0, description="Chemical solution specific gravity")
    max_daily_dose_liters: float = Field(default=50.0, ge=1.0, le=1000.0, description="Maximum daily chemical consumption (liters)")
    dose_lockout_minutes: float = Field(default=5.0, ge=0.0, le=60.0, description="Minimum time between doses (minutes)")


# =============================================================================
# BLOWDOWN CONFIGURATION
# =============================================================================

class BlowdownConfig(BaseModel):
    """Blowdown control configuration."""
    
    enabled: bool = Field(default=True, description="Enable automatic blowdown control")
    mode: BlowdownMode = Field(default=BlowdownMode.COMBINED, description="Blowdown operation mode")
    continuous_enabled: bool = Field(default=True, description="Enable continuous surface blowdown")
    continuous_min_percent: float = Field(default=1.0, ge=0.0, le=10.0, description="Minimum continuous blowdown rate (%)")
    continuous_max_percent: float = Field(default=8.0, ge=1.0, le=20.0, description="Maximum continuous blowdown rate (%)")
    continuous_target_percent: float = Field(default=3.0, ge=0.5, le=15.0, description="Target continuous blowdown rate (%)")
    intermittent_enabled: bool = Field(default=True, description="Enable intermittent bottom blowdown")
    intermittent_interval_hours: float = Field(default=8.0, ge=1.0, le=48.0, description="Interval between bottom blowdowns (hours)")
    intermittent_duration_seconds: float = Field(default=10.0, ge=1.0, le=60.0, description="Bottom blowdown duration (seconds)")
    coc_min: float = Field(default=3.0, ge=1.0, le=10.0, description="Minimum cycles of concentration")
    coc_max: float = Field(default=8.0, ge=2.0, le=25.0, description="Maximum cycles of concentration")
    coc_target: float = Field(default=5.0, ge=2.0, le=20.0, description="Target cycles of concentration")
    heat_recovery_enabled: bool = Field(default=True, description="Enable blowdown heat recovery system")
    heat_recovery_efficiency_percent: float = Field(default=80.0, ge=0.0, le=100.0, description="Heat recovery system efficiency (%)")
    high_level_interlock: bool = Field(default=True, description="Interlock blowdown on high drum level")
    low_level_lockout: bool = Field(default=True, description="Lockout blowdown on low drum level")
    
    @model_validator(mode="after")
    def validate_coc(self) -> "BlowdownConfig":
        if self.coc_min >= self.coc_target:
            raise ValueError("coc_min must be less than coc_target")
        if self.coc_target >= self.coc_max:
            raise ValueError("coc_target must be less than coc_max")
        return self


# =============================================================================
# SAFETY CONFIGURATION
# =============================================================================

class SafetyConfig(BaseModel):
    """Safety system configuration."""
    
    safety_level: SafetyLevel = Field(default=SafetyLevel.SIL_2, description="Safety Integrity Level per IEC 61511")
    watchdog_timeout_ms: int = Field(default=5000, ge=100, le=60000, description="Safety watchdog timeout (ms)")
    heartbeat_interval_ms: int = Field(default=1000, ge=100, le=10000, description="Safety heartbeat interval (ms)")
    high_conductivity_interlock: bool = Field(default=True, description="Interlock on high conductivity")
    high_silica_interlock: bool = Field(default=True, description="Interlock on high silica")
    ph_interlock: bool = Field(default=True, description="Interlock on pH out of range")
    failsafe_blowdown_rate_percent: float = Field(default=5.0, ge=1.0, le=15.0, description="Failsafe blowdown rate on communication loss")
    failsafe_dosing_enabled: bool = Field(default=False, description="Enable dosing in failsafe mode")
    alarm_delay_seconds: float = Field(default=30.0, ge=0.0, le=300.0, description="Alarm delay for transient filtering")
    max_consecutive_violations: int = Field(default=3, ge=1, le=10, description="Max violations before escalation")


# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""
    
    enabled: bool = Field(default=True, description="Enable metrics collection")
    prefix: str = Field(default="greenlang_waterguard", description="Metrics name prefix")
    port: int = Field(default=9096, ge=1024, le=65535, description="Metrics HTTP port")
    push_gateway_url: Optional[str] = Field(default=None, description="Prometheus push gateway URL")
    collection_interval_s: float = Field(default=15.0, ge=1.0, le=300.0, description="Metrics collection interval (seconds)")
    include_chemistry_metrics: bool = Field(default=True, description="Include water chemistry metrics")
    include_dosing_metrics: bool = Field(default=True, description="Include chemical dosing metrics")
    include_blowdown_metrics: bool = Field(default=True, description="Include blowdown metrics")
    histogram_buckets: List[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Histogram bucket boundaries"
    )


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

class WaterguardConfig(BaseModel):
    """
    Complete GL-016 WATERGUARD Boiler Water Treatment Agent configuration.
    
    This is the main configuration class aggregating all sub-configurations
    for comprehensive boiler water treatment optimization.
    
    Example:
        >>> config = WaterguardConfig(
        ...     agent_id="GL-016-001",
        ...     operating_mode=OperatingMode.SUPERVISED,
        ...     chemistry_limits=ChemistryLimitsConfig()
        ... )
        >>> coordinator = ChemistryCoordinator(config)
    """
    
    agent_id: str = Field(default_factory=lambda: f"GL-016-{os.getpid()}", description="Unique agent identifier")
    name: str = Field(default="WATERGUARD-Primary", description="Human-readable agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    safety_level: SafetyLevel = Field(default=SafetyLevel.SIL_2, description="Safety Integrity Level per IEC 61511")
    operating_mode: OperatingMode = Field(default=OperatingMode.RECOMMEND_ONLY, description="Agent operating mode")
    deployment_mode: DeploymentMode = Field(default=DeploymentMode.PRODUCTION, description="Deployment environment")
    kafka_config: KafkaConfig = Field(default_factory=KafkaConfig, description="Kafka integration settings")
    opcua_config: OPCUAConfig = Field(default_factory=OPCUAConfig, description="OPC-UA integration settings")
    cmms_config: CMMSConfig = Field(default_factory=CMMSConfig, description="CMMS integration settings")
    chemistry_limits: ChemistryLimitsConfig = Field(default_factory=ChemistryLimitsConfig, description="Water chemistry limits")
    phosphate_dosing: DosingConfig = Field(default_factory=lambda: DosingConfig(chemical_type=ChemicalType.PHOSPHATE), description="Phosphate dosing configuration")
    oxygen_scavenger_dosing: DosingConfig = Field(default_factory=lambda: DosingConfig(chemical_type=ChemicalType.OXYGEN_SCAVENGER), description="Oxygen scavenger dosing configuration")
    blowdown: BlowdownConfig = Field(default_factory=BlowdownConfig, description="Blowdown configuration")
    safety: SafetyConfig = Field(default_factory=SafetyConfig, description="Safety configuration")
    metrics: MetricsConfig = Field(default_factory=MetricsConfig, description="Metrics configuration")
    log_level: str = Field(default="INFO", description="Logging level")
    audit_enabled: bool = Field(default=True, description="Enable audit logging")
    provenance_tracking: bool = Field(default=True, description="Enable SHA-256 provenance tracking")
    deterministic_mode: bool = Field(default=True, description="Enable deterministic calculations (zero-hallucination)")
    random_seed: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    chemistry_calc_interval_s: int = Field(default=60, ge=10, le=3600, description="Chemistry calculation interval (seconds)")
    optimization_interval_s: int = Field(default=300, ge=60, le=3600, description="Optimization cycle interval (seconds)")
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    @classmethod
    def from_environment(cls) -> "WaterguardConfig":
        config_dict: Dict[str, Any] = {}
        prefix = "GL_WATERGUARD_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("_")
                current = config_dict
                for part in config_path[:-1]:
                    current = current.setdefault(part, {})
                current[config_path[-1]] = value
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WaterguardConfig":
        import yaml
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

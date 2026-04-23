"""
GL-007 FurnacePulse Configuration Module

This module defines all configuration classes, enums, and constraints
for the FurnacePulse Furnace Performance Monitoring Agent. It provides
type-safe configuration management with validation per NFPA 86 and
API 560 standards.

Configuration Hierarchy:
    - FurnacePulseConfig (master config)
        - FurnaceConstraints (operational limits)
        - NFPA86ComplianceConfig (safety checklist settings)
        - Integration configs (OPC-UA, Kafka, CMMS)

Example:
    >>> from core.config import FurnacePulseConfig, AlertTier
    >>> config = FurnacePulseConfig.from_yaml("furnace_config.yaml")
    >>> print(config.constraints.tmt_max_C)
    950.0

Author: GreenLang Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AlertTier(str, Enum):
    """
    Alert severity tiers for furnace monitoring.

    Tiers are aligned with NFPA 86 and plant safety management systems.

    Attributes:
        ADVISORY: Informational alert, no immediate action required
        WARNING: Attention needed, action required within shift
        URGENT: Immediate action required, potential safety impact
    """
    ADVISORY = "ADVISORY"
    WARNING = "WARNING"
    URGENT = "URGENT"

    @property
    def response_time_minutes(self) -> int:
        """Get recommended response time for each tier."""
        response_times = {
            AlertTier.ADVISORY: 480,   # 8 hours
            AlertTier.WARNING: 60,     # 1 hour
            AlertTier.URGENT: 5,       # 5 minutes
        }
        return response_times[self]

    @property
    def notification_channels(self) -> List[str]:
        """Get notification channels for each tier."""
        channels = {
            AlertTier.ADVISORY: ["dashboard", "log"],
            AlertTier.WARNING: ["dashboard", "log", "email"],
            AlertTier.URGENT: ["dashboard", "log", "email", "sms", "pager"],
        }
        return channels[self]


class SignalQuality(str, Enum):
    """
    Signal quality indicators for telemetry data.

    Quality codes follow OPC-UA quality standards and are used
    to determine data reliability for calculations.

    Attributes:
        GOOD: Signal is valid and within expected range
        BAD: Signal is invalid or sensor failure detected
        SUSPECT: Signal may be valid but outside expected bounds
        MISSING: No signal received within expected interval
    """
    GOOD = "GOOD"
    BAD = "BAD"
    SUSPECT = "SUSPECT"
    MISSING = "MISSING"

    @property
    def is_usable(self) -> bool:
        """Determine if signal quality is usable for calculations."""
        return self in (SignalQuality.GOOD, SignalQuality.SUSPECT)

    @property
    def confidence_factor(self) -> float:
        """Get confidence factor for weighted calculations."""
        factors = {
            SignalQuality.GOOD: 1.0,
            SignalQuality.SUSPECT: 0.7,
            SignalQuality.BAD: 0.0,
            SignalQuality.MISSING: 0.0,
        }
        return factors[self]


class FurnaceZone(str, Enum):
    """
    Furnace zones for temperature monitoring.

    Zones are defined per API 560 fired heater design standards.

    Attributes:
        RADIANT: Radiant section (direct flame exposure)
        CONVECTION: Convection section (flue gas heat transfer)
        SHIELD: Shield bank (transition between radiant/convection)
        CROSSOVER: Crossover tubes (connecting sections)
    """
    RADIANT = "RADIANT"
    CONVECTION = "CONVECTION"
    SHIELD = "SHIELD"
    CROSSOVER = "CROSSOVER"

    @property
    def typical_tmt_range_C(self) -> tuple:
        """Get typical TMT range for each zone (min, max)."""
        ranges = {
            FurnaceZone.RADIANT: (400, 950),
            FurnaceZone.CONVECTION: (300, 700),
            FurnaceZone.SHIELD: (350, 800),
            FurnaceZone.CROSSOVER: (350, 850),
        }
        return ranges[self]

    @property
    def criticality_weight(self) -> float:
        """Get criticality weight for zone prioritization."""
        weights = {
            FurnaceZone.RADIANT: 1.0,      # Highest priority
            FurnaceZone.CROSSOVER: 0.9,
            FurnaceZone.SHIELD: 0.8,
            FurnaceZone.CONVECTION: 0.7,
        }
        return weights[self]


class OperatingMode(str, Enum):
    """
    Furnace operating modes.

    Modes determine which calculations and limits apply.

    Attributes:
        STARTUP: Furnace is in startup sequence
        NORMAL: Normal steady-state operation
        TURNDOWN: Operating at reduced capacity
        SHUTDOWN: Controlled shutdown in progress
        EMERGENCY: Emergency shutdown or upset condition
        MAINTENANCE: Offline for maintenance
    """
    STARTUP = "STARTUP"
    NORMAL = "NORMAL"
    TURNDOWN = "TURNDOWN"
    SHUTDOWN = "SHUTDOWN"
    EMERGENCY = "EMERGENCY"
    MAINTENANCE = "MAINTENANCE"

    @property
    def allows_efficiency_calc(self) -> bool:
        """Determine if mode allows efficiency calculations."""
        return self in (OperatingMode.NORMAL, OperatingMode.TURNDOWN)

    @property
    def requires_enhanced_monitoring(self) -> bool:
        """Determine if mode requires enhanced monitoring."""
        return self in (
            OperatingMode.STARTUP,
            OperatingMode.SHUTDOWN,
            OperatingMode.EMERGENCY,
        )


class FlameQuality(str, Enum):
    """
    Flame quality indicators for burner monitoring.

    Quality assessments are based on UV/IR sensor analysis
    per NFPA 86 flame supervision requirements.

    Attributes:
        STABLE: Flame is stable and properly shaped
        UNSTABLE: Flame is fluctuating (combustion air/fuel imbalance)
        LIFTING: Flame is lifting off burner (excess velocity)
        IMPINGING: Flame is impinging on tubes (dangerous condition)
    """
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"
    LIFTING = "LIFTING"
    IMPINGING = "IMPINGING"

    @property
    def is_safe(self) -> bool:
        """Determine if flame quality is safe for continued operation."""
        return self == FlameQuality.STABLE

    @property
    def alert_tier(self) -> Optional[AlertTier]:
        """Get alert tier for abnormal flame conditions."""
        tiers = {
            FlameQuality.STABLE: None,
            FlameQuality.UNSTABLE: AlertTier.ADVISORY,
            FlameQuality.LIFTING: AlertTier.WARNING,
            FlameQuality.IMPINGING: AlertTier.URGENT,
        }
        return tiers[self]


class TrendDirection(str, Enum):
    """
    Trend direction indicators for time-series analysis.

    Attributes:
        IMPROVING: Metric is improving over time
        STABLE: Metric is stable within tolerance
        DEGRADING: Metric is degrading over time
    """
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"


class ComplianceStatus(str, Enum):
    """
    NFPA 86 compliance status indicators.

    Attributes:
        COMPLIANT: All checklist items pass
        NON_COMPLIANT: One or more critical items fail
        PENDING_REVIEW: Manual review required
    """
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"


class HotspotSeverity(str, Enum):
    """
    Hotspot severity levels for IR camera detection.

    Severity levels determine maintenance priority and
    shutdown requirements per API 530 tube thickness calculations.

    Attributes:
        LOW: Minor temperature elevation, monitor only
        MEDIUM: Moderate elevation, schedule inspection
        HIGH: Significant elevation, urgent inspection required
        CRITICAL: Immediate shutdown may be required
    """
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    @property
    def delta_t_threshold_C(self) -> float:
        """Get delta-T threshold above baseline for each severity."""
        thresholds = {
            HotspotSeverity.LOW: 25.0,
            HotspotSeverity.MEDIUM: 50.0,
            HotspotSeverity.HIGH: 75.0,
            HotspotSeverity.CRITICAL: 100.0,
        }
        return thresholds[self]

    @property
    def alert_tier(self) -> AlertTier:
        """Map severity to alert tier."""
        mapping = {
            HotspotSeverity.LOW: AlertTier.ADVISORY,
            HotspotSeverity.MEDIUM: AlertTier.ADVISORY,
            HotspotSeverity.HIGH: AlertTier.WARNING,
            HotspotSeverity.CRITICAL: AlertTier.URGENT,
        }
        return mapping[self]


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class FurnaceConstraints:
    """
    Operational constraints for furnace monitoring.

    These constraints define the safe operating envelope per
    API 560, API 530, and site-specific requirements.

    Attributes:
        tmt_max_C: Maximum allowable tube metal temperature
        tmt_warning_C: TMT threshold for warning alert
        tmt_rate_of_rise_max_C_min: Maximum TMT rate of rise
        draft_min_Pa: Minimum draft pressure (negative)
        draft_max_Pa: Maximum draft pressure
        excess_air_min_percent: Minimum excess air for combustion
        excess_air_max_percent: Maximum excess air (efficiency limit)
        efficiency_min_percent: Minimum acceptable efficiency
        efficiency_target_percent: Target efficiency for optimization
        stack_temp_max_C: Maximum stack temperature
        fuel_pressure_min_kPa: Minimum fuel gas pressure
        fuel_pressure_max_kPa: Maximum fuel gas pressure

    Example:
        >>> constraints = FurnaceConstraints(
        ...     tmt_max_C=950.0,
        ...     tmt_warning_C=900.0,
        ...     efficiency_target_percent=88.0
        ... )
        >>> is_safe = current_tmt < constraints.tmt_max_C
    """
    # TMT Limits (Tube Metal Temperature)
    tmt_max_C: float = 950.0
    tmt_warning_C: float = 900.0
    tmt_advisory_C: float = 850.0
    tmt_rate_of_rise_max_C_min: float = 10.0
    tmt_rate_of_rise_warning_C_min: float = 5.0

    # Draft Limits
    draft_min_Pa: float = -250.0  # Negative = suction
    draft_max_Pa: float = 25.0    # Positive = pressure
    draft_target_Pa: float = -50.0

    # Excess Air Limits
    excess_air_min_percent: float = 10.0
    excess_air_max_percent: float = 30.0
    excess_air_target_percent: float = 15.0

    # Efficiency Thresholds
    efficiency_min_percent: float = 75.0
    efficiency_warning_percent: float = 80.0
    efficiency_target_percent: float = 88.0

    # Stack Temperature Limits
    stack_temp_max_C: float = 450.0
    stack_temp_warning_C: float = 400.0

    # Fuel Pressure Limits
    fuel_pressure_min_kPa: float = 50.0
    fuel_pressure_max_kPa: float = 500.0

    # Hotspot Detection Thresholds
    hotspot_delta_t_low_C: float = 25.0
    hotspot_delta_t_medium_C: float = 50.0
    hotspot_delta_t_high_C: float = 75.0
    hotspot_delta_t_critical_C: float = 100.0

    # RUL Thresholds (hours)
    rul_warning_hours: float = 2000.0
    rul_urgent_hours: float = 500.0

    def validate(self) -> List[str]:
        """
        Validate constraint consistency.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.tmt_warning_C >= self.tmt_max_C:
            errors.append("tmt_warning_C must be less than tmt_max_C")

        if self.tmt_advisory_C >= self.tmt_warning_C:
            errors.append("tmt_advisory_C must be less than tmt_warning_C")

        if self.draft_min_Pa >= self.draft_max_Pa:
            errors.append("draft_min_Pa must be less than draft_max_Pa")

        if self.excess_air_min_percent >= self.excess_air_max_percent:
            errors.append("excess_air_min_percent must be less than max")

        if self.efficiency_min_percent > self.efficiency_target_percent:
            errors.append("efficiency_min must be less than target")

        return errors

    def get_tmt_alert_tier(self, tmt_C: float) -> Optional[AlertTier]:
        """
        Get alert tier for given TMT value.

        Args:
            tmt_C: Current tube metal temperature in Celsius

        Returns:
            Alert tier or None if within normal range
        """
        if tmt_C >= self.tmt_max_C:
            return AlertTier.URGENT
        elif tmt_C >= self.tmt_warning_C:
            return AlertTier.WARNING
        elif tmt_C >= self.tmt_advisory_C:
            return AlertTier.ADVISORY
        return None


@dataclass
class NFPA86ComplianceConfig:
    """
    NFPA 86 compliance checklist configuration.

    Defines the safety checklist items and verification requirements
    per NFPA 86 Standard for Ovens and Furnaces.

    Attributes:
        enabled: Enable/disable compliance checking
        checklist_items: List of required checklist items
        audit_interval_days: Days between compliance audits
        auto_lockout_on_failure: Auto-lockout on critical failures
        flame_supervision_required: Require flame supervision
        purge_cycle_required: Require purge before ignition
        purge_volume_exchanges: Number of air volume exchanges
        low_fire_start_required: Require low-fire start
        combustion_air_interlock: Require combustion air interlock
        fuel_shutoff_interlock: Require fuel shutoff interlock

    Example:
        >>> nfpa_config = NFPA86ComplianceConfig(
        ...     enabled=True,
        ...     audit_interval_days=90,
        ...     purge_volume_exchanges=4
        ... )
    """
    enabled: bool = True

    # Audit settings
    audit_interval_days: int = 90
    auto_lockout_on_failure: bool = True

    # Flame supervision (NFPA 86 Chapter 8)
    flame_supervision_required: bool = True
    flame_failure_response_seconds: float = 4.0
    flame_detector_type: str = "UV_IR"

    # Purge requirements (NFPA 86 Chapter 8)
    purge_cycle_required: bool = True
    purge_volume_exchanges: int = 4
    purge_airflow_verification: bool = True

    # Startup requirements
    low_fire_start_required: bool = True
    trial_for_ignition_seconds: float = 15.0

    # Interlocks (NFPA 86 Chapter 8)
    combustion_air_interlock: bool = True
    fuel_shutoff_interlock: bool = True
    high_temperature_interlock: bool = True
    low_draft_interlock: bool = True

    # Emergency shutdown
    emergency_shutdown_enabled: bool = True
    manual_reset_required: bool = True

    # Checklist items
    checklist_items: List[str] = field(default_factory=lambda: [
        "FLAME_SUPERVISION_OPERATIONAL",
        "COMBUSTION_AIR_INTERLOCK_VERIFIED",
        "FUEL_SHUTOFF_INTERLOCK_VERIFIED",
        "PURGE_TIMER_CALIBRATED",
        "HIGH_TEMP_INTERLOCK_TESTED",
        "LOW_DRAFT_INTERLOCK_TESTED",
        "EMERGENCY_SHUTDOWN_TESTED",
        "FLAME_DETECTOR_CALIBRATED",
        "COMBUSTION_CONTROLS_CALIBRATED",
        "RELIEF_VALVE_INSPECTED",
        "REFRACTORY_CONDITION_INSPECTED",
        "ELECTRICAL_SYSTEMS_INSPECTED",
        "VENTILATION_VERIFIED",
        "OPERATOR_TRAINING_CURRENT",
        "DOCUMENTATION_COMPLETE",
    ])

    def get_critical_items(self) -> List[str]:
        """Get list of critical checklist items that cannot fail."""
        return [
            "FLAME_SUPERVISION_OPERATIONAL",
            "COMBUSTION_AIR_INTERLOCK_VERIFIED",
            "FUEL_SHUTOFF_INTERLOCK_VERIFIED",
            "EMERGENCY_SHUTDOWN_TESTED",
        ]


@dataclass
class IntegrationConfig:
    """
    Integration configuration for external systems.

    Attributes:
        opcua_enabled: Enable OPC-UA integration
        opcua_endpoint: OPC-UA server endpoint URL
        opcua_namespace: OPC-UA namespace index
        kafka_enabled: Enable Kafka streaming
        kafka_brokers: Kafka broker list
        kafka_topic_prefix: Topic name prefix
        cmms_enabled: Enable CMMS integration
        cmms_api_url: CMMS REST API URL
        cmms_auth_type: Authentication type (oauth2, api_key)
    """
    # OPC-UA Settings
    opcua_enabled: bool = True
    opcua_endpoint: str = "opc.tcp://localhost:4840"
    opcua_namespace: int = 2
    opcua_security_mode: str = "SignAndEncrypt"
    opcua_certificate_path: Optional[str] = None
    opcua_polling_interval_ms: int = 1000

    # Kafka Settings
    kafka_enabled: bool = True
    kafka_brokers: str = "localhost:9092"
    kafka_topic_prefix: str = "furnacepulse"
    kafka_consumer_group: str = "furnacepulse-agent"
    kafka_auto_offset_reset: str = "latest"

    # CMMS Settings
    cmms_enabled: bool = True
    cmms_api_url: str = "http://localhost:8080/api/v1"
    cmms_auth_type: str = "oauth2"
    cmms_client_id: Optional[str] = None
    cmms_client_secret: Optional[str] = None

    # Database Settings
    db_connection_string: Optional[str] = None
    db_pool_size: int = 10


@dataclass
class MonitoringConfig:
    """
    Monitoring and observability configuration.

    Attributes:
        metrics_enabled: Enable Prometheus metrics
        metrics_port: Metrics endpoint port
        health_check_enabled: Enable health check endpoint
        health_check_port: Health check port
        log_level: Logging level
        trace_enabled: Enable distributed tracing
    """
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    health_check_enabled: bool = True
    health_check_port: int = 8080
    health_check_path: str = "/health"

    log_level: str = "INFO"
    log_format: str = "json"

    trace_enabled: bool = False
    trace_endpoint: Optional[str] = None
    trace_sample_rate: float = 0.1


@dataclass
class FurnacePulseConfig:
    """
    Master configuration for FurnacePulse agent.

    This is the top-level configuration class that aggregates all
    sub-configurations for the FurnacePulse agent.

    Attributes:
        agent_id: Unique agent identifier (GL-007)
        agent_name: Agent display name
        version: Configuration version
        constraints: Furnace operational constraints
        nfpa86: NFPA 86 compliance configuration
        integration: External system integration config
        monitoring: Monitoring and observability config
        furnace_ids: List of furnace IDs to monitor
        calculation_interval_seconds: KPI calculation interval
        rul_update_interval_hours: RUL prediction update interval

    Example:
        >>> config = FurnacePulseConfig.from_yaml("config.yaml")
        >>> print(config.agent_id)
        GL-007
        >>> print(config.constraints.tmt_max_C)
        950.0
    """
    # Agent Identity
    agent_id: str = "GL-007"
    agent_name: str = "FURNACEPULSE"
    version: str = "1.0.0"

    # Sub-configurations
    constraints: FurnaceConstraints = field(default_factory=FurnaceConstraints)
    nfpa86: NFPA86ComplianceConfig = field(default_factory=NFPA86ComplianceConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Furnace settings
    furnace_ids: List[str] = field(default_factory=list)

    # Calculation settings
    calculation_interval_seconds: int = 60
    rul_update_interval_hours: int = 24
    trend_analysis_window_days: int = 30

    # Data quality settings
    min_signal_quality_for_calc: SignalQuality = SignalQuality.SUSPECT
    missing_data_timeout_seconds: int = 300

    # Explainability settings
    explainability_enabled: bool = True
    shap_enabled: bool = True
    lime_enabled: bool = True

    # Audit settings
    audit_enabled: bool = True
    audit_retention_days: int = 730  # 2 years

    def validate(self) -> List[str]:
        """
        Validate complete configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate constraints
        errors.extend(self.constraints.validate())

        # Validate agent identity
        if not self.agent_id.startswith("GL-"):
            errors.append("agent_id must start with 'GL-'")

        # Validate intervals
        if self.calculation_interval_seconds < 10:
            errors.append("calculation_interval_seconds must be >= 10")

        if self.rul_update_interval_hours < 1:
            errors.append("rul_update_interval_hours must be >= 1")

        return errors

    @classmethod
    def from_yaml(cls, path: str) -> "FurnacePulseConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            FurnacePulseConfig instance

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config validation fails
        """
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        # Build nested config objects
        constraints = FurnaceConstraints(**data.get("constraints", {}))
        nfpa86 = NFPA86ComplianceConfig(**data.get("nfpa86", {}))
        integration = IntegrationConfig(**data.get("integration", {}))
        monitoring = MonitoringConfig(**data.get("monitoring", {}))

        # Build main config
        config = cls(
            agent_id=data.get("agent_id", "GL-007"),
            agent_name=data.get("agent_name", "FURNACEPULSE"),
            version=data.get("version", "1.0.0"),
            constraints=constraints,
            nfpa86=nfpa86,
            integration=integration,
            monitoring=monitoring,
            furnace_ids=data.get("furnace_ids", []),
            calculation_interval_seconds=data.get("calculation_interval_seconds", 60),
            rul_update_interval_hours=data.get("rul_update_interval_hours", 24),
            trend_analysis_window_days=data.get("trend_analysis_window_days", 30),
        )

        # Validate
        errors = config.validate()
        if errors:
            raise ValueError(f"Config validation failed: {errors}")

        logger.info(
            f"Loaded FurnacePulse config: agent_id={config.agent_id}, "
            f"furnaces={len(config.furnace_ids)}"
        )

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "constraints": {
                "tmt_max_C": self.constraints.tmt_max_C,
                "tmt_warning_C": self.constraints.tmt_warning_C,
                "draft_min_Pa": self.constraints.draft_min_Pa,
                "draft_max_Pa": self.constraints.draft_max_Pa,
                "efficiency_target_percent": self.constraints.efficiency_target_percent,
            },
            "nfpa86": {
                "enabled": self.nfpa86.enabled,
                "audit_interval_days": self.nfpa86.audit_interval_days,
                "checklist_items": self.nfpa86.checklist_items,
            },
            "furnace_ids": self.furnace_ids,
            "calculation_interval_seconds": self.calculation_interval_seconds,
        }

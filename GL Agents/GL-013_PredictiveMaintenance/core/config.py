# -*- coding: utf-8 -*-
"""
Configuration Module for GL-013 PredictiveMaintenance Agent.

Defines all configuration dataclasses for predictive maintenance
analytics with sensible defaults for industrial process heat equipment.

Author: GreenLang AI Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class EquipmentType(str, Enum):
    """Types of industrial process heat equipment."""
    BOILER = "boiler"
    FURNACE = "furnace"
    HEAT_EXCHANGER = "heat_exchanger"
    PUMP = "pump"
    COMPRESSOR = "compressor"
    MOTOR = "motor"
    TURBINE = "turbine"
    FAN = "fan"
    VALVE = "valve"
    BURNER = "burner"
    STEAM_TRAP = "steam_trap"
    CONDENSER = "condenser"


class MaintenanceStrategy(str, Enum):
    """Maintenance strategy types."""
    REACTIVE = "reactive"
    PREVENTIVE = "preventive"
    CONDITION_BASED = "condition_based"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"


class SeverityLevel(str, Enum):
    """Alert severity levels."""
    S0_INFO = "S0"
    S1_WARNING = "S1"
    S2_CRITICAL = "S2"
    S3_EMERGENCY = "S3"


class ModelType(str, Enum):
    """ML model types for predictions."""
    WEIBULL_AFT = "weibull_aft"
    COX_PH = "cox_ph"
    RANDOM_SURVIVAL_FOREST = "rsf"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    GRADIENT_BOOSTING = "gradient_boosting"
    ISOLATION_FOREST = "isolation_forest"
    AUTOENCODER = "autoencoder"


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    # RUL Estimation
    rul_model_type: ModelType = ModelType.WEIBULL_AFT
    rul_confidence_level: float = 0.95
    rul_min_samples: int = 100
    rul_horizon_days: int = 365

    # Anomaly Detection
    anomaly_model_type: ModelType = ModelType.ISOLATION_FOREST
    anomaly_contamination: float = 0.01
    anomaly_threshold: float = 0.5

    # Failure Prediction
    failure_model_type: ModelType = ModelType.GRADIENT_BOOSTING
    failure_threshold: float = 0.7
    failure_horizon_hours: int = 168  # 7 days

    # General
    random_seed: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10

    # Feature Engineering
    feature_window_hours: int = 24
    rolling_windows: List[int] = field(default_factory=lambda: [1, 6, 12, 24, 168])

    # Uncertainty Quantification
    n_bootstrap: int = 100
    calibration_method: str = "isotonic"


@dataclass
class AlertConfig:
    """Configuration for alerts and notifications."""

    # Severity thresholds
    rul_critical_days: int = 7
    rul_warning_days: int = 30
    failure_prob_critical: float = 0.8
    failure_prob_warning: float = 0.5
    anomaly_score_critical: float = 0.9
    anomaly_score_warning: float = 0.7

    # Alert rate limiting
    min_alert_interval_seconds: int = 300
    max_alerts_per_hour: int = 20
    deduplication_window_seconds: int = 3600

    # Notification channels
    enable_email: bool = False
    enable_sms: bool = False
    enable_webhook: bool = True
    enable_cmms_ticket: bool = True

    # Alert escalation
    escalation_delay_minutes: int = 30
    auto_escalate: bool = True


@dataclass
class IntegrationConfig:
    """Configuration for external integrations."""

    # OPC-UA
    opcua_enabled: bool = True
    opcua_endpoint: str = "opc.tcp://localhost:4840"
    opcua_namespace: str = "http://greenlang.ai/predictive-maintenance"
    opcua_security_mode: str = "SignAndEncrypt"
    opcua_polling_interval_ms: int = 1000

    # Kafka
    kafka_enabled: bool = True
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_prefix: str = "gl-013-pdm"
    kafka_consumer_group: str = "predictive-maintenance"
    kafka_batch_size: int = 100
    kafka_linger_ms: int = 100

    # CMMS
    cmms_enabled: bool = True
    cmms_type: str = "sap_pm"  # sap_pm, maximo, fiix, etc.
    cmms_api_url: str = ""
    cmms_sync_interval_minutes: int = 15

    # GraphQL API
    graphql_enabled: bool = True
    graphql_port: int = 8080
    graphql_path: str = "/graphql"
    graphql_cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Historian
    historian_enabled: bool = False
    historian_type: str = "osisoft_pi"
    historian_url: str = ""


@dataclass
class SafetyConfig:
    """Configuration for safety and governance controls."""

    # Human-in-the-loop
    require_human_approval_rul_critical: bool = True
    require_human_approval_emergency_stop: bool = True
    human_approval_timeout_minutes: int = 60

    # Uncertainty gating
    max_prediction_uncertainty: float = 0.3
    reject_high_uncertainty: bool = True
    uncertainty_escalation_threshold: float = 0.25

    # Model governance
    model_drift_threshold: float = 0.1
    retrain_trigger_accuracy_drop: float = 0.05
    max_model_age_days: int = 90
    require_model_validation: bool = True

    # Data quality
    min_data_completeness: float = 0.95
    max_sensor_staleness_minutes: int = 10
    outlier_rejection_sigma: float = 4.0

    # Audit
    enable_audit_logging: bool = True
    audit_retention_days: int = 365
    enable_provenance_tracking: bool = True


@dataclass
class SignalProcessingConfig:
    """Configuration for signal processing."""

    # Vibration analysis
    vibration_sample_rate_hz: float = 25600.0
    vibration_window_type: str = "hanning"
    vibration_fft_size: int = 8192

    # MCSA analysis
    mcsa_sample_rate_hz: float = 10000.0
    mcsa_fundamental_freq_hz: float = 50.0
    mcsa_harmonics: int = 10

    # Thermal analysis
    thermal_sample_rate_hz: float = 1.0
    thermal_moving_avg_window: int = 60

    # Envelope analysis
    envelope_highpass_hz: float = 500.0
    envelope_lowpass_hz: float = 5000.0

    # Bearing analysis
    bearing_defect_tolerance: float = 0.05
    bearing_amplitude_threshold: float = 0.1


@dataclass
class PredictiveMaintenanceConfig:
    """Main configuration for GL-013 PredictiveMaintenance agent."""

    # Agent identification
    agent_id: str = "GL-013"
    agent_name: str = "PredictiveMaintenance"
    version: str = "1.0.0"

    # Facility information
    facility_id: str = ""
    facility_name: str = ""
    timezone: str = "UTC"

    # Equipment types to monitor
    equipment_types: List[EquipmentType] = field(default_factory=lambda: [
        EquipmentType.BOILER,
        EquipmentType.FURNACE,
        EquipmentType.PUMP,
        EquipmentType.MOTOR,
        EquipmentType.FAN,
    ])

    # Maintenance strategy
    default_strategy: MaintenanceStrategy = MaintenanceStrategy.PREDICTIVE

    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    integrations: IntegrationConfig = field(default_factory=IntegrationConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    signal_processing: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)

    # Performance settings
    batch_size: int = 32
    max_workers: int = 4
    cache_ttl_seconds: int = 300

    # Logging
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        if self.model.rul_confidence_level < 0.5 or self.model.rul_confidence_level > 0.99:
            issues.append("RUL confidence level must be between 0.5 and 0.99")

        if self.safety.max_prediction_uncertainty < 0 or self.safety.max_prediction_uncertainty > 1:
            issues.append("Max prediction uncertainty must be between 0 and 1")

        if self.safety.min_data_completeness < 0 or self.safety.min_data_completeness > 1:
            issues.append("Min data completeness must be between 0 and 1")

        if self.model.anomaly_contamination < 0 or self.model.anomaly_contamination > 0.5:
            issues.append("Anomaly contamination must be between 0 and 0.5")

        return issues

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PredictiveMaintenanceConfig":
        """Create configuration from dictionary."""
        # Handle nested configs
        if "model" in data and isinstance(data["model"], dict):
            data["model"] = ModelConfig(**data["model"])
        if "alerts" in data and isinstance(data["alerts"], dict):
            data["alerts"] = AlertConfig(**data["alerts"])
        if "integrations" in data and isinstance(data["integrations"], dict):
            data["integrations"] = IntegrationConfig(**data["integrations"])
        if "safety" in data and isinstance(data["safety"], dict):
            data["safety"] = SafetyConfig(**data["safety"])
        if "signal_processing" in data and isinstance(data["signal_processing"], dict):
            data["signal_processing"] = SignalProcessingConfig(**data["signal_processing"])

        # Handle equipment types
        if "equipment_types" in data:
            data["equipment_types"] = [
                EquipmentType(et) if isinstance(et, str) else et
                for et in data["equipment_types"]
            ]

        return cls(**data)

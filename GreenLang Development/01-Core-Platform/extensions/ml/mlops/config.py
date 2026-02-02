"""
MLOps Configuration - Configuration management for MLOps components.

This module provides centralized configuration for all MLOps pipeline components
including MLflow integration, drift detection, monitoring, and alerting.

Example:
    >>> from greenlang.ml.mlops.config import MLOpsConfig
    >>> config = MLOpsConfig()
    >>> print(config.model_registry.storage_path)
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# =============================================================================
# Sub-Configuration Classes
# =============================================================================

class MLflowConfig(BaseModel):
    """MLflow backend configuration."""

    enabled: bool = Field(
        default=False, description="Whether MLflow backend is enabled"
    )
    tracking_uri: str = Field(
        default="file:///tmp/mlruns", description="MLflow tracking URI"
    )
    experiment_name: str = Field(
        default="greenlang-process-heat", description="Default experiment name"
    )
    artifact_location: Optional[str] = Field(
        None, description="Custom artifact storage location"
    )
    registry_uri: Optional[str] = Field(
        None, description="Model registry URI (if different from tracking)"
    )


class ModelRegistryConfig(BaseModel):
    """Model registry configuration."""

    storage_path: str = Field(
        default="./mlops_data/model_registry",
        description="Path for file-based model storage"
    )
    max_versions_per_model: int = Field(
        default=10, ge=1, le=100, description="Maximum versions to retain per model"
    )
    artifact_format: str = Field(
        default="joblib", description="Default serialization format (joblib, pickle)"
    )
    compression: bool = Field(
        default=True, description="Enable artifact compression"
    )
    metadata_file: str = Field(
        default="registry_metadata.json", description="Metadata file name"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class DriftDetectionConfig(BaseModel):
    """Drift detection configuration."""

    # Statistical test thresholds
    ks_test_threshold: float = Field(
        default=0.05, ge=0.001, le=0.5,
        description="KS test p-value threshold for drift detection"
    )
    chi_squared_threshold: float = Field(
        default=0.05, ge=0.001, le=0.5,
        description="Chi-squared test p-value threshold"
    )
    psi_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="PSI threshold for drift detection"
    )
    kl_divergence_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0,
        description="KL divergence threshold"
    )

    # Severity thresholds
    low_drift_threshold: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Low drift severity threshold"
    )
    medium_drift_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0, description="Medium drift severity threshold"
    )
    high_drift_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="High drift severity threshold"
    )
    critical_drift_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Critical drift severity threshold"
    )

    # Analysis settings
    min_samples_for_drift: int = Field(
        default=100, ge=30, description="Minimum samples required for drift analysis"
    )
    num_bins_psi: int = Field(
        default=10, ge=5, le=50, description="Number of bins for PSI calculation"
    )
    reference_window_size: int = Field(
        default=10000, ge=1000, description="Reference data window size"
    )
    storage_path: str = Field(
        default="./mlops_data/drift_reports",
        description="Path for drift report storage"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class ABTestingConfig(BaseModel):
    """A/B testing configuration."""

    storage_path: str = Field(
        default="./mlops_data/ab_experiments",
        description="Path for experiment data storage"
    )
    default_traffic_split: float = Field(
        default=0.1, ge=0.01, le=0.5,
        description="Default traffic fraction to challenger"
    )
    default_min_samples: int = Field(
        default=1000, ge=100, description="Default minimum samples for analysis"
    )
    default_significance_level: float = Field(
        default=0.05, ge=0.001, le=0.1, description="Default significance level"
    )
    max_concurrent_experiments: int = Field(
        default=5, ge=1, le=20, description="Maximum concurrent experiments per model"
    )
    auto_cleanup_after_days: int = Field(
        default=30, ge=1, description="Days after which completed experiments are cleaned"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class RetrainingConfig(BaseModel):
    """Retraining pipeline configuration."""

    storage_path: str = Field(
        default="./mlops_data/retraining",
        description="Path for retraining data storage"
    )
    default_drift_threshold: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Default drift threshold for retraining trigger"
    )
    default_schedule: str = Field(
        default="0 0 * * 0", description="Default cron schedule (weekly Sunday midnight)"
    )
    default_min_samples: int = Field(
        default=5000, ge=100, description="Default minimum new samples for data trigger"
    )
    default_validation_split: float = Field(
        default=0.2, ge=0.1, le=0.5, description="Default validation data fraction"
    )
    default_min_improvement: float = Field(
        default=0.01, ge=0.0, description="Default minimum improvement to deploy"
    )
    max_training_time_seconds: int = Field(
        default=3600, ge=60, description="Maximum training time in seconds"
    )
    cooldown_period_hours: int = Field(
        default=24, ge=1, description="Minimum hours between retraining runs"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""

    storage_path: str = Field(
        default="./mlops_data/monitoring",
        description="Path for monitoring data storage"
    )

    # Time windows for metrics aggregation
    time_windows: List[str] = Field(
        default=["1h", "24h", "7d"],
        description="Time windows for metrics aggregation"
    )

    # Prediction logging
    log_features: bool = Field(
        default=True, description="Whether to log input features"
    )
    max_feature_log_size: int = Field(
        default=1000, ge=100, description="Maximum feature dict size to log"
    )
    retention_days: int = Field(
        default=30, ge=1, le=365, description="Days to retain prediction logs"
    )

    # Performance thresholds for alerts
    latency_p95_threshold_ms: float = Field(
        default=100.0, ge=1.0, description="P95 latency threshold for alerts"
    )
    latency_p99_threshold_ms: float = Field(
        default=500.0, ge=1.0, description="P99 latency threshold for alerts"
    )
    error_rate_threshold: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Error rate threshold for alerts"
    )
    mae_degradation_threshold: float = Field(
        default=0.1, ge=0.0, description="MAE degradation threshold (relative)"
    )
    rmse_degradation_threshold: float = Field(
        default=0.1, ge=0.0, description="RMSE degradation threshold (relative)"
    )

    # Prometheus export
    prometheus_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics export"
    )
    prometheus_port: int = Field(
        default=9090, ge=1024, le=65535, description="Prometheus metrics port"
    )
    prometheus_prefix: str = Field(
        default="greenlang_mlops", description="Prometheus metric name prefix"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


class AlertConfig(BaseModel):
    """Alert configuration."""

    enabled: bool = Field(default=True, description="Enable alerting")
    storage_path: str = Field(
        default="./mlops_data/alerts",
        description="Path for alert storage"
    )

    # Alert channels
    email_enabled: bool = Field(default=False, description="Enable email alerts")
    email_recipients: List[str] = Field(
        default_factory=list, description="Email recipients for alerts"
    )
    smtp_host: Optional[str] = Field(None, description="SMTP host for email")
    smtp_port: int = Field(default=587, description="SMTP port")

    slack_enabled: bool = Field(default=False, description="Enable Slack alerts")
    slack_webhook_url: Optional[str] = Field(None, description="Slack webhook URL")

    # Alert behavior
    cooldown_minutes: int = Field(
        default=15, ge=1, description="Minutes between repeated alerts"
    )
    max_alerts_per_hour: int = Field(
        default=10, ge=1, le=100, description="Maximum alerts per hour per model"
    )
    auto_acknowledge_after_hours: int = Field(
        default=24, ge=1, description="Auto-acknowledge alerts after this many hours"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v: str) -> str:
        """Ensure storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())


# =============================================================================
# Main Configuration Class
# =============================================================================

class MLOpsConfig(BaseModel):
    """
    Main configuration class for MLOps pipeline framework.

    This class aggregates all sub-configurations and provides a single
    entry point for configuring the entire MLOps system.

    Attributes:
        mlflow: MLflow backend configuration
        model_registry: Model registry configuration
        drift_detection: Drift detection configuration
        ab_testing: A/B testing configuration
        retraining: Retraining pipeline configuration
        monitoring: Monitoring configuration
        alert: Alert configuration

    Example:
        >>> config = MLOpsConfig()
        >>> config.model_registry.storage_path
        '/path/to/model_registry'

        >>> # Or load from environment
        >>> config = MLOpsConfig.from_env()

        >>> # Or from dict
        >>> config = MLOpsConfig(**config_dict)
    """

    mlflow: MLflowConfig = Field(
        default_factory=MLflowConfig, description="MLflow configuration"
    )
    model_registry: ModelRegistryConfig = Field(
        default_factory=ModelRegistryConfig, description="Model registry configuration"
    )
    drift_detection: DriftDetectionConfig = Field(
        default_factory=DriftDetectionConfig, description="Drift detection configuration"
    )
    ab_testing: ABTestingConfig = Field(
        default_factory=ABTestingConfig, description="A/B testing configuration"
    )
    retraining: RetrainingConfig = Field(
        default_factory=RetrainingConfig, description="Retraining pipeline configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )
    alert: AlertConfig = Field(
        default_factory=AlertConfig, description="Alert configuration"
    )

    # Global settings
    base_storage_path: str = Field(
        default="./mlops_data", description="Base path for all MLOps data storage"
    )
    log_level: str = Field(
        default="INFO", description="Logging level"
    )
    enable_provenance_tracking: bool = Field(
        default=True, description="Enable SHA-256 provenance tracking"
    )
    enable_audit_logging: bool = Field(
        default=True, description="Enable audit logging"
    )
    thread_pool_size: int = Field(
        default=4, ge=1, le=32, description="Thread pool size for concurrent operations"
    )

    @validator("base_storage_path")
    def validate_base_storage_path(cls, v: str) -> str:
        """Ensure base storage path exists."""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return str(path.absolute())

    @classmethod
    def from_env(cls) -> "MLOpsConfig":
        """
        Create configuration from environment variables.

        Environment variables are prefixed with GREENLANG_MLOPS_.
        Example: GREENLANG_MLOPS_LOG_LEVEL=DEBUG

        Returns:
            MLOpsConfig instance populated from environment variables.
        """
        env_prefix = "GREENLANG_MLOPS_"

        config_dict: Dict[str, Any] = {}

        # Global settings
        if os.getenv(f"{env_prefix}BASE_STORAGE_PATH"):
            config_dict["base_storage_path"] = os.getenv(
                f"{env_prefix}BASE_STORAGE_PATH"
            )
        if os.getenv(f"{env_prefix}LOG_LEVEL"):
            config_dict["log_level"] = os.getenv(f"{env_prefix}LOG_LEVEL")

        # MLflow settings
        mlflow_config: Dict[str, Any] = {}
        if os.getenv(f"{env_prefix}MLFLOW_ENABLED"):
            mlflow_config["enabled"] = (
                os.getenv(f"{env_prefix}MLFLOW_ENABLED", "").lower() == "true"
            )
        if os.getenv(f"{env_prefix}MLFLOW_TRACKING_URI"):
            mlflow_config["tracking_uri"] = os.getenv(
                f"{env_prefix}MLFLOW_TRACKING_URI"
            )
        if os.getenv(f"{env_prefix}MLFLOW_EXPERIMENT_NAME"):
            mlflow_config["experiment_name"] = os.getenv(
                f"{env_prefix}MLFLOW_EXPERIMENT_NAME"
            )
        if mlflow_config:
            config_dict["mlflow"] = mlflow_config

        # Monitoring settings
        monitoring_config: Dict[str, Any] = {}
        if os.getenv(f"{env_prefix}PROMETHEUS_ENABLED"):
            monitoring_config["prometheus_enabled"] = (
                os.getenv(f"{env_prefix}PROMETHEUS_ENABLED", "").lower() == "true"
            )
        if os.getenv(f"{env_prefix}PROMETHEUS_PORT"):
            monitoring_config["prometheus_port"] = int(
                os.getenv(f"{env_prefix}PROMETHEUS_PORT", "9090")
            )
        if monitoring_config:
            config_dict["monitoring"] = monitoring_config

        # Alert settings
        alert_config: Dict[str, Any] = {}
        if os.getenv(f"{env_prefix}ALERT_EMAIL_ENABLED"):
            alert_config["email_enabled"] = (
                os.getenv(f"{env_prefix}ALERT_EMAIL_ENABLED", "").lower() == "true"
            )
        if os.getenv(f"{env_prefix}ALERT_SLACK_WEBHOOK_URL"):
            alert_config["slack_enabled"] = True
            alert_config["slack_webhook_url"] = os.getenv(
                f"{env_prefix}ALERT_SLACK_WEBHOOK_URL"
            )
        if alert_config:
            config_dict["alert"] = alert_config

        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: str) -> "MLOpsConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            MLOpsConfig instance populated from YAML file.

        Raises:
            FileNotFoundError: If YAML file does not exist.
            ValueError: If YAML parsing fails.
        """
        import yaml  # Optional dependency

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return self.dict()

    def save_yaml(self, path: str) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save YAML configuration file.
        """
        import yaml  # Optional dependency

        yaml_path = Path(path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)

    def get_storage_path(self, component: str) -> Path:
        """
        Get storage path for a specific component.

        Args:
            component: Component name (model_registry, drift_detection, etc.)

        Returns:
            Path object for the component's storage directory.
        """
        component_configs = {
            "model_registry": self.model_registry.storage_path,
            "drift_detection": self.drift_detection.storage_path,
            "ab_testing": self.ab_testing.storage_path,
            "retraining": self.retraining.storage_path,
            "monitoring": self.monitoring.storage_path,
            "alert": self.alert.storage_path,
        }

        if component not in component_configs:
            raise ValueError(f"Unknown component: {component}")

        path = Path(component_configs[component])
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# Default Configuration Instance
# =============================================================================

# Create a default configuration instance for convenience
default_config = MLOpsConfig()


def get_config() -> MLOpsConfig:
    """
    Get the default MLOps configuration.

    Returns:
        Default MLOpsConfig instance.
    """
    return default_config


def set_config(config: MLOpsConfig) -> None:
    """
    Set the default MLOps configuration.

    Args:
        config: New MLOpsConfig instance to use as default.
    """
    global default_config
    default_config = config

"""
GL-001 ThermalCommand Orchestrator - Configuration Module

This module defines all configuration schemas for the ThermalCommand
Orchestrator including safety, integration, MLOps, and operational settings.

All configurations use Pydantic for validation with comprehensive defaults
and documentation.
"""

from datetime import timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set
import os

from pydantic import BaseModel, Field, validator


# =============================================================================
# ENUMS
# =============================================================================

class SafetyLevel(Enum):
    """IEC 61511 Safety Integrity Levels."""
    NONE = 0
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class ProtocolType(Enum):
    """Supported communication protocols."""
    OPC_UA = "opc-ua"
    MODBUS_TCP = "modbus-tcp"
    MQTT = "mqtt"
    KAFKA = "kafka"
    HTTP_REST = "http"
    GRPC = "grpc"


class DeploymentMode(Enum):
    """Deployment modes."""
    STANDALONE = "standalone"
    DISTRIBUTED = "distributed"
    HIGH_AVAILABILITY = "ha"
    EDGE = "edge"


class CoordinationStrategy(Enum):
    """Agent coordination strategies."""
    CONTRACT_NET = "contract_net"
    HIERARCHICAL = "hierarchical"
    BLACKBOARD = "blackboard"
    AUCTION = "auction"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================

class SafetyConfig(BaseModel):
    """Safety system configuration."""

    level: SafetyLevel = Field(
        default=SafetyLevel.SIL_3,
        description="Safety Integrity Level per IEC 61511"
    )
    emergency_shutdown_enabled: bool = Field(
        default=True,
        description="Enable Emergency Shutdown (ESD) integration"
    )
    esd_endpoint: Optional[str] = Field(
        default=None,
        description="ESD system communication endpoint"
    )
    fail_safe_mode: str = Field(
        default="safe_state",
        description="Fail-safe behavior: safe_state, last_known, shutdown"
    )
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
    max_consecutive_failures: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum consecutive failures before ESD"
    )
    safety_interlock_enabled: bool = Field(
        default=True,
        description="Enable safety interlock checking"
    )
    redundant_monitoring: bool = Field(
        default=True,
        description="Enable redundant safety monitoring"
    )
    alarm_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "high_temperature_f": 1800.0,
            "low_pressure_psig": 5.0,
            "high_pressure_psig": 500.0,
            "high_emissions_ppm": 100.0,
        },
        description="Safety alarm thresholds"
    )

    class Config:
        use_enum_values = True


class IntegrationConfig(BaseModel):
    """External system integration configuration."""

    # OPC-UA
    opcua_enabled: bool = Field(
        default=True,
        description="Enable OPC-UA connectivity"
    )
    opcua_endpoints: List[str] = Field(
        default_factory=list,
        description="OPC-UA server endpoints"
    )
    opcua_security_policy: str = Field(
        default="Basic256Sha256",
        description="OPC-UA security policy"
    )

    # MQTT
    mqtt_enabled: bool = Field(
        default=True,
        description="Enable MQTT messaging"
    )
    mqtt_broker: str = Field(
        default="localhost:1883",
        description="MQTT broker address"
    )
    mqtt_topics: Dict[str, str] = Field(
        default_factory=lambda: {
            "status": "greenlang/process_heat/status",
            "commands": "greenlang/process_heat/commands",
            "events": "greenlang/process_heat/events",
            "safety": "greenlang/process_heat/safety",
        },
        description="MQTT topic mappings"
    )

    # Kafka
    kafka_enabled: bool = Field(
        default=False,
        description="Enable Kafka event streaming"
    )
    kafka_bootstrap_servers: str = Field(
        default="localhost:9092",
        description="Kafka bootstrap servers"
    )
    kafka_topics: Dict[str, str] = Field(
        default_factory=lambda: {
            "events": "process-heat-events",
            "metrics": "process-heat-metrics",
            "audit": "process-heat-audit",
        },
        description="Kafka topic mappings"
    )

    # Database
    database_url: str = Field(
        default="postgresql://localhost:5432/greenlang",
        description="Database connection URL"
    )
    timeseries_url: Optional[str] = Field(
        default=None,
        description="TimescaleDB/InfluxDB URL for metrics"
    )

    # External APIs
    erp_integration_enabled: bool = Field(
        default=False,
        description="Enable ERP system integration"
    )
    erp_api_url: Optional[str] = Field(
        default=None,
        description="ERP API endpoint"
    )
    weather_api_enabled: bool = Field(
        default=False,
        description="Enable weather data integration"
    )
    weather_api_key: Optional[str] = Field(
        default=None,
        description="Weather API key"
    )


class MLOpsConfig(BaseModel):
    """Machine Learning Operations configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable ML capabilities"
    )

    # Model Registry
    model_registry_url: str = Field(
        default="http://localhost:5000",
        description="MLflow model registry URL"
    )
    model_cache_dir: str = Field(
        default="/tmp/greenlang/models",
        description="Local model cache directory"
    )

    # Inference
    inference_timeout_ms: int = Field(
        default=5000,
        ge=100,
        le=60000,
        description="Model inference timeout"
    )
    batch_inference_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Batch size for inference"
    )

    # Explainability
    explainability_enabled: bool = Field(
        default=True,
        description="Enable SHAP/LIME explainability"
    )
    shap_sample_size: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Sample size for SHAP explanations"
    )

    # Uncertainty
    uncertainty_quantification: bool = Field(
        default=True,
        description="Enable uncertainty bounds calculation"
    )
    confidence_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for ML predictions"
    )

    # Drift Detection
    drift_detection_enabled: bool = Field(
        default=True,
        description="Enable data/concept drift detection"
    )
    drift_check_interval_s: int = Field(
        default=3600,
        ge=60,
        description="Drift check interval in seconds"
    )
    drift_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Drift detection threshold"
    )

    # A/B Testing
    ab_testing_enabled: bool = Field(
        default=False,
        description="Enable A/B testing for models"
    )
    shadow_deployment_enabled: bool = Field(
        default=True,
        description="Enable shadow model deployment"
    )


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    prefix: str = Field(
        default="greenlang_thermal_command",
        description="Metrics name prefix"
    )
    port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics HTTP port"
    )
    push_gateway_url: Optional[str] = Field(
        default=None,
        description="Prometheus push gateway URL"
    )
    collection_interval_s: float = Field(
        default=15.0,
        ge=1.0,
        le=300.0,
        description="Metrics collection interval"
    )
    histogram_buckets: List[float] = Field(
        default=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        description="Histogram bucket boundaries"
    )
    custom_labels: Dict[str, str] = Field(
        default_factory=lambda: {
            "environment": "production",
            "region": "us-west",
        },
        description="Custom metric labels"
    )


class TracingConfig(BaseModel):
    """Distributed tracing configuration."""

    enabled: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    exporter: str = Field(
        default="jaeger",
        description="Trace exporter (jaeger, zipkin, otlp)"
    )
    endpoint: str = Field(
        default="http://localhost:14268/api/traces",
        description="Trace collector endpoint"
    )
    sample_rate: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate"
    )
    service_name: str = Field(
        default="thermal-command-orchestrator",
        description="Service name for traces"
    )


class CoordinationConfig(BaseModel):
    """Multi-agent coordination configuration."""

    strategy: CoordinationStrategy = Field(
        default=CoordinationStrategy.CONTRACT_NET,
        description="Default coordination strategy"
    )
    bid_timeout_s: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Task bidding timeout"
    )
    max_concurrent_workflows: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum concurrent workflows"
    )
    agent_discovery_enabled: bool = Field(
        default=True,
        description="Enable automatic agent discovery"
    )
    agent_heartbeat_timeout_s: float = Field(
        default=60.0,
        ge=10.0,
        le=300.0,
        description="Agent heartbeat timeout"
    )
    conflict_resolution: str = Field(
        default="priority",
        description="Conflict resolution strategy"
    )
    consensus_threshold: float = Field(
        default=0.67,
        ge=0.5,
        le=1.0,
        description="Consensus threshold for voting"
    )

    class Config:
        use_enum_values = True


class APIConfig(BaseModel):
    """REST/GraphQL API configuration."""

    rest_enabled: bool = Field(
        default=True,
        description="Enable REST API"
    )
    rest_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="REST API port"
    )
    graphql_enabled: bool = Field(
        default=True,
        description="Enable GraphQL API"
    )
    graphql_port: int = Field(
        default=8001,
        ge=1024,
        le=65535,
        description="GraphQL API port"
    )
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000"],
        description="CORS allowed origins"
    )
    rate_limit_rpm: int = Field(
        default=1000,
        ge=10,
        description="Rate limit (requests per minute)"
    )
    auth_enabled: bool = Field(
        default=True,
        description="Enable API authentication"
    )
    auth_provider: str = Field(
        default="jwt",
        description="Authentication provider (jwt, oauth2, api_key)"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="API key header name"
    )


class OrchestratorConfig(BaseModel):
    """
    Complete orchestrator configuration.

    This is the main configuration class for the ThermalCommand Orchestrator.
    It aggregates all sub-configurations and provides validation.
    """

    # Identity
    orchestrator_id: str = Field(
        default_factory=lambda: f"GL-001-{os.getpid()}",
        description="Unique orchestrator identifier"
    )
    name: str = Field(
        default="ThermalCommand-Primary",
        description="Human-readable orchestrator name"
    )
    version: str = Field(
        default="1.0.0",
        description="Orchestrator version"
    )

    # Deployment
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.DISTRIBUTED,
        description="Deployment mode"
    )
    environment: str = Field(
        default="production",
        description="Environment (development, staging, production)"
    )

    # Sub-configurations
    safety: SafetyConfig = Field(
        default_factory=SafetyConfig,
        description="Safety configuration"
    )
    integration: IntegrationConfig = Field(
        default_factory=IntegrationConfig,
        description="Integration configuration"
    )
    mlops: MLOpsConfig = Field(
        default_factory=MLOpsConfig,
        description="MLOps configuration"
    )
    metrics: MetricsConfig = Field(
        default_factory=MetricsConfig,
        description="Metrics configuration"
    )
    tracing: TracingConfig = Field(
        default_factory=TracingConfig,
        description="Tracing configuration"
    )
    coordination: CoordinationConfig = Field(
        default_factory=CoordinationConfig,
        description="Coordination configuration"
    )
    api: APIConfig = Field(
        default_factory=APIConfig,
        description="API configuration"
    )

    # Operational
    max_concurrent_tasks: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum concurrent tasks"
    )
    task_timeout_s: float = Field(
        default=300.0,
        ge=1.0,
        le=3600.0,
        description="Default task timeout"
    )
    audit_enabled: bool = Field(
        default=True,
        description="Enable audit logging"
    )
    provenance_tracking: bool = Field(
        default=True,
        description="Enable SHA-256 provenance tracking"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )

    class Config:
        use_enum_values = True

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment is valid."""
        valid_envs = {"development", "staging", "production"}
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()

    @classmethod
    def from_environment(cls) -> "OrchestratorConfig":
        """
        Create configuration from environment variables.

        Environment variables follow the pattern:
        GL_ORCHESTRATOR_<SECTION>_<KEY>=value

        Example:
            GL_ORCHESTRATOR_SAFETY_LEVEL=SIL_3
            GL_ORCHESTRATOR_API_REST_PORT=8080
        """
        config_dict = {}

        # Parse relevant environment variables
        prefix = "GL_ORCHESTRATOR_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_path = key[len(prefix):].lower().split("_")
                # Build nested dict from path
                current = config_dict
                for part in config_path[:-1]:
                    current = current.setdefault(part, {})
                current[config_path[-1]] = value

        return cls(**config_dict)

"""
GL-004 BURNMASTER Configuration Module

Configuration schemas and settings for the Burner Optimization Agent.
Defines operating modes, safety constraints, optimization parameters,
and integration settings.

Author: GreenLang AI Agent Workforce
Version: 1.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class OperatingMode(str, Enum):
    """Operating modes for the burner optimization agent."""
    OBSERVE_ONLY = "observe_only"  # Monitoring only, no recommendations
    ADVISORY = "advisory"  # Recommendations provided, human implements
    CLOSED_LOOP = "closed_loop"  # Automatic setpoint adjustments
    FALLBACK = "fallback"  # Safety fallback mode


class FuelType(str, Enum):
    """Supported fuel types for combustion calculations."""
    NATURAL_GAS = "natural_gas"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL = "coal"
    BIOMASS = "biomass"
    HYDROGEN = "hydrogen"
    MIXED = "mixed"


class BurnerType(str, Enum):
    """Burner configuration types."""
    PREMIX = "premix"
    NOZZLE_MIX = "nozzle_mix"
    RAW_GAS = "raw_gas"
    LOW_NOX = "low_nox"
    ULTRA_LOW_NOX = "ultra_low_nox"


class SafetyLimits(BaseModel):
    """Safety limits per NFPA 85/86 and process requirements."""

    # Excess O2 limits (%)
    min_excess_o2: float = Field(default=1.5, ge=0.5, le=5.0)
    max_excess_o2: float = Field(default=6.0, ge=2.0, le=15.0)

    # CO limits (ppm)
    max_co_ppm: float = Field(default=100.0, ge=10.0, le=400.0)
    alarm_co_ppm: float = Field(default=200.0, ge=50.0, le=500.0)

    # NOx limits (ppm or lb/MMBtu)
    max_nox_ppm: float = Field(default=50.0, ge=5.0, le=200.0)
    max_nox_lb_mmbtu: float = Field(default=0.1, ge=0.01, le=0.5)

    # Temperature limits (°C)
    min_flame_temp: float = Field(default=800.0, ge=500.0)
    max_flame_temp: float = Field(default=1800.0, le=2000.0)
    max_stack_temp: float = Field(default=250.0, le=400.0)

    # Pressure limits (mbar)
    min_furnace_pressure: float = Field(default=-10.0, ge=-50.0)
    max_furnace_pressure: float = Field(default=10.0, le=50.0)

    # Stability limits
    min_stability_index: float = Field(default=0.7, ge=0.5, le=1.0)
    max_flame_pulsation_hz: float = Field(default=15.0, ge=5.0, le=50.0)

    # Lambda (air-fuel ratio) limits
    min_lambda: float = Field(default=1.05, ge=1.02)
    max_lambda: float = Field(default=1.3, le=2.0)

    # Turndown limits
    min_turndown_ratio: float = Field(default=0.25, ge=0.1, le=0.5)


class OptimizationConfig(BaseModel):
    """Configuration for multi-objective optimization."""

    # Objective weights
    fuel_cost_weight: float = Field(default=1.0, ge=0.0, le=10.0)
    emissions_cost_weight: float = Field(default=0.5, ge=0.0, le=10.0)
    co_penalty_weight: float = Field(default=2.0, ge=0.0, le=10.0)
    stability_penalty_weight: float = Field(default=3.0, ge=0.0, le=10.0)
    actuator_move_weight: float = Field(default=0.1, ge=0.0, le=1.0)

    # Optimization parameters
    max_iterations: int = Field(default=100, ge=10, le=1000)
    convergence_tolerance: float = Field(default=1e-6, ge=1e-9, le=1e-3)
    step_size: float = Field(default=0.01, ge=0.001, le=0.1)

    # Constraints
    max_setpoint_change_per_step: float = Field(default=0.05, ge=0.01, le=0.2)
    min_update_interval_s: float = Field(default=60.0, ge=10.0, le=600.0)

    # Actuator rate limits
    damper_rate_limit_pct_s: float = Field(default=5.0, ge=1.0, le=20.0)
    valve_rate_limit_pct_s: float = Field(default=2.0, ge=0.5, le=10.0)


class MLModelConfig(BaseModel):
    """Configuration for ML models."""

    # Model selection
    stability_model: str = Field(default="gradient_boosting")
    emissions_model: str = Field(default="neural_network")
    soft_sensor_model: str = Field(default="ensemble")

    # Training parameters
    retrain_interval_hours: int = Field(default=24, ge=1, le=168)
    min_training_samples: int = Field(default=1000, ge=100, le=100000)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.4)

    # Inference parameters
    inference_timeout_ms: int = Field(default=100, ge=10, le=1000)
    confidence_threshold: float = Field(default=0.8, ge=0.5, le=0.99)

    # Feature engineering
    lookback_window_s: int = Field(default=300, ge=60, le=3600)
    feature_aggregation_interval_s: int = Field(default=10, ge=1, le=60)


class IntegrationConfig(BaseModel):
    """Configuration for external system integrations."""

    # OPC-UA
    opcua_endpoint: str = Field(default="opc.tcp://localhost:4840")
    opcua_namespace: str = Field(default="ns=2;s=")
    opcua_poll_interval_ms: int = Field(default=1000, ge=100, le=10000)

    # Kafka
    kafka_brokers: list[str] = Field(default_factory=lambda: ["localhost:9092"])
    kafka_topic_prefix: str = Field(default="gl004.burnmaster")
    kafka_consumer_group: str = Field(default="gl004-burnmaster-group")

    # gRPC
    grpc_port: int = Field(default=50054, ge=1024, le=65535)
    grpc_max_workers: int = Field(default=10, ge=1, le=100)

    # REST API
    rest_port: int = Field(default=8084, ge=1024, le=65535)
    rest_host: str = Field(default="0.0.0.0")

    # DCS/Historian
    historian_type: str = Field(default="pi")  # pi, osisoft, wonderware, etc.
    historian_connection: str = Field(default="")
    historian_poll_interval_s: int = Field(default=5, ge=1, le=60)


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and alerting."""

    # Metrics
    prometheus_port: int = Field(default=9094, ge=1024, le=65535)
    metrics_prefix: str = Field(default="gl004_burnmaster")

    # Tracing
    otlp_endpoint: str = Field(default="http://localhost:4317")
    trace_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0)

    # Alerting
    alert_cooldown_s: int = Field(default=300, ge=60, le=3600)
    critical_alert_channels: list[str] = Field(
        default_factory=lambda: ["email", "sms", "pagerduty"]
    )
    warning_alert_channels: list[str] = Field(
        default_factory=lambda: ["email", "slack"]
    )


class AuditConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")
    log_retention_days: int = Field(default=90, ge=30, le=365)

    # Audit trail
    audit_database: str = Field(default="postgresql://localhost/gl004_audit")
    audit_retention_days: int = Field(default=730, ge=365, le=3650)

    # Evidence packaging
    evidence_storage_path: str = Field(default="/var/gl004/evidence")
    evidence_encryption: bool = Field(default=True)


class BurnmasterConfig(BaseModel):
    """Master configuration for GL-004 BURNMASTER agent."""

    # Agent metadata
    agent_id: str = Field(default="GL-004")
    agent_codename: str = Field(default="BURNMASTER")
    instance_id: str = Field(default="")

    # Operating mode
    mode: OperatingMode = Field(default=OperatingMode.ADVISORY)

    # Fuel configuration
    fuel_type: FuelType = Field(default=FuelType.NATURAL_GAS)
    burner_type: BurnerType = Field(default=BurnerType.LOW_NOX)

    # Number of burners
    num_burners: int = Field(default=1, ge=1, le=100)

    # Process parameters
    design_capacity_mw: float = Field(default=10.0, ge=0.1, le=1000.0)
    design_efficiency_pct: float = Field(default=85.0, ge=50.0, le=99.0)

    # Sub-configurations
    safety: SafetyLimits = Field(default_factory=SafetyLimits)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    ml: MLModelConfig = Field(default_factory=MLModelConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)

    @field_validator("instance_id", mode="before")
    @classmethod
    def generate_instance_id(cls, v: str) -> str:
        """Generate instance ID if not provided."""
        if not v:
            import uuid
            return str(uuid.uuid4())[:8]
        return v

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BurnmasterConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_env(cls) -> "BurnmasterConfig":
        """Load configuration from environment variables."""
        import os
        config_path = os.getenv("GL004_CONFIG_PATH")
        if config_path:
            return cls.from_yaml(config_path)
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


@dataclass
class RuntimeContext:
    """Runtime context for the agent."""

    config: BurnmasterConfig = field(default_factory=BurnmasterConfig)
    start_time: float = field(default_factory=lambda: __import__("time").time())
    current_mode: OperatingMode = field(default=OperatingMode.ADVISORY)
    last_optimization_time: float | None = None
    last_setpoint_change_time: float | None = None
    error_count: int = 0
    warning_count: int = 0

    def mode_allows_writes(self) -> bool:
        """Check if current mode allows writing setpoints."""
        return self.current_mode == OperatingMode.CLOSED_LOOP

    def mode_allows_recommendations(self) -> bool:
        """Check if current mode allows generating recommendations."""
        return self.current_mode in (
            OperatingMode.ADVISORY,
            OperatingMode.CLOSED_LOOP,
        )

    def time_since_last_optimization(self) -> float | None:
        """Get time since last optimization run."""
        if self.last_optimization_time is None:
            return None
        import time
        return time.time() - self.last_optimization_time

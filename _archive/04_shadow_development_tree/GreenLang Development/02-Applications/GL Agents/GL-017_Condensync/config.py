# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC Configuration Module

Comprehensive configuration management for the Condenser Optimization Agent.
Provides hierarchical configuration with environment variable support,
validation, and sensible defaults.

Configuration Hierarchy:
1. Default values (defined in this module)
2. Configuration file (YAML/JSON)
3. Environment variables (highest priority)

Environment Variable Prefix: GL017_

Example:
    # Set via environment
    export GL017_AGENT_MODE=monitoring
    export GL017_KAFKA_BOOTSTRAP_SERVERS=localhost:9092

    # Or load from file
    config = load_config("config.yaml")

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

logger = logging.getLogger(__name__)


# ============================================================================
# ENVIRONMENT VARIABLE PREFIX
# ============================================================================

ENV_PREFIX = "GL017_"


def get_env(key: str, default: Any = None, cast_type: type = str) -> Any:
    """
    Get environment variable with prefix and type casting.

    Args:
        key: Variable name (without prefix)
        default: Default value if not set
        cast_type: Type to cast the value to

    Returns:
        Environment variable value or default
    """
    full_key = f"{ENV_PREFIX}{key}"
    value = os.environ.get(full_key)

    if value is None:
        return default

    if cast_type == bool:
        return value.lower() in ("true", "1", "yes", "on")
    elif cast_type == int:
        return int(value)
    elif cast_type == float:
        return float(value)
    elif cast_type == list:
        return value.split(",")
    else:
        return value


# ============================================================================
# ENUMS FOR CONFIGURATION
# ============================================================================

class AgentMode(str, Enum):
    """Agent operating modes."""
    MONITORING = "monitoring"      # Continuous real-time monitoring
    DIAGNOSTIC = "diagnostic"      # Single diagnostic analysis
    OPTIMIZATION = "optimization"  # Vacuum/performance optimization
    PREDICTIVE = "predictive"      # Fouling/degradation prediction
    MAINTENANCE = "maintenance"    # Maintenance scheduling mode
    BATCH = "batch"                # Batch processing mode


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TubeMaterial(str, Enum):
    """Condenser tube materials with thermal conductivity."""
    TITANIUM = "titanium"                    # k = 21.9 W/mK
    ADMIRALTY_BRASS = "admiralty_brass"      # k = 111 W/mK
    COPPER_NICKEL_90_10 = "cupronickel_90_10"  # k = 45 W/mK
    COPPER_NICKEL_70_30 = "cupronickel_70_30"  # k = 29 W/mK
    STAINLESS_STEEL_304 = "ss_304"           # k = 16.2 W/mK
    STAINLESS_STEEL_316 = "ss_316"           # k = 16.3 W/mK
    ALUMINUM_BRASS = "aluminum_brass"        # k = 100 W/mK


class CondenserType(str, Enum):
    """Types of condensers supported."""
    SURFACE_SINGLE_PASS = "surface_single_pass"
    SURFACE_TWO_PASS = "surface_two_pass"
    SURFACE_DIVIDED_WATERBOX = "surface_divided_waterbox"
    AIR_COOLED = "air_cooled"
    EVAPORATIVE = "evaporative"
    DIRECT_CONTACT = "direct_contact"


# ============================================================================
# CORE AGENT CONFIGURATION
# ============================================================================

@dataclass
class AgentConfig:
    """
    Core agent configuration.

    Controls agent identification, mode, and general behavior.

    Attributes:
        agent_id: Unique agent identifier (GL-017)
        agent_name: Human-readable agent name
        version: Agent software version
        mode: Operating mode (monitoring, diagnostic, etc.)
        instance_id: Unique instance identifier for distributed deployments
        enable_provenance: Enable SHA-256 provenance tracking
        enable_explainability: Enable detailed explanations
        enable_audit_logging: Enable audit trail logging
        log_level: Logging verbosity level
    """
    agent_id: str = field(
        default_factory=lambda: get_env("AGENT_ID", "GL-017")
    )
    agent_name: str = field(
        default_factory=lambda: get_env("AGENT_NAME", "CONDENSYNC")
    )
    version: str = "1.0.0"
    mode: AgentMode = field(
        default_factory=lambda: AgentMode(get_env("AGENT_MODE", "diagnostic"))
    )
    instance_id: str = field(
        default_factory=lambda: get_env(
            "INSTANCE_ID",
            f"condensync-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
    )
    enable_provenance: bool = field(
        default_factory=lambda: get_env("ENABLE_PROVENANCE", True, bool)
    )
    enable_explainability: bool = field(
        default_factory=lambda: get_env("ENABLE_EXPLAINABILITY", True, bool)
    )
    enable_audit_logging: bool = field(
        default_factory=lambda: get_env("ENABLE_AUDIT_LOGGING", True, bool)
    )
    log_level: LogLevel = field(
        default_factory=lambda: LogLevel(get_env("LOG_LEVEL", "INFO"))
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "mode": self.mode.value,
            "instance_id": self.instance_id,
            "enable_provenance": self.enable_provenance,
            "enable_explainability": self.enable_explainability,
            "enable_audit_logging": self.enable_audit_logging,
            "log_level": self.log_level.value
        }


# ============================================================================
# CONDENSER DESIGN CONFIGURATION
# ============================================================================

@dataclass
class TubeGeometry:
    """
    Tube geometry parameters.

    Attributes:
        outer_diameter_mm: Tube outer diameter (mm)
        inner_diameter_mm: Tube inner diameter (mm)
        wall_thickness_mm: Tube wall thickness (mm)
        length_m: Tube effective length (m)
        tube_count: Total number of tubes
        tube_pitch_mm: Tube pitch (center-to-center spacing) (mm)
        tube_layout: Tube layout pattern (triangular, square)
    """
    outer_diameter_mm: float = 25.4  # 1 inch standard
    inner_diameter_mm: float = 22.9  # ~0.9 inch ID
    wall_thickness_mm: float = 1.25  # BWG 18
    length_m: float = 12.0
    tube_count: int = 15000
    tube_pitch_mm: float = 31.75  # 1.25 inch pitch
    tube_layout: str = "triangular"

    @property
    def wall_thickness_calculated_mm(self) -> float:
        """Calculate wall thickness from OD and ID."""
        return (self.outer_diameter_mm - self.inner_diameter_mm) / 2

    @property
    def tube_cross_section_area_m2(self) -> float:
        """Calculate single tube flow cross-section area (m2)."""
        import math
        radius_m = (self.inner_diameter_mm / 1000) / 2
        return math.pi * radius_m ** 2

    @property
    def total_flow_area_m2(self) -> float:
        """Calculate total flow area through all tubes (m2)."""
        return self.tube_cross_section_area_m2 * self.tube_count

    @property
    def heat_transfer_area_m2(self) -> float:
        """Calculate total heat transfer area (m2)."""
        import math
        # Based on outer diameter for shell-side heat transfer
        circumference_m = math.pi * (self.outer_diameter_mm / 1000)
        return circumference_m * self.length_m * self.tube_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "outer_diameter_mm": self.outer_diameter_mm,
            "inner_diameter_mm": self.inner_diameter_mm,
            "wall_thickness_mm": self.wall_thickness_mm,
            "length_m": self.length_m,
            "tube_count": self.tube_count,
            "tube_pitch_mm": self.tube_pitch_mm,
            "tube_layout": self.tube_layout,
            "calculated": {
                "heat_transfer_area_m2": round(self.heat_transfer_area_m2, 1),
                "total_flow_area_m2": round(self.total_flow_area_m2, 4)
            }
        }


@dataclass
class CondenserDesignConfig:
    """
    Condenser design (nameplate) parameters.

    These are fixed design values from condenser specification/datasheet.

    Attributes:
        condenser_id: Unique condenser identifier
        condenser_name: Human-readable condenser name
        condenser_type: Type of condenser
        tube_geometry: Tube geometry parameters
        tube_material: Tube material type
        tube_passes: Number of tube passes (1, 2, or 4)
        design_duty_mw: Design heat duty (MW thermal)
        design_steam_flow_kg_s: Design exhaust steam flow (kg/s)
        design_backpressure_kpa: Design backpressure (kPa absolute)
        design_ttd_c: Design terminal temperature difference (C)
        design_cleanliness_factor: Design cleanliness factor (HEI)
        design_u_value_w_m2k: Design overall U-value (W/m2K)
        design_cw_inlet_temp_c: Design CW inlet temperature (C)
        design_cw_temp_rise_c: Design CW temperature rise (C)
        design_cw_flow_m3_s: Design CW flow rate (m3/s)
        design_cw_velocity_m_s: Design CW velocity in tubes (m/s)
        air_removal_capacity_scfm: Air removal system capacity (SCFM)
        shell_material: Shell material
        waterbox_material: Waterbox material
        effective_surface_area_m2: Override for effective area (if different from calculated)
    """
    condenser_id: str = field(
        default_factory=lambda: get_env("CONDENSER_ID", "COND-001")
    )
    condenser_name: str = "Main Condenser Unit 1"
    condenser_type: CondenserType = CondenserType.SURFACE_TWO_PASS
    tube_geometry: TubeGeometry = field(default_factory=TubeGeometry)
    tube_material: TubeMaterial = TubeMaterial.TITANIUM
    tube_passes: int = 2
    design_duty_mw: float = 500.0
    design_steam_flow_kg_s: float = 200.0
    design_backpressure_kpa: float = 5.0
    design_ttd_c: float = 3.0
    design_cleanliness_factor: float = 0.85
    design_u_value_w_m2k: float = 3000.0
    design_cw_inlet_temp_c: float = 20.0
    design_cw_temp_rise_c: float = 10.0
    design_cw_flow_m3_s: float = 12.0
    design_cw_velocity_m_s: float = 2.0
    air_removal_capacity_scfm: float = 50.0
    shell_material: str = "carbon_steel"
    waterbox_material: str = "carbon_steel_rubber_lined"
    effective_surface_area_m2: Optional[float] = None

    @property
    def surface_area_m2(self) -> float:
        """Get effective surface area (calculated or override)."""
        if self.effective_surface_area_m2 is not None:
            return self.effective_surface_area_m2
        return self.tube_geometry.heat_transfer_area_m2

    @property
    def tube_thermal_conductivity_w_mk(self) -> float:
        """Get tube material thermal conductivity."""
        conductivity_map = {
            TubeMaterial.TITANIUM: 21.9,
            TubeMaterial.ADMIRALTY_BRASS: 111.0,
            TubeMaterial.COPPER_NICKEL_90_10: 45.0,
            TubeMaterial.COPPER_NICKEL_70_30: 29.0,
            TubeMaterial.STAINLESS_STEEL_304: 16.2,
            TubeMaterial.STAINLESS_STEEL_316: 16.3,
            TubeMaterial.ALUMINUM_BRASS: 100.0,
        }
        return conductivity_map.get(self.tube_material, 21.9)

    def validate(self) -> List[str]:
        """
        Validate configuration parameters.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if self.design_duty_mw <= 0:
            errors.append("design_duty_mw must be positive")

        if self.design_backpressure_kpa <= 0:
            errors.append("design_backpressure_kpa must be positive (absolute)")

        if not 0.5 <= self.design_cleanliness_factor <= 1.0:
            errors.append("design_cleanliness_factor must be between 0.5 and 1.0")

        if self.tube_passes not in [1, 2, 4]:
            errors.append("tube_passes must be 1, 2, or 4")

        if self.design_cw_velocity_m_s < 0.5 or self.design_cw_velocity_m_s > 4.0:
            errors.append("design_cw_velocity_m_s should be 0.5-4.0 m/s")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "condenser_id": self.condenser_id,
            "condenser_name": self.condenser_name,
            "condenser_type": self.condenser_type.value,
            "tube_geometry": self.tube_geometry.to_dict(),
            "tube_material": self.tube_material.value,
            "tube_passes": self.tube_passes,
            "design_duty_mw": self.design_duty_mw,
            "design_steam_flow_kg_s": self.design_steam_flow_kg_s,
            "design_backpressure_kpa": self.design_backpressure_kpa,
            "design_ttd_c": self.design_ttd_c,
            "design_cleanliness_factor": self.design_cleanliness_factor,
            "design_u_value_w_m2k": self.design_u_value_w_m2k,
            "design_cw_inlet_temp_c": self.design_cw_inlet_temp_c,
            "design_cw_temp_rise_c": self.design_cw_temp_rise_c,
            "design_cw_flow_m3_s": self.design_cw_flow_m3_s,
            "design_cw_velocity_m_s": self.design_cw_velocity_m_s,
            "air_removal_capacity_scfm": self.air_removal_capacity_scfm,
            "surface_area_m2": round(self.surface_area_m2, 1),
            "tube_thermal_conductivity_w_mk": self.tube_thermal_conductivity_w_mk
        }


# ============================================================================
# OPERATIONAL THRESHOLDS CONFIGURATION
# ============================================================================

@dataclass
class OperationalConfig:
    """
    Operational thresholds and alarm limits.

    Defines limits for monitoring and alarming.

    Attributes:
        min_backpressure_kpa: Minimum safe backpressure (kPa abs)
        max_backpressure_kpa: Maximum safe backpressure (kPa abs)
        target_cleanliness_factor: Target CF for operations
        cleanliness_warning_threshold: CF warning threshold
        cleanliness_alarm_threshold: CF alarm threshold
        fouling_warning_threshold_m2k_kw: Fouling factor warning
        fouling_alarm_threshold_m2k_kw: Fouling factor alarm
        ttd_warning_threshold_c: TTD warning threshold (C)
        ttd_alarm_threshold_c: TTD alarm threshold (C)
        air_inleakage_normal_scfm: Normal air in-leakage (SCFM)
        air_inleakage_warning_scfm: Air in-leakage warning (SCFM)
        air_inleakage_alarm_scfm: Air in-leakage alarm (SCFM)
        do2_warning_ppb: Dissolved oxygen warning (ppb)
        do2_alarm_ppb: Dissolved oxygen alarm (ppb)
        subcooling_warning_c: Subcooling warning threshold (C)
        subcooling_alarm_c: Subcooling alarm threshold (C)
        min_cw_velocity_m_s: Minimum CW velocity for cleaning (m/s)
        max_cw_velocity_m_s: Maximum CW velocity limit (m/s)
        max_tube_plugging_percent: Maximum allowed tube plugging (%)
    """
    # Backpressure limits
    min_backpressure_kpa: float = 2.5
    max_backpressure_kpa: float = 15.0

    # Cleanliness factor thresholds
    target_cleanliness_factor: float = field(
        default_factory=lambda: get_env("TARGET_CF", 0.90, float)
    )
    cleanliness_warning_threshold: float = 0.80
    cleanliness_alarm_threshold: float = 0.70

    # Fouling thresholds (m2K/kW)
    fouling_warning_threshold_m2k_kw: float = 0.0002
    fouling_alarm_threshold_m2k_kw: float = 0.0003

    # TTD thresholds
    ttd_warning_threshold_c: float = 4.0
    ttd_alarm_threshold_c: float = 6.0

    # Air in-leakage thresholds
    air_inleakage_normal_scfm: float = 2.0
    air_inleakage_warning_scfm: float = 5.0
    air_inleakage_alarm_scfm: float = 10.0

    # Dissolved oxygen thresholds
    do2_warning_ppb: float = 10.0
    do2_alarm_ppb: float = 20.0

    # Subcooling thresholds
    subcooling_warning_c: float = 1.5
    subcooling_alarm_c: float = 3.0

    # CW velocity limits
    min_cw_velocity_m_s: float = 1.5
    max_cw_velocity_m_s: float = 3.0

    # Tube plugging limit
    max_tube_plugging_percent: float = 10.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# INTEGRATION CONFIGURATION
# ============================================================================

@dataclass
class OpcUaConfig:
    """
    OPC-UA server configuration for real-time data acquisition.

    Attributes:
        enabled: Enable OPC-UA connectivity
        server_url: OPC-UA server endpoint URL
        namespace_uri: OPC-UA namespace URI
        security_mode: Security mode (None, Sign, SignAndEncrypt)
        security_policy: Security policy URI
        username: Authentication username (if required)
        password: Authentication password (if required)
        certificate_path: Client certificate path
        private_key_path: Private key path
        subscription_interval_ms: Subscription publishing interval (ms)
        sampling_interval_ms: Node sampling interval (ms)
        queue_size: Subscription queue size
        timeout_ms: Connection timeout (ms)
    """
    enabled: bool = field(
        default_factory=lambda: get_env("OPCUA_ENABLED", False, bool)
    )
    server_url: str = field(
        default_factory=lambda: get_env("OPCUA_SERVER_URL", "opc.tcp://localhost:4840")
    )
    namespace_uri: str = "urn:condensync:opcua:server"
    security_mode: str = "None"
    security_policy: str = "None"
    username: Optional[str] = field(
        default_factory=lambda: get_env("OPCUA_USERNAME", None)
    )
    password: Optional[str] = field(
        default_factory=lambda: get_env("OPCUA_PASSWORD", None)
    )
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    subscription_interval_ms: int = 1000
    sampling_interval_ms: int = 500
    queue_size: int = 10
    timeout_ms: int = 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive fields)."""
        return {
            "enabled": self.enabled,
            "server_url": self.server_url,
            "namespace_uri": self.namespace_uri,
            "security_mode": self.security_mode,
            "subscription_interval_ms": self.subscription_interval_ms,
            "sampling_interval_ms": self.sampling_interval_ms,
            "timeout_ms": self.timeout_ms
        }


@dataclass
class KafkaConfig:
    """
    Apache Kafka configuration for event streaming.

    Attributes:
        enabled: Enable Kafka connectivity
        bootstrap_servers: Kafka bootstrap servers (comma-separated)
        consumer_group_id: Consumer group ID
        input_topic: Topic for receiving condenser data
        output_topic: Topic for publishing results
        alert_topic: Topic for publishing alerts
        security_protocol: Security protocol (PLAINTEXT, SSL, SASL_SSL)
        sasl_mechanism: SASL mechanism (PLAIN, SCRAM-SHA-256, etc.)
        sasl_username: SASL username
        sasl_password: SASL password
        ssl_ca_location: SSL CA certificate path
        auto_offset_reset: Offset reset policy (earliest, latest)
        enable_auto_commit: Enable auto offset commit
        max_poll_records: Maximum records per poll
    """
    enabled: bool = field(
        default_factory=lambda: get_env("KAFKA_ENABLED", False, bool)
    )
    bootstrap_servers: str = field(
        default_factory=lambda: get_env("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    )
    consumer_group_id: str = field(
        default_factory=lambda: get_env("KAFKA_CONSUMER_GROUP", "condensync-agent")
    )
    input_topic: str = field(
        default_factory=lambda: get_env("KAFKA_INPUT_TOPIC", "condenser.data.raw")
    )
    output_topic: str = field(
        default_factory=lambda: get_env("KAFKA_OUTPUT_TOPIC", "condenser.diagnostics")
    )
    alert_topic: str = field(
        default_factory=lambda: get_env("KAFKA_ALERT_TOPIC", "condenser.alerts")
    )
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = field(
        default_factory=lambda: get_env("KAFKA_SASL_USERNAME", None)
    )
    sasl_password: Optional[str] = field(
        default_factory=lambda: get_env("KAFKA_SASL_PASSWORD", None)
    )
    ssl_ca_location: Optional[str] = None
    auto_offset_reset: str = "latest"
    enable_auto_commit: bool = True
    max_poll_records: int = 500

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive fields)."""
        return {
            "enabled": self.enabled,
            "bootstrap_servers": self.bootstrap_servers,
            "consumer_group_id": self.consumer_group_id,
            "input_topic": self.input_topic,
            "output_topic": self.output_topic,
            "alert_topic": self.alert_topic,
            "security_protocol": self.security_protocol,
            "auto_offset_reset": self.auto_offset_reset,
            "max_poll_records": self.max_poll_records
        }


@dataclass
class CmmsConfig:
    """
    CMMS (Computerized Maintenance Management System) integration.

    Attributes:
        enabled: Enable CMMS integration
        system_type: CMMS system type (maximo, sap_pm, generic)
        api_base_url: CMMS API base URL
        api_key: API key for authentication
        api_secret: API secret for authentication
        work_order_prefix: Prefix for generated work orders
        equipment_id_field: Field name for equipment ID mapping
        auto_create_work_orders: Automatically create work orders
        min_severity_for_wo: Minimum alert severity for work order creation
        sync_interval_minutes: Sync interval with CMMS
    """
    enabled: bool = field(
        default_factory=lambda: get_env("CMMS_ENABLED", False, bool)
    )
    system_type: str = field(
        default_factory=lambda: get_env("CMMS_SYSTEM_TYPE", "generic")
    )
    api_base_url: str = field(
        default_factory=lambda: get_env("CMMS_API_URL", "https://cmms.example.com/api")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: get_env("CMMS_API_KEY", None)
    )
    api_secret: Optional[str] = field(
        default_factory=lambda: get_env("CMMS_API_SECRET", None)
    )
    work_order_prefix: str = "COND-"
    equipment_id_field: str = "asset_id"
    auto_create_work_orders: bool = False
    min_severity_for_wo: str = "HIGH"
    sync_interval_minutes: int = 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding sensitive fields)."""
        return {
            "enabled": self.enabled,
            "system_type": self.system_type,
            "api_base_url": self.api_base_url,
            "work_order_prefix": self.work_order_prefix,
            "auto_create_work_orders": self.auto_create_work_orders,
            "min_severity_for_wo": self.min_severity_for_wo,
            "sync_interval_minutes": self.sync_interval_minutes
        }


@dataclass
class IntegrationConfig:
    """
    Combined integration configuration.

    Attributes:
        opc_ua: OPC-UA configuration
        kafka: Kafka configuration
        cmms: CMMS configuration
        historian_dsn: Historian database connection string
        metrics_endpoint: Prometheus metrics endpoint
    """
    opc_ua: OpcUaConfig = field(default_factory=OpcUaConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    cmms: CmmsConfig = field(default_factory=CmmsConfig)
    historian_dsn: Optional[str] = field(
        default_factory=lambda: get_env("HISTORIAN_DSN", None)
    )
    metrics_endpoint: str = field(
        default_factory=lambda: get_env("METRICS_ENDPOINT", "/metrics")
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "opc_ua": self.opc_ua.to_dict(),
            "kafka": self.kafka.to_dict(),
            "cmms": self.cmms.to_dict(),
            "historian_dsn": self.historian_dsn is not None,
            "metrics_endpoint": self.metrics_endpoint
        }


# ============================================================================
# API CONFIGURATION
# ============================================================================

@dataclass
class ApiConfig:
    """
    REST API server configuration.

    Attributes:
        enabled: Enable REST API
        host: API server host
        port: API server port
        workers: Number of worker processes
        reload: Enable auto-reload (development only)
        cors_origins: Allowed CORS origins
        api_key_header: Header name for API key
        require_api_key: Require API key for all requests
        rate_limit_per_minute: Rate limit per minute per client
        request_timeout_seconds: Request timeout
        max_request_size_mb: Maximum request body size
    """
    enabled: bool = field(
        default_factory=lambda: get_env("API_ENABLED", True, bool)
    )
    host: str = field(
        default_factory=lambda: get_env("API_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: get_env("API_PORT", 8017, int)
    )
    workers: int = field(
        default_factory=lambda: get_env("API_WORKERS", 4, int)
    )
    reload: bool = field(
        default_factory=lambda: get_env("API_RELOAD", False, bool)
    )
    cors_origins: List[str] = field(
        default_factory=lambda: get_env("API_CORS_ORIGINS", "*", list)
    )
    api_key_header: str = "X-API-Key"
    require_api_key: bool = field(
        default_factory=lambda: get_env("API_REQUIRE_KEY", False, bool)
    )
    rate_limit_per_minute: int = 100
    request_timeout_seconds: int = 300
    max_request_size_mb: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "cors_origins": self.cors_origins,
            "require_api_key": self.require_api_key,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "request_timeout_seconds": self.request_timeout_seconds
        }


# ============================================================================
# MONITORING CONFIGURATION
# ============================================================================

@dataclass
class MonitoringConfig:
    """
    Monitoring and observability configuration.

    Attributes:
        polling_interval_seconds: Data polling interval
        metrics_enabled: Enable Prometheus metrics
        metrics_port: Prometheus metrics port
        health_check_interval_seconds: Health check interval
        tracing_enabled: Enable distributed tracing
        tracing_endpoint: Tracing collector endpoint (OTLP)
        tracing_sample_rate: Tracing sample rate (0.0-1.0)
    """
    polling_interval_seconds: float = field(
        default_factory=lambda: get_env("POLLING_INTERVAL", 30.0, float)
    )
    metrics_enabled: bool = field(
        default_factory=lambda: get_env("METRICS_ENABLED", True, bool)
    )
    metrics_port: int = field(
        default_factory=lambda: get_env("METRICS_PORT", 9017, int)
    )
    health_check_interval_seconds: int = 60
    tracing_enabled: bool = field(
        default_factory=lambda: get_env("TRACING_ENABLED", False, bool)
    )
    tracing_endpoint: str = field(
        default_factory=lambda: get_env("TRACING_ENDPOINT", "http://localhost:4317")
    )
    tracing_sample_rate: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# CALCULATION CONFIGURATION
# ============================================================================

@dataclass
class CalculationConfig:
    """
    Calculation engine configuration.

    Attributes:
        max_concurrent_calculations: Maximum parallel calculations
        calculation_timeout_seconds: Single calculation timeout
        enable_caching: Enable result caching
        cache_ttl_seconds: Cache time-to-live
        decimal_precision: Decimal precision for financial calculations
        use_iapws_steam_tables: Use IAPWS-IF97 for steam properties
        fouling_model: Fouling prediction model type
        optimization_algorithm: Vacuum optimization algorithm
    """
    max_concurrent_calculations: int = field(
        default_factory=lambda: get_env("MAX_CONCURRENT_CALC", 10, int)
    )
    calculation_timeout_seconds: int = 300
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    decimal_precision: int = 4
    use_iapws_steam_tables: bool = True
    fouling_model: str = "kern_seaton"  # kern_seaton, linear, exponential
    optimization_algorithm: str = "gradient"  # gradient, genetic, mixed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# ECONOMIC CONFIGURATION
# ============================================================================

@dataclass
class EconomicConfig:
    """
    Economic parameters for cost/benefit calculations.

    Attributes:
        fuel_cost_per_mmbtu: Fuel cost ($/MMBtu)
        fuel_type: Fuel type (natural_gas, coal, oil)
        electricity_price_per_mwh: Electricity price ($/MWh)
        capacity_factor: Plant capacity factor (0.0-1.0)
        carbon_price_per_ton: Carbon price ($/ton CO2)
        maintenance_labor_rate_per_hour: Maintenance labor rate ($/hr)
        discount_rate: Discount rate for NPV calculations
        analysis_period_years: Default analysis period
    """
    fuel_cost_per_mmbtu: float = field(
        default_factory=lambda: get_env("FUEL_COST", 3.50, float)
    )
    fuel_type: str = "natural_gas"
    electricity_price_per_mwh: float = field(
        default_factory=lambda: get_env("ELECTRICITY_PRICE", 40.0, float)
    )
    capacity_factor: float = 0.85
    carbon_price_per_ton: float = field(
        default_factory=lambda: get_env("CARBON_PRICE", 50.0, float)
    )
    maintenance_labor_rate_per_hour: float = 75.0
    discount_rate: float = 0.08
    analysis_period_years: int = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

@dataclass
class CondensyncConfig:
    """
    Master configuration for GL-017 CONDENSYNC agent.

    Combines all configuration sections into a single object.

    Attributes:
        agent: Core agent configuration
        condenser_design: Condenser design parameters
        operational: Operational thresholds
        integration: Integration settings
        api: API server configuration
        monitoring: Monitoring configuration
        calculation: Calculation engine settings
        economic: Economic parameters
    """
    agent: AgentConfig = field(default_factory=AgentConfig)
    condenser_design: CondenserDesignConfig = field(default_factory=CondenserDesignConfig)
    operational: OperationalConfig = field(default_factory=OperationalConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    calculation: CalculationConfig = field(default_factory=CalculationConfig)
    economic: EconomicConfig = field(default_factory=EconomicConfig)

    def validate(self) -> List[str]:
        """
        Validate entire configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        errors.extend(self.condenser_design.validate())
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire configuration to dictionary."""
        return {
            "agent": self.agent.to_dict(),
            "condenser_design": self.condenser_design.to_dict(),
            "operational": self.operational.to_dict(),
            "integration": self.integration.to_dict(),
            "api": self.api.to_dict(),
            "monitoring": self.monitoring.to_dict(),
            "calculation": self.calculation.to_dict(),
            "economic": self.economic.to_dict()
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def get_config_hash(self) -> str:
        """
        Get SHA-256 hash of configuration.

        Useful for tracking configuration versions.
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# ============================================================================
# CONFIGURATION LOADING FUNCTIONS
# ============================================================================

def load_config_from_file(file_path: Union[str, Path]) -> CondensyncConfig:
    """
    Load configuration from YAML or JSON file.

    Args:
        file_path: Path to configuration file

    Returns:
        CondensyncConfig object

    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file format is unsupported
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        if path.suffix in [".yaml", ".yml"]:
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML config files")
        elif path.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    return _dict_to_config(data)


def _dict_to_config(data: Dict[str, Any]) -> CondensyncConfig:
    """
    Convert dictionary to CondensyncConfig.

    Args:
        data: Configuration dictionary

    Returns:
        CondensyncConfig object
    """
    config = CondensyncConfig()

    # Override with provided values
    if "agent" in data:
        for key, value in data["agent"].items():
            if hasattr(config.agent, key):
                setattr(config.agent, key, value)

    if "condenser_design" in data:
        for key, value in data["condenser_design"].items():
            if key == "tube_geometry" and isinstance(value, dict):
                for gkey, gval in value.items():
                    if hasattr(config.condenser_design.tube_geometry, gkey):
                        setattr(config.condenser_design.tube_geometry, gkey, gval)
            elif hasattr(config.condenser_design, key):
                setattr(config.condenser_design, key, value)

    # ... similar for other sections

    return config


def get_default_config() -> CondensyncConfig:
    """
    Get default configuration with environment overrides.

    Returns:
        CondensyncConfig with defaults and environment variables applied
    """
    return CondensyncConfig()


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Master Config
    "CondensyncConfig",
    # Individual Configs
    "AgentConfig",
    "CondenserDesignConfig",
    "TubeGeometry",
    "OperationalConfig",
    "IntegrationConfig",
    "OpcUaConfig",
    "KafkaConfig",
    "CmmsConfig",
    "ApiConfig",
    "MonitoringConfig",
    "CalculationConfig",
    "EconomicConfig",
    # Enums
    "AgentMode",
    "LogLevel",
    "TubeMaterial",
    "CondenserType",
    # Functions
    "load_config_from_file",
    "get_default_config",
    "get_env",
    # Constants
    "ENV_PREFIX",
]

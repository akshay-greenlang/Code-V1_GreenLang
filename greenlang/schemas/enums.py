# -*- coding: utf-8 -*-
"""
GreenLang Shared Enumerations
==============================

Centralized enumerations that replace 1,200+ duplicated enum definitions
across the GreenLang codebase. Agent-specific enums (e.g. ``FuelType``,
``NodeType``) remain in their respective agent modules.

Categories (30 enums total):
    - Job/Calculation Lifecycle (3): CalculationStatus, JobStatus, ProcessingStatus
    - Execution (1): ExecutionStatus
    - Severity/Priority (3): Severity, ValidationSeverity, Priority
    - Data Quality (3): DataQualityLevel, MatchStatus, ResolutionStatus
    - Reporting (2): ReportingPeriod, RegulatoryFramework
    - Risk/Compliance (2): RiskLevel, ComplianceStatus
    - Organizational (1): ControlApproach
    - Units (1): EmissionUnit
    - Config/Infrastructure (8): Environment, LogLevel, HealthStatus,
          NotificationChannel, StorageBackend, ProtocolType, SortOrder,
          ScheduleFrequency
    - Alerts (2): AlertSeverity, AlertStatus
    - Formats (2): ReportFormat, FileFormat
    - i18n (1): LanguageCode
    - Geography (1): GeographicRegion

Usage::

    from greenlang.schemas.enums import CalculationStatus, JobStatus
    from greenlang.schemas.enums import Environment, LogLevel, HealthStatus
    from greenlang.schemas.enums import ReportFormat, AlertSeverity

Author: GreenLang Platform Team
Date: March 2026 (expanded 2026-03-30)
Status: Production Ready
"""

from __future__ import annotations

from enum import Enum


# =============================================================================
# Job / Calculation Lifecycle
# =============================================================================


class CalculationStatus(str, Enum):
    """Lifecycle status of a calculation or computation task.

    Duplicated in 30+ MRV agents, 10+ pack engines.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    """Lifecycle status of an asynchronous processing job.

    Duplicated in 20+ data agents, 10+ EUDR agents.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStatus(str, Enum):
    """Generic processing pipeline status.

    Use when neither CalculationStatus nor JobStatus fits exactly.
    """

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


# =============================================================================
# Severity / Priority
# =============================================================================


class Severity(str, Enum):
    """Severity level for validation findings, alerts, and issues."""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationSeverity(str, Enum):
    """Severity specifically for validation results.

    Duplicated in 15+ data/MRV agents.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class Priority(str, Enum):
    """Priority levels for task scheduling and queue ordering."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# Data Quality
# =============================================================================


class DataQualityLevel(str, Enum):
    """Data quality tier classification per GHG Protocol guidance.

    Duplicated across MRV agents and pack engines.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    ESTIMATED = "estimated"
    DEFAULT = "default"


class MatchStatus(str, Enum):
    """Status of a data matching/reconciliation operation.

    Duplicated in duplicate_detector, cross_source_reconciliation,
    data_quality_profiler, and EUDR agents.
    """

    MATCHED = "matched"
    UNMATCHED = "unmatched"
    PARTIAL = "partial"
    CONFLICT = "conflict"
    PENDING = "pending"


class ResolutionStatus(str, Enum):
    """Status of conflict/issue resolution.

    Duplicated in cross_source_reconciliation, validation_rule_engine.
    """

    PENDING = "pending"
    RESOLVED = "resolved"
    REJECTED = "rejected"
    ESCALATED = "escalated"


# =============================================================================
# Reporting
# =============================================================================


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission/metric reporting aggregation.

    Duplicated in 30+ MRV agents.
    """

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class RegulatoryFramework(str, Enum):
    """Regulatory framework governing calculation methodology.

    Duplicated in 20+ MRV agents and 10+ pack engines.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    EPA_40CFR98 = "epa_40cfr98"
    UK_SECR = "uk_secr"
    EU_ETS = "eu_ets"
    EUDR = "eudr"
    CBAM = "cbam"
    SBTi = "sbti"
    CDP = "cdp"
    TCFD = "tcfd"


# =============================================================================
# Risk / Compliance
# =============================================================================


class RiskLevel(str, Enum):
    """Risk assessment level.

    Duplicated in EUDR agents and compliance packs.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ComplianceStatus(str, Enum):
    """Compliance assessment status.

    Duplicated in EUDR agents and compliance packs.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    UNDER_REVIEW = "under_review"
    NOT_ASSESSED = "not_assessed"


# =============================================================================
# Organizational Boundaries (GHG Protocol)
# =============================================================================


class ControlApproach(str, Enum):
    """Organizational boundary approach for emission ownership.

    Duplicated in MRV agents and GHG accounting packs.
    """

    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    EQUITY_SHARE = "equity_share"


# =============================================================================
# Units (commonly duplicated subset)
# =============================================================================


class EmissionUnit(str, Enum):
    """Standard emission reporting units.

    Duplicated in 30+ MRV agents.
    """

    KG_CO2E = "kg_co2e"
    TONNES_CO2E = "tonnes_co2e"
    MT_CO2E = "mt_co2e"
    KG = "kg"
    TONNES = "tonnes"


# =============================================================================
# Execution
# =============================================================================


class ExecutionStatus(str, Enum):
    """Execution lifecycle status for pipelines, workflows, and tasks.

    Replaces 8+ duplicate ExecutionStatus/PipelineStatus/WorkflowStatus enums
    in orchestrator, schema_migration, integration, and execution modules.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    RETRYING = "retrying"


# =============================================================================
# Config / Infrastructure
# =============================================================================


class Environment(str, Enum):
    """Deployment environment.

    Replaces 5+ duplicate Environment/EnvironmentName enums in config,
    vector DB, agent factory, feature flags, and infrastructure modules.
    """

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class LogLevel(str, Enum):
    """Standard logging levels.

    Replaces 3+ duplicate LogLevel enums in monitoring, observability,
    and orchestrator modules.
    """

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(str, Enum):
    """Service/component health status.

    Replaces 44+ duplicate HealthStatus enums across EUDR setup.py files,
    monitoring modules, and infrastructure components.
    """

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"


class NotificationChannel(str, Enum):
    """Notification delivery channel.

    Replaces 19+ duplicate NotificationChannel enums in alerting,
    monitoring, EUDR agents, and infrastructure modules.
    """

    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    LOG = "log"


class StorageBackend(str, Enum):
    """Data storage backend type.

    Replaces 5+ duplicate StorageBackend/StorageType enums in config,
    execution, and infrastructure modules.
    """

    MEMORY = "memory"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    MONGODB = "mongodb"
    S3 = "s3"
    FILE = "file"


class ProtocolType(str, Enum):
    """Communication protocol type.

    Replaces 8+ duplicate ProtocolType/BMSProtocol/CommunicationProtocol
    enums in SCADA, BMS, IoT, and execution modules.
    """

    HTTP = "http"
    HTTPS = "https"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MQTT = "mqtt"
    MODBUS = "modbus"
    BACNET = "bacnet"
    OPCUA = "opcua"


class SortOrder(str, Enum):
    """Sort direction for queries and results.

    Replaces 5+ duplicate SortOrder enums in data gateway, GraphQL,
    and EUDR API schemas.
    """

    ASC = "asc"
    DESC = "desc"


class ScheduleFrequency(str, Enum):
    """Scheduling/monitoring frequency.

    Replaces 10+ duplicate ScheduleFrequency/MonitoringFrequency/
    UpdateFrequency enums. Extends ReportingPeriod with sub-daily
    and real-time granularity.
    """

    REALTIME = "realtime"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


# =============================================================================
# Alerts
# =============================================================================


class AlertSeverity(str, Enum):
    """Alert severity level for monitoring and incident management.

    Replaces 47+ duplicate AlertSeverity enums across EUDR agents,
    process heat agents, data agents, and infrastructure modules.
    """

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert lifecycle status.

    Replaces duplicate AlertStatus enums in alerting service,
    monitoring, and EUDR agent modules.
    """

    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


# =============================================================================
# Formats
# =============================================================================


class ReportFormat(str, Enum):
    """Output report format.

    Replaces 30+ duplicate ReportFormat/ExportFormat enums across data
    agents, EUDR agents, and pack engines.
    """

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    HTML = "html"
    EXCEL = "excel"
    XML = "xml"


class FileFormat(str, Enum):
    """Input/output file format for data processing.

    Replaces 15+ duplicate FileFormat/DocumentFormat enums in data
    ingestion, parsing, and export modules.
    """

    CSV = "csv"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    PDF = "pdf"
    EXCEL = "excel"
    PARQUET = "parquet"
    PNG = "png"
    JPG = "jpg"
    TIFF = "tiff"
    SHAPEFILE = "shapefile"
    GEOJSON = "geojson"


# =============================================================================
# Internationalization
# =============================================================================


class LanguageCode(str, Enum):
    """ISO 639-1 language codes for multi-language support.

    Replaces 8+ duplicate LanguageCode/DocumentLanguage/ReportLanguage
    enums in EUDR agents and reporting modules.
    """

    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    PT = "pt"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    AR = "ar"
    HI = "hi"
    IT = "it"
    NL = "nl"


# =============================================================================
# Geography
# =============================================================================


class GeographicRegion(str, Enum):
    """Major geographic regions for compliance and reporting.

    Replaces 15+ duplicate Region/RegionType enums in MRV agents,
    finance agents, and compliance modules.
    """

    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    AFRICA = "africa"
    MIDDLE_EAST = "middle_east"
    OCEANIA = "oceania"

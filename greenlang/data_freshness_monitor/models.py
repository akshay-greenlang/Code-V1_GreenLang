# -*- coding: utf-8 -*-
"""
Data Freshness Monitor Agent Service Data Models - AGENT-DATA-016

Pydantic v2 data models for the Data Freshness Monitor SDK. Defines
enumerations, SDK models, request models, and constants for dataset
registration, SLA definition, freshness checking, staleness pattern
detection, breach alerting, refresh prediction, and monitoring pipeline
orchestration.

Re-exported Layer 1 sources:
    - greenlang.data_quality_profiler.timeliness_tracker:
        TimelinessTracker (as L1TimelinessTracker)
    - greenlang.data_quality_profiler.models:
        FreshnessResult (as L1FreshnessResult),
        QualityDimension (as L1QualityDimension),
        RuleType (as L1RuleType),
        FRESHNESS_BOUNDARIES_HOURS (as L1FreshnessBoundariesHours)

New enumerations (13):
    - DatasetStatus, DatasetPriority, RefreshCadence, FreshnessLevel,
      SLAStatus, BreachSeverity, BreachStatus, AlertChannel,
      AlertStatus, AlertSeverity, PatternType, PredictionStatus,
      MonitoringStatus

New SDK models (20):
    - DatasetDefinition, SLADefinition, FreshnessCheck, RefreshEvent,
      StalenessPattern, SLABreach, FreshnessAlert, RefreshPrediction,
      FreshnessReport, AuditEntry, DatasetGroup, FreshnessSummary,
      SourceReliability, EscalationPolicy, EscalationLevel,
      MonitoringRun

Request models (8):
    - RegisterDatasetRequest, UpdateDatasetRequest, CreateSLARequest,
      UpdateSLARequest, RunFreshnessCheckRequest, RunBatchCheckRequest,
      UpdateBreachRequest, RunPipelineRequest

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from data_quality_profiler.timeliness_tracker
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.timeliness_tracker import (
        TimelinessTracker as L1TimelinessTracker,
    )
    TimelinessTracker = L1TimelinessTracker
except ImportError:
    TimelinessTracker = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# Re-export Layer 1 models from data_quality_profiler.models
# ---------------------------------------------------------------------------

try:
    from greenlang.data_quality_profiler.models import (
        FreshnessResult as L1FreshnessResult,
    )
    FreshnessResult = L1FreshnessResult
except ImportError:
    FreshnessResult = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.models import (
        QualityDimension as L1QualityDimension,
    )
    QualityDimension = L1QualityDimension
except ImportError:
    QualityDimension = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.models import (
        RuleType as L1RuleType,
    )
    RuleType = L1RuleType
except ImportError:
    RuleType = None  # type: ignore[assignment, misc]

try:
    from greenlang.data_quality_profiler.models import (
        FRESHNESS_BOUNDARIES_HOURS as L1FreshnessBoundariesHours,
    )
    FRESHNESS_BOUNDARIES_HOURS = L1FreshnessBoundariesHours
except ImportError:
    FRESHNESS_BOUNDARIES_HOURS = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from RefreshCadence value to expected hours between refreshes.
CADENCE_HOURS: Dict[str, float] = {
    "realtime": 0.0,
    "minutely": 1.0 / 60.0,
    "hourly": 1.0,
    "daily": 24.0,
    "weekly": 168.0,
    "monthly": 720.0,
    "quarterly": 2160.0,
    "annual": 8760.0,
    "on_demand": -1.0,
}

#: Freshness score boundaries by level (maximum age hours for each level).
FRESHNESS_SCORE_BOUNDARIES: Dict[str, float] = {
    "excellent": 1.0,
    "good": 24.0,
    "fair": 168.0,
    "poor": 720.0,
    "stale": 8760.0,
}

#: Default escalation delay minutes per level [L1, L2, L3, L4].
DEFAULT_ESCALATION_DELAYS: List[int] = [15, 60, 240, 1440]

#: Maximum number of datasets allowed in a single dataset group.
MAX_DATASETS_PER_GROUP: int = 500

#: Maximum SLA warning hours (1 calendar year).
MAX_SLA_WARNING_HOURS: float = 8760.0

#: Module version string.
VERSION: str = "1.0.0"


# =============================================================================
# Enumerations (13)
# =============================================================================


class DatasetStatus(str, Enum):
    """Operational status of a registered dataset.

    ACTIVE: Dataset is actively monitored for freshness.
    INACTIVE: Dataset is registered but monitoring is paused by design.
    PAUSED: Dataset monitoring is temporarily paused.
    ARCHIVED: Dataset has been archived and is no longer monitored.
    ERROR: Dataset encountered an error during last check.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    ARCHIVED = "archived"
    ERROR = "error"


class DatasetPriority(str, Enum):
    """Priority classification for a monitored dataset.

    CRITICAL: Highest priority; immediate alerting on staleness.
    HIGH: High priority; alert within the first escalation window.
    MEDIUM: Standard priority; alert on SLA breach.
    LOW: Low priority; informational alerts only.
    INFORMATIONAL: Lowest priority; logged but no active alerts.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RefreshCadence(str, Enum):
    """Expected refresh frequency for a dataset.

    REALTIME: Continuous or near-real-time refresh.
    MINUTELY: Refresh expected every minute.
    HOURLY: Refresh expected every hour.
    DAILY: Refresh expected every day.
    WEEKLY: Refresh expected every week.
    MONTHLY: Refresh expected every month.
    QUARTERLY: Refresh expected every quarter.
    ANNUAL: Refresh expected every year.
    ON_DEMAND: Refresh occurs only when explicitly triggered.
    """

    REALTIME = "realtime"
    MINUTELY = "minutely"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class FreshnessLevel(str, Enum):
    """Qualitative freshness assessment of a dataset.

    EXCELLENT: Data is very fresh (within expected cadence).
    GOOD: Data is reasonably fresh.
    FAIR: Data is aging but still usable.
    POOR: Data is becoming stale; action recommended.
    STALE: Data is stale; immediate action required.
    """

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    STALE = "stale"


class SLAStatus(str, Enum):
    """SLA compliance status for a freshness check.

    COMPLIANT: Dataset is within SLA thresholds.
    WARNING: Dataset is approaching SLA breach threshold.
    BREACHED: Dataset has breached the SLA critical threshold.
    CRITICAL: Dataset has exceeded the critical SLA threshold significantly.
    UNKNOWN: SLA status could not be determined.
    """

    COMPLIANT = "compliant"
    WARNING = "warning"
    BREACHED = "breached"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class BreachSeverity(str, Enum):
    """Severity level of an SLA breach event.

    INFO: Informational; no material impact.
    LOW: Minor breach; logged for tracking.
    MEDIUM: Moderate breach; review recommended.
    HIGH: Significant breach; escalation recommended.
    CRITICAL: Severe breach; immediate action required.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BreachStatus(str, Enum):
    """Lifecycle status of an SLA breach.

    DETECTED: Breach has been detected but not yet acknowledged.
    ACKNOWLEDGED: Breach has been acknowledged by an operator.
    INVESTIGATING: Breach is under active investigation.
    RESOLVED: Breach has been resolved.
    EXPIRED: Breach expired without resolution (dataset archived, etc.).
    """

    DETECTED = "detected"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    EXPIRED = "expired"


class AlertChannel(str, Enum):
    """Delivery channel for a freshness alert notification.

    WEBHOOK: HTTP webhook callback.
    EMAIL: Email notification.
    SLACK: Slack channel message.
    PAGERDUTY: PagerDuty incident.
    TEAMS: Microsoft Teams message.
    LOG: Internal log entry only.
    """

    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    LOG = "log"


class AlertStatus(str, Enum):
    """Lifecycle status of a freshness alert.

    PENDING: Alert has been created but not yet sent.
    SENT: Alert has been delivered to the channel.
    ACKNOWLEDGED: Alert has been acknowledged by a recipient.
    RESOLVED: Alert has been resolved (underlying breach resolved).
    SUPPRESSED: Alert was suppressed by a suppression rule.
    """

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertSeverity(str, Enum):
    """Severity level of a freshness alert.

    INFO: Informational notification.
    WARNING: Warning-level notification.
    CRITICAL: Critical-level notification requiring action.
    EMERGENCY: Emergency-level notification requiring immediate action.
    """

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class PatternType(str, Enum):
    """Classification of a detected staleness pattern.

    RECURRING_STALENESS: Dataset repeatedly goes stale at predictable intervals.
    SEASONAL_DEGRADATION: Freshness degrades during specific seasonal periods.
    SOURCE_FAILURE: Upstream source system failures causing staleness.
    REFRESH_DRIFT: Refresh timing drifts gradually over time.
    RANDOM_GAPS: Unpredictable gaps in refresh schedule.
    SYSTEMATIC_DELAY: Consistent delay relative to expected cadence.
    """

    RECURRING_STALENESS = "recurring_staleness"
    SEASONAL_DEGRADATION = "seasonal_degradation"
    SOURCE_FAILURE = "source_failure"
    REFRESH_DRIFT = "refresh_drift"
    RANDOM_GAPS = "random_gaps"
    SYSTEMATIC_DELAY = "systematic_delay"


class PredictionStatus(str, Enum):
    """Status of a refresh prediction relative to actual outcome.

    PENDING: Prediction has not yet been evaluated against actuals.
    ON_TIME: Actual refresh occurred within the predicted window.
    LATE: Actual refresh occurred after the predicted time.
    VERY_LATE: Actual refresh was significantly later than predicted.
    MISSED: No actual refresh occurred within the evaluation window.
    """

    PENDING = "pending"
    ON_TIME = "on_time"
    LATE = "late"
    VERY_LATE = "very_late"
    MISSED = "missed"


class MonitoringStatus(str, Enum):
    """Execution status of a monitoring pipeline run.

    IDLE: Pipeline is idle and waiting for the next scheduled run.
    RUNNING: Pipeline is currently executing.
    COMPLETED: Pipeline finished successfully.
    FAILED: Pipeline terminated due to an error.
    CANCELLED: Pipeline was cancelled by the user.
    """

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# =============================================================================
# SDK Data Models (20)
# =============================================================================


class DatasetDefinition(BaseModel):
    """A registered dataset monitored for data freshness.

    Contains metadata, refresh expectations, priority, operational
    status, and timestamps for a single monitored dataset.

    Attributes:
        id: Unique identifier for this dataset.
        name: Human-readable dataset name.
        source_name: Name of the upstream data source.
        source_type: Type of the upstream data source (e.g. 'erp', 'api').
        owner: Owner or team responsible for this dataset.
        refresh_cadence: Expected refresh frequency.
        priority: Priority classification for alerting.
        status: Current operational status of the dataset.
        tags: List of tags for categorization and filtering.
        metadata: Arbitrary metadata key-value pairs.
        last_refreshed_at: Timestamp of the most recent data refresh.
        registered_at: When the dataset was registered.
        updated_at: When the dataset record was last updated.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this dataset",
    )
    name: str = Field(
        ..., description="Human-readable dataset name",
    )
    source_name: str = Field(
        default="",
        description="Name of the upstream data source",
    )
    source_type: str = Field(
        default="",
        description="Type of the upstream data source (e.g. 'erp', 'api')",
    )
    owner: str = Field(
        default="",
        description="Owner or team responsible for this dataset",
    )
    refresh_cadence: RefreshCadence = Field(
        default=RefreshCadence.DAILY,
        description="Expected refresh frequency",
    )
    priority: DatasetPriority = Field(
        default=DatasetPriority.MEDIUM,
        description="Priority classification for alerting",
    )
    status: DatasetStatus = Field(
        default=DatasetStatus.ACTIVE,
        description="Current operational status of the dataset",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization and filtering",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata key-value pairs",
    )
    last_refreshed_at: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent data refresh",
    )
    registered_at: datetime = Field(
        default_factory=_utcnow,
        description="When the dataset was registered",
    )
    updated_at: datetime = Field(
        default_factory=_utcnow,
        description="When the dataset record was last updated",
    )
    version: int = Field(
        default=1,
        description="Record version for optimistic concurrency",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class SLADefinition(BaseModel):
    """Service-level agreement definition for dataset freshness.

    Specifies warning and critical hour thresholds, breach severity,
    escalation policy, and business-hours-only enforcement for a
    single dataset.

    Attributes:
        id: Unique identifier for this SLA definition.
        dataset_id: Identifier of the dataset this SLA applies to.
        warning_hours: Hours of staleness before a warning is raised.
        critical_hours: Hours of staleness before a critical breach is raised.
        breach_severity: Default severity for breaches under this SLA.
        escalation_policy: Escalation policy for breach notifications.
        business_hours_only: Whether SLA enforcement is limited to business hours.
        created_at: When this SLA was created.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this SLA definition",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the dataset this SLA applies to",
    )
    warning_hours: float = Field(
        default=24.0, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Hours of staleness before a warning is raised",
    )
    critical_hours: float = Field(
        default=72.0, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Hours of staleness before a critical breach is raised",
    )
    breach_severity: BreachSeverity = Field(
        default=BreachSeverity.HIGH,
        description="Default severity for breaches under this SLA",
    )
    escalation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Escalation policy for breach notifications",
    )
    business_hours_only: bool = Field(
        default=False,
        description="Whether SLA enforcement is limited to business hours",
    )
    created_at: datetime = Field(
        default_factory=_utcnow,
        description="When this SLA was created",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_warning_before_critical(self) -> SLADefinition:
        """Validate that warning_hours is less than or equal to critical_hours."""
        if self.warning_hours > self.critical_hours:
            raise ValueError(
                f"warning_hours ({self.warning_hours}) must be <= "
                f"critical_hours ({self.critical_hours})"
            )
        return self


class FreshnessCheck(BaseModel):
    """Result of a single freshness check for a dataset.

    Records the check timestamp, computed data age, freshness score,
    qualitative freshness level, SLA status, and provenance hash.

    Attributes:
        id: Unique identifier for this freshness check.
        dataset_id: Identifier of the checked dataset.
        checked_at: Timestamp when the check was performed.
        age_hours: Current age of the data in hours.
        freshness_score: Computed freshness score (0.0-1.0, 1.0 = perfectly fresh).
        freshness_level: Qualitative freshness assessment.
        sla_status: SLA compliance status at the time of check.
        sla_id: Identifier of the SLA used for evaluation (if any).
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this freshness check",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the checked dataset",
    )
    checked_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the check was performed",
    )
    age_hours: float = Field(
        default=0.0, ge=0.0,
        description="Current age of the data in hours",
    )
    freshness_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Computed freshness score (0.0-1.0, 1.0 = perfectly fresh)",
    )
    freshness_level: FreshnessLevel = Field(
        default=FreshnessLevel.EXCELLENT,
        description="Qualitative freshness assessment",
    )
    sla_status: SLAStatus = Field(
        default=SLAStatus.UNKNOWN,
        description="SLA compliance status at the time of check",
    )
    sla_id: Optional[str] = Field(
        None,
        description="Identifier of the SLA used for evaluation (if any)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class RefreshEvent(BaseModel):
    """Record of a dataset refresh event.

    Captures the refresh timestamp, data size, record count, source
    information, and provenance hash for each observed refresh.

    Attributes:
        id: Unique identifier for this refresh event.
        dataset_id: Identifier of the refreshed dataset.
        refreshed_at: Timestamp when the refresh occurred.
        data_size_bytes: Size of the refreshed data in bytes.
        record_count: Number of records in the refreshed data.
        source_info: Arbitrary metadata about the refresh source.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this refresh event",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the refreshed dataset",
    )
    refreshed_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the refresh occurred",
    )
    data_size_bytes: Optional[int] = Field(
        None, ge=0,
        description="Size of the refreshed data in bytes",
    )
    record_count: Optional[int] = Field(
        None, ge=0,
        description="Number of records in the refreshed data",
    )
    source_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata about the refresh source",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class StalenessPattern(BaseModel):
    """A detected staleness pattern for a monitored dataset.

    Records the pattern type, detection timestamp, recurrence
    frequency, severity, confidence, and human-readable description.

    Attributes:
        id: Unique identifier for this staleness pattern.
        dataset_id: Identifier of the dataset exhibiting the pattern.
        pattern_type: Classification of the staleness pattern.
        detected_at: Timestamp when the pattern was detected.
        frequency_hours: Recurrence frequency in hours (if periodic).
        severity: Severity level of the pattern.
        confidence: Confidence in the pattern detection (0.0-1.0).
        description: Human-readable description of the pattern.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this staleness pattern",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the dataset exhibiting the pattern",
    )
    pattern_type: PatternType = Field(
        default=PatternType.RECURRING_STALENESS,
        description="Classification of the staleness pattern",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the pattern was detected",
    )
    frequency_hours: Optional[float] = Field(
        None, ge=0.0,
        description="Recurrence frequency in hours (if periodic)",
    )
    severity: BreachSeverity = Field(
        default=BreachSeverity.LOW,
        description="Severity level of the pattern",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the pattern detection (0.0-1.0)",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the pattern",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class SLABreach(BaseModel):
    """Record of an SLA breach event for a monitored dataset.

    Tracks the breach lifecycle from detection through acknowledgement
    to resolution, including the data age at breach time and notes.

    Attributes:
        id: Unique identifier for this SLA breach.
        dataset_id: Identifier of the breached dataset.
        sla_id: Identifier of the breached SLA definition.
        breach_severity: Severity level of the breach.
        detected_at: Timestamp when the breach was detected.
        acknowledged_at: Timestamp when the breach was acknowledged.
        resolved_at: Timestamp when the breach was resolved.
        status: Current lifecycle status of the breach.
        age_at_breach_hours: Data age in hours at the time of breach.
        resolution_notes: Human-readable notes on the resolution.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this SLA breach",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the breached dataset",
    )
    sla_id: str = Field(
        ..., description="Identifier of the breached SLA definition",
    )
    breach_severity: BreachSeverity = Field(
        default=BreachSeverity.HIGH,
        description="Severity level of the breach",
    )
    detected_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the breach was detected",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the breach was acknowledged",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the breach was resolved",
    )
    status: BreachStatus = Field(
        default=BreachStatus.DETECTED,
        description="Current lifecycle status of the breach",
    )
    age_at_breach_hours: float = Field(
        default=0.0, ge=0.0,
        description="Data age in hours at the time of breach",
    )
    resolution_notes: str = Field(
        default="",
        description="Human-readable notes on the resolution",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v

    @field_validator("sla_id")
    @classmethod
    def validate_sla_id(cls, v: str) -> str:
        """Validate sla_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("sla_id must be non-empty")
        return v


class FreshnessAlert(BaseModel):
    """A freshness alert notification sent for an SLA breach.

    Records the alert delivery channel, severity, message content,
    timestamps, and suppression status.

    Attributes:
        id: Unique identifier for this alert.
        breach_id: Identifier of the triggering SLA breach.
        alert_severity: Severity level of the alert.
        channel: Delivery channel for the alert.
        message: Alert message content.
        sent_at: Timestamp when the alert was sent.
        acknowledged_at: Timestamp when the alert was acknowledged.
        status: Current lifecycle status of the alert.
        suppressed_reason: Reason for suppression (if suppressed).
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this alert",
    )
    breach_id: str = Field(
        ..., description="Identifier of the triggering SLA breach",
    )
    alert_severity: AlertSeverity = Field(
        default=AlertSeverity.WARNING,
        description="Severity level of the alert",
    )
    channel: AlertChannel = Field(
        default=AlertChannel.LOG,
        description="Delivery channel for the alert",
    )
    message: str = Field(
        default="",
        description="Alert message content",
    )
    sent_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the alert was sent",
    )
    acknowledged_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the alert was acknowledged",
    )
    status: AlertStatus = Field(
        default=AlertStatus.PENDING,
        description="Current lifecycle status of the alert",
    )
    suppressed_reason: Optional[str] = Field(
        None,
        description="Reason for suppression (if suppressed)",
    )

    model_config = {"extra": "forbid"}

    @field_validator("breach_id")
    @classmethod
    def validate_breach_id(cls, v: str) -> str:
        """Validate breach_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("breach_id must be non-empty")
        return v


class RefreshPrediction(BaseModel):
    """A prediction for the next expected refresh of a dataset.

    Records the predicted refresh time, confidence, actual outcome
    (once known), prediction error, and evaluation status.

    Attributes:
        id: Unique identifier for this prediction.
        dataset_id: Identifier of the predicted dataset.
        predicted_refresh_at: Predicted timestamp for the next refresh.
        confidence: Confidence in the prediction (0.0-1.0).
        actual_refresh_at: Actual refresh timestamp (populated after observation).
        error_hours: Prediction error in hours (actual minus predicted).
        status: Evaluation status of the prediction.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this prediction",
    )
    dataset_id: str = Field(
        ..., description="Identifier of the predicted dataset",
    )
    predicted_refresh_at: datetime = Field(
        ..., description="Predicted timestamp for the next refresh",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the prediction (0.0-1.0)",
    )
    actual_refresh_at: Optional[datetime] = Field(
        None,
        description="Actual refresh timestamp (populated after observation)",
    )
    error_hours: Optional[float] = Field(
        None,
        description="Prediction error in hours (actual minus predicted)",
    )
    status: PredictionStatus = Field(
        default=PredictionStatus.PENDING,
        description="Evaluation status of the prediction",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class FreshnessReport(BaseModel):
    """Compliance-grade report summarizing freshness status across datasets.

    Provides aggregate counts, compliance statistics, a human-readable
    summary, and a provenance hash for regulatory audit trails.

    Attributes:
        id: Unique identifier for this report.
        report_type: Type of the report (e.g. 'daily_summary', 'sla_compliance').
        generated_at: Timestamp when the report was generated.
        dataset_count: Total number of datasets included in the report.
        compliant_count: Number of datasets that are SLA-compliant.
        breached_count: Number of datasets with active SLA breaches.
        summary: Human-readable summary of the freshness status.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    report_type: str = Field(
        default="summary",
        description="Type of the report (e.g. 'daily_summary', 'sla_compliance')",
    )
    generated_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the report was generated",
    )
    dataset_count: int = Field(
        default=0, ge=0,
        description="Total number of datasets included in the report",
    )
    compliant_count: int = Field(
        default=0, ge=0,
        description="Number of datasets that are SLA-compliant",
    )
    breached_count: int = Field(
        default=0, ge=0,
        description="Number of datasets with active SLA breaches",
    )
    summary: str = Field(
        default="",
        description="Human-readable summary of the freshness status",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


class AuditEntry(BaseModel):
    """An audit log entry for a data freshness monitoring operation.

    Records the operation type, affected entity, detail payload,
    timestamp, and provenance hash for full auditability.

    Attributes:
        id: Unique identifier for this audit entry.
        operation: Name of the operation performed (e.g. 'register_dataset').
        entity_type: Type of the affected entity (e.g. 'dataset', 'sla', 'breach').
        entity_id: Identifier of the affected entity.
        details: Arbitrary detail payload for the operation.
        timestamp: When the operation was performed.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this audit entry",
    )
    operation: str = Field(
        ..., description="Name of the operation performed (e.g. 'register_dataset')",
    )
    entity_type: str = Field(
        ..., description="Type of the affected entity (e.g. 'dataset', 'sla', 'breach')",
    )
    entity_id: str = Field(
        ..., description="Identifier of the affected entity",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary detail payload for the operation",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="When the operation was performed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}

    @field_validator("operation")
    @classmethod
    def validate_operation(cls, v: str) -> str:
        """Validate operation is non-empty."""
        if not v or not v.strip():
            raise ValueError("operation must be non-empty")
        return v

    @field_validator("entity_type")
    @classmethod
    def validate_entity_type(cls, v: str) -> str:
        """Validate entity_type is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_type must be non-empty")
        return v

    @field_validator("entity_id")
    @classmethod
    def validate_entity_id(cls, v: str) -> str:
        """Validate entity_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("entity_id must be non-empty")
        return v


class DatasetGroup(BaseModel):
    """A logical grouping of monitored datasets.

    Groups datasets for collective SLA enforcement, batch monitoring,
    and reporting. Limited to MAX_DATASETS_PER_GROUP members.

    Attributes:
        id: Unique identifier for this dataset group.
        name: Human-readable group name.
        description: Human-readable description of the group.
        dataset_ids: List of dataset identifiers in this group.
        priority: Priority classification for the group.
        sla_id: Optional SLA identifier applied to the entire group.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this dataset group",
    )
    name: str = Field(
        ..., description="Human-readable group name",
    )
    description: str = Field(
        default="",
        description="Human-readable description of the group",
    )
    dataset_ids: List[str] = Field(
        default_factory=list,
        description="List of dataset identifiers in this group",
    )
    priority: DatasetPriority = Field(
        default=DatasetPriority.MEDIUM,
        description="Priority classification for the group",
    )
    sla_id: Optional[str] = Field(
        None,
        description="Optional SLA identifier applied to the entire group",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("dataset_ids")
    @classmethod
    def validate_dataset_ids(cls, v: List[str]) -> List[str]:
        """Validate dataset_ids does not exceed MAX_DATASETS_PER_GROUP."""
        if len(v) > MAX_DATASETS_PER_GROUP:
            raise ValueError(
                f"dataset_ids cannot exceed {MAX_DATASETS_PER_GROUP} datasets, "
                f"got {len(v)}"
            )
        for did in v:
            if not did or not did.strip():
                raise ValueError("dataset_ids must contain non-empty strings")
        return v


class FreshnessSummary(BaseModel):
    """Lightweight freshness summary for a single dataset.

    Provides the current freshness snapshot without full check history,
    suitable for dashboard displays and list views.

    Attributes:
        dataset_id: Identifier of the dataset.
        name: Human-readable dataset name.
        current_age_hours: Current data age in hours.
        freshness_score: Current freshness score (0.0-1.0).
        freshness_level: Current qualitative freshness level.
        sla_status: Current SLA compliance status.
        last_checked_at: Timestamp of the most recent freshness check.
    """

    dataset_id: str = Field(
        ..., description="Identifier of the dataset",
    )
    name: str = Field(
        default="",
        description="Human-readable dataset name",
    )
    current_age_hours: float = Field(
        default=0.0, ge=0.0,
        description="Current data age in hours",
    )
    freshness_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Current freshness score (0.0-1.0)",
    )
    freshness_level: FreshnessLevel = Field(
        default=FreshnessLevel.EXCELLENT,
        description="Current qualitative freshness level",
    )
    sla_status: SLAStatus = Field(
        default=SLAStatus.UNKNOWN,
        description="Current SLA compliance status",
    )
    last_checked_at: Optional[datetime] = Field(
        None,
        description="Timestamp of the most recent freshness check",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class SourceReliability(BaseModel):
    """Reliability metrics for an upstream data source.

    Aggregates refresh statistics, on-time rate, average delay,
    and trend direction for a single upstream source.

    Attributes:
        source_name: Name of the upstream source.
        total_refreshes: Total number of observed refreshes.
        on_time_refreshes: Number of refreshes that arrived on time.
        reliability_pct: Percentage of on-time refreshes (0.0-100.0).
        avg_delay_hours: Average delay in hours for late refreshes.
        trend: Trend direction ('improving', 'stable', 'degrading').
    """

    source_name: str = Field(
        ..., description="Name of the upstream source",
    )
    total_refreshes: int = Field(
        default=0, ge=0,
        description="Total number of observed refreshes",
    )
    on_time_refreshes: int = Field(
        default=0, ge=0,
        description="Number of refreshes that arrived on time",
    )
    reliability_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Percentage of on-time refreshes (0.0-100.0)",
    )
    avg_delay_hours: float = Field(
        default=0.0, ge=0.0,
        description="Average delay in hours for late refreshes",
    )
    trend: str = Field(
        default="stable",
        description="Trend direction ('improving', 'stable', 'degrading')",
    )

    model_config = {"extra": "forbid"}

    @field_validator("source_name")
    @classmethod
    def validate_source_name(cls, v: str) -> str:
        """Validate source_name is non-empty."""
        if not v or not v.strip():
            raise ValueError("source_name must be non-empty")
        return v

    @field_validator("trend")
    @classmethod
    def validate_trend(cls, v: str) -> str:
        """Validate trend is one of the allowed values."""
        allowed = {"improving", "stable", "degrading"}
        if v not in allowed:
            raise ValueError(
                f"trend must be one of {sorted(allowed)}, got '{v}'"
            )
        return v


class EscalationLevel(BaseModel):
    """A single level within an escalation policy.

    Defines the delay before escalation, delivery channel,
    recipients, and message template for one escalation tier.

    Attributes:
        delay_minutes: Minutes to wait before triggering this escalation level.
        channel: Delivery channel for this escalation level.
        recipients: List of recipient identifiers (email, Slack handle, etc.).
        message_template: Optional message template with variable placeholders.
    """

    delay_minutes: int = Field(
        default=15, ge=0,
        description="Minutes to wait before triggering this escalation level",
    )
    channel: AlertChannel = Field(
        default=AlertChannel.LOG,
        description="Delivery channel for this escalation level",
    )
    recipients: List[str] = Field(
        default_factory=list,
        description="List of recipient identifiers (email, Slack handle, etc.)",
    )
    message_template: Optional[str] = Field(
        None,
        description="Optional message template with variable placeholders",
    )

    model_config = {"extra": "forbid"}


class EscalationPolicy(BaseModel):
    """Multi-level escalation policy for SLA breach notifications.

    Defines an ordered list of escalation levels and the maximum
    number of escalations before halting.

    Attributes:
        levels: Ordered list of escalation levels.
        max_escalations: Maximum number of escalation iterations.
    """

    levels: List[EscalationLevel] = Field(
        default_factory=list,
        description="Ordered list of escalation levels",
    )
    max_escalations: int = Field(
        default=4, ge=1,
        description="Maximum number of escalation iterations",
    )

    model_config = {"extra": "forbid"}


class MonitoringRun(BaseModel):
    """Record of a single monitoring pipeline execution.

    Tracks the run lifecycle, dataset coverage, breach and alert
    counts, and provenance hash for audit.

    Attributes:
        id: Unique identifier for this monitoring run.
        started_at: Timestamp when the run started.
        completed_at: Timestamp when the run completed.
        status: Execution status of the monitoring run.
        datasets_checked: Number of datasets checked during the run.
        breaches_found: Number of new SLA breaches detected.
        alerts_sent: Number of alerts sent during the run.
        provenance_hash: SHA-256 provenance chain hash for audit trail.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this monitoring run",
    )
    started_at: datetime = Field(
        default_factory=_utcnow,
        description="Timestamp when the run started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Timestamp when the run completed",
    )
    status: MonitoringStatus = Field(
        default=MonitoringStatus.IDLE,
        description="Execution status of the monitoring run",
    )
    datasets_checked: int = Field(
        default=0, ge=0,
        description="Number of datasets checked during the run",
    )
    breaches_found: int = Field(
        default=0, ge=0,
        description="Number of new SLA breaches detected",
    )
    alerts_sent: int = Field(
        default=0, ge=0,
        description="Number of alerts sent during the run",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 provenance chain hash for audit trail",
    )

    model_config = {"extra": "forbid"}


# =============================================================================
# Request Models (8)
# =============================================================================


class RegisterDatasetRequest(BaseModel):
    """Request body for registering a new dataset for freshness monitoring.

    Attributes:
        name: Human-readable dataset name.
        source_name: Name of the upstream data source.
        source_type: Type of the upstream data source.
        owner: Owner or team responsible for the dataset.
        refresh_cadence: Expected refresh frequency.
        priority: Priority classification for alerting.
        tags: List of tags for categorization and filtering.
        metadata: Arbitrary metadata key-value pairs.
    """

    name: str = Field(
        ..., description="Human-readable dataset name",
    )
    source_name: str = Field(
        default="",
        description="Name of the upstream data source",
    )
    source_type: str = Field(
        default="",
        description="Type of the upstream data source",
    )
    owner: str = Field(
        default="",
        description="Owner or team responsible for the dataset",
    )
    refresh_cadence: RefreshCadence = Field(
        default=RefreshCadence.DAILY,
        description="Expected refresh frequency",
    )
    priority: DatasetPriority = Field(
        default=DatasetPriority.MEDIUM,
        description="Priority classification for alerting",
    )
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization and filtering",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata key-value pairs",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty."""
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


class UpdateDatasetRequest(BaseModel):
    """Request body for updating an existing dataset registration.

    All fields are optional; only provided fields will be updated.

    Attributes:
        name: Updated human-readable dataset name.
        source_name: Updated upstream data source name.
        owner: Updated owner or team.
        refresh_cadence: Updated refresh frequency.
        priority: Updated priority classification.
        status: Updated operational status.
        tags: Updated list of tags.
        metadata: Updated metadata key-value pairs.
    """

    name: Optional[str] = Field(
        None,
        description="Updated human-readable dataset name",
    )
    source_name: Optional[str] = Field(
        None,
        description="Updated upstream data source name",
    )
    owner: Optional[str] = Field(
        None,
        description="Updated owner or team",
    )
    refresh_cadence: Optional[RefreshCadence] = Field(
        None,
        description="Updated refresh frequency",
    )
    priority: Optional[DatasetPriority] = Field(
        None,
        description="Updated priority classification",
    )
    status: Optional[DatasetStatus] = Field(
        None,
        description="Updated operational status",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Updated list of tags",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated metadata key-value pairs",
    )

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: Optional[str]) -> Optional[str]:
        """Validate name is non-empty if provided."""
        if v is not None and (not v or not v.strip()):
            raise ValueError("name must be non-empty when provided")
        return v


class CreateSLARequest(BaseModel):
    """Request body for creating a new SLA definition.

    Attributes:
        dataset_id: Identifier of the dataset this SLA applies to.
        warning_hours: Hours before a warning is raised.
        critical_hours: Hours before a critical breach is raised.
        breach_severity: Default severity for breaches under this SLA.
        escalation_policy: Optional escalation policy.
        business_hours_only: Whether to enforce only during business hours.
    """

    dataset_id: str = Field(
        ..., description="Identifier of the dataset this SLA applies to",
    )
    warning_hours: float = Field(
        default=24.0, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Hours before a warning is raised",
    )
    critical_hours: float = Field(
        default=72.0, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Hours before a critical breach is raised",
    )
    breach_severity: BreachSeverity = Field(
        default=BreachSeverity.HIGH,
        description="Default severity for breaches under this SLA",
    )
    escalation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional escalation policy",
    )
    business_hours_only: bool = Field(
        default=False,
        description="Whether to enforce only during business hours",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_warning_before_critical(self) -> CreateSLARequest:
        """Validate that warning_hours is less than or equal to critical_hours."""
        if self.warning_hours > self.critical_hours:
            raise ValueError(
                f"warning_hours ({self.warning_hours}) must be <= "
                f"critical_hours ({self.critical_hours})"
            )
        return self


class UpdateSLARequest(BaseModel):
    """Request body for updating an existing SLA definition.

    All fields are optional; only provided fields will be updated.

    Attributes:
        warning_hours: Updated warning threshold in hours.
        critical_hours: Updated critical threshold in hours.
        breach_severity: Updated default breach severity.
        escalation_policy: Updated escalation policy.
        business_hours_only: Updated business-hours-only flag.
    """

    warning_hours: Optional[float] = Field(
        None, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Updated warning threshold in hours",
    )
    critical_hours: Optional[float] = Field(
        None, ge=0.0, le=MAX_SLA_WARNING_HOURS,
        description="Updated critical threshold in hours",
    )
    breach_severity: Optional[BreachSeverity] = Field(
        None,
        description="Updated default breach severity",
    )
    escalation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Updated escalation policy",
    )
    business_hours_only: Optional[bool] = Field(
        None,
        description="Updated business-hours-only flag",
    )

    model_config = {"extra": "forbid"}


class RunFreshnessCheckRequest(BaseModel):
    """Request body for running a freshness check on a single dataset.

    Attributes:
        dataset_id: Identifier of the dataset to check.
        sla_id: Optional SLA identifier to evaluate against.
        force: Whether to force a check even if recently checked.
    """

    dataset_id: str = Field(
        ..., description="Identifier of the dataset to check",
    )
    sla_id: Optional[str] = Field(
        None,
        description="Optional SLA identifier to evaluate against",
    )
    force: bool = Field(
        default=False,
        description="Whether to force a check even if recently checked",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_id")
    @classmethod
    def validate_dataset_id(cls, v: str) -> str:
        """Validate dataset_id is non-empty."""
        if not v or not v.strip():
            raise ValueError("dataset_id must be non-empty")
        return v


class RunBatchCheckRequest(BaseModel):
    """Request body for running freshness checks on multiple datasets.

    Attributes:
        dataset_ids: List of dataset identifiers to check.
        group_id: Optional group identifier (checks all datasets in group).
        include_predictions: Whether to include refresh predictions.
        force: Whether to force checks even if recently checked.
    """

    dataset_ids: Optional[List[str]] = Field(
        None,
        description="List of dataset identifiers to check",
    )
    group_id: Optional[str] = Field(
        None,
        description="Optional group identifier (checks all datasets in group)",
    )
    include_predictions: bool = Field(
        default=False,
        description="Whether to include refresh predictions",
    )
    force: bool = Field(
        default=False,
        description="Whether to force checks even if recently checked",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_at_least_one_target(self) -> RunBatchCheckRequest:
        """Validate that at least one of dataset_ids or group_id is provided."""
        if not self.dataset_ids and not self.group_id:
            raise ValueError(
                "At least one of dataset_ids or group_id must be provided"
            )
        return self

    @field_validator("dataset_ids")
    @classmethod
    def validate_dataset_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate dataset_ids are non-empty strings if provided."""
        if v is not None:
            for did in v:
                if not did or not did.strip():
                    raise ValueError("dataset_ids must contain non-empty strings")
        return v


class UpdateBreachRequest(BaseModel):
    """Request body for updating an SLA breach status.

    Attributes:
        status: Updated breach lifecycle status.
        resolution_notes: Notes on the resolution or investigation.
    """

    status: BreachStatus = Field(
        ..., description="Updated breach lifecycle status",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Notes on the resolution or investigation",
    )

    model_config = {"extra": "forbid"}


class RunPipelineRequest(BaseModel):
    """Request body for running the full monitoring pipeline.

    Triggers end-to-end freshness checking, SLA evaluation, breach
    detection, alerting, and optional pattern analysis.

    Attributes:
        dataset_ids: Optional list of dataset identifiers to monitor.
        group_id: Optional group identifier (monitors all datasets in group).
        include_predictions: Whether to include refresh predictions.
        detect_patterns: Whether to run staleness pattern detection.
        generate_report: Whether to generate a freshness report.
    """

    dataset_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of dataset identifiers to monitor",
    )
    group_id: Optional[str] = Field(
        None,
        description="Optional group identifier (monitors all datasets in group)",
    )
    include_predictions: bool = Field(
        default=False,
        description="Whether to include refresh predictions",
    )
    detect_patterns: bool = Field(
        default=False,
        description="Whether to run staleness pattern detection",
    )
    generate_report: bool = Field(
        default=True,
        description="Whether to generate a freshness report",
    )

    model_config = {"extra": "forbid"}

    @field_validator("dataset_ids")
    @classmethod
    def validate_dataset_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate dataset_ids are non-empty strings if provided."""
        if v is not None:
            for did in v:
                if not did or not did.strip():
                    raise ValueError("dataset_ids must contain non-empty strings")
        return v


# =============================================================================
# __all__ export list
# =============================================================================

__all__ = [
    # -------------------------------------------------------------------------
    # Layer 1 re-exports (5)
    # -------------------------------------------------------------------------
    "TimelinessTracker",
    "FreshnessResult",
    "QualityDimension",
    "RuleType",
    "FRESHNESS_BOUNDARIES_HOURS",
    # -------------------------------------------------------------------------
    # Helper
    # -------------------------------------------------------------------------
    "_utcnow",
    # -------------------------------------------------------------------------
    # Constants (6)
    # -------------------------------------------------------------------------
    "CADENCE_HOURS",
    "FRESHNESS_SCORE_BOUNDARIES",
    "DEFAULT_ESCALATION_DELAYS",
    "MAX_DATASETS_PER_GROUP",
    "MAX_SLA_WARNING_HOURS",
    "VERSION",
    # -------------------------------------------------------------------------
    # Enumerations (13)
    # -------------------------------------------------------------------------
    "DatasetStatus",
    "DatasetPriority",
    "RefreshCadence",
    "FreshnessLevel",
    "SLAStatus",
    "BreachSeverity",
    "BreachStatus",
    "AlertChannel",
    "AlertStatus",
    "AlertSeverity",
    "PatternType",
    "PredictionStatus",
    "MonitoringStatus",
    # -------------------------------------------------------------------------
    # SDK data models (20)
    # -------------------------------------------------------------------------
    "DatasetDefinition",
    "SLADefinition",
    "FreshnessCheck",
    "RefreshEvent",
    "StalenessPattern",
    "SLABreach",
    "FreshnessAlert",
    "RefreshPrediction",
    "FreshnessReport",
    "AuditEntry",
    "DatasetGroup",
    "FreshnessSummary",
    "SourceReliability",
    "EscalationPolicy",
    "EscalationLevel",
    "MonitoringRun",
    # -------------------------------------------------------------------------
    # Request models (8)
    # -------------------------------------------------------------------------
    "RegisterDatasetRequest",
    "UpdateDatasetRequest",
    "CreateSLARequest",
    "UpdateSLARequest",
    "RunFreshnessCheckRequest",
    "RunBatchCheckRequest",
    "UpdateBreachRequest",
    "RunPipelineRequest",
]

# -*- coding: utf-8 -*-
"""
Audit Service Data Models - SEC-005

Pydantic models for audit events, search, and reporting.
Maps to the audit.audit_log TimescaleDB hypertable schema.

Author: GreenLang Framework Team
Date: February 2026
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class EventCategory(str, enum.Enum):
    """Audit event category (maps to audit.event_category enum)."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    CONFIGURATION = "configuration"
    INTEGRATION = "integration"
    REPORTING = "reporting"
    ADMIN_ACTION = "admin_action"
    SECURITY = "security"
    SYSTEM = "system"


class SeverityLevel(str, enum.Enum):
    """Audit event severity (maps to audit.severity_level enum)."""

    DEBUG = "debug"
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"
    EMERGENCY = "emergency"


class EventOutcome(str, enum.Enum):
    """Audit event outcome."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class AuditEvent(BaseModel):
    """Core audit event model (mirrors audit.audit_log schema)."""

    model_config = ConfigDict(from_attributes=True)

    # Event identification
    id: UUID
    event_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Time dimension
    performed_at: datetime

    # Classification
    category: EventCategory
    severity: SeverityLevel
    event_type: str

    # What changed
    schema_name: Optional[str] = None
    table_name: Optional[str] = None
    record_id: Optional[UUID] = None
    operation: str

    # Change details
    old_data: Optional[Dict[str, Any]] = None
    new_data: Optional[Dict[str, Any]] = None
    changed_fields: Optional[List[str]] = None
    change_summary: Optional[str] = None

    # Who
    organization_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    service_account: Optional[str] = None
    impersonated_by: Optional[UUID] = None

    # From where
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[UUID] = None
    session_id: Optional[UUID] = None

    # Context
    resource_type: Optional[str] = None
    resource_path: Optional[str] = None
    action: Optional[str] = None
    outcome: EventOutcome = EventOutcome.SUCCESS
    error_message: Optional[str] = None
    error_code: Optional[str] = None

    # Additional data
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: Optional[List[str]] = None

    # Compliance
    data_classification: Optional[str] = None
    retention_days: Optional[int] = None
    gdpr_relevant: bool = False


class AuditEventSummary(BaseModel):
    """Compact audit event summary for list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    performed_at: datetime
    category: EventCategory
    severity: SeverityLevel
    event_type: str
    operation: str
    user_email: Optional[str] = None
    resource_type: Optional[str] = None
    outcome: EventOutcome


class TimeRange(BaseModel):
    """Time range for queries."""

    start: datetime = Field(..., description="Start of time range (inclusive)")
    end: datetime = Field(..., description="End of time range (exclusive)")


class SearchQuery(BaseModel):
    """Structured search query for audit events."""

    # Text search
    query: Optional[str] = Field(
        None,
        description="LogQL-like query string: field:value AND/OR field:value",
    )

    # Time range
    time_range: Optional[TimeRange] = None

    # Filters
    categories: Optional[List[EventCategory]] = None
    severities: Optional[List[SeverityLevel]] = None
    event_types: Optional[List[str]] = None
    outcomes: Optional[List[EventOutcome]] = None

    # Entity filters
    user_ids: Optional[List[UUID]] = None
    organization_ids: Optional[List[UUID]] = None
    resource_types: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None

    # Aggregations
    aggregations: Optional[List[str]] = Field(
        None,
        description="Fields to aggregate: category, severity, event_type, user_id, etc.",
    )


class AggregationBucket(BaseModel):
    """Single aggregation bucket."""

    key: str
    count: int
    percentage: float = 0.0


class SearchAggregation(BaseModel):
    """Aggregation result for a single field."""

    field: str
    buckets: List[AggregationBucket]
    total: int


class EventStatistics(BaseModel):
    """Audit event statistics."""

    total_events: int
    events_by_category: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_outcome: Dict[str, int]
    unique_users: int
    unique_resources: int
    time_range: TimeRange


class ActivityTimelineEntry(BaseModel):
    """Single entry in activity timeline."""

    timestamp: datetime
    event_type: str
    category: EventCategory
    severity: SeverityLevel
    description: str
    resource_path: Optional[str] = None
    outcome: EventOutcome


class HotspotEntry(BaseModel):
    """Single hotspot entry (top user/resource/IP)."""

    identifier: str
    label: Optional[str] = None
    event_count: int
    last_seen: datetime
    severity_breakdown: Dict[str, int] = Field(default_factory=dict)


class ReportJob(BaseModel):
    """Report generation job status."""

    job_id: str
    report_type: str
    status: str  # pending, processing, completed, failed
    progress_percent: float = 0.0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None


class ExportJob(BaseModel):
    """Export job status."""

    job_id: str
    export_format: str  # csv, json, parquet
    status: str  # pending, processing, completed, failed
    progress_percent: float = 0.0
    total_records: int = 0
    exported_records: int = 0
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    download_url: Optional[str] = None
    file_size_bytes: Optional[int] = None


class StreamFilter(BaseModel):
    """Filter configuration for event streaming."""

    event_types: Optional[List[str]] = None
    tenant_id: Optional[str] = None
    severity_min: Optional[SeverityLevel] = None
    categories: Optional[List[EventCategory]] = None

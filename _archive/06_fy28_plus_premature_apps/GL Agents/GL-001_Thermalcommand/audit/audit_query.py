"""
Audit Query API for GL-001 ThermalCommand

This module provides a comprehensive query interface for audit data retrieval
and export. It supports multiple export formats (JSON, CSV, PDF) and provides
evidence pack retrieval for compliance purposes.

Key Features:
    - Flexible query interface with multiple filter options
    - Export to JSON, CSV, and PDF formats
    - Evidence pack retrieval and bundling
    - Aggregation and statistics
    - Compliance report generation

Example:
    >>> query_api = AuditQueryAPI(audit_logger, evidence_generator)
    >>> results = query_api.query(
    ...     asset_id="boiler-001",
    ...     event_type=EventType.DECISION,
    ...     start_time=datetime(2024, 1, 1)
    ... )
    >>> query_api.export(results, format=ExportFormat.CSV, output_path="audit.csv")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

from .audit_events import (
    AuditEvent,
    DecisionAuditEvent,
    ActionAuditEvent,
    SafetyAuditEvent,
    ComplianceAuditEvent,
    EventType,
    SolverStatus,
    ActionStatus,
    SafetyLevel,
    ComplianceStatus,
)
from .audit_logger import EnhancedAuditLogger
from .evidence_pack import EvidencePack, EvidencePackGenerator, EvidencePackFormat

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


class QuerySortField(str, Enum):
    """Fields available for sorting query results."""

    TIMESTAMP = "timestamp"
    ASSET_ID = "asset_id"
    EVENT_TYPE = "event_type"
    CORRELATION_ID = "correlation_id"


class QuerySortOrder(str, Enum):
    """Sort order for query results."""

    ASC = "asc"
    DESC = "desc"


class QueryFilter(BaseModel):
    """Filter criteria for audit queries."""

    # Entity filters
    asset_id: Optional[str] = Field(None, description="Filter by asset ID")
    asset_ids: Optional[List[str]] = Field(None, description="Filter by multiple asset IDs")
    facility_id: Optional[str] = Field(None, description="Filter by facility ID")
    operator_id: Optional[str] = Field(None, description="Filter by operator ID")
    agent_id: Optional[str] = Field(None, description="Filter by agent ID")

    # Event type filters
    event_type: Optional[EventType] = Field(None, description="Filter by event type")
    event_types: Optional[List[EventType]] = Field(None, description="Filter by multiple types")

    # Time filters
    start_time: Optional[datetime] = Field(None, description="Start time (inclusive)")
    end_time: Optional[datetime] = Field(None, description="End time (exclusive)")
    time_range_days: Optional[int] = Field(None, ge=1, description="Last N days")

    # Correlation/tracing filters
    correlation_id: Optional[str] = Field(None, description="Filter by correlation ID")
    correlation_ids: Optional[List[str]] = Field(None, description="Filter by multiple corr IDs")

    # Safety-specific filters
    boundary_id: Optional[str] = Field(None, description="Filter by safety boundary ID")
    safety_level: Optional[SafetyLevel] = Field(None, description="Filter by safety level")
    is_violation: Optional[bool] = Field(None, description="Filter by violation status")

    # Decision-specific filters
    solver_status: Optional[SolverStatus] = Field(None, description="Filter by solver status")

    # Action-specific filters
    action_status: Optional[ActionStatus] = Field(None, description="Filter by action status")

    # Compliance-specific filters
    regulation_id: Optional[str] = Field(None, description="Filter by regulation ID")
    compliance_status: Optional[ComplianceStatus] = Field(
        None, description="Filter by compliance status"
    )

    # Pagination
    limit: int = Field(100, ge=1, le=10000, description="Maximum results")
    offset: int = Field(0, ge=0, description="Offset for pagination")

    # Sorting
    sort_by: QuerySortField = Field(
        QuerySortField.TIMESTAMP, description="Field to sort by"
    )
    sort_order: QuerySortOrder = Field(QuerySortOrder.DESC, description="Sort order")

    @validator("end_time")
    def validate_time_range(cls, v, values):
        """Validate end_time is after start_time."""
        if v and values.get("start_time") and v <= values["start_time"]:
            raise ValueError("end_time must be after start_time")
        return v

    def apply_time_range(self) -> None:
        """Apply time_range_days to set start_time."""
        if self.time_range_days and not self.start_time:
            self.start_time = datetime.now(timezone.utc) - timedelta(days=self.time_range_days)


class QueryResult(BaseModel):
    """Result of an audit query."""

    events: List[Dict[str, Any]] = Field(default_factory=list, description="Matched events")
    total_count: int = Field(0, ge=0, description="Total matching events")
    returned_count: int = Field(0, ge=0, description="Number returned in this page")
    has_more: bool = Field(False, description="More results available")
    query_time_ms: float = Field(0, ge=0, description="Query execution time in ms")
    filter_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Filters that were applied"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AggregationResult(BaseModel):
    """Result of an aggregation query."""

    group_by: str = Field(..., description="Field used for grouping")
    aggregations: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregation results"
    )
    total_events: int = Field(0, ge=0, description="Total events aggregated")
    time_range: Dict[str, Optional[str]] = Field(
        default_factory=dict, description="Time range covered"
    )


class ComplianceReport(BaseModel):
    """Compliance report for regulatory purposes."""

    report_id: str = Field(..., description="Unique report identifier")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation time"
    )
    generated_by: str = Field(default="GL-001", description="Generator")
    report_type: str = Field(..., description="Type of report")
    period_start: datetime = Field(..., description="Reporting period start")
    period_end: datetime = Field(..., description="Reporting period end")
    facility_id: Optional[str] = Field(None, description="Facility ID")
    asset_ids: List[str] = Field(default_factory=list, description="Assets covered")

    # Summary statistics
    total_decisions: int = Field(0, ge=0, description="Total decisions")
    total_actions: int = Field(0, ge=0, description="Total actions")
    total_safety_events: int = Field(0, ge=0, description="Total safety events")
    total_violations: int = Field(0, ge=0, description="Total violations")
    compliance_checks: int = Field(0, ge=0, description="Compliance checks performed")
    compliance_failures: int = Field(0, ge=0, description="Compliance failures")

    # Detailed data
    events_summary: Dict[str, int] = Field(
        default_factory=dict, description="Events by type"
    )
    safety_summary: Dict[str, int] = Field(
        default_factory=dict, description="Safety events by level"
    )
    compliance_summary: Dict[str, int] = Field(
        default_factory=dict, description="Compliance by status"
    )

    # Evidence packs
    evidence_pack_ids: List[str] = Field(
        default_factory=list, description="Associated evidence pack IDs"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AuditQueryAPI:
    """
    Comprehensive query interface for audit data.

    This class provides methods for querying, aggregating, and exporting
    audit data for compliance and analysis purposes.

    Attributes:
        audit_logger: Enhanced audit logger instance
        evidence_generator: Evidence pack generator instance

    Example:
        >>> api = AuditQueryAPI(audit_logger)
        >>> results = api.query(QueryFilter(asset_id="boiler-001"))
        >>> api.export_to_csv(results.events, "audit_results.csv")
    """

    def __init__(
        self,
        audit_logger: EnhancedAuditLogger,
        evidence_generator: Optional[EvidencePackGenerator] = None,
    ):
        """
        Initialize audit query API.

        Args:
            audit_logger: Enhanced audit logger instance
            evidence_generator: Optional evidence pack generator
        """
        self.audit_logger = audit_logger
        self.evidence_generator = evidence_generator

        logger.info("AuditQueryAPI initialized")

    def query(self, filter: QueryFilter) -> QueryResult:
        """
        Execute an audit query.

        Args:
            filter: Query filter criteria

        Returns:
            QueryResult with matched events
        """
        start_time = datetime.now(timezone.utc)

        # Apply time range if specified
        filter.apply_time_range()

        # Build query parameters
        query_params = {
            "limit": filter.limit,
            "offset": filter.offset,
        }

        # Add filters
        if filter.asset_id:
            query_params["asset_id"] = filter.asset_id
        if filter.event_type:
            query_params["event_type"] = filter.event_type
        if filter.start_time:
            query_params["start_time"] = filter.start_time
        if filter.end_time:
            query_params["end_time"] = filter.end_time
        if filter.operator_id:
            query_params["operator_id"] = filter.operator_id
        if filter.boundary_id:
            query_params["boundary_id"] = filter.boundary_id
        if filter.correlation_id:
            query_params["correlation_id"] = filter.correlation_id

        # Execute query
        events = self.audit_logger.query(**query_params)

        # Apply additional filters not supported by storage backend
        filtered_events = self._apply_additional_filters(events, filter)

        # Apply sorting
        filtered_events = self._apply_sorting(filtered_events, filter)

        # Get total count
        total_count = self.audit_logger.count(
            asset_id=filter.asset_id,
            event_type=filter.event_type,
            start_time=filter.start_time,
            end_time=filter.end_time,
        )

        # Convert to dicts
        event_dicts = [e.dict() for e in filtered_events]

        query_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        result = QueryResult(
            events=event_dicts,
            total_count=total_count,
            returned_count=len(event_dicts),
            has_more=(filter.offset + len(event_dicts)) < total_count,
            query_time_ms=query_time,
            filter_applied=filter.dict(exclude_none=True),
        )

        logger.info(
            f"Query executed: {len(event_dicts)} events returned",
            extra={
                "total_count": total_count,
                "query_time_ms": query_time,
            }
        )

        return result

    def _apply_additional_filters(
        self,
        events: List[AuditEvent],
        filter: QueryFilter,
    ) -> List[AuditEvent]:
        """Apply filters not supported by storage backend."""
        filtered = events

        # Multiple asset IDs
        if filter.asset_ids:
            filtered = [e for e in filtered if e.asset_id in filter.asset_ids]

        # Multiple event types
        if filter.event_types:
            filtered = [e for e in filtered if e.event_type in filter.event_types]

        # Multiple correlation IDs
        if filter.correlation_ids:
            filtered = [e for e in filtered if e.correlation_id in filter.correlation_ids]

        # Facility ID
        if filter.facility_id:
            filtered = [e for e in filtered if e.facility_id == filter.facility_id]

        # Agent ID
        if filter.agent_id:
            filtered = [e for e in filtered if e.agent_id == filter.agent_id]

        # Safety-specific filters
        if filter.safety_level or filter.is_violation is not None:
            safety_filtered = []
            for e in filtered:
                if isinstance(e, SafetyAuditEvent):
                    if filter.safety_level and e.safety_level != filter.safety_level:
                        continue
                    if filter.is_violation is not None and e.is_violation != filter.is_violation:
                        continue
                    safety_filtered.append(e)
                elif filter.safety_level is None and filter.is_violation is None:
                    safety_filtered.append(e)
            filtered = safety_filtered

        # Decision-specific filters
        if filter.solver_status:
            filtered = [
                e for e in filtered
                if isinstance(e, DecisionAuditEvent) and e.solver_status == filter.solver_status
            ]

        # Action-specific filters
        if filter.action_status:
            filtered = [
                e for e in filtered
                if isinstance(e, ActionAuditEvent) and e.action_status == filter.action_status
            ]

        # Compliance-specific filters
        if filter.regulation_id or filter.compliance_status:
            compliance_filtered = []
            for e in filtered:
                if isinstance(e, ComplianceAuditEvent):
                    if filter.regulation_id and e.regulation_id != filter.regulation_id:
                        continue
                    if filter.compliance_status and e.compliance_status != filter.compliance_status:
                        continue
                    compliance_filtered.append(e)
            filtered = compliance_filtered

        return filtered

    def _apply_sorting(
        self,
        events: List[AuditEvent],
        filter: QueryFilter,
    ) -> List[AuditEvent]:
        """Apply sorting to events."""
        reverse = filter.sort_order == QuerySortOrder.DESC

        if filter.sort_by == QuerySortField.TIMESTAMP:
            return sorted(events, key=lambda e: e.timestamp, reverse=reverse)
        elif filter.sort_by == QuerySortField.ASSET_ID:
            return sorted(events, key=lambda e: e.asset_id, reverse=reverse)
        elif filter.sort_by == QuerySortField.EVENT_TYPE:
            return sorted(events, key=lambda e: e.event_type.value, reverse=reverse)
        elif filter.sort_by == QuerySortField.CORRELATION_ID:
            return sorted(events, key=lambda e: e.correlation_id, reverse=reverse)

        return events

    def query_decisions(
        self,
        asset_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        solver_status: Optional[SolverStatus] = None,
        limit: int = 100,
    ) -> List[DecisionAuditEvent]:
        """
        Query decision events specifically.

        Args:
            asset_id: Filter by asset ID
            start_time: Start time filter
            end_time: End time filter
            solver_status: Filter by solver status
            limit: Maximum results

        Returns:
            List of DecisionAuditEvents
        """
        filter = QueryFilter(
            asset_id=asset_id,
            event_type=EventType.DECISION,
            start_time=start_time,
            end_time=end_time,
            solver_status=solver_status,
            limit=limit,
        )
        result = self.query(filter)

        # Convert back to typed events
        decisions = []
        for event_dict in result.events:
            try:
                decisions.append(DecisionAuditEvent(**event_dict))
            except Exception as e:
                logger.warning(f"Failed to parse decision event: {e}")

        return decisions

    def query_safety_events(
        self,
        asset_id: Optional[str] = None,
        boundary_id: Optional[str] = None,
        safety_level: Optional[SafetyLevel] = None,
        is_violation: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SafetyAuditEvent]:
        """
        Query safety events specifically.

        Args:
            asset_id: Filter by asset ID
            boundary_id: Filter by safety boundary ID
            safety_level: Filter by safety level
            is_violation: Filter by violation status
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of SafetyAuditEvents
        """
        filter = QueryFilter(
            asset_id=asset_id,
            event_type=EventType.SAFETY,
            boundary_id=boundary_id,
            safety_level=safety_level,
            is_violation=is_violation,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        result = self.query(filter)

        safety_events = []
        for event_dict in result.events:
            try:
                safety_events.append(SafetyAuditEvent(**event_dict))
            except Exception as e:
                logger.warning(f"Failed to parse safety event: {e}")

        return safety_events

    def aggregate_by_type(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        asset_id: Optional[str] = None,
    ) -> AggregationResult:
        """
        Aggregate events by type.

        Args:
            start_time: Start time filter
            end_time: End time filter
            asset_id: Filter by asset ID

        Returns:
            Aggregation results
        """
        aggregations: Dict[str, int] = {}
        total = 0

        for event_type in EventType:
            count = self.audit_logger.count(
                asset_id=asset_id,
                event_type=event_type,
                start_time=start_time,
                end_time=end_time,
            )
            aggregations[event_type.value] = count
            total += count

        return AggregationResult(
            group_by="event_type",
            aggregations=aggregations,
            total_events=total,
            time_range={
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        )

    def aggregate_by_asset(
        self,
        asset_ids: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AggregationResult:
        """
        Aggregate events by asset.

        Args:
            asset_ids: List of asset IDs to aggregate
            start_time: Start time filter
            end_time: End time filter

        Returns:
            Aggregation results
        """
        aggregations: Dict[str, int] = {}
        total = 0

        for asset_id in asset_ids:
            count = self.audit_logger.count(
                asset_id=asset_id,
                start_time=start_time,
                end_time=end_time,
            )
            aggregations[asset_id] = count
            total += count

        return AggregationResult(
            group_by="asset_id",
            aggregations=aggregations,
            total_events=total,
            time_range={
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None,
            },
        )

    def get_audit_trail(self, correlation_id: str) -> Dict[str, Any]:
        """
        Get complete audit trail for a correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            Complete audit trail
        """
        return self.audit_logger.get_audit_trail(correlation_id)

    def get_evidence_pack(
        self,
        decision_event_id: str,
    ) -> Optional[EvidencePack]:
        """
        Get or generate evidence pack for a decision.

        Args:
            decision_event_id: Decision event ID

        Returns:
            Evidence pack if found/generated
        """
        if not self.evidence_generator:
            raise ValueError("Evidence generator not configured")

        # Get the decision event
        decision_event = self.audit_logger.get_event(decision_event_id)
        if not decision_event or not isinstance(decision_event, DecisionAuditEvent):
            return None

        # Get related events
        correlation_id = decision_event.correlation_id
        related_events = self.audit_logger.query_by_correlation(correlation_id)

        action_events = [e for e in related_events if isinstance(e, ActionAuditEvent)]
        safety_events = [e for e in related_events if isinstance(e, SafetyAuditEvent)]
        compliance_events = [e for e in related_events if isinstance(e, ComplianceAuditEvent)]

        # Generate evidence pack
        return self.evidence_generator.generate(
            decision_event=decision_event,
            action_events=action_events,
            safety_events=safety_events,
            compliance_events=compliance_events,
        )

    def export_to_json(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        pretty: bool = True,
    ) -> str:
        """
        Export events to JSON file.

        Args:
            events: Events to export
            output_path: Output file path
            pretty: Pretty print JSON

        Returns:
            Output file path
        """
        with open(output_path, "w") as f:
            if pretty:
                json.dump(events, f, indent=2, default=str)
            else:
                json.dump(events, f, default=str)

        logger.info(f"Exported {len(events)} events to JSON: {output_path}")
        return output_path

    def export_to_csv(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        fields: Optional[List[str]] = None,
    ) -> str:
        """
        Export events to CSV file.

        Args:
            events: Events to export
            output_path: Output file path
            fields: Fields to include (default: common fields)

        Returns:
            Output file path
        """
        if not events:
            # Write empty file with headers
            default_fields = [
                "event_id", "correlation_id", "event_type", "timestamp",
                "asset_id", "facility_id", "operator_id", "agent_id"
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(fields or default_fields)
            return output_path

        # Determine fields
        if not fields:
            # Use common fields plus any additional from events
            common_fields = [
                "event_id", "correlation_id", "event_type", "timestamp",
                "asset_id", "facility_id", "operator_id", "agent_id"
            ]
            all_fields = set(common_fields)
            for event in events[:10]:  # Sample first 10
                all_fields.update(event.keys())
            fields = common_fields + [f for f in sorted(all_fields) if f not in common_fields]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                # Flatten nested objects
                flat_event = self._flatten_dict(event)
                writer.writerow(flat_event)

        logger.info(f"Exported {len(events)} events to CSV: {output_path}")
        return output_path

    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "_",
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items: List[tuple] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, json.dumps(v, default=str)))
            else:
                items.append((new_key, v))
        return dict(items)

    def export_to_pdf(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
        title: str = "Audit Report",
        include_summary: bool = True,
    ) -> str:
        """
        Export events to PDF file.

        Note: This is a simplified implementation. Production would use
        a proper PDF library like reportlab or weasyprint.

        Args:
            events: Events to export
            output_path: Output file path
            title: Report title
            include_summary: Include summary section

        Returns:
            Output file path
        """
        # For now, generate a text-based report
        # Production would use proper PDF generation
        content = []
        content.append(f"{'=' * 60}")
        content.append(title)
        content.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        content.append(f"{'=' * 60}")
        content.append("")

        if include_summary:
            content.append("SUMMARY")
            content.append("-" * 40)
            content.append(f"Total Events: {len(events)}")

            # Count by type
            type_counts: Dict[str, int] = {}
            for event in events:
                event_type = event.get("event_type", "UNKNOWN")
                type_counts[event_type] = type_counts.get(event_type, 0) + 1

            content.append("Events by Type:")
            for et, count in sorted(type_counts.items()):
                content.append(f"  - {et}: {count}")
            content.append("")

        content.append("EVENTS")
        content.append("-" * 40)

        for i, event in enumerate(events[:100], 1):  # Limit to 100 for PDF
            content.append(f"\n[Event {i}]")
            content.append(f"  ID: {event.get('event_id')}")
            content.append(f"  Type: {event.get('event_type')}")
            content.append(f"  Timestamp: {event.get('timestamp')}")
            content.append(f"  Asset: {event.get('asset_id')}")
            content.append(f"  Correlation: {event.get('correlation_id')}")

        if len(events) > 100:
            content.append(f"\n... and {len(events) - 100} more events")

        # Write as text file (would be PDF in production)
        text_path = output_path.replace(".pdf", ".txt")
        with open(text_path, "w") as f:
            f.write("\n".join(content))

        logger.info(f"Exported {len(events)} events to report: {text_path}")
        return text_path

    def export(
        self,
        result: QueryResult,
        format: ExportFormat,
        output_path: str,
        **kwargs,
    ) -> str:
        """
        Export query results to specified format.

        Args:
            result: Query result to export
            format: Export format
            output_path: Output file path
            **kwargs: Additional format-specific arguments

        Returns:
            Output file path

        Raises:
            ValueError: If format not supported
        """
        if format == ExportFormat.JSON:
            return self.export_to_json(result.events, output_path, **kwargs)
        elif format == ExportFormat.CSV:
            return self.export_to_csv(result.events, output_path, **kwargs)
        elif format == ExportFormat.PDF:
            return self.export_to_pdf(result.events, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def generate_compliance_report(
        self,
        period_start: datetime,
        period_end: datetime,
        facility_id: Optional[str] = None,
        asset_ids: Optional[List[str]] = None,
        report_type: str = "COMPREHENSIVE",
    ) -> ComplianceReport:
        """
        Generate a compliance report for a time period.

        Args:
            period_start: Reporting period start
            period_end: Reporting period end
            facility_id: Filter by facility ID
            asset_ids: Filter by asset IDs
            report_type: Type of report

        Returns:
            Compliance report
        """
        from uuid import uuid4

        # Query events for period
        filter = QueryFilter(
            facility_id=facility_id,
            asset_ids=asset_ids,
            start_time=period_start,
            end_time=period_end,
            limit=100000,
        )
        result = self.query(filter)

        # Count by type
        events_summary: Dict[str, int] = {}
        safety_summary: Dict[str, int] = {}
        compliance_summary: Dict[str, int] = {}

        total_decisions = 0
        total_actions = 0
        total_safety = 0
        total_violations = 0
        compliance_checks = 0
        compliance_failures = 0

        for event in result.events:
            event_type = event.get("event_type")
            events_summary[event_type] = events_summary.get(event_type, 0) + 1

            if event_type == "DECISION":
                total_decisions += 1
            elif event_type == "ACTION":
                total_actions += 1
            elif event_type == "SAFETY":
                total_safety += 1
                level = event.get("safety_level", "UNKNOWN")
                safety_summary[level] = safety_summary.get(level, 0) + 1
                if event.get("is_violation"):
                    total_violations += 1
            elif event_type == "COMPLIANCE":
                compliance_checks += 1
                status = event.get("compliance_status", "UNKNOWN")
                compliance_summary[status] = compliance_summary.get(status, 0) + 1
                if status == "NON_COMPLIANT":
                    compliance_failures += 1

        # Collect evidence pack IDs
        evidence_pack_ids = []
        if self.evidence_generator:
            for event in result.events:
                if event.get("event_type") == "DECISION":
                    evidence_pack_ids.append(str(event.get("event_id")))

        report = ComplianceReport(
            report_id=f"report-{uuid4().hex[:12]}",
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            facility_id=facility_id,
            asset_ids=asset_ids or [],
            total_decisions=total_decisions,
            total_actions=total_actions,
            total_safety_events=total_safety,
            total_violations=total_violations,
            compliance_checks=compliance_checks,
            compliance_failures=compliance_failures,
            events_summary=events_summary,
            safety_summary=safety_summary,
            compliance_summary=compliance_summary,
            evidence_pack_ids=evidence_pack_ids[:100],  # Limit
        )

        logger.info(
            f"Generated compliance report: {report.report_id}",
            extra={
                "period": f"{period_start.date()} to {period_end.date()}",
                "total_events": result.total_count,
            }
        )

        return report

    def verify_chain_integrity(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Verify hash chain integrity.

        Args:
            start_sequence: Starting sequence number
            end_sequence: Ending sequence number

        Returns:
            Verification result
        """
        is_valid, error = self.audit_logger.verify_chain(start_sequence, end_sequence)

        return {
            "is_valid": is_valid,
            "error": error,
            "range_checked": {
                "start": start_sequence,
                "end": end_sequence or "latest",
            },
            "chain_statistics": self.audit_logger.get_chain_statistics(),
        }

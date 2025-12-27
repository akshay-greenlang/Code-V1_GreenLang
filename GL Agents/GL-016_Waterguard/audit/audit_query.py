"""
Audit Query Service for GL-016 Waterguard

This module provides a comprehensive query interface for audit log retrieval,
aggregation, and export. Supports flexible filtering, pagination, and
multiple export formats.

Key Features:
    - Multi-criteria filtering (time, asset, operator, event type)
    - Pagination support for large result sets
    - Aggregation and statistics
    - Export to JSON, CSV formats

Example:
    >>> query_service = AuditQueryService(audit_logger)
    >>> results = query_service.query(QueryFilter(
    ...     asset_id="boiler-001",
    ...     event_type=EventType.CHEMISTRY_CALCULATION,
    ...     start_time=datetime(2024, 1, 1)
    ... ))
    >>> query_service.export_to_csv(results.events, "audit_report.csv")

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from .audit_events import (
    WaterguardAuditEvent,
    ChemistryCalculationEvent,
    RecommendationGeneratedEvent,
    CommandExecutedEvent,
    ConstraintViolationEvent,
    ConfigChangeEvent,
    OperatorActionEvent,
    EventType,
    ChemistryParameter,
    SeverityLevel,
)
from .audit_logger import WaterguardAuditLogger

logger = logging.getLogger(__name__)


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "JSON"
    CSV = "CSV"
    NDJSON = "NDJSON"


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

    # Chemistry-specific filters
    parameter: Optional[ChemistryParameter] = Field(None, description="Filter by chemistry parameter")

    # Violation-specific filters
    severity: Optional[SeverityLevel] = Field(None, description="Filter by severity level")
    is_violation: Optional[bool] = Field(None, description="Filter by violation status")

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

    def apply_time_range(self) -> "QueryFilter":
        """Apply time_range_days to set start_time."""
        if self.time_range_days and not self.start_time:
            self.start_time = datetime.now(timezone.utc) - timedelta(days=self.time_range_days)
        return self


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

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ChemistryStatistics(BaseModel):
    """Statistics for chemistry parameters."""

    parameter: str = Field(..., description="Chemistry parameter")
    count: int = Field(0, ge=0, description="Number of readings")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    avg_value: Optional[float] = Field(None, description="Average value")
    std_dev: Optional[float] = Field(None, description="Standard deviation")
    out_of_spec_count: int = Field(0, ge=0, description="Out of spec readings")


class AuditQueryService:
    """
    Query service for audit log retrieval and analysis.

    Provides comprehensive query, aggregation, and export capabilities
    for Waterguard audit logs.

    Attributes:
        audit_logger: The audit logger to query against

    Example:
        >>> service = AuditQueryService(audit_logger)
        >>> results = service.query(QueryFilter(asset_id="boiler-001"))
        >>> service.export_to_csv(results.events, "report.csv")
    """

    def __init__(self, audit_logger: WaterguardAuditLogger):
        """
        Initialize audit query service.

        Args:
            audit_logger: WaterguardAuditLogger instance
        """
        self.audit_logger = audit_logger
        logger.info("AuditQueryService initialized")

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
        filter = filter.apply_time_range()

        # Execute query against storage
        events = self.audit_logger.query(
            asset_id=filter.asset_id,
            event_type=filter.event_type,
            start_time=filter.start_time,
            end_time=filter.end_time,
            operator_id=filter.operator_id,
            correlation_id=filter.correlation_id,
            limit=filter.limit,
            offset=filter.offset,
        )

        # Apply additional filters not supported by storage
        events = self._apply_additional_filters(events, filter)

        # Apply sorting
        events = self._apply_sorting(events, filter)

        # Get total count
        total_count = self.audit_logger.count(
            asset_id=filter.asset_id,
            event_type=filter.event_type,
            start_time=filter.start_time,
            end_time=filter.end_time,
        )

        # Convert to dicts
        event_dicts = [e.dict() for e in events]

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
        events: List[WaterguardAuditEvent],
        filter: QueryFilter,
    ) -> List[WaterguardAuditEvent]:
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

        # Severity filter for violations
        if filter.severity:
            severity_filtered = []
            for e in filtered:
                if isinstance(e, ConstraintViolationEvent):
                    if e.severity == filter.severity:
                        severity_filtered.append(e)
                elif filter.severity is None:
                    severity_filtered.append(e)
            filtered = severity_filtered

        return filtered

    def _apply_sorting(
        self,
        events: List[WaterguardAuditEvent],
        filter: QueryFilter,
    ) -> List[WaterguardAuditEvent]:
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

    def query_chemistry_events(
        self,
        asset_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ChemistryCalculationEvent]:
        """
        Query chemistry calculation events.

        Args:
            asset_id: Filter by asset ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of ChemistryCalculationEvent
        """
        filter = QueryFilter(
            asset_id=asset_id,
            event_type=EventType.CHEMISTRY_CALCULATION,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        result = self.query(filter)

        return [
            ChemistryCalculationEvent(**e) for e in result.events
        ]

    def query_recommendations(
        self,
        asset_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[RecommendationGeneratedEvent]:
        """
        Query recommendation events.

        Args:
            asset_id: Filter by asset ID
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of RecommendationGeneratedEvent
        """
        filter = QueryFilter(
            asset_id=asset_id,
            event_type=EventType.RECOMMENDATION_GENERATED,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        result = self.query(filter)

        return [
            RecommendationGeneratedEvent(**e) for e in result.events
        ]

    def query_violations(
        self,
        asset_id: Optional[str] = None,
        severity: Optional[SeverityLevel] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ConstraintViolationEvent]:
        """
        Query constraint violation events.

        Args:
            asset_id: Filter by asset ID
            severity: Filter by severity
            start_time: Start time filter
            end_time: End time filter
            limit: Maximum results

        Returns:
            List of ConstraintViolationEvent
        """
        filter = QueryFilter(
            asset_id=asset_id,
            event_type=EventType.CONSTRAINT_VIOLATION,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
        result = self.query(filter)

        return [
            ConstraintViolationEvent(**e) for e in result.events
        ]

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
            Aggregation results by event type
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
            Aggregation results by asset
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

    def get_chemistry_statistics(
        self,
        asset_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, ChemistryStatistics]:
        """
        Get chemistry parameter statistics.

        Args:
            asset_id: Asset ID to analyze
            start_time: Start time filter
            end_time: End time filter

        Returns:
            Dictionary of parameter statistics
        """
        events = self.query_chemistry_events(
            asset_id=asset_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000,
        )

        stats: Dict[str, List[float]] = {}

        for event in events:
            if event.conductivity_us_cm is not None:
                stats.setdefault("conductivity", []).append(event.conductivity_us_cm)
            if event.ph is not None:
                stats.setdefault("ph", []).append(event.ph)
            if event.silica_ppm is not None:
                stats.setdefault("silica", []).append(event.silica_ppm)
            if event.phosphate_ppm is not None:
                stats.setdefault("phosphate", []).append(event.phosphate_ppm)
            if event.dissolved_oxygen_ppb is not None:
                stats.setdefault("dissolved_oxygen", []).append(event.dissolved_oxygen_ppb)

        results = {}
        for param, values in stats.items():
            if values:
                import statistics
                results[param] = ChemistryStatistics(
                    parameter=param,
                    count=len(values),
                    min_value=min(values),
                    max_value=max(values),
                    avg_value=statistics.mean(values),
                    std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
                )

        return results

    def get_audit_trail(self, correlation_id: str) -> Dict[str, Any]:
        """
        Get complete audit trail for a correlation ID.

        Args:
            correlation_id: Correlation ID

        Returns:
            Complete audit trail
        """
        return self.audit_logger.get_audit_trail(correlation_id)

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
            common_fields = [
                "event_id", "correlation_id", "event_type", "timestamp",
                "asset_id", "facility_id", "operator_id", "agent_id"
            ]
            all_fields = set(common_fields)
            for event in events[:10]:
                all_fields.update(event.keys())
            fields = common_fields + [f for f in sorted(all_fields) if f not in common_fields]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                flat_event = self._flatten_dict(event)
                writer.writerow(flat_event)

        logger.info(f"Exported {len(events)} events to CSV: {output_path}")
        return output_path

    def export_to_ndjson(
        self,
        events: List[Dict[str, Any]],
        output_path: str,
    ) -> str:
        """
        Export events to NDJSON (newline-delimited JSON) file.

        Args:
            events: Events to export
            output_path: Output file path

        Returns:
            Output file path
        """
        with open(output_path, "w") as f:
            for event in events:
                f.write(json.dumps(event, default=str) + "\n")

        logger.info(f"Exported {len(events)} events to NDJSON: {output_path}")
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
        elif format == ExportFormat.NDJSON:
            return self.export_to_ndjson(result.events, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")

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

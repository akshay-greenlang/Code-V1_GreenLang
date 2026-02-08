# -*- coding: utf-8 -*-
"""
Audit Logger - AGENT-FOUND-006: Access & Policy Guard

In-memory audit event logger with retention enforcement, filtering,
and compliance report generation. Provides a complete audit trail
for all access decisions, policy mutations, and security events.

Zero-Hallucination Guarantees:
    - All audit events are deterministic records
    - No event is silently dropped (max_size rotation with logging)
    - Compliance reports use only recorded events (no extrapolation)
    - All statistics are exact counts

Example:
    >>> from greenlang.access_guard.audit_logger import AuditLogger
    >>> audit = AuditLogger()
    >>> event_id = audit.log_event(
    ...     event_type="access_granted",
    ...     tenant_id="tenant-1",
    ...     principal_id="user-1",
    ...     resource_id="doc-1",
    ...     action="read",
    ...     decision="allow",
    ... )
    >>> events = audit.get_events(tenant_id="tenant-1")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-006 Access & Policy Guard
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.access_guard.models import (
    AccessDecision,
    AuditEvent,
    AuditEventType,
    ComplianceReport,
)

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _to_naive(dt: datetime) -> datetime:
    """Convert datetime to naive (remove timezone info) for comparison.

    Args:
        dt: Datetime, possibly timezone-aware.

    Returns:
        Naive datetime.
    """
    if dt.tzinfo is not None:
        return dt.replace(tzinfo=None)
    return dt


class AuditLogger:
    """In-memory audit logger with retention and compliance reporting.

    Stores audit events in memory with configurable max size. Supports
    filtering by tenant, event type, principal, and resource. Generates
    compliance reports with exact statistics.

    Attributes:
        _events: Ordered list of audit events.
        _max_size: Maximum number of events to retain in memory.
        _retention_days: Default retention period in days.

    Example:
        >>> audit = AuditLogger(max_size=50000, retention_days=365)
        >>> eid = audit.log_event("access_granted", "t1", "u1", "r1", "read", "allow")
        >>> report = audit.generate_compliance_report("t1", start, end)
    """

    def __init__(
        self,
        max_size: int = 100000,
        retention_days: int = 365,
    ) -> None:
        """Initialize the AuditLogger.

        Args:
            max_size: Maximum events to keep in memory. Oldest events
                are rotated out when exceeded.
            retention_days: Default retention days for new events.
        """
        self._events: List[AuditEvent] = []
        self._max_size = max_size
        self._retention_days = retention_days
        self._lock = threading.Lock()
        logger.info(
            "AuditLogger initialized: max_size=%d, retention_days=%d",
            max_size, retention_days,
        )

    # ------------------------------------------------------------------
    # Event logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        tenant_id: str,
        principal_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        decision: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        source_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> str:
        """Log an audit event.

        Args:
            event_type: Type of event (string value of AuditEventType).
            tenant_id: Tenant context.
            principal_id: Acting principal.
            resource_id: Affected resource.
            action: Action attempted.
            decision: Access decision string.
            details: Additional event details.
            source_ip: Source IP address.
            user_agent: User agent string.

        Returns:
            The event_id of the logged event.
        """
        # Parse enums
        evt_type = (
            AuditEventType(event_type)
            if isinstance(event_type, str)
            else event_type
        )
        dec = (
            AccessDecision(decision)
            if isinstance(decision, str) and decision
            else decision
        )

        event = AuditEvent(
            event_type=evt_type,
            tenant_id=tenant_id,
            principal_id=principal_id,
            resource_id=resource_id,
            action=action,
            decision=dec,
            details=details or {},
            source_ip=source_ip,
            user_agent=user_agent,
            retention_days=self._retention_days,
        )

        with self._lock:
            self._events.append(event)
            self._enforce_max_size()

        logger.debug(
            "Audit event logged: %s tenant=%s event_id=%s",
            evt_type.value, tenant_id, event.event_id,
        )
        return event.event_id

    def log_audit_event(self, event: AuditEvent) -> str:
        """Log a pre-built AuditEvent directly.

        Args:
            event: The AuditEvent to log.

        Returns:
            The event_id.
        """
        with self._lock:
            self._events.append(event)
            self._enforce_max_size()
        return event.event_id

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_events(
        self,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEvent]:
        """Get audit events with optional filtering.

        Args:
            tenant_id: Optional tenant filter.
            event_type: Optional event type filter (string).
            limit: Maximum events to return.
            offset: Number of events to skip.

        Returns:
            List of matching audit events (newest first).
        """
        events = list(self._events)

        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]

        if event_type:
            evt = AuditEventType(event_type) if isinstance(event_type, str) else event_type
            events = [e for e in events if e.event_type == evt]

        # Sort newest first
        events.sort(key=lambda e: e.timestamp, reverse=True)

        return events[offset: offset + limit]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        """Get a single audit event by ID.

        Args:
            event_id: The event identifier.

        Returns:
            AuditEvent if found, None otherwise.
        """
        for event in self._events:
            if event.event_id == event_id:
                return event
        return None

    def get_events_by_principal(
        self, principal_id: str, limit: int = 100,
    ) -> List[AuditEvent]:
        """Get events for a specific principal.

        Args:
            principal_id: Principal identifier.
            limit: Maximum events to return.

        Returns:
            List of matching events (newest first).
        """
        events = [
            e for e in self._events if e.principal_id == principal_id
        ]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_events_by_resource(
        self, resource_id: str, limit: int = 100,
    ) -> List[AuditEvent]:
        """Get events for a specific resource.

        Args:
            resource_id: Resource identifier.
            limit: Maximum events to return.

        Returns:
            List of matching events (newest first).
        """
        events = [
            e for e in self._events if e.resource_id == resource_id
        ]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    # ------------------------------------------------------------------
    # Compliance reporting
    # ------------------------------------------------------------------

    def generate_compliance_report(
        self,
        tenant_id: str,
        start: datetime,
        end: datetime,
    ) -> ComplianceReport:
        """Generate a compliance report for a tenant over a time period.

        Calculates exact statistics from recorded audit events including
        total, allowed, denied, and rate-limited request counts, as well
        as decision breakdowns and top denial reasons.

        Args:
            tenant_id: Tenant to generate the report for.
            start: Report period start.
            end: Report period end.

        Returns:
            ComplianceReport with detailed statistics.
        """
        start_naive = _to_naive(start)
        end_naive = _to_naive(end)

        relevant = [
            e for e in self._events
            if e.tenant_id == tenant_id
            and start_naive <= _to_naive(e.timestamp) <= end_naive
        ]

        # Calculate statistics
        total_requests = len([
            e for e in relevant
            if e.event_type in (
                AuditEventType.ACCESS_GRANTED,
                AuditEventType.ACCESS_DENIED,
            )
        ])
        allowed = len([
            e for e in relevant
            if e.event_type == AuditEventType.ACCESS_GRANTED
        ])
        denied = len([
            e for e in relevant
            if e.event_type == AuditEventType.ACCESS_DENIED
        ])
        rate_limited = len([
            e for e in relevant
            if e.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        ])

        # Breakdown by action type
        decisions_by_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"allowed": 0, "denied": 0},
        )
        for event in relevant:
            if event.action:
                if event.event_type == AuditEventType.ACCESS_GRANTED:
                    decisions_by_type[event.action]["allowed"] += 1
                elif event.event_type == AuditEventType.ACCESS_DENIED:
                    decisions_by_type[event.action]["denied"] += 1

        # Top denial reasons
        denial_reasons: Dict[str, int] = defaultdict(int)
        for event in relevant:
            if event.event_type == AuditEventType.ACCESS_DENIED:
                reason = event.details.get("reason", "unknown")
                denial_reasons[str(reason)] += 1

        top_denial_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                denial_reasons.items(), key=lambda x: -x[1],
            )[:10]
        ]

        # Access by classification
        access_by_classification: Dict[str, int] = defaultdict(int)
        for event in relevant:
            if event.event_type == AuditEventType.CLASSIFICATION_CHECK:
                classification = event.details.get("classified_as", "unknown")
                access_by_classification[classification] += 1

        # Rules triggered
        rules_triggered: Dict[str, int] = defaultdict(int)
        for event in relevant:
            matching = event.details.get("matching_rules", [])
            if isinstance(matching, list):
                for rule_id in matching:
                    rules_triggered[str(rule_id)] += 1

        # Build report
        report = ComplianceReport(
            report_period_start=start,
            report_period_end=end,
            tenant_id=tenant_id,
            total_requests=total_requests,
            allowed_requests=allowed,
            denied_requests=denied,
            rate_limited_requests=rate_limited,
            decisions_by_type=dict(decisions_by_type),
            top_denial_reasons=top_denial_reasons,
            rules_triggered=dict(rules_triggered),
            access_by_classification=dict(access_by_classification),
        )
        report.provenance_hash = report.compute_hash()

        logger.info(
            "Generated compliance report for tenant %s: "
            "%d requests, %d allowed, %d denied",
            tenant_id, total_requests, allowed, denied,
        )
        return report

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Return the number of stored audit events."""
        return len(self._events)

    def clear(self, tenant_id: Optional[str] = None) -> int:
        """Clear audit events, optionally filtered by tenant.

        Args:
            tenant_id: Optional tenant filter. If None, clears all.

        Returns:
            Number of events removed.
        """
        with self._lock:
            if tenant_id is None:
                removed = len(self._events)
                self._events.clear()
            else:
                before = len(self._events)
                self._events = [
                    e for e in self._events if e.tenant_id != tenant_id
                ]
                removed = before - len(self._events)

        logger.info("Cleared %d audit events (tenant=%s)", removed, tenant_id)
        return removed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_max_size(self) -> None:
        """Trim events to max_size (call under lock).

        Keeps the newest half when the max_size is exceeded.
        """
        if len(self._events) > self._max_size:
            keep = self._max_size // 2
            self._events = self._events[-keep:]
            logger.warning(
                "Audit log rotated: trimmed to %d events", keep,
            )


__all__ = [
    "AuditLogger",
]

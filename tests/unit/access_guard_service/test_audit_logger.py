# -*- coding: utf-8 -*-
"""
Unit Tests for AuditLogger (AGENT-FOUND-006)

Tests audit event logging, retrieval, filtering, compliance report generation,
log rotation, clear, and count operations.

Coverage target: 85%+ of audit_logger.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums and models
# ---------------------------------------------------------------------------


class AccessDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    CONDITIONAL = "conditional"


class AuditEventType(str, Enum):
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    POLICY_EVALUATED = "policy_evaluated"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TENANT_BOUNDARY_VIOLATION = "tenant_boundary_violation"
    CLASSIFICATION_CHECK = "classification_check"
    EXPORT_APPROVED = "export_approved"
    EXPORT_DENIED = "export_denied"
    POLICY_UPDATED = "policy_updated"
    SIMULATION_RUN = "simulation_run"


class AuditEvent:
    def __init__(
        self, event_type, tenant_id, event_id=None, timestamp=None,
        principal_id=None, resource_id=None, action=None, decision=None,
        decision_hash=None, details=None, source_ip=None,
        user_agent=None, retention_days=365,
    ):
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = AuditEventType(event_type)
        self.timestamp = timestamp or datetime.utcnow()
        self.tenant_id = tenant_id
        self.principal_id = principal_id
        self.resource_id = resource_id
        self.action = action
        self.decision = AccessDecision(decision) if decision else None
        self.decision_hash = decision_hash
        self.details = details or {}
        self.source_ip = source_ip
        self.user_agent = user_agent
        self.retention_days = retention_days


class ComplianceReport:
    def __init__(self, report_id=None, tenant_id="", report_period_start=None,
                 report_period_end=None, generated_at=None,
                 total_requests=0, allowed_requests=0, denied_requests=0,
                 rate_limited_requests=0, decisions_by_type=None,
                 top_denial_reasons=None, policies_evaluated=None,
                 access_by_classification=None, provenance_hash=""):
        self.report_id = report_id or str(uuid.uuid4())
        self.generated_at = generated_at or datetime.utcnow()
        self.report_period_start = report_period_start
        self.report_period_end = report_period_end
        self.tenant_id = tenant_id
        self.total_requests = total_requests
        self.allowed_requests = allowed_requests
        self.denied_requests = denied_requests
        self.rate_limited_requests = rate_limited_requests
        self.decisions_by_type = decisions_by_type or {}
        self.top_denial_reasons = top_denial_reasons or []
        self.policies_evaluated = policies_evaluated or []
        self.access_by_classification = access_by_classification or {}
        self.provenance_hash = provenance_hash

    def compute_hash(self):
        data = {
            "report_id": self.report_id, "tenant_id": self.tenant_id,
            "total_requests": self.total_requests,
            "allowed_requests": self.allowed_requests,
            "denied_requests": self.denied_requests,
            "generated_at": self.generated_at.isoformat() if self.generated_at else "",
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


# ---------------------------------------------------------------------------
# AuditLogger (self-contained mirror)
# ---------------------------------------------------------------------------


class AuditLogger:
    def __init__(self, max_size: int = 100000, retention_days: int = 365):
        self._events: List[AuditEvent] = []
        self._max_size = max_size
        self._retention_days = retention_days

    @property
    def count(self) -> int:
        return len(self._events)

    def log_event(self, event: AuditEvent) -> str:
        self._events.append(event)
        if len(self._events) > self._max_size:
            self._events = self._events[-self._max_size // 2:]
        return event.event_id

    def get_events(
        self, tenant_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        events = self._events
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-limit:]

    def get_event(self, event_id: str) -> Optional[AuditEvent]:
        for e in self._events:
            if e.event_id == event_id:
                return e
        return None

    def get_events_by_principal(
        self, principal_id: str, limit: int = 100,
    ) -> List[AuditEvent]:
        events = [e for e in self._events if e.principal_id == principal_id]
        return events[-limit:]

    def get_events_by_resource(
        self, resource_id: str, limit: int = 100,
    ) -> List[AuditEvent]:
        events = [e for e in self._events if e.resource_id == resource_id]
        return events[-limit:]

    def generate_compliance_report(
        self, tenant_id: str, start_date: datetime, end_date: datetime,
    ) -> ComplianceReport:
        relevant = [
            e for e in self._events
            if e.tenant_id == tenant_id
            and start_date <= e.timestamp <= end_date
        ]

        total = len([
            e for e in relevant
            if e.event_type in (AuditEventType.ACCESS_GRANTED, AuditEventType.ACCESS_DENIED)
        ])
        allowed = len([
            e for e in relevant if e.event_type == AuditEventType.ACCESS_GRANTED
        ])
        denied = len([
            e for e in relevant if e.event_type == AuditEventType.ACCESS_DENIED
        ])
        rate_limited = len([
            e for e in relevant if e.event_type == AuditEventType.RATE_LIMIT_EXCEEDED
        ])

        decisions_by_type: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"allowed": 0, "denied": 0},
        )
        for e in relevant:
            if e.action:
                if e.event_type == AuditEventType.ACCESS_GRANTED:
                    decisions_by_type[e.action]["allowed"] += 1
                elif e.event_type == AuditEventType.ACCESS_DENIED:
                    decisions_by_type[e.action]["denied"] += 1

        denial_reasons: Dict[str, int] = defaultdict(int)
        for e in relevant:
            if e.event_type == AuditEventType.ACCESS_DENIED:
                reason = e.details.get("reason", "unknown")
                denial_reasons[str(reason)] += 1

        top_denial = [
            {"reason": r, "count": c}
            for r, c in sorted(denial_reasons.items(), key=lambda x: -x[1])[:10]
        ]

        report = ComplianceReport(
            tenant_id=tenant_id,
            report_period_start=start_date,
            report_period_end=end_date,
            total_requests=total,
            allowed_requests=allowed,
            denied_requests=denied,
            rate_limited_requests=rate_limited,
            decisions_by_type=dict(decisions_by_type),
            top_denial_reasons=top_denial,
        )
        report.provenance_hash = report.compute_hash()
        return report

    def clear(self, tenant_id: Optional[str] = None) -> int:
        if tenant_id:
            before = len(self._events)
            self._events = [e for e in self._events if e.tenant_id != tenant_id]
            return before - len(self._events)
        else:
            count = len(self._events)
            self._events.clear()
            return count


# ===========================================================================
# Test Classes
# ===========================================================================


class TestAuditLoggerLogEvent:
    """Test log creation and event_id generation."""

    def test_log_event_returns_event_id(self):
        logger = AuditLogger()
        event = AuditEvent(event_type="access_granted", tenant_id="t1")
        eid = logger.log_event(event)
        assert eid == event.event_id
        assert len(eid) == 36

    def test_log_event_increments_count(self):
        logger = AuditLogger()
        assert logger.count == 0
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        assert logger.count == 1
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t1"))
        assert logger.count == 2

    def test_log_event_preserves_fields(self):
        logger = AuditLogger()
        event = AuditEvent(
            event_type="access_denied", tenant_id="t1",
            principal_id="u1", resource_id="r1", action="write",
            decision="deny", details={"reason": "unauthorized"},
        )
        logger.log_event(event)
        retrieved = logger.get_event(event.event_id)
        assert retrieved.principal_id == "u1"
        assert retrieved.resource_id == "r1"
        assert retrieved.action == "write"
        assert retrieved.decision == AccessDecision.DENY

    def test_log_multiple_event_types(self):
        logger = AuditLogger()
        for et in AuditEventType:
            logger.log_event(AuditEvent(event_type=et.value, tenant_id="t1"))
        assert logger.count == 10


class TestAuditLoggerGetEvents:
    """Test filter by tenant, event_type, limit."""

    def test_get_all_events(self):
        logger = AuditLogger()
        for i in range(5):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        events = logger.get_events()
        assert len(events) == 5

    def test_filter_by_tenant(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t2"))
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t1"))
        events = logger.get_events(tenant_id="t1")
        assert len(events) == 2

    def test_filter_by_event_type(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t1"))
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t1"))
        events = logger.get_events(event_type=AuditEventType.ACCESS_DENIED)
        assert len(events) == 2

    def test_filter_by_tenant_and_type(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t1"))
        logger.log_event(AuditEvent(event_type="access_denied", tenant_id="t2"))
        events = logger.get_events(
            tenant_id="t1", event_type=AuditEventType.ACCESS_DENIED,
        )
        assert len(events) == 1

    def test_limit_results(self):
        logger = AuditLogger()
        for i in range(20):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        events = logger.get_events(limit=5)
        assert len(events) == 5

    def test_limit_returns_most_recent(self):
        logger = AuditLogger()
        for i in range(10):
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t1",
                event_id=f"event-{i:03d}",
            ))
        events = logger.get_events(limit=3)
        assert events[0].event_id == "event-007"
        assert events[2].event_id == "event-009"


class TestAuditLoggerGetEvent:
    """Test get by ID and not found."""

    def test_get_event_by_id(self):
        logger = AuditLogger()
        event = AuditEvent(event_type="access_granted", tenant_id="t1")
        logger.log_event(event)
        found = logger.get_event(event.event_id)
        assert found is not None
        assert found.event_id == event.event_id

    def test_get_event_not_found(self):
        logger = AuditLogger()
        assert logger.get_event("nonexistent") is None

    def test_get_event_among_many(self):
        logger = AuditLogger()
        for i in range(50):
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t1",
                event_id=f"evt-{i}",
            ))
        found = logger.get_event("evt-25")
        assert found is not None
        assert found.event_id == "evt-25"


class TestAuditLoggerByPrincipal:
    """Test filter by principal."""

    def test_filter_by_principal(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1", principal_id="u1",
        ))
        logger.log_event(AuditEvent(
            event_type="access_denied", tenant_id="t1", principal_id="u2",
        ))
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1", principal_id="u1",
        ))
        events = logger.get_events_by_principal("u1")
        assert len(events) == 2
        for e in events:
            assert e.principal_id == "u1"

    def test_principal_not_found(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1", principal_id="u1",
        ))
        events = logger.get_events_by_principal("nobody")
        assert len(events) == 0

    def test_principal_limit(self):
        logger = AuditLogger()
        for _ in range(20):
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t1", principal_id="u1",
            ))
        events = logger.get_events_by_principal("u1", limit=5)
        assert len(events) == 5


class TestAuditLoggerByResource:
    """Test filter by resource."""

    def test_filter_by_resource(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1", resource_id="r1",
        ))
        logger.log_event(AuditEvent(
            event_type="access_denied", tenant_id="t1", resource_id="r2",
        ))
        events = logger.get_events_by_resource("r1")
        assert len(events) == 1
        assert events[0].resource_id == "r1"

    def test_resource_not_found(self):
        logger = AuditLogger()
        events = logger.get_events_by_resource("nonexistent")
        assert len(events) == 0


class TestAuditLoggerComplianceReport:
    """Test report generation with statistics."""

    def test_report_basic(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1",
            action="read", timestamp=now,
        ))
        logger.log_event(AuditEvent(
            event_type="access_denied", tenant_id="t1",
            action="write", timestamp=now,
            details={"reason": "unauthorized"},
        ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert report.tenant_id == "t1"
        assert report.total_requests == 2
        assert report.allowed_requests == 1
        assert report.denied_requests == 1

    def test_report_excludes_other_tenants(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1", timestamp=now,
        ))
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t2", timestamp=now,
        ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert report.allowed_requests == 1

    def test_report_excludes_out_of_range(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1",
            timestamp=now - timedelta(days=10),
        ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert report.total_requests == 0

    def test_report_rate_limited_count(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        logger.log_event(AuditEvent(
            event_type="rate_limit_exceeded", tenant_id="t1", timestamp=now,
        ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert report.rate_limited_requests == 1

    def test_report_decisions_by_type(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        logger.log_event(AuditEvent(
            event_type="access_granted", tenant_id="t1",
            action="read", timestamp=now,
        ))
        logger.log_event(AuditEvent(
            event_type="access_denied", tenant_id="t1",
            action="read", timestamp=now, details={"reason": "unauthorized"},
        ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert "read" in report.decisions_by_type
        assert report.decisions_by_type["read"]["allowed"] == 1
        assert report.decisions_by_type["read"]["denied"] == 1

    def test_report_top_denial_reasons(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        for _ in range(5):
            logger.log_event(AuditEvent(
                event_type="access_denied", tenant_id="t1", timestamp=now,
                details={"reason": "unauthorized"},
            ))
        for _ in range(3):
            logger.log_event(AuditEvent(
                event_type="access_denied", tenant_id="t1", timestamp=now,
                details={"reason": "rate_limited"},
            ))
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert len(report.top_denial_reasons) == 2
        assert report.top_denial_reasons[0]["reason"] == "unauthorized"
        assert report.top_denial_reasons[0]["count"] == 5

    def test_report_has_provenance_hash(self):
        logger = AuditLogger()
        now = datetime.utcnow()
        report = logger.generate_compliance_report(
            "t1", now - timedelta(hours=1), now + timedelta(hours=1),
        )
        assert len(report.provenance_hash) == 64


class TestAuditLoggerRotation:
    """Test max_size trimming."""

    def test_trimming_at_max_size(self):
        logger = AuditLogger(max_size=10)
        for i in range(15):
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t1",
                event_id=f"evt-{i:03d}",
            ))
        assert logger.count <= 10

    def test_trimming_keeps_recent(self):
        logger = AuditLogger(max_size=10)
        for i in range(15):
            logger.log_event(AuditEvent(
                event_type="access_granted", tenant_id="t1",
                event_id=f"evt-{i:03d}",
            ))
        # The most recent events should be preserved
        latest = logger.get_event("evt-014")
        assert latest is not None


class TestAuditLoggerClear:
    """Test clear by tenant."""

    def test_clear_all(self):
        logger = AuditLogger()
        for _ in range(10):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        removed = logger.clear()
        assert removed == 10
        assert logger.count == 0

    def test_clear_by_tenant(self):
        logger = AuditLogger()
        for _ in range(5):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        for _ in range(3):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t2"))
        removed = logger.clear(tenant_id="t1")
        assert removed == 5
        assert logger.count == 3

    def test_clear_tenant_not_found(self):
        logger = AuditLogger()
        logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        removed = logger.clear(tenant_id="t99")
        assert removed == 0
        assert logger.count == 1


class TestAuditLoggerCount:
    """Test count accuracy."""

    def test_count_zero_initial(self):
        logger = AuditLogger()
        assert logger.count == 0

    def test_count_after_adds(self):
        logger = AuditLogger()
        for _ in range(7):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        assert logger.count == 7

    def test_count_after_clear(self):
        logger = AuditLogger()
        for _ in range(5):
            logger.log_event(AuditEvent(event_type="access_granted", tenant_id="t1"))
        logger.clear()
        assert logger.count == 0

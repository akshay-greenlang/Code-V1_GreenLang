"""
AuditLogger - Comprehensive audit trail management.

This module provides enterprise-grade audit logging for the GreenLang
process heat ecosystem. It supports regulatory compliance, security
auditing, and operational diagnostics.

Features:
    - Structured audit events with severity levels
    - Compliance-specific audit trails (SOX, ISO 14064)
    - Tamper-evident logging with hash chains
    - Real-time audit event streaming
    - Audit report generation
    - Long-term retention support

Example:
    >>> from greenlang.agents.process_heat.shared import AuditLogger
    >>>
    >>> audit = AuditLogger(agent_id="GL-002-001")
    >>> audit.log_event(
    ...     event_type="CALCULATION_COMPLETED",
    ...     level=AuditLevel.INFO,
    ...     data={"efficiency": 85.5}
    ... )
"""

from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60


class AuditCategory(Enum):
    """Audit event categories."""
    AUTHENTICATION = auto()
    AUTHORIZATION = auto()
    CONFIGURATION = auto()
    DATA_ACCESS = auto()
    DATA_MODIFICATION = auto()
    CALCULATION = auto()
    VALIDATION = auto()
    INTEGRATION = auto()
    SAFETY = auto()
    COMPLIANCE = auto()
    PERFORMANCE = auto()
    SYSTEM = auto()


class ComplianceStandard(Enum):
    """Compliance standards for audit trails."""
    SOX = "sox"
    ISO_14064 = "iso_14064"
    ISO_50001 = "iso_50001"
    GHG_PROTOCOL = "ghg_protocol"
    EPA_40_CFR = "epa_40_cfr"
    EU_ETS = "eu_ets"
    CSRD = "csrd"
    SEC_CLIMATE = "sec_climate"


# =============================================================================
# DATA MODELS
# =============================================================================

class AuditEvent(BaseModel):
    """Structured audit event."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    level: AuditLevel = Field(..., description="Event severity level")
    category: AuditCategory = Field(..., description="Event category")
    event_type: str = Field(..., description="Specific event type")
    agent_id: str = Field(..., description="Source agent ID")
    agent_version: str = Field(default="1.0.0", description="Agent version")
    user_id: Optional[str] = Field(default=None, description="User ID if applicable")
    session_id: Optional[str] = Field(default=None, description="Session ID")
    correlation_id: Optional[str] = Field(
        default=None,
        description="Correlation ID for tracking"
    )
    message: str = Field(..., description="Human-readable message")
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event data payload"
    )
    source_ip: Optional[str] = Field(default=None, description="Source IP address")
    resource: Optional[str] = Field(default=None, description="Affected resource")
    action: Optional[str] = Field(default=None, description="Action performed")
    outcome: Optional[str] = Field(
        default=None,
        description="Action outcome (success, failure)"
    )
    duration_ms: Optional[float] = Field(
        default=None,
        description="Action duration in milliseconds"
    )
    previous_hash: Optional[str] = Field(
        default=None,
        description="Hash of previous event (chain)"
    )
    event_hash: Optional[str] = Field(
        default=None,
        description="Hash of this event"
    )
    compliance_tags: List[ComplianceStandard] = Field(
        default_factory=list,
        description="Applicable compliance standards"
    )

    class Config:
        use_enum_values = True


class ComplianceAuditTrail(BaseModel):
    """Compliance-specific audit trail."""

    trail_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Trail identifier"
    )
    standard: ComplianceStandard = Field(..., description="Compliance standard")
    start_time: datetime = Field(..., description="Trail start time")
    end_time: Optional[datetime] = Field(default=None, description="Trail end time")
    event_count: int = Field(default=0, description="Number of events")
    hash_chain_root: Optional[str] = Field(
        default=None,
        description="Root hash of event chain"
    )
    events: List[str] = Field(
        default_factory=list,
        description="Event IDs in trail"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Trail metadata"
    )

    class Config:
        use_enum_values = True


class AuditReport(BaseModel):
    """Audit report summary."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report generation time"
    )
    period_start: datetime = Field(..., description="Reporting period start")
    period_end: datetime = Field(..., description="Reporting period end")
    total_events: int = Field(default=0, description="Total events in period")
    events_by_level: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by level"
    )
    events_by_category: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by category"
    )
    critical_events: List[AuditEvent] = Field(
        default_factory=list,
        description="Critical/security events"
    )
    compliance_summary: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Compliance summary by standard"
    )
    agent_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Event counts by agent"
    )


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Enterprise-grade audit logging system.

    This class provides comprehensive audit logging with support for
    regulatory compliance, tamper-evident logging, and real-time
    event streaming.

    Features:
        - Structured audit events
        - Hash chain for tamper detection
        - Compliance-specific trails
        - Real-time event streaming
        - Report generation
        - Long-term retention

    Example:
        >>> audit = AuditLogger(agent_id="GL-002-001")
        >>> audit.log_event(
        ...     event_type="EFFICIENCY_CALCULATED",
        ...     level=AuditLevel.INFO,
        ...     category=AuditCategory.CALCULATION,
        ...     message="Boiler efficiency calculated",
        ...     data={"efficiency": 85.5, "boiler_id": "B-001"}
        ... )
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "1.0.0",
        enable_hash_chain: bool = True,
        retention_days: int = 365,
    ) -> None:
        """
        Initialize the audit logger.

        Args:
            agent_id: Agent identifier
            agent_version: Agent version string
            enable_hash_chain: Enable tamper-evident hash chain
            retention_days: Default retention period in days
        """
        self.agent_id = agent_id
        self.agent_version = agent_version
        self.enable_hash_chain = enable_hash_chain
        self.retention_days = retention_days

        self._events: List[AuditEvent] = []
        self._compliance_trails: Dict[ComplianceStandard, ComplianceAuditTrail] = {}
        self._event_handlers: List[Callable[[AuditEvent], None]] = []
        self._lock = threading.RLock()
        self._last_hash: Optional[str] = None

        logger.info(f"AuditLogger initialized for agent {agent_id}")

    # =========================================================================
    # CORE LOGGING
    # =========================================================================

    def log_event(
        self,
        event_type: str,
        level: AuditLevel,
        category: AuditCategory,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        duration_ms: Optional[float] = None,
        compliance_tags: Optional[List[ComplianceStandard]] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Specific event type identifier
            level: Event severity level
            category: Event category
            message: Human-readable message
            data: Event data payload
            user_id: User ID if applicable
            session_id: Session ID for tracking
            correlation_id: Correlation ID for distributed tracing
            resource: Affected resource identifier
            action: Action performed
            outcome: Action outcome
            duration_ms: Action duration
            compliance_tags: Applicable compliance standards

        Returns:
            Created AuditEvent
        """
        with self._lock:
            # Create event
            event = AuditEvent(
                level=level,
                category=category,
                event_type=event_type,
                agent_id=self.agent_id,
                agent_version=self.agent_version,
                user_id=user_id,
                session_id=session_id,
                correlation_id=correlation_id,
                message=message,
                data=data or {},
                resource=resource,
                action=action,
                outcome=outcome,
                duration_ms=duration_ms,
                compliance_tags=compliance_tags or [],
            )

            # Add to hash chain
            if self.enable_hash_chain:
                event.previous_hash = self._last_hash
                event.event_hash = self._calculate_event_hash(event)
                self._last_hash = event.event_hash

            # Store event
            self._events.append(event)

            # Update compliance trails
            for tag in event.compliance_tags:
                self._add_to_compliance_trail(tag, event)

            # Notify handlers
            for handler in self._event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Audit handler error: {e}")

            # Log to standard logger as well
            log_level = self._audit_to_log_level(level)
            logger.log(
                log_level,
                f"[AUDIT] {event_type}: {message}",
                extra={"audit_event_id": event.event_id}
            )

            return event

    def log_calculation(
        self,
        calculation_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        formula_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
        provenance_hash: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log a calculation event.

        Args:
            calculation_type: Type of calculation
            inputs: Calculation inputs
            outputs: Calculation outputs
            formula_id: Formula identifier
            duration_ms: Calculation duration
            provenance_hash: Provenance hash reference

        Returns:
            AuditEvent
        """
        return self.log_event(
            event_type="CALCULATION_EXECUTED",
            level=AuditLevel.INFO,
            category=AuditCategory.CALCULATION,
            message=f"Calculation executed: {calculation_type}",
            data={
                "calculation_type": calculation_type,
                "inputs_summary": self._summarize_data(inputs),
                "outputs_summary": self._summarize_data(outputs),
                "formula_id": formula_id,
                "provenance_hash": provenance_hash,
            },
            action="calculate",
            outcome="success",
            duration_ms=duration_ms,
            compliance_tags=[
                ComplianceStandard.ISO_14064,
                ComplianceStandard.GHG_PROTOCOL,
            ],
        )

    def log_validation(
        self,
        validation_type: str,
        target: str,
        rules_checked: int,
        rules_passed: int,
        failures: Optional[List[str]] = None,
    ) -> AuditEvent:
        """
        Log a validation event.

        Args:
            validation_type: Type of validation
            target: Validation target
            rules_checked: Number of rules checked
            rules_passed: Number of rules passed
            failures: List of failure messages

        Returns:
            AuditEvent
        """
        outcome = "success" if rules_checked == rules_passed else "failure"
        level = AuditLevel.INFO if outcome == "success" else AuditLevel.WARNING

        return self.log_event(
            event_type="VALIDATION_COMPLETED",
            level=level,
            category=AuditCategory.VALIDATION,
            message=f"Validation {outcome}: {validation_type}",
            data={
                "validation_type": validation_type,
                "target": target,
                "rules_checked": rules_checked,
                "rules_passed": rules_passed,
                "rules_failed": rules_checked - rules_passed,
                "failures": failures or [],
            },
            action="validate",
            outcome=outcome,
        )

    def log_safety_event(
        self,
        event_type: str,
        severity: str,
        description: str,
        affected_equipment: Optional[List[str]] = None,
        response_action: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log a safety-related event.

        Args:
            event_type: Safety event type
            severity: Event severity (low, medium, high, critical)
            description: Event description
            affected_equipment: List of affected equipment
            response_action: Action taken in response

        Returns:
            AuditEvent
        """
        level_map = {
            "low": AuditLevel.INFO,
            "medium": AuditLevel.WARNING,
            "high": AuditLevel.ERROR,
            "critical": AuditLevel.CRITICAL,
        }
        level = level_map.get(severity.lower(), AuditLevel.WARNING)

        return self.log_event(
            event_type=f"SAFETY_{event_type.upper()}",
            level=level,
            category=AuditCategory.SAFETY,
            message=f"Safety event: {description}",
            data={
                "severity": severity,
                "affected_equipment": affected_equipment or [],
                "response_action": response_action,
            },
            action="safety_response",
            outcome="logged",
        )

    def log_data_access(
        self,
        resource: str,
        action: str,
        user_id: str,
        granted: bool,
        reason: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log a data access event.

        Args:
            resource: Accessed resource
            action: Access action (read, write, delete)
            user_id: Accessing user ID
            granted: Whether access was granted
            reason: Denial reason if not granted

        Returns:
            AuditEvent
        """
        level = AuditLevel.INFO if granted else AuditLevel.WARNING
        outcome = "granted" if granted else "denied"

        return self.log_event(
            event_type="DATA_ACCESS",
            level=level,
            category=AuditCategory.DATA_ACCESS,
            message=f"Data access {outcome}: {action} on {resource}",
            data={
                "access_action": action,
                "granted": granted,
                "denial_reason": reason,
            },
            user_id=user_id,
            resource=resource,
            action=action,
            outcome=outcome,
            compliance_tags=[ComplianceStandard.SOX],
        )

    def log_security_event(
        self,
        event_type: str,
        description: str,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        threat_level: str = "medium",
    ) -> AuditEvent:
        """
        Log a security event.

        Args:
            event_type: Security event type
            description: Event description
            source_ip: Source IP address
            user_id: Associated user ID
            threat_level: Threat level (low, medium, high, critical)

        Returns:
            AuditEvent
        """
        return self.log_event(
            event_type=f"SECURITY_{event_type.upper()}",
            level=AuditLevel.SECURITY,
            category=AuditCategory.AUTHORIZATION,
            message=description,
            data={
                "threat_level": threat_level,
            },
            user_id=user_id,
            source_ip=source_ip,
            action="security_alert",
            outcome="logged",
            compliance_tags=[ComplianceStandard.SOX],
        )

    # =========================================================================
    # COMPLIANCE TRAILS
    # =========================================================================

    def start_compliance_trail(
        self,
        standard: ComplianceStandard,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceAuditTrail:
        """
        Start a compliance-specific audit trail.

        Args:
            standard: Compliance standard
            metadata: Trail metadata

        Returns:
            Created ComplianceAuditTrail
        """
        with self._lock:
            trail = ComplianceAuditTrail(
                standard=standard,
                start_time=datetime.now(timezone.utc),
                metadata=metadata or {},
            )
            self._compliance_trails[standard] = trail
            logger.info(f"Started compliance trail for {standard.value}")
            return trail

    def close_compliance_trail(
        self,
        standard: ComplianceStandard,
    ) -> Optional[ComplianceAuditTrail]:
        """
        Close a compliance audit trail.

        Args:
            standard: Compliance standard

        Returns:
            Closed trail or None if not found
        """
        with self._lock:
            trail = self._compliance_trails.get(standard)
            if trail:
                trail.end_time = datetime.now(timezone.utc)
                trail.hash_chain_root = self._calculate_trail_hash(trail)
                logger.info(f"Closed compliance trail for {standard.value}")
            return trail

    def _add_to_compliance_trail(
        self,
        standard: ComplianceStandard,
        event: AuditEvent,
    ) -> None:
        """Add event to compliance trail."""
        trail = self._compliance_trails.get(standard)
        if trail and trail.end_time is None:
            trail.events.append(event.event_id)
            trail.event_count += 1

    def get_compliance_trail(
        self,
        standard: ComplianceStandard,
    ) -> Optional[ComplianceAuditTrail]:
        """Get compliance trail for a standard."""
        return self._compliance_trails.get(standard)

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def register_handler(
        self,
        handler: Callable[[AuditEvent], None],
    ) -> None:
        """Register an event handler for real-time notifications."""
        self._event_handlers.append(handler)

    def unregister_handler(
        self,
        handler: Callable[[AuditEvent], None],
    ) -> None:
        """Unregister an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_events(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[AuditLevel] = None,
        category: Optional[AuditCategory] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            level: Filter by severity level
            category: Filter by category
            event_type: Filter by event type
            user_id: Filter by user ID
            correlation_id: Filter by correlation ID
            limit: Maximum events to return

        Returns:
            List of matching AuditEvents
        """
        with self._lock:
            events = self._events.copy()

        # Apply filters
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if level:
            events = [e for e in events if e.level == level]
        if category:
            events = [e for e in events if e.category == category]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if correlation_id:
            events = [e for e in events if e.correlation_id == correlation_id]

        # Sort by timestamp descending and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    def get_event_by_id(self, event_id: str) -> Optional[AuditEvent]:
        """Get a specific event by ID."""
        with self._lock:
            for event in self._events:
                if event.event_id == event_id:
                    return event
        return None

    # =========================================================================
    # REPORTS
    # =========================================================================

    def generate_report(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> AuditReport:
        """
        Generate an audit report for a time period.

        Args:
            start_time: Report period start
            end_time: Report period end

        Returns:
            AuditReport summary
        """
        events = self.get_events(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
        )

        # Count by level
        events_by_level: Dict[str, int] = defaultdict(int)
        for event in events:
            level_name = event.level.name if hasattr(event.level, "name") else str(event.level)
            events_by_level[level_name] += 1

        # Count by category
        events_by_category: Dict[str, int] = defaultdict(int)
        for event in events:
            cat_name = event.category.name if hasattr(event.category, "name") else str(event.category)
            events_by_category[cat_name] += 1

        # Count by agent
        agent_summary: Dict[str, int] = defaultdict(int)
        for event in events:
            agent_summary[event.agent_id] += 1

        # Get critical events
        critical_events = [
            e for e in events
            if e.level in [AuditLevel.CRITICAL, AuditLevel.SECURITY]
        ]

        # Compliance summary
        compliance_summary: Dict[str, Dict[str, Any]] = {}
        for standard, trail in self._compliance_trails.items():
            standard_name = standard.value if hasattr(standard, "value") else str(standard)
            compliance_summary[standard_name] = {
                "event_count": trail.event_count,
                "start_time": trail.start_time.isoformat() if trail.start_time else None,
                "end_time": trail.end_time.isoformat() if trail.end_time else None,
                "status": "closed" if trail.end_time else "active",
            }

        return AuditReport(
            period_start=start_time,
            period_end=end_time,
            total_events=len(events),
            events_by_level=dict(events_by_level),
            events_by_category=dict(events_by_category),
            critical_events=critical_events[:100],
            compliance_summary=compliance_summary,
            agent_summary=dict(agent_summary),
        )

    # =========================================================================
    # INTEGRITY
    # =========================================================================

    def verify_hash_chain(self) -> Tuple[bool, List[str]]:
        """
        Verify the integrity of the hash chain.

        Returns:
            Tuple of (is_valid, list of issues)
        """
        if not self.enable_hash_chain:
            return True, []

        issues = []
        previous_hash = None

        with self._lock:
            for i, event in enumerate(self._events):
                # Verify previous hash link
                if event.previous_hash != previous_hash:
                    issues.append(
                        f"Event {i} ({event.event_id}): "
                        f"previous_hash mismatch"
                    )

                # Verify event hash
                calculated_hash = self._calculate_event_hash(event)
                if event.event_hash != calculated_hash:
                    issues.append(
                        f"Event {i} ({event.event_id}): "
                        f"event_hash mismatch (tampering detected)"
                    )

                previous_hash = event.event_hash

        return len(issues) == 0, issues

    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================

    def _calculate_event_hash(self, event: AuditEvent) -> str:
        """Calculate hash for an event."""
        hash_data = {
            "event_id": event.event_id,
            "timestamp": event.timestamp.isoformat(),
            "level": event.level.value if hasattr(event.level, "value") else event.level,
            "category": event.category.value if hasattr(event.category, "value") else event.category,
            "event_type": event.event_type,
            "agent_id": event.agent_id,
            "message": event.message,
            "data": event.data,
            "previous_hash": event.previous_hash,
        }
        hash_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def _calculate_trail_hash(self, trail: ComplianceAuditTrail) -> str:
        """Calculate hash for a compliance trail."""
        hash_data = {
            "trail_id": trail.trail_id,
            "standard": trail.standard.value if hasattr(trail.standard, "value") else trail.standard,
            "event_count": trail.event_count,
            "events": trail.events,
        }
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def _audit_to_log_level(self, level: AuditLevel) -> int:
        """Convert audit level to logging level."""
        level_map = {
            AuditLevel.DEBUG: logging.DEBUG,
            AuditLevel.INFO: logging.INFO,
            AuditLevel.WARNING: logging.WARNING,
            AuditLevel.ERROR: logging.ERROR,
            AuditLevel.CRITICAL: logging.CRITICAL,
            AuditLevel.SECURITY: logging.CRITICAL,
        }
        return level_map.get(level, logging.INFO)

    def _summarize_data(self, data: Dict[str, Any], max_length: int = 100) -> Dict[str, Any]:
        """Summarize data for audit log storage."""
        summary = {}
        for key, value in data.items():
            if isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, str):
                summary[key] = value[:max_length] + "..." if len(value) > max_length else value
            elif isinstance(value, (list, dict)):
                summary[key] = f"<{type(value).__name__}:{len(value)} items>"
            else:
                summary[key] = f"<{type(value).__name__}>"
        return summary

    # =========================================================================
    # STATISTICS
    # =========================================================================

    @property
    def event_count(self) -> int:
        """Get total event count."""
        return len(self._events)

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit statistics."""
        with self._lock:
            events = self._events

        if not events:
            return {
                "total_events": 0,
                "agent_id": self.agent_id,
            }

        return {
            "total_events": len(events),
            "agent_id": self.agent_id,
            "oldest_event": min(e.timestamp for e in events).isoformat(),
            "newest_event": max(e.timestamp for e in events).isoformat(),
            "hash_chain_enabled": self.enable_hash_chain,
            "compliance_trails": list(self._compliance_trails.keys()),
            "handler_count": len(self._event_handlers),
        }

    def export_events(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> str:
        """
        Export audit events.

        Args:
            format: Export format (json, csv)
            start_time: Filter start time
            end_time: Filter end time

        Returns:
            Exported events as string
        """
        events = self.get_events(start_time=start_time, end_time=end_time)

        if format == "json":
            events_data = [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "level": e.level.name if hasattr(e.level, "name") else str(e.level),
                    "category": e.category.name if hasattr(e.category, "name") else str(e.category),
                    "event_type": e.event_type,
                    "agent_id": e.agent_id,
                    "message": e.message,
                    "data": e.data,
                    "event_hash": e.event_hash,
                }
                for e in events
            ]
            return json.dumps(events_data, indent=2, default=str)

        else:  # CSV
            lines = ["event_id,timestamp,level,category,event_type,agent_id,message"]
            for e in events:
                level_str = e.level.name if hasattr(e.level, "name") else str(e.level)
                cat_str = e.category.name if hasattr(e.category, "name") else str(e.category)
                line = f"{e.event_id},{e.timestamp.isoformat()},{level_str},{cat_str},{e.event_type},{e.agent_id},{e.message}"
                lines.append(line)
            return "\n".join(lines)

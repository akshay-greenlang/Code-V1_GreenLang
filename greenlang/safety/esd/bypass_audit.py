"""
BypassAudit - Comprehensive Bypass Event Logging

This module implements comprehensive bypass audit logging for Emergency
Shutdown Systems per IEC 61511-1 Clause 11.7. Provides complete event
logging with justification tracking, duration tracking, authorizer
identification, and compliance reporting.

Key features:
- Comprehensive bypass event logging
- Bypass justification tracking
- Duration tracking
- Authorizer identification
- Compliance reporting
- Provenance hashing for all records

Reference: IEC 61511-1 Clause 11.7, ISA TR84.00.09

Example:
    >>> from greenlang.safety.esd.bypass_audit import BypassAuditLogger
    >>> logger = BypassAuditLogger(system_id="ESD-001")
    >>> logger.log_bypass_event(event_data)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import json

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events for bypass logging."""

    REQUEST_CREATED = "request_created"
    REQUEST_SUBMITTED = "request_submitted"
    REQUEST_APPROVED = "request_approved"
    REQUEST_REJECTED = "request_rejected"
    BYPASS_ACTIVATED = "bypass_activated"
    BYPASS_DEACTIVATED = "bypass_deactivated"
    BYPASS_EXTENDED = "bypass_extended"
    BYPASS_EXPIRED = "bypass_expired"
    BYPASS_MODIFIED = "bypass_modified"
    ALARM_GENERATED = "alarm_generated"
    ALARM_ACKNOWLEDGED = "alarm_acknowledged"
    COMPENSATING_MEASURE_ADDED = "compensating_measure_added"
    INCIDENT_DURING_BYPASS = "incident_during_bypass"
    PERIODIC_STATUS_CHECK = "periodic_status_check"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ALERT = "alert"


class BypassAuditEvent(BaseModel):
    """Individual bypass audit event."""

    event_id: str = Field(
        default_factory=lambda: f"BAE-{uuid.uuid4().hex[:12].upper()}",
        description="Event identifier"
    )
    event_type: AuditEventType = Field(
        ...,
        description="Type of event"
    )
    severity: AuditSeverity = Field(
        default=AuditSeverity.INFO,
        description="Event severity"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    bypass_id: str = Field(
        ...,
        description="Bypass/workflow identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF identifier"
    )
    sif_name: str = Field(
        default="",
        description="SIF name"
    )
    sil_level: int = Field(
        default=0,
        description="SIL level"
    )
    actor: str = Field(
        ...,
        description="Person performing action"
    )
    actor_role: str = Field(
        default="",
        description="Actor's role"
    )
    action_description: str = Field(
        ...,
        description="Description of action"
    )
    justification: str = Field(
        default="",
        description="Justification for action"
    )
    duration_hours: Optional[float] = Field(
        None,
        description="Duration (hours)"
    )
    remaining_hours: Optional[float] = Field(
        None,
        description="Remaining hours at event time"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures in place"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Work permit reference"
    )
    risk_assessment_ref: Optional[str] = Field(
        None,
        description="Risk assessment reference"
    )
    additional_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional event data"
    )
    previous_event_id: Optional[str] = Field(
        None,
        description="Previous event in chain"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BypassDurationRecord(BaseModel):
    """Record of bypass duration for tracking."""

    record_id: str = Field(
        default_factory=lambda: f"BDR-{uuid.uuid4().hex[:8].upper()}",
        description="Record identifier"
    )
    bypass_id: str = Field(
        ...,
        description="Bypass identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF identifier"
    )
    sil_level: int = Field(
        default=0,
        description="SIL level"
    )
    activated_at: datetime = Field(
        ...,
        description="Activation timestamp"
    )
    deactivated_at: Optional[datetime] = Field(
        None,
        description="Deactivation timestamp"
    )
    approved_duration_hours: float = Field(
        ...,
        description="Approved duration"
    )
    actual_duration_hours: Optional[float] = Field(
        None,
        description="Actual duration"
    )
    extensions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Extension records"
    )
    exceeded_approved: bool = Field(
        default=False,
        description="Did actual exceed approved"
    )
    authorized_by: str = Field(
        ...,
        description="Final authorizer"
    )
    justification: str = Field(
        ...,
        description="Bypass justification"
    )


class ComplianceViolation(BaseModel):
    """Compliance violation record."""

    violation_id: str = Field(
        default_factory=lambda: f"CV-{uuid.uuid4().hex[:8].upper()}",
        description="Violation identifier"
    )
    bypass_id: str = Field(
        ...,
        description="Bypass identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF identifier"
    )
    violation_type: str = Field(
        ...,
        description="Type of violation"
    )
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp"
    )
    description: str = Field(
        ...,
        description="Violation description"
    )
    severity: AuditSeverity = Field(
        default=AuditSeverity.WARNING,
        description="Violation severity"
    )
    resolved: bool = Field(
        default=False,
        description="Is violation resolved"
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="Resolution timestamp"
    )
    resolved_by: Optional[str] = Field(
        None,
        description="Person who resolved"
    )
    corrective_action: Optional[str] = Field(
        None,
        description="Corrective action taken"
    )


class BypassAuditLogger:
    """
    Bypass Audit Logger for ESD/SIS Systems.

    Provides comprehensive audit logging for all bypass activities
    per IEC 61511-1 Clause 11.7 requirements.

    Key features:
    - Complete event logging
    - Justification tracking
    - Duration monitoring
    - Authorizer identification
    - Compliance reporting
    - Provenance chain for all records

    The logger follows IEC 61511 principles:
    - All events are logged
    - Events are immutable
    - Complete chain of custody
    - Tamper-evident records

    Attributes:
        system_id: ESD system identifier
        events: All audit events
        duration_records: Duration tracking records

    Example:
        >>> logger = BypassAuditLogger(system_id="ESD-001")
        >>> event = logger.log_bypass_activated(bypass_id, sif_id, actor, ...)
    """

    def __init__(
        self,
        system_id: str,
        retention_days: int = 365 * 5  # 5 years per IEC 61511
    ):
        """
        Initialize BypassAuditLogger.

        Args:
            system_id: ESD system identifier
            retention_days: Record retention period (days)
        """
        self.system_id = system_id
        self.retention_days = retention_days

        self.events: Dict[str, BypassAuditEvent] = {}
        self.duration_records: Dict[str, BypassDurationRecord] = {}
        self.violations: Dict[str, ComplianceViolation] = {}

        # Event chain tracking
        self._last_event_by_bypass: Dict[str, str] = {}

        # Genesis block for provenance chain
        self._genesis_hash = hashlib.sha256(
            f"{system_id}|{datetime.utcnow().isoformat()}|GENESIS".encode()
        ).hexdigest()

        logger.info(f"BypassAuditLogger initialized: {system_id}")

    def log_event(
        self,
        event_type: AuditEventType,
        bypass_id: str,
        sif_id: str,
        actor: str,
        action_description: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        sif_name: str = "",
        sil_level: int = 0,
        actor_role: str = "",
        justification: str = "",
        duration_hours: Optional[float] = None,
        remaining_hours: Optional[float] = None,
        compensating_measures: Optional[List[str]] = None,
        work_permit_ref: Optional[str] = None,
        risk_assessment_ref: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> BypassAuditEvent:
        """
        Log a bypass audit event.

        Args:
            event_type: Type of event
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            actor: Person performing action
            action_description: Description of action
            severity: Event severity
            sif_name: SIF name
            sil_level: SIL level
            actor_role: Actor's role
            justification: Justification for action
            duration_hours: Duration in hours
            remaining_hours: Remaining hours
            compensating_measures: Compensating measures
            work_permit_ref: Work permit reference
            risk_assessment_ref: Risk assessment reference
            additional_data: Additional data

        Returns:
            BypassAuditEvent
        """
        # Get previous event in chain
        previous_event_id = self._last_event_by_bypass.get(bypass_id)

        event = BypassAuditEvent(
            event_type=event_type,
            severity=severity,
            bypass_id=bypass_id,
            sif_id=sif_id,
            sif_name=sif_name,
            sil_level=sil_level,
            actor=actor,
            actor_role=actor_role,
            action_description=action_description,
            justification=justification,
            duration_hours=duration_hours,
            remaining_hours=remaining_hours,
            compensating_measures=compensating_measures or [],
            work_permit_ref=work_permit_ref,
            risk_assessment_ref=risk_assessment_ref,
            additional_data=additional_data or {},
            previous_event_id=previous_event_id,
        )

        # Calculate provenance hash (chain)
        event.provenance_hash = self._calculate_provenance(
            event,
            previous_event_id
        )

        # Store event
        self.events[event.event_id] = event
        self._last_event_by_bypass[bypass_id] = event.event_id

        # Log at appropriate level
        if severity == AuditSeverity.CRITICAL:
            logger.critical(
                f"AUDIT [{event_type.value}]: {action_description}"
            )
        elif severity == AuditSeverity.WARNING:
            logger.warning(
                f"AUDIT [{event_type.value}]: {action_description}"
            )
        else:
            logger.info(
                f"AUDIT [{event_type.value}]: {action_description}"
            )

        return event

    def log_bypass_activated(
        self,
        bypass_id: str,
        sif_id: str,
        actor: str,
        authorized_by: str,
        duration_hours: float,
        justification: str,
        sif_name: str = "",
        sil_level: int = 0,
        compensating_measures: Optional[List[str]] = None,
        work_permit_ref: Optional[str] = None
    ) -> BypassAuditEvent:
        """
        Log bypass activation event.

        Args:
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            actor: Person activating
            authorized_by: Person who authorized
            duration_hours: Approved duration
            justification: Bypass justification
            sif_name: SIF name
            sil_level: SIL level
            compensating_measures: Compensating measures
            work_permit_ref: Work permit reference

        Returns:
            BypassAuditEvent
        """
        # Create duration record
        duration_record = BypassDurationRecord(
            bypass_id=bypass_id,
            sif_id=sif_id,
            sil_level=sil_level,
            activated_at=datetime.utcnow(),
            approved_duration_hours=duration_hours,
            authorized_by=authorized_by,
            justification=justification,
        )
        self.duration_records[bypass_id] = duration_record

        return self.log_event(
            event_type=AuditEventType.BYPASS_ACTIVATED,
            bypass_id=bypass_id,
            sif_id=sif_id,
            actor=actor,
            action_description=(
                f"Bypass activated for {sif_id} ({sif_name}) "
                f"for {duration_hours}h, authorized by {authorized_by}"
            ),
            severity=AuditSeverity.WARNING,
            sif_name=sif_name,
            sil_level=sil_level,
            justification=justification,
            duration_hours=duration_hours,
            remaining_hours=duration_hours,
            compensating_measures=compensating_measures,
            work_permit_ref=work_permit_ref,
            additional_data={
                "authorized_by": authorized_by,
            }
        )

    def log_bypass_deactivated(
        self,
        bypass_id: str,
        sif_id: str,
        actor: str,
        reason: str,
        sif_name: str = "",
        sil_level: int = 0
    ) -> BypassAuditEvent:
        """
        Log bypass deactivation event.

        Args:
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            actor: Person deactivating
            reason: Deactivation reason
            sif_name: SIF name
            sil_level: SIL level

        Returns:
            BypassAuditEvent
        """
        # Update duration record
        if bypass_id in self.duration_records:
            record = self.duration_records[bypass_id]
            record.deactivated_at = datetime.utcnow()
            record.actual_duration_hours = (
                record.deactivated_at - record.activated_at
            ).total_seconds() / 3600
            record.exceeded_approved = (
                record.actual_duration_hours > record.approved_duration_hours
            )

            if record.exceeded_approved:
                self._log_violation(
                    bypass_id=bypass_id,
                    sif_id=sif_id,
                    violation_type="duration_exceeded",
                    description=(
                        f"Bypass duration {record.actual_duration_hours:.1f}h "
                        f"exceeded approved {record.approved_duration_hours}h"
                    ),
                    severity=AuditSeverity.WARNING
                )

        return self.log_event(
            event_type=AuditEventType.BYPASS_DEACTIVATED,
            bypass_id=bypass_id,
            sif_id=sif_id,
            actor=actor,
            action_description=(
                f"Bypass deactivated for {sif_id} ({sif_name}): {reason}"
            ),
            severity=AuditSeverity.INFO,
            sif_name=sif_name,
            sil_level=sil_level,
            justification=reason,
            additional_data={
                "actual_duration_hours": (
                    self.duration_records[bypass_id].actual_duration_hours
                    if bypass_id in self.duration_records else None
                ),
            }
        )

    def log_bypass_extended(
        self,
        bypass_id: str,
        sif_id: str,
        actor: str,
        extension_hours: float,
        new_total_hours: float,
        justification: str,
        sif_name: str = "",
        sil_level: int = 0,
        extension_number: int = 1
    ) -> BypassAuditEvent:
        """
        Log bypass extension event.

        Args:
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            actor: Person extending
            extension_hours: Additional hours
            new_total_hours: New total duration
            justification: Extension justification
            sif_name: SIF name
            sil_level: SIL level
            extension_number: Extension count

        Returns:
            BypassAuditEvent
        """
        # Update duration record
        if bypass_id in self.duration_records:
            record = self.duration_records[bypass_id]
            record.approved_duration_hours = new_total_hours
            record.extensions.append({
                "extension_number": extension_number,
                "extension_hours": extension_hours,
                "new_total_hours": new_total_hours,
                "extended_by": actor,
                "extended_at": datetime.utcnow().isoformat(),
                "justification": justification,
            })

        return self.log_event(
            event_type=AuditEventType.BYPASS_EXTENDED,
            bypass_id=bypass_id,
            sif_id=sif_id,
            actor=actor,
            action_description=(
                f"Bypass extended for {sif_id} by {extension_hours}h "
                f"(total: {new_total_hours}h)"
            ),
            severity=AuditSeverity.WARNING,
            sif_name=sif_name,
            sil_level=sil_level,
            justification=justification,
            duration_hours=new_total_hours,
            additional_data={
                "extension_hours": extension_hours,
                "extension_number": extension_number,
            }
        )

    def log_bypass_expired(
        self,
        bypass_id: str,
        sif_id: str,
        sif_name: str = "",
        sil_level: int = 0
    ) -> BypassAuditEvent:
        """
        Log bypass expiration event.

        Args:
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            sif_name: SIF name
            sil_level: SIL level

        Returns:
            BypassAuditEvent
        """
        # Update duration record
        if bypass_id in self.duration_records:
            record = self.duration_records[bypass_id]
            record.deactivated_at = datetime.utcnow()
            record.actual_duration_hours = (
                record.deactivated_at - record.activated_at
            ).total_seconds() / 3600

        return self.log_event(
            event_type=AuditEventType.BYPASS_EXPIRED,
            bypass_id=bypass_id,
            sif_id=sif_id,
            actor="SYSTEM",
            action_description=(
                f"Bypass expired for {sif_id} ({sif_name}) - "
                f"SIF protection automatically restored"
            ),
            severity=AuditSeverity.WARNING,
            sif_name=sif_name,
            sil_level=sil_level,
        )

    def log_incident_during_bypass(
        self,
        bypass_id: str,
        sif_id: str,
        incident_description: str,
        reported_by: str,
        sif_name: str = "",
        sil_level: int = 0
    ) -> BypassAuditEvent:
        """
        Log an incident that occurred during bypass.

        Args:
            bypass_id: Bypass identifier
            sif_id: SIF identifier
            incident_description: Incident description
            reported_by: Person reporting
            sif_name: SIF name
            sil_level: SIL level

        Returns:
            BypassAuditEvent
        """
        self._log_violation(
            bypass_id=bypass_id,
            sif_id=sif_id,
            violation_type="incident_during_bypass",
            description=incident_description,
            severity=AuditSeverity.CRITICAL
        )

        return self.log_event(
            event_type=AuditEventType.INCIDENT_DURING_BYPASS,
            bypass_id=bypass_id,
            sif_id=sif_id,
            actor=reported_by,
            action_description=(
                f"INCIDENT during bypass of {sif_id}: {incident_description}"
            ),
            severity=AuditSeverity.CRITICAL,
            sif_name=sif_name,
            sil_level=sil_level,
            additional_data={
                "incident_description": incident_description,
            }
        )

    def get_bypass_audit_trail(
        self,
        bypass_id: str
    ) -> List[BypassAuditEvent]:
        """
        Get complete audit trail for a bypass.

        Args:
            bypass_id: Bypass identifier

        Returns:
            List of audit events in chronological order
        """
        events = [
            e for e in self.events.values()
            if e.bypass_id == bypass_id
        ]
        return sorted(events, key=lambda e: e.timestamp)

    def get_events_by_sif(
        self,
        sif_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[BypassAuditEvent]:
        """
        Get all events for a SIF.

        Args:
            sif_id: SIF identifier
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of matching events
        """
        events = [
            e for e in self.events.values()
            if e.sif_id == sif_id
        ]

        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        return sorted(events, key=lambda e: e.timestamp)

    def get_events_by_actor(
        self,
        actor: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[BypassAuditEvent]:
        """
        Get all events by a specific actor.

        Args:
            actor: Actor name
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of matching events
        """
        events = [
            e for e in self.events.values()
            if e.actor == actor
        ]

        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]

        return sorted(events, key=lambda e: e.timestamp)

    def get_duration_statistics(
        self,
        sif_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get bypass duration statistics.

        Args:
            sif_id: Optional SIF filter
            start_date: Start date filter
            end_date: End date filter

        Returns:
            Statistics dictionary
        """
        records = list(self.duration_records.values())

        if sif_id:
            records = [r for r in records if r.sif_id == sif_id]
        if start_date:
            records = [r for r in records if r.activated_at >= start_date]
        if end_date:
            records = [r for r in records if r.activated_at <= end_date]

        if not records:
            return {"error": "No records found"}

        completed = [r for r in records if r.actual_duration_hours is not None]

        if completed:
            durations = [r.actual_duration_hours for r in completed]
            import statistics as stats

            avg_duration = stats.mean(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration = max_duration = min_duration = None

        exceeded_count = sum(1 for r in completed if r.exceeded_approved)
        extension_count = sum(len(r.extensions) for r in records)

        # Calculate by SIL
        by_sil = {}
        for sil in range(0, 5):
            sil_records = [r for r in records if r.sil_level == sil]
            if sil_records:
                by_sil[f"SIL_{sil}"] = {
                    "count": len(sil_records),
                    "exceeded": sum(1 for r in sil_records if r.exceeded_approved),
                }

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "total_bypasses": len(records),
            "completed_bypasses": len(completed),
            "active_bypasses": len(records) - len(completed),
            "avg_duration_hours": round(avg_duration, 2) if avg_duration else None,
            "max_duration_hours": round(max_duration, 2) if max_duration else None,
            "min_duration_hours": round(min_duration, 2) if min_duration else None,
            "exceeded_approved_count": exceeded_count,
            "total_extensions": extension_count,
            "by_sil_level": by_sil,
        }

    def get_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Generate compliance report for bypass auditing.

        Args:
            start_date: Report start date
            end_date: Report end date

        Returns:
            Compliance report dictionary
        """
        # Filter events in period
        events = [
            e for e in self.events.values()
            if start_date <= e.timestamp <= end_date
        ]

        # Filter violations in period
        violations = [
            v for v in self.violations.values()
            if start_date <= v.detected_at <= end_date
        ]

        # Count by event type
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Get unique bypasses
        unique_bypasses = set(e.bypass_id for e in events)

        # Duration analysis
        duration_stats = self.get_duration_statistics(
            start_date=start_date,
            end_date=end_date
        )

        # Unresolved violations
        unresolved = [v for v in violations if not v.resolved]

        # Calculate compliance score
        total_bypasses = len(unique_bypasses)
        violation_count = len(violations)

        if total_bypasses > 0:
            compliance_score = max(0, 100 - (violation_count / total_bypasses * 20))
        else:
            compliance_score = 100

        report = {
            "report_id": f"BCR-{uuid.uuid4().hex[:8].upper()}",
            "report_timestamp": datetime.utcnow().isoformat(),
            "reporting_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "summary": {
                "total_events": len(events),
                "unique_bypasses": total_bypasses,
                "total_violations": violation_count,
                "unresolved_violations": len(unresolved),
                "compliance_score": round(compliance_score, 1),
            },
            "event_breakdown": event_counts,
            "duration_statistics": duration_stats,
            "violations": [
                {
                    "violation_id": v.violation_id,
                    "type": v.violation_type,
                    "sif_id": v.sif_id,
                    "severity": v.severity.value,
                    "resolved": v.resolved,
                    "description": v.description,
                }
                for v in violations
            ],
            "recommendations": [],
            "provenance_hash": hashlib.sha256(
                f"{len(events)}|{total_bypasses}|{compliance_score}|{end_date.isoformat()}".encode()
            ).hexdigest()
        }

        # Add recommendations
        if duration_stats.get("exceeded_approved_count", 0) > 0:
            report["recommendations"].append(
                "Review bypass duration approval process - multiple instances of exceeded duration"
            )

        if len(unresolved) > 0:
            report["recommendations"].append(
                f"Address {len(unresolved)} unresolved compliance violations"
            )

        if event_counts.get("bypass_extended", 0) > total_bypasses * 0.5:
            report["recommendations"].append(
                "High extension rate detected - consider reviewing initial duration estimates"
            )

        return report

    def verify_audit_chain(
        self,
        bypass_id: str
    ) -> Dict[str, Any]:
        """
        Verify the integrity of the audit chain for a bypass.

        Args:
            bypass_id: Bypass identifier

        Returns:
            Verification result
        """
        events = self.get_bypass_audit_trail(bypass_id)

        if not events:
            return {
                "bypass_id": bypass_id,
                "verified": False,
                "error": "No events found"
            }

        verified = True
        broken_links = []

        for i, event in enumerate(events):
            # Verify provenance hash
            expected_hash = self._calculate_provenance(
                event,
                event.previous_event_id
            )

            if event.provenance_hash != expected_hash:
                verified = False
                broken_links.append({
                    "event_id": event.event_id,
                    "issue": "Hash mismatch - possible tampering"
                })

            # Verify chain linkage
            if i > 0:
                if event.previous_event_id != events[i-1].event_id:
                    verified = False
                    broken_links.append({
                        "event_id": event.event_id,
                        "issue": "Chain link broken"
                    })

        return {
            "bypass_id": bypass_id,
            "verified": verified,
            "event_count": len(events),
            "chain_start": events[0].timestamp.isoformat(),
            "chain_end": events[-1].timestamp.isoformat(),
            "broken_links": broken_links,
        }

    def export_audit_log(
        self,
        bypass_id: Optional[str] = None,
        format: str = "json"
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Export audit log for archival.

        Args:
            bypass_id: Optional bypass filter
            format: Export format (json, dict)

        Returns:
            Exported data
        """
        if bypass_id:
            events = self.get_bypass_audit_trail(bypass_id)
        else:
            events = sorted(
                self.events.values(),
                key=lambda e: e.timestamp
            )

        export_data = [
            {
                "event_id": e.event_id,
                "event_type": e.event_type.value,
                "severity": e.severity.value,
                "timestamp": e.timestamp.isoformat(),
                "bypass_id": e.bypass_id,
                "sif_id": e.sif_id,
                "sif_name": e.sif_name,
                "sil_level": e.sil_level,
                "actor": e.actor,
                "actor_role": e.actor_role,
                "action_description": e.action_description,
                "justification": e.justification,
                "duration_hours": e.duration_hours,
                "remaining_hours": e.remaining_hours,
                "compensating_measures": e.compensating_measures,
                "work_permit_ref": e.work_permit_ref,
                "risk_assessment_ref": e.risk_assessment_ref,
                "additional_data": e.additional_data,
                "previous_event_id": e.previous_event_id,
                "provenance_hash": e.provenance_hash,
            }
            for e in events
        ]

        if format == "json":
            return json.dumps(export_data, indent=2)

        return export_data

    def _log_violation(
        self,
        bypass_id: str,
        sif_id: str,
        violation_type: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.WARNING
    ) -> ComplianceViolation:
        """Log a compliance violation."""
        violation = ComplianceViolation(
            bypass_id=bypass_id,
            sif_id=sif_id,
            violation_type=violation_type,
            description=description,
            severity=severity,
        )

        self.violations[violation.violation_id] = violation

        logger.warning(
            f"Compliance violation: {violation_type} - {description}"
        )

        return violation

    def _calculate_provenance(
        self,
        event: BypassAuditEvent,
        previous_event_id: Optional[str]
    ) -> str:
        """Calculate SHA-256 provenance hash with chain link."""
        # Get previous hash for chain
        if previous_event_id and previous_event_id in self.events:
            prev_hash = self.events[previous_event_id].provenance_hash
        else:
            prev_hash = self._genesis_hash

        provenance_str = (
            f"{prev_hash}|"
            f"{event.event_id}|"
            f"{event.event_type.value}|"
            f"{event.bypass_id}|"
            f"{event.sif_id}|"
            f"{event.actor}|"
            f"{event.timestamp.isoformat()}"
        )

        return hashlib.sha256(provenance_str.encode()).hexdigest()

"""
SafetyAuditTrail - Complete audit trail for regulatory compliance.

This module provides comprehensive logging and reporting of all safety-related
events, constraint violations, and envelope changes. Full audit trail is
maintained for regulatory compliance and incident investigation.

CRITICAL: All safety events are logged immutably with SHA-256 provenance hashes.

Example:
    >>> audit = SafetyAuditTrail(unit_id="BLR-001")
    >>> record = audit.log_safety_check(check)
    >>> report = audit.generate_safety_report(period)
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
import json
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Type of safety event."""
    SAFETY_CHECK = "safety_check"
    CONSTRAINT_VIOLATION = "constraint_violation"
    ENVELOPE_CHANGE = "envelope_change"
    INTERLOCK_EVENT = "interlock_event"
    TRIP_EVENT = "trip_event"
    HAZARD_DETECTION = "hazard_detection"
    EMERGENCY_EVENT = "emergency_event"
    OPERATOR_ACTION = "operator_action"
    SYSTEM_ACTION = "system_action"


class EventSeverity(str, Enum):
    """Severity of safety event."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DateRange(BaseModel):
    """Date range for report generation."""
    start_date: datetime = Field(..., description="Start of date range")
    end_date: datetime = Field(..., description="End of date range")

    def contains(self, dt: datetime) -> bool:
        """Check if datetime is within range."""
        return self.start_date <= dt <= self.end_date


class SafetyCheck(BaseModel):
    """Safety check performed by the system."""
    check_id: str = Field(..., description="Unique check identifier")
    unit_id: str = Field(..., description="Unit identifier")
    check_type: str = Field(..., description="Type of check performed")
    passed: bool = Field(..., description="Whether check passed")
    checked_values: Dict[str, float] = Field(default_factory=dict)
    limits: Dict[str, float] = Field(default_factory=dict)
    result_details: str = Field(default="", description="Detailed result")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConstraintViolation(BaseModel):
    """Constraint violation event."""
    violation_id: str = Field(..., description="Unique violation identifier")
    unit_id: str = Field(..., description="Unit identifier")
    constraint_name: str = Field(..., description="Name of violated constraint")
    actual_value: float = Field(..., description="Actual value at violation")
    limit_value: float = Field(..., description="Limit that was violated")
    severity: EventSeverity = Field(..., description="Violation severity")
    duration_seconds: Optional[float] = Field(None, description="Duration of violation")
    action_taken: str = Field(..., description="Action taken in response")
    resolved: bool = Field(default=False, description="Whether violation is resolved")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class EnvelopeChange(BaseModel):
    """Safety envelope modification event."""
    change_id: str = Field(..., description="Unique change identifier")
    unit_id: str = Field(..., description="Unit identifier")
    change_type: str = Field(..., description="shrink, expand, or redefine")
    factor: Optional[float] = Field(None, description="Change factor if applicable")
    reason: str = Field(..., description="Reason for change")
    approval: Optional[str] = Field(None, description="Approval string if required")
    previous_limits: Dict[str, float] = Field(default_factory=dict)
    new_limits: Dict[str, float] = Field(default_factory=dict)
    approved_by: Optional[str] = Field(None, description="Approver if expansion")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AuditRecord(BaseModel):
    """Immutable audit record with provenance."""
    record_id: str = Field(..., description="Unique record identifier")
    unit_id: str = Field(..., description="Unit identifier")
    event_type: EventType = Field(..., description="Type of event")
    severity: EventSeverity = Field(..., description="Event severity")
    event_data: Dict[str, Any] = Field(..., description="Event-specific data")
    source: str = Field(default="system", description="Event source")
    operator_id: Optional[str] = Field(None, description="Operator if manual action")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    chain_hash: Optional[str] = Field(None, description="Hash of previous record")


class SafetyMetrics(BaseModel):
    """Safety metrics for a period."""
    period: DateRange = Field(..., description="Reporting period")
    total_checks: int = Field(default=0, description="Total safety checks")
    checks_passed: int = Field(default=0, description="Checks that passed")
    checks_failed: int = Field(default=0, description="Checks that failed")
    pass_rate: float = Field(default=0, description="Pass rate percentage")
    total_violations: int = Field(default=0, description="Total violations")
    critical_violations: int = Field(default=0, description="Critical violations")
    violations_resolved: int = Field(default=0, description="Violations resolved")
    envelope_shrinks: int = Field(default=0, description="Envelope shrink events")
    envelope_expansions: int = Field(default=0, description="Envelope expansions")
    average_violation_duration: Optional[float] = Field(None)
    observe_only_time_percentage: float = Field(default=0)


class SafetyReport(BaseModel):
    """Comprehensive safety report for compliance."""
    report_id: str = Field(..., description="Unique report identifier")
    unit_id: str = Field(..., description="Unit identifier")
    report_period: DateRange = Field(..., description="Reporting period")
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    metrics: SafetyMetrics = Field(..., description="Safety metrics")
    violations_summary: List[Dict[str, Any]] = Field(default_factory=list)
    envelope_changes: List[Dict[str, Any]] = Field(default_factory=list)
    significant_events: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_status: str = Field(..., description="Overall compliance status")
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class SafetyAuditTrail:
    """
    SafetyAuditTrail maintains complete audit trail for regulatory compliance.

    CRITICAL SAFETY INVARIANT:
    - All records are immutable once created
    - SHA-256 provenance hashes ensure data integrity
    - Chain hashes link records for tamper detection
    - Export functionality for regulatory submission

    Attributes:
        unit_id: Identifier for the combustion unit
        records: List of all audit records
        chain_hash: Current chain hash for linking

    Example:
        >>> audit = SafetyAuditTrail(unit_id="BLR-001")
        >>> check = SafetyCheck(check_id="CHK-001", ...)
        >>> record = audit.log_safety_check(check)
        >>> report = audit.generate_safety_report(DateRange(...))
    """

    def __init__(self, unit_id: str, storage_path: Optional[str] = None):
        """
        Initialize SafetyAuditTrail.

        Args:
            unit_id: Unit identifier
            storage_path: Optional path for persistent storage
        """
        self.unit_id = unit_id
        self.records: List[AuditRecord] = []
        self._chain_hash: Optional[str] = None
        self._storage_path = Path(storage_path) if storage_path else None
        self._creation_time = datetime.utcnow()

        # Initialize chain with genesis hash
        self._chain_hash = hashlib.sha256(
            f"GENESIS_{unit_id}_{self._creation_time.isoformat()}".encode()
        ).hexdigest()

        logger.info(f"SafetyAuditTrail initialized for unit {unit_id}")

    def log_safety_check(self, check: SafetyCheck) -> AuditRecord:
        """
        Log a safety check event.

        Args:
            check: SafetyCheck to log

        Returns:
            AuditRecord with provenance hash
        """
        severity = EventSeverity.INFO if check.passed else EventSeverity.WARNING

        record = self._create_record(
            event_type=EventType.SAFETY_CHECK,
            severity=severity,
            event_data={
                "check_id": check.check_id,
                "check_type": check.check_type,
                "passed": check.passed,
                "checked_values": check.checked_values,
                "limits": check.limits,
                "result_details": check.result_details,
                "timestamp": check.timestamp.isoformat()
            }
        )

        logger.info(
            f"Safety check logged: {check.check_type} - "
            f"{'PASSED' if check.passed else 'FAILED'}"
        )

        return record

    def log_constraint_violation(self, violation: ConstraintViolation) -> AuditRecord:
        """
        Log a constraint violation event.

        Args:
            violation: ConstraintViolation to log

        Returns:
            AuditRecord with provenance hash
        """
        record = self._create_record(
            event_type=EventType.CONSTRAINT_VIOLATION,
            severity=violation.severity,
            event_data={
                "violation_id": violation.violation_id,
                "constraint_name": violation.constraint_name,
                "actual_value": violation.actual_value,
                "limit_value": violation.limit_value,
                "severity": violation.severity.value,
                "duration_seconds": violation.duration_seconds,
                "action_taken": violation.action_taken,
                "resolved": violation.resolved,
                "timestamp": violation.timestamp.isoformat()
            }
        )

        logger.warning(
            f"Constraint violation logged: {violation.constraint_name} - "
            f"{violation.actual_value} vs limit {violation.limit_value}"
        )

        return record

    def log_envelope_change(self, change: EnvelopeChange) -> AuditRecord:
        """
        Log a safety envelope change event.

        Args:
            change: EnvelopeChange to log

        Returns:
            AuditRecord with provenance hash
        """
        severity = (
            EventSeverity.INFO if change.change_type == "shrink"
            else EventSeverity.WARNING
        )

        record = self._create_record(
            event_type=EventType.ENVELOPE_CHANGE,
            severity=severity,
            event_data={
                "change_id": change.change_id,
                "change_type": change.change_type,
                "factor": change.factor,
                "reason": change.reason,
                "approval": change.approval,
                "previous_limits": change.previous_limits,
                "new_limits": change.new_limits,
                "approved_by": change.approved_by,
                "timestamp": change.timestamp.isoformat()
            }
        )

        if change.change_type == "expand":
            logger.warning(
                f"Envelope EXPANSION logged: factor={change.factor}, "
                f"approved_by={change.approved_by}"
            )
        else:
            logger.info(
                f"Envelope {change.change_type} logged: factor={change.factor}"
            )

        return record

    def log_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        event_data: Dict[str, Any],
        operator_id: Optional[str] = None
    ) -> AuditRecord:
        """
        Log a generic safety event.

        Args:
            event_type: Type of event
            severity: Event severity
            event_data: Event-specific data
            operator_id: Optional operator identifier

        Returns:
            AuditRecord with provenance hash
        """
        return self._create_record(
            event_type=event_type,
            severity=severity,
            event_data=event_data,
            operator_id=operator_id
        )

    def generate_safety_report(self, period: DateRange) -> SafetyReport:
        """
        Generate comprehensive safety report for a period.

        Args:
            period: DateRange for the report

        Returns:
            SafetyReport with metrics and summaries
        """
        # Filter records for period
        period_records = [
            r for r in self.records
            if period.contains(r.timestamp)
        ]

        # Calculate metrics
        safety_checks = [
            r for r in period_records
            if r.event_type == EventType.SAFETY_CHECK
        ]
        checks_passed = sum(
            1 for r in safety_checks
            if r.event_data.get("passed", False)
        )
        checks_failed = len(safety_checks) - checks_passed
        pass_rate = (checks_passed / len(safety_checks) * 100) if safety_checks else 100

        violations = [
            r for r in period_records
            if r.event_type == EventType.CONSTRAINT_VIOLATION
        ]
        critical_violations = sum(
            1 for r in violations
            if r.severity == EventSeverity.CRITICAL
        )
        violations_resolved = sum(
            1 for r in violations
            if r.event_data.get("resolved", False)
        )

        envelope_changes = [
            r for r in period_records
            if r.event_type == EventType.ENVELOPE_CHANGE
        ]
        shrinks = sum(
            1 for r in envelope_changes
            if r.event_data.get("change_type") == "shrink"
        )
        expansions = sum(
            1 for r in envelope_changes
            if r.event_data.get("change_type") == "expand"
        )

        # Calculate average violation duration
        durations = [
            r.event_data.get("duration_seconds")
            for r in violations
            if r.event_data.get("duration_seconds") is not None
        ]
        avg_duration = sum(durations) / len(durations) if durations else None

        metrics = SafetyMetrics(
            period=period,
            total_checks=len(safety_checks),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            pass_rate=pass_rate,
            total_violations=len(violations),
            critical_violations=critical_violations,
            violations_resolved=violations_resolved,
            envelope_shrinks=shrinks,
            envelope_expansions=expansions,
            average_violation_duration=avg_duration
        )

        # Summarize violations
        violations_summary = [
            {
                "violation_id": r.event_data.get("violation_id"),
                "constraint": r.event_data.get("constraint_name"),
                "severity": r.severity.value,
                "timestamp": r.event_data.get("timestamp"),
                "resolved": r.event_data.get("resolved")
            }
            for r in violations
        ]

        # Summarize envelope changes
        envelope_summary = [
            {
                "change_id": r.event_data.get("change_id"),
                "type": r.event_data.get("change_type"),
                "factor": r.event_data.get("factor"),
                "reason": r.event_data.get("reason"),
                "timestamp": r.event_data.get("timestamp")
            }
            for r in envelope_changes
        ]

        # Identify significant events
        significant = [
            r for r in period_records
            if r.severity in [EventSeverity.ERROR, EventSeverity.CRITICAL]
        ]
        significant_summary = [
            {
                "event_type": r.event_type.value,
                "severity": r.severity.value,
                "timestamp": r.timestamp.isoformat(),
                "summary": self._summarize_event(r)
            }
            for r in significant[:20]  # Limit to 20 most significant
        ]

        # Determine compliance status
        if critical_violations > 0:
            compliance_status = "NON-COMPLIANT"
        elif len(violations) > 10:
            compliance_status = "MARGINAL"
        elif pass_rate >= 99:
            compliance_status = "COMPLIANT"
        else:
            compliance_status = "ACCEPTABLE"

        # Generate recommendations
        recommendations = []
        if critical_violations > 0:
            recommendations.append(
                f"Address {critical_violations} critical violations immediately"
            )
        if expansions > shrinks:
            recommendations.append(
                "Review envelope expansions - more expansions than shrinks"
            )
        if pass_rate < 99:
            recommendations.append(
                f"Improve safety check pass rate (currently {pass_rate:.1f}%)"
            )
        if avg_duration and avg_duration > 300:
            recommendations.append(
                f"Reduce violation duration (avg {avg_duration:.0f}s)"
            )

        report_id = hashlib.sha256(
            f"{self.unit_id}_{period.start_date}_{period.end_date}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        provenance_hash = hashlib.sha256(
            f"{report_id}{metrics.json()}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        report = SafetyReport(
            report_id=report_id,
            unit_id=self.unit_id,
            report_period=period,
            metrics=metrics,
            violations_summary=violations_summary,
            envelope_changes=envelope_summary,
            significant_events=significant_summary,
            compliance_status=compliance_status,
            recommendations=recommendations,
            provenance_hash=provenance_hash
        )

        logger.info(
            f"Safety report generated: {report_id} - {compliance_status}"
        )

        return report

    def export_for_compliance(self, format: str = "json") -> bytes:
        """
        Export audit trail for regulatory compliance.

        Args:
            format: Export format ("json", "csv")

        Returns:
            Exported data as bytes
        """
        if format.lower() == "json":
            export_data = {
                "unit_id": self.unit_id,
                "export_timestamp": datetime.utcnow().isoformat(),
                "genesis_hash": self._chain_hash,
                "record_count": len(self.records),
                "records": [r.dict() for r in self.records]
            }

            # Add export provenance
            export_hash = hashlib.sha256(
                json.dumps(export_data, default=str).encode()
            ).hexdigest()
            export_data["export_hash"] = export_hash

            return json.dumps(export_data, default=str, indent=2).encode()

        elif format.lower() == "csv":
            lines = [
                "record_id,timestamp,event_type,severity,unit_id,provenance_hash"
            ]
            for r in self.records:
                lines.append(
                    f"{r.record_id},{r.timestamp.isoformat()},{r.event_type.value},"
                    f"{r.severity.value},{r.unit_id},{r.provenance_hash}"
                )
            return "\n".join(lines).encode()

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def verify_chain_integrity(self) -> bool:
        """
        Verify audit trail chain integrity.

        Returns:
            True if chain is intact, False if tampered
        """
        if not self.records:
            return True

        expected_chain_hash = hashlib.sha256(
            f"GENESIS_{self.unit_id}_{self._creation_time.isoformat()}".encode()
        ).hexdigest()

        for record in self.records:
            if record.chain_hash != expected_chain_hash:
                logger.error(
                    f"Chain integrity violation at record {record.record_id}"
                )
                return False
            expected_chain_hash = record.provenance_hash

        logger.info("Audit trail chain integrity verified")
        return True

    def _create_record(
        self,
        event_type: EventType,
        severity: EventSeverity,
        event_data: Dict[str, Any],
        operator_id: Optional[str] = None
    ) -> AuditRecord:
        """Create and store an audit record."""
        record_id = hashlib.sha256(
            f"{self.unit_id}_{event_type}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]

        # Create provenance hash
        provenance_str = f"{record_id}{event_type}{severity}{json.dumps(event_data, default=str)}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        record = AuditRecord(
            record_id=record_id,
            unit_id=self.unit_id,
            event_type=event_type,
            severity=severity,
            event_data=event_data,
            operator_id=operator_id,
            provenance_hash=provenance_hash,
            chain_hash=self._chain_hash
        )

        # Update chain hash
        self._chain_hash = provenance_hash

        # Store record
        self.records.append(record)

        # Persist if storage configured
        if self._storage_path:
            self._persist_record(record)

        return record

    def _persist_record(self, record: AuditRecord) -> None:
        """Persist record to storage."""
        try:
            if self._storage_path:
                file_path = self._storage_path / f"{record.record_id}.json"
                with open(file_path, 'w') as f:
                    json.dump(record.dict(), f, default=str, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist record: {e}")

    def _summarize_event(self, record: AuditRecord) -> str:
        """Create brief summary of an event."""
        if record.event_type == EventType.CONSTRAINT_VIOLATION:
            return f"Violation: {record.event_data.get('constraint_name', 'unknown')}"
        elif record.event_type == EventType.TRIP_EVENT:
            return f"Trip: {record.event_data.get('trip_type', 'unknown')}"
        elif record.event_type == EventType.EMERGENCY_EVENT:
            return f"Emergency: {record.event_data.get('event_type', 'unknown')}"
        else:
            return f"{record.event_type.value}: {record.severity.value}"

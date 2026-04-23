"""
Audit Schemas - Data models for audit trails and provenance tracking.

This module defines Pydantic models for audit records, provenance links,
evidence packs, and compliance reports. These schemas support complete
traceability and regulatory compliance for combustion optimization.

Example:
    >>> from audit_schemas import AuditRecord, EvidencePack
    >>> record = AuditRecord(
    ...     event_type="setpoint_change",
    ...     user="operator_1",
    ...     details={"tag": "FC-101.SP", "old_value": 2.0, "new_value": 2.5}
    ... )
"""

from datetime import datetime, date
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator, computed_field
import hashlib
import json


class AuditEventType(str, Enum):
    """Type of audit event."""
    # Control events
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    CONTROL_ACTION = "control_action"
    OPTIMIZATION_RUN = "optimization_run"

    # Safety events
    SAFETY_CHECK = "safety_check"
    INTERLOCK_TRIP = "interlock_trip"
    INTERLOCK_RESET = "interlock_reset"
    INTERLOCK_BYPASS = "interlock_bypass"
    EMERGENCY_STOP = "emergency_stop"

    # System events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    CONFIG_CHANGE = "config_change"
    MODEL_UPDATE = "model_update"

    # User events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"

    # Data events
    DATA_SNAPSHOT = "data_snapshot"
    DATA_EXPORT = "data_export"
    REPORT_GENERATION = "report_generation"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    VIOLATION_DETECTED = "violation_detected"
    CERTIFICATION_UPDATE = "certification_update"


class AuditSeverity(str, Enum):
    """Severity level for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceStatus(str, Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_REVIEW = "pending_review"
    EXEMPT = "exempt"
    UNKNOWN = "unknown"


class CertificationStatus(str, Enum):
    """Certification status."""
    VALID = "valid"
    EXPIRED = "expired"
    PENDING = "pending"
    REVOKED = "revoked"
    NOT_REQUIRED = "not_required"


class AuditRecord(BaseModel):
    """
    Single audit record capturing an auditable event.

    Provides complete traceability for all system actions with
    cryptographic hash for tamper detection.

    Attributes:
        event_type: Type of event being recorded
        timestamp: When the event occurred
        user: User or system that initiated the event
        details: Event-specific details
        hash: SHA-256 hash for integrity verification
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    record_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique record identifier")
    event_type: AuditEventType = Field(..., description="Type of audit event")
    severity: AuditSeverity = Field(default=AuditSeverity.INFO, description="Event severity")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When event occurred")

    # Actor information
    user: Optional[str] = Field(default=None, max_length=100, description="User who initiated event")
    user_role: Optional[str] = Field(default=None, max_length=50, description="User's role")
    system: str = Field(default="BURNMASTER", max_length=100, description="System that generated record")
    session_id: Optional[str] = Field(default=None, max_length=100, description="Session identifier")
    source_ip: Optional[str] = Field(default=None, max_length=50, description="Source IP address")

    # Event details
    details: Dict[str, Any] = Field(default_factory=dict, description="Event-specific details")
    description: str = Field(default="", max_length=1000, description="Human-readable event description")

    # Related entities
    equipment_id: Optional[str] = Field(default=None, max_length=100, description="Related equipment")
    tag: Optional[str] = Field(default=None, max_length=100, description="Related tag/point")
    optimization_id: Optional[str] = Field(default=None, max_length=100, description="Related optimization")
    cycle_id: Optional[str] = Field(default=None, max_length=100, description="Related control cycle")

    # Result
    success: bool = Field(default=True, description="Whether event completed successfully")
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message if failed")

    # Integrity
    hash: Optional[str] = Field(default=None, description="SHA-256 hash for integrity verification")
    previous_hash: Optional[str] = Field(default=None, description="Hash of previous record in chain")

    # Retention
    retention_days: int = Field(default=2555, ge=1, description="Days to retain this record (default 7 years)")

    @computed_field
    @property
    def is_user_action(self) -> bool:
        """Check if this was a user-initiated action."""
        return self.user is not None

    @computed_field
    @property
    def is_safety_event(self) -> bool:
        """Check if this is a safety-related event."""
        safety_events = [
            AuditEventType.SAFETY_CHECK,
            AuditEventType.INTERLOCK_TRIP,
            AuditEventType.INTERLOCK_RESET,
            AuditEventType.INTERLOCK_BYPASS,
            AuditEventType.EMERGENCY_STOP
        ]
        return self.event_type in safety_events

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of record contents."""
        data = {
            "record_id": self.record_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "system": self.system,
            "details": self.details,
            "success": self.success,
            "previous_hash": self.previous_hash
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def seal(self, previous_hash: Optional[str] = None) -> None:
        """Seal the record with hash (includes previous hash for chain)."""
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def verify(self) -> bool:
        """Verify record integrity."""
        if self.hash is None:
            return False
        return self.hash == self.calculate_hash()


class ProvenanceLink(BaseModel):
    """
    Provenance information linking data to its source.

    Captures the complete lineage of data including source snapshots,
    model versions, and code versions for reproducibility.

    Attributes:
        data_snapshot_id: ID of the source data snapshot
        model_version: Version of the ML model used
        code_version: Version of the code that processed data
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    provenance_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique provenance identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When provenance was recorded")

    # Data source
    data_snapshot_id: str = Field(..., min_length=1, max_length=100, description="ID of source data snapshot")
    data_source: str = Field(..., min_length=1, max_length=200, description="Source system for data")
    data_timestamp: datetime = Field(..., description="Timestamp of source data")
    data_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 hash of source data")

    # Model information
    model_version: str = Field(..., min_length=1, max_length=50, description="Version of ML model used")
    model_id: str = Field(..., min_length=1, max_length=100, description="Model identifier")
    model_training_date: Optional[datetime] = Field(default=None, description="When model was trained")
    model_hash: Optional[str] = Field(default=None, description="Hash of model weights")

    # Code information
    code_version: str = Field(..., min_length=1, max_length=50, description="Version of code")
    code_commit: Optional[str] = Field(default=None, max_length=50, description="Git commit hash")
    code_repository: Optional[str] = Field(default=None, max_length=200, description="Code repository URL")

    # Configuration
    config_version: Optional[str] = Field(default=None, max_length=50, description="Configuration version")
    config_hash: Optional[str] = Field(default=None, description="Hash of configuration")

    # Calculation information
    calculation_id: Optional[str] = Field(default=None, max_length=100, description="Related calculation ID")
    formula_version: Optional[str] = Field(default=None, max_length=50, description="Formula/equation version")

    # Parent provenance (for derived data)
    parent_provenance_ids: List[str] = Field(default_factory=list, description="Parent provenance IDs for derived data")

    # Result
    output_hash: Optional[str] = Field(default=None, description="Hash of output data")

    def calculate_chain_hash(self) -> str:
        """Calculate hash of entire provenance chain."""
        data = {
            "provenance_id": self.provenance_id,
            "data_snapshot_id": self.data_snapshot_id,
            "data_hash": self.data_hash,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "code_version": self.code_version,
            "code_commit": self.code_commit,
            "parent_provenance_ids": sorted(self.parent_provenance_ids)
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()


class DataSnapshot(BaseModel):
    """Snapshot of data at a point in time for audit purposes."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    snapshot_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique snapshot identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Snapshot timestamp")

    # Content
    data_type: str = Field(..., max_length=100, description="Type of data in snapshot")
    data: Dict[str, Any] = Field(..., description="Snapshot data")
    data_count: int = Field(default=0, ge=0, description="Number of data points")

    # Integrity
    data_hash: str = Field(..., description="SHA-256 hash of data")

    # Source
    source_system: str = Field(default="", max_length=100, description="Source system")
    source_tags: List[str] = Field(default_factory=list, description="Source tag names")

    # Retention
    retention_days: int = Field(default=365, ge=1, description="Days to retain snapshot")

    def calculate_data_hash(self) -> str:
        """Calculate hash of snapshot data."""
        data_str = json.dumps(self.data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify snapshot integrity."""
        return self.data_hash == self.calculate_data_hash()


class CalculationRecord(BaseModel):
    """Record of a calculation for audit purposes."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    calculation_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique calculation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Calculation timestamp")

    # Calculation details
    calculation_type: str = Field(..., max_length=100, description="Type of calculation")
    formula_id: Optional[str] = Field(default=None, max_length=100, description="Formula identifier")
    formula_version: Optional[str] = Field(default=None, max_length=50, description="Formula version")

    # Inputs and outputs
    inputs: Dict[str, Any] = Field(..., description="Calculation inputs")
    outputs: Dict[str, Any] = Field(..., description="Calculation outputs")
    intermediate_values: Dict[str, Any] = Field(default_factory=dict, description="Intermediate calculation values")

    # Integrity
    input_hash: str = Field(..., description="Hash of inputs")
    output_hash: str = Field(..., description="Hash of outputs")

    # Execution
    execution_time_ms: Optional[float] = Field(default=None, ge=0.0, description="Execution time")
    success: bool = Field(default=True, description="Whether calculation succeeded")
    error_message: Optional[str] = Field(default=None, max_length=500, description="Error message if failed")

    def verify_outputs(self) -> bool:
        """Verify output hash matches current outputs."""
        output_str = json.dumps(self.outputs, sort_keys=True, default=str)
        calculated_hash = hashlib.sha256(output_str.encode()).hexdigest()
        return self.output_hash == calculated_hash


class EvidencePack(BaseModel):
    """
    Complete evidence pack for audit and compliance.

    Bundles all audit records, data snapshots, and calculations
    for a specific period or event with cryptographic seal.

    Attributes:
        records: Audit records included in pack
        snapshots: Data snapshots included
        calculations: Calculation records included
        seal: Cryptographic seal for entire pack
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    pack_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Unique pack identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Pack name")
    description: str = Field(default="", max_length=1000, description="Pack description")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    created_by: str = Field(default="BURNMASTER", max_length=100, description="Creator")

    # Time range
    period_start: datetime = Field(..., description="Start of covered period")
    period_end: datetime = Field(..., description="End of covered period")

    # Evidence contents
    records: List[AuditRecord] = Field(default_factory=list, description="Audit records")
    snapshots: List[DataSnapshot] = Field(default_factory=list, description="Data snapshots")
    calculations: List[CalculationRecord] = Field(default_factory=list, description="Calculation records")
    provenance_links: List[ProvenanceLink] = Field(default_factory=list, description="Provenance links")

    # Summary statistics
    record_count: int = Field(default=0, ge=0, description="Number of audit records")
    snapshot_count: int = Field(default=0, ge=0, description="Number of snapshots")
    calculation_count: int = Field(default=0, ge=0, description="Number of calculations")

    # Integrity
    seal: Optional[str] = Field(default=None, description="Cryptographic seal (hash) of entire pack")
    sealed_at: Optional[datetime] = Field(default=None, description="When pack was sealed")
    sealed_by: Optional[str] = Field(default=None, max_length=100, description="Who sealed the pack")

    # Verification
    verified: bool = Field(default=False, description="Whether pack has been verified")
    verified_at: Optional[datetime] = Field(default=None, description="When pack was verified")
    verified_by: Optional[str] = Field(default=None, max_length=100, description="Who verified the pack")

    @computed_field
    @property
    def is_sealed(self) -> bool:
        """Check if pack is sealed."""
        return self.seal is not None

    def calculate_seal(self) -> str:
        """Calculate cryptographic seal for entire pack."""
        data = {
            "pack_id": self.pack_id,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "record_hashes": [r.hash for r in self.records if r.hash],
            "snapshot_hashes": [s.data_hash for s in self.snapshots],
            "calculation_hashes": [c.output_hash for c in self.calculations],
            "record_count": len(self.records),
            "snapshot_count": len(self.snapshots),
            "calculation_count": len(self.calculations)
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def seal_pack(self, sealed_by: str) -> None:
        """Seal the evidence pack."""
        self.record_count = len(self.records)
        self.snapshot_count = len(self.snapshots)
        self.calculation_count = len(self.calculations)
        self.seal = self.calculate_seal()
        self.sealed_at = datetime.utcnow()
        self.sealed_by = sealed_by

    def verify_pack(self) -> bool:
        """Verify pack integrity."""
        if not self.is_sealed:
            return False

        # Verify individual records
        for record in self.records:
            if not record.verify():
                return False

        # Verify snapshots
        for snapshot in self.snapshots:
            if not snapshot.verify():
                return False

        # Verify overall seal
        return self.seal == self.calculate_seal()


class EmissionsRecord(BaseModel):
    """Record of emissions for compliance reporting."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    record_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Record identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    period_start: datetime = Field(..., description="Reporting period start")
    period_end: datetime = Field(..., description="Reporting period end")

    # Emissions quantities
    co2_kg: float = Field(..., ge=0.0, description="CO2 emissions in kg")
    co2e_kg: Optional[float] = Field(default=None, ge=0.0, description="CO2 equivalent emissions in kg")
    nox_kg: float = Field(..., ge=0.0, description="NOx emissions in kg")
    co_kg: float = Field(..., ge=0.0, description="CO emissions in kg")
    so2_kg: Optional[float] = Field(default=None, ge=0.0, description="SO2 emissions in kg")
    particulate_kg: Optional[float] = Field(default=None, ge=0.0, description="Particulate emissions in kg")

    # Intensity metrics
    co2_intensity_kg_per_mwh: Optional[float] = Field(default=None, ge=0.0, description="CO2 intensity")
    fuel_consumed_mj: float = Field(..., ge=0.0, description="Total fuel consumed in MJ")

    # Operating hours
    operating_hours: float = Field(..., ge=0.0, description="Total operating hours")
    load_factor_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Average load factor")

    # Calculation method
    calculation_method: str = Field(default="measured", max_length=100, description="Calculation method")
    measurement_uncertainty_percent: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="Measurement uncertainty")

    # Data quality
    data_quality_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Data quality score")
    data_gaps_hours: float = Field(default=0.0, ge=0.0, description="Hours of missing data")

    # Provenance
    provenance_id: Optional[str] = Field(default=None, max_length=100, description="Related provenance link")


class Certification(BaseModel):
    """Certification record."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    certification_id: str = Field(..., min_length=1, max_length=100, description="Certification identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Certification name")
    issuer: str = Field(..., min_length=1, max_length=200, description="Issuing authority")
    status: CertificationStatus = Field(default=CertificationStatus.VALID, description="Current status")

    # Dates
    issue_date: date = Field(..., description="Issue date")
    expiry_date: Optional[date] = Field(default=None, description="Expiry date")
    last_audit_date: Optional[date] = Field(default=None, description="Last audit date")

    # Scope
    scope: str = Field(default="", max_length=1000, description="Certification scope")
    applicable_equipment: List[str] = Field(default_factory=list, description="Applicable equipment IDs")

    # Documents
    certificate_number: Optional[str] = Field(default=None, max_length=100, description="Certificate number")
    document_url: Optional[str] = Field(default=None, max_length=500, description="Certificate document URL")


class ComplianceViolation(BaseModel):
    """Record of a compliance violation."""
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    violation_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Violation identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp")

    # Violation details
    regulation: str = Field(..., min_length=1, max_length=200, description="Violated regulation")
    parameter: str = Field(..., min_length=1, max_length=100, description="Parameter that violated")
    limit: float = Field(..., description="Regulatory limit")
    actual_value: float = Field(..., description="Actual value")
    exceedance_percent: float = Field(..., description="Percentage exceedance")
    duration_minutes: float = Field(default=0.0, ge=0.0, description="Duration of violation")

    # Status
    status: str = Field(default="open", max_length=50, description="Violation status")
    resolved_at: Optional[datetime] = Field(default=None, description="Resolution timestamp")
    resolution_notes: Optional[str] = Field(default=None, max_length=1000, description="Resolution notes")

    # Corrective actions
    corrective_actions: List[str] = Field(default_factory=list, description="Corrective actions taken")
    root_cause: Optional[str] = Field(default=None, max_length=500, description="Root cause")


class ComplianceReport(BaseModel):
    """
    Compliance report for regulatory submission.

    Comprehensive report covering emissions, violations, and
    certifications for a reporting period.

    Attributes:
        period: Reporting period identifier
        emissions: Emissions records for the period
        violations: Any violations during period
        certifications: Current certification status
    """
    model_config = ConfigDict(strict=True, frozen=False, validate_assignment=True)

    report_id: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d%H%M%S%f"), description="Report identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Report name")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    created_by: str = Field(default="BURNMASTER", max_length=100, description="Report creator")

    # Reporting period
    period: str = Field(..., min_length=1, max_length=50, description="Reporting period (e.g., '2024-Q1')")
    period_start: date = Field(..., description="Period start date")
    period_end: date = Field(..., description="Period end date")

    # Facility information
    facility_id: str = Field(..., min_length=1, max_length=100, description="Facility identifier")
    facility_name: str = Field(..., min_length=1, max_length=200, description="Facility name")
    equipment_ids: List[str] = Field(default_factory=list, description="Equipment covered")

    # Compliance status
    overall_status: ComplianceStatus = Field(default=ComplianceStatus.COMPLIANT, description="Overall compliance status")

    # Emissions
    emissions: List[EmissionsRecord] = Field(default_factory=list, description="Emissions records")
    total_co2_tonnes: float = Field(default=0.0, ge=0.0, description="Total CO2 in tonnes")
    total_nox_kg: float = Field(default=0.0, ge=0.0, description="Total NOx in kg")
    total_co_kg: float = Field(default=0.0, ge=0.0, description="Total CO in kg")

    # Violations
    violations: List[ComplianceViolation] = Field(default_factory=list, description="Violations during period")
    violation_count: int = Field(default=0, ge=0, description="Number of violations")

    # Certifications
    certifications: List[Certification] = Field(default_factory=list, description="Current certifications")

    # Evidence
    evidence_pack_id: Optional[str] = Field(default=None, max_length=100, description="Related evidence pack ID")

    # Submission
    submitted: bool = Field(default=False, description="Whether report has been submitted")
    submitted_at: Optional[datetime] = Field(default=None, description="Submission timestamp")
    submitted_by: Optional[str] = Field(default=None, max_length=100, description="Who submitted")
    submission_reference: Optional[str] = Field(default=None, max_length=100, description="Submission reference number")

    # Approval
    approved: bool = Field(default=False, description="Whether report has been approved")
    approved_at: Optional[datetime] = Field(default=None, description="Approval timestamp")
    approved_by: Optional[str] = Field(default=None, max_length=100, description="Who approved")

    # Report hash
    report_hash: Optional[str] = Field(default=None, description="SHA-256 hash of report contents")

    @computed_field
    @property
    def is_compliant(self) -> bool:
        """Check if overall status is compliant."""
        return self.overall_status == ComplianceStatus.COMPLIANT

    @computed_field
    @property
    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return self.violation_count > 0 or len(self.violations) > 0

    def calculate_report_hash(self) -> str:
        """Calculate hash of report contents."""
        data = {
            "report_id": self.report_id,
            "period": self.period,
            "facility_id": self.facility_id,
            "total_co2_tonnes": self.total_co2_tonnes,
            "total_nox_kg": self.total_nox_kg,
            "violation_count": self.violation_count,
            "overall_status": self.overall_status.value
        }
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def seal_report(self) -> None:
        """Seal the report with hash."""
        self.violation_count = len(self.violations)
        self.report_hash = self.calculate_report_hash()


__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "ComplianceStatus",
    "CertificationStatus",
    "AuditRecord",
    "ProvenanceLink",
    "DataSnapshot",
    "CalculationRecord",
    "EvidencePack",
    "EmissionsRecord",
    "Certification",
    "ComplianceViolation",
    "ComplianceReport",
]

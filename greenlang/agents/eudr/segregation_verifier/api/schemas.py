# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-010 Segregation Verifier

Pydantic v2 request/response models for all Segregation Verifier REST API
endpoints. Organized by domain: segregation control points, storage zones,
transport vehicles, processing lines, contamination events, facility
assessment, reports, batch jobs, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

Model Count: 60+ schemas covering 37 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-010, Section 7.4
Agent ID: GL-EUDR-SGV-010
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# =============================================================================
# Enumerations
# =============================================================================


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class SCPType(str, Enum):
    """Types of segregation control points."""

    RECEIVING = "receiving"
    STORAGE = "storage"
    PROCESSING = "processing"
    PACKAGING = "packaging"
    DISPATCH = "dispatch"
    TRANSFER = "transfer"
    CLEANING = "cleaning"
    INSPECTION = "inspection"


class SCPStatus(str, Enum):
    """Status of a segregation control point."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_REVIEW = "pending_review"
    DECOMMISSIONED = "decommissioned"


class SegregationLevel(str, Enum):
    """Level of physical segregation at a control point."""

    FULL_PHYSICAL = "full_physical"
    PARTIAL_PHYSICAL = "partial_physical"
    TEMPORAL = "temporal"
    ADMINISTRATIVE = "administrative"
    NONE = "none"


class StorageZoneType(str, Enum):
    """Types of storage zones."""

    WAREHOUSE = "warehouse"
    SILO = "silo"
    TANK = "tank"
    BAY = "bay"
    COLD_ROOM = "cold_room"
    OPEN_YARD = "open_yard"
    CONTAINER_YARD = "container_yard"
    DEDICATED_AREA = "dedicated_area"


class StorageZoneStatus(str, Enum):
    """Status of a storage zone."""

    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"
    MAINTENANCE = "maintenance"
    RESTRICTED = "restricted"
    DECOMMISSIONED = "decommissioned"


class StorageEventType(str, Enum):
    """Types of storage events."""

    INBOUND = "inbound"
    OUTBOUND = "outbound"
    TRANSFER = "transfer"
    INSPECTION = "inspection"
    CLEANING = "cleaning"
    SEAL_CHECK = "seal_check"
    TEMPERATURE_LOG = "temperature_log"


class VehicleType(str, Enum):
    """Types of transport vehicles."""

    TRUCK = "truck"
    CONTAINER = "container"
    RAIL_CAR = "rail_car"
    VESSEL = "vessel"
    BARGE = "barge"
    TANKER = "tanker"
    VAN = "van"
    BULK_CARRIER = "bulk_carrier"


class VehicleStatus(str, Enum):
    """Status of a transport vehicle."""

    AVAILABLE = "available"
    IN_USE = "in_use"
    CLEANING = "cleaning"
    MAINTENANCE = "maintenance"
    QUARANTINE = "quarantine"
    DECOMMISSIONED = "decommissioned"


class CleaningVerificationStatus(str, Enum):
    """Status of cleaning verification."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"
    WAIVED = "waived"


class ProcessingLineType(str, Enum):
    """Types of processing lines."""

    DEDICATED = "dedicated"
    SHARED = "shared"
    MULTI_PURPOSE = "multi_purpose"
    BATCH_LINE = "batch_line"
    CONTINUOUS_LINE = "continuous_line"


class ProcessingLineStatus(str, Enum):
    """Status of a processing line."""

    RUNNING = "running"
    IDLE = "idle"
    CHANGEOVER = "changeover"
    MAINTENANCE = "maintenance"
    CLEANING = "cleaning"
    DECOMMISSIONED = "decommissioned"


class ChangeoverStatus(str, Enum):
    """Status of a line changeover."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"
    FAILED = "failed"


class ContaminationSeverity(str, Enum):
    """Severity level of contamination events."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


class ContaminationType(str, Enum):
    """Types of contamination."""

    CROSS_CONTACT = "cross_contact"
    RESIDUE = "residue"
    CO_MINGLING = "co_mingling"
    LABELLING_ERROR = "labelling_error"
    DOCUMENT_MISMATCH = "document_mismatch"
    PHYSICAL = "physical"
    CHEMICAL = "chemical"


class ContaminationStatus(str, Enum):
    """Status of a contamination event."""

    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class RiskLevel(str, Enum):
    """Risk assessment levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class AssessmentStatus(str, Enum):
    """Status of a facility assessment."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"


class LabelStatus(str, Enum):
    """Status of a segregation label."""

    ACTIVE = "active"
    VERIFIED = "verified"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"


class ReportType(str, Enum):
    """Types of segregation reports."""

    AUDIT = "audit"
    CONTAMINATION = "contamination"
    EVIDENCE_PACKAGE = "evidence_package"
    COMPLIANCE = "compliance"


class ReportFormat(str, Enum):
    """Output formats for reports."""

    PDF = "pdf"
    XLSX = "xlsx"
    JSON = "json"
    CSV = "csv"


class BatchJobStatus(str, Enum):
    """Status of an async batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobType(str, Enum):
    """Types of batch jobs for segregation verifier."""

    SCP_IMPORT = "scp_import"
    STORAGE_AUDIT = "storage_audit"
    TRANSPORT_VERIFICATION = "transport_verification"
    PROCESSING_VERIFICATION = "processing_verification"
    CONTAMINATION_SCAN = "contamination_scan"
    FACILITY_ASSESSMENT = "facility_assessment"
    REPORT_GENERATION = "report_generation"


class UnitOfMeasure(str, Enum):
    """Units for quantity measurements."""

    KG = "kg"
    TONNES = "tonnes"
    LBS = "lbs"
    LITRES = "litres"
    CUBIC_METRES = "cubic_metres"
    BAGS_60KG = "bags_60kg"
    BAGS_69KG = "bags_69kg"


# =============================================================================
# Shared / Common Models
# =============================================================================


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


class GeoCoordinate(BaseModel):
    """WGS84 geographic coordinate."""

    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude")
    longitude: float = Field(
        ..., ge=-180.0, le=180.0, description="Longitude"
    )

    model_config = ConfigDict(from_attributes=True)


class QuantitySpec(BaseModel):
    """Quantity with unit of measure."""

    amount: Decimal = Field(
        ..., gt=0, description="Quantity amount (positive)"
    )
    unit: UnitOfMeasure = Field(..., description="Unit of measure")

    model_config = ConfigDict(from_attributes=True)


class ProvenanceInfo(BaseModel):
    """Provenance tracking information for audit trail."""

    provenance_hash: str = Field(
        ..., description="SHA-256 hash of input data"
    )
    created_by: str = Field(..., description="User ID who created the record")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp (UTC)"
    )
    source: str = Field(
        default="api", description="Data source (api, import, system)"
    )

    model_config = ConfigDict(from_attributes=True)


class ScoreBreakdown(BaseModel):
    """Breakdown of a segregation compliance score."""

    category: str = Field(..., description="Score category")
    score: float = Field(..., ge=0.0, le=100.0, description="Category score (0-100)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Category weight (0-1)")
    findings: List[str] = Field(
        default_factory=list, description="Findings for this category"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# SCP (Segregation Control Point) Schemas
# =============================================================================


class RegisterSCPRequest(BaseModel):
    """Request to register a new Segregation Control Point."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    scp_name: str = Field(
        ..., min_length=1, max_length=200, description="Control point name"
    )
    scp_type: SCPType = Field(
        ..., description="Type of segregation control point"
    )
    commodity: EUDRCommodity = Field(
        ..., description="EUDR commodity this SCP handles"
    )
    segregation_level: SegregationLevel = Field(
        ..., description="Physical segregation level"
    )
    location: Optional[GeoCoordinate] = Field(
        None, description="GPS coordinates of the SCP"
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Detailed SCP description"
    )
    responsible_person: Optional[str] = Field(
        None, max_length=200, description="Person responsible for the SCP"
    )
    operating_procedures: Optional[List[str]] = Field(
        None, description="List of applicable SOPs"
    )
    inspection_frequency_days: int = Field(
        default=30, ge=1, le=365, description="Inspection frequency in days"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional key-value metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-BR-001",
                    "scp_name": "Soya Receiving Bay A",
                    "scp_type": "receiving",
                    "commodity": "soya",
                    "segregation_level": "full_physical",
                    "inspection_frequency_days": 14,
                }
            ]
        },
    )


class UpdateSCPRequest(BaseModel):
    """Request to update an existing Segregation Control Point."""

    scp_name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Updated SCP name"
    )
    scp_type: Optional[SCPType] = Field(
        None, description="Updated SCP type"
    )
    segregation_level: Optional[SegregationLevel] = Field(
        None, description="Updated segregation level"
    )
    status: Optional[SCPStatus] = Field(
        None, description="Updated SCP status"
    )
    location: Optional[GeoCoordinate] = Field(
        None, description="Updated GPS coordinates"
    )
    description: Optional[str] = Field(
        None, max_length=2000, description="Updated description"
    )
    responsible_person: Optional[str] = Field(
        None, max_length=200, description="Updated responsible person"
    )
    operating_procedures: Optional[List[str]] = Field(
        None, description="Updated SOPs"
    )
    inspection_frequency_days: Optional[int] = Field(
        None, ge=1, le=365, description="Updated inspection frequency"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata"
    )

    model_config = ConfigDict(extra="forbid")


class ValidateSCPRequest(BaseModel):
    """Request to validate SCP compliance."""

    scp_id: str = Field(
        ..., min_length=1, max_length=100, description="SCP identifier to validate"
    )
    check_physical_segregation: bool = Field(
        default=True, description="Verify physical segregation controls"
    )
    check_documentation: bool = Field(
        default=True, description="Verify documentation completeness"
    )
    check_inspection_schedule: bool = Field(
        default=True, description="Verify inspection schedule adherence"
    )
    check_training_records: bool = Field(
        default=True, description="Verify staff training records"
    )

    model_config = ConfigDict(extra="forbid")


class SCPSearchRequest(BaseModel):
    """Request to search SCPs with filters."""

    facility_id: Optional[str] = Field(
        None, max_length=100, description="Filter by facility"
    )
    scp_type: Optional[SCPType] = Field(
        None, description="Filter by SCP type"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    status: Optional[SCPStatus] = Field(
        None, description="Filter by status"
    )
    segregation_level: Optional[SegregationLevel] = Field(
        None, description="Filter by segregation level"
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter by creation date (from)"
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter by creation date (to)"
    )
    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Results to skip")
    sort_by: str = Field(
        default="created_at", description="Sort field"
    )
    sort_order: str = Field(
        default="desc", description="Sort order (asc, desc)"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: str) -> str:
        """Validate sort field."""
        allowed = {"created_at", "scp_name", "scp_type", "status", "facility_id"}
        if v not in allowed:
            raise ValueError(f"sort_by must be one of: {allowed}")
        return v

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v: str) -> str:
        """Validate sort order."""
        if v not in {"asc", "desc"}:
            raise ValueError("sort_order must be 'asc' or 'desc'")
        return v


class SCPBatchImportRequest(BaseModel):
    """Request for bulk SCP import."""

    scps: List[RegisterSCPRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of SCPs to import (max 500)",
    )
    validate_only: bool = Field(
        default=False, description="If true, validate without persisting"
    )

    model_config = ConfigDict(extra="forbid")


class SCPResponse(BaseModel):
    """Response for a single Segregation Control Point."""

    scp_id: str = Field(..., description="Unique SCP identifier")
    facility_id: str = Field(..., description="Facility identifier")
    scp_name: str = Field(..., description="Control point name")
    scp_type: SCPType = Field(..., description="SCP type")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    segregation_level: SegregationLevel = Field(..., description="Segregation level")
    status: SCPStatus = Field(
        default=SCPStatus.ACTIVE, description="SCP status"
    )
    location: Optional[GeoCoordinate] = Field(None)
    description: Optional[str] = Field(None)
    responsible_person: Optional[str] = Field(None)
    operating_procedures: List[str] = Field(default_factory=list)
    inspection_frequency_days: int = Field(default=30)
    last_inspection_at: Optional[datetime] = Field(None)
    next_inspection_due: Optional[datetime] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)


class SCPListResponse(BaseModel):
    """Response for SCP list/search results."""

    scps: List[SCPResponse] = Field(
        default_factory=list, description="Matching SCPs"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class SCPValidationFinding(BaseModel):
    """Individual SCP validation finding."""

    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    category: str = Field(
        ..., description="Category (physical, documentation, inspection, training)"
    )
    passed: bool = Field(..., description="Whether the check passed")
    severity: ContaminationSeverity = Field(
        ..., description="Finding severity"
    )
    message: str = Field(..., description="Finding description")
    remediation: Optional[str] = Field(
        None, description="Suggested remediation"
    )

    model_config = ConfigDict(from_attributes=True)


class SCPValidationResponse(BaseModel):
    """Response from SCP compliance validation."""

    scp_id: str = Field(..., description="SCP identifier")
    is_valid: bool = Field(..., description="Overall validation result")
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    findings: List[SCPValidationFinding] = Field(
        default_factory=list, description="Validation findings"
    )
    validated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class SCPBatchImportResponse(BaseModel):
    """Response for bulk SCP import."""

    total_submitted: int = Field(..., ge=0, description="Total SCPs submitted")
    total_accepted: int = Field(..., ge=0, description="SCPs accepted")
    total_rejected: int = Field(..., ge=0, description="SCPs rejected")
    scps: List[SCPResponse] = Field(
        default_factory=list, description="Accepted SCPs"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rejection details"
    )
    validate_only: bool = Field(default=False)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Storage Zone Schemas
# =============================================================================


class RegisterZoneRequest(BaseModel):
    """Request to register a storage zone."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    zone_name: str = Field(
        ..., min_length=1, max_length=200, description="Storage zone name"
    )
    zone_type: StorageZoneType = Field(
        ..., description="Type of storage zone"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Dedicated commodity for this zone"
    )
    capacity: Optional[QuantitySpec] = Field(
        None, description="Zone storage capacity"
    )
    is_dedicated: bool = Field(
        default=True, description="Whether zone is dedicated to single commodity"
    )
    segregation_level: SegregationLevel = Field(
        default=SegregationLevel.FULL_PHYSICAL, description="Segregation level"
    )
    location: Optional[GeoCoordinate] = Field(
        None, description="GPS coordinates"
    )
    temperature_controlled: bool = Field(
        default=False, description="Whether zone has temperature control"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-BR-001",
                    "zone_name": "Silo A - Certified Soya",
                    "zone_type": "silo",
                    "commodity": "soya",
                    "is_dedicated": True,
                    "segregation_level": "full_physical",
                }
            ]
        },
    )


class RecordStorageEventRequest(BaseModel):
    """Request to record a storage event."""

    zone_id: str = Field(
        ..., min_length=1, max_length=100, description="Storage zone identifier"
    )
    event_type: StorageEventType = Field(
        ..., description="Type of storage event"
    )
    batch_id: Optional[str] = Field(
        None, max_length=100, description="Associated batch/lot ID"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity involved"
    )
    quantity: Optional[QuantitySpec] = Field(
        None, description="Quantity involved"
    )
    timestamp: Optional[datetime] = Field(
        None, description="Event timestamp (defaults to now)"
    )
    operator_name: Optional[str] = Field(
        None, max_length=200, description="Operator who performed the action"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(extra="forbid")


class StorageAuditRequest(BaseModel):
    """Request to run a storage segregation audit."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility to audit"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    check_physical_barriers: bool = Field(
        default=True, description="Check physical barrier integrity"
    )
    check_labelling: bool = Field(
        default=True, description="Check zone labelling"
    )
    check_cleaning_records: bool = Field(
        default=True, description="Check cleaning records"
    )
    check_access_controls: bool = Field(
        default=True, description="Check access control logs"
    )
    period_start: Optional[datetime] = Field(
        None, description="Audit period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Audit period end"
    )

    model_config = ConfigDict(extra="forbid")


class ZoneResponse(BaseModel):
    """Response for a storage zone."""

    zone_id: str = Field(..., description="Unique zone identifier")
    facility_id: str = Field(..., description="Facility identifier")
    zone_name: str = Field(..., description="Zone name")
    zone_type: StorageZoneType = Field(..., description="Zone type")
    commodity: EUDRCommodity = Field(..., description="Dedicated commodity")
    status: StorageZoneStatus = Field(
        default=StorageZoneStatus.AVAILABLE, description="Zone status"
    )
    capacity: Optional[QuantitySpec] = Field(None)
    current_occupancy: Optional[QuantitySpec] = Field(None)
    is_dedicated: bool = Field(default=True)
    segregation_level: SegregationLevel = Field(
        default=SegregationLevel.FULL_PHYSICAL
    )
    location: Optional[GeoCoordinate] = Field(None)
    temperature_controlled: bool = Field(default=False)
    last_cleaning_at: Optional[datetime] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ZoneListResponse(BaseModel):
    """Response for listing storage zones at a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    zones: List[ZoneResponse] = Field(
        default_factory=list, description="Storage zones"
    )
    total_zones: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class StorageEventResponse(BaseModel):
    """Response for a storage event."""

    event_id: str = Field(..., description="Unique event identifier")
    zone_id: str = Field(..., description="Storage zone ID")
    event_type: StorageEventType = Field(..., description="Event type")
    batch_id: Optional[str] = Field(None)
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: Optional[QuantitySpec] = Field(None)
    timestamp: datetime = Field(default_factory=_utcnow)
    operator_name: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class StorageAuditFinding(BaseModel):
    """Individual storage audit finding."""

    finding_id: str = Field(..., description="Finding identifier")
    category: str = Field(
        ..., description="Category (physical, labelling, cleaning, access)"
    )
    zone_id: Optional[str] = Field(None, description="Affected zone")
    severity: ContaminationSeverity = Field(..., description="Severity")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Finding description")
    remediation: Optional[str] = Field(None, description="Suggested fix")

    model_config = ConfigDict(from_attributes=True)


class StorageAuditResponse(BaseModel):
    """Response from a storage segregation audit."""

    audit_id: str = Field(..., description="Unique audit identifier")
    facility_id: str = Field(..., description="Facility audited")
    status: AssessmentStatus = Field(..., description="Audit outcome")
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    findings: List[StorageAuditFinding] = Field(
        default_factory=list, description="Audit findings"
    )
    zones_audited: int = Field(default=0, ge=0)
    audited_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class StorageScoreResponse(BaseModel):
    """Response for facility storage segregation score."""

    facility_id: str = Field(..., description="Facility identifier")
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall storage score (0-100)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    breakdown: List[ScoreBreakdown] = Field(
        default_factory=list, description="Score breakdown by category"
    )
    total_zones: int = Field(default=0, ge=0)
    compliant_zones: int = Field(default=0, ge=0)
    non_compliant_zones: int = Field(default=0, ge=0)
    last_audit_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Transport Schemas
# =============================================================================


class RegisterVehicleRequest(BaseModel):
    """Request to register a transport vehicle."""

    vehicle_reference: str = Field(
        ..., min_length=1, max_length=100, description="Vehicle reference/plate"
    )
    vehicle_type: VehicleType = Field(
        ..., description="Type of vehicle"
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=100, description="Transport operator ID"
    )
    commodity_restrictions: List[EUDRCommodity] = Field(
        default_factory=list,
        description="Commodities this vehicle is restricted to",
    )
    is_dedicated: bool = Field(
        default=False, description="Whether vehicle is dedicated to single commodity"
    )
    capacity: Optional[QuantitySpec] = Field(
        None, description="Vehicle capacity"
    )
    cleaning_protocol: Optional[str] = Field(
        None, max_length=2000, description="Cleaning protocol reference"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "vehicle_reference": "TRK-BR-2026-001",
                    "vehicle_type": "truck",
                    "operator_id": "OPR-BR-001",
                    "is_dedicated": True,
                    "commodity_restrictions": ["soya"],
                }
            ]
        },
    )


class VerifyTransportRequest(BaseModel):
    """Request to verify transport segregation."""

    vehicle_id: str = Field(
        ..., min_length=1, max_length=100, description="Vehicle identifier"
    )
    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch being transported"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity being transported"
    )
    origin_facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Origin facility"
    )
    destination_facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Destination facility"
    )
    check_cleaning_status: bool = Field(
        default=True, description="Verify cleaning status before loading"
    )
    check_previous_cargo: bool = Field(
        default=True, description="Check previous cargo compatibility"
    )
    check_seals: bool = Field(
        default=True, description="Verify seal integrity"
    )

    model_config = ConfigDict(extra="forbid")


class RecordCleaningRequest(BaseModel):
    """Request to record a vehicle cleaning verification."""

    vehicle_id: str = Field(
        ..., min_length=1, max_length=100, description="Vehicle identifier"
    )
    cleaning_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Cleaning type (dry_sweep, wet_wash, steam_clean, fumigation)",
    )
    previous_cargo_commodity: Optional[EUDRCommodity] = Field(
        None, description="Previous cargo commodity"
    )
    next_cargo_commodity: Optional[EUDRCommodity] = Field(
        None, description="Next planned cargo commodity"
    )
    inspector_name: str = Field(
        ..., min_length=1, max_length=200, description="Inspector name"
    )
    inspection_date: Optional[datetime] = Field(
        None, description="Inspection timestamp (defaults to now)"
    )
    passed: bool = Field(
        ..., description="Whether cleaning verification passed"
    )
    findings: Optional[str] = Field(
        None, max_length=2000, description="Inspection findings"
    )
    evidence_photos: List[str] = Field(
        default_factory=list, description="Evidence photo URLs"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(extra="forbid")


class VehicleResponse(BaseModel):
    """Response for a transport vehicle."""

    vehicle_id: str = Field(..., description="Unique vehicle identifier")
    vehicle_reference: str = Field(..., description="Vehicle reference/plate")
    vehicle_type: VehicleType = Field(..., description="Vehicle type")
    operator_id: str = Field(..., description="Transport operator ID")
    status: VehicleStatus = Field(
        default=VehicleStatus.AVAILABLE, description="Vehicle status"
    )
    commodity_restrictions: List[EUDRCommodity] = Field(default_factory=list)
    is_dedicated: bool = Field(default=False)
    capacity: Optional[QuantitySpec] = Field(None)
    cleaning_protocol: Optional[str] = Field(None)
    last_cleaning_at: Optional[datetime] = Field(None)
    last_cleaning_status: Optional[CleaningVerificationStatus] = Field(None)
    last_cargo_commodity: Optional[EUDRCommodity] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class TransportVerificationFinding(BaseModel):
    """Individual transport verification finding."""

    finding_id: str = Field(..., description="Finding identifier")
    category: str = Field(
        ..., description="Category (cleaning, cargo_history, seals, compatibility)"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Finding description")
    remediation: Optional[str] = Field(None, description="Suggested fix")

    model_config = ConfigDict(from_attributes=True)


class TransportVerificationResponse(BaseModel):
    """Response from transport segregation verification."""

    verification_id: str = Field(..., description="Unique verification ID")
    vehicle_id: str = Field(..., description="Vehicle verified")
    batch_id: str = Field(..., description="Batch verified")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    is_approved: bool = Field(
        ..., description="Whether transport is approved"
    )
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    findings: List[TransportVerificationFinding] = Field(
        default_factory=list, description="Verification findings"
    )
    verified_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class CargoHistoryEntry(BaseModel):
    """Single entry in vehicle cargo history."""

    entry_id: str = Field(..., description="Entry identifier")
    batch_id: Optional[str] = Field(None)
    commodity: EUDRCommodity = Field(..., description="Cargo commodity")
    origin_facility_id: Optional[str] = Field(None)
    destination_facility_id: Optional[str] = Field(None)
    loaded_at: Optional[datetime] = Field(None)
    unloaded_at: Optional[datetime] = Field(None)
    cleaning_after: Optional[CleaningVerificationStatus] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class VehicleHistoryResponse(BaseModel):
    """Response for vehicle cargo history."""

    vehicle_id: str = Field(..., description="Vehicle identifier")
    entries: List[CargoHistoryEntry] = Field(
        default_factory=list, description="Cargo history entries"
    )
    total_entries: int = Field(default=0, ge=0)
    commodities_transported: List[EUDRCommodity] = Field(
        default_factory=list, description="All commodities transported"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class TransportScoreResponse(BaseModel):
    """Response for transport segregation score."""

    vehicle_id: str = Field(..., description="Vehicle identifier")
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall transport score (0-100)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    cleaning_compliance: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Cleaning compliance rate"
    )
    cargo_compatibility_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Cargo compatibility score"
    )
    total_trips: int = Field(default=0, ge=0)
    trips_with_issues: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Processing Line Schemas
# =============================================================================


class RegisterLineRequest(BaseModel):
    """Request to register a processing line."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    line_name: str = Field(
        ..., min_length=1, max_length=200, description="Processing line name"
    )
    line_type: ProcessingLineType = Field(
        ..., description="Type of processing line"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Primary commodity for this line"
    )
    is_dedicated: bool = Field(
        default=True, description="Whether line is dedicated to single commodity"
    )
    changeover_protocol: Optional[str] = Field(
        None, max_length=2000, description="Changeover protocol reference"
    )
    minimum_changeover_time_minutes: int = Field(
        default=60, ge=0, le=1440, description="Minimum changeover time"
    )
    capacity_per_hour: Optional[QuantitySpec] = Field(
        None, description="Line capacity per hour"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-ID-001",
                    "line_name": "Palm Oil Press Line A",
                    "line_type": "dedicated",
                    "commodity": "oil_palm",
                    "is_dedicated": True,
                    "minimum_changeover_time_minutes": 120,
                }
            ]
        },
    )


class RecordChangeoverRequest(BaseModel):
    """Request to record a processing line changeover."""

    line_id: str = Field(
        ..., min_length=1, max_length=100, description="Processing line identifier"
    )
    from_commodity: EUDRCommodity = Field(
        ..., description="Commodity before changeover"
    )
    to_commodity: EUDRCommodity = Field(
        ..., description="Commodity after changeover"
    )
    changeover_start: Optional[datetime] = Field(
        None, description="Changeover start time"
    )
    changeover_end: Optional[datetime] = Field(
        None, description="Changeover end time"
    )
    cleaning_performed: bool = Field(
        default=True, description="Whether cleaning was performed"
    )
    cleaning_method: Optional[str] = Field(
        None, max_length=200, description="Cleaning method used"
    )
    inspector_name: Optional[str] = Field(
        None, max_length=200, description="Inspector name"
    )
    verification_status: CleaningVerificationStatus = Field(
        default=CleaningVerificationStatus.PENDING,
        description="Verification status",
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(extra="forbid")


class VerifyProcessingRequest(BaseModel):
    """Request to verify processing line segregation."""

    line_id: str = Field(
        ..., min_length=1, max_length=100, description="Processing line identifier"
    )
    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch being processed"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity being processed"
    )
    check_changeover_compliance: bool = Field(
        default=True, description="Verify changeover was proper"
    )
    check_line_dedication: bool = Field(
        default=True, description="Verify line dedication status"
    )
    check_residue_testing: bool = Field(
        default=True, description="Verify residue testing results"
    )

    model_config = ConfigDict(extra="forbid")


class LineResponse(BaseModel):
    """Response for a processing line."""

    line_id: str = Field(..., description="Unique line identifier")
    facility_id: str = Field(..., description="Facility identifier")
    line_name: str = Field(..., description="Line name")
    line_type: ProcessingLineType = Field(..., description="Line type")
    commodity: EUDRCommodity = Field(..., description="Primary commodity")
    status: ProcessingLineStatus = Field(
        default=ProcessingLineStatus.IDLE, description="Line status"
    )
    is_dedicated: bool = Field(default=True)
    changeover_protocol: Optional[str] = Field(None)
    minimum_changeover_time_minutes: int = Field(default=60)
    capacity_per_hour: Optional[QuantitySpec] = Field(None)
    current_commodity: Optional[EUDRCommodity] = Field(
        None, description="Currently running commodity"
    )
    last_changeover_at: Optional[datetime] = Field(None)
    total_changeovers: int = Field(default=0, ge=0)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ChangeoverResponse(BaseModel):
    """Response for a line changeover record."""

    changeover_id: str = Field(..., description="Unique changeover identifier")
    line_id: str = Field(..., description="Processing line ID")
    from_commodity: EUDRCommodity = Field(..., description="Previous commodity")
    to_commodity: EUDRCommodity = Field(..., description="New commodity")
    changeover_start: datetime = Field(..., description="Changeover start")
    changeover_end: Optional[datetime] = Field(None, description="Changeover end")
    duration_minutes: Optional[float] = Field(
        None, ge=0.0, description="Changeover duration"
    )
    cleaning_performed: bool = Field(default=True)
    cleaning_method: Optional[str] = Field(None)
    inspector_name: Optional[str] = Field(None)
    verification_status: CleaningVerificationStatus = Field(
        default=CleaningVerificationStatus.PENDING
    )
    meets_minimum_time: bool = Field(
        default=True, description="Whether minimum changeover time was met"
    )
    notes: Optional[str] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ProcessingVerificationFinding(BaseModel):
    """Individual processing verification finding."""

    finding_id: str = Field(..., description="Finding identifier")
    category: str = Field(
        ..., description="Category (changeover, dedication, residue, compatibility)"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Finding description")
    remediation: Optional[str] = Field(None, description="Suggested fix")

    model_config = ConfigDict(from_attributes=True)


class ProcessingVerificationResponse(BaseModel):
    """Response from processing line segregation verification."""

    verification_id: str = Field(..., description="Unique verification ID")
    line_id: str = Field(..., description="Processing line verified")
    batch_id: str = Field(..., description="Batch verified")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    is_approved: bool = Field(
        ..., description="Whether processing is approved"
    )
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    findings: List[ProcessingVerificationFinding] = Field(
        default_factory=list, description="Verification findings"
    )
    verified_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ProcessingScoreResponse(BaseModel):
    """Response for facility processing segregation score."""

    facility_id: str = Field(..., description="Facility identifier")
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall processing score (0-100)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level assessment")
    breakdown: List[ScoreBreakdown] = Field(
        default_factory=list, description="Score breakdown by category"
    )
    total_lines: int = Field(default=0, ge=0)
    dedicated_lines: int = Field(default=0, ge=0)
    shared_lines: int = Field(default=0, ge=0)
    changeover_compliance_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Changeover compliance rate"
    )
    last_audit_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Contamination Schemas
# =============================================================================


class DetectContaminationRequest(BaseModel):
    """Request to run contamination detection analysis."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility to scan"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Target commodity to check"
    )
    scan_storage: bool = Field(
        default=True, description="Scan storage zones"
    )
    scan_transport: bool = Field(
        default=True, description="Scan transport records"
    )
    scan_processing: bool = Field(
        default=True, description="Scan processing lines"
    )
    scan_labelling: bool = Field(
        default=True, description="Scan labelling records"
    )
    sensitivity: str = Field(
        default="standard",
        description="Detection sensitivity (standard, high, forensic)",
    )
    period_start: Optional[datetime] = Field(
        None, description="Scan period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Scan period end"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("sensitivity")
    @classmethod
    def validate_sensitivity(cls, v: str) -> str:
        """Validate sensitivity level."""
        allowed = {"standard", "high", "forensic"}
        if v not in allowed:
            raise ValueError(f"sensitivity must be one of: {allowed}")
        return v


class RecordContaminationRequest(BaseModel):
    """Request to record a contamination event."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    contamination_type: ContaminationType = Field(
        ..., description="Type of contamination"
    )
    severity: ContaminationSeverity = Field(
        ..., description="Contamination severity"
    )
    commodity_affected: EUDRCommodity = Field(
        ..., description="Affected commodity"
    )
    commodity_contaminant: Optional[EUDRCommodity] = Field(
        None, description="Contaminant commodity"
    )
    location_type: str = Field(
        ..., min_length=1, max_length=50,
        description="Where contamination occurred (storage, transport, processing)",
    )
    location_id: Optional[str] = Field(
        None, max_length=100, description="Zone/vehicle/line ID"
    )
    batch_ids_affected: List[str] = Field(
        default_factory=list, description="Affected batch IDs"
    )
    quantity_affected: Optional[QuantitySpec] = Field(
        None, description="Quantity of affected material"
    )
    detected_at: Optional[datetime] = Field(
        None, description="Detection timestamp"
    )
    detected_by: Optional[str] = Field(
        None, max_length=200, description="Person who detected"
    )
    root_cause: Optional[str] = Field(
        None, max_length=2000, description="Root cause analysis"
    )
    corrective_actions: Optional[str] = Field(
        None, max_length=2000, description="Corrective actions taken"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(extra="forbid")


class AssessImpactRequest(BaseModel):
    """Request to assess contamination impact."""

    event_id: str = Field(
        ..., min_length=1, max_length=100, description="Contamination event ID"
    )
    include_downstream_batches: bool = Field(
        default=True, description="Trace downstream batch impact"
    )
    include_financial_impact: bool = Field(
        default=True, description="Estimate financial impact"
    )
    include_regulatory_impact: bool = Field(
        default=True, description="Assess regulatory implications"
    )

    model_config = ConfigDict(extra="forbid")


class ContaminationEventResponse(BaseModel):
    """Response for a contamination event."""

    event_id: str = Field(..., description="Unique event identifier")
    facility_id: str = Field(..., description="Facility identifier")
    contamination_type: ContaminationType = Field(
        ..., description="Contamination type"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    status: ContaminationStatus = Field(
        default=ContaminationStatus.DETECTED, description="Event status"
    )
    commodity_affected: EUDRCommodity = Field(..., description="Affected commodity")
    commodity_contaminant: Optional[EUDRCommodity] = Field(None)
    location_type: str = Field(..., description="Location type")
    location_id: Optional[str] = Field(None)
    batch_ids_affected: List[str] = Field(default_factory=list)
    quantity_affected: Optional[QuantitySpec] = Field(None)
    detected_at: datetime = Field(default_factory=_utcnow)
    detected_by: Optional[str] = Field(None)
    root_cause: Optional[str] = Field(None)
    corrective_actions: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ContaminationDetectionFinding(BaseModel):
    """Individual contamination detection finding."""

    finding_id: str = Field(..., description="Finding identifier")
    area: str = Field(
        ..., description="Area (storage, transport, processing, labelling)"
    )
    contamination_type: ContaminationType = Field(
        ..., description="Contamination type"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    location_id: Optional[str] = Field(None)
    batch_ids_at_risk: List[str] = Field(default_factory=list)
    message: str = Field(..., description="Finding description")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Detection confidence (0-1)"
    )
    remediation: Optional[str] = Field(None, description="Suggested action")

    model_config = ConfigDict(from_attributes=True)


class ContaminationDetectionResponse(BaseModel):
    """Response from contamination detection analysis."""

    detection_id: str = Field(..., description="Unique detection run ID")
    facility_id: str = Field(..., description="Facility scanned")
    commodity: EUDRCommodity = Field(..., description="Target commodity")
    risk_level: RiskLevel = Field(..., description="Overall risk level")
    total_areas_scanned: int = Field(default=0, ge=0)
    issues_found: int = Field(default=0, ge=0)
    findings: List[ContaminationDetectionFinding] = Field(
        default_factory=list, description="Detection findings"
    )
    scanned_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ContaminationImpactResponse(BaseModel):
    """Response from contamination impact assessment."""

    impact_id: str = Field(..., description="Unique impact assessment ID")
    event_id: str = Field(..., description="Contamination event ID")
    severity: ContaminationSeverity = Field(..., description="Impact severity")
    batches_affected: List[str] = Field(
        default_factory=list, description="All affected batch IDs"
    )
    total_batches_affected: int = Field(default=0, ge=0)
    quantity_at_risk: Optional[QuantitySpec] = Field(
        None, description="Total quantity at risk"
    )
    downstream_facilities: List[str] = Field(
        default_factory=list, description="Downstream facilities impacted"
    )
    estimated_financial_impact: Optional[Dict[str, Any]] = Field(
        None, description="Financial impact estimate"
    )
    regulatory_implications: List[str] = Field(
        default_factory=list, description="Regulatory implications"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended corrective actions"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class HeatmapCell(BaseModel):
    """Single cell in a risk heatmap."""

    zone_id: str = Field(..., description="Zone/area identifier")
    zone_name: str = Field(..., description="Zone/area name")
    risk_level: RiskLevel = Field(..., description="Risk level")
    risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Risk score (0-100)"
    )
    contamination_events: int = Field(
        default=0, ge=0, description="Number of contamination events"
    )
    last_event_at: Optional[datetime] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class RiskHeatmapResponse(BaseModel):
    """Response for facility risk heatmap."""

    facility_id: str = Field(..., description="Facility identifier")
    overall_risk: RiskLevel = Field(..., description="Overall facility risk")
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall risk score"
    )
    cells: List[HeatmapCell] = Field(
        default_factory=list, description="Heatmap cells"
    )
    total_zones: int = Field(default=0, ge=0)
    high_risk_zones: int = Field(default=0, ge=0)
    generated_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Label Schemas
# =============================================================================


class RegisterLabelRequest(BaseModel):
    """Request to register a segregation label."""

    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Associated batch ID"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity on label"
    )
    label_text: str = Field(
        ..., min_length=1, max_length=500, description="Label text content"
    )
    label_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Label type (segregation, certification, origin, warning)",
    )
    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility where label applied"
    )
    applied_at: Optional[datetime] = Field(
        None, description="When label was applied"
    )
    applied_by: Optional[str] = Field(
        None, max_length=200, description="Person who applied label"
    )
    barcode: Optional[str] = Field(
        None, max_length=200, description="Label barcode/QR code"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(extra="forbid")


class VerifyLabelRequest(BaseModel):
    """Request to verify a segregation label."""

    label_id: str = Field(
        ..., min_length=1, max_length=100, description="Label identifier"
    )
    check_content_accuracy: bool = Field(
        default=True, description="Verify label content matches batch"
    )
    check_placement: bool = Field(
        default=True, description="Verify label placement"
    )
    check_legibility: bool = Field(
        default=True, description="Verify label is legible"
    )

    model_config = ConfigDict(extra="forbid")


class LabelAuditRequest(BaseModel):
    """Request for label audit at a facility."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility to audit"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    check_consistency: bool = Field(
        default=True, description="Check label consistency across batches"
    )
    check_completeness: bool = Field(
        default=True, description="Check all batches have labels"
    )
    period_start: Optional[datetime] = Field(
        None, description="Audit period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Audit period end"
    )

    model_config = ConfigDict(extra="forbid")


class LabelResponse(BaseModel):
    """Response for a segregation label."""

    label_id: str = Field(..., description="Unique label identifier")
    batch_id: str = Field(..., description="Associated batch ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    label_text: str = Field(..., description="Label text")
    label_type: str = Field(..., description="Label type")
    facility_id: str = Field(..., description="Facility")
    status: LabelStatus = Field(
        default=LabelStatus.ACTIVE, description="Label status"
    )
    applied_at: datetime = Field(default_factory=_utcnow)
    applied_by: Optional[str] = Field(None)
    barcode: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class LabelVerificationFinding(BaseModel):
    """Individual label verification finding."""

    finding_id: str = Field(..., description="Finding identifier")
    category: str = Field(
        ..., description="Category (content, placement, legibility)"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Finding description")

    model_config = ConfigDict(from_attributes=True)


class LabelVerificationResponse(BaseModel):
    """Response from label verification."""

    verification_id: str = Field(..., description="Unique verification ID")
    label_id: str = Field(..., description="Label verified")
    is_valid: bool = Field(..., description="Overall validity")
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    findings: List[LabelVerificationFinding] = Field(
        default_factory=list, description="Verification findings"
    )
    verified_at: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class LabelAuditFinding(BaseModel):
    """Individual label audit finding."""

    finding_id: str = Field(..., description="Finding identifier")
    category: str = Field(
        ..., description="Category (consistency, completeness)"
    )
    severity: ContaminationSeverity = Field(..., description="Severity")
    passed: bool = Field(..., description="Whether the check passed")
    batch_id: Optional[str] = Field(None)
    message: str = Field(..., description="Finding description")

    model_config = ConfigDict(from_attributes=True)


class LabelAuditResponse(BaseModel):
    """Response from label audit."""

    audit_id: str = Field(..., description="Unique audit identifier")
    facility_id: str = Field(..., description="Facility audited")
    is_compliant: bool = Field(..., description="Overall compliance result")
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_labels_checked: int = Field(default=0, ge=0)
    labels_valid: int = Field(default=0, ge=0)
    labels_invalid: int = Field(default=0, ge=0)
    batches_without_labels: int = Field(default=0, ge=0)
    findings: List[LabelAuditFinding] = Field(
        default_factory=list, description="Audit findings"
    )
    audited_at: datetime = Field(default_factory=_utcnow)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Facility Assessment Schemas
# =============================================================================


class RunAssessmentRequest(BaseModel):
    """Request to run a facility segregation assessment."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility to assess"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Focus commodity (all if omitted)"
    )
    include_storage: bool = Field(
        default=True, description="Include storage segregation"
    )
    include_transport: bool = Field(
        default=True, description="Include transport segregation"
    )
    include_processing: bool = Field(
        default=True, description="Include processing segregation"
    )
    include_contamination: bool = Field(
        default=True, description="Include contamination risk"
    )
    include_labelling: bool = Field(
        default=True, description="Include labelling compliance"
    )
    include_documentation: bool = Field(
        default=True, description="Include documentation completeness"
    )

    model_config = ConfigDict(extra="forbid")


class AssessmentCategoryResult(BaseModel):
    """Assessment result for a single category."""

    category: str = Field(..., description="Assessment category")
    score: float = Field(..., ge=0.0, le=100.0, description="Category score")
    weight: float = Field(..., ge=0.0, le=1.0, description="Category weight")
    status: AssessmentStatus = Field(..., description="Category status")
    findings_count: int = Field(default=0, ge=0)
    critical_issues: int = Field(default=0, ge=0)

    model_config = ConfigDict(from_attributes=True)


class AssessmentResponse(BaseModel):
    """Response for a facility segregation assessment."""

    assessment_id: str = Field(..., description="Unique assessment ID")
    facility_id: str = Field(..., description="Facility assessed")
    commodity: Optional[EUDRCommodity] = Field(None)
    overall_status: AssessmentStatus = Field(
        ..., description="Overall assessment status"
    )
    overall_score: float = Field(
        ..., ge=0.0, le=100.0, description="Overall score (0-100)"
    )
    risk_level: RiskLevel = Field(..., description="Risk level")
    categories: List[AssessmentCategoryResult] = Field(
        default_factory=list, description="Category-level results"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    critical_issues: int = Field(default=0, ge=0)
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    assessed_at: datetime = Field(default_factory=_utcnow)
    valid_until: Optional[datetime] = Field(
        None, description="Assessment validity period"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class AssessmentHistoryResponse(BaseModel):
    """Response for assessment history."""

    facility_id: str = Field(..., description="Facility identifier")
    assessments: List[AssessmentResponse] = Field(
        default_factory=list, description="Historical assessments"
    )
    total_assessments: int = Field(default=0, ge=0)
    trend: str = Field(
        default="stable",
        description="Score trend (improving, stable, declining)",
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas
# =============================================================================


class GenerateAuditReportRequest(BaseModel):
    """Request to generate a segregation audit report."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    period_start: datetime = Field(
        ..., description="Report period start"
    )
    period_end: datetime = Field(
        ..., description="Report period end"
    )
    include_storage: bool = Field(default=True)
    include_transport: bool = Field(default=True)
    include_processing: bool = Field(default=True)
    include_labelling: bool = Field(default=True)
    output_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Report output format"
    )
    language: str = Field(
        default="en", max_length=5, description="Report language"
    )

    model_config = ConfigDict(extra="forbid")


class GenerateContaminationReportRequest(BaseModel):
    """Request to generate a contamination report."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    period_start: datetime = Field(
        ..., description="Report period start"
    )
    period_end: datetime = Field(
        ..., description="Report period end"
    )
    include_impact_analysis: bool = Field(
        default=True, description="Include impact analysis"
    )
    include_corrective_actions: bool = Field(
        default=True, description="Include corrective action details"
    )
    output_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Report output format"
    )

    model_config = ConfigDict(extra="forbid")


class GenerateEvidencePackageRequest(BaseModel):
    """Request to generate a regulatory evidence package."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility identifier"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity for evidence package"
    )
    batch_ids: List[str] = Field(
        default_factory=list,
        description="Specific batches to include (all if empty)",
    )
    period_start: datetime = Field(
        ..., description="Evidence period start"
    )
    period_end: datetime = Field(
        ..., description="Evidence period end"
    )
    include_scp_records: bool = Field(default=True)
    include_storage_records: bool = Field(default=True)
    include_transport_records: bool = Field(default=True)
    include_processing_records: bool = Field(default=True)
    include_contamination_records: bool = Field(default=True)
    include_assessment_results: bool = Field(default=True)
    output_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Report output format"
    )

    model_config = ConfigDict(extra="forbid")


class ReportResponse(BaseModel):
    """Response for a generated report."""

    report_id: str = Field(..., description="Unique report ID")
    report_type: ReportType = Field(..., description="Report type")
    title: str = Field(..., description="Report title")
    status: str = Field(default="generated", description="Generation status")
    output_format: ReportFormat = Field(..., description="Output format")
    facility_id: str = Field(..., description="Facility ID")
    commodity: Optional[EUDRCommodity] = Field(None)
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Report summary data"
    )
    download_url: Optional[str] = Field(
        None, description="URL to download report"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size in bytes"
    )
    generated_at: datetime = Field(default_factory=_utcnow)
    expires_at: Optional[datetime] = Field(
        None, description="Download link expiry"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ReportDownloadResponse(BaseModel):
    """Response for report download."""

    report_id: str = Field(..., description="Report ID")
    download_url: str = Field(..., description="Signed download URL")
    output_format: ReportFormat = Field(..., description="File format")
    file_size_bytes: Optional[int] = Field(None, ge=0)
    content_type: str = Field(
        default="application/pdf", description="MIME content type"
    )
    expires_at: datetime = Field(
        default_factory=_utcnow, description="URL expiry"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Job Schemas
# =============================================================================


class BatchJobSubmitRequest(BaseModel):
    """Request to submit an async batch job."""

    job_type: BatchJobType = Field(
        ..., description="Type of batch job"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Job priority (1=highest, 10=lowest)"
    )
    callback_url: Optional[str] = Field(
        None, max_length=2000, description="Webhook URL for completion notification"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Job notes"
    )

    model_config = ConfigDict(extra="forbid")


class BatchJobResponse(BaseModel):
    """Response for a batch job."""

    job_id: str = Field(..., description="Unique job ID")
    job_type: BatchJobType = Field(..., description="Job type")
    status: BatchJobStatus = Field(..., description="Current job status")
    priority: int = Field(default=5, ge=1, le=10)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    total_items: Optional[int] = Field(None, ge=0)
    processed_items: Optional[int] = Field(None, ge=0)
    failed_items: Optional[int] = Field(None, ge=0)
    result: Optional[Dict[str, Any]] = Field(None)
    error: Optional[str] = Field(None)
    callback_url: Optional[str] = Field(None)
    submitted_at: datetime = Field(default_factory=_utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    cancelled_at: Optional[datetime] = Field(None)
    provenance_hash: str = Field(default="")

    model_config = ConfigDict(from_attributes=True)


class BatchJobCancelResponse(BaseModel):
    """Response after cancelling a batch job."""

    job_id: str = Field(..., description="Cancelled job ID")
    status: BatchJobStatus = Field(
        default=BatchJobStatus.CANCELLED, description="Updated status"
    )
    cancelled_at: datetime = Field(default_factory=_utcnow)
    message: str = Field(default="Job cancelled successfully")

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Health Check Schema
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response for the Segregation Verifier API."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-SGV-010")
    agent_name: str = Field(
        default="EUDR Segregation Verifier Agent"
    )
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=_utcnow)
    components: Dict[str, str] = Field(
        default_factory=lambda: {
            "scp_manager": "healthy",
            "storage_verifier": "healthy",
            "transport_verifier": "healthy",
            "processing_verifier": "healthy",
            "contamination_detector": "healthy",
            "assessment_engine": "healthy",
            "report_generator": "healthy",
        },
        description="Component health statuses",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Forward reference rebuilds
# =============================================================================

SCPBatchImportRequest.model_rebuild()
RunAssessmentRequest.model_rebuild()
GenerateEvidencePackageRequest.model_rebuild()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "AssessmentStatus",
    "BatchJobStatus",
    "BatchJobType",
    "ChangeoverStatus",
    "CleaningVerificationStatus",
    "ContaminationSeverity",
    "ContaminationStatus",
    "ContaminationType",
    "EUDRCommodity",
    "LabelStatus",
    "ProcessingLineStatus",
    "ProcessingLineType",
    "ReportFormat",
    "ReportType",
    "RiskLevel",
    "SCPStatus",
    "SCPType",
    "SegregationLevel",
    "StorageEventType",
    "StorageZoneStatus",
    "StorageZoneType",
    "UnitOfMeasure",
    "VehicleStatus",
    "VehicleType",
    # Common
    "GeoCoordinate",
    "HeatmapCell",
    "PaginatedMeta",
    "ProvenanceInfo",
    "QuantitySpec",
    "ScoreBreakdown",
    # SCP
    "RegisterSCPRequest",
    "SCPBatchImportRequest",
    "SCPBatchImportResponse",
    "SCPListResponse",
    "SCPResponse",
    "SCPSearchRequest",
    "SCPValidationFinding",
    "SCPValidationResponse",
    "UpdateSCPRequest",
    "ValidateSCPRequest",
    # Storage
    "RecordStorageEventRequest",
    "RegisterZoneRequest",
    "StorageAuditFinding",
    "StorageAuditRequest",
    "StorageAuditResponse",
    "StorageEventResponse",
    "StorageScoreResponse",
    "ZoneListResponse",
    "ZoneResponse",
    # Transport
    "CargoHistoryEntry",
    "RecordCleaningRequest",
    "RegisterVehicleRequest",
    "TransportScoreResponse",
    "TransportVerificationFinding",
    "TransportVerificationResponse",
    "VehicleHistoryResponse",
    "VehicleResponse",
    "VerifyTransportRequest",
    # Processing
    "ChangeoverResponse",
    "LineResponse",
    "ProcessingScoreResponse",
    "ProcessingVerificationFinding",
    "ProcessingVerificationResponse",
    "RecordChangeoverRequest",
    "RegisterLineRequest",
    "VerifyProcessingRequest",
    # Contamination
    "AssessImpactRequest",
    "ContaminationDetectionFinding",
    "ContaminationDetectionResponse",
    "ContaminationEventResponse",
    "ContaminationImpactResponse",
    "DetectContaminationRequest",
    "RecordContaminationRequest",
    "RiskHeatmapResponse",
    # Labels
    "LabelAuditFinding",
    "LabelAuditRequest",
    "LabelAuditResponse",
    "LabelResponse",
    "LabelVerificationFinding",
    "LabelVerificationResponse",
    "RegisterLabelRequest",
    "VerifyLabelRequest",
    # Assessment
    "AssessmentCategoryResult",
    "AssessmentHistoryResponse",
    "AssessmentResponse",
    "RunAssessmentRequest",
    # Reports
    "GenerateAuditReportRequest",
    "GenerateContaminationReportRequest",
    "GenerateEvidencePackageRequest",
    "ReportDownloadResponse",
    "ReportResponse",
    # Batch Jobs
    "BatchJobCancelResponse",
    "BatchJobResponse",
    "BatchJobSubmitRequest",
    # Health
    "HealthResponse",
]

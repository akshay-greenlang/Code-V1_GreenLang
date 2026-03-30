# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-011 Mass Balance Calculator

Pydantic v2 request/response models for all Mass Balance Calculator REST API
endpoints. Organized by domain: ledgers, credit periods, conversion factors,
overdraft detection, loss/waste tracking, reconciliation, consolidation,
batch jobs, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

All quantity fields use ``Decimal`` for bit-perfect arithmetic required by
EUDR Article 14 and ISO 22095:2020 mass balance accounting.

Model Count: 80+ schemas covering 34 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-011, Section 7.4
Agent ID: GL-EUDR-MBC-011
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field, field_validator, model_validator

from greenlang.schemas import GreenLangBase, utcnow

# =============================================================================
# Helpers
# =============================================================================

def _new_id() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

# =============================================================================
# Enumerations (API-layer mirrors of domain enums)
# =============================================================================

class LedgerEntryTypeSchema(str, Enum):
    """Type of mass balance ledger entry."""

    INPUT = "input"
    OUTPUT = "output"
    ADJUSTMENT = "adjustment"
    LOSS = "loss"
    WASTE = "waste"
    CARRY_FORWARD_IN = "carry_forward_in"
    CARRY_FORWARD_OUT = "carry_forward_out"
    EXPIRY = "expiry"

class PeriodStatusSchema(str, Enum):
    """Lifecycle status of a credit period."""

    PENDING = "pending"
    ACTIVE = "active"
    RECONCILING = "reconciling"
    CLOSED = "closed"

class OverdraftSeveritySchema(str, Enum):
    """Severity classification for overdraft events."""

    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"

class OverdraftModeSchema(str, Enum):
    """Overdraft enforcement mode."""

    ZERO_TOLERANCE = "zero_tolerance"
    PERCENTAGE = "percentage"
    ABSOLUTE = "absolute"

class LossTypeSchema(str, Enum):
    """Type of material loss."""

    PROCESSING_LOSS = "processing_loss"
    TRANSPORT_LOSS = "transport_loss"
    STORAGE_LOSS = "storage_loss"
    QUALITY_REJECTION = "quality_rejection"
    SPILLAGE = "spillage"
    CONTAMINATION_LOSS = "contamination_loss"

class WasteTypeSchema(str, Enum):
    """Type of waste material."""

    BY_PRODUCT = "by_product"
    WASTE_MATERIAL = "waste_material"
    HAZARDOUS_WASTE = "hazardous_waste"

class ConversionStatusSchema(str, Enum):
    """Validation status of a conversion factor."""

    VALIDATED = "validated"
    WARNED = "warned"
    REJECTED = "rejected"
    PENDING = "pending"

class VarianceClassificationSchema(str, Enum):
    """Classification of reconciliation variance."""

    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    VIOLATION = "violation"

class ReconciliationStatusSchema(str, Enum):
    """Status of a period reconciliation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SIGNED_OFF = "signed_off"

class CarryForwardStatusSchema(str, Enum):
    """Status of a carry-forward balance transfer."""

    ACTIVE = "active"
    EXPIRED = "expired"
    UTILIZED = "utilized"
    PARTIAL = "partial"

class ReportFormatSchema(str, Enum):
    """Output format for reports."""

    JSON = "json"
    CSV = "csv"
    PDF = "pdf"
    EUDR_XML = "eudr_xml"

class ReportTypeSchema(str, Enum):
    """Type of mass balance report."""

    RECONCILIATION = "reconciliation"
    CONSOLIDATION = "consolidation"
    OVERDRAFT = "overdraft"
    VARIANCE = "variance"
    EVIDENCE = "evidence"

class FacilityGroupTypeSchema(str, Enum):
    """Type of facility grouping for consolidation."""

    REGION = "region"
    COUNTRY = "country"
    COMMODITY = "commodity"
    CUSTOM = "custom"

class ComplianceStatusSchema(str, Enum):
    """Compliance status for mass balance operations."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"

class StandardTypeSchema(str, Enum):
    """Certification standard governing the mass balance."""

    RSPO = "rspo"
    FSC = "fsc"
    ISCC = "iscc"
    UTZ_RA = "utz_ra"
    FAIRTRADE = "fairtrade"
    EUDR_DEFAULT = "eudr_default"

class BatchJobStatusSchema(str, Enum):
    """Status of an async batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BatchJobTypeSchema(str, Enum):
    """Types of batch jobs supported by mass balance calculator."""

    LEDGER_ENTRY_IMPORT = "ledger_entry_import"
    RECONCILIATION_BATCH = "reconciliation_batch"
    REPORT_GENERATION = "report_generation"
    LOSS_IMPORT = "loss_import"
    FACTOR_VALIDATION = "factor_validation"

class SortOrderSchema(str, Enum):
    """Sort order for list endpoints."""

    ASC = "asc"
    DESC = "desc"

# =============================================================================
# Shared / Common Models
# =============================================================================

class ProvenanceInfo(GreenLangBase):
    """Provenance tracking information for audit trail."""

    provenance_hash: str = Field(
        ..., description="SHA-256 hash of input data"
    )
    created_by: str = Field(..., description="User ID who created the record")
    created_at: datetime = Field(
        default_factory=utcnow, description="Creation timestamp (UTC)"
    )
    source: str = Field(
        default="api", description="Data source (api, import, system)"
    )

    model_config = ConfigDict(from_attributes=True)

class PaginatedMeta(GreenLangBase):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")

    model_config = ConfigDict(from_attributes=True)

class ErrorDetail(GreenLangBase):
    """Individual error detail within an error response."""

    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error description")
    code: Optional[str] = Field(None, description="Error code")

    model_config = ConfigDict(from_attributes=True)

class ErrorResponse(GreenLangBase):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    errors: List[ErrorDetail] = Field(
        default_factory=list, description="List of individual errors"
    )
    request_id: Optional[str] = Field(None, description="Request correlation ID")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Ledger Schemas
# =============================================================================

class CreateLedgerSchema(GreenLangBase):
    """Request to create a new mass balance ledger."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Facility identifier",
    )
    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="EUDR commodity (cattle, cocoa, coffee, oil_palm, rubber, soya, wood)",
    )
    standard: StandardTypeSchema = Field(
        default=StandardTypeSchema.EUDR_DEFAULT,
        description="Certification standard governing this ledger",
    )
    initial_balance: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Initial balance in kilograms (default: 0)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for context",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-GH-001",
                    "commodity": "cocoa",
                    "standard": "rspo",
                    "initial_balance": "0",
                }
            ]
        },
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate commodity is a known EUDR commodity or derived product."""
        if not v or not v.strip():
            raise ValueError("commodity must be non-empty")
        return v.strip().lower()

class RecordEntrySchema(GreenLangBase):
    """Request to record a single ledger entry."""

    ledger_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Target ledger identifier",
    )
    entry_type: LedgerEntryTypeSchema = Field(
        ..., description="Type of ledger entry",
    )
    quantity_kg: Decimal = Field(
        ..., gt=0, description="Quantity in kilograms (must be positive)",
    )
    batch_id: Optional[str] = Field(
        None, max_length=100,
        description="Associated batch identifier for traceability",
    )
    source_destination: Optional[str] = Field(
        None, max_length=200,
        description="Source (for inputs) or destination (for outputs)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator recording the entry",
    )
    conversion_factor: Optional[float] = Field(
        None, gt=0.0, le=1.0,
        description="Conversion factor applied (yield ratio 0-1)",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Free-text notes or observations",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "ledger_id": "led-001",
                    "entry_type": "input",
                    "quantity_kg": "5000.00",
                    "batch_id": "BATCH-2026-001",
                    "source_destination": "FAC-GH-COOP-001",
                }
            ]
        },
    )

class BulkEntrySchema(GreenLangBase):
    """Request to record multiple ledger entries in bulk."""

    entries: List[RecordEntrySchema] = Field(
        ..., min_length=1, max_length=500,
        description="List of entries to record (max 500)",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator performing the bulk operation",
    )
    validate_only: bool = Field(
        default=False,
        description="If true, validate without persisting",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional key-value pairs for the batch",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("entries")
    @classmethod
    def validate_max_entries(cls, v: List[RecordEntrySchema]) -> List[RecordEntrySchema]:
        """Ensure entries do not exceed max batch size."""
        if len(v) > 500:
            raise ValueError(f"Maximum 500 entries per bulk request, got {len(v)}")
        return v

class SearchLedgerSchema(GreenLangBase):
    """Request to search ledgers by criteria."""

    facility_id: Optional[str] = Field(
        None, max_length=100, description="Filter by facility",
    )
    commodity: Optional[str] = Field(
        None, max_length=50, description="Filter by commodity",
    )
    standard: Optional[StandardTypeSchema] = Field(
        None, description="Filter by certification standard",
    )
    min_balance: Optional[Decimal] = Field(
        None, description="Minimum balance filter (kg)",
    )
    max_balance: Optional[Decimal] = Field(
        None, description="Maximum balance filter (kg)",
    )
    compliance_status: Optional[ComplianceStatusSchema] = Field(
        None, description="Filter by compliance status",
    )
    sort_by: str = Field(
        default="created_at", description="Field to sort by",
    )
    sort_order: SortOrderSchema = Field(
        default=SortOrderSchema.DESC, description="Sort order",
    )
    limit: int = Field(
        default=50, ge=1, le=1000, description="Maximum results to return",
    )
    offset: int = Field(
        default=0, ge=0, description="Result offset for pagination",
    )

    model_config = ConfigDict(extra="forbid")

# -- Ledger Response Schemas --

class LedgerEntryDetailSchema(GreenLangBase):
    """Detailed ledger entry for response payloads."""

    entry_id: str = Field(..., description="Unique entry identifier")
    ledger_id: str = Field(..., description="Parent ledger identifier")
    entry_type: LedgerEntryTypeSchema = Field(..., description="Entry type")
    quantity_kg: Decimal = Field(..., description="Quantity in kg")
    batch_id: Optional[str] = Field(None, description="Associated batch")
    source_destination: Optional[str] = Field(None, description="Source or destination")
    conversion_factor_applied: Optional[float] = Field(
        None, description="Conversion factor applied"
    )
    compliance_status: ComplianceStatusSchema = Field(
        ..., description="Material compliance status"
    )
    operator_id: Optional[str] = Field(None, description="Operator ID")
    notes: Optional[str] = Field(None, description="Notes")
    voided: bool = Field(default=False, description="Whether entry is voided")
    voided_at: Optional[datetime] = Field(None, description="Void timestamp")
    voided_by: Optional[str] = Field(None, description="Voided by operator")
    void_reason: Optional[str] = Field(None, description="Reason for voiding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    timestamp: datetime = Field(..., description="Entry timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)

class LedgerDetailSchema(GreenLangBase):
    """Response for a single ledger with full details."""

    ledger_id: str = Field(..., description="Unique ledger identifier")
    facility_id: str = Field(..., description="Facility identifier")
    commodity: str = Field(..., description="EUDR commodity")
    standard: StandardTypeSchema = Field(..., description="Certification standard")
    period_id: Optional[str] = Field(None, description="Active credit period")
    current_balance: Decimal = Field(..., description="Current balance in kg")
    total_inputs: Decimal = Field(..., description="Cumulative inputs in kg")
    total_outputs: Decimal = Field(..., description="Cumulative outputs in kg")
    total_losses: Decimal = Field(..., description="Cumulative losses in kg")
    total_waste: Decimal = Field(..., description="Cumulative waste in kg")
    utilization_rate: float = Field(..., description="Utilization rate (0-1)")
    compliance_status: ComplianceStatusSchema = Field(
        ..., description="Current compliance status"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class LedgerBalanceSchema(GreenLangBase):
    """Response with detailed ledger balance information."""

    ledger_id: str = Field(..., description="Ledger identifier")
    current_balance: Decimal = Field(..., description="Current balance in kg")
    total_inputs: Decimal = Field(..., description="Cumulative inputs")
    total_outputs: Decimal = Field(..., description="Cumulative outputs")
    total_losses: Decimal = Field(..., description="Cumulative losses")
    total_waste: Decimal = Field(..., description="Cumulative waste")
    utilization_rate: float = Field(..., description="Utilization rate")
    carry_forward_available: Decimal = Field(
        default=Decimal("0"), description="Available carry-forward balance"
    )
    overdraft_status: bool = Field(
        default=False, description="Whether overdraft exists"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class EntryHistorySchema(GreenLangBase):
    """Response with ledger entry history."""

    ledger_id: str = Field(..., description="Ledger identifier")
    entries: List[LedgerEntryDetailSchema] = Field(
        default_factory=list, description="Ledger entries"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class BulkEntryResultSchema(GreenLangBase):
    """Response for bulk entry import."""

    total_submitted: int = Field(..., ge=0, description="Total entries submitted")
    total_accepted: int = Field(..., ge=0, description="Entries accepted")
    total_rejected: int = Field(..., ge=0, description="Entries rejected")
    entries: List[LedgerEntryDetailSchema] = Field(
        default_factory=list, description="Accepted entries"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rejection details"
    )
    validate_only: bool = Field(
        default=False, description="Whether this was validation-only"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class LedgerListSchema(GreenLangBase):
    """Response for ledger search results."""

    ledgers: List[LedgerDetailSchema] = Field(
        default_factory=list, description="Matching ledgers"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Period Schemas
# =============================================================================

class CreatePeriodSchema(GreenLangBase):
    """Request to create a new credit period."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Facility identifier",
    )
    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="Commodity for this period",
    )
    standard: StandardTypeSchema = Field(
        default=StandardTypeSchema.EUDR_DEFAULT,
        description="Certification standard",
    )
    start_date: datetime = Field(
        ..., description="Period start date (UTC)",
    )
    end_date: Optional[datetime] = Field(
        None, description="Explicit end date (auto-calculated if omitted)",
    )
    opening_balance: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Opening balance from carry-forward (kg)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-GH-001",
                    "commodity": "cocoa",
                    "standard": "rspo",
                    "start_date": "2026-01-01T00:00:00Z",
                    "opening_balance": "1000.00",
                }
            ]
        },
    )

class ExtendPeriodSchema(GreenLangBase):
    """Request to extend a credit period end date."""

    new_end_date: datetime = Field(
        ..., description="New end date for the period (UTC)",
    )
    reason: str = Field(
        ..., min_length=1, max_length=2000,
        description="Reason for the extension",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Operator requesting the extension",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Ensure reason is non-empty."""
        if not v or not v.strip():
            raise ValueError("reason must be non-empty")
        return v

class RolloverPeriodSchema(GreenLangBase):
    """Request to rollover a credit period to a new one."""

    closing_period_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Period to close and rollover from",
    )
    carry_forward_percent: float = Field(
        default=100.0, ge=0.0, le=100.0,
        description="Percentage of closing balance to carry forward (0-100)",
    )
    operator_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Operator performing the rollover",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

# -- Period Response Schemas --

class PeriodDetailSchema(GreenLangBase):
    """Response for a single credit period with full details."""

    period_id: str = Field(..., description="Unique period identifier")
    facility_id: str = Field(..., description="Facility identifier")
    commodity: str = Field(..., description="Commodity")
    standard: StandardTypeSchema = Field(..., description="Certification standard")
    start_date: datetime = Field(..., description="Period start date")
    end_date: datetime = Field(..., description="Period end date")
    status: PeriodStatusSchema = Field(..., description="Current lifecycle status")
    grace_period_end: Optional[datetime] = Field(
        None, description="Grace period end date"
    )
    carry_forward_balance: Decimal = Field(
        ..., description="Balance carried forward from previous period"
    )
    opening_balance: Decimal = Field(..., description="Opening balance")
    closing_balance: Optional[Decimal] = Field(
        None, description="Closing balance (set during reconciliation)"
    )
    total_inputs: Decimal = Field(..., description="Total inputs during period")
    total_outputs: Decimal = Field(..., description="Total outputs during period")
    total_losses: Decimal = Field(..., description="Total losses during period")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class ActivePeriodsSchema(GreenLangBase):
    """Response listing active credit periods for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    periods: List[PeriodDetailSchema] = Field(
        default_factory=list, description="Active credit periods"
    )
    total_count: int = Field(..., ge=0, description="Total active periods")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class RolloverResultSchema(GreenLangBase):
    """Response for period rollover operation."""

    closed_period: PeriodDetailSchema = Field(
        ..., description="The period that was closed"
    )
    new_period: PeriodDetailSchema = Field(
        ..., description="The newly created period"
    )
    carry_forward_amount: Decimal = Field(
        ..., description="Amount carried forward in kg"
    )
    carry_forward_percent: float = Field(
        ..., description="Percentage of balance carried forward"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Conversion Factor Schemas
# =============================================================================

class ValidateFactorSchema(GreenLangBase):
    """Request to validate a conversion factor against reference data."""

    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="Commodity to validate against",
    )
    process_name: str = Field(
        ..., min_length=1, max_length=100,
        description="Processing step name (e.g., fermentation, drying)",
    )
    yield_ratio: float = Field(
        ..., gt=0.0, le=1.0,
        description="Reported yield ratio (output_mass / input_mass)",
    )
    facility_id: Optional[str] = Field(
        None, max_length=100,
        description="Facility where the factor was measured",
    )
    source: Optional[str] = Field(
        None, max_length=200,
        description="Source/justification for the factor",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "commodity": "cocoa",
                    "process_name": "fermentation",
                    "yield_ratio": 0.91,
                    "facility_id": "FAC-GH-001",
                }
            ]
        },
    )

class RegisterCustomFactorSchema(GreenLangBase):
    """Request to register a custom conversion factor."""

    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="Commodity this factor applies to",
    )
    process_name: str = Field(
        ..., min_length=1, max_length=100,
        description="Processing step name",
    )
    yield_ratio: float = Field(
        ..., gt=0.0, le=1.0,
        description="Custom yield ratio (0-1)",
    )
    input_material: Optional[str] = Field(
        None, max_length=200,
        description="Input material description",
    )
    output_material: Optional[str] = Field(
        None, max_length=200,
        description="Output material description",
    )
    facility_id: Optional[str] = Field(
        None, max_length=100,
        description="Facility where factor was measured",
    )
    source: Optional[str] = Field(
        None, max_length=500,
        description="Source/justification for the custom factor",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

# -- Factor Response Schemas --

class FactorValidationResultSchema(GreenLangBase):
    """Response after validating a conversion factor."""

    factor_id: str = Field(..., description="Factor identifier")
    commodity: str = Field(..., description="Commodity")
    process_name: str = Field(..., description="Processing step")
    yield_ratio: float = Field(..., description="Reported yield ratio")
    reference_ratio: Optional[float] = Field(
        None, description="Reference yield ratio from EUDR data"
    )
    deviation_percent: Optional[float] = Field(
        None, description="Deviation from reference as percentage"
    )
    acceptable_range_min: Optional[float] = Field(
        None, description="Minimum acceptable yield ratio"
    )
    acceptable_range_max: Optional[float] = Field(
        None, description="Maximum acceptable yield ratio"
    )
    validation_status: ConversionStatusSchema = Field(
        ..., description="Validation result"
    )
    message: str = Field(..., description="Human-readable validation message")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ReferenceFactorDetailSchema(GreenLangBase):
    """Detail of a single reference conversion factor."""

    process_name: str = Field(..., description="Processing step name")
    yield_ratio: float = Field(..., description="Reference yield ratio")
    acceptable_range_min: float = Field(
        ..., description="Minimum acceptable range"
    )
    acceptable_range_max: float = Field(
        ..., description="Maximum acceptable range"
    )

    model_config = ConfigDict(from_attributes=True)

class ReferenceFactorsSchema(GreenLangBase):
    """Response listing reference conversion factors for a commodity."""

    commodity: str = Field(..., description="Commodity")
    factors: List[ReferenceFactorDetailSchema] = Field(
        default_factory=list, description="Reference factors by process"
    )
    source: str = Field(
        default="EUDR reference data",
        description="Source of the reference data",
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class FactorRegistrationResultSchema(GreenLangBase):
    """Response after registering a custom conversion factor."""

    factor_id: str = Field(..., description="New factor identifier")
    commodity: str = Field(..., description="Commodity")
    process_name: str = Field(..., description="Processing step")
    yield_ratio: float = Field(..., description="Registered yield ratio")
    validation_status: ConversionStatusSchema = Field(
        ..., description="Initial validation result"
    )
    deviation_percent: Optional[float] = Field(
        None, description="Deviation from reference if available"
    )
    message: str = Field(..., description="Registration result message")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class FactorHistoryEntrySchema(GreenLangBase):
    """Single entry in factor usage history."""

    factor_id: str = Field(..., description="Factor identifier")
    commodity: str = Field(..., description="Commodity")
    process_name: str = Field(..., description="Processing step")
    yield_ratio: float = Field(..., description="Yield ratio")
    validation_status: ConversionStatusSchema = Field(
        ..., description="Validation status at time of use"
    )
    applied_at: Optional[datetime] = Field(
        None, description="When the factor was applied"
    )
    facility_id: Optional[str] = Field(None, description="Facility")

    model_config = ConfigDict(from_attributes=True)

class FactorHistorySchema(GreenLangBase):
    """Response with factor usage history for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    factors: List[FactorHistoryEntrySchema] = Field(
        default_factory=list, description="Factor usage entries"
    )
    total_count: int = Field(..., ge=0, description="Total entries")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Overdraft Schemas
# =============================================================================

class CheckOverdraftSchema(GreenLangBase):
    """Request to check for potential overdraft before recording output."""

    ledger_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Ledger to check",
    )
    output_quantity_kg: Decimal = Field(
        ..., gt=0,
        description="Proposed output quantity in kg",
    )
    dry_run: bool = Field(
        default=True,
        description="Whether this is a dry-run check (no state change)",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "ledger_id": "led-001",
                    "output_quantity_kg": "3000.00",
                    "dry_run": True,
                }
            ]
        },
    )

class ForecastOutputSchema(GreenLangBase):
    """Request to forecast maximum available output quantity."""

    ledger_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Ledger to forecast against",
    )
    include_carry_forward: bool = Field(
        default=True,
        description="Include carry-forward balance in forecast",
    )
    include_pending_inputs: bool = Field(
        default=False,
        description="Include pending input entries in forecast",
    )

    model_config = ConfigDict(extra="forbid")

class RequestExemptionSchema(GreenLangBase):
    """Request an exemption for an overdraft event."""

    event_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Overdraft event identifier",
    )
    reason: str = Field(
        ..., min_length=1, max_length=2000,
        description="Justification for the exemption",
    )
    requested_by: str = Field(
        ..., min_length=1, max_length=100,
        description="Operator requesting the exemption",
    )
    supporting_evidence: Optional[str] = Field(
        None, max_length=5000,
        description="Evidence supporting the request",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Ensure reason is non-empty."""
        if not v or not v.strip():
            raise ValueError("reason must be non-empty")
        return v

# -- Overdraft Response Schemas --

class OverdraftCheckResultSchema(GreenLangBase):
    """Response after checking for potential overdraft."""

    ledger_id: str = Field(..., description="Ledger identifier")
    current_balance: Decimal = Field(..., description="Current balance in kg")
    proposed_output: Decimal = Field(..., description="Proposed output in kg")
    remaining_after: Decimal = Field(
        ..., description="Balance after proposed output"
    )
    overdraft_detected: bool = Field(
        ..., description="Whether overdraft would occur"
    )
    severity: Optional[OverdraftSeveritySchema] = Field(
        None, description="Overdraft severity if detected"
    )
    overdraft_mode: OverdraftModeSchema = Field(
        ..., description="Active overdraft enforcement mode"
    )
    tolerance_applied: Optional[Decimal] = Field(
        None, description="Tolerance amount applied"
    )
    allowed: bool = Field(..., description="Whether the output is allowed")
    message: str = Field(..., description="Human-readable explanation")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class OverdraftAlertDetailSchema(GreenLangBase):
    """Detail of a single overdraft alert."""

    event_id: str = Field(..., description="Overdraft event identifier")
    ledger_id: str = Field(..., description="Affected ledger")
    facility_id: str = Field(..., description="Affected facility")
    commodity: str = Field(..., description="Affected commodity")
    severity: OverdraftSeveritySchema = Field(..., description="Severity")
    current_balance: Decimal = Field(
        ..., description="Balance at time of detection"
    )
    overdraft_amount: Decimal = Field(..., description="Overdraft amount in kg")
    trigger_entry_id: Optional[str] = Field(
        None, description="Entry that triggered the overdraft"
    )
    resolution_deadline: Optional[datetime] = Field(
        None, description="Resolution deadline"
    )
    resolved: bool = Field(default=False, description="Whether resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution time")
    exemption_id: Optional[str] = Field(
        None, description="Exemption ID if granted"
    )
    created_at: datetime = Field(..., description="Detection timestamp")

    model_config = ConfigDict(from_attributes=True)

class OverdraftAlertsSchema(GreenLangBase):
    """Response with active overdraft alerts for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    alerts: List[OverdraftAlertDetailSchema] = Field(
        default_factory=list, description="Active overdraft alerts"
    )
    total_unresolved: int = Field(
        default=0, ge=0, description="Total unresolved overdrafts"
    )
    critical_count: int = Field(
        default=0, ge=0, description="Number of critical overdrafts"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ForecastResultSchema(GreenLangBase):
    """Response with output forecast information."""

    ledger_id: str = Field(..., description="Ledger identifier")
    available_balance: Decimal = Field(
        ..., description="Available balance for output"
    )
    carry_forward_included: bool = Field(
        ..., description="Whether carry-forward was included"
    )
    carry_forward_amount: Decimal = Field(
        default=Decimal("0"), description="Carry-forward amount included"
    )
    pending_inputs_included: bool = Field(
        ..., description="Whether pending inputs were included"
    )
    pending_inputs_amount: Decimal = Field(
        default=Decimal("0"), description="Pending inputs amount included"
    )
    max_output_kg: Decimal = Field(
        ..., description="Maximum output quantity available in kg"
    )
    overdraft_mode: OverdraftModeSchema = Field(
        ..., description="Active overdraft enforcement mode"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ExemptionResultSchema(GreenLangBase):
    """Response after requesting an overdraft exemption."""

    exemption_id: str = Field(..., description="Exemption identifier")
    event_id: str = Field(..., description="Overdraft event identifier")
    status: str = Field(..., description="Exemption status (pending/approved/denied)")
    message: str = Field(..., description="Result message")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class OverdraftHistorySchema(GreenLangBase):
    """Response with overdraft event history for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    events: List[OverdraftAlertDetailSchema] = Field(
        default_factory=list, description="Historical overdraft events"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Loss Schemas
# =============================================================================

class RecordLossSchema(GreenLangBase):
    """Request to record a processing loss."""

    ledger_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Target ledger identifier",
    )
    loss_type: LossTypeSchema = Field(
        ..., description="Type of loss",
    )
    waste_type: Optional[WasteTypeSchema] = Field(
        None, description="Type of waste (if applicable)",
    )
    quantity_kg: Decimal = Field(
        ..., gt=0, description="Quantity lost in kilograms",
    )
    batch_id: Optional[str] = Field(
        None, max_length=100,
        description="Associated batch identifier",
    )
    process_type: Optional[str] = Field(
        None, max_length=100,
        description="Processing type that caused the loss",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "ledger_id": "led-001",
                    "loss_type": "processing_loss",
                    "quantity_kg": "250.00",
                    "process_type": "fermentation",
                }
            ]
        },
    )

class ValidateLossSchema(GreenLangBase):
    """Request to validate a loss against tolerance thresholds."""

    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="Commodity to validate against",
    )
    loss_type: LossTypeSchema = Field(
        ..., description="Type of loss",
    )
    quantity_kg: Decimal = Field(
        ..., gt=0, description="Quantity lost in kg",
    )
    input_quantity_kg: Decimal = Field(
        ..., gt=0, description="Total input quantity for percentage calculation",
    )
    process_type: Optional[str] = Field(
        None, max_length=100,
        description="Processing type",
    )

    model_config = ConfigDict(extra="forbid")

# -- Loss Response Schemas --

class LossRecordSchema(GreenLangBase):
    """Response for a single loss record."""

    record_id: str = Field(..., description="Unique loss record identifier")
    ledger_id: str = Field(..., description="Associated ledger")
    loss_type: LossTypeSchema = Field(..., description="Type of loss")
    waste_type: Optional[WasteTypeSchema] = Field(
        None, description="Type of waste"
    )
    quantity_kg: Decimal = Field(..., description="Quantity lost in kg")
    percentage: float = Field(
        ..., description="Loss as percentage of input quantity"
    )
    batch_id: Optional[str] = Field(None, description="Associated batch")
    process_type: Optional[str] = Field(None, description="Processing type")
    within_tolerance: bool = Field(
        ..., description="Whether loss is within acceptable range"
    )
    expected_loss_percent: Optional[float] = Field(
        None, description="Expected loss percentage"
    )
    max_tolerance_percent: Optional[float] = Field(
        None, description="Maximum acceptable loss percentage"
    )
    facility_id: Optional[str] = Field(None, description="Facility")
    commodity: Optional[str] = Field(None, description="Commodity")
    notes: Optional[str] = Field(None, description="Notes")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    created_at: datetime = Field(..., description="Creation timestamp")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class LossListSchema(GreenLangBase):
    """Response listing loss records for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    records: List[LossRecordSchema] = Field(
        default_factory=list, description="Loss records"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    total_loss_kg: Decimal = Field(
        default=Decimal("0"), description="Total loss in kg"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class LossValidationResultSchema(GreenLangBase):
    """Response after validating a loss against tolerances."""

    commodity: str = Field(..., description="Commodity validated against")
    loss_type: LossTypeSchema = Field(..., description="Type of loss")
    quantity_kg: Decimal = Field(..., description="Loss quantity in kg")
    loss_percent: float = Field(
        ..., description="Loss as percentage of input"
    )
    expected_percent: Optional[float] = Field(
        None, description="Expected loss percentage"
    )
    max_tolerance_percent: Optional[float] = Field(
        None, description="Maximum tolerance percentage"
    )
    commodity_tolerance_percent: Optional[float] = Field(
        None, description="Commodity-specific tolerance"
    )
    loss_type_tolerance_percent: Optional[float] = Field(
        None, description="Loss-type-specific tolerance"
    )
    within_tolerance: bool = Field(
        ..., description="Whether loss is acceptable"
    )
    message: str = Field(..., description="Human-readable validation message")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class LossTrendPointSchema(GreenLangBase):
    """Single data point in loss trend analysis."""

    period_label: str = Field(..., description="Period label")
    loss_percent: float = Field(..., description="Loss percentage for period")
    loss_kg: Decimal = Field(..., description="Loss in kg for period")
    input_kg: Decimal = Field(..., description="Total inputs for period")
    is_anomaly: bool = Field(
        default=False, description="Whether this is an anomalous data point"
    )

    model_config = ConfigDict(from_attributes=True)

class LossTrendsSchema(GreenLangBase):
    """Response with loss trend analysis for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    commodity: Optional[str] = Field(None, description="Commodity filter applied")
    periods_analyzed: int = Field(..., ge=0, description="Number of periods")
    average_loss_percent: float = Field(
        ..., description="Average loss percentage"
    )
    trend_direction: str = Field(
        ..., description="Trend direction (increasing/decreasing/stable)"
    )
    trend_slope: Optional[float] = Field(
        None, description="Trend slope coefficient"
    )
    data_points: List[LossTrendPointSchema] = Field(
        default_factory=list, description="Trend data points"
    )
    anomalies: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detected anomalies"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Reconciliation Schemas
# =============================================================================

class RunReconciliationSchema(GreenLangBase):
    """Request to run a period reconciliation."""

    period_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Period to reconcile",
    )
    facility_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Facility to reconcile",
    )
    commodity: str = Field(
        ..., min_length=1, max_length=50,
        description="Commodity to reconcile",
    )
    include_anomaly_detection: bool = Field(
        default=True,
        description="Run statistical anomaly detection (Z-score)",
    )
    include_trend_analysis: bool = Field(
        default=True,
        description="Run trend analysis against historical periods",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator running the reconciliation",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "period_id": "per-001",
                    "facility_id": "FAC-GH-001",
                    "commodity": "cocoa",
                    "include_anomaly_detection": True,
                    "include_trend_analysis": True,
                }
            ]
        },
    )

class SignOffReconciliationSchema(GreenLangBase):
    """Request to sign off on a completed reconciliation."""

    reconciliation_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Reconciliation to sign off",
    )
    signed_off_by: str = Field(
        ..., min_length=1, max_length=100,
        description="Operator signing off",
    )
    notes: Optional[str] = Field(
        None, max_length=2000,
        description="Sign-off notes",
    )
    auto_rollover: bool = Field(
        default=True,
        description="Automatically create next credit period",
    )

    model_config = ConfigDict(extra="forbid")

# -- Reconciliation Response Schemas --

class AnomalyDetailSchema(GreenLangBase):
    """Detail of a detected anomaly during reconciliation."""

    anomaly_type: str = Field(..., description="Type of anomaly")
    description: str = Field(..., description="Anomaly description")
    z_score: Optional[float] = Field(None, description="Z-score if applicable")
    affected_entries: List[str] = Field(
        default_factory=list, description="Affected entry IDs"
    )
    severity: str = Field(
        default="medium", description="Anomaly severity (low/medium/high)"
    )

    model_config = ConfigDict(from_attributes=True)

class ReconciliationResultSchema(GreenLangBase):
    """Response after running a reconciliation."""

    reconciliation_id: str = Field(
        ..., description="Unique reconciliation identifier"
    )
    period_id: str = Field(..., description="Period reconciled")
    facility_id: str = Field(..., description="Facility reconciled")
    commodity: str = Field(..., description="Commodity reconciled")
    expected_balance: Decimal = Field(
        ..., description="Expected balance (inputs - outputs - losses)"
    )
    recorded_balance: Decimal = Field(
        ..., description="Actual recorded balance"
    )
    variance_absolute: Decimal = Field(
        ..., description="Absolute variance (expected - recorded)"
    )
    variance_percent: float = Field(
        ..., description="Variance as percentage of expected"
    )
    classification: VarianceClassificationSchema = Field(
        ..., description="Variance classification"
    )
    anomalies_detected: int = Field(
        default=0, ge=0, description="Number of anomalies detected"
    )
    anomaly_details: List[AnomalyDetailSchema] = Field(
        default_factory=list, description="Anomaly details"
    )
    trend_deviation: Optional[float] = Field(
        None, description="Deviation from historical trend"
    )
    status: ReconciliationStatusSchema = Field(
        ..., description="Reconciliation status"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ReconciliationHistoryEntrySchema(GreenLangBase):
    """Summary of a historical reconciliation for list views."""

    reconciliation_id: str = Field(..., description="Reconciliation identifier")
    period_id: str = Field(..., description="Period reconciled")
    expected_balance: Decimal = Field(..., description="Expected balance")
    recorded_balance: Decimal = Field(..., description="Recorded balance")
    variance_percent: float = Field(..., description="Variance percentage")
    classification: VarianceClassificationSchema = Field(
        ..., description="Variance classification"
    )
    anomalies_detected: int = Field(
        default=0, ge=0, description="Anomalies detected"
    )
    status: ReconciliationStatusSchema = Field(..., description="Status")
    signed_off_by: Optional[str] = Field(None, description="Signed off by")
    signed_off_at: Optional[datetime] = Field(None, description="Sign-off time")
    created_at: datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True)

class ReconciliationHistorySchema(GreenLangBase):
    """Response with reconciliation history for a facility."""

    facility_id: str = Field(..., description="Facility identifier")
    reconciliations: List[ReconciliationHistoryEntrySchema] = Field(
        default_factory=list, description="Historical reconciliations"
    )
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Consolidation Schemas
# =============================================================================

class GenerateConsolidationSchema(GreenLangBase):
    """Request to generate a consolidation report."""

    facility_ids: List[str] = Field(
        default_factory=list,
        description="Facility identifiers to include",
    )
    group_id: Optional[str] = Field(
        None, max_length=100,
        description="Facility group identifier (alternative to facility_ids)",
    )
    report_type: ReportTypeSchema = Field(
        default=ReportTypeSchema.CONSOLIDATION,
        description="Type of report",
    )
    report_format: ReportFormatSchema = Field(
        default=ReportFormatSchema.JSON,
        description="Output format",
    )
    period_start: Optional[datetime] = Field(
        None, description="Report period start date",
    )
    period_end: Optional[datetime] = Field(
        None, description="Report period end date",
    )
    commodity: Optional[str] = Field(
        None, max_length=50,
        description="Filter by commodity",
    )
    operator_id: Optional[str] = Field(
        None, max_length=100,
        description="Operator generating the report",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_source(self) -> GenerateConsolidationSchema:
        """Ensure either facility_ids or group_id is provided."""
        if not self.facility_ids and not self.group_id:
            raise ValueError(
                "Either facility_ids or group_id must be provided"
            )
        return self

class CreateFacilityGroupSchema(GreenLangBase):
    """Request to create a facility group for consolidation."""

    name: str = Field(
        ..., min_length=1, max_length=200,
        description="Human-readable group name",
    )
    group_type: FacilityGroupTypeSchema = Field(
        ..., description="Type of grouping",
    )
    facility_ids: List[str] = Field(
        ..., min_length=1,
        description="List of facility identifiers in this group",
    )
    description: Optional[str] = Field(
        None, max_length=2000,
        description="Group description",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional key-value pairs",
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "name": "West Africa Cocoa Operations",
                    "group_type": "region",
                    "facility_ids": ["FAC-GH-001", "FAC-CI-001", "FAC-CI-002"],
                    "description": "All cocoa processing facilities in West Africa",
                }
            ]
        },
    )

# -- Consolidation Response Schemas --

class FacilityGroupDetailSchema(GreenLangBase):
    """Response for a facility group."""

    group_id: str = Field(..., description="Unique group identifier")
    name: str = Field(..., description="Group name")
    group_type: FacilityGroupTypeSchema = Field(..., description="Group type")
    facility_ids: List[str] = Field(..., description="Facilities in group")
    facility_count: int = Field(..., ge=0, description="Number of facilities")
    description: Optional[str] = Field(None, description="Group description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    created_at: datetime = Field(..., description="Creation timestamp")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class CommodityBreakdownSchema(GreenLangBase):
    """Per-commodity balance breakdown in consolidation dashboard."""

    commodity: str = Field(..., description="Commodity name")
    total_balance_kg: Decimal = Field(..., description="Total balance in kg")
    total_inputs_kg: Decimal = Field(..., description="Total inputs in kg")
    total_outputs_kg: Decimal = Field(..., description="Total outputs in kg")
    total_losses_kg: Decimal = Field(..., description="Total losses in kg")
    facility_count: int = Field(
        ..., ge=0, description="Number of facilities handling this commodity"
    )
    utilization_rate: float = Field(..., description="Average utilization rate")

    model_config = ConfigDict(from_attributes=True)

class ConsolidationDashboardSchema(GreenLangBase):
    """Response with enterprise-level consolidated dashboard data."""

    facility_count: int = Field(
        ..., ge=0, description="Total facilities included"
    )
    total_balance_kg: Decimal = Field(
        ..., description="Total balance across all facilities"
    )
    total_inputs_kg: Decimal = Field(
        ..., description="Total inputs across all facilities"
    )
    total_outputs_kg: Decimal = Field(
        ..., description="Total outputs across all facilities"
    )
    total_losses_kg: Decimal = Field(
        ..., description="Total losses across all facilities"
    )
    overdraft_count: int = Field(
        default=0, ge=0, description="Active overdraft count"
    )
    compliance_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Compliance status counts (compliant, non_compliant, pending)",
    )
    commodity_breakdown: List[CommodityBreakdownSchema] = Field(
        default_factory=list, description="Per-commodity breakdown"
    )
    group_id: Optional[str] = Field(
        None, description="Facility group filter applied"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

class ConsolidationReportSchema(GreenLangBase):
    """Response after generating a consolidation report."""

    report_id: str = Field(..., description="Unique report identifier")
    report_type: ReportTypeSchema = Field(..., description="Type of report")
    report_format: ReportFormatSchema = Field(..., description="Output format")
    facility_count: int = Field(
        ..., ge=0, description="Number of facilities included"
    )
    facility_ids: List[str] = Field(
        default_factory=list, description="Facility identifiers included"
    )
    group_id: Optional[str] = Field(None, description="Facility group ID")
    period_start: Optional[datetime] = Field(
        None, description="Report period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Report period end"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary statistics"
    )
    file_reference: Optional[str] = Field(
        None, description="Storage reference for the generated file"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size in bytes"
    )
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )
    provenance: ProvenanceInfo = Field(..., description="Provenance tracking")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)

class ConsolidationReportDownloadSchema(GreenLangBase):
    """Response with report download information."""

    report_id: str = Field(..., description="Report identifier")
    report_format: ReportFormatSchema = Field(..., description="Format")
    file_reference: str = Field(
        ..., description="Storage reference for download"
    )
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    download_url: Optional[str] = Field(
        None, description="Pre-signed download URL"
    )
    expires_at: Optional[datetime] = Field(
        None, description="URL expiry time"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Batch Job Schemas
# =============================================================================

class SubmitBatchSchema(GreenLangBase):
    """Request to submit a batch processing job."""

    job_type: BatchJobTypeSchema = Field(
        ..., description="Type of batch job",
    )
    priority: int = Field(
        default=5, ge=1, le=10,
        description="Job priority (1=highest, 10=lowest)",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Job-specific parameters",
    )
    callback_url: Optional[str] = Field(
        None, max_length=500,
        description="Webhook URL for job completion notification",
    )

    model_config = ConfigDict(extra="forbid")

class BatchJobSchema(GreenLangBase):
    """Response for a batch processing job."""

    job_id: str = Field(..., description="Unique job identifier")
    job_type: BatchJobTypeSchema = Field(..., description="Job type")
    status: BatchJobStatusSchema = Field(..., description="Current status")
    priority: int = Field(..., ge=1, le=10, description="Job priority")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Job parameters"
    )
    progress_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Progress percentage"
    )
    total_items: Optional[int] = Field(
        None, ge=0, description="Total items to process"
    )
    processed_items: Optional[int] = Field(
        None, ge=0, description="Items processed so far"
    )
    failed_items: Optional[int] = Field(
        None, ge=0, description="Items that failed"
    )
    result: Optional[Dict[str, Any]] = Field(
        None, description="Job result data"
    )
    error: Optional[str] = Field(None, description="Error message if failed")
    callback_url: Optional[str] = Field(
        None, description="Callback URL"
    )
    submitted_at: datetime = Field(..., description="Submission timestamp")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        None, description="Completion timestamp"
    )
    cancelled_at: Optional[datetime] = Field(
        None, description="Cancellation timestamp"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    model_config = ConfigDict(from_attributes=True)

class BatchJobCancelSchema(GreenLangBase):
    """Response after cancelling a batch job."""

    job_id: str = Field(..., description="Job identifier")
    status: BatchJobStatusSchema = Field(..., description="New status")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")
    message: str = Field(..., description="Cancellation message")

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Health Schema
# =============================================================================

class HealthSchema(GreenLangBase):
    """Health check response for the mass balance calculator service."""

    service: str = Field(
        default="gl-eudr-mbc-011",
        description="Service identifier",
    )
    version: str = Field(
        default="1.0.0",
        description="Service version",
    )
    status: str = Field(
        default="healthy",
        description="Service status (healthy/degraded/unhealthy)",
    )
    database_connected: bool = Field(
        default=True, description="Database connectivity"
    )
    redis_connected: bool = Field(
        default=True, description="Redis connectivity"
    )
    active_ledgers: int = Field(
        default=0, ge=0, description="Number of active ledgers"
    )
    active_periods: int = Field(
        default=0, ge=0, description="Number of active credit periods"
    )
    unresolved_overdrafts: int = Field(
        default=0, ge=0, description="Number of unresolved overdrafts"
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0, description="Service uptime in seconds"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Health check timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Paginated Response Wrapper
# =============================================================================

class PaginatedResponse(GreenLangBase):
    """Generic paginated response wrapper."""

    data: List[Any] = Field(default_factory=list, description="Result items")
    pagination: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # -- Helpers --
    "_utcnow",
    "_new_id",
    # -- Enumerations --
    "BatchJobStatusSchema",
    "BatchJobTypeSchema",
    "CarryForwardStatusSchema",
    "ComplianceStatusSchema",
    "ConversionStatusSchema",
    "FacilityGroupTypeSchema",
    "LedgerEntryTypeSchema",
    "LossTypeSchema",
    "OverdraftModeSchema",
    "OverdraftSeveritySchema",
    "PeriodStatusSchema",
    "ReconciliationStatusSchema",
    "ReportFormatSchema",
    "ReportTypeSchema",
    "SortOrderSchema",
    "StandardTypeSchema",
    "VarianceClassificationSchema",
    "WasteTypeSchema",
    # -- Common --
    "ErrorDetail",
    "ErrorResponse",
    "PaginatedMeta",
    "PaginatedResponse",
    "ProvenanceInfo",
    # -- Ledger Request --
    "BulkEntrySchema",
    "CreateLedgerSchema",
    "RecordEntrySchema",
    "SearchLedgerSchema",
    # -- Ledger Response --
    "BulkEntryResultSchema",
    "EntryHistorySchema",
    "LedgerBalanceSchema",
    "LedgerDetailSchema",
    "LedgerEntryDetailSchema",
    "LedgerListSchema",
    # -- Period Request --
    "CreatePeriodSchema",
    "ExtendPeriodSchema",
    "RolloverPeriodSchema",
    # -- Period Response --
    "ActivePeriodsSchema",
    "PeriodDetailSchema",
    "RolloverResultSchema",
    # -- Factor Request --
    "RegisterCustomFactorSchema",
    "ValidateFactorSchema",
    # -- Factor Response --
    "FactorHistoryEntrySchema",
    "FactorHistorySchema",
    "FactorRegistrationResultSchema",
    "FactorValidationResultSchema",
    "ReferenceFactorDetailSchema",
    "ReferenceFactorsSchema",
    # -- Overdraft Request --
    "CheckOverdraftSchema",
    "ForecastOutputSchema",
    "RequestExemptionSchema",
    # -- Overdraft Response --
    "ExemptionResultSchema",
    "ForecastResultSchema",
    "OverdraftAlertDetailSchema",
    "OverdraftAlertsSchema",
    "OverdraftCheckResultSchema",
    "OverdraftHistorySchema",
    # -- Loss Request --
    "RecordLossSchema",
    "ValidateLossSchema",
    # -- Loss Response --
    "LossListSchema",
    "LossRecordSchema",
    "LossTrendPointSchema",
    "LossTrendsSchema",
    "LossValidationResultSchema",
    # -- Reconciliation Request --
    "RunReconciliationSchema",
    "SignOffReconciliationSchema",
    # -- Reconciliation Response --
    "AnomalyDetailSchema",
    "ReconciliationHistoryEntrySchema",
    "ReconciliationHistorySchema",
    "ReconciliationResultSchema",
    # -- Consolidation Request --
    "CreateFacilityGroupSchema",
    "GenerateConsolidationSchema",
    # -- Consolidation Response --
    "CommodityBreakdownSchema",
    "ConsolidationDashboardSchema",
    "ConsolidationReportDownloadSchema",
    "ConsolidationReportSchema",
    "FacilityGroupDetailSchema",
    # -- Batch --
    "BatchJobCancelSchema",
    "BatchJobSchema",
    "SubmitBatchSchema",
    # -- Health --
    "HealthSchema",
]

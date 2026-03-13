# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-009 Chain of Custody

Pydantic v2 request/response models for all Chain of Custody REST API
endpoints. Organized by domain: custody events, batches, CoC model
assignment, mass balance, transformations, documents, verification,
reports, batch jobs, and health.

All models use ``ConfigDict(from_attributes=True)`` for ORM compatibility
and ``Field()`` with descriptions for OpenAPI documentation.

Model Count: 90+ schemas covering 35 endpoints.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-009, Section 7.4
Agent ID: GL-EUDR-COC-009
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


class CustodyEventType(str, Enum):
    """Types of chain-of-custody events."""

    RECEIPT = "receipt"
    TRANSFER = "transfer"
    TRANSFORMATION = "transformation"
    STORAGE = "storage"
    DISPATCH = "dispatch"
    INSPECTION = "inspection"
    CERTIFICATION = "certification"
    CUSTOMS_CLEARANCE = "customs_clearance"
    QUALITY_CHECK = "quality_check"
    REJECTION = "rejection"


class CustodyModelType(str, Enum):
    """ISO 22095 chain-of-custody models."""

    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"
    CONTROLLED_BLENDING = "controlled_blending"


class BatchStatus(str, Enum):
    """Lifecycle status of a batch."""

    CREATED = "created"
    ACTIVE = "active"
    IN_TRANSIT = "in_transit"
    RECEIVED = "received"
    PROCESSING = "processing"
    SPLIT = "split"
    MERGED = "merged"
    BLENDED = "blended"
    CONSUMED = "consumed"
    EXPIRED = "expired"
    REJECTED = "rejected"


class BatchOriginType(str, Enum):
    """How a batch was originated."""

    HARVEST = "harvest"
    PURCHASE = "purchase"
    SPLIT = "split"
    MERGE = "merge"
    BLEND = "blend"
    TRANSFORMATION = "transformation"
    IMPORT = "import"


class EUDRCommodity(str, Enum):
    """EUDR-regulated commodities."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class BalanceEntryType(str, Enum):
    """Type of mass balance ledger entry."""

    INPUT = "input"
    OUTPUT = "output"
    ADJUSTMENT = "adjustment"
    LOSS = "loss"
    WASTE = "waste"


class ReconciliationStatus(str, Enum):
    """Outcome of a balance reconciliation."""

    BALANCED = "balanced"
    SURPLUS = "surplus"
    DEFICIT = "deficit"
    PENDING = "pending"
    FAILED = "failed"


class TransformationType(str, Enum):
    """Types of product transformations."""

    PROCESSING = "processing"
    REFINING = "refining"
    ROASTING = "roasting"
    FERMENTATION = "fermentation"
    DRYING = "drying"
    MILLING = "milling"
    PRESSING = "pressing"
    BLENDING = "blending"
    PACKAGING = "packaging"
    CUTTING = "cutting"


class DocumentType(str, Enum):
    """Types of supporting documents."""

    BILL_OF_LADING = "bill_of_lading"
    PHYTOSANITARY_CERTIFICATE = "phytosanitary_certificate"
    CERTIFICATE_OF_ORIGIN = "certificate_of_origin"
    CUSTOMS_DECLARATION = "customs_declaration"
    WEIGHT_CERTIFICATE = "weight_certificate"
    QUALITY_CERTIFICATE = "quality_certificate"
    SUSTAINABILITY_CERTIFICATE = "sustainability_certificate"
    INVOICE = "invoice"
    PACKING_LIST = "packing_list"
    TRANSPORT_DOCUMENT = "transport_document"
    GPS_LOG = "gps_log"
    PHOTO_EVIDENCE = "photo_evidence"


class VerificationStatus(str, Enum):
    """Outcome of a custody chain verification."""

    VERIFIED = "verified"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"
    EXPIRED = "expired"


class VerificationSeverity(str, Enum):
    """Severity of verification findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ReportType(str, Enum):
    """Types of chain-of-custody reports."""

    TRACEABILITY = "traceability"
    MASS_BALANCE = "mass_balance"
    COMPLIANCE = "compliance"
    AUDIT = "audit"


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
    """Types of batch jobs."""

    EVENT_IMPORT = "event_import"
    BATCH_VERIFICATION = "batch_verification"
    BALANCE_RECONCILIATION = "balance_reconciliation"
    REPORT_GENERATION = "report_generation"
    DOCUMENT_VALIDATION = "document_validation"


class ComplianceLevel(str, Enum):
    """Compliance level assessment."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_ASSESSED = "not_assessed"


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


# =============================================================================
# Event Schemas
# =============================================================================


class CustodyEventCreateRequest(BaseModel):
    """Request to record a custody event."""

    event_type: CustodyEventType = Field(
        ..., description="Type of custody event"
    )
    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch/lot identifier"
    )
    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility where event occurs"
    )
    commodity: EUDRCommodity = Field(
        ..., description="EUDR commodity involved"
    )
    quantity: QuantitySpec = Field(
        ..., description="Quantity involved in event"
    )
    timestamp: Optional[datetime] = Field(
        None, description="Event timestamp (defaults to now if omitted)"
    )
    source_facility_id: Optional[str] = Field(
        None, max_length=100, description="Source facility for transfers"
    )
    destination_facility_id: Optional[str] = Field(
        None, max_length=100, description="Destination facility for transfers"
    )
    transport_mode: Optional[str] = Field(
        None, max_length=50, description="Transport mode (truck, ship, rail)"
    )
    transport_document_ref: Optional[str] = Field(
        None, max_length=200, description="Transport document reference"
    )
    location: Optional[GeoCoordinate] = Field(
        None, description="GPS coordinates of event location"
    )
    custody_model: Optional[CustodyModelType] = Field(
        None, description="Applicable CoC model"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional key-value metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "event_type": "receipt",
                    "batch_id": "BATCH-2026-001",
                    "facility_id": "FAC-GH-001",
                    "commodity": "cocoa",
                    "quantity": {"amount": "5000.00", "unit": "kg"},
                    "source_facility_id": "FAC-GH-COOP-001",
                    "custody_model": "segregated",
                }
            ]
        },
    )


class CustodyEventBatchRequest(BaseModel):
    """Request for bulk custody event import."""

    events: List[CustodyEventCreateRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of events to import (max 1000)",
    )
    validate_only: bool = Field(
        default=False,
        description="If true, validate without persisting",
    )

    model_config = ConfigDict(extra="forbid")


class CustodyEventAmendRequest(BaseModel):
    """Request to amend an existing custody event (immutable append)."""

    reason: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Reason for amendment",
    )
    corrected_quantity: Optional[QuantitySpec] = Field(
        None, description="Corrected quantity if applicable"
    )
    corrected_timestamp: Optional[datetime] = Field(
        None, description="Corrected event timestamp"
    )
    corrected_facility_id: Optional[str] = Field(
        None, max_length=100, description="Corrected facility ID"
    )
    corrected_notes: Optional[str] = Field(
        None, max_length=2000, description="Corrected notes"
    )
    corrected_metadata: Optional[Dict[str, Any]] = Field(
        None, description="Corrected metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "reason": "Weight correction after re-measurement",
                    "corrected_quantity": {"amount": "4980.50", "unit": "kg"},
                }
            ]
        },
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, v: str) -> str:
        """Ensure amendment reason is non-empty."""
        if not v or not v.strip():
            raise ValueError("reason must be non-empty")
        return v


class CustodyEventResponse(BaseModel):
    """Response for a single custody event."""

    event_id: str = Field(..., description="Unique event identifier")
    event_type: CustodyEventType = Field(..., description="Type of event")
    batch_id: str = Field(..., description="Associated batch ID")
    facility_id: str = Field(..., description="Facility ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: QuantitySpec = Field(..., description="Event quantity")
    timestamp: datetime = Field(..., description="Event timestamp")
    source_facility_id: Optional[str] = Field(None)
    destination_facility_id: Optional[str] = Field(None)
    transport_mode: Optional[str] = Field(None)
    transport_document_ref: Optional[str] = Field(None)
    location: Optional[GeoCoordinate] = Field(None)
    custody_model: Optional[CustodyModelType] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Optional[Dict[str, Any]] = Field(None)
    is_amendment: bool = Field(
        default=False, description="Whether this is an amendment"
    )
    amends_event_id: Optional[str] = Field(
        None, description="Original event ID if amendment"
    )
    amendment_reason: Optional[str] = Field(
        None, description="Reason for amendment"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)


class CustodyEventBatchResponse(BaseModel):
    """Response for bulk custody event import."""

    total_submitted: int = Field(..., ge=0, description="Total events submitted")
    total_accepted: int = Field(..., ge=0, description="Events accepted")
    total_rejected: int = Field(..., ge=0, description="Events rejected")
    events: List[CustodyEventResponse] = Field(
        default_factory=list, description="Accepted events"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Rejection details"
    )
    validate_only: bool = Field(
        default=False, description="Whether this was validation-only"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of batch import"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Total processing time in ms"
    )

    model_config = ConfigDict(from_attributes=True)


class CustodyEventChainResponse(BaseModel):
    """Response for event chain query on a batch."""

    batch_id: str = Field(..., description="Batch ID queried")
    events: List[CustodyEventResponse] = Field(
        default_factory=list, description="Ordered chain of events"
    )
    total_events: int = Field(default=0, ge=0, description="Total events in chain")
    chain_start: Optional[datetime] = Field(
        None, description="Earliest event timestamp"
    )
    chain_end: Optional[datetime] = Field(
        None, description="Latest event timestamp"
    )
    chain_duration_hours: Optional[float] = Field(
        None, ge=0.0, description="Chain duration in hours"
    )
    custody_models_used: List[CustodyModelType] = Field(
        default_factory=list, description="CoC models applied in chain"
    )
    facilities_involved: List[str] = Field(
        default_factory=list, description="Facility IDs in chain"
    )
    has_amendments: bool = Field(
        default=False, description="Whether chain contains amendments"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of chain data"
    )

    model_config = ConfigDict(from_attributes=True)


class CustodyEventAmendResponse(BaseModel):
    """Response after amending a custody event."""

    amendment_event_id: str = Field(
        ..., description="New event ID for the amendment"
    )
    original_event_id: str = Field(
        ..., description="Original event that was amended"
    )
    reason: str = Field(..., description="Amendment reason")
    status: str = Field(default="amended", description="Amendment status")
    amended_at: datetime = Field(
        default_factory=_utcnow, description="Amendment timestamp"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(
        default=0.0, ge=0.0, description="Processing duration in ms"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Schemas
# =============================================================================


class BatchCreateRequest(BaseModel):
    """Request to create a new batch/lot."""

    batch_reference: Optional[str] = Field(
        None, max_length=200, description="External batch reference"
    )
    commodity: EUDRCommodity = Field(
        ..., description="EUDR commodity for this batch"
    )
    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    origin_type: BatchOriginType = Field(
        ..., description="How the batch originated"
    )
    quantity: QuantitySpec = Field(
        ..., description="Initial batch quantity"
    )
    origin_country: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2 origin country"
    )
    origin_region: Optional[str] = Field(
        None, max_length=200, description="Origin region/province"
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list, description="Source production plot IDs"
    )
    harvest_date: Optional[datetime] = Field(
        None, description="Harvest or production date"
    )
    custody_model: CustodyModelType = Field(
        ..., description="Chain-of-custody model for this batch"
    )
    certifications: List[str] = Field(
        default_factory=list, description="Applicable certifications"
    )
    supplier_id: Optional[str] = Field(
        None, max_length=100, description="Supplier/producer ID"
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
                    "commodity": "cocoa",
                    "facility_id": "FAC-GH-001",
                    "origin_type": "harvest",
                    "quantity": {"amount": "10000.00", "unit": "kg"},
                    "origin_country": "GH",
                    "origin_region": "Ashanti",
                    "origin_plot_ids": ["PLOT-GH-001", "PLOT-GH-002"],
                    "custody_model": "segregated",
                }
            ]
        },
    )

    @field_validator("origin_country")
    @classmethod
    def validate_origin_country(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "origin_country must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v


class BatchResponse(BaseModel):
    """Response for a single batch."""

    batch_id: str = Field(..., description="Unique batch identifier")
    batch_reference: Optional[str] = Field(None, description="External reference")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    facility_id: str = Field(..., description="Current facility ID")
    status: BatchStatus = Field(..., description="Batch lifecycle status")
    origin_type: BatchOriginType = Field(..., description="Origin type")
    quantity: QuantitySpec = Field(..., description="Current quantity")
    original_quantity: QuantitySpec = Field(
        ..., description="Original quantity at creation"
    )
    origin_country: str = Field(..., description="Origin country code")
    origin_region: Optional[str] = Field(None)
    origin_plot_ids: List[str] = Field(default_factory=list)
    harvest_date: Optional[datetime] = Field(None)
    custody_model: CustodyModelType = Field(..., description="CoC model")
    certifications: List[str] = Field(default_factory=list)
    supplier_id: Optional[str] = Field(None)
    parent_batch_ids: List[str] = Field(
        default_factory=list, description="Parent batch IDs (from split/merge)"
    )
    child_batch_ids: List[str] = Field(
        default_factory=list, description="Child batch IDs"
    )
    event_count: int = Field(default=0, ge=0, description="Number of events")
    notes: Optional[str] = Field(None)
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


class BatchSplitPart(BaseModel):
    """One part of a batch split allocation."""

    quantity: QuantitySpec = Field(
        ..., description="Quantity for this split part"
    )
    destination_facility_id: Optional[str] = Field(
        None, max_length=100, description="Destination facility"
    )
    batch_reference: Optional[str] = Field(
        None, max_length=200, description="Reference for new child batch"
    )
    notes: Optional[str] = Field(None, max_length=1000)

    model_config = ConfigDict(from_attributes=True)


class BatchSplitRequest(BaseModel):
    """Request to split a batch into multiple child batches."""

    source_batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch to split"
    )
    splits: List[BatchSplitPart] = Field(
        ...,
        min_length=2,
        max_length=50,
        description="Split allocations (must sum to source quantity)",
    )
    reason: str = Field(
        ..., min_length=1, max_length=2000, description="Reason for split"
    )

    model_config = ConfigDict(extra="forbid")


class BatchSplitResponse(BaseModel):
    """Response after splitting a batch."""

    source_batch_id: str = Field(..., description="Source batch that was split")
    child_batches: List[BatchResponse] = Field(
        default_factory=list, description="Newly created child batches"
    )
    total_splits: int = Field(default=0, ge=0)
    source_status: BatchStatus = Field(
        default=BatchStatus.SPLIT, description="Updated source batch status"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BatchMergeRequest(BaseModel):
    """Request to merge multiple batches into one."""

    source_batch_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Batch IDs to merge (min 2)",
    )
    destination_facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Destination facility"
    )
    batch_reference: Optional[str] = Field(
        None, max_length=200, description="Reference for merged batch"
    )
    reason: str = Field(
        ..., min_length=1, max_length=2000, description="Reason for merge"
    )
    notes: Optional[str] = Field(None, max_length=2000)

    model_config = ConfigDict(extra="forbid")


class BatchMergeResponse(BaseModel):
    """Response after merging batches."""

    merged_batch: BatchResponse = Field(
        ..., description="Newly created merged batch"
    )
    source_batch_ids: List[str] = Field(
        default_factory=list, description="Source batches that were merged"
    )
    total_quantity: QuantitySpec = Field(
        ..., description="Combined quantity"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BatchBlendRequest(BaseModel):
    """Request to blend batches (controlled blending model)."""

    source_batch_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Batch IDs to blend (min 2)",
    )
    blend_ratios: List[Decimal] = Field(
        ...,
        min_length=2,
        description="Blend ratios for each source (must sum to 1.0)",
    )
    destination_facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Blending facility"
    )
    total_output_quantity: QuantitySpec = Field(
        ..., description="Total output quantity after blending"
    )
    batch_reference: Optional[str] = Field(
        None, max_length=200, description="Reference for blended batch"
    )
    reason: str = Field(
        ..., min_length=1, max_length=2000, description="Reason for blending"
    )
    notes: Optional[str] = Field(None, max_length=2000)

    model_config = ConfigDict(extra="forbid")

    @field_validator("blend_ratios")
    @classmethod
    def validate_blend_ratios(cls, v: List[Decimal]) -> List[Decimal]:
        """Validate blend ratios sum to 1.0."""
        ratio_sum = sum(v)
        if abs(float(ratio_sum) - 1.0) > 0.001:
            raise ValueError(
                f"blend_ratios must sum to 1.0, got {float(ratio_sum):.4f}"
            )
        for ratio in v:
            if ratio <= 0:
                raise ValueError("All blend ratios must be positive")
        return v


class BatchBlendResponse(BaseModel):
    """Response after blending batches."""

    blended_batch: BatchResponse = Field(
        ..., description="Newly created blended batch"
    )
    source_batch_ids: List[str] = Field(default_factory=list)
    blend_ratios: List[Decimal] = Field(default_factory=list)
    total_output_quantity: QuantitySpec = Field(
        ..., description="Output quantity"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class GenealogyNode(BaseModel):
    """Node in a batch genealogy tree."""

    batch_id: str = Field(..., description="Batch identifier")
    batch_reference: Optional[str] = Field(None)
    status: BatchStatus = Field(..., description="Batch status")
    origin_type: BatchOriginType = Field(..., description="Origin type")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: QuantitySpec = Field(..., description="Quantity")
    custody_model: CustodyModelType = Field(..., description="CoC model")
    facility_id: str = Field(..., description="Current facility")
    origin_country: str = Field(..., description="Origin country")
    depth: int = Field(default=0, ge=0, description="Tree depth")
    parent_batch_ids: List[str] = Field(default_factory=list)
    child_batch_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


class BatchGenealogyResponse(BaseModel):
    """Response for batch genealogy tree."""

    batch_id: str = Field(..., description="Root batch ID")
    nodes: List[GenealogyNode] = Field(
        default_factory=list, description="Genealogy tree nodes"
    )
    total_nodes: int = Field(default=0, ge=0)
    max_depth: int = Field(default=0, ge=0, description="Maximum tree depth")
    origin_countries: List[str] = Field(
        default_factory=list, description="All origin countries in tree"
    )
    origin_plot_ids: List[str] = Field(
        default_factory=list, description="All origin plot IDs in tree"
    )
    custody_models_used: List[CustodyModelType] = Field(
        default_factory=list, description="All CoC models used"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BatchSearchRequest(BaseModel):
    """Request to search batches with filters."""

    commodity: Optional[EUDRCommodity] = Field(
        None, description="Filter by commodity"
    )
    facility_id: Optional[str] = Field(
        None, max_length=100, description="Filter by facility"
    )
    status: Optional[BatchStatus] = Field(
        None, description="Filter by batch status"
    )
    custody_model: Optional[CustodyModelType] = Field(
        None, description="Filter by CoC model"
    )
    origin_country: Optional[str] = Field(
        None, min_length=2, max_length=2, description="Filter by origin country"
    )
    supplier_id: Optional[str] = Field(
        None, max_length=100, description="Filter by supplier"
    )
    date_from: Optional[datetime] = Field(
        None, description="Filter by creation date (from)"
    )
    date_to: Optional[datetime] = Field(
        None, description="Filter by creation date (to)"
    )
    batch_reference: Optional[str] = Field(
        None, max_length=200, description="Partial match on batch reference"
    )
    min_quantity: Optional[Decimal] = Field(
        None, gt=0, description="Minimum quantity filter"
    )
    max_quantity: Optional[Decimal] = Field(
        None, gt=0, description="Maximum quantity filter"
    )
    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Results to skip")
    sort_by: str = Field(
        default="created_at",
        description="Sort field (created_at, quantity, batch_id)",
    )
    sort_order: str = Field(
        default="desc", description="Sort order (asc, desc)"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, v: str) -> str:
        """Validate sort field."""
        allowed = {"created_at", "quantity", "batch_id", "status", "commodity"}
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


class BatchSearchResponse(BaseModel):
    """Response for batch search results."""

    batches: List[BatchResponse] = Field(
        default_factory=list, description="Matching batches"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# CoC Model Schemas
# =============================================================================


class ModelAssignRequest(BaseModel):
    """Request to assign a CoC model to a facility."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    model_type: CustodyModelType = Field(
        ..., description="CoC model to assign (IP/SG/MB/CB)"
    )
    commodity: EUDRCommodity = Field(
        ..., description="Commodity this assignment covers"
    )
    effective_date: Optional[datetime] = Field(
        None, description="When the model becomes effective"
    )
    certification_id: Optional[str] = Field(
        None, max_length=200, description="Supporting certification ID"
    )
    auditor_name: Optional[str] = Field(
        None, max_length=200, description="Auditor or certifying body name"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Assignment notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "facility_id": "FAC-GH-001",
                    "model_type": "segregated",
                    "commodity": "cocoa",
                    "certification_id": "CERT-RA-2026-001",
                }
            ]
        },
    )


class ModelAssignResponse(BaseModel):
    """Response after assigning a CoC model."""

    assignment_id: str = Field(..., description="Unique assignment ID")
    facility_id: str = Field(..., description="Facility ID")
    model_type: CustodyModelType = Field(..., description="Assigned CoC model")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    effective_date: datetime = Field(..., description="Effective date")
    status: str = Field(default="active", description="Assignment status")
    certification_id: Optional[str] = Field(None)
    auditor_name: Optional[str] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class FacilityModelResponse(BaseModel):
    """Response for facility CoC model details."""

    facility_id: str = Field(..., description="Facility ID")
    assignments: List[ModelAssignResponse] = Field(
        default_factory=list, description="Active model assignments"
    )
    active_models: List[CustodyModelType] = Field(
        default_factory=list, description="Currently active CoC models"
    )
    commodities_covered: List[EUDRCommodity] = Field(
        default_factory=list, description="Commodities covered by assignments"
    )
    total_assignments: int = Field(default=0, ge=0)

    model_config = ConfigDict(from_attributes=True)


class ModelValidateRequest(BaseModel):
    """Request to validate an operation against CoC model rules."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    operation_type: str = Field(
        ..., min_length=1, max_length=50,
        description="Operation type (receipt, transfer, blend, split)",
    )
    batch_ids: List[str] = Field(
        ..., min_length=1, description="Batch IDs involved"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: Optional[QuantitySpec] = Field(
        None, description="Quantity involved"
    )

    model_config = ConfigDict(extra="forbid")


class ModelValidationFinding(BaseModel):
    """Individual validation finding."""

    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    severity: VerificationSeverity = Field(
        ..., description="Finding severity"
    )
    passed: bool = Field(..., description="Whether the rule passed")
    message: str = Field(..., description="Finding description")
    details: Optional[Dict[str, Any]] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class ModelValidateResponse(BaseModel):
    """Response from CoC model validation."""

    facility_id: str = Field(..., description="Facility ID")
    model_type: CustodyModelType = Field(
        ..., description="CoC model validated against"
    )
    operation_type: str = Field(..., description="Operation type")
    is_valid: bool = Field(
        ..., description="Overall validation result"
    )
    findings: List[ModelValidationFinding] = Field(
        default_factory=list, description="Validation findings"
    )
    total_rules_checked: int = Field(default=0, ge=0)
    rules_passed: int = Field(default=0, ge=0)
    rules_failed: int = Field(default=0, ge=0)
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Compliance score (0-100)",
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class ModelComplianceResponse(BaseModel):
    """Response for facility CoC model compliance score."""

    facility_id: str = Field(..., description="Facility ID")
    model_type: CustodyModelType = Field(
        ..., description="CoC model assessed"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    compliance_level: ComplianceLevel = Field(
        ..., description="Overall compliance level"
    )
    compliance_score: float = Field(
        ..., ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_rules: int = Field(default=0, ge=0)
    rules_met: int = Field(default=0, ge=0)
    rules_not_met: int = Field(default=0, ge=0)
    critical_findings: List[ModelValidationFinding] = Field(
        default_factory=list, description="Critical non-compliance findings"
    )
    last_assessment_at: datetime = Field(
        default_factory=_utcnow, description="Last assessment timestamp"
    )
    next_assessment_due: Optional[datetime] = Field(
        None, description="Next assessment due date"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Mass Balance Schemas
# =============================================================================


class BalanceInputRequest(BaseModel):
    """Request to record a mass balance input."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch being inputted"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: QuantitySpec = Field(
        ..., description="Input quantity"
    )
    entry_type: BalanceEntryType = Field(
        default=BalanceEntryType.INPUT, description="Ledger entry type"
    )
    source_facility_id: Optional[str] = Field(
        None, max_length=100, description="Source facility"
    )
    is_certified: bool = Field(
        default=False, description="Whether input is certified material"
    )
    certification_id: Optional[str] = Field(
        None, max_length=200, description="Certification reference"
    )
    period_start: Optional[datetime] = Field(
        None, description="Accounting period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Accounting period end"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )

    model_config = ConfigDict(extra="forbid")


class BalanceOutputRequest(BaseModel):
    """Request to record a mass balance output."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Output batch ID"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    quantity: QuantitySpec = Field(
        ..., description="Output quantity"
    )
    entry_type: BalanceEntryType = Field(
        default=BalanceEntryType.OUTPUT, description="Ledger entry type"
    )
    destination_facility_id: Optional[str] = Field(
        None, max_length=100, description="Destination facility"
    )
    is_certified_claim: bool = Field(
        default=False,
        description="Whether output carries a certified claim",
    )
    conversion_factor: Optional[Decimal] = Field(
        None, gt=0, le=10,
        description="Input-to-output conversion factor",
    )
    loss_quantity: Optional[QuantitySpec] = Field(
        None, description="Processing loss quantity"
    )
    period_start: Optional[datetime] = Field(
        None, description="Accounting period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Accounting period end"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )

    model_config = ConfigDict(extra="forbid")


class BalanceEntryResponse(BaseModel):
    """Response for a single balance entry."""

    entry_id: str = Field(..., description="Unique entry ID")
    facility_id: str = Field(..., description="Facility ID")
    batch_id: str = Field(..., description="Batch ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    entry_type: BalanceEntryType = Field(..., description="Entry type")
    quantity: QuantitySpec = Field(..., description="Entry quantity")
    is_certified: bool = Field(default=False)
    certification_id: Optional[str] = Field(None)
    source_facility_id: Optional[str] = Field(None)
    destination_facility_id: Optional[str] = Field(None)
    conversion_factor: Optional[Decimal] = Field(None)
    loss_quantity: Optional[QuantitySpec] = Field(None)
    period_start: Optional[datetime] = Field(None)
    period_end: Optional[datetime] = Field(None)
    notes: Optional[str] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    created_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class FacilityBalanceResponse(BaseModel):
    """Response for current facility mass balance."""

    facility_id: str = Field(..., description="Facility ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    total_input: QuantitySpec = Field(
        ..., description="Total input quantity"
    )
    total_output: QuantitySpec = Field(
        ..., description="Total output quantity"
    )
    total_loss: QuantitySpec = Field(
        ..., description="Total processing loss"
    )
    current_balance: QuantitySpec = Field(
        ..., description="Current balance (input - output - loss)"
    )
    certified_input: QuantitySpec = Field(
        ..., description="Total certified input"
    )
    certified_output: QuantitySpec = Field(
        ..., description="Total certified output"
    )
    certified_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Ratio of certified material (0-1)",
    )
    entry_count: int = Field(default=0, ge=0, description="Total entries")
    period_start: Optional[datetime] = Field(
        None, description="Accounting period start"
    )
    period_end: Optional[datetime] = Field(
        None, description="Accounting period end"
    )
    last_updated: datetime = Field(default_factory=_utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BalanceReconcileRequest(BaseModel):
    """Request to reconcile mass balance for a period."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    period_start: datetime = Field(
        ..., description="Reconciliation period start"
    )
    period_end: datetime = Field(
        ..., description="Reconciliation period end"
    )
    expected_loss_rate: Optional[Decimal] = Field(
        None, ge=0, le=1,
        description="Expected processing loss rate (0-1)",
    )
    tolerance_percent: Decimal = Field(
        default=Decimal("0.05"), ge=0, le=1,
        description="Acceptable variance tolerance (0-1)",
    )
    include_adjustments: bool = Field(
        default=True, description="Include prior adjustments"
    )

    model_config = ConfigDict(extra="forbid")


class BalanceReconcileResponse(BaseModel):
    """Response from balance reconciliation."""

    reconciliation_id: str = Field(
        ..., description="Unique reconciliation ID"
    )
    facility_id: str = Field(..., description="Facility ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    status: ReconciliationStatus = Field(
        ..., description="Reconciliation outcome"
    )
    total_input: QuantitySpec = Field(
        ..., description="Total input in period"
    )
    total_output: QuantitySpec = Field(
        ..., description="Total output in period"
    )
    total_loss: QuantitySpec = Field(
        ..., description="Total loss in period"
    )
    variance: QuantitySpec = Field(
        ..., description="Variance (input - output - loss)"
    )
    variance_percent: Decimal = Field(
        ..., description="Variance as percentage"
    )
    within_tolerance: bool = Field(
        ..., description="Whether variance is within tolerance"
    )
    certified_ratio: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Certified ratio"
    )
    findings: List[str] = Field(
        default_factory=list, description="Reconciliation findings"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BalanceHistoryEntry(BaseModel):
    """Single entry in balance history."""

    entry_id: str = Field(..., description="Entry ID")
    entry_type: BalanceEntryType = Field(..., description="Entry type")
    batch_id: str = Field(..., description="Batch ID")
    quantity: QuantitySpec = Field(..., description="Quantity")
    running_balance: QuantitySpec = Field(
        ..., description="Running balance after entry"
    )
    is_certified: bool = Field(default=False)
    source: Optional[str] = Field(None, description="Source facility")
    destination: Optional[str] = Field(None, description="Destination facility")
    timestamp: datetime = Field(default_factory=_utcnow)

    model_config = ConfigDict(from_attributes=True)


class BalanceHistoryResponse(BaseModel):
    """Response for balance history."""

    facility_id: str = Field(..., description="Facility ID")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    entries: List[BalanceHistoryEntry] = Field(
        default_factory=list, description="History entries"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")
    current_balance: QuantitySpec = Field(
        ..., description="Current balance"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Transformation Schemas
# =============================================================================


class TransformInputSpec(BaseModel):
    """Input batch specification for transformation."""

    batch_id: str = Field(..., min_length=1, max_length=100)
    quantity: QuantitySpec = Field(..., description="Quantity consumed")
    commodity: EUDRCommodity = Field(..., description="Input commodity")

    model_config = ConfigDict(from_attributes=True)


class TransformOutputSpec(BaseModel):
    """Output batch specification for transformation."""

    batch_reference: Optional[str] = Field(
        None, max_length=200, description="Reference for output batch"
    )
    quantity: QuantitySpec = Field(..., description="Output quantity")
    commodity: EUDRCommodity = Field(..., description="Output commodity")

    model_config = ConfigDict(from_attributes=True)


class TransformCreateRequest(BaseModel):
    """Request to record a product transformation."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Processing facility"
    )
    transformation_type: TransformationType = Field(
        ..., description="Type of transformation"
    )
    input_batches: List[TransformInputSpec] = Field(
        ..., min_length=1, max_length=50,
        description="Input batch specifications",
    )
    output_batches: List[TransformOutputSpec] = Field(
        ..., min_length=1, max_length=50,
        description="Output batch specifications",
    )
    conversion_factor: Decimal = Field(
        ..., gt=0, le=10, description="Input-to-output conversion factor"
    )
    loss_rate: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Processing loss rate (0-1)"
    )
    timestamp: Optional[datetime] = Field(
        None, description="Transformation timestamp"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )
    metadata: Optional[Dict[str, Any]] = Field(None)

    model_config = ConfigDict(extra="forbid")


class TransformResponse(BaseModel):
    """Response for a recorded transformation."""

    transform_id: str = Field(
        ..., description="Unique transformation ID"
    )
    facility_id: str = Field(..., description="Processing facility")
    transformation_type: TransformationType = Field(
        ..., description="Transformation type"
    )
    input_batches: List[TransformInputDetail] = Field(
        default_factory=list, description="Input details"
    )
    output_batches: List[TransformOutputDetail] = Field(
        default_factory=list, description="Output details"
    )
    conversion_factor: Decimal = Field(..., description="Conversion factor")
    loss_rate: Optional[Decimal] = Field(None)
    total_input_quantity: QuantitySpec = Field(
        ..., description="Total input quantity"
    )
    total_output_quantity: QuantitySpec = Field(
        ..., description="Total output quantity"
    )
    total_loss_quantity: Optional[QuantitySpec] = Field(
        None, description="Total loss quantity"
    )
    timestamp: datetime = Field(default_factory=_utcnow)
    status: str = Field(default="completed")
    notes: Optional[str] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class TransformInputDetail(BaseModel):
    """Detailed input batch information for transformation response."""

    batch_id: str = Field(..., description="Input batch ID")
    quantity: QuantitySpec = Field(..., description="Quantity consumed")
    commodity: EUDRCommodity = Field(..., description="Commodity")
    origin_country: Optional[str] = Field(None)
    custody_model: Optional[CustodyModelType] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class TransformOutputDetail(BaseModel):
    """Detailed output batch information for transformation response."""

    batch_id: str = Field(..., description="Output batch ID")
    batch_reference: Optional[str] = Field(None)
    quantity: QuantitySpec = Field(..., description="Output quantity")
    commodity: EUDRCommodity = Field(..., description="Commodity")

    model_config = ConfigDict(from_attributes=True)


class TransformBatchRequest(BaseModel):
    """Request for batch transformation import."""

    transformations: List[TransformCreateRequest] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of transformations to import (max 500)",
    )
    validate_only: bool = Field(
        default=False, description="If true, validate without persisting"
    )

    model_config = ConfigDict(extra="forbid")


class TransformBatchResponse(BaseModel):
    """Response for batch transformation import."""

    total_submitted: int = Field(..., ge=0)
    total_accepted: int = Field(..., ge=0)
    total_rejected: int = Field(..., ge=0)
    transformations: List[TransformResponse] = Field(
        default_factory=list
    )
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    validate_only: bool = Field(default=False)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Document Schemas
# =============================================================================


class DocumentLinkRequest(BaseModel):
    """Request to link a document to a custody event or batch."""

    event_id: Optional[str] = Field(
        None, max_length=100, description="Associated event ID"
    )
    batch_id: Optional[str] = Field(
        None, max_length=100, description="Associated batch ID"
    )
    document_type: DocumentType = Field(
        ..., description="Type of document"
    )
    document_reference: str = Field(
        ..., min_length=1, max_length=500,
        description="Document reference or identifier",
    )
    document_url: Optional[str] = Field(
        None, max_length=2000, description="URL to document"
    )
    document_hash: Optional[str] = Field(
        None, max_length=128,
        description="SHA-256 hash of document content",
    )
    issuer: Optional[str] = Field(
        None, max_length=500, description="Document issuer"
    )
    issue_date: Optional[datetime] = Field(
        None, description="Document issue date"
    )
    expiry_date: Optional[datetime] = Field(
        None, description="Document expiry date"
    )
    notes: Optional[str] = Field(
        None, max_length=2000, description="Additional notes"
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("document_reference")
    @classmethod
    def validate_reference(cls, v: str) -> str:
        """Validate document reference is non-empty."""
        if not v or not v.strip():
            raise ValueError("document_reference must be non-empty")
        return v


class DocumentResponse(BaseModel):
    """Response for a linked document."""

    document_id: str = Field(..., description="Unique document ID")
    event_id: Optional[str] = Field(None)
    batch_id: Optional[str] = Field(None)
    document_type: DocumentType = Field(..., description="Document type")
    document_reference: str = Field(..., description="Document reference")
    document_url: Optional[str] = Field(None)
    document_hash: Optional[str] = Field(None)
    issuer: Optional[str] = Field(None)
    issue_date: Optional[datetime] = Field(None)
    expiry_date: Optional[datetime] = Field(None)
    is_valid: bool = Field(
        default=True, description="Whether document is currently valid"
    )
    notes: Optional[str] = Field(None)
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    created_at: datetime = Field(default_factory=_utcnow)
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    """Response for listing documents for a batch."""

    batch_id: str = Field(..., description="Batch ID queried")
    documents: List[DocumentResponse] = Field(
        default_factory=list, description="Linked documents"
    )
    total_documents: int = Field(default=0, ge=0)
    document_types_present: List[DocumentType] = Field(
        default_factory=list, description="Document types found"
    )
    has_complete_chain: bool = Field(
        default=False,
        description="Whether required documents are complete",
    )
    missing_document_types: List[DocumentType] = Field(
        default_factory=list, description="Required but missing document types"
    )

    model_config = ConfigDict(from_attributes=True)


class DocumentValidateRequest(BaseModel):
    """Request to validate document chain for a batch."""

    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch ID"
    )
    required_document_types: List[DocumentType] = Field(
        default_factory=list,
        description="Required document types to check",
    )
    check_expiry: bool = Field(
        default=True, description="Check for expired documents"
    )
    check_hash_integrity: bool = Field(
        default=True, description="Verify document hash integrity"
    )

    model_config = ConfigDict(extra="forbid")


class DocumentValidationFinding(BaseModel):
    """Individual document validation finding."""

    document_id: Optional[str] = Field(None)
    document_type: DocumentType = Field(...)
    finding_type: str = Field(
        ..., description="Type of finding (missing, expired, invalid_hash)"
    )
    severity: VerificationSeverity = Field(...)
    message: str = Field(..., description="Finding description")

    model_config = ConfigDict(from_attributes=True)


class DocumentValidateResponse(BaseModel):
    """Response from document chain validation."""

    batch_id: str = Field(..., description="Batch ID")
    is_valid: bool = Field(..., description="Overall validation result")
    findings: List[DocumentValidationFinding] = Field(
        default_factory=list, description="Validation findings"
    )
    total_documents_checked: int = Field(default=0, ge=0)
    documents_valid: int = Field(default=0, ge=0)
    documents_invalid: int = Field(default=0, ge=0)
    missing_types: List[DocumentType] = Field(default_factory=list)
    expired_documents: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Verification Schemas
# =============================================================================


class ChainVerifyRequest(BaseModel):
    """Request to verify a complete custody chain."""

    batch_id: str = Field(
        ..., min_length=1, max_length=100, description="Batch to verify"
    )
    verify_documents: bool = Field(
        default=True, description="Include document verification"
    )
    verify_mass_balance: bool = Field(
        default=True, description="Include mass balance checks"
    )
    verify_model_compliance: bool = Field(
        default=True, description="Include CoC model compliance checks"
    )
    verify_geo_coordinates: bool = Field(
        default=True, description="Include GPS coordinate verification"
    )
    verify_temporal_consistency: bool = Field(
        default=True, description="Check chronological event ordering"
    )
    cutoff_date: Optional[datetime] = Field(
        None,
        description="EUDR cutoff date for deforestation check (default: 2020-12-31)",
    )

    model_config = ConfigDict(extra="forbid")


class VerificationFinding(BaseModel):
    """Individual verification finding."""

    finding_id: str = Field(..., description="Unique finding ID")
    category: str = Field(
        ..., description="Category (document, balance, model, geo, temporal)"
    )
    severity: VerificationSeverity = Field(..., description="Severity")
    rule_id: str = Field(..., description="Rule identifier")
    rule_name: str = Field(..., description="Human-readable rule name")
    passed: bool = Field(..., description="Whether the check passed")
    message: str = Field(..., description="Finding description")
    affected_event_ids: List[str] = Field(
        default_factory=list, description="Events related to this finding"
    )
    affected_batch_ids: List[str] = Field(
        default_factory=list, description="Batches related to this finding"
    )
    remediation: Optional[str] = Field(
        None, description="Suggested remediation"
    )
    details: Optional[Dict[str, Any]] = Field(None)

    model_config = ConfigDict(from_attributes=True)


class ChainVerifyResponse(BaseModel):
    """Response from chain verification."""

    verification_id: str = Field(
        ..., description="Unique verification ID"
    )
    batch_id: str = Field(..., description="Batch verified")
    status: VerificationStatus = Field(
        ..., description="Overall verification status"
    )
    compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Compliance score (0-100)"
    )
    total_checks: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    checks_warnings: int = Field(default=0, ge=0)
    findings: List[VerificationFinding] = Field(
        default_factory=list, description="Verification findings"
    )
    chain_length: int = Field(
        default=0, ge=0, description="Number of events in chain"
    )
    facilities_verified: int = Field(default=0, ge=0)
    documents_verified: int = Field(default=0, ge=0)
    origin_countries: List[str] = Field(default_factory=list)
    origin_plot_ids: List[str] = Field(default_factory=list)
    custody_models_used: List[CustodyModelType] = Field(
        default_factory=list
    )
    verified_at: datetime = Field(default_factory=_utcnow)
    expires_at: Optional[datetime] = Field(
        None, description="Verification validity expiry"
    )
    provenance: ProvenanceInfo = Field(
        ..., description="Provenance tracking data"
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


class BatchVerifyRequest(BaseModel):
    """Request for batch verification of multiple chains."""

    batch_ids: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch IDs to verify (max 100)",
    )
    verify_documents: bool = Field(default=True)
    verify_mass_balance: bool = Field(default=True)
    verify_model_compliance: bool = Field(default=True)

    model_config = ConfigDict(extra="forbid")


class BatchVerifyResponse(BaseModel):
    """Response from batch verification."""

    total_submitted: int = Field(..., ge=0)
    total_verified: int = Field(..., ge=0)
    total_passed: int = Field(..., ge=0)
    total_failed: int = Field(..., ge=0)
    results: List[ChainVerifyResponse] = Field(
        default_factory=list, description="Individual verification results"
    )
    overall_compliance_score: float = Field(
        default=0.0, ge=0.0, le=100.0
    )
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0, ge=0.0)

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas
# =============================================================================


class TraceabilityReportRequest(BaseModel):
    """Request for Article 9 traceability report."""

    batch_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="Batch IDs to include in report",
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    operator_id: str = Field(
        ..., min_length=1, max_length=100, description="Operator ID"
    )
    report_period_start: datetime = Field(
        ..., description="Report period start"
    )
    report_period_end: datetime = Field(
        ..., description="Report period end"
    )
    include_genealogy: bool = Field(
        default=True, description="Include batch genealogy trees"
    )
    include_documents: bool = Field(
        default=True, description="Include linked documents"
    )
    include_verification: bool = Field(
        default=True, description="Include verification results"
    )
    output_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Report output format"
    )
    language: str = Field(
        default="en", max_length=5, description="Report language (ISO 639-1)"
    )

    model_config = ConfigDict(extra="forbid")


class MassBalanceReportRequest(BaseModel):
    """Request for mass balance period report."""

    facility_id: str = Field(
        ..., min_length=1, max_length=100, description="Facility ID"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    include_entries: bool = Field(
        default=True, description="Include individual ledger entries"
    )
    include_reconciliation: bool = Field(
        default=True, description="Include reconciliation results"
    )
    output_format: ReportFormat = Field(
        default=ReportFormat.PDF, description="Report output format"
    )

    model_config = ConfigDict(extra="forbid")


class ReportResponse(BaseModel):
    """Response for generated report."""

    report_id: str = Field(..., description="Unique report ID")
    report_type: ReportType = Field(..., description="Report type")
    title: str = Field(..., description="Report title")
    status: str = Field(
        default="generated", description="Report generation status"
    )
    output_format: ReportFormat = Field(
        ..., description="Output format"
    )
    commodity: EUDRCommodity = Field(..., description="Commodity")
    period_start: datetime = Field(..., description="Period start")
    period_end: datetime = Field(..., description="Period end")
    total_batches: int = Field(default=0, ge=0)
    total_events: int = Field(default=0, ge=0)
    compliance_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Compliance score"
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Report summary data"
    )
    download_url: Optional[str] = Field(
        None, description="URL to download report file"
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
        None, description="Job result when completed"
    )
    error: Optional[str] = Field(
        None, description="Error message if failed"
    )
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
    """Health check response for the Chain of Custody API."""

    status: str = Field(default="healthy")
    agent_id: str = Field(default="GL-EUDR-COC-009")
    agent_name: str = Field(
        default="EUDR Chain of Custody Agent"
    )
    version: str = Field(default="1.0.0")
    timestamp: datetime = Field(default_factory=_utcnow)
    components: Dict[str, str] = Field(
        default_factory=lambda: {
            "event_tracker": "healthy",
            "batch_manager": "healthy",
            "model_enforcer": "healthy",
            "balance_engine": "healthy",
        },
        description="Component health statuses",
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Forward reference rebuilds
# =============================================================================

# Rebuild models to resolve any forward references
BatchSplitRequest.model_rebuild()
BatchGenealogyResponse.model_rebuild()
TransformCreateRequest.model_rebuild()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Enumerations
    "BalanceEntryType",
    "BatchJobStatus",
    "BatchJobType",
    "BatchOriginType",
    "BatchStatus",
    "ComplianceLevel",
    "CustodyEventType",
    "CustodyModelType",
    "DocumentType",
    "EUDRCommodity",
    "ReconciliationStatus",
    "ReportFormat",
    "ReportType",
    "TransformationType",
    "UnitOfMeasure",
    "VerificationSeverity",
    "VerificationStatus",
    # Common
    "GeoCoordinate",
    "PaginatedMeta",
    "ProvenanceInfo",
    "QuantitySpec",
    # Events
    "CustodyEventAmendRequest",
    "CustodyEventAmendResponse",
    "CustodyEventBatchRequest",
    "CustodyEventBatchResponse",
    "CustodyEventChainResponse",
    "CustodyEventCreateRequest",
    "CustodyEventResponse",
    # Batches
    "BatchBlendRequest",
    "BatchBlendResponse",
    "BatchCreateRequest",
    "BatchGenealogyResponse",
    "BatchMergeRequest",
    "BatchMergeResponse",
    "BatchResponse",
    "BatchSearchRequest",
    "BatchSearchResponse",
    "BatchSplitPart",
    "BatchSplitRequest",
    "BatchSplitResponse",
    "GenealogyNode",
    # Models
    "FacilityModelResponse",
    "ModelAssignRequest",
    "ModelAssignResponse",
    "ModelComplianceResponse",
    "ModelValidateRequest",
    "ModelValidateResponse",
    "ModelValidationFinding",
    # Balance
    "BalanceEntryResponse",
    "BalanceHistoryEntry",
    "BalanceHistoryResponse",
    "BalanceInputRequest",
    "BalanceOutputRequest",
    "BalanceReconcileRequest",
    "BalanceReconcileResponse",
    "FacilityBalanceResponse",
    # Transformations
    "TransformBatchRequest",
    "TransformBatchResponse",
    "TransformCreateRequest",
    "TransformInputDetail",
    "TransformInputSpec",
    "TransformOutputDetail",
    "TransformOutputSpec",
    "TransformResponse",
    # Documents
    "DocumentLinkRequest",
    "DocumentListResponse",
    "DocumentResponse",
    "DocumentValidateRequest",
    "DocumentValidateResponse",
    "DocumentValidationFinding",
    # Verification
    "BatchVerifyRequest",
    "BatchVerifyResponse",
    "ChainVerifyRequest",
    "ChainVerifyResponse",
    "VerificationFinding",
    # Reports
    "MassBalanceReportRequest",
    "ReportDownloadResponse",
    "ReportResponse",
    "TraceabilityReportRequest",
    # Batch Jobs
    "BatchJobCancelResponse",
    "BatchJobResponse",
    "BatchJobSubmitRequest",
    # Health
    "HealthResponse",
]

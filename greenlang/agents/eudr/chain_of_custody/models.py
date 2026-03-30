# -*- coding: utf-8 -*-
"""
Chain of Custody Data Models - AGENT-EUDR-009

Pydantic v2 data models for the Chain of Custody Agent covering custody
event tracking, batch lifecycle management, batch operations (split, merge,
blend, transform), CoC model assignment (IP, SG, MB, CB), mass balance
ledger entries, transformation processing records, document management,
chain verification, origin allocation, and compliance reporting.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all chain of custody operations per EU 2023/1115
Article 10(2)(f) and ISO 22095:2020.

Enumerations (11):
    - CustodyEventType, BatchStatus, CoCModelType, DocumentType,
      ProcessType, OriginAllocationType, ReportFormat,
      BatchOperationType, GapSeverity, VerificationStatus,
      LedgerEntryType

Core Models (10):
    - CustodyEvent, Batch, BatchOrigin, BatchOperation,
      CoCModelAssignment, MassBalanceEntry, TransformationRecord,
      CustodyDocument, ChainVerificationResult, OriginAllocation

Request Models (12):
    - RecordEventRequest, CreateBatchRequest, SplitBatchRequest,
      MergeBatchRequest, BlendBatchRequest, AssignModelRequest,
      RecordInputRequest, RecordOutputRequest, RecordTransformRequest,
      LinkDocumentRequest, VerifyChainRequest, GenerateReportRequest

Response Models (8):
    - RecordEventResponse, CreateBatchResponse, BatchOperationResponse,
      BalanceResponse, TransformResponse, VerificationResponse,
      ReportResponse, BatchResult

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, and AGENT-EUDR-008
    Multi-Tier Supplier Tracker.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody (GL-EUDR-COC-009)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: EUDR deforestation cutoff date (31 December 2020), per Article 2(1).
EUDR_DEFORESTATION_CUTOFF: str = "2020-12-31"

#: Maximum number of records in a single batch processing job.
MAX_BATCH_SIZE: int = 100_000

#: Maximum number of parts in a single split operation.
MAX_SPLIT_PARTS: int = 100

#: Maximum number of batches in a single merge operation.
MAX_MERGE_BATCHES: int = 50

#: Maximum custody gap threshold in hours (default).
DEFAULT_GAP_THRESHOLD_HOURS: int = 72

#: Maximum amendment depth (default).
DEFAULT_MAX_AMENDMENT_DEPTH: int = 5

#: EUDR Article 31 data retention in years.
EUDR_RETENTION_YEARS: int = 5

#: Default mass balance short credit period in months.
MB_SHORT_CREDIT_MONTHS: int = 3

#: Default mass balance long credit period in months.
MB_LONG_CREDIT_MONTHS: int = 12

#: Default mass balance overdraft threshold percentage.
MB_OVERDRAFT_THRESHOLD_PCT: float = 5.0

#: EUDR-regulated primary commodities (Annex I).
PRIMARY_COMMODITIES: List[str] = [
    "cattle", "cocoa", "coffee", "oil_palm",
    "rubber", "soya", "wood",
]

#: Mapping of derived products to primary commodities per EUDR Annex I.
DERIVED_TO_PRIMARY: Dict[str, str] = {
    "beef": "cattle",
    "leather": "cattle",
    "chocolate": "cocoa",
    "cocoa_butter": "cocoa",
    "cocoa_powder": "cocoa",
    "palm_oil": "oil_palm",
    "palm_kernel_oil": "oil_palm",
    "natural_rubber": "rubber",
    "tyres": "rubber",
    "soybean_oil": "soya",
    "soybean_meal": "soya",
    "timber": "wood",
    "furniture": "wood",
    "paper": "wood",
    "charcoal": "wood",
    "plywood": "wood",
}

#: Default yield ratios for common process types.
DEFAULT_YIELD_RATIOS: Dict[str, float] = {
    "fermentation": 0.92,
    "drying": 0.88,
    "roasting": 0.85,
    "milling": 0.85,
    "refining": 0.92,
    "pressing": 0.45,
    "extraction": 0.22,
    "slaughtering": 0.55,
    "sawing": 0.55,
    "tanning": 0.30,
    "blending": 0.99,
}

# =============================================================================
# Enumerations
# =============================================================================

class CustodyEventType(str, Enum):
    """Type of custody event in the commodity chain.

    Represents the specific custodial action taking place as a
    commodity batch moves through the supply chain. Each event type
    has distinct document requirements and validation rules per
    EUDR Article 10(2)(f).

    TRANSFER: Physical transfer of custody between two supply chain
        actors. Requires bill of lading and commercial invoice.
    RECEIPT: Receipt and acceptance of a commodity batch at a facility.
        Triggers weight verification and quality checks.
    STORAGE_IN: Entry of a batch into a storage facility (warehouse,
        silo, cold storage). Generates a warehouse receipt.
    STORAGE_OUT: Exit of a batch from a storage facility for dispatch.
        Generates a warehouse release note.
    PROCESSING_IN: Entry of raw material into a processing facility
        (mill, refinery, factory). Records input mass and quality.
    PROCESSING_OUT: Exit of processed product from a processing
        facility. Records output mass, yield ratio, and batch certificate.
    EXPORT: Export of a commodity batch from a country of origin.
        Requires export declaration and phytosanitary certificate.
    IMPORT_: Import of a commodity batch into the EU. Requires import
        declaration, customs clearance, and DDS reference.
    INSPECTION: Physical inspection of a batch by a competent authority
        or third-party auditor. Generates an inspection report.
    SAMPLING: Collection of samples from a batch for laboratory analysis.
        Maintains sample chain of custody form.
    """

    TRANSFER = "transfer"
    RECEIPT = "receipt"
    STORAGE_IN = "storage_in"
    STORAGE_OUT = "storage_out"
    PROCESSING_IN = "processing_in"
    PROCESSING_OUT = "processing_out"
    EXPORT = "export"
    IMPORT_ = "import_"
    INSPECTION = "inspection"
    SAMPLING = "sampling"

class BatchStatus(str, Enum):
    """Lifecycle status of a commodity batch.

    Tracks the current state of a batch as it moves through the
    supply chain from creation to final consumption.

    CREATED: Batch has been created with an initial quantity and origin
        allocation but has not yet entered physical custody.
    IN_TRANSIT: Batch is currently being transported between facilities.
    AT_FACILITY: Batch is at a facility (storage, processing, port)
        awaiting the next custodial action.
    PROCESSING: Batch is currently undergoing processing/transformation
        at a facility (e.g., milling, refining, roasting).
    PROCESSED: Batch has completed processing and is ready for dispatch
        as a derived product or intermediate.
    DISPATCHED: Batch has been dispatched from a facility and is en
        route to the next actor in the chain.
    DELIVERED: Batch has been delivered to and accepted by the final
        recipient (EU operator or downstream customer).
    CONSUMED: Batch has been fully consumed (processed, sold, or used)
        and is no longer active in the custody chain.
    """

    CREATED = "created"
    IN_TRANSIT = "in_transit"
    AT_FACILITY = "at_facility"
    PROCESSING = "processing"
    PROCESSED = "processed"
    DISPATCHED = "dispatched"
    DELIVERED = "delivered"
    CONSUMED = "consumed"

class CoCModelType(str, Enum):
    """Chain of Custody model type per ISO 22095:2020.

    Defines how EUDR-relevant commodities are tracked through the
    supply chain from plot of origin to final product placement.
    Each model has different requirements for physical separation,
    administrative tracking, and mixing rules.

    IDENTITY_PRESERVED: Compliant material is physically separated
        throughout the entire supply chain. Full plot-to-product
        traceability is maintained. Highest assurance level.
    SEGREGATED: Compliant material is kept separate from non-compliant
        material but may be mixed with other compliant material from
        different certified sources. Origin tracking maintained.
    MASS_BALANCE: Compliant and non-compliant material may be
        physically mixed but quantities are tracked administratively
        to ensure the correct volume of compliant product is sold.
        Credit-based system with periodic reconciliation.
    CONTROLLED_BLENDING: A controlled blend of compliant and
        non-compliant material where the ratio is tracked. The blend
        ratio determines the percentage claim on the output.
    """

    IDENTITY_PRESERVED = "identity_preserved"
    SEGREGATED = "segregated"
    MASS_BALANCE = "mass_balance"
    CONTROLLED_BLENDING = "controlled_blending"

class DocumentType(str, Enum):
    """Document types required for EUDR custody chain evidence.

    Each custody event type requires specific documents to establish
    a complete audit trail per EUDR Article 10(2) and Article 14.

    BILL_OF_LADING: Transport document for sea freight shipments.
    COMMERCIAL_INVOICE: Financial document for goods traded.
    PACKING_LIST: Itemized list of goods in a shipment.
    PHYTOSANITARY_CERTIFICATE: Plant health certificate for export.
    CERTIFICATE_OF_ORIGIN: Document certifying country of origin.
    GOODS_RECEIVED_NOTE: Acknowledgement of goods received.
    QUALITY_INSPECTION_REPORT: Report from quality inspection.
    WEIGHT_CERTIFICATE: Certified weight measurement document.
    WAREHOUSE_RECEIPT: Receipt for goods stored in a warehouse.
    STORAGE_CERTIFICATE: Certificate for storage conditions.
    WAREHOUSE_RELEASE_NOTE: Authorization to release stored goods.
    DISPATCH_NOTE: Document accompanying dispatched goods.
    PROCESSING_ORDER: Work order for processing operations.
    RAW_MATERIAL_RECEIPT: Receipt for raw materials entering processing.
    PROCESSING_REPORT: Report on processing operations and outcomes.
    YIELD_CERTIFICATE: Certificate documenting yield from processing.
    QUALITY_CERTIFICATE: Quality assurance certificate for products.
    BATCH_CERTIFICATE: Certificate identifying a specific batch.
    EXPORT_DECLARATION: Customs declaration for goods being exported.
    IMPORT_DECLARATION: Customs declaration for goods being imported.
    CUSTOMS_CLEARANCE: Customs clearance documentation.
    DDS_REFERENCE: Due Diligence Statement reference number.
    FUMIGATION_CERTIFICATE: Certificate for fumigation treatment.
    INSPECTION_REPORT: Report from physical inspection.
    SAMPLE_ANALYSIS_REPORT: Laboratory analysis results for samples.
    COMPLIANCE_CERTIFICATE: Certificate of regulatory compliance.
    SAMPLE_COLLECTION_FORM: Form documenting sample collection.
    CHAIN_OF_CUSTODY_FORM: Form documenting custody chain for samples.
    LABORATORY_ANALYSIS_REPORT: Full laboratory analysis report.
    """

    BILL_OF_LADING = "bill_of_lading"
    COMMERCIAL_INVOICE = "commercial_invoice"
    PACKING_LIST = "packing_list"
    PHYTOSANITARY_CERTIFICATE = "phytosanitary_certificate"
    CERTIFICATE_OF_ORIGIN = "certificate_of_origin"
    GOODS_RECEIVED_NOTE = "goods_received_note"
    QUALITY_INSPECTION_REPORT = "quality_inspection_report"
    WEIGHT_CERTIFICATE = "weight_certificate"
    WAREHOUSE_RECEIPT = "warehouse_receipt"
    STORAGE_CERTIFICATE = "storage_certificate"
    WAREHOUSE_RELEASE_NOTE = "warehouse_release_note"
    DISPATCH_NOTE = "dispatch_note"
    PROCESSING_ORDER = "processing_order"
    RAW_MATERIAL_RECEIPT = "raw_material_receipt"
    PROCESSING_REPORT = "processing_report"
    YIELD_CERTIFICATE = "yield_certificate"
    QUALITY_CERTIFICATE = "quality_certificate"
    BATCH_CERTIFICATE = "batch_certificate"
    EXPORT_DECLARATION = "export_declaration"
    IMPORT_DECLARATION = "import_declaration"
    CUSTOMS_CLEARANCE = "customs_clearance"
    DDS_REFERENCE = "dds_reference"
    FUMIGATION_CERTIFICATE = "fumigation_certificate"
    INSPECTION_REPORT = "inspection_report"
    SAMPLE_ANALYSIS_REPORT = "sample_analysis_report"
    COMPLIANCE_CERTIFICATE = "compliance_certificate"
    SAMPLE_COLLECTION_FORM = "sample_collection_form"
    CHAIN_OF_CUSTODY_FORM = "chain_of_custody_form"
    LABORATORY_ANALYSIS_REPORT = "laboratory_analysis_report"

class ProcessType(str, Enum):
    """Processing/transformation types for commodity conversion.

    Represents the specific industrial process applied to convert
    raw commodities into derived products. Each process type has
    an expected yield ratio (output mass / input mass) that is used
    for mass balance reconciliation.

    Organized by commodity processing chain with typical yield ratios
    documented in parentheses.
    """

    # Cocoa processing chain
    FERMENTATION = "fermentation"        # 0.92
    DRYING = "drying"                    # 0.88
    ROASTING = "roasting"                # 0.85
    WINNOWING = "winnowing"              # 0.80
    GRINDING = "grinding"                # 0.98
    PRESSING = "pressing"                # 0.45
    CONCHING = "conching"                # 0.97
    TEMPERING = "tempering"              # 0.99

    # Coffee processing chain
    WET_PROCESSING = "wet_processing"    # 0.60
    DRY_PROCESSING = "dry_processing"    # 0.50
    HULLING = "hulling"                  # 0.80
    POLISHING = "polishing"              # 0.98

    # Oil palm processing chain
    STERILIZATION = "sterilization"      # 0.95
    THRESHING = "threshing"              # 0.65
    DIGESTION = "digestion"              # 0.90
    EXTRACTION = "extraction"            # 0.22
    CLARIFICATION = "clarification"      # 0.95
    REFINING = "refining"                # 0.92
    FRACTIONATION = "fractionation"      # 0.90

    # Wood processing chain
    DEBARKING = "debarking"              # 0.90
    SAWING = "sawing"                    # 0.55
    PLANING = "planing"                  # 0.90
    KILN_DRYING = "kiln_drying"          # 0.92

    # Rubber processing chain
    COAGULATION = "coagulation"          # 0.60
    SHEETING = "sheeting"                # 0.95
    SMOKING = "smoking"                  # 0.88
    CRUMBLING = "crumbling"              # 0.92

    # Soya processing chain
    CLEANING = "cleaning"                # 0.98
    DEHULLING = "dehulling"              # 0.92
    FLAKING = "flaking"                  # 0.97
    SOLVENT_EXTRACTION = "solvent_extraction"  # 0.82

    # Cattle processing chain
    SLAUGHTERING = "slaughtering"        # 0.55
    DEBONING = "deboning"                # 0.70
    TANNING = "tanning"                  # 0.30

    # General / cross-commodity
    MILLING = "milling"                  # 0.85
    BLENDING = "blending"                # 0.99
    PACKAGING = "packaging"              # 0.99
    SORTING = "sorting"                  # 0.95

class OriginAllocationType(str, Enum):
    """Method of allocating origin to a batch.

    Determines how the origin (country, region, plot) is attributed
    to a batch, particularly relevant when batches are split, merged,
    or processed.

    DIRECT: Origin is directly attributed from a single source plot
        or producer. Used in Identity Preserved chains.
    PROPORTIONAL: Origin is attributed proportionally based on input
        quantities from multiple sources. Used in Segregated chains.
    CREDIT_BASED: Origin is attributed via mass balance credit system.
        Used in Mass Balance chains with periodic reconciliation.
    BLENDED: Origin is attributed based on blend ratios of compliant
        and non-compliant inputs. Used in Controlled Blending chains.
    """

    DIRECT = "direct"
    PROPORTIONAL = "proportional"
    CREDIT_BASED = "credit_based"
    BLENDED = "blended"

class BatchOperationType(str, Enum):
    """Type of batch operation.

    SPLIT: Divide a single batch into multiple sub-batches.
    MERGE: Combine multiple batches into a single batch.
    BLEND: Mix inputs from multiple batches under controlled blending.
    TRANSFORM: Process a batch to produce a derived product.
    """

    SPLIT = "split"
    MERGE = "merge"
    BLEND = "blend"
    TRANSFORM = "transform"

class GapSeverity(str, Enum):
    """Severity classification for custody chain gaps.

    CRITICAL: Gap blocks DDS submission entirely. Requires immediate
        remediation (e.g., missing origin data, broken custody chain).
    HIGH: Gap creates significant regulatory risk but may not block
        DDS with appropriate mitigation documentation.
    MEDIUM: Gap is a data quality issue that should be addressed but
        does not directly impact compliance status.
    LOW: Minor gap or informational finding. Does not affect
        compliance or DDS eligibility.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class VerificationStatus(str, Enum):
    """Status of a chain of custody verification.

    PASSED: All verification checks passed. Chain integrity confirmed.
    FAILED: One or more verification checks failed. Chain integrity
        compromised. Requires investigation and remediation.
    PARTIAL: Some checks passed but others could not be completed
        due to missing data or pending documents.
    PENDING: Verification has been requested but not yet completed.
    """

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"

class LedgerEntryType(str, Enum):
    """Type of mass balance ledger entry.

    INPUT: Material received into the mass balance account. Increases
        the available compliant balance.
    OUTPUT: Material dispatched from the mass balance account. Decreases
        the available compliant balance.
    ADJUSTMENT: Manual or automated adjustment to the balance (e.g.,
        shrinkage, quality downgrade, reclassification).
    CARRY_FORWARD: Balance carried forward to the next credit period
        during reconciliation.
    """

    INPUT = "input"
    OUTPUT = "output"
    ADJUSTMENT = "adjustment"
    CARRY_FORWARD = "carry_forward"

# =============================================================================
# Core Models
# =============================================================================

class CustodyEvent(GreenLangBase):
    """A single custody event in the commodity chain.

    Represents a specific custodial action (transfer, receipt, storage,
    processing, export, import, inspection, sampling) that occurs as a
    commodity batch moves through the supply chain. Each event is
    immutable once recorded and is linked to a specific batch.

    Attributes:
        event_id: Unique identifier for this custody event.
        batch_id: Identifier of the batch this event relates to.
        event_type: Type of custody event.
        operator_id: Identifier of the operator performing the action.
        operator_name: Human-readable name of the operator.
        facility_id: Identifier of the facility where the event occurs.
        facility_name: Human-readable name of the facility.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: EUDR commodity or derived product identifier.
        quantity_kg: Quantity in kilograms at this event.
        unit: Unit of measurement (default: kg).
        coc_model: Chain of custody model applied.
        timestamp: UTC timestamp when the event occurred.
        previous_event_id: Identifier of the preceding event in the chain.
        amendment_of: Event ID this event amends (if amendment).
        amendment_depth: Depth of amendment chain (0 = original).
        notes: Free-text notes or observations.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash for audit trail.
        created_at: UTC timestamp when the record was created.
        updated_at: UTC timestamp when the record was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this custody event",
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the batch this event relates to",
    )
    event_type: CustodyEventType = Field(
        ...,
        description="Type of custody event",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the operator performing the action",
    )
    operator_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable name of the operator",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Identifier of the facility where the event occurs",
    )
    facility_name: Optional[str] = Field(
        None,
        description="Human-readable name of the facility",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="EUDR commodity or derived product identifier",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity in kilograms at this event",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement",
    )
    coc_model: Optional[CoCModelType] = Field(
        None,
        description="Chain of custody model applied",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the event occurred",
    )
    previous_event_id: Optional[str] = Field(
        None,
        description="Identifier of the preceding event in the chain",
    )
    amendment_of: Optional[str] = Field(
        None,
        description="Event ID this event amends (if amendment)",
    )
    amendment_depth: int = Field(
        default=0,
        ge=0,
        description="Depth of amendment chain (0 = original)",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was last updated",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()

class BatchOrigin(GreenLangBase):
    """Origin allocation for a commodity batch.

    Tracks the geographic and producer origin of material in a batch,
    including plot-level coordinates for EUDR compliance.

    Attributes:
        origin_id: Unique identifier for this origin record.
        country_code: ISO 3166-1 alpha-2 country code of origin.
        region: Sub-national region or province of origin.
        plot_id: Identifier of the production plot (for IP/SG models).
        latitude: Latitude of the production plot center.
        longitude: Longitude of the production plot center.
        producer_id: Identifier of the producer/farmer.
        producer_name: Name of the producer/farmer.
        quantity_kg: Quantity from this origin in kilograms.
        percentage: Percentage of batch from this origin (0-100).
        deforestation_free: Whether origin is verified deforestation-free.
        verification_date: Date of last deforestation-free verification.
    """

    model_config = ConfigDict(from_attributes=True)

    origin_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this origin record",
    )
    country_code: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code of origin",
    )
    region: Optional[str] = Field(
        None,
        description="Sub-national region or province of origin",
    )
    plot_id: Optional[str] = Field(
        None,
        description="Identifier of the production plot",
    )
    latitude: Optional[float] = Field(
        None,
        ge=-90.0,
        le=90.0,
        description="Latitude of the production plot center",
    )
    longitude: Optional[float] = Field(
        None,
        ge=-180.0,
        le=180.0,
        description="Longitude of the production plot center",
    )
    producer_id: Optional[str] = Field(
        None,
        description="Identifier of the producer/farmer",
    )
    producer_name: Optional[str] = Field(
        None,
        description="Name of the producer/farmer",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity from this origin in kilograms",
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of batch from this origin (0-100)",
    )
    deforestation_free: Optional[bool] = Field(
        None,
        description="Whether origin is verified deforestation-free",
    )
    verification_date: Optional[datetime] = Field(
        None,
        description="Date of last deforestation-free verification",
    )

    @field_validator("country_code")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        return v.upper()

class Batch(GreenLangBase):
    """A commodity batch tracked through the custody chain.

    Represents a discrete quantity of a commodity (or derived product)
    that is tracked as a unit through the supply chain. Batches can be
    split, merged, blended, or transformed into new batches.

    Attributes:
        batch_id: Unique identifier for this batch.
        parent_batch_id: Identifier of the parent batch (for splits).
        commodity: EUDR commodity or derived product identifier.
        quantity_kg: Current quantity in kilograms.
        initial_quantity_kg: Original quantity at creation.
        unit: Unit of measurement.
        status: Current lifecycle status of the batch.
        coc_model: Chain of custody model applied to this batch.
        origins: List of origin allocations for this batch.
        current_operator_id: ID of the operator currently holding custody.
        current_facility_id: ID of the facility currently holding the batch.
        country_code: ISO 3166-1 alpha-2 country code of current location.
        dds_reference: Due Diligence Statement reference if applicable.
        harvest_date: Date of harvest or production.
        expiry_date: Best-before or quality expiry date.
        certifications: List of certification identifiers.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash for audit trail.
        created_at: UTC timestamp when the batch was created.
        updated_at: UTC timestamp when the batch was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch",
    )
    parent_batch_id: Optional[str] = Field(
        None,
        description="Identifier of the parent batch (for splits)",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="EUDR commodity or derived product identifier",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Current quantity in kilograms",
    )
    initial_quantity_kg: Optional[Decimal] = Field(
        None,
        gt=0,
        description="Original quantity at creation",
    )
    unit: str = Field(
        default="kg",
        description="Unit of measurement",
    )
    status: BatchStatus = Field(
        default=BatchStatus.CREATED,
        description="Current lifecycle status of the batch",
    )
    coc_model: Optional[CoCModelType] = Field(
        None,
        description="Chain of custody model applied to this batch",
    )
    origins: List[BatchOrigin] = Field(
        default_factory=list,
        description="List of origin allocations for this batch",
    )
    current_operator_id: Optional[str] = Field(
        None,
        description="ID of the operator currently holding custody",
    )
    current_facility_id: Optional[str] = Field(
        None,
        description="ID of the facility currently holding the batch",
    )
    country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code of current location",
    )
    dds_reference: Optional[str] = Field(
        None,
        description="Due Diligence Statement reference if applicable",
    )
    harvest_date: Optional[datetime] = Field(
        None,
        description="Date of harvest or production",
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Best-before or quality expiry date",
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="List of certification identifiers",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the batch was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the batch was last updated",
    )

    @model_validator(mode="after")
    def set_initial_quantity(self) -> Batch:
        """Set initial_quantity_kg from quantity_kg if not provided."""
        if self.initial_quantity_kg is None:
            self.initial_quantity_kg = self.quantity_kg
        return self

class BatchOperation(GreenLangBase):
    """Record of a batch operation (split, merge, blend, or transform).

    Captures the details of a batch operation including input and output
    batch references, quantities, and the operation type. Provides a
    complete audit trail for batch lifecycle changes.

    Attributes:
        operation_id: Unique identifier for this operation.
        operation_type: Type of batch operation performed.
        input_batch_ids: List of input batch identifiers.
        output_batch_ids: List of output batch identifiers.
        input_quantities_kg: Input quantities in kilograms.
        output_quantities_kg: Output quantities in kilograms.
        yield_ratio: Actual yield ratio (output total / input total).
        process_type: Processing type if transformation operation.
        operator_id: Identifier of the operator performing the operation.
        facility_id: Identifier of the facility.
        coc_model: CoC model governing this operation.
        notes: Free-text notes or observations.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp when the operation occurred.
    """

    model_config = ConfigDict(from_attributes=True)

    operation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this operation",
    )
    operation_type: BatchOperationType = Field(
        ...,
        description="Type of batch operation performed",
    )
    input_batch_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of input batch identifiers",
    )
    output_batch_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of output batch identifiers",
    )
    input_quantities_kg: List[Decimal] = Field(
        default_factory=list,
        description="Input quantities in kilograms",
    )
    output_quantities_kg: List[Decimal] = Field(
        default_factory=list,
        description="Output quantities in kilograms",
    )
    yield_ratio: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Actual yield ratio (output total / input total)",
    )
    process_type: Optional[ProcessType] = Field(
        None,
        description="Processing type if transformation operation",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the operator performing the operation",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Identifier of the facility",
    )
    coc_model: Optional[CoCModelType] = Field(
        None,
        description="CoC model governing this operation",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the operation occurred",
    )

class CoCModelAssignment(GreenLangBase):
    """Assignment of a CoC model to a batch or chain segment.

    Records the specific chain of custody model (IP, SG, MB, CB)
    assigned to a batch, the rationale, and any applicable constraints.

    Attributes:
        assignment_id: Unique identifier for this assignment.
        batch_id: Batch to which the model is assigned.
        coc_model: Chain of custody model type.
        assigned_by: Identifier of the user or system making assignment.
        rationale: Reason for the model selection.
        constraints: Model-specific constraints or requirements.
        effective_from: Start date of model applicability.
        effective_until: End date of model applicability (if temporary).
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the assignment was created.
    """

    model_config = ConfigDict(from_attributes=True)

    assignment_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this assignment",
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        description="Batch to which the model is assigned",
    )
    coc_model: CoCModelType = Field(
        ...,
        description="Chain of custody model type",
    )
    assigned_by: str = Field(
        ...,
        min_length=1,
        description="Identifier of the user or system making assignment",
    )
    rationale: Optional[str] = Field(
        None,
        description="Reason for the model selection",
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model-specific constraints or requirements",
    )
    effective_from: datetime = Field(
        default_factory=utcnow,
        description="Start date of model applicability",
    )
    effective_until: Optional[datetime] = Field(
        None,
        description="End date of model applicability (if temporary)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the assignment was created",
    )

class MassBalanceEntry(GreenLangBase):
    """A single entry in the mass balance ledger.

    Records a debit or credit to the mass balance account for a
    specific commodity at a specific facility. Used for mass balance
    CoC model reconciliation per ISO 22095:2020 Section 7.

    Attributes:
        entry_id: Unique identifier for this ledger entry.
        facility_id: Facility where the mass balance applies.
        commodity: Commodity tracked in this ledger.
        entry_type: Type of ledger entry (input/output/adjustment/carry).
        quantity_kg: Quantity in kilograms (positive for input, negative
            for output/adjustment).
        batch_id: Associated batch identifier.
        credit_period_start: Start of the credit period.
        credit_period_end: End of the credit period.
        running_balance_kg: Running balance after this entry.
        overdraft: Whether this entry creates an overdraft condition.
        notes: Free-text notes or observations.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp when the entry was recorded.
    """

    model_config = ConfigDict(from_attributes=True)

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this ledger entry",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Facility where the mass balance applies",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Commodity tracked in this ledger",
    )
    entry_type: LedgerEntryType = Field(
        ...,
        description="Type of ledger entry",
    )
    quantity_kg: Decimal = Field(
        ...,
        description="Quantity in kilograms",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Associated batch identifier",
    )
    credit_period_start: Optional[datetime] = Field(
        None,
        description="Start of the credit period",
    )
    credit_period_end: Optional[datetime] = Field(
        None,
        description="End of the credit period",
    )
    running_balance_kg: Optional[Decimal] = Field(
        None,
        description="Running balance after this entry",
    )
    overdraft: bool = Field(
        default=False,
        description="Whether this entry creates an overdraft condition",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the entry was recorded",
    )

class TransformationRecord(GreenLangBase):
    """Record of a commodity transformation/processing event.

    Captures the complete details of a processing operation that
    converts raw material into a derived product, including input
    and output quantities, yield ratios, and loss accounting.

    Attributes:
        transformation_id: Unique identifier for this transformation.
        process_type: Type of processing operation.
        facility_id: Processing facility identifier.
        operator_id: Processing operator identifier.
        input_batch_id: Input batch identifier.
        output_batch_id: Output batch identifier.
        input_commodity: Input commodity type.
        output_commodity: Output commodity type.
        input_quantity_kg: Input quantity in kilograms.
        output_quantity_kg: Output quantity in kilograms.
        expected_yield_ratio: Expected yield ratio for this process.
        actual_yield_ratio: Actual yield ratio achieved.
        loss_kg: Processing loss in kilograms.
        loss_percentage: Processing loss as a percentage.
        within_tolerance: Whether loss is within acceptable tolerance.
        processing_start: Start time of processing.
        processing_end: End time of processing.
        quality_grade: Quality grade of the output product.
        notes: Free-text notes or observations.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the record was created.
    """

    model_config = ConfigDict(from_attributes=True)

    transformation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this transformation",
    )
    process_type: ProcessType = Field(
        ...,
        description="Type of processing operation",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Processing facility identifier",
    )
    operator_id: str = Field(
        ...,
        min_length=1,
        description="Processing operator identifier",
    )
    input_batch_id: str = Field(
        ...,
        min_length=1,
        description="Input batch identifier",
    )
    output_batch_id: str = Field(
        ...,
        min_length=1,
        description="Output batch identifier",
    )
    input_commodity: str = Field(
        ...,
        min_length=1,
        description="Input commodity type",
    )
    output_commodity: str = Field(
        ...,
        min_length=1,
        description="Output commodity type",
    )
    input_quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Input quantity in kilograms",
    )
    output_quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Output quantity in kilograms",
    )
    expected_yield_ratio: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Expected yield ratio for this process",
    )
    actual_yield_ratio: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Actual yield ratio achieved",
    )
    loss_kg: Decimal = Field(
        ...,
        ge=0,
        description="Processing loss in kilograms",
    )
    loss_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Processing loss as a percentage",
    )
    within_tolerance: bool = Field(
        ...,
        description="Whether loss is within acceptable tolerance",
    )
    processing_start: Optional[datetime] = Field(
        None,
        description="Start time of processing",
    )
    processing_end: Optional[datetime] = Field(
        None,
        description="End time of processing",
    )
    quality_grade: Optional[str] = Field(
        None,
        description="Quality grade of the output product",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was created",
    )

class CustodyDocument(GreenLangBase):
    """A document linked to a custody event or batch.

    Represents a supporting document in the custody chain evidence
    trail. Documents can be mandatory or optional depending on the
    event type and EUDR requirements.

    Attributes:
        document_id: Unique identifier for this document record.
        document_type: Type of document.
        event_id: Linked custody event identifier (if applicable).
        batch_id: Linked batch identifier (if applicable).
        document_number: External document reference number.
        issuer: Organization that issued the document.
        issue_date: Date the document was issued.
        expiry_date: Date the document expires (if applicable).
        file_reference: Storage reference (S3 key, file path, URL).
        file_hash: SHA-256 hash of the document file for integrity.
        is_mandatory: Whether this document is mandatory for the event.
        is_verified: Whether the document has been verified.
        verified_by: Identifier of the verifier.
        verified_at: Timestamp of verification.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        created_at: UTC timestamp when the record was created.
    """

    model_config = ConfigDict(from_attributes=True)

    document_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this document record",
    )
    document_type: DocumentType = Field(
        ...,
        description="Type of document",
    )
    event_id: Optional[str] = Field(
        None,
        description="Linked custody event identifier",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Linked batch identifier",
    )
    document_number: Optional[str] = Field(
        None,
        description="External document reference number",
    )
    issuer: Optional[str] = Field(
        None,
        description="Organization that issued the document",
    )
    issue_date: Optional[datetime] = Field(
        None,
        description="Date the document was issued",
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Date the document expires",
    )
    file_reference: Optional[str] = Field(
        None,
        description="Storage reference (S3 key, file path, URL)",
    )
    file_hash: Optional[str] = Field(
        None,
        description="SHA-256 hash of the document file for integrity",
    )
    is_mandatory: bool = Field(
        default=False,
        description="Whether this document is mandatory for the event",
    )
    is_verified: bool = Field(
        default=False,
        description="Whether the document has been verified",
    )
    verified_by: Optional[str] = Field(
        None,
        description="Identifier of the verifier",
    )
    verified_at: Optional[datetime] = Field(
        None,
        description="Timestamp of verification",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was created",
    )

    @model_validator(mode="after")
    def validate_linkage(self) -> CustodyDocument:
        """Ensure at least one of event_id or batch_id is provided."""
        if not self.event_id and not self.batch_id:
            raise ValueError(
                "At least one of event_id or batch_id must be provided"
            )
        return self

class ChainVerificationResult(GreenLangBase):
    """Result of a chain of custody verification operation.

    Captures the outcome of verifying the integrity and completeness
    of a custody chain for a specific batch or supply chain segment.

    Attributes:
        verification_id: Unique identifier for this verification.
        batch_id: Batch identifier being verified.
        status: Overall verification status.
        completeness_score: Chain completeness score (0.0-1.0).
        integrity_score: Chain integrity score (0.0-1.0).
        total_events: Total custody events in the chain.
        verified_events: Number of events that passed verification.
        gaps_found: Number of custody gaps detected.
        gap_details: List of gap descriptions with severity.
        missing_documents: List of missing mandatory documents.
        coc_model_violations: List of CoC model rule violations.
        mass_balance_status: Mass balance reconciliation status.
        recommendations: List of recommended remediation actions.
        verified_by: Identifier of the verifier (user or system).
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp of the verification.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this verification",
    )
    batch_id: str = Field(
        ...,
        min_length=1,
        description="Batch identifier being verified",
    )
    status: VerificationStatus = Field(
        ...,
        description="Overall verification status",
    )
    completeness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Chain completeness score (0.0-1.0)",
    )
    integrity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Chain integrity score (0.0-1.0)",
    )
    total_events: int = Field(
        ...,
        ge=0,
        description="Total custody events in the chain",
    )
    verified_events: int = Field(
        ...,
        ge=0,
        description="Number of events that passed verification",
    )
    gaps_found: int = Field(
        default=0,
        ge=0,
        description="Number of custody gaps detected",
    )
    gap_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of gap descriptions with severity",
    )
    missing_documents: List[str] = Field(
        default_factory=list,
        description="List of missing mandatory documents",
    )
    coc_model_violations: List[str] = Field(
        default_factory=list,
        description="List of CoC model rule violations",
    )
    mass_balance_status: Optional[str] = Field(
        None,
        description="Mass balance reconciliation status",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="List of recommended remediation actions",
    )
    verified_by: Optional[str] = Field(
        None,
        description="Identifier of the verifier (user or system)",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the verification",
    )

class OriginAllocation(GreenLangBase):
    """Origin allocation record for mass balance and blending operations.

    Tracks how origin is allocated to output batches based on the
    CoC model in use. For mass balance, this uses credit-based
    allocation. For controlled blending, this uses ratio-based.

    Attributes:
        allocation_id: Unique identifier for this allocation.
        output_batch_id: Output batch receiving the allocation.
        input_batch_id: Input batch providing the origin.
        allocation_type: Method of origin allocation.
        allocated_quantity_kg: Quantity allocated in kilograms.
        percentage: Percentage of output attributed to this origin.
        origin_country_code: Country code of the allocated origin.
        origin_plot_id: Plot identifier of the allocated origin.
        compliant: Whether this allocation is from compliant origin.
        metadata: Additional contextual key-value pairs.
        timestamp: UTC timestamp of the allocation.
    """

    model_config = ConfigDict(from_attributes=True)

    allocation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this allocation",
    )
    output_batch_id: str = Field(
        ...,
        min_length=1,
        description="Output batch receiving the allocation",
    )
    input_batch_id: str = Field(
        ...,
        min_length=1,
        description="Input batch providing the origin",
    )
    allocation_type: OriginAllocationType = Field(
        ...,
        description="Method of origin allocation",
    )
    allocated_quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity allocated in kilograms",
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of output attributed to this origin",
    )
    origin_country_code: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Country code of the allocated origin",
    )
    origin_plot_id: Optional[str] = Field(
        None,
        description="Plot identifier of the allocated origin",
    )
    compliant: bool = Field(
        default=True,
        description="Whether this allocation is from compliant origin",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp of the allocation",
    )

# =============================================================================
# Request Models
# =============================================================================

class RecordEventRequest(GreenLangBase):
    """Request to record a custody event.

    Attributes:
        batch_id: Batch identifier.
        event_type: Type of custody event.
        operator_id: Operator performing the action.
        operator_name: Name of the operator.
        facility_id: Facility where event occurs.
        facility_name: Name of the facility.
        country_code: ISO 3166-1 alpha-2 country code.
        commodity: Commodity or derived product identifier.
        quantity_kg: Quantity in kilograms.
        coc_model: Chain of custody model applied.
        previous_event_id: Preceding event in the chain.
        amendment_of: Event being amended (if amendment).
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(..., min_length=1)
    event_type: CustodyEventType = Field(...)
    operator_id: str = Field(..., min_length=1)
    operator_name: str = Field(..., min_length=1)
    facility_id: Optional[str] = Field(None)
    facility_name: Optional[str] = Field(None)
    country_code: str = Field(..., min_length=2, max_length=2)
    commodity: str = Field(..., min_length=1)
    quantity_kg: Decimal = Field(..., gt=0)
    coc_model: Optional[CoCModelType] = Field(None)
    previous_event_id: Optional[str] = Field(None)
    amendment_of: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CreateBatchRequest(GreenLangBase):
    """Request to create a new commodity batch.

    Attributes:
        commodity: Commodity or derived product identifier.
        quantity_kg: Initial quantity in kilograms.
        operator_id: Operator creating the batch.
        facility_id: Facility where batch is created.
        country_code: ISO 3166-1 alpha-2 country code.
        coc_model: Chain of custody model to apply.
        origins: Origin allocations for the batch.
        harvest_date: Date of harvest or production.
        certifications: Certification identifiers.
        dds_reference: DDS reference if applicable.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(..., min_length=1)
    quantity_kg: Decimal = Field(..., gt=0)
    operator_id: str = Field(..., min_length=1)
    facility_id: Optional[str] = Field(None)
    country_code: Optional[str] = Field(None, min_length=2, max_length=2)
    coc_model: Optional[CoCModelType] = Field(None)
    origins: List[BatchOrigin] = Field(default_factory=list)
    harvest_date: Optional[datetime] = Field(None)
    certifications: List[str] = Field(default_factory=list)
    dds_reference: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SplitBatchRequest(GreenLangBase):
    """Request to split a batch into sub-batches.

    Attributes:
        source_batch_id: Batch to be split.
        split_quantities_kg: Quantity for each sub-batch in kg.
        operator_id: Operator performing the split.
        facility_id: Facility where split occurs.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    source_batch_id: str = Field(..., min_length=1)
    split_quantities_kg: List[Decimal] = Field(..., min_length=2)
    operator_id: str = Field(..., min_length=1)
    facility_id: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("split_quantities_kg")
    @classmethod
    def validate_positive_quantities(
        cls, v: List[Decimal],
    ) -> List[Decimal]:
        """Ensure all split quantities are positive."""
        for i, qty in enumerate(v):
            if qty <= 0:
                raise ValueError(
                    f"split_quantities_kg[{i}] must be > 0, got {qty}"
                )
        return v

class MergeBatchRequest(GreenLangBase):
    """Request to merge multiple batches into one.

    Attributes:
        source_batch_ids: Batches to be merged.
        operator_id: Operator performing the merge.
        facility_id: Facility where merge occurs.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    source_batch_ids: List[str] = Field(..., min_length=2)
    operator_id: str = Field(..., min_length=1)
    facility_id: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BlendBatchRequest(GreenLangBase):
    """Request to blend multiple batches under controlled blending rules.

    Attributes:
        input_batch_ids: Input batch identifiers.
        input_quantities_kg: Quantity from each input batch in kg.
        operator_id: Operator performing the blend.
        facility_id: Facility where blend occurs.
        compliant_input_ids: Which inputs are from compliant origins.
        target_blend_ratio: Target ratio of compliant material (0-1).
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    input_batch_ids: List[str] = Field(..., min_length=2)
    input_quantities_kg: List[Decimal] = Field(..., min_length=2)
    operator_id: str = Field(..., min_length=1)
    facility_id: Optional[str] = Field(None)
    compliant_input_ids: List[str] = Field(default_factory=list)
    target_blend_ratio: Optional[float] = Field(
        None, gt=0.0, le=1.0,
    )
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_lengths_match(self) -> BlendBatchRequest:
        """Ensure input_batch_ids and input_quantities_kg have same length."""
        if len(self.input_batch_ids) != len(self.input_quantities_kg):
            raise ValueError(
                "input_batch_ids and input_quantities_kg must have "
                "the same length"
            )
        return self

class AssignModelRequest(GreenLangBase):
    """Request to assign a CoC model to a batch.

    Attributes:
        batch_id: Batch to assign the model to.
        coc_model: Chain of custody model type.
        assigned_by: User or system making the assignment.
        rationale: Reason for model selection.
        constraints: Model-specific constraints.
        effective_from: Start of model applicability.
        effective_until: End of model applicability.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(..., min_length=1)
    coc_model: CoCModelType = Field(...)
    assigned_by: str = Field(..., min_length=1)
    rationale: Optional[str] = Field(None)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    effective_from: Optional[datetime] = Field(None)
    effective_until: Optional[datetime] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecordInputRequest(GreenLangBase):
    """Request to record a mass balance input entry.

    Attributes:
        facility_id: Facility for the mass balance ledger.
        commodity: Commodity being tracked.
        batch_id: Associated batch identifier.
        quantity_kg: Input quantity in kilograms.
        credit_period_months: Credit period length in months.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    batch_id: Optional[str] = Field(None)
    quantity_kg: Decimal = Field(..., gt=0)
    credit_period_months: Optional[int] = Field(None, ge=1, le=36)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecordOutputRequest(GreenLangBase):
    """Request to record a mass balance output entry.

    Attributes:
        facility_id: Facility for the mass balance ledger.
        commodity: Commodity being tracked.
        batch_id: Associated batch identifier.
        quantity_kg: Output quantity in kilograms.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    batch_id: Optional[str] = Field(None)
    quantity_kg: Decimal = Field(..., gt=0)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecordTransformRequest(GreenLangBase):
    """Request to record a processing transformation.

    Attributes:
        process_type: Type of processing operation.
        facility_id: Processing facility identifier.
        operator_id: Processing operator identifier.
        input_batch_id: Input batch identifier.
        input_commodity: Input commodity type.
        output_commodity: Output commodity type.
        input_quantity_kg: Input quantity in kilograms.
        output_quantity_kg: Output quantity in kilograms.
        processing_start: Start time of processing.
        processing_end: End time of processing.
        quality_grade: Quality grade of the output product.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    process_type: ProcessType = Field(...)
    facility_id: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)
    input_batch_id: str = Field(..., min_length=1)
    input_commodity: str = Field(..., min_length=1)
    output_commodity: str = Field(..., min_length=1)
    input_quantity_kg: Decimal = Field(..., gt=0)
    output_quantity_kg: Decimal = Field(..., gt=0)
    processing_start: Optional[datetime] = Field(None)
    processing_end: Optional[datetime] = Field(None)
    quality_grade: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class LinkDocumentRequest(GreenLangBase):
    """Request to link a document to a custody event or batch.

    Attributes:
        document_type: Type of document being linked.
        event_id: Custody event to link to.
        batch_id: Batch to link to.
        document_number: External document reference number.
        issuer: Document issuer organization.
        issue_date: Date of document issuance.
        expiry_date: Document expiry date.
        file_reference: Storage reference (S3 key, URL, etc.).
        file_hash: SHA-256 hash of the document file.
        is_mandatory: Whether the document is mandatory.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    document_type: DocumentType = Field(...)
    event_id: Optional[str] = Field(None)
    batch_id: Optional[str] = Field(None)
    document_number: Optional[str] = Field(None)
    issuer: Optional[str] = Field(None)
    issue_date: Optional[datetime] = Field(None)
    expiry_date: Optional[datetime] = Field(None)
    file_reference: Optional[str] = Field(None)
    file_hash: Optional[str] = Field(None)
    is_mandatory: bool = Field(default=False)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_linkage(self) -> LinkDocumentRequest:
        """Ensure at least one of event_id or batch_id is provided."""
        if not self.event_id and not self.batch_id:
            raise ValueError(
                "At least one of event_id or batch_id must be provided"
            )
        return self

class VerifyChainRequest(GreenLangBase):
    """Request to verify chain of custody integrity.

    Attributes:
        batch_id: Batch identifier to verify.
        include_documents: Whether to verify document completeness.
        include_mass_balance: Whether to verify mass balance.
        include_coc_model_rules: Whether to verify CoC model rules.
        depth: Maximum chain depth to verify (None = full chain).
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(..., min_length=1)
    include_documents: bool = Field(default=True)
    include_mass_balance: bool = Field(default=True)
    include_coc_model_rules: bool = Field(default=True)
    depth: Optional[int] = Field(None, ge=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class GenerateReportRequest(GreenLangBase):
    """Request to generate a chain of custody report.

    Attributes:
        batch_id: Batch identifier (optional, for batch-specific reports).
        facility_id: Facility identifier (for facility-level reports).
        commodity: Commodity filter.
        report_format: Desired output format.
        date_from: Report start date.
        date_to: Report end date.
        include_mass_balance: Include mass balance summary.
        include_transformations: Include transformation details.
        include_documents: Include document inventory.
        include_verification: Include verification results.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: Optional[str] = Field(None)
    facility_id: Optional[str] = Field(None)
    commodity: Optional[str] = Field(None)
    report_format: ReportFormat = Field(default=ReportFormat.JSON)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    include_mass_balance: bool = Field(default=True)
    include_transformations: bool = Field(default=True)
    include_documents: bool = Field(default=True)
    include_verification: bool = Field(default=True)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Response Models
# =============================================================================

class RecordEventResponse(GreenLangBase):
    """Response after recording a custody event.

    Attributes:
        event_id: Identifier of the recorded event.
        batch_id: Associated batch identifier.
        event_type: Type of custody event recorded.
        provenance_hash: SHA-256 provenance hash.
        gap_detected: Whether a custody gap was detected.
        gap_hours: Gap duration in hours (if gap detected).
        gap_severity: Severity of the gap (if gap detected).
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(...)
    batch_id: str = Field(...)
    event_type: CustodyEventType = Field(...)
    provenance_hash: str = Field(...)
    gap_detected: bool = Field(default=False)
    gap_hours: Optional[float] = Field(None)
    gap_severity: Optional[GapSeverity] = Field(None)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class CreateBatchResponse(GreenLangBase):
    """Response after creating a new batch.

    Attributes:
        batch_id: Identifier of the created batch.
        commodity: Commodity type.
        quantity_kg: Initial quantity.
        status: Initial batch status.
        coc_model: Assigned CoC model.
        origin_count: Number of origin allocations.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    batch_id: str = Field(...)
    commodity: str = Field(...)
    quantity_kg: Decimal = Field(...)
    status: BatchStatus = Field(...)
    coc_model: Optional[CoCModelType] = Field(None)
    origin_count: int = Field(default=0)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class BatchOperationResponse(GreenLangBase):
    """Response after a batch operation (split/merge/blend).

    Attributes:
        operation_id: Identifier of the operation.
        operation_type: Type of batch operation.
        input_batch_ids: Input batch identifiers.
        output_batch_ids: Output batch identifiers.
        total_input_kg: Total input quantity.
        total_output_kg: Total output quantity.
        yield_ratio: Actual yield ratio.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    operation_id: str = Field(...)
    operation_type: BatchOperationType = Field(...)
    input_batch_ids: List[str] = Field(...)
    output_batch_ids: List[str] = Field(...)
    total_input_kg: Decimal = Field(...)
    total_output_kg: Decimal = Field(...)
    yield_ratio: Optional[float] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class BalanceResponse(GreenLangBase):
    """Response for mass balance operations.

    Attributes:
        entry_id: Ledger entry identifier.
        facility_id: Facility identifier.
        commodity: Commodity tracked.
        entry_type: Type of ledger entry.
        quantity_kg: Entry quantity.
        running_balance_kg: Balance after this entry.
        overdraft: Whether overdraft condition exists.
        credit_period_end: End of credit period.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    entry_id: str = Field(...)
    facility_id: str = Field(...)
    commodity: str = Field(...)
    entry_type: LedgerEntryType = Field(...)
    quantity_kg: Decimal = Field(...)
    running_balance_kg: Decimal = Field(...)
    overdraft: bool = Field(default=False)
    credit_period_end: Optional[datetime] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class TransformResponse(GreenLangBase):
    """Response after recording a transformation.

    Attributes:
        transformation_id: Transformation identifier.
        process_type: Type of processing.
        input_batch_id: Input batch identifier.
        output_batch_id: Output batch identifier.
        input_quantity_kg: Input quantity.
        output_quantity_kg: Output quantity.
        actual_yield_ratio: Actual yield achieved.
        expected_yield_ratio: Expected yield ratio.
        loss_kg: Processing loss.
        within_tolerance: Whether loss is acceptable.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    transformation_id: str = Field(...)
    process_type: ProcessType = Field(...)
    input_batch_id: str = Field(...)
    output_batch_id: str = Field(...)
    input_quantity_kg: Decimal = Field(...)
    output_quantity_kg: Decimal = Field(...)
    actual_yield_ratio: float = Field(...)
    expected_yield_ratio: float = Field(...)
    loss_kg: Decimal = Field(...)
    within_tolerance: bool = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class VerificationResponse(GreenLangBase):
    """Response after chain verification.

    Attributes:
        verification_id: Verification identifier.
        batch_id: Batch that was verified.
        status: Overall verification status.
        completeness_score: Completeness score (0.0-1.0).
        integrity_score: Integrity score (0.0-1.0).
        total_events: Total events in chain.
        verified_events: Events that passed.
        gaps_found: Number of gaps found.
        gap_details: Gap descriptions.
        missing_documents: Missing mandatory documents.
        coc_model_violations: CoC model violations.
        recommendations: Remediation recommendations.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    verification_id: str = Field(...)
    batch_id: str = Field(...)
    status: VerificationStatus = Field(...)
    completeness_score: float = Field(...)
    integrity_score: float = Field(...)
    total_events: int = Field(...)
    verified_events: int = Field(...)
    gaps_found: int = Field(default=0)
    gap_details: List[Dict[str, Any]] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    coc_model_violations: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ReportResponse(GreenLangBase):
    """Response after generating a report.

    Attributes:
        report_id: Report identifier.
        report_format: Format of the generated report.
        file_reference: Storage reference for the generated report.
        file_size_bytes: Size of the generated report in bytes.
        batch_count: Number of batches included.
        event_count: Number of events included.
        date_from: Report period start.
        date_to: Report period end.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing duration in milliseconds.
        timestamp: UTC timestamp of the operation.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
    )
    report_format: ReportFormat = Field(...)
    file_reference: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    batch_count: int = Field(default=0, ge=0)
    event_count: int = Field(default=0, ge=0)
    date_from: Optional[datetime] = Field(None)
    date_to: Optional[datetime] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class BatchResult(GreenLangBase):
    """Generic batch processing result.

    Used for reporting the outcome of bulk operations across
    multiple records.

    Attributes:
        total_records: Total records in the batch.
        successful: Number of successfully processed records.
        failed: Number of failed records.
        errors: List of error descriptions.
        processing_time_ms: Total processing time in milliseconds.
        provenance_hash: SHA-256 provenance hash of the batch job.
        timestamp: UTC timestamp of the batch completion.
    """

    model_config = ConfigDict(from_attributes=True)

    total_records: int = Field(..., ge=0)
    successful: int = Field(..., ge=0)
    failed: int = Field(default=0, ge=0)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(...)
    provenance_hash: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def validate_counts(self) -> BatchResult:
        """Ensure successful + failed equals total_records."""
        if self.successful + self.failed != self.total_records:
            raise ValueError(
                f"successful ({self.successful}) + failed ({self.failed}) "
                f"must equal total_records ({self.total_records})"
            )
        return self

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "MAX_SPLIT_PARTS",
    "MAX_MERGE_BATCHES",
    "DEFAULT_GAP_THRESHOLD_HOURS",
    "DEFAULT_MAX_AMENDMENT_DEPTH",
    "EUDR_RETENTION_YEARS",
    "MB_SHORT_CREDIT_MONTHS",
    "MB_LONG_CREDIT_MONTHS",
    "MB_OVERDRAFT_THRESHOLD_PCT",
    "PRIMARY_COMMODITIES",
    "DERIVED_TO_PRIMARY",
    "DEFAULT_YIELD_RATIOS",
    # Enumerations
    "CustodyEventType",
    "BatchStatus",
    "CoCModelType",
    "DocumentType",
    "ProcessType",
    "OriginAllocationType",
    "ReportFormat",
    "BatchOperationType",
    "GapSeverity",
    "VerificationStatus",
    "LedgerEntryType",
    # Core Models
    "CustodyEvent",
    "Batch",
    "BatchOrigin",
    "BatchOperation",
    "CoCModelAssignment",
    "MassBalanceEntry",
    "TransformationRecord",
    "CustodyDocument",
    "ChainVerificationResult",
    "OriginAllocation",
    # Request Models
    "RecordEventRequest",
    "CreateBatchRequest",
    "SplitBatchRequest",
    "MergeBatchRequest",
    "BlendBatchRequest",
    "AssignModelRequest",
    "RecordInputRequest",
    "RecordOutputRequest",
    "RecordTransformRequest",
    "LinkDocumentRequest",
    "VerifyChainRequest",
    "GenerateReportRequest",
    # Response Models
    "RecordEventResponse",
    "CreateBatchResponse",
    "BatchOperationResponse",
    "BalanceResponse",
    "TransformResponse",
    "VerificationResponse",
    "ReportResponse",
    "BatchResult",
]

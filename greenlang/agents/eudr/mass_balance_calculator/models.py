# -*- coding: utf-8 -*-
"""
Mass Balance Calculator Data Models - AGENT-EUDR-011

Pydantic v2 data models for the Mass Balance Calculator Agent covering
double-entry ledger management, credit period lifecycle, conversion factor
validation, overdraft detection and enforcement, loss/waste tracking,
carry-forward with expiry, reconciliation with anomaly detection, and
multi-facility consolidation reporting.

Every model is designed for deterministic serialization and SHA-256
provenance hashing to ensure zero-hallucination, bit-perfect
reproducibility across all mass balance operations per EU 2023/1115
Article 14 and ISO 22095:2020 mass balance chain of custody.

Enumerations (15):
    - LedgerEntryType, PeriodStatus, OverdraftSeverity, OverdraftMode,
      LossType, WasteType, ConversionStatus, VarianceClassification,
      ReconciliationStatus, CarryForwardStatus, ReportFormat, ReportType,
      FacilityGroupType, ComplianceStatus, StandardType

Core Models (11):
    - Ledger, LedgerEntry, CreditPeriod, ConversionFactor,
      OverdraftEvent, LossRecord, CarryForward, Reconciliation,
      FacilityGroup, ConsolidationReport, BatchJob

Request Models (15):
    - CreateLedgerRequest, RecordEntryRequest, BulkEntryRequest,
      SearchLedgerRequest, CreatePeriodRequest, ExtendPeriodRequest,
      RolloverPeriodRequest, ValidateFactorRequest,
      RegisterCustomFactorRequest, CheckOverdraftRequest,
      ForecastOutputRequest, RequestExemptionRequest,
      RecordLossRequest, ValidateLossRequest,
      RunReconciliationRequest, SignOffReconciliationRequest,
      GenerateConsolidationRequest, CreateFacilityGroupRequest

Response Models (18):
    - LedgerResponse, LedgerBalanceResponse, EntryHistoryResponse,
      PeriodResponse, ActivePeriodsResponse,
      FactorValidationResponse, ReferenceFactorsResponse,
      OverdraftCheckResponse, OverdraftAlertResponse, ForecastResponse,
      LossValidationResponse, LossTrendsResponse,
      ReconciliationResponse, ReconciliationHistoryResponse,
      ConsolidationDashboardResponse, ConsolidationReportResponse,
      BatchJobResponse, HealthResponse

Compatibility:
    Imports EUDRCommodity from greenlang.agents.data.eudr_traceability.models for
    cross-agent consistency with AGENT-DATA-005 EUDR Traceability
    Connector, AGENT-EUDR-001 Supply Chain Mapper, and AGENT-EUDR-009
    Chain of Custody Agent.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-011 Mass Balance Calculator (GL-EUDR-MBC-011)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    Field,
    field_validator,
    model_validator,
)
from greenlang.schemas import GreenLangBase, utcnow
from greenlang.schemas.enums import ReportFormat

# ---------------------------------------------------------------------------
# Cross-agent commodity import (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.data.eudr_traceability.models import EUDRCommodity
except ImportError:
    EUDRCommodity = None  # type: ignore[assignment,misc]

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
MAX_BATCH_SIZE: int = 500

#: Default credit period length in days for EUDR.
DEFAULT_CREDIT_PERIOD_DAYS: int = 365

#: RSPO mass balance credit period in days (SCC 2020).
RSPO_CREDIT_PERIOD_DAYS: int = 90

#: FSC mass balance credit period in days (FSC-STD-40-004).
FSC_CREDIT_PERIOD_DAYS: int = 365

#: ISCC mass balance credit period in days (ISCC 203).
ISCC_CREDIT_PERIOD_DAYS: int = 365

#: Default grace period in days after credit period expiry.
DEFAULT_GRACE_PERIOD_DAYS: int = 5

#: Default overdraft resolution deadline in hours.
DEFAULT_OVERDRAFT_RESOLUTION_HOURS: int = 48

#: Default variance acceptable threshold (%).
DEFAULT_VARIANCE_ACCEPTABLE_PCT: float = 1.0

#: Default variance warning threshold (%).
DEFAULT_VARIANCE_WARNING_PCT: float = 3.0

#: Default conversion factor warn deviation (fraction).
DEFAULT_CF_WARN_DEVIATION: float = 0.05

#: Default conversion factor reject deviation (fraction).
DEFAULT_CF_REJECT_DEVIATION: float = 0.15

#: EUDR Article 14 data retention in years.
EUDR_RETENTION_YEARS: int = 5

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

# =============================================================================
# Enumerations
# =============================================================================

class LedgerEntryType(str, Enum):
    """Type of mass balance ledger entry.

    Represents the specific debit or credit action in the double-entry
    mass balance ledger system.

    INPUT: Compliant material received into the mass balance account.
        Increases the available certified balance.
    OUTPUT: Compliant material dispatched from the mass balance account.
        Decreases the available certified balance.
    ADJUSTMENT: Manual or automated balance adjustment (e.g., shrinkage,
        reclassification, correction). May be positive or negative.
    LOSS: Material lost during processing, transport, or storage.
        Always decreases the balance.
    WASTE: Material classified as waste (by-product, hazardous,
        non-recoverable). Always decreases the balance.
    CARRY_FORWARD_IN: Balance carried forward from a previous credit
        period into the current period. Increases current period balance.
    CARRY_FORWARD_OUT: Balance carried forward from the current credit
        period to the next period. Decreases current period balance.
    EXPIRY: Expired carry-forward balance that was not utilized within
        the grace period. Always decreases the balance.
    """

    INPUT = "input"
    OUTPUT = "output"
    ADJUSTMENT = "adjustment"
    LOSS = "loss"
    WASTE = "waste"
    CARRY_FORWARD_IN = "carry_forward_in"
    CARRY_FORWARD_OUT = "carry_forward_out"
    EXPIRY = "expiry"

class PeriodStatus(str, Enum):
    """Lifecycle status of a credit period.

    PENDING: Period has been defined but has not yet started.
        Typically pre-created for scheduling purposes.
    ACTIVE: Period is currently active and accepting ledger entries.
        Only one period per facility/commodity/standard should be active.
    RECONCILING: Period is in the reconciliation process. No new
        entries are accepted; only adjustments are permitted.
    CLOSED: Period has been reconciled and signed off. Immutable;
        no further changes are allowed.
    """

    PENDING = "pending"
    ACTIVE = "active"
    RECONCILING = "reconciling"
    CLOSED = "closed"

class OverdraftSeverity(str, Enum):
    """Severity classification for overdraft events.

    WARNING: Output exceeds balance but within the configured tolerance
        threshold. Logged but does not block the operation.
    VIOLATION: Output exceeds the tolerance threshold. Must be resolved
        within the configured overdraft_resolution_hours.
    CRITICAL: Output significantly exceeds the tolerance threshold or
        multiple unresolved violations exist. Blocks further outputs
        and escalates to compliance team.
    """

    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"

class OverdraftMode(str, Enum):
    """Overdraft enforcement mode.

    ZERO_TOLERANCE: Any output exceeding the current balance is an
        immediate violation. Strictest mode per EUDR requirements.
    PERCENTAGE: Allows overdraft up to a configured percentage of
        total period inputs before triggering a violation.
    ABSOLUTE: Allows overdraft up to a configured absolute quantity
        (in kg) before triggering a violation.
    """

    ZERO_TOLERANCE = "zero_tolerance"
    PERCENTAGE = "percentage"
    ABSOLUTE = "absolute"

class LossType(str, Enum):
    """Type of material loss in the supply chain.

    PROCESSING_LOSS: Material lost during industrial processing
        (milling, refining, roasting, etc.). Expected per yield ratios.
    TRANSPORT_LOSS: Material lost during transportation between
        facilities (spillage, damage, shrinkage).
    STORAGE_LOSS: Material lost during storage (moisture loss, pest
        damage, spoilage, evaporation).
    QUALITY_REJECTION: Material rejected during quality inspection
        that cannot be reprocessed or sold as certified.
    SPILLAGE: Accidental material spillage during handling operations.
    CONTAMINATION_LOSS: Material contaminated and rendered unsuitable
        for certified use (chemical, biological, physical).
    """

    PROCESSING_LOSS = "processing_loss"
    TRANSPORT_LOSS = "transport_loss"
    STORAGE_LOSS = "storage_loss"
    QUALITY_REJECTION = "quality_rejection"
    SPILLAGE = "spillage"
    CONTAMINATION_LOSS = "contamination_loss"

class WasteType(str, Enum):
    """Type of waste material generated during processing.

    BY_PRODUCT: Useful by-product from processing that may generate
        credits (e.g., cocoa butter from pressing, palm kernel oil
        from extraction). May credit the balance if by_product_credit
        is enabled.
    WASTE_MATERIAL: Non-recoverable waste from processing (e.g.,
        husks, shells, chaff). Does not generate credits.
    HAZARDOUS_WASTE: Waste classified as hazardous requiring special
        handling and disposal. Does not generate credits.
    """

    BY_PRODUCT = "by_product"
    WASTE_MATERIAL = "waste_material"
    HAZARDOUS_WASTE = "hazardous_waste"

class ConversionStatus(str, Enum):
    """Validation status of a conversion factor.

    VALIDATED: Conversion factor is within the acceptable deviation
        range from the reference factor. Approved for use.
    WARNED: Conversion factor deviates from the reference by more
        than the warn threshold but less than the reject threshold.
        Approved with a warning flag.
    REJECTED: Conversion factor deviates from the reference by more
        than the reject threshold. Not approved for use.
    PENDING: Conversion factor has been submitted but not yet
        validated against reference data.
    """

    VALIDATED = "validated"
    WARNED = "warned"
    REJECTED = "rejected"
    PENDING = "pending"

class VarianceClassification(str, Enum):
    """Classification of reconciliation variance.

    ACCEPTABLE: Variance is within the acceptable threshold. No
        action required. Period can be signed off.
    WARNING: Variance exceeds the acceptable threshold but is below
        the violation threshold. Investigation recommended.
    VIOLATION: Variance exceeds the violation threshold. Investigation
        required before sign-off. May indicate data integrity issues.
    """

    ACCEPTABLE = "acceptable"
    WARNING = "warning"
    VIOLATION = "violation"

class ReconciliationStatus(str, Enum):
    """Status of a period reconciliation process.

    PENDING: Reconciliation has been requested but not yet started.
    IN_PROGRESS: Reconciliation is currently being processed
        (variance calculation, anomaly detection, trend analysis).
    COMPLETED: Reconciliation has been completed and results are
        available. Awaiting sign-off.
    SIGNED_OFF: Reconciliation has been reviewed and signed off
        by an authorized operator. Period may be closed.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SIGNED_OFF = "signed_off"

class CarryForwardStatus(str, Enum):
    """Status of a carry-forward balance transfer.

    ACTIVE: Carry-forward balance is active and available for
        utilization in the target credit period.
    EXPIRED: Carry-forward balance has expired (grace period elapsed)
        and is no longer available for utilization.
    UTILIZED: Carry-forward balance has been fully utilized via
        output entries against the target period.
    PARTIAL: Carry-forward balance has been partially utilized.
        Remaining balance is still available until expiry.
    """

    ACTIVE = "active"
    EXPIRED = "expired"
    UTILIZED = "utilized"
    PARTIAL = "partial"

class ReportType(str, Enum):
    """Type of mass balance report.

    RECONCILIATION: Period-end reconciliation report with variance
        analysis, anomaly detection, and trend metrics.
    CONSOLIDATION: Multi-facility consolidated mass balance report.
    OVERDRAFT: Overdraft event history and resolution report.
    VARIANCE: Detailed variance analysis report across periods.
    EVIDENCE: Evidence package for EUDR compliance submission.
    """

    RECONCILIATION = "reconciliation"
    CONSOLIDATION = "consolidation"
    OVERDRAFT = "overdraft"
    VARIANCE = "variance"
    EVIDENCE = "evidence"

class FacilityGroupType(str, Enum):
    """Type of facility grouping for consolidation.

    REGION: Facilities grouped by geographic region.
    COUNTRY: Facilities grouped by country.
    COMMODITY: Facilities grouped by primary commodity handled.
    CUSTOM: Arbitrary custom grouping defined by the operator.
    """

    REGION = "region"
    COUNTRY = "country"
    COMMODITY = "commodity"
    CUSTOM = "custom"

class ComplianceStatus(str, Enum):
    """Compliance status for mass balance operations.

    COMPLIANT: All mass balance requirements are met. No overdrafts,
        all losses within tolerance, reconciliation passed.
    NON_COMPLIANT: One or more mass balance requirements are violated.
        Immediate remediation required.
    PENDING: Compliance status has not yet been determined. Awaiting
        reconciliation or verification.
    UNDER_REVIEW: Compliance status is under review following an
        anomaly detection or external audit finding.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    UNDER_REVIEW = "under_review"

class StandardType(str, Enum):
    """Certification standard governing the mass balance.

    RSPO: Roundtable on Sustainable Palm Oil (SCC 2020).
        90-day credit period.
    FSC: Forest Stewardship Council (FSC-STD-40-004).
        365-day credit period.
    ISCC: International Sustainability & Carbon Certification (ISCC 203).
        365-day credit period.
    UTZ_RA: UTZ / Rainforest Alliance standard.
        365-day credit period.
    FAIRTRADE: Fairtrade International mass balance standard.
        365-day credit period.
    EUDR_DEFAULT: Default EUDR mass balance rules when no specific
        standard applies. 365-day credit period.
    """

    RSPO = "rspo"
    FSC = "fsc"
    ISCC = "iscc"
    UTZ_RA = "utz_ra"
    FAIRTRADE = "fairtrade"
    EUDR_DEFAULT = "eudr_default"

# =============================================================================
# Core Models
# =============================================================================

class Ledger(GreenLangBase):
    """A double-entry mass balance ledger for a facility/commodity/standard.

    Represents a single mass balance account that tracks certified
    material inputs and outputs for a specific facility, commodity, and
    certification standard combination. Supports real-time balance
    tracking with utilization rate calculation.

    Attributes:
        ledger_id: Unique identifier for this ledger.
        facility_id: Identifier of the facility this ledger belongs to.
        commodity: EUDR commodity tracked by this ledger.
        standard: Certification standard governing this ledger.
        period_id: Currently active credit period identifier.
        current_balance: Current certified balance in kilograms.
        total_inputs: Cumulative inputs in kilograms.
        total_outputs: Cumulative outputs in kilograms.
        total_losses: Cumulative losses in kilograms.
        total_waste: Cumulative waste in kilograms.
        utilization_rate: Balance utilization rate (outputs / inputs).
        compliance_status: Current compliance status.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash for audit trail.
        created_at: UTC timestamp when the ledger was created.
        updated_at: UTC timestamp when the ledger was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this ledger",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility this ledger belongs to",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="EUDR commodity tracked by this ledger",
    )
    standard: StandardType = Field(
        default=StandardType.EUDR_DEFAULT,
        description="Certification standard governing this ledger",
    )
    period_id: Optional[str] = Field(
        None,
        description="Currently active credit period identifier",
    )
    current_balance: Decimal = Field(
        default=Decimal("0"),
        description="Current certified balance in kilograms",
    )
    total_inputs: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cumulative inputs in kilograms",
    )
    total_outputs: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cumulative outputs in kilograms",
    )
    total_losses: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cumulative losses in kilograms",
    )
    total_waste: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cumulative waste in kilograms",
    )
    utilization_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Balance utilization rate (outputs / inputs)",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="Current compliance status",
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
        description="UTC timestamp when the ledger was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the ledger was last updated",
    )

class LedgerEntry(GreenLangBase):
    """A single entry in the mass balance ledger.

    Records a debit or credit to the mass balance account. Every entry
    is immutable once recorded and contributes to the SHA-256 provenance
    chain for tamper-evident audit trails.

    Attributes:
        entry_id: Unique identifier for this ledger entry.
        ledger_id: Identifier of the parent ledger.
        entry_type: Type of ledger entry.
        batch_id: Associated batch identifier for traceability.
        quantity_kg: Quantity in kilograms (positive value; sign
            determined by entry_type).
        compliance_status: Compliance status of the source material.
        source_destination: Source (for inputs) or destination (for
            outputs) facility/operator identifier.
        conversion_factor_applied: Conversion factor applied to this
            entry (if applicable).
        timestamp: UTC timestamp when the entry was recorded.
        operator_id: Identifier of the operator recording the entry.
        provenance_hash: SHA-256 provenance hash for audit trail.
        notes: Free-text notes or observations.
        voided: Whether this entry has been voided.
        voided_at: UTC timestamp when the entry was voided.
        voided_by: Identifier of the operator who voided the entry.
        void_reason: Reason for voiding the entry.
        metadata: Additional contextual key-value pairs.
        created_at: UTC timestamp when the record was created.
    """

    model_config = ConfigDict(from_attributes=True)

    entry_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this ledger entry",
    )
    ledger_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the parent ledger",
    )
    entry_type: LedgerEntryType = Field(
        ...,
        description="Type of ledger entry",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Associated batch identifier for traceability",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity in kilograms (positive value)",
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING,
        description="Compliance status of the source material",
    )
    source_destination: Optional[str] = Field(
        None,
        description="Source or destination facility/operator identifier",
    )
    conversion_factor_applied: Optional[float] = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Conversion factor applied to this entry",
    )
    timestamp: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the entry was recorded",
    )
    operator_id: Optional[str] = Field(
        None,
        description="Identifier of the operator recording the entry",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes or observations",
    )
    voided: bool = Field(
        default=False,
        description="Whether this entry has been voided",
    )
    voided_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the entry was voided",
    )
    voided_by: Optional[str] = Field(
        None,
        description="Identifier of the operator who voided the entry",
    )
    void_reason: Optional[str] = Field(
        None,
        description="Reason for voiding the entry",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the record was created",
    )

class CreditPeriod(GreenLangBase):
    """A credit period for mass balance accounting.

    Represents a defined time window within which mass balance inputs
    can be offset against outputs. Credit periods are standard-specific
    (e.g., 90 days for RSPO, 365 days for FSC/ISCC).

    Attributes:
        period_id: Unique identifier for this credit period.
        facility_id: Facility this period applies to.
        commodity: Commodity tracked in this period.
        standard: Certification standard defining the period length.
        start_date: Period start date (UTC).
        end_date: Period end date (UTC).
        status: Current lifecycle status.
        grace_period_end: Grace period end date for carry-forward.
        carry_forward_balance: Balance carried forward from the
            previous period.
        opening_balance: Opening balance at period start.
        closing_balance: Closing balance at period end (set during
            reconciliation).
        total_inputs: Total inputs during this period.
        total_outputs: Total outputs during this period.
        total_losses: Total losses during this period.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the period was created.
        updated_at: UTC timestamp when the period was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    period_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this credit period",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Facility this period applies to",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Commodity tracked in this period",
    )
    standard: StandardType = Field(
        default=StandardType.EUDR_DEFAULT,
        description="Certification standard defining the period length",
    )
    start_date: datetime = Field(
        ...,
        description="Period start date (UTC)",
    )
    end_date: datetime = Field(
        ...,
        description="Period end date (UTC)",
    )
    status: PeriodStatus = Field(
        default=PeriodStatus.PENDING,
        description="Current lifecycle status",
    )
    grace_period_end: Optional[datetime] = Field(
        None,
        description="Grace period end date for carry-forward",
    )
    carry_forward_balance: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Balance carried forward from the previous period",
    )
    opening_balance: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Opening balance at period start",
    )
    closing_balance: Optional[Decimal] = Field(
        None,
        description="Closing balance at period end",
    )
    total_inputs: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total inputs during this period",
    )
    total_outputs: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total outputs during this period",
    )
    total_losses: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total losses during this period",
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
        description="UTC timestamp when the period was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the period was last updated",
    )

    @model_validator(mode="after")
    def validate_date_range(self) -> CreditPeriod:
        """Ensure end_date is after start_date."""
        if self.end_date <= self.start_date:
            raise ValueError(
                f"end_date ({self.end_date}) must be after "
                f"start_date ({self.start_date})"
            )
        return self

class ConversionFactor(GreenLangBase):
    """A conversion factor for commodity processing.

    Represents the yield ratio (output_mass / input_mass) for a specific
    commodity processing step. Validated against reference data to detect
    deviations that might indicate data quality issues.

    Attributes:
        factor_id: Unique identifier for this conversion factor.
        commodity: Commodity this factor applies to.
        process_name: Name of the processing step.
        input_material: Description of the input material.
        output_material: Description of the output material.
        yield_ratio: Reported yield ratio (0.0-1.0).
        acceptable_range_min: Minimum acceptable yield ratio.
        acceptable_range_max: Maximum acceptable yield ratio.
        source: Source of the conversion factor data.
        validation_status: Validation result against reference data.
        deviation_percent: Deviation from reference as a percentage.
        facility_id: Facility where this factor was measured.
        applied_at: UTC timestamp when the factor was last applied.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the factor was created.
    """

    model_config = ConfigDict(from_attributes=True)

    factor_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this conversion factor",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Commodity this factor applies to",
    )
    process_name: str = Field(
        ...,
        min_length=1,
        description="Name of the processing step",
    )
    input_material: Optional[str] = Field(
        None,
        description="Description of the input material",
    )
    output_material: Optional[str] = Field(
        None,
        description="Description of the output material",
    )
    yield_ratio: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Reported yield ratio (0.0-1.0)",
    )
    acceptable_range_min: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable yield ratio",
    )
    acceptable_range_max: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum acceptable yield ratio",
    )
    source: Optional[str] = Field(
        None,
        description="Source of the conversion factor data",
    )
    validation_status: ConversionStatus = Field(
        default=ConversionStatus.PENDING,
        description="Validation result against reference data",
    )
    deviation_percent: Optional[float] = Field(
        None,
        description="Deviation from reference as a percentage",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Facility where this factor was measured",
    )
    applied_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the factor was last applied",
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
        description="UTC timestamp when the factor was created",
    )

class OverdraftEvent(GreenLangBase):
    """An overdraft event detected in the mass balance ledger.

    Records when an output entry causes the ledger balance to go
    negative (or below the configured tolerance threshold).

    Attributes:
        event_id: Unique identifier for this overdraft event.
        ledger_id: Identifier of the affected ledger.
        facility_id: Identifier of the facility.
        commodity: Commodity affected by the overdraft.
        severity: Overdraft severity classification.
        current_balance: Balance at the time of overdraft detection.
        overdraft_amount: Amount by which the balance was exceeded.
        trigger_entry_id: Identifier of the entry that triggered
            the overdraft.
        resolution_deadline: Deadline by which the overdraft must
            be resolved.
        resolved: Whether the overdraft has been resolved.
        resolved_at: UTC timestamp when the overdraft was resolved.
        resolved_by: Identifier of the operator who resolved it.
        resolution_notes: Notes on how the overdraft was resolved.
        exemption_id: Exemption identifier if an exemption was granted.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the event was detected.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this overdraft event",
    )
    ledger_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the affected ledger",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the facility",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Commodity affected by the overdraft",
    )
    severity: OverdraftSeverity = Field(
        ...,
        description="Overdraft severity classification",
    )
    current_balance: Decimal = Field(
        ...,
        description="Balance at the time of overdraft detection",
    )
    overdraft_amount: Decimal = Field(
        ...,
        gt=0,
        description="Amount by which the balance was exceeded",
    )
    trigger_entry_id: Optional[str] = Field(
        None,
        description="Entry that triggered the overdraft",
    )
    resolution_deadline: Optional[datetime] = Field(
        None,
        description="Deadline for overdraft resolution",
    )
    resolved: bool = Field(
        default=False,
        description="Whether the overdraft has been resolved",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when resolved",
    )
    resolved_by: Optional[str] = Field(
        None,
        description="Operator who resolved it",
    )
    resolution_notes: Optional[str] = Field(
        None,
        description="Notes on resolution",
    )
    exemption_id: Optional[str] = Field(
        None,
        description="Exemption identifier if granted",
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
        description="UTC timestamp when the event was detected",
    )

class LossRecord(GreenLangBase):
    """A record of material loss or waste in the supply chain.

    Captures the details of material lost during processing, transport,
    storage, or due to quality issues. Used for mass balance
    reconciliation and tolerance validation.

    Attributes:
        record_id: Unique identifier for this loss record.
        ledger_id: Identifier of the associated ledger.
        loss_type: Type of loss.
        waste_type: Type of waste (if applicable).
        quantity_kg: Quantity lost in kilograms.
        percentage: Loss as a percentage of input quantity.
        batch_id: Associated batch identifier.
        process_type: Processing type that caused the loss.
        within_tolerance: Whether the loss is within acceptable range.
        expected_loss_percent: Expected loss percentage for this
            commodity/process combination.
        max_tolerance_percent: Maximum acceptable loss percentage.
        facility_id: Facility where the loss occurred.
        commodity: Commodity that was lost.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the loss was recorded.
    """

    model_config = ConfigDict(from_attributes=True)

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this loss record",
    )
    ledger_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the associated ledger",
    )
    loss_type: LossType = Field(
        ...,
        description="Type of loss",
    )
    waste_type: Optional[WasteType] = Field(
        None,
        description="Type of waste (if applicable)",
    )
    quantity_kg: Decimal = Field(
        ...,
        gt=0,
        description="Quantity lost in kilograms",
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Loss as a percentage of input quantity",
    )
    batch_id: Optional[str] = Field(
        None,
        description="Associated batch identifier",
    )
    process_type: Optional[str] = Field(
        None,
        description="Processing type that caused the loss",
    )
    within_tolerance: bool = Field(
        ...,
        description="Whether the loss is within acceptable range",
    )
    expected_loss_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Expected loss percentage",
    )
    max_tolerance_percent: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Maximum acceptable loss percentage",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Facility where the loss occurred",
    )
    commodity: Optional[str] = Field(
        None,
        description="Commodity that was lost",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
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
        description="UTC timestamp when the loss was recorded",
    )

class CarryForward(GreenLangBase):
    """A carry-forward balance transfer between credit periods.

    Records the transfer of unused certified balance from one credit
    period to the next, subject to maximum carry-forward percentage
    and grace period expiry rules.

    Attributes:
        carry_forward_id: Unique identifier for this carry-forward.
        from_period_id: Source credit period identifier.
        to_period_id: Target credit period identifier.
        amount_kg: Amount carried forward in kilograms.
        status: Current carry-forward status.
        expiry_date: Date when the carry-forward balance expires.
        utilized_amount: Amount utilized from the carry-forward.
        remaining_amount: Remaining available carry-forward balance.
        facility_id: Facility this carry-forward applies to.
        commodity: Commodity carried forward.
        standard: Certification standard.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the carry-forward was created.
        updated_at: UTC timestamp when the carry-forward was last
            updated.
    """

    model_config = ConfigDict(from_attributes=True)

    carry_forward_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this carry-forward",
    )
    from_period_id: str = Field(
        ...,
        min_length=1,
        description="Source credit period identifier",
    )
    to_period_id: str = Field(
        ...,
        min_length=1,
        description="Target credit period identifier",
    )
    amount_kg: Decimal = Field(
        ...,
        gt=0,
        description="Amount carried forward in kilograms",
    )
    status: CarryForwardStatus = Field(
        default=CarryForwardStatus.ACTIVE,
        description="Current carry-forward status",
    )
    expiry_date: Optional[datetime] = Field(
        None,
        description="Date when the carry-forward balance expires",
    )
    utilized_amount: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Amount utilized from the carry-forward",
    )
    remaining_amount: Optional[Decimal] = Field(
        None,
        description="Remaining available carry-forward balance",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Facility this carry-forward applies to",
    )
    commodity: Optional[str] = Field(
        None,
        description="Commodity carried forward",
    )
    standard: Optional[StandardType] = Field(
        None,
        description="Certification standard",
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
        description="UTC timestamp when the carry-forward was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the carry-forward was last updated",
    )

    @model_validator(mode="after")
    def set_remaining(self) -> CarryForward:
        """Set remaining_amount from amount_kg minus utilized_amount."""
        if self.remaining_amount is None:
            self.remaining_amount = self.amount_kg - self.utilized_amount
        return self

class Reconciliation(GreenLangBase):
    """A period-end reconciliation record.

    Captures the results of reconciling a credit period including
    expected vs recorded balance, variance analysis, anomaly detection,
    and sign-off tracking.

    Attributes:
        reconciliation_id: Unique identifier for this reconciliation.
        period_id: Credit period being reconciled.
        facility_id: Facility being reconciled.
        commodity: Commodity being reconciled.
        expected_balance: Expected balance based on inputs minus
            outputs minus losses.
        recorded_balance: Actual recorded balance in the ledger.
        variance_absolute: Absolute variance (expected - recorded).
        variance_percent: Variance as a percentage of expected balance.
        classification: Variance classification result.
        anomalies_detected: Number of anomalies detected.
        anomaly_details: List of anomaly descriptions.
        trend_deviation: Deviation from historical trend (if available).
        signed_off_by: Identifier of the operator who signed off.
        signed_off_at: UTC timestamp of sign-off.
        status: Current reconciliation status.
        notes: Free-text notes.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
        created_at: UTC timestamp when the reconciliation was created.
        updated_at: UTC timestamp when the reconciliation was last
            updated.
    """

    model_config = ConfigDict(from_attributes=True)

    reconciliation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this reconciliation",
    )
    period_id: str = Field(
        ...,
        min_length=1,
        description="Credit period being reconciled",
    )
    facility_id: str = Field(
        ...,
        min_length=1,
        description="Facility being reconciled",
    )
    commodity: str = Field(
        ...,
        min_length=1,
        description="Commodity being reconciled",
    )
    expected_balance: Decimal = Field(
        ...,
        description="Expected balance based on inputs minus outputs",
    )
    recorded_balance: Decimal = Field(
        ...,
        description="Actual recorded balance in the ledger",
    )
    variance_absolute: Decimal = Field(
        ...,
        description="Absolute variance (expected - recorded)",
    )
    variance_percent: float = Field(
        ...,
        description="Variance as a percentage of expected balance",
    )
    classification: VarianceClassification = Field(
        ...,
        description="Variance classification result",
    )
    anomalies_detected: int = Field(
        default=0,
        ge=0,
        description="Number of anomalies detected",
    )
    anomaly_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of anomaly descriptions",
    )
    trend_deviation: Optional[float] = Field(
        None,
        description="Deviation from historical trend",
    )
    signed_off_by: Optional[str] = Field(
        None,
        description="Operator who signed off",
    )
    signed_off_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp of sign-off",
    )
    status: ReconciliationStatus = Field(
        default=ReconciliationStatus.PENDING,
        description="Current reconciliation status",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text notes",
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
        description="UTC timestamp when the reconciliation was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when last updated",
    )

class FacilityGroup(GreenLangBase):
    """A group of facilities for consolidated reporting.

    Organizes multiple facilities into a logical group for
    multi-facility consolidation reporting purposes.

    Attributes:
        group_id: Unique identifier for this facility group.
        name: Human-readable group name.
        group_type: Type of grouping.
        facility_ids: List of facility identifiers in this group.
        description: Free-text description.
        metadata: Additional contextual key-value pairs.
        created_at: UTC timestamp when the group was created.
        updated_at: UTC timestamp when the group was last updated.
    """

    model_config = ConfigDict(from_attributes=True)

    group_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this facility group",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable group name",
    )
    group_type: FacilityGroupType = Field(
        ...,
        description="Type of grouping",
    )
    facility_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of facility identifiers in this group",
    )
    description: Optional[str] = Field(
        None,
        description="Free-text description",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    created_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the group was created",
    )
    updated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the group was last updated",
    )

class ConsolidationReport(GreenLangBase):
    """A consolidated multi-facility mass balance report.

    Aggregates mass balance data across multiple facilities for
    regulatory reporting and management oversight.

    Attributes:
        report_id: Unique identifier for this report.
        report_type: Type of report.
        report_format: Output format of the report.
        facility_ids: Facilities included in this report.
        group_id: Facility group identifier (if generated for a group).
        period_start: Report period start date.
        period_end: Report period end date.
        generated_at: UTC timestamp when the report was generated.
        generated_by: Identifier of the operator who generated it.
        data: Report data payload.
        file_reference: Storage reference for the generated file.
        file_size_bytes: Size of the generated file.
        summary: Summary statistics for the report.
        provenance_hash: SHA-256 provenance hash.
        metadata: Additional contextual key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report",
    )
    report_type: ReportType = Field(
        ...,
        description="Type of report",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Output format of the report",
    )
    facility_ids: List[str] = Field(
        default_factory=list,
        description="Facilities included in this report",
    )
    group_id: Optional[str] = Field(
        None,
        description="Facility group identifier",
    )
    period_start: Optional[datetime] = Field(
        None,
        description="Report period start date",
    )
    period_end: Optional[datetime] = Field(
        None,
        description="Report period end date",
    )
    generated_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the report was generated",
    )
    generated_by: Optional[str] = Field(
        None,
        description="Operator who generated the report",
    )
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Report data payload",
    )
    file_reference: Optional[str] = Field(
        None,
        description="Storage reference for the generated file",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="Size of the generated file",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics for the report",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )

class BatchJob(GreenLangBase):
    """A batch processing job record.

    Tracks the lifecycle of a batch processing job including progress,
    errors, and completion status.

    Attributes:
        job_id: Unique identifier for this batch job.
        job_type: Type of batch operation being performed.
        status: Current job status.
        total_items: Total number of items to process.
        processed_items: Number of items processed so far.
        failed_items: Number of items that failed processing.
        errors: List of error descriptions.
        started_at: UTC timestamp when the job started.
        completed_at: UTC timestamp when the job completed.
        processing_time_ms: Total processing time in milliseconds.
        metadata: Additional contextual key-value pairs.
        provenance_hash: SHA-256 provenance hash.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this batch job",
    )
    job_type: str = Field(
        ...,
        min_length=1,
        description="Type of batch operation being performed",
    )
    status: str = Field(
        default="pending",
        description="Current job status",
    )
    total_items: int = Field(
        ...,
        ge=0,
        description="Total number of items to process",
    )
    processed_items: int = Field(
        default=0,
        ge=0,
        description="Number of items processed so far",
    )
    failed_items: int = Field(
        default=0,
        ge=0,
        description="Number of items that failed processing",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of error descriptions",
    )
    started_at: datetime = Field(
        default_factory=utcnow,
        description="UTC timestamp when the job started",
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="UTC timestamp when the job completed",
    )
    processing_time_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Total processing time in milliseconds",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional contextual key-value pairs",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )

# =============================================================================
# Request Models
# =============================================================================

class CreateLedgerRequest(GreenLangBase):
    """Request to create a new mass balance ledger.

    Attributes:
        facility_id: Facility identifier.
        commodity: EUDR commodity.
        standard: Certification standard.
        initial_balance: Optional initial balance in kg.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    standard: StandardType = Field(default=StandardType.EUDR_DEFAULT)
    initial_balance: Decimal = Field(default=Decimal("0"), ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecordEntryRequest(GreenLangBase):
    """Request to record a single ledger entry.

    Attributes:
        ledger_id: Target ledger identifier.
        entry_type: Type of ledger entry.
        quantity_kg: Quantity in kilograms.
        batch_id: Associated batch identifier.
        source_destination: Source or destination identifier.
        operator_id: Operator recording the entry.
        conversion_factor: Conversion factor applied.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(..., min_length=1)
    entry_type: LedgerEntryType = Field(...)
    quantity_kg: Decimal = Field(..., gt=0)
    batch_id: Optional[str] = Field(None)
    source_destination: Optional[str] = Field(None)
    operator_id: Optional[str] = Field(None)
    conversion_factor: Optional[float] = Field(None, gt=0.0, le=1.0)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BulkEntryRequest(GreenLangBase):
    """Request to record multiple ledger entries in bulk.

    Attributes:
        entries: List of individual entry requests.
        operator_id: Operator performing the bulk operation.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    entries: List[RecordEntryRequest] = Field(..., min_length=1)
    operator_id: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("entries")
    @classmethod
    def validate_max_entries(
        cls, v: List[RecordEntryRequest],
    ) -> List[RecordEntryRequest]:
        """Ensure bulk entries do not exceed max batch size."""
        if len(v) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Maximum {MAX_BATCH_SIZE} entries per bulk request, "
                f"got {len(v)}"
            )
        return v

class SearchLedgerRequest(GreenLangBase):
    """Request to search ledgers by criteria.

    Attributes:
        facility_id: Filter by facility.
        commodity: Filter by commodity.
        standard: Filter by standard.
        min_balance: Minimum balance filter.
        max_balance: Maximum balance filter.
        compliance_status: Filter by compliance status.
        limit: Maximum results to return.
        offset: Result offset for pagination.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: Optional[str] = Field(None)
    commodity: Optional[str] = Field(None)
    standard: Optional[StandardType] = Field(None)
    min_balance: Optional[Decimal] = Field(None)
    max_balance: Optional[Decimal] = Field(None)
    compliance_status: Optional[ComplianceStatus] = Field(None)
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

class CreatePeriodRequest(GreenLangBase):
    """Request to create a new credit period.

    Attributes:
        facility_id: Facility identifier.
        commodity: Commodity for this period.
        standard: Certification standard.
        start_date: Period start date.
        end_date: Optional explicit end date.
        opening_balance: Opening balance from carry-forward.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    standard: StandardType = Field(default=StandardType.EUDR_DEFAULT)
    start_date: datetime = Field(...)
    end_date: Optional[datetime] = Field(None)
    opening_balance: Decimal = Field(default=Decimal("0"), ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExtendPeriodRequest(GreenLangBase):
    """Request to extend a credit period's end date.

    Attributes:
        period_id: Period to extend.
        new_end_date: New end date.
        reason: Reason for extension.
        operator_id: Operator requesting the extension.
    """

    model_config = ConfigDict(from_attributes=True)

    period_id: str = Field(..., min_length=1)
    new_end_date: datetime = Field(...)
    reason: str = Field(..., min_length=1)
    operator_id: str = Field(..., min_length=1)

class RolloverPeriodRequest(GreenLangBase):
    """Request to rollover a credit period to a new one.

    Attributes:
        closing_period_id: Period to close and rollover from.
        carry_forward_percent: Percentage of balance to carry forward.
        operator_id: Operator performing the rollover.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    closing_period_id: str = Field(..., min_length=1)
    carry_forward_percent: float = Field(
        default=100.0, ge=0.0, le=100.0,
    )
    operator_id: str = Field(..., min_length=1)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidateFactorRequest(GreenLangBase):
    """Request to validate a conversion factor against reference data.

    Attributes:
        commodity: Commodity to validate against.
        process_name: Processing step name.
        yield_ratio: Reported yield ratio.
        facility_id: Facility where measured.
        source: Source of the data.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(..., min_length=1)
    process_name: str = Field(..., min_length=1)
    yield_ratio: float = Field(..., gt=0.0, le=1.0)
    facility_id: Optional[str] = Field(None)
    source: Optional[str] = Field(None)

class RegisterCustomFactorRequest(GreenLangBase):
    """Request to register a custom conversion factor.

    Attributes:
        commodity: Commodity this factor applies to.
        process_name: Processing step name.
        yield_ratio: Custom yield ratio.
        input_material: Input material description.
        output_material: Output material description.
        facility_id: Facility where measured.
        source: Source/justification for the custom factor.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(..., min_length=1)
    process_name: str = Field(..., min_length=1)
    yield_ratio: float = Field(..., gt=0.0, le=1.0)
    input_material: Optional[str] = Field(None)
    output_material: Optional[str] = Field(None)
    facility_id: Optional[str] = Field(None)
    source: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class CheckOverdraftRequest(GreenLangBase):
    """Request to check for potential overdraft before recording output.

    Attributes:
        ledger_id: Ledger to check.
        output_quantity_kg: Proposed output quantity.
        dry_run: Whether this is a dry-run check (no state change).
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(..., min_length=1)
    output_quantity_kg: Decimal = Field(..., gt=0)
    dry_run: bool = Field(default=True)

class ForecastOutputRequest(GreenLangBase):
    """Request to forecast maximum available output quantity.

    Attributes:
        ledger_id: Ledger to forecast against.
        include_carry_forward: Include carry-forward balance.
        include_pending_inputs: Include pending input entries.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(..., min_length=1)
    include_carry_forward: bool = Field(default=True)
    include_pending_inputs: bool = Field(default=False)

class RequestExemptionRequest(GreenLangBase):
    """Request an exemption for an overdraft event.

    Attributes:
        event_id: Overdraft event identifier.
        reason: Justification for the exemption.
        requested_by: Operator requesting the exemption.
        supporting_evidence: Evidence supporting the request.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)
    requested_by: str = Field(..., min_length=1)
    supporting_evidence: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RecordLossRequest(GreenLangBase):
    """Request to record a material loss.

    Attributes:
        ledger_id: Target ledger.
        loss_type: Type of loss.
        waste_type: Type of waste (if applicable).
        quantity_kg: Quantity lost in kg.
        batch_id: Associated batch.
        process_type: Processing type that caused the loss.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(..., min_length=1)
    loss_type: LossType = Field(...)
    waste_type: Optional[WasteType] = Field(None)
    quantity_kg: Decimal = Field(..., gt=0)
    batch_id: Optional[str] = Field(None)
    process_type: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ValidateLossRequest(GreenLangBase):
    """Request to validate a loss against tolerance thresholds.

    Attributes:
        commodity: Commodity to validate against.
        loss_type: Type of loss.
        quantity_kg: Quantity lost.
        input_quantity_kg: Total input quantity for percentage calc.
        process_type: Processing type.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(..., min_length=1)
    loss_type: LossType = Field(...)
    quantity_kg: Decimal = Field(..., gt=0)
    input_quantity_kg: Decimal = Field(..., gt=0)
    process_type: Optional[str] = Field(None)

class RunReconciliationRequest(GreenLangBase):
    """Request to run a period reconciliation.

    Attributes:
        period_id: Period to reconcile.
        facility_id: Facility to reconcile.
        commodity: Commodity to reconcile.
        include_anomaly_detection: Run anomaly detection.
        include_trend_analysis: Run trend analysis.
        operator_id: Operator running the reconciliation.
        notes: Free-text notes.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    period_id: str = Field(..., min_length=1)
    facility_id: str = Field(..., min_length=1)
    commodity: str = Field(..., min_length=1)
    include_anomaly_detection: bool = Field(default=True)
    include_trend_analysis: bool = Field(default=True)
    operator_id: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SignOffReconciliationRequest(GreenLangBase):
    """Request to sign off on a completed reconciliation.

    Attributes:
        reconciliation_id: Reconciliation to sign off.
        signed_off_by: Operator signing off.
        notes: Free-text notes.
        auto_rollover: Whether to auto-create the next period.
    """

    model_config = ConfigDict(from_attributes=True)

    reconciliation_id: str = Field(..., min_length=1)
    signed_off_by: str = Field(..., min_length=1)
    notes: Optional[str] = Field(None)
    auto_rollover: bool = Field(default=True)

class GenerateConsolidationRequest(GreenLangBase):
    """Request to generate a consolidation report.

    Attributes:
        facility_ids: Facilities to include.
        group_id: Optional facility group identifier.
        report_type: Type of report.
        report_format: Output format.
        period_start: Report period start.
        period_end: Report period end.
        commodity: Filter by commodity.
        operator_id: Operator generating the report.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_ids: List[str] = Field(default_factory=list)
    group_id: Optional[str] = Field(None)
    report_type: ReportType = Field(default=ReportType.CONSOLIDATION)
    report_format: ReportFormat = Field(default=ReportFormat.JSON)
    period_start: Optional[datetime] = Field(None)
    period_end: Optional[datetime] = Field(None)
    commodity: Optional[str] = Field(None)
    operator_id: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source(self) -> GenerateConsolidationRequest:
        """Ensure either facility_ids or group_id is provided."""
        if not self.facility_ids and not self.group_id:
            raise ValueError(
                "Either facility_ids or group_id must be provided"
            )
        return self

class CreateFacilityGroupRequest(GreenLangBase):
    """Request to create a facility group.

    Attributes:
        name: Group name.
        group_type: Type of grouping.
        facility_ids: Facilities in the group.
        description: Group description.
        metadata: Additional key-value pairs.
    """

    model_config = ConfigDict(from_attributes=True)

    name: str = Field(..., min_length=1)
    group_type: FacilityGroupType = Field(...)
    facility_ids: List[str] = Field(..., min_length=1)
    description: Optional[str] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# Response Models
# =============================================================================

class LedgerResponse(GreenLangBase):
    """Response after creating or retrieving a ledger.

    Attributes:
        ledger_id: Ledger identifier.
        facility_id: Facility identifier.
        commodity: Commodity.
        standard: Certification standard.
        current_balance: Current balance in kg.
        compliance_status: Compliance status.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing time in milliseconds.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    facility_id: str = Field(...)
    commodity: str = Field(...)
    standard: StandardType = Field(...)
    current_balance: Decimal = Field(...)
    compliance_status: ComplianceStatus = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class LedgerBalanceResponse(GreenLangBase):
    """Response with detailed ledger balance information.

    Attributes:
        ledger_id: Ledger identifier.
        current_balance: Current balance in kg.
        total_inputs: Total inputs.
        total_outputs: Total outputs.
        total_losses: Total losses.
        total_waste: Total waste.
        utilization_rate: Utilization rate.
        carry_forward_available: Available carry-forward balance.
        overdraft_status: Whether overdraft exists.
        processing_time_ms: Processing time in milliseconds.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    current_balance: Decimal = Field(...)
    total_inputs: Decimal = Field(...)
    total_outputs: Decimal = Field(...)
    total_losses: Decimal = Field(...)
    total_waste: Decimal = Field(...)
    utilization_rate: float = Field(...)
    carry_forward_available: Decimal = Field(default=Decimal("0"))
    overdraft_status: bool = Field(default=False)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class EntryHistoryResponse(GreenLangBase):
    """Response with ledger entry history.

    Attributes:
        ledger_id: Ledger identifier.
        entries: List of ledger entries.
        total_count: Total number of entries.
        page_size: Number of entries returned.
        page_offset: Offset of the current page.
        processing_time_ms: Processing time in milliseconds.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    entries: List[LedgerEntry] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)
    page_size: int = Field(..., ge=0)
    page_offset: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class PeriodResponse(GreenLangBase):
    """Response after creating or retrieving a credit period.

    Attributes:
        period_id: Period identifier.
        facility_id: Facility identifier.
        commodity: Commodity.
        standard: Certification standard.
        start_date: Period start.
        end_date: Period end.
        status: Period status.
        grace_period_end: Grace period end.
        opening_balance: Opening balance.
        current_balance: Current balance (if active).
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    period_id: str = Field(...)
    facility_id: str = Field(...)
    commodity: str = Field(...)
    standard: StandardType = Field(...)
    start_date: datetime = Field(...)
    end_date: datetime = Field(...)
    status: PeriodStatus = Field(...)
    grace_period_end: Optional[datetime] = Field(None)
    opening_balance: Decimal = Field(...)
    current_balance: Optional[Decimal] = Field(None)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ActivePeriodsResponse(GreenLangBase):
    """Response listing active credit periods.

    Attributes:
        periods: List of active credit periods.
        total_count: Total number of active periods.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    periods: List[CreditPeriod] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class FactorValidationResponse(GreenLangBase):
    """Response after validating a conversion factor.

    Attributes:
        factor_id: Factor identifier.
        commodity: Commodity.
        process_name: Process name.
        yield_ratio: Reported yield ratio.
        reference_ratio: Reference yield ratio.
        deviation_percent: Deviation as percentage.
        validation_status: Validation result.
        message: Validation message.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    factor_id: str = Field(...)
    commodity: str = Field(...)
    process_name: str = Field(...)
    yield_ratio: float = Field(...)
    reference_ratio: Optional[float] = Field(None)
    deviation_percent: Optional[float] = Field(None)
    validation_status: ConversionStatus = Field(...)
    message: str = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ReferenceFactorsResponse(GreenLangBase):
    """Response listing reference conversion factors.

    Attributes:
        commodity: Commodity.
        factors: Dictionary of process_name to yield_ratio.
        source: Source of the reference data.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(...)
    factors: Dict[str, float] = Field(default_factory=dict)
    source: str = Field(default="EUDR reference data")
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class OverdraftCheckResponse(GreenLangBase):
    """Response after checking for potential overdraft.

    Attributes:
        ledger_id: Ledger identifier.
        current_balance: Current balance.
        proposed_output: Proposed output quantity.
        remaining_after: Balance after proposed output.
        overdraft_detected: Whether overdraft would occur.
        severity: Overdraft severity (if applicable).
        allowed: Whether the output is allowed.
        message: Explanation message.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    current_balance: Decimal = Field(...)
    proposed_output: Decimal = Field(...)
    remaining_after: Decimal = Field(...)
    overdraft_detected: bool = Field(...)
    severity: Optional[OverdraftSeverity] = Field(None)
    allowed: bool = Field(...)
    message: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class OverdraftAlertResponse(GreenLangBase):
    """Response with overdraft alert details.

    Attributes:
        event_id: Overdraft event identifier.
        ledger_id: Ledger identifier.
        facility_id: Facility identifier.
        commodity: Commodity.
        severity: Overdraft severity.
        overdraft_amount: Overdraft amount.
        resolution_deadline: Resolution deadline.
        unresolved_count: Total unresolved overdrafts for this ledger.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    event_id: str = Field(...)
    ledger_id: str = Field(...)
    facility_id: str = Field(...)
    commodity: str = Field(...)
    severity: OverdraftSeverity = Field(...)
    overdraft_amount: Decimal = Field(...)
    resolution_deadline: Optional[datetime] = Field(None)
    unresolved_count: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ForecastResponse(GreenLangBase):
    """Response with output forecast information.

    Attributes:
        ledger_id: Ledger identifier.
        available_balance: Available balance for output.
        carry_forward_included: Whether carry-forward was included.
        pending_inputs_included: Whether pending inputs were included.
        max_output_kg: Maximum output quantity available.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    available_balance: Decimal = Field(...)
    carry_forward_included: bool = Field(...)
    pending_inputs_included: bool = Field(...)
    max_output_kg: Decimal = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class LossValidationResponse(GreenLangBase):
    """Response after validating a loss against tolerances.

    Attributes:
        commodity: Commodity.
        loss_type: Type of loss.
        quantity_kg: Loss quantity.
        loss_percent: Loss as percentage of input.
        expected_percent: Expected loss percentage.
        max_tolerance_percent: Maximum tolerance.
        within_tolerance: Whether loss is acceptable.
        message: Validation message.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    commodity: str = Field(...)
    loss_type: LossType = Field(...)
    quantity_kg: Decimal = Field(...)
    loss_percent: float = Field(...)
    expected_percent: Optional[float] = Field(None)
    max_tolerance_percent: Optional[float] = Field(None)
    within_tolerance: bool = Field(...)
    message: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class LossTrendsResponse(GreenLangBase):
    """Response with loss trend analysis.

    Attributes:
        ledger_id: Ledger identifier.
        commodity: Commodity.
        periods_analyzed: Number of periods in trend.
        average_loss_percent: Average loss percentage.
        trend_direction: Trend direction (increasing/decreasing/stable).
        anomalies: List of anomalous loss events.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    ledger_id: str = Field(...)
    commodity: str = Field(...)
    periods_analyzed: int = Field(..., ge=0)
    average_loss_percent: float = Field(...)
    trend_direction: str = Field(...)
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ReconciliationResponse(GreenLangBase):
    """Response after running a reconciliation.

    Attributes:
        reconciliation_id: Reconciliation identifier.
        period_id: Period reconciled.
        facility_id: Facility reconciled.
        commodity: Commodity.
        expected_balance: Expected balance.
        recorded_balance: Recorded balance.
        variance_absolute: Absolute variance.
        variance_percent: Variance percentage.
        classification: Variance classification.
        anomalies_detected: Number of anomalies.
        status: Reconciliation status.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    reconciliation_id: str = Field(...)
    period_id: str = Field(...)
    facility_id: str = Field(...)
    commodity: str = Field(...)
    expected_balance: Decimal = Field(...)
    recorded_balance: Decimal = Field(...)
    variance_absolute: Decimal = Field(...)
    variance_percent: float = Field(...)
    classification: VarianceClassification = Field(...)
    anomalies_detected: int = Field(default=0, ge=0)
    status: ReconciliationStatus = Field(...)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ReconciliationHistoryResponse(GreenLangBase):
    """Response with reconciliation history.

    Attributes:
        facility_id: Facility identifier.
        commodity: Commodity.
        reconciliations: List of historical reconciliations.
        total_count: Total number of reconciliations.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_id: str = Field(...)
    commodity: str = Field(...)
    reconciliations: List[Reconciliation] = Field(default_factory=list)
    total_count: int = Field(..., ge=0)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ConsolidationDashboardResponse(GreenLangBase):
    """Response with consolidated dashboard data.

    Attributes:
        facility_count: Number of facilities.
        total_balance_kg: Total balance across facilities.
        total_inputs_kg: Total inputs across facilities.
        total_outputs_kg: Total outputs across facilities.
        total_losses_kg: Total losses across facilities.
        overdraft_count: Active overdraft count.
        compliance_summary: Compliance status counts.
        commodity_breakdown: Per-commodity balances.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    facility_count: int = Field(..., ge=0)
    total_balance_kg: Decimal = Field(...)
    total_inputs_kg: Decimal = Field(...)
    total_outputs_kg: Decimal = Field(...)
    total_losses_kg: Decimal = Field(...)
    overdraft_count: int = Field(default=0, ge=0)
    compliance_summary: Dict[str, int] = Field(default_factory=dict)
    commodity_breakdown: Dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class ConsolidationReportResponse(GreenLangBase):
    """Response after generating a consolidation report.

    Attributes:
        report_id: Report identifier.
        report_type: Type of report.
        report_format: Output format.
        facility_count: Number of facilities included.
        file_reference: Storage reference.
        file_size_bytes: File size.
        provenance_hash: SHA-256 provenance hash.
        processing_time_ms: Processing time.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    report_id: str = Field(...)
    report_type: ReportType = Field(...)
    report_format: ReportFormat = Field(...)
    facility_count: int = Field(..., ge=0)
    file_reference: Optional[str] = Field(None)
    file_size_bytes: Optional[int] = Field(None, ge=0)
    provenance_hash: str = Field(...)
    processing_time_ms: float = Field(...)
    timestamp: datetime = Field(default_factory=utcnow)

class BatchJobResponse(GreenLangBase):
    """Response for a batch processing job.

    Attributes:
        job_id: Job identifier.
        job_type: Type of batch operation.
        status: Job status.
        total_items: Total items.
        processed_items: Processed items.
        failed_items: Failed items.
        processing_time_ms: Processing time.
        provenance_hash: SHA-256 provenance hash.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    job_id: str = Field(...)
    job_type: str = Field(...)
    status: str = Field(...)
    total_items: int = Field(..., ge=0)
    processed_items: int = Field(..., ge=0)
    failed_items: int = Field(default=0, ge=0)
    processing_time_ms: float = Field(...)
    provenance_hash: Optional[str] = Field(None)
    timestamp: datetime = Field(default_factory=utcnow)

class HealthResponse(GreenLangBase):
    """Health check response for the mass balance calculator service.

    Attributes:
        service: Service name.
        version: Service version.
        status: Service status (healthy/unhealthy).
        database_connected: Database connectivity status.
        redis_connected: Redis connectivity status.
        active_ledgers: Number of active ledgers.
        active_periods: Number of active credit periods.
        unresolved_overdrafts: Number of unresolved overdrafts.
        uptime_seconds: Service uptime in seconds.
        timestamp: UTC timestamp.
    """

    model_config = ConfigDict(from_attributes=True)

    service: str = Field(default="gl-eudr-mbc-011")
    version: str = Field(default=VERSION)
    status: str = Field(default="healthy")
    database_connected: bool = Field(default=True)
    redis_connected: bool = Field(default=True)
    active_ledgers: int = Field(default=0, ge=0)
    active_periods: int = Field(default=0, ge=0)
    unresolved_overdrafts: int = Field(default=0, ge=0)
    uptime_seconds: float = Field(default=0.0, ge=0.0)
    timestamp: datetime = Field(default_factory=utcnow)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "EUDR_DEFORESTATION_CUTOFF",
    "MAX_BATCH_SIZE",
    "DEFAULT_CREDIT_PERIOD_DAYS",
    "RSPO_CREDIT_PERIOD_DAYS",
    "FSC_CREDIT_PERIOD_DAYS",
    "ISCC_CREDIT_PERIOD_DAYS",
    "DEFAULT_GRACE_PERIOD_DAYS",
    "DEFAULT_OVERDRAFT_RESOLUTION_HOURS",
    "DEFAULT_VARIANCE_ACCEPTABLE_PCT",
    "DEFAULT_VARIANCE_WARNING_PCT",
    "DEFAULT_CF_WARN_DEVIATION",
    "DEFAULT_CF_REJECT_DEVIATION",
    "EUDR_RETENTION_YEARS",
    "PRIMARY_COMMODITIES",
    "DERIVED_TO_PRIMARY",
    # Enumerations
    "LedgerEntryType",
    "PeriodStatus",
    "OverdraftSeverity",
    "OverdraftMode",
    "LossType",
    "WasteType",
    "ConversionStatus",
    "VarianceClassification",
    "ReconciliationStatus",
    "CarryForwardStatus",
    "ReportFormat",
    "ReportType",
    "FacilityGroupType",
    "ComplianceStatus",
    "StandardType",
    # Core Models
    "Ledger",
    "LedgerEntry",
    "CreditPeriod",
    "ConversionFactor",
    "OverdraftEvent",
    "LossRecord",
    "CarryForward",
    "Reconciliation",
    "FacilityGroup",
    "ConsolidationReport",
    "BatchJob",
    # Request Models
    "CreateLedgerRequest",
    "RecordEntryRequest",
    "BulkEntryRequest",
    "SearchLedgerRequest",
    "CreatePeriodRequest",
    "ExtendPeriodRequest",
    "RolloverPeriodRequest",
    "ValidateFactorRequest",
    "RegisterCustomFactorRequest",
    "CheckOverdraftRequest",
    "ForecastOutputRequest",
    "RequestExemptionRequest",
    "RecordLossRequest",
    "ValidateLossRequest",
    "RunReconciliationRequest",
    "SignOffReconciliationRequest",
    "GenerateConsolidationRequest",
    "CreateFacilityGroupRequest",
    # Response Models
    "LedgerResponse",
    "LedgerBalanceResponse",
    "EntryHistoryResponse",
    "PeriodResponse",
    "ActivePeriodsResponse",
    "FactorValidationResponse",
    "ReferenceFactorsResponse",
    "OverdraftCheckResponse",
    "OverdraftAlertResponse",
    "ForecastResponse",
    "LossValidationResponse",
    "LossTrendsResponse",
    "ReconciliationResponse",
    "ReconciliationHistoryResponse",
    "ConsolidationDashboardResponse",
    "ConsolidationReportResponse",
    "BatchJobResponse",
    "HealthResponse",
]

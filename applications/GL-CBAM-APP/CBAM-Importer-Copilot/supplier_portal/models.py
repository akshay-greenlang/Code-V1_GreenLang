# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal - Data Models

Pydantic v2 data models for the CBAM Supplier Portal, covering supplier
profiles, installations, emissions submissions, verification records,
data quality scoring, and importer-supplier data exchange.

All monetary and emissions values use ``Decimal`` with ``ROUND_HALF_UP``
to guarantee deterministic arithmetic and avoid floating-point drift in
regulatory calculations.

Reference:
  - EU CBAM Regulation 2023/956
  - EU Implementing Regulation 2023/1773
  - GHG Protocol Corporate Standard (Scope 1 + 2)

Version: 1.1.0
Author: GreenLang CBAM Team
"""

from __future__ import annotations

import re
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator, model_validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

# ============================================================================
# ENUMS
# ============================================================================


class SupplierStatus(str, Enum):
    """Lifecycle status of a registered supplier."""

    REGISTERED = "registered"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    SUSPENDED = "suspended"
    EXPIRED = "expired"


class VerificationStatus(str, Enum):
    """Verification status for a supplier or installation."""

    UNVERIFIED = "unverified"
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class InstallationType(str, Enum):
    """Type of production installation covered by CBAM."""

    CEMENT_PLANT = "cement_plant"
    STEEL_MILL = "steel_mill"
    ALUMINIUM_SMELTER = "aluminium_smelter"
    FERTILIZER_PLANT = "fertilizer_plant"
    POWER_PLANT = "power_plant"
    HYDROGEN_PLANT = "hydrogen_plant"
    MIXED = "mixed"


class CBAMSector(str, Enum):
    """CBAM sectors as defined in Annex I of Regulation 2023/956."""

    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    FERTILIZERS = "fertilizers"
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


class SubmissionStatus(str, Enum):
    """Lifecycle status of an emissions data submission."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"


class EvidenceType(str, Enum):
    """Type of supporting evidence document."""

    VERIFICATION_REPORT = "verification_report"
    PROCESS_DESCRIPTION = "process_description"
    PAYMENT_RECEIPT = "payment_receipt"
    TAX_CERTIFICATE = "tax_certificate"
    ACCREDITATION = "accreditation"


class CalculationMethod(str, Enum):
    """Emissions calculation methodology reference."""

    EU_IMPLEMENTING_REG = "eu_implementing_regulation"
    GHG_PROTOCOL = "ghg_protocol"
    NATIONAL_METHOD = "national_method"
    DEFAULT_VALUES = "default_values"
    DIRECT_MEASUREMENT = "direct_measurement"


class ExportFormat(str, Enum):
    """Supported data export formats."""

    CSV = "csv"
    JSON = "json"
    XML = "xml"


class AccessRequestStatus(str, Enum):
    """Status of an importer data-access request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    REVOKED = "revoked"


class VisitType(str, Enum):
    """Type of verification site visit."""

    ON_SITE = "on_site"
    REMOTE = "remote"


class VerificationOutcome(str, Enum):
    """Outcome of a verification assessment."""

    PASS = "pass"
    FAIL = "fail"
    CONDITIONAL = "conditional"


# ============================================================================
# CONSTANTS
# ============================================================================

# Mapping of 8-digit CN code prefixes to CBAM sectors (Annex I).
# Keys are 4-digit chapter/heading prefixes; values are CBAMSector.
CN_CODE_SECTORS: Dict[str, CBAMSector] = {
    # Cement
    "2523": CBAMSector.CEMENT,
    "2507": CBAMSector.CEMENT,
    # Iron and steel
    "7201": CBAMSector.IRON_STEEL,
    "7202": CBAMSector.IRON_STEEL,
    "7203": CBAMSector.IRON_STEEL,
    "7204": CBAMSector.IRON_STEEL,
    "7205": CBAMSector.IRON_STEEL,
    "7206": CBAMSector.IRON_STEEL,
    "7207": CBAMSector.IRON_STEEL,
    "7208": CBAMSector.IRON_STEEL,
    "7209": CBAMSector.IRON_STEEL,
    "7210": CBAMSector.IRON_STEEL,
    "7211": CBAMSector.IRON_STEEL,
    "7212": CBAMSector.IRON_STEEL,
    "7213": CBAMSector.IRON_STEEL,
    "7214": CBAMSector.IRON_STEEL,
    "7215": CBAMSector.IRON_STEEL,
    "7216": CBAMSector.IRON_STEEL,
    "7217": CBAMSector.IRON_STEEL,
    "7218": CBAMSector.IRON_STEEL,
    "7219": CBAMSector.IRON_STEEL,
    "7220": CBAMSector.IRON_STEEL,
    "7221": CBAMSector.IRON_STEEL,
    "7222": CBAMSector.IRON_STEEL,
    "7223": CBAMSector.IRON_STEEL,
    "7224": CBAMSector.IRON_STEEL,
    "7225": CBAMSector.IRON_STEEL,
    "7226": CBAMSector.IRON_STEEL,
    "7228": CBAMSector.IRON_STEEL,
    "7229": CBAMSector.IRON_STEEL,
    "7301": CBAMSector.IRON_STEEL,
    "7302": CBAMSector.IRON_STEEL,
    "7303": CBAMSector.IRON_STEEL,
    "7304": CBAMSector.IRON_STEEL,
    "7305": CBAMSector.IRON_STEEL,
    "7306": CBAMSector.IRON_STEEL,
    "7307": CBAMSector.IRON_STEEL,
    "7308": CBAMSector.IRON_STEEL,
    # Aluminium
    "7601": CBAMSector.ALUMINIUM,
    "7602": CBAMSector.ALUMINIUM,
    "7603": CBAMSector.ALUMINIUM,
    "7604": CBAMSector.ALUMINIUM,
    "7605": CBAMSector.ALUMINIUM,
    "7606": CBAMSector.ALUMINIUM,
    "7607": CBAMSector.ALUMINIUM,
    "7608": CBAMSector.ALUMINIUM,
    "7609": CBAMSector.ALUMINIUM,
    "7610": CBAMSector.ALUMINIUM,
    "7611": CBAMSector.ALUMINIUM,
    "7612": CBAMSector.ALUMINIUM,
    "7613": CBAMSector.ALUMINIUM,
    "7614": CBAMSector.ALUMINIUM,
    "7616": CBAMSector.ALUMINIUM,
    # Fertilizers
    "2808": CBAMSector.FERTILIZERS,
    "2814": CBAMSector.FERTILIZERS,
    "3102": CBAMSector.FERTILIZERS,
    "3105": CBAMSector.FERTILIZERS,
    # Electricity
    "2716": CBAMSector.ELECTRICITY,
    # Hydrogen
    "2804": CBAMSector.HYDROGEN,
}

CBAM_PRODUCT_GROUPS: List[str] = [
    "cement",
    "iron_steel",
    "aluminium",
    "fertilizers",
    "electricity",
    "hydrogen",
]

# EORI pattern: 2-letter ISO country code followed by up to 15 alphanumeric chars
EORI_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{1,15}$")

# ISO 3166-1 alpha-2 country code pattern
ISO_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")

# CN code: exactly 8 digits
CN_CODE_PATTERN = re.compile(r"^[0-9]{8}$")

# Reporting period: YYYYQN format
REPORTING_PERIOD_PATTERN = re.compile(r"^20[2-9][0-9]Q[1-4]$")

# Phase-in markup schedule: Year -> fraction of EU ETS price as carbon price threshold
DEFAULT_MARKUP_SCHEDULE: Dict[int, Decimal] = {
    2026: Decimal("0.10"),
    2027: Decimal("0.20"),
    2028: Decimal("0.30"),
}

# Materiality threshold: emissions below 5% of total may use default values
MATERIALITY_THRESHOLD: Decimal = Decimal("0.05")

# Valid ISO 3166-1 alpha-2 country codes for EORI validation
# (subset of most common; production should use a full list)
VALID_ISO_COUNTRY_CODES = frozenset({
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT",
    "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI",
    "BJ", "BL", "BM", "BN", "BO", "BQ", "BR", "BS", "BT", "BV", "BW", "BY",
    "BZ", "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN",
    "CO", "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM",
    "DO", "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK",
    "FM", "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL",
    "GM", "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM",
    "HN", "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR",
    "IS", "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN",
    "KP", "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS",
    "LT", "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK",
    "ML", "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW",
    "MX", "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP",
    "NR", "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM",
    "PN", "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW",
    "SA", "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM",
    "SN", "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF",
    "TG", "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW",
    "TZ", "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI",
    "VN", "VU", "WF", "WS", "XK", "YE", "YT", "ZA", "ZM", "ZW",
})

# Default decimal precision context
DECIMAL_PLACES = 6
DECIMAL_QUANTIZE = Decimal(10) ** -DECIMAL_PLACES

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard CBAM precision."""
    return value.quantize(DECIMAL_QUANTIZE, rounding=ROUND_HALF_UP)


def _utc_now() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _generate_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


# ============================================================================
# PYDANTIC MODELS - Supporting Types
# ============================================================================


class ValidationIssue(GreenLangBase):
    """A single validation issue found during emissions data checks."""

    field: str = Field(..., description="Field name that failed validation")
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    severity: str = Field(
        "error",
        description="Severity level: error, warning, or info",
    )

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: str) -> str:
        """Ensure severity is one of the allowed values."""
        allowed = {"error", "warning", "info"}
        if v not in allowed:
            raise ValueError(f"severity must be one of {allowed}")
        return v


class AccessRequest(GreenLangBase):
    """An importer's request for data access to a supplier's emissions data."""

    request_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique identifier for this access request",
    )
    importer_id: str = Field(..., description="EU importer identifier")
    importer_name: str = Field(..., description="Legal name of the EU importer")
    supplier_id: str = Field(..., description="Target supplier identifier")
    installation_ids: Optional[List[str]] = Field(
        None,
        description="Specific installations requested; None means all",
    )
    scope: str = Field(
        "emissions_data",
        description="Scope of access: emissions_data, verification_reports, or all",
    )
    purpose: str = Field(
        ...,
        description="Purpose of the access request (e.g. CBAM quarterly reporting)",
    )
    status: AccessRequestStatus = Field(
        default=AccessRequestStatus.PENDING,
        description="Current status of the access request",
    )
    requested_at: datetime = Field(
        default_factory=_utc_now,
        description="When the request was created",
    )
    resolved_at: Optional[datetime] = Field(
        None,
        description="When the request was approved/denied",
    )
    access_duration_days: int = Field(
        365,
        ge=1,
        le=730,
        description="Duration of access in days",
    )
    restrictions: Optional[List[str]] = Field(
        None,
        description="Restrictions placed on the data access",
    )
    notes: Optional[str] = Field(None, max_length=2000)


class AccessEvent(GreenLangBase):
    """An audit log entry for data access events."""

    event_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique event identifier",
    )
    timestamp: datetime = Field(
        default_factory=_utc_now,
        description="When the event occurred",
    )
    actor_id: str = Field(..., description="ID of the entity performing the action")
    actor_type: str = Field(
        ...,
        description="Type of actor: importer, supplier, or system",
    )
    action: str = Field(
        ...,
        description="Action performed: view, download, export, revoke",
    )
    resource_type: str = Field(
        ...,
        description="Type of resource accessed: submission, installation, report",
    )
    resource_id: str = Field(..., description="ID of the resource accessed")
    supplier_id: str = Field(..., description="Supplier whose data was accessed")
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional event details",
    )
    ip_address: Optional[str] = Field(None, description="Client IP address")


class Deadline(GreenLangBase):
    """An upcoming deadline for the supplier."""

    deadline_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique deadline identifier",
    )
    deadline_type: str = Field(
        ...,
        description="Type: submission_due, verification_expiry, report_due",
    )
    description: str = Field(..., description="Human-readable deadline description")
    due_date: date = Field(..., description="Deadline date")
    installation_id: Optional[str] = Field(
        None,
        description="Related installation, if applicable",
    )
    reporting_period: Optional[str] = Field(
        None,
        description="Related reporting period in YYYYQN format",
    )
    is_overdue: bool = Field(False, description="Whether this deadline has passed")
    priority: str = Field(
        "medium",
        description="Priority level: high, medium, low",
    )


# ============================================================================
# PYDANTIC MODELS - Core Entities
# ============================================================================


class PrecursorEmission(GreenLangBase):
    """
    Emissions data for a precursor material used in production.

    CBAM requires tracking of embedded emissions from precursor materials
    (e.g., clinker in cement, pig iron in steel) that are themselves
    CBAM-covered goods.
    """

    precursor_cn_code: str = Field(
        ...,
        description="8-digit CN code of the precursor material",
    )
    precursor_name: str = Field(
        ...,
        max_length=200,
        description="Name of the precursor material",
    )
    quantity_per_unit: Decimal = Field(
        ...,
        ge=0,
        description="Quantity of precursor per unit of final product (tonnes/tonne)",
    )
    embedded_emissions_tCO2e_per_mt: Decimal = Field(
        ...,
        ge=0,
        description="Embedded emissions of the precursor in tCO2e per metric tonne",
    )
    origin_installation_id: Optional[str] = Field(
        None,
        description="Installation ID where the precursor was produced",
    )

    @field_validator("precursor_cn_code")
    @classmethod
    def validate_cn_code(cls, v: str) -> str:
        """Validate 8-digit CN code format."""
        if not CN_CODE_PATTERN.match(v):
            raise ValueError("precursor_cn_code must be exactly 8 digits")
        return v

    @field_validator("quantity_per_unit", "embedded_emissions_tCO2e_per_mt")
    @classmethod
    def quantize_decimal(cls, v: Decimal) -> Decimal:
        """Quantize decimal values to standard precision."""
        try:
            return _quantize(v)
        except InvalidOperation:
            raise ValueError(f"Invalid decimal value: {v}")


class SupportingDocument(GreenLangBase):
    """A supporting document attached to an emissions submission."""

    doc_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique document identifier",
    )
    submission_id: str = Field(
        ...,
        description="ID of the parent submission",
    )
    doc_type: EvidenceType = Field(
        ...,
        description="Type of evidence document",
    )
    filename: str = Field(
        ...,
        max_length=500,
        description="Original filename",
    )
    upload_date: datetime = Field(
        default_factory=_utc_now,
        description="When the document was uploaded",
    )
    file_hash_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the file content for integrity verification",
    )
    file_size_bytes: Optional[int] = Field(
        None,
        ge=0,
        description="File size in bytes",
    )
    content_type: Optional[str] = Field(
        None,
        max_length=100,
        description="MIME content type",
    )

    @field_validator("file_hash_sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        """Validate SHA-256 hash format."""
        if not re.match(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError("file_hash_sha256 must be a 64-character hex string")
        return v.lower()


class DataQualityScore(GreenLangBase):
    """
    Data quality assessment for an emissions submission.

    Each dimension is scored 0-100. The overall score is the weighted average
    of completeness (30%), consistency (25%), timeliness (20%), accuracy (25%).
    """

    completeness: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Completeness score (0-100): all required fields present",
    )
    consistency: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Consistency score (0-100): values internally consistent",
    )
    timeliness: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Timeliness score (0-100): data submitted within deadline",
    )
    accuracy: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Accuracy score (0-100): values verified/plausible",
    )
    overall: Decimal = Field(
        ...,
        ge=0,
        le=100,
        description="Weighted overall score (0-100)",
    )

    @model_validator(mode="after")
    def validate_overall_score(self) -> "DataQualityScore":
        """Validate that the overall score is plausibly derived from dimensions."""
        # Compute expected weighted average for cross-check
        expected = _quantize(
            self.completeness * Decimal("0.30")
            + self.consistency * Decimal("0.25")
            + self.timeliness * Decimal("0.20")
            + self.accuracy * Decimal("0.25")
        )
        # Allow 1-point tolerance for manually adjusted scores
        if abs(self.overall - expected) > Decimal("1.0"):
            # Auto-correct to computed weighted average
            object.__setattr__(self, "overall", expected)
        return self

    @field_validator(
        "completeness", "consistency", "timeliness", "accuracy", "overall"
    )
    @classmethod
    def quantize_score(cls, v: Decimal) -> Decimal:
        """Quantize scores to standard precision."""
        try:
            return _quantize(v)
        except InvalidOperation:
            raise ValueError(f"Invalid decimal score: {v}")


class VerificationRecord(GreenLangBase):
    """
    Record of a third-party verification visit for an installation.

    CBAM requires that installation-level emissions data be verified by
    an accredited verifier from a recognized National Accreditation Body.
    """

    verification_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique verification record identifier",
    )
    installation_id: str = Field(
        ...,
        description="Installation that was verified",
    )
    verifier_name: str = Field(
        ...,
        max_length=300,
        description="Name of the verification body/person",
    )
    verifier_accreditation: str = Field(
        ...,
        max_length=200,
        description="Accreditation number or reference",
    )
    nab_country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country of the National Accreditation Body (ISO 3166-1 alpha-2)",
    )
    verification_date: date = Field(
        ...,
        description="Date the verification was conducted",
    )
    next_visit_date: Optional[date] = Field(
        None,
        description="Scheduled date for the next verification visit",
    )
    visit_type: VisitType = Field(
        ...,
        description="Type of visit: on_site or remote",
    )
    outcome: VerificationOutcome = Field(
        ...,
        description="Verification outcome: pass, fail, or conditional",
    )
    materiality_assessment: Optional[str] = Field(
        None,
        max_length=2000,
        description="Verifier's materiality assessment notes",
    )
    report_reference: Optional[str] = Field(
        None,
        max_length=200,
        description="Reference to the full verification report",
    )
    conditions: Optional[List[str]] = Field(
        None,
        description="Conditions attached to a conditional outcome",
    )

    @field_validator("nab_country")
    @classmethod
    def validate_nab_country(cls, v: str) -> str:
        """Validate NAB country code."""
        v = v.upper()
        if not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("nab_country must be a 2-letter ISO country code")
        return v

    @model_validator(mode="after")
    def validate_dates(self) -> "VerificationRecord":
        """Ensure next_visit_date is after verification_date."""
        if self.next_visit_date and self.next_visit_date <= self.verification_date:
            raise ValueError(
                "next_visit_date must be after verification_date"
            )
        return self


class Installation(GreenLangBase):
    """
    A production installation (facility) registered under the Supplier Portal.

    Each installation belongs to a single supplier and may produce goods
    in one or more CBAM sectors.
    """

    installation_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique installation identifier",
    )
    supplier_id: str = Field(
        ...,
        description="Parent supplier identifier",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Installation facility name",
    )
    address: str = Field(
        ...,
        max_length=500,
        description="Physical address of the installation",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country where installation is located (ISO 3166-1 alpha-2)",
    )
    installation_type: InstallationType = Field(
        ...,
        description="Type of production installation",
    )
    cbam_sectors: List[CBAMSector] = Field(
        ...,
        min_length=1,
        description="CBAM sectors this installation produces goods for",
    )
    production_processes: Optional[List[str]] = Field(
        None,
        description="Production processes at this installation",
    )
    capacity_mt_per_year: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Annual production capacity in metric tonnes",
    )
    verified_until: Optional[date] = Field(
        None,
        description="Date until which the installation verification is valid",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Current verification status",
    )
    cn_codes: Optional[List[str]] = Field(
        None,
        description="CN codes of products manufactured at this installation",
    )
    energy_source: Optional[str] = Field(
        None,
        max_length=200,
        description="Primary energy source",
    )
    monitoring_methodology: Optional[str] = Field(
        None,
        max_length=500,
        description="Emissions monitoring methodology used",
    )
    is_active: bool = Field(True, description="Whether the installation is active")
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Registration timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp",
    )

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate ISO country code format."""
        v = v.upper()
        if not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("country must be a 2-letter ISO country code")
        return v

    @field_validator("cn_codes")
    @classmethod
    def validate_cn_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate CN code format."""
        if v is not None:
            invalid = [c for c in v if not CN_CODE_PATTERN.match(c)]
            if invalid:
                raise ValueError(f"CN codes must be exactly 8 digits: {invalid}")
        return v

    @field_validator("capacity_mt_per_year")
    @classmethod
    def quantize_capacity(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize capacity to standard precision."""
        if v is not None:
            try:
                return _quantize(v)
            except InvalidOperation:
                raise ValueError(f"Invalid decimal capacity: {v}")
        return v


class EmissionsDataSubmission(GreenLangBase):
    """
    An emissions data submission from a supplier for a specific product
    at a specific installation during a reporting period.

    This is the core data exchange record between third-country operators
    and EU importers under CBAM.
    """

    submission_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique submission identifier",
    )
    installation_id: str = Field(
        ...,
        description="Installation producing the goods",
    )
    reporting_period_year: int = Field(
        ...,
        ge=2023,
        le=2040,
        description="Reporting year",
    )
    reporting_period_quarter: int = Field(
        ...,
        ge=1,
        le=4,
        description="Reporting quarter (1-4)",
    )
    cn_code: str = Field(
        ...,
        description="8-digit CN code of the product",
    )
    product_description: str = Field(
        ...,
        max_length=500,
        description="Human-readable product description",
    )
    quantity_mt: Decimal = Field(
        ...,
        gt=0,
        description="Quantity of goods in metric tonnes",
    )
    direct_emissions_tCO2e_per_mt: Decimal = Field(
        ...,
        ge=0,
        description="Direct (Scope 1) specific embedded emissions in tCO2e/mt",
    )
    indirect_emissions_tCO2e_per_mt: Decimal = Field(
        ...,
        ge=0,
        description="Indirect (Scope 2) specific embedded emissions in tCO2e/mt",
    )
    total_embedded_emissions_tCO2e_per_mt: Decimal = Field(
        ...,
        ge=0,
        description="Total specific embedded emissions in tCO2e/mt",
    )
    calculation_method: CalculationMethod = Field(
        ...,
        description="Emissions calculation methodology used",
    )
    electricity_source: Optional[str] = Field(
        None,
        max_length=200,
        description="Source of electricity (grid, PPA, on-site generation)",
    )
    grid_emission_factor: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Grid emission factor in tCO2e/MWh",
    )
    precursor_emissions: Optional[List[PrecursorEmission]] = Field(
        None,
        description="Precursor material embedded emissions",
    )
    supporting_docs: Optional[List[SupportingDocument]] = Field(
        None,
        description="Supporting evidence documents",
    )
    submission_status: SubmissionStatus = Field(
        default=SubmissionStatus.DRAFT,
        description="Current submission lifecycle status",
    )
    version: int = Field(
        default=1,
        ge=1,
        description="Submission version (increments on amendment)",
    )
    submitted_at: Optional[datetime] = Field(
        None,
        description="When the submission was officially submitted",
    )
    reviewed_at: Optional[datetime] = Field(
        None,
        description="When the submission was reviewed",
    )
    reviewer_notes: Optional[str] = Field(
        None,
        max_length=2000,
        description="Notes from the reviewer",
    )
    reviewer_id: Optional[str] = Field(
        None,
        description="ID of the reviewer",
    )
    carbon_price_paid_eur_per_tco2: Optional[Decimal] = Field(
        None,
        ge=0,
        description="Carbon price already paid in country of origin (EUR/tCO2e)",
    )
    carbon_price_instrument: Optional[str] = Field(
        None,
        max_length=200,
        description="Name of the carbon pricing instrument",
    )
    data_quality_score: Optional[DataQualityScore] = Field(
        None,
        description="Computed data quality assessment",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code(cls, v: str) -> str:
        """Validate 8-digit CN code format."""
        if not CN_CODE_PATTERN.match(v):
            raise ValueError("cn_code must be exactly 8 digits")
        return v

    @field_validator(
        "quantity_mt",
        "direct_emissions_tCO2e_per_mt",
        "indirect_emissions_tCO2e_per_mt",
        "total_embedded_emissions_tCO2e_per_mt",
        "grid_emission_factor",
        "carbon_price_paid_eur_per_tco2",
    )
    @classmethod
    def quantize_emissions(cls, v: Optional[Decimal]) -> Optional[Decimal]:
        """Quantize emissions values to standard precision."""
        if v is not None:
            try:
                return _quantize(v)
            except InvalidOperation:
                raise ValueError(f"Invalid decimal value: {v}")
        return v

    @model_validator(mode="after")
    def validate_total_emissions(self) -> "EmissionsDataSubmission":
        """
        Validate that total_embedded_emissions equals
        direct + indirect + sum(precursor emissions).
        """
        precursor_sum = Decimal("0")
        if self.precursor_emissions:
            for p in self.precursor_emissions:
                contribution = _quantize(
                    p.quantity_per_unit * p.embedded_emissions_tCO2e_per_mt
                )
                precursor_sum += contribution
            precursor_sum = _quantize(precursor_sum)

        expected_total = _quantize(
            self.direct_emissions_tCO2e_per_mt
            + self.indirect_emissions_tCO2e_per_mt
            + precursor_sum
        )

        # Allow a small tolerance for rounding
        tolerance = Decimal("0.01")
        if abs(self.total_embedded_emissions_tCO2e_per_mt - expected_total) > tolerance:
            # Auto-correct total to computed value
            object.__setattr__(
                self,
                "total_embedded_emissions_tCO2e_per_mt",
                expected_total,
            )
        return self

    @model_validator(mode="after")
    def validate_reporting_period(self) -> "EmissionsDataSubmission":
        """Validate reporting period is within CBAM scope."""
        if self.reporting_period_year < 2023:
            raise ValueError("Reporting period cannot be before 2023 (CBAM start)")
        return self

    @property
    def reporting_period(self) -> str:
        """Return reporting period as YYYYQN string."""
        return f"{self.reporting_period_year}Q{self.reporting_period_quarter}"

    @property
    def total_absolute_emissions_tco2e(self) -> Decimal:
        """Calculate total absolute emissions: specific * quantity."""
        return _quantize(
            self.total_embedded_emissions_tCO2e_per_mt * self.quantity_mt
        )


class SupplierProfile(GreenLangBase):
    """
    A registered third-country supplier (installation operator) in the
    CBAM Supplier Portal.

    The profile tracks company details, EORI number, verification status,
    and linked installations.
    """

    supplier_id: str = Field(
        default_factory=_generate_uuid,
        description="Unique supplier identifier",
    )
    company_name: str = Field(
        ...,
        min_length=1,
        max_length=300,
        description="Legal company name",
    )
    eori_number: Optional[str] = Field(
        None,
        max_length=17,
        description="EORI number (2-letter country + up to 15 alphanumeric)",
    )
    country: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="Country of registration (ISO 3166-1 alpha-2)",
    )
    address: str = Field(
        ...,
        max_length=500,
        description="Registered address",
    )
    contact_email: str = Field(
        ...,
        max_length=254,
        description="Primary contact email address",
    )
    status: SupplierStatus = Field(
        default=SupplierStatus.REGISTERED,
        description="Current supplier lifecycle status",
    )
    installations: List[Installation] = Field(
        default_factory=list,
        description="List of registered installations",
    )
    registration_date: datetime = Field(
        default_factory=_utc_now,
        description="When the supplier was registered",
    )
    verification_expiry: Optional[date] = Field(
        None,
        description="Date when current verification expires",
    )
    cbam_sectors: Optional[List[CBAMSector]] = Field(
        None,
        description="CBAM sectors the supplier operates in",
    )
    tax_id: Optional[str] = Field(
        None,
        max_length=50,
        description="Tax identification number",
    )
    certifications: Optional[List[str]] = Field(
        None,
        description="Environmental certifications held",
    )
    is_active: bool = Field(True, description="Whether the supplier is active")
    linked_importers: List[str] = Field(
        default_factory=list,
        description="IDs of EU importers authorized to access this supplier's data",
    )
    created_at: datetime = Field(
        default_factory=_utc_now,
        description="Record creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=_utc_now,
        description="Last update timestamp",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash for audit trail",
    )

    @field_validator("eori_number")
    @classmethod
    def validate_eori(cls, v: Optional[str]) -> Optional[str]:
        """Validate EORI number format when provided."""
        if v is not None:
            v = v.upper()
            if not EORI_PATTERN.match(v):
                raise ValueError(
                    "EORI must match: 2-letter country code + 1-15 alphanumeric chars"
                )
        return v

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Validate ISO country code format."""
        v = v.upper()
        if not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("country must be a 2-letter ISO country code")
        return v

    @field_validator("contact_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email format validation."""
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("contact_email must be a valid email address")
        return v.lower()


class SupplierSearchResult(GreenLangBase):
    """
    A summarized supplier record returned from search queries.

    Contains only the fields necessary for search result display,
    avoiding full profile loading for performance.
    """

    supplier_id: str = Field(..., description="Unique supplier identifier")
    company_name: str = Field(..., description="Legal company name")
    country: str = Field(..., description="Country code (ISO 3166-1 alpha-2)")
    sectors: List[CBAMSector] = Field(
        default_factory=list,
        description="CBAM sectors the supplier operates in",
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Current verification status",
    )
    installations_count: int = Field(
        default=0,
        ge=0,
        description="Number of registered installations",
    )
    last_submission_date: Optional[datetime] = Field(
        None,
        description="Date of the most recent emissions submission",
    )


class SupplierDashboard(GreenLangBase):
    """
    Aggregated dashboard data for a supplier.

    Provides a comprehensive overview of the supplier's status,
    submissions, data quality, deadlines, and importer relationships.
    """

    supplier_id: str = Field(..., description="Supplier identifier")
    company_name: str = Field(..., description="Company name")
    status: SupplierStatus = Field(..., description="Current supplier status")
    installations_count: int = Field(0, ge=0, description="Number of installations")
    total_submissions: int = Field(0, ge=0, description="Total submission count")
    submission_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Submissions by status",
    )
    period_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Submissions by reporting period",
    )
    data_quality_overview: Optional[DataQualityScore] = Field(
        None,
        description="Average data quality scores",
    )
    upcoming_deadlines: List[Deadline] = Field(
        default_factory=list,
        description="Upcoming deadlines",
    )
    verification_timeline: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verification status and next visit",
    )
    emissions_trend: Dict[str, Any] = Field(
        default_factory=dict,
        description="Emissions data by period",
    )
    linked_importers_count: int = Field(
        0,
        ge=0,
        description="Number of linked importers",
    )
    recent_activity: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activity log entries",
    )
    generated_at: datetime = Field(
        default_factory=_utc_now,
        description="When the dashboard data was generated",
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 provenance hash",
    )

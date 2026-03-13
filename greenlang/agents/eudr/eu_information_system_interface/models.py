# -*- coding: utf-8 -*-
"""
EU Information System Interface Models - AGENT-EUDR-036

Pydantic v2 models for Due Diligence Statement (DDS) submission,
operator registration, geolocation formatting, document package assembly,
submission status tracking, EU Information System API interaction, and
Article 31 audit recording.

All models use Decimal for numeric values to ensure deterministic,
bit-perfect reproducibility in compliance data submissions.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-036 EU Information System Interface (GL-EUDR-EUIS-036)
Regulation: EU 2023/1115 (EUDR) Articles 4, 12, 13, 14, 31, 33
Status: Production Ready
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EUDRCommodity(str, enum.Enum):
    """EUDR regulated commodities (Article 1)."""

    CATTLE = "cattle"
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    RUBBER = "rubber"
    SOYA = "soya"
    WOOD = "wood"


class OperatorType(str, enum.Enum):
    """Operator classification under EUDR Articles 4-6."""

    OPERATOR = "operator"
    TRADER = "trader"
    SME_OPERATOR = "sme_operator"
    SME_TRADER = "sme_trader"


class DDSType(str, enum.Enum):
    """Due Diligence Statement types per EUDR Article 4."""

    PLACING = "placing"
    MAKING_AVAILABLE = "making_available"
    EXPORT = "export"


class DDSStatus(str, enum.Enum):
    """DDS lifecycle states in the EU Information System."""

    DRAFT = "draft"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    RECEIVED = "received"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    AMENDED = "amended"
    EXPIRED = "expired"


class SubmissionStatus(str, enum.Enum):
    """Submission request lifecycle status."""

    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class RegistrationStatus(str, enum.Enum):
    """Operator registration status in EU IS."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    EXPIRED = "expired"
    REVOKED = "revoked"


class GeolocationFormat(str, enum.Enum):
    """Geolocation data format types per EUDR Annex II."""

    POINT = "point"
    POLYGON = "polygon"
    MULTIPOLYGON = "multipolygon"


class CoordinateSystem(str, enum.Enum):
    """Supported coordinate reference systems."""

    WGS84 = "EPSG:4326"
    WEB_MERCATOR = "EPSG:3857"


class DocumentType(str, enum.Enum):
    """Document types included in DDS packages."""

    DDS_FORM = "dds_form"
    RISK_ASSESSMENT = "risk_assessment"
    MITIGATION_REPORT = "mitigation_report"
    GEOLOCATION_DATA = "geolocation_data"
    SUPPLY_CHAIN_MAP = "supply_chain_map"
    CERTIFICATE = "certificate"
    AUDIT_REPORT = "audit_report"
    SATELLITE_IMAGERY = "satellite_imagery"
    LEGAL_COMPLIANCE = "legal_compliance"
    IMPROVEMENT_PLAN = "improvement_plan"
    EVIDENCE_PACKAGE = "evidence_package"
    OPERATOR_DECLARATION = "operator_declaration"


class AuditEventType(str, enum.Enum):
    """Article 31 audit event types."""

    DDS_CREATED = "dds_created"
    DDS_VALIDATED = "dds_validated"
    DDS_SUBMITTED = "dds_submitted"
    DDS_RECEIVED = "dds_received"
    DDS_ACCEPTED = "dds_accepted"
    DDS_REJECTED = "dds_rejected"
    DDS_WITHDRAWN = "dds_withdrawn"
    DDS_AMENDED = "dds_amended"
    OPERATOR_REGISTERED = "operator_registered"
    OPERATOR_UPDATED = "operator_updated"
    GEOLOCATION_FORMATTED = "geolocation_formatted"
    PACKAGE_ASSEMBLED = "package_assembled"
    API_CALL_MADE = "api_call_made"
    API_CALL_FAILED = "api_call_failed"
    STATUS_CHECKED = "status_checked"
    STATUS_CHANGED = "status_changed"


class CompetentAuthority(str, enum.Enum):
    """EU Member State competent authority references."""

    DE = "DE"
    FR = "FR"
    IT = "IT"
    ES = "ES"
    NL = "NL"
    BE = "BE"
    AT = "AT"
    PL = "PL"
    SE = "SE"
    DK = "DK"
    FI = "FI"
    PT = "PT"
    IE = "IE"
    CZ = "CZ"
    RO = "RO"
    HU = "HU"
    BG = "BG"
    HR = "HR"
    SK = "SK"
    LT = "LT"
    SI = "SI"
    LV = "LV"
    EE = "EE"
    CY = "CY"
    LU = "LU"
    MT = "MT"
    EL = "EL"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_ID = "GL-EUDR-EUIS-036"
AGENT_VERSION = "1.0.0"

SUPPORTED_COMMODITIES: List[str] = [c.value for c in EUDRCommodity]

# Article 4(2): DDS reference number format
DDS_REFERENCE_PREFIX = "EUDR-DDS"

# Article 31: Minimum retention period
MIN_AUDIT_RETENTION_YEARS = 5

# Annex II: Maximum geolocation points for single plot
MAX_GEOLOCATION_POINTS = 500

# Article 12: Required DDS fields per regulation
REQUIRED_DDS_FIELDS = [
    "operator_id",
    "eori_number",
    "commodity",
    "description",
    "quantity",
    "country_of_production",
    "geolocation",
    "risk_assessment_conclusion",
    "dds_type",
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class Coordinate(BaseModel):
    """Geographic coordinate with WGS84 precision.

    Per EUDR Annex II, latitude and longitude must be provided
    in decimal degrees with sufficient precision for plot identification.
    """

    latitude: Decimal = Field(
        ..., ge=-90, le=90, description="Latitude in decimal degrees"
    )
    longitude: Decimal = Field(
        ..., ge=-180, le=180, description="Longitude in decimal degrees"
    )

    model_config = {"frozen": False, "extra": "ignore"}


class GeoPolygon(BaseModel):
    """Geographic polygon boundary for plot identification.

    Per EUDR Annex II, plots of land above 4 hectares require polygon
    coordinates rather than single point geolocation.
    """

    coordinates: List[Coordinate] = Field(
        ...,
        min_length=3,
        description="Polygon vertices (min 3 points, must close)",
    )
    area_hectares: Optional[Decimal] = Field(
        default=None, ge=0, description="Area in hectares"
    )
    crs: str = Field(
        default="EPSG:4326",
        description="Coordinate Reference System",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class GeolocationData(BaseModel):
    """Geolocation data for DDS submission per EUDR Annex II.

    Supports point, polygon, and multipolygon formats depending
    on the plot size and number of production areas.
    """

    format: GeolocationFormat = Field(
        ..., description="Geolocation format type"
    )
    point: Optional[Coordinate] = Field(
        default=None, description="Point coordinate (plots < 4 ha)"
    )
    polygon: Optional[GeoPolygon] = Field(
        default=None, description="Polygon boundary (plots >= 4 ha)"
    )
    polygons: List[GeoPolygon] = Field(
        default_factory=list,
        description="Multiple polygons for multi-plot declarations",
    )
    country_code: str = Field(
        ..., min_length=2, max_length=2, description="ISO 3166-1 alpha-2"
    )
    region: str = Field(
        default="", description="Sub-national region or province"
    )
    formatted_for_eu: bool = Field(
        default=False,
        description="Whether coordinates have been formatted to EU specs",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class OperatorRegistration(BaseModel):
    """Operator registration record for the EU Information System.

    Per EUDR Article 4, all operators and traders must register
    in the EU Information System before submitting DDS.
    """

    registration_id: str = Field(
        ..., description="Internal registration identifier"
    )
    operator_id: str = Field(
        ..., description="GreenLang operator identifier"
    )
    eori_number: str = Field(
        ..., description="Economic Operators Registration and Identification"
    )
    operator_type: OperatorType = Field(
        ..., description="Operator classification"
    )
    company_name: str = Field(..., description="Legal entity name")
    member_state: CompetentAuthority = Field(
        ..., description="EU Member State of registration"
    )
    address: str = Field(default="", description="Registered address")
    contact_email: str = Field(default="", description="Contact email")
    registration_status: RegistrationStatus = RegistrationStatus.PENDING
    eu_system_id: Optional[str] = Field(
        default=None,
        description="Assigned ID from EU Information System",
    )
    registered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DDSCommodityLine(BaseModel):
    """Individual commodity line item within a DDS.

    Each DDS may contain multiple commodity lines for different
    products or production sources.
    """

    line_id: str = Field(..., description="Line item identifier")
    commodity: EUDRCommodity = Field(
        ..., description="EUDR regulated commodity"
    )
    hs_code: str = Field(
        default="", description="Harmonized System commodity code"
    )
    description: str = Field(
        ..., description="Product description"
    )
    quantity: Decimal = Field(
        ..., gt=0, description="Quantity in specified unit"
    )
    unit: str = Field(
        default="kg", description="Unit of measurement"
    )
    country_of_production: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    geolocation: GeolocationData = Field(
        ..., description="Geolocation of production area"
    )
    supplier_ids: List[str] = Field(
        default_factory=list,
        description="Supply chain supplier identifiers",
    )
    risk_assessment_conclusion: str = Field(
        default="negligible",
        description="Risk assessment result (negligible/low/standard/high)",
    )

    model_config = {"frozen": False, "extra": "ignore"}


class DueDiligenceStatement(BaseModel):
    """Complete Due Diligence Statement for EU Information System.

    Per EUDR Articles 4 and 12, operators must submit a DDS before
    placing, making available, or exporting relevant commodities.
    The DDS references the operator, commodity details, geolocation
    of production, risk assessment conclusions, and supporting evidence.
    """

    dds_id: str = Field(..., description="Internal DDS identifier")
    dds_reference: str = Field(
        default="", description="EU IS reference number (after submission)"
    )
    operator_id: str = Field(..., description="GreenLang operator ID")
    eori_number: str = Field(..., description="EORI number")
    dds_type: DDSType = Field(..., description="DDS type per Article 4")
    status: DDSStatus = DDSStatus.DRAFT
    commodity_lines: List[DDSCommodityLine] = Field(
        default_factory=list,
        description="Commodity line items",
    )
    total_quantity: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total quantity across all lines",
    )
    risk_assessment_id: Optional[str] = Field(
        default=None, description="Reference to risk assessment"
    )
    mitigation_plan_id: Optional[str] = Field(
        default=None, description="Reference to mitigation plan if applicable"
    )
    improvement_plan_id: Optional[str] = Field(
        default=None, description="Reference to improvement plan (EUDR-035)"
    )
    competent_authority: Optional[CompetentAuthority] = None
    declaration_date: Optional[datetime] = None
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None
    rejected_reason: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class DocumentPackage(BaseModel):
    """Assembled document package for DDS submission.

    Contains all supporting documents required for a DDS submission,
    including risk assessments, mitigation reports, evidence files,
    and geolocation data. Each document is hashed for integrity.
    """

    package_id: str = Field(..., description="Package identifier")
    dds_id: str = Field(..., description="Associated DDS identifier")
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of document references with metadata",
    )
    total_size_bytes: int = Field(
        default=0, ge=0, description="Total package size in bytes"
    )
    document_count: int = Field(
        default=0, ge=0, description="Number of documents"
    )
    compressed: bool = Field(
        default=False, description="Whether package is compressed"
    )
    assembled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class SubmissionRequest(BaseModel):
    """DDS submission request to the EU Information System.

    Tracks the full submission lifecycle from creation through
    confirmation, including retry state and error details.
    """

    submission_id: str = Field(
        ..., description="Internal submission identifier"
    )
    dds_id: str = Field(..., description="DDS being submitted")
    package_id: str = Field(
        ..., description="Document package for submission"
    )
    status: SubmissionStatus = SubmissionStatus.PENDING
    eu_reference_number: Optional[str] = Field(
        default=None, description="Reference from EU IS"
    )
    attempt_count: int = Field(
        default=0, ge=0, description="Number of submission attempts"
    )
    last_attempt_at: Optional[datetime] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    submitted_at: Optional[datetime] = None
    confirmed_at: Optional[datetime] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class StatusCheckResult(BaseModel):
    """Result from checking DDS status on the EU Information System."""

    check_id: str = Field(..., description="Status check identifier")
    dds_id: str = Field(..., description="DDS being checked")
    eu_reference: str = Field(
        ..., description="EU IS reference number"
    )
    previous_status: DDSStatus = Field(
        ..., description="Status before this check"
    )
    current_status: DDSStatus = Field(
        ..., description="Current status from EU IS"
    )
    status_changed: bool = Field(
        default=False, description="Whether status changed"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional status details from EU IS",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class AuditRecord(BaseModel):
    """Article 31 audit trail record.

    EUDR Article 31 requires operators to maintain records of
    due diligence statements and supporting information for at
    least 5 years, making them available to competent authorities
    on request.
    """

    audit_id: str = Field(..., description="Audit record identifier")
    event_type: AuditEventType = Field(
        ..., description="Type of audited event"
    )
    entity_type: str = Field(
        ..., description="Entity type (dds/operator/submission)"
    )
    entity_id: str = Field(
        ..., description="Entity identifier"
    )
    actor: str = Field(
        ..., description="User or system performing the action"
    )
    action: str = Field(
        ..., description="Action performed"
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event details and context",
    )
    request_summary: Optional[str] = Field(
        default=None, description="API request summary if applicable"
    )
    response_summary: Optional[str] = Field(
        default=None, description="API response summary if applicable"
    )
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    retention_until: Optional[datetime] = None
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class APICallRecord(BaseModel):
    """Record of an API call to the EU Information System.

    Tracks request/response details, timing, and outcome for
    audit trail and debugging purposes.
    """

    call_id: str = Field(..., description="API call identifier")
    method: str = Field(..., description="HTTP method (GET/POST/PUT/PATCH)")
    endpoint: str = Field(..., description="API endpoint path")
    status_code: int = Field(
        default=0, ge=0, description="HTTP response status code"
    )
    request_size_bytes: int = Field(
        default=0, ge=0, description="Request body size"
    )
    response_size_bytes: int = Field(
        default=0, ge=0, description="Response body size"
    )
    duration_ms: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Call duration in milliseconds",
    )
    success: bool = Field(default=False, description="Whether call succeeded")
    error_message: Optional[str] = None
    retry_count: int = Field(
        default=0, ge=0, description="Number of retries"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"frozen": False, "extra": "ignore"}


class DDSSummary(BaseModel):
    """Summary of a DDS for listing and reporting."""

    dds_id: str
    dds_reference: str = ""
    operator_id: str
    dds_type: DDSType
    status: DDSStatus
    commodity_count: int = 0
    total_quantity: Decimal = Decimal("0")
    submitted_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None

    model_config = {"frozen": False, "extra": "ignore"}


class SubmissionReport(BaseModel):
    """Submission report aggregating DDS submission outcomes.

    Provides summary statistics for batch submission operations
    including success/failure rates and timing data.
    """

    report_id: str = Field(..., description="Report identifier")
    total_submissions: int = Field(default=0, ge=0)
    successful: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    pending: int = Field(default=0, ge=0)
    average_duration_ms: Decimal = Field(default=Decimal("0"), ge=0)
    submissions: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    model_config = {"frozen": False, "extra": "ignore"}


class HealthStatus(BaseModel):
    """Health check response for the EU Information System Interface."""

    agent_id: str = AGENT_ID
    status: str = "healthy"
    version: str = AGENT_VERSION
    engines: Dict[str, str] = Field(default_factory=dict)
    eu_api_reachable: bool = False
    database: bool = False
    redis: bool = False
    uptime_seconds: float = 0.0

    model_config = {"frozen": False, "extra": "ignore"}

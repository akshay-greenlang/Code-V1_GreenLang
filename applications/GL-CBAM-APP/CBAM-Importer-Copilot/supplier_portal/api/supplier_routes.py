# -*- coding: utf-8 -*-
"""
CBAM Supplier Portal API - Supplier Routes

This module implements the complete FastAPI router for the CBAM Supplier Portal,
enabling third-country suppliers to register installations, submit emissions data,
manage data exchange with EU importers, and view dashboard analytics.

Endpoints cover:
  - Supplier Registration & Profile Management
  - Installation Management (production facilities)
  - Emissions Data Submission & Review Workflow
  - Dashboard & Analytics
  - Data Exchange (importer access grants)

All endpoints enforce SHA-256 provenance hashing, standardized error handling,
pagination for list responses, and OpenAPI documentation.

Version: 1.1.0
Author: GreenLang CBAM Team
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone, date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

CBAM_PRODUCT_GROUPS = [
    "cement", "steel", "aluminum", "fertilizers", "hydrogen", "electricity",
]

EORI_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{1,15}$")

ISO_COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")

CN_CODE_PATTERN = re.compile(r"^[0-9]{8}$")


# ============================================================================
# ENUMS
# ============================================================================

class VerificationStatus(str, Enum):
    """Verification status for supplier or installation."""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SubmissionStatus(str, Enum):
    """Status of an emissions data submission."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    AMENDED = "amended"


class DataQuality(str, Enum):
    """Data quality assessment tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    XML = "xml"


class ReviewDecision(str, Enum):
    """Decision made during a submission review."""
    ACCEPT = "accept"
    REJECT = "reject"
    REQUEST_AMENDMENT = "request_amendment"


class AccessRequestStatus(str, Enum):
    """Status of an importer data-access request."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    REVOKED = "revoked"


class AuditAction(str, Enum):
    """Audit log action types."""
    SUPPLIER_REGISTERED = "supplier_registered"
    SUPPLIER_UPDATED = "supplier_updated"
    SUPPLIER_DEACTIVATED = "supplier_deactivated"
    INSTALLATION_CREATED = "installation_created"
    INSTALLATION_UPDATED = "installation_updated"
    EMISSIONS_SUBMITTED = "emissions_submitted"
    EMISSIONS_AMENDED = "emissions_amended"
    EMISSIONS_REVIEWED = "emissions_reviewed"
    ACCESS_REQUESTED = "access_requested"
    ACCESS_APPROVED = "access_approved"
    ACCESS_REVOKED = "access_revoked"
    DOCUMENT_UPLOADED = "document_uploaded"
    DATA_EXPORTED = "data_exported"


# ============================================================================
# REQUEST MODELS - SUPPLIER REGISTRATION
# ============================================================================

class AddressModel(BaseModel):
    """Physical address of a supplier or installation."""
    street: Optional[str] = Field(None, max_length=200, description="Street address")
    city: str = Field(..., max_length=100, description="City name")
    region: Optional[str] = Field(None, max_length=100, description="State, province, or region")
    postal_code: Optional[str] = Field(None, max_length=20, description="Postal / ZIP code")
    country: str = Field(..., max_length=100, description="Country name")


class ContactModel(BaseModel):
    """Primary contact information for CBAM correspondence."""
    person: str = Field(..., max_length=100, description="Contact person full name")
    email: str = Field(..., max_length=100, description="Contact email address")
    phone: Optional[str] = Field(None, max_length=30, description="Phone with country code")
    website: Optional[str] = Field(None, max_length=200, description="Company website URL")


class SupplierRegistrationRequest(BaseModel):
    """Request body for registering a new supplier."""
    company_name: str = Field(
        ..., min_length=1, max_length=200,
        description="Legal company name of the supplier",
    )
    country_iso: str = Field(
        ..., min_length=2, max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    eori_number: Optional[str] = Field(
        None, max_length=17,
        description="EORI number (Economic Operators Registration and Identification)",
    )
    tax_id: Optional[str] = Field(None, max_length=50, description="Tax identification number")
    product_groups: List[str] = Field(
        ..., min_length=1,
        description="CBAM product groups this supplier produces",
    )
    cn_codes_produced: Optional[List[str]] = Field(
        None, description="8-digit CN codes for products manufactured",
    )
    address: Optional[AddressModel] = Field(None, description="Primary facility address")
    contact: ContactModel = Field(..., description="Primary contact information")
    certifications: Optional[List[str]] = Field(
        None, description="Environmental certifications held (e.g. ISO 14001)",
    )
    production_capacity_tons_per_year: Optional[float] = Field(
        None, ge=0, description="Annual production capacity in metric tonnes",
    )
    notes: Optional[str] = Field(None, max_length=2000, description="Additional notes")

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Validate ISO 3166-1 alpha-2 format."""
        if not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("country_iso must be a 2-letter uppercase ISO code (e.g. CN, DE)")
        return v

    @field_validator("eori_number")
    @classmethod
    def validate_eori(cls, v: Optional[str]) -> Optional[str]:
        """Validate EORI number format when provided."""
        if v is not None and not EORI_PATTERN.match(v):
            raise ValueError(
                "EORI must match pattern: 2-letter country prefix + 1-15 alphanumeric chars"
            )
        return v

    @field_validator("product_groups")
    @classmethod
    def validate_product_groups(cls, v: List[str]) -> List[str]:
        """Validate each product group is a recognised CBAM category."""
        invalid = [pg for pg in v if pg not in CBAM_PRODUCT_GROUPS]
        if invalid:
            raise ValueError(
                f"Invalid product groups: {invalid}. "
                f"Allowed: {CBAM_PRODUCT_GROUPS}"
            )
        return v

    @field_validator("cn_codes_produced")
    @classmethod
    def validate_cn_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate 8-digit CN code format."""
        if v is not None:
            invalid = [c for c in v if not CN_CODE_PATTERN.match(c)]
            if invalid:
                raise ValueError(f"CN codes must be exactly 8 digits: {invalid}")
        return v


class SupplierUpdateRequest(BaseModel):
    """Request body for updating an existing supplier profile."""
    company_name: Optional[str] = Field(None, max_length=200)
    country_iso: Optional[str] = Field(None, min_length=2, max_length=2)
    eori_number: Optional[str] = Field(None, max_length=17)
    tax_id: Optional[str] = Field(None, max_length=50)
    product_groups: Optional[List[str]] = Field(None)
    cn_codes_produced: Optional[List[str]] = Field(None)
    address: Optional[AddressModel] = Field(None)
    contact: Optional[ContactModel] = Field(None)
    certifications: Optional[List[str]] = Field(None)
    production_capacity_tons_per_year: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Validate ISO country code when provided."""
        if v is not None and not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("country_iso must be a 2-letter uppercase ISO code")
        return v

    @field_validator("eori_number")
    @classmethod
    def validate_eori(cls, v: Optional[str]) -> Optional[str]:
        """Validate EORI format when provided."""
        if v is not None and not EORI_PATTERN.match(v):
            raise ValueError("Invalid EORI format")
        return v

    @field_validator("product_groups")
    @classmethod
    def validate_product_groups(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate product groups when provided."""
        if v is not None:
            invalid = [pg for pg in v if pg not in CBAM_PRODUCT_GROUPS]
            if invalid:
                raise ValueError(f"Invalid product groups: {invalid}")
        return v


# ============================================================================
# REQUEST MODELS - INSTALLATION MANAGEMENT
# ============================================================================

class InstallationRegistrationRequest(BaseModel):
    """Request body for registering a production installation."""
    installation_name: str = Field(
        ..., min_length=1, max_length=200,
        description="Name of the production installation / facility",
    )
    country_iso: str = Field(
        ..., min_length=2, max_length=2,
        description="Country where installation is located (ISO 3166-1 alpha-2)",
    )
    address: AddressModel = Field(..., description="Installation physical address")
    product_groups: List[str] = Field(
        ..., min_length=1,
        description="CBAM product groups produced at this installation",
    )
    cn_codes: Optional[List[str]] = Field(
        None, description="CN codes of products manufactured",
    )
    production_capacity_tons_per_year: Optional[float] = Field(
        None, ge=0, description="Annual capacity in metric tonnes",
    )
    energy_source: Optional[str] = Field(
        None, max_length=200,
        description="Primary energy source (e.g. natural gas, coal, electricity grid)",
    )
    monitoring_methodology: Optional[str] = Field(
        None, max_length=500,
        description="Emissions monitoring methodology used at this installation",
    )
    accreditation_body: Optional[str] = Field(
        None, max_length=200,
        description="Name of accreditation or verification body",
    )
    start_date: Optional[date] = Field(
        None, description="Date the installation began operation",
    )
    notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Validate ISO country code."""
        if not ISO_COUNTRY_PATTERN.match(v):
            raise ValueError("country_iso must be a 2-letter uppercase ISO code")
        return v

    @field_validator("product_groups")
    @classmethod
    def validate_product_groups(cls, v: List[str]) -> List[str]:
        """Validate product groups."""
        invalid = [pg for pg in v if pg not in CBAM_PRODUCT_GROUPS]
        if invalid:
            raise ValueError(f"Invalid product groups: {invalid}")
        return v


class InstallationUpdateRequest(BaseModel):
    """Request body for updating an installation."""
    installation_name: Optional[str] = Field(None, max_length=200)
    address: Optional[AddressModel] = Field(None)
    product_groups: Optional[List[str]] = Field(None)
    cn_codes: Optional[List[str]] = Field(None)
    production_capacity_tons_per_year: Optional[float] = Field(None, ge=0)
    energy_source: Optional[str] = Field(None, max_length=200)
    monitoring_methodology: Optional[str] = Field(None, max_length=500)
    accreditation_body: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = Field(None, max_length=2000)


# ============================================================================
# REQUEST MODELS - EMISSIONS SUBMISSION
# ============================================================================

class PrecursorMaterial(BaseModel):
    """Precursor material emissions data for complex goods."""
    material_name: str = Field(..., max_length=200, description="Name of precursor material")
    cn_code: Optional[str] = Field(None, description="8-digit CN code of precursor")
    mass_fraction: float = Field(
        ..., ge=0, le=1,
        description="Mass fraction of precursor in final product (0.0 - 1.0)",
    )
    emissions_tco2_per_ton: float = Field(
        ..., ge=0,
        description="Embedded emissions of precursor in tCO2e/tonne",
    )

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate CN code format."""
        if v is not None and not CN_CODE_PATTERN.match(v):
            raise ValueError("CN code must be exactly 8 digits")
        return v


class EmissionsSubmissionRequest(BaseModel):
    """Request body for submitting emissions data for an installation."""
    installation_id: str = Field(
        ..., min_length=1, max_length=50,
        description="ID of the installation this submission relates to",
    )
    reporting_period: str = Field(
        ..., description="Reporting period in YYYYQN format (e.g. 2026Q1)",
    )
    product_group: str = Field(
        ..., description="CBAM product group for this submission",
    )
    cn_code: str = Field(
        ..., description="8-digit CN code of the product",
    )
    direct_emissions_tco2_per_ton: float = Field(
        ..., ge=0,
        description="Direct (Scope 1) emissions in tCO2e per tonne of product",
    )
    indirect_emissions_tco2_per_ton: float = Field(
        ..., ge=0,
        description="Indirect (Scope 2) emissions in tCO2e per tonne of product",
    )
    total_emissions_tco2_per_ton: Optional[float] = Field(
        None, ge=0,
        description="Total emissions (auto-calculated if omitted)",
    )
    production_volume_tons: float = Field(
        ..., gt=0,
        description="Total production volume in metric tonnes for the period",
    )
    methodology: str = Field(
        ..., max_length=500,
        description="Emissions calculation methodology used",
    )
    data_quality: DataQuality = Field(
        ..., description="Self-assessed data quality tier",
    )
    data_completeness_pct: float = Field(
        ..., ge=0, le=100,
        description="Percentage of data that is measured vs estimated",
    )
    scope_1_included: bool = Field(True, description="Whether Scope 1 emissions are included")
    scope_2_included: bool = Field(True, description="Whether Scope 2 emissions are included")
    scope_3_included: bool = Field(False, description="Whether Scope 3 emissions are included")
    boundary: Optional[str] = Field(
        None, max_length=500,
        description="System boundary description (e.g. cradle-to-gate)",
    )
    precursor_materials: Optional[List[PrecursorMaterial]] = Field(
        None, description="Precursor materials for complex goods",
    )
    carbon_price_paid_eur_per_ton: Optional[float] = Field(
        None, ge=0,
        description="Carbon price already paid in country of origin (EUR/tCO2e)",
    )
    carbon_price_instrument: Optional[str] = Field(
        None, max_length=200,
        description="Name of the carbon pricing instrument (e.g. China ETS)",
    )
    verification_report_ref: Optional[str] = Field(
        None, max_length=200,
        description="Reference to third-party verification report",
    )
    notes: Optional[str] = Field(None, max_length=2000)

    @field_validator("reporting_period")
    @classmethod
    def validate_reporting_period(cls, v: str) -> str:
        """Validate YYYYQN format."""
        if not re.match(r"^20[2-9][0-9]Q[1-4]$", v):
            raise ValueError("reporting_period must match YYYYQN format (e.g. 2026Q1)")
        return v

    @field_validator("product_group")
    @classmethod
    def validate_product_group(cls, v: str) -> str:
        """Validate product group."""
        if v not in CBAM_PRODUCT_GROUPS:
            raise ValueError(f"Invalid product group. Allowed: {CBAM_PRODUCT_GROUPS}")
        return v

    @field_validator("cn_code")
    @classmethod
    def validate_cn_code(cls, v: str) -> str:
        """Validate CN code."""
        if not CN_CODE_PATTERN.match(v):
            raise ValueError("CN code must be exactly 8 digits")
        return v


class EmissionsAmendRequest(BaseModel):
    """Request body for amending a previously submitted emissions record."""
    amendment_reason: str = Field(
        ..., min_length=5, max_length=1000,
        description="Reason for the amendment",
    )
    direct_emissions_tco2_per_ton: Optional[float] = Field(None, ge=0)
    indirect_emissions_tco2_per_ton: Optional[float] = Field(None, ge=0)
    total_emissions_tco2_per_ton: Optional[float] = Field(None, ge=0)
    production_volume_tons: Optional[float] = Field(None, gt=0)
    methodology: Optional[str] = Field(None, max_length=500)
    data_quality: Optional[DataQuality] = Field(None)
    data_completeness_pct: Optional[float] = Field(None, ge=0, le=100)
    precursor_materials: Optional[List[PrecursorMaterial]] = Field(None)
    carbon_price_paid_eur_per_ton: Optional[float] = Field(None, ge=0)
    notes: Optional[str] = Field(None, max_length=2000)


class SubmissionReviewRequest(BaseModel):
    """Request body for reviewing (accept/reject) a submission."""
    decision: ReviewDecision = Field(
        ..., description="Review decision: accept, reject, or request_amendment",
    )
    reviewer_id: str = Field(
        ..., min_length=1, max_length=50,
        description="ID of the reviewer performing the action",
    )
    review_comments: Optional[str] = Field(
        None, max_length=2000,
        description="Detailed review comments",
    )
    rejection_reasons: Optional[List[str]] = Field(
        None, description="Specific reasons for rejection (if rejected)",
    )


# ============================================================================
# REQUEST MODELS - DATA EXCHANGE
# ============================================================================

class AccessRequestBody(BaseModel):
    """Request body for an importer requesting data access from a supplier."""
    importer_id: str = Field(
        ..., min_length=1, max_length=50, description="EU importer identifier",
    )
    importer_name: str = Field(
        ..., max_length=200, description="Legal name of the EU importer",
    )
    importer_eori: Optional[str] = Field(None, max_length=17, description="Importer EORI")
    supplier_id: str = Field(
        ..., min_length=1, max_length=50, description="Supplier to request access from",
    )
    installation_ids: Optional[List[str]] = Field(
        None, description="Specific installations to access (None = all)",
    )
    purpose: str = Field(
        ..., max_length=500,
        description="Purpose of the data access request (e.g. CBAM quarterly report)",
    )
    requested_periods: Optional[List[str]] = Field(
        None, description="Reporting periods requested (YYYYQN format)",
    )
    access_duration_days: int = Field(
        365, ge=1, le=730,
        description="Requested access duration in days (max 2 years)",
    )


class AccessApprovalRequest(BaseModel):
    """Request body for supplier approving/denying an access request."""
    decision: str = Field(
        ..., description="approved or denied",
    )
    restrictions: Optional[List[str]] = Field(
        None, description="Any data access restrictions or conditions",
    )
    approved_installation_ids: Optional[List[str]] = Field(
        None, description="Subset of installations approved (None = all requested)",
    )
    notes: Optional[str] = Field(None, max_length=1000)

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v: str) -> str:
        """Validate decision value."""
        if v not in ("approved", "denied"):
            raise ValueError("decision must be 'approved' or 'denied'")
        return v


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PaginationMeta(BaseModel):
    """Pagination metadata included in list responses."""
    total: int = Field(..., description="Total number of records matching the query")
    offset: int = Field(..., description="Current offset")
    limit: int = Field(..., description="Page size")
    has_more: bool = Field(..., description="Whether more pages are available")


class ProvenanceMeta(BaseModel):
    """Provenance metadata for audit trail."""
    provenance_hash: str = Field(
        ..., description="SHA-256 hash of the response payload for integrity verification",
    )
    generated_at: str = Field(..., description="ISO 8601 timestamp of response generation")
    api_version: str = Field(default="1.1.0", description="API version that generated this response")


class SupplierProfileResponse(BaseModel):
    """Response model for a single supplier profile."""
    supplier_id: str
    company_name: str
    country_iso: str
    country_name: Optional[str] = None
    eori_number: Optional[str] = None
    tax_id: Optional[str] = None
    product_groups: List[str]
    cn_codes_produced: Optional[List[str]] = None
    address: Optional[AddressModel] = None
    contact: Optional[ContactModel] = None
    certifications: Optional[List[str]] = None
    production_capacity_tons_per_year: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    is_active: bool = True
    actual_emissions_available: bool = False
    installation_count: int = 0
    created_at: str
    updated_at: str
    notes: Optional[str] = None
    provenance: ProvenanceMeta


class SupplierListResponse(BaseModel):
    """Response model for paginated supplier list."""
    suppliers: List[SupplierProfileResponse]
    pagination: PaginationMeta
    provenance: ProvenanceMeta


class InstallationResponse(BaseModel):
    """Response model for a single installation."""
    installation_id: str
    supplier_id: str
    installation_name: str
    country_iso: str
    address: AddressModel
    product_groups: List[str]
    cn_codes: Optional[List[str]] = None
    production_capacity_tons_per_year: Optional[float] = None
    energy_source: Optional[str] = None
    monitoring_methodology: Optional[str] = None
    accreditation_body: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.PENDING
    start_date: Optional[str] = None
    is_active: bool = True
    created_at: str
    updated_at: str
    notes: Optional[str] = None
    provenance: ProvenanceMeta


class InstallationListResponse(BaseModel):
    """Response model for paginated installation list."""
    installations: List[InstallationResponse]
    pagination: PaginationMeta
    provenance: ProvenanceMeta


class InstallationVerificationResponse(BaseModel):
    """Response model for installation verification status."""
    installation_id: str
    supplier_id: str
    verification_status: VerificationStatus
    verified_by: Optional[str] = None
    verification_date: Optional[str] = None
    expiry_date: Optional[str] = None
    accreditation_body: Optional[str] = None
    verification_report_ref: Optional[str] = None
    issues: Optional[List[str]] = None
    provenance: ProvenanceMeta


class EmissionsSubmissionResponse(BaseModel):
    """Response model for a single emissions submission."""
    submission_id: str
    installation_id: str
    supplier_id: str
    reporting_period: str
    product_group: str
    cn_code: str
    direct_emissions_tco2_per_ton: float
    indirect_emissions_tco2_per_ton: float
    total_emissions_tco2_per_ton: float
    production_volume_tons: float
    total_embedded_emissions_tco2: float
    methodology: str
    data_quality: str
    data_completeness_pct: float
    scope_1_included: bool
    scope_2_included: bool
    scope_3_included: bool
    boundary: Optional[str] = None
    precursor_materials: Optional[List[Dict[str, Any]]] = None
    carbon_price_paid_eur_per_ton: Optional[float] = None
    carbon_price_instrument: Optional[str] = None
    verification_report_ref: Optional[str] = None
    status: SubmissionStatus
    submitted_at: str
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None
    review_comments: Optional[str] = None
    amendment_version: int = 1
    notes: Optional[str] = None
    provenance: ProvenanceMeta


class EmissionsListResponse(BaseModel):
    """Response model for paginated emissions submissions list."""
    submissions: List[EmissionsSubmissionResponse]
    pagination: PaginationMeta
    provenance: ProvenanceMeta


class AmendmentHistoryEntry(BaseModel):
    """Single entry in the amendment history."""
    version: int
    amended_at: str
    amendment_reason: str
    changed_fields: Dict[str, Any]
    previous_values: Dict[str, Any]
    new_values: Dict[str, Any]
    provenance_hash: str


class AmendmentHistoryResponse(BaseModel):
    """Response model for submission amendment history."""
    submission_id: str
    current_version: int
    history: List[AmendmentHistoryEntry]
    provenance: ProvenanceMeta


class DocumentUploadResponse(BaseModel):
    """Response model for supporting document upload."""
    document_id: str
    submission_id: str
    file_name: str
    file_size_bytes: int
    content_type: str
    file_hash: str
    uploaded_at: str
    provenance: ProvenanceMeta


class ExportResponse(BaseModel):
    """Response model for data export."""
    export_id: str
    format: str
    record_count: int
    file_size_bytes: int
    download_url: str
    expires_at: str
    provenance: ProvenanceMeta


class DashboardResponse(BaseModel):
    """Response model for supplier dashboard."""
    supplier_id: str
    company_name: str
    summary: Dict[str, Any]
    installations: Dict[str, Any]
    submissions: Dict[str, Any]
    compliance: Dict[str, Any]
    recent_activity: List[Dict[str, Any]]
    provenance: ProvenanceMeta


class DataQualityResponse(BaseModel):
    """Response model for data quality overview."""
    supplier_id: str
    overall_score: float
    dimensions: Dict[str, float]
    installation_scores: List[Dict[str, Any]]
    recommendations: List[str]
    provenance: ProvenanceMeta


class DeadlineResponse(BaseModel):
    """Response model for upcoming deadlines."""
    supplier_id: str
    deadlines: List[Dict[str, Any]]
    overdue: List[Dict[str, Any]]
    provenance: ProvenanceMeta


class EmissionsTrendResponse(BaseModel):
    """Response model for emissions trend data."""
    supplier_id: str
    periods: List[str]
    direct_emissions: List[float]
    indirect_emissions: List[float]
    total_emissions: List[float]
    production_volumes: List[float]
    intensity_trend: List[float]
    provenance: ProvenanceMeta


class AccessRequestResponse(BaseModel):
    """Response model for a data access request."""
    request_id: str
    importer_id: str
    importer_name: str
    supplier_id: str
    installation_ids: Optional[List[str]] = None
    purpose: str
    requested_periods: Optional[List[str]] = None
    access_duration_days: int
    status: AccessRequestStatus
    created_at: str
    resolved_at: Optional[str] = None
    restrictions: Optional[List[str]] = None
    notes: Optional[str] = None
    provenance: ProvenanceMeta


class AuthorizedDataResponse(BaseModel):
    """Response model for authorized supplier data provided to an importer."""
    importer_id: str
    supplier_id: str
    supplier_name: str
    access_expires_at: str
    installations: List[Dict[str, Any]]
    emissions_data: List[Dict[str, Any]]
    provenance: ProvenanceMeta


class InstallationSearchResponse(BaseModel):
    """Response model for third-country installation search."""
    installations: List[Dict[str, Any]]
    pagination: PaginationMeta
    provenance: ProvenanceMeta


class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    log_id: str
    timestamp: str
    action: str
    actor_id: str
    actor_type: str
    resource_type: str
    resource_id: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    provenance_hash: str


class AuditLogResponse(BaseModel):
    """Response model for audit trail."""
    supplier_id: str
    entries: List[AuditLogEntry]
    pagination: PaginationMeta
    provenance: ProvenanceMeta


class MessageResponse(BaseModel):
    """Generic message response for operations without rich payloads."""
    message: str
    resource_id: Optional[str] = None
    provenance: ProvenanceMeta


# ============================================================================
# IN-MEMORY STORES (replaced by database in production)
# ============================================================================

_suppliers: Dict[str, Dict[str, Any]] = {}
_installations: Dict[str, Dict[str, Any]] = {}
_submissions: Dict[str, Dict[str, Any]] = {}
_amendment_history: Dict[str, List[Dict[str, Any]]] = {}
_documents: Dict[str, Dict[str, Any]] = {}
_access_requests: Dict[str, Dict[str, Any]] = {}
_audit_log: List[Dict[str, Any]] = []


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _generate_id(prefix: str) -> str:
    """Generate a unique identifier with the given prefix."""
    short_uuid = uuid.uuid4().hex[:12].upper()
    return f"{prefix}-{short_uuid}"


def _now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _compute_provenance_hash(payload: Any) -> str:
    """Compute SHA-256 provenance hash over a serialisable payload."""
    raw = str(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _provenance(payload: Any) -> ProvenanceMeta:
    """Build a ProvenanceMeta from any serialisable payload."""
    return ProvenanceMeta(
        provenance_hash=_compute_provenance_hash(payload),
        generated_at=_now_iso(),
    )


def _record_audit(
    action: AuditAction,
    actor_id: str,
    actor_type: str,
    resource_type: str,
    resource_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Append an entry to the in-memory audit log."""
    entry = {
        "log_id": _generate_id("AUD"),
        "timestamp": _now_iso(),
        "action": action.value,
        "actor_id": actor_id,
        "actor_type": actor_type,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details or {},
        "provenance_hash": _compute_provenance_hash(
            f"{action.value}:{resource_id}:{_now_iso()}"
        ),
    }
    _audit_log.append(entry)
    logger.info(
        "Audit recorded: %s on %s/%s by %s",
        action.value, resource_type, resource_id, actor_id,
    )


def _get_supplier_or_404(supplier_id: str) -> Dict[str, Any]:
    """Retrieve supplier dict or raise 404."""
    supplier = _suppliers.get(supplier_id)
    if supplier is None or not supplier.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Supplier '{supplier_id}' not found or is deactivated",
        )
    return supplier


def _get_installation_or_404(installation_id: str) -> Dict[str, Any]:
    """Retrieve installation dict or raise 404."""
    inst = _installations.get(installation_id)
    if inst is None or not inst.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Installation '{installation_id}' not found",
        )
    return inst


def _get_submission_or_404(submission_id: str) -> Dict[str, Any]:
    """Retrieve submission dict or raise 404."""
    sub = _submissions.get(submission_id)
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Submission '{submission_id}' not found",
        )
    return sub


def _build_supplier_response(supplier: Dict[str, Any]) -> SupplierProfileResponse:
    """Build a SupplierProfileResponse from the internal dict."""
    inst_count = sum(
        1 for i in _installations.values()
        if i.get("supplier_id") == supplier["supplier_id"] and i.get("is_active", True)
    )
    payload = {**supplier, "installation_count": inst_count}
    return SupplierProfileResponse(
        **payload,
        provenance=_provenance(payload),
    )


def _build_installation_response(inst: Dict[str, Any]) -> InstallationResponse:
    """Build an InstallationResponse from the internal dict."""
    return InstallationResponse(
        **inst,
        provenance=_provenance(inst),
    )


def _build_submission_response(sub: Dict[str, Any]) -> EmissionsSubmissionResponse:
    """Build an EmissionsSubmissionResponse from the internal dict."""
    return EmissionsSubmissionResponse(
        **sub,
        provenance=_provenance(sub),
    )


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/api/v1/cbam/suppliers",
    tags=["CBAM Supplier Portal"],
    responses={
        400: {"description": "Bad Request - Invalid input data"},
        404: {"description": "Not Found - Resource does not exist"},
        409: {"description": "Conflict - Resource already exists or state conflict"},
        422: {"description": "Validation Error - Input failed schema validation"},
        500: {"description": "Internal Server Error"},
    },
)


# ============================================================================
# SUPPLIER REGISTRATION ENDPOINTS
# ============================================================================

@router.post(
    "/register",
    response_model=SupplierProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new CBAM supplier",
    description=(
        "Register a new third-country supplier (manufacturer/producer) of CBAM-covered "
        "goods. Validates EORI format, country code, and product groups against the "
        "official CBAM Annex I list. Returns the created supplier profile with a unique "
        "supplier_id and SHA-256 provenance hash."
    ),
)
async def register_supplier(
    body: SupplierRegistrationRequest,
) -> SupplierProfileResponse:
    """Register a new CBAM supplier."""
    start_time = datetime.now(timezone.utc)

    # Check for duplicate EORI
    if body.eori_number:
        for existing in _suppliers.values():
            if (
                existing.get("eori_number") == body.eori_number
                and existing.get("is_active", True)
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Supplier with EORI '{body.eori_number}' already registered",
                )

    supplier_id = _generate_id("SUP")
    now = _now_iso()

    supplier = {
        "supplier_id": supplier_id,
        "company_name": body.company_name,
        "country_iso": body.country_iso,
        "country_name": None,
        "eori_number": body.eori_number,
        "tax_id": body.tax_id,
        "product_groups": body.product_groups,
        "cn_codes_produced": body.cn_codes_produced,
        "address": body.address.model_dump() if body.address else None,
        "contact": body.contact.model_dump() if body.contact else None,
        "certifications": body.certifications,
        "production_capacity_tons_per_year": body.production_capacity_tons_per_year,
        "verification_status": VerificationStatus.PENDING.value,
        "is_active": True,
        "actual_emissions_available": False,
        "created_at": now,
        "updated_at": now,
        "notes": body.notes,
    }

    _suppliers[supplier_id] = supplier

    _record_audit(
        action=AuditAction.SUPPLIER_REGISTERED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="supplier",
        resource_id=supplier_id,
        details={"company_name": body.company_name, "country_iso": body.country_iso},
    )

    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    logger.info(
        "Supplier registered: %s (%s) in %.1f ms",
        supplier_id, body.company_name, duration_ms,
    )

    return _build_supplier_response(supplier)


@router.get(
    "/search",
    response_model=SupplierListResponse,
    summary="Search suppliers",
    description=(
        "Search registered suppliers by country, CBAM sector, company name, or "
        "verification status. Supports pagination via offset/limit parameters."
    ),
)
async def search_suppliers(
    country: Optional[str] = Query(None, min_length=2, max_length=2, description="ISO country code filter"),
    sector: Optional[str] = Query(None, description="CBAM product group filter"),
    name: Optional[str] = Query(None, max_length=200, description="Company name search (case-insensitive contains)"),
    verification_status: Optional[VerificationStatus] = Query(None, description="Verification status filter"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Page size (max 100)"),
) -> SupplierListResponse:
    """Search suppliers with optional filters."""
    results = [
        s for s in _suppliers.values()
        if s.get("is_active", True)
    ]

    # Apply filters
    if country:
        results = [s for s in results if s.get("country_iso") == country.upper()]
    if sector:
        results = [s for s in results if sector in s.get("product_groups", [])]
    if name:
        name_lower = name.lower()
        results = [s for s in results if name_lower in s.get("company_name", "").lower()]
    if verification_status:
        results = [s for s in results if s.get("verification_status") == verification_status.value]

    total = len(results)
    page = results[offset: offset + limit]

    supplier_responses = [_build_supplier_response(s) for s in page]
    payload = {"suppliers": [s.model_dump() for s in supplier_responses], "total": total}

    return SupplierListResponse(
        suppliers=supplier_responses,
        pagination=PaginationMeta(
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        ),
        provenance=_provenance(payload),
    )


@router.get(
    "/{supplier_id}",
    response_model=SupplierProfileResponse,
    summary="Get supplier profile",
    description="Retrieve the full profile of a registered CBAM supplier by ID.",
)
async def get_supplier(supplier_id: str) -> SupplierProfileResponse:
    """Get supplier profile by ID."""
    supplier = _get_supplier_or_404(supplier_id)
    return _build_supplier_response(supplier)


@router.put(
    "/{supplier_id}",
    response_model=SupplierProfileResponse,
    summary="Update supplier profile",
    description=(
        "Update one or more fields on an existing supplier profile. Only the fields "
        "provided in the request body are changed; others remain untouched."
    ),
)
async def update_supplier(
    supplier_id: str,
    body: SupplierUpdateRequest,
) -> SupplierProfileResponse:
    """Update supplier profile."""
    supplier = _get_supplier_or_404(supplier_id)

    update_data = body.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update",
        )

    # Serialize nested models
    if "address" in update_data and update_data["address"] is not None:
        update_data["address"] = body.address.model_dump() if body.address else None
    if "contact" in update_data and update_data["contact"] is not None:
        update_data["contact"] = body.contact.model_dump() if body.contact else None

    for key, value in update_data.items():
        supplier[key] = value

    supplier["updated_at"] = _now_iso()
    _suppliers[supplier_id] = supplier

    _record_audit(
        action=AuditAction.SUPPLIER_UPDATED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="supplier",
        resource_id=supplier_id,
        details={"updated_fields": list(update_data.keys())},
    )

    logger.info("Supplier updated: %s, fields: %s", supplier_id, list(update_data.keys()))
    return _build_supplier_response(supplier)


@router.delete(
    "/{supplier_id}",
    response_model=MessageResponse,
    summary="Deactivate supplier",
    description=(
        "Soft-delete a supplier by setting is_active to false. The supplier record "
        "is retained for audit purposes but will no longer appear in search results."
    ),
)
async def deactivate_supplier(supplier_id: str) -> MessageResponse:
    """Deactivate a supplier (soft delete)."""
    supplier = _get_supplier_or_404(supplier_id)

    supplier["is_active"] = False
    supplier["updated_at"] = _now_iso()
    _suppliers[supplier_id] = supplier

    # Deactivate installations
    for inst in _installations.values():
        if inst.get("supplier_id") == supplier_id:
            inst["is_active"] = False
            inst["updated_at"] = _now_iso()

    _record_audit(
        action=AuditAction.SUPPLIER_DEACTIVATED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="supplier",
        resource_id=supplier_id,
    )

    logger.info("Supplier deactivated: %s", supplier_id)
    payload = {"message": f"Supplier '{supplier_id}' deactivated", "supplier_id": supplier_id}
    return MessageResponse(
        message=f"Supplier '{supplier_id}' has been deactivated",
        resource_id=supplier_id,
        provenance=_provenance(payload),
    )


# ============================================================================
# INSTALLATION MANAGEMENT ENDPOINTS
# ============================================================================

@router.post(
    "/{supplier_id}/installations",
    response_model=InstallationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new installation",
    description=(
        "Register a production installation (factory/facility) under a supplier. "
        "An installation represents a specific production site that produces "
        "CBAM-covered goods."
    ),
)
async def register_installation(
    supplier_id: str,
    body: InstallationRegistrationRequest,
) -> InstallationResponse:
    """Register a new installation for a supplier."""
    _get_supplier_or_404(supplier_id)

    installation_id = _generate_id("INST")
    now = _now_iso()

    inst = {
        "installation_id": installation_id,
        "supplier_id": supplier_id,
        "installation_name": body.installation_name,
        "country_iso": body.country_iso,
        "address": body.address.model_dump(),
        "product_groups": body.product_groups,
        "cn_codes": body.cn_codes,
        "production_capacity_tons_per_year": body.production_capacity_tons_per_year,
        "energy_source": body.energy_source,
        "monitoring_methodology": body.monitoring_methodology,
        "accreditation_body": body.accreditation_body,
        "verification_status": VerificationStatus.PENDING.value,
        "start_date": body.start_date.isoformat() if body.start_date else None,
        "is_active": True,
        "created_at": now,
        "updated_at": now,
        "notes": body.notes,
    }

    _installations[installation_id] = inst

    _record_audit(
        action=AuditAction.INSTALLATION_CREATED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="installation",
        resource_id=installation_id,
        details={
            "installation_name": body.installation_name,
            "country_iso": body.country_iso,
        },
    )

    logger.info(
        "Installation registered: %s for supplier %s",
        installation_id, supplier_id,
    )
    return _build_installation_response(inst)


@router.get(
    "/{supplier_id}/installations",
    response_model=InstallationListResponse,
    summary="List supplier installations",
    description="List all active installations belonging to a supplier with pagination.",
)
async def list_installations(
    supplier_id: str,
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Page size"),
) -> InstallationListResponse:
    """List installations for a supplier."""
    _get_supplier_or_404(supplier_id)

    results = [
        i for i in _installations.values()
        if i.get("supplier_id") == supplier_id and i.get("is_active", True)
    ]

    total = len(results)
    page = results[offset: offset + limit]
    inst_responses = [_build_installation_response(i) for i in page]

    payload = {"installations_count": total, "supplier_id": supplier_id}
    return InstallationListResponse(
        installations=inst_responses,
        pagination=PaginationMeta(
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        ),
        provenance=_provenance(payload),
    )


@router.get(
    "/{supplier_id}/installations/{installation_id}",
    response_model=InstallationResponse,
    summary="Get installation details",
    description="Retrieve full details of a specific installation.",
)
async def get_installation(
    supplier_id: str,
    installation_id: str,
) -> InstallationResponse:
    """Get installation details."""
    _get_supplier_or_404(supplier_id)
    inst = _get_installation_or_404(installation_id)

    if inst.get("supplier_id") != supplier_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Installation '{installation_id}' does not belong to supplier '{supplier_id}'",
        )

    return _build_installation_response(inst)


@router.put(
    "/{supplier_id}/installations/{installation_id}",
    response_model=InstallationResponse,
    summary="Update installation",
    description="Update fields on an existing installation. Only provided fields are changed.",
)
async def update_installation(
    supplier_id: str,
    installation_id: str,
    body: InstallationUpdateRequest,
) -> InstallationResponse:
    """Update an installation."""
    _get_supplier_or_404(supplier_id)
    inst = _get_installation_or_404(installation_id)

    if inst.get("supplier_id") != supplier_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Installation '{installation_id}' does not belong to supplier '{supplier_id}'",
        )

    update_data = body.model_dump(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update",
        )

    if "address" in update_data and update_data["address"] is not None:
        update_data["address"] = body.address.model_dump() if body.address else None

    for key, value in update_data.items():
        inst[key] = value

    inst["updated_at"] = _now_iso()
    _installations[installation_id] = inst

    _record_audit(
        action=AuditAction.INSTALLATION_UPDATED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="installation",
        resource_id=installation_id,
        details={"updated_fields": list(update_data.keys())},
    )

    logger.info("Installation updated: %s, fields: %s", installation_id, list(update_data.keys()))
    return _build_installation_response(inst)


@router.get(
    "/{supplier_id}/installations/{installation_id}/verification",
    response_model=InstallationVerificationResponse,
    summary="Get installation verification status",
    description=(
        "Retrieve the current verification status for an installation, including "
        "verifier details, verification date, expiry, and any outstanding issues."
    ),
)
async def get_installation_verification(
    supplier_id: str,
    installation_id: str,
) -> InstallationVerificationResponse:
    """Get verification status for an installation."""
    _get_supplier_or_404(supplier_id)
    inst = _get_installation_or_404(installation_id)

    if inst.get("supplier_id") != supplier_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Installation '{installation_id}' does not belong to supplier '{supplier_id}'",
        )

    verification_payload = {
        "installation_id": installation_id,
        "supplier_id": supplier_id,
        "verification_status": inst.get("verification_status", VerificationStatus.PENDING.value),
        "accreditation_body": inst.get("accreditation_body"),
    }

    return InstallationVerificationResponse(
        installation_id=installation_id,
        supplier_id=supplier_id,
        verification_status=VerificationStatus(
            inst.get("verification_status", VerificationStatus.PENDING.value)
        ),
        verified_by=None,
        verification_date=None,
        expiry_date=None,
        accreditation_body=inst.get("accreditation_body"),
        verification_report_ref=None,
        issues=None,
        provenance=_provenance(verification_payload),
    )


# ============================================================================
# EMISSIONS DATA SUBMISSION ENDPOINTS
# ============================================================================

@router.post(
    "/emissions/submit",
    response_model=EmissionsSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit emissions data",
    description=(
        "Submit emissions data for an installation and reporting period. The total "
        "emissions are deterministically calculated (direct + indirect) following "
        "the zero-hallucination principle. Duplicate submissions for the same "
        "installation/period/product are rejected."
    ),
)
async def submit_emissions(
    body: EmissionsSubmissionRequest,
) -> EmissionsSubmissionResponse:
    """Submit emissions data for an installation."""
    start_time = datetime.now(timezone.utc)

    # Validate installation exists
    inst = _get_installation_or_404(body.installation_id)
    supplier_id = inst["supplier_id"]
    _get_supplier_or_404(supplier_id)

    # Check for duplicate submission (same installation + period + cn_code)
    for existing in _submissions.values():
        if (
            existing.get("installation_id") == body.installation_id
            and existing.get("reporting_period") == body.reporting_period
            and existing.get("cn_code") == body.cn_code
            and existing.get("status") not in (
                SubmissionStatus.REJECTED.value,
                SubmissionStatus.AMENDED.value,
            )
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Active submission already exists for installation "
                    f"'{body.installation_id}', period '{body.reporting_period}', "
                    f"CN code '{body.cn_code}'. Use the amend endpoint to modify."
                ),
            )

    # Zero-hallucination: deterministic total calculation
    total_emissions = body.total_emissions_tco2_per_ton
    if total_emissions is None:
        total_emissions = body.direct_emissions_tco2_per_ton + body.indirect_emissions_tco2_per_ton

    # Deterministic embedded emissions calculation
    total_embedded = total_emissions * body.production_volume_tons

    submission_id = _generate_id("EMSUB")
    now = _now_iso()

    submission = {
        "submission_id": submission_id,
        "installation_id": body.installation_id,
        "supplier_id": supplier_id,
        "reporting_period": body.reporting_period,
        "product_group": body.product_group,
        "cn_code": body.cn_code,
        "direct_emissions_tco2_per_ton": body.direct_emissions_tco2_per_ton,
        "indirect_emissions_tco2_per_ton": body.indirect_emissions_tco2_per_ton,
        "total_emissions_tco2_per_ton": round(total_emissions, 6),
        "production_volume_tons": body.production_volume_tons,
        "total_embedded_emissions_tco2": round(total_embedded, 6),
        "methodology": body.methodology,
        "data_quality": body.data_quality.value,
        "data_completeness_pct": body.data_completeness_pct,
        "scope_1_included": body.scope_1_included,
        "scope_2_included": body.scope_2_included,
        "scope_3_included": body.scope_3_included,
        "boundary": body.boundary,
        "precursor_materials": (
            [pm.model_dump() for pm in body.precursor_materials]
            if body.precursor_materials else None
        ),
        "carbon_price_paid_eur_per_ton": body.carbon_price_paid_eur_per_ton,
        "carbon_price_instrument": body.carbon_price_instrument,
        "verification_report_ref": body.verification_report_ref,
        "status": SubmissionStatus.SUBMITTED.value,
        "submitted_at": now,
        "reviewed_at": None,
        "reviewed_by": None,
        "review_comments": None,
        "amendment_version": 1,
        "notes": body.notes,
    }

    _submissions[submission_id] = submission
    _amendment_history[submission_id] = []

    # Update supplier flag
    supplier = _suppliers.get(supplier_id)
    if supplier:
        supplier["actual_emissions_available"] = True
        supplier["updated_at"] = now

    _record_audit(
        action=AuditAction.EMISSIONS_SUBMITTED,
        actor_id=supplier_id,
        actor_type="supplier",
        resource_type="submission",
        resource_id=submission_id,
        details={
            "installation_id": body.installation_id,
            "reporting_period": body.reporting_period,
            "total_emissions_tco2_per_ton": round(total_emissions, 6),
        },
    )

    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    logger.info(
        "Emissions submitted: %s for installation %s, period %s (%.1f ms)",
        submission_id, body.installation_id, body.reporting_period, duration_ms,
    )

    return _build_submission_response(submission)


@router.get(
    "/emissions/export",
    response_model=ExportResponse,
    summary="Export emissions submissions",
    description=(
        "Export filtered emissions submissions in CSV, JSON, or XML format. "
        "Returns a download reference that can be used to retrieve the file."
    ),
)
async def export_emissions(
    format: ExportFormat = Query(ExportFormat.CSV, description="Export format"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier"),
    installation_id: Optional[str] = Query(None, description="Filter by installation"),
    reporting_period: Optional[str] = Query(None, description="Filter by period (YYYYQN)"),
    status_filter: Optional[SubmissionStatus] = Query(None, alias="status", description="Filter by status"),
) -> ExportResponse:
    """Export emissions data."""
    results = list(_submissions.values())

    if supplier_id:
        results = [s for s in results if s.get("supplier_id") == supplier_id]
    if installation_id:
        results = [s for s in results if s.get("installation_id") == installation_id]
    if reporting_period:
        results = [s for s in results if s.get("reporting_period") == reporting_period]
    if status_filter:
        results = [s for s in results if s.get("status") == status_filter.value]

    export_id = _generate_id("EXP")
    now = _now_iso()

    # Simulate file generation
    simulated_size = len(str(results).encode("utf-8"))
    expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

    payload = {
        "export_id": export_id,
        "format": format.value,
        "record_count": len(results),
        "file_size_bytes": simulated_size,
    }

    _record_audit(
        action=AuditAction.DATA_EXPORTED,
        actor_id="system",
        actor_type="system",
        resource_type="export",
        resource_id=export_id,
        details={"format": format.value, "record_count": len(results)},
    )

    logger.info("Export generated: %s, format=%s, records=%d", export_id, format.value, len(results))

    return ExportResponse(
        export_id=export_id,
        format=format.value,
        record_count=len(results),
        file_size_bytes=simulated_size,
        download_url=f"/api/v1/cbam/suppliers/emissions/export/{export_id}/download",
        expires_at=expires_at,
        provenance=_provenance(payload),
    )


@router.get(
    "/emissions",
    response_model=EmissionsListResponse,
    summary="List emissions submissions",
    description=(
        "List all emissions submissions with optional filters by installation, "
        "reporting period, and status. Supports pagination."
    ),
)
async def list_emissions(
    installation_id: Optional[str] = Query(None, description="Filter by installation"),
    reporting_period: Optional[str] = Query(None, description="Filter by period (YYYYQN)"),
    status_filter: Optional[SubmissionStatus] = Query(None, alias="status", description="Filter by status"),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Page size"),
) -> EmissionsListResponse:
    """List emissions submissions with filters."""
    results = list(_submissions.values())

    if installation_id:
        results = [s for s in results if s.get("installation_id") == installation_id]
    if reporting_period:
        results = [s for s in results if s.get("reporting_period") == reporting_period]
    if status_filter:
        results = [s for s in results if s.get("status") == status_filter.value]
    if supplier_id:
        results = [s for s in results if s.get("supplier_id") == supplier_id]

    total = len(results)
    page = results[offset: offset + limit]
    sub_responses = [_build_submission_response(s) for s in page]

    payload = {"total": total, "offset": offset, "limit": limit}
    return EmissionsListResponse(
        submissions=sub_responses,
        pagination=PaginationMeta(
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        ),
        provenance=_provenance(payload),
    )


@router.get(
    "/emissions/{submission_id}",
    response_model=EmissionsSubmissionResponse,
    summary="Get submission details",
    description="Retrieve full details of an emissions data submission by ID.",
)
async def get_submission(submission_id: str) -> EmissionsSubmissionResponse:
    """Get emissions submission by ID."""
    sub = _get_submission_or_404(submission_id)
    return _build_submission_response(sub)


@router.put(
    "/emissions/{submission_id}/amend",
    response_model=EmissionsSubmissionResponse,
    summary="Amend existing submission",
    description=(
        "Amend a previously submitted emissions record. Creates a new version while "
        "preserving the full amendment history for audit purposes. Only submissions "
        "in 'submitted', 'accepted', or 'under_review' status can be amended."
    ),
)
async def amend_submission(
    submission_id: str,
    body: EmissionsAmendRequest,
) -> EmissionsSubmissionResponse:
    """Amend an existing emissions submission."""
    sub = _get_submission_or_404(submission_id)

    # Only allow amending certain statuses
    allowed_statuses = {
        SubmissionStatus.SUBMITTED.value,
        SubmissionStatus.ACCEPTED.value,
        SubmissionStatus.UNDER_REVIEW.value,
    }
    if sub.get("status") not in allowed_statuses:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot amend submission in '{sub.get('status')}' status. "
                f"Allowed statuses: {sorted(allowed_statuses)}"
            ),
        )

    # Record amendment history
    update_data = body.model_dump(exclude_unset=True, exclude={"amendment_reason"})

    # Serialize nested models
    if "precursor_materials" in update_data and update_data["precursor_materials"] is not None:
        update_data["precursor_materials"] = [
            pm.model_dump() for pm in body.precursor_materials
        ]

    previous_values = {}
    new_values = {}
    for key, value in update_data.items():
        if key in sub and sub[key] != value:
            previous_values[key] = sub[key]
            new_values[key] = value

    if not new_values:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No actual changes detected in the amendment",
        )

    # Increment version
    old_version = sub.get("amendment_version", 1)
    new_version = old_version + 1

    history_entry = {
        "version": new_version,
        "amended_at": _now_iso(),
        "amendment_reason": body.amendment_reason,
        "changed_fields": list(new_values.keys()),
        "previous_values": previous_values,
        "new_values": new_values,
        "provenance_hash": _compute_provenance_hash(
            f"{submission_id}:v{new_version}:{new_values}"
        ),
    }

    if submission_id not in _amendment_history:
        _amendment_history[submission_id] = []
    _amendment_history[submission_id].append(history_entry)

    # Apply changes
    for key, value in update_data.items():
        sub[key] = value

    # Recalculate total emissions if components changed
    if "direct_emissions_tco2_per_ton" in new_values or "indirect_emissions_tco2_per_ton" in new_values:
        sub["total_emissions_tco2_per_ton"] = round(
            sub["direct_emissions_tco2_per_ton"] + sub["indirect_emissions_tco2_per_ton"],
            6,
        )

    # Recalculate total embedded emissions
    sub["total_embedded_emissions_tco2"] = round(
        sub["total_emissions_tco2_per_ton"] * sub["production_volume_tons"],
        6,
    )

    sub["amendment_version"] = new_version
    sub["status"] = SubmissionStatus.AMENDED.value
    sub["updated_at"] = _now_iso()
    _submissions[submission_id] = sub

    _record_audit(
        action=AuditAction.EMISSIONS_AMENDED,
        actor_id=sub.get("supplier_id", "unknown"),
        actor_type="supplier",
        resource_type="submission",
        resource_id=submission_id,
        details={
            "version": new_version,
            "amendment_reason": body.amendment_reason,
            "changed_fields": list(new_values.keys()),
        },
    )

    logger.info(
        "Submission amended: %s -> v%d, reason: %s",
        submission_id, new_version, body.amendment_reason,
    )

    return _build_submission_response(sub)


@router.post(
    "/emissions/{submission_id}/review",
    response_model=EmissionsSubmissionResponse,
    summary="Review submission",
    description=(
        "Accept, reject, or request amendment of an emissions data submission. "
        "Only submissions in 'submitted' or 'amended' status can be reviewed."
    ),
)
async def review_submission(
    submission_id: str,
    body: SubmissionReviewRequest,
) -> EmissionsSubmissionResponse:
    """Review (accept/reject) an emissions submission."""
    sub = _get_submission_or_404(submission_id)

    reviewable = {SubmissionStatus.SUBMITTED.value, SubmissionStatus.AMENDED.value}
    if sub.get("status") not in reviewable:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Cannot review submission in '{sub.get('status')}' status. "
                f"Must be in: {sorted(reviewable)}"
            ),
        )

    now = _now_iso()

    # Map decision to status
    decision_status_map = {
        ReviewDecision.ACCEPT: SubmissionStatus.ACCEPTED.value,
        ReviewDecision.REJECT: SubmissionStatus.REJECTED.value,
        ReviewDecision.REQUEST_AMENDMENT: SubmissionStatus.UNDER_REVIEW.value,
    }

    sub["status"] = decision_status_map[body.decision]
    sub["reviewed_at"] = now
    sub["reviewed_by"] = body.reviewer_id
    sub["review_comments"] = body.review_comments
    _submissions[submission_id] = sub

    _record_audit(
        action=AuditAction.EMISSIONS_REVIEWED,
        actor_id=body.reviewer_id,
        actor_type="reviewer",
        resource_type="submission",
        resource_id=submission_id,
        details={
            "decision": body.decision.value,
            "rejection_reasons": body.rejection_reasons,
        },
    )

    logger.info(
        "Submission reviewed: %s, decision=%s by %s",
        submission_id, body.decision.value, body.reviewer_id,
    )

    return _build_submission_response(sub)


@router.get(
    "/emissions/{submission_id}/history",
    response_model=AmendmentHistoryResponse,
    summary="Get amendment history",
    description="Retrieve the full amendment history for an emissions submission.",
)
async def get_amendment_history(submission_id: str) -> AmendmentHistoryResponse:
    """Get amendment history for a submission."""
    sub = _get_submission_or_404(submission_id)

    history = _amendment_history.get(submission_id, [])
    history_entries = [
        AmendmentHistoryEntry(**entry)
        for entry in history
    ]

    payload = {
        "submission_id": submission_id,
        "current_version": sub.get("amendment_version", 1),
        "entry_count": len(history_entries),
    }

    return AmendmentHistoryResponse(
        submission_id=submission_id,
        current_version=sub.get("amendment_version", 1),
        history=history_entries,
        provenance=_provenance(payload),
    )


@router.post(
    "/emissions/{submission_id}/documents",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload supporting document",
    description=(
        "Upload a supporting document (e.g. verification report, methodology "
        "description, or lab certificate) for an emissions submission."
    ),
)
async def upload_document(
    submission_id: str,
    file: UploadFile = File(..., description="Supporting document file"),
) -> DocumentUploadResponse:
    """Upload a supporting document for a submission."""
    _get_submission_or_404(submission_id)

    # Read file content for hashing
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate file
    for existing in _documents.values():
        if (
            existing.get("submission_id") == submission_id
            and existing.get("file_hash") == file_hash
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Identical document already uploaded for this submission",
            )

    doc_id = _generate_id("DOC")
    now = _now_iso()

    doc = {
        "document_id": doc_id,
        "submission_id": submission_id,
        "file_name": file.filename or "unnamed",
        "file_size_bytes": len(content),
        "content_type": file.content_type or "application/octet-stream",
        "file_hash": file_hash,
        "uploaded_at": now,
    }

    _documents[doc_id] = doc

    _record_audit(
        action=AuditAction.DOCUMENT_UPLOADED,
        actor_id="system",
        actor_type="system",
        resource_type="document",
        resource_id=doc_id,
        details={
            "submission_id": submission_id,
            "file_name": doc["file_name"],
            "file_hash": file_hash,
        },
    )

    logger.info(
        "Document uploaded: %s (%s, %d bytes) for submission %s",
        doc_id, doc["file_name"], len(content), submission_id,
    )

    return DocumentUploadResponse(
        **doc,
        provenance=_provenance(doc),
    )


# ============================================================================
# DASHBOARD ENDPOINTS
# ============================================================================

@router.get(
    "/{supplier_id}/dashboard",
    response_model=DashboardResponse,
    summary="Get supplier dashboard",
    description=(
        "Retrieve aggregated dashboard data for a supplier, including summary "
        "statistics, installation overview, submission counts by status, "
        "compliance indicators, and recent activity."
    ),
)
async def get_dashboard(supplier_id: str) -> DashboardResponse:
    """Get supplier dashboard data."""
    supplier = _get_supplier_or_404(supplier_id)

    # Aggregate installations
    installations = [
        i for i in _installations.values()
        if i.get("supplier_id") == supplier_id and i.get("is_active", True)
    ]
    verified_installations = [
        i for i in installations
        if i.get("verification_status") == VerificationStatus.VERIFIED.value
    ]

    # Aggregate submissions
    submissions = [
        s for s in _submissions.values()
        if s.get("supplier_id") == supplier_id
    ]

    status_counts: Dict[str, int] = {}
    for sub in submissions:
        st = sub.get("status", "unknown")
        status_counts[st] = status_counts.get(st, 0) + 1

    total_embedded = sum(
        s.get("total_embedded_emissions_tco2", 0)
        for s in submissions
        if s.get("status") == SubmissionStatus.ACCEPTED.value
    )

    # Recent activity from audit log
    recent = [
        e for e in _audit_log
        if (
            e.get("resource_id") == supplier_id
            or e.get("actor_id") == supplier_id
            or e.get("details", {}).get("supplier_id") == supplier_id
        )
    ]
    recent_sorted = sorted(recent, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]

    summary = {
        "company_name": supplier.get("company_name"),
        "country_iso": supplier.get("country_iso"),
        "verification_status": supplier.get("verification_status"),
        "product_groups": supplier.get("product_groups", []),
        "total_installations": len(installations),
        "total_submissions": len(submissions),
        "total_embedded_emissions_tco2": round(total_embedded, 2),
    }

    installations_overview = {
        "total": len(installations),
        "verified": len(verified_installations),
        "pending_verification": len(installations) - len(verified_installations),
        "countries": list({i.get("country_iso") for i in installations}),
    }

    submissions_overview = {
        "total": len(submissions),
        "by_status": status_counts,
        "accepted_count": status_counts.get(SubmissionStatus.ACCEPTED.value, 0),
        "pending_review": status_counts.get(SubmissionStatus.SUBMITTED.value, 0),
    }

    compliance = {
        "has_verified_installations": len(verified_installations) > 0,
        "has_accepted_submissions": status_counts.get(SubmissionStatus.ACCEPTED.value, 0) > 0,
        "data_coverage_pct": (
            100.0 if len(installations) > 0
            and all(
                any(
                    s.get("installation_id") == i.get("installation_id")
                    for s in submissions
                )
                for i in installations
            )
            else 0.0
        ),
    }

    payload = {
        "supplier_id": supplier_id,
        "summary": summary,
        "timestamp": _now_iso(),
    }

    return DashboardResponse(
        supplier_id=supplier_id,
        company_name=supplier.get("company_name", ""),
        summary=summary,
        installations=installations_overview,
        submissions=submissions_overview,
        compliance=compliance,
        recent_activity=recent_sorted,
        provenance=_provenance(payload),
    )


@router.get(
    "/{supplier_id}/quality",
    response_model=DataQualityResponse,
    summary="Get data quality overview",
    description=(
        "Retrieve a data quality assessment for a supplier across all submissions, "
        "broken down by quality dimension and installation."
    ),
)
async def get_data_quality(supplier_id: str) -> DataQualityResponse:
    """Get data quality overview for a supplier."""
    _get_supplier_or_404(supplier_id)

    submissions = [
        s for s in _submissions.values()
        if s.get("supplier_id") == supplier_id
    ]

    if not submissions:
        payload = {"supplier_id": supplier_id, "overall_score": 0.0}
        return DataQualityResponse(
            supplier_id=supplier_id,
            overall_score=0.0,
            dimensions={
                "completeness": 0.0,
                "accuracy": 0.0,
                "timeliness": 0.0,
                "consistency": 0.0,
                "methodology": 0.0,
            },
            installation_scores=[],
            recommendations=[
                "Submit emissions data for at least one installation to receive quality scores."
            ],
            provenance=_provenance(payload),
        )

    # Deterministic quality scoring (zero-hallucination)
    avg_completeness = sum(s.get("data_completeness_pct", 0) for s in submissions) / len(submissions)

    quality_map = {"high": 90.0, "medium": 65.0, "low": 35.0}
    avg_accuracy = sum(
        quality_map.get(s.get("data_quality", "low"), 35.0)
        for s in submissions
    ) / len(submissions)

    # Methodology score: higher if scope 1+2 included
    methodology_scores = []
    for s in submissions:
        score = 50.0
        if s.get("scope_1_included"):
            score += 20.0
        if s.get("scope_2_included"):
            score += 20.0
        if s.get("verification_report_ref"):
            score += 10.0
        methodology_scores.append(score)
    avg_methodology = sum(methodology_scores) / len(methodology_scores)

    # Consistency score: check for large variances across submissions
    consistency = 80.0
    if len(submissions) >= 2:
        totals = [s.get("total_emissions_tco2_per_ton", 0) for s in submissions]
        if max(totals) > 0:
            cv = (max(totals) - min(totals)) / max(totals) * 100
            consistency = max(0.0, 100.0 - cv)

    timeliness = 75.0  # Default timeliness score

    overall = round(
        (avg_completeness * 0.25 + avg_accuracy * 0.25 + timeliness * 0.15
         + consistency * 0.15 + avg_methodology * 0.20),
        1,
    )

    dimensions = {
        "completeness": round(avg_completeness, 1),
        "accuracy": round(avg_accuracy, 1),
        "timeliness": round(timeliness, 1),
        "consistency": round(consistency, 1),
        "methodology": round(avg_methodology, 1),
    }

    # Per-installation scores
    installation_ids = list({s.get("installation_id") for s in submissions})
    installation_scores = []
    for inst_id in installation_ids:
        inst_subs = [s for s in submissions if s.get("installation_id") == inst_id]
        inst_avg = sum(s.get("data_completeness_pct", 0) for s in inst_subs) / len(inst_subs)
        installation_scores.append({
            "installation_id": inst_id,
            "submission_count": len(inst_subs),
            "avg_completeness_pct": round(inst_avg, 1),
        })

    recommendations = []
    if avg_completeness < 80:
        recommendations.append("Increase measured data coverage to at least 80% for higher quality scores.")
    if avg_accuracy < 70:
        recommendations.append("Improve data quality tier to 'high' through third-party verification.")
    if avg_methodology < 80:
        recommendations.append("Include both Scope 1 and Scope 2 emissions and attach verification reports.")

    payload = {"supplier_id": supplier_id, "overall_score": overall}
    return DataQualityResponse(
        supplier_id=supplier_id,
        overall_score=overall,
        dimensions=dimensions,
        installation_scores=installation_scores,
        recommendations=recommendations,
        provenance=_provenance(payload),
    )


@router.get(
    "/{supplier_id}/deadlines",
    response_model=DeadlineResponse,
    summary="Get upcoming deadlines",
    description=(
        "Retrieve upcoming CBAM reporting deadlines and any overdue submissions "
        "for the supplier."
    ),
)
async def get_deadlines(supplier_id: str) -> DeadlineResponse:
    """Get upcoming deadlines for a supplier."""
    _get_supplier_or_404(supplier_id)

    now = datetime.now(timezone.utc)
    current_year = now.year
    current_quarter = (now.month - 1) // 3 + 1

    # Generate upcoming deadlines (next 4 quarters)
    deadlines = []
    for offset in range(4):
        q = current_quarter + offset
        y = current_year
        while q > 4:
            q -= 4
            y += 1

        period = f"{y}Q{q}"

        # CBAM deadline: end of month following the quarter
        if q == 1:
            deadline_date = date(y, 4, 30)
        elif q == 2:
            deadline_date = date(y, 7, 31)
        elif q == 3:
            deadline_date = date(y, 10, 31)
        else:
            deadline_date = date(y + 1, 1, 31)

        days_remaining = (deadline_date - now.date()).days

        # Check if submission exists for this period
        has_submission = any(
            s.get("supplier_id") == supplier_id
            and s.get("reporting_period") == period
            for s in _submissions.values()
        )

        deadlines.append({
            "reporting_period": period,
            "deadline_date": deadline_date.isoformat(),
            "days_remaining": days_remaining,
            "status": "submitted" if has_submission else ("overdue" if days_remaining < 0 else "pending"),
        })

    overdue = [d for d in deadlines if d["status"] == "overdue"]
    upcoming = [d for d in deadlines if d["status"] != "overdue"]

    payload = {"supplier_id": supplier_id, "deadline_count": len(deadlines)}
    return DeadlineResponse(
        supplier_id=supplier_id,
        deadlines=upcoming,
        overdue=overdue,
        provenance=_provenance(payload),
    )


@router.get(
    "/{supplier_id}/emissions-trend",
    response_model=EmissionsTrendResponse,
    summary="Get emissions trend",
    description=(
        "Retrieve emissions trend data across reporting periods for a supplier, "
        "including direct, indirect, total emissions, production volumes, and "
        "emissions intensity."
    ),
)
async def get_emissions_trend(
    supplier_id: str,
    periods: Optional[int] = Query(8, ge=1, le=20, description="Number of periods to include"),
) -> EmissionsTrendResponse:
    """Get emissions trend data for a supplier."""
    _get_supplier_or_404(supplier_id)

    submissions = [
        s for s in _submissions.values()
        if s.get("supplier_id") == supplier_id
        and s.get("status") in (SubmissionStatus.ACCEPTED.value, SubmissionStatus.SUBMITTED.value)
    ]

    # Group by period, aggregate
    period_data: Dict[str, Dict[str, float]] = {}
    for sub in submissions:
        period = sub.get("reporting_period", "unknown")
        if period not in period_data:
            period_data[period] = {
                "direct": 0.0,
                "indirect": 0.0,
                "total": 0.0,
                "volume": 0.0,
            }
        period_data[period]["direct"] += sub.get("direct_emissions_tco2_per_ton", 0)
        period_data[period]["indirect"] += sub.get("indirect_emissions_tco2_per_ton", 0)
        period_data[period]["total"] += sub.get("total_emissions_tco2_per_ton", 0)
        period_data[period]["volume"] += sub.get("production_volume_tons", 0)

    # Sort periods chronologically and limit
    sorted_periods = sorted(period_data.keys())[-periods:]

    period_list = sorted_periods
    direct_list = [round(period_data[p]["direct"], 4) for p in sorted_periods]
    indirect_list = [round(period_data[p]["indirect"], 4) for p in sorted_periods]
    total_list = [round(period_data[p]["total"], 4) for p in sorted_periods]
    volume_list = [round(period_data[p]["volume"], 2) for p in sorted_periods]
    intensity_list = [
        round(period_data[p]["total"] / period_data[p]["volume"], 6)
        if period_data[p]["volume"] > 0 else 0.0
        for p in sorted_periods
    ]

    payload = {
        "supplier_id": supplier_id,
        "period_count": len(period_list),
    }

    return EmissionsTrendResponse(
        supplier_id=supplier_id,
        periods=period_list,
        direct_emissions=direct_list,
        indirect_emissions=indirect_list,
        total_emissions=total_list,
        production_volumes=volume_list,
        intensity_trend=intensity_list,
        provenance=_provenance(payload),
    )


# ============================================================================
# DATA EXCHANGE ENDPOINTS
# ============================================================================

@router.post(
    "/exchange/request-access",
    response_model=AccessRequestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Request data access from supplier",
    description=(
        "An EU importer requests access to a supplier's emissions data. "
        "The supplier must approve the request before data becomes accessible."
    ),
)
async def request_access(body: AccessRequestBody) -> AccessRequestResponse:
    """Importer requests data access from a supplier."""
    _get_supplier_or_404(body.supplier_id)

    # Check for existing pending request
    for existing in _access_requests.values():
        if (
            existing.get("importer_id") == body.importer_id
            and existing.get("supplier_id") == body.supplier_id
            and existing.get("status") == AccessRequestStatus.PENDING.value
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=(
                    f"Pending access request already exists from importer "
                    f"'{body.importer_id}' to supplier '{body.supplier_id}'"
                ),
            )

    request_id = _generate_id("AREQ")
    now = _now_iso()

    req = {
        "request_id": request_id,
        "importer_id": body.importer_id,
        "importer_name": body.importer_name,
        "importer_eori": body.importer_eori,
        "supplier_id": body.supplier_id,
        "installation_ids": body.installation_ids,
        "purpose": body.purpose,
        "requested_periods": body.requested_periods,
        "access_duration_days": body.access_duration_days,
        "status": AccessRequestStatus.PENDING.value,
        "created_at": now,
        "resolved_at": None,
        "restrictions": None,
        "notes": None,
    }

    _access_requests[request_id] = req

    _record_audit(
        action=AuditAction.ACCESS_REQUESTED,
        actor_id=body.importer_id,
        actor_type="importer",
        resource_type="access_request",
        resource_id=request_id,
        details={
            "supplier_id": body.supplier_id,
            "purpose": body.purpose,
        },
    )

    logger.info(
        "Access requested: %s from importer %s to supplier %s",
        request_id, body.importer_id, body.supplier_id,
    )

    return AccessRequestResponse(
        **req,
        provenance=_provenance(req),
    )


@router.put(
    "/exchange/{request_id}/approve",
    response_model=AccessRequestResponse,
    summary="Approve or deny access request",
    description=(
        "Supplier approves or denies an importer's data access request. "
        "Optional restrictions can be placed on the approved access."
    ),
)
async def approve_access(
    request_id: str,
    body: AccessApprovalRequest,
) -> AccessRequestResponse:
    """Supplier approves or denies an access request."""
    req = _access_requests.get(request_id)
    if req is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Access request '{request_id}' not found",
        )

    if req.get("status") != AccessRequestStatus.PENDING.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Access request is already '{req.get('status')}', cannot update",
        )

    now = _now_iso()
    req["status"] = (
        AccessRequestStatus.APPROVED.value
        if body.decision == "approved"
        else AccessRequestStatus.DENIED.value
    )
    req["resolved_at"] = now
    req["restrictions"] = body.restrictions
    req["notes"] = body.notes

    if body.approved_installation_ids is not None:
        req["installation_ids"] = body.approved_installation_ids

    _access_requests[request_id] = req

    audit_action = (
        AuditAction.ACCESS_APPROVED
        if body.decision == "approved"
        else AuditAction.ACCESS_REVOKED
    )
    _record_audit(
        action=audit_action,
        actor_id=req.get("supplier_id", "unknown"),
        actor_type="supplier",
        resource_type="access_request",
        resource_id=request_id,
        details={"decision": body.decision, "restrictions": body.restrictions},
    )

    logger.info(
        "Access request %s %s for importer %s",
        request_id, body.decision, req.get("importer_id"),
    )

    return AccessRequestResponse(
        **req,
        provenance=_provenance(req),
    )


@router.delete(
    "/exchange/{importer_id}/revoke",
    response_model=MessageResponse,
    summary="Revoke importer access",
    description="Revoke all active data access grants for a specific importer.",
)
async def revoke_access(importer_id: str) -> MessageResponse:
    """Revoke all access for an importer."""
    revoked_count = 0
    for req in _access_requests.values():
        if (
            req.get("importer_id") == importer_id
            and req.get("status") == AccessRequestStatus.APPROVED.value
        ):
            req["status"] = AccessRequestStatus.REVOKED.value
            req["resolved_at"] = _now_iso()
            revoked_count += 1

    if revoked_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active access grants found for importer '{importer_id}'",
        )

    _record_audit(
        action=AuditAction.ACCESS_REVOKED,
        actor_id="system",
        actor_type="system",
        resource_type="access_grant",
        resource_id=importer_id,
        details={"revoked_count": revoked_count},
    )

    logger.info("Revoked %d access grants for importer %s", revoked_count, importer_id)

    payload = {"importer_id": importer_id, "revoked_count": revoked_count}
    return MessageResponse(
        message=f"Revoked {revoked_count} access grant(s) for importer '{importer_id}'",
        resource_id=importer_id,
        provenance=_provenance(payload),
    )


@router.get(
    "/exchange/installations/search",
    response_model=InstallationSearchResponse,
    summary="Search third-country installations",
    description=(
        "Search for registered third-country installations across all suppliers. "
        "Useful for importers looking for supplier installations by country or product."
    ),
)
async def search_installations(
    country: Optional[str] = Query(None, min_length=2, max_length=2, description="ISO country code"),
    product_group: Optional[str] = Query(None, description="CBAM product group"),
    name: Optional[str] = Query(None, max_length=200, description="Installation name (contains)"),
    verification_status: Optional[VerificationStatus] = Query(None, description="Verification status"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(20, ge=1, le=100, description="Page size"),
) -> InstallationSearchResponse:
    """Search third-country installations."""
    results = [
        i for i in _installations.values()
        if i.get("is_active", True)
    ]

    if country:
        results = [i for i in results if i.get("country_iso") == country.upper()]
    if product_group:
        results = [i for i in results if product_group in i.get("product_groups", [])]
    if name:
        name_lower = name.lower()
        results = [i for i in results if name_lower in i.get("installation_name", "").lower()]
    if verification_status:
        results = [
            i for i in results
            if i.get("verification_status") == verification_status.value
        ]

    total = len(results)
    page = results[offset: offset + limit]

    # Return summary dicts (not full installation details for privacy)
    installation_summaries = [
        {
            "installation_id": i.get("installation_id"),
            "installation_name": i.get("installation_name"),
            "supplier_id": i.get("supplier_id"),
            "country_iso": i.get("country_iso"),
            "product_groups": i.get("product_groups", []),
            "verification_status": i.get("verification_status"),
        }
        for i in page
    ]

    payload = {"total": total, "offset": offset}
    return InstallationSearchResponse(
        installations=installation_summaries,
        pagination=PaginationMeta(
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        ),
        provenance=_provenance(payload),
    )


@router.get(
    "/exchange/{importer_id}/data",
    response_model=AuthorizedDataResponse,
    summary="Get authorized supplier data for importer",
    description=(
        "Retrieve the emissions data that a supplier has authorized an importer "
        "to access. Only returns data covered by approved access grants."
    ),
)
async def get_authorized_data(
    importer_id: str,
    supplier_id: Optional[str] = Query(None, description="Filter by specific supplier"),
) -> AuthorizedDataResponse:
    """Get authorized supplier data for an importer."""
    # Find approved access grants for this importer
    grants = [
        r for r in _access_requests.values()
        if (
            r.get("importer_id") == importer_id
            and r.get("status") == AccessRequestStatus.APPROVED.value
        )
    ]

    if supplier_id:
        grants = [g for g in grants if g.get("supplier_id") == supplier_id]

    if not grants:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No approved access grants found for importer '{importer_id}'"
            + (f" with supplier '{supplier_id}'" if supplier_id else ""),
        )

    # Use the first matching grant for response
    grant = grants[0]
    grant_supplier_id = grant.get("supplier_id", "")

    supplier = _suppliers.get(grant_supplier_id, {})

    # Get authorized installations
    authorized_inst_ids = grant.get("installation_ids")
    installations = [
        i for i in _installations.values()
        if (
            i.get("supplier_id") == grant_supplier_id
            and i.get("is_active", True)
            and (authorized_inst_ids is None or i.get("installation_id") in authorized_inst_ids)
        )
    ]

    installation_summaries = [
        {
            "installation_id": i.get("installation_id"),
            "installation_name": i.get("installation_name"),
            "country_iso": i.get("country_iso"),
            "product_groups": i.get("product_groups", []),
            "verification_status": i.get("verification_status"),
        }
        for i in installations
    ]

    # Get authorized emissions data
    authorized_periods = grant.get("requested_periods")
    emissions = [
        s for s in _submissions.values()
        if (
            s.get("supplier_id") == grant_supplier_id
            and s.get("status") == SubmissionStatus.ACCEPTED.value
            and (
                authorized_inst_ids is None
                or s.get("installation_id") in authorized_inst_ids
            )
            and (
                authorized_periods is None
                or s.get("reporting_period") in authorized_periods
            )
        )
    ]

    emissions_data = [
        {
            "submission_id": s.get("submission_id"),
            "installation_id": s.get("installation_id"),
            "reporting_period": s.get("reporting_period"),
            "product_group": s.get("product_group"),
            "cn_code": s.get("cn_code"),
            "direct_emissions_tco2_per_ton": s.get("direct_emissions_tco2_per_ton"),
            "indirect_emissions_tco2_per_ton": s.get("indirect_emissions_tco2_per_ton"),
            "total_emissions_tco2_per_ton": s.get("total_emissions_tco2_per_ton"),
            "production_volume_tons": s.get("production_volume_tons"),
            "data_quality": s.get("data_quality"),
            "methodology": s.get("methodology"),
        }
        for s in emissions
    ]

    # Calculate access expiry
    created_at_str = grant.get("created_at", _now_iso())
    try:
        created_dt = datetime.fromisoformat(created_at_str)
    except (ValueError, TypeError):
        created_dt = datetime.now(timezone.utc)

    access_duration = grant.get("access_duration_days", 365)
    expires_at = (created_dt + timedelta(days=access_duration)).isoformat()

    payload = {
        "importer_id": importer_id,
        "supplier_id": grant_supplier_id,
        "emission_records": len(emissions_data),
    }

    return AuthorizedDataResponse(
        importer_id=importer_id,
        supplier_id=grant_supplier_id,
        supplier_name=supplier.get("company_name", "Unknown"),
        access_expires_at=expires_at,
        installations=installation_summaries,
        emissions_data=emissions_data,
        provenance=_provenance(payload),
    )


@router.get(
    "/exchange/{supplier_id}/audit-log",
    response_model=AuditLogResponse,
    summary="Get supplier audit log",
    description=(
        "Retrieve the complete audit trail for a supplier, including all actions "
        "performed on the supplier's profile, installations, submissions, and "
        "data access grants."
    ),
)
async def get_audit_log(
    supplier_id: str,
    action_filter: Optional[AuditAction] = Query(None, alias="action", description="Filter by action type"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(50, ge=1, le=200, description="Page size (max 200)"),
) -> AuditLogResponse:
    """Get audit log for a supplier."""
    _get_supplier_or_404(supplier_id)

    # Collect all audit entries related to this supplier
    entries = [
        e for e in _audit_log
        if (
            e.get("resource_id") == supplier_id
            or e.get("actor_id") == supplier_id
            or e.get("details", {}).get("supplier_id") == supplier_id
        )
    ]

    # Also include entries for supplier's installations and submissions
    supplier_inst_ids = {
        i.get("installation_id")
        for i in _installations.values()
        if i.get("supplier_id") == supplier_id
    }
    supplier_sub_ids = {
        s.get("submission_id")
        for s in _submissions.values()
        if s.get("supplier_id") == supplier_id
    }
    related_ids = supplier_inst_ids | supplier_sub_ids

    for e in _audit_log:
        if e.get("resource_id") in related_ids and e not in entries:
            entries.append(e)

    if action_filter:
        entries = [e for e in entries if e.get("action") == action_filter.value]

    # Sort by timestamp descending
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    total = len(entries)
    page = entries[offset: offset + limit]

    log_entries = [
        AuditLogEntry(
            log_id=e.get("log_id", ""),
            timestamp=e.get("timestamp", ""),
            action=e.get("action", ""),
            actor_id=e.get("actor_id", ""),
            actor_type=e.get("actor_type", ""),
            resource_type=e.get("resource_type", ""),
            resource_id=e.get("resource_id", ""),
            details=e.get("details", {}),
            ip_address=e.get("ip_address"),
            provenance_hash=e.get("provenance_hash", ""),
        )
        for e in page
    ]

    payload = {"supplier_id": supplier_id, "total_entries": total}
    return AuditLogResponse(
        supplier_id=supplier_id,
        entries=log_entries,
        pagination=PaginationMeta(
            total=total,
            offset=offset,
            limit=limit,
            has_more=(offset + limit) < total,
        ),
        provenance=_provenance(payload),
    )

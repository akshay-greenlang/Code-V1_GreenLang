# -*- coding: utf-8 -*-
"""
API Schemas - AGENT-EUDR-008 Multi-Tier Supplier Tracker

Pydantic v2 request/response models for the Multi-Tier Supplier Tracker
REST API. Covers supplier discovery, profile management, tier depth
tracking, relationship lifecycle, risk propagation, compliance monitoring,
gap analysis, audit reporting, and batch processing operations.

Core domain models are imported from the main models module; this file
defines API-level request wrappers, response envelopes, and batch schemas.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EUDR_COMMODITIES: List[str] = [
    "cattle",
    "cocoa",
    "coffee",
    "oil_palm",
    "rubber",
    "soy",
    "wood",
]

CERTIFICATION_TYPES: List[str] = [
    "fsc",
    "rspo",
    "utz",
    "rainforest_alliance",
    "fairtrade",
    "organic",
    "iscc",
    "bonsucro",
    "pefc",
    "roundtable_soy",
]

RISK_CATEGORIES: List[str] = [
    "deforestation_proximity",
    "country_risk",
    "certification_gap",
    "compliance_history",
    "data_quality",
    "concentration_risk",
]

PROPAGATION_METHODS: List[str] = [
    "max",
    "weighted_average",
    "volume_weighted",
]

RELATIONSHIP_STATES: List[str] = [
    "prospective",
    "onboarding",
    "active",
    "suspended",
    "terminated",
]

CONFIDENCE_LEVELS: List[str] = [
    "verified",
    "declared",
    "inferred",
    "suspected",
]

REPORT_FORMATS: List[str] = [
    "json",
    "pdf",
    "csv",
    "eudr_xml",
]

GAP_SEVERITIES: List[str] = [
    "critical",
    "major",
    "minor",
]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ComplianceStatusEnum(str, Enum):
    """EUDR supplier compliance status values."""

    COMPLIANT = "compliant"
    CONDITIONALLY_COMPLIANT = "conditionally_compliant"
    NON_COMPLIANT = "non_compliant"
    UNVERIFIED = "unverified"
    EXPIRED = "expired"


class DiscoverySourceEnum(str, Enum):
    """Source type for supplier discovery."""

    DECLARATION = "declaration"
    QUESTIONNAIRE = "questionnaire"
    SHIPPING_DOCUMENT = "shipping_document"
    CERTIFICATION_DB = "certification_db"
    ERP = "erp"
    MANUAL = "manual"


class BatchJobStatusEnum(str, Enum):
    """Batch job processing status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportTypeEnum(str, Enum):
    """Report type identifiers."""

    AUDIT = "audit"
    TIER_SUMMARY = "tier_summary"
    GAP_ANALYSIS = "gap_analysis"
    RISK_PROPAGATION = "risk_propagation"
    COMPLIANCE = "compliance"


# =============================================================================
# Pagination
# =============================================================================


class PaginatedMeta(BaseModel):
    """Pagination metadata for list responses."""

    total: int = Field(..., ge=0, description="Total number of results")
    limit: int = Field(..., ge=1, description="Maximum results returned")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")


class PaginatedResponse(BaseModel):
    """Generic paginated response wrapper."""

    items: List[Dict[str, Any]] = Field(
        default_factory=list, description="Page of result items"
    )
    meta: PaginatedMeta = Field(..., description="Pagination metadata")

    model_config = ConfigDict(from_attributes=True)


class PaginationParams(BaseModel):
    """Standard pagination query parameters."""

    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


# =============================================================================
# Response Wrappers
# =============================================================================


class ApiResponse(BaseModel):
    """Standard API success response wrapper."""

    status: str = Field(default="success", description="Response status")
    message: str = Field(default="", description="Response message")
    data: Optional[Any] = Field(None, description="Response payload")
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class ErrorResponse(BaseModel):
    """Structured error response for all API endpoints."""

    error: str = Field(..., description="Error type identifier")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request correlation ID")


# =============================================================================
# Health Check
# =============================================================================


class HealthResponseSchema(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service health status")
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent display name")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Current server timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Supplier Location Schema
# =============================================================================


class SupplierLocationSchema(BaseModel):
    """Supplier location data."""

    country_iso: str = Field(
        ...,
        min_length=2,
        max_length=2,
        description="ISO 3166-1 alpha-2 country code",
    )
    admin_region: Optional[str] = Field(
        None, max_length=200, description="Administrative region or province"
    )
    address: Optional[str] = Field(
        None, max_length=500, description="Full postal address"
    )
    latitude: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Latitude in decimal degrees"
    )
    longitude: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Longitude in decimal degrees"
    )

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: str) -> str:
        """Normalize country code to uppercase."""
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Certification Schema
# =============================================================================


class CertificationSchema(BaseModel):
    """Supplier certification record."""

    certification_type: str = Field(
        ..., description="Certification type (e.g., fsc, rspo, utz)"
    )
    certificate_id: str = Field(
        ..., max_length=200, description="Certificate identifier"
    )
    issued_date: Optional[datetime] = Field(
        None, description="Certificate issue date"
    )
    expiry_date: Optional[datetime] = Field(
        None, description="Certificate expiry date"
    )
    issuing_body: Optional[str] = Field(
        None, max_length=200, description="Issuing organization"
    )
    scope: Optional[str] = Field(
        None, max_length=500, description="Certification scope description"
    )
    status: str = Field(
        default="active",
        description="Certification status: active, expired, suspended, revoked",
    )

    @field_validator("certification_type")
    @classmethod
    def validate_cert_type(cls, v: str) -> str:
        """Validate certification type."""
        v = v.lower().strip()
        if v not in CERTIFICATION_TYPES:
            raise ValueError(
                f"Invalid certification type: '{v}'. "
                f"Valid values: {CERTIFICATION_TYPES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Contact Schema
# =============================================================================


class ContactSchema(BaseModel):
    """Supplier contact information."""

    name: str = Field(..., max_length=200, description="Contact name")
    email: Optional[str] = Field(
        None, max_length=200, description="Contact email"
    )
    phone: Optional[str] = Field(
        None, max_length=50, description="Contact phone number"
    )
    role: str = Field(
        default="primary", description="Contact role: primary, compliance, logistics"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Discovery Schemas
# =============================================================================


class DiscoverRequestSchema(BaseModel):
    """Request to discover sub-tier suppliers from a Tier 1 supplier."""

    supplier_id: str = Field(
        ..., max_length=100, description="Tier 1 supplier identifier"
    )
    commodity: str = Field(
        ..., description="EUDR commodity to discover suppliers for"
    )
    max_depth: int = Field(
        default=15,
        ge=1,
        le=20,
        description="Maximum discovery depth (default 15)",
    )
    sources: List[str] = Field(
        default_factory=lambda: ["declaration", "certification_db"],
        description="Discovery data sources to use",
    )
    include_inferred: bool = Field(
        default=True,
        description="Include inferred (low-confidence) supplier relationships",
    )
    country_filter: Optional[str] = Field(
        None,
        min_length=2,
        max_length=2,
        description="Limit discovery to specific country (ISO 3166-1 alpha-2)",
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("country_filter")
    @classmethod
    def validate_country_filter(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country filter to uppercase."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_filter must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "supplier_id": "SUP-001",
                    "commodity": "cocoa",
                    "max_depth": 8,
                    "sources": ["declaration", "certification_db"],
                    "include_inferred": True,
                    "country_filter": "GH",
                }
            ]
        },
    )


class DiscoveredSupplierSchema(BaseModel):
    """A single discovered sub-tier supplier."""

    supplier_id: str = Field(..., description="Generated or existing supplier ID")
    name: str = Field(..., max_length=300, description="Supplier name")
    tier: int = Field(..., ge=1, description="Tier level in the supply chain")
    parent_supplier_id: Optional[str] = Field(
        None, description="Direct upstream supplier ID"
    )
    commodity: str = Field(..., description="EUDR commodity handled")
    country_iso: Optional[str] = Field(
        None, description="ISO 3166-1 alpha-2 country code"
    )
    confidence: str = Field(
        default="inferred", description="Confidence level of discovery"
    )
    source: str = Field(
        default="certification_db", description="Discovery data source"
    )
    certifications: List[str] = Field(
        default_factory=list, description="Known certification types"
    )
    profile_completeness: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Profile completeness score (0-100)",
    )

    model_config = ConfigDict(from_attributes=True)


class DiscoverResponseSchema(BaseModel):
    """Response from supplier discovery operation."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique request identifier",
    )
    supplier_id: str = Field(..., description="Root supplier that was analyzed")
    commodity: str = Field(..., description="EUDR commodity")
    total_discovered: int = Field(
        ..., ge=0, description="Total suppliers discovered"
    )
    max_depth_reached: int = Field(
        ..., ge=0, description="Maximum tier depth reached"
    )
    discovered_suppliers: List[DiscoveredSupplierSchema] = Field(
        default_factory=list, description="List of discovered suppliers"
    )
    discovery_sources_used: List[str] = Field(
        default_factory=list, description="Data sources used for discovery"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchDiscoverRequestSchema(BaseModel):
    """Batch supplier discovery request."""

    discoveries: List[DiscoverRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of discovery requests (max 100)",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchDiscoverResponseSchema(BaseModel):
    """Batch supplier discovery response."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch request identifier",
    )
    total_requests: int = Field(
        ..., ge=0, description="Total discovery requests processed"
    )
    total_discovered: int = Field(
        ..., ge=0, description="Total suppliers discovered across all requests"
    )
    results: List[DiscoverResponseSchema] = Field(
        default_factory=list, description="Individual discovery results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Errors for failed discovery requests"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class DeclarationDiscoverRequestSchema(BaseModel):
    """Discover suppliers from a supplier declaration document."""

    declaration_text: str = Field(
        ...,
        min_length=10,
        max_length=50000,
        description="Raw supplier declaration text content",
    )
    declaring_supplier_id: str = Field(
        ..., max_length=100, description="ID of the supplier making the declaration"
    )
    commodity: str = Field(
        ..., description="EUDR commodity context"
    )
    max_depth: int = Field(
        default=15, ge=1, le=20, description="Maximum discovery depth"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class QuestionnaireDiscoverRequestSchema(BaseModel):
    """Discover suppliers from questionnaire responses."""

    questionnaire_data: Dict[str, Any] = Field(
        ..., description="Structured questionnaire response data"
    )
    responding_supplier_id: str = Field(
        ..., max_length=100, description="ID of the supplier who responded"
    )
    commodity: str = Field(
        ..., description="EUDR commodity context"
    )
    max_depth: int = Field(
        default=15, ge=1, le=20, description="Maximum discovery depth"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Supplier Profile Schemas
# =============================================================================


class CreateSupplierSchema(BaseModel):
    """Request to create a new supplier profile."""

    name: str = Field(
        ..., min_length=1, max_length=300, description="Legal entity name"
    )
    registration_id: Optional[str] = Field(
        None, max_length=100, description="Business registration ID"
    )
    tax_id: Optional[str] = Field(
        None, max_length=100, description="Tax identification number"
    )
    duns_number: Optional[str] = Field(
        None, max_length=20, description="DUNS number"
    )
    location: SupplierLocationSchema = Field(
        ..., description="Supplier location data"
    )
    commodities: List[str] = Field(
        ...,
        min_length=1,
        description="EUDR commodities handled by this supplier",
    )
    tier: int = Field(
        ..., ge=1, le=20, description="Tier level in the supply chain"
    )
    certifications: List[CertificationSchema] = Field(
        default_factory=list, description="Supplier certifications"
    )
    contacts: List[ContactSchema] = Field(
        default_factory=list, description="Supplier contacts"
    )
    annual_volume_tonnes: Optional[float] = Field(
        None, ge=0, description="Annual volume in metric tonnes"
    )
    processing_capacity_tonnes: Optional[float] = Field(
        None, ge=0, description="Processing capacity in metric tonnes per year"
    )
    upstream_supplier_count: Optional[int] = Field(
        None, ge=0, description="Number of known upstream suppliers"
    )
    dds_references: List[str] = Field(
        default_factory=list,
        description="Linked DDS IDs from EU Information System",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("commodities")
    @classmethod
    def validate_commodities(cls, v: List[str]) -> List[str]:
        """Validate all commodities are valid EUDR commodities."""
        validated = []
        for c in v:
            c = c.lower().strip()
            if c not in EUDR_COMMODITIES:
                raise ValueError(
                    f"Invalid EUDR commodity: '{c}'. "
                    f"Valid values: {EUDR_COMMODITIES}"
                )
            validated.append(c)
        return validated

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "name": "Ghana Cocoa Cooperative Ltd",
                    "registration_id": "GH-REG-2024-001",
                    "location": {
                        "country_iso": "GH",
                        "admin_region": "Ashanti",
                        "latitude": 6.6884,
                        "longitude": -1.6244,
                    },
                    "commodities": ["cocoa"],
                    "tier": 3,
                    "certifications": [
                        {
                            "certification_type": "utz",
                            "certificate_id": "UTZ-GH-2024-001",
                            "status": "active",
                        }
                    ],
                    "annual_volume_tonnes": 5000.0,
                }
            ]
        },
    )


class SupplierProfileSchema(BaseModel):
    """Complete supplier profile response."""

    supplier_id: str = Field(..., description="Unique supplier identifier")
    name: str = Field(..., description="Legal entity name")
    registration_id: Optional[str] = Field(
        None, description="Business registration ID"
    )
    tax_id: Optional[str] = Field(None, description="Tax identification number")
    duns_number: Optional[str] = Field(None, description="DUNS number")
    location: SupplierLocationSchema = Field(
        ..., description="Supplier location data"
    )
    commodities: List[str] = Field(
        ..., description="EUDR commodities handled"
    )
    tier: int = Field(..., ge=1, description="Tier level in the supply chain")
    certifications: List[CertificationSchema] = Field(
        default_factory=list, description="Supplier certifications"
    )
    contacts: List[ContactSchema] = Field(
        default_factory=list, description="Supplier contacts"
    )
    annual_volume_tonnes: Optional[float] = Field(
        None, description="Annual volume in metric tonnes"
    )
    processing_capacity_tonnes: Optional[float] = Field(
        None, description="Processing capacity in metric tonnes per year"
    )
    upstream_supplier_count: Optional[int] = Field(
        None, description="Number of known upstream suppliers"
    )
    dds_references: List[str] = Field(
        default_factory=list, description="Linked DDS IDs"
    )
    profile_completeness: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Profile completeness score (0-100)",
    )
    missing_fields: List[str] = Field(
        default_factory=list, description="Fields missing for full completeness"
    )
    compliance_status: str = Field(
        default="unverified", description="Current compliance status"
    )
    risk_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Composite risk score (0-100)"
    )
    is_active: bool = Field(default=True, description="Whether supplier is active")
    created_at: datetime = Field(
        default_factory=_utcnow, description="Profile creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Profile last update timestamp"
    )
    version: int = Field(default=1, ge=1, description="Profile version number")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class UpdateSupplierSchema(BaseModel):
    """Request to update an existing supplier profile (partial update)."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=300, description="Legal entity name"
    )
    registration_id: Optional[str] = Field(
        None, max_length=100, description="Business registration ID"
    )
    tax_id: Optional[str] = Field(
        None, max_length=100, description="Tax identification number"
    )
    duns_number: Optional[str] = Field(
        None, max_length=20, description="DUNS number"
    )
    location: Optional[SupplierLocationSchema] = Field(
        None, description="Updated location data"
    )
    commodities: Optional[List[str]] = Field(
        None, description="Updated EUDR commodities"
    )
    tier: Optional[int] = Field(
        None, ge=1, le=20, description="Updated tier level"
    )
    certifications: Optional[List[CertificationSchema]] = Field(
        None, description="Updated certifications"
    )
    contacts: Optional[List[ContactSchema]] = Field(
        None, description="Updated contacts"
    )
    annual_volume_tonnes: Optional[float] = Field(
        None, ge=0, description="Updated annual volume"
    )
    processing_capacity_tonnes: Optional[float] = Field(
        None, ge=0, description="Updated processing capacity"
    )
    upstream_supplier_count: Optional[int] = Field(
        None, ge=0, description="Updated upstream supplier count"
    )
    dds_references: Optional[List[str]] = Field(
        None, description="Updated DDS references"
    )
    is_active: Optional[bool] = Field(None, description="Active status")
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata"
    )

    @field_validator("commodities")
    @classmethod
    def validate_commodities(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate all commodities if provided."""
        if v is None:
            return v
        validated = []
        for c in v:
            c = c.lower().strip()
            if c not in EUDR_COMMODITIES:
                raise ValueError(
                    f"Invalid EUDR commodity: '{c}'. "
                    f"Valid values: {EUDR_COMMODITIES}"
                )
            validated.append(c)
        return validated

    model_config = ConfigDict(from_attributes=True)


class SearchCriteriaSchema(BaseModel):
    """Search criteria for supplier profiles."""

    query: Optional[str] = Field(
        None,
        max_length=500,
        description="Free-text search query against supplier name and IDs",
    )
    commodity: Optional[str] = Field(
        None, description="Filter by EUDR commodity"
    )
    country_iso: Optional[str] = Field(
        None, min_length=2, max_length=2, description="Filter by country"
    )
    tier: Optional[int] = Field(
        None, ge=1, le=20, description="Filter by tier level"
    )
    min_tier: Optional[int] = Field(
        None, ge=1, le=20, description="Minimum tier level (inclusive)"
    )
    max_tier: Optional[int] = Field(
        None, ge=1, le=20, description="Maximum tier level (inclusive)"
    )
    compliance_status: Optional[str] = Field(
        None,
        description="Filter by compliance status",
    )
    certification_type: Optional[str] = Field(
        None, description="Filter by certification type"
    )
    min_risk_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Minimum risk score filter"
    )
    max_risk_score: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Maximum risk score filter"
    )
    is_active: Optional[bool] = Field(
        None, description="Filter by active status"
    )
    has_gps: Optional[bool] = Field(
        None, description="Filter by GPS coordinate availability"
    )
    has_certification: Optional[bool] = Field(
        None, description="Filter by certification availability"
    )
    created_after: Optional[datetime] = Field(
        None, description="Filter by creation date (after)"
    )
    created_before: Optional[datetime] = Field(
        None, description="Filter by creation date (before)"
    )
    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Results to skip")
    sort_by: str = Field(
        default="name",
        description="Sort field: name, tier, risk_score, compliance_status, created_at",
    )
    sort_order: str = Field(
        default="asc", description="Sort order: asc or desc"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate commodity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(cls, v: str) -> str:
        """Validate sort order."""
        v = v.lower().strip()
        if v not in ("asc", "desc"):
            raise ValueError("sort_order must be 'asc' or 'desc'")
        return v

    model_config = ConfigDict(from_attributes=True)


class SupplierSearchResponseSchema(BaseModel):
    """Supplier search results."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    total: int = Field(..., ge=0, description="Total matching suppliers")
    suppliers: List[SupplierProfileSchema] = Field(
        default_factory=list, description="Matching supplier profiles"
    )
    limit: int = Field(..., ge=1, description="Results per page")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchSupplierRequestSchema(BaseModel):
    """Batch create/update supplier profiles."""

    suppliers: List[CreateSupplierSchema] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="List of supplier profiles to create/update (max 500)",
    )
    upsert: bool = Field(
        default=True,
        description="If true, update existing suppliers; if false, skip duplicates",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchSupplierResponseSchema(BaseModel):
    """Batch supplier operation response."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch request identifier",
    )
    total_submitted: int = Field(..., ge=0, description="Total suppliers submitted")
    created: int = Field(..., ge=0, description="Suppliers created")
    updated: int = Field(..., ge=0, description="Suppliers updated")
    skipped: int = Field(..., ge=0, description="Suppliers skipped")
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-supplier errors"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Tier Depth Schemas
# =============================================================================


class TierDepthRequestSchema(BaseModel):
    """Request to assess tier depth for a supply chain."""

    root_supplier_id: str = Field(
        ..., max_length=100, description="Root supplier to analyze from"
    )
    commodity: str = Field(
        ..., description="EUDR commodity for tier assessment"
    )
    include_inactive: bool = Field(
        default=False,
        description="Include inactive/terminated suppliers in analysis",
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class TierLevelSummarySchema(BaseModel):
    """Summary statistics for a single tier level."""

    tier: int = Field(..., ge=1, description="Tier level number")
    supplier_count: int = Field(
        ..., ge=0, description="Number of suppliers at this tier"
    )
    visibility_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Visibility percentage at this tier (0-100)",
    )
    avg_completeness: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Average profile completeness at this tier",
    )
    certified_count: int = Field(
        ..., ge=0, description="Suppliers with valid certifications"
    )
    gps_coverage_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of suppliers with GPS coordinates",
    )

    model_config = ConfigDict(from_attributes=True)


class TierDepthResponseSchema(BaseModel):
    """Response for tier depth assessment."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    root_supplier_id: str = Field(
        ..., description="Root supplier analyzed"
    )
    commodity: str = Field(..., description="EUDR commodity analyzed")
    max_depth: int = Field(
        ..., ge=0, description="Maximum tier depth found"
    )
    total_suppliers: int = Field(
        ..., ge=0, description="Total suppliers in the chain"
    )
    overall_visibility_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall supply chain visibility score (0-100)",
    )
    volume_coverage_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of volume with full origin traceability",
    )
    tier_summaries: List[TierLevelSummarySchema] = Field(
        default_factory=list, description="Per-tier summary statistics"
    )
    industry_avg_depth: Optional[float] = Field(
        None, description="Industry average tier depth for comparison"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class VisibilityScoreSchema(BaseModel):
    """Visibility score response across multiple chains."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    scores: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Visibility scores per commodity/supplier chain",
    )
    overall_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall visibility score across all chains",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class TierGapSchema(BaseModel):
    """Tier coverage gap details."""

    tier: int = Field(..., ge=1, description="Tier level with gap")
    gap_type: str = Field(
        ..., description="Type: no_suppliers, low_visibility, low_completeness"
    )
    severity: str = Field(
        ..., description="Gap severity: critical, major, minor"
    )
    description: str = Field(
        ..., description="Human-readable gap description"
    )
    affected_volume_pct: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of volume affected by this gap",
    )
    remediation_priority: int = Field(
        ..., ge=1, le=10, description="Remediation priority (1=highest)"
    )

    model_config = ConfigDict(from_attributes=True)


class TierGapsResponseSchema(BaseModel):
    """Response for tier coverage gap analysis."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    total_gaps: int = Field(..., ge=0, description="Total gaps identified")
    critical_gaps: int = Field(..., ge=0, description="Critical gaps count")
    major_gaps: int = Field(..., ge=0, description="Major gaps count")
    minor_gaps: int = Field(..., ge=0, description="Minor gaps count")
    gaps: List[TierGapSchema] = Field(
        default_factory=list, description="List of identified gaps"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Relationship Schemas
# =============================================================================


class CreateRelationshipSchema(BaseModel):
    """Request to create a supplier relationship."""

    parent_supplier_id: str = Field(
        ..., max_length=100, description="Upstream (parent) supplier ID"
    )
    child_supplier_id: str = Field(
        ..., max_length=100, description="Downstream (child) supplier ID"
    )
    commodity: str = Field(
        ..., description="EUDR commodity for this relationship"
    )
    relationship_state: str = Field(
        default="active", description="Relationship state"
    )
    volume_tonnes: Optional[float] = Field(
        None, ge=0, description="Annual volume in metric tonnes"
    )
    frequency: Optional[str] = Field(
        None,
        description="Transaction frequency: daily, weekly, monthly, seasonal, annual",
    )
    is_exclusive: bool = Field(
        default=False,
        description="Whether this is an exclusive supplier relationship",
    )
    start_date: Optional[datetime] = Field(
        None, description="Relationship start date"
    )
    end_date: Optional[datetime] = Field(
        None, description="Relationship end date (for terminated/seasonal)"
    )
    confidence: str = Field(
        default="declared", description="Confidence level of this relationship"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("relationship_state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        """Validate relationship state."""
        v = v.lower().strip()
        if v not in RELATIONSHIP_STATES:
            raise ValueError(
                f"Invalid state: '{v}'. Valid values: {RELATIONSHIP_STATES}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: str) -> str:
        """Validate confidence level."""
        v = v.lower().strip()
        if v not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"Invalid confidence: '{v}'. Valid values: {CONFIDENCE_LEVELS}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class RelationshipSchema(BaseModel):
    """Supplier relationship record."""

    relationship_id: str = Field(
        ..., description="Unique relationship identifier"
    )
    parent_supplier_id: str = Field(
        ..., description="Upstream (parent) supplier ID"
    )
    child_supplier_id: str = Field(
        ..., description="Downstream (child) supplier ID"
    )
    commodity: str = Field(..., description="EUDR commodity")
    relationship_state: str = Field(
        ..., description="Current relationship state"
    )
    volume_tonnes: Optional[float] = Field(
        None, description="Annual volume in metric tonnes"
    )
    frequency: Optional[str] = Field(None, description="Transaction frequency")
    is_exclusive: bool = Field(
        default=False, description="Whether relationship is exclusive"
    )
    start_date: Optional[datetime] = Field(
        None, description="Relationship start date"
    )
    end_date: Optional[datetime] = Field(
        None, description="Relationship end date"
    )
    confidence: str = Field(
        default="declared", description="Confidence level"
    )
    strength_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Relationship strength score (0-100)",
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=_utcnow, description="Last update timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class UpdateRelationshipSchema(BaseModel):
    """Request to update a supplier relationship."""

    relationship_state: Optional[str] = Field(
        None, description="Updated relationship state"
    )
    volume_tonnes: Optional[float] = Field(
        None, ge=0, description="Updated annual volume"
    )
    frequency: Optional[str] = Field(
        None, description="Updated transaction frequency"
    )
    is_exclusive: Optional[bool] = Field(
        None, description="Updated exclusivity status"
    )
    end_date: Optional[datetime] = Field(
        None, description="Updated end date"
    )
    confidence: Optional[str] = Field(
        None, description="Updated confidence level"
    )
    reason: Optional[str] = Field(
        None, max_length=500, description="Reason for the update"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Updated metadata"
    )

    @field_validator("relationship_state")
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Validate relationship state if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in RELATIONSHIP_STATES:
            raise ValueError(
                f"Invalid state: '{v}'. Valid values: {RELATIONSHIP_STATES}"
            )
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: Optional[str]) -> Optional[str]:
        """Validate confidence level if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"Invalid confidence: '{v}'. Valid values: {CONFIDENCE_LEVELS}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class RelationshipHistoryRequestSchema(BaseModel):
    """Request for relationship change history."""

    supplier_id: str = Field(
        ..., max_length=100, description="Supplier ID to get history for"
    )
    relationship_id: Optional[str] = Field(
        None, description="Specific relationship ID (optional)"
    )
    start_date: Optional[datetime] = Field(
        None, description="History start date filter"
    )
    end_date: Optional[datetime] = Field(
        None, description="History end date filter"
    )
    limit: int = Field(default=50, ge=1, le=1000, description="Results per page")
    offset: int = Field(default=0, ge=0, description="Results to skip")

    model_config = ConfigDict(from_attributes=True)


class RelationshipChangeSchema(BaseModel):
    """A single relationship change history entry."""

    change_id: str = Field(..., description="Change event identifier")
    relationship_id: str = Field(..., description="Relationship identifier")
    field_changed: str = Field(..., description="Field that was changed")
    old_value: Optional[str] = Field(None, description="Previous value")
    new_value: Optional[str] = Field(None, description="New value")
    reason: Optional[str] = Field(None, description="Reason for change")
    changed_by: str = Field(..., description="User who made the change")
    changed_at: datetime = Field(
        ..., description="Timestamp of the change"
    )

    model_config = ConfigDict(from_attributes=True)


class RelationshipHistoryResponseSchema(BaseModel):
    """Response for relationship history query."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    supplier_id: str = Field(..., description="Supplier analyzed")
    total_changes: int = Field(
        ..., ge=0, description="Total change events found"
    )
    changes: List[RelationshipChangeSchema] = Field(
        default_factory=list, description="Change history entries"
    )
    limit: int = Field(..., ge=1, description="Results per page")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class SupplierRelationshipsResponseSchema(BaseModel):
    """Response listing all relationships for a supplier."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    supplier_id: str = Field(..., description="Supplier analyzed")
    upstream: List[RelationshipSchema] = Field(
        default_factory=list, description="Upstream (parent) relationships"
    )
    downstream: List[RelationshipSchema] = Field(
        default_factory=list, description="Downstream (child) relationships"
    )
    total_relationships: int = Field(
        ..., ge=0, description="Total relationship count"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Risk Assessment Schemas
# =============================================================================


class RiskAssessmentRequestSchema(BaseModel):
    """Request to assess risk for a supplier."""

    supplier_id: str = Field(
        ..., max_length=100, description="Supplier to assess"
    )
    commodity: Optional[str] = Field(
        None, description="Commodity context for risk assessment"
    )
    risk_categories: List[str] = Field(
        default_factory=lambda: list(RISK_CATEGORIES),
        description="Risk categories to evaluate",
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        None, description="Custom weights per risk category (must sum to 1.0)"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("risk_categories")
    @classmethod
    def validate_risk_categories(cls, v: List[str]) -> List[str]:
        """Validate all risk categories."""
        validated = []
        for c in v:
            c = c.lower().strip()
            if c not in RISK_CATEGORIES:
                raise ValueError(
                    f"Invalid risk category: '{c}'. "
                    f"Valid values: {RISK_CATEGORIES}"
                )
            validated.append(c)
        return validated

    model_config = ConfigDict(from_attributes=True)


class RiskCategoryScoreSchema(BaseModel):
    """Score for a single risk category."""

    category: str = Field(..., description="Risk category name")
    score: float = Field(
        ..., ge=0.0, le=100.0, description="Category risk score (0-100)"
    )
    weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight applied to this category"
    )
    weighted_score: float = Field(
        ..., ge=0.0, le=100.0, description="Score * weight contribution"
    )
    factors: List[str] = Field(
        default_factory=list, description="Contributing risk factors"
    )
    trend: str = Field(
        default="stable",
        description="Risk trend: improving, stable, degrading",
    )

    model_config = ConfigDict(from_attributes=True)


class RiskScoreSchema(BaseModel):
    """Complete risk assessment response for a supplier."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    supplier_id: str = Field(..., description="Assessed supplier ID")
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite risk score (0-100)",
    )
    risk_level: str = Field(
        ..., description="Risk level: low, medium, high, critical"
    )
    category_scores: List[RiskCategoryScoreSchema] = Field(
        default_factory=list, description="Per-category risk breakdown"
    )
    overall_trend: str = Field(
        default="stable",
        description="Overall risk trend: improving, stable, degrading",
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp"
    )
    next_assessment_due: Optional[datetime] = Field(
        None, description="Next scheduled assessment date"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class RiskPropagationRequestSchema(BaseModel):
    """Request to propagate risk through a supply chain."""

    root_supplier_id: str = Field(
        ..., max_length=100, description="Root supplier to propagate from"
    )
    commodity: str = Field(
        ..., description="EUDR commodity for propagation"
    )
    propagation_method: str = Field(
        default="weighted_average",
        description="Propagation method: max, weighted_average, volume_weighted",
    )
    max_depth: int = Field(
        default=15, ge=1, le=20, description="Maximum propagation depth"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("propagation_method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate propagation method."""
        v = v.lower().strip()
        if v not in PROPAGATION_METHODS:
            raise ValueError(
                f"Invalid method: '{v}'. Valid values: {PROPAGATION_METHODS}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class RiskPathNodeSchema(BaseModel):
    """A node in the risk propagation path."""

    supplier_id: str = Field(..., description="Supplier ID")
    supplier_name: str = Field(..., description="Supplier name")
    tier: int = Field(..., ge=1, description="Tier level")
    own_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Supplier's own risk score"
    )
    propagated_risk_score: float = Field(
        ..., ge=0.0, le=100.0, description="Risk score after propagation"
    )
    contributing_children: int = Field(
        ..., ge=0, description="Number of child suppliers contributing risk"
    )

    model_config = ConfigDict(from_attributes=True)


class RiskPropagationResponseSchema(BaseModel):
    """Risk propagation through supply chain response."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    root_supplier_id: str = Field(..., description="Root supplier")
    commodity: str = Field(..., description="EUDR commodity")
    propagation_method: str = Field(
        ..., description="Method used for propagation"
    )
    root_propagated_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Final propagated risk score at root",
    )
    max_depth_reached: int = Field(
        ..., ge=0, description="Maximum depth reached during propagation"
    )
    total_suppliers_assessed: int = Field(
        ..., ge=0, description="Total suppliers in the propagation chain"
    )
    risk_path: List[RiskPathNodeSchema] = Field(
        default_factory=list,
        description="Risk propagation path from deepest tier to root",
    )
    high_risk_suppliers: List[str] = Field(
        default_factory=list,
        description="Supplier IDs with risk score above threshold",
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchRiskRequestSchema(BaseModel):
    """Batch risk assessment request."""

    assessments: List[RiskAssessmentRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of risk assessment requests (max 200)",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchRiskResponseSchema(BaseModel):
    """Batch risk assessment response."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch request identifier",
    )
    total_assessed: int = Field(
        ..., ge=0, description="Total suppliers assessed"
    )
    results: List[RiskScoreSchema] = Field(
        default_factory=list, description="Individual risk assessment results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-assessment errors"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Compliance Schemas
# =============================================================================


class ComplianceCheckRequestSchema(BaseModel):
    """Request to check supplier compliance."""

    supplier_id: str = Field(
        ..., max_length=100, description="Supplier to check"
    )
    commodity: Optional[str] = Field(
        None, description="EUDR commodity context"
    )
    dimensions: List[str] = Field(
        default_factory=lambda: [
            "dds_validity",
            "certification_status",
            "geolocation_coverage",
            "deforestation_free",
        ],
        description="Compliance dimensions to check",
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class ComplianceDimensionSchema(BaseModel):
    """Score for a single compliance dimension."""

    dimension: str = Field(..., description="Compliance dimension name")
    status: str = Field(
        ..., description="Dimension status: pass, fail, warning, unknown"
    )
    score: float = Field(
        ..., ge=0.0, le=100.0, description="Dimension score (0-100)"
    )
    details: str = Field(
        default="", description="Human-readable status details"
    )
    expiry_date: Optional[datetime] = Field(
        None, description="Relevant expiry date (DDS, certification)"
    )
    days_until_expiry: Optional[int] = Field(
        None, description="Days until expiry (negative = expired)"
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceCheckResponseSchema(BaseModel):
    """Compliance check result for a supplier."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    supplier_id: str = Field(..., description="Checked supplier ID")
    compliance_status: str = Field(
        ..., description="Overall compliance status"
    )
    composite_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Composite compliance score (0-100)",
    )
    dimensions: List[ComplianceDimensionSchema] = Field(
        default_factory=list, description="Per-dimension compliance results"
    )
    dds_impact: str = Field(
        default="",
        description="Impact on DDS: can_include, can_include_with_disclosure, cannot_include",
    )
    remediation_required: bool = Field(
        default=False,
        description="Whether remediation actions are needed",
    )
    remediation_actions: List[str] = Field(
        default_factory=list,
        description="Recommended remediation actions",
    )
    assessment_date: datetime = Field(
        default_factory=_utcnow, description="Assessment timestamp"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceStatusSchema(BaseModel):
    """Current compliance status for a supplier."""

    supplier_id: str = Field(..., description="Supplier ID")
    compliance_status: str = Field(
        ..., description="Current compliance status"
    )
    composite_score: float = Field(
        ..., ge=0.0, le=100.0, description="Composite compliance score"
    )
    dimensions: List[ComplianceDimensionSchema] = Field(
        default_factory=list, description="Per-dimension status"
    )
    last_checked: datetime = Field(
        ..., description="Last compliance check timestamp"
    )
    next_check_due: Optional[datetime] = Field(
        None, description="Next scheduled check"
    )
    trend: str = Field(
        default="stable",
        description="Compliance trend: improving, stable, degrading",
    )
    history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent compliance status changes",
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceAlertSchema(BaseModel):
    """Compliance alert for status changes or approaching expirations."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Alert identifier",
    )
    supplier_id: str = Field(..., description="Affected supplier ID")
    supplier_name: str = Field(..., description="Supplier name")
    alert_type: str = Field(
        ...,
        description="Alert type: status_change, expiry_warning, non_compliance, data_gap",
    )
    severity: str = Field(
        ..., description="Severity: critical, high, medium, low"
    )
    message: str = Field(
        ..., description="Human-readable alert message"
    )
    dimension: Optional[str] = Field(
        None, description="Affected compliance dimension"
    )
    old_status: Optional[str] = Field(
        None, description="Previous compliance status"
    )
    new_status: Optional[str] = Field(
        None, description="New compliance status"
    )
    days_until_expiry: Optional[int] = Field(
        None, description="Days until relevant expiry"
    )
    created_at: datetime = Field(
        default_factory=_utcnow, description="Alert creation timestamp"
    )
    acknowledged: bool = Field(
        default=False, description="Whether alert has been acknowledged"
    )

    model_config = ConfigDict(from_attributes=True)


class ComplianceAlertsResponseSchema(BaseModel):
    """Response listing compliance alerts."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Request identifier",
    )
    total_alerts: int = Field(..., ge=0, description="Total alerts found")
    critical_count: int = Field(
        ..., ge=0, description="Critical alerts count"
    )
    high_count: int = Field(..., ge=0, description="High alerts count")
    alerts: List[ComplianceAlertSchema] = Field(
        default_factory=list, description="Compliance alerts"
    )
    limit: int = Field(..., ge=1, description="Results per page")
    offset: int = Field(..., ge=0, description="Results skipped")
    has_more: bool = Field(..., description="Whether more results exist")
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


class BatchComplianceRequestSchema(BaseModel):
    """Batch compliance check request."""

    checks: List[ComplianceCheckRequestSchema] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="List of compliance check requests (max 200)",
    )

    model_config = ConfigDict(from_attributes=True)


class BatchComplianceResponseSchema(BaseModel):
    """Batch compliance check response."""

    request_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Batch request identifier",
    )
    total_checked: int = Field(
        ..., ge=0, description="Total suppliers checked"
    )
    compliant_count: int = Field(..., ge=0, description="Compliant count")
    non_compliant_count: int = Field(
        ..., ge=0, description="Non-compliant count"
    )
    results: List[ComplianceCheckResponseSchema] = Field(
        default_factory=list, description="Individual check results"
    )
    errors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Per-check errors"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Total processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )
    timestamp: datetime = Field(
        default_factory=_utcnow, description="Response timestamp"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Report Schemas
# =============================================================================


class AuditReportRequestSchema(BaseModel):
    """Request to generate an EUDR Article 14 audit report."""

    root_supplier_id: str = Field(
        ..., max_length=100, description="Root supplier for the audit chain"
    )
    commodity: str = Field(
        ..., description="EUDR commodity"
    )
    report_format: str = Field(
        default="json",
        description="Report format: json, pdf, csv, eudr_xml",
    )
    include_risk_details: bool = Field(
        default=True, description="Include risk assessment details"
    )
    include_compliance_details: bool = Field(
        default=True, description="Include compliance check details"
    )
    include_relationship_history: bool = Field(
        default=False, description="Include full relationship change history"
    )
    date_from: Optional[datetime] = Field(
        None, description="Report period start date"
    )
    date_to: Optional[datetime] = Field(
        None, description="Report period end date"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate EUDR commodity."""
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("report_format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate report format."""
        v = v.lower().strip()
        if v not in REPORT_FORMATS:
            raise ValueError(
                f"Invalid format: '{v}'. Valid values: {REPORT_FORMATS}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class AuditReportSchema(BaseModel):
    """Generated audit report response."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique report identifier",
    )
    report_type: str = Field(
        default="audit", description="Report type"
    )
    root_supplier_id: str = Field(
        ..., description="Root supplier of the chain"
    )
    commodity: str = Field(..., description="EUDR commodity")
    report_format: str = Field(..., description="Report format")
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Report generation timestamp"
    )
    total_suppliers: int = Field(
        ..., ge=0, description="Total suppliers in the chain"
    )
    max_tier_depth: int = Field(
        ..., ge=0, description="Maximum tier depth in the chain"
    )
    compliance_summary: Dict[str, int] = Field(
        default_factory=dict,
        description="Compliance status counts (compliant, non_compliant, etc.)",
    )
    risk_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Risk score summary statistics",
    )
    data: Optional[Dict[str, Any]] = Field(
        None, description="Full report data (for JSON format)"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL (for PDF/CSV/XML formats)"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Report download expiry timestamp"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class TierSummaryRequestSchema(BaseModel):
    """Request to generate a tier depth summary report."""

    commodity: Optional[str] = Field(
        None, description="EUDR commodity filter"
    )
    country_iso: Optional[str] = Field(
        None, min_length=2, max_length=2, description="Country filter"
    )
    include_benchmarks: bool = Field(
        default=True, description="Include industry benchmark comparisons"
    )
    report_format: str = Field(
        default="json", description="Report format: json, pdf, csv"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("country_iso")
    @classmethod
    def validate_country_iso(cls, v: Optional[str]) -> Optional[str]:
        """Normalize country code."""
        if v is None:
            return v
        v = v.upper().strip()
        if len(v) != 2 or not v.isalpha():
            raise ValueError(
                "country_iso must be a two-letter ISO 3166-1 alpha-2 code"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class TierSummarySchema(BaseModel):
    """Tier depth summary report."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier",
    )
    report_type: str = Field(
        default="tier_summary", description="Report type"
    )
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Generation timestamp"
    )
    total_supply_chains: int = Field(
        ..., ge=0, description="Total supply chains analyzed"
    )
    avg_tier_depth: float = Field(
        ..., ge=0, description="Average tier depth across all chains"
    )
    max_tier_depth: int = Field(
        ..., ge=0, description="Maximum tier depth found"
    )
    total_suppliers: int = Field(
        ..., ge=0, description="Total unique suppliers"
    )
    overall_visibility_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall visibility score (0-100)",
    )
    commodity_breakdown: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tier depth stats per commodity",
    )
    benchmarks: Optional[Dict[str, Any]] = Field(
        None, description="Industry benchmark comparisons"
    )
    report_format: str = Field(
        default="json", description="Report format"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL for non-JSON formats"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class GapReportRequestSchema(BaseModel):
    """Request to generate a gap analysis report."""

    root_supplier_id: Optional[str] = Field(
        None, max_length=100, description="Root supplier to analyze (optional)"
    )
    commodity: Optional[str] = Field(
        None, description="EUDR commodity filter"
    )
    severity_filter: Optional[str] = Field(
        None, description="Gap severity filter: critical, major, minor"
    )
    include_remediation_plans: bool = Field(
        default=True, description="Include remediation action plans"
    )
    report_format: str = Field(
        default="json", description="Report format: json, pdf, csv"
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: Optional[str]) -> Optional[str]:
        """Validate EUDR commodity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in EUDR_COMMODITIES:
            raise ValueError(
                f"Invalid EUDR commodity: '{v}'. Valid values: {EUDR_COMMODITIES}"
            )
        return v

    @field_validator("severity_filter")
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> Optional[str]:
        """Validate gap severity if provided."""
        if v is None:
            return v
        v = v.lower().strip()
        if v not in GAP_SEVERITIES:
            raise ValueError(
                f"Invalid severity: '{v}'. Valid values: {GAP_SEVERITIES}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class DataGapSchema(BaseModel):
    """A single data gap identified for a supplier."""

    gap_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Gap identifier",
    )
    supplier_id: str = Field(..., description="Affected supplier ID")
    supplier_name: str = Field(..., description="Supplier name")
    tier: int = Field(..., ge=1, description="Supplier tier level")
    gap_type: str = Field(
        ...,
        description="Gap type: missing_gps, missing_certification, "
        "missing_legal_entity, missing_dds, outdated_data, unverified",
    )
    severity: str = Field(
        ..., description="Gap severity: critical, major, minor"
    )
    description: str = Field(
        ..., description="Human-readable gap description"
    )
    dds_blocking: bool = Field(
        ..., description="Whether this gap blocks DDS submission"
    )
    remediation_steps: List[str] = Field(
        default_factory=list,
        description="Recommended remediation steps",
    )
    estimated_effort_days: Optional[int] = Field(
        None, ge=0, description="Estimated remediation effort in days"
    )

    model_config = ConfigDict(from_attributes=True)


class GapReportSchema(BaseModel):
    """Gap analysis report response."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier",
    )
    report_type: str = Field(
        default="gap_analysis", description="Report type"
    )
    generated_at: datetime = Field(
        default_factory=_utcnow, description="Generation timestamp"
    )
    total_gaps: int = Field(..., ge=0, description="Total gaps identified")
    critical_gaps: int = Field(..., ge=0, description="Critical gaps")
    major_gaps: int = Field(..., ge=0, description="Major gaps")
    minor_gaps: int = Field(..., ge=0, description="Minor gaps")
    dds_blocking_gaps: int = Field(
        ..., ge=0, description="Gaps that block DDS submission"
    )
    gaps: List[DataGapSchema] = Field(
        default_factory=list, description="Identified gaps"
    )
    gap_trend: str = Field(
        default="stable",
        description="Gap trend vs previous period: improving, stable, worsening",
    )
    remediation_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Remediation effort summary statistics",
    )
    report_format: str = Field(
        default="json", description="Report format"
    )
    download_url: Optional[str] = Field(
        None, description="Download URL for non-JSON formats"
    )
    elapsed_ms: float = Field(
        ..., ge=0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class ReportDownloadSchema(BaseModel):
    """Report download response with pre-signed URL."""

    report_id: str = Field(..., description="Report identifier")
    report_type: str = Field(..., description="Report type")
    report_format: str = Field(..., description="Report format")
    download_url: str = Field(
        ..., description="Pre-signed download URL"
    )
    file_size_bytes: Optional[int] = Field(
        None, ge=0, description="File size in bytes"
    )
    expires_at: datetime = Field(
        ..., description="URL expiry timestamp"
    )
    generated_at: datetime = Field(
        ..., description="Report generation timestamp"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


class ReportMetadataSchema(BaseModel):
    """Report metadata for retrieval."""

    report_id: str = Field(..., description="Report identifier")
    report_type: str = Field(..., description="Report type")
    report_format: str = Field(..., description="Report format")
    status: str = Field(
        ..., description="Report status: generating, ready, expired, failed"
    )
    generated_at: Optional[datetime] = Field(
        None, description="Generation timestamp"
    )
    expires_at: Optional[datetime] = Field(
        None, description="Expiry timestamp"
    )
    file_size_bytes: Optional[int] = Field(
        None, description="File size in bytes"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Batch Job Schemas
# =============================================================================


class BatchJobRequestSchema(BaseModel):
    """Request to submit a batch processing job."""

    job_type: str = Field(
        ...,
        description="Batch job type: discovery, risk_assessment, compliance_check, "
        "profile_import, gap_analysis",
    )
    parameters: Dict[str, Any] = Field(
        ..., description="Job-specific parameters"
    )
    priority: str = Field(
        default="normal",
        description="Job priority: low, normal, high, critical",
    )
    callback_url: Optional[str] = Field(
        None, max_length=500, description="Webhook URL for completion notification"
    )

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: str) -> str:
        """Validate batch job type."""
        v = v.lower().strip()
        allowed = {
            "discovery",
            "risk_assessment",
            "compliance_check",
            "profile_import",
            "gap_analysis",
        }
        if v not in allowed:
            raise ValueError(
                f"Invalid job type: '{v}'. Valid values: {sorted(allowed)}"
            )
        return v

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: str) -> str:
        """Validate priority level."""
        v = v.lower().strip()
        allowed = {"low", "normal", "high", "critical"}
        if v not in allowed:
            raise ValueError(
                f"Invalid priority: '{v}'. Valid values: {sorted(allowed)}"
            )
        return v

    model_config = ConfigDict(from_attributes=True)


class BatchJobResponseSchema(BaseModel):
    """Batch job submission response."""

    batch_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique batch job identifier",
    )
    job_type: str = Field(..., description="Batch job type")
    status: str = Field(
        default="pending", description="Job status"
    )
    priority: str = Field(
        default="normal", description="Job priority"
    )
    submitted_at: datetime = Field(
        default_factory=_utcnow, description="Submission timestamp"
    )
    estimated_duration_seconds: Optional[int] = Field(
        None, ge=0, description="Estimated processing duration"
    )
    total_items: Optional[int] = Field(
        None, ge=0, description="Total items to process"
    )
    processed_items: int = Field(
        default=0, ge=0, description="Items processed so far"
    )
    failed_items: int = Field(
        default=0, ge=0, description="Items that failed processing"
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Processing progress percentage",
    )
    result_summary: Optional[Dict[str, Any]] = Field(
        None, description="Summary of results (when completed)"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if job failed"
    )
    started_at: Optional[datetime] = Field(
        None, description="Processing start timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Processing completion timestamp"
    )
    callback_url: Optional[str] = Field(
        None, description="Webhook callback URL"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# Delete / Deactivate Response
# =============================================================================


class DeactivateResponseSchema(BaseModel):
    """Response for supplier deactivation."""

    supplier_id: str = Field(..., description="Deactivated supplier ID")
    status: str = Field(
        default="deactivated", description="Supplier status"
    )
    message: str = Field(
        ..., description="Deactivation confirmation message"
    )
    deactivated_at: datetime = Field(
        default_factory=_utcnow, description="Deactivation timestamp"
    )
    provenance_hash: str = Field(
        ..., description="SHA-256 hash for audit trail"
    )

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "EUDR_COMMODITIES",
    "CERTIFICATION_TYPES",
    "RISK_CATEGORIES",
    "PROPAGATION_METHODS",
    "RELATIONSHIP_STATES",
    "CONFIDENCE_LEVELS",
    "REPORT_FORMATS",
    "GAP_SEVERITIES",
    # Enums
    "ComplianceStatusEnum",
    "DiscoverySourceEnum",
    "BatchJobStatusEnum",
    "ReportTypeEnum",
    # Pagination
    "PaginatedMeta",
    "PaginatedResponse",
    "PaginationParams",
    # Response Wrappers
    "ApiResponse",
    "ErrorResponse",
    # Health
    "HealthResponseSchema",
    # Location / Certification / Contact
    "SupplierLocationSchema",
    "CertificationSchema",
    "ContactSchema",
    # Discovery
    "DiscoverRequestSchema",
    "DiscoveredSupplierSchema",
    "DiscoverResponseSchema",
    "BatchDiscoverRequestSchema",
    "BatchDiscoverResponseSchema",
    "DeclarationDiscoverRequestSchema",
    "QuestionnaireDiscoverRequestSchema",
    # Profiles
    "CreateSupplierSchema",
    "SupplierProfileSchema",
    "UpdateSupplierSchema",
    "SearchCriteriaSchema",
    "SupplierSearchResponseSchema",
    "BatchSupplierRequestSchema",
    "BatchSupplierResponseSchema",
    "DeactivateResponseSchema",
    # Tiers
    "TierDepthRequestSchema",
    "TierLevelSummarySchema",
    "TierDepthResponseSchema",
    "VisibilityScoreSchema",
    "TierGapSchema",
    "TierGapsResponseSchema",
    # Relationships
    "CreateRelationshipSchema",
    "RelationshipSchema",
    "UpdateRelationshipSchema",
    "RelationshipHistoryRequestSchema",
    "RelationshipChangeSchema",
    "RelationshipHistoryResponseSchema",
    "SupplierRelationshipsResponseSchema",
    # Risk
    "RiskAssessmentRequestSchema",
    "RiskCategoryScoreSchema",
    "RiskScoreSchema",
    "RiskPropagationRequestSchema",
    "RiskPathNodeSchema",
    "RiskPropagationResponseSchema",
    "BatchRiskRequestSchema",
    "BatchRiskResponseSchema",
    # Compliance
    "ComplianceCheckRequestSchema",
    "ComplianceDimensionSchema",
    "ComplianceCheckResponseSchema",
    "ComplianceStatusSchema",
    "ComplianceAlertSchema",
    "ComplianceAlertsResponseSchema",
    "BatchComplianceRequestSchema",
    "BatchComplianceResponseSchema",
    # Reports
    "AuditReportRequestSchema",
    "AuditReportSchema",
    "TierSummaryRequestSchema",
    "TierSummarySchema",
    "GapReportRequestSchema",
    "DataGapSchema",
    "GapReportSchema",
    "ReportDownloadSchema",
    "ReportMetadataSchema",
    # Batch Jobs
    "BatchJobRequestSchema",
    "BatchJobResponseSchema",
]

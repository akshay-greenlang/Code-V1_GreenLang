# -*- coding: utf-8 -*-
"""
Supplier Risk Scorer API Schemas - AGENT-EUDR-017

Pydantic v2 request/response schemas for the Supplier Risk Scorer
REST API. All schemas are designed for OpenAPI/Swagger documentation
with comprehensive field descriptions, validation constraints, and
JSON schema examples.

Schema Groups:
    - Common: PaginationSchema, ErrorSchema, HealthSchema, SuccessSchema
    - Supplier: AssessSupplierRequest, SupplierRiskResponse,
      CompareSupplierRequest, TrendResponse, BatchAssessmentRequest
    - Due Diligence: DDRecordRequest, DDHistoryResponse, DDGapsResponse,
      EscalateIssueRequest
    - Documentation: AnalyzeDocumentRequest, DocumentProfileResponse,
      DocumentGapsResponse, RequestDocumentRequest
    - Certification: ValidateCertificationRequest, CertStatusResponse,
      CertExpiryResponse, VerifyScopeRequest
    - Geographic: AnalyzeSourcingRequest, SourcingProfileResponse,
      RiskZonesResponse, ConcentrationRequest
    - Network: AnalyzeNetworkRequest, NetworkResponse, SubSuppliersResponse,
      RiskPropagationRequest
    - Monitoring: ConfigureMonitoringRequest, AlertResponse,
      WatchlistResponse, PortfolioRiskResponse
    - Report: GenerateReportRequest, ReportResponse, DownloadReportResponse

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# =============================================================================
# Common Schemas
# =============================================================================


class PaginationSchema(BaseModel):
    """Standard pagination metadata for list responses."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total: int = Field(
        default=0, ge=0,
        description="Total number of records matching the query.",
    )
    limit: int = Field(
        default=50, ge=1, le=500,
        description="Maximum number of records per page.",
    )
    offset: int = Field(
        default=0, ge=0,
        description="Number of records skipped from the start.",
    )
    has_more: bool = Field(
        default=False,
        description="Whether additional pages are available.",
    )


class ErrorSchema(BaseModel):
    """Structured error response schema per GreenLang API standards."""

    model_config = ConfigDict(str_strip_whitespace=True)

    error: str = Field(
        ...,
        description="Machine-readable error code (e.g., 'validation_error').",
    )
    message: str = Field(
        ...,
        description="Human-readable error description.",
    )
    details: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Additional error details (field-level validation errors).",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing and support.",
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the error occurred (UTC).",
    )


class HealthSchema(BaseModel):
    """Health check response for the Supplier Risk Scorer service."""

    model_config = ConfigDict(str_strip_whitespace=True)

    status: str = Field(
        default="healthy",
        description="Service health status (healthy/degraded/unhealthy).",
    )
    version: str = Field(
        default="1.0.0",
        description="Service version.",
    )
    agent_id: str = Field(
        default="GL-EUDR-SRS-017",
        description="Agent identifier.",
    )
    suppliers_assessed: int = Field(
        default=0, ge=0,
        description="Number of suppliers with active risk assessments.",
    )
    high_risk_suppliers: int = Field(
        default=0, ge=0,
        description="Number of suppliers classified as high or critical risk.",
    )
    active_alerts: int = Field(
        default=0, ge=0,
        description="Number of active supplier alerts.",
    )
    database_connected: bool = Field(
        default=False,
        description="Whether database connection is healthy.",
    )
    cache_connected: bool = Field(
        default=False,
        description="Whether Redis cache connection is healthy.",
    )
    uptime_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Service uptime in seconds.",
    )
    checked_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp of health check (UTC).",
    )


class SuccessSchema(BaseModel):
    """Generic success response."""

    model_config = ConfigDict(str_strip_whitespace=True)

    success: bool = Field(
        default=True,
        description="Whether the operation succeeded.",
    )
    message: str = Field(
        default="Operation completed successfully.",
        description="Human-readable success message.",
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing.",
    )


# =============================================================================
# Supplier Schemas
# =============================================================================


class FactorScoreSchema(BaseModel):
    """Individual risk factor score breakdown."""

    model_config = ConfigDict(str_strip_whitespace=True)

    factor_name: str = Field(
        ...,
        description="Risk factor name (e.g., geographic_sourcing).",
    )
    raw_score: float = Field(
        ..., ge=0, le=100,
        description="Raw score before normalization (0-100).",
    )
    normalized_score: float = Field(
        ..., ge=0, le=100,
        description="Normalized score (0-100).",
    )
    weight: float = Field(
        ..., ge=0, le=1,
        description="Factor weight in composite score (0.0-1.0).",
    )
    weighted_score: float = Field(
        ..., ge=0, le=100,
        description="Weighted score contribution to composite (0-100).",
    )
    data_sources: List[str] = Field(
        default_factory=list,
        description="Data sources used for scoring.",
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Confidence in factor score (0.0-1.0).",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC).",
    )


class AssessSupplierRequest(BaseModel):
    """Request schema for single supplier risk assessment."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "commodities": ["coffee", "cocoa"],
                "custom_weights": None,
                "include_trend": True,
                "include_network": False,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique supplier identifier.",
    )
    commodities: Optional[List[str]] = Field(
        default=None,
        description="Commodities to assess (defaults to all supplier commodities).",
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Custom factor weights (must sum to 1.0). Default: "
            "geographic_sourcing=0.20, compliance_history=0.15, "
            "documentation_quality=0.15, certification_status=0.15, "
            "traceability_completeness=0.10, financial_stability=0.10, "
            "environmental_performance=0.10, social_compliance=0.05."
        ),
    )
    include_trend: bool = Field(
        default=False,
        description="Include historical risk trend analysis.",
    )
    include_network: bool = Field(
        default=False,
        description="Include supplier network analysis.",
    )

    @field_validator("custom_weights")
    @classmethod
    def validate_custom_weights(
        cls, v: Optional[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        """Validate custom weights sum to 1.0."""
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"Custom weights must sum to 1.0, got {total}")
        return v


class SupplierRiskResponse(BaseModel):
    """Response schema for supplier risk assessment."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "assessment_id": "sra-abc123",
                "supplier_id": "SUP-12345",
                "supplier_name": "Coffee Growers Co-op",
                "risk_score": 45.5,
                "risk_level": "medium",
                "factor_scores": [],
                "confidence": 0.85,
                "trend": "stable",
                "assessed_at": "2026-03-09T10:30:00Z",
            }
        },
    )

    assessment_id: str = Field(
        ...,
        description="Unique assessment identifier.",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    supplier_name: str = Field(
        ...,
        description="Supplier name.",
    )
    risk_score: float = Field(
        ..., ge=0, le=100,
        description="Composite risk score (0-100).",
    )
    risk_level: str = Field(
        ...,
        description="Risk level: low, medium, high, critical.",
    )
    factor_scores: List[FactorScoreSchema] = Field(
        ...,
        description="Individual factor scores (8 factors).",
    )
    confidence: float = Field(
        ..., ge=0, le=1,
        description="Overall confidence in assessment (0.0-1.0).",
    )
    trend: Optional[str] = Field(
        default=None,
        description="Risk trend direction: increasing, stable, decreasing.",
    )
    assessed_at: Optional[datetime] = Field(
        default=None,
        description="Assessment timestamp (UTC).",
    )
    operator_id: Optional[str] = Field(
        default=None,
        description="EUDR operator identifier.",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata.",
    )


class BatchAssessmentRequest(BaseModel):
    """Request schema for batch supplier assessment."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_ids": ["SUP-001", "SUP-002", "SUP-003"],
                "custom_weights": None,
            }
        },
    )

    supplier_ids: List[str] = Field(
        ..., min_length=1, max_length=500,
        description="List of supplier IDs (max 500).",
    )
    custom_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom factor weights for all assessments.",
    )


class BatchAssessmentResponse(BaseModel):
    """Response schema for batch supplier assessment."""

    model_config = ConfigDict(str_strip_whitespace=True)

    assessments: List[SupplierRiskResponse] = Field(
        ...,
        description="List of supplier risk assessments.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of assessments.",
    )
    successful: int = Field(
        ..., ge=0,
        description="Number of successful assessments.",
    )
    failed: int = Field(
        ..., ge=0,
        description="Number of failed assessments.",
    )
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="List of errors for failed assessments.",
    )


class CompareSupplierRequest(BaseModel):
    """Request schema for supplier comparison."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_ids": ["SUP-001", "SUP-002"],
                "factors": ["geographic_sourcing", "compliance_history"],
            }
        },
    )

    supplier_ids: List[str] = Field(
        ..., min_length=2, max_length=10,
        description="List of supplier IDs to compare (2-10).",
    )
    factors: Optional[List[str]] = Field(
        default=None,
        description="Risk factors to compare (defaults to all 8).",
    )


class ComparisonResponse(BaseModel):
    """Response schema for supplier comparison."""

    model_config = ConfigDict(str_strip_whitespace=True)

    suppliers: List[SupplierRiskResponse] = Field(
        ...,
        description="Supplier risk assessments for comparison.",
    )
    factor_comparison: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Factor-level comparison matrix.",
    )
    rankings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Supplier rankings by risk score.",
    )
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Comparison statistics (mean, median, std_dev).",
    )
    comparison_date: Optional[datetime] = Field(
        default=None,
        description="Comparison timestamp (UTC).",
    )


class TrendPointSchema(BaseModel):
    """Single point in risk trend time series."""

    model_config = ConfigDict(str_strip_whitespace=True)

    timestamp: datetime = Field(
        ...,
        description="Timestamp of data point (UTC).",
    )
    risk_score: float = Field(
        ..., ge=0, le=100,
        description="Risk score at this point.",
    )
    risk_level: str = Field(
        ...,
        description="Risk level at this point.",
    )


class TrendResponse(BaseModel):
    """Response schema for supplier risk trend."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    supplier_name: str = Field(
        ...,
        description="Supplier name.",
    )
    trend_direction: str = Field(
        ...,
        description="Trend direction: increasing, stable, decreasing.",
    )
    data_points: List[TrendPointSchema] = Field(
        ...,
        description="Time series data points.",
    )
    statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Trend statistics (mean, median, std_dev, min, max).",
    )
    retrieved_at: Optional[datetime] = Field(
        default=None,
        description="Retrieval timestamp (UTC).",
    )


class RankingsResponse(BaseModel):
    """Response schema for supplier rankings."""

    model_config = ConfigDict(str_strip_whitespace=True)

    rankings: List[Dict[str, Any]] = Field(
        ...,
        description="Ranked list of suppliers with scores.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of suppliers.",
    )
    high_risk_count: int = Field(
        ..., ge=0,
        description="Number of high/critical risk suppliers.",
    )
    medium_risk_count: int = Field(
        ..., ge=0,
        description="Number of medium risk suppliers.",
    )
    low_risk_count: int = Field(
        ..., ge=0,
        description="Number of low risk suppliers.",
    )
    retrieved_at: Optional[datetime] = Field(
        default=None,
        description="Retrieval timestamp (UTC).",
    )


# =============================================================================
# Due Diligence Schemas
# =============================================================================


class DDRecordRequest(BaseModel):
    """Request schema for recording due diligence activity."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "activity_type": "site_visit",
                "description": "On-site audit of production facilities.",
                "findings": ["Minor documentation gap"],
                "auditor": "John Smith",
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    activity_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Due diligence activity type (e.g., site_visit, document_review).",
    )
    description: str = Field(
        ..., min_length=1, max_length=2000,
        description="Activity description.",
    )
    findings: List[str] = Field(
        default_factory=list,
        description="List of findings from the activity.",
    )
    non_conformances: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Non-conformances detected (type, severity, description).",
    )
    auditor: str = Field(
        ..., min_length=1, max_length=100,
        description="Auditor identifier or name.",
    )


class DDHistoryResponse(BaseModel):
    """Response schema for due diligence history."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    dd_level: str = Field(
        ...,
        description="Due diligence level: simplified, standard, enhanced.",
    )
    status: str = Field(
        ...,
        description="DD status: not_started, in_progress, completed, overdue.",
    )
    activities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of DD activities with timestamps.",
    )
    non_conformances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of non-conformances detected.",
    )
    corrective_actions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of corrective actions taken.",
    )
    last_review_date: Optional[datetime] = Field(
        default=None,
        description="Last review date (UTC).",
    )
    next_review_date: Optional[datetime] = Field(
        default=None,
        description="Next scheduled review date (UTC).",
    )


class DDGapsResponse(BaseModel):
    """Response schema for due diligence gaps."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    dd_level: str = Field(
        ...,
        description="Required due diligence level.",
    )
    missing_activities: List[str] = Field(
        default_factory=list,
        description="List of missing DD activities.",
    )
    overdue_activities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of overdue activities with deadlines.",
    )
    open_non_conformances: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of open non-conformances.",
    )
    gap_score: float = Field(
        ..., ge=0, le=100,
        description="Due diligence gap score (0=no gaps, 100=major gaps).",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations to close gaps.",
    )


class EscalateIssueRequest(BaseModel):
    """Request schema for escalating due diligence issue."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "issue_type": "critical_non_conformance",
                "description": "Deforestation detected in sourcing region.",
                "severity": "critical",
                "escalate_to": "compliance_manager",
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    issue_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Issue type (e.g., critical_non_conformance, document_fraud).",
    )
    description: str = Field(
        ..., min_length=1, max_length=2000,
        description="Issue description.",
    )
    severity: str = Field(
        ...,
        description="Issue severity: minor, major, critical.",
    )
    escalate_to: str = Field(
        ..., min_length=1, max_length=100,
        description="Escalation target (role or user).",
    )
    supporting_evidence: Optional[List[str]] = Field(
        default=None,
        description="List of supporting evidence (document IDs, URLs).",
    )


# =============================================================================
# Documentation Schemas
# =============================================================================


class AnalyzeDocumentRequest(BaseModel):
    """Request schema for document analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "document_type": "geolocation",
                "document_url": "https://storage.example.com/geo-doc.pdf",
                "validate_eudr": True,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    document_type: str = Field(
        ..., min_length=1, max_length=100,
        description="Document type (e.g., geolocation, certificate).",
    )
    document_url: Optional[str] = Field(
        default=None, max_length=500,
        description="Document URL (for remote documents).",
    )
    document_data: Optional[str] = Field(
        default=None,
        description="Base64-encoded document data (for inline upload).",
    )
    validate_eudr: bool = Field(
        default=True,
        description="Validate against EUDR requirements.",
    )


class DocumentProfileResponse(BaseModel):
    """Response schema for supplier documentation profile."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of documents with status and metadata.",
    )
    completeness_score: float = Field(
        ..., ge=0, le=1,
        description="Documentation completeness score (0.0-1.0).",
    )
    quality_score: float = Field(
        ..., ge=0, le=100,
        description="Documentation quality score (0-100).",
    )
    gaps: List[str] = Field(
        default_factory=list,
        description="List of missing or incomplete documents.",
    )
    expiring_soon: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Documents expiring within warning period.",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC).",
    )


class DocumentGapsResponse(BaseModel):
    """Response schema for documentation gaps."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    missing_documents: List[str] = Field(
        default_factory=list,
        description="List of missing required documents.",
    )
    expired_documents: List[str] = Field(
        default_factory=list,
        description="List of expired documents.",
    )
    pending_validation: List[str] = Field(
        default_factory=list,
        description="Documents pending validation.",
    )
    gap_details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Detailed gap information.",
    )
    priority_actions: List[str] = Field(
        default_factory=list,
        description="Priority actions to close gaps.",
    )


class RequestDocumentRequest(BaseModel):
    """Request schema for requesting documents from supplier."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "document_types": ["geolocation", "harvest_date"],
                "due_date": "2026-04-01",
                "message": "Please provide missing documentation.",
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    document_types: List[str] = Field(
        ..., min_length=1,
        description="List of document types to request.",
    )
    due_date: str = Field(
        ...,
        description="Due date for document submission (YYYY-MM-DD).",
    )
    message: Optional[str] = Field(
        default=None, max_length=2000,
        description="Message to supplier.",
    )


# =============================================================================
# Certification Schemas
# =============================================================================


class ValidateCertificationRequest(BaseModel):
    """Request schema for certification validation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "scheme": "FSC",
                "certificate_number": "FSC-C123456",
                "verify_chain_of_custody": True,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    scheme: str = Field(
        ..., min_length=1, max_length=100,
        description="Certification scheme (e.g., FSC, RSPO, PEFC).",
    )
    certificate_number: str = Field(
        ..., min_length=1, max_length=100,
        description="Certificate number.",
    )
    verify_chain_of_custody: bool = Field(
        default=False,
        description="Verify chain-of-custody status.",
    )


class CertStatusResponse(BaseModel):
    """Response schema for certification status."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    certifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of certifications with status.",
    )
    valid_count: int = Field(
        ..., ge=0,
        description="Number of valid certifications.",
    )
    expired_count: int = Field(
        ..., ge=0,
        description="Number of expired certifications.",
    )
    expiring_soon_count: int = Field(
        ..., ge=0,
        description="Certifications expiring within warning period.",
    )
    coverage_score: float = Field(
        ..., ge=0, le=1,
        description="Certification coverage score (0.0-1.0).",
    )
    last_updated: Optional[datetime] = Field(
        default=None,
        description="Last update timestamp (UTC).",
    )


class CertExpiryResponse(BaseModel):
    """Response schema for certification expiry check."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    expiring_soon: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Certifications expiring within threshold.",
    )
    expired: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Expired certifications.",
    )
    days_threshold: int = Field(
        ..., ge=0,
        description="Days threshold for expiry warning.",
    )
    checked_at: Optional[datetime] = Field(
        default=None,
        description="Check timestamp (UTC).",
    )


class VerifyScopeRequest(BaseModel):
    """Request schema for certification scope verification."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "certificate_number": "FSC-C123456",
                "commodity": "wood",
                "product_type": "lumber",
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    certificate_number: str = Field(
        ..., min_length=1, max_length=100,
        description="Certificate number.",
    )
    commodity: str = Field(
        ..., min_length=1, max_length=100,
        description="Commodity to verify.",
    )
    product_type: Optional[str] = Field(
        default=None, max_length=100,
        description="Specific product type.",
    )


class SchemesListResponse(BaseModel):
    """Response schema for supported certification schemes."""

    model_config = ConfigDict(str_strip_whitespace=True)

    schemes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of supported schemes with metadata.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of schemes.",
    )


# =============================================================================
# Geographic Schemas
# =============================================================================


class AnalyzeSourcingRequest(BaseModel):
    """Request schema for geographic sourcing analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "commodity": "coffee",
                "include_deforestation_overlay": True,
                "include_country_risk": True,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    commodity: Optional[str] = Field(
        default=None, max_length=100,
        description="Commodity filter (defaults to all).",
    )
    include_deforestation_overlay: bool = Field(
        default=False,
        description="Include deforestation overlay analysis.",
    )
    include_country_risk: bool = Field(
        default=False,
        description="Include country risk scores (AGENT-EUDR-016).",
    )


class SourcingProfileResponse(BaseModel):
    """Response schema for geographic sourcing profile."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    sourcing_countries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sourcing countries with volumes and risk scores.",
    )
    concentration_risk: float = Field(
        ..., ge=0, le=100,
        description="Geographic concentration risk score (0-100).",
    )
    high_risk_exposure: float = Field(
        ..., ge=0, le=1,
        description="Proportion of sourcing from high-risk countries (0.0-1.0).",
    )
    deforestation_risk: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="Deforestation risk score (0-100).",
    )
    analyzed_at: Optional[datetime] = Field(
        default=None,
        description="Analysis timestamp (UTC).",
    )


class RiskZonesResponse(BaseModel):
    """Response schema for risk zones analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    risk_zones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of risk zones with coordinates and risk levels.",
    )
    high_risk_count: int = Field(
        ..., ge=0,
        description="Number of high-risk zones.",
    )
    critical_risk_count: int = Field(
        ..., ge=0,
        description="Number of critical-risk zones.",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for risk mitigation.",
    )


class ConcentrationRequest(BaseModel):
    """Request schema for geographic concentration analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "threshold": 0.5,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    threshold: float = Field(
        default=0.5, ge=0, le=1,
        description="Concentration risk threshold (0.0-1.0).",
    )


class SourcingChangesResponse(BaseModel):
    """Response schema for sourcing changes detection."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    changes_detected: bool = Field(
        ...,
        description="Whether significant changes were detected.",
    )
    new_countries: List[str] = Field(
        default_factory=list,
        description="Newly added sourcing countries.",
    )
    removed_countries: List[str] = Field(
        default_factory=list,
        description="Removed sourcing countries.",
    )
    volume_changes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Significant volume changes by country.",
    )
    risk_impact: Optional[float] = Field(
        default=None,
        description="Impact on overall risk score.",
    )


# =============================================================================
# Network Schemas
# =============================================================================


class AnalyzeNetworkRequest(BaseModel):
    """Request schema for supplier network analysis."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "depth": 2,
                "include_sub_suppliers": True,
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    depth: int = Field(
        default=1, ge=1, le=5,
        description="Network depth to analyze (1-5 tiers).",
    )
    include_sub_suppliers: bool = Field(
        default=True,
        description="Include sub-supplier details.",
    )


class NetworkResponse(BaseModel):
    """Response schema for supplier network."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Primary supplier identifier.",
    )
    network_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Network nodes (suppliers) with risk scores.",
    )
    network_edges: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Network edges (relationships) with volumes.",
    )
    network_risk_score: float = Field(
        ..., ge=0, le=100,
        description="Overall network risk score (0-100).",
    )
    high_risk_nodes_count: int = Field(
        ..., ge=0,
        description="Number of high-risk nodes in network.",
    )
    analyzed_at: Optional[datetime] = Field(
        default=None,
        description="Analysis timestamp (UTC).",
    )


class SubSuppliersResponse(BaseModel):
    """Response schema for sub-suppliers list."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Primary supplier identifier.",
    )
    sub_suppliers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of sub-suppliers with risk scores.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of sub-suppliers.",
    )
    high_risk_count: int = Field(
        ..., ge=0,
        description="Number of high-risk sub-suppliers.",
    )


class RiskPropagationRequest(BaseModel):
    """Request schema for network risk propagation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "source_supplier_id": "SUP-12345",
                "risk_increase": 25.0,
                "propagation_depth": 2,
            }
        },
    )

    source_supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Source supplier ID for risk propagation.",
    )
    risk_increase: float = Field(
        ..., ge=0, le=100,
        description="Risk score increase at source (0-100).",
    )
    propagation_depth: int = Field(
        default=1, ge=1, le=5,
        description="Propagation depth (1-5 tiers).",
    )


class NetworkGraphResponse(BaseModel):
    """Response schema for network graph data."""

    model_config = ConfigDict(str_strip_whitespace=True)

    supplier_id: str = Field(
        ...,
        description="Primary supplier identifier.",
    )
    graph_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Graph data in D3.js-compatible format.",
    )
    statistics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Network statistics (node count, edge count, density).",
    )


# =============================================================================
# Monitoring Schemas
# =============================================================================


class ConfigureMonitoringRequest(BaseModel):
    """Request schema for configuring supplier monitoring."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "frequency": "weekly",
                "alert_thresholds": {"risk_score": 70, "cert_expiry_days": 30},
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    frequency: str = Field(
        ...,
        description="Monitoring frequency: daily, weekly, biweekly, monthly, quarterly.",
    )
    alert_thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Alert thresholds for various metrics.",
    )
    alert_channels: Optional[List[str]] = Field(
        default=None,
        description="Alert notification channels (email, slack, webhook).",
    )


class AlertResponse(BaseModel):
    """Response schema for supplier alerts."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alert_id: str = Field(
        ...,
        description="Unique alert identifier.",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    alert_type: str = Field(
        ...,
        description="Alert type (e.g., risk_threshold, cert_expiry).",
    )
    severity: str = Field(
        ...,
        description="Alert severity: info, warning, high, critical.",
    )
    message: str = Field(
        ...,
        description="Alert message.",
    )
    created_at: Optional[datetime] = Field(
        default=None,
        description="Alert creation timestamp (UTC).",
    )
    acknowledged: bool = Field(
        default=False,
        description="Whether alert has been acknowledged.",
    )


class AlertListResponse(BaseModel):
    """Response schema for alert list."""

    model_config = ConfigDict(str_strip_whitespace=True)

    alerts: List[AlertResponse] = Field(
        default_factory=list,
        description="List of alerts.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of alerts.",
    )
    critical_count: int = Field(
        ..., ge=0,
        description="Number of critical alerts.",
    )
    unacknowledged_count: int = Field(
        ..., ge=0,
        description="Number of unacknowledged alerts.",
    )


class WatchlistResponse(BaseModel):
    """Response schema for monitoring watchlist."""

    model_config = ConfigDict(str_strip_whitespace=True)

    watchlist_suppliers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Suppliers on watchlist with monitoring config.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total suppliers on watchlist.",
    )


class AddToWatchlistRequest(BaseModel):
    """Request schema for adding supplier to watchlist."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "reason": "Recent risk score increase",
                "monitoring_frequency": "daily",
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    reason: str = Field(
        ..., min_length=1, max_length=500,
        description="Reason for adding to watchlist.",
    )
    monitoring_frequency: str = Field(
        default="weekly",
        description="Monitoring frequency: daily, weekly, biweekly, monthly.",
    )


class PortfolioRiskResponse(BaseModel):
    """Response schema for portfolio risk analysis."""

    model_config = ConfigDict(str_strip_whitespace=True)

    total_suppliers: int = Field(
        ..., ge=0,
        description="Total number of suppliers in portfolio.",
    )
    average_risk_score: float = Field(
        ..., ge=0, le=100,
        description="Average risk score across portfolio.",
    )
    risk_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Risk level distribution (low, medium, high, critical).",
    )
    top_risks: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top risk suppliers.",
    )
    trend: str = Field(
        ...,
        description="Portfolio risk trend: improving, stable, deteriorating.",
    )
    analyzed_at: Optional[datetime] = Field(
        default=None,
        description="Analysis timestamp (UTC).",
    )


# =============================================================================
# Report Schemas
# =============================================================================


class GenerateReportRequest(BaseModel):
    """Request schema for generating supplier risk report."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_id": "SUP-12345",
                "report_type": "individual",
                "format": "pdf",
                "include_sections": ["risk_assessment", "due_diligence", "certifications"],
            }
        },
    )

    supplier_id: str = Field(
        ..., min_length=1, max_length=100,
        description="Supplier identifier.",
    )
    report_type: str = Field(
        default="individual",
        description="Report type: individual, portfolio, comparative, trend, audit_package, executive.",
    )
    format: str = Field(
        default="pdf",
        description="Output format: pdf, json, html, excel, csv.",
    )
    include_sections: Optional[List[str]] = Field(
        default=None,
        description="Sections to include (defaults to all).",
    )
    language: str = Field(
        default="en",
        description="Report language (ISO 639-1 code).",
    )


class BatchReportRequest(BaseModel):
    """Request schema for batch report generation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "supplier_ids": ["SUP-001", "SUP-002"],
                "report_type": "individual",
                "format": "pdf",
            }
        },
    )

    supplier_ids: List[str] = Field(
        ..., min_length=1, max_length=100,
        description="List of supplier IDs (max 100).",
    )
    report_type: str = Field(
        default="individual",
        description="Report type for all suppliers.",
    )
    format: str = Field(
        default="pdf",
        description="Output format: pdf, json, html, excel, csv.",
    )


class ReportResponse(BaseModel):
    """Response schema for generated report."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(
        ...,
        description="Unique report identifier.",
    )
    supplier_id: str = Field(
        ...,
        description="Supplier identifier.",
    )
    report_type: str = Field(
        ...,
        description="Report type.",
    )
    format: str = Field(
        ...,
        description="Report format.",
    )
    status: str = Field(
        ...,
        description="Report status: pending, completed, failed.",
    )
    download_url: Optional[str] = Field(
        default=None,
        description="Download URL (if completed).",
    )
    file_size_bytes: Optional[int] = Field(
        default=None, ge=0,
        description="File size in bytes.",
    )
    generated_at: Optional[datetime] = Field(
        default=None,
        description="Generation timestamp (UTC).",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="Download URL expiry timestamp (UTC).",
    )


class ReportListResponse(BaseModel):
    """Response schema for report list."""

    model_config = ConfigDict(str_strip_whitespace=True)

    reports: List[ReportResponse] = Field(
        default_factory=list,
        description="List of reports.",
    )
    total: int = Field(
        ..., ge=0,
        description="Total number of reports.",
    )
    pagination: PaginationSchema = Field(
        default_factory=PaginationSchema,
        description="Pagination metadata.",
    )


class DownloadReportResponse(BaseModel):
    """Response schema for report download."""

    model_config = ConfigDict(str_strip_whitespace=True)

    report_id: str = Field(
        ...,
        description="Report identifier.",
    )
    download_url: str = Field(
        ...,
        description="Signed download URL.",
    )
    file_name: str = Field(
        ...,
        description="Suggested file name.",
    )
    content_type: str = Field(
        ...,
        description="MIME content type.",
    )
    file_size_bytes: int = Field(
        ..., ge=0,
        description="File size in bytes.",
    )
    expires_at: Optional[datetime] = Field(
        default=None,
        description="URL expiry timestamp (UTC).",
    )


class PortfolioReportRequest(BaseModel):
    """Request schema for portfolio report generation."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "portfolio_name": "Coffee Suppliers 2026",
                "supplier_ids": ["SUP-001", "SUP-002", "SUP-003"],
                "format": "pdf",
            }
        },
    )

    portfolio_name: str = Field(
        ..., min_length=1, max_length=255,
        description="Portfolio name.",
    )
    supplier_ids: Optional[List[str]] = Field(
        default=None,
        description="Supplier IDs (defaults to all).",
    )
    format: str = Field(
        default="pdf",
        description="Output format: pdf, excel.",
    )
    include_executive_summary: bool = Field(
        default=True,
        description="Include executive summary.",
    )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Common
    "PaginationSchema",
    "ErrorSchema",
    "HealthSchema",
    "SuccessSchema",
    # Supplier
    "FactorScoreSchema",
    "AssessSupplierRequest",
    "SupplierRiskResponse",
    "BatchAssessmentRequest",
    "BatchAssessmentResponse",
    "CompareSupplierRequest",
    "ComparisonResponse",
    "TrendPointSchema",
    "TrendResponse",
    "RankingsResponse",
    # Due Diligence
    "DDRecordRequest",
    "DDHistoryResponse",
    "DDGapsResponse",
    "EscalateIssueRequest",
    # Documentation
    "AnalyzeDocumentRequest",
    "DocumentProfileResponse",
    "DocumentGapsResponse",
    "RequestDocumentRequest",
    # Certification
    "ValidateCertificationRequest",
    "CertStatusResponse",
    "CertExpiryResponse",
    "VerifyScopeRequest",
    "SchemesListResponse",
    # Geographic
    "AnalyzeSourcingRequest",
    "SourcingProfileResponse",
    "RiskZonesResponse",
    "ConcentrationRequest",
    "SourcingChangesResponse",
    # Network
    "AnalyzeNetworkRequest",
    "NetworkResponse",
    "SubSuppliersResponse",
    "RiskPropagationRequest",
    "NetworkGraphResponse",
    # Monitoring
    "ConfigureMonitoringRequest",
    "AlertResponse",
    "AlertListResponse",
    "WatchlistResponse",
    "AddToWatchlistRequest",
    "PortfolioRiskResponse",
    # Report
    "GenerateReportRequest",
    "BatchReportRequest",
    "ReportResponse",
    "ReportListResponse",
    "DownloadReportResponse",
    "PortfolioReportRequest",
]

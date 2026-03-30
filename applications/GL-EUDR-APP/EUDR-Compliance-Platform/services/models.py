# -*- coding: utf-8 -*-
"""
GL-EUDR-APP Domain Models - EU Deforestation Regulation Compliance Platform

All Pydantic domain models for the EUDR compliance platform including:
- Core entities: Supplier, Plot, Procurement, Document, DueDiligenceStatement
- Pipeline models: PipelineRun, StageResult
- Risk models: RiskAssessment, RiskAlert, RiskTrendPoint
- Dashboard models: DashboardMetrics
- Request/Response models for all API operations

All IDs use UUID v4. Timestamps use timezone-aware UTC datetime.
Models use Pydantic v2 with model_config for serialization control.

Example:
    >>> from services.models import Supplier, Plot
    >>> supplier = Supplier(name="AgroTrade Ltd", country="BRA")
    >>> plot = Plot(supplier_id=supplier.id, name="Farm Block A", ...)

Author: GreenLang Platform Team
Date: March 2026
Application: GL-EUDR-APP v1.0
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator
from greenlang.schemas import GreenLangBase, utcnow, new_uuid

from services.config import (
    ComplianceStatus,
    DDSStatus,
    DocumentType,
    EUDRCommodity,
    PipelineStage,
    PipelineStatus,
    ProcurementStatus,
    RiskLevel,
    SatelliteAssessmentStatus,
    VerificationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_id() -> str:
    """Generate a new UUID v4 string."""
    return str(uuid.uuid4())


# ===========================================================================
# Core Entity Models
# ===========================================================================


class Supplier(GreenLangBase):
    """Supplier entity for EUDR compliance tracking.

    Represents a supplier or operator placing relevant commodities on the
    EU market. Tracks compliance status, risk level, associated plots,
    and commodity types.

    Attributes:
        id: Unique supplier identifier (UUID v4).
        name: Legal name of the supplier or operator.
        country: ISO-3166 alpha-3 country code of registration.
        tax_id: Tax identification number or VAT number.
        address: Registered business address.
        contact_email: Primary contact email.
        contact_phone: Primary contact phone number.
        commodities: List of EUDR commodities this supplier deals in.
        risk_level: Current risk classification.
        compliance_status: Current compliance status.
        plots: List of associated plot IDs.
        certifications: List of certifications held.
        last_audit_date: Date of last compliance audit.
        notes: Free-text notes.
        created_at: Record creation timestamp (UTC).
        updated_at: Record last-update timestamp (UTC).
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 identifier")
    name: str = Field(..., min_length=1, max_length=500, description="Legal name")
    country: str = Field(
        ..., min_length=3, max_length=3, description="ISO-3166 alpha-3 country code"
    )
    tax_id: Optional[str] = Field(None, max_length=50, description="Tax/VAT ID")
    address: Optional[str] = Field(None, max_length=1000, description="Business address")
    contact_email: Optional[str] = Field(None, max_length=254, description="Contact email")
    contact_phone: Optional[str] = Field(None, max_length=30, description="Contact phone")
    commodities: List[EUDRCommodity] = Field(
        default_factory=list, description="EUDR commodities"
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD, description="Risk classification"
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.PENDING, description="Compliance status"
    )
    plots: List[str] = Field(
        default_factory=list, description="Associated plot IDs"
    )
    certifications: List[str] = Field(
        default_factory=list, description="Certifications held"
    )
    last_audit_date: Optional[date] = Field(
        None, description="Last compliance audit date"
    )
    notes: Optional[str] = Field(None, max_length=5000, description="Free-text notes")
    created_at: datetime = Field(default_factory=_utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Updated timestamp")

    @field_validator("country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Ensure country code is uppercase 3-letter ISO."""
        return v.upper()

    model_config = {"arbitrary_types_allowed": True}


class GeoJSONPolygon(GreenLangBase):
    """GeoJSON Polygon geometry for plot boundaries.

    Coordinates follow the GeoJSON specification: an array of linear ring
    coordinate arrays. Each coordinate is [longitude, latitude].

    Attributes:
        type: Must be "Polygon".
        coordinates: Nested array of [lon, lat] coordinate pairs.
    """

    type: str = Field(default="Polygon", description="GeoJSON geometry type")
    coordinates: List[List[List[float]]] = Field(
        ..., description="Polygon coordinates [[[lon, lat], ...]]"
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Ensure type is Polygon."""
        if v != "Polygon":
            raise ValueError("GeoJSON type must be 'Polygon'")
        return v

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: List[List[List[float]]]) -> List[List[List[float]]]:
        """Validate polygon has at least one ring with 4+ points."""
        if not v or len(v) < 1:
            raise ValueError("Polygon must have at least one linear ring")
        for ring in v:
            if len(ring) < 4:
                raise ValueError(
                    "Each linear ring must have at least 4 coordinate pairs "
                    "(first and last must match)"
                )
            # Verify ring closure
            if ring[0] != ring[-1]:
                raise ValueError("Linear ring must be closed (first == last)")
        return v


class Plot(GreenLangBase):
    """Production plot for EUDR geolocation tracking.

    Represents a geographic area of land used for commodity production.
    Per EUDR Article 9, geolocation data must be provided for all
    plots of land where the commodity was produced.

    Attributes:
        id: Unique plot identifier (UUID v4).
        supplier_id: Owning supplier's ID.
        name: Human-readable plot name.
        coordinates: GeoJSON Polygon boundary.
        centroid_lat: Centroid latitude (derived).
        centroid_lon: Centroid longitude (derived).
        area_hectares: Plot area in hectares.
        commodity: Primary commodity produced.
        country_iso3: ISO-3166 alpha-3 country code.
        region: Sub-national region or state.
        risk_level: Current risk classification.
        satellite_status: Satellite assessment status.
        last_assessed: Last satellite assessment date.
        ndvi_baseline: Baseline NDVI value at cutoff date.
        ndvi_current: Most recent NDVI value.
        forest_cover_pct: Forest cover percentage at baseline.
        is_deforestation_free: Whether plot passes EUDR Art 3 check.
        legal_compliance: Whether plot has legal production evidence.
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 identifier")
    supplier_id: str = Field(..., description="Owning supplier ID")
    name: str = Field(..., min_length=1, max_length=500, description="Plot name")
    coordinates: Optional[GeoJSONPolygon] = Field(
        None, description="GeoJSON Polygon boundary"
    )
    centroid_lat: Optional[float] = Field(
        None, ge=-90.0, le=90.0, description="Centroid latitude"
    )
    centroid_lon: Optional[float] = Field(
        None, ge=-180.0, le=180.0, description="Centroid longitude"
    )
    area_hectares: Optional[float] = Field(
        None, ge=0.0, description="Area in hectares"
    )
    commodity: EUDRCommodity = Field(..., description="Primary commodity")
    country_iso3: str = Field(
        ..., min_length=3, max_length=3, description="ISO-3166 alpha-3 country code"
    )
    region: Optional[str] = Field(None, max_length=200, description="Sub-national region")
    risk_level: RiskLevel = Field(
        default=RiskLevel.STANDARD, description="Risk classification"
    )
    satellite_status: SatelliteAssessmentStatus = Field(
        default=SatelliteAssessmentStatus.NOT_ASSESSED,
        description="Satellite assessment status",
    )
    last_assessed: Optional[datetime] = Field(
        None, description="Last satellite assessment timestamp"
    )
    ndvi_baseline: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Baseline NDVI at cutoff"
    )
    ndvi_current: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Current NDVI"
    )
    forest_cover_pct: Optional[float] = Field(
        None, ge=0.0, le=100.0, description="Forest cover percentage at baseline"
    )
    is_deforestation_free: Optional[bool] = Field(
        None, description="Passes EUDR Art 3 deforestation-free check"
    )
    legal_compliance: Optional[bool] = Field(
        None, description="Has evidence of legal production"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Updated timestamp")

    @field_validator("country_iso3")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Uppercase country code."""
        return v.upper()

    model_config = {"arbitrary_types_allowed": True}


class Procurement(GreenLangBase):
    """Procurement record tracking commodity purchases from suppliers.

    Links a purchase order to its origin plots for EUDR traceability.

    Attributes:
        id: Unique procurement identifier.
        supplier_id: Supplier this procurement is from.
        po_number: Purchase order number.
        commodity: Commodity type.
        quantity: Quantity purchased.
        unit: Unit of measure (kg, tonnes, m3).
        harvest_date: Date commodity was harvested.
        shipment_date: Date commodity was shipped.
        arrival_date: Expected or actual arrival date.
        origin_plot_ids: IDs of plots commodity originates from.
        status: Procurement status.
        dds_id: Linked Due Diligence Statement ID (if any).
        notes: Additional notes.
        created_at: Record creation timestamp.
        updated_at: Record last-update timestamp.
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 identifier")
    supplier_id: str = Field(..., description="Supplier ID")
    po_number: str = Field(..., min_length=1, max_length=100, description="PO number")
    commodity: EUDRCommodity = Field(..., description="Commodity type")
    quantity: float = Field(..., gt=0, description="Quantity purchased")
    unit: str = Field(
        default="tonnes", description="Unit of measure (kg, tonnes, m3, heads)"
    )
    harvest_date: Optional[date] = Field(None, description="Harvest date")
    shipment_date: Optional[date] = Field(None, description="Shipment date")
    arrival_date: Optional[date] = Field(None, description="Arrival date")
    origin_plot_ids: List[str] = Field(
        default_factory=list, description="Origin plot IDs"
    )
    status: ProcurementStatus = Field(
        default=ProcurementStatus.DRAFT, description="Procurement status"
    )
    dds_id: Optional[str] = Field(None, description="Linked DDS ID")
    notes: Optional[str] = Field(None, max_length=5000, description="Notes")
    created_at: datetime = Field(default_factory=_utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Updated timestamp")

    model_config = {"arbitrary_types_allowed": True}


class Document(GreenLangBase):
    """Compliance document for EUDR verification.

    Represents an uploaded document (certificate, permit, land title,
    invoice, transport document) that supports EUDR due diligence.

    Attributes:
        id: Unique document identifier.
        name: Document filename.
        doc_type: Document type classification.
        file_path: Storage path.
        file_size_bytes: File size in bytes.
        mime_type: MIME type.
        verification_status: Verification outcome.
        verification_score: Confidence score (0.0 to 1.0).
        verified_at: Verification timestamp.
        linked_supplier_id: Linked supplier ID.
        linked_plot_id: Linked plot ID.
        linked_dds_id: Linked DDS ID.
        linked_procurement_id: Linked procurement ID.
        ocr_text: Extracted text from OCR.
        compliance_findings: List of compliance check findings.
        expiry_date: Document expiry date (for certificates, permits).
        issuer: Document issuer name.
        metadata: Additional key-value metadata.
        created_at: Upload timestamp.
        updated_at: Last-update timestamp.
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 identifier")
    name: str = Field(..., min_length=1, max_length=500, description="Filename")
    doc_type: DocumentType = Field(..., description="Document type")
    file_path: Optional[str] = Field(None, max_length=1000, description="Storage path")
    file_size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(None, max_length=100, description="MIME type")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.PENDING, description="Verification status"
    )
    verification_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Verification confidence score"
    )
    verified_at: Optional[datetime] = Field(None, description="Verification timestamp")
    linked_supplier_id: Optional[str] = Field(None, description="Linked supplier ID")
    linked_plot_id: Optional[str] = Field(None, description="Linked plot ID")
    linked_dds_id: Optional[str] = Field(None, description="Linked DDS ID")
    linked_procurement_id: Optional[str] = Field(
        None, description="Linked procurement ID"
    )
    ocr_text: Optional[str] = Field(None, description="OCR-extracted text")
    compliance_findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Compliance check findings"
    )
    expiry_date: Optional[date] = Field(None, description="Document expiry date")
    issuer: Optional[str] = Field(None, max_length=500, description="Document issuer")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Upload timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Updated timestamp")

    model_config = {"arbitrary_types_allowed": True}


# ===========================================================================
# DDS Models
# ===========================================================================


class DDSRiskAssessment(GreenLangBase):
    """Risk assessment section within a Due Diligence Statement.

    Captures the four-dimension risk analysis required by EUDR Article 10.

    Attributes:
        country_risk: Country-level deforestation risk (0-1).
        commodity_risk: Commodity-level risk (0-1).
        supplier_risk: Supplier history risk (0-1).
        satellite_risk: Satellite-based deforestation risk (0-1).
        overall_risk: Weighted overall risk score (0-1).
        risk_level: Classified risk level.
        factors: Contributing risk factors.
        data_sources: Data sources used for assessment.
        assessment_date: Date of assessment.
    """

    country_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    commodity_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    satellite_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    factors: List[str] = Field(default_factory=list, description="Risk factors")
    data_sources: List[str] = Field(default_factory=list, description="Data sources")
    assessment_date: Optional[datetime] = Field(None, description="Assessment date")


class DDSMitigationMeasure(GreenLangBase):
    """Risk mitigation measure within a DDS per EUDR Article 11.

    Attributes:
        id: Measure identifier.
        risk_factor: Risk factor being mitigated.
        measure: Description of the mitigation action.
        status: Implementation status.
        responsible_party: Party responsible for implementation.
        due_date: Target completion date.
        evidence: Supporting evidence references.
    """

    id: str = Field(default_factory=_new_id)
    risk_factor: str = Field(..., description="Risk factor being mitigated")
    measure: str = Field(..., description="Mitigation action description")
    status: str = Field(default="planned", description="planned|in_progress|completed")
    responsible_party: Optional[str] = Field(None, description="Responsible party")
    due_date: Optional[date] = Field(None, description="Target completion date")
    evidence: List[str] = Field(default_factory=list, description="Evidence references")


class DueDiligenceStatement(GreenLangBase):
    """Due Diligence Statement per EU Regulation 2023/1115 Articles 4, 9-12.

    The DDS is the core compliance artifact operators must submit to the
    EU Information System before placing relevant commodities on the market.

    Attributes:
        id: Unique DDS identifier (UUID v4).
        reference_number: Official reference (EUDR-{ISO3}-{YEAR}-{SEQ}).
        supplier_id: Supplier this DDS covers.
        operator_name: Name of the operator (may differ from supplier).
        operator_country: Operator country of registration.
        year: Reporting year.
        commodity: Commodity covered.
        product_description: Description of products covered.
        country_of_production: Production country ISO-3 code.
        plots: List of plot IDs included.
        plot_details: Detailed plot information for the DDS.
        procurement_ids: Linked procurement record IDs.
        status: DDS lifecycle status.
        risk_assessment: Risk assessment section.
        mitigation_measures: Risk mitigation measures.
        documents: Supporting document IDs.
        conclusion: DDS conclusion statement.
        submission_date: Date submitted to EU system.
        eu_response: Response from EU system.
        eu_submission_id: EU system submission tracking ID.
        rejection_reason: Reason for rejection (if applicable).
        amendment_history: History of amendments.
        created_at: Creation timestamp.
        updated_at: Last-update timestamp.
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 identifier")
    reference_number: str = Field(
        ..., description="Official reference EUDR-{ISO3}-{YEAR}-{SEQ}"
    )
    supplier_id: str = Field(..., description="Supplier ID")
    operator_name: Optional[str] = Field(None, max_length=500, description="Operator name")
    operator_country: Optional[str] = Field(
        None, max_length=3, description="Operator country ISO-3"
    )
    year: int = Field(..., ge=2024, le=2100, description="Reporting year")
    commodity: EUDRCommodity = Field(..., description="Commodity covered")
    product_description: Optional[str] = Field(
        None, max_length=2000, description="Product description"
    )
    country_of_production: Optional[str] = Field(
        None, max_length=3, description="Production country ISO-3"
    )
    plots: List[str] = Field(default_factory=list, description="Plot IDs")
    plot_details: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed plot information"
    )
    procurement_ids: List[str] = Field(
        default_factory=list, description="Procurement record IDs"
    )
    status: DDSStatus = Field(default=DDSStatus.DRAFT, description="DDS lifecycle status")
    risk_assessment: Optional[DDSRiskAssessment] = Field(
        None, description="Risk assessment section"
    )
    mitigation_measures: List[DDSMitigationMeasure] = Field(
        default_factory=list, description="Risk mitigation measures"
    )
    documents: List[str] = Field(
        default_factory=list, description="Supporting document IDs"
    )
    conclusion: Optional[str] = Field(
        None, max_length=5000, description="DDS conclusion statement"
    )
    submission_date: Optional[datetime] = Field(
        None, description="Submission date to EU system"
    )
    eu_response: Optional[Dict[str, Any]] = Field(
        None, description="EU system response"
    )
    eu_submission_id: Optional[str] = Field(
        None, description="EU system submission tracking ID"
    )
    rejection_reason: Optional[str] = Field(
        None, max_length=5000, description="Rejection reason"
    )
    amendment_history: List[Dict[str, Any]] = Field(
        default_factory=list, description="Amendment history"
    )
    supply_chain_section: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Supply chain mapping data from AGENT-EUDR-001 "
            "RegulatoryExporter, including node list, tier depth, "
            "traceability score, and gap counts per EUDR Article 4(2)."
        ),
    )
    supply_chain_graph_id: Optional[str] = Field(
        None,
        description="Supply chain graph identifier linking to AGENT-EUDR-001",
    )
    created_at: datetime = Field(default_factory=_utcnow, description="Created timestamp")
    updated_at: datetime = Field(default_factory=_utcnow, description="Updated timestamp")

    model_config = {"arbitrary_types_allowed": True}


# ===========================================================================
# Pipeline Models
# ===========================================================================


class StageResult(GreenLangBase):
    """Result of a single pipeline stage execution.

    Attributes:
        stage: Pipeline stage that was executed.
        status: Outcome status (pending, running, completed, failed, skipped).
        started_at: Stage start timestamp.
        completed_at: Stage completion timestamp.
        duration_ms: Stage duration in milliseconds.
        result_data: Stage-specific result payload.
        error: Error message if stage failed.
        retry_count: Number of retries attempted.
    """

    stage: PipelineStage = Field(..., description="Pipeline stage")
    status: str = Field(default="pending", description="Stage status")
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_ms: Optional[float] = Field(None, ge=0, description="Duration in ms")
    result_data: Dict[str, Any] = Field(
        default_factory=dict, description="Result payload"
    )
    error: Optional[str] = Field(None, description="Error message")
    retry_count: int = Field(default=0, ge=0, description="Retry count")


class PipelineRun(GreenLangBase):
    """A single execution of the 5-stage EUDR compliance pipeline.

    Attributes:
        id: Unique run identifier.
        supplier_id: Supplier being processed.
        commodity: Commodity being assessed.
        plot_ids: Plot IDs included in this run.
        status: Overall pipeline status.
        current_stage: Currently executing stage (or last completed).
        stages: Results for each pipeline stage.
        started_at: Pipeline start timestamp.
        completed_at: Pipeline completion timestamp.
        duration_ms: Total duration in milliseconds.
        error: Error message if pipeline failed.
        triggered_by: User or system that triggered the run.
        provenance_hash: SHA-256 hash of run inputs for audit trail.
    """

    id: str = Field(default_factory=_new_id, description="UUID v4 run identifier")
    supplier_id: str = Field(..., description="Supplier ID")
    commodity: Optional[EUDRCommodity] = Field(None, description="Commodity type")
    plot_ids: List[str] = Field(default_factory=list, description="Plot IDs")
    status: PipelineStatus = Field(
        default=PipelineStatus.PENDING, description="Pipeline status"
    )
    current_stage: Optional[PipelineStage] = Field(
        None, description="Current/last stage"
    )
    stages: Dict[str, StageResult] = Field(
        default_factory=dict, description="Stage results keyed by stage name"
    )
    started_at: Optional[datetime] = Field(None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    duration_ms: Optional[float] = Field(None, ge=0, description="Total duration in ms")
    error: Optional[str] = Field(None, description="Error message")
    triggered_by: Optional[str] = Field(
        default="system", description="Trigger source"
    )
    provenance_hash: Optional[str] = Field(
        None, description="SHA-256 provenance hash"
    )

    model_config = {"arbitrary_types_allowed": True}


# ===========================================================================
# Risk Models
# ===========================================================================


class RiskAssessment(GreenLangBase):
    """Unified risk assessment combining multiple risk dimensions.

    Attributes:
        id: Assessment identifier.
        supplier_id: Assessed supplier.
        plot_id: Assessed plot (if specific).
        plot_risk: Plot-level deforestation risk (0-1).
        country_risk: Country-level risk (0-1).
        supplier_risk: Supplier history risk (0-1).
        satellite_risk: Satellite-derived risk (0-1).
        document_risk: Document completeness risk (0-1).
        overall_risk: Weighted overall risk (0-1).
        risk_level: Classified risk level.
        factors: Contributing risk factors with descriptions.
        recommendations: Risk mitigation recommendations.
        data_sources: Data sources used.
        assessed_at: Assessment timestamp.
        valid_until: Assessment validity expiry.
        provenance_hash: SHA-256 hash for audit.
    """

    id: str = Field(default_factory=_new_id, description="Assessment ID")
    supplier_id: Optional[str] = Field(None, description="Supplier ID")
    plot_id: Optional[str] = Field(None, description="Plot ID")
    plot_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    country_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    satellite_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    document_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    factors: List[Dict[str, Any]] = Field(
        default_factory=list, description="Risk factors"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Mitigation recommendations"
    )
    data_sources: List[str] = Field(
        default_factory=list, description="Data sources used"
    )
    assessed_at: datetime = Field(default_factory=_utcnow, description="Assessment time")
    valid_until: Optional[datetime] = Field(None, description="Validity expiry")
    provenance_hash: Optional[str] = Field(None, description="SHA-256 provenance hash")

    model_config = {"arbitrary_types_allowed": True}


class RiskAlert(GreenLangBase):
    """Risk alert raised when a threshold is exceeded.

    Attributes:
        id: Alert identifier.
        alert_type: Type of alert (deforestation, country_risk, etc.).
        severity: Alert severity level.
        supplier_id: Related supplier.
        plot_id: Related plot.
        title: Alert title.
        description: Alert description.
        risk_score: Score that triggered the alert.
        threshold: Threshold that was exceeded.
        recommended_action: Recommended response.
        acknowledged: Whether alert has been acknowledged.
        acknowledged_by: Who acknowledged the alert.
        created_at: Alert creation timestamp.
    """

    id: str = Field(default_factory=_new_id)
    alert_type: str = Field(..., description="Alert type")
    severity: RiskLevel = Field(default=RiskLevel.HIGH)
    supplier_id: Optional[str] = Field(None)
    plot_id: Optional[str] = Field(None)
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    recommended_action: Optional[str] = Field(None)
    acknowledged: bool = Field(default=False)
    acknowledged_by: Optional[str] = Field(None)
    created_at: datetime = Field(default_factory=_utcnow)


class RiskTrendPoint(GreenLangBase):
    """Single data point in a risk trend time series.

    Attributes:
        date: Point-in-time date.
        overall_risk: Overall risk score at this point.
        satellite_risk: Satellite risk component.
        country_risk: Country risk component.
        supplier_risk: Supplier risk component.
        document_risk: Document risk component.
        ndvi_value: NDVI value (if available).
        event: Notable event at this point (if any).
    """

    date: datetime = Field(..., description="Timestamp")
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    satellite_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    country_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    document_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    ndvi_value: Optional[float] = Field(None, ge=-1.0, le=1.0)
    event: Optional[str] = Field(None, description="Notable event")


# ===========================================================================
# Dashboard Models
# ===========================================================================


class DashboardMetrics(GreenLangBase):
    """Aggregated metrics for the EUDR compliance dashboard.

    Attributes:
        total_suppliers: Total number of registered suppliers.
        compliant_count: Suppliers with COMPLIANT status.
        pending_count: Suppliers with PENDING status.
        non_compliant_count: Suppliers with NON_COMPLIANT status.
        under_review_count: Suppliers under review.
        total_plots: Total registered plots.
        assessed_plots: Plots with satellite assessment completed.
        high_risk_plots: Plots classified as HIGH or CRITICAL risk.
        deforestation_free_plots: Plots confirmed deforestation-free.
        dds_total: Total DDS records.
        dds_draft: DDS in DRAFT status.
        dds_submitted: DDS submitted to EU system.
        dds_accepted: DDS accepted by EU authority.
        dds_rejected: DDS rejected by EU authority.
        pipeline_active: Currently running pipeline executions.
        pipeline_completed_today: Pipelines completed today.
        compliance_rate: Overall compliance rate (0-100%).
        average_risk_score: Average risk score across all suppliers.
        commodities_breakdown: Count of suppliers by commodity.
        country_breakdown: Count of suppliers by country.
        last_updated: Metrics last-refresh timestamp.
    """

    total_suppliers: int = Field(default=0, ge=0)
    compliant_count: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    non_compliant_count: int = Field(default=0, ge=0)
    under_review_count: int = Field(default=0, ge=0)
    total_plots: int = Field(default=0, ge=0)
    assessed_plots: int = Field(default=0, ge=0)
    high_risk_plots: int = Field(default=0, ge=0)
    deforestation_free_plots: int = Field(default=0, ge=0)
    dds_total: int = Field(default=0, ge=0)
    dds_draft: int = Field(default=0, ge=0)
    dds_submitted: int = Field(default=0, ge=0)
    dds_accepted: int = Field(default=0, ge=0)
    dds_rejected: int = Field(default=0, ge=0)
    pipeline_active: int = Field(default=0, ge=0)
    pipeline_completed_today: int = Field(default=0, ge=0)
    compliance_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    average_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    commodities_breakdown: Dict[str, int] = Field(default_factory=dict)
    country_breakdown: Dict[str, int] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=_utcnow)


# ===========================================================================
# Request Models
# ===========================================================================


class SupplierCreateRequest(GreenLangBase):
    """Request to create a new supplier."""

    name: str = Field(..., min_length=1, max_length=500)
    country: str = Field(..., min_length=3, max_length=3)
    tax_id: Optional[str] = Field(None, max_length=50)
    address: Optional[str] = Field(None, max_length=1000)
    contact_email: Optional[str] = Field(None, max_length=254)
    contact_phone: Optional[str] = Field(None, max_length=30)
    commodities: List[EUDRCommodity] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=5000)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Uppercase country code."""
        return v.upper()


class SupplierUpdateRequest(GreenLangBase):
    """Request to update an existing supplier."""

    name: Optional[str] = Field(None, min_length=1, max_length=500)
    country: Optional[str] = Field(None, min_length=3, max_length=3)
    tax_id: Optional[str] = Field(None, max_length=50)
    address: Optional[str] = Field(None, max_length=1000)
    contact_email: Optional[str] = Field(None, max_length=254)
    contact_phone: Optional[str] = Field(None, max_length=30)
    commodities: Optional[List[EUDRCommodity]] = None
    compliance_status: Optional[ComplianceStatus] = None
    certifications: Optional[List[str]] = None
    notes: Optional[str] = Field(None, max_length=5000)

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        """Uppercase country code if provided."""
        return v.upper() if v else v


class PlotCreateRequest(GreenLangBase):
    """Request to create a new plot."""

    supplier_id: str = Field(...)
    name: str = Field(..., min_length=1, max_length=500)
    coordinates: Optional[GeoJSONPolygon] = None
    centroid_lat: Optional[float] = Field(None, ge=-90.0, le=90.0)
    centroid_lon: Optional[float] = Field(None, ge=-180.0, le=180.0)
    area_hectares: Optional[float] = Field(None, ge=0.0)
    commodity: EUDRCommodity = Field(...)
    country_iso3: str = Field(..., min_length=3, max_length=3)
    region: Optional[str] = Field(None, max_length=200)

    @field_validator("country_iso3")
    @classmethod
    def validate_country(cls, v: str) -> str:
        """Uppercase country code."""
        return v.upper()


class ProcurementCreateRequest(GreenLangBase):
    """Request to create a new procurement record."""

    po_number: str = Field(..., min_length=1, max_length=100)
    commodity: EUDRCommodity = Field(...)
    quantity: float = Field(..., gt=0)
    unit: str = Field(default="tonnes")
    harvest_date: Optional[date] = None
    shipment_date: Optional[date] = None
    arrival_date: Optional[date] = None
    origin_plot_ids: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=5000)


class DocumentUploadRequest(GreenLangBase):
    """Request to upload a compliance document."""

    name: str = Field(..., min_length=1, max_length=500)
    doc_type: DocumentType = Field(...)
    file_path: Optional[str] = Field(None, max_length=1000)
    file_size_bytes: int = Field(default=0, ge=0)
    mime_type: Optional[str] = Field(None, max_length=100)
    linked_supplier_id: Optional[str] = None
    linked_plot_id: Optional[str] = None
    linked_dds_id: Optional[str] = None
    linked_procurement_id: Optional[str] = None
    issuer: Optional[str] = Field(None, max_length=500)
    expiry_date: Optional[date] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DDSGenerateRequest(GreenLangBase):
    """Request to generate a Due Diligence Statement."""

    supplier_id: str = Field(...)
    commodity: EUDRCommodity = Field(...)
    year: int = Field(..., ge=2024, le=2100)
    plots: List[str] = Field(..., min_length=1)
    procurement_ids: List[str] = Field(default_factory=list)
    operator_name: Optional[str] = Field(None, max_length=500)
    operator_country: Optional[str] = Field(None, max_length=3)
    product_description: Optional[str] = Field(None, max_length=2000)

    @field_validator("operator_country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        """Uppercase country code if provided."""
        return v.upper() if v else v


class PipelineStartRequest(GreenLangBase):
    """Request to start a compliance pipeline run."""

    supplier_id: str = Field(...)
    commodity: Optional[EUDRCommodity] = None
    plot_ids: List[str] = Field(default_factory=list)
    triggered_by: str = Field(default="user")


class SupplierFilterRequest(GreenLangBase):
    """Filters for listing suppliers."""

    country: Optional[str] = None
    commodity: Optional[EUDRCommodity] = None
    risk_level: Optional[RiskLevel] = None
    compliance_status: Optional[ComplianceStatus] = None
    search_query: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class DocumentFilterRequest(GreenLangBase):
    """Filters for listing documents."""

    doc_type: Optional[DocumentType] = None
    verification_status: Optional[VerificationStatus] = None
    linked_supplier_id: Optional[str] = None
    linked_plot_id: Optional[str] = None
    linked_dds_id: Optional[str] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class DDSFilterRequest(GreenLangBase):
    """Filters for listing DDS records."""

    supplier_id: Optional[str] = None
    commodity: Optional[EUDRCommodity] = None
    year: Optional[int] = None
    status: Optional[DDSStatus] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class PipelineFilterRequest(GreenLangBase):
    """Filters for listing pipeline runs."""

    supplier_id: Optional[str] = None
    status: Optional[PipelineStatus] = None
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


# ===========================================================================
# Response Models
# ===========================================================================


class BulkImportResult(GreenLangBase):
    """Result of a bulk supplier import operation.

    Attributes:
        total_records: Total records in the import.
        imported: Successfully imported count.
        failed: Failed import count.
        errors: List of error details.
        suppliers: List of created supplier IDs.
    """

    total_records: int = Field(default=0, ge=0)
    imported: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    suppliers: List[str] = Field(default_factory=list)


class SupplierComplianceStatus(GreenLangBase):
    """Comprehensive compliance status for a supplier.

    Attributes:
        supplier_id: Supplier identifier.
        compliance_status: Overall compliance status.
        risk_level: Current risk level.
        plots_total: Total plots.
        plots_compliant: Compliant plots.
        plots_at_risk: Plots at high/critical risk.
        documents_total: Total documents.
        documents_verified: Verified documents.
        documents_missing: Missing required documents.
        dds_status: Latest DDS status.
        last_pipeline_run: Last pipeline run timestamp.
        issues: List of outstanding compliance issues.
    """

    supplier_id: str = Field(...)
    compliance_status: ComplianceStatus = Field(default=ComplianceStatus.PENDING)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    plots_total: int = Field(default=0, ge=0)
    plots_compliant: int = Field(default=0, ge=0)
    plots_at_risk: int = Field(default=0, ge=0)
    documents_total: int = Field(default=0, ge=0)
    documents_verified: int = Field(default=0, ge=0)
    documents_missing: int = Field(default=0, ge=0)
    dds_status: Optional[DDSStatus] = None
    last_pipeline_run: Optional[datetime] = None
    issues: List[str] = Field(default_factory=list)


class SupplierRiskSummary(GreenLangBase):
    """Risk summary for a supplier.

    Attributes:
        supplier_id: Supplier identifier.
        overall_risk: Overall risk score.
        risk_level: Classified risk level.
        country_risk: Country risk component.
        satellite_risk: Satellite risk component.
        document_risk: Document completeness risk.
        supplier_history_risk: Historical compliance risk.
        high_risk_plots: Number of high-risk plots.
        alerts: Active risk alerts.
        last_assessed: Last assessment timestamp.
    """

    supplier_id: str = Field(...)
    overall_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_level: RiskLevel = Field(default=RiskLevel.STANDARD)
    country_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    satellite_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    document_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    supplier_history_risk: float = Field(default=0.0, ge=0.0, le=1.0)
    high_risk_plots: int = Field(default=0, ge=0)
    alerts: List[RiskAlert] = Field(default_factory=list)
    last_assessed: Optional[datetime] = None


class DocumentVerificationResult(GreenLangBase):
    """Result of verifying a compliance document.

    Attributes:
        doc_id: Document identifier.
        status: Verification outcome.
        score: Confidence score (0-1).
        checks_passed: Number of checks passed.
        checks_failed: Number of checks failed.
        checks_total: Total number of checks run.
        findings: Detailed findings.
        missing_fields: Fields required but not found.
        recommendations: Recommended next steps.
        verified_at: Verification timestamp.
    """

    doc_id: str = Field(...)
    status: VerificationStatus = Field(default=VerificationStatus.PENDING)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    checks_passed: int = Field(default=0, ge=0)
    checks_failed: int = Field(default=0, ge=0)
    checks_total: int = Field(default=0, ge=0)
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    missing_fields: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    verified_at: datetime = Field(default_factory=_utcnow)


class ComplianceCheckResult(GreenLangBase):
    """Result of a single EUDR compliance check.

    Attributes:
        rule_id: EUDR article/rule identifier.
        rule_name: Human-readable rule name.
        passed: Whether the check passed.
        severity: Severity if failed (info, warning, error, critical).
        message: Explanation of the result.
        evidence: Supporting evidence.
        remediation: Recommended remediation if failed.
    """

    rule_id: str = Field(..., description="EUDR rule identifier (e.g., EUDR-ART-3)")
    rule_name: str = Field(..., description="Rule name")
    passed: bool = Field(default=False)
    severity: str = Field(default="info", description="info|warning|error|critical")
    message: str = Field(default="", description="Explanation")
    evidence: List[str] = Field(default_factory=list, description="Evidence references")
    remediation: Optional[str] = Field(None, description="Remediation guidance")


class DocumentGapAnalysis(GreenLangBase):
    """Analysis of document gaps for a supplier's EUDR compliance.

    Attributes:
        supplier_id: Supplier identifier.
        required_documents: List of required document types.
        available_documents: List of available documents.
        missing_documents: List of missing document types.
        expired_documents: List of expired document IDs.
        coverage_pct: Document coverage percentage.
        gaps: Detailed gap descriptions.
        recommendations: Recommended actions.
    """

    supplier_id: str = Field(...)
    required_documents: List[str] = Field(default_factory=list)
    available_documents: List[Dict[str, Any]] = Field(default_factory=list)
    missing_documents: List[str] = Field(default_factory=list)
    expired_documents: List[str] = Field(default_factory=list)
    coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    gaps: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DDSValidationResult(GreenLangBase):
    """Result of validating a Due Diligence Statement.

    Attributes:
        dds_id: DDS identifier.
        is_valid: Whether the DDS passes all validation checks.
        errors: Validation errors (block submission).
        warnings: Validation warnings (non-blocking).
        completeness_pct: Section completeness percentage.
        missing_sections: Sections that are incomplete.
        checked_at: Validation timestamp.
    """

    dds_id: str = Field(...)
    is_valid: bool = Field(default=False)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    missing_sections: List[str] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=_utcnow)


class DDSSubmissionResult(GreenLangBase):
    """Result of submitting a DDS to the EU Information System.

    Attributes:
        dds_id: DDS identifier.
        reference_number: DDS reference number.
        submitted: Whether submission was successful.
        eu_submission_id: EU system tracking ID.
        submission_date: Submission timestamp.
        status: Resulting DDS status.
        response_message: EU system response message.
        errors: Submission errors.
    """

    dds_id: str = Field(...)
    reference_number: str = Field(...)
    submitted: bool = Field(default=False)
    eu_submission_id: Optional[str] = None
    submission_date: Optional[datetime] = None
    status: DDSStatus = Field(default=DDSStatus.DRAFT)
    response_message: Optional[str] = None
    errors: List[str] = Field(default_factory=list)


class DDSAnnualSummary(GreenLangBase):
    """Annual summary of DDS activity.

    Attributes:
        year: Reporting year.
        total_dds: Total DDS generated.
        by_status: Count by DDS status.
        by_commodity: Count by commodity.
        by_country: Count by production country.
        acceptance_rate: Percentage accepted by EU system.
        average_processing_days: Average days from draft to submission.
        top_rejection_reasons: Most common rejection reasons.
    """

    year: int = Field(..., ge=2024)
    total_dds: int = Field(default=0, ge=0)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_commodity: Dict[str, int] = Field(default_factory=dict)
    by_country: Dict[str, int] = Field(default_factory=dict)
    acceptance_rate: float = Field(default=0.0, ge=0.0, le=100.0)
    average_processing_days: float = Field(default=0.0, ge=0.0)
    top_rejection_reasons: List[Dict[str, Any]] = Field(default_factory=list)

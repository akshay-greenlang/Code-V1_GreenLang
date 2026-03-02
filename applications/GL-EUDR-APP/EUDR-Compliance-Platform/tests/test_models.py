"""
Unit tests for GL-EUDR-APP v1.0 Pydantic domain models.

Tests all domain models: Supplier, Plot, Procurement, Document,
DueDiligenceStatement, PipelineRun, StageResult, RiskAssessment,
and DashboardMetrics. Validates creation, field constraints,
serialization, enum validation, and edge cases.

Test count target: 30+ tests
"""

import re
import uuid
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError, field_validator


# ---------------------------------------------------------------------------
# Domain Models (self-contained for testing)
# ---------------------------------------------------------------------------
# These models represent the GL-EUDR-APP domain. They mirror what will be
# defined in services/models.py once that file is built.


class EUDRCommodity:
    ALLOWED = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}


class Supplier(BaseModel):
    """Supplier profile for EUDR compliance."""
    supplier_id: str = Field(default_factory=lambda: f"sup_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1, max_length=500)
    country_iso3: str = Field(..., min_length=3, max_length=3)
    country_iso2: Optional[str] = Field(None, min_length=2, max_length=2)
    tax_id: Optional[str] = Field(None, max_length=100)
    commodities: List[str] = Field(default_factory=list)
    compliance_status: str = Field(default="pending")
    risk_level: str = Field(default="standard")
    overall_risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    erp_source: Optional[str] = None
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("commodities")
    @classmethod
    def validate_commodities(cls, v: List[str]) -> List[str]:
        for c in v:
            if c.lower() not in EUDRCommodity.ALLOWED:
                raise ValueError(f"Invalid commodity '{c}'. Allowed: {sorted(EUDRCommodity.ALLOWED)}")
        return [c.lower() for c in v]

    @field_validator("compliance_status")
    @classmethod
    def validate_compliance_status(cls, v: str) -> str:
        allowed = {"compliant", "pending", "non_compliant", "under_review"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        allowed = {"low", "standard", "high", "critical"}
        if v not in allowed:
            raise ValueError(f"Invalid risk_level '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("country_iso3")
    @classmethod
    def validate_country_iso3(cls, v: str) -> str:
        if not v.isalpha():
            raise ValueError("country_iso3 must contain only letters")
        return v.upper()


class Plot(BaseModel):
    """Production plot with GeoJSON polygon."""
    plot_id: str = Field(default_factory=lambda: f"plot_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1, max_length=500)
    supplier_id: str
    coordinates: Dict[str, Any] = Field(...)
    commodity: str
    country_iso3: str = Field(..., min_length=3, max_length=3)
    area_hectares: Optional[float] = Field(None, ge=0)
    risk_level: str = Field(default="standard")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("coordinates")
    @classmethod
    def validate_geojson(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if v.get("type") != "Polygon":
            raise ValueError("Geometry type must be 'Polygon'")
        coords = v.get("coordinates")
        if not coords or not isinstance(coords, list) or not coords[0]:
            raise ValueError("Coordinates must have at least one ring")
        ring = coords[0]
        if len(ring) < 4:
            raise ValueError("A polygon ring must have at least 4 points")
        if ring[0] != ring[-1]:
            raise ValueError("Polygon ring must be closed (first == last)")
        return v

    @field_validator("risk_level")
    @classmethod
    def validate_risk_level(cls, v: str) -> str:
        allowed = {"low", "standard", "high", "critical"}
        if v not in allowed:
            raise ValueError(f"Invalid risk_level '{v}'")
        return v


class Procurement(BaseModel):
    """Procurement record for EUDR-regulated commodity."""
    procurement_id: str = Field(default_factory=lambda: f"proc_{uuid.uuid4().hex[:12]}")
    supplier_id: str
    commodity: str
    quantity_kg: float = Field(..., gt=0)
    po_number: str = Field(..., min_length=1, max_length=100)
    procurement_date: datetime
    delivery_date: Optional[datetime] = None
    status: str = Field(default="draft")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("delivery_date")
    @classmethod
    def validate_delivery_after_procurement(cls, v, info):
        if v is not None and "procurement_date" in info.data:
            if v < info.data["procurement_date"]:
                raise ValueError("delivery_date cannot be before procurement_date")
        return v


class Document(BaseModel):
    """Compliance document for verification."""
    document_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}")
    name: str = Field(..., min_length=1, max_length=500)
    doc_type: str
    file_path: Optional[str] = None
    file_size_bytes: Optional[int] = Field(None, ge=0)
    verification_status: str = Field(default="pending")
    verification_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    linked_supplier_id: Optional[str] = None
    linked_plot_id: Optional[str] = None
    linked_dds_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("doc_type")
    @classmethod
    def validate_doc_type(cls, v: str) -> str:
        allowed = {"CERTIFICATE", "PERMIT", "LAND_TITLE", "INVOICE", "TRANSPORT", "OTHER"}
        if v not in allowed:
            raise ValueError(f"Invalid doc_type '{v}'. Allowed: {sorted(allowed)}")
        return v

    @field_validator("verification_status")
    @classmethod
    def validate_verification_status(cls, v: str) -> str:
        allowed = {"pending", "verified", "failed", "expired"}
        if v not in allowed:
            raise ValueError(f"Invalid verification_status '{v}'")
        return v


class DueDiligenceStatement(BaseModel):
    """Due Diligence Statement per EU Regulation 2023/1115."""
    dds_id: str = Field(default_factory=lambda: f"dds_{uuid.uuid4().hex[:12]}")
    reference_number: str  # EUDR-{ISO3}-{YEAR}-{SEQ:06d}
    supplier_id: str
    commodity: str
    year: int = Field(..., ge=2024, le=2100)
    status: str = Field(default="draft")
    operator_info: Dict[str, Any] = Field(default_factory=dict)
    product_description: Dict[str, Any] = Field(default_factory=dict)
    country_of_production: Dict[str, Any] = Field(default_factory=dict)
    geolocation_data: Dict[str, Any] = Field(default_factory=dict)
    risk_assessment: Dict[str, Any] = Field(default_factory=dict)
    risk_mitigation: Dict[str, Any] = Field(default_factory=dict)
    conclusion: Dict[str, Any] = Field(default_factory=dict)
    plot_ids: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    overall_risk_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    submission_date: Optional[datetime] = None
    amendment_of: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("reference_number")
    @classmethod
    def validate_reference_number(cls, v: str) -> str:
        pattern = r"^EUDR-[A-Z]{3}-\d{4}-\d{6}$"
        if not re.match(pattern, v):
            raise ValueError(
                f"reference_number must match EUDR-XXX-YYYY-NNNNNN format, got '{v}'"
            )
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"draft", "review", "validated", "submitted", "accepted", "rejected", "amended"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'")
        return v


class PipelineRun(BaseModel):
    """Pipeline execution run tracking."""
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    supplier_id: str
    commodity: Optional[str] = None
    status: str = Field(default="pending")
    current_stage: Optional[str] = None
    stages: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        allowed = {"pending", "running", "completed", "failed", "cancelled"}
        if v not in allowed:
            raise ValueError(f"Invalid status '{v}'")
        return v

    @field_validator("current_stage")
    @classmethod
    def validate_current_stage(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"intake", "geo_validation", "deforestation_risk", "document_verification", "dds_reporting"}
        if v not in allowed:
            raise ValueError(f"Invalid stage '{v}'")
        return v


class StageResult(BaseModel):
    """Result of a single pipeline stage execution."""
    stage: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    output: Dict[str, Any] = Field(default_factory=dict)

    @property
    def computed_duration_ms(self) -> Optional[float]:
        """Compute duration from timestamps if not explicitly set."""
        if self.duration_ms is not None:
            return self.duration_ms
        if self.completed_at is not None:
            delta = self.completed_at - self.started_at
            return delta.total_seconds() * 1000
        return None


class RiskAssessment(BaseModel):
    """Risk assessment combining multiple risk sources."""
    assessment_id: str = Field(default_factory=lambda: f"ra_{uuid.uuid4().hex[:12]}")
    satellite_risk: float = Field(0.0, ge=0.0, le=1.0)
    country_risk: float = Field(0.0, ge=0.0, le=1.0)
    supplier_risk: float = Field(0.0, ge=0.0, le=1.0)
    document_risk: float = Field(0.0, ge=0.0, le=1.0)
    overall_risk: float = Field(0.0, ge=0.0, le=1.0)
    risk_level: Optional[str] = None
    factors: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    assessed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def derived_risk_level(self) -> str:
        """Derive risk level from overall score."""
        if self.overall_risk >= 0.9:
            return "critical"
        elif self.overall_risk >= 0.7:
            return "high"
        elif self.overall_risk >= 0.4:
            return "standard"
        return "low"


class DashboardMetrics(BaseModel):
    """Dashboard aggregated metrics."""
    total_suppliers: int = Field(0, ge=0)
    compliant_suppliers: int = Field(0, ge=0)
    non_compliant_suppliers: int = Field(0, ge=0)
    pending_suppliers: int = Field(0, ge=0)
    total_dds: int = Field(0, ge=0)
    submitted_dds: int = Field(0, ge=0)
    accepted_dds: int = Field(0, ge=0)
    rejected_dds: int = Field(0, ge=0)
    total_plots: int = Field(0, ge=0)
    high_risk_plots: int = Field(0, ge=0)
    active_alerts: int = Field(0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def compliance_rate(self) -> float:
        """Percentage of suppliers that are compliant."""
        if self.total_suppliers == 0:
            return 0.0
        return (self.compliant_suppliers / self.total_suppliers) * 100.0


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def _valid_polygon_coords():
    """Return a simple valid closed polygon (square in Amazon region)."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [-60.0, -3.0], [-60.0, -2.0], [-59.0, -2.0],
            [-59.0, -3.0], [-60.0, -3.0]
        ]]
    }


# ---------------------------------------------------------------------------
# TestSupplier
# ---------------------------------------------------------------------------

class TestSupplier:
    """Tests for the Supplier domain model."""

    def test_create_supplier_valid(self):
        """Supplier creation with all required fields succeeds."""
        s = Supplier(name="Amazonia Timber Co.", country_iso3="BRA", commodities=["wood", "soya"])
        assert s.name == "Amazonia Timber Co."
        assert s.country_iso3 == "BRA"
        assert s.commodities == ["wood", "soya"]
        assert s.supplier_id.startswith("sup_")

    def test_supplier_default_values(self):
        """Supplier defaults are applied correctly."""
        s = Supplier(name="Test Co.", country_iso3="DEU")
        assert s.compliance_status == "pending"
        assert s.risk_level == "standard"
        assert s.overall_risk_score == 0.0
        assert s.erp_source is None
        assert s.tax_id is None
        assert s.notes is None
        assert s.created_at is not None
        assert s.updated_at is not None

    def test_supplier_commodity_validation_valid(self):
        """Valid EUDR commodities are accepted and lowercased."""
        s = Supplier(name="Test", country_iso3="BRA", commodities=["COCOA", "Coffee"])
        assert s.commodities == ["cocoa", "coffee"]

    def test_supplier_commodity_validation_invalid(self):
        """Invalid commodity raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Supplier(name="Test", country_iso3="BRA", commodities=["bananas"])
        assert "Invalid commodity" in str(exc_info.value)

    def test_supplier_all_seven_commodities(self):
        """All seven EUDR commodities are accepted."""
        all_seven = ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"]
        s = Supplier(name="Full Scope", country_iso3="IDN", commodities=all_seven)
        assert len(s.commodities) == 7

    def test_supplier_country_code_uppercase(self):
        """Country code is uppercased automatically."""
        s = Supplier(name="Test", country_iso3="bra")
        assert s.country_iso3 == "BRA"

    def test_supplier_country_code_invalid_chars(self):
        """Country code with non-alpha characters is rejected."""
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="12A")

    def test_supplier_country_code_wrong_length(self):
        """Country code with wrong length is rejected."""
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="BR")

    def test_supplier_serialization_roundtrip(self):
        """Supplier can be serialized to dict and back."""
        s = Supplier(name="SerTest", country_iso3="DEU", commodities=["wood"])
        d = s.model_dump()
        assert d["name"] == "SerTest"
        assert d["country_iso3"] == "DEU"
        s2 = Supplier(**d)
        assert s2.name == s.name
        assert s2.supplier_id == s.supplier_id

    def test_supplier_compliance_status_invalid(self):
        """Invalid compliance_status raises ValidationError."""
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="BRA", compliance_status="unknown")

    def test_supplier_risk_level_invalid(self):
        """Invalid risk_level raises ValidationError."""
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="BRA", risk_level="medium")

    def test_supplier_risk_score_bounds(self):
        """Risk score outside [0, 1] is rejected."""
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="BRA", overall_risk_score=1.5)
        with pytest.raises(ValidationError):
            Supplier(name="Test", country_iso3="BRA", overall_risk_score=-0.1)

    def test_supplier_name_empty_rejected(self):
        """Empty supplier name is rejected."""
        with pytest.raises(ValidationError):
            Supplier(name="", country_iso3="BRA")


# ---------------------------------------------------------------------------
# TestPlot
# ---------------------------------------------------------------------------

class TestPlot:
    """Tests for the Plot domain model."""

    def test_create_plot_valid(self):
        """Plot creation with valid GeoJSON polygon succeeds."""
        p = Plot(
            name="Plot Alpha",
            supplier_id="sup_abc123",
            coordinates=_valid_polygon_coords(),
            commodity="soya",
            country_iso3="BRA",
        )
        assert p.name == "Plot Alpha"
        assert p.plot_id.startswith("plot_")

    def test_plot_geojson_polygon_validation(self):
        """GeoJSON type must be Polygon."""
        with pytest.raises(ValidationError) as exc_info:
            Plot(
                name="Bad",
                supplier_id="sup_x",
                coordinates={"type": "Point", "coordinates": [1, 2]},
                commodity="wood",
                country_iso3="BRA",
            )
        assert "Polygon" in str(exc_info.value)

    def test_plot_polygon_not_closed(self):
        """Polygon ring that is not closed is rejected."""
        with pytest.raises(ValidationError):
            Plot(
                name="Open",
                supplier_id="sup_x",
                coordinates={
                    "type": "Polygon",
                    "coordinates": [[[-60, -3], [-60, -2], [-59, -2], [-59, -3]]]
                },
                commodity="wood",
                country_iso3="BRA",
            )

    def test_plot_polygon_too_few_points(self):
        """Polygon with fewer than 4 points is rejected."""
        with pytest.raises(ValidationError):
            Plot(
                name="Tiny",
                supplier_id="sup_x",
                coordinates={
                    "type": "Polygon",
                    "coordinates": [[[-60, -3], [-60, -2], [-60, -3]]]
                },
                commodity="wood",
                country_iso3="BRA",
            )

    def test_plot_area_non_negative(self):
        """Negative area_hectares is rejected."""
        with pytest.raises(ValidationError):
            Plot(
                name="NegArea",
                supplier_id="sup_x",
                coordinates=_valid_polygon_coords(),
                commodity="wood",
                country_iso3="BRA",
                area_hectares=-10.0,
            )

    def test_plot_risk_level_enum(self):
        """Invalid risk_level is rejected."""
        with pytest.raises(ValidationError):
            Plot(
                name="Bad Risk",
                supplier_id="sup_x",
                coordinates=_valid_polygon_coords(),
                commodity="wood",
                country_iso3="BRA",
                risk_level="unknown",
            )


# ---------------------------------------------------------------------------
# TestProcurement
# ---------------------------------------------------------------------------

class TestProcurement:
    """Tests for the Procurement domain model."""

    def test_create_procurement_valid(self):
        """Procurement creation with required fields succeeds."""
        now = datetime.now(timezone.utc)
        p = Procurement(
            supplier_id="sup_abc",
            commodity="coffee",
            quantity_kg=5000.0,
            po_number="PO-2026-001",
            procurement_date=now,
        )
        assert p.po_number == "PO-2026-001"
        assert p.status == "draft"

    def test_procurement_date_validation(self):
        """delivery_date before procurement_date is rejected."""
        now = datetime.now(timezone.utc)
        past = now - timedelta(days=10)
        with pytest.raises(ValidationError):
            Procurement(
                supplier_id="sup_abc",
                commodity="wood",
                quantity_kg=100.0,
                po_number="PO-001",
                procurement_date=now,
                delivery_date=past,
            )

    def test_procurement_quantity_positive(self):
        """Zero or negative quantity_kg is rejected."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            Procurement(
                supplier_id="sup_abc",
                commodity="wood",
                quantity_kg=0,
                po_number="PO-001",
                procurement_date=now,
            )

    def test_procurement_po_number_required(self):
        """Empty po_number is rejected."""
        now = datetime.now(timezone.utc)
        with pytest.raises(ValidationError):
            Procurement(
                supplier_id="sup_abc",
                commodity="wood",
                quantity_kg=100.0,
                po_number="",
                procurement_date=now,
            )


# ---------------------------------------------------------------------------
# TestDocument
# ---------------------------------------------------------------------------

class TestDocument:
    """Tests for the Document domain model."""

    def test_create_document_valid(self):
        """Document creation with valid doc_type succeeds."""
        d = Document(name="Forest Cert.pdf", doc_type="CERTIFICATE")
        assert d.doc_type == "CERTIFICATE"
        assert d.verification_status == "pending"

    def test_document_type_enum(self):
        """Invalid doc_type is rejected."""
        with pytest.raises(ValidationError):
            Document(name="Bad.pdf", doc_type="SPREADSHEET")

    def test_document_all_types_valid(self):
        """All six document types are accepted."""
        for dt in ["CERTIFICATE", "PERMIT", "LAND_TITLE", "INVOICE", "TRANSPORT", "OTHER"]:
            d = Document(name=f"{dt}.pdf", doc_type=dt)
            assert d.doc_type == dt

    def test_document_verification_status_valid(self):
        """Valid verification statuses are accepted."""
        for vs in ["pending", "verified", "failed", "expired"]:
            d = Document(name="test.pdf", doc_type="OTHER", verification_status=vs)
            assert d.verification_status == vs

    def test_document_verification_status_invalid(self):
        """Invalid verification_status is rejected."""
        with pytest.raises(ValidationError):
            Document(name="test.pdf", doc_type="OTHER", verification_status="approved")

    def test_document_verification_score_bounds(self):
        """verification_score outside [0, 1] is rejected."""
        with pytest.raises(ValidationError):
            Document(name="test.pdf", doc_type="OTHER", verification_score=1.5)


# ---------------------------------------------------------------------------
# TestDueDiligenceStatement
# ---------------------------------------------------------------------------

class TestDueDiligenceStatement:
    """Tests for the DueDiligenceStatement domain model."""

    def test_create_dds_valid(self):
        """DDS creation with valid reference number succeeds."""
        dds = DueDiligenceStatement(
            reference_number="EUDR-BRA-2026-000001",
            supplier_id="sup_abc",
            commodity="soya",
            year=2026,
        )
        assert dds.status == "draft"
        assert dds.reference_number == "EUDR-BRA-2026-000001"

    def test_dds_reference_number_format_valid(self):
        """Multiple valid reference number formats are accepted."""
        valid_refs = [
            "EUDR-BRA-2026-000001",
            "EUDR-IDN-2025-999999",
            "EUDR-DEU-2030-000100",
        ]
        for ref in valid_refs:
            dds = DueDiligenceStatement(
                reference_number=ref, supplier_id="sup_x", commodity="wood", year=2026
            )
            assert dds.reference_number == ref

    def test_dds_reference_number_format_invalid(self):
        """Invalid reference number formats are rejected."""
        invalid_refs = [
            "EUDR-BR-2026-000001",    # 2-char country
            "EUDR-BRA-2026-0001",     # 4-digit sequence
            "DDS-BRA-2026-000001",    # wrong prefix
            "EUDR-bra-2026-000001",   # lowercase country
            "EUDR-BRA-26-000001",     # 2-digit year
            "EUDR-BRA-2026-1",        # 1-digit sequence
        ]
        for ref in invalid_refs:
            with pytest.raises(ValidationError):
                DueDiligenceStatement(
                    reference_number=ref, supplier_id="sup_x", commodity="wood", year=2026
                )

    def test_dds_status_transitions(self):
        """All valid DDS statuses are accepted."""
        for st in ["draft", "review", "validated", "submitted", "accepted", "rejected", "amended"]:
            dds = DueDiligenceStatement(
                reference_number="EUDR-BRA-2026-000001",
                supplier_id="sup_x",
                commodity="wood",
                year=2026,
                status=st,
            )
            assert dds.status == st

    def test_dds_status_invalid(self):
        """Invalid DDS status is rejected."""
        with pytest.raises(ValidationError):
            DueDiligenceStatement(
                reference_number="EUDR-BRA-2026-000001",
                supplier_id="sup_x",
                commodity="wood",
                year=2026,
                status="approved",
            )

    def test_dds_section_completeness(self):
        """All 7 DDS sections can be populated."""
        dds = DueDiligenceStatement(
            reference_number="EUDR-BRA-2026-000001",
            supplier_id="sup_x",
            commodity="soya",
            year=2026,
            operator_info={"name": "Test Corp", "eori": "DE1234567890"},
            product_description={"hs_code": "1201.90"},
            country_of_production={"iso3": "BRA", "region": "Mato Grosso"},
            geolocation_data={"plots": [{"lat": -12.0, "lon": -55.0}]},
            risk_assessment={"score": 0.3, "level": "standard"},
            risk_mitigation={"measures": ["satellite monitoring"]},
            conclusion={"compliant": True, "notes": "All checks passed"},
        )
        assert dds.operator_info["name"] == "Test Corp"
        assert dds.conclusion["compliant"] is True

    def test_dds_year_bounds(self):
        """Year outside [2024, 2100] is rejected."""
        with pytest.raises(ValidationError):
            DueDiligenceStatement(
                reference_number="EUDR-BRA-2023-000001",
                supplier_id="sup_x",
                commodity="wood",
                year=2023,
            )

    def test_dds_amendment_link(self):
        """DDS can reference an amendment of another DDS."""
        dds = DueDiligenceStatement(
            reference_number="EUDR-BRA-2026-000002",
            supplier_id="sup_x",
            commodity="wood",
            year=2026,
            status="amended",
            amendment_of="dds_original123",
        )
        assert dds.amendment_of == "dds_original123"


# ---------------------------------------------------------------------------
# TestPipelineRun
# ---------------------------------------------------------------------------

class TestPipelineRun:
    """Tests for the PipelineRun domain model."""

    def test_create_pipeline_run(self):
        """PipelineRun creation with defaults."""
        pr = PipelineRun(supplier_id="sup_abc")
        assert pr.status == "pending"
        assert pr.current_stage is None
        assert pr.run_id.startswith("run_")

    def test_pipeline_stage_enumeration(self):
        """All valid pipeline stages are accepted."""
        for stage in ["intake", "geo_validation", "deforestation_risk",
                       "document_verification", "dds_reporting"]:
            pr = PipelineRun(supplier_id="sup_x", current_stage=stage)
            assert pr.current_stage == stage

    def test_pipeline_stage_invalid(self):
        """Invalid stage is rejected."""
        with pytest.raises(ValidationError):
            PipelineRun(supplier_id="sup_x", current_stage="cleanup")

    def test_pipeline_status_tracking(self):
        """All valid pipeline statuses are accepted."""
        for st in ["pending", "running", "completed", "failed", "cancelled"]:
            pr = PipelineRun(supplier_id="sup_x", status=st)
            assert pr.status == st

    def test_pipeline_status_invalid(self):
        """Invalid pipeline status is rejected."""
        with pytest.raises(ValidationError):
            PipelineRun(supplier_id="sup_x", status="paused")


# ---------------------------------------------------------------------------
# TestStageResult
# ---------------------------------------------------------------------------

class TestStageResult:
    """Tests for the StageResult domain model."""

    def test_stage_result_timing(self):
        """Stage duration is computed from timestamps."""
        start = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 12, 0, 2, 500000, tzinfo=timezone.utc)
        sr = StageResult(stage="intake", status="completed", started_at=start, completed_at=end)
        assert sr.computed_duration_ms == pytest.approx(2500.0, abs=1)

    def test_stage_result_explicit_duration(self):
        """Explicit duration_ms takes precedence over computed."""
        start = datetime.now(timezone.utc)
        sr = StageResult(stage="intake", status="completed", started_at=start, duration_ms=150.5)
        assert sr.computed_duration_ms == 150.5

    def test_stage_result_no_completion(self):
        """Duration is None when stage is not completed."""
        start = datetime.now(timezone.utc)
        sr = StageResult(stage="geo_validation", status="running", started_at=start)
        assert sr.computed_duration_ms is None

    def test_stage_result_error_handling(self):
        """StageResult can carry error information."""
        sr = StageResult(
            stage="deforestation_risk",
            status="failed",
            started_at=datetime.now(timezone.utc),
            error="Satellite API timeout",
        )
        assert sr.error == "Satellite API timeout"
        assert sr.status == "failed"


# ---------------------------------------------------------------------------
# TestRiskAssessment
# ---------------------------------------------------------------------------

class TestRiskAssessment:
    """Tests for the RiskAssessment domain model."""

    def test_risk_score_bounds(self):
        """All risk scores must be in [0, 1]."""
        ra = RiskAssessment(satellite_risk=0.8, country_risk=0.5, supplier_risk=0.3,
                            document_risk=0.2, overall_risk=0.45)
        assert 0 <= ra.satellite_risk <= 1
        assert 0 <= ra.overall_risk <= 1

    def test_risk_score_out_of_bounds(self):
        """Risk score > 1 is rejected."""
        with pytest.raises(ValidationError):
            RiskAssessment(overall_risk=1.1)

    def test_risk_score_negative_rejected(self):
        """Negative risk score is rejected."""
        with pytest.raises(ValidationError):
            RiskAssessment(satellite_risk=-0.1)

    def test_risk_level_derivation_critical(self):
        """Overall risk >= 0.9 derives as critical."""
        ra = RiskAssessment(overall_risk=0.95)
        assert ra.derived_risk_level == "critical"

    def test_risk_level_derivation_high(self):
        """Overall risk >= 0.7 derives as high."""
        ra = RiskAssessment(overall_risk=0.75)
        assert ra.derived_risk_level == "high"

    def test_risk_level_derivation_standard(self):
        """Overall risk >= 0.4 derives as standard."""
        ra = RiskAssessment(overall_risk=0.5)
        assert ra.derived_risk_level == "standard"

    def test_risk_level_derivation_low(self):
        """Overall risk < 0.4 derives as low."""
        ra = RiskAssessment(overall_risk=0.2)
        assert ra.derived_risk_level == "low"


# ---------------------------------------------------------------------------
# TestDashboardMetrics
# ---------------------------------------------------------------------------

class TestDashboardMetrics:
    """Tests for the DashboardMetrics domain model."""

    def test_compliance_rate_calculation(self):
        """compliance_rate is correctly computed."""
        dm = DashboardMetrics(total_suppliers=100, compliant_suppliers=75)
        assert dm.compliance_rate == pytest.approx(75.0)

    def test_compliance_rate_zero_suppliers(self):
        """compliance_rate returns 0 when no suppliers."""
        dm = DashboardMetrics(total_suppliers=0, compliant_suppliers=0)
        assert dm.compliance_rate == 0.0

    def test_non_negative_values(self):
        """Negative counts are rejected."""
        with pytest.raises(ValidationError):
            DashboardMetrics(total_suppliers=-1)
        with pytest.raises(ValidationError):
            DashboardMetrics(active_alerts=-5)

    def test_dashboard_defaults(self):
        """All dashboard fields default to zero."""
        dm = DashboardMetrics()
        assert dm.total_suppliers == 0
        assert dm.total_dds == 0
        assert dm.total_plots == 0
        assert dm.active_alerts == 0
        assert dm.timestamp is not None

    def test_dashboard_all_fields(self):
        """All dashboard fields can be set."""
        dm = DashboardMetrics(
            total_suppliers=200,
            compliant_suppliers=150,
            non_compliant_suppliers=30,
            pending_suppliers=20,
            total_dds=180,
            submitted_dds=160,
            accepted_dds=140,
            rejected_dds=20,
            total_plots=500,
            high_risk_plots=50,
            active_alerts=15,
        )
        assert dm.compliance_rate == pytest.approx(75.0)
        assert dm.high_risk_plots == 50

# -*- coding: utf-8 -*-
"""
Unit tests for models.py - AGENT-EUDR-030

Tests all enumerations, model creation, defaults, Decimal fields,
constants, serialization, and optional fields.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.documentation_generator.models import (
    AGENT_ID,
    AGENT_VERSION,
    ARTICLE9_MANDATORY_ELEMENTS,
    DDS_MANDATORY_FIELDS,
    SUPPORTED_COMMODITIES,
    Article9Element,
    Article9Package,
    AuditAction,
    CompliancePackage,
    ComplianceSection,
    DDSContent,
    DDSDocument,
    DDSStatus,
    DocumentType,
    DocumentVersion,
    EUDRCommodity,
    GeolocationReference,
    HealthStatus,
    MeasureSummary,
    MitigationDoc,
    PackageFormat,
    ProductEntry,
    RetentionStatus,
    RiskAssessmentDoc,
    RiskLevel,
    SubmissionRecord,
    SubmissionStatus,
    SupplierReference,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    VersionStatus,
)


class TestEnums:
    """Test all enum definitions and membership."""

    def test_eudr_commodity_values(self):
        """Test all EUDR commodity enum values."""
        assert EUDRCommodity.CATTLE == "cattle"
        assert EUDRCommodity.COCOA == "cocoa"
        assert EUDRCommodity.COFFEE == "coffee"
        assert EUDRCommodity.PALM_OIL == "palm_oil"
        assert EUDRCommodity.RUBBER == "rubber"
        assert EUDRCommodity.SOYA == "soya"
        assert EUDRCommodity.WOOD == "wood"
        assert len(EUDRCommodity) == 7

    def test_risk_level_values(self):
        """Test all risk level enum values."""
        assert RiskLevel.NEGLIGIBLE == "negligible"
        assert RiskLevel.LOW == "low"
        assert RiskLevel.STANDARD == "standard"
        assert RiskLevel.HIGH == "high"
        assert RiskLevel.CRITICAL == "critical"
        assert len(RiskLevel) == 5

    def test_dds_status_values(self):
        """Test all DDS status enum values."""
        expected = {"draft", "validated", "submitted", "acknowledged", "rejected", "amended"}
        actual = {s.value for s in DDSStatus}
        assert actual == expected

    def test_document_type_values(self):
        """Test all document type enum values."""
        expected = {"dds", "article9_package", "risk_assessment", "mitigation_report",
                    "compliance_package", "audit_report"}
        actual = {d.value for d in DocumentType}
        assert actual == expected

    def test_submission_status_values(self):
        """Test all submission status enum values."""
        expected = {"pending", "validating", "submitted", "acknowledged", "rejected", "resubmitted"}
        actual = {s.value for s in SubmissionStatus}
        assert actual == expected

    def test_package_format_values(self):
        """Test all package format enum values."""
        expected = {"json", "xml", "pdf_structured"}
        actual = {f.value for f in PackageFormat}
        assert actual == expected

    def test_validation_severity_values(self):
        """Test all validation severity enum values."""
        expected = {"error", "warning", "info"}
        actual = {v.value for v in ValidationSeverity}
        assert actual == expected

    def test_version_status_values(self):
        """Test all version status enum values."""
        expected = {"draft", "final", "submitted", "amended", "superseded", "archived"}
        actual = {v.value for v in VersionStatus}
        assert actual == expected

    def test_article9_element_values(self):
        """Test all Article 9 element enum values."""
        expected = {
            "product_description", "quantity", "country_of_production",
            "geolocation", "production_date", "supplier_info",
            "buyer_info", "certifications", "trade_codes", "supporting_evidence"
        }
        actual = {e.value for e in Article9Element}
        assert actual == expected
        assert len(Article9Element) == 10

    def test_compliance_section_values(self):
        """Test all compliance section enum values."""
        expected = {"information_gathering", "risk_assessment", "risk_mitigation", "compliance_conclusion"}
        actual = {s.value for s in ComplianceSection}
        assert actual == expected

    def test_retention_status_values(self):
        """Test all retention status enum values."""
        expected = {"active", "expiring_soon", "expired", "archived"}
        actual = {r.value for r in RetentionStatus}
        assert actual == expected

    def test_audit_action_values(self):
        """Test all audit action enum values."""
        expected = {"create", "update", "validate", "submit", "reject", "amend", "archive", "delete"}
        actual = {a.value for a in AuditAction}
        assert actual == expected


class TestConstants:
    """Test module-level constants."""

    def test_agent_id(self):
        """Test AGENT_ID constant value."""
        assert AGENT_ID == "GL-EUDR-DGN-030"

    def test_agent_version(self):
        """Test AGENT_VERSION constant value."""
        assert AGENT_VERSION == "1.0.0"

    def test_article9_mandatory_elements_count(self):
        """Test that all 10 Article 9 mandatory elements are listed."""
        assert len(ARTICLE9_MANDATORY_ELEMENTS) == 10

    def test_article9_mandatory_elements_includes_product_description(self):
        """Test Article 9 mandatory elements includes product description."""
        assert Article9Element.PRODUCT_DESCRIPTION in ARTICLE9_MANDATORY_ELEMENTS

    def test_article9_mandatory_elements_includes_geolocation(self):
        """Test Article 9 mandatory elements includes geolocation."""
        assert Article9Element.GEOLOCATION in ARTICLE9_MANDATORY_ELEMENTS

    def test_dds_mandatory_fields(self):
        """Test DDS mandatory fields list."""
        expected = [
            "operator_id", "commodity", "products", "article9_ref",
            "risk_assessment_ref", "compliance_conclusion"
        ]
        assert DDS_MANDATORY_FIELDS == expected

    def test_supported_commodities_count(self):
        """Test that all 7 commodities are supported."""
        assert len(SUPPORTED_COMMODITIES) == 7

    def test_supported_commodities_includes_coffee(self):
        """Test supported commodities includes coffee."""
        assert "coffee" in SUPPORTED_COMMODITIES

    def test_supported_commodities_includes_wood(self):
        """Test supported commodities includes wood."""
        assert "wood" in SUPPORTED_COMMODITIES


class TestProductEntryModel:
    """Test ProductEntry model creation and defaults."""

    def test_create_valid_product_entry(self):
        """Test creating a valid ProductEntry."""
        product = ProductEntry(
            product_id="prod-001",
            description="Premium Arabica Coffee",
            hs_code="0901.11",
            cn_code="09011100",
            quantity=Decimal("1000"),
            unit="kg",
        )
        assert product.product_id == "prod-001"
        assert product.description == "Premium Arabica Coffee"
        assert product.quantity == Decimal("1000")
        assert product.unit == "kg"

    def test_product_entry_defaults(self):
        """Test ProductEntry default values."""
        product = ProductEntry(
            product_id="prod-002",
            description="Test Product",
            quantity=Decimal("500"),
            unit="tonnes",
        )
        assert product.hs_code == ""
        assert product.cn_code == ""

    def test_product_entry_model_dump(self):
        """Test ProductEntry serialization via model_dump."""
        product = ProductEntry(
            product_id="prod-003",
            description="Test",
            quantity=Decimal("100"),
            unit="m3",
        )
        dumped = product.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["product_id"] == "prod-003"
        assert dumped["unit"] == "m3"

    def test_product_entry_quantity_must_be_non_negative(self):
        """Test ProductEntry quantity validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            ProductEntry(
                product_id="prod-bad",
                description="Invalid",
                quantity=Decimal("-100"),
                unit="kg",
            )


class TestGeolocationReferenceModel:
    """Test GeolocationReference model."""

    def test_create_valid_geolocation(self):
        """Test creating a valid GeolocationReference."""
        geo = GeolocationReference(
            plot_id="plot-001",
            latitude=Decimal("4.5709"),
            longitude=Decimal("-74.2973"),
            polygon=None,
            area_hectares=Decimal("2.5"),
            country_code="CO",
        )
        assert geo.plot_id == "plot-001"
        assert geo.latitude == Decimal("4.5709")
        assert geo.longitude == Decimal("-74.2973")
        assert geo.country_code == "CO"

    def test_geolocation_with_polygon(self):
        """Test GeolocationReference with polygon coordinates."""
        polygon = [
            [Decimal("4.5709"), Decimal("-74.2973")],
            [Decimal("4.5710"), Decimal("-74.2974")],
            [Decimal("4.5711"), Decimal("-74.2975")],
        ]
        geo = GeolocationReference(
            plot_id="plot-002",
            latitude=Decimal("4.5710"),
            longitude=Decimal("-74.2973"),
            polygon=polygon,
            area_hectares=Decimal("5.0"),
            country_code="CO",
        )
        assert geo.polygon is not None
        assert len(geo.polygon) == 3

    def test_geolocation_latitude_bounds(self):
        """Test GeolocationReference latitude bounds validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            GeolocationReference(
                plot_id="bad",
                latitude=Decimal("95"),  # Outside [-90, 90]
                longitude=Decimal("0"),
                area_hectares=Decimal("1"),
                country_code="XX",
            )

    def test_geolocation_longitude_bounds(self):
        """Test GeolocationReference longitude bounds validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            GeolocationReference(
                plot_id="bad",
                latitude=Decimal("0"),
                longitude=Decimal("185"),  # Outside [-180, 180]
                area_hectares=Decimal("1"),
                country_code="XX",
            )

    def test_geolocation_country_code_length(self):
        """Test GeolocationReference country code validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            GeolocationReference(
                plot_id="bad",
                latitude=Decimal("0"),
                longitude=Decimal("0"),
                area_hectares=Decimal("1"),
                country_code="X",  # Too short
            )


class TestSupplierReferenceModel:
    """Test SupplierReference model."""

    def test_create_valid_supplier(self):
        """Test creating a valid SupplierReference."""
        supplier = SupplierReference(
            supplier_id="sup-001",
            name="Acme Coffee Co.",
            country="CO",
            registration_number="123456789",
        )
        assert supplier.supplier_id == "sup-001"
        assert supplier.name == "Acme Coffee Co."
        assert supplier.country == "CO"

    def test_supplier_defaults(self):
        """Test SupplierReference default values."""
        supplier = SupplierReference(
            supplier_id="sup-002",
            name="Test Supplier",
            country="BR",
        )
        assert supplier.registration_number == ""


class TestMeasureSummaryModel:
    """Test MeasureSummary model."""

    def test_create_valid_measure_summary(self):
        """Test creating a valid MeasureSummary."""
        summary = MeasureSummary(
            measure_id="msr-001",
            title="Enhanced Audit",
            category="independent_audit",
            status="completed",
            reduction=Decimal("25"),
        )
        assert summary.measure_id == "msr-001"
        assert summary.reduction == Decimal("25")

    def test_measure_summary_defaults(self):
        """Test MeasureSummary default values."""
        summary = MeasureSummary(
            measure_id="msr-002",
            title="Test",
            category="other_measures",
            status="proposed",
        )
        assert summary.reduction == Decimal("0")

    def test_measure_summary_reduction_bounds(self):
        """Test MeasureSummary reduction validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            MeasureSummary(
                measure_id="bad",
                title="Test",
                category="test",
                status="test",
                reduction=Decimal("150"),  # > 100
            )


class TestValidationIssueModel:
    """Test ValidationIssue model."""

    def test_create_valid_issue(self):
        """Test creating a valid ValidationIssue."""
        issue = ValidationIssue(
            field="products",
            severity=ValidationSeverity.ERROR,
            message="Products list cannot be empty",
            article_reference="Article 9(1)(a)",
        )
        assert issue.field == "products"
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.message == "Products list cannot be empty"

    def test_validation_issue_defaults(self):
        """Test ValidationIssue default values."""
        issue = ValidationIssue(
            field="test",
            severity=ValidationSeverity.WARNING,
            message="Warning message",
        )
        assert issue.article_reference == ""


class TestDDSContentModel:
    """Test DDSContent model."""

    def test_create_valid_dds_content(self):
        """Test creating a valid DDSContent."""
        content = DDSContent(
            operator_info={"id": "op-001", "name": "Acme Corp"},
            products=[],
            article9_data={"status": "complete"},
            risk_summary={"score": 72},
            mitigation_summary={"measures": 3},
            conclusion="compliant",
        )
        assert content.conclusion == "compliant"
        assert content.operator_info["id"] == "op-001"

    def test_dds_content_defaults(self):
        """Test DDSContent default values."""
        content = DDSContent()
        assert content.operator_info == {}
        assert content.products == []
        assert content.article9_data == {}
        assert content.risk_summary == {}
        assert content.mitigation_summary == {}
        assert content.conclusion == ""


class TestDDSDocumentModel:
    """Test DDSDocument model."""

    def test_create_valid_dds_document(self):
        """Test creating a valid DDSDocument."""
        doc = DDSDocument(
            dds_id="dds-001",
            reference_number="DDS-2026-00001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
            products=[],
            article9_ref="art9-001",
            risk_assessment_ref="risk-001",
            mitigation_ref="mit-001",
            status=DDSStatus.DRAFT,
            compliance_conclusion="compliant",
        )
        assert doc.dds_id == "dds-001"
        assert doc.commodity == EUDRCommodity.COFFEE
        assert doc.status == DDSStatus.DRAFT

    def test_dds_document_defaults(self):
        """Test DDSDocument default values."""
        doc = DDSDocument(
            dds_id="dds-002",
            reference_number="DDS-2026-00002",
            operator_id="op-002",
            commodity=EUDRCommodity.WOOD,
        )
        assert doc.products == []
        assert doc.article9_ref == ""
        assert doc.risk_assessment_ref == ""
        assert doc.mitigation_ref == ""
        assert doc.status == DDSStatus.DRAFT
        assert doc.compliance_conclusion == ""
        assert doc.submitted_at is None
        assert doc.provenance_hash == ""

    def test_dds_document_generated_at_is_datetime(self):
        """Test DDSDocument generated_at field is datetime."""
        doc = DDSDocument(
            dds_id="dds-003",
            reference_number="DDS-2026-00003",
            operator_id="op-003",
            commodity=EUDRCommodity.COCOA,
        )
        assert isinstance(doc.generated_at, datetime)


class TestArticle9PackageModel:
    """Test Article9Package model."""

    def test_create_valid_article9_package(self):
        """Test creating a valid Article9Package."""
        package = Article9Package(
            package_id="art9-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
            elements={"product_description": "Coffee beans"},
            completeness_score=Decimal("0.95"),
            missing_elements=[],
        )
        assert package.package_id == "art9-001"
        assert package.completeness_score == Decimal("0.95")

    def test_article9_package_defaults(self):
        """Test Article9Package default values."""
        package = Article9Package(
            package_id="art9-002",
            operator_id="op-002",
            commodity=EUDRCommodity.SOYA,
        )
        assert package.elements == {}
        assert package.completeness_score == Decimal("0")
        assert package.missing_elements == []

    def test_article9_package_completeness_bounds(self):
        """Test Article9Package completeness score validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            Article9Package(
                package_id="bad",
                operator_id="op",
                commodity=EUDRCommodity.COFFEE,
                completeness_score=Decimal("1.5"),  # > 1.0
            )


class TestRiskAssessmentDocModel:
    """Test RiskAssessmentDoc model."""

    def test_create_valid_risk_assessment_doc(self):
        """Test creating a valid RiskAssessmentDoc."""
        doc = RiskAssessmentDoc(
            doc_id="risk-001",
            assessment_id="assess-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
            composite_score=Decimal("72"),
            risk_level=RiskLevel.HIGH,
            criterion_evaluations=[],
            country_benchmark="high_risk",
            simplified_dd_eligible=False,
        )
        assert doc.composite_score == Decimal("72")
        assert doc.risk_level == RiskLevel.HIGH

    def test_risk_assessment_doc_defaults(self):
        """Test RiskAssessmentDoc default values."""
        doc = RiskAssessmentDoc(
            doc_id="risk-002",
            assessment_id="assess-002",
            operator_id="op-002",
            commodity=EUDRCommodity.WOOD,
            composite_score=Decimal("30"),
            risk_level=RiskLevel.LOW,
        )
        assert doc.criterion_evaluations == []
        assert doc.country_benchmark == ""
        assert doc.simplified_dd_eligible is False

    def test_risk_assessment_doc_score_bounds(self):
        """Test RiskAssessmentDoc score validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            RiskAssessmentDoc(
                doc_id="bad",
                assessment_id="assess",
                operator_id="op",
                commodity=EUDRCommodity.COFFEE,
                composite_score=Decimal("150"),  # > 100
                risk_level=RiskLevel.HIGH,
            )


class TestMitigationDocModel:
    """Test MitigationDoc model."""

    def test_create_valid_mitigation_doc(self):
        """Test creating a valid MitigationDoc."""
        doc = MitigationDoc(
            doc_id="mit-001",
            strategy_id="stg-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
            pre_score=Decimal("72"),
            post_score=Decimal("28"),
            measures_summary=[],
            verification_result="sufficient",
        )
        assert doc.pre_score == Decimal("72")
        assert doc.post_score == Decimal("28")

    def test_mitigation_doc_defaults(self):
        """Test MitigationDoc default values."""
        doc = MitigationDoc(
            doc_id="mit-002",
            strategy_id="stg-002",
            operator_id="op-002",
            commodity=EUDRCommodity.RUBBER,
            pre_score=Decimal("50"),
            post_score=Decimal("25"),
        )
        assert doc.measures_summary == []
        assert doc.verification_result is None


class TestCompliancePackageModel:
    """Test CompliancePackage model."""

    def test_create_valid_compliance_package(self):
        """Test creating a valid CompliancePackage."""
        package = CompliancePackage(
            package_id="pkg-001",
            dds_id="dds-001",
            operator_id="op-001",
            commodity=EUDRCommodity.COFFEE,
            sections={},
            table_of_contents=[],
            cross_references={},
        )
        assert package.package_id == "pkg-001"
        assert package.dds_id == "dds-001"

    def test_compliance_package_defaults(self):
        """Test CompliancePackage default values."""
        package = CompliancePackage(
            package_id="pkg-002",
            dds_id="dds-002",
            operator_id="op-002",
            commodity=EUDRCommodity.PALM_OIL,
        )
        assert package.sections == {}
        assert package.table_of_contents == []
        assert package.cross_references == {}
        assert package.provenance_hash == ""


class TestDocumentVersionModel:
    """Test DocumentVersion model."""

    def test_create_valid_document_version(self):
        """Test creating a valid DocumentVersion."""
        version = DocumentVersion(
            version_id="ver-001",
            document_id="dds-001",
            document_type=DocumentType.DDS,
            version_number=1,
            status=VersionStatus.FINAL,
            content_hash="abc123",
            created_by="agent-030",
            amendment_reason="Initial version",
        )
        assert version.version_number == 1
        assert version.status == VersionStatus.FINAL

    def test_document_version_defaults(self):
        """Test DocumentVersion default values."""
        version = DocumentVersion(
            version_id="ver-002",
            document_id="dds-002",
            document_type=DocumentType.DDS,
            version_number=2,
            content_hash="def456",
        )
        assert version.status == VersionStatus.DRAFT
        assert version.created_by == AGENT_ID
        assert version.amendment_reason is None

    def test_document_version_number_validation(self):
        """Test DocumentVersion version number validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            DocumentVersion(
                version_id="bad",
                document_id="doc",
                document_type=DocumentType.DDS,
                version_number=0,  # Must be >= 1
                content_hash="hash",
            )


class TestSubmissionRecordModel:
    """Test SubmissionRecord model."""

    def test_create_valid_submission_record(self):
        """Test creating a valid SubmissionRecord."""
        record = SubmissionRecord(
            submission_id="sub-001",
            dds_id="dds-001",
            status=SubmissionStatus.PENDING,
        )
        assert record.submission_id == "sub-001"
        assert record.status == SubmissionStatus.PENDING

    def test_submission_record_defaults(self):
        """Test SubmissionRecord default values."""
        record = SubmissionRecord(
            submission_id="sub-002",
            dds_id="dds-002",
        )
        assert record.status == SubmissionStatus.PENDING
        assert record.submitted_at is None
        assert record.acknowledged_at is None
        assert record.rejected_at is None
        assert record.rejection_reason is None
        assert record.receipt_number is None
        assert record.resubmission_count == 0


class TestValidationResultModel:
    """Test ValidationResult model."""

    def test_create_valid_validation_result(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(
            validation_id="val-001",
            document_id="dds-001",
            document_type=DocumentType.DDS,
            is_valid=True,
            errors=[],
            warnings=[],
        )
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validation_result_with_errors(self):
        """Test ValidationResult with validation errors."""
        issue = ValidationIssue(
            field="products",
            severity=ValidationSeverity.ERROR,
            message="Products required",
        )
        result = ValidationResult(
            validation_id="val-002",
            document_id="dds-002",
            document_type=DocumentType.DDS,
            is_valid=False,
            errors=[issue],
        )
        assert result.is_valid is False
        assert len(result.errors) == 1


class TestHealthStatusModel:
    """Test HealthStatus model."""

    def test_health_status_defaults(self):
        """Test HealthStatus default values."""
        health = HealthStatus()
        assert health.agent_id == AGENT_ID
        assert health.status == "healthy"
        assert health.version == AGENT_VERSION
        assert health.database is False
        assert health.redis is False
        assert health.uptime_seconds == 0.0

    def test_health_status_custom_values(self):
        """Test HealthStatus with custom values."""
        health = HealthStatus(
            status="degraded",
            database=True,
            redis=True,
            uptime_seconds=3600.0,
            engines={"dds_generator": "ok", "article9_assembler": "ok"},
        )
        assert health.status == "degraded"
        assert health.database is True
        assert health.uptime_seconds == 3600.0

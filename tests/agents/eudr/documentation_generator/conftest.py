# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-030 Documentation Generator tests.

Provides reusable test fixtures for config, models, DDS documents,
Article 9 packages, risk/mitigation documentation, and compliance
packages across all test modules.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.documentation_generator.config import (
    DocumentationGeneratorConfig,
    reset_config,
)
from greenlang.agents.eudr.documentation_generator.models import (
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
from greenlang.agents.eudr.documentation_generator.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


# ---------------------------------------------------------------------------
# Auto-reset config singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config() -> DocumentationGeneratorConfig:
    """Create a default DocumentationGeneratorConfig instance."""
    return DocumentationGeneratorConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Model fixtures - Product & Geolocation
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_product() -> ProductEntry:
    """Create a sample ProductEntry for testing."""
    return ProductEntry(
        description="Premium Arabica Coffee Beans",
        quantity="1000 kg",
        hs_code="0901.11",
        trade_name="Colombian Supremo",
        scientific_name="Coffea arabica",
    )


@pytest.fixture
def sample_geolocation() -> GeolocationReference:
    """Create a sample GeolocationReference for testing."""
    return GeolocationReference(
        latitude=4.5709,
        longitude=-74.2973,
        country_code="CO",
        plot_id="CO-ANT-001",
        region="Antioquia",
    )


@pytest.fixture
def sample_supplier() -> SupplierReference:
    """Create a sample SupplierReference for testing."""
    return SupplierReference(
        supplier_id="SUP-CO-001",
        name="Cafe Verde S.A.",
        tier=1,
        country_code="CO",
        certification="Rainforest Alliance",
    )


# ---------------------------------------------------------------------------
# Model fixtures - Article 9 Package
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_article9_package(
    sample_product: ProductEntry,
    sample_geolocation: GeolocationReference,
    sample_supplier: SupplierReference,
) -> Article9Package:
    """Create a sample Article9Package for testing."""
    return Article9Package(
        package_id="art9-test-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        products=[sample_product],
        geolocations=[sample_geolocation],
        suppliers=[sample_supplier],
        production_date="2025-06-15T00:00:00Z",
        completeness_score=Decimal("100.0"),
        missing_elements=[],
    )


# ---------------------------------------------------------------------------
# Model fixtures - Risk Assessment Documentation
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_risk_assessment_doc() -> RiskAssessmentDoc:
    """Create a sample RiskAssessmentDoc for testing."""
    return RiskAssessmentDoc(
        doc_id="riskdoc-test-001",
        assessment_id="assess-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score="72",
        risk_level=RiskLevel.HIGH,
        contributing_factors=[
            {
                "dimension": "country",
                "score": "65",
                "weight": "0.25",
            },
            {
                "dimension": "supplier",
                "score": "80",
                "weight": "0.30",
            },
        ],
        regulatory_references=["EUDR Art. 10(2)(a)", "EUDR Art. 10(2)(b)"],
    )


# ---------------------------------------------------------------------------
# Model fixtures - Mitigation Documentation
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_measure_summary() -> MeasureSummary:
    """Create a sample MeasureSummary for testing."""
    return MeasureSummary(
        measure_id="msr-test-001",
        title="Enhanced Supplier Audit",
        category="independent_audit",
        expected_reduction="25",
        status="completed",
    )


@pytest.fixture
def sample_mitigation_doc(
    sample_measure_summary: MeasureSummary,
) -> MitigationDoc:
    """Create a sample MitigationDoc for testing."""
    return MitigationDoc(
        doc_id="mitdoc-test-001",
        strategy_id="stg-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        pre_mitigation_score="72",
        post_mitigation_score="28",
        risk_reduction="44",
        measures=[sample_measure_summary],
        verification_status="verified",
        article11_compliant=True,
    )


# ---------------------------------------------------------------------------
# Model fixtures - DDS Document
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_dds_content(
    sample_article9_package: Article9Package,
    sample_risk_assessment_doc: RiskAssessmentDoc,
    sample_mitigation_doc: MitigationDoc,
) -> DDSContent:
    """Create a sample DDSContent for testing."""
    return DDSContent(
        article9_data=sample_article9_package,
        risk_documentation=sample_risk_assessment_doc,
        mitigation_documentation=sample_mitigation_doc,
        compliance_conclusion="deforestation_free_compliant",
    )


@pytest.fixture
def sample_dds_document(
    sample_dds_content: DDSContent,
) -> DDSDocument:
    """Create a sample DDSDocument for testing."""
    return DDSDocument(
        dds_id="dds-test-001",
        operator_id="operator-001",
        commodity=EUDRCommodity.COFFEE,
        reference_number="OP001-2026-0001",
        content=sample_dds_content,
        status=DDSStatus.DRAFT,
        provenance_hash="abc123def456",
    )


# ---------------------------------------------------------------------------
# Model fixtures - Compliance Package
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_compliance_package(
    sample_dds_document: DDSDocument,
) -> CompliancePackage:
    """Create a sample CompliancePackage for testing."""
    return CompliancePackage(
        package_id="pkg-test-001",
        dds_id="dds-test-001",
        dds_document=sample_dds_document,
        format=PackageFormat.JSON,
        sections={
            ComplianceSection.DDS: "dds-content-hash",
            ComplianceSection.ARTICLE9: "art9-content-hash",
            ComplianceSection.RISK: "risk-content-hash",
            ComplianceSection.MITIGATION: "mitigation-content-hash",
        },
        package_hash="package-integrity-hash",
    )


# ---------------------------------------------------------------------------
# Model fixtures - Submission & Validation
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_submission_record() -> SubmissionRecord:
    """Create a sample SubmissionRecord for testing."""
    return SubmissionRecord(
        submission_id="sub-test-001",
        dds_id="dds-test-001",
        status=SubmissionStatus.PENDING,
        submitted_at=datetime.now(tz=timezone.utc),
        eu_reference_number=None,
        retry_count=0,
    )


@pytest.fixture
def sample_validation_issue() -> ValidationIssue:
    """Create a sample ValidationIssue for testing."""
    return ValidationIssue(
        field="article9_data.products",
        severity=ValidationSeverity.ERROR,
        message="At least one product entry is required",
        code="MISSING_PRODUCTS",
    )


@pytest.fixture
def sample_validation_result(
    sample_validation_issue: ValidationIssue,
) -> ValidationResult:
    """Create a sample ValidationResult for testing."""
    return ValidationResult(
        dds_id="dds-test-001",
        passed=False,
        issues=[sample_validation_issue],
        validated_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Model fixtures - Document Versioning
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document_version() -> DocumentVersion:
    """Create a sample DocumentVersion for testing."""
    return DocumentVersion(
        version_id="ver-test-001",
        document_id="dds-test-001",
        version_number=1,
        status=VersionStatus.FINAL,
        created_by="operator-001",
        change_summary="Initial DDS creation",
        content_hash="abc123def456",
    )


# ---------------------------------------------------------------------------
# Fixture collections
# ---------------------------------------------------------------------------

@pytest.fixture
def multiple_products(
    sample_product: ProductEntry,
) -> List[ProductEntry]:
    """Provide multiple ProductEntry instances for testing."""
    return [
        sample_product,
        ProductEntry(
            description="Organic Cocoa Beans",
            quantity="500 kg",
            hs_code="1801.00",
            trade_name="Ghana Premium",
            scientific_name="Theobroma cacao",
        ),
    ]


@pytest.fixture
def multiple_geolocations(
    sample_geolocation: GeolocationReference,
) -> List[GeolocationReference]:
    """Provide multiple GeolocationReference instances for testing."""
    return [
        sample_geolocation,
        GeolocationReference(
            latitude=6.6666,
            longitude=-1.6163,
            country_code="GH",
            plot_id="GH-ASH-001",
            region="Ashanti",
        ),
    ]


@pytest.fixture
def multiple_suppliers(
    sample_supplier: SupplierReference,
) -> List[SupplierReference]:
    """Provide multiple SupplierReference instances for testing."""
    return [
        sample_supplier,
        SupplierReference(
            supplier_id="SUP-GH-001",
            name="Ghana Cocoa Co.",
            tier=1,
            country_code="GH",
            certification="Fairtrade",
        ),
    ]

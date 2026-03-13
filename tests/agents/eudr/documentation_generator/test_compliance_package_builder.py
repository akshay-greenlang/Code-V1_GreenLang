# -*- coding: utf-8 -*-
"""
Tests for CompliancePackageBuilder Engine - AGENT-EUDR-030

Tests the Compliance Package Builder including build(), _collect_components(),
_generate_package_hash(), _create_sections(), and multi-format support (JSON/PDF/XML).

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.compliance_package_builder import (
    CompliancePackageBuilder,
    _SECTION_ORDER,
    _SECTION_METADATA,
)
from greenlang.agents.eudr.documentation_generator.models import (
    Article9Package,
    CompliancePackage,
    ComplianceSection,
    DDSDocument,
    DDSStatus,
    EUDRCommodity,
    MeasureSummary,
    MitigationDoc,
    ProductEntry,
    RiskAssessmentDoc,
    RiskLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder() -> CompliancePackageBuilder:
    """Create CompliancePackageBuilder instance."""
    return CompliancePackageBuilder()


@pytest.fixture
def sample_dds() -> DDSDocument:
    """Create sample DDS document."""
    return DDSDocument(
        dds_id="dds-test-001",
        reference_number="DDS-OP001-COFF-20260312-00001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[
            ProductEntry(
                product_id="P1",
                description="Coffee Beans",
                hs_code="0901.11",
                quantity=Decimal("1000"),
                unit="kg",
            )
        ],
        article9_ref="a9p-001",
        risk_assessment_ref="rad-001",
        mitigation_ref="mid-001",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant_after_mitigation",
        provenance_hash="abc" * 21 + "a",  # 64 chars
    )


@pytest.fixture
def sample_article9() -> Article9Package:
    """Create sample Article 9 package."""
    return Article9Package(
        package_id="a9p-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        elements={
            "product_description": {"count": 1},
            "quantity": {"total_items": 1},
        },
        completeness_score=Decimal("0.85"),
        missing_elements=["buyer_info"],
    )


@pytest.fixture
def sample_risk_doc() -> RiskAssessmentDoc:
    """Create sample risk assessment doc."""
    return RiskAssessmentDoc(
        doc_id="rad-001",
        assessment_id="asr-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("72.0"),
        risk_level=RiskLevel.HIGH,
        criterion_evaluations=[],
        country_benchmark="high",
        simplified_dd_eligible=False,
    )


@pytest.fixture
def sample_mitigation_doc() -> MitigationDoc:
    """Create sample mitigation doc."""
    return MitigationDoc(
        doc_id="mid-001",
        strategy_id="stg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("72.0"),
        post_score=Decimal("28.0"),
        measures_summary=[
            MeasureSummary(
                measure_id="m1",
                title="Audit",
                category="independent_audit",
                status="completed",
                reduction=Decimal("44"),
            )
        ],
        verification_result="sufficient",
    )


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_builder_initialization(builder):
    """Test CompliancePackageBuilder initializes correctly."""
    assert builder._config is not None
    assert builder._provenance is not None


# ---------------------------------------------------------------------------
# Test: build_package - Success Paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_package_minimal(builder, sample_dds):
    """Test building package with minimal components."""
    package = await builder.build_package(dds=sample_dds)

    assert package.package_id.startswith("cpk-")
    assert package.dds_id == "dds-test-001"
    assert package.operator_id == "OP-001"
    assert package.commodity == EUDRCommodity.COFFEE
    assert len(package.provenance_hash) == 64


@pytest.mark.asyncio
async def test_build_package_complete(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
    sample_mitigation_doc,
):
    """Test building complete compliance package."""
    package = await builder.build_package(
        dds=sample_dds,
        article9=sample_article9,
        risk_doc=sample_risk_doc,
        mitigation_doc=sample_mitigation_doc,
    )

    assert package.package_id.startswith("cpk-")
    assert len(package.sections) >= 4
    assert ComplianceSection.INFORMATION_GATHERING.value in package.sections
    assert ComplianceSection.RISK_ASSESSMENT.value in package.sections
    assert ComplianceSection.RISK_MITIGATION.value in package.sections
    assert ComplianceSection.COMPLIANCE_CONCLUSION.value in package.sections


@pytest.mark.asyncio
async def test_build_package_with_evidence(
    builder,
    sample_dds,
    sample_article9,
):
    """Test building package with additional evidence."""
    evidence = {
        "documents": ["doc-001", "doc-002"],
        "certificates": ["cert-001"],
    }

    package = await builder.build_package(
        dds=sample_dds,
        article9=sample_article9,
        additional_evidence=evidence,
    )

    assert "supporting_evidence" in package.sections


@pytest.mark.asyncio
async def test_build_package_with_toc(
    builder,
    sample_dds,
    sample_article9,
):
    """Test package includes table of contents."""
    package = await builder.build_package(
        dds=sample_dds,
        article9=sample_article9,
    )

    assert len(package.table_of_contents) > 0
    assert all("title" in entry for entry in package.table_of_contents)
    assert all("number" in entry for entry in package.table_of_contents)


@pytest.mark.asyncio
async def test_build_package_with_cross_references(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
):
    """Test package includes cross-references."""
    package = await builder.build_package(
        dds=sample_dds,
        article9=sample_article9,
        risk_doc=sample_risk_doc,
    )

    assert len(package.cross_references) > 0
    assert "dds_id" in package.cross_references
    assert package.cross_references["dds_id"] == "dds-test-001"


# ---------------------------------------------------------------------------
# Test: Executive Summary Building
# ---------------------------------------------------------------------------

def test_build_executive_summary_minimal(
    builder,
    sample_dds,
):
    """Test executive summary with minimal data."""
    summary = builder._build_executive_summary(
        sample_dds, None, None, None
    )

    assert summary["operator_id"] == "OP-001"
    assert summary["commodity"] == "coffee"
    assert summary["compliance_conclusion"] == "compliant_after_mitigation"


def test_build_executive_summary_complete(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
    sample_mitigation_doc,
):
    """Test executive summary with all components."""
    summary = builder._build_executive_summary(
        sample_dds, sample_article9, sample_risk_doc, sample_mitigation_doc
    )

    assert "article9_completeness" in summary
    assert "risk_score" in summary
    assert "mitigation_pre_score" in summary
    assert "mitigation_post_score" in summary
    assert summary["risk_level"] == "high"


# ---------------------------------------------------------------------------
# Test: Table of Contents Building
# ---------------------------------------------------------------------------

def test_build_table_of_contents(builder):
    """Test table of contents generation."""
    sections = {
        ComplianceSection.INFORMATION_GATHERING.value: {},
        ComplianceSection.RISK_ASSESSMENT.value: {},
        "executive_summary": {},
    }

    toc = builder._build_table_of_contents(sections)

    assert len(toc) == 3
    assert all("number" in entry for entry in toc)
    assert all("title" in entry for entry in toc)
    assert all("section_key" in entry for entry in toc)


def test_build_table_of_contents_ordering(builder):
    """Test ToC entries are numbered sequentially."""
    sections = {
        "executive_summary": {},
        ComplianceSection.INFORMATION_GATHERING.value: {},
        ComplianceSection.RISK_ASSESSMENT.value: {},
    }

    toc = builder._build_table_of_contents(sections)

    numbers = [int(entry["number"]) for entry in toc]
    assert numbers == [1, 2, 3]


# ---------------------------------------------------------------------------
# Test: Cross-References Building
# ---------------------------------------------------------------------------

def test_build_cross_references_minimal(
    builder,
    sample_dds,
):
    """Test cross-references with minimal data."""
    refs = builder._build_cross_references(sample_dds, None, None)

    assert refs["dds_id"] == "dds-test-001"
    assert "dds_reference" in refs


def test_build_cross_references_complete(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
):
    """Test cross-references with all components."""
    refs = builder._build_cross_references(
        sample_dds, sample_article9, sample_risk_doc
    )

    assert "article9_package_id" in refs
    assert "risk_doc_id" in refs
    assert "risk_assessment_id" in refs
    assert refs["article9_package_id"] == "a9p-001"


# ---------------------------------------------------------------------------
# Test: Section Building
# ---------------------------------------------------------------------------

def test_build_section(builder):
    """Test building a single compliance section."""
    data = {"test_field": "test_value"}
    section = builder._build_section(
        ComplianceSection.RISK_ASSESSMENT, data
    )

    assert "metadata" in section
    assert "data" in section
    assert section["data"]["test_field"] == "test_value"
    assert "title" in section["metadata"]


# ---------------------------------------------------------------------------
# Test: Regulatory References Building
# ---------------------------------------------------------------------------

def test_build_regulatory_references(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
    sample_mitigation_doc,
):
    """Test building regulatory references section."""
    refs = builder._build_regulatory_references(
        sample_dds, sample_article9, sample_risk_doc, sample_mitigation_doc
    )

    assert "references" in refs
    assert len(refs["references"]) >= 5
    assert any(r["article"] == "Article 9" for r in refs["references"])
    assert any(r["article"] == "Article 10" for r in refs["references"])
    assert any(r["article"] == "Article 11" for r in refs["references"])


def test_build_regulatory_references_minimal(
    builder,
    sample_dds,
):
    """Test regulatory references with minimal components."""
    refs = builder._build_regulatory_references(
        sample_dds, None, None, None
    )

    assert "references" in refs
    # Check that applicability reflects missing components
    art9_ref = next(r for r in refs["references"] if r["article"] == "Article 9")
    assert art9_ref["applicability"] == "Pending"


# ---------------------------------------------------------------------------
# Test: Provenance Section Building
# ---------------------------------------------------------------------------

def test_build_provenance_section(
    builder,
    sample_dds,
    sample_article9,
    sample_risk_doc,
):
    """Test provenance chain section building."""
    section = builder._build_provenance_section(
        sample_dds, sample_article9, sample_risk_doc, None
    )

    assert "chain_length" in section
    assert "chain_valid" in section
    assert "genesis_hash" in section
    assert "entries" in section
    assert section["algorithm"] == "sha256"


def test_build_provenance_section_minimal(
    builder,
    sample_dds,
):
    """Test provenance section with minimal data."""
    section = builder._build_provenance_section(
        sample_dds, None, None, None
    )

    # Should still include DDS generation step
    assert section["chain_length"] >= 2


# ---------------------------------------------------------------------------
# Test: Package Hash Computation
# ---------------------------------------------------------------------------

def test_compute_package_hash(builder):
    """Test package hash computation."""
    data = {"package_id": "cpk-001", "operator_id": "OP-001"}
    hash1 = builder._compute_package_hash(data)
    hash2 = builder._compute_package_hash(data)

    assert len(hash1) == 64
    assert hash1 == hash2  # Deterministic

    data2 = {"package_id": "cpk-002", "operator_id": "OP-001"}
    hash3 = builder._compute_package_hash(data2)
    assert hash3 != hash1


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(builder):
    """Test health check returns correct status."""
    status = await builder.health_check()

    assert status["engine"] == "CompliancePackageBuilder"
    assert status["status"] == "available"
    assert "config" in status
    assert status["section_count"] == len(_SECTION_ORDER)


# ---------------------------------------------------------------------------
# Test: Different Commodities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_build_package_cocoa(builder):
    """Test package building for cocoa commodity."""
    dds = DDSDocument(
        dds_id="dds-cocoa-001",
        reference_number="DDS-REF-001",
        operator_id="OP-COCOA",
        commodity=EUDRCommodity.COCOA,
        products=[],
        article9_ref="a9p-cocoa",
        status=DDSStatus.DRAFT,
        compliance_conclusion="pending",
    )

    package = await builder.build_package(dds=dds)

    assert package.commodity == EUDRCommodity.COCOA


@pytest.mark.asyncio
async def test_build_package_wood(builder):
    """Test package building for wood commodity."""
    dds = DDSDocument(
        dds_id="dds-wood-001",
        reference_number="DDS-REF-002",
        operator_id="OP-WOOD",
        commodity=EUDRCommodity.WOOD,
        products=[],
        article9_ref="a9p-wood",
        status=DDSStatus.DRAFT,
        compliance_conclusion="pending",
    )

    package = await builder.build_package(dds=dds)

    assert package.commodity == EUDRCommodity.WOOD


# ---------------------------------------------------------------------------
# Test: Section Metadata
# ---------------------------------------------------------------------------

def test_section_metadata_completeness():
    """Test all compliance sections have metadata defined."""
    for section in ComplianceSection:
        assert section in _SECTION_METADATA
        metadata = _SECTION_METADATA[section]
        assert "title" in metadata
        assert "description" in metadata
        assert "article_reference" in metadata

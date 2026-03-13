# -*- coding: utf-8 -*-
"""
Tests for DDSStatementGenerator Engine - AGENT-EUDR-030

Tests the DDS Statement Generator including generate_dds(),
validate_dds_content(), _build_dds_content(), and integration
with Article9DataAssembler, RiskAssessmentDocumenter, and
MitigationDocumenter.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.dds_statement_generator import (
    DDSStatementGenerator,
    _COMPLIANCE_CONCLUSIONS,
    _DDS_MANDATORY_FIELDS,
    _COMMODITY_HS_PREFIXES,
)
from greenlang.agents.eudr.documentation_generator.models import (
    Article9Package,
    DDSDocument,
    DDSStatus,
    EUDRCommodity,
    MitigationDoc,
    ProductEntry,
    RiskAssessmentDoc,
    RiskLevel,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dds_generator() -> DDSStatementGenerator:
    """Create DDSStatementGenerator instance."""
    return DDSStatementGenerator()


@pytest.fixture
def sample_products() -> list[ProductEntry]:
    """Create sample product entries."""
    return [
        ProductEntry(
            product_id="PROD-001",
            description="Arabica Coffee Beans",
            hs_code="0901.11",
            cn_code="09011100",
            quantity=Decimal("1000"),
            unit="kg",
        ),
        ProductEntry(
            product_id="PROD-002",
            description="Robusta Coffee Beans",
            hs_code="0901.12",
            cn_code="09011200",
            quantity=Decimal("500"),
            unit="kg",
        ),
    ]


@pytest.fixture
def sample_article9_basic() -> Article9Package:
    """Create basic Article 9 package."""
    return Article9Package(
        package_id="a9p-test-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        elements={
            "product_description": {"count": 2},
            "quantity": {"total_items": 2},
            "country_of_production": {"countries": ["CO"]},
            "geolocation": {"total_plots": 3},
        },
        completeness_score=Decimal("0.80"),
        missing_elements=["buyer_info", "certifications"],
    )


@pytest.fixture
def sample_risk_doc_negligible() -> RiskAssessmentDoc:
    """Create risk doc with negligible risk."""
    return RiskAssessmentDoc(
        doc_id="rad-test-001",
        assessment_id="asr-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("10.5"),
        risk_level=RiskLevel.NEGLIGIBLE,
        criterion_evaluations=[],
        country_benchmark="low",
        simplified_dd_eligible=True,
    )


@pytest.fixture
def sample_risk_doc_high() -> RiskAssessmentDoc:
    """Create risk doc with high risk."""
    return RiskAssessmentDoc(
        doc_id="rad-test-002",
        assessment_id="asr-002",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("72.0"),
        risk_level=RiskLevel.HIGH,
        criterion_evaluations=[
            {"criterion_id": "10.2.a", "score": "75", "assessment": "high_risk"}
        ],
        country_benchmark="high",
        simplified_dd_eligible=False,
    )


@pytest.fixture
def sample_mitigation_doc() -> MitigationDoc:
    """Create mitigation documentation."""
    from greenlang.agents.eudr.documentation_generator.models import MeasureSummary

    return MitigationDoc(
        doc_id="mid-test-001",
        strategy_id="stg-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("72.0"),
        post_score=Decimal("25.0"),
        measures_summary=[
            MeasureSummary(
                measure_id="msr-001",
                title="Enhanced Audit",
                category="independent_audit",
                status="completed",
                reduction=Decimal("47.0"),
            )
        ],
        verification_result="sufficient",
    )


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_generator_initialization(dds_generator):
    """Test DDSStatementGenerator initializes with correct defaults."""
    assert dds_generator._config is not None
    assert dds_generator._provenance is not None
    assert dds_generator._sequence == 0


def test_generator_initialization_with_custom_config():
    """Test initialization with custom configuration."""
    from greenlang.agents.eudr.documentation_generator.config import (
        DocumentationGeneratorConfig,
    )

    config = DocumentationGeneratorConfig(
        dds_reference_prefix="TEST-DDS",
        max_products_per_dds=50,
    )
    generator = DDSStatementGenerator(config=config)
    assert generator._config.dds_reference_prefix == "TEST-DDS"
    assert generator._config.max_products_per_dds == 50


# ---------------------------------------------------------------------------
# Test: generate_dds - Success Paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_dds_minimal(
    dds_generator,
    sample_products,
    sample_article9_basic,
):
    """Test generating DDS with minimal required inputs."""
    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
    )

    assert dds.dds_id.startswith("dds-")
    assert dds.reference_number.startswith("DDS-")
    assert dds.operator_id == "OP-001"
    assert dds.commodity == EUDRCommodity.COFFEE
    assert len(dds.products) == 2
    assert dds.article9_ref == "a9p-test-001"
    assert dds.status == DDSStatus.DRAFT
    assert dds.compliance_conclusion == "pending"
    assert len(dds.provenance_hash) == 64


@pytest.mark.asyncio
async def test_generate_dds_with_negligible_risk(
    dds_generator,
    sample_products,
    sample_article9_basic,
    sample_risk_doc_negligible,
):
    """Test DDS generation with negligible risk leads to compliant conclusion."""
    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
        risk_doc=sample_risk_doc_negligible,
    )

    assert dds.compliance_conclusion == "compliant"
    assert dds.risk_assessment_ref == "rad-test-001"
    assert "negligible" in dds.compliance_conclusion.lower() or dds.compliance_conclusion == "compliant"


@pytest.mark.asyncio
async def test_generate_dds_with_standard_risk(
    dds_generator,
    sample_products,
    sample_article9_basic,
):
    """Test DDS with standard risk leads to monitoring conclusion."""
    risk_doc = RiskAssessmentDoc(
        doc_id="rad-standard",
        assessment_id="asr-std",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("45.0"),
        risk_level=RiskLevel.STANDARD,
        criterion_evaluations=[],
        country_benchmark="standard",
        simplified_dd_eligible=False,
    )

    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
        risk_doc=risk_doc,
    )

    assert dds.compliance_conclusion == "compliant_standard_monitoring"


@pytest.mark.asyncio
async def test_generate_dds_with_high_risk_and_sufficient_mitigation(
    dds_generator,
    sample_products,
    sample_article9_basic,
    sample_risk_doc_high,
    sample_mitigation_doc,
):
    """Test high risk with sufficient mitigation leads to compliant conclusion."""
    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
        risk_doc=sample_risk_doc_high,
        mitigation_doc=sample_mitigation_doc,
    )

    assert dds.compliance_conclusion == "compliant_after_mitigation"
    assert dds.mitigation_ref == "mid-test-001"


@pytest.mark.asyncio
async def test_generate_dds_with_high_risk_no_mitigation(
    dds_generator,
    sample_products,
    sample_article9_basic,
    sample_risk_doc_high,
):
    """Test high risk without mitigation leads to non-compliant conclusion."""
    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
        risk_doc=sample_risk_doc_high,
    )

    assert dds.compliance_conclusion == "non_compliant"


@pytest.mark.asyncio
async def test_generate_dds_with_critical_risk(
    dds_generator,
    sample_products,
    sample_article9_basic,
):
    """Test critical risk level handling."""
    risk_doc = RiskAssessmentDoc(
        doc_id="rad-critical",
        assessment_id="asr-crit",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        composite_score=Decimal("92.0"),
        risk_level=RiskLevel.CRITICAL,
        criterion_evaluations=[],
        country_benchmark="high",
        simplified_dd_eligible=False,
    )

    dds = await dds_generator.generate_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=sample_products,
        article9_package=sample_article9_basic,
        risk_doc=risk_doc,
    )

    assert dds.compliance_conclusion == "non_compliant"


# ---------------------------------------------------------------------------
# Test: generate_dds - Validation Errors
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_dds_missing_operator_id(
    dds_generator,
    sample_products,
    sample_article9_basic,
):
    """Test DDS generation fails with missing operator_id."""
    with pytest.raises(ValueError, match="operator_id is required"):
        await dds_generator.generate_dds(
            operator_id="",
            commodity=EUDRCommodity.COFFEE,
            products=sample_products,
            article9_package=sample_article9_basic,
        )


@pytest.mark.asyncio
async def test_generate_dds_empty_products(
    dds_generator,
    sample_article9_basic,
):
    """Test DDS generation fails with empty product list."""
    with pytest.raises(ValueError, match="At least one product entry is required"):
        await dds_generator.generate_dds(
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
            products=[],
            article9_package=sample_article9_basic,
        )


@pytest.mark.asyncio
async def test_generate_dds_missing_article9(
    dds_generator,
    sample_products,
):
    """Test DDS generation fails without Article 9 package."""
    with pytest.raises(ValueError, match="Article 9 package is required"):
        await dds_generator.generate_dds(
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
            products=sample_products,
            article9_package=None,
        )


@pytest.mark.asyncio
async def test_generate_dds_exceeds_max_products(
    dds_generator,
    sample_article9_basic,
):
    """Test DDS generation fails when exceeding max products limit."""
    # Create 1001 products (default max is 1000)
    many_products = [
        ProductEntry(
            product_id=f"PROD-{i:04d}",
            description=f"Product {i}",
            hs_code="0901.11",
            quantity=Decimal("10"),
            unit="kg",
        )
        for i in range(1001)
    ]

    with pytest.raises(ValueError, match="exceeds maximum"):
        await dds_generator.generate_dds(
            operator_id="OP-001",
            commodity=EUDRCommodity.COFFEE,
            products=many_products,
            article9_package=sample_article9_basic,
        )


# ---------------------------------------------------------------------------
# Test: Reference Number Generation
# ---------------------------------------------------------------------------

def test_generate_reference_number(dds_generator):
    """Test reference number generation format."""
    ref1 = dds_generator._generate_reference_number("OP-001", "coffee")
    assert ref1.startswith("DDS-")
    assert "OP001" in ref1
    assert "COFF" in ref1
    assert len(ref1.split("-")) == 5  # DDS-OPID-COMM-DATE-SEQ

    ref2 = dds_generator._generate_reference_number("OP-001", "coffee")
    assert ref1 != ref2  # Sequence should increment


def test_reference_number_sequence_increments(dds_generator):
    """Test reference number sequence increments correctly."""
    ref1 = dds_generator._generate_reference_number("OP-001", "coffee")
    ref2 = dds_generator._generate_reference_number("OP-001", "coffee")
    ref3 = dds_generator._generate_reference_number("OP-001", "coffee")

    seq1 = int(ref1.split("-")[-1])
    seq2 = int(ref2.split("-")[-1])
    seq3 = int(ref3.split("-")[-1])

    assert seq2 == seq1 + 1
    assert seq3 == seq2 + 1


# ---------------------------------------------------------------------------
# Test: Section Building
# ---------------------------------------------------------------------------

def test_build_operator_section(dds_generator):
    """Test operator section building."""
    section = dds_generator._build_operator_section("OP-TEST-001")

    assert section["operator_id"] == "OP-TEST-001"
    assert section["role"] == "operator"
    assert section["registration_type"] == "eudr_registered"
    assert "generated_by" in section
    assert "generated_at" in section


def test_build_products_section(dds_generator, sample_products):
    """Test products section building."""
    section = dds_generator._build_products_section(sample_products)

    assert len(section) == 2
    assert section[0]["index"] == 1
    assert section[0]["product_id"] == "PROD-001"
    assert section[0]["description"] == "Arabica Coffee Beans"
    assert section[0]["hs_code"] == "0901.11"
    assert section[0]["quantity"] == "1000"
    assert section[0]["unit"] == "kg"


def test_build_article9_section(dds_generator, sample_article9_basic):
    """Test Article 9 section building."""
    section = dds_generator._build_article9_section(sample_article9_basic)

    assert section["package_id"] == "a9p-test-001"
    assert section["completeness_score"] == "0.80"
    assert section["elements_present"] == 4
    assert "product_description" in str(section)
    assert section["article_reference"] == "EUDR Article 9"


def test_build_risk_summary_section_no_risk(dds_generator):
    """Test risk summary when no risk doc provided."""
    section = dds_generator._build_risk_summary_section(None)

    assert section["status"] == "not_assessed"
    assert "Risk assessment has not been performed" in section["message"]
    assert section["article_reference"] == "EUDR Article 10"


def test_build_risk_summary_section_with_risk(
    dds_generator,
    sample_risk_doc_negligible,
):
    """Test risk summary section with risk doc."""
    section = dds_generator._build_risk_summary_section(
        sample_risk_doc_negligible
    )

    assert section["doc_id"] == "rad-test-001"
    assert section["assessment_id"] == "asr-001"
    assert section["composite_score"] == "10.5"
    assert section["risk_level"] == "negligible"
    assert section["simplified_dd_eligible"] is True
    assert section["country_benchmark"] == "low"


def test_build_mitigation_summary_section_no_mitigation(dds_generator):
    """Test mitigation summary when not applicable."""
    section = dds_generator._build_mitigation_summary_section(None)

    assert section["status"] == "not_applicable"
    assert "No mitigation measures were required" in section["message"]
    assert section["article_reference"] == "EUDR Article 11"


def test_build_mitigation_summary_section_with_mitigation(
    dds_generator,
    sample_mitigation_doc,
):
    """Test mitigation summary section with mitigation doc."""
    section = dds_generator._build_mitigation_summary_section(
        sample_mitigation_doc
    )

    assert section["doc_id"] == "mid-test-001"
    assert section["strategy_id"] == "stg-001"
    assert section["pre_score"] == "72.0"
    assert section["post_score"] == "25.0"
    assert section["measure_count"] == 1
    assert section["verification_result"] == "sufficient"


# ---------------------------------------------------------------------------
# Test: Compliance Conclusion Logic
# ---------------------------------------------------------------------------

def test_determine_compliance_conclusion_no_risk(dds_generator):
    """Test conclusion when no risk assessment."""
    conclusion = dds_generator._determine_compliance_conclusion(None, None)
    assert conclusion == "pending"


def test_determine_compliance_conclusion_negligible_risk(
    dds_generator,
    sample_risk_doc_negligible,
):
    """Test conclusion with negligible risk."""
    conclusion = dds_generator._determine_compliance_conclusion(
        sample_risk_doc_negligible, None
    )
    assert conclusion == "compliant"


def test_determine_compliance_conclusion_high_with_sufficient_mitigation(
    dds_generator,
    sample_risk_doc_high,
    sample_mitigation_doc,
):
    """Test high risk with sufficient mitigation."""
    conclusion = dds_generator._determine_compliance_conclusion(
        sample_risk_doc_high, sample_mitigation_doc
    )
    assert conclusion == "compliant_after_mitigation"


def test_determine_compliance_conclusion_high_with_partial_mitigation(
    dds_generator,
    sample_risk_doc_high,
):
    """Test high risk with partial mitigation (still high post-score)."""
    from greenlang.agents.eudr.documentation_generator.models import MeasureSummary

    partial_mitigation = MitigationDoc(
        doc_id="mid-partial",
        strategy_id="stg-partial",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        pre_score=Decimal("72.0"),
        post_score=Decimal("50.0"),  # Still too high
        measures_summary=[],
        verification_result="partial",
    )

    conclusion = dds_generator._determine_compliance_conclusion(
        sample_risk_doc_high, partial_mitigation
    )
    assert conclusion == "non_compliant"


def test_determine_compliance_conclusion_high_without_mitigation(
    dds_generator,
    sample_risk_doc_high,
):
    """Test high risk without mitigation."""
    conclusion = dds_generator._determine_compliance_conclusion(
        sample_risk_doc_high, None
    )
    assert conclusion == "non_compliant"


# ---------------------------------------------------------------------------
# Test: DDS Completeness Validation
# ---------------------------------------------------------------------------

def test_validate_dds_completeness_valid(dds_generator):
    """Test validation of complete DDS."""
    dds = DDSDocument(
        dds_id="dds-test-complete",
        reference_number="DDS-TEST-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[
            ProductEntry(
                product_id="P1",
                description="Coffee",
                hs_code="0901.11",
                quantity=Decimal("100"),
                unit="kg",
            )
        ],
        article9_ref="a9p-001",
        risk_assessment_ref="rad-001",
        mitigation_ref="",
        status=DDSStatus.DRAFT,
        compliance_conclusion="compliant",
        provenance_hash="abc123" * 10 + "abcd",  # 64 chars
    )

    result = dds_generator.validate_dds_completeness(dds)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.document_type.value == "dds"


def test_validate_dds_completeness_missing_operator(dds_generator):
    """Test validation fails with missing operator_id."""
    dds = DDSDocument(
        dds_id="dds-test",
        reference_number="DDS-TEST-001",
        operator_id="",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        article9_ref="a9p-001",
    )

    result = dds_generator.validate_dds_completeness(dds)

    assert result.is_valid is False
    assert len(result.errors) > 0
    assert any("Operator ID is required" in e.message for e in result.errors)


def test_validate_dds_completeness_missing_products(dds_generator):
    """Test validation fails with no products."""
    dds = DDSDocument(
        dds_id="dds-test",
        reference_number="DDS-TEST-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        article9_ref="a9p-001",
    )

    result = dds_generator.validate_dds_completeness(dds)

    assert result.is_valid is False
    assert any("product" in e.message.lower() for e in result.errors)


def test_validate_dds_completeness_warnings(dds_generator):
    """Test validation with warnings (not errors)."""
    dds = DDSDocument(
        dds_id="dds-test",
        reference_number="DDS-TEST-001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[
            ProductEntry(
                product_id="P1",
                description="Coffee",
                hs_code="",  # Missing HS code -> warning
                quantity=Decimal("100"),
                unit="kg",
            )
        ],
        article9_ref="a9p-001",
        risk_assessment_ref="",  # Missing -> warning
        compliance_conclusion="pending",
        provenance_hash="",  # Missing -> warning
    )

    result = dds_generator.validate_dds_completeness(dds)

    # Should be valid (no errors, only warnings)
    assert result.is_valid is True
    assert len(result.warnings) > 0


# ---------------------------------------------------------------------------
# Test: HS Code Commodity Matching
# ---------------------------------------------------------------------------

def test_validate_hs_code_commodity_match_valid(dds_generator):
    """Test HS code validation with correct codes."""
    products = [
        ProductEntry(
            product_id="P1",
            description="Coffee",
            hs_code="0901.11",
            quantity=Decimal("100"),
            unit="kg",
        )
    ]

    issues = dds_generator.validate_hs_code_commodity_match(
        products, EUDRCommodity.COFFEE
    )

    assert len(issues) == 0


def test_validate_hs_code_commodity_match_mismatch(dds_generator):
    """Test HS code validation detects commodity mismatch."""
    products = [
        ProductEntry(
            product_id="P1",
            description="Cocoa (wrong for coffee)",
            hs_code="1801.00",  # Cocoa HS code
            quantity=Decimal("100"),
            unit="kg",
        )
    ]

    issues = dds_generator.validate_hs_code_commodity_match(
        products, EUDRCommodity.COFFEE
    )

    assert len(issues) > 0
    assert any("may not match" in i.message for i in issues)
    assert issues[0].severity == ValidationSeverity.WARNING


def test_validate_hs_code_commodity_match_missing_code(dds_generator):
    """Test HS code validation skips products without HS code."""
    products = [
        ProductEntry(
            product_id="P1",
            description="Coffee",
            hs_code="",
            quantity=Decimal("100"),
            unit="kg",
        )
    ]

    issues = dds_generator.validate_hs_code_commodity_match(
        products, EUDRCommodity.COFFEE
    )

    # No issues because we skip empty HS codes
    assert len(issues) == 0


# ---------------------------------------------------------------------------
# Test: Batch DDS Generation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_batch_dds(
    dds_generator,
    sample_article9_basic,
):
    """Test batch DDS generation."""
    batch1 = [
        ProductEntry(
            product_id=f"P{i}",
            description="Coffee",
            hs_code="0901.11",
            quantity=Decimal("100"),
            unit="kg",
        )
        for i in range(10)
    ]
    batch2 = [
        ProductEntry(
            product_id=f"P{i}",
            description="Coffee",
            hs_code="0901.12",
            quantity=Decimal("50"),
            unit="kg",
        )
        for i in range(10, 20)
    ]

    results = await dds_generator.generate_batch_dds(
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        product_batches=[batch1, batch2],
        article9_package=sample_article9_basic,
    )

    assert len(results) == 2
    assert all(dds.dds_id.startswith("dds-") for dds in results)
    assert results[0].dds_id != results[1].dds_id


# ---------------------------------------------------------------------------
# Test: Utility Methods
# ---------------------------------------------------------------------------

def test_get_conclusion_text(dds_generator):
    """Test getting conclusion text from key."""
    text = dds_generator.get_conclusion_text("compliant")
    assert "compliant" in text.lower()
    assert "negligible" in text.lower()

    text_unknown = dds_generator.get_conclusion_text("unknown_key")
    assert "in progress" in text_unknown.lower()


@pytest.mark.asyncio
async def test_health_check(dds_generator):
    """Test health check returns correct status."""
    status = await dds_generator.health_check()

    assert status["engine"] == "DDSStatementGenerator"
    assert status["status"] == "available"
    assert "config" in status
    assert "current_sequence" in status
    assert status["config"]["max_products_per_dds"] > 0


# ---------------------------------------------------------------------------
# Test: Provenance Hash Generation
# ---------------------------------------------------------------------------

def test_compute_dds_hash(dds_generator):
    """Test DDS hash computation."""
    content = {"dds_id": "test", "operator_id": "OP-001"}
    hash1 = dds_generator._compute_dds_hash(content)
    hash2 = dds_generator._compute_dds_hash(content)

    assert len(hash1) == 64
    assert hash1 == hash2  # Deterministic

    content2 = {"dds_id": "test2", "operator_id": "OP-001"}
    hash3 = dds_generator._compute_dds_hash(content2)
    assert hash3 != hash1


# ---------------------------------------------------------------------------
# Test: Integration with Different Commodities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_generate_dds_cocoa_commodity(dds_generator):
    """Test DDS generation for cocoa commodity."""
    products = [
        ProductEntry(
            product_id="COC-001",
            description="Cocoa Beans",
            hs_code="1801.00",
            quantity=Decimal("500"),
            unit="kg",
        )
    ]

    article9 = Article9Package(
        package_id="a9p-cocoa",
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        elements={"product_description": {"count": 1}},
        completeness_score=Decimal("0.70"),
        missing_elements=[],
    )

    dds = await dds_generator.generate_dds(
        operator_id="OP-002",
        commodity=EUDRCommodity.COCOA,
        products=products,
        article9_package=article9,
    )

    assert dds.commodity == EUDRCommodity.COCOA
    assert "COCO" in dds.reference_number


@pytest.mark.asyncio
async def test_generate_dds_wood_commodity(dds_generator):
    """Test DDS generation for wood commodity."""
    products = [
        ProductEntry(
            product_id="WOOD-001",
            description="Timber Logs",
            hs_code="4403.11",
            quantity=Decimal("1000"),
            unit="m3",
        )
    ]

    article9 = Article9Package(
        package_id="a9p-wood",
        operator_id="OP-003",
        commodity=EUDRCommodity.WOOD,
        elements={"product_description": {"count": 1}},
        completeness_score=Decimal("0.75"),
        missing_elements=[],
    )

    dds = await dds_generator.generate_dds(
        operator_id="OP-003",
        commodity=EUDRCommodity.WOOD,
        products=products,
        article9_package=article9,
    )

    assert dds.commodity == EUDRCommodity.WOOD
    assert "WOOD" in dds.reference_number

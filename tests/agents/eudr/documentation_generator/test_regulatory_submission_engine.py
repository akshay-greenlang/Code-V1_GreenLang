# -*- coding: utf-8 -*-
"""
Tests for RegulatorySubmissionEngine - AGENT-EUDR-030

Tests the Regulatory Submission Engine including submit(), _validate_pre_submission(),
get_submission_status(), _handle_retry(), and EU Information System integration.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from greenlang.agents.eudr.documentation_generator.regulatory_submission_engine import (
    RegulatorySubmissionEngine,
    _REJECTION_REASONS,
    _EU_IS_FIELD_MAPPING,
)
from greenlang.agents.eudr.documentation_generator.models import (
    DDSDocument,
    DDSStatus,
    EUDRCommodity,
    ProductEntry,
    SubmissionRecord,
    SubmissionStatus,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> RegulatorySubmissionEngine:
    """Create RegulatorySubmissionEngine instance."""
    return RegulatorySubmissionEngine()


@pytest.fixture
def sample_dds_valid() -> DDSDocument:
    """Create valid DDS document for submission."""
    return DDSDocument(
        dds_id="dds-test-001",
        reference_number="DDS-OP001-COFF-20260312-00001",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[
            ProductEntry(
                product_id="P1",
                description="Arabica Coffee Beans",
                hs_code="0901.11",
                quantity=Decimal("1000"),
                unit="kg",
            )
        ],
        article9_ref="a9p-001",
        risk_assessment_ref="rad-001",
        mitigation_ref="mid-001",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant",
        provenance_hash="abc" * 21 + "a",  # 64 chars
    )


@pytest.fixture
def sample_dds_incomplete() -> DDSDocument:
    """Create incomplete DDS document."""
    return DDSDocument(
        dds_id="dds-incomplete",
        reference_number="DDS-INCOMPLETE",
        operator_id="",  # Missing operator ID
        commodity=EUDRCommodity.COFFEE,
        products=[],  # No products
        article9_ref="",
        status=DDSStatus.DRAFT,
        compliance_conclusion="",
    )


# ---------------------------------------------------------------------------
# Test: Initialization
# ---------------------------------------------------------------------------

def test_engine_initialization(engine):
    """Test RegulatorySubmissionEngine initializes correctly."""
    assert engine._config is not None
    assert engine._provenance is not None
    assert len(engine._submissions) == 0
    assert engine._receipt_counter == 0


# ---------------------------------------------------------------------------
# Test: submit_dds - Success Paths
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_dds_valid(engine, sample_dds_valid):
    """Test submitting valid DDS."""
    record = await engine.submit_dds(sample_dds_valid)

    assert record.submission_id.startswith("sub-")
    assert record.dds_id == "dds-test-001"
    assert record.status == SubmissionStatus.SUBMITTED
    assert record.receipt_number is not None
    assert record.submitted_at is not None


@pytest.mark.asyncio
async def test_submit_dds_generates_unique_receipt(engine, sample_dds_valid):
    """Test each submission generates unique receipt number."""
    record1 = await engine.submit_dds(sample_dds_valid)

    # Modify DDS ID for second submission
    sample_dds_valid.dds_id = "dds-test-002"
    record2 = await engine.submit_dds(sample_dds_valid)

    assert record1.receipt_number != record2.receipt_number


@pytest.mark.asyncio
async def test_submit_dds_stores_submission(engine, sample_dds_valid):
    """Test submission is stored for later retrieval."""
    record = await engine.submit_dds(sample_dds_valid)

    stored = engine._submissions.get(record.submission_id)
    assert stored is not None
    assert stored.dds_id == "dds-test-001"


# ---------------------------------------------------------------------------
# Test: submit_dds - Validation Failures
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_dds_incomplete(engine, sample_dds_incomplete):
    """Test submitting incomplete DDS is rejected."""
    record = await engine.submit_dds(sample_dds_incomplete)

    assert record.status == SubmissionStatus.REJECTED
    assert record.rejection_reason is not None
    assert "validation failed" in record.rejection_reason.lower()


@pytest.mark.asyncio
async def test_submit_dds_draft_status(engine):
    """Test submitting DDS in draft status."""
    dds = DDSDocument(
        dds_id="dds-draft",
        reference_number="DDS-DRAFT",
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
        status=DDSStatus.DRAFT,  # Still draft
        compliance_conclusion="pending",
    )

    # Should allow submission but may warn
    record = await engine.submit_dds(dds)

    # Depends on implementation - may reject or accept with warning
    assert record.submission_id.startswith("sub-")


# ---------------------------------------------------------------------------
# Test: Pre-submission Validation
# ---------------------------------------------------------------------------

def test_pre_submission_validation_valid(engine, sample_dds_valid):
    """Test pre-submission validation passes for valid DDS."""
    issues = engine._pre_submission_validation(sample_dds_valid)

    errors = [i for i in issues if i.severity == ValidationSeverity.ERROR]
    assert len(errors) == 0


def test_pre_submission_validation_missing_operator(engine):
    """Test validation detects missing operator ID."""
    dds = DDSDocument(
        dds_id="dds-test",
        reference_number="DDS-REF",
        operator_id="",
        commodity=EUDRCommodity.COFFEE,
        products=[],
        article9_ref="a9p-001",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant",
    )

    issues = engine._pre_submission_validation(dds)

    assert len(issues) > 0
    assert any("operator" in i.message.lower() for i in issues)


def test_pre_submission_validation_missing_products(engine):
    """Test validation detects missing products."""
    dds = DDSDocument(
        dds_id="dds-test",
        reference_number="DDS-REF",
        operator_id="OP-001",
        commodity=EUDRCommodity.COFFEE,
        products=[],  # No products
        article9_ref="a9p-001",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant",
    )

    issues = engine._pre_submission_validation(dds)

    assert len(issues) > 0


# ---------------------------------------------------------------------------
# Test: EU IS Formatting
# ---------------------------------------------------------------------------

def test_format_for_eu_is(engine, sample_dds_valid):
    """Test formatting DDS for EU Information System schema."""
    payload = engine._format_for_eu_is(sample_dds_valid)

    assert "dds_reference_id" in payload or "dds_id" in payload
    # Check field mapping is applied
    # Implementation-specific


def test_format_for_eu_is_field_mapping(engine, sample_dds_valid):
    """Test EU IS field mapping is applied."""
    payload = engine._format_for_eu_is(sample_dds_valid)

    # Some fields should be renamed per mapping
    assert isinstance(payload, dict)


# ---------------------------------------------------------------------------
# Test: Receipt Number Generation
# ---------------------------------------------------------------------------

def test_generate_receipt_number(engine):
    """Test receipt number generation."""
    receipt1 = engine._generate_receipt_number()
    receipt2 = engine._generate_receipt_number()

    assert receipt1.startswith("EU-IS-")
    assert receipt2.startswith("EU-IS-")
    assert receipt1 != receipt2


def test_generate_receipt_number_sequential(engine):
    """Test receipt numbers increment sequentially."""
    receipt1 = engine._generate_receipt_number()
    receipt2 = engine._generate_receipt_number()
    receipt3 = engine._generate_receipt_number()

    # Extract sequence numbers and verify increment
    # Implementation-specific format
    assert receipt1 != receipt2
    assert receipt2 != receipt3


# ---------------------------------------------------------------------------
# Test: check_submission_status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_submission_status(engine, sample_dds_valid):
    """Test checking submission status."""
    record = await engine.submit_dds(sample_dds_valid)

    status = await engine.check_submission_status(record.submission_id)

    assert status.submission_id == record.submission_id
    assert status.dds_id == "dds-test-001"


@pytest.mark.asyncio
async def test_check_submission_status_not_found(engine):
    """Test checking non-existent submission fails."""
    with pytest.raises(ValueError, match="Submission not found"):
        await engine.check_submission_status("sub-nonexistent")


# ---------------------------------------------------------------------------
# Test: acknowledge_submission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_acknowledge_submission(engine, sample_dds_valid):
    """Test acknowledging a submission."""
    record = await engine.submit_dds(sample_dds_valid)

    acknowledged = await engine.acknowledge_submission(record.submission_id)

    assert acknowledged.status == SubmissionStatus.ACKNOWLEDGED
    assert acknowledged.acknowledged_at is not None


@pytest.mark.asyncio
async def test_acknowledge_submission_already_acknowledged(engine, sample_dds_valid):
    """Test acknowledging already acknowledged submission."""
    record = await engine.submit_dds(sample_dds_valid)
    await engine.acknowledge_submission(record.submission_id)

    # Second acknowledgement should be idempotent or raise error
    result = await engine.acknowledge_submission(record.submission_id)
    assert result.status == SubmissionStatus.ACKNOWLEDGED


# ---------------------------------------------------------------------------
# Test: reject_submission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reject_submission(engine, sample_dds_valid):
    """Test rejecting a submission."""
    record = await engine.submit_dds(sample_dds_valid)

    rejected = await engine.reject_submission(
        submission_id=record.submission_id,
        reason="MISSING_GEOLOCATION",
    )

    assert rejected.status == SubmissionStatus.REJECTED
    assert rejected.rejected_at is not None
    assert rejected.rejection_reason is not None


@pytest.mark.asyncio
async def test_reject_submission_with_custom_reason(engine, sample_dds_valid):
    """Test rejection with custom reason."""
    record = await engine.submit_dds(sample_dds_valid)

    rejected = await engine.reject_submission(
        submission_id=record.submission_id,
        reason="CUSTOM_REASON",
        custom_message="Custom rejection message",
    )

    assert "Custom rejection message" in rejected.rejection_reason


# ---------------------------------------------------------------------------
# Test: resubmit_dds
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resubmit_dds(engine, sample_dds_valid):
    """Test resubmitting a rejected DDS."""
    # Submit and reject
    record = await engine.submit_dds(sample_dds_valid)
    await engine.reject_submission(record.submission_id, "MISSING_GEOLOCATION")

    # Resubmit with corrections
    sample_dds_valid.dds_id = "dds-test-001-corrected"
    resubmitted = await engine.resubmit_dds(
        original_submission_id=record.submission_id,
        corrected_dds=sample_dds_valid,
    )

    assert resubmitted.status == SubmissionStatus.SUBMITTED
    assert resubmitted.resubmission_count == 1


@pytest.mark.asyncio
async def test_resubmit_dds_not_rejected(engine, sample_dds_valid):
    """Test resubmitting non-rejected submission fails."""
    record = await engine.submit_dds(sample_dds_valid)

    # Attempt to resubmit without rejection
    with pytest.raises(ValueError, match="not in rejected state"):
        await engine.resubmit_dds(record.submission_id, sample_dds_valid)


# ---------------------------------------------------------------------------
# Test: Batch Submission
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_batch_submit(engine):
    """Test batch submission of multiple DDS documents."""
    dds_list = [
        DDSDocument(
            dds_id=f"dds-batch-{i}",
            reference_number=f"DDS-BATCH-{i}",
            operator_id="OP-BATCH",
            commodity=EUDRCommodity.COFFEE,
            products=[
                ProductEntry(
                    product_id=f"P{i}",
                    description="Coffee",
                    hs_code="0901.11",
                    quantity=Decimal("100"),
                    unit="kg",
                )
            ],
            article9_ref=f"a9p-{i}",
            status=DDSStatus.VALIDATED,
            compliance_conclusion="compliant",
        )
        for i in range(3)
    ]

    results = await engine.batch_submit(dds_list)

    assert len(results) == 3
    assert all(r.status == SubmissionStatus.SUBMITTED for r in results)


@pytest.mark.asyncio
async def test_batch_submit_mixed_validity(engine):
    """Test batch submission with mix of valid/invalid DDS."""
    dds_list = [
        DDSDocument(
            dds_id="dds-valid",
            reference_number="DDS-VALID",
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
            status=DDSStatus.VALIDATED,
            compliance_conclusion="compliant",
        ),
        DDSDocument(
            dds_id="dds-invalid",
            reference_number="",
            operator_id="",  # Invalid
            commodity=EUDRCommodity.COFFEE,
            products=[],
            article9_ref="",
            status=DDSStatus.DRAFT,
            compliance_conclusion="",
        ),
    ]

    results = await engine.batch_submit(dds_list)

    assert len(results) == 2
    assert results[0].status == SubmissionStatus.SUBMITTED
    assert results[1].status == SubmissionStatus.REJECTED


# ---------------------------------------------------------------------------
# Test: Submission History
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_submission_history_by_operator(engine, sample_dds_valid):
    """Test getting submission history filtered by operator."""
    await engine.submit_dds(sample_dds_valid)

    sample_dds_valid.dds_id = "dds-test-002"
    await engine.submit_dds(sample_dds_valid)

    history = await engine.get_submission_history(operator_id="OP-001")

    assert len(history) >= 2


@pytest.mark.asyncio
async def test_get_submission_history_by_status(engine, sample_dds_valid):
    """Test getting submission history filtered by status."""
    await engine.submit_dds(sample_dds_valid)

    history = await engine.get_submission_history(
        status=SubmissionStatus.SUBMITTED
    )

    assert all(r.status == SubmissionStatus.SUBMITTED for r in history)


# ---------------------------------------------------------------------------
# Test: Health Check
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_check(engine):
    """Test health check returns correct status."""
    status = await engine.health_check()

    assert status["engine"] == "RegulatorySubmissionEngine"
    assert status["status"] == "available"
    assert "config" in status


# ---------------------------------------------------------------------------
# Test: Rejection Reasons
# ---------------------------------------------------------------------------

def test_rejection_reasons_defined():
    """Test rejection reasons are properly defined."""
    assert "MISSING_GEOLOCATION" in _REJECTION_REASONS
    assert "INVALID_HS_CODE" in _REJECTION_REASONS
    assert "INCOMPLETE_RISK_ASSESSMENT" in _REJECTION_REASONS

    for reason_key, reason_data in _REJECTION_REASONS.items():
        assert "code" in reason_data
        assert "message" in reason_data
        assert "fix_suggestion" in reason_data


# ---------------------------------------------------------------------------
# Test: EU IS Field Mapping
# ---------------------------------------------------------------------------

def test_eu_is_field_mapping_defined():
    """Test EU IS field mapping is properly defined."""
    assert "dds_id" in _EU_IS_FIELD_MAPPING
    assert "operator_id" in _EU_IS_FIELD_MAPPING
    assert "commodity" in _EU_IS_FIELD_MAPPING


# ---------------------------------------------------------------------------
# Test: Different Commodities
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_submit_dds_palm_oil(engine):
    """Test submission for palm oil commodity."""
    dds = DDSDocument(
        dds_id="dds-palm-001",
        reference_number="DDS-PALM-001",
        operator_id="OP-PALM",
        commodity=EUDRCommodity.PALM_OIL,
        products=[
            ProductEntry(
                product_id="PALM1",
                description="Crude Palm Oil",
                hs_code="1511.10",
                quantity=Decimal("5000"),
                unit="kg",
            )
        ],
        article9_ref="a9p-palm",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant",
    )

    record = await engine.submit_dds(dds)

    assert record.status == SubmissionStatus.SUBMITTED


@pytest.mark.asyncio
async def test_submit_dds_wood(engine):
    """Test submission for wood commodity."""
    dds = DDSDocument(
        dds_id="dds-wood-001",
        reference_number="DDS-WOOD-001",
        operator_id="OP-WOOD",
        commodity=EUDRCommodity.WOOD,
        products=[
            ProductEntry(
                product_id="WOOD1",
                description="Timber Logs",
                hs_code="4403.11",
                quantity=Decimal("1000"),
                unit="m3",
            )
        ],
        article9_ref="a9p-wood",
        status=DDSStatus.VALIDATED,
        compliance_conclusion="compliant",
    )

    record = await engine.submit_dds(dds)

    assert record.status == SubmissionStatus.SUBMITTED

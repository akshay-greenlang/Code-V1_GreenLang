# -*- coding: utf-8 -*-
"""
Unit tests for DDSSubmitter engine - AGENT-EUDR-036

Tests DDS creation, validation, submission, withdrawal, and amendment.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.eu_information_system_interface.config import (
    EUInformationSystemInterfaceConfig,
)
from greenlang.agents.eudr.eu_information_system_interface.dds_submitter import (
    DDSSubmitter,
)
from greenlang.agents.eudr.eu_information_system_interface.models import (
    DDSStatus,
    DDSType,
    SubmissionStatus,
)
from greenlang.agents.eudr.eu_information_system_interface.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def submitter() -> DDSSubmitter:
    """Create a DDSSubmitter instance with default config."""
    config = EUInformationSystemInterfaceConfig()
    return DDSSubmitter(config=config, provenance=ProvenanceTracker())


@pytest.fixture
def non_strict_submitter() -> DDSSubmitter:
    """Create a DDSSubmitter with non-strict validation."""
    config = EUInformationSystemInterfaceConfig(dds_validation_strict=False)
    return DDSSubmitter(config=config, provenance=ProvenanceTracker())


class TestCreateDDS:
    """Test DDSSubmitter.create_dds()."""

    @pytest.mark.asyncio
    async def test_create_dds_basic(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        assert dds.dds_id.startswith("dds-")
        assert dds.operator_id == "op-001"
        assert dds.eori_number == "DE123456789012"
        assert dds.dds_type == DDSType.PLACING
        assert dds.status == DDSStatus.DRAFT

    @pytest.mark.asyncio
    async def test_create_dds_total_quantity(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        assert dds.total_quantity == Decimal("75000.00")

    @pytest.mark.asyncio
    async def test_create_dds_provenance_hash(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        assert len(dds.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_create_dds_with_references(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="export",
            commodity_lines=sample_commodity_line_data,
            risk_assessment_id="ra-001",
            mitigation_plan_id="mp-001",
            improvement_plan_id="ip-035-001",
        )
        assert dds.risk_assessment_id == "ra-001"
        assert dds.mitigation_plan_id == "mp-001"
        assert dds.improvement_plan_id == "ip-035-001"

    @pytest.mark.asyncio
    async def test_create_dds_too_many_lines(self, submitter):
        config = EUInformationSystemInterfaceConfig(dds_max_commodities_per_statement=1)
        sub = DDSSubmitter(config=config, provenance=ProvenanceTracker())
        lines = [
            {"commodity": "cocoa", "description": "Beans", "quantity": "1000", "country_of_production": "GH",
             "geolocation": {"format": "point", "point": {"latitude": "6.0", "longitude": "-1.0"}, "country_code": "GH"}},
            {"commodity": "coffee", "description": "Green", "quantity": "2000", "country_of_production": "CO",
             "geolocation": {"format": "point", "point": {"latitude": "4.0", "longitude": "-75.0"}, "country_code": "CO"}},
        ]
        with pytest.raises(ValueError, match="Too many commodity lines"):
            await sub.create_dds(
                operator_id="op-001",
                eori_number="DE123456789012",
                dds_type="placing",
                commodity_lines=lines,
            )

    @pytest.mark.asyncio
    async def test_create_dds_commodity_lines_parsed(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        assert len(dds.commodity_lines) == 2


class TestValidateDDS:
    """Test DDSSubmitter.validate_dds()."""

    @pytest.mark.asyncio
    async def test_validate_valid_dds(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
            risk_assessment_id="ra-001",
        )
        result = await submitter.validate_dds(dds)
        assert result["valid"] is True
        assert result["error_count"] == 0

    @pytest.mark.asyncio
    async def test_validate_missing_operator_id(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        dds.operator_id = ""
        result = await submitter.validate_dds(dds)
        assert result["valid"] is False
        assert any("operator_id" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_missing_risk_assessment_strict(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        result = await submitter.validate_dds(dds)
        assert result["valid"] is False
        assert any("risk_assessment" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_non_strict_no_risk_assessment(self, non_strict_submitter, sample_commodity_line_data):
        dds = await non_strict_submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
        )
        result = await non_strict_submitter.validate_dds(dds)
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_validate_has_duration_ms(self, submitter, sample_commodity_line_data):
        dds = await submitter.create_dds(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=sample_commodity_line_data,
            risk_assessment_id="ra-001",
        )
        result = await submitter.validate_dds(dds)
        assert "duration_ms" in result


class TestSubmitDDS:
    """Test DDSSubmitter.submit_dds()."""

    @pytest.mark.asyncio
    async def test_submit_draft_dds(self, submitter, sample_dds):
        submission = await submitter.submit_dds(sample_dds)
        assert submission.submission_id.startswith("sub-")
        assert submission.dds_id == "dds-test-001"
        assert submission.status == SubmissionStatus.PENDING
        assert sample_dds.status == DDSStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_submit_validated_dds(self, submitter, validated_dds):
        submission = await submitter.submit_dds(validated_dds)
        assert submission.status == SubmissionStatus.PENDING
        assert validated_dds.status == DDSStatus.SUBMITTED

    @pytest.mark.asyncio
    async def test_submit_already_submitted_raises(self, submitter, submitted_dds):
        with pytest.raises(ValueError, match="submitted"):
            await submitter.submit_dds(submitted_dds)

    @pytest.mark.asyncio
    async def test_submit_sets_submitted_at(self, submitter, sample_dds):
        await submitter.submit_dds(sample_dds)
        assert sample_dds.submitted_at is not None

    @pytest.mark.asyncio
    async def test_submit_has_provenance_hash(self, submitter, sample_dds):
        submission = await submitter.submit_dds(sample_dds)
        assert len(submission.provenance_hash) == 64


class TestWithdrawDDS:
    """Test DDSSubmitter.withdraw_dds()."""

    @pytest.mark.asyncio
    async def test_withdraw_submitted(self, submitter, submitted_dds):
        result = await submitter.withdraw_dds(submitted_dds, "Changed requirements")
        assert result["status"] == "withdrawn"
        assert result["reason"] == "Changed requirements"
        assert submitted_dds.status == DDSStatus.WITHDRAWN

    @pytest.mark.asyncio
    async def test_withdraw_draft_raises(self, submitter, sample_dds):
        with pytest.raises(ValueError, match="cannot be withdrawn"):
            await submitter.withdraw_dds(sample_dds, "Test")

    @pytest.mark.asyncio
    async def test_withdraw_accepted_raises(self, submitter, accepted_dds):
        with pytest.raises(ValueError, match="cannot be withdrawn"):
            await submitter.withdraw_dds(accepted_dds, "Test")


class TestAmendDDS:
    """Test DDSSubmitter.amend_dds()."""

    @pytest.mark.asyncio
    async def test_amend_submitted(self, submitter, submitted_dds):
        amended = await submitter.amend_dds(
            submitted_dds,
            {"risk_assessment_id": "ra-updated-001"},
        )
        assert amended.dds_id.startswith("dds-")
        assert amended.risk_assessment_id == "ra-updated-001"
        assert amended.status == DDSStatus.DRAFT
        assert submitted_dds.status == DDSStatus.AMENDED

    @pytest.mark.asyncio
    async def test_amend_accepted(self, submitter, accepted_dds):
        amended = await submitter.amend_dds(
            accepted_dds,
            {"improvement_plan_id": "ip-035-new"},
        )
        assert amended.improvement_plan_id == "ip-035-new"

    @pytest.mark.asyncio
    async def test_amend_draft_raises(self, submitter, sample_dds):
        with pytest.raises(ValueError, match="cannot be amended"):
            await submitter.amend_dds(sample_dds, {})

    @pytest.mark.asyncio
    async def test_amend_has_provenance(self, submitter, submitted_dds):
        amended = await submitter.amend_dds(submitted_dds, {})
        assert len(amended.provenance_hash) == 64


class TestHealthCheck:
    """Test DDSSubmitter.health_check()."""

    @pytest.mark.asyncio
    async def test_health_check(self, submitter):
        health = await submitter.health_check()
        assert health["engine"] == "DDSSubmitter"
        assert health["status"] == "available"
        assert "config" in health

# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceValidator - AGENT-EUDR-037

Tests Article 4 mandatory field validation, geolocation checks,
risk assessment checks, supply chain checks, and validation reports.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.config import DDSCreatorConfig
from greenlang.agents.eudr.due_diligence_statement_creator.compliance_validator import ComplianceValidator
from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    CommodityType, DDSStatement, DDSStatus, DDSValidationReport,
    GeolocationData, RiskReference, SupplyChainData, ValidationResult,
)


@pytest.fixture
def config():
    return DDSCreatorConfig()


@pytest.fixture
def validator(config):
    return ComplianceValidator(config=config)


@pytest.fixture
def minimal_statement():
    """Statement with only required fields, missing most data."""
    return DDSStatement(
        statement_id="DDS-MIN", reference_number="REF-MIN",
        operator_id="OP-001", operator_name="Test Corp",
        commodities=[CommodityType.COCOA])


class TestValidateCompleteStatement:
    @pytest.mark.asyncio
    async def test_complete_statement_passes(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert isinstance(report, DDSValidationReport)
        assert report.overall_result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_complete_statement_mandatory_fields(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert report.mandatory_fields_complete is True

    @pytest.mark.asyncio
    async def test_complete_statement_geolocation_valid(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert report.geolocation_valid is True

    @pytest.mark.asyncio
    async def test_complete_statement_risk_included(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert report.risk_assessment_included is True

    @pytest.mark.asyncio
    async def test_complete_statement_supply_chain(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert report.supply_chain_complete is True

    @pytest.mark.asyncio
    async def test_complete_statement_has_provenance(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        assert len(report.provenance_hash) == 64


class TestValidateMinimalStatement:
    @pytest.mark.asyncio
    async def test_minimal_statement_fails(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        assert report.overall_result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_minimal_statement_has_failed_checks(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        assert report.failed_checks > 0

    @pytest.mark.asyncio
    async def test_minimal_statement_missing_geolocation(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        assert report.geolocation_valid is False

    @pytest.mark.asyncio
    async def test_minimal_statement_missing_risk(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        assert report.risk_assessment_included is False

    @pytest.mark.asyncio
    async def test_minimal_statement_missing_supply_chain(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        assert report.supply_chain_complete is False


class TestValidationReportCounts:
    @pytest.mark.asyncio
    async def test_total_checks_positive(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        assert report.total_checks > 0

    @pytest.mark.asyncio
    async def test_passed_plus_failed_plus_warnings(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        counted = report.passed_checks + report.failed_checks + report.warning_checks
        not_applicable = sum(
            1 for c in report.checks if c.result == ValidationResult.NOT_APPLICABLE
        )
        assert counted + not_applicable == report.total_checks

    @pytest.mark.asyncio
    async def test_report_has_statement_id(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        assert report.statement_id == sample_statement.statement_id

    @pytest.mark.asyncio
    async def test_report_id_prefix(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        assert report.report_id.startswith("VR-")


class TestValidationChecks:
    @pytest.mark.asyncio
    async def test_checks_have_ids(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        for check in report.checks:
            assert check.check_id.startswith("CHK-")

    @pytest.mark.asyncio
    async def test_checks_have_field_names(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        for check in report.checks:
            assert check.field_name != ""

    @pytest.mark.asyncio
    async def test_checks_have_article_references(self, validator, sample_statement):
        report = await validator.validate_statement(sample_statement)
        for check in report.checks:
            assert check.article_reference != ""

    @pytest.mark.asyncio
    async def test_failed_checks_have_severity_error(self, validator, minimal_statement):
        report = await validator.validate_statement(minimal_statement)
        failed = [c for c in report.checks if c.result == ValidationResult.FAIL]
        for check in failed:
            assert check.severity == "error"


class TestDeforestationAndLegalChecks:
    @pytest.mark.asyncio
    async def test_deforestation_free_warning_when_false(self, validator, sample_statement):
        sample_statement.deforestation_free = False
        report = await validator.validate_statement(sample_statement)
        deforestation_checks = [
            c for c in report.checks if c.field_name == "deforestation_free"
        ]
        assert len(deforestation_checks) == 1
        assert deforestation_checks[0].result == ValidationResult.WARNING

    @pytest.mark.asyncio
    async def test_legally_produced_warning_when_false(self, validator, sample_statement):
        sample_statement.legally_produced = False
        report = await validator.validate_statement(sample_statement)
        legal_checks = [
            c for c in report.checks if c.field_name == "legally_produced"
        ]
        assert len(legal_checks) == 1
        assert legal_checks[0].result == ValidationResult.WARNING

    @pytest.mark.asyncio
    async def test_deforestation_free_pass_when_true(self, validator, complete_statement):
        report = await validator.validate_statement(complete_statement)
        deforestation_checks = [
            c for c in report.checks if c.field_name == "deforestation_free"
        ]
        assert len(deforestation_checks) == 1
        assert deforestation_checks[0].result == ValidationResult.PASS


class TestValidationCount:
    @pytest.mark.asyncio
    async def test_validation_count_increments(self, validator, sample_statement):
        await validator.validate_statement(sample_statement)
        await validator.validate_statement(sample_statement)
        health = await validator.health_check()
        assert health["validations_completed"] == 2


class TestComplianceHealth:
    @pytest.mark.asyncio
    async def test_health_check(self, validator):
        health = await validator.health_check()
        assert health["engine"] == "ComplianceValidator"
        assert health["status"] == "healthy"

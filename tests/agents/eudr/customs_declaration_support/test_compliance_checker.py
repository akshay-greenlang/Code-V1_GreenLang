# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceChecker engine - AGENT-EUDR-039

Tests EUDR Article 4 compliance checks, DDS reference validation,
CN code EUDR coverage verification, origin checks, deforestation-free
declarations, risk level assessments, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from greenlang.agents.eudr.customs_declaration_support.config import CustomsDeclarationConfig
from greenlang.agents.eudr.customs_declaration_support.compliance_checker import ComplianceChecker
from greenlang.agents.eudr.customs_declaration_support.models import (
    ComplianceCheck, ComplianceStatus, ValidationResult,
)


@pytest.fixture
def config():
    return CustomsDeclarationConfig()


@pytest.fixture
def checker(config):
    return ComplianceChecker(config=config)


# ====================================================================
# DDS Reference Validation Tests
# ====================================================================


class TestDDSReferenceValidation:
    @pytest.mark.asyncio
    async def test_valid_dds_reference_passes(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_empty_dds_reference_fails(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="",
        )
        assert result.result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_invalid_format_dds_fails(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="INVALID-REF-123",
        )
        assert result.result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_check_type_is_dds_reference(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.check_type == "dds_reference_validation"

    @pytest.mark.asyncio
    async def test_article_reference_present(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert "Art" in result.article_reference or "4" in result.article_reference

    @pytest.mark.asyncio
    async def test_check_id_generated(self, checker):
        result = await checker.check_dds_reference(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.check_id != ""


# ====================================================================
# CN Code EUDR Coverage Tests
# ====================================================================


class TestCNCodeEUDRCoverage:
    @pytest.mark.asyncio
    async def test_eudr_cn_code_passes(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=["18010000"],
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_non_eudr_cn_code_info(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=["87032100"],
        )
        # Non-EUDR code might pass (no EUDR requirements) or be info
        assert result.result in (ValidationResult.PASS, ValidationResult.NOT_APPLICABLE)

    @pytest.mark.asyncio
    async def test_mixed_cn_codes(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=["18010000", "87032100"],
        )
        # Should identify EUDR-regulated items
        assert result.result in (ValidationResult.PASS, ValidationResult.WARNING)

    @pytest.mark.asyncio
    async def test_empty_cn_codes_fails(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=[],
        )
        assert result.result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_multiple_eudr_cn_codes_pass(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=["18010000", "09011100", "44011100"],
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_check_type_is_cn_code_coverage(self, checker):
        result = await checker.check_cn_code_coverage(
            declaration_id="DECL-001",
            cn_codes=["18010000"],
        )
        assert result.check_type == "cn_code_eudr_coverage"


# ====================================================================
# Origin Verification Check Tests
# ====================================================================


class TestOriginCheck:
    @pytest.mark.asyncio
    async def test_verified_origin_passes(self, checker):
        result = await checker.check_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI", "GH"],
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_mismatched_origin_fails(self, checker):
        result = await checker.check_origin(
            declaration_id="DECL-001",
            declared_origin="BR",
            supply_chain_origins=["CI", "GH"],
        )
        assert result.result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_no_supply_chain_warning(self, checker):
        result = await checker.check_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=[],
        )
        assert result.result in (ValidationResult.FAIL, ValidationResult.WARNING)

    @pytest.mark.asyncio
    async def test_check_type_is_origin_verification(self, checker):
        result = await checker.check_origin(
            declaration_id="DECL-001",
            declared_origin="CI",
            supply_chain_origins=["CI"],
        )
        assert result.check_type == "origin_verification"


# ====================================================================
# Deforestation-Free Declaration Tests
# ====================================================================


class TestDeforestationFreeCheck:
    @pytest.mark.asyncio
    async def test_confirmed_deforestation_free_passes(self, checker):
        result = await checker.check_deforestation_free(
            declaration_id="DECL-001",
            deforestation_free=True,
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_not_confirmed_fails(self, checker):
        result = await checker.check_deforestation_free(
            declaration_id="DECL-001",
            deforestation_free=False,
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.result == ValidationResult.FAIL

    @pytest.mark.asyncio
    async def test_no_dds_reference_fails(self, checker):
        result = await checker.check_deforestation_free(
            declaration_id="DECL-001",
            deforestation_free=True,
            dds_reference="",
        )
        assert result.result in (ValidationResult.FAIL, ValidationResult.WARNING)

    @pytest.mark.asyncio
    async def test_check_type_is_deforestation_free(self, checker):
        result = await checker.check_deforestation_free(
            declaration_id="DECL-001",
            deforestation_free=True,
            dds_reference="GL-DDS-20260313-ABCDEF",
        )
        assert result.check_type == "deforestation_free_declaration"


# ====================================================================
# Risk Level Check Tests
# ====================================================================


class TestRiskLevelCheck:
    @pytest.mark.asyncio
    async def test_low_risk_passes(self, checker):
        result = await checker.check_risk_level(
            declaration_id="DECL-001",
            risk_level="low",
            country_code="DE",
        )
        assert result.result == ValidationResult.PASS

    @pytest.mark.asyncio
    async def test_standard_risk_passes(self, checker):
        result = await checker.check_risk_level(
            declaration_id="DECL-001",
            risk_level="standard",
            country_code="BR",
        )
        assert result.result in (ValidationResult.PASS, ValidationResult.WARNING)

    @pytest.mark.asyncio
    async def test_high_risk_warning(self, checker):
        result = await checker.check_risk_level(
            declaration_id="DECL-001",
            risk_level="high",
            country_code="CM",
        )
        assert result.result == ValidationResult.WARNING

    @pytest.mark.asyncio
    async def test_critical_risk_warning(self, checker):
        result = await checker.check_risk_level(
            declaration_id="DECL-001",
            risk_level="critical",
            country_code="CD",
        )
        assert result.result in (ValidationResult.WARNING, ValidationResult.FAIL)


# ====================================================================
# Full Compliance Report Tests
# ====================================================================


class TestFullComplianceReport:
    @pytest.mark.asyncio
    async def test_run_all_checks(self, checker):
        report = await checker.run_full_compliance_check(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
            cn_codes=["18010000"],
            declared_origin="CI",
            supply_chain_origins=["CI"],
            deforestation_free=True,
            risk_level="low",
            country_code="CI",
        )
        assert isinstance(report, dict)
        assert "checks" in report
        assert "overall_status" in report
        assert len(report["checks"]) >= 4

    @pytest.mark.asyncio
    async def test_all_pass_is_compliant(self, checker):
        report = await checker.run_full_compliance_check(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
            cn_codes=["18010000"],
            declared_origin="CI",
            supply_chain_origins=["CI"],
            deforestation_free=True,
            risk_level="low",
            country_code="CI",
        )
        assert report["overall_status"] in ("compliant", ComplianceStatus.COMPLIANT.value)

    @pytest.mark.asyncio
    async def test_missing_dds_is_non_compliant(self, checker):
        report = await checker.run_full_compliance_check(
            declaration_id="DECL-001",
            dds_reference="",
            cn_codes=["18010000"],
            declared_origin="CI",
            supply_chain_origins=["CI"],
            deforestation_free=True,
            risk_level="low",
            country_code="CI",
        )
        assert report["overall_status"] in (
            "non_compliant", ComplianceStatus.NON_COMPLIANT.value,
            "partially_compliant", ComplianceStatus.PARTIALLY_COMPLIANT.value,
        )

    @pytest.mark.asyncio
    async def test_report_has_provenance(self, checker):
        report = await checker.run_full_compliance_check(
            declaration_id="DECL-001",
            dds_reference="GL-DDS-20260313-ABCDEF",
            cn_codes=["18010000"],
            declared_origin="CI",
            supply_chain_origins=["CI"],
            deforestation_free=True,
            risk_level="low",
            country_code="CI",
        )
        assert "provenance_hash" in report
        assert len(report["provenance_hash"]) == 64


# ====================================================================
# Health Check Tests
# ====================================================================


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_returns_dict(self, checker):
        health = await checker.health_check()
        assert isinstance(health, dict)
        assert health["engine"] == "ComplianceChecker"

    @pytest.mark.asyncio
    async def test_status_healthy(self, checker):
        health = await checker.health_check()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_checks_performed_zero(self, checker):
        health = await checker.health_check()
        assert health["checks_performed"] == 0

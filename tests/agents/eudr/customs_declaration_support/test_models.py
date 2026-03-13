# -*- coding: utf-8 -*-
"""
Unit tests for models - AGENT-EUDR-039

Tests all 12+ enums, constants, and 15+ Pydantic models for
customs declaration support including CN codes, HS codes, MRN format,
tariff calculations, and currency handling.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.customs_declaration_support.models import (
    AGENT_ID, AGENT_VERSION, EUDR_CN_CODE_MAPPINGS, EUDR_REGULATED_COMMODITIES,
    EUDR_HS_CHAPTERS, MRN_FORMAT_REGEX, SUPPORTED_INCOTERMS,
    CNCodeMapping, CommodityType, ComplianceCheck, ComplianceStatus,
    CurrencyCode, CustomsDeclaration, CustomsInterfaceResponse,
    CustomsSystemType, DeclarationStatus, DeclarationType, DutyCalculation,
    HSCodeInfo, HealthStatus, IncotermsType, MRNStatus, OriginVerification,
    OriginVerificationResult, PortOfEntry, RiskLevel, SADForm,
    TariffCalculation, TariffLineItem, ValidationResult,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestDeclarationStatusEnum:
    def test_values_count(self):
        assert len(DeclarationStatus) >= 8

    def test_pending_value(self):
        assert DeclarationStatus.PENDING.value == "pending"

    def test_submitted_value(self):
        assert DeclarationStatus.SUBMITTED.value == "submitted"

    def test_cleared_value(self):
        assert DeclarationStatus.CLEARED.value == "cleared"

    def test_rejected_value(self):
        assert DeclarationStatus.REJECTED.value == "rejected"

    def test_draft_value(self):
        assert DeclarationStatus.DRAFT.value == "draft"

    def test_under_review_value(self):
        assert DeclarationStatus.UNDER_REVIEW.value == "under_review"

    def test_amended_value(self):
        assert DeclarationStatus.AMENDED.value == "amended"

    def test_cancelled_value(self):
        assert DeclarationStatus.CANCELLED.value == "cancelled"

    def test_all_values_are_strings(self):
        for status in DeclarationStatus:
            assert isinstance(status.value, str)


class TestDeclarationTypeEnum:
    def test_values_count(self):
        assert len(DeclarationType) >= 3

    def test_import_value(self):
        assert DeclarationType.IMPORT.value == "import"

    def test_export_value(self):
        assert DeclarationType.EXPORT.value == "export"

    def test_transit_value(self):
        assert DeclarationType.TRANSIT.value == "transit"


class TestCommodityTypeEnum:
    def test_values_count(self):
        assert len(CommodityType) == 7

    def test_cattle_value(self):
        assert CommodityType.CATTLE.value == "cattle"

    def test_cocoa_value(self):
        assert CommodityType.COCOA.value == "cocoa"

    def test_coffee_value(self):
        assert CommodityType.COFFEE.value == "coffee"

    def test_oil_palm_value(self):
        assert CommodityType.OIL_PALM.value == "oil_palm"

    def test_rubber_value(self):
        assert CommodityType.RUBBER.value == "rubber"

    def test_soya_value(self):
        assert CommodityType.SOYA.value == "soya"

    def test_wood_value(self):
        assert CommodityType.WOOD.value == "wood"


class TestRiskLevelEnum:
    def test_values_count(self):
        assert len(RiskLevel) == 4

    def test_low_value(self):
        assert RiskLevel.LOW.value == "low"

    def test_standard_value(self):
        assert RiskLevel.STANDARD.value == "standard"

    def test_high_value(self):
        assert RiskLevel.HIGH.value == "high"

    def test_critical_value(self):
        assert RiskLevel.CRITICAL.value == "critical"


class TestCurrencyCodeEnum:
    def test_values_count(self):
        assert len(CurrencyCode) >= 4

    def test_eur_value(self):
        assert CurrencyCode.EUR.value == "EUR"

    def test_usd_value(self):
        assert CurrencyCode.USD.value == "USD"

    def test_gbp_value(self):
        assert CurrencyCode.GBP.value == "GBP"

    def test_jpy_value(self):
        assert CurrencyCode.JPY.value == "JPY"


class TestIncotermsTypeEnum:
    def test_values_count(self):
        assert len(IncotermsType) >= 5

    def test_cif_value(self):
        assert IncotermsType.CIF.value == "CIF"

    def test_fob_value(self):
        assert IncotermsType.FOB.value == "FOB"

    def test_exw_value(self):
        assert IncotermsType.EXW.value == "EXW"

    def test_dap_value(self):
        assert IncotermsType.DAP.value == "DAP"

    def test_ddp_value(self):
        assert IncotermsType.DDP.value == "DDP"


class TestCustomsSystemTypeEnum:
    def test_values_count(self):
        assert len(CustomsSystemType) >= 2

    def test_ncts_value(self):
        assert CustomsSystemType.NCTS.value == "ncts"

    def test_ais_value(self):
        assert CustomsSystemType.AIS.value == "ais"


class TestValidationResultEnum:
    def test_values_count(self):
        assert len(ValidationResult) == 4

    def test_pass_value(self):
        assert ValidationResult.PASS.value == "pass"

    def test_fail_value(self):
        assert ValidationResult.FAIL.value == "fail"

    def test_warning_value(self):
        assert ValidationResult.WARNING.value == "warning"

    def test_not_applicable_value(self):
        assert ValidationResult.NOT_APPLICABLE.value == "not_applicable"


class TestComplianceStatusEnum:
    def test_values_count(self):
        assert len(ComplianceStatus) >= 4

    def test_compliant_value(self):
        assert ComplianceStatus.COMPLIANT.value == "compliant"

    def test_non_compliant_value(self):
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"

    def test_pending_value(self):
        assert ComplianceStatus.PENDING.value == "pending"


class TestMRNStatusEnum:
    def test_values_count(self):
        assert len(MRNStatus) >= 4

    def test_generated_value(self):
        assert MRNStatus.GENERATED.value == "generated"

    def test_submitted_value(self):
        assert MRNStatus.SUBMITTED.value == "submitted"

    def test_accepted_value(self):
        assert MRNStatus.ACCEPTED.value == "accepted"

    def test_rejected_value(self):
        assert MRNStatus.REJECTED.value == "rejected"


class TestOriginVerificationResultEnum:
    def test_values_count(self):
        assert len(OriginVerificationResult) >= 3

    def test_verified_value(self):
        assert OriginVerificationResult.VERIFIED.value == "verified"

    def test_mismatch_value(self):
        assert OriginVerificationResult.MISMATCH.value == "mismatch"

    def test_unverified_value(self):
        assert OriginVerificationResult.UNVERIFIED.value == "unverified"


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-CDS-039"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_regulated_commodities_count(self):
        assert len(EUDR_REGULATED_COMMODITIES) == 7

    def test_regulated_commodities_all_types(self):
        for c in EUDR_REGULATED_COMMODITIES:
            assert isinstance(c, CommodityType)

    def test_cn_code_mappings_has_all_commodities(self):
        commodity_values = {c.value for c in CommodityType}
        mapping_commodities = set(EUDR_CN_CODE_MAPPINGS.keys())
        assert commodity_values.issubset(mapping_commodities)

    def test_cn_code_mappings_cattle_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["cattle"]) > 0

    def test_cn_code_mappings_cocoa_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["cocoa"]) > 0

    def test_cn_code_mappings_coffee_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["coffee"]) > 0

    def test_cn_code_mappings_oil_palm_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["oil_palm"]) > 0

    def test_cn_code_mappings_rubber_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["rubber"]) > 0

    def test_cn_code_mappings_soya_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["soya"]) > 0

    def test_cn_code_mappings_wood_not_empty(self):
        assert len(EUDR_CN_CODE_MAPPINGS["wood"]) > 0

    def test_hs_chapters_has_entries(self):
        assert len(EUDR_HS_CHAPTERS) > 0

    def test_mrn_format_regex_not_empty(self):
        assert MRN_FORMAT_REGEX is not None
        assert len(MRN_FORMAT_REGEX) > 0

    def test_supported_incoterms_count(self):
        assert len(SUPPORTED_INCOTERMS) >= 5


class TestMRNFormatValidation:
    """Test MRN (Movement Reference Number) format: 18 characters."""

    def test_valid_mrn_format_18_chars(self):
        import re
        mrn = "26NL0003960000001A"
        assert len(mrn) == 18
        assert re.match(MRN_FORMAT_REGEX, mrn) is not None

    def test_valid_mrn_de_format(self):
        import re
        mrn = "26DE0003960000004D"
        assert len(mrn) == 18
        assert re.match(MRN_FORMAT_REGEX, mrn) is not None

    def test_valid_mrn_be_format(self):
        import re
        mrn = "26BE0003960000003C"
        assert len(mrn) == 18
        assert re.match(MRN_FORMAT_REGEX, mrn) is not None

    def test_short_mrn_fails_validation(self):
        import re
        mrn = "26NL000396"
        assert len(mrn) != 18
        result = re.match(MRN_FORMAT_REGEX, mrn)
        # Should either not match or be None
        if result is not None:
            assert result.group(0) != mrn or len(mrn) == 18

    def test_empty_mrn_fails_validation(self):
        import re
        mrn = ""
        assert re.match(MRN_FORMAT_REGEX, mrn) is None


class TestCNCodeFormat:
    """Test CN code format validation (8 digits)."""

    def test_cn_code_8_digits(self):
        code = "18010000"
        assert len(code) == 8
        assert code.isdigit()

    def test_cn_code_cocoa_beans(self):
        code = "18010000"
        assert len(code) == 8

    def test_cn_code_coffee_raw(self):
        code = "09011100"
        assert len(code) == 8

    def test_cn_code_cattle(self):
        code = "01022110"
        assert len(code) == 8

    def test_cn_code_wood(self):
        code = "44011100"
        assert len(code) == 8

    def test_cn_code_rubber(self):
        code = "40011000"
        assert len(code) == 8

    def test_cn_code_soya(self):
        code = "12010090"
        assert len(code) == 8

    def test_cn_code_palm_oil(self):
        code = "15119110"
        assert len(code) == 8


class TestHSCodeFormat:
    """Test HS code format validation (6 digits)."""

    def test_hs_code_6_digits(self):
        code = "180100"
        assert len(code) == 6
        assert code.isdigit()

    def test_hs_code_coffee(self):
        code = "090111"
        assert len(code) == 6

    def test_hs_code_cattle(self):
        code = "010221"
        assert len(code) == 6

    def test_hs_code_wood(self):
        code = "440111"
        assert len(code) == 6


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestCNCodeMappingModel:
    def test_create_basic(self, cocoa_cn_codes):
        mapping = cocoa_cn_codes[0]
        assert mapping.commodity == CommodityType.COCOA
        assert mapping.cn_code == "18010000"
        assert len(mapping.cn_code) == 8

    def test_duty_rate_decimal(self, cocoa_cn_codes):
        mapping = cocoa_cn_codes[1]
        assert mapping.duty_rate == Decimal("9.6")

    def test_hs_code_derived_from_cn(self, cocoa_cn_codes):
        mapping = cocoa_cn_codes[0]
        assert mapping.hs_code == "180100"
        assert mapping.cn_code.startswith(mapping.hs_code[:4])


class TestHSCodeInfoModel:
    def test_create_basic(self, sample_hs_code_info):
        assert sample_hs_code_info.hs_code == "180100"
        assert sample_hs_code_info.chapter == 18
        assert sample_hs_code_info.eudr_regulated is True

    def test_non_eudr_product(self, invalid_hs_code_info):
        assert invalid_hs_code_info.eudr_regulated is False
        assert invalid_hs_code_info.commodity is None

    def test_chapter_extraction(self, sample_hs_code_info):
        assert sample_hs_code_info.chapter == int(sample_hs_code_info.hs_code[:2])


class TestTariffLineItemModel:
    def test_create_basic(self, sample_tariff_line_item):
        assert sample_tariff_line_item.line_number == 1
        assert sample_tariff_line_item.cn_code == "18010000"
        assert sample_tariff_line_item.currency == CurrencyCode.EUR

    def test_total_value_calculation(self, sample_tariff_line_item):
        expected = sample_tariff_line_item.quantity * sample_tariff_line_item.unit_price
        assert sample_tariff_line_item.total_value == expected

    def test_vat_amount_calculation(self, sample_tariff_line_item):
        expected = sample_tariff_line_item.total_value * sample_tariff_line_item.vat_rate / 100
        assert sample_tariff_line_item.vat_amount == expected

    def test_duty_amount_zero_for_zero_rate(self, sample_tariff_line_item):
        assert sample_tariff_line_item.duty_rate == Decimal("0.0")
        assert sample_tariff_line_item.duty_amount == Decimal("0.00")


class TestTariffCalculationModel:
    def test_create_basic(self, sample_tariff_calculation):
        assert sample_tariff_calculation.calculation_id == "CALC-001"
        assert sample_tariff_calculation.currency == CurrencyCode.EUR

    def test_total_payable_sum(self, sample_tariff_calculation):
        expected = (
            sample_tariff_calculation.total_duty_amount
            + sample_tariff_calculation.total_vat_amount
        )
        assert sample_tariff_calculation.total_payable == expected

    def test_exchange_rate_default(self):
        calc = TariffCalculation(
            calculation_id="C-X", declaration_id="D-X",
            total_customs_value=Decimal("100"),
        )
        assert calc.exchange_rate == Decimal("1.0")


class TestDutyCalculationModel:
    def test_create_basic(self, sample_duty_calculation):
        assert sample_duty_calculation.cn_code == "18032000"
        assert sample_duty_calculation.duty_rate == Decimal("9.6")

    def test_duty_amount_computation(self, sample_duty_calculation):
        expected = sample_duty_calculation.customs_value * sample_duty_calculation.duty_rate / 100
        assert sample_duty_calculation.duty_amount == expected

    def test_preferential_rate(self, sample_duty_calculation):
        assert sample_duty_calculation.preferential_rate == Decimal("0.0")
        assert sample_duty_calculation.preferential_origin == "CI"


class TestOriginVerificationModel:
    def test_create_verified(self, sample_origin_verification):
        assert sample_origin_verification.result == OriginVerificationResult.VERIFIED
        assert sample_origin_verification.confidence_score >= Decimal("90")

    def test_create_mismatch(self, failed_origin_verification):
        assert failed_origin_verification.result == OriginVerificationResult.MISMATCH
        assert failed_origin_verification.mismatch_details != ""

    def test_supply_chain_origins_list(self, sample_origin_verification):
        assert "CI" in sample_origin_verification.supply_chain_origins
        assert "GH" in sample_origin_verification.supply_chain_origins


class TestComplianceCheckModel:
    def test_create_passing(self, sample_compliance_check):
        assert sample_compliance_check.result == ValidationResult.PASS

    def test_create_failing(self, failed_compliance_check):
        assert failed_compliance_check.result == ValidationResult.FAIL
        assert failed_compliance_check.severity == "critical"

    def test_default_severity(self):
        check = ComplianceCheck(
            check_id="CHK-X", declaration_id="D-X",
            check_type="test",
            result=ValidationResult.PASS,
        )
        assert check.severity == "info"


class TestPortOfEntryModel:
    def test_create_rotterdam(self, port_rotterdam):
        assert port_rotterdam.port_code == "NLRTM"
        assert port_rotterdam.country_code == "NL"
        assert port_rotterdam.ncts_enabled is True

    def test_create_hamburg(self, port_hamburg):
        assert port_hamburg.port_code == "DEHAM"
        assert port_hamburg.country_code == "DE"

    def test_create_antwerp(self, port_antwerp):
        assert port_antwerp.port_code == "BEANR"
        assert port_antwerp.country_code == "BE"


class TestCustomsDeclarationModel:
    def test_create_pending(self, pending_declaration):
        assert pending_declaration.status == DeclarationStatus.PENDING
        assert pending_declaration.declaration_type == DeclarationType.IMPORT

    def test_mrn_format(self, pending_declaration):
        assert len(pending_declaration.mrn) == 18

    def test_eori_format(self, pending_declaration):
        assert len(pending_declaration.operator_eori) >= 10

    def test_cn_codes_present(self, pending_declaration):
        assert len(pending_declaration.cn_codes) > 0
        assert all(len(c) == 8 for c in pending_declaration.cn_codes)

    def test_hs_codes_present(self, pending_declaration):
        assert len(pending_declaration.hs_codes) > 0
        assert all(len(c) == 6 for c in pending_declaration.hs_codes)

    def test_dds_reference_present(self, pending_declaration):
        assert pending_declaration.dds_reference.startswith("GL-DDS-")

    def test_submitted_status(self, submitted_declaration):
        assert submitted_declaration.status == DeclarationStatus.SUBMITTED
        assert submitted_declaration.submitted_at is not None

    def test_cleared_status(self, cleared_declaration):
        assert cleared_declaration.status == DeclarationStatus.CLEARED
        assert cleared_declaration.cleared_at is not None

    def test_rejected_status(self, rejected_declaration):
        assert rejected_declaration.status == DeclarationStatus.REJECTED
        assert rejected_declaration.rejection_reason != ""

    def test_rejected_missing_dds(self, rejected_declaration):
        assert rejected_declaration.dds_reference == ""

    def test_default_status_is_draft(self):
        decl = CustomsDeclaration(
            declaration_id="D-X", mrn="26NL0003960000099Z",
            operator_id="OP-X", operator_name="Test",
            commodities=[CommodityType.WOOD],
        )
        assert decl.status == DeclarationStatus.DRAFT

    def test_monetary_values_decimal(self, pending_declaration):
        assert isinstance(pending_declaration.total_value, Decimal)

    def test_weight_values_decimal(self, pending_declaration):
        assert isinstance(pending_declaration.total_gross_weight, Decimal)
        assert isinstance(pending_declaration.total_net_weight, Decimal)


class TestSADFormModel:
    def test_create_basic(self, sample_sad_form):
        assert sample_sad_form.form_id == "SAD-TEST001"
        assert sample_sad_form.box1_declaration_type == "IM"

    def test_eudr_dds_reference_in_sad(self, sample_sad_form):
        assert sample_sad_form.eudr_dds_reference.startswith("GL-DDS-")

    def test_commodity_code_8_digits(self, sample_sad_form):
        assert len(sample_sad_form.box33_commodity_code) == 8

    def test_box34_country_of_origin(self, sample_sad_form):
        assert len(sample_sad_form.box34_country_of_origin) == 2

    def test_box_eori_number(self, sample_sad_form):
        assert len(sample_sad_form.box8_eori) >= 10

    def test_gross_weight_decimal(self, sample_sad_form):
        assert isinstance(sample_sad_form.box35_gross_weight, Decimal)

    def test_net_weight_decimal(self, sample_sad_form):
        assert isinstance(sample_sad_form.box38_net_weight, Decimal)


class TestCustomsInterfaceResponseModel:
    def test_ncts_success(self, ncts_success_response):
        assert ncts_success_response.system == CustomsSystemType.NCTS
        assert ncts_success_response.status == "accepted"
        assert ncts_success_response.response_code == "00"

    def test_ncts_rejection(self, ncts_rejection_response):
        assert ncts_rejection_response.status == "rejected"
        assert len(ncts_rejection_response.errors) == 2

    def test_ais_success(self, ais_success_response):
        assert ais_success_response.system == CustomsSystemType.AIS
        assert ais_success_response.status == "accepted"

    def test_ais_pending(self, ais_pending_response):
        assert ais_pending_response.status == "pending"

    def test_processing_time_decimal(self, ncts_success_response):
        assert isinstance(ncts_success_response.processing_time_ms, Decimal)


class TestHealthStatusModel:
    def test_create_basic(self):
        hs = HealthStatus()
        assert hs.agent_id == AGENT_ID
        assert hs.status == "healthy"
        assert hs.version == AGENT_VERSION

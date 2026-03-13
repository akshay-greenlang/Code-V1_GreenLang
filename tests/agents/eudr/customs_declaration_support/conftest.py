# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-039 Customs Declaration Support tests.

Provides reusable test fixtures for config, models, provenance, engines,
customs declarations, CN code mappings, HS codes, tariff calculations,
country origin verifications, compliance checks, ports of entry, and
mock customs system responses (NCTS/AIS).

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from greenlang.agents.eudr.customs_declaration_support.config import (
    CustomsDeclarationConfig,
    reset_config,
)
from greenlang.agents.eudr.customs_declaration_support.models import (
    AGENT_ID,
    AGENT_VERSION,
    EUDR_CN_CODE_MAPPINGS,
    EUDR_REGULATED_COMMODITIES,
    CNCodeMapping,
    CommodityType,
    ComplianceCheck,
    ComplianceStatus,
    CurrencyCode,
    CustomsDeclaration,
    CustomsInterfaceResponse,
    CustomsSystemType,
    DeclarationStatus,
    DeclarationType,
    DutyCalculation,
    HealthStatus,
    HSCodeInfo,
    IncotermsType,
    MRNStatus,
    OriginVerification,
    OriginVerificationResult,
    PortOfEntry,
    RiskLevel,
    SADForm,
    TariffCalculation,
    TariffLineItem,
    ValidationResult,
)
from greenlang.agents.eudr.customs_declaration_support.provenance import (
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
def sample_config() -> CustomsDeclarationConfig:
    """Create a default CustomsDeclarationConfig instance."""
    return CustomsDeclarationConfig()


# ---------------------------------------------------------------------------
# Provenance fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    """Create a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# CN Code mapping fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cattle_cn_codes() -> List[CNCodeMapping]:
    """Sample CN code mappings for cattle commodity."""
    return [
        CNCodeMapping(
            commodity=CommodityType.CATTLE,
            cn_code="01022110",
            description="Live pure-bred breeding cattle",
            hs_code="010221",
            duty_rate=Decimal("0.0"),
            unit="head",
            provenance_hash="a" * 64,
        ),
        CNCodeMapping(
            commodity=CommodityType.CATTLE,
            cn_code="02011000",
            description="Carcases and half-carcases of bovine animals, fresh or chilled",
            hs_code="020110",
            duty_rate=Decimal("12.8"),
            unit="kg",
            provenance_hash="a" * 64,
        ),
    ]


@pytest.fixture
def cocoa_cn_codes() -> List[CNCodeMapping]:
    """Sample CN code mappings for cocoa commodity."""
    return [
        CNCodeMapping(
            commodity=CommodityType.COCOA,
            cn_code="18010000",
            description="Cocoa beans, whole or broken, raw or roasted",
            hs_code="180100",
            duty_rate=Decimal("0.0"),
            unit="kg",
            provenance_hash="b" * 64,
        ),
        CNCodeMapping(
            commodity=CommodityType.COCOA,
            cn_code="18032000",
            description="Cocoa paste, wholly or partly defatted",
            hs_code="180320",
            duty_rate=Decimal("9.6"),
            unit="kg",
            provenance_hash="b" * 64,
        ),
    ]


@pytest.fixture
def coffee_cn_codes() -> List[CNCodeMapping]:
    """Sample CN code mappings for coffee commodity."""
    return [
        CNCodeMapping(
            commodity=CommodityType.COFFEE,
            cn_code="09011100",
            description="Coffee, not roasted, not decaffeinated",
            hs_code="090111",
            duty_rate=Decimal("0.0"),
            unit="kg",
            provenance_hash="c" * 64,
        ),
    ]


@pytest.fixture
def wood_cn_codes() -> List[CNCodeMapping]:
    """Sample CN code mappings for wood commodity."""
    return [
        CNCodeMapping(
            commodity=CommodityType.WOOD,
            cn_code="44011100",
            description="Fuel wood, in logs, billets, twigs, faggots",
            hs_code="440111",
            duty_rate=Decimal("0.0"),
            unit="m3",
            provenance_hash="d" * 64,
        ),
    ]


@pytest.fixture
def all_commodity_cn_codes(
    cattle_cn_codes, cocoa_cn_codes, coffee_cn_codes, wood_cn_codes
) -> Dict[str, List[CNCodeMapping]]:
    """CN code mappings grouped by commodity."""
    return {
        "cattle": cattle_cn_codes,
        "cocoa": cocoa_cn_codes,
        "coffee": coffee_cn_codes,
        "wood": wood_cn_codes,
    }


# ---------------------------------------------------------------------------
# HS Code fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hs_code_info() -> HSCodeInfo:
    """Create a sample HS code info record."""
    return HSCodeInfo(
        hs_code="180100",
        description="Cocoa beans, whole or broken, raw or roasted",
        chapter=18,
        heading="1801",
        subheading="180100",
        commodity=CommodityType.COCOA,
        eudr_regulated=True,
        notes="Chapter 18 - Cocoa and cocoa preparations",
        provenance_hash="e" * 64,
    )


@pytest.fixture
def invalid_hs_code_info() -> HSCodeInfo:
    """Create an HS code info record for a non-EUDR regulated product."""
    return HSCodeInfo(
        hs_code="870321",
        description="Motor cars - cylinder capacity <= 1000cc",
        chapter=87,
        heading="8703",
        subheading="870321",
        commodity=None,
        eudr_regulated=False,
        provenance_hash="f" * 64,
    )


# ---------------------------------------------------------------------------
# Tariff calculation fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_tariff_line_item() -> TariffLineItem:
    """Create a sample tariff line item."""
    return TariffLineItem(
        line_number=1,
        cn_code="18010000",
        description="Cocoa beans, raw",
        quantity=Decimal("10000.00"),
        unit="kg",
        unit_price=Decimal("2.50"),
        total_value=Decimal("25000.00"),
        currency=CurrencyCode.EUR,
        duty_rate=Decimal("0.0"),
        duty_amount=Decimal("0.00"),
        vat_rate=Decimal("21.0"),
        vat_amount=Decimal("5250.00"),
        origin_country="CI",
        provenance_hash="g" * 64,
    )


@pytest.fixture
def sample_tariff_calculation() -> TariffCalculation:
    """Create a sample tariff calculation."""
    return TariffCalculation(
        calculation_id="CALC-001",
        declaration_id="DECL-001",
        total_customs_value=Decimal("25000.00"),
        total_duty_amount=Decimal("0.00"),
        total_vat_amount=Decimal("5250.00"),
        total_payable=Decimal("5250.00"),
        currency=CurrencyCode.EUR,
        exchange_rate=Decimal("1.0"),
        exchange_rate_date=datetime.now(tz=timezone.utc),
        calculation_method="standard",
        provenance_hash="h" * 64,
    )


@pytest.fixture
def sample_duty_calculation() -> DutyCalculation:
    """Create a sample duty calculation with multiple rates."""
    return DutyCalculation(
        cn_code="18032000",
        customs_value=Decimal("50000.00"),
        duty_rate=Decimal("9.6"),
        duty_amount=Decimal("4800.00"),
        preferential_rate=Decimal("0.0"),
        preferential_origin="CI",
        anti_dumping_duty=Decimal("0.00"),
        countervailing_duty=Decimal("0.00"),
        total_duty=Decimal("4800.00"),
        provenance_hash="i" * 64,
    )


# ---------------------------------------------------------------------------
# Origin verification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_origin_verification() -> OriginVerification:
    """Create a sample country origin verification."""
    return OriginVerification(
        verification_id="OV-001",
        declaration_id="DECL-001",
        declared_origin="CI",
        supply_chain_origins=["CI", "GH"],
        dds_reference="GL-DDS-20260313-ABCDEF",
        result=OriginVerificationResult.VERIFIED,
        confidence_score=Decimal("95.00"),
        verification_method="supply_chain_cross_reference",
        verified_at=datetime.now(tz=timezone.utc),
        provenance_hash="j" * 64,
    )


@pytest.fixture
def failed_origin_verification() -> OriginVerification:
    """Create a failed origin verification."""
    return OriginVerification(
        verification_id="OV-002",
        declaration_id="DECL-002",
        declared_origin="BR",
        supply_chain_origins=["CI", "GH"],
        dds_reference="GL-DDS-20260313-XYZABC",
        result=OriginVerificationResult.MISMATCH,
        confidence_score=Decimal("15.00"),
        verification_method="supply_chain_cross_reference",
        mismatch_details="Declared origin BR not found in supply chain origins [CI, GH]",
        provenance_hash="k" * 64,
    )


# ---------------------------------------------------------------------------
# Compliance check fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_compliance_check() -> ComplianceCheck:
    """Create a sample EUDR compliance check for customs."""
    return ComplianceCheck(
        check_id="CHK-001",
        declaration_id="DECL-001",
        check_type="dds_reference_validation",
        article_reference="Art. 4",
        result=ValidationResult.PASS,
        message="DDS reference number is valid and active",
        dds_reference="GL-DDS-20260313-ABCDEF",
        provenance_hash="l" * 64,
    )


@pytest.fixture
def failed_compliance_check() -> ComplianceCheck:
    """Create a failed compliance check."""
    return ComplianceCheck(
        check_id="CHK-002",
        declaration_id="DECL-002",
        check_type="dds_reference_validation",
        article_reference="Art. 4",
        result=ValidationResult.FAIL,
        message="No valid DDS reference number provided",
        severity="critical",
        suggested_fix="Provide a valid DDS reference number from the EU Information System",
        provenance_hash="m" * 64,
    )


@pytest.fixture
def multiple_compliance_checks() -> List[ComplianceCheck]:
    """Create multiple compliance checks covering different aspects."""
    now = datetime.now(tz=timezone.utc)
    return [
        ComplianceCheck(
            check_id="CHK-001",
            declaration_id="DECL-001",
            check_type="dds_reference_validation",
            result=ValidationResult.PASS,
            message="DDS reference valid",
            provenance_hash="n" * 64,
        ),
        ComplianceCheck(
            check_id="CHK-002",
            declaration_id="DECL-001",
            check_type="cn_code_eudr_coverage",
            result=ValidationResult.PASS,
            message="CN code 18010000 is EUDR-regulated",
            provenance_hash="o" * 64,
        ),
        ComplianceCheck(
            check_id="CHK-003",
            declaration_id="DECL-001",
            check_type="origin_verification",
            result=ValidationResult.PASS,
            message="Origin verified against supply chain",
            provenance_hash="p" * 64,
        ),
        ComplianceCheck(
            check_id="CHK-004",
            declaration_id="DECL-001",
            check_type="deforestation_free_declaration",
            result=ValidationResult.PASS,
            message="Deforestation-free declaration confirmed via DDS",
            provenance_hash="q" * 64,
        ),
        ComplianceCheck(
            check_id="CHK-005",
            declaration_id="DECL-001",
            check_type="risk_level_check",
            result=ValidationResult.WARNING,
            message="Country risk level is HIGH - enhanced due diligence required",
            severity="warning",
            provenance_hash="r" * 64,
        ),
    ]


# ---------------------------------------------------------------------------
# Port of entry fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def port_rotterdam() -> PortOfEntry:
    """Create Rotterdam port of entry."""
    return PortOfEntry(
        port_code="NLRTM",
        port_name="Port of Rotterdam",
        country_code="NL",
        customs_office_code="NL000396",
        timezone="Europe/Amsterdam",
        ncts_enabled=True,
        ais_enabled=True,
        provenance_hash="s" * 64,
    )


@pytest.fixture
def port_hamburg() -> PortOfEntry:
    """Create Hamburg port of entry."""
    return PortOfEntry(
        port_code="DEHAM",
        port_name="Port of Hamburg",
        country_code="DE",
        customs_office_code="DE000396",
        timezone="Europe/Berlin",
        ncts_enabled=True,
        ais_enabled=True,
        provenance_hash="t" * 64,
    )


@pytest.fixture
def port_antwerp() -> PortOfEntry:
    """Create Antwerp port of entry."""
    return PortOfEntry(
        port_code="BEANR",
        port_name="Port of Antwerp",
        country_code="BE",
        customs_office_code="BE000396",
        timezone="Europe/Brussels",
        ncts_enabled=True,
        ais_enabled=True,
        provenance_hash="u" * 64,
    )


@pytest.fixture
def eu_ports(port_rotterdam, port_hamburg, port_antwerp) -> List[PortOfEntry]:
    """Create list of major EU ports of entry."""
    return [port_rotterdam, port_hamburg, port_antwerp]


# ---------------------------------------------------------------------------
# Customs declaration fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pending_declaration() -> CustomsDeclaration:
    """Create a customs declaration in PENDING status."""
    now = datetime.now(tz=timezone.utc)
    return CustomsDeclaration(
        declaration_id="DECL-TEST001",
        mrn="26NL0003960000001A",
        declaration_type=DeclarationType.IMPORT,
        status=DeclarationStatus.PENDING,
        operator_id="OP-001",
        operator_name="Acme Trading Ltd",
        operator_eori="BE1234567890",
        consignee_name="Acme Trading Ltd",
        consignee_eori="BE1234567890",
        port_of_entry="NLRTM",
        customs_office_code="NL000396",
        incoterms=IncotermsType.CIF,
        total_value=Decimal("25000.00"),
        currency=CurrencyCode.EUR,
        total_gross_weight=Decimal("10000.00"),
        total_net_weight=Decimal("9500.00"),
        weight_unit="kg",
        commodities=[CommodityType.COCOA],
        cn_codes=["18010000"],
        hs_codes=["180100"],
        country_of_origin="CI",
        country_of_dispatch="CI",
        dds_reference="GL-DDS-20260313-ABCDEF",
        created_at=now,
        updated_at=now,
        provenance_hash="v" * 64,
    )


@pytest.fixture
def submitted_declaration() -> CustomsDeclaration:
    """Create a customs declaration in SUBMITTED status."""
    now = datetime.now(tz=timezone.utc)
    return CustomsDeclaration(
        declaration_id="DECL-TEST002",
        mrn="26NL0003960000002B",
        declaration_type=DeclarationType.IMPORT,
        status=DeclarationStatus.SUBMITTED,
        operator_id="OP-002",
        operator_name="Global Coffee GmbH",
        operator_eori="DE9876543210",
        consignee_name="Global Coffee GmbH",
        consignee_eori="DE9876543210",
        port_of_entry="DEHAM",
        customs_office_code="DE000396",
        incoterms=IncotermsType.FOB,
        total_value=Decimal("150000.00"),
        currency=CurrencyCode.EUR,
        total_gross_weight=Decimal("50000.00"),
        total_net_weight=Decimal("48000.00"),
        weight_unit="kg",
        commodities=[CommodityType.COFFEE],
        cn_codes=["09011100"],
        hs_codes=["090111"],
        country_of_origin="BR",
        country_of_dispatch="BR",
        dds_reference="GL-DDS-20260312-BCDEFG",
        submitted_at=now - timedelta(hours=2),
        created_at=now - timedelta(hours=4),
        updated_at=now,
        provenance_hash="w" * 64,
    )


@pytest.fixture
def cleared_declaration() -> CustomsDeclaration:
    """Create a customs declaration in CLEARED status."""
    now = datetime.now(tz=timezone.utc)
    return CustomsDeclaration(
        declaration_id="DECL-TEST003",
        mrn="26BE0003960000003C",
        declaration_type=DeclarationType.IMPORT,
        status=DeclarationStatus.CLEARED,
        operator_id="OP-003",
        operator_name="Euro Palm BV",
        operator_eori="NL1122334455",
        consignee_name="Euro Palm BV",
        consignee_eori="NL1122334455",
        port_of_entry="BEANR",
        customs_office_code="BE000396",
        incoterms=IncotermsType.CIF,
        total_value=Decimal("500000.00"),
        currency=CurrencyCode.EUR,
        total_gross_weight=Decimal("200000.00"),
        total_net_weight=Decimal("195000.00"),
        weight_unit="kg",
        commodities=[CommodityType.OIL_PALM],
        cn_codes=["15119110"],
        hs_codes=["151191"],
        country_of_origin="MY",
        country_of_dispatch="MY",
        dds_reference="GL-DDS-20260310-CDEFGH",
        submitted_at=now - timedelta(days=2),
        cleared_at=now - timedelta(days=1),
        created_at=now - timedelta(days=3),
        updated_at=now,
        provenance_hash="x" * 64,
    )


@pytest.fixture
def rejected_declaration() -> CustomsDeclaration:
    """Create a customs declaration in REJECTED status."""
    now = datetime.now(tz=timezone.utc)
    return CustomsDeclaration(
        declaration_id="DECL-TEST004",
        mrn="26DE0003960000004D",
        declaration_type=DeclarationType.IMPORT,
        status=DeclarationStatus.REJECTED,
        operator_id="OP-004",
        operator_name="Timber Import AG",
        operator_eori="DE5544332211",
        consignee_name="Timber Import AG",
        consignee_eori="DE5544332211",
        port_of_entry="DEHAM",
        customs_office_code="DE000396",
        incoterms=IncotermsType.CIF,
        total_value=Decimal("75000.00"),
        currency=CurrencyCode.EUR,
        total_gross_weight=Decimal("30000.00"),
        total_net_weight=Decimal("28500.00"),
        weight_unit="kg",
        commodities=[CommodityType.WOOD],
        cn_codes=["44011100"],
        hs_codes=["440111"],
        country_of_origin="CM",
        country_of_dispatch="CM",
        dds_reference="",
        rejection_reason="Missing DDS reference number - EUDR Article 4 non-compliance",
        submitted_at=now - timedelta(hours=6),
        rejected_at=now - timedelta(hours=4),
        created_at=now - timedelta(hours=8),
        updated_at=now,
        provenance_hash="y" * 64,
    )


# ---------------------------------------------------------------------------
# SAD Form fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_sad_form() -> SADForm:
    """Create a sample Single Administrative Document (SAD) form."""
    now = datetime.now(tz=timezone.utc)
    return SADForm(
        form_id="SAD-TEST001",
        declaration_id="DECL-TEST001",
        box1_declaration_type="IM",
        box2_consignor="Cocoa Cooperative CI, San Pedro, Ivory Coast",
        box3_forms_count=1,
        box5_items_count=1,
        box6_total_packages=100,
        box8_consignee="Acme Trading Ltd, Brussels, Belgium",
        box8_eori="BE1234567890",
        box11_trading_country="CI",
        box14_declarant="Acme Trading Ltd",
        box14_eori="BE1234567890",
        box15_country_of_dispatch="CI",
        box17_country_of_destination="BE",
        box22_currency_and_total="EUR 25000.00",
        box25_mode_of_transport=1,
        box26_inland_mode_of_transport=3,
        box29_office_of_entry="NL000396",
        box31_package_description="100 jute bags of raw cocoa beans",
        box33_commodity_code="18010000",
        box34_country_of_origin="CI",
        box35_gross_weight=Decimal("10000.00"),
        box37_procedure="4000",
        box38_net_weight=Decimal("9500.00"),
        box44_additional_info="DDS Ref: GL-DDS-20260313-ABCDEF",
        box46_statistical_value=Decimal("25000.00"),
        box47_duty_calculation="Duty: EUR 0.00; VAT 21%: EUR 5250.00",
        box54_date_and_signature=now.isoformat(),
        eudr_dds_reference="GL-DDS-20260313-ABCDEF",
        eudr_compliance_status="compliant",
        generated_at=now,
        provenance_hash="z" * 64,
    )


# ---------------------------------------------------------------------------
# Mock customs system response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ncts_success_response() -> CustomsInterfaceResponse:
    """Create a successful NCTS submission response."""
    return CustomsInterfaceResponse(
        system=CustomsSystemType.NCTS,
        request_id="NCTS-REQ-001",
        mrn="26NL0003960000001A",
        status="accepted",
        timestamp=datetime.now(tz=timezone.utc),
        response_code="00",
        response_message="Declaration accepted for processing",
        processing_time_ms=Decimal("245.50"),
        provenance_hash="aa" * 32,
    )


@pytest.fixture
def ncts_rejection_response() -> CustomsInterfaceResponse:
    """Create a rejected NCTS submission response."""
    return CustomsInterfaceResponse(
        system=CustomsSystemType.NCTS,
        request_id="NCTS-REQ-002",
        mrn="26DE0003960000004D",
        status="rejected",
        timestamp=datetime.now(tz=timezone.utc),
        response_code="12",
        response_message="Missing EUDR compliance documentation",
        errors=["E001: DDS reference not found in EU IS", "E002: EUDR Article 4 violation"],
        processing_time_ms=Decimal("120.00"),
        provenance_hash="bb" * 32,
    )


@pytest.fixture
def ais_success_response() -> CustomsInterfaceResponse:
    """Create a successful AIS submission response."""
    return CustomsInterfaceResponse(
        system=CustomsSystemType.AIS,
        request_id="AIS-REQ-001",
        mrn="26BE0003960000003C",
        status="accepted",
        timestamp=datetime.now(tz=timezone.utc),
        response_code="00",
        response_message="Import declaration registered",
        processing_time_ms=Decimal("330.00"),
        provenance_hash="cc" * 32,
    )


@pytest.fixture
def ais_pending_response() -> CustomsInterfaceResponse:
    """Create a pending AIS submission response."""
    return CustomsInterfaceResponse(
        system=CustomsSystemType.AIS,
        request_id="AIS-REQ-002",
        mrn="26NL0003960000005E",
        status="pending",
        timestamp=datetime.now(tz=timezone.utc),
        response_code="01",
        response_message="Declaration queued for manual review",
        processing_time_ms=Decimal("85.00"),
        provenance_hash="dd" * 32,
    )


# ---------------------------------------------------------------------------
# Engine fixtures (initialized with config only)
# ---------------------------------------------------------------------------

@pytest.fixture
def cn_code_mapper(sample_config):
    """Create a CNCodeMapper engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.cn_code_mapper import CNCodeMapper
    return CNCodeMapper(config=sample_config)


@pytest.fixture
def hs_code_validator(sample_config):
    """Create an HSCodeValidator engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.hs_code_validator import HSCodeValidator
    return HSCodeValidator(config=sample_config)


@pytest.fixture
def declaration_generator(sample_config):
    """Create a DeclarationGenerator engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.declaration_generator import DeclarationGenerator
    return DeclarationGenerator(config=sample_config)


@pytest.fixture
def origin_validator(sample_config):
    """Create an OriginValidator engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.origin_validator import OriginValidator
    return OriginValidator(config=sample_config)


@pytest.fixture
def value_calculator(sample_config):
    """Create a ValueCalculator engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.value_calculator import ValueCalculator
    return ValueCalculator(config=sample_config)


@pytest.fixture
def compliance_checker(sample_config):
    """Create a ComplianceChecker engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.compliance_checker import ComplianceChecker
    return ComplianceChecker(config=sample_config)


@pytest.fixture
def customs_interface(sample_config):
    """Create a CustomsInterface engine instance."""
    from greenlang.agents.eudr.customs_declaration_support.customs_interface import CustomsInterface
    return CustomsInterface(config=sample_config)

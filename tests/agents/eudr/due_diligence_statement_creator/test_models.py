# -*- coding: utf-8 -*-
"""
Unit tests for models - AGENT-EUDR-037

Tests all 12 enums, constants, and 15+ Pydantic models.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.due_diligence_statement_creator.models import (
    AGENT_ID, AGENT_VERSION, ARTICLE_4_MANDATORY_FIELDS,
    EU_OFFICIAL_LANGUAGES, EUDR_REGULATED_COMMODITIES,
    AmendmentReason, AmendmentRecord, CommodityType, ComplianceCheck,
    ComplianceStatus, DDSStatement, DDSStatus, DDSValidationReport,
    DigitalSignature, DocumentPackage, DocumentType, GeolocationData,
    GeolocationMethod, HealthStatus, LanguageCode, LanguageTranslation,
    RiskLevel, RiskReference, SignatureType, StatementSummary,
    StatementType, StatementVersion, SubmissionPackage, SubmissionStatus,
    SupplyChainData, TemplateConfig, ValidationResult,
)


# ---------------------------------------------------------------------------
# Enum Tests
# ---------------------------------------------------------------------------

class TestDDSStatusEnum:
    def test_values_count(self):
        assert len(DDSStatus) == 14

    def test_draft_value(self):
        assert DDSStatus.DRAFT.value == "draft"

    def test_submitted_value(self):
        assert DDSStatus.SUBMITTED.value == "submitted"

    def test_accepted_value(self):
        assert DDSStatus.ACCEPTED.value == "accepted"

    def test_rejected_value(self):
        assert DDSStatus.REJECTED.value == "rejected"

    def test_all_values_are_strings(self):
        for status in DDSStatus:
            assert isinstance(status.value, str)


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


class TestComplianceStatusEnum:
    def test_values_count(self):
        assert len(ComplianceStatus) == 5

    def test_compliant_value(self):
        assert ComplianceStatus.COMPLIANT.value == "compliant"

    def test_non_compliant_value(self):
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"


class TestDocumentTypeEnum:
    def test_values_count(self):
        assert len(DocumentType) == 14

    def test_certificate_of_origin(self):
        assert DocumentType.CERTIFICATE_OF_ORIGIN.value == "certificate_of_origin"

    def test_other(self):
        assert DocumentType.OTHER.value == "other"


class TestSignatureTypeEnum:
    def test_values_count(self):
        assert len(SignatureType) == 3

    def test_qualified(self):
        assert SignatureType.QUALIFIED_ELECTRONIC.value == "qualified_electronic"


class TestValidationResultEnum:
    def test_values_count(self):
        assert len(ValidationResult) == 4

    def test_pass_value(self):
        assert ValidationResult.PASS.value == "pass"

    def test_fail_value(self):
        assert ValidationResult.FAIL.value == "fail"


class TestSubmissionStatusEnum:
    def test_values_count(self):
        assert len(SubmissionStatus) == 7


class TestAmendmentReasonEnum:
    def test_values_count(self):
        assert len(AmendmentReason) == 8

    def test_correction_of_error(self):
        assert AmendmentReason.CORRECTION_OF_ERROR.value == "correction_of_error"


class TestGeolocationMethodEnum:
    def test_values_count(self):
        assert len(GeolocationMethod) == 7

    def test_gps_field_survey(self):
        assert GeolocationMethod.GPS_FIELD_SURVEY.value == "gps_field_survey"


class TestLanguageCodeEnum:
    def test_values_count(self):
        assert len(LanguageCode) == 24

    def test_en_value(self):
        assert LanguageCode.EN.value == "en"


class TestStatementTypeEnum:
    def test_values_count(self):
        assert len(StatementType) == 3

    def test_placing_value(self):
        assert StatementType.PLACING.value == "placing"

    def test_export_value(self):
        assert StatementType.EXPORT.value == "export"


# ---------------------------------------------------------------------------
# Constant Tests
# ---------------------------------------------------------------------------

class TestConstants:
    def test_agent_id(self):
        assert AGENT_ID == "GL-EUDR-DDSC-037"

    def test_agent_version(self):
        assert AGENT_VERSION == "1.0.0"

    def test_regulated_commodities_count(self):
        assert len(EUDR_REGULATED_COMMODITIES) == 7

    def test_mandatory_fields_count(self):
        assert len(ARTICLE_4_MANDATORY_FIELDS) == 14

    def test_official_languages_count(self):
        assert len(EU_OFFICIAL_LANGUAGES) == 24

    def test_mandatory_fields_include_operator_name(self):
        assert "operator_name" in ARTICLE_4_MANDATORY_FIELDS

    def test_mandatory_fields_include_geolocation(self):
        assert "geolocation_of_plots" in ARTICLE_4_MANDATORY_FIELDS

    def test_regulated_commodities_all_types(self):
        for c in EUDR_REGULATED_COMMODITIES:
            assert isinstance(c, CommodityType)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestGeolocationDataModel:
    def test_create_basic(self, sample_geolocation):
        assert sample_geolocation.plot_id == "PLT-001"
        assert sample_geolocation.latitude == Decimal("5.123456")

    def test_longitude_range(self):
        geo = GeolocationData(
            plot_id="PLT-X", latitude=Decimal("0"), longitude=Decimal("-180"))
        assert geo.longitude == Decimal("-180")

    def test_area_default(self):
        geo = GeolocationData(
            plot_id="PLT-X", latitude=Decimal("0"), longitude=Decimal("0"))
        assert geo.area_hectares == Decimal("0")


class TestRiskReferenceModel:
    def test_create_basic(self, sample_risk_reference):
        assert sample_risk_reference.risk_id == "RISK-001"
        assert sample_risk_reference.risk_level == RiskLevel.STANDARD

    def test_default_risk_level(self):
        ref = RiskReference(risk_id="R1", source_agent="E016", risk_category="country")
        assert ref.risk_level == RiskLevel.STANDARD


class TestSupplyChainDataModel:
    def test_create_basic(self, sample_supply_chain):
        assert sample_supply_chain.supply_chain_id == "SC-001"
        assert sample_supply_chain.tier_count == 3

    def test_default_custody_model(self):
        sc = SupplyChainData(
            supply_chain_id="SC-X", operator_id="OP-X",
            commodity=CommodityType.COCOA)
        assert sc.chain_of_custody_model == "segregation"


class TestComplianceCheckModel:
    def test_create_basic(self):
        cc = ComplianceCheck(
            check_id="CHK-001", field_name="operator_name",
            article_reference="Art. 4(2)",
            result=ValidationResult.PASS)
        assert cc.check_id == "CHK-001"
        assert cc.result == ValidationResult.PASS


class TestDocumentPackageModel:
    def test_create_basic(self, sample_document):
        assert sample_document.document_id == "DOC-TEST001"
        assert sample_document.document_type == DocumentType.CERTIFICATE_OF_ORIGIN


class TestStatementVersionModel:
    def test_create_basic(self, sample_version):
        assert sample_version.version_id == "VER-TEST001"
        assert sample_version.version_number == 1


class TestDigitalSignatureModel:
    def test_create_basic(self, sample_signature):
        assert sample_signature.signature_id == "SIG-TEST001"
        assert sample_signature.is_valid is True


class TestDDSValidationReportModel:
    def test_create_basic(self):
        report = DDSValidationReport(
            report_id="VR-001", statement_id="DDS-001")
        assert report.overall_result == ValidationResult.PASS
        assert report.total_checks == 0


class TestSubmissionPackageModel:
    def test_create_basic(self):
        pkg = SubmissionPackage(
            package_id="PKG-001", statement_id="DDS-001",
            operator_id="OP-001")
        assert pkg.submission_status == SubmissionStatus.PENDING


class TestAmendmentRecordModel:
    def test_create_basic(self):
        amd = AmendmentRecord(
            amendment_id="AMD-001", statement_id="DDS-001",
            reason=AmendmentReason.CORRECTION_OF_ERROR,
            description="Fix typo", previous_version=1, new_version=2)
        assert amd.new_version == 2


class TestTemplateConfigModel:
    def test_create_basic(self):
        tmpl = TemplateConfig(template_id="TMPL-001")
        assert tmpl.language == LanguageCode.EN


class TestLanguageTranslationModel:
    def test_create_basic(self):
        tr = LanguageTranslation(
            translation_id="TR-001", target_language=LanguageCode.FR,
            field_key="operator_name", source_text="Name",
            translated_text="Nom")
        assert tr.target_language == LanguageCode.FR


class TestStatementSummaryModel:
    def test_create_basic(self):
        ss = StatementSummary(
            statement_id="DDS-001", reference_number="GL-DDS-001",
            operator_id="OP-001")
        assert ss.status == DDSStatus.DRAFT


class TestDDSStatementModel:
    def test_create_basic(self, sample_statement):
        assert sample_statement.statement_id == "DDS-TEST001"
        assert sample_statement.status == DDSStatus.DRAFT
        assert len(sample_statement.commodities) == 2

    def test_operator_fields(self, sample_statement):
        assert sample_statement.operator_name == "Acme Trading Ltd"
        assert sample_statement.operator_eori_number == "BE1234567890"

    def test_default_values(self):
        stmt = DDSStatement(
            statement_id="DDS-X", reference_number="REF-X",
            operator_id="OP-X", operator_name="Test",
            commodities=[CommodityType.WOOD])
        assert stmt.status == DDSStatus.DRAFT
        assert stmt.version_number == 1
        assert stmt.language == "en"


class TestHealthStatusModel:
    def test_create_basic(self):
        hs = HealthStatus()
        assert hs.agent_id == AGENT_ID
        assert hs.status == "healthy"
        assert hs.version == AGENT_VERSION
